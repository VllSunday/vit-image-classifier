"""Инференс обученной модели на одиночных изображениях.

Лёгкий модуль для демо и прода: загружает чекпойнт, прогоняет картинку через тот
же пайплайн предобработки, что и при обучении, и возвращает вероятности классов.
Вход модели строго соответствует стандарту (Batch_Size, 3, 224, 224).

Дополнительно умеет отдавать «расширенный» результат для UI: время инференса,
превью обработанного изображения (что реально видит модель) и attention map —
карту внимания [CLS] токена к патчам последнего блока энкодера.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

from src.config import Config
from src.data.transforms import build_eval_transform
from src.models.vit import ViTClassifier


def resolve_checkpoint(
    checkpoint: str | Path,
    repo_id: str | None = None,
    filename: str | None = None,
) -> Path:
    """Находим чекпойнт локально, иначе скачиваем с Hugging Face Hub.

    Это позволяет запустить демо «из коробки» (например, в Docker) без обучения:
    если локального файла нет, но задан repo_id, веса один раз скачиваются с Hub
    и кешируются (повторные запуски берут их из кеша, заново не качают).
    """
    path = Path(checkpoint)
    if path.exists():
        return path

    if repo_id:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=repo_id, filename=filename or path.name)
        return Path(downloaded)

    raise FileNotFoundError(
        f"Не найден чекпойнт: {checkpoint}. Сначала обучите модель "
        "(python -m src.train) или задайте repo_id на Hugging Face Hub."
    )


class Predictor:
    """Обёртка над моделью для предсказания вероятностей классов по изображению."""

    def __init__(
        self,
        cfg: Config,
        checkpoint_path: str | Path,
        attn_implementation: str = "sdpa",
        capture_attention: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = cfg.device

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.class_names: list[str] = checkpoint.get("class_names", list(cfg.class_names))

        # Архитектуру не качаем заново — поднимаем случайную и грузим веса из чекпойнта.
        model = ViTClassifier(cfg, pretrained=False, attn_implementation=attn_implementation)
        model.load_state_dict(checkpoint["model_state"])
        self.model = model.to(self.device).eval()

        self.transform = build_eval_transform(cfg)

        # Hooks для перехвата весов внимания со ВСЕХ блоков энкодера — нужны для
        # attention rollout (перемножения карт внимания по всем слоям).
        self._attentions: list[torch.Tensor] = []
        self.capture_attention = capture_attention
        if capture_attention:
            for layer in self.model.layers:
                layer.attention.register_forward_hook(self._attention_hook)

    def _attention_hook(self, module, inputs, output) -> None:
        # ViTAttention возвращает (attn_output, attn_weights) — копим веса по слоям.
        self._attentions.append(output[1].detach())

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL-изображение -> тензор (1, 3, 224, 224)."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict[str, float]:
        """Возвращаем словарь {класс: вероятность} для одного изображения."""
        batch = self.preprocess(image)  # (1, 3, 224, 224)
        probs = torch.softmax(self.model(batch), dim=1)[0]
        return {cls: float(p) for cls, p in zip(self.class_names, probs.tolist(), strict=True)}

    @torch.no_grad()
    def predict_detailed(self, image: Image.Image) -> dict:
        """Расширенный результат для UI: вероятности, тайминг, превью и attention map."""
        batch = self.preprocess(image)
        self._attentions = []  # очищаем перед прогоном, hooks наполнят заново

        if self.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = self.model(batch)
        if self.device == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - start) * 1000

        probs_tensor = torch.softmax(logits, dim=1)[0]
        probs = {
            cls: float(p) for cls, p in zip(self.class_names, probs_tensor.tolist(), strict=True)
        }

        preview = self._denormalize(batch[0])
        attention = None
        if self._attentions:
            attention = self._attention_overlay(preview, self._attentions)

        return {
            "probs": probs,
            "inference_ms": inference_ms,
            "preview": preview,
            "attention": attention,
            "max_prob": max(probs.values()),
        }

    def _denormalize(self, tensor: torch.Tensor) -> Image.Image:
        """Тензор (3, 224, 224) -> PIL: отменяем нормализацию, чтобы показать вход модели."""
        mean = torch.tensor(self.cfg.norm_mean, device=tensor.device).view(3, 1, 1)
        std = torch.tensor(self.cfg.norm_std, device=tensor.device).view(3, 1, 1)
        denorm = (tensor * std + mean).clamp(0, 1)
        array = (denorm.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        return Image.fromarray(array)

    def _attention_overlay(
        self, base_image: Image.Image, attentions: list[torch.Tensor]
    ) -> Image.Image:
        """Attention rollout по всем слоям -> тепловая карта поверх изображения.

        Rollout (Abnar & Zuidema, 2020): усредняем внимание по головам, добавляем
        единичную матрицу (учёт residual-связи) и перемножаем карты всех слоёв.
        Так получается, насколько каждый патч влияет на итоговый [CLS] токен —
        локализация объекта чище, чем у одиночного последнего слоя.
        """
        grid = self.cfg.image_size // self.cfg.patch_size  # 14

        num_tokens = attentions[0].size(-1)
        rollout = torch.eye(num_tokens, device=attentions[0].device)
        for attn in attentions:
            averaged = attn[0].mean(dim=0)  # усреднение по головам -> (N, N)
            averaged = averaged + torch.eye(num_tokens, device=averaged.device)  # residual
            averaged = averaged / averaged.sum(dim=-1, keepdim=True)  # нормировка строк
            rollout = averaged @ rollout

        # Влияние патчей на [CLS] токен (первая строка, без самого [CLS]).
        cls_attention = rollout[0, 1:]  # (196,)
        heat = cls_attention.reshape(grid, grid).float().cpu().numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

        # Растягиваем 14x14 до размера изображения и раскрашиваем colormap.
        heat_img = Image.fromarray((heat * 255).astype("uint8")).resize(
            (self.cfg.image_size, self.cfg.image_size), Image.BILINEAR
        )
        heat_norm = np.asarray(heat_img) / 255.0
        colored = (colormaps["jet"](heat_norm)[..., :3] * 255).astype("uint8")

        return Image.blend(base_image.convert("RGB"), Image.fromarray(colored), alpha=0.5)
