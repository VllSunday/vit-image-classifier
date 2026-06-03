"""Инференс обученной модели на одиночных изображениях.

Лёгкий модуль для демо и прода: загружает чекпойнт, прогоняет картинку через тот
же пайплайн предобработки, что и при обучении, и возвращает вероятности классов.
Вход модели строго соответствует стандарту (Batch_Size, 3, 224, 224).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.config import Config
from src.data.transforms import build_eval_transform
from src.models.vit import ViTClassifier


class Predictor:
    """Обёртка над моделью для предсказания вероятностей классов по изображению."""

    def __init__(self, cfg: Config, checkpoint_path: str | Path) -> None:
        self.cfg = cfg
        self.device = cfg.device

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.class_names: list[str] = checkpoint.get("class_names", list(cfg.class_names))

        # Архитектуру не качаем заново — поднимаем случайную и грузим веса из чекпойнта.
        model = ViTClassifier(cfg, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        self.model = model.to(self.device).eval()

        self.transform = build_eval_transform(cfg)

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
