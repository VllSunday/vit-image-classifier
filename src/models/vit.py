"""Полная модель ViT-классификатора.

Собирает три части:
  1. кастомный patch embedding (наш, см. patch_embedding.py);
  2. Transformer Encoder из предобученного google/vit-base-patch16-224
     (берём «из коробки» — это переиспользование базовых весов);
  3. кастомную голову-классификатор поверх выхода [CLS] токена.

Здесь же — логика дообучения: linear probing (заморозка backbone) и
постепенная разморозка последних блоков энкодера (gradual unfreezing).
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import Config
from src.models.patch_embedding import PatchEmbedding


class ViTClassifier(nn.Module):
    """ViT для классификации: (B, 3, 224, 224) -> логиты (B, num_classes)."""

    def __init__(
        self, cfg: Config, pretrained: bool = True, attn_implementation: str = "sdpa"
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # 1. Кастомный patch embedding.
        self.patch_embedding = PatchEmbedding(cfg)

        # 2. Энкодер и финальный LayerNorm берём из ViT.
        # attn_implementation="eager" нужен, чтобы можно было достать веса внимания
        # (для визуализации attention map в демо); по умолчанию "sdpa" — быстрее.
        from transformers import ViTConfig, ViTModel

        if pretrained:
            backbone = ViTModel.from_pretrained(
                cfg.model_name, attn_implementation=attn_implementation
            )
            # Инициализируем наш embedding теми же предобученными весами.
            self.patch_embedding.load_from_vit_embeddings(backbone.embeddings)
        else:
            # Случайная инициализация той же архитектуры — для тестов без скачивания.
            vit_config = ViTConfig(
                hidden_size=cfg.embed_dim,
                num_hidden_layers=cfg.num_layers,
                num_attention_heads=cfg.num_heads,
                image_size=cfg.image_size,
                patch_size=cfg.patch_size,
                num_channels=cfg.in_channels,
            )
            vit_config._attn_implementation = attn_implementation
            backbone = ViTModel(vit_config)

        # В transformers 5.x стек блоков энкодера лежит прямо в ViTModel.layers
        # (ModuleList из ViTLayer), а не во вложенном объекте encoder.
        self.layers = backbone.layers
        self.layernorm = backbone.layernorm

        # 3. Голова-классификатор поверх [CLS].
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        # Применяем выбранную стратегию заморозки.
        self.configure_finetuning()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.patch_embedding(x)  # (B, 197, 768)
        # Прогоняем последовательность токенов через блоки Transformer Encoder.
        for layer in self.layers:
            output = layer(hidden)
            # ViTLayer может вернуть кортеж (при output_attentions) — берём hidden.
            hidden = output[0] if isinstance(output, tuple) else output
        hidden = self.layernorm(hidden)
        cls = hidden[:, 0]  # выход [CLS] токена -> (B, 768)
        return self.head(cls)  # (B, num_classes)

    # --- Управление заморозкой слоёв ---

    def freeze_backbone(self) -> None:
        """Замораживаем всё, кроме головы."""
        for module in (self.patch_embedding, self.layers, self.layernorm):
            for param in module.parameters():
                param.requires_grad_(False)

    def unfreeze_last_encoder_layers(self, num_layers: int) -> None:
        """Размораживаем последние num_layers блоков энкодера и финальный LayerNorm."""
        if num_layers > 0:
            for layer in self.layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad_(True)
            for param in self.layernorm.parameters():
                param.requires_grad_(True)

    def configure_finetuning(self) -> None:
        """Настраиваем требования к градиентам согласно cfg.strategy."""
        strategy = self.cfg.strategy
        if strategy == "linear_probe":
            # Учим только голову — backbone полностью заморожен.
            self.freeze_backbone()
        elif strategy == "gradual_unfreeze":
            # Замораживаем всё, затем «приоткрываем» последние блоки энкодера.
            self.freeze_backbone()
            self.unfreeze_last_encoder_layers(self.cfg.num_unfrozen_layers)
        elif strategy == "full":
            # Обучаем всю сеть целиком (для обучения с нуля) — ничего не замораживаем.
            pass
        else:
            raise ValueError(
                f"Неизвестная стратегия дообучения: {strategy!r}. "
                "Ожидается 'linear_probe', 'gradual_unfreeze' или 'full'."
            )

    # --- Группы параметров для дискриминативного learning rate ---

    def param_groups(self) -> list[dict]:
        """Группы параметров с разными LR: голова — крупнее, backbone — мельче.

        Возвращаются только параметры с requires_grad=True.
        """
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        backbone_params = [
            p
            for name, p in self.named_parameters()
            if p.requires_grad and not name.startswith("head.")
        ]

        groups: list[dict] = [{"params": head_params, "lr": self.cfg.lr_head}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": self.cfg.lr_backbone})
        return groups

    def num_trainable_parameters(self) -> int:
        """Число обучаемых параметров (для логов)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
