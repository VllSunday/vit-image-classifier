"""Patch embedding для Vision Transformer (реализация вручную).

Здесь собрана вся «входная» часть ViT согласно ТЗ:
  1. нарезка изображения на патчи 16x16;
  2. линейная проекция каждого флаттен-патча в вектор размерности embed_dim;
  3. добавление обучаемого [CLS] токена в начало последовательности;
  4. прибавление позиционных эмбеддингов.

Нарезка на патчи сделана явно через reshape/permute (а не спрятана в Conv2d),
чтобы было видно, что происходит. При этом операция математически эквивалентна
свёртке с ядром 16x16 и шагом 16 — это используется для загрузки предобученных
весов из google/vit-base-patch16-224 (см. load_pretrained).
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import Config


class PatchEmbedding(nn.Module):
    """Превращает картинку (B, 3, 224, 224) в последовательность токенов (B, 197, 768)."""

    def __init__(self, cfg: Config, dropout: float = 0.0) -> None:
        super().__init__()
        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size
        self.in_channels = cfg.in_channels
        self.embed_dim = cfg.embed_dim
        self.num_patches = cfg.num_patches

        # Размерность одного флаттен-патча: C * P * P = 3 * 16 * 16 = 768.
        patch_dim = self.in_channels * self.patch_size * self.patch_size

        # Линейная проекция флаттен-патча в пространство эмбеддингов.
        self.projection = nn.Linear(patch_dim, self.embed_dim)

        # Обучаемый [CLS] токен — его выход потом идёт в классификатор.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Позиционные эмбеддинги для патчей + [CLS] (всего num_patches + 1).
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Инициализация по умолчанию (перезапишется при загрузке предобученных весов)."""
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Нарезаем (B, C, H, W) на флаттен-патчи (B, num_patches, C*P*P).

        Порядок флаттенинга внутри патча — (канал, строка, столбец). Именно он
        совпадает с раскладкой весов свёрточной проекции в ViT, что и позволяет
        переиспользовать предобученные веса.
        """
        b, c, h, w = x.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(
                f"Ожидался размер {self.image_size}x{self.image_size}, получено {h}x{w}."
            )

        p = self.patch_size
        n_h, n_w = h // p, w // p

        # (B, C, n_h, P, n_w, P)
        x = x.reshape(b, c, n_h, p, n_w, p)
        # (B, n_h, n_w, C, P, P) — собираем оси патча рядом, каналы впереди.
        x = x.permute(0, 2, 4, 1, 3, 5)
        # (B, n_h * n_w, C * P * P)
        x = x.reshape(b, n_h * n_w, c * p * p)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) -> (B, 197, 768)."""
        batch_size = x.shape[0]

        patches = self._to_patches(x)
        tokens = self.projection(patches)  # (B, num_patches, embed_dim)

        # Добавляем [CLS] токен в начало каждой последовательности.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, num_patches + 1, embed_dim)

        # Прибавляем позиционные эмбеддинги (broadcast по батчу).
        tokens = tokens + self.position_embeddings
        return self.dropout(tokens)

    @torch.no_grad()
    def load_projection_from_conv(self, conv: nn.Conv2d) -> None:
        """Загружаем веса линейной проекции из эквивалентной свёртки 16x16.

        Conv2d-вес имеет форму (embed_dim, C, P, P); его развёртка в
        (embed_dim, C*P*P) совпадает с весом нашего Linear-слоя.
        """
        self.projection.weight.copy_(conv.weight.reshape(self.embed_dim, -1))
        self.projection.bias.copy_(conv.bias)

    @torch.no_grad()
    def load_pretrained(self, model_name: str) -> None:
        """Инициализируем patch embedding весами предобученного ViT.

        Берём из google/vit-base-patch16-224 свёрточную проекцию патчей,
        [CLS] токен и позиционные эмбеддинги.
        """
        from transformers import ViTModel

        backbone = ViTModel.from_pretrained(model_name)
        embeddings = backbone.embeddings

        self.load_projection_from_conv(embeddings.patch_embeddings.projection)
        self.cls_token.copy_(embeddings.cls_token)
        self.position_embeddings.copy_(embeddings.position_embeddings)
