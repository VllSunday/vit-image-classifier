"""Тесты пайплайна предобработки: форма и тип выходного тензора."""

from __future__ import annotations

import torch
from PIL import Image

from src.config import Config
from src.data.transforms import build_eval_transform, build_train_transform


def _dummy_image(width: int = 100, height: int = 80) -> Image.Image:
    # Изображение нестандартного размера, чтобы проверить ресайз до 224x224.
    return Image.new("RGB", (width, height), color=(120, 30, 200))


def test_eval_transform_output_shape() -> None:
    cfg = Config()
    tensor = build_eval_transform(cfg)(_dummy_image())
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_train_transform_output_shape() -> None:
    cfg = Config()
    tensor = build_train_transform(cfg)(_dummy_image())
    assert tensor.shape == (3, 224, 224)


def test_grayscale_is_converted_to_three_channels() -> None:
    cfg = Config()
    gray = Image.new("L", (64, 64), color=128)
    tensor = build_eval_transform(cfg)(gray.convert("RGB"))
    assert tensor.shape[0] == 3
