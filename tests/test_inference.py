"""Тесты инференса: формат входа (1, 3, 224, 224) и вероятности классов."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.config import Config
from src.inference import Predictor
from src.models.vit import ViTClassifier


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> tuple[Config, Path]:
    """Сохраняем крошечную модель в чекпойнт, чтобы не качать веса."""
    cfg = Config(embed_dim=32, num_layers=2, num_heads=4)
    model = ViTClassifier(cfg, pretrained=False)
    path = tmp_path / "tiny.pt"
    torch.save(
        {"model_state": model.state_dict(), "class_names": list(cfg.class_names)},
        path,
    )
    return cfg, path


def test_preprocess_produces_standard_input(tiny_checkpoint: tuple[Config, Path]) -> None:
    cfg, path = tiny_checkpoint
    predictor = Predictor(cfg, path)
    tensor = predictor.preprocess(Image.new("RGB", (300, 200)))
    # Строгий формат инференса: (Batch_Size, 3, 224, 224).
    assert tensor.shape == (1, 3, 224, 224)


def test_predict_returns_probability_distribution(tiny_checkpoint: tuple[Config, Path]) -> None:
    cfg, path = tiny_checkpoint
    predictor = Predictor(cfg, path)
    probs = predictor.predict(Image.new("RGB", (224, 224), color=(50, 100, 150)))

    # Три класса, вероятности в [0, 1] и суммируются в 1.
    assert set(probs) == set(cfg.class_names)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)


def test_predict_handles_grayscale(tiny_checkpoint: tuple[Config, Path]) -> None:
    cfg, path = tiny_checkpoint
    predictor = Predictor(cfg, path)
    # Серое изображение должно корректно приводиться к трём каналам.
    probs = predictor.predict(Image.new("L", (224, 224), color=128))
    assert len(probs) == 3
