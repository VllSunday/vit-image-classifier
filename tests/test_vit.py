"""Тесты сборки модели: форма выхода и логика заморозки слоёв.

Используем pretrained=False, чтобы тесты не качали веса в CI.
"""

from __future__ import annotations

import torch

from src.config import Config
from src.models.vit import ViTClassifier


def test_forward_output_shape() -> None:
    cfg = Config()
    model = ViTClassifier(cfg, pretrained=False).eval()
    x = torch.randn(2, cfg.in_channels, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        logits = model(x)
    # Строгое соответствие формату инференса: (Batch, num_classes).
    assert logits.shape == (2, cfg.num_classes)


def test_linear_probe_freezes_backbone() -> None:
    cfg = Config(strategy="linear_probe")
    model = ViTClassifier(cfg, pretrained=False)

    # Обучаемы только параметры головы.
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"head.weight", "head.bias"}


def test_gradual_unfreeze_opens_last_layers() -> None:
    cfg = Config(strategy="gradual_unfreeze", num_unfrozen_layers=2)
    model = ViTClassifier(cfg, pretrained=False)

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}

    # Голова обучается.
    assert "head.weight" in trainable
    # Последние два блока энкодера разморожены.
    assert any(name.startswith("layers.10.") for name in trainable)
    assert any(name.startswith("layers.11.") for name in trainable)
    # Ранние блоки остаются замороженными.
    assert not any(name.startswith("layers.0.") for name in trainable)
    # Patch embedding заморожен.
    assert not any(name.startswith("patch_embedding.") for name in trainable)


def test_param_groups_have_discriminative_lr() -> None:
    cfg = Config(strategy="gradual_unfreeze", num_unfrozen_layers=1)
    model = ViTClassifier(cfg, pretrained=False)

    groups = model.param_groups()
    # Две группы: голова (lr_head) и backbone (lr_backbone).
    assert len(groups) == 2
    assert groups[0]["lr"] == cfg.lr_head
    assert groups[1]["lr"] == cfg.lr_backbone


def test_param_groups_linear_probe_head_only() -> None:
    cfg = Config(strategy="linear_probe")
    model = ViTClassifier(cfg, pretrained=False)

    groups = model.param_groups()
    # При заморозке backbone остаётся только группа головы.
    assert len(groups) == 1
    assert groups[0]["lr"] == cfg.lr_head


def test_trainable_count_linear_probe() -> None:
    cfg = Config(strategy="linear_probe")
    model = ViTClassifier(cfg, pretrained=False)
    # Голова: 768 * 3 + 3 = 2307 параметров.
    expected = cfg.embed_dim * cfg.num_classes + cfg.num_classes
    assert model.num_trainable_parameters() == expected
