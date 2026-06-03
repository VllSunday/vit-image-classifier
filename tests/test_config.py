"""Smoke-тесты конфигурации: проверяем инварианты, а не конкретные значения."""

from __future__ import annotations

from src.config import Config, get_device, set_seed


def test_num_classes_matches_class_names() -> None:
    cfg = Config()
    assert cfg.num_classes == len(cfg.class_names) == 3


def test_patch_geometry() -> None:
    cfg = Config()
    # 224 / 16 = 14 патчей по стороне → 196 патчей, +1 [CLS] токен = 197.
    assert cfg.num_patches == 196
    assert cfg.seq_length == 197


def test_image_divisible_by_patch() -> None:
    cfg = Config()
    assert cfg.image_size % cfg.patch_size == 0


def test_get_device_returns_known_backend() -> None:
    assert get_device() in {"cuda", "mps", "cpu"}


def test_set_seed_is_deterministic() -> None:
    import random

    set_seed(123)
    first = [random.random() for _ in range(5)]
    set_seed(123)
    second = [random.random() for _ in range(5)]
    assert first == second
