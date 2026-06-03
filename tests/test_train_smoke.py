"""Smoke-тест обучения: маленькая модель на синтетике, один полный цикл.

pretrained=False и крошечная архитектура — чтобы тест не качал веса и был быстрым.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.config import Config
from src.train import train

CLASS_NAMES = ("cat", "dog", "panda")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    for cls in CLASS_NAMES:
        class_dir = tmp_path / cls
        class_dir.mkdir()
        for i in range(12):
            Image.new("RGB", (32, 32), color=(i * 5, i * 3, i)).save(class_dir / f"{cls}_{i}.jpg")
    return tmp_path


def test_training_runs_and_saves_checkpoint(tmp_path: Path, data_dir: Path) -> None:
    cfg = Config(
        data_dir=data_dir,
        checkpoints_dir=tmp_path / "checkpoints",
        runs_dir=tmp_path / "runs",
        # Крошечная архитектура ради скорости теста.
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        batch_size=4,
        epochs=1,
        num_workers=0,
        use_amp=False,
        warmup_ratio=0.5,
        strategy="linear_probe",
    )

    result = train(cfg, pretrained=False)

    assert len(result["history"]) == 1
    assert 0.0 <= result["best_val_acc"] <= 1.0
    assert Path(result["best_checkpoint"]).exists()
