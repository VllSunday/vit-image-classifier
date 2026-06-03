"""Тесты загрузки данных: сканирование, стратифицированный split, Dataset."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.config import Config
from src.data.dataset import (
    AnimalsDataset,
    build_dataloaders,
    scan_samples,
    split_samples,
    subsample_stratified,
)
from src.data.transforms import build_eval_transform

CLASS_NAMES = ("cat", "dog", "panda")
PER_CLASS = 20


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Создаём временный датасет: по PER_CLASS картинок на класс."""
    for cls in CLASS_NAMES:
        class_dir = tmp_path / cls
        class_dir.mkdir()
        for i in range(PER_CLASS):
            Image.new("RGB", (32, 32), color=(i, i, i)).save(class_dir / f"{cls}_{i}.jpg")
    return tmp_path


def test_scan_samples_counts_and_labels(data_dir: Path) -> None:
    samples = scan_samples(data_dir, CLASS_NAMES)
    assert len(samples) == PER_CLASS * len(CLASS_NAMES)
    # Метки лежат в диапазоне индексов классов.
    assert {label for _, label in samples} == {0, 1, 2}


def test_split_sizes_and_no_leakage(data_dir: Path) -> None:
    samples = scan_samples(data_dir, CLASS_NAMES)
    train, val, test = split_samples(samples, val_split=0.15, test_split=0.15, seed=42)

    # Сумма сплитов равна исходному размеру.
    assert len(train) + len(val) + len(test) == len(samples)

    # Нет пересечений между сплитами (отсутствие утечки).
    paths_train = {p for p, _ in train}
    paths_val = {p for p, _ in val}
    paths_test = {p for p, _ in test}
    assert paths_train.isdisjoint(paths_val)
    assert paths_train.isdisjoint(paths_test)
    assert paths_val.isdisjoint(paths_test)

    # Каждый класс представлен во всех трёх сплитах (стратификация).
    for split in (train, val, test):
        assert {label for _, label in split} == {0, 1, 2}


def test_split_is_deterministic(data_dir: Path) -> None:
    samples = scan_samples(data_dir, CLASS_NAMES)
    first = split_samples(samples, 0.15, 0.15, seed=42)
    second = split_samples(samples, 0.15, 0.15, seed=42)
    assert [p for p, _ in first[0]] == [p for p, _ in second[0]]


def test_subsample_stratified_keeps_balance(data_dir: Path) -> None:
    samples = scan_samples(data_dir, CLASS_NAMES)
    subset = subsample_stratified(samples, fraction=0.5, seed=42)

    # Половина выборки, баланс классов сохранён.
    assert len(subset) == len(samples) // 2
    counts = {label: 0 for label in range(len(CLASS_NAMES))}
    for _, label in subset:
        counts[label] += 1
    assert all(c == PER_CLASS // 2 for c in counts.values())


def test_subsample_full_returns_all(data_dir: Path) -> None:
    samples = scan_samples(data_dir, CLASS_NAMES)
    assert subsample_stratified(samples, fraction=1.0, seed=42) is samples


def test_dataset_returns_tensor_and_label(data_dir: Path) -> None:
    cfg = Config()
    samples = scan_samples(data_dir, CLASS_NAMES)
    ds = AnimalsDataset(samples, build_eval_transform(cfg))
    image, label = ds[0]
    assert image.shape == (3, 224, 224)
    assert isinstance(label, int)


def test_build_dataloaders_yields_correct_batch(data_dir: Path) -> None:
    cfg = Config(data_dir=data_dir, batch_size=8, num_workers=0)
    bundle = build_dataloaders(cfg)

    images, labels = next(iter(bundle.train_loader))
    assert images.shape == (8, 3, 224, 224)
    assert images.dtype == torch.float32
    assert labels.shape == (8,)
