"""Загрузка датасета и разбиение на train/val/test.

Сканируем папки data/<class>, формируем список (путь, метка), делаем
стратифицированный split и оборачиваем всё в torch Dataset / DataLoader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.config import Config
from src.data.transforms import build_eval_transform, build_train_transform

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Один пример: путь к изображению и индекс класса.
Sample = tuple[Path, int]


def scan_samples(data_dir: Path, class_names: tuple[str, ...]) -> list[Sample]:
    """Собираем список (путь, метка) по папкам data/<class>.

    Индекс класса определяется позицией в class_names, что фиксирует
    соответствие «имя класса → метка» во всём проекте.
    """
    samples: list[Sample] = []
    for label, cls in enumerate(class_names):
        class_dir = data_dir / cls
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Не найдена папка класса: {class_dir}. " "Запустите scripts/download_data.py."
            )
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))

    if not samples:
        raise RuntimeError(f"В {data_dir} не найдено ни одного изображения.")
    return samples


def split_samples(
    samples: list[Sample],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Стратифицированное разбиение на train/val/test без утечек.

    Стратификация по меткам сохраняет баланс классов во всех трёх подвыборках.
    """
    labels = [label for _, label in samples]

    # Сначала отделяем test, затем из оставшегося — val.
    train_val, test = train_test_split(
        samples,
        test_size=test_split,
        stratify=labels,
        random_state=seed,
    )
    train_val_labels = [label for _, label in train_val]
    # Долю val пересчитываем относительно оставшейся части выборки.
    val_relative = val_split / (1.0 - test_split)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        stratify=train_val_labels,
        random_state=seed,
    )
    return train, val, test


class AnimalsDataset(Dataset):
    """Датасет изображений животных: отдаёт (тензор, метка)."""

    def __init__(self, samples: list[Sample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        # convert("RGB") гарантирует три канала даже для серых/палитровых картинок.
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


@dataclass
class DataBundle:
    """Готовые DataLoader-ы и сырые сплиты для удобства."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_samples: list[Sample]
    val_samples: list[Sample]
    test_samples: list[Sample]


def build_dataloaders(cfg: Config) -> DataBundle:
    """Собираем DataLoader-ы для train/val/test согласно конфигу."""
    samples = scan_samples(cfg.data_dir, cfg.class_names)
    train, val, test = split_samples(samples, cfg.val_split, cfg.test_split, cfg.seed)

    train_tf = build_train_transform(cfg)
    eval_tf = build_eval_transform(cfg)

    train_ds = AnimalsDataset(train, train_tf)
    val_ds = AnimalsDataset(val, eval_tf)
    test_ds = AnimalsDataset(test, eval_tf)

    common = {"num_workers": cfg.num_workers, "pin_memory": cfg.device == "cuda"}
    return DataBundle(
        train_loader=DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **common),
        val_loader=DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **common),
        test_loader=DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **common),
        train_samples=train,
        val_samples=val,
        test_samples=test,
    )
