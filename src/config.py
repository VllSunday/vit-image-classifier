"""Центральная конфигурация проекта.

Все гиперпараметры и пути собраны в одном месте, чтобы остальной код не
хардкодил значения и оставался воспроизводимым. Конфиг намеренно не зависит от
torch на уровне импорта — это позволяет линтеру и лёгким утилитам подтягивать
его без установленного PyTorch.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

# Корень проекта вычисляем относительно этого файла, чтобы пути работали
# независимо от текущей рабочей директории, из которой запущен скрипт.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_device() -> str:
    """Автоопределение устройства: CUDA → Apple MPS → CPU.

    torch импортируется лениво, чтобы модуль можно было использовать и без него.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    # MPS — бэкенд для Apple Silicon (M1/M2/...).
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    """Фиксируем все источники случайности ради воспроизводимости экспериментов."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class Config:
    """Единый набор гиперпараметров и путей проекта."""

    # --- Пути ---
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints"
    runs_dir: Path = PROJECT_ROOT / "runs"

    # --- Целевые классы (порядок фиксирует индексы меток) ---
    class_names: tuple[str, ...] = ("cat", "dog", "panda")

    # --- Изображение и патчи ---
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    # Нормализация под предобученный ViT: процессор google/vit-base-patch16-224
    # использует mean = std = 0.5 по каждому каналу.
    norm_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    norm_std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    # --- Базовая модель (backbone) ---
    model_name: str = "google/vit-base-patch16-224"
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12

    # --- Данные и DataLoader ---
    batch_size: int = 32
    num_workers: int = 4
    val_split: float = 0.15
    test_split: float = 0.15
    # Доля train-выборки для обучения (для экспериментов с урезанными данными).
    train_fraction: float = 1.0

    # --- Стратегия дообучения ---
    # "linear_probe"     — заморозить backbone, учить только голову.
    # "gradual_unfreeze" — разморозить последние N блоков энкодера.
    strategy: str = "linear_probe"
    num_unfrozen_layers: int = 2

    # --- Гиперпараметры обучения ---
    epochs: int = 10
    lr_head: float = 1e-3
    lr_backbone: float = 1e-5
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.1
    grad_clip: float = 1.0  # макс. норма градиента (0 — отключить клиппинг)
    use_amp: bool = True
    early_stopping_patience: int = 5

    # --- Прочее ---
    seed: int = 42
    device: str = field(default_factory=get_device)

    @property
    def num_classes(self) -> int:
        """Количество целевых классов."""
        return len(self.class_names)

    @property
    def num_patches(self) -> int:
        """Число патчей на изображении: (224 / 16)^2 = 196."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def seq_length(self) -> int:
        """Длина последовательности токенов: патчи + [CLS]."""
        return self.num_patches + 1

    def ensure_dirs(self) -> None:
        """Создаём служебные директории, если их ещё нет."""
        for directory in (self.data_dir, self.checkpoints_dir, self.runs_dir):
            directory.mkdir(parents=True, exist_ok=True)
