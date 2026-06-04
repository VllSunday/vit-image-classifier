"""Пайплайн предобработки изображений.

Базовое преобразование (eval): приведение к 224x224 и нормализация по трём
каналам RGB под статистику предобученного ViT. Тренировочное преобразование
дополнительно включает аугментации для устойчивости модели.
"""

from __future__ import annotations

from torchvision import transforms

from src.config import Config


def build_eval_transform(cfg: Config) -> transforms.Compose:
    """Преобразование для валидации/теста/инференса.

    resize до 224x224 и нормализация трёх каналов.
    """
    return transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),  # PIL -> тензор [0, 1], формат (C, H, W)
            transforms.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
        ]
    )


def build_train_transform(cfg: Config) -> transforms.Compose:
    """Преобразование для обучения: базовый пайплайн + аугментации.

    Аугментации умеренные — животные не должны переворачиваться вверх ногами,
    поэтому ограничиваемся горизонтальным отражением, лёгким поворотом, случайным
    кадрированием и небольшим изменением цвета.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
        ]
    )
