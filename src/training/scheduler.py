"""Свой планировщик learning rate: линейный warm-up + косинусное затухание.

Реализован вручную (без torch.optim.lr_scheduler), чтобы наглядно показать
логику. Расписание задаёт общий множитель в диапазоне [0, 1], на который
домножается базовый LR каждой группы параметров. Это корректно работает с
дискриминативным LR (у головы и backbone разные базовые значения).
"""

from __future__ import annotations

import math

from torch.optim import Optimizer


class WarmupCosineScheduler:
    """Линейный разогрев до базового LR, затем косинусное затухание до min_lr."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps должно быть положительным.")

        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = int(total_steps)
        self.min_lr_ratio = float(min_lr_ratio)

        # Запоминаем базовый LR каждой группы — расписание масштабирует именно их.
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        self._step_count = 0
        # Выставляем стартовый LR (в начале warm-up он близок к нулю).
        self._apply(self._factor(0))

    def _factor(self, step: int) -> float:
        """Множитель LR на заданном шаге."""
        # Фаза разогрева: линейный рост от 0 до 1.
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return step / self.warmup_steps

        # Фаза затухания: косинус от 1 до min_lr_ratio.
        denom = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / denom)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _apply(self, factor: float) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * factor

    def step(self) -> None:
        """Шаг расписания — вызывать после optimizer.step()."""
        self._step_count += 1
        self._apply(self._factor(self._step_count))

    def get_last_lr(self) -> list[float]:
        """Текущий LR каждой группы параметров."""
        return [group["lr"] for group in self.optimizer.param_groups]
