"""Тесты планировщика LR: форма расписания и дискриминативный LR."""

from __future__ import annotations

import pytest
import torch

from src.scheduler import WarmupCosineScheduler


def _make_optimizer() -> torch.optim.Optimizer:
    # Две группы параметров с разными базовыми LR (как голова и backbone).
    p_head = torch.nn.Parameter(torch.zeros(1))
    p_backbone = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD(
        [
            {"params": [p_head], "lr": 1.0},
            {"params": [p_backbone], "lr": 0.1},
        ]
    )


def test_warmup_then_decay_shape() -> None:
    opt = _make_optimizer()
    sched = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)

    lrs = [sched.get_last_lr()[0]]  # начальный LR (шаг 0)
    for _ in range(100):
        sched.step()
        lrs.append(sched.get_last_lr()[0])

    # В начале warm-up LR близок к нулю, на пике (шаг 10) равен базовому.
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[10] == pytest.approx(1.0)

    # Разогрев строго возрастает.
    warmup_segment = lrs[1:11]
    assert all(b > a for a, b in zip(warmup_segment, warmup_segment[1:], strict=False))

    # После пика — затухание до ~0 к концу.
    assert lrs[100] == pytest.approx(0.0, abs=1e-6)
    assert lrs[50] < lrs[10]


def test_discriminative_lr_ratio_preserved() -> None:
    opt = _make_optimizer()
    sched = WarmupCosineScheduler(opt, warmup_steps=5, total_steps=50)

    for _ in range(7):  # выходим из warm-up, LR обеих групп ненулевой
        sched.step()

    lr_head, lr_backbone = sched.get_last_lr()
    # Базовое отношение 1.0 : 0.1 сохраняется на любом шаге расписания.
    assert lr_head / lr_backbone == pytest.approx(10.0)


def test_min_lr_ratio_floor() -> None:
    opt = _make_optimizer()
    sched = WarmupCosineScheduler(opt, warmup_steps=0, total_steps=20, min_lr_ratio=0.1)

    for _ in range(20):
        sched.step()

    # В конце LR не падает ниже min_lr_ratio * base_lr.
    assert sched.get_last_lr()[0] == pytest.approx(0.1, abs=1e-6)
