"""Обучение ViT-классификатора.

Поддерживает обе стратегии дообучения (linear_probe / gradual_unfreeze),
mixed precision (AMP) на CUDA, свой warmup+cosine планировщик LR,
дискриминативный learning rate, клиппинг градиентов, чекпойнты лучшей модели,
логирование в TensorBoard и раннюю остановку.

Запуск:
    python -m src.train
    python -m src.train --strategy gradual_unfreeze --unfrozen-layers 4 --epochs 15
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import Config, set_seed
from src.data.dataset import build_dataloaders
from src.models.vit import ViTClassifier
from src.training.scheduler import WarmupCosineScheduler


def build_optimizer(model: ViTClassifier, cfg: Config) -> AdamW:
    """AdamW с дискриминативным LR (группы задаёт сама модель)."""
    return AdamW(model.param_groups(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> int:
    """Число верных предсказаний в батче."""
    return int((logits.argmax(dim=1) == labels).sum().item())


def run_train_epoch(
    model: ViTClassifier,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: WarmupCosineScheduler,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    cfg: Config,
    epoch: int,
) -> tuple[float, float]:
    """Одна эпоха обучения. Возвращает (средний loss, accuracy)."""
    model.train()
    device = cfg.device
    use_amp = cfg.use_amp and device == "cuda"
    amp_device = "cuda" if device == "cuda" else "cpu"

    total_loss, total_correct, total_seen = 0.0, 0, 0
    progress = tqdm(loader, desc=f"epoch {epoch} [train]", leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        # backward + шаг оптимизатора (с поддержкой AMP).
        if use_amp:
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        # Шаг планировщика LR — после шага оптимизатора.
        scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += _accuracy(logits, labels)
        total_seen += batch_size
        progress.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def evaluate(
    model: ViTClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    cfg: Config,
) -> tuple[float, float]:
    """Оценка на валидации. Возвращает (средний loss, accuracy)."""
    model.eval()
    device = cfg.device
    use_amp = cfg.use_amp and device == "cuda"
    amp_device = "cuda" if device == "cuda" else "cpu"

    total_loss, total_correct, total_seen = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += _accuracy(logits, labels)
        total_seen += batch_size

    return total_loss / total_seen, total_correct / total_seen


def save_checkpoint(
    path: Path, model: ViTClassifier, cfg: Config, epoch: int, val_acc: float
) -> None:
    """Сохраняем веса модели и метаданные, нужные для инференса."""
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": list(cfg.class_names),
            "model_name": cfg.model_name,
            "strategy": cfg.strategy,
            "epoch": epoch,
            "val_acc": val_acc,
        },
        path,
    )


def train(cfg: Config, pretrained: bool = True) -> dict:
    """Полный цикл обучения. Возвращает историю метрик и путь к лучшей модели."""
    set_seed(cfg.seed)
    cfg.ensure_dirs()
    device = cfg.device

    data = build_dataloaders(cfg)
    model = ViTClassifier(cfg, pretrained=pretrained).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg)

    # Планировщик считаем по числу шагов (батчей) за всё обучение.
    steps_per_epoch = len(data.train_loader)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler(enabled=cfg.use_amp and device == "cuda")

    run_name = f"{cfg.strategy}_{datetime.now():%Y%m%d_%H%M%S}"
    writer = SummaryWriter(cfg.runs_dir / run_name)
    best_path = cfg.checkpoints_dir / f"{cfg.strategy}_best.pt"

    print(
        f"Старт обучения: strategy={cfg.strategy}, device={device}, "
        f"обучаемых параметров={model.num_trainable_parameters():,}"
    )

    history: list[dict] = []
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_train_epoch(
            model, data.train_loader, optimizer, scheduler, criterion, scaler, cfg, epoch
        )
        val_loss, val_acc = evaluate(model, data.val_loader, criterion, cfg)

        # Логирование в TensorBoard.
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        print(
            f"epoch {epoch:02d}/{cfg.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        # Сохраняем лучшую модель и считаем шаги без улучшения (ранняя остановка).
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, cfg, epoch, val_acc)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stopping_patience:
                print(f"Ранняя остановка на эпохе {epoch} (нет улучшения val acc).")
                break

    writer.close()
    print(f"Готово. Лучшая val accuracy: {best_val_acc:.4f}. Чекпойнт: {best_path}")

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "best_checkpoint": str(best_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение ViT-классификатора cat/dog/panda.")
    parser.add_argument(
        "--strategy",
        choices=["linear_probe", "gradual_unfreeze"],
        help="Стратегия дообучения.",
    )
    parser.add_argument("--epochs", type=int, help="Число эпох.")
    parser.add_argument("--batch-size", type=int, help="Размер батча.")
    parser.add_argument("--lr-head", type=float, help="LR для головы.")
    parser.add_argument("--lr-backbone", type=float, help="LR для размороженного backbone.")
    parser.add_argument("--unfrozen-layers", type=int, help="Сколько блоков энкодера разморозить.")
    parser.add_argument("--seed", type=int, help="Сид.")
    parser.add_argument("--no-amp", action="store_true", help="Отключить mixed precision.")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> Config:
    """Накладываем переданные аргументы CLI поверх значений по умолчанию."""
    overrides: dict = {}
    if args.strategy is not None:
        overrides["strategy"] = args.strategy
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr_head is not None:
        overrides["lr_head"] = args.lr_head
    if args.lr_backbone is not None:
        overrides["lr_backbone"] = args.lr_backbone
    if args.unfrozen_layers is not None:
        overrides["num_unfrozen_layers"] = args.unfrozen_layers
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.no_amp:
        overrides["use_amp"] = False
    return dataclasses.replace(Config(), **overrides)


def main() -> None:
    cfg = _config_from_args(_parse_args())
    train(cfg)


if __name__ == "__main__":
    main()
