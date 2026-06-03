"""Оценка обученной модели на тестовой выборке.

Считает accuracy, per-class precision/recall/F1 и строит confusion matrix.

Запуск:
    python -m src.evaluate --checkpoint checkpoints/linear_probe_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Без графического бэкенда — чтобы работало в headless-окружении и CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader  # noqa: E402

from src.config import Config  # noqa: E402
from src.data.dataset import build_dataloaders  # noqa: E402
from src.models.vit import ViTClassifier  # noqa: E402


def load_model(cfg: Config, checkpoint_path: Path) -> tuple[ViTClassifier, dict]:
    """Восстанавливаем модель из чекпойнта (архитектуру не качаем заново)."""
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    model = ViTClassifier(cfg, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(cfg.device).eval()
    return model, checkpoint


@torch.no_grad()
def predict(model: ViTClassifier, loader: DataLoader, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Прогоняем модель по загрузчику и собираем истинные/предсказанные метки."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in loader:
        images = images.to(cfg.device, non_blocking=True)
        logits = model(images)
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(labels.tolist())
    return np.array(y_true), np.array(y_pred)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: tuple[str, ...]
) -> tuple[float, dict, np.ndarray]:
    """Возвращаем (accuracy, classification_report как dict, confusion matrix)."""
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return accuracy, report, matrix


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: tuple[str, ...],
    out_path: Path,
    title: str = "Confusion matrix",
) -> Path:
    """Рисуем confusion matrix и сохраняем в PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Подписываем числа в ячейках, выбирая контрастный цвет текста.
    threshold = matrix.max() / 2 if matrix.max() > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                int(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def print_report(accuracy: float, report: dict, class_names: tuple[str, ...]) -> None:
    """Печатаем краткий отчёт в консоль."""
    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Macro F1:      {report['macro avg']['f1-score']:.4f}\n")
    print(f"{'class':<10}{'precision':>10}{'recall':>10}{'f1':>10}")
    for cls in class_names:
        row = report[cls]
        print(f"{cls:<10}{row['precision']:>10.4f}{row['recall']:>10.4f}{row['f1-score']:>10.4f}")


def evaluate_checkpoint(cfg: Config, checkpoint_path: Path, report_dir: Path) -> dict:
    """Полная оценка чекпойнта на тестовой выборке + сохранение confusion matrix."""
    data = build_dataloaders(cfg)
    model, checkpoint = load_model(cfg, checkpoint_path)

    y_true, y_pred = predict(model, data.test_loader, cfg)
    accuracy, report, matrix = compute_metrics(y_true, y_pred, cfg.class_names)

    print_report(accuracy, report, cfg.class_names)

    cm_path = report_dir / f"confusion_matrix_{checkpoint_path.stem}.png"
    plot_confusion_matrix(matrix, cfg.class_names, cm_path, title=checkpoint_path.stem)

    return {
        "checkpoint": str(checkpoint_path),
        "strategy": checkpoint.get("strategy"),
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "report": report,
        "confusion_matrix_png": str(cm_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценка ViT-классификатора на тесте.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Путь к .pt чекпойнту.")
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports"), help="Куда сохранять отчёты."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    evaluate_checkpoint(Config(), args.checkpoint, args.report_dir)


if __name__ == "__main__":
    main()
