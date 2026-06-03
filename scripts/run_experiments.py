"""Прогон и сравнение экспериментов дообучения.

Обучает обе стратегии (linear probe / gradual unfreeze) в двух режимах данных
(полные данные и урезанный train), оценивает каждую модель на тесте, сохраняет
confusion matrix и сводную таблицу результатов в reports/.

Запуск:
    python scripts/run_experiments.py
"""

from __future__ import annotations

import dataclasses
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.evaluate import evaluate_checkpoint  # noqa: E402
from src.models.vit import ViTClassifier  # noqa: E402
from src.train import train  # noqa: E402

REPORT_DIR = Path("reports")

# Описание экспериментов: slug используется в именах чекпойнтов и картинок.
EXPERIMENTS = [
    {
        "slug": "linear_probe_full",
        "name": "Linear probe (full data)",
        "overrides": {"strategy": "linear_probe", "epochs": 5, "train_fraction": 1.0},
    },
    {
        "slug": "gradual_unfreeze_full",
        "name": "Gradual unfreeze (full data)",
        "overrides": {
            "strategy": "gradual_unfreeze",
            "num_unfrozen_layers": 4,
            "epochs": 5,
            "train_fraction": 1.0,
        },
    },
    {
        "slug": "linear_probe_small",
        "name": "Linear probe (small train)",
        "overrides": {"strategy": "linear_probe", "epochs": 12, "train_fraction": 0.07},
    },
    {
        "slug": "gradual_unfreeze_small",
        "name": "Gradual unfreeze (small train)",
        "overrides": {
            "strategy": "gradual_unfreeze",
            "num_unfrozen_layers": 4,
            "epochs": 12,
            "train_fraction": 0.07,
        },
    },
    {
        # Контрольный эксперимент: та же архитектура, но БЕЗ предобученных весов.
        # Показывает, насколько результат держится на transfer learning.
        "slug": "from_scratch_full",
        "name": "From scratch (no pretraining)",
        "pretrained": False,
        "overrides": {
            "strategy": "full",
            "epochs": 20,
            "train_fraction": 1.0,
            "lr_head": 3e-4,
            "lr_backbone": 3e-4,
            "early_stopping_patience": 20,
        },
    },
]


def _epochs_to_best(history: list[dict]) -> int:
    """Номер эпохи, на которой достигнута лучшая val accuracy."""
    best = max(history, key=lambda row: row["val_acc"])
    return best["epoch"]


def run() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for exp in EXPERIMENTS:
        print("\n" + "=" * 70)
        print(f"Эксперимент: {exp['name']}")
        print("=" * 70)

        cfg = dataclasses.replace(Config(), **exp["overrides"])

        # Осмысленное имя рана для TensorBoard (вместо стратегия+timestamp).
        result = train(cfg, pretrained=exp.get("pretrained", True), run_name=exp["slug"])

        # Копируем лучший чекпойнт под уникальным именем эксперимента.
        unique_ckpt = cfg.checkpoints_dir / f"{exp['slug']}.pt"
        shutil.copy(result["best_checkpoint"], unique_ckpt)

        metrics = evaluate_checkpoint(cfg, unique_ckpt, REPORT_DIR)

        trainable = ViTClassifier(cfg, pretrained=False).num_trainable_parameters()
        rows.append(
            {
                "name": exp["name"],
                "trainable": trainable,
                "epochs_to_best": _epochs_to_best(result["history"]),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "cm": Path(metrics["confusion_matrix_png"]).name,
            }
        )

    _write_results_table(rows)


def _write_results_table(rows: list[dict]) -> None:
    """Пишем сводную таблицу в reports/results.md и в консоль."""
    lines = [
        "| Experiment | Trainable params | Epochs to best | Test accuracy | Macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['trainable']:,} | {r['epochs_to_best']} | "
            f"{r['accuracy']:.4f} | {r['macro_f1']:.4f} |"
        )
    table = "\n".join(lines)

    (REPORT_DIR / "results.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nТаблица сохранена в {REPORT_DIR / 'results.md'}")


if __name__ == "__main__":
    run()
