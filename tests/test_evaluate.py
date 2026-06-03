"""Тесты метрик и построения confusion matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluate import compute_metrics, plot_confusion_matrix

CLASS_NAMES = ("cat", "dog", "panda")


def test_perfect_prediction_metrics() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    accuracy, report, matrix = compute_metrics(y_true, y_pred, CLASS_NAMES)

    assert accuracy == 1.0
    assert report["macro avg"]["f1-score"] == 1.0
    # Идеальная диагональ.
    assert np.array_equal(matrix, np.eye(3, dtype=matrix.dtype) * 2)


def test_metrics_on_known_errors() -> None:
    # Один cat предсказан как dog.
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2])
    accuracy, report, matrix = compute_metrics(y_true, y_pred, CLASS_NAMES)

    assert accuracy == 0.75
    # cat: recall 0.5 (1 из 2 верно).
    assert report["cat"]["recall"] == 0.5
    # Матрица 3x3.
    assert matrix.shape == (3, 3)


def test_plot_confusion_matrix_saves_png(tmp_path: Path) -> None:
    matrix = np.array([[5, 0, 0], [0, 4, 1], [0, 0, 6]])
    out = tmp_path / "cm.png"
    result = plot_confusion_matrix(matrix, CLASS_NAMES, out, title="test")

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0
