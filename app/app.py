"""Gradio-демо: загрузка изображения -> топ-3 вероятности классов.

Чекпойнт берётся из переменной окружения CHECKPOINT (по умолчанию — лучшая
linear-probe модель). Перед запуском нужно обучить модель (python -m src.train).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Позволяем запускать как `python app/app.py` из корня репозитория.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402

from src.config import Config  # noqa: E402
from src.inference import Predictor  # noqa: E402

DEFAULT_CHECKPOINT = "checkpoints/linear_probe_best.pt"


def build_demo(predictor: Predictor) -> gr.Interface:
    """Собираем Gradio-интерфейс вокруг предиктора."""
    return gr.Interface(
        fn=predictor.predict,
        inputs=gr.Image(type="pil", label="Изображение"),
        # gr.Label с num_top_classes=3 покажет топ-3 класса с вероятностями.
        outputs=gr.Label(num_top_classes=3, label="Предсказание"),
        title="ViT: cat / dog / panda",
        description="Загрузите изображение животного — модель вернёт топ-3 вероятности.",
    )


def main() -> None:
    checkpoint = os.environ.get("CHECKPOINT", DEFAULT_CHECKPOINT)
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"Не найден чекпойнт: {checkpoint}. Сначала обучите модель "
            "(python -m src.train) или задайте путь через переменную CHECKPOINT."
        )

    predictor = Predictor(Config(), checkpoint)
    demo = build_demo(predictor)
    # server_name=0.0.0.0 нужен, чтобы демо было доступно из Docker-контейнера.
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
