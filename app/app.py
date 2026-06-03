"""Gradio-демо: загрузка изображения -> топ-3 вероятности классов.

Возможности интерфейса:
  - топ-3 класса с вероятностями;
  - время инференса;
  - превью обработанного изображения (что реально видит модель после resize/norm);
  - attention map — куда «смотрел» ViT;
  - предупреждение, когда максимальная вероятность < 0.5;
  - готовые примеры cat / dog / panda.

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
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
CONFIDENCE_THRESHOLD = 0.5


def _example_images() -> list[list[str]]:
    """Список готовых примеров для gr.Examples (если папка с ними существует)."""
    if not EXAMPLES_DIR.is_dir():
        return []
    return [[str(p)] for p in sorted(EXAMPLES_DIR.glob("*.jpg"))]


def build_demo(predictor: Predictor) -> gr.Blocks:
    """Собираем Gradio-интерфейс вокруг предиктора."""

    def classify(image):
        # Пустой ввод (например, после очистки) — ничего не считаем.
        if image is None:
            return {}, None, None, ""

        result = predictor.predict_detailed(image)
        info = f"⏱️ Обработано за **{result['inference_ms']:.0f} мс**"
        if result["max_prob"] < CONFIDENCE_THRESHOLD:
            info += "\n\n⚠️ Модель не уверена (максимальная вероятность < 0.5)."
        return result["probs"], result["preview"], result["attention"], info

    with gr.Blocks(title="ViT: cat / dog / panda") as demo:
        gr.Markdown(
            "# ViT: cat / dog / panda\n"
            "Загрузите изображение животного — модель вернёт топ-3 вероятности, "
            "покажет, что она видит после предобработки, и куда «смотрит» (attention map)."
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Изображение")
                classify_button = gr.Button("Классифицировать", variant="primary")
                examples = _example_images()
                if examples:
                    gr.Examples(examples=examples, inputs=image_input)
            with gr.Column():
                label_output = gr.Label(num_top_classes=3, label="Предсказание (топ-3)")
                info_output = gr.Markdown()
                with gr.Row():
                    preview_output = gr.Image(label="Что видит модель (224×224)")
                    attention_output = gr.Image(label="Attention map")

        outputs = [label_output, preview_output, attention_output, info_output]
        classify_button.click(classify, inputs=image_input, outputs=outputs)
        # Автоматически классифицируем сразу после загрузки картинки.
        image_input.upload(classify, inputs=image_input, outputs=outputs)

    return demo


def main() -> None:
    checkpoint = os.environ.get("CHECKPOINT", DEFAULT_CHECKPOINT)
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"Не найден чекпойнт: {checkpoint}. Сначала обучите модель "
            "(python -m src.train) или задайте путь через переменную CHECKPOINT."
        )

    # eager-внимание + перехват весов нужны для attention map.
    predictor = Predictor(Config(), checkpoint, attn_implementation="eager", capture_attention=True)
    demo = build_demo(predictor)
    # server_name=0.0.0.0 нужен, чтобы демо было доступно из Docker-контейнера.
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
