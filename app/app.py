"""Gradio-демо: загрузка изображения -> топ-3 вероятности классов.

Возможности интерфейса:
  - топ-3 класса с вероятностями;
  - время инференса;
  - превью обработанного изображения (что реально видит модель после resize/norm);
  - серия attention rollout-карт по нарастающей глубине слоёв — как ViT от слоя
    к слою фокусируется на объекте;
  - предупреждение, когда максимальная вероятность < 0.68;
  - готовые примеры cat / dog / panda.

Чекпойнт берётся из переменной окружения CHECKPOINT (по умолчанию — лучшая
linear-probe модель). Если локального файла нет, веса автоматически качаются с
Hugging Face Hub (репозиторий из HF_REPO_ID, по умолчанию — DEFAULT_HF_REPO),
поэтому демо запускается «из коробки» без обучения.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Позволяем запускать как `python app/app.py` из корня репозитория.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402

from src.config import Config  # noqa: E402
from src.inference import Predictor, resolve_checkpoint  # noqa: E402

DEFAULT_CHECKPOINT = "checkpoints/linear_probe_best.pt"
# Репозиторий с весами на Hugging Face Hub — отсюда скачиваем, если локально нет.
DEFAULT_HF_REPO = "A11Sunday/vit-cat-dog-panda"
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
# Ниже этого порога максимальной вероятности показываем «модель не уверена».
CONFIDENCE_THRESHOLD = 0.68


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
            info += (
                f"\n\n⚠️ Модель не уверена (максимальная вероятность < {CONFIDENCE_THRESHOLD:.0%})."
            )
        return result["probs"], result["preview"], result["attention_maps"], info

    with gr.Blocks(title="ViT: cat / dog / panda") as demo:
        gr.Markdown(
            "# ViT: cat / dog / panda\n"
            "Загрузите изображение животного — модель вернёт топ-3 вероятности, "
            "покажет, что она видит после предобработки, и как от слоя к слою "
            "фокусируется на объекте (attention rollout по нарастающей глубине)."
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
                preview_output = gr.Image(label="Что видит модель (224×224)")
                attention_gallery = gr.Gallery(
                    label="Attention rollout: ранние слои → все слои (фокус сжимается на объект)",
                    columns=2,
                    height="auto",
                    object_fit="contain",
                )

        outputs = [label_output, preview_output, attention_gallery, info_output]
        classify_button.click(classify, inputs=image_input, outputs=outputs)
        # Автоматически классифицируем сразу после загрузки картинки.
        image_input.upload(classify, inputs=image_input, outputs=outputs)

    return demo


def main() -> None:
    checkpoint = os.environ.get("CHECKPOINT", DEFAULT_CHECKPOINT)
    repo_id = os.environ.get("HF_REPO_ID", DEFAULT_HF_REPO)
    # Локальный путь, иначе скачиваем веса с Hugging Face Hub (кешируется).
    checkpoint = resolve_checkpoint(checkpoint, repo_id=repo_id)

    # eager-внимание + перехват весов нужны для attention rollout-карт по слоям.
    predictor = Predictor(Config(), checkpoint, attn_implementation="eager", capture_attention=True)
    demo = build_demo(predictor)
    # server_name=0.0.0.0 нужен, чтобы демо было доступно из Docker-контейнера.
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
