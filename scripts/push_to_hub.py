"""Публикация обученного чекпойнта на Hugging Face Hub.

Заливает .pt-файл с весами и Model Card в репозиторий модели, чтобы демо можно
было запустить без обучения: достаточно `docker run` — веса подтянутся с Hub.

Перед запуском нужно залогиниться (нужен токен с правом write):
    huggingface-cli login

Запуск:
    python scripts/push_to_hub.py --repo-id USERNAME/vit-cat-dog-panda
    python scripts/push_to_hub.py --repo-id USERNAME/vit-cat-dog-panda \
        --checkpoint checkpoints/linear_probe_best.pt --private
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Корректный вывод кириллицы в Windows-консоли (cp1251 не кодирует часть символов).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Позволяем запускать скрипт напрямую (python scripts/push_to_hub.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.config import Config  # noqa: E402

DEFAULT_CHECKPOINT = "checkpoints/linear_probe_best.pt"


def _build_model_card(repo_id: str, checkpoint_name: str, meta: dict) -> str:
    """Собираем Model Card (README.md репозитория) из метаданных чекпойнта."""
    cfg = Config()
    val_acc = meta.get("val_acc")
    acc_line = f"- **Validation accuracy:** {val_acc:.4f}\n" if val_acc is not None else ""
    classes = ", ".join(meta.get("class_names", cfg.class_names))
    base_model = meta.get("model_name", cfg.model_name)

    return f"""---
license: mit
tags:
  - image-classification
  - vision-transformer
  - pytorch
library_name: pytorch
pipeline_tag: image-classification
base_model: {base_model}
---

# ViT cat / dog / panda

Fine-tuned [`{base_model}`](https://huggingface.co/{base_model}) для классификации
изображений на три класса: **{classes}**.

Это веса для проекта [vit-image-classifier](https://github.com/VllSunday/vit-image-classifier).
Модель = кастомный patch embedding + предобученный энкодер ViT + голова-классификатор
на `[CLS]` токене. Стратегия дообучения: `{meta.get("strategy", cfg.strategy)}`.

{acc_line}- **Классы (порядок индексов):** {classes}
- **Вход:** изображение RGB, нормализация mean=std=0.5, размер 224×224 → тензор `(B, 3, 224, 224)`.

## Файлы

- `{checkpoint_name}` — чекпойнт PyTorch (`torch.save`) со словарём:
  `model_state`, `class_names`, `model_name`, `strategy`, `epoch`, `val_acc`.

## Использование

```python
from huggingface_hub import hf_hub_download
from src.config import Config
from src.inference import Predictor

path = hf_hub_download(repo_id="{repo_id}", filename="{checkpoint_name}")
predictor = Predictor(Config(), path)
print(predictor.predict(image))  # {{"cat": ..., "dog": ..., "panda": ...}}
```

Запуск демо целиком (Gradio) — см. README проекта.
"""


def push(repo_id: str, checkpoint: str, private: bool) -> None:
    """Создаём репозиторий (если нужно) и заливаем веса + Model Card."""
    from huggingface_hub import HfApi

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Не найден чекпойнт: {checkpoint}. Сначала обучите модель (python -m src.train)."
        )

    # Метаданные читаем из чекпойнта, чтобы Model Card была честной.
    meta = torch.load(checkpoint_path, map_location="cpu")
    meta.pop("model_state", None)  # сами веса в карточку не нужны

    api = HfApi()
    print(f"Создаю/проверяю репозиторий {repo_id} (private={private}) ...")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    size_mb = checkpoint_path.stat().st_size / 1e6
    print(f"Заливаю веса: {checkpoint_path.name} ({size_mb:.0f} МБ) ...")
    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=checkpoint_path.name,
        repo_id=repo_id,
        repo_type="model",
    )

    print("Заливаю Model Card (README.md) ...")
    card = _build_model_card(repo_id, checkpoint_path.name, meta)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Готово: https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Опубликовать чекпойнт на Hugging Face Hub.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Репозиторий модели на Hub, например USERNAME/vit-cat-dog-panda.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Путь к чекпойнту (по умолчанию {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Сделать репозиторий приватным (по умолчанию публичный).",
    )
    args = parser.parse_args()

    push(args.repo_id, args.checkpoint, args.private)


if __name__ == "__main__":
    main()
