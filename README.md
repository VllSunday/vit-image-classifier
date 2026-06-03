# vit-image-classifier

[![CI](https://github.com/VllSunday/vit-image-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/VllSunday/vit-image-classifier/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Дообучение Vision Transformer (`google/vit-base-patch16-224`) на три класса:
cat / dog / panda.

Базовые кубики ViT (patch embedding, токен `[CLS]`, позиционные эмбеддинги
и классификационная голова) написаны руками поверх предобученного энкодера.
К проекту приложено небольшое демо на Gradio.

## Что умеет

- препроцессинг (ресайз 224×224 + нормализация RGB)
- patch embedding руками (патчи 16×16), токен `[CLS]`, позиционные эмбеддинги
- transfer learning на предобученном vit-base-patch16-224
- обучение с mixed precision, косинусным LR с warmup и логами в TensorBoard
- оценка: accuracy, F1 по классам, confusion matrix
- демо на Gradio (загрузил картинку → топ-3 вероятности)

## Структура

```
vit-image-classifier/
├── data/                       # датасет (не в гите)
├── scripts/
│   └── download_data.py        # скачать датасет с Kaggle
├── src/
│   ├── data/
│   │   ├── dataset.py          # загрузка + сплит train/val/test
│   │   └── transforms.py       # препроцессинг + аугментации
│   ├── models/
│   │   ├── patch_embedding.py  # патчи + CLS + позиционные
│   │   ├── vit.py              # вся модель: свои части + backbone + голова
│   │   └── encoder_from_scratch.py  # отдельный Transformer-энкодер
│   ├── config.py               # гиперпараметры
│   ├── train.py                # цикл обучения
│   └── evaluate.py             # метрики на тесте
├── app/
│   └── app.py                  # демо на Gradio
├── tests/                      # pytest
├── checkpoints/                # веса (не в гите)
├── runs/                       # логи TensorBoard (не в гите)
├── .github/workflows/ci.yml    # линт + тесты на push / PR
├── pyproject.toml              # конфиг тулинга (ruff, black, pytest)
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
└── README.md
```

## Датасет

[Animal Image Dataset (DOG, CAT and PANDA)](https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda) — 3000 картинок, по 1000 на класс.

Скачать скриптом (нужен токен Kaggle API, см. [доку kagglehub](https://github.com/Kaggle/kagglehub#authenticate)):

```bash
python scripts/download_data.py
```

Скрипт раскладывает картинки в `data/cat`, `data/dog`, `data/panda`. Если токена
Kaggle нет — скачай архив по ссылке руками и разложи так же.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# .venv/bin/activate          # Linux / macOS
```

PyTorch ставится отдельно, сборка зависит от железа:

```bash
# NVIDIA GPU (CUDA 13.2)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
# только CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Дальше остальное:

```bash
pip install -r requirements.txt          # только runtime
pip install -r requirements-dev.txt      # + линтер / форматтер / тесты
```

## Разработка

```bash
pre-commit install     # ruff + black на каждый коммит
ruff check .
black .
pytest
```

Те же проверки гоняются в CI на каждый push и PR.

## Запуск

```bash
python -m src.train        # обучение
python -m src.evaluate     # оценка на тесте
python app/app.py          # демо на Gradio
```

## Стек

PyTorch, torchvision, Hugging Face Transformers, Gradio, TensorBoard
