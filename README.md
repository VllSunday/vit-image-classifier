# vit-image-classifier

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
├── checkpoints/                # веса (не в гите)
├── runs/                       # логи TensorBoard (не в гите)
├── requirements.txt
├── Dockerfile
└── README.md
```

## Датасет

[Animal Image Dataset (DOG, CAT and PANDA)](https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda) — 3000 картинок, по 1000 на класс.

Скачать и разложить папки классов в `data/`:

```
data/
├── cats/
├── dogs/
└── panda/
```

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

CUDA-сборку PyTorch под свою GPU ставь с https://pytorch.org/get-started/locally/.

## Запуск

```bash
python -m src.train        # обучение
python -m src.evaluate     # оценка на тесте
python app/app.py          # демо на Gradio
```

## Стек

PyTorch, torchvision, Hugging Face Transformers, Gradio, TensorBoard
