# vit-image-classifier

[![CI](https://github.com/VllSunday/vit-image-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/VllSunday/vit-image-classifier/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Fine-tuning a Vision Transformer (`google/vit-base-patch16-224`) to classify images into three classes: **cat / dog / panda**.

The project reimplements the core ViT building blocks (patch embedding, `[CLS]` token, positional encoding and the classification head) on top of a pretrained Transformer encoder, and ships a small Gradio demo for inference.

## Features

- Custom preprocessing pipeline (resize to 224×224 + RGB normalization)
- Hand-written patch embedding (16×16 patches), `[CLS]` token and positional encoding
- Transfer learning on a pretrained `vit-base-patch16-224` backbone
- Training with mixed precision, cosine LR schedule with warmup and TensorBoard logging
- Evaluation with accuracy, per-class F1 and a confusion matrix
- Gradio web demo (upload an image → top-3 class probabilities)

## Project structure

```
vit-image-classifier/
├── data/                       # dataset (not tracked)
├── scripts/
│   └── download_data.py        # download dataset from Kaggle
├── src/
│   ├── data/
│   │   ├── dataset.py          # loading + train/val/test split
│   │   └── transforms.py       # preprocessing + augmentations
│   ├── models/
│   │   ├── patch_embedding.py  # patches + CLS token + positional encoding
│   │   ├── vit.py              # full model: custom parts + backbone + head
│   │   └── encoder_from_scratch.py  # standalone Transformer encoder
│   ├── config.py               # hyperparameters
│   ├── train.py                # training loop
│   └── evaluate.py             # metrics on the test set
├── app/
│   └── app.py                  # Gradio demo
├── tests/                      # pytest suite
├── checkpoints/                # model weights (not tracked)
├── runs/                       # TensorBoard logs (not tracked)
├── .github/workflows/ci.yml    # lint + tests on push / PR
├── pyproject.toml              # tooling config (ruff, black, pytest)
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
└── README.md
```

## Dataset

[Animal Image Dataset (DOG, CAT and PANDA)](https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda) — 3000 images, 1000 per class.

Download it with the helper script (requires a Kaggle API token, see [kagglehub docs](https://github.com/Kaggle/kagglehub#authenticate)):

```bash
python scripts/download_data.py
```

The script downloads the dataset and normalizes it into:

```
data/
├── cat/
├── dog/
└── panda/
```

> No Kaggle token? Download the archive manually from the link above and lay the
> images out under `data/cat`, `data/dog`, `data/panda`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# .venv/bin/activate          # Linux / macOS
```

PyTorch is installed separately because the build depends on your hardware:

```bash
# NVIDIA GPU (CUDA 13.2)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> Pick the build matching your machine from https://pytorch.org/get-started/locally/.

Then install the rest of the dependencies:

```bash
pip install -r requirements.txt          # runtime only
pip install -r requirements-dev.txt      # + linter / formatter / tests
```

## Development

```bash
pre-commit install     # run ruff + black on every commit
ruff check .           # lint
black .                # format
pytest                 # run tests
```

The same checks run in CI (GitHub Actions) on every push and pull request.

## Usage

```bash
python -m src.train        # train the model
python -m src.evaluate     # evaluate on the test set
python app/app.py          # launch the Gradio demo
```

## Tech stack

PyTorch · torchvision · Hugging Face Transformers · Gradio · TensorBoard
