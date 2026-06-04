# vit-image-classifier

[![CI](https://github.com/VllSunday/vit-image-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/VllSunday/vit-image-classifier/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Fine-tuning a Vision Transformer (`google/vit-base-patch16-224`) to classify images into three classes: **cat / dog / panda**.

The project reimplements the core ViT building blocks (patch embedding, `[CLS]` token, positional encoding and the classification head) on top of a pretrained Transformer encoder, and ships a small Gradio demo for inference.

## Quick start

No training and no dataset required — try the demo straight away. The fine-tuned weights are pulled from the [Hugging Face Hub](https://huggingface.co/A11Sunday/vit-cat-dog-panda) on first run:

```bash
docker build -t vit-classifier .
docker run -p 7860:7860 vit-classifier      # then open http://localhost:7860
```

Upload an image (or pick a built-in example) and you get the top-3 class probabilities, a preview of what the model actually sees after preprocessing, and an attention rollout that shows how the focus narrows onto the object from the early layers to the full encoder.

## Features

- Custom preprocessing pipeline (resize to 224×224 + RGB normalization)
- Hand-written patch embedding (16×16 patches), `[CLS]` token and positional encoding
- Transfer learning on a pretrained `vit-base-patch16-224` backbone
- Training with mixed precision, cosine LR schedule with warmup and TensorBoard logging
- Evaluation with accuracy, per-class F1 and a confusion matrix
- Gradio web demo: top-3 probabilities, a preview of the model's input, and an attention rollout (early layers → full encoder) visualizing where the model looks

## Project structure

```
vit-image-classifier/
├── data/                       # dataset (not tracked)
├── scripts/
│   ├── download_data.py        # download dataset from Kaggle
│   └── run_experiments.py      # train + evaluate both strategies, build results table
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
│   ├── evaluate.py             # metrics on the test set
│   └── inference.py            # single-image prediction (used by the demo)
├── app/
│   └── app.py                  # Gradio demo
├── reports/                    # confusion matrices + results table
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
# Train (defaults to linear probing; see --help for all options)
python -m src.train --strategy linear_probe --epochs 5
python -m src.train --strategy gradual_unfreeze --unfrozen-layers 4 --epochs 5

# Evaluate a checkpoint on the test set (accuracy, per-class F1, confusion matrix)
python -m src.evaluate --checkpoint checkpoints/linear_probe_best.pt

# Reproduce the full experiment comparison
python scripts/run_experiments.py

# Launch the Gradio demo
python app/app.py
```

The demo loads `checkpoints/linear_probe_best.pt` if present; otherwise it downloads
the weights from the Hugging Face Hub (`A11Sunday/vit-cat-dog-panda`) and caches them,
so **you don't need to train anything to try it**. Override the local path with the
`CHECKPOINT` environment variable or the Hub repo with `HF_REPO_ID`. The model input
strictly follows the standard `(Batch_Size, 3, 224, 224)` format.

### Pretrained weights

The fine-tuned checkpoint lives on the Hugging Face Hub:
[`A11Sunday/vit-cat-dog-panda`](https://huggingface.co/A11Sunday/vit-cat-dog-panda).
It is fetched automatically on first run. To publish a (re)trained checkpoint yourself:

```bash
hf auth login                                              # token with write access
python scripts/push_to_hub.py --repo-id A11Sunday/vit-cat-dog-panda
```

Training logs go to TensorBoard:

```bash
tensorboard --logdir runs
```

### Docker

```bash
docker build -t vit-classifier .
docker run -p 7860:7860 vit-classifier      # open http://localhost:7860
```

No training and no local weights required: on first launch the container downloads the
checkpoint from the Hugging Face Hub. Point it at a different Hub repo with
`-e HF_REPO_ID=...`, or mount your own checkpoint and set `-e CHECKPOINT=...`.

## Results

Two fine-tuning strategies are compared, each in two data regimes — full training
set (~2100 images) and a small one (~150 images) — plus a **from-scratch baseline**
(identical architecture, no pretrained weights). Evaluation is always on the same
held-out test set (450 images, 150 per class).

| Experiment | Pretrained | Trainable params | Epochs to best | Test accuracy | Macro F1 |
|---|:---:|---:|---:|---:|---:|
| Linear probe (full data) | ✅ | 2,307 | 3 | **0.9956** | 0.9955 |
| Gradual unfreeze (full data) | ✅ | 28,355,331 | 1 | 0.9933 | 0.9933 |
| Linear probe (small train) | ✅ | 2,307 | 3 | 0.9933 | 0.9933 |
| Gradual unfreeze (small train) | ✅ | 28,355,331 | 3 | 0.9933 | 0.9933 |
| From scratch (no pretraining) | ❌ | 85,800,963 | 17 | 0.6689 | 0.6647 |

<img src="reports/confusion_matrix_linear_probe_full.png" width="420" alt="Confusion matrix (linear probe, full data)">

**Takeaways**

- The pretrained ViT already separates cat / dog / panda almost perfectly (ImageNet,
  which the backbone was trained on, already contains cats, dogs and "giant panda"),
  so all transfer-learning strategies land at ~99% test accuracy.
- **Transfer learning is doing the heavy lifting.** The same architecture trained
  *from scratch* on the same data reaches only **~67%** — a ~32-point gap that is the
  whole point of fine-tuning a pretrained backbone instead of training one.
- Because the task is easy for a pretrained model, **linear probing wins on efficiency**:
  it matches full fine-tuning while training only **2,307** parameters vs **28M**.
- With a small training set (~50 images per class) the pretrained model still reaches
  ~99%, while the from-scratch model is data-starved (ViTs need a lot of data).

Confusion matrices for every run are saved under [`reports/`](reports/).

## Tech stack

PyTorch · torchvision · Hugging Face Transformers · Gradio · TensorBoard · scikit-learn
