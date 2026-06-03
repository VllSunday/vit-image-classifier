# vit-image-classifier

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
├── checkpoints/                # model weights (not tracked)
├── runs/                       # TensorBoard logs (not tracked)
├── requirements.txt
├── Dockerfile
└── README.md
```

## Dataset

[Animal Image Dataset (DOG, CAT and PANDA)](https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda) — 3000 images, 1000 per class.

Download it and place the class folders under `data/`:

```
data/
├── cats/
├── dogs/
└── panda/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

> Install a CUDA-enabled PyTorch build matching your GPU from https://pytorch.org/get-started/locally/.

## Usage

```bash
python -m src.train        # train the model
python -m src.evaluate     # evaluate on the test set
python app/app.py          # launch the Gradio demo
```

## Tech stack

PyTorch · torchvision · Hugging Face Transformers · Gradio · TensorBoard
