"""Тесты patch embedding: формы, [CLS] токен и эквивалентность свёртке."""

from __future__ import annotations

import torch
from torch import nn

from src.config import Config
from src.models.patch_embedding import PatchEmbedding


def test_output_shape() -> None:
    cfg = Config()
    emb = PatchEmbedding(cfg)
    x = torch.randn(2, cfg.in_channels, cfg.image_size, cfg.image_size)
    out = emb(x)
    # (B, num_patches + 1, embed_dim) = (2, 197, 768)
    assert out.shape == (2, cfg.seq_length, cfg.embed_dim)


def test_wrong_input_size_raises() -> None:
    cfg = Config()
    emb = PatchEmbedding(cfg)
    with torch.no_grad():
        bad = torch.randn(1, 3, 128, 128)
        try:
            emb(bad)
        except ValueError:
            return
    raise AssertionError("Ожидалась ошибка на неверном размере входа.")


def test_cls_token_is_prepended() -> None:
    cfg = Config()
    emb = PatchEmbedding(cfg)
    # Обнулим позиционные эмбеддинги, чтобы изолировать вклад [CLS] токена.
    with torch.no_grad():
        emb.position_embeddings.zero_()
        emb.cls_token.fill_(0.123)
    x = torch.randn(1, cfg.in_channels, cfg.image_size, cfg.image_size)
    out = emb(x)
    # [CLS] токен конкатенируется уже в пространстве эмбеддингов (не проецируется),
    # поэтому при нулевых позиционных эмбеддингах первый токен — это сам [CLS].
    expected = torch.full((cfg.embed_dim,), 0.123)
    assert torch.allclose(out[0, 0], expected, atol=1e-5)


def test_manual_patching_matches_conv() -> None:
    """Ручная нарезка + Linear должна совпадать со строеной Conv2d 16x16."""
    cfg = Config()
    emb = PatchEmbedding(cfg)

    conv = nn.Conv2d(
        cfg.in_channels,
        cfg.embed_dim,
        kernel_size=cfg.patch_size,
        stride=cfg.patch_size,
    )
    emb.load_projection_from_conv(conv)

    x = torch.randn(2, cfg.in_channels, cfg.image_size, cfg.image_size)

    # Наш путь: нарезка на патчи + линейная проекция.
    manual = emb.projection(emb._to_patches(x))
    # Эталон: свёртка, затем развёртка карты признаков в последовательность.
    conv_out = conv(x).flatten(2).transpose(1, 2)

    assert manual.shape == conv_out.shape
    assert torch.allclose(manual, conv_out, atol=1e-5)
