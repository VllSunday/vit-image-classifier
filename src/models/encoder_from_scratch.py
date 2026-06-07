"""Transformer Encoder для ViT — пишу руками, с нуля.

Зачем я завёл этот файл
=======================
В основном коде проекта (src/models/vit.py) руками у меня сделан только
patch embedding, а сам стек блоков Transformer Encoder я беру готовым из
HuggingFace:

    self.layers = backbone.layers   # ModuleList из 12 готовых ViTLayer

Мне не нравится, что внутренности энкодера для меня остаются чёрным ящиком.
Поэтому здесь я расписываю один блок (и весь стек) сам, без transformers,
чтобы окончательно разобраться, что там происходит. Файл самодостаточный, в
основной пайплайн его встраивать необязательно — это в первую очередь мой
разбор темы.

Размерности беру те же, что и в проекте (ViT-base/16):
    embed_dim   = 768       размер вектора одного токена
    num_heads   = 12        число голов внимания (768 / 12 = 64 на голову)
    num_layers  = 12        число блоков энкодера
    mlp_ratio   = 4         скрытый слой MLP = 4 * 768 = 3072
    seq_len     = 197       196 патчей + 1 токен [CLS]

На входе энкодер получает последовательность токенов (B, 197, 768) — ровно то,
что выдаёт мой PatchEmbedding, — и возвращает обновлённую последовательность
той же формы (B, 197, 768).


Как устроен энкодер (общая схема)
=================================
Энкодер ViT — это стопка одинаковых блоков. Важный момент: ViT использует
PRE-NORM (LayerNorm стоит ПЕРЕД под-слоем, а не после), и на каждый блок
приходится две остаточные (residual) связи:

    вход x ─┬─────────────────────────────────────────┐
            │                                          │
          LayerNorm ─► Multi-Head Self-Attention ─► (+)│   <- 1-я residual-связь
            │                                          │
    x = x + Attn(LN(x))                                ▼
            ┌──────────────────────────────────────────┐
            │                                           │
          LayerNorm ─► MLP (Linear→GELU→Linear) ───► (+)│   <- 2-я residual-связь
            │                                           │
    x = x + MLP(LN(x))                                  ▼
                                                    выход блока

После всех N блоков идёт ещё один финальный LayerNorm.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Шаг 1. Scaled Dot-Product Attention
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Внимание из статьи «Attention Is All You Need».

    Аргументы (формы):
        q, k, v: (B, num_heads, seq_len, head_dim)

    Возвращает:
        out:     (B, num_heads, seq_len, head_dim) — взвешенная сумма V;
        attn:    (B, num_heads, seq_len, seq_len)  — карты внимания (веса).
    """
    # размер головы — по нему нормируем скоры, иначе при больших d_k
    # скалярные произведения разъезжаются и softmax уходит в насыщение
    d_k = q.shape[-1]

    # для каждой пары токенов (i, j) считаю, насколько i «смотрит» на j:
    # (B, h, seq, d) @ (B, h, d, seq) -> (B, h, seq, seq)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    # по последней оси (по всем j) превращаю скоры в распределение весов
    attn = F.softmax(scores, dim=-1)

    # взвешенно собираю V: (B, h, seq, seq) @ (B, h, seq, d) -> (B, h, seq, d)
    out = attn @ v
    return out, attn


# ---------------------------------------------------------------------------
# Шаг 2. Multi-Head Self-Attention
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """Многоголовое самовнимание: (B, seq, dim) -> (B, seq, dim).

    Идея: вместо одной «большой» головы режу embed_dim на num_heads кусков
    по head_dim и считаю внимание в каждом куске независимо. Так модель может
    параллельно смотреть на разные связи между токенами.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim должен делиться на num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # три отдельных Linear под Q, K, V — как в HF ViT (bias=True по умолчанию)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # выходная проекция, чтобы перемешать информацию между головами
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, seq, dim) -> (B, seq, heads, head_dim) -> (B, heads, seq, head_dim)
        B, seq, _ = x.shape
        x = x.view(B, seq, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, seq, dim = x.shape

        # считаю Q, K, V и сразу раскладываю по головам
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # внимание в каждой голове независимо (карты внимания тут не нужны)
        out, _ = scaled_dot_product_attention(q, k, v)

        # склеиваю головы обратно: (B, heads, seq, head_dim) -> (B, seq, dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, seq, dim)

        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Шаг 3. MLP (FeedForward)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Позиционный MLP блока энкодера: Linear -> GELU -> Linear.

    Работает с каждым токеном независимо. Внимание перемешивает информацию
    МЕЖДУ токенами, а этот MLP — ВНУТРИ одного токена, между его признаками.
    """

    def __init__(self, embed_dim: int = 768, mlp_ratio: int = 4) -> None:
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# Шаг 4. Encoder Block (аналог одного ViTLayer)
# ---------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    """Один блок Transformer Encoder (pre-norm): (B, seq, dim) -> (B, seq, dim)."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: int = 4) -> None:
        super().__init__()
        # eps=1e-12 — как в HF ViT, чтобы поведение совпадало
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-12)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-12)
        self.mlp = MLP(embed_dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm: нормирую ВХОД под-слоя, а результат прибавляю к x (residual)
        x = x + self.attn(self.norm1(x))  # 1-я residual-связь
        x = x + self.mlp(self.norm2(x))  # 2-я residual-связь
        return x


# ---------------------------------------------------------------------------
# Шаг 5. Полный стек энкодера
# ---------------------------------------------------------------------------
class TransformerEncoder(nn.Module):
    """Стек из num_layers блоков + финальный LayerNorm.

    Полный аналог того, что в проекте я беру как backbone.layers + layernorm.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [EncoderBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Шаг 6 (опционально). Быстрая самопроверка форм
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    x = torch.randn(2, 197, 768)
    encoder = TransformerEncoder()
    y = encoder(x)
    assert y.shape == x.shape, f"форма не совпала: {y.shape}"
    assert torch.isfinite(y).all(), "в выходе появились nan/inf"
    print("OK:", tuple(y.shape))
