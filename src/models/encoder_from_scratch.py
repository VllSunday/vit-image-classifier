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


Мой план реализации (заполняю заглушки сверху вниз)
===================================================
Шаг 1. Scaled Dot-Product Attention — сердце трансформера.
       Формула: Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
       Заодно разобраться: зачем скалярное произведение, зачем делить на √d_k,
       зачем softmax, что вообще такое «карта внимания».

Шаг 2. MultiHeadSelfAttention — несколько голов внимания параллельно.
       - линейные проекции входа в Q, K, V;
       - «разрезаю» вектор на num_heads голов по 64 числа;
       - применяю attention из шага 1 в каждой голове;
       - склеиваю головы обратно и прогоняю через выходную проекцию.

Шаг 3. MLP (FeedForward) — позиционный двухслойный персептрон.
       Linear(768→3072) → GELU → Linear(3072→768). Работает с каждым
       токеном независимо; именно тут «перемешиваются» признаки внутри токена.

Шаг 4. EncoderBlock (аналог ViTLayer) — собираю шаги 2 и 3 вместе
       с двумя LayerNorm и двумя residual-связями по pre-norm схеме.

Шаг 5. TransformerEncoder — ModuleList из num_layers блоков + финальный
       LayerNorm. Полный аналог backbone.layers + backbone.layernorm.

Шаг 6 (опционально). Проверка корректности:
       - прогнать случайный тензор и убедиться в формах/конечности значений;
       - если будет настроение — загрузить веса из HF ViTLayer и сверить выход
         с эталоном, чтобы доказать себе, что моя математика идентична библиотечной.

Пока всё заглушки с TODO. Реализую по одному шагу.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


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

    TODO (Шаг 1):
      1. d_k = head_dim (последняя ось q).
      2. scores = q @ k.transpose(-2, -1) / sqrt(d_k)   -> (..., seq, seq)
      3. attn = softmax(scores, dim=-1)
      4. out = attn @ v
    """
    raise NotImplementedError("Шаг 1: дописать scaled dot-product attention")


# ---------------------------------------------------------------------------
# Шаг 2. Multi-Head Self-Attention
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """Многоголовое самовнимание: (B, seq, dim) -> (B, seq, dim).

    TODO (Шаг 2):
      init:
        - сохранить num_heads, head_dim = embed_dim // num_heads;
        - проекции q_proj, k_proj, v_proj: nn.Linear(embed_dim, embed_dim);
          (в HF ViT это три отдельных Linear с bias=True);
        - выходная проекция out_proj: nn.Linear(embed_dim, embed_dim).
      forward(x):
        - посчитать Q, K, V из x;
        - reshape (B, seq, num_heads, head_dim) и permute -> (B, heads, seq, head_dim);
        - вызвать scaled_dot_product_attention;
        - склеить головы обратно в (B, seq, embed_dim);
        - прогнать через out_proj.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        raise NotImplementedError("Шаг 2: дописать multi-head self-attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Шаг 2: дописать forward MHSA")


# ---------------------------------------------------------------------------
# Шаг 3. MLP (FeedForward)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Позиционный MLP блока энкодера: Linear -> GELU -> Linear.

    TODO (Шаг 3):
      - fc1: nn.Linear(embed_dim, embed_dim * mlp_ratio)
      - act: nn.GELU()
      - fc2: nn.Linear(embed_dim * mlp_ratio, embed_dim)
      - forward: fc2(act(fc1(x)))
    """

    def __init__(self, embed_dim: int = 768, mlp_ratio: int = 4) -> None:
        super().__init__()
        raise NotImplementedError("Шаг 3: дописать MLP")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Шаг 3: дописать forward MLP")


# ---------------------------------------------------------------------------
# Шаг 4. Encoder Block (аналог одного ViTLayer)
# ---------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    """Один блок Transformer Encoder (pre-norm): (B, seq, dim) -> (B, seq, dim).

    TODO (Шаг 4):
      - norm1 = nn.LayerNorm(embed_dim, eps=1e-12)  # eps как в HF ViT
      - attn  = MultiHeadSelfAttention(...)
      - norm2 = nn.LayerNorm(embed_dim, eps=1e-12)
      - mlp   = MLP(...)
      forward(x):
        x = x + attn(norm1(x))   # 1-я residual-связь
        x = x + mlp(norm2(x))    # 2-я residual-связь
        return x
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: int = 4) -> None:
        super().__init__()
        raise NotImplementedError("Шаг 4: собрать encoder block")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Шаг 4: дописать forward блока")


# ---------------------------------------------------------------------------
# Шаг 5. Полный стек энкодера
# ---------------------------------------------------------------------------
class TransformerEncoder(nn.Module):
    """Стек из num_layers блоков + финальный LayerNorm.

    Полный аналог того, что в проекте я беру как backbone.layers + layernorm.

    TODO (Шаг 5):
      - blocks: nn.ModuleList([EncoderBlock(...) for _ in range(num_layers)])
      - norm:   nn.LayerNorm(embed_dim, eps=1e-12)
      forward(x):
        for block in blocks: x = block(x)
        return norm(x)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        raise NotImplementedError("Шаг 5: собрать полный стек энкодера")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Шаг 5: дописать forward стека")


# ---------------------------------------------------------------------------
# Шаг 6 (опционально). Быстрая самопроверка форм
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # После реализации прогоню фиктивный батч и проверю формы.
    # x = torch.randn(2, 197, 768)
    # encoder = TransformerEncoder()
    # y = encoder(x)
    # assert y.shape == x.shape
    # print("OK:", y.shape)
    raise SystemExit("Дописать шаги 1-5, затем раскомментировать проверку.")
