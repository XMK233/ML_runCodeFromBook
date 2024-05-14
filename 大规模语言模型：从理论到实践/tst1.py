import math

import warnings
import torch
import torch.nn as nn
from einops import rearrange

from typing import Optional


def scaled_multihead_dot_product_attention(
        query,

        key,
        value,
        n_heads,
        past_key_value=None,
        softmax_scale=None,
        attn_bias=None,
        key_padding_mask=None,
        is_causal=False,
        dropout_p=0.0,
        training=False,
        needs_weights=False,
        multiquery=False,
):
    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)  # (1, 512, 768) -> (1, 8, 512, 96)
    kv_n_heads = 1 if multiquery else n_heads
    k = rearrange(key, 'b s (h d) -> b h d s', h=kv_n_heads)  # (1, 512, 768) -> (1, 8, 96, 512) if not multiquery
    # (1, 512, 96) -> (1, 1, 96, 512)  if multiquery
    v = rearrange(value, 'b s (h d) -> b h s d', h=kv_n_heads)  # (1, 512, 768) -> (1, 8, 512, 96) if not multiquery
    # (1, 512, 96) -> (1, 1, 512, 96)  if multiquery

    attn_weight = q.matmul(k) * softmax_scale  # (1, 8, 512, 512)
    attn_weight = torch.softmax(attn_weight, dim=-1)  # (1, 8, 512, 512)

    out = attn_weight.matmul(v)  # (1, 8, 512, 512) * (1, 1, 512, 96) = (1, 8, 512, 96)
    out = rearrange(out, 'b h s d -> b s (h d)')  # (1, 512, 768)

    return out, attn_weight, past_key_value


class MultiheadAttention(nn.Module):
    """Multi-head self attention.

    Using torch or triton attention implemetation enables user to also use
    additive bias.
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            attn_impl: str = 'triton',
            clip_qkv: Optional[float] = None,
            qk_ln: bool = False,
            softmax_scale: Optional[float] = None,
            attn_pdrop: float = 0.0,
            low_precision_layernorm: bool = False,
            verbose: int = 0,
            device: Optional[str] = None,
    ):
        super().__init__()

        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln

        self.d_model = d_model
        self.n_heads = n_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = attn_pdrop

        self.Wqkv = nn.Linear(self.d_model, 3 * self.d_model, device=device)

        fuse_splits = (d_model, 2 * d_model)
        self.Wqkv._fused = (0, fuse_splits)  # type: ignore

        self.attn_fn = scaled_multihead_dot_product_attention
        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
        self.out_proj._is_residual = True  # type: ignore

    def forward(
            self,
            x,
            past_key_value=None,
            attn_bias=None,
            attention_mask=None,
            is_causal=True,
            needs_weights=False,
    ):
        qkv = self.Wqkv(x)  # (1, 512, 2304)

        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        query, key, value = qkv.chunk(3, dim=2)  # both q, k, v: (1, 512, 768)

        key_padding_mask = attention_mask

        context, attn_weights, past_key_value = self.attn_fn(
            query,
            key,
            value,
            self.n_heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p,
            training=self.training,
            needs_weights=needs_weights,
        )

        return self.out_proj(context), attn_weights, past_key_value


class MultiQueryAttention(nn.Module):
    """Multi-Query self attention.

    Using torch or triton attention implemetation enables user to also use
    additive bias.
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            attn_impl: str = 'triton',
            clip_qkv: Optional[float] = None,
            qk_ln: bool = False,
            softmax_scale: Optional[float] = None,
            attn_pdrop: float = 0.0,
            low_precision_layernorm: bool = False,
            verbose: int = 0,
            device: Optional[str] = None,
    ):
        super().__init__()

        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.head_dim)
        self.attn_dropout_p = attn_pdrop

        self.Wqkv = nn.Linear(
            d_model,
            d_model + 2 * self.head_dim,
            device=device,
        )

        fuse_splits = (d_model, d_model + self.head_dim)
        self.Wqkv._fused = (0, fuse_splits)  # type: ignore

        self.attn_fn = scaled_multihead_dot_product_attention
        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
        self.out_proj._is_residual = True  # type: ignore

    def forward(
            self,
            x,
            past_key_value=None,
            attn_bias=None,
            attention_mask=None,
            is_causal=True,
            needs_weights=False,
    ):
        qkv = self.Wqkv(x)  # (1, 512, 960)

        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        query, key, value = qkv.split(  # query -> (1, 512, 768)
            [self.d_model, self.head_dim, self.head_dim],  # key   -> (1, 512, 96)
            dim=2  # value -> (1, 512, 96)
        )

        key_padding_mask = attention_mask

        if self.qk_ln:
            # Applying layernorm to qk
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)

        context, attn_weights, past_key_value = self.attn_fn(
            query,
            key,
            value,
            self.n_heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p,
            training=self.training,
            needs_weights=needs_weights,
            multiquery=True,
        )

        return self.out_proj(context), attn_weights, past_key_value


if __name__ == '__main__':
    # attn = MultiQueryAttention(
    #     768,
    #     8,
    #     'torch'
    # )

    attn = MultiheadAttention(
        768,
        8,
        'torch'
    )

    attn(
        torch.ones(size=(1, 512, 768))
    )