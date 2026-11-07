#!/usr/bin/env python3
from __future__ import annotations

"""
Mask-free native-resolution inference for ONE image with multi-face geometry support.

Key behavior
------------
1) --input must be a single image file.
2) MediaPipe detects up to --max_faces faces.
3) Eye / iris / periocular / spectacle-lens priors from all detected faces are merged.
4) OcuFlow is executed once on the whole image; the fixed global token budget is shared
   across all detected faces.

Example
-------
    python infer_single_multiface.py \
        --input /path/to/glare.png \
        --checkpoint /path/to/checkpoint.pt \
        --output /path/to/pred.png \
        --max_faces 8 \
        --force_redetect
"""

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import fnmatch
import hashlib
from io import BytesIO
import json
import math
import multiprocessing as mp_ctx
import os
from pathlib import Path
import platform
import statistics
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Optional, Sequence

# MediaPipe/absl otherwise emits several C++ INFO/WARNING lines for every
# detector instance.  Set these before the detector module is imported lazily.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "reglass_mplconfig")
)

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from functools import partial
from math import sqrt
import heapq

# === BEGIN INLINED PROJECT RUNTIME ===


"""Project runtime vendored from the Stage-7 implementation.

This block intentionally contains the checkpoint architecture, adaptive-grid,
renderer, and geometry code needed by this standalone inference script.
"""


# --- inlined from models/torch_models.py ---
class TorchLinear(nn.Module):
    """A linear layer similar to torch.nn.Linear."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        weight_init="scaled_variance",
        init_constant=1.0,
        bias_init="zeros",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight_init = weight_init
        self.init_constant = init_constant
        self.bias_init = bias_init

        if self.weight_init == "scaled_variance":
            std = self.init_constant / sqrt(self.in_features)
            weight_initializer = partial(nn.init.normal_, std=std)
        elif self.weight_init == "zeros":
            weight_initializer = nn.init.zeros_
        else:
            raise ValueError(f"Invalid weight_init: {self.weight_init}")

        if self.bias_init == "zeros":
            bias_initializer = nn.init.zeros_
        else:
            raise ValueError(f"Invalid bias_init: {self.bias_init}")

        self._flax_linear = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias,
        )
        weight_initializer(self._flax_linear.weight)
        if self.bias:
            bias_initializer(self._flax_linear.bias)

    def forward(self, x):
        return self._flax_linear(x)


class TorchEmbedding(nn.Module):
    """A embedding layer similar to torch.nn.Embedding."""

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        weight_init="scaled_variance",
        init_constant=1.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight_init = weight_init
        self.init_constant = init_constant

        if self.weight_init is None:
            std = 0.02
        elif self.weight_init == "scaled_variance":
            std = self.init_constant / sqrt(self.embedding_dim)
        else:
            raise ValueError(f"Invalid weight_init: {self.weight_init}")

        self._flax_embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
        )
        nn.init.normal_(self._flax_embedding.weight, std=std)

    def forward(self, x):
        return self._flax_embedding(x)


class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(self.dim))

    def _norm(self, x):
        mean_square = torch.mean(torch.square(x), dim=-1, keepdim=True)
        return x * torch.rsqrt(mean_square + self.eps)

    def forward(self, x):
        output = self._norm(x).to(x.dtype)
        return output * self.weight


class SwiGLUMlp(nn.Module):
    """Swish-Gated Linear Unit MLP."""

    def __init__(
        self,
        in_features,
        hidden_features,
        weight_init="scaled_variance",
        weight_init_constant=1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.weight_init = weight_init
        self.weight_init_constant = weight_init_constant

        init_kwargs = dict(
            bias=False,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )

        self.w1 = TorchLinear(self.in_features, self.hidden_features, **init_kwargs)
        self.w3 = TorchLinear(self.in_features, self.hidden_features, **init_kwargs)
        self.w2 = TorchLinear(self.hidden_features, self.in_features, **init_kwargs)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# --- inlined from models/embedder.py ---
class TimestepEmbedder(nn.Module):
    """Embeds a scalar timestep (or scalar conditioning) into a vector."""

    def __init__(
        self,
        hidden_size,
        frequency_embedding_size=256,
        weight_init="scaled_variance",
        init_constant=1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.weight_init = weight_init
        self.init_constant = init_constant

        init_kwargs = dict(
            out_features=self.hidden_size,
            bias=True,
            weight_init=self.weight_init,
            init_constant=self.init_constant,
            bias_init="zeros",
        )
        self.mlp = nn.Sequential(
            TorchLinear(self.frequency_embedding_size, **init_kwargs),
            nn.SiLU(),
            TorchLinear(self.hidden_size, **init_kwargs),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        args = t[:, None].to(torch.float32) * freqs[None].to(t.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], axis=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations with token dropout."""

    def __init__(
        self, num_classes, hidden_size, weight_init="scaled_variance", init_constant=1.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.weight_init = weight_init
        self.init_constant = init_constant

        self.embedding_table = TorchEmbedding(
            self.num_classes + 1,
            self.hidden_size,
            weight_init=self.weight_init,
            init_constant=self.init_constant,
        )

    def forward(self, labels):
        return self.embedding_table(labels)


class BottleneckPatchEmbedder(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self, input_size, initial_patch_size, pca_channels, in_channels, hidden_size, bias=True
    ):
        super().__init__()
        self.input_size = input_size
        self.initial_patch_size = initial_patch_size
        self.in_channels = in_channels
        self.pca_channels = pca_channels
        self.hidden_size = hidden_size
        self.bias = bias

        self.patch_size = (self.initial_patch_size, self.initial_patch_size)
        self.img_size = self.input_size
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(
            self.img_size
        )

        self.flatten = True
        self.proj1 = nn.Conv2d(
            self.in_channels,
            self.pca_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=self.bias,
        )
        self.proj2 = nn.Conv2d(
            self.pca_channels,
            self.hidden_size,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=self.bias,
        )

        # init proj1 weights like nn.Linear, instead of nn.Conv2d
        kh = kw = self.patch_size[0]
        fan_in = kh * kw * self.in_channels
        fan_out = self.pca_channels
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.proj1.weight, -limit, limit)

        # init proj2 weights like nn.Linear, instead of nn.Conv2d
        fan_in = self.pca_channels
        fan_out = self.hidden_size
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.proj2.weight, -limit, limit)
        if self.bias:
            nn.init.zeros_(self.proj1.bias)
            nn.init.zeros_(self.proj2.bias)
    
    def _init_img_size(self, img_size):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            img_size = (int(img_size[0]), int(img_size[1]))
        else:
            raise ValueError(f"input_size must be an int or (height, width), got {img_size!r}")
        if any(size % patch for size, patch in zip(img_size, self.patch_size)):
            raise ValueError(f"input_size {img_size} must be divisible by patch_size {self.patch_size}")
        grid_size = tuple(size // patch for size, patch in zip(img_size, self.patch_size))
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def forward(self, x, return_grid_size: bool = False):
        """Embed any patch-aligned rectangular image.

        ``input_size`` defines the training/reference grid used to initialize
        positional parameters, but inference may use any height and width that
        are divisible by ``patch_size``.  This keeps the patch projection fully
        convolutional and lets the pMF wrapper generate dynamic 2D position
        encodings for native-resolution images.
        """
        batch, _, height, width = x.shape
        patch_h, patch_w = self.patch_size
        if height % patch_h or width % patch_w:
            raise ValueError(
                f"Input spatial size {(height, width)} must be divisible by "
                f"patch_size {self.patch_size}"
            )
        x = self.proj2(self.proj1(x))
        grid_size = (int(x.shape[-2]), int(x.shape[-1]))
        tokens = x.permute(0, 2, 3, 1).reshape(batch, -1, x.shape[1])
        return (tokens, grid_size) if return_grid_size else tokens


# --- inlined from models/pmf.py ---
def unsqueeze(t: torch.Tensor, dim: int) -> torch.Tensor:
    return t.unsqueeze(dim)


def _rope_phase(head_dim: int, positions_yx: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Complex axial 2D RoPE phase for arbitrary continuous y/x positions."""
    complex_dim = head_dim // 2
    if complex_dim % 2:
        raise ValueError(f"head_dim={head_dim} is incompatible with axial 2D RoPE")
    freq = 1.0 / (
        theta
        ** (
            torch.arange(0, complex_dim, 2, device=positions_yx.device, dtype=torch.float32)
            / complex_dim
        )
    )
    y_phase = positions_yx[..., 0:1].float() * freq
    x_phase = positions_yx[..., 1:2].float() * freq
    phase = torch.cat([y_phase, x_phase], dim=-1)
    return torch.complex(torch.cos(phase), torch.sin(phase))


def apply_rotary_positions(x: torch.Tensor, positions_yx: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to [N,H,D] using one continuous 2D position per token."""
    if x.ndim != 3:
        raise ValueError(f"expected [N,H,D], got {tuple(x.shape)}")
    x_float = x.float()
    x_complex = torch.view_as_complex(
        x_float.reshape(x.shape[0], x.shape[1], -1, 2).contiguous()
    )
    phase = _rope_phase(x.shape[-1], positions_yx)[:, None, :]
    output = x_complex * phase
    return torch.view_as_real(output).flatten(-2).to(x.dtype)


def apply_rotary_phase(x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Apply a precomputed complex RoPE phase to [N,H,D]."""
    if x.ndim != 3:
        raise ValueError(f"expected [N,H,D], got {tuple(x.shape)}")
    if phase.ndim != 2 or phase.shape[0] != x.shape[0]:
        raise ValueError(f"phase {tuple(phase.shape)} is incompatible with {tuple(x.shape)}")
    if x.shape[0] == 0:
        return x
    x_complex = torch.view_as_complex(
        x.float().reshape(x.shape[0], x.shape[1], -1, 2).contiguous()
    )
    output = x_complex * phase[:, None, :]
    return torch.view_as_real(output).flatten(-2).to(x.dtype)


def apply_rotary_phase_batched(x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Apply a precomputed complex RoPE phase to ``[B,N,H,D]``."""
    if x.ndim != 4:
        raise ValueError(f"expected [B,N,H,D], got {tuple(x.shape)}")
    if phase.ndim != 3 or phase.shape[:2] != x.shape[:2]:
        raise ValueError(f"phase {tuple(phase.shape)} is incompatible with {tuple(x.shape)}")
    if x.shape[1] == 0:
        return x
    x_complex = torch.view_as_complex(
        x.float().reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2).contiguous()
    )
    output = x_complex * phase[:, :, None, :]
    return torch.view_as_real(output).flatten(-2).to(x.dtype)


class RoPEAttention(nn.Module):
    """Official pMF attention plus dense padded mixed-scale attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        weight_init: str = "scaled_variance",
        weight_init_constant: float = 1.0,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.use_sdpa = bool(use_sdpa)
        # Production adaptive attention backend.  The ordinary dense path already
        # uses SDPA when enabled; the padded mixed-scale path is selectable so the
        # legacy implementation remains available for numerical/performance A/B.
        self.padded_backend = "legacy"
        kwargs = dict(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False,
            weight_init=weight_init,
            init_constant=weight_init_constant,
        )
        self.q_proj = TorchLinear(**kwargs)
        self.k_proj = TorchLinear(**kwargs)
        self.v_proj = TorchLinear(**kwargs)
        self.out_proj = TorchLinear(**kwargs)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def _qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nshape = (*x.shape[:-1], self.num_heads, self.head_dim)
        q = self.q_norm(self.q_proj(x).reshape(nshape))
        k = self.k_norm(self.k_proj(x).reshape(nshape))
        v = self.v_proj(x).reshape(nshape)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        key_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q, k, v = self._qkv(x)
        q = apply_rotary_pos_emb(q, rope_freqs)
        k = apply_rotary_pos_emb(k, rope_freqs)
        if self.use_sdpa:
            mask = None if key_valid is None else key_valid[:, None, None, :]
            attended = F.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
            ).permute(0, 2, 1, 3)
        else:
            logits = torch.einsum(
                "bqhd,bkhd->bhqk", q / math.sqrt(self.head_dim), k
            )
            if key_valid is not None:
                logits = logits.masked_fill(~key_valid[:, None, None, :], -1.0e4)
            weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(v.dtype)
            attended = torch.einsum("bhqk,bkhd->bqhd", weights, v)
        return self.out_proj(attended.reshape(batch, seq_len, self.hidden_size))

    def forward_padded(
        self,
        x: torch.Tensor,
        key_valid: torch.Tensor,
        query_groups,
        rope_phases,
        key_area_bias: torch.Tensor,
        all_keys_valid: bool = False,
    ) -> torch.Tensor:
        """Dense bucket-padded mixed-scale attention with one output reorder.

        ``legacy`` preserves the original einsum -> FP32 softmax -> einsum path.
        ``sdpa`` expresses the identical per-key area bias as an additive SDPA mask
        so PyTorch can dispatch to an optimized CUDA attention backend when legal.
        Query-scale RoPE remains unchanged: each p4/p8/p16/p32 group is still
        evaluated against the corresponding rotated K tensor.
        """
        if x.ndim != 3:
            raise ValueError(f"expected [B,L,D], got {tuple(x.shape)}")
        backend = str(getattr(self, "padded_backend", "legacy")).lower()
        if backend not in ("legacy", "sdpa"):
            raise ValueError(f"unknown padded attention backend {backend!r}")

        q, k, v = self._qkv(x)
        scale = 1.0 / math.sqrt(self.head_dim)
        group_phase = (2, 0, 1, 2, 3)
        k_rotated = tuple(apply_rotary_phase_batched(k, phase) for phase in rope_phases)
        key_mask = None if all_keys_valid else key_valid[:, None, None, :]
        area = key_area_bias[:, None, None, :]

        # SDPA accepts a broadcastable floating attention mask that is added to
        # QK^T scores.  Build it once per network call, not once per Transformer
        # block/group.  finfo.min matches the legacy masked_fill policy.
        sdpa_mask = None
        if backend == "sdpa":
            sdpa_mask = area.to(dtype=q.dtype)
            if key_mask is not None:
                sdpa_mask = sdpa_mask.masked_fill(
                    ~key_mask, torch.finfo(q.dtype).min
                )

        attended_groups = []
        index_groups = []
        valid_groups = []

        for group_id, (query_index, query_valid) in enumerate(query_groups):
            if query_index.shape[1] == 0:
                continue
            phase_id = group_phase[group_id]
            gather_index = query_index[:, :, None, None].expand(
                -1, -1, self.num_heads, self.head_dim
            )
            q_group = torch.gather(q, 1, gather_index)
            if group_id != 0:
                phase_index = query_index[:, :, None].expand(
                    -1, -1, rope_phases[phase_id].shape[-1]
                )
                q_phase = torch.gather(rope_phases[phase_id], 1, phase_index)
                q_group = apply_rotary_phase_batched(q_group, q_phase)

            if backend == "sdpa":
                attended = F.scaled_dot_product_attention(
                    q_group.permute(0, 2, 1, 3),
                    k_rotated[phase_id].permute(0, 2, 1, 3),
                    v.permute(0, 2, 1, 3),
                    attn_mask=sdpa_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scale,
                ).permute(0, 2, 1, 3)
            else:
                logits = torch.einsum(
                    "bqhd,bkhd->bhqk", q_group * scale, k_rotated[phase_id]
                )
                logits = logits + area.to(logits.dtype)
                if key_mask is not None:
                    logits = logits.masked_fill(
                        ~key_mask, torch.finfo(logits.dtype).min
                    )
                weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(v.dtype)
                attended = torch.einsum("bhqk,bkhd->bqhd", weights, v)

            attended_groups.append(attended)
            index_groups.append(query_index)
            valid_groups.append(query_valid)

        output = v.new_zeros(q.shape)
        if attended_groups:
            attended_all = torch.cat(attended_groups, dim=1)
            index_all = torch.cat(index_groups, dim=1)
            valid_all = torch.cat(valid_groups, dim=1)
            attended_all = attended_all * valid_all[:, :, None, None].to(
                attended_all.dtype
            )
            scatter_index = index_all[:, :, None, None].expand(
                -1, -1, self.num_heads, self.head_dim
            )
            output = output.scatter_add(1, scatter_index, attended_all)
        if not all_keys_valid:
            output = output * key_valid[:, :, None, None].to(output.dtype)
        batch, seq_len = x.shape[:2]
        return self.out_proj(output.reshape(batch, seq_len, self.hidden_size))

    @torch.no_grad()
    def padded_attention_weights(
        self,
        x: torch.Tensor,
        key_valid: torch.Tensor,
        query_groups,
        rope_phases,
        key_area_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Return exact mean-head attention as ``[B,L,L]`` for diagnostics."""
        q, k, _ = self._qkv(x)
        batch, seq_len = x.shape[:2]
        scale = 1.0 / math.sqrt(self.head_dim)
        group_phase = (2, 0, 1, 2, 3)
        k_rotated = tuple(
            apply_rotary_phase_batched(k, phase) for phase in rope_phases
        )
        key_mask = key_valid[:, None, None, :]
        area = key_area_bias[:, None, None, :]
        matrix = torch.zeros(
            batch, seq_len, seq_len, device=x.device, dtype=torch.float32
        )
        for group_id, (query_index, query_valid) in enumerate(query_groups):
            if query_index.shape[1] == 0:
                continue
            phase_id = group_phase[group_id]
            gather_index = query_index[:, :, None, None].expand(
                -1, -1, self.num_heads, self.head_dim
            )
            q_group = torch.gather(q, 1, gather_index)
            if group_id != 0:
                phase_index = query_index[:, :, None].expand(
                    -1, -1, rope_phases[phase_id].shape[-1]
                )
                q_phase = torch.gather(rope_phases[phase_id], 1, phase_index)
                q_group = apply_rotary_phase_batched(q_group, q_phase)
            logits = torch.einsum(
                "bqhd,bkhd->bhqk", q_group * scale, k_rotated[phase_id]
            )
            logits = logits + area.to(logits.dtype)
            logits = logits.masked_fill(~key_mask, torch.finfo(logits.dtype).min)
            weights = F.softmax(logits, dim=-1, dtype=torch.float32).mean(dim=1)
            weights = weights * query_valid[:, :, None].to(weights.dtype)
            row_index = query_index[:, :, None].expand(-1, -1, seq_len)
            matrix = matrix.scatter_add(1, row_index, weights)
        return matrix


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 8 / 3,
        weight_init: str = "scaled_variance",
        weight_init_constant: float = 1.0,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = RoPEAttention(
            hidden_size,
            num_heads,
            weight_init=weight_init,
            weight_init_constant=weight_init_constant,
            use_sdpa=use_sdpa,
        )
        self.norm2 = RMSNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if hidden_size > 1024:
            mlp_hidden_dim = (mlp_hidden_dim + 7) // 8 * 8
        self.mlp = SwiGLUMlp(
            hidden_size,
            mlp_hidden_dim,
            weight_init=weight_init,
            weight_init_constant=weight_init_constant,
        )
        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self, x: torch.Tensor, rope_freqs: torch.Tensor, key_valid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_freqs, key_valid=key_valid) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x

    def forward_padded(
        self,
        x: torch.Tensor,
        key_valid: torch.Tensor,
        query_groups,
        rope_phases,
        key_area_bias: torch.Tensor,
        all_keys_valid: bool = False,
    ) -> torch.Tensor:
        valid = None if all_keys_valid else key_valid[:, :, None].to(x.dtype)
        x = x + self.attn.forward_padded(
            self.norm1(x),
            key_valid,
            query_groups,
            rope_phases,
            key_area_bias,
            all_keys_valid=all_keys_valid,
        ) * self.attn_scale
        if valid is not None:
            x = x * valid
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x if valid is None else x * valid


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm = RMSNorm(hidden_size)
        self.linear = TorchLinear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
            weight_init="zeros",
            bias_init="zeros",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class pmfDiT(nn.Module):
    """Official-compatible pMF with an adaptive dense-padded grid path."""

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        canonical_size: Optional[int] = None,
        in_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 16,
        num_heads: int = 12,
        mlp_ratio: float = 8 / 3,
        num_classes: int = 1000,
        aux_head_depth: int = 8,
        num_class_tokens: int = 8,
        num_time_tokens: int = 4,
        num_cfg_tokens: int = 4,
        num_interval_tokens: int = 2,
        token_init_constant: float = 1.0,
        embedding_init_constant: float = 1.0,
        weight_init_constant: float = 0.32,
        pca_channels: Optional[int] = None,
        eval_mode: bool = False,
        use_sdpa: bool = True,
        gradient_checkpointing: bool = False,
        enable_v_head: bool = True,
        region_condition_dim: int = 4,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        # patch_size is the logical-grid reference scale (16 for the unchanged
        # V29.4 p4/p8/p16/p32 grid). canonical_size is the per-leaf pixel
        # resolution consumed and directly predicted by the adaptive PMF.
        self.patch_size = int(patch_size)
        self.canonical_size = int(canonical_size if canonical_size is not None else patch_size)
        if self.patch_size <= 0 or self.canonical_size <= 0:
            raise ValueError("patch_size and canonical_size must be positive")
        if self.input_size % self.patch_size:
            raise ValueError("input_size must be divisible by logical patch_size")
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.num_classes = int(num_classes)
        self.aux_head_depth = int(aux_head_depth)
        self.num_class_tokens = int(num_class_tokens)
        self.num_time_tokens = int(num_time_tokens)
        self.num_cfg_tokens = int(num_cfg_tokens)
        self.num_interval_tokens = int(num_interval_tokens)
        self.token_init_constant = float(token_init_constant)
        self.embedding_init_constant = float(embedding_init_constant)
        self.weight_init_constant = float(weight_init_constant)
        self.pca_channels = pca_channels
        self.eval_mode = bool(eval_mode)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.enable_v_head = bool(enable_v_head)
        self.region_condition_dim = int(region_condition_dim)
        self.out_channels = self.in_channels

        bottleneck = pca_channels if pca_channels is not None else (128 if hidden_size <= 1024 else 256)
        # Keep the positional reference grid identical to B/16 (16x16 at a
        # 256 logical canvas) while allowing each adaptive leaf to carry a 24x24
        # canonical raster. A 384x384 reference with 24px embedding kernels also
        # yields a 16x16 token table, so transformer positional semantics and the
        # cached logical grid remain unchanged.
        reference_grid = self.input_size // self.patch_size
        embed_reference_size = reference_grid * self.canonical_size
        self.x_embedder = BottleneckPatchEmbedder(
            embed_reference_size, self.canonical_size, bottleneck, in_channels, hidden_size, bias=True
        )
        embed_kwargs = dict(
            hidden_size=hidden_size,
            weight_init="scaled_variance",
            init_constant=embedding_init_constant,
        )
        self.h_embedder = TimestepEmbedder(**embed_kwargs)
        self.omega_embedder = TimestepEmbedder(**embed_kwargs)
        self.cfg_t_start_embedder = TimestepEmbedder(**embed_kwargs)
        self.cfg_t_end_embedder = TimestepEmbedder(**embed_kwargs)
        self.y_embedder = LabelEmbedder(num_classes, **embed_kwargs)

        token_initializer = partial(
            nn.init.normal_, std=token_init_constant / math.sqrt(hidden_size)
        )
        self.time_tokens = nn.Parameter(token_initializer(torch.empty(1, num_time_tokens, hidden_size)))
        self.class_tokens = nn.Parameter(token_initializer(torch.empty(1, num_class_tokens, hidden_size)))
        self.omega_tokens = nn.Parameter(token_initializer(torch.empty(1, num_cfg_tokens, hidden_size)))
        self.t_min_tokens = nn.Parameter(token_initializer(torch.empty(1, num_interval_tokens, hidden_size)))
        self.t_max_tokens = nn.Parameter(token_initializer(torch.empty(1, num_interval_tokens, hidden_size)))

        context_tokens = num_class_tokens + num_cfg_tokens + 2 * num_interval_tokens
        self.prefix_tokens = context_tokens + num_time_tokens
        total_tokens = self.x_embedder.num_patches + self.prefix_tokens
        self.head_dim = hidden_size // num_heads
        self.base_grid_size = tuple(map(int, self.x_embedder.grid_size))
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.head_dim, self.base_grid_size),
            persistent=False,
        )
        self.pos_embed = nn.Parameter(
            nn.init.normal_(torch.empty(1, total_tokens, hidden_size), std=0.02)
        )

        # Fine/coarse absolute-position and scale information start at exactly
        # zero, so base-patch tokens use the official positional representation.
        offset_hidden = max(32, hidden_size // 8)
        self.adaptive_offset_mlp = nn.Sequential(
            nn.Linear(3, offset_hidden),
            nn.SiLU(),
            nn.Linear(offset_hidden, hidden_size),
        )
        nn.init.zeros_(self.adaptive_offset_mlp[-1].weight)
        nn.init.zeros_(self.adaptive_offset_mlp[-1].bias)
        self.adaptive_scale_residual = nn.Parameter(torch.zeros(3, hidden_size))

        # Mask-free eye-geometry conditioning.  The projection is zero initialized so
        # V16/official warm starts preserve their learned function at step 0.
        self.eye_region_proj = (
            nn.Linear(self.region_condition_dim, hidden_size, bias=False)
            if self.region_condition_dim > 0 else None
        )
        if self.eye_region_proj is not None:
            nn.init.zeros_(self.eye_region_proj.weight)

        head_depth = aux_head_depth
        shared_depth = depth - head_depth
        if shared_depth < 0:
            raise ValueError("aux_head_depth cannot exceed depth")
        block_kwargs = dict(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            weight_init="scaled_variance",
            weight_init_constant=weight_init_constant,
            use_sdpa=use_sdpa,
        )
        self.shared_blocks = nn.ModuleList([TransformerBlock(**block_kwargs) for _ in range(shared_depth)])
        self.u_heads = nn.ModuleList([TransformerBlock(**block_kwargs) for _ in range(head_depth)])
        self.v_heads = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(head_depth if (self.enable_v_head and not eval_mode) else 0)]
        )
        # Direct canonical prediction: this is still the original x0/flow head,
        # only widened spatially. There is no residual/detail branch.
        self.u_final_layer = FinalLayer(hidden_size, self.canonical_size, self.out_channels)
        self.v_final_layer = (
            FinalLayer(hidden_size, self.canonical_size, self.out_channels)
            if self.enable_v_head
            else nn.Identity()
        )

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def set_sdpa_enabled(self, enabled: bool = True) -> None:
        for module in self.modules():
            if isinstance(module, RoPEAttention):
                module.use_sdpa = bool(enabled)

    def sdpa_enabled(self) -> bool:
        return any(isinstance(module, RoPEAttention) and module.use_sdpa for module in self.modules())

    def _prefix(self, t: torch.Tensor, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del t
        h_embed = self.h_embedder(h)
        y_embed = self.y_embedder(y)
        omega_embed = self.omega_embedder(torch.zeros_like(h))
        t_min_embed = self.cfg_t_start_embedder(torch.zeros_like(h))
        t_max_embed = self.cfg_t_end_embedder(torch.ones_like(h))
        pieces = [self.class_tokens + unsqueeze(y_embed, 1)]
        if self.num_cfg_tokens:
            pieces.append(self.omega_tokens + unsqueeze(omega_embed, 1))
        if self.num_interval_tokens:
            pieces.extend(
                [
                    self.t_min_tokens + unsqueeze(t_min_embed, 1),
                    self.t_max_tokens + unsqueeze(t_max_embed, 1),
                ]
            )
        pieces.append(self.time_tokens + unsqueeze(h_embed, 1))
        return torch.cat(pieces, dim=1)

    def unpatchify(self, x: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        c = self.out_channels
        p = self.canonical_size
        if grid_size is None:
            grid_size = self.base_grid_size
        h, w = map(int, grid_size)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def _interpolated_pos_embed(self, grid_size: Tuple[int, int], device, dtype) -> torch.Tensor:
        grid_h, grid_w = map(int, grid_size)
        prefix = self.pos_embed[:, : self.prefix_tokens]
        image = self.pos_embed[:, self.prefix_tokens :]
        base_h, base_w = self.base_grid_size
        image = image.reshape(1, base_h, base_w, self.hidden_size).permute(0, 3, 1, 2)
        if (grid_h, grid_w) != (base_h, base_w):
            image = F.interpolate(
                image.float(),
                size=(grid_h, grid_w),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).to(self.pos_embed.dtype)
        image = image.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, self.hidden_size)
        return torch.cat([prefix, image], dim=1).to(device=device, dtype=dtype)

    def rope_for_grid(self, grid_size: Tuple[int, int], device: torch.device) -> torch.Tensor:
        if tuple(map(int, grid_size)) == self.base_grid_size:
            return self.rope_freqs.to(device=device)
        return precompute_rope_freqs(self.head_dim, grid_size).to(device=device)

    def build_sequence_with_layout(self, x, t, h, y):
        x_embed, grid_size = self.x_embedder(x, return_grid_size=True)
        prefix = self._prefix(t, h, y)
        seq = torch.cat([prefix, x_embed], dim=1)
        seq = seq + self._interpolated_pos_embed(grid_size, seq.device, seq.dtype)
        return seq, self.rope_for_grid(grid_size, x.device), grid_size

    def _run_block(self, block, seq, rope_freqs, key_valid=None):
        def fn(value):
            return block(value, rope_freqs, key_valid=key_valid)
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return activation_checkpoint(fn, seq, use_reentrant=False)
        return fn(seq)

    def forward(self, x, t, h, y, need_v: bool = True, key_valid=None):
        seq, rope_freqs, grid_size = self.build_sequence_with_layout(x, t, h, y)
        for block in self.shared_blocks:
            seq = self._run_block(block, seq, rope_freqs, key_valid=key_valid)
        u_seq = seq
        for block in self.u_heads:
            u_seq = self._run_block(block, u_seq, rope_freqs, key_valid=key_valid)
        u = self.unpatchify(self.u_final_layer(u_seq[:, self.prefix_tokens :]), grid_size)
        v = None
        if need_v:
            if not self.enable_v_head:
                raise RuntimeError("this pmfDiT instance was constructed with enable_v_head=False")
            v_seq = seq
            for block in self.v_heads:
                v_seq = self._run_block(block, v_seq, rope_freqs, key_valid=key_valid)
            v = self.unpatchify(self.v_final_layer(v_seq[:, self.prefix_tokens :]), grid_size)
        return u, v

    def _canonical_patch_embed(self, patches: torch.Tensor) -> torch.Tensor:
        p = self.canonical_size
        if patches.ndim != 4 or tuple(patches.shape[-2:]) != (p, p):
            raise ValueError(f"canonical patches must be [T,C,{p},{p}], got {tuple(patches.shape)}")
        x = self.x_embedder.proj2(self.x_embedder.proj1(patches))
        if tuple(x.shape[-2:]) != (1, 1):
            raise RuntimeError("canonical patch embedder must produce one token per patch")
        return x[:, :, 0, 0]

    def _adaptive_image_pos(
        self,
        leaves: torch.Tensor,
        canvas_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base = int(self.patch_size)
        grid = int(canvas_size) // base
        table = self._interpolated_pos_embed((grid, grid), leaves.device, dtype)[
            0, self.prefix_tokens :
        ].reshape(grid, grid, self.hidden_size)
        x = leaves[:, 0].long()
        y = leaves[:, 1].long()
        size = leaves[:, 2].long()
        parent_x = torch.div(x, base, rounding_mode="floor").clamp(0, grid - 1)
        parent_y = torch.div(y, base, rounding_mode="floor").clamp(0, grid - 1)
        pos = table[parent_y, parent_x]

        coarse = size == base * 2
        # Keep this branch entirely on-device.  bool(coarse.any()) would force a
        # D2H synchronization on every adaptive inference call.
        coarse_max = max(grid - 2, 0)
        cx = parent_x.clamp(0, coarse_max)
        cy = parent_y.clamp(0, coarse_max)
        cx1 = (cx + 1).clamp_max(grid - 1)
        cy1 = (cy + 1).clamp_max(grid - 1)
        coarse_pos = (
            table[cy, cx]
            + table[cy, cx1]
            + table[cy1, cx]
            + table[cy1, cx1]
        ) * 0.25
        pos = torch.where(coarse[:, None], coarse_pos, pos)

        center_x = x.float() + size.float() * 0.5
        center_y = y.float() + size.float() * 0.5
        parent_center_x = parent_x.float() * base + base * 0.5
        parent_center_y = parent_y.float() * base + base * 0.5
        offset = torch.stack(
            [
                (center_y - parent_center_y) / base,
                (center_x - parent_center_x) / base,
                torch.log2(size.float() / base),
            ],
            dim=-1,
        )
        pos = pos + self.adaptive_offset_mlp(offset.to(pos.dtype))

        residual = torch.zeros_like(pos)
        scale_rows = ((base // 4, 0), (base // 2, 1), (base * 2, 2))
        for patch_size, row in scale_rows:
            mask = (size == patch_size).to(pos.dtype)[:, None]
            residual = residual + mask * self.adaptive_scale_residual[row].to(pos.dtype)
        return pos + residual

    @staticmethod
    def _make_padded_query_group(mask: torch.Tensor):
        """Convert a ``[B,L]`` query mask into dense gather indices.

        The returned query dimension is the largest count in this batch. This
        operation runs once per network call, not once per Transformer block.
        """
        if mask.ndim != 2:
            raise ValueError(f"query mask must be [B,L], got {tuple(mask.shape)}")
        batch, seq_len = mask.shape
        counts = mask.sum(dim=1)
        max_count = int(counts.max().detach().cpu()) if batch else 0
        if max_count == 0:
            return (
                torch.zeros(batch, 0, device=mask.device, dtype=torch.long),
                torch.zeros(batch, 0, device=mask.device, dtype=torch.bool),
            )
        positions = torch.arange(seq_len, device=mask.device, dtype=torch.long)
        positions = positions[None, :].expand(batch, -1)
        sort_key = torch.where(mask, positions, positions + seq_len)
        query_index = torch.sort(sort_key, dim=1).values[:, :max_count].clamp_max(seq_len - 1)
        query_valid = torch.arange(max_count, device=mask.device)[None, :] < counts[:, None]
        return query_index, query_valid

    def prepare_adaptive_layout(
        self,
        *,
        slot_index: torch.Tensor,
        key_valid: torch.Tensor,
        query_groups,
        rope_positions: torch.Tensor,
        key_area_bias: torch.Tensor,
        lengths: Sequence[int],
        padded_image_tokens: int,
    ):
        """Materialize RoPE phases once for the adaptive attention path."""
        if rope_positions.ndim != 4 or rope_positions.shape[0] != 4:
            raise ValueError(
                f"rope_positions must be [4,B,L,2], got {tuple(rope_positions.shape)}"
            )
        phases = tuple(
            _rope_phase(self.hidden_size // self.num_heads, rope_positions[level])
            for level in range(4)
        )
        dense_image_tokens = bool(
            lengths and all(int(value) == int(padded_image_tokens) for value in lengths)
        )
        return {
            "slot_index": slot_index,
            "key_valid": key_valid,
            "dense_image_tokens": dense_image_tokens,
            "all_keys_valid": dense_image_tokens,
            "query_groups": tuple(query_groups),
            "rope_phases": phases,
            "key_area_bias": key_area_bias,
            "lengths": tuple(int(value) for value in lengths),
            "padded_image_tokens": int(padded_image_tokens),
        }

    def build_adaptive_sequence(
        self,
        canonical_patches: torch.Tensor,
        image_offsets: torch.Tensor,
        leaves: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        y: torch.Tensor,
        canvas_size: int,
        padded_image_tokens: int,
        adaptive_layout=None,
        region_probs: Optional[torch.Tensor] = None,
        image_tokens_override: Optional[torch.Tensor] = None,
        query_groups_override=None,
    ):
        if image_tokens_override is None:
            image_embed = self._canonical_patch_embed(canonical_patches)
        else:
            if image_tokens_override.ndim != 2:
                raise ValueError(f"image_tokens_override must be [T,D], got {tuple(image_tokens_override.shape)}")
            if image_tokens_override.shape[0] != leaves.shape[0] or image_tokens_override.shape[1] != self.hidden_size:
                raise ValueError(
                    "image_tokens_override must match leaves and hidden_size: "
                    f"tokens={tuple(image_tokens_override.shape)} leaves={tuple(leaves.shape)} hidden={self.hidden_size}"
                )
            image_embed = image_tokens_override
        if region_probs is not None:
            if self.eye_region_proj is None:
                raise RuntimeError("region_probs were supplied while semantic conditioning is disabled")
            if region_probs.ndim != 2 or region_probs.shape[0] != image_embed.shape[0]:
                raise ValueError(f"region_probs must be [T,K], got {tuple(region_probs.shape)}")
            if region_probs.shape[1] != self.region_condition_dim:
                raise ValueError(
                    f"region_probs has K={region_probs.shape[1]}, expected {self.region_condition_dim}"
                )
            image_embed = image_embed + self.eye_region_proj(
                region_probs.to(device=image_embed.device, dtype=image_embed.dtype)
            )
        image_pos = self._adaptive_image_pos(leaves, canvas_size, image_embed.dtype)
        image_embed = image_embed + image_pos
        prefix = self._prefix(t, h, y) + self.pos_embed[:, : self.prefix_tokens].to(
            device=image_embed.device, dtype=image_embed.dtype
        )

        if adaptive_layout is not None:
            batch = int(prefix.shape[0])
            padded_image_tokens = int(adaptive_layout["padded_image_tokens"])
            slot_index = adaptive_layout["slot_index"]
            total_slots = batch * padded_image_tokens
            if adaptive_layout.get("dense_image_tokens", False):
                padded_embed = image_embed.reshape(
                    batch, padded_image_tokens, self.hidden_size
                )
            else:
                padded_embed = image_embed.new_zeros(total_slots, self.hidden_size)
                padded_embed = padded_embed.index_copy(0, slot_index, image_embed)
                padded_embed = padded_embed.reshape(
                    batch, padded_image_tokens, self.hidden_size
                )
            sequence = torch.cat([prefix, padded_embed], dim=1)
            return (
                sequence,
                adaptive_layout["key_valid"],
                adaptive_layout["query_groups"],
                adaptive_layout["rope_phases"],
                adaptive_layout["key_area_bias"],
                slot_index,
                adaptive_layout["lengths"],
            )

        batch = int(prefix.shape[0])
        padded_image_tokens = int(padded_image_tokens)
        if padded_image_tokens <= 0:
            raise ValueError("padded_image_tokens must be positive")

        if query_groups_override is not None:
            # High-throughput inference uses a fixed token budget for every sample.
            # Do not read image_offsets CUDA scalars back to Python just to rediscover
            # lengths that are already known from padded_image_tokens.
            lengths = [padded_image_tokens] * batch
            total_slots = batch * padded_image_tokens
            if image_embed.shape[0] != total_slots or leaves.shape[0] != total_slots:
                raise ValueError(
                    "precomputed dense query groups require exactly "
                    "batch*padded_image_tokens image rows"
                )
            slot_index = torch.arange(
                total_slots, device=image_embed.device, dtype=torch.long
            )
            padded_embed = image_embed.reshape(
                batch, padded_image_tokens, self.hidden_size
            )
            padded_leaves = leaves.reshape(batch, padded_image_tokens, 3)
            image_valid = torch.ones(
                batch, padded_image_tokens, device=leaves.device, dtype=torch.bool
            )
        else:
            lengths = [
                int(image_offsets[index + 1]) - int(image_offsets[index])
                for index in range(batch)
            ]
            max_length = max(lengths, default=0)
            if padded_image_tokens < max_length:
                raise ValueError(
                    f"padded_image_tokens={padded_image_tokens} is smaller than batch max {max_length}"
                )
            slot_parts = [
                sample * padded_image_tokens
                + torch.arange(length, device=image_embed.device, dtype=torch.long)
                for sample, length in enumerate(lengths)
            ]
            slot_index = (
                torch.cat(slot_parts, dim=0)
                if slot_parts else
                torch.empty(0, device=image_embed.device, dtype=torch.long)
            )
            total_slots = batch * padded_image_tokens
            padded_embed = image_embed.new_zeros(total_slots, self.hidden_size)
            padded_embed = padded_embed.index_copy(0, slot_index, image_embed)
            padded_embed = padded_embed.reshape(
                batch, padded_image_tokens, self.hidden_size
            )
            padded_leaves = leaves.new_zeros(total_slots, 3)
            padded_leaves[:, 2] = self.patch_size
            padded_leaves = padded_leaves.index_copy(0, slot_index, leaves)
            padded_leaves = padded_leaves.reshape(batch, padded_image_tokens, 3)
            image_valid_flat = torch.zeros(
                total_slots, device=leaves.device, dtype=torch.bool
            )
            image_valid_flat = image_valid_flat.index_fill(0, slot_index, True)
            image_valid = image_valid_flat.reshape(batch, padded_image_tokens)

        sequence = torch.cat([prefix, padded_embed], dim=1)
        prefix_valid = torch.ones(
            batch, self.prefix_tokens, device=leaves.device, dtype=torch.bool
        )
        key_valid = torch.cat([prefix_valid, image_valid], dim=1)

        centers_image = torch.stack(
            [
                padded_leaves[..., 1].float() + padded_leaves[..., 2].float() * 0.5,
                padded_leaves[..., 0].float() + padded_leaves[..., 2].float() * 0.5,
            ],
            dim=-1,
        )
        centers = torch.cat(
            [
                torch.zeros(batch, self.prefix_tokens, 2, device=leaves.device, dtype=torch.float32),
                centers_image,
            ],
            dim=1,
        )
        sizes = torch.cat(
            [
                torch.zeros(batch, self.prefix_tokens, device=leaves.device, dtype=torch.long),
                padded_leaves[..., 2].long(),
            ],
            dim=1,
        )
        is_image = torch.cat(
            [
                torch.zeros(batch, self.prefix_tokens, device=leaves.device, dtype=torch.bool),
                image_valid,
            ],
            dim=1,
        )

        key_area_bias = torch.zeros_like(sizes, dtype=torch.float32)
        safe_sizes = sizes.clamp_min(1).float()
        key_area_bias = torch.where(
            is_image,
            2.0 * torch.log(safe_sizes / float(self.patch_size)),
            key_area_bias,
        )

        levels = (
            self.patch_size // 4,
            self.patch_size // 2,
            self.patch_size,
            self.patch_size * 2,
        )
        rope_phases = []
        for level in levels:
            positions = torch.where(
                is_image[..., None],
                centers / float(level) - 0.5,
                torch.zeros_like(centers),
            )
            rope_phases.append(_rope_phase(self.hidden_size // self.num_heads, positions))

        if query_groups_override is None:
            prefix_query = torch.zeros_like(key_valid)
            prefix_query[:, : self.prefix_tokens] = True
            group_masks = [prefix_query]
            group_masks.extend(is_image & (sizes == level) for level in levels)
            query_groups = tuple(
                self._make_padded_query_group(mask) for mask in group_masks
            )
        else:
            query_groups = tuple(query_groups_override)
            if len(query_groups) != 5:
                raise ValueError(
                    f"expected 5 adaptive query groups, got {len(query_groups)}"
                )

        return (
            sequence,
            key_valid,
            query_groups,
            tuple(rope_phases),
            key_area_bias,
            slot_index,
            tuple(lengths),
        )

    def _run_padded_block(
        self,
        block: TransformerBlock,
        sequence: torch.Tensor,
        key_valid: torch.Tensor,
        query_groups,
        rope_phases,
        key_area_bias: torch.Tensor,
        all_keys_valid: bool = False,
    ) -> torch.Tensor:
        def fn(value):
            return block.forward_padded(
                value,
                key_valid,
                query_groups,
                rope_phases,
                key_area_bias,
                all_keys_valid=all_keys_valid,
            )
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return activation_checkpoint(fn, sequence, use_reentrant=False)
        return fn(sequence)

    def encode_adaptive(
        self,
        canonical_patches: torch.Tensor,
        image_offsets: torch.Tensor | Sequence[int],
        leaves: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        y: torch.Tensor,
        canvas_size: int,
        padded_image_tokens: int,
        adaptive_layout=None,
    ) -> torch.Tensor:
        """Return pretrained PMF semantic image-token features without RGB heads.

        This legacy helper reuses the shared transformer and U/refinement blocks
        but deliberately skips image-space output heads.  V26's active MeanFlow
        generator does not use this helper: it calls ``forward_adaptive`` so the
        learned U/x clean-proposal head remains in the generative path.
        """
        (
            sequence,
            key_valid,
            query_groups,
            rope_phases,
            key_area_bias,
            slot_index,
            lengths,
        ) = self.build_adaptive_sequence(
            canonical_patches,
            image_offsets,
            leaves,
            t,
            h,
            y,
            canvas_size,
            padded_image_tokens,
            adaptive_layout=adaptive_layout,
        )
        dense_image_tokens = (
            bool(adaptive_layout.get("dense_image_tokens", False))
            if adaptive_layout is not None
            else bool(lengths and all(int(v) == int(padded_image_tokens) for v in lengths))
        )
        all_keys_valid = (
            bool(adaptive_layout.get("all_keys_valid", False))
            if adaptive_layout is not None
            else dense_image_tokens
        )
        for block in self.shared_blocks:
            sequence = self._run_padded_block(
                block,
                sequence,
                key_valid,
                query_groups,
                rope_phases,
                key_area_bias,
                all_keys_valid=all_keys_valid,
            )
        u_seq = sequence
        for block in self.u_heads:
            u_seq = self._run_padded_block(
                block,
                u_seq,
                key_valid,
                query_groups,
                rope_phases,
                key_area_bias,
                all_keys_valid=all_keys_valid,
            )
        dense_u = u_seq[:, self.prefix_tokens :].reshape(-1, self.hidden_size)
        return dense_u if dense_image_tokens else dense_u.index_select(0, slot_index)

    def forward_adaptive(
        self,
        canonical_patches: torch.Tensor,
        image_offsets: torch.Tensor,
        leaves: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        y: torch.Tensor,
        canvas_size: int,
        need_v: bool = True,
        padded_image_tokens: Optional[int] = None,
        adaptive_layout=None,
        region_probs: Optional[torch.Tensor] = None,
        return_u_latents: bool = False,
        image_tokens_override: Optional[torch.Tensor] = None,
        query_groups_override=None,
    ):
        if padded_image_tokens is None:
            padded_image_tokens = max(
                int(image_offsets[index + 1]) - int(image_offsets[index])
                for index in range(int(image_offsets.numel() - 1))
            )
        (
            sequence,
            key_valid,
            query_groups,
            rope_phases,
            key_area_bias,
            slot_index,
            _lengths,
        ) = self.build_adaptive_sequence(
            canonical_patches, image_offsets, leaves, t, h, y, canvas_size,
            padded_image_tokens, adaptive_layout=adaptive_layout,
            region_probs=region_probs, image_tokens_override=image_tokens_override,
            query_groups_override=query_groups_override,
        )
        dense_image_tokens = (
            bool(adaptive_layout.get("dense_image_tokens", False))
            if adaptive_layout is not None
            else bool(_lengths and all(length == int(padded_image_tokens) for length in _lengths))
        )
        all_keys_valid = (
            bool(adaptive_layout.get("all_keys_valid", False))
            if adaptive_layout is not None
            else dense_image_tokens
        )
        for block in self.shared_blocks:
            sequence = self._run_padded_block(
                block, sequence, key_valid, query_groups, rope_phases, key_area_bias,
                all_keys_valid=all_keys_valid,
            )
        u_seq = sequence
        for block in self.u_heads:
            u_seq = self._run_padded_block(
                block, u_seq, key_valid, query_groups, rope_phases, key_area_bias,
                all_keys_valid=all_keys_valid,
            )
        dense_u = u_seq[:, self.prefix_tokens :].reshape(-1, self.hidden_size)
        u_tokens = dense_u if dense_image_tokens else dense_u.index_select(0, slot_index)
        p = self.canonical_size
        u = self.u_final_layer(u_tokens).reshape(-1, p, p, self.out_channels).permute(0, 3, 1, 2)
        v = None
        if need_v:
            if not self.enable_v_head:
                raise RuntimeError("this pmfDiT instance was constructed with enable_v_head=False")
            v_seq = sequence
            for block in self.v_heads:
                v_seq = self._run_padded_block(
                    block, v_seq, key_valid, query_groups, rope_phases, key_area_bias,
                    all_keys_valid=all_keys_valid,
                )
            dense_v = v_seq[:, self.prefix_tokens :].reshape(-1, self.hidden_size)
            v_tokens = dense_v if dense_image_tokens else dense_v.index_select(0, slot_index)
            v = self.v_final_layer(v_tokens).reshape(-1, p, p, self.out_channels).permute(0, 3, 1, 2)
        if return_u_latents:
            return u, v, u_tokens
        return u, v

    @torch.no_grad()
    def adaptive_attention_diagnostics(
        self,
        canonical_patches: torch.Tensor,
        image_offsets: torch.Tensor,
        leaves: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        y: torch.Tensor,
        canvas_size: int,
        layer: int = -1,
        padded_image_tokens: Optional[int] = None,
        adaptive_layout=None,
    ):
        """Capture exact dense-padded self-attention at one shared pMF block."""
        if padded_image_tokens is None:
            padded_image_tokens = max(
                int(image_offsets[index + 1]) - int(image_offsets[index])
                for index in range(int(image_offsets.numel() - 1))
            )
        (
            sequence,
            key_valid,
            query_groups,
            rope_phases,
            key_area_bias,
            _slot_index,
            lengths,
        ) = self.build_adaptive_sequence(
            canonical_patches, image_offsets, leaves, t, h, y, canvas_size,
            padded_image_tokens, adaptive_layout=adaptive_layout
        )
        depth = len(self.shared_blocks)
        if depth == 0:
            raise RuntimeError("attention diagnostics require at least one shared block")
        resolved = int(layer)
        if resolved < 0:
            resolved += depth
        if not 0 <= resolved < depth:
            raise ValueError(f"attention layer {layer} resolves to {resolved}, depth={depth}")

        all_keys_valid = (
            bool(adaptive_layout.get("all_keys_valid", False))
            if adaptive_layout is not None
            else bool(lengths and all(length == int(padded_image_tokens) for length in lengths))
        )
        dense_matrix = None
        for index, block in enumerate(self.shared_blocks):
            if index == resolved:
                dense_matrix = block.attn.padded_attention_weights(
                    block.norm1(sequence),
                    key_valid,
                    query_groups,
                    rope_phases,
                    key_area_bias,
                )
            sequence = self._run_padded_block(
                block, sequence, key_valid, query_groups, rope_phases, key_area_bias,
                all_keys_valid=all_keys_valid,
            )
            if index >= resolved:
                break
        if dense_matrix is None:
            raise RuntimeError("failed to capture attention diagnostics")
        matrices = tuple(
            dense_matrix[sample, : self.prefix_tokens + length, : self.prefix_tokens + length]
            for sample, length in enumerate(lengths)
        )
        area_rows = tuple(
            key_area_bias[sample, : self.prefix_tokens + length].detach().float()
            for sample, length in enumerate(lengths)
        )
        return {
            "layer": resolved,
            "prefix_tokens": self.prefix_tokens,
            "attention": matrices,
            "key_area_bias": area_rows,
        }


def precompute_rope_freqs(dim: int, grid_size, theta: float = 10000.0):
    dim = dim // 2
    if isinstance(grid_size, int):
        side = int(round(math.sqrt(grid_size)))
        if side * side != int(grid_size):
            raise ValueError("integer grid_size must be square")
        height = width = side
    else:
        height, width = map(int, grid_size)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions_h = torch.arange(height, dtype=torch.float32)
    positions_w = torch.arange(width, dtype=torch.float32)
    freqs_h = torch.einsum("i,j->ij", positions_h, freqs)
    freqs_w = torch.einsum("i,j->ij", positions_w, freqs)
    phase = torch.cat(
        [
            torch.tile(freqs_h[:, None, :], (1, width, 1)),
            torch.tile(freqs_w[None, :, :], (height, 1, 1)),
        ],
        dim=-1,
    )
    count = height * width
    return torch.complex(torch.cos(phase).reshape(count, dim), torch.sin(phase).reshape(count, dim))


def apply_rotary_pos_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_float = x.float()
    x_complex = torch.view_as_complex(
        x_float.reshape(*x_float.shape[:-1], -1, 2).contiguous()
    )
    freqs_cis = freqs_cis[None, :, None, :]
    count = freqs_cis.shape[1]
    rotated = x_complex.clone()
    rotated[:, -count:] = x_complex[:, -count:] * freqs_cis
    return torch.view_as_real(rotated).flatten(-2).to(x.dtype)


pmfDiT_B_16 = partial(pmfDiT, input_size=256, depth=16, hidden_size=768, patch_size=16, num_heads=12, aux_head_depth=8)


pmfDiT_B_32 = partial(pmfDiT, input_size=512, depth=16, hidden_size=768, patch_size=32, num_heads=12, aux_head_depth=8)


pmfDiT_L_16 = partial(pmfDiT, input_size=256, depth=32, hidden_size=1024, patch_size=16, num_heads=16, aux_head_depth=8)


pmfDiT_L_32 = partial(pmfDiT, input_size=512, depth=32, hidden_size=1024, patch_size=32, num_heads=16, aux_head_depth=8)


pmfDiT_H_16 = partial(pmfDiT, input_size=256, depth=48, hidden_size=1280, patch_size=16, num_heads=16, aux_head_depth=8)


pmfDiT_H_32 = partial(pmfDiT, input_size=512, depth=48, hidden_size=1280, patch_size=32, num_heads=16, aux_head_depth=8)


pmfDiT_Tiny_4 = partial(pmfDiT, input_size=32, depth=4, hidden_size=96, patch_size=4, num_heads=3, aux_head_depth=2, num_class_tokens=4, num_time_tokens=2, pca_channels=32)


pmfDiT_Nano_4 = partial(pmfDiT, input_size=16, depth=2, hidden_size=32, patch_size=4, num_heads=2, aux_head_depth=1, num_class_tokens=2, num_time_tokens=2, num_cfg_tokens=0, num_interval_tokens=0, pca_channels=8)


# --- inlined from models/spatial.py ---
LANDMARK_SCHEMA = "lrda_8ch_v1"


GEOMETRY_VERSION = "landmark_detail_adapter_v3_multiface"


def source_signature(path: Path) -> dict:
    """Stable signature used to bind cached geometry to its exact source bytes."""
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"algorithm": "sha256", "digest": digest.hexdigest(), "size_bytes": size}


@dataclass(frozen=True)
class AdaptiveLeaf:
    x: int
    y: int
    size: int
    reason: str = ""


@dataclass(frozen=True)
class AdaptiveGridConfig:
    """Geometry-only exact-budget quadtree configuration.

    The default 256-token budget matches the official pMF-B/16 image-token
    length on a 256x256 canvas.  The image starts as 64 independent 32x32
    quadtree roots.  Dynamic programming chooses an exact-budget forest with
    leaves in {4, 8, 16, 32}.
    """

    canvas_size: int = 256
    base_patch_size: int = 16
    target_tokens: int = 256

    # Additive per-pixel importance weights.  canvas_weight is deliberately
    # positive: without geometry, exact-budget DP returns the official uniform
    # 16x16 grid rather than placing arbitrary extra splits in zero-cost areas.
    canvas_weight: float = 0.02
    face_weight: float = 0.20
    periocular_weight: float = 1.00
    eye_weight: float = 3.00
    iris_weight: float = 8.00
    eyelid_weight: float = 5.00
    lens_weight: float = 4.00

    # V29.4: estimate a glasses/lens band from eye contours. This is geometry-only
    # inference-time information. The band receives high token density so thin
    # high-frequency reflections are less likely to fall into p16/p32 leaves.
    lens_expand_x: float = 0.45
    lens_expand_y: float = 1.00
    hard_lens_max_patch: int = 8

    # Legacy eye-prior support is retained for importance/diagnostics. V29.4 uses
    # a more targeted geometry-derived spectacle lens band as the preferred hard
    # p4/p8 capacity constraint.
    correction_threshold: int = 64
    correction_dilate_radius: int = 1
    hard_correction_max_patch: int = 8

    # Sparse eye anchors prevent the geometry prior from forcing the entire eye
    # region to the finest scale.  Only landmark-centered pixels are constrained
    # to <= 8x8; the remaining exact token budget is allocated by geometry cost.
    hard_anchor_radius: int = 1
    hard_anchor_max_patch: int = 8

    eyelid_band_radius: int = 2
    iris_dilate_radius: int = 1
    enforce_hard_eye_constraints: bool = True
    hard_iris_max_patch: int = 8
    hard_eyelid_max_patch: int = 8
    relax_hard_constraints_if_infeasible: bool = True

    head_expand_left: float = 0.65
    head_expand_right: float = 0.65
    head_expand_top: float = 1.40
    head_expand_bottom: float = 3.00

    def __post_init__(self) -> None:
        p = int(self.base_patch_size)
        if p <= 0 or p % 4:
            raise ValueError("base_patch_size must be a positive multiple of 4")
        if self.canvas_size <= 0 or self.canvas_size % (2 * p):
            raise ValueError("canvas_size must be divisible by 2 * base_patch_size")
        if int(self.target_tokens) <= 0:
            raise ValueError("target_tokens must be positive")
        root_tokens = (self.canvas_size // (2 * p)) ** 2
        max_tokens = (self.canvas_size // (p // 4)) ** 2
        if not root_tokens <= self.target_tokens <= max_tokens:
            raise ValueError(
                f"target_tokens must be in [{root_tokens}, {max_tokens}], got {self.target_tokens}"
            )
        # A four-way split changes the number of leaves by +3.
        if (self.target_tokens - root_tokens) % 3:
            raise ValueError(
                "target_tokens is unreachable for this quadtree forest: "
                f"({self.target_tokens} - {root_tokens}) must be divisible by 3"
            )
        allowed = set(self.patch_sizes)
        if self.hard_iris_max_patch not in allowed:
            raise ValueError("hard_iris_max_patch must be one of patch_sizes")
        if self.hard_eyelid_max_patch not in allowed:
            raise ValueError("hard_eyelid_max_patch must be one of patch_sizes")
        if self.hard_anchor_max_patch not in allowed:
            raise ValueError("hard_anchor_max_patch must be one of patch_sizes")
        if self.hard_correction_max_patch not in allowed:
            raise ValueError("hard_correction_max_patch must be one of patch_sizes")
        if self.hard_lens_max_patch not in allowed:
            raise ValueError("hard_lens_max_patch must be one of patch_sizes")
        if not 0 <= int(self.correction_threshold) <= 255:
            raise ValueError("correction_threshold must be in [0,255]")
        if int(self.correction_dilate_radius) < 0:
            raise ValueError("correction_dilate_radius must be non-negative")
        if int(self.hard_anchor_radius) < 0:
            raise ValueError("hard_anchor_radius must be non-negative")
        if int(self.eyelid_band_radius) < 0 or int(self.iris_dilate_radius) < 0:
            raise ValueError("eyelid_band_radius and iris_dilate_radius must be non-negative")
        for name in (
            "canvas_weight", "face_weight", "periocular_weight", "eye_weight",
            "iris_weight", "eyelid_weight", "lens_weight", "lens_expand_x", "lens_expand_y",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def patch4(self) -> int:
        return self.base_patch_size // 4

    @property
    def patch8(self) -> int:
        return self.base_patch_size // 2

    @property
    def patch16(self) -> int:
        return self.base_patch_size

    @property
    def patch32(self) -> int:
        return self.base_patch_size * 2

    @property
    def patch_sizes(self) -> Tuple[int, int, int, int]:
        return self.patch4, self.patch8, self.patch16, self.patch32


@dataclass
class GridBuildResult:
    leaves: List[AdaptiveLeaf]
    geometry_success: bool
    geometry_source: str
    target_token_count: int
    dp_total_cost: float
    patch4_count: int
    patch8_count: int
    patch16_count: int
    patch32_count: int
    eye_token_count: int
    hard_constraints_relaxed: bool
    internal_splits: int
    correction_token_count: int = 0
    anchor_token_count: int = 0
    lens_token_count: int = 0
    constraint_mode: str = "legacy_eye"

    # Backward-compatible aliases retained so old logging/checkpoint utilities
    # can read a v15 layout record during migration.
    mandatory_token_count: int = 0
    hard_splits_used: int = 0
    soft_candidates_count: int = 0
    optional_splits_used: int = 0
    budget_clipped: bool = False


def build_uniform_grid(cfg: AdaptiveGridConfig) -> GridBuildResult:
    """Return the exact 16x16 p16 baseline with the same 256-token budget."""
    patch = int(cfg.base_patch_size)
    side = int(cfg.canvas_size)
    leaves = [
        AdaptiveLeaf(x, y, patch, "uniform_grid")
        for y in range(0, side, patch)
        for x in range(0, side, patch)
    ]
    if len(leaves) != int(cfg.target_tokens):
        raise ValueError(
            "uniform grid requires target_tokens=(canvas/base_patch)^2; "
            f"got {len(leaves)} leaves and target_tokens={cfg.target_tokens}"
        )
    root_tokens = (side // (2 * patch)) ** 2
    return GridBuildResult(
        leaves=leaves,
        geometry_success=True,
        geometry_source="uniform_p16",
        target_token_count=len(leaves),
        dp_total_cost=0.0,
        patch4_count=0,
        patch8_count=0,
        patch16_count=len(leaves),
        patch32_count=0,
        eye_token_count=0,
        hard_constraints_relaxed=False,
        internal_splits=(len(leaves) - root_tokens) // 3,
        correction_token_count=0,
        anchor_token_count=0,
        lens_token_count=0,
        constraint_mode="uniform_p16",
    )


@dataclass(frozen=True)
class _DPDecision:
    cost: float
    child_budgets: Optional[Tuple[int, int, int, int]]


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, :, None]
    tensor = torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0
    return tensor.contiguous()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach().float().clamp(-1, 1).add(1).mul(127.5)
        .permute(1, 2, 0).cpu().numpy().round().astype(np.uint8)
    )
    return Image.fromarray(array)


def square_edge_canvas(image: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Pad an original image to a square without resizing, using edge replication."""
    image = image.convert("RGB")
    width, height = image.size
    side = max(width, height)
    pad_left = (side - width) // 2
    pad_top = (side - height) // 2
    pad_right = side - width - pad_left
    pad_bottom = side - height - pad_top
    arr = np.asarray(image)
    arr = np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="edge",
    )
    return Image.fromarray(arr), (pad_left, pad_top, pad_left + width, pad_top + height)


def square_zero_canvas(mask: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    mask = mask.convert("L")
    width, height = mask.size
    side = max(width, height)
    pad_left = (side - width) // 2
    pad_top = (side - height) // 2
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(mask, (pad_left, pad_top))
    return canvas, (pad_left, pad_top, pad_left + width, pad_top + height)


def logical_canvas(image: Image.Image, canvas_size: int, is_mask: bool = False) -> Image.Image:
    if is_mask:
        square, _ = square_zero_canvas(image)
        resample = Image.Resampling.NEAREST
    else:
        square, _ = square_edge_canvas(image)
        resample = Image.Resampling.BICUBIC
    return square.resize((canvas_size, canvas_size), resample=resample)


def _rect_mask(rect: Tuple[int, int, int, int], size: int) -> np.ndarray:
    out = np.zeros((size, size), dtype=np.uint8)
    x1, y1, x2, y2 = rect
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = 1
    return out


def _clip_rect(values: Sequence[float], size: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, values)
    left = max(0, min(size, int(math.floor(x1))))
    top = max(0, min(size, int(math.floor(y1))))
    right = max(left, min(size, int(math.ceil(x2))))
    bottom = max(top, min(size, int(math.ceil(y2))))
    return left, top, right, bottom


def _roi_to_logical_rect(
    roi: Sequence[float], original_size: Tuple[int, int], canvas_size: int
) -> Tuple[int, int, int, int]:
    width, height = map(int, original_size)
    side = max(width, height)
    pad_x = (side - width) / 2.0
    pad_y = (side - height) / 2.0
    x1, y1, x2, y2 = map(float, roi)
    scale = canvas_size / float(side)
    return _clip_rect(
        (
            (pad_x + x1 * width) * scale,
            (pad_y + y1 * height) * scale,
            (pad_x + x2 * width) * scale,
            (pad_y + y2 * height) * scale,
        ),
        canvas_size,
    )


def _face_metadata_entries(metadata: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Return per-face metadata when available, otherwise an empty list."""
    faces = metadata.get("faces", [])
    if not isinstance(faces, (list, tuple)):
        return []
    return [face for face in faces if isinstance(face, Mapping)]


def _landmark_group_entries(metadata: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Return one landmark-group mapping per detected face.

    Multi-face v3 metadata stores a ``faces`` list.  Older single-face metadata
    stores only the top-level ``landmark_groups_normalized`` mapping.  Supporting
    both keeps old checkpoints independent from the geometry-cache format.
    """
    groups = []
    for face in _face_metadata_entries(metadata):
        value = face.get("landmark_groups_normalized", {})
        if isinstance(value, Mapping):
            groups.append(value)
    if groups:
        return groups
    value = metadata.get("landmark_groups_normalized", {})
    return [value] if isinstance(value, Mapping) else []


def _head_mask(
    metadata: Mapping[str, object], original_size: Tuple[int, int], cfg: AdaptiveGridConfig
) -> Tuple[np.ndarray, str]:
    # Multi-face path: build one expanded face-context rectangle per valid face and
    # take their union.  This avoids turning the space between distant people into
    # one giant face-context rectangle.
    face_entries = _face_metadata_entries(metadata)
    if face_entries:
        union = np.zeros((cfg.canvas_size, cfg.canvas_size), dtype=np.uint8)
        made = False
        for face in face_entries:
            explicit = None
            for key in (
                "head_protect_box_normalized",
                "head_bbox_normalized",
                "face_bbox_normalized",
                "face_box_normalized",
            ):
                value = face.get(key)
                if isinstance(value, (list, tuple)) and len(value) == 4:
                    explicit = value
                    break
            if explicit is not None:
                rect = _roi_to_logical_rect(explicit, original_size, cfg.canvas_size)
            else:
                roi = face.get("roi_normalized")
                if not isinstance(roi, (list, tuple)) or len(roi) != 4:
                    continue
                x1, y1, x2, y2 = _roi_to_logical_rect(
                    roi, original_size, cfg.canvas_size
                )
                span_x = max(1.0, x2 - x1)
                span_y = max(1.0, y2 - y1)
                rect = _clip_rect(
                    (
                        x1 - cfg.head_expand_left * span_x,
                        y1 - cfg.head_expand_top * span_y,
                        x2 + cfg.head_expand_right * span_x,
                        y2 + cfg.head_expand_bottom * span_y,
                    ),
                    cfg.canvas_size,
                )
            union = np.maximum(union, _rect_mask(rect, cfg.canvas_size))
            made = True
        if made:
            return union, "multi_face_roi_union"

    # Backward-compatible single-face path.
    explicit_keys = (
        "head_protect_box_normalized",
        "head_bbox_normalized",
        "face_bbox_normalized",
        "face_box_normalized",
    )
    for key in explicit_keys:
        value = metadata.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 4:
            rect = _roi_to_logical_rect(value, original_size, cfg.canvas_size)
            return _rect_mask(rect, cfg.canvas_size), "explicit_meta_bbox"

    roi = metadata.get("roi_normalized")
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        return np.ones((cfg.canvas_size, cfg.canvas_size), dtype=np.uint8), "full_canvas_fallback"
    x1, y1, x2, y2 = _roi_to_logical_rect(roi, original_size, cfg.canvas_size)
    span_x = max(1.0, x2 - x1)
    span_y = max(1.0, y2 - y1)
    rect = _clip_rect(
        (
            x1 - cfg.head_expand_left * span_x,
            y1 - cfg.head_expand_top * span_y,
            x2 + cfg.head_expand_right * span_x,
            y2 + cfg.head_expand_bottom * span_y,
        ),
        cfg.canvas_size,
    )
    return _rect_mask(rect, cfg.canvas_size), "roi_expand_fallback"


def _morph_gradient(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.zeros_like(mask, dtype=np.uint8)
    k = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return (cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0).astype(np.uint8)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0).astype(np.uint8)
    k = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return (cv2.dilate(mask.astype(np.uint8), kernel) > 0).astype(np.uint8)


def _split(leaf: AdaptiveLeaf, reason: str) -> List[AdaptiveLeaf]:
    child = leaf.size // 2
    return [
        AdaptiveLeaf(leaf.x, leaf.y, child, reason),
        AdaptiveLeaf(leaf.x + child, leaf.y, child, reason),
        AdaptiveLeaf(leaf.x, leaf.y + child, child, reason),
        AdaptiveLeaf(leaf.x + child, leaf.y + child, child, reason),
    ]


def uniform_grid(cfg: AdaptiveGridConfig) -> List[AdaptiveLeaf]:
    p = cfg.base_patch_size
    return [
        AdaptiveLeaf(x, y, p, "uniform_geometry_failure")
        for y in range(0, cfg.canvas_size, p)
        for x in range(0, cfg.canvas_size, p)
    ]


def validate_grid(leaves: Sequence[AdaptiveLeaf], cfg: AdaptiveGridConfig) -> None:
    canvas = np.zeros((cfg.canvas_size, cfg.canvas_size), dtype=np.int16)
    allowed = set(cfg.patch_sizes)
    for leaf in leaves:
        if leaf.size not in allowed:
            raise ValueError(f"invalid leaf size {leaf.size}")
        if leaf.x % leaf.size or leaf.y % leaf.size:
            raise ValueError(f"misaligned leaf {leaf}")
        if leaf.x < 0 or leaf.y < 0 or leaf.x + leaf.size > cfg.canvas_size or leaf.y + leaf.size > cfg.canvas_size:
            raise ValueError(f"out-of-bounds leaf {leaf}")
        canvas[leaf.y : leaf.y + leaf.size, leaf.x : leaf.x + leaf.size] += 1
    if np.any(canvas == 0):
        raise ValueError("adaptive grid has holes")
    if np.any(canvas > 1):
        raise ValueError("adaptive grid has overlaps")
    if len(leaves) != cfg.target_tokens:
        raise ValueError(
            f"exact-budget grid has {len(leaves)} leaves, expected {cfg.target_tokens}"
        )


def _integral(array: np.ndarray) -> np.ndarray:
    return cv2.integral(array.astype(np.float64))


def _integral_sum(integral: np.ndarray, x: int, y: int, size: int) -> float:
    x2, y2 = x + size, y + size
    return float(integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x])


def _combine_children(
    child_tables: Sequence[Dict[int, _DPDecision]],
    budget_cap: int,
) -> Dict[int, Tuple[float, Tuple[int, ...]]]:
    """Sparse min-plus convolution over four child budget curves."""
    combined: Dict[int, Tuple[float, Tuple[int, ...]]] = {0: (0.0, ())}
    for table in child_tables:
        next_combined: Dict[int, Tuple[float, Tuple[int, ...]]] = {}
        for used, (prefix_cost, prefix_alloc) in combined.items():
            for child_budget, decision in table.items():
                total = used + child_budget
                if total > budget_cap:
                    continue
                candidate = prefix_cost + decision.cost
                previous = next_combined.get(total)
                allocation = prefix_alloc + (child_budget,)
                if previous is None or candidate < previous[0] - 1.0e-12:
                    next_combined[total] = (candidate, allocation)
        combined = next_combined
    return combined


def _solve_exact_budget(
    importance: np.ndarray,
    iris_hard: np.ndarray,
    eyelid_hard: np.ndarray,
    cfg: AdaptiveGridConfig,
    enforce_hard: bool,
    correction_hard: Optional[np.ndarray] = None,
    anchor_hard: Optional[np.ndarray] = None,
    lens_hard: Optional[np.ndarray] = None,
) -> Tuple[List[AdaptiveLeaf], float]:
    importance_integral = _integral(importance)
    iris_integral = _integral(iris_hard)
    eyelid_integral = _integral(eyelid_hard)
    correction_integral = _integral(
        np.zeros_like(iris_hard) if correction_hard is None else correction_hard
    )
    anchor_integral = _integral(
        np.zeros_like(iris_hard) if anchor_hard is None else anchor_hard
    )
    lens_integral = _integral(
        np.zeros_like(iris_hard) if lens_hard is None else lens_hard
    )
    tables: Dict[Tuple[int, int, int], Dict[int, _DPDecision]] = {}

    def solve_node(x: int, y: int, size: int) -> Dict[int, _DPDecision]:
        key = (x, y, size)
        cached = tables.get(key)
        if cached is not None:
            return cached

        scale_penalty = math.log2(float(size) / float(cfg.patch4))
        leaf_cost = scale_penalty * _integral_sum(importance_integral, x, y, size)
        leaf_allowed = True
        if enforce_hard:
            if size > cfg.hard_iris_max_patch and _integral_sum(iris_integral, x, y, size) > 0:
                leaf_allowed = False
            if size > cfg.hard_eyelid_max_patch and _integral_sum(eyelid_integral, x, y, size) > 0:
                leaf_allowed = False
            if (
                correction_hard is not None
                and size > cfg.hard_correction_max_patch
                and _integral_sum(correction_integral, x, y, size) > 0
            ):
                leaf_allowed = False
        # Structural anchors are independent of the dense eye-constraint switch:
        # if supplied, they remain hard until the caller explicitly omits them.
        if (
            anchor_hard is not None
            and size > cfg.hard_anchor_max_patch
            and _integral_sum(anchor_integral, x, y, size) > 0
        ):
            leaf_allowed = False
        if (
            lens_hard is not None
            and size > cfg.hard_lens_max_patch
            and _integral_sum(lens_integral, x, y, size) > 0
        ):
            leaf_allowed = False

        table: Dict[int, _DPDecision] = {}
        if leaf_allowed:
            table[1] = _DPDecision(leaf_cost, None)

        if size > cfg.patch4:
            child = size // 2
            child_keys = (
                (x, y, child),
                (x + child, y, child),
                (x, y + child, child),
                (x + child, y + child, child),
            )
            child_tables = [solve_node(*child_key) for child_key in child_keys]
            combined = _combine_children(child_tables, cfg.target_tokens)
            for budget, (cost, allocation) in combined.items():
                if len(allocation) != 4:
                    continue
                previous = table.get(budget)
                if previous is None or cost < previous.cost - 1.0e-12:
                    table[budget] = _DPDecision(cost, tuple(map(int, allocation)))
        tables[key] = table
        return table

    roots = [
        (x, y, cfg.patch32)
        for y in range(0, cfg.canvas_size, cfg.patch32)
        for x in range(0, cfg.canvas_size, cfg.patch32)
    ]
    root_tables = [solve_node(*root) for root in roots]

    # Exact-budget forest knapsack.  remaining-root lower bounds prune states
    # that can no longer reach the target.
    global_costs: Dict[int, float] = {0: 0.0}
    backpointers: List[Dict[int, Tuple[int, int]]] = []
    total_roots = len(roots)
    for root_index, table in enumerate(root_tables):
        remaining = total_roots - root_index - 1
        next_costs: Dict[int, float] = {}
        next_back: Dict[int, Tuple[int, int]] = {}
        for used, prefix_cost in global_costs.items():
            for local_budget, decision in table.items():
                total = used + local_budget
                if total + remaining > cfg.target_tokens:
                    continue
                if total > cfg.target_tokens:
                    continue
                candidate = prefix_cost + decision.cost
                previous = next_costs.get(total)
                if previous is None or candidate < previous - 1.0e-12:
                    next_costs[total] = candidate
                    next_back[total] = (used, local_budget)
        global_costs = next_costs
        backpointers.append(next_back)

    if cfg.target_tokens not in global_costs:
        raise RuntimeError(
            "No exact-budget quadtree satisfies the current hard constraints "
            f"for target_tokens={cfg.target_tokens}"
        )

    root_budgets = [0] * len(roots)
    cursor = cfg.target_tokens
    for index in range(len(roots) - 1, -1, -1):
        previous, local_budget = backpointers[index][cursor]
        root_budgets[index] = local_budget
        cursor = previous
    if cursor != 0:
        raise AssertionError("invalid exact-budget backtrace")

    leaves: List[AdaptiveLeaf] = []

    def recover(x: int, y: int, size: int, budget: int) -> None:
        decision = tables[(x, y, size)][budget]
        if decision.child_budgets is None:
            leaves.append(AdaptiveLeaf(x, y, size, "exact_budget_dp"))
            return
        child = size // 2
        child_keys = (
            (x, y, child),
            (x + child, y, child),
            (x, y + child, child),
            (x + child, y + child, child),
        )
        for child_key, child_budget in zip(child_keys, decision.child_budgets):
            recover(*child_key, child_budget)

    for root, budget in zip(roots, root_budgets):
        recover(*root, budget)
    return leaves, float(global_costs[cfg.target_tokens])


def _solve_exact_budget_fast(
    importance: np.ndarray,
    iris_hard: np.ndarray,
    eyelid_hard: np.ndarray,
    cfg: AdaptiveGridConfig,
    enforce_hard: bool,
    correction_hard: Optional[np.ndarray] = None,
    anchor_hard: Optional[np.ndarray] = None,
    lens_hard: Optional[np.ndarray] = None,
) -> Tuple[List[AdaptiveLeaf], float]:
    """Exact-budget heap solver specialized to the current scale-linear cost.

    For a node with importance mass M and size s, the legacy leaf cost is
    ``log2(s / patch4) * M``. Splitting once lowers the scale penalty by exactly
    one for all four children, whose masses sum to M; therefore the benefit of
    that split is exactly M. Since importance is non-negative, descendants never
    have larger benefit than their ancestors, so the fixed-size ancestor-closed
    optimum is obtained by repeatedly splitting the available node with maximum
    importance mass. Hard max-patch constraints are expanded first as mandatory
    splits.
    """
    importance = np.asarray(importance, dtype=np.float64)
    expected_shape = (cfg.canvas_size, cfg.canvas_size)
    if importance.shape != expected_shape:
        raise ValueError(f"importance must be {expected_shape}, got {importance.shape}")
    minimum = float(np.min(importance))
    if minimum < -1.0e-12:
        raise ValueError(
            "fast exact-budget solver requires non-negative importance; "
            f"minimum={minimum}"
        )

    importance_integral = _integral(importance)
    iris_integral = _integral(iris_hard)
    eyelid_integral = _integral(eyelid_hard)
    correction_integral = _integral(
        np.zeros_like(iris_hard) if correction_hard is None else correction_hard
    )
    anchor_integral = _integral(
        np.zeros_like(iris_hard) if anchor_hard is None else anchor_hard
    )
    lens_integral = _integral(
        np.zeros_like(iris_hard) if lens_hard is None else lens_hard
    )

    def intersects(integral: np.ndarray, x: int, y: int, size: int) -> bool:
        return _integral_sum(integral, x, y, size) > 0.0

    def must_split(x: int, y: int, size: int) -> bool:
        if size <= cfg.patch4:
            return False
        if enforce_hard:
            if size > cfg.hard_iris_max_patch and intersects(iris_integral, x, y, size):
                return True
            if size > cfg.hard_eyelid_max_patch and intersects(eyelid_integral, x, y, size):
                return True
            if (
                correction_hard is not None
                and size > cfg.hard_correction_max_patch
                and intersects(correction_integral, x, y, size)
            ):
                return True
        # Preserve legacy semantics: anchors/lens remain hard independently of
        # the dense eye constraint switch.
        if (
            anchor_hard is not None
            and size > cfg.hard_anchor_max_patch
            and intersects(anchor_integral, x, y, size)
        ):
            return True
        if (
            lens_hard is not None
            and size > cfg.hard_lens_max_patch
            and intersects(lens_integral, x, y, size)
        ):
            return True
        return False

    roots = [
        (x, y, cfg.patch32)
        for y in range(0, cfg.canvas_size, cfg.patch32)
        for x in range(0, cfg.canvas_size, cfg.patch32)
    ]
    root_count = len(roots)
    split_budget = (int(cfg.target_tokens) - root_count) // 3

    active: set[Tuple[int, int, int]] = set()
    mandatory_splits = 0

    def force_constraints(x: int, y: int, size: int) -> None:
        nonlocal mandatory_splits
        if must_split(x, y, size):
            mandatory_splits += 1
            if mandatory_splits > split_budget:
                raise RuntimeError(
                    "No exact-budget quadtree satisfies the current hard constraints "
                    f"for target_tokens={cfg.target_tokens}"
                )
            child = size // 2
            force_constraints(x, y, child)
            force_constraints(x + child, y, child)
            force_constraints(x, y + child, child)
            force_constraints(x + child, y + child, child)
        else:
            active.add((x, y, size))

    for root in roots:
        force_constraints(*root)

    remaining_splits = split_budget - mandatory_splits
    heap: List[Tuple[float, int, int, int, int]] = []
    push_counter = 0

    def push_candidate(x: int, y: int, size: int) -> None:
        nonlocal push_counter
        if size <= cfg.patch4:
            return
        mass = _integral_sum(importance_integral, x, y, size)
        heapq.heappush(heap, (-mass, push_counter, x, y, size))
        push_counter += 1

    # Deterministic tie handling. Stable geometric insertion is close to the
    # legacy DP's first-found preference; equal-cost solutions may still differ
    # in constant-importance background regions while remaining exactly optimal.
    for x, y, size in sorted(active, key=lambda item: (item[1], item[0], -item[2])):
        push_candidate(x, y, size)

    for _ in range(remaining_splits):
        while heap:
            _negative_mass, _order, x, y, size = heapq.heappop(heap)
            node = (x, y, size)
            if node in active:
                break
        else:
            raise RuntimeError(
                "No exact-budget quadtree satisfies the current hard constraints "
                f"for target_tokens={cfg.target_tokens}"
            )

        active.remove(node)
        child = size // 2
        children = (
            (x, y, child),
            (x + child, y, child),
            (x, y + child, child),
            (x + child, y + child, child),
        )
        for child_node in children:
            active.add(child_node)
            push_candidate(*child_node)

    leaves = [
        AdaptiveLeaf(x, y, size, "exact_budget_heap")
        for x, y, size in active
    ]
    if len(leaves) != int(cfg.target_tokens):
        raise AssertionError(
            f"fast exact-budget solver produced {len(leaves)} leaves, "
            f"expected {cfg.target_tokens}"
        )

    total_cost = 0.0
    for leaf in leaves:
        scale_penalty = math.log2(float(leaf.size) / float(cfg.patch4))
        total_cost += scale_penalty * _integral_sum(
            importance_integral, leaf.x, leaf.y, leaf.size
        )
    return leaves, float(total_cost)


def _solve_exact_budget_selected(
    importance: np.ndarray,
    iris_hard: np.ndarray,
    eyelid_hard: np.ndarray,
    cfg: AdaptiveGridConfig,
    enforce_hard: bool,
    correction_hard: Optional[np.ndarray] = None,
    anchor_hard: Optional[np.ndarray] = None,
    lens_hard: Optional[np.ndarray] = None,
    *,
    solver: str = "dp",
) -> Tuple[List[AdaptiveLeaf], float]:
    """Select legacy DP, fast heap, or a dual-solver regression verifier.

    ``verify`` runs both implementations and returns the legacy-DP leaves. It
    checks equal feasibility and equal objective value; exact leaf identity is
    intentionally not required because constant-weight regions can contain
    multiple exactly tied optimal quadtrees.
    """
    mode = str(solver).lower()
    kwargs = dict(
        correction_hard=correction_hard,
        anchor_hard=anchor_hard,
        lens_hard=lens_hard,
    )
    if mode == "dp":
        return _solve_exact_budget(
            importance, iris_hard, eyelid_hard, cfg, enforce_hard, **kwargs
        )
    if mode == "fast":
        return _solve_exact_budget_fast(
            importance, iris_hard, eyelid_hard, cfg, enforce_hard, **kwargs
        )
    if mode != "verify":
        raise ValueError(
            f"unknown adaptive-grid solver {solver!r}; expected dp, fast, or verify"
        )

    dp_error = None
    fast_error = None
    try:
        dp_result = _solve_exact_budget(
            importance, iris_hard, eyelid_hard, cfg, enforce_hard, **kwargs
        )
    except RuntimeError as exc:
        dp_error = exc
        dp_result = None
    try:
        fast_result = _solve_exact_budget_fast(
            importance, iris_hard, eyelid_hard, cfg, enforce_hard, **kwargs
        )
    except RuntimeError as exc:
        fast_error = exc
        fast_result = None

    if (dp_error is None) != (fast_error is None):
        raise AssertionError(
            "adaptive-grid feasibility mismatch: "
            f"dp={dp_error!r}, fast={fast_error!r}"
        )
    if dp_error is not None:
        raise RuntimeError(str(dp_error))

    assert dp_result is not None and fast_result is not None
    dp_leaves, dp_cost = dp_result
    fast_leaves, fast_cost = fast_result
    validate_grid(fast_leaves, cfg)
    tolerance = 1.0e-8 * max(1.0, abs(float(dp_cost)), abs(float(fast_cost)))
    if abs(float(dp_cost) - float(fast_cost)) > tolerance:
        raise AssertionError(
            "adaptive-grid objective mismatch: "
            f"dp={dp_cost:.12g}, fast={fast_cost:.12g}, tol={tolerance:.3g}"
        )
    return dp_leaves, float(dp_cost)


def _result_from_leaves(
    leaves: List[AdaptiveLeaf],
    geometry_success: bool,
    geometry_source: str,
    dp_total_cost: float,
    eye_mask: np.ndarray,
    cfg: AdaptiveGridConfig,
    hard_constraints_relaxed: bool,
    *,
    correction_mask: Optional[np.ndarray] = None,
    anchor_mask: Optional[np.ndarray] = None,
    lens_mask: Optional[np.ndarray] = None,
    constraint_mode: str = "legacy_eye",
) -> GridBuildResult:
    leaves.sort(key=lambda leaf: (leaf.y, leaf.x, leaf.size))
    validate_grid(leaves, cfg)
    counts = {size: 0 for size in cfg.patch_sizes}
    eye_tokens = 0
    correction_tokens = 0
    anchor_tokens = 0
    lens_tokens = 0
    correction_mask = eye_mask if correction_mask is None else correction_mask
    anchor_mask = np.zeros_like(eye_mask) if anchor_mask is None else anchor_mask
    lens_mask = np.zeros_like(eye_mask) if lens_mask is None else lens_mask
    for leaf in leaves:
        counts[leaf.size] += 1
        if np.any(eye_mask[leaf.y : leaf.y + leaf.size, leaf.x : leaf.x + leaf.size]):
            eye_tokens += 1
        if np.any(correction_mask[leaf.y : leaf.y + leaf.size, leaf.x : leaf.x + leaf.size]):
            correction_tokens += 1
        if np.any(anchor_mask[leaf.y : leaf.y + leaf.size, leaf.x : leaf.x + leaf.size]):
            anchor_tokens += 1
        if np.any(lens_mask[leaf.y : leaf.y + leaf.size, leaf.x : leaf.x + leaf.size]):
            lens_tokens += 1
    root_count = (cfg.canvas_size // cfg.patch32) ** 2
    internal_splits = (len(leaves) - root_count) // 3
    return GridBuildResult(
        leaves=leaves,
        geometry_success=geometry_success,
        geometry_source=geometry_source,
        target_token_count=cfg.target_tokens,
        dp_total_cost=float(dp_total_cost),
        patch4_count=counts[cfg.patch4],
        patch8_count=counts[cfg.patch8],
        patch16_count=counts[cfg.patch16],
        patch32_count=counts[cfg.patch32],
        eye_token_count=eye_tokens,
        hard_constraints_relaxed=hard_constraints_relaxed,
        internal_splits=internal_splits,
        correction_token_count=correction_tokens,
        anchor_token_count=anchor_tokens,
        lens_token_count=lens_tokens,
        constraint_mode=str(constraint_mode),
        mandatory_token_count=len(leaves),
        hard_splits_used=counts[cfg.patch4],
        soft_candidates_count=0,
        optional_splits_used=internal_splits,
        budget_clipped=False,
    )


def build_adaptive_grid(
    eye_prior: Image.Image,
    metadata: Mapping[str, object],
    original_size: Tuple[int, int],
    cfg: AdaptiveGridConfig,
    *,
    solver: str = "dp",
) -> GridBuildResult:
    """Build an exact-budget p4/p8/p16/p32 geometry grid.

    V29.4 prioritizes the *estimated spectacle lens band* rather than a very broad
    periocular rectangle.  The preferred exact-budget solve forces lens-band leaves
    to <= p8 while retaining iris/eyelid and sparse landmark constraints.  If that
    is infeasible, fallback order is eye+anchors, anchors only, then unconstrained.
    The lens estimate uses only glare-time geometry and never clean/glare difference.
    """
    geometry_success = bool(metadata.get("success", metadata.get("prior_valid", metadata.get("has_eye", False))))
    if not geometry_success:
        leaves = uniform_grid(cfg)
        prior = np.zeros((cfg.canvas_size, cfg.canvas_size), dtype=np.uint8)
        if len(leaves) == cfg.target_tokens:
            return _result_from_leaves(
                leaves, False, "uniform_geometry_failure", 0.0, prior, cfg, False,
                correction_mask=prior, lens_mask=prior, constraint_mode="uniform_geometry_failure",
            )
        importance = np.full_like(prior, cfg.canvas_weight, dtype=np.float64)
        leaves, total_cost = _solve_exact_budget_selected(importance, prior, prior, cfg, enforce_hard=False, solver=solver)
        return _result_from_leaves(
            leaves, False, "canvas_only_geometry_failure", total_cost, prior, cfg, False,
            correction_mask=prior, lens_mask=prior, constraint_mode="canvas_only",
        )

    maps = build_geometry_maps(eye_prior, metadata, original_size, cfg)
    prior = maps["prior"]
    periocular = maps["periocular"]
    iris_hard = maps["iris"]
    eyelid_hard = maps["eyelid"]
    anchor_hard = maps["hard_anchor"]
    lens_band = maps["lens_band"]
    importance = maps["importance"]
    head, geometry_source = _head_mask(metadata, original_size, cfg)
    del head

    correction = (prior >= int(cfg.correction_threshold)).astype(np.uint8)
    correction = _dilate(correction, int(cfg.correction_dilate_radius))

    # Stage 1: reserve p4/p8 capacity over the whole estimated lens band.  This is
    # deliberately more targeted than hard-constraining the full periocular prior.
    try:
        leaves, total_cost = _solve_exact_budget_selected(
            importance, iris_hard, eyelid_hard, cfg, enforce_hard=True,
            anchor_hard=anchor_hard, lens_hard=lens_band, solver=solver,
        )
        return _result_from_leaves(
            leaves, True, geometry_source, total_cost, periocular, cfg, False,
            correction_mask=correction, anchor_mask=anchor_hard, lens_mask=lens_band,
            constraint_mode="lens_band_plus_eye_anchor_p8",
        )
    except RuntimeError:
        if not cfg.relax_hard_constraints_if_infeasible:
            raise

    # Stage 2: if the complete lens band is too expensive, preserve the actual eye
    # geometry and sparse structural anchors. The lens still has strong soft cost via
    # lens_weight, so DP preferentially spends remaining tokens there.
    try:
        leaves, total_cost = _solve_exact_budget_selected(
            importance, iris_hard, eyelid_hard, cfg,
            enforce_hard=cfg.enforce_hard_eye_constraints, anchor_hard=anchor_hard, solver=solver,
        )
        return _result_from_leaves(
            leaves, True, geometry_source, total_cost, periocular, cfg, True,
            correction_mask=correction, anchor_mask=anchor_hard, lens_mask=lens_band,
            constraint_mode="iris_eyelid_plus_anchor_fallback",
        )
    except RuntimeError:
        pass

    # Stage 3: retain sparse structural anchors even when dense eye constraints are
    # incompatible with the exact 256-token budget.
    zeros = np.zeros_like(iris_hard)
    try:
        leaves, total_cost = _solve_exact_budget_selected(
            importance, zeros, zeros, cfg, enforce_hard=True, anchor_hard=anchor_hard, solver=solver,
        )
        return _result_from_leaves(
            leaves, True, geometry_source, total_cost, periocular, cfg, True,
            correction_mask=correction, anchor_mask=anchor_hard, lens_mask=lens_band,
            constraint_mode="anchor_only_fallback",
        )
    except RuntimeError:
        pass

    leaves, total_cost = _solve_exact_budget_selected(
        importance, zeros, zeros, cfg, enforce_hard=False, solver=solver
    )
    return _result_from_leaves(
        leaves, True, geometry_source, total_cost, periocular, cfg, True,
        correction_mask=correction, anchor_mask=anchor_hard, lens_mask=lens_band,
        constraint_mode="unconstrained_fallback",
    )


def _normalized_point_to_logical(
    point: Sequence[float],
    original_size: Tuple[int, int],
    canvas_size: int,
) -> Tuple[float, float]:
    width, height = map(int, original_size)
    side = max(width, height)
    pad_x = (side - width) / 2.0
    pad_y = (side - height) / 2.0
    x, y = map(float, point)
    scale = canvas_size / float(side)
    return (pad_x + x * width) * scale, (pad_y + y * height) * scale


def _polygon_mask_from_normalized_points(
    points: Sequence[Sequence[float]],
    original_size: Tuple[int, int],
    canvas_size: int,
) -> np.ndarray:
    output = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    if not points:
        return output
    polygon = np.asarray(
        [
            _normalized_point_to_logical(point, original_size, canvas_size)
            for point in points
        ],
        dtype=np.float32,
    )
    if polygon.shape[0] >= 3:
        polygon = np.round(polygon).astype(np.int32)
        cv2.fillPoly(output, [polygon], 1)
    return output


def _lens_band_from_landmarks(
    metadata: Mapping[str, object],
    original_size: Tuple[int, int],
    cfg: AdaptiveGridConfig,
    fallback_periocular: np.ndarray,
) -> np.ndarray:
    """Estimate a glasses/lens support from all available eye contours.

    For multi-face inputs, every detected face contributes its left/right eye
    contours and the resulting rectangular lens bands are unioned.  If no detailed
    contour is available, the periocular prior remains the conservative fallback.
    """
    group_entries = _landmark_group_entries(metadata)
    if not group_entries:
        return fallback_periocular.astype(np.uint8, copy=True)

    out = np.zeros_like(fallback_periocular, dtype=np.uint8)
    made = False
    for groups in group_entries:
        for key in ("left_eye_contour", "right_eye_contour"):
            contour = groups.get(key)
            if not isinstance(contour, (list, tuple)) or len(contour) < 3:
                continue
            pts = []
            for point in contour:
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    pts.append(
                        _normalized_point_to_logical(
                            point, original_size, cfg.canvas_size
                        )
                    )
            if len(pts) < 3:
                continue
            arr = np.asarray(pts, dtype=np.float32)
            x1, y1 = arr.min(axis=0)
            x2, y2 = arr.max(axis=0)
            span_x = max(1.0, float(x2 - x1))
            span_y = max(1.0, float(y2 - y1))
            cx = 0.5 * float(x1 + x2)
            cy = 0.5 * float(y1 + y2)
            width = span_x * (1.0 + 2.0 * float(cfg.lens_expand_x))
            height = span_y * (1.0 + 2.0 * float(cfg.lens_expand_y))
            height = max(height, 0.55 * width)
            xa = max(0, int(math.floor(cx - 0.5 * width)))
            xb = min(cfg.canvas_size, int(math.ceil(cx + 0.5 * width)))
            ya = max(0, int(math.floor(cy - 0.5 * height)))
            yb = min(cfg.canvas_size, int(math.ceil(cy + 0.5 * height)))
            if xb > xa and yb > ya:
                out[ya:yb, xa:xb] = 1
                made = True
    if not made:
        return fallback_periocular.astype(np.uint8, copy=True)
    return out


def build_renderer_lens_alpha(
    metadata: Mapping[str, object],
    original_size: Tuple[int, int],
    *,
    canvas_size: int = 256,
    expand_x: float = 0.60,
    expand_y: float = 1.25,
    min_height_ratio: float = 0.65,
    feather: float = 2.0,
    superellipse_power: float = 4.0,
) -> Optional[np.ndarray]:
    """Build a soft, rotated rounded lens support for all detected faces.

    Every available left/right eye contour contributes one soft spectacle-lens
    support.  Multi-face supports are combined by a pixelwise maximum so the final
    renderer can preserve/restore all faces in one whole-image forward pass.

    Returns a float32 HxW map in [0,1], or ``None`` when no eye contour is available.
    """
    canvas = int(canvas_size)
    if canvas <= 0:
        raise ValueError("canvas_size must be positive")
    if float(expand_x) < 0 or float(expand_y) < 0:
        raise ValueError("renderer lens expansion must be non-negative")
    if float(min_height_ratio) <= 0:
        raise ValueError("min_height_ratio must be positive")
    if float(feather) < 0:
        raise ValueError("feather must be non-negative")
    power = float(superellipse_power)
    if power < 2.0:
        raise ValueError("superellipse_power must be >= 2")

    group_entries = _landmark_group_entries(metadata)
    if not group_entries:
        return None

    yy, xx = np.mgrid[0:canvas, 0:canvas].astype(np.float32)
    out = np.zeros((canvas, canvas), dtype=np.float32)
    made = False

    for groups in group_entries:
        for key in ("left_eye_contour", "right_eye_contour"):
            contour = groups.get(key)
            if not isinstance(contour, (list, tuple)) or len(contour) < 3:
                continue
            pts = []
            for point in contour:
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    pts.append(
                        _normalized_point_to_logical(point, original_size, canvas)
                    )
            if len(pts) < 3:
                continue

            arr = np.asarray(pts, dtype=np.float32)
            center0 = arr.mean(axis=0)
            centered = arr - center0[None, :]
            cov = centered.T @ centered / max(1, arr.shape[0] - 1)
            values, vectors = np.linalg.eigh(cov)
            axis_x = vectors[:, int(np.argmax(values))].astype(np.float32)
            norm = float(np.linalg.norm(axis_x))
            if norm < 1e-6:
                axis_x = np.array([1.0, 0.0], dtype=np.float32)
            else:
                axis_x /= norm
            if axis_x[0] < 0:
                axis_x = -axis_x
            axis_y = np.array([-axis_x[1], axis_x[0]], dtype=np.float32)

            proj_x = centered @ axis_x
            proj_y = centered @ axis_y
            x_min, x_max = float(proj_x.min()), float(proj_x.max())
            y_min, y_max = float(proj_y.min()), float(proj_y.max())
            span_x = max(1.0, x_max - x_min)
            span_y = max(1.0, y_max - y_min)

            center = (
                center0
                + axis_x * (0.5 * (x_min + x_max))
                + axis_y * (0.5 * (y_min + y_max))
            )
            half_w = 0.5 * span_x * (1.0 + 2.0 * float(expand_x))
            half_h = 0.5 * span_y * (1.0 + 2.0 * float(expand_y))
            half_h = max(half_h, float(min_height_ratio) * half_w)
            half_w = max(1.0, half_w)
            half_h = max(1.0, half_h)

            dx = xx - float(center[0])
            dy = yy - float(center[1])
            u = (dx * axis_x[0] + dy * axis_x[1]) / half_w
            v = (dx * axis_y[0] + dy * axis_y[1]) / half_h
            radius = (
                np.abs(u) ** power + np.abs(v) ** power
            ) ** (1.0 / power)

            distance = (radius - 1.0) * min(half_w, half_h)
            if float(feather) <= 0:
                alpha = (distance <= 0).astype(np.float32)
            else:
                t = np.clip(distance / float(feather), 0.0, 1.0)
                smooth = t * t * (3.0 - 2.0 * t)
                alpha = np.where(
                    distance <= 0.0, 1.0, 1.0 - smooth
                ).astype(np.float32)
                alpha[distance >= float(feather)] = 0.0
            out = np.maximum(out, alpha)
            made = True

    return out if made else None


def build_geometry_maps(
    eye_prior: Image.Image,
    metadata: Mapping[str, object],
    original_size: Tuple[int, int],
    cfg: AdaptiveGridConfig,
) -> Dict[str, np.ndarray]:
    """Return logical-canvas task masks shared by preprocessing and DP.

    The function is deterministic and uses only inference-time information.
    """
    prior = np.asarray(
        logical_canvas(eye_prior, cfg.canvas_size, is_mask=True), dtype=np.uint8
    )
    periocular = (prior >= 64).astype(np.uint8)
    eye = (prior >= 128).astype(np.uint8)
    iris = (prior >= 192).astype(np.uint8)
    iris_soft = _dilate(iris, cfg.iris_dilate_radius)
    eyelid = _morph_gradient(eye, cfg.eyelid_band_radius)
    head, _ = _head_mask(metadata, original_size, cfg)
    lens_band = _lens_band_from_landmarks(metadata, original_size, cfg, periocular)

    importance = np.full(
        (cfg.canvas_size, cfg.canvas_size), cfg.canvas_weight, dtype=np.float64
    )
    importance += cfg.face_weight * head
    importance += cfg.periocular_weight * periocular
    importance += cfg.eye_weight * eye
    importance += cfg.iris_weight * iris_soft
    importance += cfg.eyelid_weight * eyelid
    importance += cfg.lens_weight * lens_band

    group_entries = _landmark_group_entries(metadata)
    hard_anchor = np.zeros_like(prior, dtype=np.uint8)
    anchor_points: List[Tuple[float, float]] = []
    for groups in group_entries:
        for key in (
            "left_iris_center",
            "right_iris_center",
            "left_inner_corner",
            "left_outer_corner",
            "right_inner_corner",
            "right_outer_corner",
        ):
            point = groups.get(key)
            if isinstance(point, (list, tuple)) and len(point) == 2:
                anchor_points.append(
                    _normalized_point_to_logical(
                        point, original_size, cfg.canvas_size
                    )
                )
        # Sparse samples from both eye contours of every detected face.
        for key in ("left_eye_contour", "right_eye_contour"):
            contour = groups.get(key)
            if isinstance(contour, (list, tuple)) and contour:
                step = max(1, len(contour) // 6)
                for point in contour[::step]:
                    if isinstance(point, (list, tuple)) and len(point) == 2:
                        anchor_points.append(
                            _normalized_point_to_logical(
                                point, original_size, cfg.canvas_size
                            )
                        )
    radius = max(0, int(cfg.hard_anchor_radius))
    for x, y in anchor_points:
        cv2.circle(
            hard_anchor,
            (int(round(x)), int(round(y))),
            radius,
            1,
            thickness=-1,
        )


    return {
        "prior": prior,
        "head": head,
        "periocular": periocular,
        "eye": eye,
        "iris": iris_soft,
        "eyelid": eyelid,
        "importance": importance,
        "hard_anchor": hard_anchor,
        "lens_band": lens_band,
    }


def extract_canonical_patches(
    image: Image.Image,
    leaves: Sequence[AdaptiveLeaf],
    canvas_size: int,
    canonical_size: int,
) -> torch.Tensor:
    """Crop from the original-resolution square canvas, then resize each leaf to P0."""
    square, _ = square_edge_canvas(image)
    side = square.size[0]
    outputs: List[torch.Tensor] = []
    for leaf in leaves:
        x1 = int(math.floor(leaf.x * side / canvas_size))
        y1 = int(math.floor(leaf.y * side / canvas_size))
        x2 = int(math.ceil((leaf.x + leaf.size) * side / canvas_size))
        y2 = int(math.ceil((leaf.y + leaf.size) * side / canvas_size))
        x1 = max(0, min(side - 1, x1))
        y1 = max(0, min(side - 1, y1))
        x2 = max(x1 + 1, min(side, x2))
        y2 = max(y1 + 1, min(side, y2))
        patch = square.crop((x1, y1, x2, y2)).resize(
            (canonical_size, canonical_size), Image.Resampling.BICUBIC
        )
        outputs.append(pil_to_tensor(patch))
    return torch.stack(outputs, dim=0)


def load_metadata(path: Optional[Path]) -> Dict[str, object]:
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# --- inlined from models/learned_frequency_renderer.py ---
def _group_count(channels: int) -> int:
    for groups in range(min(8, int(channels)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


class _DepthwiseBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=dilation, dilation=dilation,
            groups=channels, bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.pointwise(self.depthwise(x))))


class LearnedFrequencyGate(nn.Module):
    """Predict residual corrections to the handcrafted low/HF gates.

    The last convolution is initialized to zero, so a newly constructed module
    reproduces the existing handcrafted renderer exactly.  Learning changes only
    the two scalar spatial maps; the original frequency decomposition is retained.
    """

    input_channels = 15

    def __init__(self, width: int = 32, map_size: int = 256,
                 residual_scale: float = 1.0) -> None:
        super().__init__()
        width = int(width)
        if width <= 0 or int(map_size) <= 0:
            raise ValueError("renderer gate width and map_size must be positive")
        self.map_size = int(map_size)
        self.residual_scale = float(residual_scale)
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            _DepthwiseBlock(width, 1),
            _DepthwiseBlock(width, 2),
            _DepthwiseBlock(width, 4),
        )
        self.head = nn.Conv2d(width, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.last_residual: Optional[torch.Tensor] = None

    def _small_batch(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(f"renderer batch maps must be [B,C,H,W], got {tuple(value.shape)}")
        return F.interpolate(
            value.float(),
            size=(self.map_size, self.map_size),
            mode="bilinear", align_corners=False,
        )

    def _forward_core(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(features)))

    def _run_core(self, features: torch.Tensor) -> torch.Tensor:
        """Run the fixed 256x256 gate core, compiling once on CUDA when enabled.

        Only the CNN core is compiled: native-resolution resizes stay eager so
        dynamic H/W do not cause shape-driven recompiles. Any compiler/runtime
        failure permanently falls back to eager execution for this module.
        """
        if not bool(getattr(self, "compile_core", False)) or not features.is_cuda:
            return self._forward_core(features)
        compiled = getattr(self, "_compiled_core_fn", None)
        if compiled is False:
            return self._forward_core(features)
        signature = (
            tuple(features.shape), tuple(features.stride()), str(features.dtype),
            features.device.type, features.device.index, bool(features.requires_grad),
        )
        validated_signature = getattr(self, "_compiled_core_signature", None)
        if validated_signature is not None and validated_signature != signature:
            return self._forward_core(features)
        if compiled is None:
            try:
                compiled = torch.compile(
                    self._forward_core,
                    dynamic=True,
                    mode="default",
                )
                self._compiled_core_fn = compiled
            except Exception:
                self._compiled_core_fn = False
                return self._forward_core(features)
        try:
            candidate = compiled(features)
            if not bool(getattr(self, "_compiled_core_validated", False)):
                reference = self._forward_core(features)
                if float((candidate - reference).abs().amax().item()) > 5e-6:
                    self._compiled_core_fn = False
                    return reference
                self._compiled_core_validated = True
                self._compiled_core_signature = signature
            return candidate
        except Exception:
            self._compiled_core_fn = False
            return self._forward_core(features)

    def forward_batch(
        self,
        native_glare: torch.Tensor,
        clean_candidate: torch.Tensor,
        correction: torch.Tensor,
        correction_magnitude: torch.Tensor,
        eye_mask: torch.Tensor,
        iris_mask: torch.Tensor,
        lens_alpha: torch.Tensor,
        handcrafted_low: torch.Tensor,
        handcrafted_hf: torch.Tensor,
        correction_normalizer: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if native_glare.ndim != 4:
            raise ValueError("batched learned renderer expects [B,C,H,W] maps")
        norm = max(float(correction_normalizer), 1e-6)
        features = torch.cat([
            self._small_batch(native_glare),
            self._small_batch(clean_candidate),
            self._small_batch((0.5 * correction).clamp(-1.0, 1.0)),
            self._small_batch((correction_magnitude / norm).clamp(0.0, 2.0)),
            self._small_batch(eye_mask),
            self._small_batch(iris_mask),
            self._small_batch(lens_alpha),
            self._small_batch(handcrafted_low),
            self._small_batch(handcrafted_hf),
        ], dim=1)
        if features.shape[1] != self.input_channels:
            raise RuntimeError(
                f"learned renderer built {features.shape[1]} channels; "
                f"expected {self.input_channels}"
            )
        residual = self._run_core(features)
        residual = F.interpolate(
            residual, size=native_glare.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        residual = self.residual_scale * torch.tanh(residual)
        self.last_residual = residual * lens_alpha
        learned_low = (
            handcrafted_low + lens_alpha * residual[:, 0:1]
        ).clamp(0.0, 1.0)
        learned_hf = (
            handcrafted_hf + lens_alpha * residual[:, 1:2]
        ).clamp(0.0, 1.0)
        return learned_low, learned_hf

    def forward(
        self,
        native_glare: torch.Tensor,
        clean_candidate: torch.Tensor,
        correction: torch.Tensor,
        correction_magnitude: torch.Tensor,
        eye_mask: torch.Tensor,
        iris_mask: torch.Tensor,
        lens_alpha: torch.Tensor,
        handcrafted_low: torch.Tensor,
        handcrafted_hf: torch.Tensor,
        correction_normalizer: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if native_glare.ndim != 3:
            raise ValueError("learned renderer expects unbatched [C,H,W] maps")
        values = (
            native_glare, clean_candidate, correction, correction_magnitude,
            eye_mask, iris_mask, lens_alpha, handcrafted_low, handcrafted_hf,
        )
        low, high = self.forward_batch(
            *(value.unsqueeze(0) for value in values),
            correction_normalizer=correction_normalizer,
        )
        return low[0], high[0]

    def residual_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.last_residual is None:
            zero = self.head.weight.new_zeros(())
            return zero, zero
        value = self.last_residual
        magnitude = value.abs().mean()
        tv_h = (value[..., 1:, :] - value[..., :-1, :]).abs().mean()
        tv_w = (value[..., :, 1:] - value[..., :, :-1]).abs().mean()
        return magnitude, 0.5 * (tv_h + tv_w)


# --- inlined from models/adaptive_conditioning.py ---
class CanonicalPatchTokenEmbedder(nn.Module):
    def __init__(self, canonical_size: int, hidden_size: int, bottleneck: int = 128) -> None:
        super().__init__()
        self.p = int(canonical_size)
        self.d = int(hidden_size)
        self.embedder = BottleneckPatchEmbedder(
            input_size=self.p,
            initial_patch_size=self.p,
            pca_channels=int(bottleneck),
            in_channels=3,
            hidden_size=self.d,
            bias=True,
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 5:
            raise ValueError(f"patches must be [B,T,3,P,P], got {tuple(patches.shape)}")
        b,t,c,h,w = patches.shape
        if c != 3 or h != self.p or w != self.p:
            raise ValueError(f"expected [B,T,3,{self.p},{self.p}], got {tuple(patches.shape)}")
        x = self.embedder(patches.reshape(b*t,c,h,w))
        if x.shape[1] != 1:
            raise RuntimeError("canonical patch embedder must emit one token per adaptive leaf")
        return x[:,0].reshape(b,t,self.d)


class GeometryEmbedding(nn.Module):
    def __init__(self, hidden_size: int, canvas_size: int = 256, base_patch: int = 16) -> None:
        super().__init__()
        self.canvas_size = float(canvas_size)
        self.base_patch = float(base_patch)
        inner = max(32, int(hidden_size)//4)
        self.net = nn.Sequential(nn.Linear(3,inner), nn.SiLU(), nn.Linear(inner,int(hidden_size)))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)

    def forward(self, leaves: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        v = leaves.float()
        x,y,s = v.unbind(-1)
        f = torch.stack([
            (y + .5*s)/self.canvas_size*2-1,
            (x + .5*s)/self.canvas_size*2-1,
            torch.log2(s.clamp_min(1)/self.base_patch),
        ],-1)
        return self.net(f).to(dtype=dtype)


class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        d=int(hidden_size); h=int(num_heads)
        if d % h: raise ValueError("hidden_size must divide num_heads")
        self.h=h; self.dh=d//h; self.d=d
        self.q=nn.Linear(d,d,bias=False); self.k=nn.Linear(d,d,bias=False)
        self.v=nn.Linear(d,d,bias=False); self.o=nn.Linear(d,d,bias=False)
        # The conditioner is added to a pretrained PMF backbone. Zero output
        # projection makes that residual branch an exact no-op at step 0 while
        # preserving gradients into the projection so conditioning can grow safely.
        nn.init.zeros_(self.o.weight)
        self.qn=nn.LayerNorm(self.dh, elementwise_affine=False)
        self.kn=nn.LayerNorm(self.dh, elementwise_affine=False)
        self.backend="legacy"

    def forward(self, q_content: torch.Tensor, kv_content: torch.Tensor,
                q_geom: Optional[torch.Tensor]=None, kv_geom: Optional[torch.Tensor]=None,
                key_valid: Optional[torch.Tensor]=None) -> torch.Tensor:
        b,nq,d=q_content.shape; nk=kv_content.shape[1]
        qin=q_content if q_geom is None else q_content+q_geom.to(q_content.dtype)
        kin=kv_content if kv_geom is None else kv_content+kv_geom.to(kv_content.dtype)
        q=self.qn(self.q(qin).reshape(b,nq,self.h,self.dh))
        k=self.kn(self.k(kin).reshape(b,nk,self.h,self.dh))
        v=self.v(kv_content).reshape(b,nk,self.h,self.dh)
        backend=str(getattr(self,"backend","legacy")).lower()
        if backend not in ("legacy","sdpa"):
            raise ValueError(f"unknown conditioner attention backend {backend!r}")

        any_valid=None
        safe=None
        if key_valid is not None:
            valid=key_valid.bool(); any_valid=valid.any(-1)
            safe=valid.clone(); safe[:,0] |= ~any_valid

        if backend == "sdpa":
            mask=None if safe is None else safe[:,None,None,:]
            out=F.scaled_dot_product_attention(
                q.permute(0,2,1,3),
                k.permute(0,2,1,3),
                v.permute(0,2,1,3),
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0/math.sqrt(self.dh),
            ).permute(0,2,1,3)
            if any_valid is not None:
                out=out*any_valid[:,None,None,None].to(out.dtype)
        else:
            logits=torch.einsum('bqhd,bkhd->bhqk',q/math.sqrt(self.dh),k)
            if safe is not None:
                logits=logits.masked_fill(~safe[:,None,None,:],-1e4)
            w=F.softmax(logits.float(),dim=-1).to(v.dtype)
            if any_valid is not None:
                w=w*any_valid[:,None,None,None].to(w.dtype)
            out=torch.einsum('bhqk,bkhd->bqhd',w,v)
        return self.o(out.reshape(b,nq,d))



@dataclass
class ConditioningOutput:
    conditioned_state: torch.Tensor
    state_tokens: torch.Tensor
    glare_tokens: torch.Tensor


class AdaptiveConditioner(nn.Module):
    def __init__(self, canonical_size:int, hidden_size:int, num_heads:int,
                 bottleneck:int=128, canvas_size:int=256, base_patch:int=16,
                 condition_gate_init:float=.1, context_gate_init:float=.05,
                 use_geometry_token:bool=True,
                 use_coarse_to_fine_context:bool=True) -> None:
        super().__init__(); d=int(hidden_size)
        self.use_geometry_token=bool(use_geometry_token)
        self.use_coarse_to_fine_context=bool(use_coarse_to_fine_context)
        self.glare_embed=CanonicalPatchTokenEmbedder(canonical_size,d,bottleneck)
        self.geom=GeometryEmbedding(d,canvas_size,base_patch) if self.use_geometry_token else None
        self.sg_q=nn.LayerNorm(d); self.sg_k=nn.LayerNorm(d); self.sg=CrossAttention(d,num_heads)
        if self.use_coarse_to_fine_context:
            self.gl_q=nn.LayerNorm(d); self.gl_k=nn.LayerNorm(d); self.gl=CrossAttention(d,num_heads)
        else:
            self.gl_q=self.gl_k=self.gl=None
        self.g_sg=nn.Parameter(torch.full((d,),float(condition_gate_init)))
        self.g_gl=(
            nn.Parameter(torch.full((d,),float(context_gate_init)))
            if self.use_coarse_to_fine_context else None
        )

    @staticmethod
    def masks(leaves: torch.Tensor, region_probs: torch.Tensor):
        sizes=leaves[...,2]
        fine=sizes<=8; coarse=sizes>=16
        semantic=(1.0-region_probs[...,0])>1e-4
        # Fine leaves are also the geometry-derived spectacle-band proxy.
        active=semantic | fine
        return active,fine,coarse

    @staticmethod
    def geometry_support(leaves: torch.Tensor, region_probs: torch.Tensor) -> torch.Tensor:
        sizes=leaves[...,2]
        fine=(sizes<=8).to(region_probs.dtype)
        semantic=(1.0-region_probs[...,0]).clamp(0,1)
        # Full semantic eye/periocular support; 0.75 support for fine lens-band leaves.
        return torch.maximum(semantic, 0.75*fine).clamp(0,1)

    def gate_metrics(self):
        zero=self.g_sg.new_zeros(())
        condition=self.g_sg.abs().mean()
        context=self.g_gl.abs().mean() if self.g_gl is not None else zero
        return {'condition_gate_mean':condition,'context_gate_mean':context}

    def _attention_without_coarse_to_fine(self,attention,q,kv,q_geom,kv_geom,key_valid,fine,coarse):
        delta=attention(q,kv,q_geom,kv_geom,key_valid)
        if not self.use_coarse_to_fine_context:
            fine_delta=attention(q,kv,q_geom,kv_geom,key_valid & ~coarse)
            delta=torch.where(fine[...,None],fine_delta,delta)
        return delta

    def forward_from_state_tokens(self,state_tokens,glare_patches,region_probs,leaves):
        b,t,d=state_tokens.shape
        # Disabling geometry tokens removes the learned (x,y,scale) route. The grid
        # stays fixed, so token count, crops, PMF RoPE and compute remain comparable.
        g=self.geom(leaves,state_tokens.dtype) if self.geom is not None else None
        active,fine,coarse=self.masks(leaves,region_probs)
        glare=self.glare_embed(glare_patches)
        delta=self._attention_without_coarse_to_fine(
            self.sg,self.sg_q(state_tokens),self.sg_k(glare),g,g,active,fine,coarse
        )
        state2=torch.where(active[...,None],state_tokens+self.g_sg.view(1,1,-1)*delta,state_tokens)
        if self.use_coarse_to_fine_context:
            assert self.gl is not None and self.gl_q is not None and self.gl_k is not None
            assert self.g_gl is not None
            delta=self.gl(self.gl_q(state2),self.gl_k(state2),g,g,coarse)
            conditioned=torch.where(fine[...,None],state2+self.g_gl.view(1,1,-1)*delta,state2)
        else:
            # Strict switch: fine tokens receive no coarse-token context. The
            # glare -> state conditioning before this point is unchanged.
            conditioned=state2
        return ConditioningOutput(conditioned,state_tokens,glare)


# --- inlined from core.py ---
def make_source(glare: torch.Tensor, noise_scale: float, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    noise = torch.randn(glare.shape, device=glare.device, dtype=glare.dtype, generator=generator)
    return (glare + noise * float(noise_scale)).clamp(-1.0, 1.0)


def token_vote_loss(token_error: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Uniform one-leaf-one-vote loss.

    Every adaptive token has exactly the same weight.  We deliberately do not use
    clean/glare change magnitude as a gain: a low-change eye-detail token therefore
    retains the same vote as a high-change glare token.  Division by ``target_tokens``
    is only a fixed global scale because every image has exactly T leaves.
    """
    if token_error.ndim != 2:
        raise ValueError(f"token_error must be [B,T], got {tuple(token_error.shape)}")
    if int(target_tokens) <= 0 or token_error.shape[1] != int(target_tokens):
        raise ValueError(
            f"expected exactly target_tokens={target_tokens}, got T={token_error.shape[1]}"
        )
    return (token_error.sum(dim=1) / float(target_tokens)).mean()


def raised_cosine(rel: torch.Tensor, halo: float) -> torch.Tensor:
    left = torch.where(rel < 0, 0.5 * (1.0 + torch.cos(math.pi * rel / halo)), torch.ones_like(rel))
    right = torch.where(rel > 1, 0.5 * (1.0 + torch.cos(math.pi * (rel - 1.0) / halo)), torch.ones_like(rel))
    w = torch.where(rel < 0, left, torch.where(rel > 1, right, torch.ones_like(rel)))
    return torch.where((rel < -halo) | (rel > 1.0 + halo), torch.zeros_like(w), w).clamp(0.0, 1.0)


def render_continuous_field_batch(
    token_fields: torch.Tensor,
    leaves_cpu_batch: Sequence[Sequence[Sequence[int]]],
    sides: Sequence[int],
    canvas_size: int,
    halo_ratio: float = 0.05,
) -> list[torch.Tensor]:
    """Render arbitrary C-channel adaptive token fields with continuous PoU overlap.

    This is the renderer primitive.  It is linear in token values and supports
    ragged native sizes without padding. It supports both continuous residual fields
    and the current hybrid absolute-clean/residual reconstruction.
    """
    if token_fields.ndim != 5:
        raise ValueError("token_fields must be [B,T,C,P,P]")
    batch, token_count, channels = token_fields.shape[:3]
    if channels <= 0:
        raise ValueError("token_fields must have at least one channel")
    if len(leaves_cpu_batch) != batch or len(sides) != batch:
        raise ValueError("renderer batch metadata mismatch")
    if any(len(rows) != token_count for rows in leaves_cpu_batch):
        raise ValueError("leaf/token count mismatch")

    canvas_size = int(canvas_size)
    device = token_fields.device
    sides_i = [int(v) for v in sides]
    pixel_offsets = []
    cursor = 0
    for side in sides_i:
        pixel_offsets.append(cursor)
        cursor += side * side
    total_pixels = cursor

    groups: Dict[tuple[int, int], list[tuple[int, int, int, int, int, int, int, int]]] = {}
    halo = float(halo_ratio)
    for b, (rows, side) in enumerate(zip(leaves_cpu_batch, sides_i)):
        for local_index, row in enumerate(rows):
            x, y, size = map(int, row)
            lx1 = x - halo * size
            ly1 = y - halo * size
            lx2 = x + size + halo * size
            ly2 = y + size + halo * size
            x1 = max(0, int(math.floor(lx1 * side / canvas_size)))
            y1 = max(0, int(math.floor(ly1 * side / canvas_size)))
            x2 = min(side, int(math.ceil(lx2 * side / canvas_size)))
            y2 = min(side, int(math.ceil(ly2 * side / canvas_size)))
            if x2 <= x1 or y2 <= y1:
                continue
            height, width = y2 - y1, x2 - x1
            flat_token_index = b * token_count + local_index
            groups.setdefault((height, width), []).append(
                (flat_token_index, x, y, size, x1, y1, side, pixel_offsets[b])
            )

    accum = torch.zeros(channels, total_pixels, device=device, dtype=torch.float32)
    wsum = torch.zeros(total_pixels, device=device, dtype=torch.float32)
    flat_tokens = token_fields.reshape(batch * token_count, *token_fields.shape[2:])

    for (height, width), rows in groups.items():
        select = torch.tensor([r[0] for r in rows], device=device, dtype=torch.long)
        x = torch.tensor([r[1] for r in rows], device=device, dtype=torch.float32)
        y = torch.tensor([r[2] for r in rows], device=device, dtype=torch.float32)
        size = torch.tensor([r[3] for r in rows], device=device, dtype=torch.float32)
        x1 = torch.tensor([r[4] for r in rows], device=device, dtype=torch.long)
        y1 = torch.tensor([r[5] for r in rows], device=device, dtype=torch.long)
        side = torch.tensor([r[6] for r in rows], device=device, dtype=torch.long)
        pixel_offset = torch.tensor([r[7] for r in rows], device=device, dtype=torch.long)

        px = torch.arange(width, device=device, dtype=torch.float32)[None, :] + 0.5
        py = torch.arange(height, device=device, dtype=torch.float32)[None, :] + 0.5
        native_x = x1.float()[:, None] + px
        native_y = y1.float()[:, None] + py
        scale = float(canvas_size) / side.float()
        rel_x = (native_x * scale[:, None] - x[:, None]) / size[:, None]
        rel_y = (native_y * scale[:, None] - y[:, None]) / size[:, None]
        gx = rel_x * 2.0 - 1.0
        gy = rel_y * 2.0 - 1.0
        grid = torch.stack([
            gx[:, None, :].expand(-1, height, -1),
            gy[:, :, None].expand(-1, -1, width),
        ], dim=-1)

        patches = F.grid_sample(
            flat_tokens.index_select(0, select).float(), grid,
            mode="bicubic", padding_mode="border", align_corners=True,
        )
        weight = raised_cosine(rel_y, halo)[:, :, None] * raised_cosine(rel_x, halo)[:, None, :]

        ix = x1[:, None, None] + torch.arange(width, device=device)[None, None, :]
        iy = y1[:, None, None] + torch.arange(height, device=device)[None, :, None]
        local_idx = iy * side[:, None, None] + ix
        flat_idx = (local_idx + pixel_offset[:, None, None]).reshape(-1).long()
        flat_values = (patches * weight[:, None]).permute(1, 0, 2, 3).reshape(channels, -1)
        flat_weight = weight.reshape(-1)
        accum.scatter_add_(1, flat_idx[None].expand(channels, -1), flat_values)
        wsum.scatter_add_(0, flat_idx, flat_weight)

    normalized = accum / wsum.clamp_min(1e-6)[None]
    output = []
    for offset, side in zip(pixel_offsets, sides_i):
        output.append(normalized[:, offset:offset + side * side].reshape(channels, side, side))
    return output


_RENDER_ARANGE_CACHE: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _renderer_aranges(device: torch.device, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache integer coordinates and pixel-center float coordinates on-device."""
    index = -1 if device.index is None else int(device.index)
    key = (device.type, index, int(length))
    cached = _RENDER_ARANGE_CACHE.get(key)
    if cached is None or cached[0].device != device:
        integer = torch.arange(int(length), device=device, dtype=torch.long)
        center = integer.to(torch.float32).add(0.5)
        cached = (integer, center)
        _RENDER_ARANGE_CACHE[key] = cached
    return cached


_RENDER_COMPILED_HELPERS: dict[str, object] = {}
_RENDER_COMPILED_SIGNATURES: dict[str, tuple] = {}
_RENDER_DISABLED_HELPERS: set[str] = set()
_RENDER_VALIDATED_HELPERS: set[str] = set()


def _helper_tensor_sequence(value):
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_helper_tensor_sequence(item))
        return tuple(result)
    return tuple()


def _compiled_helper_matches(name: str, candidate, reference) -> bool:
    cand = _helper_tensor_sequence(candidate)
    ref = _helper_tensor_sequence(reference)
    if len(cand) != len(ref):
        return False
    # Geometry/source preparation feeds bicubic/scatter directly: require exact
    # equality there. Decode pointwise fusion may change the final FP32 ulps, but
    # the tolerance remains orders of magnitude below one uint8 output level.
    tolerance = 0.0 if name.startswith("pou_") else 5e-6
    for a, b in zip(cand, ref):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not (a.dtype.is_floating_point or a.dtype.is_complex):
            if not torch.equal(a, b):
                return False
        elif tolerance == 0.0:
            if not torch.equal(a, b):
                return False
        else:
            if float((a - b).abs().amax().item()) > tolerance:
                return False
    return True


def _renderer_signature_item(value):
    if isinstance(value, torch.Tensor):
        return (
            "tensor", tuple(value.shape), tuple(value.stride()), str(value.dtype),
            value.device.type, value.device.index, bool(value.requires_grad),
        )
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_renderer_signature_item(v) for v in value))
    if isinstance(value, (int, float, bool, str, type(None))):
        return (type(value).__name__, value)
    return (type(value).__name__, repr(value))


def _renderer_args_signature(args: tuple) -> tuple:
    return tuple(_renderer_signature_item(v) for v in args)


def _run_renderer_helper(name: str, fn, args: tuple, compile_enabled: bool):
    """Use a warmup-validated compiled graph only for its exact input signature.

    Native renderer shapes vary substantially.  Calling an existing Dynamo graph
    with a new shape can trigger multi-second recompilation inside the measured
    production loop.  We therefore pin each helper to the signature on which it
    was first compiled/validated (normally during warmup) and execute eager for
    every other signature.  This preserves the common-shape fusion win without
    allowing production-time recompilation.
    """
    if not compile_enabled or not args or not isinstance(args[0], torch.Tensor) or not args[0].is_cuda:
        return fn(*args)
    if name in _RENDER_DISABLED_HELPERS:
        return fn(*args)
    signature = _renderer_args_signature(args)
    validated_signature = _RENDER_COMPILED_SIGNATURES.get(name)
    if validated_signature is not None and validated_signature != signature:
        return fn(*args)
    compiled = _RENDER_COMPILED_HELPERS.get(name)
    if compiled is None:
        try:
            compiled = torch.compile(fn, dynamic=True, fullgraph=True, mode="default")
            _RENDER_COMPILED_HELPERS[name] = compiled
        except Exception:
            _RENDER_DISABLED_HELPERS.add(name)
            return fn(*args)
    try:
        candidate = compiled(*args)
        if name not in _RENDER_VALIDATED_HELPERS:
            reference = fn(*args)
            if not _compiled_helper_matches(name, candidate, reference):
                _RENDER_DISABLED_HELPERS.add(name)
                _RENDER_COMPILED_HELPERS.pop(name, None)
                _RENDER_COMPILED_SIGNATURES.pop(name, None)
                return reference
            _RENDER_VALIDATED_HELPERS.add(name)
            _RENDER_COMPILED_SIGNATURES[name] = signature
        return candidate
    except Exception:
        _RENDER_DISABLED_HELPERS.add(name)
        _RENDER_COMPILED_HELPERS.pop(name, None)
        _RENDER_COMPILED_SIGNATURES.pop(name, None)
        return fn(*args)


def _pou_geometry_eager(
    meta_i: torch.Tensor,
    meta_f: torch.Tensor,
    ix_base: torch.Tensor,
    px_center: torch.Tensor,
    iy_base: torch.Tensor,
    py_center: torch.Tensor,
    canvas: float,
    halo: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one PoU group's grid, separable weight and flat destination index."""
    x = meta_f[:, 0]
    y = meta_f[:, 1]
    size = meta_f[:, 2]
    x1f = meta_f[:, 3]
    y1f = meta_f[:, 4]
    sidef = meta_f[:, 5]
    native_x = x1f[:, None] + px_center[None, :]
    native_y = y1f[:, None] + py_center[None, :]
    scale = float(canvas) / sidef
    rel_x = (native_x * scale[:, None] - x[:, None]) / size[:, None]
    rel_y = (native_y * scale[:, None] - y[:, None]) / size[:, None]
    gx = rel_x.mul(2.0).sub(1.0)
    gy = rel_y.mul(2.0).sub(1.0)
    grid = torch.stack([
        gx[:, None, :].expand(-1, py_center.shape[0], -1),
        gy[:, :, None].expand(-1, -1, px_center.shape[0]),
    ], dim=-1)
    weight = (
        raised_cosine(rel_y, float(halo))[:, :, None]
        * raised_cosine(rel_x, float(halo))[:, None, :]
    )

    x1 = meta_i[:, 1]
    y1 = meta_i[:, 2]
    side = meta_i[:, 3]
    pixel_offset = meta_i[:, 4]
    row_start = (
        (y1[:, None] + iy_base[None, :]) * side[:, None]
        + x1[:, None]
        + pixel_offset[:, None]
    )
    flat_idx = (row_start[:, :, None] + ix_base[None, None, :]).reshape(-1)
    return grid, weight, flat_idx


def _pou_scatter_source_eager(
    patches: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Materialize weighted channels + PoU weight in scatter-major layout."""
    channels = patches.shape[1]
    flat_values = (
        (patches * weight[:, None])
        .permute(1, 0, 2, 3)
        .reshape(channels, -1)
    )
    return torch.cat((flat_values, weight.reshape(1, -1)), dim=0)


def render_continuous_field_batch_optimized(
    token_fields: torch.Tensor,
    samples: Sequence[object],
    canvas_size: int,
    halo_ratio: float = 0.05,
    renderer_plan: Optional[
        tuple[tuple[tuple[int, int, int, int], ...], torch.Tensor, torch.Tensor]
    ] = None,
    compile_helpers: bool = True,
) -> list[torch.Tensor]:
    """Inference PoU renderer with one metadata upload and one token gather/batch.

    The hot loop receives views into two flat metadata tensors instead of one
    CUDA tensor per native (H,W) group. Tokens are gathered in the exact group
    order once before the loop. Pure geometry/source preparation can be fused by
    Inductor; any compiler failure falls back to the same eager equations.
    """
    if token_fields.ndim != 5:
        raise ValueError("token_fields must be [B,T,C,P,P]")
    batch, token_count, channels = token_fields.shape[:3]
    if len(samples) != batch:
        raise ValueError("renderer sample batch mismatch")
    device = token_fields.device
    sides_i = [int(sample.side) for sample in samples]
    pixel_offsets: list[int] = []
    cursor = 0
    for side in sides_i:
        pixel_offsets.append(cursor)
        cursor += side * side
    total_pixels = cursor

    if renderer_plan is None:
        specs, host_i, host_f = pack_renderer_plan_cpu(samples, pin_memory=False)
        renderer_plan = (
            specs,
            host_i.to(device),
            host_f.to(device),
        )
    group_specs, meta_i_all, meta_f_all = renderer_plan
    if meta_i_all.device != device:
        meta_i_all = meta_i_all.to(device, non_blocking=(device.type == "cuda"))
    if meta_f_all.device != device:
        meta_f_all = meta_f_all.to(device, non_blocking=(device.type == "cuda"))

    combined = torch.zeros(
        channels + 1, total_pixels, device=device, dtype=torch.float32
    )
    flat_tokens = token_fields.reshape(batch * token_count, *token_fields.shape[2:])
    # One gather replaces one index_select/gather per native shape group.
    ordered_tokens = flat_tokens.index_select(0, meta_i_all[:, 0]).float()
    halo = float(halo_ratio)
    canvas = float(int(canvas_size))

    for height, width, start, end in group_specs:
        if end <= start:
            continue
        meta_i = meta_i_all[start:end]
        meta_f = meta_f_all[start:end]
        ix_base, px_center = _renderer_aranges(device, int(width))
        iy_base, py_center = _renderer_aranges(device, int(height))
        grid, weight, flat_idx = _run_renderer_helper(
            "pou_geometry",
            _pou_geometry_eager,
            (meta_i, meta_f, ix_base, px_center, iy_base, py_center, canvas, halo),
            bool(compile_helpers),
        )
        patches = F.grid_sample(
            ordered_tokens[start:end], grid,
            mode="bicubic", padding_mode="border", align_corners=True,
        )
        scatter_src = _run_renderer_helper(
            "pou_scatter_source",
            _pou_scatter_source_eager,
            (patches, weight),
            bool(compile_helpers),
        )
        combined.scatter_add_(
            1, flat_idx[None].expand(channels + 1, -1), scatter_src
        )

    accum = combined[:channels]
    wsum = combined[channels]
    normalized = accum / wsum.clamp_min(1e-6)[None]
    return [
        normalized[:, offset:offset + side * side].reshape(channels, side, side)
        for offset, side in zip(pixel_offsets, sides_i)
    ]

def render_continuous_residual_batch(
    delta_tokens: torch.Tensor,
    leaves_cpu_batch: Sequence[Sequence[Sequence[int]]],
    sides: Sequence[int],
    canvas_size: int,
    halo_ratio: float = 0.05,
) -> list[torch.Tensor]:
    if delta_tokens.ndim != 5 or delta_tokens.shape[2] != 3:
        raise ValueError("delta_tokens must be [B,T,3,P,P]")
    return render_continuous_field_batch(
        delta_tokens, leaves_cpu_batch, sides, canvas_size, halo_ratio
    )


def render_continuous_residual(
    delta_tokens: torch.Tensor,
    leaves_cpu: Sequence[Sequence[int]],
    side: int,
    canvas_size: int,
    halo_ratio: float = 0.05,
) -> torch.Tensor:
    return render_continuous_residual_batch(
        delta_tokens.unsqueeze(0), [leaves_cpu], [int(side)], canvas_size, halo_ratio
    )[0]


def _logical_alpha_to_native(
    logical_alpha: np.ndarray,
    *,
    side: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if logical_alpha.ndim != 2:
        raise ValueError("logical lens alpha must be HxW")
    field = torch.from_numpy(np.asarray(logical_alpha, dtype=np.float32)).view(
        1, 1, logical_alpha.shape[0], logical_alpha.shape[1]
    ).to(device=device, dtype=torch.float32)
    if field.shape[-1] != int(side) or field.shape[-2] != int(side):
        field = F.interpolate(field, size=(int(side), int(side)), mode="bilinear", align_corners=False)
    return field[0].to(dtype=dtype).clamp(0.0, 1.0)


def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _blur_native_scalar(field: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Box-blur a native [1,H,W] scalar field without changing its size."""
    k = int(kernel_size)
    if k <= 1:
        return field
    if k % 2 == 0:
        raise ValueError("native blur kernel must be odd")
    if field.ndim != 3 or field.shape[0] != 1:
        raise ValueError(f"expected scalar field [1,H,W], got {tuple(field.shape)}")
    return F.avg_pool2d(field.float().unsqueeze(0), kernel_size=k, stride=1, padding=k // 2)[0]


def _blur_native_rgb(field: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Box-blur a native [3,H,W] RGB field without changing its size."""
    k = int(kernel_size)
    if k <= 1:
        return field.float()
    if k % 2 == 0:
        raise ValueError("native RGB blur kernel must be odd")
    if field.ndim != 3 or field.shape[0] != 3:
        raise ValueError(f"expected RGB field [3,H,W], got {tuple(field.shape)}")
    return F.avg_pool2d(field.float().unsqueeze(0), kernel_size=k, stride=1, padding=k // 2)[0]


def render_hybrid_clean_residual(
    x0_tokens: torch.Tensor,
    glare_tokens: torch.Tensor,
    native_glare: torch.Tensor,
    leaves_cpu: Sequence[Sequence[int]],
    side: int,
    canvas_size: int,
    halo_ratio: float,
    *,
    lens_alpha_native: Optional[torch.Tensor] = None,
    eye_prior_native: Optional[torch.Tensor] = None,
    reconstruction_low: float = 0.03,
    reconstruction_high: float = 0.10,
    reconstruction_blur_kernel: int = 11,
    eye_core_feather_kernel: int = 7,
    texture_lowpass_kernel: int = 15,
    native_hf_skin: float = 0.90,
    native_hf_eye: float = 0.30,
    native_hf_iris: float = 0.15,
    correction_hf_suppression: float = 0.75,
    low_clean_skin: float = 0.00,
    low_clean_eye: float = 0.35,
    low_clean_iris: float = 0.55,
    low_clean_highlight: float = 0.90,
    highlight_low: float = 0.82,
    highlight_high: float = 0.96,
    highlight_chroma_max: float = 0.16,
    learned_gate: Optional[nn.Module] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frequency-preserving renderer (TRUE native-HF variant).

    Low/mid frequency reconstruction and photographic high-frequency texture are handled
    separately instead of asking one alpha map to choose an entire RGB image branch.

      clean_candidate    = R(x0)
      residual_candidate = G_native + R(x0 - G_token)

    Both candidates are split with the same native low-pass filter.  Low-frequency
    mixing is deliberately *semantic and smooth*, not leaf-size/block driven:

      output_low  = a_low * low(clean) + (1-a_low) * low(residual)
      output_high = beta * high(native photo) + (1-beta) * high(clean)
      output      = output_low + output_high

    a_low is near zero on ordinary lens skin (so the residual branch keeps the native
    tone/illumination base), moderate on semantic eye/iris pixels, and high only for
    strong specular highlights.  It does NOT use p4/p8 ownership, so the old block-shaped
    direct-clean alpha cannot imprint a different low-frequency skin domain inside the
    glasses.  beta separately preserves photographic high-frequency skin texture while
    reserving more generated detail for eye/iris pixels.  Inside the lens the
    photographic term is TRUE native-image high frequency, not high(residual_candidate).
    The final lens compositing is an explicit identity constraint: outside the
    feathered lens support, output pixels are copied from the native input exactly.
    """
    if x0_tokens.shape != glare_tokens.shape or x0_tokens.ndim != 4:
        raise ValueError("x0_tokens and glare_tokens must match [T,3,P,P]")
    if native_glare.ndim != 3 or native_glare.shape[0] != 3:
        raise ValueError("native_glare must be [3,H,W]")
    if native_glare.shape[-2:] != (int(side), int(side)):
        raise ValueError("native_glare size must match side")

    low_thr, high_thr = float(reconstruction_low), float(reconstruction_high)
    if low_thr < 0.0 or high_thr <= low_thr:
        raise ValueError("reconstruction thresholds require 0 <= low < high")
    for name, value in (
        ("native_hf_skin", native_hf_skin), ("native_hf_eye", native_hf_eye),
        ("native_hf_iris", native_hf_iris), ("correction_hf_suppression", correction_hf_suppression),
        ("low_clean_skin", low_clean_skin), ("low_clean_eye", low_clean_eye),
        ("low_clean_iris", low_clean_iris), ("low_clean_highlight", low_clean_highlight),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    if not 0.0 <= float(highlight_chroma_max) <= 1.0:
        raise ValueError("highlight_chroma_max must be in [0,1]")
    if not 0.0 <= float(highlight_low) < float(highlight_high) <= 1.0:
        raise ValueError("highlight thresholds require 0 <= low < high <= 1")

    clean_native = render_continuous_field_batch(
        x0_tokens.unsqueeze(0), [leaves_cpu], [int(side)], int(canvas_size), float(halo_ratio)
    )[0].float()
    delta_native = render_continuous_field_batch(
        (x0_tokens - glare_tokens).unsqueeze(0), [leaves_cpu], [int(side)], int(canvas_size), float(halo_ratio)
    )[0].float()
    residual_candidate = native_glare.float() + delta_native

    if lens_alpha_native is None:
        lens_native = torch.ones((1, int(side), int(side)), device=x0_tokens.device, dtype=torch.float32)
    else:
        lens_native = lens_alpha_native.to(device=x0_tokens.device, dtype=torch.float32)
        if lens_native.ndim == 2:
            lens_native = lens_native.unsqueeze(0)
        if lens_native.shape != (1, int(side), int(side)):
            raise ValueError(
                f"lens_alpha_native must have shape {(1, int(side), int(side))}, got {tuple(lens_native.shape)}"
            )
        lens_native = lens_native.clamp(0.0, 1.0)

    # A weak coherent-correction diagnostic.  Unlike V1/V2 it NEVER selects a whole
    # RGB branch; it can only attenuate native high-frequency injection by a bounded
    # amount, so model-vs-photo low-frequency bias cannot erase skin texture wholesale.
    k_corr = int(reconstruction_blur_kernel)
    if k_corr <= 0 or k_corr % 2 == 0:
        raise ValueError("reconstruction_blur_kernel must be a positive odd integer")
    signed_correction = _blur_native_rgb(delta_native, k_corr)
    correction_magnitude = signed_correction.abs().mean(dim=0, keepdim=True)
    reconstruction_need = _smoothstep01((correction_magnitude - low_thr) / (high_thr - low_thr))

    # Semantic masks. eye_prior encoding: periocular 64..127, eye 128..191, iris>=192.
    eye_mask = torch.zeros_like(lens_native)
    iris_mask = torch.zeros_like(lens_native)
    if eye_prior_native is not None:
        prior = eye_prior_native.to(device=x0_tokens.device)
        if prior.ndim == 2:
            prior = prior.unsqueeze(0)
        if prior.shape != (1, int(side), int(side)):
            raise ValueError(
                f"eye_prior_native must have shape {(1, int(side), int(side))}, got {tuple(prior.shape)}"
            )
        eye_mask = ((prior >= 128) & (prior < 192)).float()
        iris_mask = (prior >= 192).float()
        eye_mask = _smoothstep01(_blur_native_scalar(eye_mask, int(eye_core_feather_kernel)).clamp(0, 1))
        iris_mask = _smoothstep01(_blur_native_scalar(iris_mask, int(eye_core_feather_kernel)).clamp(0, 1))

    # Start from photographic texture on lens skin, then reserve more generated high
    # frequency for the eye/iris where the trained x0 carries eyelashes and iris detail.
    beta_inside = torch.full_like(lens_native, float(native_hf_skin))
    beta_inside = beta_inside * (1.0 - eye_mask) + float(native_hf_eye) * eye_mask
    beta_inside = beta_inside * (1.0 - iris_mask) + float(native_hf_iris) * iris_mask

    # Coherent correction strongly attenuates TRUE native HF where the model
    # consistently requests a correction (default max 75%). This is safe because this
    # gate only controls the high-frequency injection, never the whole RGB branch.
    beta_inside = beta_inside * (1.0 - float(correction_hf_suppression) * reconstruction_need)

    # Explicit near-white specular suppression. Work in [0,1] RGB. A highlight must be
    # both bright and low-chroma; bright skin therefore is much less likely to be removed.
    rgb01 = ((native_glare.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    lum = rgb01.mean(dim=0, keepdim=True)
    chroma = rgb01.amax(dim=0, keepdim=True) - rgb01.amin(dim=0, keepdim=True)
    bright = _smoothstep01((lum - float(highlight_low)) / (float(highlight_high) - float(highlight_low)))
    white = _smoothstep01((float(highlight_chroma_max) - chroma) / max(float(highlight_chroma_max), 1e-6))
    highlight_suppression = (bright * white * lens_native).clamp(0.0, 1.0)

    # LOW-FREQUENCY CLEAN GATE: smooth semantic/specular policy, independent of leaf size.
    # Ordinary lens skin stays on residual_low by default, preserving the same native
    # low-frequency skin/tone domain as the surrounding face.  Eye/iris get a moderate
    # clean contribution for reconstructed anatomy; near-white specular glare can force
    # clean_low strongly.  No p4/p8 ownership enters this gate, eliminating block alpha.
    low_inside = torch.full_like(lens_native, float(low_clean_skin))
    low_inside = torch.maximum(low_inside, float(low_clean_eye) * eye_mask)
    low_inside = torch.maximum(low_inside, float(low_clean_iris) * iris_mask)
    low_inside = torch.maximum(low_inside, float(low_clean_highlight) * highlight_suppression)
    low_clean_alpha = (lens_native * low_inside).clamp(0.0, 1.0)

    beta_inside = beta_inside * (1.0 - highlight_suppression)
    beta_inside = beta_inside.clamp(0.0, 1.0)

    # These gates are defined over the full canvas for convenient diagnostics. The
    # final compositor below is the actual hard outside-lens identity constraint.
    native_hf_weight = (1.0 - lens_native) + lens_native * beta_inside
    native_hf_weight = native_hf_weight.clamp(0.0, 1.0)

    if learned_gate is not None:
        low_clean_alpha, native_hf_weight = learned_gate(
            native_glare.float(), clean_native, delta_native,
            correction_magnitude, eye_mask, iris_mask, lens_native,
            low_clean_alpha, native_hf_weight,
            correction_normalizer=high_thr,
        )

    k_tex = int(texture_lowpass_kernel)
    if k_tex <= 0 or k_tex % 2 == 0:
        raise ValueError("texture_lowpass_kernel must be a positive odd integer")
    clean_low = _blur_native_rgb(clean_native, k_tex)
    residual_low = _blur_native_rgb(residual_candidate, k_tex)
    native_low = _blur_native_rgb(native_glare.float(), k_tex)

    clean_high = clean_native - clean_low
    residual_high = residual_candidate - residual_low
    native_high = native_glare.float() - native_low

    # Critical V4 change: inside the spectacle support, the photographic HF source is
    # the TRUE native image rather than residual_candidate.  x0-glare correction can
    # contain the negative of pores/wrinkles (because x0 is smoother), so residual_high
    # is already texture-erased.  Re-injecting it cannot recover photographic skin.
    #
    # Cross-fade the photographic HF source across the feathered boundary. A final
    # spatial compositor still copies the input exactly wherever lens_native is zero.
    photographic_high = (1.0 - lens_native) * residual_high + lens_native * native_high

    output_low = low_clean_alpha * clean_low + (1.0 - low_clean_alpha) * residual_low
    output_high = native_hf_weight * photographic_high + (1.0 - native_hf_weight) * clean_high
    rendered = (output_low + output_high).clamp(-1.0, 1.0)
    prediction = (
        (1.0 - lens_native) * native_glare.float() + lens_native * rendered
    ).clamp(-1.0, 1.0)
    actual_delta = prediction - native_glare.float()
    return (
        prediction, actual_delta, low_clean_alpha, reconstruction_need,
        correction_magnitude, native_hf_weight, highlight_suppression,
    )


class MaskFreeResidualPMF(nn.Module):
    """C24 old-V16 objective + multi-low anchor + TRUE-native-HF frequency-preserving renderer with semantic low-frequency mixing.

    The detail-preserving core remains the original V16-style per-image adaptive-
    normalized MeanFlow U/V objective.  A deliberately weak direct one-step x0 MSE
    term is added only as a fidelity anchor so recovered eyelashes/iris texture are
    nudged toward the paired clean target without returning to the strong smoothing
    pressure of V29.4.  Structure loss remains disabled.
    """

    def __init__(self, args: argparse.Namespace, *, enable_training_v: bool = True) -> None:
        super().__init__()
        self.enable_training_v = bool(enable_training_v)
        self.backbone = pmfDiT_B_16(
            input_size=args.canvas_size,
            patch_size=args.base_patch_size,
            canonical_size=args.canonical_size,
            in_channels=3,
            num_classes=1,
            num_class_tokens=4,
            num_time_tokens=4,
            num_cfg_tokens=0,
            num_interval_tokens=0,
            eval_mode=False,
            use_sdpa=not args.no_sdpa,
            gradient_checkpointing=args.gradient_checkpointing,
            # V is an auxiliary training head only. one_step_x0()/predict() request
            # need_v=False, so inference never executes the V branch.
            enable_v_head=self.enable_training_v,
            region_condition_dim=(
                4 if getattr(self, "use_semantic_conditioning", True) else 0
            ),
        )
        self.canvas_size = int(args.canvas_size)
        self.base_patch_size = int(args.base_patch_size)
        self.canonical_size = int(args.canonical_size)
        self.target_tokens = int(args.target_tokens)
        self.t_eps = float(args.t_eps)
        self.p_mean = float(args.p_mean)
        self.p_std = float(args.p_std)
        self.fm_ratio = float(args.fm_ratio)
        self.noise_scale = float(args.noise_scale)
        self.halo_ratio = float(args.halo_ratio)
        self.renderer_lens_expand_x = float(args.renderer_lens_expand_x)
        self.renderer_lens_expand_y = float(args.renderer_lens_expand_y)
        self.renderer_lens_min_height_ratio = float(args.renderer_lens_min_height_ratio)
        self.renderer_lens_feather = float(args.renderer_lens_feather)
        self.renderer_lens_power = float(args.renderer_lens_power)
        self.reconstruction_low = float(args.reconstruction_low)
        self.reconstruction_high = float(args.reconstruction_high)
        self.reconstruction_blur_kernel = int(args.reconstruction_blur_kernel)
        self.eye_core_feather_kernel = int(args.eye_core_feather_kernel)
        self.texture_lowpass_kernel = int(args.texture_lowpass_kernel)
        self.native_hf_skin = float(args.native_hf_skin)
        self.native_hf_eye = float(args.native_hf_eye)
        self.native_hf_iris = float(args.native_hf_iris)
        self.correction_hf_suppression = float(args.correction_hf_suppression)
        self.low_clean_skin = float(args.low_clean_skin)
        self.low_clean_eye = float(args.low_clean_eye)
        self.low_clean_iris = float(args.low_clean_iris)
        self.low_clean_highlight = float(args.low_clean_highlight)
        self.highlight_low = float(args.highlight_low)
        self.highlight_high = float(args.highlight_high)
        self.highlight_chroma_max = float(args.highlight_chroma_max)
        self.prior_dropout = float(args.eye_prior_dropout)
        self.norm_p = float(args.norm_p)
        self.norm_eps = float(args.norm_eps)
        self.lambda_u = float(args.lambda_u)
        self.lambda_v = float(args.lambda_v)
        # Low-frequency x0 fidelity anchor.  Full-resolution x0 MSE is deliberately
        # NOT optimized: the old adaptive U/V objective owns the 24x24 high-frequency
        # detail.  The two area-pooled scales constrain only medium/coarse fidelity.
        self.lambda_x0 = float(args.lambda_x0)
        self.x0_mid_size = int(args.x0_mid_size)
        self.x0_coarse_size = int(args.x0_coarse_size)
        self.x0_coarse_weight = float(args.x0_coarse_weight)

        # V29.4 summed pixel/gradient errors inside each canonical token. Moving
        # from c16 to c24 would otherwise silently multiply U/V/x0 loss scale by
        # (24/16)^2=2.25 and the first-order structure scale by about 2.30. Keep
        # the original c16 optimization scale while changing only representation
        # bandwidth. The logical base patch remains the V29.4 reference (16).
        ref = int(self.base_patch_size)
        cur = int(self.canonical_size)
        self.canonical_pixel_loss_scale = float(ref * ref) / float(cur * cur)

    @staticmethod
    def _offsets(batch_size: int, token_count: int, device: torch.device) -> torch.Tensor:
        return torch.arange(batch_size + 1, device=device, dtype=torch.long) * int(token_count)

    def _sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        def draw():
            z = torch.randn(batch_size, 1, 1, 1, 1, device=device, dtype=dtype)
            return torch.sigmoid(z * self.p_std + self.p_mean)
        t, r = draw(), draw()
        fm = torch.rand(batch_size, 1, 1, 1, 1, device=device) < self.fm_ratio
        r = torch.where(fm, t, r)
        return torch.maximum(t, r), torch.minimum(t, r)

    def _condition_batch(self, region_probs: torch.Tensor, training_dropout: bool) -> torch.Tensor:
        if not training_dropout or self.prior_dropout <= 0:
            return region_probs
        batch = region_probs.shape[0]
        keep = (torch.rand(batch, 1, 1, device=region_probs.device) >= self.prior_dropout).to(region_probs.dtype)
        neutral = torch.zeros_like(region_probs)
        neutral[..., 0] = 1.0
        return region_probs * keep + neutral * (1.0 - keep)

    def _network_batch(self, state: torch.Tensor, t: torch.Tensor, r: torch.Tensor,
                       leaves: torch.Tensor, region_probs: torch.Tensor,
                       *, need_v: bool) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if state.ndim != 5:
            raise ValueError(f"state must be [B,T,C,P,P], got {tuple(state.shape)}")
        batch, tokens = state.shape[:2]
        if tokens != self.target_tokens:
            raise ValueError(f"expected exactly {self.target_tokens} adaptive tokens, got {tokens}")
        flat_state = state.reshape(batch * tokens, *state.shape[2:])
        flat_leaves = leaves.reshape(batch * tokens, 3)
        flat_region = region_probs.reshape(batch * tokens, region_probs.shape[-1])
        offsets = self._offsets(batch, tokens, state.device)
        labels = torch.zeros(batch, device=state.device, dtype=torch.long)
        u_x0, v_x0 = self.backbone.forward_adaptive(
            flat_state, offsets, flat_leaves, t.reshape(batch), (t-r).reshape(batch), labels,
            self.canvas_size, need_v=need_v, padded_image_tokens=tokens,
            region_probs=flat_region,
        )
        u_x0 = u_x0.reshape(batch, tokens, *state.shape[2:])
        u = (state - u_x0) / t.clamp_min(self.t_eps)
        if not need_v:
            return u, None
        if v_x0 is None:
            raise RuntimeError("V head was requested but backbone returned None")
        v_x0 = v_x0.reshape(batch, tokens, *state.shape[2:])
        v = (state - v_x0) / t.clamp_min(self.t_eps)
        return u, v

    def _adaptive_normalize(self, per_sample_loss: torch.Tensor) -> torch.Tensor:
        """Original V16 detached per-image adaptive normalization.

        For the default norm_p=1 this scales each image gradient approximately by
        1 / (per_sample_loss + norm_eps), preventing large-error/glare-heavy images
        from monopolizing the batch gradient while preserving pressure on small
        residual eye-detail errors.
        """
        weight = (per_sample_loss + self.norm_eps).pow(self.norm_p)
        return per_sample_loss / weight.detach()

    def _low_frequency_token_error(
        self, pred: torch.Tensor, target: torch.Tensor, out_size: int
    ) -> torch.Tensor:
        """Per-token low-frequency MSE at an area-pooled spatial scale.

        ``pred``/``target`` are [B,T,3,P,P].  We area-pool P->out_size before
        computing MSE, so intra-cell variation (eyelashes, pores, iris microtexture)
        is invisible to this anchor as long as the pooled average is correct.

        The summed error is rescaled to the old c16 pixel-loss convention.  Thus
        12x12 and 6x6 terms have comparable numerical scale despite containing
        different numbers of pixels.
        """
        if pred.shape != target.shape or pred.ndim != 5:
            raise ValueError("pred/target must match [B,T,C,P,P]")
        size = int(out_size)
        if size <= 0 or size > int(pred.shape[-1]):
            raise ValueError(f"invalid low-frequency size {size} for P={pred.shape[-1]}")
        b, t, c, _, _ = pred.shape
        pred_low = F.interpolate(
            pred.float().reshape(b * t, c, pred.shape[-2], pred.shape[-1]),
            size=(size, size), mode="area",
        ).reshape(b, t, c, size, size)
        target_low = F.interpolate(
            target.float().reshape(b * t, c, target.shape[-2], target.shape[-1]),
            size=(size, size), mode="area",
        ).reshape(b, t, c, size, size)
        reference_area = float(self.base_patch_size * self.base_patch_size)
        scale = reference_area / float(size * size)
        return (pred_low - target_low).square().flatten(2).sum(2) * scale

    def _meanflow_loss_batch(self, clean: torch.Tensor, glare: torch.Tensor, leaves: torch.Tensor,
                             region_probs: torch.Tensor
                             ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.enable_training_v:
            raise RuntimeError("training loss requires the auxiliary V head")
        batch, tokens = clean.shape[:2]
        t, r = self._sample_time(batch, clean.device, clean.dtype)
        source = make_source(glare, self.noise_scale)
        state = (1.0 - t) * clean + t * source
        v_target = ((state - clean) / t.clamp_min(self.t_eps)).detach()

        # V16-style training: the JVP tangent remains the ground-truth instantaneous
        # velocity.  V is returned as an auxiliary branch and supervised directly;
        # it is not required at inference.
        def fn(state_arg, t_arg, r_arg):
            u, v = self._network_batch(
                state_arg, t_arg, r_arg, leaves, region_probs, need_v=True
            )
            if v is None:
                raise RuntimeError("training-only V head unexpectedly missing")
            return u, v

        original_checkpoint = self.backbone.gradient_checkpointing
        self.backbone.set_gradient_checkpointing(False)
        try:
            u, du_dt, v = torch.func.jvp(
                fn,
                (state, t, r),
                (v_target, torch.ones_like(t), torch.zeros_like(r)),
                has_aux=True,
            )
        finally:
            self.backbone.set_gradient_checkpointing(original_checkpoint)

        capital_v = u + (t-r) * du_dt.detach()
        error_u = capital_v - v_target
        error_v = v - v_target

        # Restore the original V16 optimization dynamics.  First sum the squared
        # canonical error inside every token, then sum all token errors belonging to
        # the same image.  Only after that do adaptive normalization.  This is
        # intentionally *not* uniform averaging of raw token MSE.
        token_u = error_u.float().square().flatten(2).sum(2) * self.canonical_pixel_loss_scale
        token_v = error_v.float().square().flatten(2).sum(2) * self.canonical_pixel_loss_scale
        per_sample_u = token_u.sum(dim=1)
        per_sample_v = token_v.sum(dim=1)
        adaptive_u = self._adaptive_normalize(per_sample_u)
        adaptive_v = self._adaptive_normalize(per_sample_v)
        loss_u = adaptive_u.mean()
        loss_v = adaptive_v.mean()
        return loss_u, loss_v, {
            "u": loss_u.detach(),
            "v": loss_v.detach(),
            "adaptive_u": adaptive_u.mean().detach(),
            "adaptive_v": adaptive_v.mean().detach(),
            "sample_u_raw": per_sample_u.mean().detach(),
            "sample_v_raw": per_sample_v.mean().detach(),
            "raw_u": error_u.float().square().mean().detach(),
            "raw_v": error_v.float().square().mean().detach(),
            "meanflow_t": t.mean().detach(),
            "meanflow_interval": (t-r).mean().detach(),
        }

    def one_step_x0_batch(self, glare: torch.Tensor, leaves: torch.Tensor, region_probs: torch.Tensor,
                          *, seed: Optional[int] = None) -> torch.Tensor:
        if glare.ndim != 5:
            raise ValueError(f"glare must be [B,T,C,P,P], got {tuple(glare.shape)}")
        batch, tokens = glare.shape[:2]
        generator = None
        if seed is not None:
            generator = torch.Generator(device=glare.device)
            generator.manual_seed(int(seed))
        source = make_source(glare, self.noise_scale, generator=generator)
        flat_source = source.reshape(batch * tokens, *source.shape[2:])
        flat_leaves = leaves.reshape(batch * tokens, 3)
        flat_region = region_probs.reshape(batch * tokens, region_probs.shape[-1])
        offsets = self._offsets(batch, tokens, glare.device)
        labels = torch.zeros(batch, device=glare.device, dtype=torch.long)
        ones = torch.ones(batch, device=glare.device, dtype=glare.dtype)
        x0, _ = self.backbone.forward_adaptive(
            flat_source, offsets, flat_leaves, ones, ones, labels, self.canvas_size,
            need_v=False, padded_image_tokens=tokens, region_probs=flat_region,
        )
        return x0.reshape(batch, tokens, *glare.shape[2:])

    def one_step_x0(self, glare: torch.Tensor, leaves: torch.Tensor, region_probs: torch.Tensor,
                    *, seed: Optional[int] = None) -> torch.Tensor:
        return self.one_step_x0_batch(
            glare.unsqueeze(0), leaves.unsqueeze(0), region_probs.unsqueeze(0), seed=seed
        )[0]

    def _renderer_lens_native(self, sample: Dict[str, object], dtype: torch.dtype) -> torch.Tensor:
        # Renderer lens support is deliberately separate from the cache/grid lens
        # rectangle.  That lets V29.4 checkpoints and cache-v4 leaves be tested
        # immediately while using a broader rotated rounded spectacle support.
        logical = build_renderer_lens_alpha(
            sample.get("metadata", {}),
            tuple(sample["original_size"]),
            canvas_size=self.canvas_size,
            expand_x=self.renderer_lens_expand_x,
            expand_y=self.renderer_lens_expand_y,
            min_height_ratio=self.renderer_lens_min_height_ratio,
            feather=self.renderer_lens_feather,
            superellipse_power=self.renderer_lens_power,
        )
        if logical is not None:
            return _logical_alpha_to_native(
                logical, side=int(sample["native_side"]),
                device=sample["glare_patches"].device, dtype=dtype,
            )
        # Fallback for old/incomplete face metadata: use the V29.4 binary geometry
        # mask already prepared from inference-time information.
        fallback = sample["native_lens_u8"].to(
            device=sample["glare_patches"].device, dtype=dtype
        ) / 255.0
        return fallback.clamp(0.0, 1.0)

    def decode(self, x0: torch.Tensor, sample: Dict[str, object]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lens_native = self._renderer_lens_native(sample, torch.float32)
        prior_native = (
            sample.get("native_prior_u8")
            if getattr(self, "use_semantic_conditioning", True)
            else None
        )
        pred, delta, alpha, reconstruction_need, correction_magnitude, native_hf_weight, highlight_suppression = render_hybrid_clean_residual(
            x0, sample["glare_patches"], sample["native_glare"],
            sample["leaves_cpu"], int(sample["native_side"]),
            self.canvas_size, self.halo_ratio,
            lens_alpha_native=lens_native, eye_prior_native=prior_native,
            reconstruction_low=self.reconstruction_low, reconstruction_high=self.reconstruction_high,
            reconstruction_blur_kernel=self.reconstruction_blur_kernel,
            eye_core_feather_kernel=self.eye_core_feather_kernel,
            texture_lowpass_kernel=self.texture_lowpass_kernel,
            native_hf_skin=self.native_hf_skin, native_hf_eye=self.native_hf_eye, native_hf_iris=self.native_hf_iris,
            correction_hf_suppression=self.correction_hf_suppression,
            low_clean_skin=self.low_clean_skin, low_clean_eye=self.low_clean_eye,
            low_clean_iris=self.low_clean_iris, low_clean_highlight=self.low_clean_highlight,
            highlight_low=self.highlight_low, highlight_high=self.highlight_high,
            highlight_chroma_max=self.highlight_chroma_max,
            learned_gate=(
                getattr(self, "learned_frequency_gate", None)
                if getattr(self, "use_learned_frequency_renderer", False)
                else None
            ),
        )
        return pred, delta, alpha, lens_native, reconstruction_need, correction_magnitude, native_hf_weight, highlight_suppression

    def forward_loss_batch(self, batch_data: Dict[str, object]) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        clean = batch_data["clean_patches"]
        glare = batch_data["glare_patches"]
        leaves = batch_data["leaves"]
        region = self._condition_batch(batch_data["region_probs"], training_dropout=self.training)
        token_count = clean.shape[1]
        if token_count != self.target_tokens:
            raise ValueError(f"expected {self.target_tokens} tokens, got {token_count}")

        # Detail-preserving core: original V16-style adaptive-normalized MeanFlow U/V.
        loss_u, loss_v, metrics = self._meanflow_loss_batch(
            clean, glare, leaves, region
        )

        # Multi-scale LOW-FREQUENCY one-step x0 anchor.  The full 24x24 x0 MSE is
        # intentionally absent.  24->12 constrains medium-scale eye/eyelid/iris
        # geometry and tone; 24->6 constrains coarser color/shape.  High-frequency
        # detail remains governed by the old adaptive-normalized U/V objective.
        x0 = self.one_step_x0_batch(glare, leaves, region)
        token_x0_mid = self._low_frequency_token_error(x0, clean, self.x0_mid_size)
        token_x0_coarse = self._low_frequency_token_error(x0, clean, self.x0_coarse_size)
        x0_mid_loss = token_vote_loss(token_x0_mid, self.target_tokens)
        x0_coarse_loss = token_vote_loss(token_x0_coarse, self.target_tokens)
        x0_loss = x0_mid_loss + self.x0_coarse_weight * x0_coarse_loss

        loss = (
            self.lambda_u * loss_u
            + self.lambda_v * loss_v
            + self.lambda_x0 * x0_loss
        )
        metrics.update({
            "loss": loss.detach(),
            "x0": x0_loss.detach(),
            "x0_low_mid": x0_mid_loss.detach(),
            "x0_low_coarse": x0_coarse_loss.detach(),
            # Logging only: full-resolution x0 MSE never enters backward.
            "x0_raw": (x0.float() - clean.float()).square().mean().detach(),
            "delta_abs": (x0.float() - glare.float()).abs().mean().detach(),
        })
        return loss, metrics

    def forward_loss(self, sample: Dict[str, object]) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = collate_batch([sample])
        batch = move_batch(batch, sample["glare_patches"].device)
        return self.forward_loss_batch(batch)

    @torch.inference_mode()
    def predict(self, sample: Dict[str, object], seed: int = 0):
        region = sample["region_probs"]
        x0 = self.one_step_x0(sample["glare_patches"], sample["leaves"], region, seed=seed)
        pred, delta, alpha, lens_native, reconstruction_need, correction_magnitude, native_hf_weight, highlight_suppression = self.decode(x0, sample)
        return pred, delta, x0, alpha, lens_native, reconstruction_need, correction_magnitude, native_hf_weight, highlight_suppression


def _inference_subset(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {str(k): v for k, v in state.items() if torch.is_tensor(v) and not _is_v_key(str(k))}


def configure_cuda_speed(device: torch.device) -> None:
    """Safe throughput-oriented CUDA switches; no approximate model changes."""
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    # SDPA itself still decides whether flash/memory-efficient kernels are legal
    # for the current attention shape/mask.
    for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
        fn = getattr(torch.backends.cuda, name, None)
        if callable(fn):
            fn(True)


# Preserve the former core-module namespace used by the inference/model code.
base = SimpleNamespace(
    MaskFreeResidualPMF=MaskFreeResidualPMF,
    _inference_subset=_inference_subset,
    _blur_native_rgb=_blur_native_rgb,
    _blur_native_scalar=_blur_native_scalar,
    _smoothstep01=_smoothstep01,
    render_continuous_field_batch=render_continuous_field_batch,
    token_vote_loss=token_vote_loss,
    configure_cuda_speed=configure_cuda_speed,
)


# --- inlined from model.py ---
def load_inference_state(model:torch.nn.Module,state:Mapping[str,torch.Tensor]):
    """Load only checkpoints created by this cleaned implementation."""
    return model.load_state_dict(state,strict=True)


class CleanResidualPMF(base.MaskFreeResidualPMF):
    def __init__(self,args,*,enable_training_v:bool=True):
        self.use_geometry_token=not bool(getattr(args,'disable_geometry_token',False))
        self.use_semantic_conditioning=not bool(getattr(args,'disable_semantic_conditioning',False))
        self.use_coarse_to_fine_context=not bool(getattr(args,'disable_coarse_to_fine_context',False))
        self.use_geometry_guided_source=not bool(getattr(args,'disable_geometry_guided_source',False))
        self.use_adaptive_grid=not bool(getattr(args,'uniform_grid',False))
        self.use_frequency_renderer=True
        self.use_learned_frequency_renderer=(
            bool(getattr(args,'learned_frequency_renderer',True))
        )
        super().__init__(args,enable_training_v=enable_training_v)
        self.lambda_eye_gradient=(
            0.0 if not self.use_semantic_conditioning
            else float(getattr(args,'lambda_eye_gradient',0.005))
        )
        self.eye_gradient_periocular_weight=float(
            getattr(args,'eye_gradient_periocular_weight',0.05)
        )
        self.eye_gradient_eye_weight=float(getattr(args,'eye_gradient_eye_weight',1.0))
        self.eye_gradient_iris_weight=float(getattr(args,'eye_gradient_iris_weight',2.0))
        self.lambda_preserve=float(getattr(args,'lambda_preserve',0.01))
        self.conditioner=AdaptiveConditioner(
            canonical_size=args.canonical_size,
            hidden_size=self.backbone.hidden_size,
            num_heads=self.backbone.num_heads,
            bottleneck=int(getattr(args,'conditioner_bottleneck',128)),
            canvas_size=args.canvas_size,
            base_patch=args.base_patch_size,
            condition_gate_init=float(getattr(args,'condition_gate_init',0.1)),
            context_gate_init=float(getattr(args,'context_gate_init',0.05)),
            use_geometry_token=self.use_geometry_token,
            use_coarse_to_fine_context=self.use_coarse_to_fine_context,
        )
        self.learned_frequency_gate=LearnedFrequencyGate(
            width=int(getattr(args,'renderer_gate_width',32)),
            map_size=int(getattr(args,'renderer_gate_map_size',256)),
            residual_scale=float(getattr(args,'renderer_gate_residual_scale',1.0)),
        )

    def _backbone_region_probs(self,region_probs):
        return region_probs if self.use_semantic_conditioning else None

    def _semantic_region_probs(self,region_probs):
        if self.use_semantic_conditioning:
            return region_probs
        neutral=torch.zeros_like(region_probs)
        neutral[...,0]=1
        return neutral

    def _condition_tokens(self,state,glare,leaves,region_probs):
        b,t=state.shape[:2]
        region_probs=self._semantic_region_probs(region_probs)
        raw=self.backbone._canonical_patch_embed(state.reshape(b*t,*state.shape[2:])).reshape(b,t,self.backbone.hidden_size)
        return self.conditioner.forward_from_state_tokens(raw,glare,region_probs,leaves).conditioned_state

    def _network_batch(self,state,t,r,leaves,region_probs,*,need_v:bool,
                       glare_patches:Optional[torch.Tensor]=None):
        if glare_patches is None:
            raise ValueError('adaptive PMF requires glare_patches for conditioning')
        batch,tokens=state.shape[:2]
        conditioned=self._condition_tokens(state,glare_patches,leaves,region_probs)
        flat_state=state.reshape(batch*tokens,*state.shape[2:])
        flat_leaves=leaves.reshape(batch*tokens,3)
        model_region=self._semantic_region_probs(region_probs)
        flat_region=model_region.reshape(batch*tokens,model_region.shape[-1])
        flat_cond=conditioned.reshape(batch*tokens,self.backbone.hidden_size)
        offsets=self._offsets(batch,tokens,state.device)
        labels=torch.zeros(batch,device=state.device,dtype=torch.long)
        ux0,vx0=self.backbone.forward_adaptive(
            flat_state,offsets,flat_leaves,t.reshape(batch),(t-r).reshape(batch),labels,
            self.canvas_size,need_v=need_v,padded_image_tokens=tokens,
            region_probs=self._backbone_region_probs(flat_region),image_tokens_override=flat_cond,
        )
        ux0=ux0.reshape(batch,tokens,*state.shape[2:])
        u=(state-ux0)/t.clamp_min(self.t_eps)
        if not need_v: return u,None
        if vx0 is None: raise RuntimeError('V head requested but missing')
        vx0=vx0.reshape(batch,tokens,*state.shape[2:])
        return u,(state-vx0)/t.clamp_min(self.t_eps)

    def _source_support(self,leaves,region_probs):
        if not self.use_geometry_guided_source:
            return torch.ones_like(region_probs[...,0])
        model_region=self._semantic_region_probs(region_probs)
        return self.conditioner.geometry_support(leaves,model_region)

    def _geometry_source(self,glare,leaves,region_probs,generator=None):
        support=self._source_support(leaves,region_probs)[...,None,None,None].to(glare.dtype)
        noise=torch.randn(glare.shape,device=glare.device,dtype=glare.dtype,generator=generator)
        return (glare+support*noise*self.noise_scale).clamp(-1,1)

    def _preservation_support(self,leaves,region_probs):
        """Inference-time geometry support, independent of the source-noise switch."""
        semantic=self._semantic_region_probs(region_probs)
        return self.conditioner.geometry_support(leaves,semantic)

    def _meanflow_loss_batch(self,clean,glare,leaves,region_probs):
        if not self.enable_training_v: raise RuntimeError('training requires V head')
        batch=clean.shape[0]
        t,r=self._sample_time(batch,clean.device,clean.dtype)
        source=self._geometry_source(glare,leaves,region_probs)
        state=(1-t)*clean+t*source
        v_target=((state-clean)/t.clamp_min(self.t_eps)).detach()
        def fn(state_arg,t_arg,r_arg):
            u,v=self._network_batch(state_arg,t_arg,r_arg,leaves,region_probs,need_v=True,
                                    glare_patches=glare)
            if v is None: raise RuntimeError('V missing')
            return u,v
        original=self.backbone.gradient_checkpointing
        self.backbone.set_gradient_checkpointing(False)
        try:
            u,du_dt,v=torch.func.jvp(fn,(state,t,r),(v_target,torch.ones_like(t),torch.zeros_like(r)),has_aux=True)
        finally:
            self.backbone.set_gradient_checkpointing(original)
        capital_v=u+(t-r)*du_dt.detach()
        eu=capital_v-v_target; ev=v-v_target
        token_u=eu.float().square().flatten(2).sum(2)*self.canonical_pixel_loss_scale
        token_v=ev.float().square().flatten(2).sum(2)*self.canonical_pixel_loss_scale
        per_u=token_u.sum(1); per_v=token_v.sum(1)
        au=self._adaptive_normalize(per_u); av=self._adaptive_normalize(per_v)
        lu=au.mean(); lv=av.mean()
        return lu,lv,{
            'u':lu.detach(),'v':lv.detach(),'adaptive_u':au.mean().detach(),'adaptive_v':av.mean().detach(),
            'sample_u_raw':per_u.mean().detach(),'sample_v_raw':per_v.mean().detach(),
            'raw_u':eu.float().square().mean().detach(),'raw_v':ev.float().square().mean().detach(),
            'meanflow_t':t.mean().detach(),'meanflow_interval':(t-r).mean().detach(),
        }

    def one_step_x0_batch(self,glare,leaves,region_probs,*,seed:Optional[int]=None):
        batch,tokens=glare.shape[:2]
        generator=None
        if seed is not None:
            generator=torch.Generator(device=glare.device); generator.manual_seed(int(seed))
        source=self._geometry_source(glare,leaves,region_probs,generator)
        cond=self._condition_tokens(source,glare,leaves,region_probs)
        fs=source.reshape(batch*tokens,*source.shape[2:]); fl=leaves.reshape(batch*tokens,3)
        fr=region_probs.reshape(batch*tokens,region_probs.shape[-1]); fc=cond.reshape(batch*tokens,self.backbone.hidden_size)
        offsets=self._offsets(batch,tokens,glare.device); labels=torch.zeros(batch,device=glare.device,dtype=torch.long)
        ones=torch.ones(batch,device=glare.device,dtype=glare.dtype)
        x0,_=self.backbone.forward_adaptive(fs,offsets,fl,ones,ones,labels,self.canvas_size,need_v=False,
            padded_image_tokens=tokens,region_probs=self._backbone_region_probs(fr),image_tokens_override=fc)
        return x0.reshape(batch,tokens,*glare.shape[2:])

    def one_step_x0(self,glare,leaves,region_probs,*,seed:Optional[int]=None):
        return self.one_step_x0_batch(
            glare.unsqueeze(0),leaves.unsqueeze(0),region_probs.unsqueeze(0),
            seed=seed,
        )[0]

    def _eye_gradient_loss(self,x0,clean,region_probs):
        pred=x0.float(); target=clean.float()
        dx=(pred[...,1:]-pred[...,:-1])-(target[...,1:]-target[...,:-1])
        dy=(pred[...,1:,:]-pred[...,:-1,:])-(target[...,1:,:]-target[...,:-1,:])
        token_error=0.5*(dx.abs().flatten(2).mean(2)+dy.abs().flatten(2).mean(2))
        weights=(
            self.eye_gradient_periocular_weight*region_probs[...,1].float()
            +self.eye_gradient_eye_weight*region_probs[...,2].float()
            +self.eye_gradient_iris_weight*region_probs[...,3].float()
        )
        denominator=weights.sum(1)
        per_image=(token_error*weights).sum(1)/denominator.clamp_min(1e-6)
        return torch.where(denominator>0,per_image,torch.zeros_like(per_image)).mean()

    @staticmethod
    def _preserve_loss(x0,glare,support,leaves):
        token_change=(x0.float()-glare.float()).abs().flatten(2).mean(2)
        outside=(1.0-support.float()).clamp(0.0,1.0)
        # Adaptive tokens cover different native areas. A p32 leaf represents 64x
        # the area of a p4 leaf and must therefore contribute 64x as much to an
        # outside-region preservation average.
        area=leaves[...,2].float().square()
        weight=outside*area
        denominator=weight.sum(1)
        per_image=(token_change*weight).sum(1)/denominator.clamp_min(1e-6)
        return torch.where(denominator>0,per_image,torch.zeros_like(per_image)).mean()

    def forward_loss_batch(self,batch_data:Dict[str,object],*,return_x0:bool=False):
        clean=batch_data['clean_patches']; glare=batch_data['glare_patches']; leaves=batch_data['leaves']
        supervision_region=batch_data['region_probs']
        region=self._condition_batch(supervision_region,training_dropout=self.training)
        loss_u,loss_v,metrics=self._meanflow_loss_batch(clean,glare,leaves,region)
        zero=loss_u.detach().new_zeros(())
        need_x0=(return_x0 or self.lambda_x0>0 or self.lambda_eye_gradient>0 or self.lambda_preserve>0)
        x0=None
        if need_x0:
            x0=self.one_step_x0_batch(glare,leaves,region)
        if self.lambda_x0<=0:
            x0_loss=mid=coarse=zero
        else:
            assert x0 is not None
            token_mid=self._low_frequency_token_error(x0,clean,self.x0_mid_size)
            token_coarse=self._low_frequency_token_error(x0,clean,self.x0_coarse_size)
            mid=base.token_vote_loss(token_mid,self.target_tokens)
            coarse=base.token_vote_loss(token_coarse,self.target_tokens)
            x0_loss=mid+self.x0_coarse_weight*coarse
        if x0 is None:
            eye_gradient=preserve=x0_raw=delta_abs=zero
            preserve_support=self._preservation_support(leaves,supervision_region)
        else:
            x0_raw=(x0.float()-clean.float()).square().mean().detach()
            delta_abs=(x0.float()-glare.float()).abs().mean().detach()
            eye_gradient=(
                self._eye_gradient_loss(x0,clean,supervision_region)
                if self.lambda_eye_gradient>0 else zero
            )
            preserve_support=self._preservation_support(leaves,supervision_region)
            preserve=(
                self._preserve_loss(x0,glare,preserve_support,leaves)
                if self.lambda_preserve>0 else zero
            )
        loss=(
            self.lambda_u*loss_u+self.lambda_v*loss_v+self.lambda_x0*x0_loss
            +self.lambda_eye_gradient*eye_gradient
            +self.lambda_preserve*preserve
        )
        metrics.update({'loss':loss.detach(),'x0':x0_loss.detach(),'x0_low_mid':mid.detach(),'x0_low_coarse':coarse.detach(),
                        'eye_gradient':eye_gradient.detach(),'preserve':preserve.detach(),
                        'x0_raw':x0_raw,'delta_abs':delta_abs})
        metrics.update({k:v.detach() for k,v in self.conditioner.gate_metrics().items()})
        source_support=self._source_support(leaves,region)
        metrics['geometry_support_mean']=source_support.mean().detach()
        metrics['preserve_outside_mean']=(1.0-preserve_support).mean().detach()
        if return_x0:
            assert x0 is not None
            return loss,metrics,x0
        return loss,metrics

    @torch.inference_mode()
    def predict(self,sample:Dict[str,object],seed:int=0):
        region=sample['region_probs']
        x0=self.one_step_x0(sample['glare_patches'],sample['leaves'],region,seed=seed)
        pred,delta,alpha,lens_native,reconstruction_need,correction_magnitude,native_hf_weight,highlight_suppression=self.decode(x0,sample)
        return pred,delta,x0,alpha,lens_native,reconstruction_need,correction_magnitude,native_hf_weight,highlight_suppression

    def decode(self,x0:torch.Tensor,sample:Dict[str,object]):
        if self.use_frequency_renderer:
            return super().decode(x0,sample)
        # Explicit renderer bypass: keep adaptive PMF and linear PoU rendering,
        # but bypass all low/high splitting, native-HF injection and semantic
        # frequency gates in render_hybrid_clean_residual().
        delta_native=base.render_continuous_field_batch(
            (x0-sample['glare_patches']).unsqueeze(0),
            [sample['leaves_cpu']],[int(sample['native_side'])],
            self.canvas_size,self.halo_ratio,
        )[0].float()
        native_glare=sample['native_glare'].float()
        lens_native=self._renderer_lens_native(sample,torch.float32)
        rendered=(native_glare+delta_native).clamp(-1,1)
        pred=((1.0-lens_native)*native_glare+lens_native*rendered).clamp(-1,1)
        delta=pred-native_glare
        zero=torch.zeros_like(lens_native)
        return pred,delta,zero,lens_native,zero,zero,zero,zero


# --- inlined from config.py ---
def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value}")
    return result


def _nonnegative(name: str, value: float) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return result


def _probability(name: str, value: float) -> float:
    result = _finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0,1], got {value}")
    return result


def _positive_odd(name: str, value: int) -> int:
    result = int(value)
    if result <= 0 or result % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {value}")
    return result


def make_grid_config(args: argparse.Namespace) -> AdaptiveGridConfig:
    return AdaptiveGridConfig(
        canvas_size=args.canvas_size,
        base_patch_size=args.base_patch_size,
        target_tokens=args.target_tokens,
        canvas_weight=args.canvas_weight,
        face_weight=args.face_weight,
        periocular_weight=args.periocular_weight,
        eye_weight=args.eye_weight,
        iris_weight=args.iris_weight,
        eyelid_weight=args.eyelid_weight,
        lens_weight=args.lens_weight,
        lens_expand_x=args.lens_expand_x,
        lens_expand_y=args.lens_expand_y,
        hard_lens_max_patch=args.hard_lens_max_patch,
        correction_threshold=args.correction_threshold,
        correction_dilate_radius=args.correction_dilate_radius,
        hard_correction_max_patch=args.hard_correction_max_patch,
        hard_anchor_radius=args.hard_anchor_radius,
        hard_anchor_max_patch=args.hard_anchor_max_patch,
    )


def validate_common_args(args: argparse.Namespace) -> None:
    """Validate model/grid/renderer values shared by training and checkpoints."""
    make_grid_config(args)
    if int(args.base_patch_size) != 16:
        raise ValueError("--base_patch_size must remain 16 for the current grid/checkpoint format")
    if int(args.canonical_size) <= 0:
        raise ValueError("--canonical_size must be > 0")

    _nonnegative("--noise_scale", args.noise_scale)
    _positive("--t_eps", args.t_eps)
    _finite("--p_mean", args.p_mean)
    _nonnegative("--p_std", args.p_std)
    _probability("--fm_ratio", args.fm_ratio)
    _nonnegative("--halo_ratio", args.halo_ratio)

    for name in ("renderer_lens_expand_x", "renderer_lens_expand_y", "renderer_lens_feather"):
        _nonnegative(f"--{name}", getattr(args, name))
    _positive("--renderer_lens_min_height_ratio", args.renderer_lens_min_height_ratio)
    if _finite("--renderer_lens_power", args.renderer_lens_power) < 2.0:
        raise ValueError("--renderer_lens_power must be >= 2")

    _nonnegative("--reconstruction_low", args.reconstruction_low)
    if _finite("--reconstruction_high", args.reconstruction_high) <= float(args.reconstruction_low):
        raise ValueError("--reconstruction_high must be > --reconstruction_low")
    for name in ("reconstruction_blur_kernel", "eye_core_feather_kernel", "texture_lowpass_kernel"):
        _positive_odd(f"--{name}", getattr(args, name))

    for name in (
        "native_hf_skin", "native_hf_eye", "native_hf_iris", "correction_hf_suppression",
        "low_clean_skin", "low_clean_eye", "low_clean_iris", "low_clean_highlight",
        "highlight_chroma_max", "eye_prior_dropout",
    ):
        _probability(f"--{name}", getattr(args, name))
    low = _probability("--highlight_low", args.highlight_low)
    high = _probability("--highlight_high", args.highlight_high)
    if low >= high:
        raise ValueError("--highlight_low/high require 0 <= low < high <= 1")

    if int(args.renderer_gate_width) <= 0 or int(args.renderer_gate_map_size) <= 0:
        raise ValueError("--renderer_gate_width and --renderer_gate_map_size must be positive")
    _nonnegative("--renderer_gate_residual_scale", args.renderer_gate_residual_scale)
    _nonnegative("--norm_p", args.norm_p)
    _positive("--norm_eps", args.norm_eps)
    for name in (
        "lambda_u", "lambda_v", "lambda_x0", "x0_coarse_weight",
        "lambda_eye_gradient", "eye_gradient_periocular_weight",
        "eye_gradient_eye_weight", "eye_gradient_iris_weight", "lambda_preserve",
    ):
        _nonnegative(f"--{name}", getattr(args, name))

    for name in ("x0_mid_size", "x0_coarse_size"):
        size = int(getattr(args, name))
        if size <= 0 or size > int(args.canonical_size):
            raise ValueError(f"--{name} must be in [1, canonical_size]")
        if int(args.canonical_size) % size:
            raise ValueError(f"--{name} must evenly divide canonical_size={args.canonical_size}")
    if int(args.x0_mid_size) <= int(args.x0_coarse_size):
        raise ValueError("--x0_mid_size must be larger than --x0_coarse_size")


def validate_inference_args(args: argparse.Namespace) -> None:
    """Validate inference-only CLI values before checkpoint/model allocation."""
    input_path = Path(args.input).expanduser()
    checkpoint = Path(args.checkpoint).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"--input does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"--input must be a single image file: {input_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"--checkpoint not found: {checkpoint}")
    if not str(args.output).strip():
        raise ValueError("--output must not be empty")
    for name in (
        "min_detection_confidence", "min_presence_confidence", "min_tracking_confidence",
    ):
        _probability(f"--{name}", getattr(args, name))
    if int(getattr(args, "max_faces", 1)) < 1:
        raise ValueError("--max_faces must be >= 1")
    if int(getattr(args, "geometry_detect_max_side", 0)) < 0:
        raise ValueError("--geometry_detect_max_side must be >= 0")
    if int(getattr(args, "geometry_retry_max_side", 0)) < 0:
        raise ValueError("--geometry_retry_max_side must be >= 0")
    for name in ("batch_size", "preprocess_workers", "prefetch_batches", "save_workers"):
        if int(getattr(args, name, 1)) < 1:
            raise ValueError(f"--{name} must be >= 1")
    for name in ("renderer_lens_expand_x", "renderer_lens_expand_y", "renderer_lens_feather"):
        value = getattr(args, name, None)
        if value is not None:
            _nonnegative(f"--{name}", value)
    minimum_height = getattr(args, "renderer_lens_min_height_ratio", None)
    if minimum_height is not None:
        _positive("--renderer_lens_min_height_ratio", minimum_height)
    if bool(args.force_redetect) and args.backend in ("auto", "tasks"):
        asset = Path(args.model_asset_path).expanduser()
        if not asset.is_file():
            raise FileNotFoundError(f"--model_asset_path not found: {asset}")


# Preserve the former config-module namespace used below.
config = SimpleNamespace(
    make_grid_config=make_grid_config,
    validate_common_args=validate_common_args,
    validate_inference_args=validate_inference_args,
)


# MediaPipe is loaded lazily only when online detection is needed.
mp = None


# --- inlined from prepare_geometry.py ---
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


LEFT_BROW = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]


RIGHT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]


LEFT_IRIS = [474, 475, 476, 477]


RIGHT_IRIS = [469, 470, 471, 472]


LEFT_INNER_CORNER = 362


LEFT_OUTER_CORNER = 263


RIGHT_INNER_CORNER = 133


RIGHT_OUTER_CORNER = 33


def valid_indices(indices: Sequence[int], count: int) -> list[int]:
    return [index for index in indices if 0 <= index < count]


def points_for(points: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    valid = valid_indices(indices, len(points))
    return points[valid] if valid else np.empty((0, 2), dtype=np.float32)


def sort_clockwise(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angle = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angle)]


def draw_polygon(prior: np.ndarray, points: np.ndarray, indices: Sequence[int], value: int) -> bool:
    selected = points_for(points, indices)
    if len(selected) < 3:
        return False
    polygon = sort_clockwise(selected).round().astype(np.int32)
    cv2.fillPoly(prior, [polygon], int(value))
    return True


def draw_iris(prior: np.ndarray, points: np.ndarray, indices: Sequence[int], value: int, dilation_ratio: float) -> bool:
    selected = points_for(points, indices)
    if len(selected) < 3:
        return False
    center = selected.mean(axis=0)
    radius = max(float(np.linalg.norm(selected - center[None], axis=1).mean()), 2.0)
    radius *= max(1.0, 1.0 + float(dilation_ratio))
    cv2.circle(
        prior,
        tuple(np.round(center).astype(np.int32).tolist()),
        max(1, int(round(radius))),
        int(value),
        thickness=-1,
    )
    return True


def clipped_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, int(math.floor(x1))))
    top = max(0, min(height - 1, int(math.floor(y1))))
    right = max(left + 1, min(width, int(math.ceil(x2))))
    bottom = max(top + 1, min(height, int(math.ceil(y2))))
    return left, top, right, bottom


def periocular_box(
    points: np.ndarray,
    eye_indices: Sequence[int],
    brow_indices: Sequence[int],
    width: int,
    height: int,
    horizontal_ratio: float,
    vertical_ratio: float,
) -> Optional[tuple[int, int, int, int]]:
    selected = np.concatenate([points_for(points, eye_indices), points_for(points, brow_indices)], axis=0)
    if len(selected) < 3:
        return None
    x1, y1 = selected.min(axis=0)
    x2, y2 = selected.max(axis=0)
    span_x = max(1.0, float(x2 - x1))
    span_y = max(1.0, float(y2 - y1))
    return clipped_box(
        x1 - horizontal_ratio * span_x,
        y1 - vertical_ratio * span_y,
        x2 + horizontal_ratio * span_x,
        y2 + 0.7 * vertical_ratio * span_y,
        width,
        height,
    )


def eye_center(points: np.ndarray, indices: Sequence[int]) -> Optional[np.ndarray]:
    selected = points_for(points, indices)
    return selected.mean(axis=0) if len(selected) else None


def normalized_group(points: np.ndarray, indices: Sequence[int], width: int, height: int) -> list[list[float]]:
    selected = points_for(points, indices)
    return [[float(x) / width, float(y) / height] for x, y in selected]


def normalized_point(points: np.ndarray, index: int, width: int, height: int) -> Optional[list[float]]:
    if not 0 <= int(index) < len(points):
        return None
    x, y = points[int(index)]
    return [float(x) / width, float(y) / height]


def normalized_center(points: np.ndarray, indices: Sequence[int], width: int, height: int) -> Optional[list[float]]:
    center = eye_center(points, indices)
    if center is None:
        return None
    return [float(center[0]) / width, float(center[1]) / height]


def landmark_groups_normalized(points: np.ndarray, width: int, height: int) -> dict:
    """Return the fixed eight-channel LRDA semantic landmark schema."""
    return {
        "left_eye_contour": normalized_group(points, LEFT_EYE, width, height),
        "right_eye_contour": normalized_group(points, RIGHT_EYE, width, height),
        "left_iris_center": normalized_center(points, LEFT_IRIS, width, height),
        "right_iris_center": normalized_center(points, RIGHT_IRIS, width, height),
        "left_inner_corner": normalized_point(points, LEFT_INNER_CORNER, width, height),
        "left_outer_corner": normalized_point(points, LEFT_OUTER_CORNER, width, height),
        "right_inner_corner": normalized_point(points, RIGHT_INNER_CORNER, width, height),
        "right_outer_corner": normalized_point(points, RIGHT_OUTER_CORNER, width, height),
    }


def _build_single_face_prior_and_meta(
    image_bgr: np.ndarray,
    points: Optional[np.ndarray],
    args: argparse.Namespace,
    *,
    face_index: int = 0,
) -> tuple[np.ndarray, dict]:
    """Build the semantic prior for one detected face."""
    height, width = image_bgr.shape[:2]
    prior = np.zeros((height, width), dtype=np.uint8)
    meta = {
        "face_index": int(face_index),
        "success": False,
        "fail_reason": "face_landmarker_failed",
        "source_width": width,
        "source_height": height,
        "roi_normalized": None,
        "roi_pixels": None,
        "eye_line_angle_degrees": 0.0,
        "has_eye": False,
        "has_iris": False,
        "prior_valid": False,
        "geometry_version": GEOMETRY_VERSION,
        "landmark_schema": LANDMARK_SCHEMA,
        "landmark_groups_normalized": None,
    }
    if points is None:
        return prior, meta

    # Keep whatever landmark groups are available even if bilateral geometry later
    # fails.  This lets partial eye information remain available to the renderer and
    # semantic conditioning when another face in the same image is valid.
    meta["landmark_groups_normalized"] = landmark_groups_normalized(
        points, width, height
    )
    meta["landmark_count"] = int(len(points))

    periocular_boxes = []
    for eye_indices, brow_indices in ((LEFT_EYE, LEFT_BROW), (RIGHT_EYE, RIGHT_BROW)):
        box = periocular_box(
            points,
            eye_indices,
            brow_indices,
            width,
            height,
            args.periocular_horizontal_ratio,
            args.periocular_vertical_ratio,
        )
        if box is not None:
            left, top, right, bottom = box
            periocular_boxes.append(box)
            prior[top:bottom, left:right] = np.maximum(
                prior[top:bottom, left:right], 64
            )
            meta["has_eye"] = True

    left_eye = draw_polygon(prior, points, LEFT_EYE, 128)
    right_eye = draw_polygon(prior, points, RIGHT_EYE, 128)
    left_iris = draw_iris(prior, points, LEFT_IRIS, 192, args.iris_dilation_ratio)
    right_iris = draw_iris(prior, points, RIGHT_IRIS, 192, args.iris_dilation_ratio)
    meta["has_eye"] = bool(meta["has_eye"] or left_eye or right_eye)
    meta["has_iris"] = bool(left_iris and right_iris)
    meta["prior_valid"] = bool(meta["has_eye"])

    if len(periocular_boxes) != 2:
        meta["fail_reason"] = "periocular_roi_failed"
        return prior, meta

    left = min(box[0] for box in periocular_boxes)
    top = min(box[1] for box in periocular_boxes)
    right = max(box[2] for box in periocular_boxes)
    bottom = max(box[3] for box in periocular_boxes)
    left_center = eye_center(points, LEFT_EYE)
    right_center = eye_center(points, RIGHT_EYE)
    if left_center is None or right_center is None:
        meta["fail_reason"] = "eye_center_failed"
        return prior, meta

    vector = left_center - right_center
    angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    meta.update(
        {
            "success": True,
            "fail_reason": "",
            "roi_pixels": [left, top, right, bottom],
            "roi_normalized": [
                left / width, top / height, right / width, bottom / height
            ],
            "eye_line_angle_degrees": angle,
            "roi_definition": "exact_union_of_yellow_periocular_boxes",
        }
    )
    return prior, meta


def build_prior_and_meta(
    image_bgr: np.ndarray,
    face_points: Optional[Sequence[np.ndarray] | np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    """Merge ocular geometry from all detected faces into one image prior.

    The model still runs exactly once on the complete image.  Multi-face support is
    implemented by unioning per-face semantic priors and retaining one metadata entry
    per face for lens support, structural anchors, and face-context construction.
    """
    height, width = image_bgr.shape[:2]
    merged_prior = np.zeros((height, width), dtype=np.uint8)

    if face_points is None:
        points_list: list[np.ndarray] = []
    elif isinstance(face_points, np.ndarray):
        points_list = [face_points]
    else:
        points_list = [
            np.asarray(points, dtype=np.float32)
            for points in face_points
            if points is not None
        ]

    faces = []
    for face_index, points in enumerate(points_list):
        face_prior, face_meta = _build_single_face_prior_and_meta(
            image_bgr, points, args, face_index=face_index
        )
        merged_prior = np.maximum(merged_prior, face_prior)
        faces.append(face_meta)

    valid_faces = [face for face in faces if bool(face.get("success"))]
    eye_faces = [face for face in faces if bool(face.get("has_eye"))]
    iris_faces = [face for face in faces if bool(face.get("has_iris"))]

    meta = {
        "success": bool(valid_faces),
        "fail_reason": "",
        "source_image": "input",
        "source_width": width,
        "source_height": height,
        "roi_normalized": None,
        "roi_pixels": None,
        "eye_line_angle_degrees": 0.0,
        "has_eye": bool(eye_faces),
        "has_iris": bool(iris_faces),
        "prior_valid": bool(eye_faces),
        "geometry_version": GEOMETRY_VERSION,
        "landmark_schema": LANDMARK_SCHEMA,
        "landmark_groups_normalized": None,
        "face_count_detected": int(len(faces)),
        "face_count_valid": int(len(valid_faces)),
        "face_count_with_eye": int(len(eye_faces)),
        "face_count_with_bilateral_iris": int(len(iris_faces)),
        "faces": faces,
        "partial_face_failures": [
            {
                "face_index": int(face.get("face_index", -1)),
                "reason": str(face.get("fail_reason", "unknown")),
            }
            for face in faces
            if not bool(face.get("success"))
        ],
    }

    if not faces:
        meta["fail_reason"] = "face_landmarker_failed"
        return merged_prior, meta
    if not valid_faces:
        reasons = sorted(
            {str(face.get("fail_reason", "unknown")) for face in faces}
        )
        meta["fail_reason"] = "no_valid_face_geometry:" + ",".join(reasons)
        return merged_prior, meta

    # Backward-compatible top-level fields use the largest valid periocular ROI as
    # the primary face.  Multi-face-aware code below consumes ``faces`` directly.
    def _roi_area(face):
        roi = face.get("roi_pixels")
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            return -1
        return max(0, int(roi[2]) - int(roi[0])) * max(0, int(roi[3]) - int(roi[1]))

    primary = max(valid_faces, key=_roi_area)
    meta["landmark_groups_normalized"] = primary.get(
        "landmark_groups_normalized"
    )
    meta["eye_line_angle_degrees"] = float(
        primary.get("eye_line_angle_degrees", 0.0)
    )

    rois = [face.get("roi_pixels") for face in valid_faces]
    rois = [roi for roi in rois if isinstance(roi, (list, tuple)) and len(roi) == 4]
    if rois:
        left = min(int(roi[0]) for roi in rois)
        top = min(int(roi[1]) for roi in rois)
        right = max(int(roi[2]) for roi in rois)
        bottom = max(int(roi[3]) for roi in rois)
        meta["roi_pixels"] = [left, top, right, bottom]
        meta["roi_normalized"] = [
            left / width, top / height, right / width, bottom / height
        ]
        meta["roi_definition"] = "bounding_union_of_valid_face_periocular_rois"

    return merged_prior, meta


class LandmarkDetector:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.backend = args.backend
        self.detector = None
        if self.backend == "auto":
            self.backend = "tasks" if args.model_asset_path and Path(args.model_asset_path).is_file() else "legacy"
        if self.backend == "tasks":
            if not args.model_asset_path or not Path(args.model_asset_path).is_file():
                raise FileNotFoundError("--model_asset_path is required for --backend tasks")
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            delegate_name = str(getattr(args, "geometry_delegate", "cpu")).lower()
            if delegate_name not in ("cpu", "gpu"):
                raise ValueError(f"Unsupported geometry delegate: {delegate_name!r}")
            delegate = (
                python.BaseOptions.Delegate.GPU
                if delegate_name == "gpu"
                else python.BaseOptions.Delegate.CPU
            )
            base_options = python.BaseOptions(
                model_asset_path=args.model_asset_path,
                delegate=delegate,
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=int(args.max_faces),
                min_face_detection_confidence=args.min_detection_confidence,
                min_face_presence_confidence=args.min_presence_confidence,
                min_tracking_confidence=args.min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        elif self.backend == "legacy":
            solutions = getattr(mp, "solutions", None)
            if solutions is None or not hasattr(solutions, "face_mesh"):
                raise RuntimeError(
                    "This MediaPipe build does not expose mp.solutions.face_mesh. "
                    "Use --backend tasks with --model_asset_path /path/to/face_landmarker.task."
                )
            self.detector = solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=int(args.max_faces),
                refine_landmarks=True,
                min_detection_confidence=args.min_detection_confidence,
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()

    def detect(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "tasks":
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(image)
            face_sets = result.face_landmarks or []
        else:
            result = self.detector.process(rgb)
            face_sets = [
                item.landmark for item in (result.multi_face_landmarks or [])
            ]
        return [
            np.asarray(
                [[point.x * width, point.y * height] for point in landmarks],
                dtype=np.float32,
            )
            for landmarks in face_sets
        ]


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


# === END INLINED PROJECT RUNTIME ===

SCRIPT_BUILD = "infer_single_multiface_v1_20260918"

# Only these MediaPipe points are consumed by the current eye/brow/iris prior.
# Preserve the original dense index space so prepare_geometry remains unchanged.
GEOMETRY_LANDMARK_INDICES = np.asarray([
    7, 33, 46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 133, 144, 145,
    153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246, 249, 263,
    276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 362, 373, 374,
    380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466, 469, 470,
    471, 472, 474, 475, 476, 477,
], dtype=np.int32)
GEOMETRY_LANDMARK_DENSE_COUNT = int(GEOMETRY_LANDMARK_INDICES.max()) + 1
LEGACY_GEOMETRY_VERSION = "landmark_detail_adapter_v1"

_WARNING_LOCK = threading.Lock()
_WARNING_COUNTS: dict[tuple[str, str], int] = {}
_WARNING_EXAMPLES: dict[tuple[str, str], list[str]] = {}


def reset_warning_summary() -> None:
    with _WARNING_LOCK:
        _WARNING_COUNTS.clear()
        _WARNING_EXAMPLES.clear()


def record_warning(args, category: str, path: Path, detail: str) -> None:
    """Aggregate repetitive per-image warnings; verbose mode prints each one."""
    if bool(getattr(args, "verbose_warnings", False)):
        print(f"[warn] {path}: {detail}", flush=True)
    key = (str(category), str(detail))
    with _WARNING_LOCK:
        _WARNING_COUNTS[key] = _WARNING_COUNTS.get(key, 0) + 1
        examples = _WARNING_EXAMPLES.setdefault(key, [])
        if len(examples) < 3:
            examples.append(str(path))


def print_warning_summary() -> None:
    with _WARNING_LOCK:
        rows = [
            (category, detail, count, list(_WARNING_EXAMPLES.get((category, detail), [])))
            for (category, detail), count in _WARNING_COUNTS.items()
        ]
    if not rows:
        return
    print(f"[WARN-SUMMARY] {sum(row[2] for row in rows)} event(s)", flush=True)
    for category, detail, count, examples in sorted(rows):
        sample_text = "; ".join(examples)
        print(
            f"  {category}: {count} | {detail}"
            + (f" | examples: {sample_text}" if sample_text else ""),
            flush=True,
        )


def detector_args(a):
    return SimpleNamespace(
        backend=a.backend,
        model_asset_path=a.model_asset_path,
        geometry_delegate=getattr(a, "geometry_delegate", "cpu"),
        min_detection_confidence=a.min_detection_confidence,
        min_presence_confidence=a.min_presence_confidence,
        min_tracking_confidence=a.min_tracking_confidence,
        max_faces=int(getattr(a, "max_faces", 8)),
        periocular_horizontal_ratio=.22,
        periocular_vertical_ratio=.35,
        iris_dilation_ratio=.35,
    )


class LazyGeometryBuilder:
    """
    Lazily import MediaPipe-dependent geometry code only when needed.
    This allows batch inference to run even without mediapipe installed,
    as long as eye_prior.png + face_meta.json already exist.
    """
    def __init__(self, args):
        self.args = args
        self._ctx = None
        self._detector = None
        self._build_prior_and_meta = None
        self._dargs = detector_args(args)
        self.last_build_timing_ms: dict[str, float] = {}

    def _ensure_detector(self):
        if self._detector is not None:
            return
        global mp
        if mp is None:
            import mediapipe as imported_mediapipe
            mp = imported_mediapipe
        self._ctx = LandmarkDetector(self._dargs)
        self._detector = self._ctx.__enter__()
        self._build_prior_and_meta = build_prior_and_meta

    @staticmethod
    def _resize_detection_rgb(
        rgb: np.ndarray,
        original_w: int,
        original_h: int,
        max_side: int,
        interpolation_name: str,
    ) -> np.ndarray:
        longest = max(original_h, original_w)
        if max_side <= 0 or longest <= max_side:
            return rgb if rgb.flags.c_contiguous else np.ascontiguousarray(rgb)
        scale = float(max_side) / float(longest)
        interpolation = {
            "area": cv2.INTER_AREA,
            "linear": cv2.INTER_LINEAR,
            "linear_exact": cv2.INTER_LINEAR_EXACT,
        }[interpolation_name]
        resized = cv2.resize(
            rgb,
            (max(1, round(original_w * scale)), max(1, round(original_h * scale))),
            interpolation=interpolation,
        )
        return resized if resized.flags.c_contiguous else np.ascontiguousarray(resized)

    def _detect_landmarks(self, detect_rgb: np.ndarray):
        wrapper = self._detector
        if wrapper.backend == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=detect_rgb)
            result = wrapper.detector.detect(mp_image)
            return list(result.face_landmarks or [])
        result = wrapper.detector.process(detect_rgb)
        return [item.landmark for item in (result.multi_face_landmarks or [])]


    def build(self, image: Image.Image):
        build_started = time.perf_counter()
        detector_init_started = time.perf_counter()
        self._ensure_detector()
        detector_init_ms = (time.perf_counter() - detector_init_started) * 1000.0

        resize_started = time.perf_counter()
        # image is already RGB uint8; omitting dtype avoids an unnecessary copy on
        # Pillow/Numpy combinations that can expose the underlying buffer directly.
        rgb = np.asarray(image)
        original_h, original_w = rgb.shape[:2]
        max_side = int(getattr(self.args, "geometry_detect_max_side", 768))
        interpolation_name = str(
            getattr(self.args, "geometry_resize_interp", "area")
        ).lower()
        if interpolation_name not in ("area", "linear", "linear_exact"):
            raise ValueError(
                f"unknown geometry resize interpolation {interpolation_name!r}"
            )
        detect_rgb = self._resize_detection_rgb(
            rgb, original_w, original_h, max_side, interpolation_name
        )
        resize_ms = (time.perf_counter() - resize_started) * 1000.0

        detection_started = time.perf_counter()
        landmarks = self._detect_landmarks(detect_rgb)
        detection_ms = (time.perf_counter() - detection_started) * 1000.0

        # Fast mode may use a smaller/linear pre-resize.  Preserve robustness by
        # retrying failed detections at a larger AREA-resampled size.  Successful
        # fast detections pay no retry cost.
        retry_resize_ms = 0.0
        retry_detection_ms = 0.0
        retry_count = 0.0
        retry_side = int(getattr(self.args, "geometry_retry_max_side", 0))
        should_retry = (
            not landmarks
            and retry_side > 0
            and (
                max_side == 0
                or retry_side > max_side
                or interpolation_name != "area"
            )
        )
        if should_retry:
            retry_count = 1.0
            retry_resize_started = time.perf_counter()
            retry_rgb = self._resize_detection_rgb(
                rgb, original_w, original_h, retry_side, "area"
            )
            retry_resize_ms = (time.perf_counter() - retry_resize_started) * 1000.0
            retry_detection_started = time.perf_counter()
            landmarks = self._detect_landmarks(retry_rgb)
            retry_detection_ms = (
                time.perf_counter() - retry_detection_started
            ) * 1000.0

        unpack_started = time.perf_counter()
        points_list = []
        for face_landmarks in (landmarks or []):
            if len(face_landmarks) > int(GEOMETRY_LANDMARK_INDICES.max()):
                points = np.zeros(
                    (GEOMETRY_LANDMARK_DENSE_COUNT, 2), dtype=np.float32
                )
                for idx in GEOMETRY_LANDMARK_INDICES:
                    point = face_landmarks[int(idx)]
                    points[int(idx), 0] = float(point.x) * float(original_w)
                    points[int(idx), 1] = float(point.y) * float(original_h)
            else:
                points = np.asarray(
                    [
                        (float(point.x) * float(original_w),
                         float(point.y) * float(original_h))
                        for point in face_landmarks
                    ],
                    dtype=np.float32,
                )
            points_list.append(points)
        unpack_ms = (time.perf_counter() - unpack_started) * 1000.0

        prior_started = time.perf_counter()
        prior_np, meta = self._build_prior_and_meta(rgb, points_list, self._dargs)
        prior = Image.fromarray(prior_np)
        prior_ms = (time.perf_counter() - prior_started) * 1000.0
        self.last_build_timing_ms = {
            "geometry_detector_init": float(detector_init_ms),
            "geometry_detection_resize": float(resize_ms),
            "geometry_landmark_detection": float(detection_ms),
            "geometry_retry_resize": float(retry_resize_ms),
            "geometry_retry_detection": float(retry_detection_ms),
            "geometry_retry_count": float(retry_count),
            "geometry_landmark_unpack": float(unpack_ms),
            "geometry_prior_meta_build": float(prior_ms),
            "geometry_detect_total": float(
                (time.perf_counter() - build_started) * 1000.0
            ),
        }
        return prior, meta

    def close(self):
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
            self._detector = None
            self._build_prior_and_meta = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def load_checkpoint(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        # fallback for older torch versions
        return torch.load(path, map_location='cpu')


def load_model(checkpoint_path: str, device: torch.device):
    ck = load_checkpoint(checkpoint_path)
    if 'args' not in ck:
        raise KeyError("checkpoint missing 'args'")

    saved = SimpleNamespace(**ck['args'])
    config.validate_common_args(saved)
    model = CleanResidualPMF(saved, enable_training_v=False).to(device)

    state = ck.get('ema') or ck.get('model')
    if state is None:
        raise KeyError("checkpoint has neither 'ema' nor 'model'")
    if 'ema' not in ck:
        state = base._inference_subset(state)

    load_inference_state(model, state)
    model.eval().requires_grad_(False)
    return model, saved


def configure_attention_backend(model: torch.nn.Module, backend: str) -> dict[str, int]:
    """Configure inference attention kernels without changing checkpoint weights."""
    mode = str(backend).lower()
    if mode not in ("legacy", "sdpa"):
        raise ValueError(f"unknown --attention_backend {backend!r}")
    rope_count = 0
    cross_count = 0
    for module in model.modules():
        if isinstance(module, RoPEAttention):
            module.use_sdpa = (mode == "sdpa")
            module.padded_backend = mode
            rope_count += 1
        elif isinstance(module, CrossAttention):
            module.backend = mode
            cross_count += 1
    return {"rope_padded": rope_count, "conditioner_cross": cross_count}


def apply_renderer_geometry_overrides(model, args) -> dict[str, float]:
    applied = {}
    for name in (
        "renderer_lens_expand_x", "renderer_lens_expand_y",
        "renderer_lens_min_height_ratio", "renderer_lens_feather",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(model, name, float(value))
            applied[name] = float(value)
    return applied


def prediction_run_signature(checkpoint_path: Path, model, args) -> dict:
    stat = checkpoint_path.stat()
    return {
        "version": 2,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
        "seed": int(args.seed),
        "nfe": int(args.nfe),
        "patch_backend": str(args.patch_backend),
        "amp": str(args.amp),
        "attention_backend": str(args.attention_backend),
        "renderer_backend": str(args.renderer_backend),
        "renderer_blur_backend": str(args.renderer_blur_backend),
        "disable_renderer_compile": bool(args.disable_renderer_compile),
        "grid_solver": str(args.grid_solver),
        "geometry_detect_max_side": int(args.geometry_detect_max_side),
        "geometry_resize_interp": str(args.geometry_resize_interp),
        "geometry_retry_max_side": int(args.geometry_retry_max_side),
        "geometry_delegate": str(args.geometry_delegate),
        "max_faces": int(args.max_faces),
        "pmf_only": bool(args.pmf_only),
        "disable_learned_renderer": bool(args.disable_learned_renderer),
        "disable_frequency_renderer": bool(args.disable_frequency_renderer),
        "renderer_lens_expand_x": float(model.renderer_lens_expand_x),
        "renderer_lens_expand_y": float(model.renderer_lens_expand_y),
        "renderer_lens_min_height_ratio": float(model.renderer_lens_min_height_ratio),
        "renderer_lens_feather": float(model.renderer_lens_feather),
        "geometry_version": GEOMETRY_VERSION,
        "landmark_schema": LANDMARK_SCHEMA,
    }


def prediction_meta_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".meta.json")


def expected_prediction_manifest(
    glare_path: Path,
    run_signature: dict,
    input_signature: Optional[dict] = None,
) -> dict:
    return {
        "run": run_signature,
        "source_signature": (
            input_signature
            if input_signature is not None
            else source_signature(glare_path)
        ),
    }


def prediction_is_current(
    glare_path: Path, output_path: Path, run_signature: dict
) -> bool:
    if not output_path.is_file():
        return False
    meta_path = prediction_meta_path(output_path)
    if not meta_path.is_file():
        return False
    try:
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    # If the checkpoint/config changed, the output is stale regardless of the
    # source hash. Avoid reading a large input file merely to reach that result.
    if saved.get("run") != run_signature:
        return False
    return saved.get("source_signature") == source_signature(glare_path)


def write_prediction_manifest(
    glare_path: Path,
    output_path: Path,
    run_signature: dict,
    input_signature: Optional[dict] = None,
) -> None:
    path = prediction_meta_path(output_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            expected_prediction_manifest(
                glare_path, run_signature, input_signature=input_signature
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def discover_inputs(
    input_path: Path, pattern: str, max_items: int = 0
) -> list[Path]:
    """Discover inputs with early stopping for the common root/sample layout."""
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    limit = int(max_items)
    files = []
    entries = sorted(
        (
            entry for entry in os.scandir(input_path)
            if entry.is_dir(follow_symlinks=False)
        ),
        key=lambda entry: entry.name,
    )
    if not any(char in pattern for char in "*?[]"):
        for entry in entries:
            candidate = Path(entry.path) / pattern
            if candidate.is_file():
                files.append(candidate)
                if limit > 0 and len(files) >= limit:
                    return files
    else:
        for entry in entries:
            for candidate in sorted(Path(entry.path).glob(pattern)):
                if candidate.is_file():
                    files.append(candidate)
                    if limit > 0 and len(files) >= limit:
                        return files
    if files:
        return files

    # General nested-layout fallback, still stopping as soon as max_items is met.
    for directory, dirnames, filenames in os.walk(input_path):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            if not fnmatch.fnmatch(filename, pattern):
                continue
            files.append(Path(directory) / filename)
            if limit > 0 and len(files) >= limit:
                return files
    if not files:
        raise FileNotFoundError(
            f"No files matched pattern '{pattern}' under directory: {input_path}"
        )
    return files


def validate_cached_geometry(
    glare_path: Path,
    image: Image.Image,
    prior_path: Path,
    meta_path: Path,
    actual_signature: Optional[dict] = None,
    allow_legacy_v1: bool = True,
) -> tuple[Optional[Image.Image], Optional[dict], Optional[str]]:
    if not prior_path.is_file() or not meta_path.is_file():
        return None, None, None

    try:
        prior = Image.open(prior_path).convert('L')
    except Exception as e:
        return None, None, f"failed to open prior: {e}"

    meta = load_metadata(meta_path)
    if not meta:
        return None, None, "face_meta.json is empty or invalid"

    if prior.size != image.size:
        return None, None, f"prior size mismatch: prior={prior.size}, image={image.size}"

    expected_size = [int(image.size[0]), int(image.size[1])]
    recorded_size = [meta.get('source_width'), meta.get('source_height')]
    if recorded_size != expected_size:
        return None, None, f"metadata source size mismatch: meta={recorded_size}, image={expected_size}"
    geometry_version = meta.get('geometry_version')
    legacy_v1_compatible = bool(
        allow_legacy_v1
        and geometry_version == LEGACY_GEOMETRY_VERSION
        and meta.get('landmark_schema') == LANDMARK_SCHEMA
        and meta.get('success')
        and meta.get('landmark_groups_normalized')
    )
    if geometry_version != GEOMETRY_VERSION and not legacy_v1_compatible:
        return None, None, (
            f"geometry version mismatch: {geometry_version!r} != "
            f"{GEOMETRY_VERSION!r}"
        )
    if meta.get('landmark_schema') != LANDMARK_SCHEMA:
        return None, None, (
            f"landmark schema mismatch: {meta.get('landmark_schema')!r} != "
            f"{LANDMARK_SCHEMA!r}"
        )
    # v1 and v2 use the same prior/ROI/landmark schema. v2 only added source
    # hashing. For v1 we already verified dimensions and semantic fields above.
    if not legacy_v1_compatible:
        if actual_signature is None:
            actual_signature = source_signature(glare_path)
        if meta.get('source_signature') != actual_signature:
            return None, None, "source image signature/hash mismatch"

    return prior, meta, None



def _persist_geometry_cache(
    prior: Image.Image,
    meta: dict,
    prior_path: Path,
    meta_path: Path,
    input_signature: Optional[dict],
) -> None:
    """Atomically persist successful geometry for zero-detection repeat runs."""
    signature = input_signature
    meta_to_write = dict(meta)
    if signature is not None:
        meta_to_write["source_signature"] = signature
    token = f"{os.getpid()}.{threading.get_ident()}"
    prior_tmp = prior_path.with_name(prior_path.name + f".{token}.tmp")
    meta_tmp = meta_path.with_name(meta_path.name + f".{token}.tmp")
    try:
        prior.save(prior_tmp, format="PNG", compress_level=1)
        meta_tmp.write_text(
            json.dumps(meta_to_write, indent=2), encoding="utf-8"
        )
        os.replace(prior_tmp, prior_path)
        os.replace(meta_tmp, meta_path)
    finally:
        for path in (prior_tmp, meta_tmp):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass


def load_or_build_geometry(
    glare_path: Path,
    image: Image.Image,
    args,
    geometry_builder: LazyGeometryBuilder,
    input_signature: Optional[dict] = None,
    return_timing: bool = False,
):
    timing_ms = {
        "geometry_cache_lookup": 0.0,
        "geometry_detector_init": 0.0,
        "geometry_detection_resize": 0.0,
        "geometry_landmark_detection": 0.0,
        "geometry_landmark_unpack": 0.0,
        "geometry_prior_meta_build": 0.0,
        "geometry_detect_total": 0.0,
        "geometry_cache_write": 0.0,
    }
    sample_dir = glare_path.parent
    prior_path = sample_dir / 'eye_prior.png'
    meta_path = sample_dir / 'face_meta.json'

    if args.use_existing_geometry:
        cache_started = time.perf_counter()
        prior, meta, reason = validate_cached_geometry(
            glare_path,
            image,
            prior_path,
            meta_path,
            actual_signature=input_signature,
            allow_legacy_v1=(not bool(getattr(args, "strict_geometry_cache", False)) and int(getattr(args, "max_faces", 1)) == 1),
        )
        timing_ms["geometry_cache_lookup"] = (
            time.perf_counter() - cache_started
        ) * 1000.0
        if prior is not None and meta is not None:
            result = (prior, meta, "cached")
            return (*result, timing_ms) if return_timing else result
        if reason is not None:
            record_warning(
                args,
                "cache_unavailable",
                glare_path,
                f"cached geometry unavailable -> {reason}",
            )

    if args.no_detection_fallback:
        raise RuntimeError(
            f"{glare_path}: missing/invalid eye_prior.png or face_meta.json, "
            f"and --no_detection_fallback is enabled"
        )

    prior, meta = geometry_builder.build(image)
    timing_ms.update(geometry_builder.last_build_timing_ms)
    if bool(getattr(args, "write_geometry_cache", False)) and bool(meta.get("success")):
        cache_write_started = time.perf_counter()
        _persist_geometry_cache(
            prior, meta, prior_path, meta_path, input_signature=input_signature
        )
        timing_ms["geometry_cache_write"] = (
            time.perf_counter() - cache_write_started
        ) * 1000.0
        # Keep the in-memory metadata identical to what strict cache validation sees.
        if input_signature is not None:
            meta = dict(meta)
            meta["source_signature"] = input_signature
    result = (prior, meta, "detected")
    return (*result, timing_ms) if return_timing else result


@dataclass(frozen=True)
class InputRecord:
    glare_path: Path
    output_path: Path


@dataclass
class PreparedSample:
    record: InputRecord
    geometry_source: str
    original_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    side: int
    rgb_square_u8: np.ndarray
    prior_square_u8: np.ndarray
    leaves: np.ndarray
    leaves_cpu: list[list[int]]
    region_probs: np.ndarray
    lens_logical: np.ndarray
    lens_interpolation: str
    canonical_patches: Optional[np.ndarray]
    patch_boxes: np.ndarray
    # CPU preprocessing plans. These deliberately stay host-side until batching so
    # the CUDA hot path never has to discover patch groups or pull query counts
    # back from the GPU.
    patch_groups: tuple
    query_group_positions: tuple[np.ndarray, ...]
    renderer_groups: tuple
    input_signature: Optional[dict] = None
    cpu_prepare_ms: float = 0.0
    prepare_stage_ms: dict[str, float] = field(default_factory=dict)


class ThreadLocalGeometryBuilders:
    """Give each preprocessing thread its own MediaPipe detector instance."""

    def __init__(self, args):
        self.args = args
        self.local = threading.local()
        self.lock = threading.Lock()
        self.builders: list[LazyGeometryBuilder] = []

    def get(self) -> LazyGeometryBuilder:
        builder = getattr(self.local, "builder", None)
        if builder is None:
            builder = LazyGeometryBuilder(self.args)
            self.local.builder = builder
            with self.lock:
                self.builders.append(builder)
        return builder

    def close_all(self) -> None:
        with self.lock:
            builders, self.builders = self.builders, []
        for builder in builders:
            builder.close()


def _square_edge_rgb_u8(image: Image.Image):
    width, height = image.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    right = side - width - left
    bottom = side - height - top
    square = np.pad(
        np.asarray(image, dtype=np.uint8),
        ((top, bottom), (left, right), (0, 0)),
        mode="edge",
    )
    return (
        np.ascontiguousarray(square),
        (left, top, left + width, top + height),
    )


def _square_zero_u8(prior: Image.Image):
    width, height = prior.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    right = side - width - left
    bottom = side - height - top
    return np.ascontiguousarray(np.pad(
        np.asarray(prior, dtype=np.uint8),
        ((top, bottom), (left, right)),
        mode="constant",
        constant_values=0,
    ))


def _region_probs_from_logical_u8(
    logical_prior_u8: np.ndarray, leaves_rows: np.ndarray
) -> np.ndarray:
    """Vectorized equivalent of core.region_probs_from_prior()."""
    arr = np.asarray(logical_prior_u8, dtype=np.uint8)
    maps = (
        arr < 64,
        (arr >= 64) & (arr < 128),
        (arr >= 128) & (arr < 192),
        arr >= 192,
    )
    integrals = [cv2.integral(x.astype(np.float32), sdepth=cv2.CV_64F) for x in maps]
    rows = np.asarray(leaves_rows, dtype=np.int64)
    x, y, size = rows[:, 0], rows[:, 1], rows[:, 2]
    area = np.square(size.astype(np.float64))
    result = np.empty((rows.shape[0], 4), dtype=np.float32)
    for channel, integral in enumerate(integrals):
        values = (
            integral[y + size, x + size] - integral[y, x + size]
            - integral[y + size, x] + integral[y, x]
        )
        result[:, channel] = values / area
    totals = result.sum(axis=1)
    valid = totals > 0
    result[valid] /= totals[valid, None]
    result[~valid] = np.asarray([1, 0, 0, 0], dtype=np.float32)
    return result


def _precompute_patch_boxes(
    leaves: np.ndarray, side: int, canvas_size: int
) -> np.ndarray:
    rows = np.asarray(leaves, dtype=np.int64)
    boxes = np.empty((rows.shape[0], 4), dtype=np.int64)
    for index, (x, y, size) in enumerate(rows):
        x1 = max(0, min(side - 1, math.floor(int(x) * side / canvas_size)))
        y1 = max(0, min(side - 1, math.floor(int(y) * side / canvas_size)))
        x2 = max(x1 + 1, min(side, math.ceil((int(x) + int(size)) * side / canvas_size)))
        y2 = max(y1 + 1, min(side, math.ceil((int(y) + int(size)) * side / canvas_size)))
        boxes[index] = (x1, y1, x2, y2)
    return boxes


def _precompute_patch_groups(patch_boxes: np.ndarray) -> tuple:
    """Group native crop boxes by shape in CPU preprocessing.

    The previous CUDA path rediscovered these groups for every batch by walking
    every token.  Store immutable Python rows here instead; the GPU path only
    merges already-grouped rows across the samples in a batch.
    """
    groups: dict[tuple[int, int], list[tuple[int, int, int, int, int]]] = {}
    for token_index, row in enumerate(np.asarray(patch_boxes, dtype=np.int64)):
        x1, y1, x2, y2 = map(int, row)
        groups.setdefault((y2 - y1, x2 - x1), []).append(
            (int(token_index), x1, y1, x2, y2)
        )
    return tuple(
        (shape, tuple(rows)) for shape, rows in sorted(groups.items())
    )


def _precompute_renderer_groups(
    leaves: np.ndarray, side: int, canvas_size: int, halo_ratio: float
) -> tuple:
    """Precompute PoU renderer geometry groups during CPU preparation.

    Each row stores local token index plus the native crop geometry.  Batch-only
    offsets are deliberately omitted and added cheaply when a batch is rendered.
    This removes the per-batch B*T floor/ceil/grouping loop from the CUDA hot path.
    """
    rows = np.asarray(leaves, dtype=np.int64)
    side_i = int(side)
    canvas_i = int(canvas_size)
    halo = float(halo_ratio)
    grouped: dict[tuple[int, int], list[tuple[int, int, int, int, int, int, int]]] = {}
    for local_index, (x0, y0, size0) in enumerate(rows):
        x, y, size = int(x0), int(y0), int(size0)
        lx1 = x - halo * size
        ly1 = y - halo * size
        lx2 = x + size + halo * size
        ly2 = y + size + halo * size
        x1 = max(0, int(math.floor(lx1 * side_i / canvas_i)))
        y1 = max(0, int(math.floor(ly1 * side_i / canvas_i)))
        x2 = min(side_i, int(math.ceil(lx2 * side_i / canvas_i)))
        y2 = min(side_i, int(math.ceil(ly2 * side_i / canvas_i)))
        if x2 <= x1 or y2 <= y1:
            continue
        shape = (y2 - y1, x2 - x1)
        grouped.setdefault(shape, []).append(
            (int(local_index), x, y, size, x1, y1, side_i)
        )
    return tuple(
        (shape, np.ascontiguousarray(group_rows, dtype=np.int64))
        for shape, group_rows in sorted(grouped.items())
    )


def _precompute_query_group_positions(
    leaves: np.ndarray, prefix_tokens: int, base_patch: int
) -> tuple[np.ndarray, ...]:
    """Precompute exact adaptive attention query positions on the CPU.

    Sequence positions are prefix tokens followed by adaptive image tokens.  The
    five groups exactly match pmfDiT.build_adaptive_sequence(): prefix plus the
    p4/p8/p16/p32 image scales (relative to the logical base patch).
    """
    rows = np.asarray(leaves, dtype=np.int64)
    prefix = int(prefix_tokens)
    base = int(base_patch)
    image_positions = prefix + np.arange(rows.shape[0], dtype=np.int64)
    sizes = rows[:, 2]
    levels = (base // 4, base // 2, base, base * 2)
    groups = [np.arange(prefix, dtype=np.int64)]
    groups.extend(
        np.ascontiguousarray(image_positions[sizes == level], dtype=np.int64)
        for level in levels
    )
    return tuple(groups)


def _needs_input_signature(args) -> bool:
    """Hash source bytes only when a downstream cache/manifest actually needs it."""
    return bool(
        getattr(args, "use_existing_geometry", False)
        or getattr(args, "write_geometry_cache", False)
        or not bool(getattr(args, "no_save", False))
    )


def _load_rgb_and_signature(
    path: Path, *, need_signature: bool
) -> tuple[Image.Image, Optional[dict]]:
    """Decode once; skip SHA256 entirely for force-redetect/no-save benchmarks."""
    if not need_signature:
        with Image.open(path) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB").copy()
        return image, None
    encoded = Path(path).read_bytes()
    signature = {
        "algorithm": "sha256",
        "digest": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
    }
    with Image.open(BytesIO(encoded)) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB").copy()
    return image, signature


def _grid_worker_ping() -> int:
    return os.getpid()


def _grid_worker_job(
    logical_prior_u8: np.ndarray,
    meta: Mapping[str, object],
    original_size: tuple[int, int],
    cfg_payload: dict,
    solver: str,
):
    """Build the exact adaptive grid in a spawned CPU process."""
    cfg = AdaptiveGridConfig(**cfg_payload)
    prior = Image.fromarray(np.asarray(logical_prior_u8, dtype=np.uint8))
    grid = build_adaptive_grid(prior, meta, tuple(original_size), cfg, solver=str(solver))
    leaves_obj = list(grid.leaves)
    leaves = np.asarray(
        [[int(v.x), int(v.y), int(v.size)] for v in leaves_obj], dtype=np.int64
    )
    return leaves


def prepare_one(
    record: InputRecord,
    args,
    saved,
    cfg,
    model,
    geometry_builders: ThreadLocalGeometryBuilders,
    grid_executor=None,
) -> PreparedSample:
    prepare_started = time.perf_counter()
    prepare_stage_ms: dict[str, float] = {}
    # The former path opened the file for PIL and then read it again for SHA256.
    # Decode and hash the same compressed byte buffer to halve source-file I/O.
    stage_started = time.perf_counter()
    image, input_signature = _load_rgb_and_signature(
        record.glare_path, need_signature=_needs_input_signature(args)
    )
    prepare_stage_ms["input_read_decode_hash"] = (
        time.perf_counter() - stage_started
    ) * 1000.0

    stage_started = time.perf_counter()
    geometry_builder = geometry_builders.get()
    prior, meta, geometry_source, geometry_detail_ms = load_or_build_geometry(
        record.glare_path,
        image,
        args,
        geometry_builder,
        input_signature=input_signature,
        return_timing=True,
    )
    prepare_stage_ms["geometry_total"] = (
        time.perf_counter() - stage_started
    ) * 1000.0
    prepare_stage_ms.update(geometry_detail_ms)
    if not bool(meta.get("success")):
        record_warning(
            args,
            "geometry_fallback",
            record.glare_path,
            f"eye detection/meta not fully successful: "
            f"{meta.get('fail_reason', 'unknown')}; using fallback geometry",
        )
    elif int(meta.get("face_count_valid", 1)) < int(meta.get("face_count_detected", 1)):
        record_warning(
            args,
            "geometry_partial_face",
            record.glare_path,
            f"valid geometry for {meta.get('face_count_valid', 0)}/"
            f"{meta.get('face_count_detected', 0)} detected faces; "
            "using all available face cues",
        )

    # Compute the tiny logical prior first.  When grid_workers>0, launch the exact
    # adaptive-grid solver in a separate process and overlap it with independent
    # square-padding and renderer-lens preparation in this thread.
    logical_prior = np.asarray(
        logical_canvas(prior, cfg.canvas_size, is_mask=True), dtype=np.uint8
    )
    grid_future = None
    grid_started = time.perf_counter()
    if model.use_adaptive_grid and grid_executor is not None:
        grid_future = grid_executor.submit(
            _grid_worker_job,
            np.ascontiguousarray(logical_prior),
            dict(meta),
            tuple(image.size),
            asdict(cfg),
            args.grid_solver,
        )

    stage_started = time.perf_counter()
    rgb_square, crop_box = _square_edge_rgb_u8(image)
    prior_square = _square_zero_u8(prior)
    side = int(rgb_square.shape[0])
    prepare_stage_ms["square_padding"] = (
        time.perf_counter() - stage_started
    ) * 1000.0

    stage_started = time.perf_counter()
    logical_lens = build_renderer_lens_alpha(
        meta,
        tuple(image.size),
        canvas_size=cfg.canvas_size,
        expand_x=float(model.renderer_lens_expand_x),
        expand_y=float(model.renderer_lens_expand_y),
        min_height_ratio=float(model.renderer_lens_min_height_ratio),
        feather=float(model.renderer_lens_feather),
        superellipse_power=float(model.renderer_lens_power),
    )
    if logical_lens is None:
        maps = build_geometry_maps(prior, meta, image.size, cfg)
        lens_logical = maps["lens_band"].astype(np.float32, copy=False)[None]
        lens_interpolation = "nearest"
    else:
        lens_logical = logical_lens.astype(np.float32, copy=False)[None]
        lens_interpolation = "bilinear"
    prepare_stage_ms["renderer_lens_mask"] = (
        time.perf_counter() - stage_started
    ) * 1000.0

    if model.use_adaptive_grid:
        if grid_future is not None:
            leaves = np.ascontiguousarray(grid_future.result(), dtype=np.int64)
            leaves_obj = [AdaptiveLeaf(int(x), int(y), int(size)) for x, y, size in leaves]
        else:
            grid = build_adaptive_grid(prior, meta, image.size, cfg, solver=args.grid_solver)
            leaves_obj = list(grid.leaves)
            leaves = np.asarray(
                [[int(v.x), int(v.y), int(v.size)] for v in leaves_obj], dtype=np.int64
            )
    else:
        grid = build_uniform_grid(cfg)
        leaves_obj = list(grid.leaves)
        leaves = np.asarray(
            [[int(v.x), int(v.y), int(v.size)] for v in leaves_obj], dtype=np.int64
        )
    prepare_stage_ms["grid_build"] = (
        time.perf_counter() - grid_started
    ) * 1000.0

    stage_started = time.perf_counter()
    region = _region_probs_from_logical_u8(logical_prior, leaves)
    prepare_stage_ms["region_probabilities"] = (
        time.perf_counter() - stage_started
    ) * 1000.0

    stage_started = time.perf_counter()
    canonical_patches = None
    if args.patch_backend == "pil":
        canonical_patches = (
            extract_canonical_patches(
                image, leaves_obj, cfg.canvas_size, int(saved.canonical_size)
            ).numpy().astype(np.float32, copy=False)
        )
    patch_boxes = _precompute_patch_boxes(leaves, side, cfg.canvas_size)
    patch_groups = _precompute_patch_groups(patch_boxes)
    query_group_positions = _precompute_query_group_positions(
        leaves, model.backbone.prefix_tokens, model.backbone.patch_size
    )
    renderer_groups = _precompute_renderer_groups(
        leaves, side, cfg.canvas_size, model.halo_ratio
    )
    prepare_stage_ms["patch_extract_or_plan"] = (
        time.perf_counter() - stage_started
    ) * 1000.0
    cpu_prepare_ms = (time.perf_counter() - prepare_started) * 1000.0
    prepare_stage_ms["prepare_total"] = float(cpu_prepare_ms)

    return PreparedSample(
        record=record,
        geometry_source=geometry_source,
        original_size=tuple(map(int, image.size)),
        crop_box=tuple(map(int, crop_box)),
        side=side,
        rgb_square_u8=rgb_square,
        prior_square_u8=prior_square,
        leaves=leaves,
        leaves_cpu=[[int(x), int(y), int(size)] for x, y, size in leaves],
        region_probs=np.ascontiguousarray(region),
        lens_logical=np.ascontiguousarray(lens_logical),
        lens_interpolation=lens_interpolation,
        canonical_patches=(
            None if canonical_patches is None else np.ascontiguousarray(canonical_patches)
        ),
        patch_boxes=patch_boxes,
        patch_groups=patch_groups,
        query_group_positions=query_group_positions,
        renderer_groups=renderer_groups,
        input_signature=input_signature,
        cpu_prepare_ms=cpu_prepare_ms,
        prepare_stage_ms=prepare_stage_ms,
    )


def _pin_empty(shape, dtype, pin: bool):
    return torch.empty(shape, dtype=dtype, pin_memory=bool(pin))


_PACK_CACHE_LOCAL = threading.local()


def pack_batch_cpu(samples: Sequence[PreparedSample], pin_memory: bool):
    batch = len(samples)
    max_side = max(sample.side for sample in samples)
    tokens = int(samples[0].leaves.shape[0])
    canvas = int(samples[0].lens_logical.shape[-1])
    patch_size = (
        int(samples[0].canonical_patches.shape[-1])
        if samples[0].canonical_patches is not None else 0
    )
    cache_key = (
        bool(pin_memory), batch, max_side, tokens, canvas, patch_size
    )
    cached = getattr(_PACK_CACHE_LOCAL, "value", None)
    if cached is not None and cached[0] == cache_key:
        rgb, prior, leaves, region, lens, patches = cached[1]
    else:
        rgb = _pin_empty((batch, 3, max_side, max_side), torch.uint8, pin_memory)
        prior = _pin_empty((batch, 1, max_side, max_side), torch.uint8, pin_memory)
        leaves = _pin_empty((batch, tokens, 3), torch.long, pin_memory)
        region = _pin_empty((batch, tokens, 4), torch.float32, pin_memory)
        lens = _pin_empty((batch, 1, canvas, canvas), torch.float32, pin_memory)
        patches = (
            _pin_empty(
                (batch, tokens, 3, patch_size, patch_size),
                torch.float32,
                pin_memory,
            )
            if patch_size > 0 else None
        )
        _PACK_CACHE_LOCAL.value = (
            cache_key, (rgb, prior, leaves, region, lens, patches)
        )
    # Equal-side buckets overwrite the full native tensor. Mixed final batches
    # need zero padding outside each smaller sample's square.
    if any(sample.side != max_side for sample in samples):
        rgb.zero_()
        prior.zero_()
    for index, sample in enumerate(samples):
        if sample.leaves.shape[0] != tokens:
            raise ValueError("all samples in a batch must have the same token count")
        side = sample.side
        rgb[index, :, :side, :side].copy_(
            torch.from_numpy(sample.rgb_square_u8).permute(2, 0, 1)
        )
        prior[index, :, :side, :side].copy_(
            torch.from_numpy(sample.prior_square_u8)[None]
        )
        leaves[index].copy_(torch.from_numpy(sample.leaves))
        region[index].copy_(torch.from_numpy(sample.region_probs))
        lens[index].copy_(torch.from_numpy(sample.lens_logical))
        if patches is not None:
            if sample.canonical_patches is None:
                raise RuntimeError("mixed patch backends inside one batch")
            patches[index].copy_(torch.from_numpy(sample.canonical_patches))
    return rgb, prior, leaves, region, lens, patches


def pack_query_groups_cpu(
    samples: Sequence[PreparedSample], pin_memory: bool
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Pad CPU-precomputed query positions without touching CUDA state."""
    if not samples:
        return tuple()
    group_count = len(samples[0].query_group_positions)
    if any(len(sample.query_group_positions) != group_count for sample in samples):
        raise ValueError("inconsistent adaptive query-group plans inside batch")
    batch = len(samples)
    packed = []
    for group_id in range(group_count):
        max_count = max(
            int(sample.query_group_positions[group_id].shape[0])
            for sample in samples
        )
        indices = _pin_empty((batch, max_count), torch.long, pin_memory)
        valid = _pin_empty((batch, max_count), torch.bool, pin_memory)
        if max_count:
            # Legacy torch.sort(...).clamp_max(seq_len - 1) pads missing query
            # slots with the final sequence position. Match it exactly even though
            # query_valid later zeros those rows before scatter.
            seq_len = (
                int(samples[0].query_group_positions[0].shape[0])
                + int(samples[0].leaves.shape[0])
            )
            indices.fill_(seq_len - 1)
            valid.zero_()
            for batch_index, sample in enumerate(samples):
                positions = sample.query_group_positions[group_id]
                count = int(positions.shape[0])
                if count:
                    indices[batch_index, :count].copy_(torch.from_numpy(positions))
                    valid[batch_index, :count] = True
        packed.append((indices, valid))
    return tuple(packed)


def pack_patch_plan_cpu(
    samples: Sequence[PreparedSample], pin_memory: bool
) -> tuple[tuple[tuple, torch.Tensor], ...]:
    """Merge pre-grouped crop rows and stage output indices in host memory."""
    if not samples:
        return tuple()
    tokens = int(samples[0].leaves.shape[0])
    merged: dict[tuple[int, int], list[tuple[int, tuple]]] = {}
    for batch_index, sample in enumerate(samples):
        for shape, rows in sample.patch_groups:
            merged.setdefault(tuple(shape), []).append((batch_index, rows))
    plan = []
    for shape in sorted(merged):
        crop_rows = []
        flat_indices = []
        for batch_index, rows in merged[shape]:
            for token_index, x1, y1, x2, y2 in rows:
                crop_rows.append((batch_index, x1, y1, x2, y2))
                flat_indices.append(batch_index * tokens + token_index)
        indices = _pin_empty((len(flat_indices),), torch.long, pin_memory)
        if flat_indices:
            indices.copy_(torch.from_numpy(np.asarray(flat_indices, dtype=np.int64)))
        plan.append((tuple(crop_rows), indices))
    return tuple(plan)


def pack_renderer_plan_cpu(
    samples: Sequence[PreparedSample], pin_memory: bool
) -> tuple[tuple[tuple[int, int, int, int], ...], torch.Tensor, torch.Tensor]:
    """Pack all PoU metadata into two pinned tensors and one CPU group table.

    Previous optimized revisions uploaded one small metadata tensor per
    (height,width) renderer group. A typical batch contains ~30 groups, so this
    produced many tiny H2D copies and repeated token gathers. Here every row is
    concatenated in renderer-group order and uploaded once per dtype.

    integer rows: [flat_token_index, x1, y1, side, pixel_offset]
    float rows:   [x, y, size, x1, y1, side]
    group table:  (height, width, start_row, end_row)
    """
    if not samples:
        empty_i = _pin_empty((0, 5), torch.long, pin_memory)
        empty_f = _pin_empty((0, 6), torch.float32, pin_memory)
        return tuple(), empty_i, empty_f

    tokens = int(samples[0].leaves.shape[0])
    pixel_offsets: list[int] = []
    cursor = 0
    for sample in samples:
        pixel_offsets.append(cursor)
        cursor += int(sample.side) * int(sample.side)

    merged: dict[tuple[int, int], list[np.ndarray]] = {}
    for batch_index, sample in enumerate(samples):
        token_base = batch_index * tokens
        pixel_offset = int(pixel_offsets[batch_index])
        for shape, local_rows in sample.renderer_groups:
            rows = np.asarray(local_rows, dtype=np.int64)
            if rows.size == 0:
                continue
            # Local renderer rows are [token_index, x, y, size, x1, y1, side].
            meta = np.empty((rows.shape[0], 8), dtype=np.int64)
            meta[:, 0] = rows[:, 0] + token_base
            meta[:, 1:7] = rows[:, 1:7]
            meta[:, 7] = pixel_offset
            merged.setdefault(tuple(shape), []).append(meta)

    group_specs: list[tuple[int, int, int, int]] = []
    int_parts: list[np.ndarray] = []
    float_parts: list[np.ndarray] = []
    row_cursor = 0
    for shape in sorted(merged):
        parts = merged[shape]
        host_meta = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        count = int(host_meta.shape[0])
        height, width = map(int, shape)
        group_specs.append((height, width, row_cursor, row_cursor + count))

        meta_i = np.empty((count, 5), dtype=np.int64)
        meta_i[:, 0] = host_meta[:, 0]  # global token index
        meta_i[:, 1] = host_meta[:, 4]  # x1
        meta_i[:, 2] = host_meta[:, 5]  # y1
        meta_i[:, 3] = host_meta[:, 6]  # side
        meta_i[:, 4] = host_meta[:, 7]  # pixel offset
        int_parts.append(meta_i)

        # Integer geometry values are exactly representable as FP32 at the
        # native sizes used here, so staging these columns as float removes six
        # per-group int->float CUDA conversions without changing their values.
        meta_f = np.ascontiguousarray(host_meta[:, 1:7], dtype=np.float32)
        float_parts.append(meta_f)
        row_cursor += count

    if int_parts:
        host_i = np.ascontiguousarray(np.concatenate(int_parts, axis=0))
        host_f = np.ascontiguousarray(np.concatenate(float_parts, axis=0))
    else:
        host_i = np.empty((0, 5), dtype=np.int64)
        host_f = np.empty((0, 6), dtype=np.float32)

    packed_i = _pin_empty(host_i.shape, torch.long, pin_memory)
    packed_f = _pin_empty(host_f.shape, torch.float32, pin_memory)
    if host_i.size:
        packed_i.copy_(torch.from_numpy(host_i))
        packed_f.copy_(torch.from_numpy(host_f))
    return tuple(group_specs), packed_i, packed_f

def gpu_extract_canonical_patches(
    native_rgb: torch.Tensor,
    patch_plan: Sequence[tuple[tuple, torch.Tensor]],
    token_count: int,
    canonical_size: int,
) -> torch.Tensor:
    """Resize CPU-preplanned native crops; no grouping/index H2D occurs here."""
    batch = int(native_rgb.shape[0])
    tokens = int(token_count)
    output = torch.empty(
        (batch * tokens, 3, canonical_size, canonical_size),
        device=native_rgb.device,
        dtype=native_rgb.dtype,
    )
    for crop_rows, indices in patch_plan:
        if not crop_rows:
            continue
        crops = torch.stack([
            native_rgb[batch_index, :, y1:y2, x1:x2]
            for batch_index, x1, y1, x2, y2 in crop_rows
        ])
        resized = F.interpolate(
            crops,
            size=(canonical_size, canonical_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        output.index_copy_(0, indices, resized)
    return output.reshape(batch, tokens, 3, canonical_size, canonical_size)


def _same_seed_noise(model, glare: torch.Tensor, seed: int) -> torch.Tensor:
    key = (str(glare.device), str(glare.dtype), tuple(glare.shape[1:]), int(seed))
    cache = getattr(model, "_inference_noise_cache", None)
    if cache is None:
        cache = {}; model._inference_noise_cache = cache
    noise = cache.get(key)
    if noise is None:
        generator = torch.Generator(device=glare.device)
        generator.manual_seed(int(seed))
        noise = torch.randn(
            (1, *glare.shape[1:]), device=glare.device, dtype=glare.dtype,
            generator=generator,
        )
        cache[key] = noise
    return noise


def one_step_x0_batch_same_seed(
    model, glare, leaves, region_probs, seed: int, query_groups=None
):
    """Batch PMF while preserving the old per-image fixed-seed noise exactly."""
    batch, tokens = glare.shape[:2]
    noise = _same_seed_noise(model, glare, seed)
    if model.use_geometry_guided_source:
        support = model._source_support(leaves, region_probs)
    else:
        support = torch.ones_like(region_probs[..., 0])
    source = (
        glare + support[..., None, None, None].to(glare.dtype)
        * noise * model.noise_scale
    ).clamp(-1, 1)
    conditioned = model._condition_tokens(source, glare, leaves, region_probs)
    flat_source = source.reshape(batch * tokens, *source.shape[2:])
    flat_leaves = leaves.reshape(batch * tokens, 3)
    flat_region = region_probs.reshape(batch * tokens, region_probs.shape[-1])
    flat_conditioned = conditioned.reshape(batch * tokens, model.backbone.hidden_size)
    offsets = model._offsets(batch, tokens, glare.device)
    labels = torch.zeros(batch, device=glare.device, dtype=torch.long)
    ones = torch.ones(batch, device=glare.device, dtype=glare.dtype)
    x0, _ = model.backbone.forward_adaptive(
        flat_source, offsets, flat_leaves, ones, ones, labels,
        model.canvas_size, need_v=False, padded_image_tokens=tokens,
        region_probs=model._backbone_region_probs(flat_region),
        image_tokens_override=flat_conditioned,
        query_groups_override=query_groups,
    )
    return x0.reshape(batch, tokens, *glare.shape[2:])


def multistep_x0_batch_same_seed(
    model, glare, leaves, region_probs, seed: int, nfe: int,
    query_groups=None,
):
    """Equal MeanFlow intervals from t=1 to r=0 with one network call each."""
    nfe = int(nfe)
    if nfe == 1:
        return one_step_x0_batch_same_seed(
            model, glare, leaves, region_probs, int(seed),
            query_groups=query_groups,
        )
    if nfe < 1:
        raise ValueError("nfe must be >= 1")

    batch = int(glare.shape[0])
    noise = _same_seed_noise(model, glare, int(seed))
    support = (
        model._source_support(leaves, region_probs)
        if model.use_geometry_guided_source
        else torch.ones_like(region_probs[..., 0])
    )
    state = (
        glare
        + support[..., None, None, None].to(glare.dtype)
        * noise
        * model.noise_scale
    ).clamp(-1, 1)
    for index in range(nfe):
        t_value = 1.0 - index / nfe
        r_value = 1.0 - (index + 1) / nfe
        t = torch.full(
            (batch, 1, 1, 1, 1), t_value,
            device=glare.device, dtype=glare.dtype,
        )
        r = torch.full_like(t, r_value)
        velocity, _ = model._network_batch(
            state, t, r, leaves, region_probs, need_v=False,
            glare_patches=glare,
        )
        state = state - (t_value - r_value) * velocity
    return state


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


class DeviceStageTimer:
    """Record consecutive device stages without adding intermediate syncs."""

    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.marks: list[tuple[str, object]] = []

    def mark(self, name: str) -> None:
        if self.is_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            value = event
        else:
            value = time.perf_counter()
        self.marks.append((str(name), value))

    def consecutive_elapsed_ms(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for (start_name, start), (end_name, end) in zip(
            self.marks, self.marks[1:]
        ):
            if self.is_cuda:
                elapsed = float(start.elapsed_time(end))
            else:
                elapsed = float((end - start) * 1000.0)
            result[f"{start_name}_to_{end_name}"] = elapsed
        return result


@dataclass
class InferenceBatchResult:
    images: list[Image.Image]
    observations: list[dict]
    oom_retries: int = 0


def resize_lenses(lens_logical: torch.Tensor, samples: Sequence[PreparedSample]):
    # After side bucketing, most batches share one native side and interpolation
    # mode. One batched interpolation launch is equivalent and much cheaper.
    first = samples[0]
    if all(
        sample.side == first.side
        and sample.lens_interpolation == first.lens_interpolation
        for sample in samples
    ):
        mode = first.lens_interpolation
        kwargs = {"align_corners": False} if mode == "bilinear" else {}
        resized = F.interpolate(
            lens_logical,
            size=(first.side, first.side),
            mode=mode,
            **kwargs,
        ).float().clamp(0, 1)
        # Keep the same-side result batched; callers can index it like the old
        # list while avoiding an immediate stack/copy in the renderer.
        return resized

    result = []
    for index, sample in enumerate(samples):
        mode = sample.lens_interpolation
        kwargs = {"align_corners": False} if mode == "bilinear" else {}
        result.append(F.interpolate(
            lens_logical[index:index + 1], size=(sample.side, sample.side),
            mode=mode, **kwargs,
        )[0].float().clamp(0, 1))
    return result


def _blur_native_scalar_batch(field: torch.Tensor, kernel_size: int) -> torch.Tensor:
    k = int(kernel_size)
    if k <= 1:
        return field.float()
    if k % 2 == 0:
        raise ValueError("native blur kernel must be odd")
    if field.ndim != 4 or field.shape[1] != 1:
        raise ValueError(f"expected scalar batch [B,1,H,W], got {tuple(field.shape)}")
    return F.avg_pool2d(field.float(), kernel_size=k, stride=1, padding=k // 2)


def _blur_native_rgb_batch(field: torch.Tensor, kernel_size: int) -> torch.Tensor:
    k = int(kernel_size)
    if k <= 1:
        return field.float()
    if k % 2 == 0:
        raise ValueError("native RGB blur kernel must be odd")
    if field.ndim != 4 or field.shape[1] != 3:
        raise ValueError(f"expected RGB batch [B,3,H,W], got {tuple(field.shape)}")
    return F.avg_pool2d(field.float(), kernel_size=k, stride=1, padding=k // 2)


def decode_from_rendered_batch_legacy(
    model, clean_native, delta_native, native_glare, lens_native, eye_prior_native,
):
    """Exact renderer policy for a same-native-side batch."""
    if clean_native.ndim != 4 or clean_native.shape[1] != 3:
        raise ValueError("clean_native must be [B,3,H,W]")
    residual_candidate = native_glare.float() + delta_native.float()
    lens_native = lens_native.float().clamp(0, 1)
    signed_correction = _blur_native_rgb_batch(
        delta_native.float(), int(model.reconstruction_blur_kernel)
    )
    correction_magnitude = signed_correction.abs().mean(dim=1, keepdim=True)
    reconstruction_need = base._smoothstep01(
        (correction_magnitude - float(model.reconstruction_low))
        / (float(model.reconstruction_high) - float(model.reconstruction_low))
    )
    eye_mask = torch.zeros_like(lens_native)
    iris_mask = torch.zeros_like(lens_native)
    if eye_prior_native is not None:
        prior = eye_prior_native
        eye_mask = ((prior >= 128) & (prior < 192)).float()
        iris_mask = (prior >= 192).float()
        eye_mask = base._smoothstep01(_blur_native_scalar_batch(
            eye_mask, int(model.eye_core_feather_kernel)
        ).clamp(0, 1))
        iris_mask = base._smoothstep01(_blur_native_scalar_batch(
            iris_mask, int(model.eye_core_feather_kernel)
        ).clamp(0, 1))
    beta_inside = torch.full_like(lens_native, float(model.native_hf_skin))
    beta_inside = beta_inside * (1 - eye_mask) + float(model.native_hf_eye) * eye_mask
    beta_inside = beta_inside * (1 - iris_mask) + float(model.native_hf_iris) * iris_mask
    beta_inside = beta_inside * (
        1 - float(model.correction_hf_suppression) * reconstruction_need
    )
    rgb01 = ((native_glare.float() + 1) * .5).clamp(0, 1)
    luminance = rgb01.mean(dim=1, keepdim=True)
    chroma = rgb01.amax(dim=1, keepdim=True) - rgb01.amin(dim=1, keepdim=True)
    bright = base._smoothstep01(
        (luminance - float(model.highlight_low))
        / (float(model.highlight_high) - float(model.highlight_low))
    )
    white = base._smoothstep01(
        (float(model.highlight_chroma_max) - chroma)
        / max(float(model.highlight_chroma_max), 1e-6)
    )
    highlight = (bright * white * lens_native).clamp(0, 1)
    low_inside = torch.full_like(lens_native, float(model.low_clean_skin))
    low_inside = torch.maximum(low_inside, float(model.low_clean_eye) * eye_mask)
    low_inside = torch.maximum(low_inside, float(model.low_clean_iris) * iris_mask)
    low_inside = torch.maximum(low_inside, float(model.low_clean_highlight) * highlight)
    low_clean_alpha = (lens_native * low_inside).clamp(0, 1)
    beta_inside = (beta_inside * (1 - highlight)).clamp(0, 1)
    native_hf_weight = ((1 - lens_native) + lens_native * beta_inside).clamp(0, 1)
    if model.use_learned_frequency_renderer:
        low_clean_alpha, native_hf_weight = model.learned_frequency_gate.forward_batch(
            native_glare.float(), clean_native.float(), delta_native.float(),
            correction_magnitude, eye_mask, iris_mask, lens_native,
            low_clean_alpha, native_hf_weight,
            correction_normalizer=float(model.reconstruction_high),
        )
    kernel = int(model.texture_lowpass_kernel)
    clean_low = _blur_native_rgb_batch(clean_native.float(), kernel)
    residual_low = _blur_native_rgb_batch(residual_candidate, kernel)
    native_low = _blur_native_rgb_batch(native_glare.float(), kernel)
    clean_high = clean_native.float() - clean_low
    residual_high = residual_candidate - residual_low
    native_high = native_glare.float() - native_low
    photographic_high = (1 - lens_native) * residual_high + lens_native * native_high
    rendered = (
        low_clean_alpha * clean_low + (1 - low_clean_alpha) * residual_low
        + native_hf_weight * photographic_high
        + (1 - native_hf_weight) * clean_high
    ).clamp(-1, 1)
    return ((1 - lens_native) * native_glare.float() + lens_native * rendered).clamp(-1, 1)



_RENDER_BLUR_BACKEND = "auto"
_RENDER_BLUR_CHOICE_CACHE: dict[tuple, str] = {}
_RENDER_BLUR_WEIGHT_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_RENDER_BLUR_TUNE_LOG: dict[str, dict[str, float]] = {}


def _blur_avgpool_sep(value: torch.Tensor, k: int) -> torch.Tensor:
    pad = int(k) // 2
    out = F.avg_pool2d(value, kernel_size=(1, int(k)), stride=1, padding=(0, pad))
    return F.avg_pool2d(out, kernel_size=(int(k), 1), stride=1, padding=(pad, 0))


def _blur_avgpool_direct(value: torch.Tensor, k: int) -> torch.Tensor:
    return F.avg_pool2d(value, kernel_size=int(k), stride=1, padding=int(k) // 2)


def _blur_weights(value: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    channels = int(value.shape[1])
    key = (str(value.device), str(value.dtype), channels, int(k))
    cached = _RENDER_BLUR_WEIGHT_CACHE.get(key)
    if cached is None:
        scale = 1.0 / float(int(k))
        wh = torch.full(
            (channels, 1, 1, int(k)), scale,
            device=value.device, dtype=value.dtype,
        )
        wv = torch.full(
            (channels, 1, int(k), 1), scale,
            device=value.device, dtype=value.dtype,
        )
        cached = (wh, wv)
        _RENDER_BLUR_WEIGHT_CACHE[key] = cached
    return cached


def _blur_conv_sep(value: torch.Tensor, k: int) -> torch.Tensor:
    pad = int(k) // 2
    wh, wv = _blur_weights(value, int(k))
    # Avoid TF32 changing the renderer's FP32 box-filter semantics.
    with torch.backends.cudnn.flags(
        enabled=torch.backends.cudnn.enabled,
        benchmark=torch.backends.cudnn.benchmark,
        deterministic=torch.backends.cudnn.deterministic,
        allow_tf32=False,
    ):
        out = F.conv2d(value, wh, stride=1, padding=(0, pad), groups=value.shape[1])
        return F.conv2d(out, wv, stride=1, padding=(pad, 0), groups=value.shape[1])


def _blur_hybrid_vertical_conv(value: torch.Tensor, k: int) -> torch.Tensor:
    pad = int(k) // 2
    _, wv = _blur_weights(value, int(k))
    out = F.avg_pool2d(value, kernel_size=(1, int(k)), stride=1, padding=(0, pad))
    with torch.backends.cudnn.flags(
        enabled=torch.backends.cudnn.enabled,
        benchmark=torch.backends.cudnn.benchmark,
        deterministic=torch.backends.cudnn.deterministic,
        allow_tf32=False,
    ):
        return F.conv2d(out, wv, stride=1, padding=(pad, 0), groups=value.shape[1])


def _blur_transpose_avg(value: torch.Tensor, k: int) -> torch.Tensor:
    """Use the faster horizontal pool for both axes; include layout restore cost."""
    pad = int(k) // 2
    out = F.avg_pool2d(value, kernel_size=(1, int(k)), stride=1, padding=(0, pad))
    out_t = out.transpose(-2, -1)
    out_t = F.avg_pool2d(out_t, kernel_size=(1, int(k)), stride=1, padding=(0, pad))
    return out_t.transpose(-2, -1).contiguous()


_RENDER_BLUR_IMPLS = {
    "avgpool": _blur_avgpool_sep,
    "avgpool2d": _blur_avgpool_direct,
    "conv": _blur_conv_sep,
    "hybrid": _blur_hybrid_vertical_conv,
    "transpose": _blur_transpose_avg,
}


def _blur_tune_key(value: torch.Tensor, k: int) -> tuple:
    # Choose once per channel-count/kernel on this device. With the normal
    # --warmup_batches path all three renderer blur families are tuned before
    # production timing, avoiding first-seen native-size tuning inside the run.
    return (str(value.device), str(value.dtype), int(value.shape[1]), int(k))


def _time_cuda_blur(fn, value: torch.Tensor, k: int, repeats: int = 2) -> float:
    # One warm call lets cuDNN/autotuners settle before CUDA-event timing.
    tmp = fn(value, int(k))
    del tmp
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(int(repeats)):
        tmp = fn(value, int(k))
    end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end)) / float(repeats)
    del tmp
    return elapsed


def _choose_blur_backend(value: torch.Tensor, k: int) -> str:
    requested = str(_RENDER_BLUR_BACKEND).lower()
    if requested != "auto":
        return requested
    if not value.is_cuda:
        return "avgpool"
    key = _blur_tune_key(value, int(k))
    cached = _RENDER_BLUR_CHOICE_CACHE.get(key)
    if cached is not None:
        return cached

    reference = _blur_avgpool_sep(value, int(k))
    timings: dict[str, float] = {}
    # Keep only numerically equivalent candidates. The threshold is far below a
    # uint8 level after [-1,1] -> [0,255] quantization.
    for name, fn in _RENDER_BLUR_IMPLS.items():
        try:
            probe = fn(value, int(k))
            max_err = float((probe - reference).abs().amax().item())
            if max_err > 5e-6:
                continue
            timings[name] = _time_cuda_blur(fn, value, int(k))
            del probe
        except Exception:
            continue
    del reference
    if not timings:
        choice = "avgpool"
    else:
        choice = min(timings, key=timings.get)
    _RENDER_BLUR_CHOICE_CACHE[key] = choice
    _RENDER_BLUR_TUNE_LOG[str(key)] = {
        **{name: float(ms) for name, ms in timings.items()},
        "chosen_index": float(list(_RENDER_BLUR_IMPLS).index(choice)),
    }
    return choice


def _box_blur_separable_batch(field: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Autotuned zero-padded box blur with a numerically guarded backend."""
    k = int(kernel_size)
    value = field.float()
    if k <= 1:
        return value
    if k % 2 == 0:
        raise ValueError("native blur kernel must be odd")
    backend = _choose_blur_backend(value, k)
    fn = _RENDER_BLUR_IMPLS.get(backend)
    if fn is None:
        raise ValueError(f"unknown renderer blur backend {backend!r}")
    return fn(value, k)


def _renderer_gate_maps_eager(
    glare_f: torch.Tensor,
    lens_f: torch.Tensor,
    signed_correction: torch.Tensor,
    eye_mask: torch.Tensor,
    iris_mask: torch.Tensor,
    reconstruction_low: float,
    reconstruction_high: float,
    correction_hf_suppression: float,
    highlight_low: float,
    highlight_high: float,
    highlight_chroma_max: float,
    low_clean_skin: float,
    low_clean_eye: float,
    low_clean_iris: float,
    low_clean_highlight: float,
    native_hf_skin: float,
    native_hf_eye: float,
    native_hf_iris: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    correction_magnitude = signed_correction.abs().mean(dim=1, keepdim=True)
    reconstruction_need = _smoothstep01(
        (correction_magnitude - float(reconstruction_low))
        / (float(reconstruction_high) - float(reconstruction_low))
    )
    beta_inside = torch.full_like(lens_f, float(native_hf_skin))
    beta_inside = torch.lerp(
        beta_inside, torch.full_like(beta_inside, float(native_hf_eye)), eye_mask
    )
    beta_inside = torch.lerp(
        beta_inside, torch.full_like(beta_inside, float(native_hf_iris)), iris_mask
    )
    beta_inside = beta_inside * (
        1.0 - float(correction_hf_suppression) * reconstruction_need
    )

    rgb01 = ((glare_f + 1.0) * 0.5).clamp(0.0, 1.0)
    luminance = rgb01.mean(dim=1, keepdim=True)
    chroma = rgb01.amax(dim=1, keepdim=True) - rgb01.amin(dim=1, keepdim=True)
    bright = _smoothstep01(
        (luminance - float(highlight_low))
        / (float(highlight_high) - float(highlight_low))
    )
    white = _smoothstep01(
        (float(highlight_chroma_max) - chroma)
        / max(float(highlight_chroma_max), 1e-6)
    )
    highlight = (bright * white * lens_f).clamp(0.0, 1.0)

    low_inside = torch.full_like(lens_f, float(low_clean_skin))
    low_inside = torch.maximum(low_inside, float(low_clean_eye) * eye_mask)
    low_inside = torch.maximum(low_inside, float(low_clean_iris) * iris_mask)
    low_inside = torch.maximum(
        low_inside, float(low_clean_highlight) * highlight
    )
    low_clean_alpha = (lens_f * low_inside).clamp(0.0, 1.0)
    beta_inside = (beta_inside * (1.0 - highlight)).clamp(0.0, 1.0)
    native_hf_weight = ((1.0 - lens_f) + lens_f * beta_inside).clamp(0.0, 1.0)
    return correction_magnitude, low_clean_alpha, native_hf_weight


def _renderer_composite_eager(
    clean_f: torch.Tensor,
    residual_candidate: torch.Tensor,
    glare_f: torch.Tensor,
    lens_f: torch.Tensor,
    low_clean_alpha: torch.Tensor,
    native_hf_weight: torch.Tensor,
    clean_low: torch.Tensor,
    residual_low: torch.Tensor,
    native_low: torch.Tensor,
) -> torch.Tensor:
    clean_high = clean_f - clean_low
    residual_high = residual_candidate - residual_low
    native_high = glare_f - native_low
    output_low = torch.lerp(residual_low, clean_low, low_clean_alpha)
    photographic_high = torch.lerp(residual_high, native_high, lens_f)
    output_high = torch.lerp(clean_high, photographic_high, native_hf_weight)
    rendered = (output_low + output_high).clamp(-1.0, 1.0)
    return torch.lerp(glare_f, rendered, lens_f).clamp(-1.0, 1.0)

def decode_from_rendered_batch_optimized(
    model, clean_native, delta_native, native_glare, lens_native, eye_prior_native,
):
    """Autotuned/fused renderer policy; equations match the v3 hybrid path."""
    if clean_native.ndim != 4 or clean_native.shape[1] != 3:
        raise ValueError("clean_native must be [B,3,H,W]")
    clean_f = clean_native.float()
    delta_f = delta_native.float()
    glare_f = native_glare.float()
    lens_f = lens_native.float().clamp(0, 1)
    residual_candidate = glare_f + delta_f
    compile_helpers = bool(getattr(model, "renderer_compile_helpers", True))

    signed_correction = _box_blur_separable_batch(
        delta_f, int(model.reconstruction_blur_kernel)
    )

    eye_mask = torch.zeros_like(lens_f)
    iris_mask = torch.zeros_like(lens_f)
    if eye_prior_native is not None:
        prior = eye_prior_native
        eye_mask = ((prior >= 128) & (prior < 192)).float()
        iris_mask = (prior >= 192).float()
        eye_mask = _smoothstep01(
            _box_blur_separable_batch(
                eye_mask, int(model.eye_core_feather_kernel)
            ).clamp(0, 1)
        )
        iris_mask = _smoothstep01(
            _box_blur_separable_batch(
                iris_mask, int(model.eye_core_feather_kernel)
            ).clamp(0, 1)
        )

    correction_magnitude, low_clean_alpha, native_hf_weight = _run_renderer_helper(
        "decode_gate_maps",
        _renderer_gate_maps_eager,
        (
            glare_f, lens_f, signed_correction, eye_mask, iris_mask,
            float(model.reconstruction_low), float(model.reconstruction_high),
            float(model.correction_hf_suppression),
            float(model.highlight_low), float(model.highlight_high),
            float(model.highlight_chroma_max),
            float(model.low_clean_skin), float(model.low_clean_eye),
            float(model.low_clean_iris), float(model.low_clean_highlight),
            float(model.native_hf_skin), float(model.native_hf_eye),
            float(model.native_hf_iris),
        ),
        compile_helpers,
    )

    if model.use_learned_frequency_renderer:
        low_clean_alpha, native_hf_weight = model.learned_frequency_gate.forward_batch(
            glare_f, clean_f, delta_f,
            correction_magnitude, eye_mask, iris_mask, lens_f,
            low_clean_alpha, native_hf_weight,
            correction_normalizer=float(model.reconstruction_high),
        )

    kernel = int(model.texture_lowpass_kernel)
    clean_low = _box_blur_separable_batch(clean_f, kernel)
    residual_low = _box_blur_separable_batch(residual_candidate, kernel)
    native_low = _box_blur_separable_batch(glare_f, kernel)

    return _run_renderer_helper(
        "decode_composite",
        _renderer_composite_eager,
        (
            clean_f, residual_candidate, glare_f, lens_f,
            low_clean_alpha, native_hf_weight,
            clean_low, residual_low, native_low,
        ),
        compile_helpers,
    )

def decode_from_rendered_batch(
    model, clean_native, delta_native, native_glare, lens_native, eye_prior_native,
):
    backend = getattr(model, "renderer_execution_backend", "optimized")
    if backend == "legacy":
        return decode_from_rendered_batch_legacy(
            model, clean_native, delta_native, native_glare, lens_native,
            eye_prior_native,
        )
    if backend != "optimized":
        raise ValueError(f"unknown renderer_execution_backend {backend!r}")
    return decode_from_rendered_batch_optimized(
        model, clean_native, delta_native, native_glare, lens_native,
        eye_prior_native,
    )

def decode_from_rendered(
    model, clean_native, delta_native, native_glare, lens_native, eye_prior_native,
):
    """Single-image compatibility wrapper around the batched renderer."""
    prior = None if eye_prior_native is None else eye_prior_native.unsqueeze(0)
    return decode_from_rendered_batch(
        model,
        clean_native.unsqueeze(0),
        delta_native.unsqueeze(0),
        native_glare.unsqueeze(0),
        lens_native.unsqueeze(0),
        prior,
    )[0]


def _same_side_groups(samples: Sequence[PreparedSample]) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(int(sample.side), []).append(index)
    return list(groups.values())


def _full_batch_indices(indices: Sequence[int], batch: int) -> bool:
    return len(indices) == int(batch) and all(int(v) == i for i, v in enumerate(indices))


def _native_group_view(
    tensor: torch.Tensor, indices: Sequence[int], side: int
) -> torch.Tensor:
    """Avoid copying the overwhelmingly common full same-side batch."""
    if _full_batch_indices(indices, int(tensor.shape[0])):
        return tensor[:, :, :int(side), :int(side)]
    idx = torch.as_tensor(indices, device=tensor.device, dtype=torch.long)
    return tensor.index_select(0, idx)[:, :, :int(side), :int(side)]


def _lens_group_view(lenses, indices: Sequence[int]) -> torch.Tensor:
    if isinstance(lenses, torch.Tensor):
        if _full_batch_indices(indices, int(lenses.shape[0])):
            return lenses
        idx = torch.as_tensor(indices, device=lenses.device, dtype=torch.long)
        return lenses.index_select(0, idx)
    return torch.stack([lenses[index] for index in indices])


def render_decode_batch(
    model, x0, glare_patches, native_rgb, native_prior, lens, samples,
    stage_timer: Optional[DeviceStageTimer] = None,
    renderer_plan: Optional[
        tuple[tuple[tuple[int, int, int, int], ...], torch.Tensor, torch.Tensor]
    ] = None,
):
    leaves = [sample.leaves_cpu for sample in samples]
    sides = [sample.side for sample in samples]
    lenses = resize_lenses(lens, samples)
    if stage_timer is not None:
        stage_timer.mark("renderer_lens_end")
    predictions: list[Optional[torch.Tensor]] = [None] * len(samples)
    # PoU and decode backends are deliberately split in the hybrid path. The
    # original renderer wins the measured PoU A/B, while Phase-2 contributes its
    # same-side batched/autotuned frequency decode.
    optimized = getattr(model, "renderer_pou_backend", "legacy") == "optimized"
    compile_helpers = bool(getattr(model, "renderer_compile_helpers", True))

    if not model.use_frequency_renderer:
        fields = x0.float() - glare_patches.float()
        rendered_delta = (
            render_continuous_field_batch_optimized(
                fields, samples, model.canvas_size, model.halo_ratio,
                renderer_plan=renderer_plan,
                compile_helpers=compile_helpers,
            )
            if optimized else
            base.render_continuous_field_batch(
                fields, leaves, sides, model.canvas_size, model.halo_ratio
            )
        )
        if stage_timer is not None:
            stage_timer.mark("renderer_pou_end")
        for indices in _same_side_groups(samples):
            side = int(samples[indices[0]].side)
            glare_batch = _native_group_view(native_rgb, indices, side).float()
            delta_batch = torch.stack([rendered_delta[index] for index in indices])
            lens_batch = _lens_group_view(lenses, indices)
            rendered_batch = (glare_batch + delta_batch).clamp(-1, 1)
            output_batch = torch.lerp(glare_batch, rendered_batch, lens_batch).clamp(-1, 1)
            for local_index, output in zip(indices, output_batch):
                predictions[local_index] = output
        return [value for value in predictions if value is not None]

    fields = torch.cat([x0.float(), (x0 - glare_patches).float()], dim=2)
    rendered = (
        render_continuous_field_batch_optimized(
            fields, samples, model.canvas_size, model.halo_ratio,
            renderer_plan=renderer_plan,
            compile_helpers=compile_helpers,
        )
        if optimized else
        base.render_continuous_field_batch(
            fields, leaves, sides, model.canvas_size, model.halo_ratio
        )
    )
    if stage_timer is not None:
        stage_timer.mark("renderer_pou_end")
    for indices in _same_side_groups(samples):
        side = int(samples[indices[0]].side)
        # One 6-channel stack replaces separate clean/delta 3-channel stacks.
        rendered_batch = torch.stack([rendered[index] for index in indices])
        clean_batch = rendered_batch[:, :3]
        delta_batch = rendered_batch[:, 3:]
        glare_batch = _native_group_view(native_rgb, indices, side)
        lens_batch = _lens_group_view(lenses, indices)
        prior_batch = (
            _native_group_view(native_prior, indices, side)
            if model.use_semantic_conditioning else None
        )
        output_batch = decode_from_rendered_batch(
            model, clean_batch, delta_batch, glare_batch, lens_batch, prior_batch
        )
        for local_index, output in zip(indices, output_batch):
            predictions[local_index] = output
    if any(value is None for value in predictions):
        raise RuntimeError("renderer failed to populate every batch output")
    return [value for value in predictions if value is not None]

def assemble_raw_pmf_batch(model, x0, samples):
    """Convert raw PMF x0 tokens to native images without post-render decoding.

    PMF predicts one canonical RGB patch per adaptive leaf.  A full-resolution
    image therefore needs only the linear partition-of-unity (PoU) spatial
    assembly below.  This path deliberately bypasses residual decoding,
    frequency decomposition, learned renderer gates, lens compositing, and
    native high-frequency injection.
    """
    return base.render_continuous_field_batch(
        x0.float(),
        [sample.leaves_cpu for sample in samples],
        [sample.side for sample in samples],
        model.canvas_size,
        model.halo_ratio,
    )


def tensors_to_pils(
    crops: Sequence[torch.Tensor],
    device: torch.device,
    stage_timer: Optional[DeviceStageTimer] = None,
):
    device_arrays = [
        crop.detach().float().clamp(-1, 1).add(1).mul(127.5).round()
        .clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()
        for crop in crops
    ]
    if device.type == "cuda":
        host_arrays = [
            torch.empty(value.shape, dtype=torch.uint8, pin_memory=True)
            for value in device_arrays
        ]
        for host, value in zip(host_arrays, device_arrays):
            host.copy_(value, non_blocking=True)
        if stage_timer is not None:
            stage_timer.mark("output_end")
        torch.cuda.synchronize(device)
    else:
        host_arrays = [value.cpu() for value in device_arrays]
        if stage_timer is not None:
            stage_timer.mark("output_end")
    pil_started = time.perf_counter()
    images = [Image.fromarray(value.numpy()) for value in host_arrays]
    pil_wrap_ms = (time.perf_counter() - pil_started) * 1000.0
    if stage_timer is None:
        return images
    return images, float(pil_wrap_ms)


def infer_batch_once(samples, model, saved, device, args):
    batch_started = time.perf_counter()
    pin = device.type == "cuda"
    pack_started = time.perf_counter()
    rgb_cpu, prior_cpu, leaves_cpu, region_cpu, lens_cpu, patches_cpu = pack_batch_cpu(
        samples, pin_memory=pin
    )
    query_groups_cpu = pack_query_groups_cpu(samples, pin_memory=pin)
    patch_plan_cpu = (
        pack_patch_plan_cpu(samples, pin_memory=pin)
        if patches_cpu is None else tuple()
    )
    renderer_plan_cpu = (
        pack_renderer_plan_cpu(samples, pin_memory=pin)
        if (not args.pmf_only and str(args.renderer_backend) == "optimized") else None
    )
    cpu_pack_ms = (time.perf_counter() - pack_started) * 1000.0
    non_blocking = device.type == "cuda"
    timer = DeviceStageTimer(device)
    timer.mark("device_start")
    rgb_u8 = rgb_cpu.to(device, non_blocking=non_blocking)
    prior_u8 = prior_cpu.to(device, non_blocking=non_blocking)
    leaves = leaves_cpu.to(device, non_blocking=non_blocking)
    region = region_cpu.to(device, non_blocking=non_blocking)
    lens = lens_cpu.to(device, non_blocking=non_blocking)
    query_groups = tuple(
        (
            indices.to(device, non_blocking=non_blocking),
            valid.to(device, non_blocking=non_blocking),
        )
        for indices, valid in query_groups_cpu
    )
    patch_plan = tuple(
        (rows, indices.to(device, non_blocking=non_blocking))
        for rows, indices in patch_plan_cpu
    )
    renderer_plan = None
    if renderer_plan_cpu is not None:
        renderer_specs, renderer_meta_i_cpu, renderer_meta_f_cpu = renderer_plan_cpu
        renderer_plan = (
            renderer_specs,
            renderer_meta_i_cpu.to(device, non_blocking=non_blocking),
            renderer_meta_f_cpu.to(device, non_blocking=non_blocking),
        )
    timer.mark("h2d_end")
    # Match pil_to_tensor's operation order in the legacy single-image path.
    native_rgb = rgb_u8.float().div_(127.5).sub_(1.0)
    patches = (
        patches_cpu.to(device, non_blocking=non_blocking)
        if patches_cpu is not None else gpu_extract_canonical_patches(
            native_rgb, patch_plan,
            int(samples[0].leaves.shape[0]), int(saved.canonical_size),
        )
    )
    timer.mark("patch_end")
    with torch.inference_mode(), autocast_context(device, args.amp):
        x0 = multistep_x0_batch_same_seed(
            model, patches, leaves, region, args.seed, args.nfe,
            query_groups=query_groups,
        )
    timer.mark("pmf_end")
    with torch.inference_mode():
        if args.pmf_only:
            timer.mark("renderer_lens_end")
            predictions = assemble_raw_pmf_batch(model, x0, samples)
            timer.mark("renderer_pou_end")
        else:
            predictions = render_decode_batch(
                model, x0, patches, native_rgb, prior_u8, lens, samples,
                stage_timer=timer, renderer_plan=renderer_plan,
            )
    timer.mark("renderer_end")
    crops = []
    for prediction, sample in zip(predictions, samples):
        left, top, right, bottom = sample.crop_box
        crops.append(prediction[:, top:bottom, left:right])
    images, pil_wrap_ms = tensors_to_pils(crops, device, stage_timer=timer)
    raw_device_ms = timer.consecutive_elapsed_ms()
    stage_ms = {
        "cpu_batch_pack": float(cpu_pack_ms),
        "h2d_inputs": raw_device_ms["device_start_to_h2d_end"],
        "normalize_and_patch_extract_or_transfer": raw_device_ms[
            "h2d_end_to_patch_end"
        ],
        "pmf_network": raw_device_ms["patch_end_to_pmf_end"],
        "renderer_lens_resize": raw_device_ms.get("pmf_end_to_renderer_lens_end", 0.0),
        "renderer_pou_assembly": raw_device_ms.get("renderer_lens_end_to_renderer_pou_end", 0.0),
        "renderer_decode_policy": raw_device_ms.get("renderer_pou_end_to_renderer_end", 0.0),
        "native_renderer": (
            raw_device_ms.get("pmf_end_to_renderer_lens_end", 0.0)
            + raw_device_ms.get("renderer_lens_end_to_renderer_pou_end", 0.0)
            + raw_device_ms.get("renderer_pou_end_to_renderer_end", 0.0)
        ),
        "output_quantize_and_d2h": raw_device_ms[
            "renderer_end_to_output_end"
        ],
        "pil_wrap": float(pil_wrap_ms),
    }
    observation = {
        "batch_size": int(len(samples)),
        "side_pixels": [int(sample.side) for sample in samples],
        "stage_total_ms": stage_ms,
        "successful_batch_wall_ms": float(
            (time.perf_counter() - batch_started) * 1000.0
        ),
    }
    return InferenceBatchResult(images=images, observations=[observation])


def infer_batch_adaptive(samples, model, saved, device, args):
    try:
        return infer_batch_once(samples, model, saved, device, args)
    except torch.cuda.OutOfMemoryError:
        if len(samples) == 1:
            raise
        torch.cuda.empty_cache()
        middle = len(samples) // 2
        print(f"[OOM] batch={len(samples)} -> retry {middle}+{len(samples)-middle}", flush=True)
        left = infer_batch_adaptive(samples[:middle], model, saved, device, args)
        right = infer_batch_adaptive(samples[middle:], model, saved, device, args)
        return InferenceBatchResult(
            images=left.images + right.images,
            observations=left.observations + right.observations,
            oom_retries=1 + left.oom_retries + right.oom_retries,
        )


def save_prediction(
    sample: PreparedSample,
    image: Image.Image,
    run_signature: dict,
    png_compress_level: int,
):
    save_started = time.perf_counter()
    output_path = sample.record.output_path
    mkdir_started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mkdir_ms = (time.perf_counter() - mkdir_started) * 1000.0
    image_started = time.perf_counter()
    if output_path.suffix.lower() == ".png":
        # Compression level changes file size only; decoded pixels stay lossless.
        image.save(output_path, compress_level=int(png_compress_level))
    else:
        image.save(output_path)
    image_write_ms = (time.perf_counter() - image_started) * 1000.0
    manifest_started = time.perf_counter()
    write_prediction_manifest(
        sample.record.glare_path,
        output_path,
        run_signature,
        input_signature=sample.input_signature,
    )
    manifest_ms = (time.perf_counter() - manifest_started) * 1000.0
    return {
        "output_mkdir": float(mkdir_ms),
        "image_encode_and_write": float(image_write_ms),
        "manifest_write": float(manifest_ms),
        "save_total": float((time.perf_counter() - save_started) * 1000.0),
    }


def make_output_path(input_root: Path, glare_path: Path, output_root_or_file: Path, output_name: str) -> Path:
    if input_root.is_file():
        return output_root_or_file
    rel_dir = glare_path.parent.relative_to(input_root)
    return output_root_or_file / rel_dir / output_name


def _numeric_summary(values: Sequence[float]) -> dict:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "total_ms": 0.0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    array = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(array.size),
        "total_ms": float(array.sum()),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def _value_summary(values: Sequence[float]) -> dict:
    timed = _numeric_summary(values)
    return {
        "count": timed["count"],
        "total": timed["total_ms"],
        "mean": timed["mean_ms"],
        "median": timed["median_ms"],
        "p95": timed["p95_ms"],
        "min": timed["min_ms"],
        "max": timed["max_ms"],
    }


def _batch_stage_summary(observations: Sequence[dict], stage_name: str) -> dict:
    batch_values = []
    image_weighted_values = []
    total_images = 0
    for observation in observations:
        batch_size = int(observation["batch_size"])
        value = float(observation["stage_total_ms"][stage_name])
        batch_values.append(value)
        image_weighted_values.extend([value / batch_size] * batch_size)
        total_images += batch_size
    result = _numeric_summary(batch_values)
    result["total_images"] = int(total_images)
    result["throughput_normalized_mean_ms_per_image"] = (
        float(sum(batch_values) / total_images) if total_images else None
    )
    result["image_weighted_median_ms_per_image"] = (
        float(statistics.median(image_weighted_values))
        if image_weighted_values else None
    )
    result["image_weighted_p95_ms_per_image"] = (
        float(np.percentile(np.asarray(image_weighted_values), 95))
        if image_weighted_values else None
    )
    return result


def _hardware_report(device: torch.device) -> dict:
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "logical_cpu_count": os.cpu_count(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        ),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report.update({
            "cuda_device_name": properties.name,
            "cuda_total_memory_mib": float(properties.total_memory / (2 ** 20)),
            "cuda_capability": [int(properties.major), int(properties.minor)],
        })
    return report


def _resolve_timing_json_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_dir() or path.suffix.lower() != ".json":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = path / f"infer_time_{stamp}.json"
    return path


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def run_timing_warmup(records, model, saved, device, args, cfg) -> tuple[int, float]:
    """Warm the production compute path outside the authoritative wall clock."""
    repeats = max(0, int(args.warmup_batches))
    if repeats == 0 or not records:
        return 0, 0.0
    selected = list(records[:max(1, int(args.batch_size))])
    builders = ThreadLocalGeometryBuilders(args)
    warmup_started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=args.preprocess_workers) as executor:
            futures = [
                executor.submit(
                    prepare_one, record, args, saved, cfg, model, builders
                )
                for record in selected
            ]
            samples = [future.result() for future in futures]
        samples.sort(key=lambda sample: sample.side)
        for _ in range(repeats):
            infer_batch_adaptive(samples, model, saved, device, args)
    finally:
        builders.close_all()
    return repeats, float((time.perf_counter() - warmup_started) * 1000.0)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input', required=True,
                   help='Single input glare image path.')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True,
                   help='Output image path.')

    p.add_argument('--pattern', default='glare.png',
                   help='Deprecated in single-image mode; ignored.')
    p.add_argument('--output_name', default='pred.png',
                   help='Deprecated in single-image mode; ignored.')
    p.add_argument('--skip_existing', action='store_true',
                   help='Skip samples whose output file already exists.')

    p.add_argument('--use_existing_geometry', action='store_true', default=False,
                   help='Reuse compatible multi-face eye_prior.png and face_meta.json if present.')
    p.add_argument('--force_redetect', action='store_true',
                   help='Ignore existing eye_prior.png / face_meta.json and redetect geometry.')
    p.add_argument('--strict_geometry_cache', action='store_true',
                   help='Require the current multi-face geometry cache plus matching source SHA256.')
    p.add_argument('--no_detection_fallback', action='store_true',
                   help='If cached geometry is missing/invalid, fail instead of running detector.')

    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--backend', choices=('auto', 'tasks', 'legacy'), default='auto')
    p.add_argument('--model_asset_path', default=str(Path(__file__).parent/'assets'/'face_landmarker.task'))
    p.add_argument('--min_detection_confidence', type=float, default=.5)
    p.add_argument('--min_presence_confidence', type=float, default=.5)
    p.add_argument('--min_tracking_confidence', type=float, default=.5)
    p.add_argument('--max_faces', type=int, default=8,
                   help='Maximum number of faces MediaPipe may return; all valid faces are merged into one geometry prior.')
    p.add_argument('--geometry_delegate', choices=('cpu','gpu'), default='cpu',
                   help='MediaPipe Tasks delegate. GPU is supported by MediaPipe on Ubuntu but can contend with the PMF CUDA workload.')
    p.add_argument('--geometry_detect_max_side',type=int,default=768,
                   help='Downscale only online landmark detection to this maximum side; 0 preserves full-resolution detection.')
    p.add_argument('--geometry_resize_interp',
                   choices=('area','linear','linear_exact'), default='area',
                   help='Pre-resize interpolation for online landmark detection. linear is the fast candidate; area preserves the previous path.')
    p.add_argument('--geometry_retry_max_side',type=int,default=0,
                   help='If fast detection fails, retry once at this max side with INTER_AREA; 0 disables retry.')
    p.add_argument('--write_geometry_cache',action='store_true',
                   help='After successful online detection, atomically write eye_prior.png and face_meta.json for future cache hits.')

    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--nfe', type=int, default=1,
                   help='Number of equal MeanFlow intervals; 1 preserves the original one-step fast path.')
    p.add_argument('--pmf_only', action='store_true',
                   help='save raw PMF x0 after linear PoU assembly; bypass all post-render decoding')
    p.add_argument('--disable_learned_renderer', action='store_true',
                   help='keep frequency decomposition but use handcrafted gates only')
    p.add_argument('--disable_frequency_renderer', action='store_true',
                   help='bypass the entire low/high-frequency renderer')
    p.add_argument('--max_items', type=int, default=0,
                   help='Deprecated in single-image mode; ignored.')
    speed=p.add_argument_group('high-throughput inference')
    speed.add_argument('--batch_size',type=int,default=1,
                       help='Batched PMF/renderer size; CUDA OOM automatically splits the batch.')
    speed.add_argument('--preprocess_workers',type=int,
                       default=1,
                       help='Parallel image/geometry/grid preparation workers.')
    speed.add_argument('--grid_workers',type=int,default=0,
                       help='Spawned CPU processes for adaptive-grid solving; 0 keeps grid construction in preprocessing threads.')
    speed.add_argument('--prefetch_batches',type=int,default=1,
                       help='Number of batches prepared ahead of GPU execution.')
    speed.add_argument('--save_workers',type=int,default=1,
                       help='Asynchronous PNG/manifest writer threads.')
    speed.add_argument('--png_compress_level',type=int,choices=range(0,10),default=1,
                       metavar='0..9',
                       help='PNG compression level; pixels remain lossless, lower is faster/larger.')
    speed.add_argument('--patch_backend',choices=('gpu','pil'),default='gpu',
                       help='gpu is fast bicubic+antialias extraction; pil exactly matches the legacy preprocessing.')
    speed.add_argument('--amp',choices=('off','bf16','fp16'),default='bf16',
                       help='Mixed precision for PMF only; renderer always remains FP32.')
    speed.add_argument('--attention_backend',choices=('legacy','sdpa'),default='legacy',
                       help='Attention implementation. legacy is the measured-fast default for grouped adaptive attention; sdpa remains available for A/B.')
    speed.add_argument('--renderer_backend',choices=('legacy','hybrid','optimized'),default='hybrid',
                       help=('Renderer execution path. hybrid combines the original fast/stable PoU assembly with Phase-2 batched, autotuned decode; '
                             'optimized also enables the Phase-2 metadata/scatter PoU; legacy keeps original PoU plus the batched legacy equations.'))
    speed.add_argument('--renderer_blur_backend',
                       choices=('auto','avgpool','avgpool2d','conv','hybrid','transpose'),
                       default='auto',
                       help='Native box-blur backend. auto benchmarks numerically-equivalent candidates per size bucket and caches the fastest.')
    speed.add_argument('--disable_renderer_compile',action='store_true',
                       help='Disable best-effort torch.compile fusion for PoU geometry/source prep, renderer pointwise maps, and the fixed 256x256 learned gate core.')
    speed.add_argument('--opencv_threads',type=int,default=1,
                       help='OpenCV internal threads per process. Keep at 1 while preprocessing already uses a worker pool to avoid CPU oversubscription.')
    speed.add_argument('--grid_solver',choices=('fast','verify','dp'),default='fast',
                       help='Exact heap grid solver, regression verifier, or legacy dynamic programming.')
    speed.add_argument('--progress_interval',type=int,default=100,
                       help='Print one progress line after roughly this many processed images; 0 prints only the final summary.')
    speed.add_argument('--verbose_warnings',action='store_true',
                       help='Print every per-image geometry warning instead of only a final grouped summary.')
    timing=p.add_argument_group('runtime measurement')
    timing.add_argument('--no_save',action='store_true',
                        help='Benchmark through D2H/PIL materialization without PNG encoding, prediction writes, manifests, or save threads.')
    timing.add_argument('--warmup_batches',type=int,default=3,
                        help='Run this many GPU warm-up repeats on the first prepared batch before starting the measured production pass.')
    timing.add_argument('--timing_json',default=None,
                        help='Write a machine-readable timing report. A directory value gets a UTC-stamped JSON filename.')
    lens=p.add_argument_group('renderer lens-support overrides')
    lens.add_argument('--renderer_lens_expand_x',type=float,default=None)
    lens.add_argument('--renderer_lens_expand_y',type=float,default=None)
    lens.add_argument('--renderer_lens_min_height_ratio',type=float,default=None)
    lens.add_argument('--renderer_lens_feather',type=float,default=None)
    return p.parse_args()


def main():
    global _RENDER_BLUR_BACKEND
    program_started = time.perf_counter()
    run_timestamp_utc = datetime.now(timezone.utc).isoformat()
    args = parse_args()
    reset_warning_summary()
    validation_started = time.perf_counter()
    try:
        config.validate_inference_args(args)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"inference configuration error: {error}") from error
    if args.warmup_batches < 0:
        raise SystemExit("inference configuration error: --warmup_batches must be >= 0")
    if int(args.nfe) < 1:
        raise SystemExit("inference configuration error: --nfe must be >= 1")
    if args.no_save and args.skip_existing:
        raise SystemExit(
            "inference configuration error: --no_save and --skip_existing cannot "
            "be combined because no prediction files are written"
        )
    if args.force_redetect:
        args.use_existing_geometry = False
    if int(args.grid_workers) < 0:
        raise SystemExit("inference configuration error: --grid_workers must be >= 0")
    if str(args.geometry_delegate).lower() == "gpu" and int(args.preprocess_workers) != 1:
        print(
            "[GEOMETRY] GPU delegate selected: forcing --preprocess_workers 1 "
            "to avoid multiple MediaPipe GPU contexts; grid work can still use --grid_workers.",
            flush=True,
        )
        args.preprocess_workers = 1
    if int(args.opencv_threads) < 0:
        raise SystemExit(
            "inference configuration error: --opencv_threads must be >= 0"
        )
    cv2.setNumThreads(int(args.opencv_threads))
    startup_stage_ms = {
        "argument_validation": (time.perf_counter() - validation_started) * 1000.0
    }

    input_path = Path(args.input)
    output_path = Path(args.output)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit(f"CUDA device {args.device!r} requested but CUDA is unavailable")
    base.configure_cuda_speed(device)
    if args.amp == 'bf16' and device.type == 'cuda':
        if hasattr(torch.cuda,'is_bf16_supported') and not torch.cuda.is_bf16_supported():
            print('[warn] BF16 unsupported; falling back to FP16',flush=True)
            args.amp='fp16'

    model_started = time.perf_counter()
    model, saved = load_model(args.checkpoint, device)
    startup_stage_ms["model_checkpoint_load"] = (
        time.perf_counter() - model_started
    ) * 1000.0
    attention_modules = configure_attention_backend(model, args.attention_backend)
    # Keep the two Phase-2 renderer changes independently selectable. Measurements
    # showed that its batched/autotuned decode is useful in isolation, while its
    # metadata/scatter PoU path can lose to the original shape-grouped renderer.
    model.renderer_execution_backend = (
        "optimized" if args.renderer_backend in ("hybrid", "optimized")
        else "legacy"
    )
    model.renderer_pou_backend = (
        "optimized" if args.renderer_backend == "optimized" else "legacy"
    )
    _RENDER_BLUR_BACKEND = str(args.renderer_blur_backend)
    model.renderer_compile_helpers = not bool(args.disable_renderer_compile)
    if hasattr(model, 'learned_frequency_gate'):
        model.learned_frequency_gate.compile_core = not bool(args.disable_renderer_compile)
    renderer_overrides=apply_renderer_geometry_overrides(model,args)
    if renderer_overrides:
        print(f"[INFO] Renderer lens overrides: {renderer_overrides}",flush=True)
    if args.disable_learned_renderer:
        model.use_learned_frequency_renderer = False
    if args.disable_frequency_renderer:
        model.use_frequency_renderer = False
    cfg = config.make_grid_config(saved)
    run_signature=prediction_run_signature(Path(args.checkpoint),model,args)

    discovery_started = time.perf_counter()
    glare_files = [input_path]
    startup_stage_ms["input_discovery"] = (
        time.perf_counter() - discovery_started
    ) * 1000.0

    is_batch = False
    total = 1
    done = 0
    skipped = 0
    failed = 0

    print(f"[BUILD] {SCRIPT_BUILD}",flush=True)
    print(f"[INFO] Found {total} input file(s).", flush=True)
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Batch mode: {is_batch}", flush=True)
    print(
        "[INFO] Output stage: "
        + ("raw PMF x0 (PoU assembly only; post-render bypassed)" if args.pmf_only else "final rendered prediction"),
        flush=True,
    )
    print(
        "[INFO] Prediction saving: "
        + (
            "disabled (timing ends after D2H/PIL materialization)"
            if args.no_save else
            "enabled (PNG and manifest writes included in wall time)"
        ),
        flush=True,
    )
    print(
        f"[FAST] batch={args.batch_size} preprocess_workers={args.preprocess_workers} "
        f"grid_workers={args.grid_workers} prefetch_batches={args.prefetch_batches} patch={args.patch_backend} "
        f"amp={args.amp} nfe={args.nfe} attention={args.attention_backend} renderer={args.renderer_backend} "
        f"blur={args.renderer_blur_backend} renderer_compile={not args.disable_renderer_compile} "
        f"opencv_threads={cv2.getNumThreads()} "
        f"grid_solver={args.grid_solver} "
        f"geometry_max_side={args.geometry_detect_max_side} "
        f"geometry_interp={args.geometry_resize_interp} "
        f"geometry_retry={args.geometry_retry_max_side} "
        f"geometry_delegate={args.geometry_delegate} "
        f"max_faces={args.max_faces} "
        f"geometry_cache_write={args.write_geometry_cache} "
        f"input_hash={_needs_input_signature(args)} "
        f"png_level={args.png_compress_level}",flush=True,
    )
    print(
        f"[ATTN] backend={args.attention_backend} "
        f"adaptive_modules={attention_modules['rope_padded']} "
        f"conditioner_modules={attention_modules['conditioner_cross']}",
        flush=True,
    )
    print(
        f"[RENDER] mode={args.renderer_backend} "
        f"pou={model.renderer_pou_backend} "
        f"decode={model.renderer_execution_backend}",
        flush=True,
    )

    record_scan_started = time.perf_counter()
    records=[]
    for index,glare_path in enumerate(glare_files,start=1):
        out_path=output_path
        if args.skip_existing and prediction_is_current(glare_path,out_path,run_signature):
            skipped+=1
            print(f"[SKIP] [{index}/{total}] {glare_path} -> {out_path} already exists",flush=True)
            continue
        if args.skip_existing and out_path.exists():
            print(
                f"[STALE] [{index}/{total}] {out_path} has no matching inference manifest; regenerating",
                flush=True,
            )
        records.append(InputRecord(glare_path,out_path))
    startup_stage_ms["output_cache_scan"] = (
        time.perf_counter() - record_scan_started
    ) * 1000.0

    warmup_repeats, warmup_ms = run_timing_warmup(
        records, model, saved, device, args, cfg
    )
    if warmup_repeats:
        print(
            f"[WARMUP] repeats={warmup_repeats} first_batch_images="
            f"{min(len(records), args.batch_size)} elapsed={warmup_ms:.1f} ms "
            "(excluded from measured production wall)",
            flush=True,
        )
        if _RENDER_BLUR_CHOICE_CACHE:
            choices = ", ".join(
                f"C{key[2]}/k{key[3]}={value}"
                for key, value in sorted(_RENDER_BLUR_CHOICE_CACHE.items(), key=lambda item: str(item[0]))
            )
            print(f"[RENDER-TUNE] blur {choices}", flush=True)
        if not args.disable_renderer_compile:
            print(
                f"[RENDER-COMPILE] active={sorted(_RENDER_COMPILED_HELPERS.keys())} "
                f"disabled={sorted(_RENDER_DISABLED_HELPERS)} "
                f"gate_core={getattr(model.learned_frequency_gate, '_compiled_core_fn', None) is not False if hasattr(model, 'learned_frequency_gate') else False}",
                flush=True,
            )
        # The measured pass should report its own warnings only.
        reset_warning_summary()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    executor_started = time.perf_counter()
    geometry_builders=ThreadLocalGeometryBuilders(args)
    preprocess_executor=ThreadPoolExecutor(max_workers=args.preprocess_workers)
    grid_executor = (
        ProcessPoolExecutor(
            max_workers=args.grid_workers,
            mp_context=mp_ctx.get_context("spawn"),
        )
        if int(args.grid_workers) > 0 and model.use_adaptive_grid
        else None
    )
    if grid_executor is not None:
        # Start worker processes before the authoritative production wall begins.
        warm = [grid_executor.submit(_grid_worker_ping) for _ in range(int(args.grid_workers))]
        for future in warm:
            future.result()
    save_executor=(
        None if args.no_save
        else ThreadPoolExecutor(max_workers=args.save_workers)
    )
    startup_stage_ms["executor_setup"] = (
        time.perf_counter() - executor_started
    ) * 1000.0
    save_futures=[]
    cpu_prepare_total_ms=0.0
    cpu_prepare_images=0
    pipeline_total_ms=0.0
    pipeline_images=0
    save_total_ms=0.0
    save_images=0
    prepare_stage_values: dict[str, list[float]] = defaultdict(list)
    save_stage_values: dict[str, list[float]] = defaultdict(list)
    inference_observations: list[dict] = []
    geometry_source_counts: Counter = Counter()
    input_widths: list[float] = []
    input_heights: list[float] = []
    square_sides: list[float] = []
    token_counts: list[float] = []
    requested_batch_sizes: list[int] = []
    preprocess_exposed_wait_ms=0.0
    final_save_drain_wait_ms=0.0
    oom_retries=0
    last_progress=0
    window_size=max(args.batch_size,args.batch_size*args.prefetch_batches)
    windows=[records[i:i+window_size] for i in range(0,len(records),window_size)]

    def submit_window(window):
        return [preprocess_executor.submit(
            prepare_one,record,args,saved,cfg,model,geometry_builders,grid_executor
        ) for record in window]

    started=time.perf_counter()
    work_completed=None
    try:
        current=submit_window(windows[0]) if windows else []
        for window_index in range(len(windows)):
            prepared=[]
            for future,record in zip(current,windows[window_index]):
                try:
                    wait_started=time.perf_counter()
                    sample=future.result()
                    preprocess_exposed_wait_ms+=(
                        time.perf_counter()-wait_started
                    )*1000.0
                    prepared.append(sample)
                    cpu_prepare_total_ms+=float(sample.cpu_prepare_ms)
                    cpu_prepare_images+=1
                    for name,value in sample.prepare_stage_ms.items():
                        prepare_stage_values[name].append(float(value))
                    geometry_source_counts[sample.geometry_source]+=1
                    input_widths.append(float(sample.original_size[0]))
                    input_heights.append(float(sample.original_size[1]))
                    square_sides.append(float(sample.side))
                    token_counts.append(float(sample.leaves.shape[0]))
                except Exception as error:
                    failed+=1
                    print(f"[FAIL] {record.glare_path}: {error}",flush=True)
            following=(
                submit_window(windows[window_index+1])
                if window_index+1<len(windows) else []
            )
            # Similar native sizes reduce batch padding and improve renderer grouping.
            prepared.sort(key=lambda sample:sample.side)
            for start in range(0,len(prepared),args.batch_size):
                batch=prepared[start:start+args.batch_size]
                requested_batch_sizes.append(len(batch))
                try:
                    pipeline_started=time.perf_counter()
                    batch_result=infer_batch_adaptive(batch,model,saved,device,args)
                    images=batch_result.images
                    # tensors_to_pils() has already synchronized the batch's D2H
                    # copy. Extra before/after batch synchronizations only add
                    # driver round-trips and cannot improve timing correctness.
                    pipeline_total_ms+=(time.perf_counter()-pipeline_started)*1000.0
                    pipeline_images+=len(batch)
                    inference_observations.extend(batch_result.observations)
                    oom_retries+=int(batch_result.oom_retries)
                    if save_executor is not None:
                        for sample,image in zip(batch,images):
                            save_futures.append(save_executor.submit(
                                save_prediction,
                                sample,
                                image,
                                run_signature,
                                args.png_compress_level,
                            ))
                    done+=len(batch)
                    elapsed=max(time.perf_counter()-started,1e-9)
                    processed=done+skipped+failed
                    progress_interval=max(0,int(args.progress_interval))
                    if (
                        processed == total
                        or (
                            progress_interval > 0
                            and processed-last_progress >= progress_interval
                        )
                    ):
                        mean_pipeline=pipeline_total_ms/max(pipeline_images,1)
                        print(
                            f"[PASS] {processed}/{total} "
                            f"throughput={done/elapsed:.2f} img/s "
                            f"avg_pipeline={mean_pipeline:.2f} ms/img",
                            flush=True,
                        )
                        last_progress=processed
                except Exception as error:
                    failed+=len(batch)
                    for sample in batch:
                        print(f"[FAIL] {sample.record.glare_path}: {error}",flush=True)
            current=following
        if save_executor is not None:
            save_drain_started=time.perf_counter()
            for future in save_futures:
                try:
                    save_timing=future.result()
                    save_total_ms+=float(save_timing["save_total"])
                    save_images+=1
                    for name,value in save_timing.items():
                        save_stage_values[name].append(float(value))
                except Exception as error:
                    failed+=1; done=max(0,done-1)
                    print(f"[FAIL] asynchronous save: {error}",flush=True)
            final_save_drain_wait_ms=(
                time.perf_counter()-save_drain_started
            )*1000.0
        work_completed=time.perf_counter()
    finally:
        preprocess_executor.shutdown(wait=True,cancel_futures=True)
        geometry_builders.close_all()
        if grid_executor is not None:
            grid_executor.shutdown(wait=True, cancel_futures=True)
        if save_executor is not None:
            save_executor.shutdown(wait=True,cancel_futures=True)

    print("=" * 72, flush=True)
    elapsed=(work_completed or time.perf_counter())-started
    program_elapsed=(work_completed or time.perf_counter())-program_started
    end_to_end_mean_ms=(elapsed*1000.0/done) if done else 0.0
    pipeline_mean_ms=pipeline_total_ms/max(pipeline_images,1)
    cpu_prepare_mean_ms=cpu_prepare_total_ms/max(cpu_prepare_images,1)
    save_mean_ms=save_total_ms/max(save_images,1)
    print(
        f"[SUMMARY] total={total}, done={done}, skipped={skipped}, failed={failed} "
        f"wall={elapsed:.1f}s throughput={(done/max(elapsed,1e-9)):.2f} img/s",
        flush=True,
    )
    print(
        f"[TIMING] average_end_to_end_ms_per_image={end_to_end_mean_ms:.3f}",
        flush=True,
    )
    print(
        f"[TIMING] average_inference_pipeline_ms_per_image={pipeline_mean_ms:.3f}",
        flush=True,
    )
    print(
        f"[TIMING] average_cpu_prepare_ms_per_image={cpu_prepare_mean_ms:.3f}",
        flush=True,
    )
    if args.no_save:
        print("[TIMING] prediction_save=disabled",flush=True)
    else:
        print(
            f"[TIMING] average_async_save_ms_per_image={save_mean_ms:.3f}",
            flush=True,
        )
    additive_prepare_stages = (
        "input_read_decode_hash", "geometry_total", "grid_build",
        "region_probabilities", "square_padding", "renderer_lens_mask",
        "patch_extract_or_plan",
    )
    for name in additive_prepare_stages:
        summary = _numeric_summary(prepare_stage_values.get(name, []))
        if summary["mean_ms"] is not None:
            print(
                f"[TIMING][CPU-PREP] {name}="
                f"{summary['mean_ms']:.3f} ms/img "
                f"p95={summary['p95_ms']:.3f}",
                flush=True,
            )
    inference_stage_names = (
        "cpu_batch_pack", "h2d_inputs",
        "normalize_and_patch_extract_or_transfer", "pmf_network",
        "renderer_lens_resize", "renderer_pou_assembly", "renderer_decode_policy",
        "native_renderer", "output_quantize_and_d2h", "pil_wrap",
    )
    inference_stage_report = {
        name: _batch_stage_summary(inference_observations, name)
        for name in inference_stage_names
    }
    for name in inference_stage_names:
        value = inference_stage_report[name][
            "throughput_normalized_mean_ms_per_image"
        ]
        if value is not None:
            print(
                f"[TIMING][INFER] {name}={value:.3f} ms/img",
                flush=True,
            )
    for name in ("image_encode_and_write", "manifest_write"):
        summary = _numeric_summary(save_stage_values.get(name, []))
        if summary["mean_ms"] is not None:
            print(
                f"[TIMING][SAVE] {name}={summary['mean_ms']:.3f} ms/img",
                flush=True,
            )
    print(
        f"[TIMING][OVERLAP] exposed_preprocess_wait="
        f"{preprocess_exposed_wait_ms/max(done,1):.3f} ms/img "
        f"final_save_drain_wait={final_save_drain_wait_ms/max(done,1):.3f} ms/img",
        flush=True,
    )
    print(
        "[TIMING] component latencies overlap; "
        "average_end_to_end_ms_per_image is the authoritative wall-clock average.",
        flush=True,
    )

    batch_wall_values=[]
    batch_wall_image_values=[]
    for observation in inference_observations:
        value=float(observation["successful_batch_wall_ms"])
        batch_size=int(observation["batch_size"])
        batch_wall_values.append(value)
        batch_wall_image_values.extend([value/batch_size]*batch_size)
    peak_cuda_mib=(
        float(torch.cuda.max_memory_allocated(device)/(2**20))
        if device.type == "cuda" else None
    )
    timing_report = {
        "schema_version": 1,
        "script_build": SCRIPT_BUILD,
        "run_timestamp_utc": run_timestamp_utc,
        "paths": {
            "script": str(Path(__file__).resolve()),
            "input": str(input_path.resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "output": str(output_path.resolve()),
        },
        "configuration": {
            "batch_size": int(args.batch_size),
            "preprocess_workers": int(args.preprocess_workers),
            "grid_workers": int(args.grid_workers),
            "prefetch_batches": int(args.prefetch_batches),
            "save_predictions": bool(not args.no_save),
            "save_workers": int(0 if args.no_save else args.save_workers),
            "requested_save_workers": int(args.save_workers),
            "png_compress_level": int(args.png_compress_level),
            "patch_backend": str(args.patch_backend),
            "amp": str(args.amp),
            "nfe": int(args.nfe),
            "attention_backend": str(args.attention_backend),
            "renderer_backend": str(args.renderer_backend),
            "renderer_pou_backend": str(model.renderer_pou_backend),
            "renderer_decode_backend": str(model.renderer_execution_backend),
            "renderer_blur_backend": str(args.renderer_blur_backend),
            "renderer_compile_enabled": bool(not args.disable_renderer_compile),
            "opencv_threads": int(cv2.getNumThreads()),
            "grid_solver": str(args.grid_solver),
            "geometry_detect_max_side": int(args.geometry_detect_max_side),
            "geometry_resize_interp": str(args.geometry_resize_interp),
            "geometry_retry_max_side": int(args.geometry_retry_max_side),
            "geometry_delegate": str(args.geometry_delegate),
            "write_geometry_cache": bool(args.write_geometry_cache),
            "input_hash_enabled": bool(_needs_input_signature(args)),
            "use_existing_geometry": bool(args.use_existing_geometry),
            "strict_geometry_cache": bool(args.strict_geometry_cache),
            "pmf_only": bool(args.pmf_only),
            "frequency_renderer": bool(model.use_frequency_renderer),
            "learned_frequency_renderer": bool(
                model.use_learned_frequency_renderer
            ),
            "seed": int(args.seed),
            "warmup_batches": int(args.warmup_batches),
        },
        "hardware": _hardware_report(device),
        "renderer_runtime": {
            "blur_requested": str(_RENDER_BLUR_BACKEND),
            "blur_choices": {str(key): value for key, value in _RENDER_BLUR_CHOICE_CACHE.items()},
            "blur_tuning_ms": dict(_RENDER_BLUR_TUNE_LOG),
            "compiled_helpers": sorted(_RENDER_COMPILED_HELPERS.keys()),
            "compiled_helper_signature_locked": sorted(_RENDER_COMPILED_SIGNATURES.keys()),
            "disabled_compile_helpers": sorted(_RENDER_DISABLED_HELPERS),
        },
        "counts": {
            "discovered": int(total),
            "scheduled": int(len(records)),
            "completed": int(done),
            "completed_and_saved": int(done if not args.no_save else 0),
            "skipped": int(skipped),
            "failed": int(failed),
            "geometry_source": dict(sorted(geometry_source_counts.items())),
            "requested_batch_sizes": requested_batch_sizes,
            "effective_successful_batch_sizes": [
                int(value["batch_size"]) for value in inference_observations
            ],
            "oom_split_retries": int(oom_retries),
        },
        "input_shape": {
            "width_pixels": _value_summary(input_widths),
            "height_pixels": _value_summary(input_heights),
            "square_side_pixels": _value_summary(square_sides),
            "adaptive_tokens": _value_summary(token_counts),
        },
        "startup_excluded_from_production_wall": {
            "stages": {
                name: float(value) for name,value in startup_stage_ms.items()
            },
            "warmup_repeats": int(warmup_repeats),
            "warmup_images": int(min(len(records), args.batch_size))
                if warmup_repeats else 0,
            "warmup_total_ms": float(warmup_ms),
            "cold_cli_to_work_complete_seconds": float(program_elapsed),
        },
        "authoritative_production_wall": {
            "scope": (
                (
                    "first preprocessing submission through final D2H and PIL "
                    "materialization; PNG encoding, prediction and manifest "
                    "writes, and save threads are disabled; "
                    if args.no_save else
                    "first preprocessing submission through completion of the "
                    "last asynchronous image and manifest write; "
                )
                + "model load, discovery, cache scan, optional warm-up, and "
                "executor teardown are excluded"
            ),
            "includes_prediction_save": bool(not args.no_save),
            "wall_seconds": float(elapsed),
            "images": int(done),
            "average_ms_per_image": float(end_to_end_mean_ms) if done else None,
            "throughput_images_per_second": (
                float(done/max(elapsed,1e-9)) if done else None
            ),
            "valid_full_run": bool(
                done == total and skipped == 0 and failed == 0
            ),
        },
        "overlap_diagnostics": {
            "cpu_prepare_work_total_ms": float(cpu_prepare_total_ms),
            "inference_pipeline_wall_total_ms": float(pipeline_total_ms),
            "async_save_work_total_ms": float(save_total_ms),
            "exposed_preprocess_wait_total_ms": float(
                preprocess_exposed_wait_ms
            ),
            "final_save_drain_wait_total_ms": float(final_save_drain_wait_ms),
            "note": (
                (
                    "CPU preparation and inference overlap; saving is disabled. "
                    if args.no_save else
                    "CPU preparation, inference, and asynchronous saving overlap; "
                )
                + "Component totals must not be added. Exposed waits are already "
                "part of the authoritative production wall."
            ),
        },
        "preprocessing_per_image_task_latency": {
            name: _numeric_summary(values)
            for name,values in sorted(prepare_stage_values.items())
        },
        "inference_successful_batches": {
            "outer_pipeline_average_ms_per_image": (
                float(pipeline_mean_ms) if pipeline_images else None
            ),
            "successful_batch_wall_ms": _numeric_summary(batch_wall_values),
            "successful_batch_wall_image_weighted": _numeric_summary(
                batch_wall_image_values
            ),
            "stages": inference_stage_report,
            "observations": inference_observations,
        },
        "asynchronous_save_per_image_task_latency": {
            name: _numeric_summary(values)
            for name,values in sorted(save_stage_values.items())
        },
        "peak_cuda_memory_mib": peak_cuda_mib,
        "measurement_notes": [
            "CUDA stages use events on the active stream and add no intermediate synchronization.",
            "CPU preprocessing values are per-task latencies and overlap across worker threads.",
            (
                "PNG encoding, prediction writes, manifests, and save threads are excluded."
                if args.no_save else
                "Save values are per-task latencies and overlap across writer threads and inference."
            ),
            "The first CUDA batch is cold unless warmup_batches is greater than zero.",
            "Use a full run with skipped=0 and failed=0 for a paper throughput result.",
        ],
    }
    if args.timing_json:
        timing_json_path=_resolve_timing_json_path(args.timing_json)
        _write_json_atomic(timing_json_path,timing_report)
        print(f"[TIMING-JSON] {timing_json_path.resolve()}",flush=True)
    print_warning_summary()
    print("=" * 72, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
