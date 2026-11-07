from __future__ import annotations

import math
from functools import partial
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .embedder import BottleneckPatchEmbedder, LabelEmbedder, TimestepEmbedder
from .torch_models import RMSNorm, SwiGLUMlp, TorchLinear


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

        All samples in the bucket share one sequence length. Each query scale is
        evaluated by one batched matrix multiplication. The five group outputs
        are concatenated and restored to token order with a single scatter.
        """
        if x.ndim != 3:
            raise ValueError(f"expected [B,L,D], got {tuple(x.shape)}")
        q, k, v = self._qkv(x)
        scale = 1.0 / math.sqrt(self.head_dim)
        group_phase = (2, 0, 1, 2, 3)
        k_rotated = tuple(apply_rotary_phase_batched(k, phase) for phase in rope_phases)
        key_mask = None if all_keys_valid else key_valid[:, None, None, :]
        area = key_area_bias[:, None, None, :]
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
            logits = torch.einsum(
                "bqhd,bkhd->bhqk", q_group * scale, k_rotated[phase_id]
            )
            logits = logits + area.to(logits.dtype)
            if key_mask is not None:
                logits = logits.masked_fill(~key_mask, torch.finfo(logits.dtype).min)
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
            attended_all = attended_all * valid_all[:, :, None, None].to(attended_all.dtype)
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
        if bool(coarse.any()):
            cx = parent_x[coarse].clamp(0, grid - 2)
            cy = parent_y[coarse].clamp(0, grid - 2)
            coarse_pos = (
                table[cy, cx]
                + table[cy, cx + 1]
                + table[cy + 1, cx]
                + table[cy + 1, cx + 1]
            ) * 0.25
            pos = pos.clone()
            pos[coarse] = coarse_pos

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

        batch = int(image_offsets.numel() - 1)
        lengths = [
            int(image_offsets[index + 1]) - int(image_offsets[index])
            for index in range(batch)
        ]
        max_length = max(lengths, default=0)
        padded_image_tokens = int(padded_image_tokens)
        if padded_image_tokens < max_length:
            raise ValueError(
                f"padded_image_tokens={padded_image_tokens} is smaller than batch max {max_length}"
            )
        if padded_image_tokens <= 0:
            raise ValueError("padded_image_tokens must be positive")

        slot_parts = [
            sample * padded_image_tokens
            + torch.arange(length, device=image_embed.device, dtype=torch.long)
            for sample, length in enumerate(lengths)
        ]
        slot_index = torch.cat(slot_parts, dim=0) if slot_parts else torch.empty(0, device=image_embed.device, dtype=torch.long)
        total_slots = batch * padded_image_tokens

        padded_embed = image_embed.new_zeros(total_slots, self.hidden_size)
        padded_embed = padded_embed.index_copy(0, slot_index, image_embed)
        padded_embed = padded_embed.reshape(batch, padded_image_tokens, self.hidden_size)

        padded_leaves = leaves.new_zeros(total_slots, 3)
        padded_leaves[:, 2] = self.patch_size
        padded_leaves = padded_leaves.index_copy(0, slot_index, leaves)
        padded_leaves = padded_leaves.reshape(batch, padded_image_tokens, 3)

        image_valid_flat = torch.zeros(total_slots, device=leaves.device, dtype=torch.bool)
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

        prefix_query = torch.zeros_like(key_valid)
        prefix_query[:, : self.prefix_tokens] = True
        group_masks = [prefix_query]
        group_masks.extend(is_image & (sizes == level) for level in levels)
        query_groups = tuple(
            self._make_padded_query_group(mask) for mask in group_masks
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
            region_probs=region_probs, image_tokens_override=image_tokens_override
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
