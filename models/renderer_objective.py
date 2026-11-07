from __future__ import annotations

"""Shared lens-supported objective for renderer warm-up and joint fine-tuning."""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RendererLosses:
    """Separate data fidelity from gate-only regularization."""

    total: torch.Tensor
    data: torch.Tensor
    gate_regularization: torch.Tensor
    metrics: Dict[str, torch.Tensor]


def sample_from_batch(batch: Dict[str, object], index: int) -> Dict[str, object]:
    """Extract one ragged native-resolution sample from ``collate_batch`` output."""
    return {
        "glare_patches": batch["glare_patches"][index],
        "leaves_cpu": batch["leaves_cpu"][index],
        "native_glare": batch["native_glare"][index],
        "native_prior_u8": batch["native_prior_u8"][index],
        "native_lens_u8": batch["native_lens_u8"][index],
        "original_size": batch["original_size"][index],
        "native_side": batch["native_side"][index],
        "metadata": batch["metadata"][index],
    }


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / (weight.sum() * value.shape[0]).clamp_min(1e-6)


def renderer_loss(pred, clean, lens, gate, args):
    """Native renderer loss normalized only over the editable lens support."""
    weight = lens
    error = torch.sqrt(
        (pred.float() - clean.float()).square() + float(args.charbonnier_eps) ** 2
    )
    reconstruction = weighted_mean(error, weight)

    kernel = int(args.renderer_lowpass_kernel)
    pred_low = F.avg_pool2d(pred.float().unsqueeze(0), kernel, 1, kernel // 2)[0]
    clean_low = F.avg_pool2d(clean.float().unsqueeze(0), kernel, 1, kernel // 2)[0]
    low = weighted_mean((pred_low - clean_low).abs(), weight)

    dx = (pred[..., 1:] - pred[..., :-1]) - (clean[..., 1:] - clean[..., :-1])
    dy = (pred[..., 1:, :] - pred[..., :-1, :]) - (clean[..., 1:, :] - clean[..., :-1, :])
    grad_x = weighted_mean(dx.abs(), weight[..., 1:])
    grad_y = weighted_mean(dy.abs(), weight[..., 1:, :])
    gradient = 0.5 * (grad_x + grad_y)

    prior, tv = gate.residual_regularization()
    data = (
        reconstruction
        + float(args.lambda_renderer_low) * low
        + float(args.lambda_renderer_gradient) * gradient
    )
    gate_regularization = (
        float(args.lambda_gate_prior) * prior
        + float(args.lambda_gate_tv) * tv
    )
    total = data + gate_regularization
    metrics = {
        "renderer": total.detach(),
        "renderer_data": data.detach(),
        "renderer_gate_regularization": gate_regularization.detach(),
        "renderer_reconstruction": reconstruction.detach(),
        "renderer_low": low.detach(),
        "renderer_gradient": gradient.detach(),
        "gate_prior": prior.detach(),
        "gate_tv": tv.detach(),
    }
    return RendererLosses(total, data, gate_regularization, metrics)
