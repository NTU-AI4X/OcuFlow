from __future__ import annotations

"""Small content-aware residual gate for the native-resolution frequency renderer."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def _small(self, value: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            value.unsqueeze(0).float(),
            size=(self.map_size, self.map_size),
            mode="bilinear", align_corners=False,
        )

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
        norm = max(float(correction_normalizer), 1e-6)
        features = torch.cat([
            self._small(native_glare),
            self._small(clean_candidate),
            self._small((0.5 * correction).clamp(-1.0, 1.0)),
            self._small((correction_magnitude / norm).clamp(0.0, 2.0)),
            self._small(eye_mask),
            self._small(iris_mask),
            self._small(lens_alpha),
            self._small(handcrafted_low),
            self._small(handcrafted_hf),
        ], dim=1)
        if features.shape[1] != self.input_channels:
            raise RuntimeError(
                f"learned renderer built {features.shape[1]} channels; "
                f"expected {self.input_channels}"
            )
        residual = self.head(self.blocks(self.stem(features)))
        residual = F.interpolate(
            residual, size=native_glare.shape[-2:],
            mode="bilinear", align_corners=False,
        )[0]
        residual = self.residual_scale * torch.tanh(residual)
        # Regularize only the support where the gate can affect the rendered image.
        self.last_residual = residual * lens_alpha

        # Lens multiplication makes the renderer immutable outside its geometric
        # support. Additive residuals preserve the old gates exactly at init and
        # can move a handcrafted zero upward or a handcrafted one downward.
        learned_low = (
            handcrafted_low + lens_alpha * residual[0:1]
        ).clamp(0.0, 1.0)
        learned_hf = (
            handcrafted_hf + lens_alpha * residual[1:2]
        ).clamp(0.0, 1.0)
        return learned_low, learned_hf

    def residual_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.last_residual is None:
            zero = self.head.weight.new_zeros(())
            return zero, zero
        value = self.last_residual
        magnitude = value.abs().mean()
        tv_h = (value[..., 1:, :] - value[..., :-1, :]).abs().mean()
        tv_w = (value[..., :, 1:] - value[..., :, :-1]).abs().mean()
        return magnitude, 0.5 * (tv_h + tv_w)
