from __future__ import annotations

"""Argument definitions and fail-fast validation for training and inference."""

import argparse
import math
from pathlib import Path

from models.spatial import AdaptiveGridConfig


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


def validate_training_args(args: argparse.Namespace) -> None:
    """Validate the complete three-stage trainer configuration before side effects."""
    validate_common_args(args)
    if not Path(args.data_root).expanduser().is_dir():
        raise FileNotFoundError(f"--data_root is not a directory: {args.data_root}")
    if not bool(getattr(args, "uniform_grid", False)) and not Path(args.grid_cache).expanduser().is_dir():
        raise FileNotFoundError(f"--grid_cache is not a directory: {args.grid_cache}")
    if not str(args.output).strip():
        raise ValueError("--output must not be empty")
    if Path(args.output).expanduser().is_file():
        raise ValueError(f"--output points to a file: {args.output}")
    for name in ("resume", "init_weights"):
        value = str(getattr(args, name, "") or "")
        if value and not Path(value).expanduser().is_file():
            raise FileNotFoundError(f"--{name} not found: {value}")

    for name in ("num_steps", "batch_size", "grad_accum_steps"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    if int(args.num_workers) < 0:
        raise ValueError("--num_workers must be >= 0")
    if int(args.seed) < 0:
        raise ValueError("--seed must be >= 0")
    if int(args.conditioner_bottleneck) <= 0:
        raise ValueError("--conditioner_bottleneck must be positive")
    _finite("--condition_gate_init", args.condition_gate_init)
    _finite("--context_gate_init", args.context_gate_init)

    for name in ("lr", "joint_backbone_lr", "joint_conditioner_lr", "renderer_lr"):
        _positive(f"--{name}", getattr(args, name))
    _nonnegative("--weight_decay", args.weight_decay)
    _nonnegative("--grad_clip", args.grad_clip)
    if not 0.0 <= _finite("--ema_decay", args.ema_decay) < 1.0:
        raise ValueError("--ema_decay must be in [0,1)")
    if int(args.log_every) <= 0:
        raise ValueError("--log_every must be positive")
    for name in ("save_every", "vis_every", "vis_count"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name} must be non-negative")

    if int(args.pmf_only_steps) <= 0 or int(args.gate_warmup_steps) <= 0:
        raise ValueError("--pmf_only_steps and --gate_warmup_steps must be positive")
    joint_steps = int(args.num_steps) - int(args.pmf_only_steps) - int(args.gate_warmup_steps)
    if joint_steps <= 0:
        raise ValueError("--pmf_only_steps + --gate_warmup_steps must be smaller than --num_steps")
    if int(args.renderer_ramp_steps) <= 0 or int(args.renderer_ramp_steps) > joint_steps:
        raise ValueError("--renderer_ramp_steps must be positive and not exceed joint-phase duration")
    _positive_odd("--renderer_lowpass_kernel", args.renderer_lowpass_kernel)
    _positive("--charbonnier_eps", args.charbonnier_eps)
    for name in (
        "lambda_renderer", "lambda_renderer_low", "lambda_renderer_gradient",
        "lambda_gate_prior", "lambda_gate_tv",
    ):
        _nonnegative(f"--{name}", getattr(args, name))
    if not bool(args.learned_frequency_renderer):
        raise ValueError("three-stage training requires --learned_frequency_renderer")
    if args.sampling == "balanced":
        if int(args.batch_size) < 2:
            raise ValueError("balanced domain batches require --batch_size >= 2")
        fraction = _finite("--real_batch_fraction", args.real_batch_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("--real_batch_fraction must be in (0,1)")

    if int(args.val_every) < 0:
        raise ValueError("--val_every must be >= 0")
    for name in ("val_synthetic_count", "val_real_count"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    _probability("--val_mixed_mask_weight", args.val_mixed_mask_weight)
    if int(args.val_ssim_win_size) < 3 or int(args.val_ssim_win_size) % 2 == 0:
        raise ValueError("--val_ssim_win_size must be an odd integer >= 3")
    if int(args.early_stop_patience) < 0:
        raise ValueError("--early_stop_patience must be >= 0")
    _nonnegative("--early_stop_min_delta", args.early_stop_min_delta)
    joint_start = int(args.pmf_only_steps) + int(args.gate_warmup_steps)
    early_start = int(args.early_stop_start_step)
    if early_start != -1 and not joint_start <= early_start < int(args.num_steps):
        raise ValueError(
            "--early_stop_start_step must be -1 (automatic) or within the joint phase"
        )
    if int(args.early_stop_patience) > 0 and int(args.val_every) == 0:
        raise ValueError("early stopping requires --val_every > 0")


def validate_inference_args(args: argparse.Namespace) -> None:
    """Validate inference-only CLI values before checkpoint/model allocation."""
    input_path = Path(args.input).expanduser()
    checkpoint = Path(args.checkpoint).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"--input does not exist: {input_path}")
    if not input_path.is_file() and not input_path.is_dir():
        raise ValueError(f"--input must be a file or directory: {input_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"--checkpoint not found: {checkpoint}")
    if not str(args.output).strip():
        raise ValueError("--output must not be empty")
    if input_path.is_dir() and not str(args.pattern):
        raise ValueError("--pattern must not be empty in directory mode")
    for name in (
        "min_detection_confidence", "min_presence_confidence", "min_tracking_confidence",
    ):
        _probability(f"--{name}", getattr(args, name))
    if int(args.max_items) < 0:
        raise ValueError("--max_items must be >= 0")
    if int(getattr(args, "geometry_detect_max_side", 0)) < 0:
        raise ValueError("--geometry_detect_max_side must be >= 0")
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


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data_root", required=True)
    p.add_argument("--grid_cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="")
    p.add_argument("--canvas_size", type=int, default=256)
    p.add_argument("--base_patch_size", type=int, default=16,
                   help="logical-grid reference patch size; keep 16 to reuse the current cache")
    p.add_argument("--canonical_size", type=int, default=24)
    p.add_argument("--target_tokens", type=int, default=256)
    p.add_argument("--canvas_weight", type=float, default=0.02)
    p.add_argument("--face_weight", type=float, default=0.20)
    p.add_argument("--periocular_weight", type=float, default=1.00)
    p.add_argument("--eye_weight", type=float, default=3.00)
    p.add_argument("--iris_weight", type=float, default=8.00)
    p.add_argument("--eyelid_weight", type=float, default=5.00)
    p.add_argument("--lens_weight", type=float, default=4.00)
    p.add_argument("--lens_expand_x", type=float, default=0.45)
    p.add_argument("--lens_expand_y", type=float, default=1.00)
    p.add_argument("--hard_lens_max_patch", type=int, default=8)
    p.add_argument("--correction_threshold", type=int, default=64)
    p.add_argument("--correction_dilate_radius", type=int, default=1)
    p.add_argument("--hard_correction_max_patch", type=int, default=8)
    p.add_argument("--hard_anchor_radius", type=int, default=1)
    p.add_argument("--hard_anchor_max_patch", type=int, default=8)
    p.add_argument("--noise_scale", type=float, default=0.05)
    p.add_argument("--t_eps", type=float, default=1e-4)
    p.add_argument("--p_mean", type=float, default=-0.4)
    p.add_argument("--p_std", type=float, default=1.0)
    p.add_argument("--fm_ratio", type=float, default=0.5)
    p.add_argument("--halo_ratio", type=float, default=0.05)
    p.add_argument("--renderer_lens_expand_x", type=float, default=1.50)
    p.add_argument("--renderer_lens_expand_y", type=float, default=2.50)
    p.add_argument("--renderer_lens_min_height_ratio", type=float, default=1.10)
    p.add_argument("--renderer_lens_feather", type=float, default=2.0)
    p.add_argument("--renderer_lens_power", type=float, default=4.0)
    p.add_argument("--reconstruction_low", type=float, default=0.03)
    p.add_argument("--reconstruction_high", type=float, default=0.10)
    p.add_argument("--reconstruction_blur_kernel", type=int, default=11)
    p.add_argument("--eye_core_feather_kernel", type=int, default=7)
    p.add_argument("--texture_lowpass_kernel", type=int, default=15)
    p.add_argument("--native_hf_skin", type=float, default=0.90)
    p.add_argument("--native_hf_eye", type=float, default=0.30)
    p.add_argument("--native_hf_iris", type=float, default=0.15)
    p.add_argument("--correction_hf_suppression", type=float, default=0.75)
    p.add_argument("--low_clean_skin", type=float, default=0.00)
    p.add_argument("--low_clean_eye", type=float, default=0.35)
    p.add_argument("--low_clean_iris", type=float, default=0.55)
    p.add_argument("--low_clean_highlight", type=float, default=0.90)
    p.add_argument("--highlight_low", type=float, default=0.82)
    p.add_argument("--highlight_high", type=float, default=0.96)
    p.add_argument("--highlight_chroma_max", type=float, default=0.16)
    p.add_argument("--learned_frequency_renderer", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--renderer_gate_width", type=int, default=32)
    p.add_argument("--renderer_gate_map_size", type=int, default=256)
    p.add_argument("--renderer_gate_residual_scale", type=float, default=1.0)
    p.add_argument("--eye_prior_dropout", type=float, default=0.0)
    p.add_argument("--norm_p", type=float, default=1.0)
    p.add_argument("--norm_eps", type=float, default=0.01)
    p.add_argument("--lambda_u", type=float, default=1.0)
    p.add_argument("--lambda_v", type=float, default=1.0)
    p.add_argument("--lambda_x0", type=float, default=0.05)
    p.add_argument("--lambda_eye_gradient", type=float, default=0.005)
    p.add_argument("--eye_gradient_periocular_weight", type=float, default=0.05)
    p.add_argument("--eye_gradient_eye_weight", type=float, default=1.0)
    p.add_argument("--eye_gradient_iris_weight", type=float, default=2.0)
    p.add_argument("--lambda_preserve", type=float, default=0.01)
    p.add_argument("--x0_mid_size", type=int, default=12)
    p.add_argument("--x0_coarse_size", type=int, default=6)
    p.add_argument("--x0_coarse_weight", type=float, default=0.5)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--no_sdpa", action="store_true")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--vis_seed", type=int, default=0)
