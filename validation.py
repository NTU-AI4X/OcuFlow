from __future__ import annotations

"""Fast in-training validation matching the image metrics in eval v3.3.

The expensive MediaPipe NME and LPIPS metrics remain part of the final test pass.
Training-time model selection uses a fixed holdout from ``train`` and never the
paper test folders.
"""

import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch

import core as base

try:
    from skimage.metrics import structural_similarity
except Exception as error:  # pragma: no cover - exercised by startup validation
    structural_similarity = None
    STRUCTURAL_SIMILARITY_ERROR = repr(error)
else:
    STRUCTURAL_SIMILARITY_ERROR = ""


FAST_METRICS = (
    "PSNR_mask", "PSNR_mixed", "PSNR_outside", "PSNR_full",
    "SSIM_mask", "SSIM_mixed", "SSIM_outside", "SSIM_full",
)


def select_holdout_names(
    data_root, *, seed: int, synthetic_count: int, real_count: int,
    synthetic_prefix: str = "syn_", real_prefix: str = "pub_",
):
    train_root = Path(data_root) / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"training split not found: {train_root}")
    candidates = {"synthetic": [], "real": []}
    for folder in sorted(train_root.iterdir()):
        if not folder.is_dir():
            continue
        domain = (
            "synthetic" if folder.name.startswith(str(synthetic_prefix))
            else "real" if folder.name.startswith(str(real_prefix)) else None
        )
        if domain is None:
            continue
        required = ("clean.png", "glare.png", "mask.png", "eye_prior.png", "face_meta.json")
        if all((folder / filename).is_file() for filename in required):
            candidates[domain].append(folder.name)

    requested = {"synthetic": int(synthetic_count), "real": int(real_count)}
    selected = {}
    for offset, domain in enumerate(("synthetic", "real")):
        count = requested[domain]
        if count <= 0:
            raise ValueError(f"validation {domain}_count must be positive")
        if len(candidates[domain]) <= count:
            raise ValueError(
                f"not enough {domain} samples for a disjoint holdout: "
                f"available={len(candidates[domain])}, requested={count}"
            )
        rng = random.Random(int(seed) + 70_001 + offset)
        ordered = list(candidates[domain])
        rng.shuffle(ordered)
        valid_names = []
        for name in ordered:
            mask = cv2.imread(
                str(train_root / name / "mask.png"), cv2.IMREAD_GRAYSCALE
            )
            if mask is None:
                continue
            mask_bool = mask > 127
            if not mask_bool.any() or mask_bool.all():
                continue
            valid_names.append(name)
            if len(valid_names) == count:
                break
        if len(valid_names) < count:
            raise ValueError(
                f"not enough {domain} samples with a non-empty, non-full mask: "
                f"available={len(valid_names)}, requested={count}"
            )
        selected[domain] = sorted(valid_names)
    return selected


def _mse(gt_bgr, pred_bgr, region):
    if region is None or not region.any():
        return np.nan
    gt = gt_bgr.astype(np.float32) / 255.0
    pred = pred_bgr.astype(np.float32) / 255.0
    return float(np.mean((gt[region] - pred[region]) ** 2))


def _psnr(mse):
    if not np.isfinite(mse):
        return np.nan
    if mse <= 0:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def _erode(mask, radius):
    if radius <= 0:
        return mask.copy()
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def compute_fast_metrics(
    gt_bgr, pred_bgr, mask_bool, *, mixed_mask_weight=0.5, ssim_win_size=7
):
    """Return the eight non-LPIPS/non-landmark metrics from eval v3.3."""
    if structural_similarity is None:
        raise RuntimeError(
            "scikit-image is required for in-training SSIM validation: "
            f"{STRUCTURAL_SIMILARITY_ERROR}"
        )
    if gt_bgr.shape != pred_bgr.shape:
        raise ValueError(f"validation image shape mismatch: {gt_bgr.shape} != {pred_bgr.shape}")
    valid = np.ones(gt_bgr.shape[:2], dtype=bool)
    mask = mask_bool.astype(bool) & valid
    outside = valid & ~mask
    if not mask.any() or not outside.any():
        raise ValueError("validation mask must contain both editable and outside pixels")

    mse_mask = _mse(gt_bgr, pred_bgr, mask)
    mse_outside = _mse(gt_bgr, pred_bgr, outside)
    mse_full = _mse(gt_bgr, pred_bgr, valid)
    weight = float(mixed_mask_weight)
    mse_mixed = weight * mse_mask + (1.0 - weight) * mse_outside

    gt_gray = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    pred_gray = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    win = min(int(ssim_win_size), min(gt_gray.shape))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        raise ValueError("validation image is too small for SSIM")
    _, ssim_map = structural_similarity(
        gt_gray, pred_gray, data_range=1.0, win_size=win,
        channel_axis=None, full=True,
    )
    ssim_map = np.asarray(ssim_map, dtype=np.float32)
    radius = win // 2
    valid_support = _erode(valid, radius)
    mask_support = _erode(mask, radius) & valid_support
    outside_support = _erode(outside, radius) & valid_support
    if not mask_support.any():
        mask_support = mask & valid_support
    if not mask_support.any():
        mask_support = mask
    if not outside_support.any():
        outside_support = outside & valid_support
    if not outside_support.any():
        outside_support = outside
    full_support = valid_support if valid_support.any() else valid

    def region_mean(region):
        values = ssim_map[region]
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else np.nan

    ssim_mask = region_mean(mask_support)
    ssim_outside = region_mean(outside_support)
    return {
        "PSNR_mask": _psnr(mse_mask),
        "PSNR_mixed": _psnr(mse_mixed),
        "PSNR_outside": _psnr(mse_outside),
        "PSNR_full": _psnr(mse_full),
        "SSIM_mask": ssim_mask,
        "SSIM_mixed": weight * ssim_mask + (1.0 - weight) * ssim_outside,
        "SSIM_outside": ssim_outside,
        "SSIM_full": region_mean(full_support),
    }


def _tensor_bgr_u8(tensor):
    rgb = (
        ((tensor.detach().float().clamp(-1, 1) + 1.0) * 127.5)
        .round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    )
    return np.ascontiguousarray(rgb[..., ::-1])


def _aggregate(rows):
    output = {"num_samples": len(rows)}
    for metric in FAST_METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        valid = np.isfinite(values) | np.isposinf(values)
        output[metric] = float(values[valid].mean()) if valid.any() else None
        output[f"{metric}_count"] = int(valid.sum())
    return output


@torch.inference_mode()
def evaluate_model(
    model, dataset, device, *, seed: int, mixed_mask_weight: float, ssim_win_size: int
):
    rows = {"synthetic": [], "real": []}
    for index in range(len(dataset)):
        sample = base.move_sample(dataset[index], device)
        pred, *_ = model.predict(sample, seed=int(seed) + index)
        box = sample["crop_box"]
        pred_crop = base.crop_original(pred.clamp(-1, 1), box)
        clean_crop = base.crop_original(sample["native_clean"], box)
        mask_crop = base.crop_original(sample["native_mask_u8"], box)[0]
        metrics = compute_fast_metrics(
            _tensor_bgr_u8(clean_crop), _tensor_bgr_u8(pred_crop),
            mask_crop.detach().cpu().numpy() > 127,
            mixed_mask_weight=mixed_mask_weight, ssim_win_size=ssim_win_size,
        )
        domain = "synthetic" if str(sample["name"]).startswith("syn_") else "real"
        rows[domain].append(metrics)

    summaries = {domain: _aggregate(values) for domain, values in rows.items()}
    macro = {"num_samples": sum(value["num_samples"] for value in summaries.values())}
    for metric in FAST_METRICS:
        domain_values = [summaries[domain][metric] for domain in ("synthetic", "real")]
        if any(value is None for value in domain_values):
            raise RuntimeError(f"validation metric {metric} is unavailable for one domain")
        macro[metric] = float(np.mean(domain_values))
    return {"macro": macro, **summaries}


def flatten_summary(summary):
    row = {}
    for domain in ("macro", "synthetic", "real"):
        row[f"{domain}_num_samples"] = summary[domain]["num_samples"]
        for metric in FAST_METRICS:
            row[f"{domain}_{metric}"] = summary[domain][metric]
    return row
