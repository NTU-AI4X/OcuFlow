#!/usr/bin/env python3
from __future__ import annotations

"""
Mask-free native-resolution inference.
Supports:
1) single image inference
2) recursive batch inference over a folder (e.g. test_real / test_syn)

Examples
--------
Single image:
    python infer.py \
        --input /path/to/glare.png \
        --checkpoint /path/to/checkpoint.pt \
        --output /path/to/pred.png

Batch folder:
    python infer.py \
        --input /media/liutao/B/demo/Datasets/data/test_real \
        --checkpoint /path/to/checkpoint.pt \
        --output /media/liutao/B/demo/Datasets/preds/test_real

Batch entire dataset root (will recursively find all glare.png):
    python infer.py \
        --input /media/liutao/B/demo/Datasets/data \
        --checkpoint /path/to/checkpoint.pt \
        --output /media/liutao/B/demo/Datasets/preds
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import fnmatch
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Optional, Sequence

# MediaPipe/absl otherwise emits several C++ INFO/WARNING lines for every
# detector instance.  Set these before the detector module is imported lazily.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

import config
import core as base
from model import CleanResidualPMF, load_inference_state
from models.spatial import (
    GEOMETRY_VERSION,
    LANDMARK_SCHEMA,
    build_adaptive_grid,
    build_uniform_grid,
    build_geometry_maps,
    build_renderer_lens_alpha,
    extract_canonical_patches,
    logical_canvas,
    source_signature,
    load_metadata,
)


SCRIPT_BUILD = "infer_fast_v4_2_timing_quiet_20260911"

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
        min_detection_confidence=a.min_detection_confidence,
        min_presence_confidence=a.min_presence_confidence,
        min_tracking_confidence=a.min_tracking_confidence,
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

    def _ensure_detector(self):
        if self._detector is not None:
            return
        from prepare_geometry import LandmarkDetector, build_prior_and_meta
        self._ctx = LandmarkDetector(self._dargs)
        self._detector = self._ctx.__enter__()
        self._build_prior_and_meta = build_prior_and_meta

    def build(self, image: Image.Image):
        self._ensure_detector()
        rgb = np.asarray(image, dtype=np.uint8)
        original_h, original_w = rgb.shape[:2]
        detect_rgb = rgb
        longest = max(original_h, original_w)
        max_side = int(getattr(self.args, "geometry_detect_max_side", 768))
        if max_side > 0 and longest > max_side:
            scale = float(max_side) / float(longest)
            detect_rgb = cv2.resize(
                rgb,
                (max(1, round(original_w * scale)), max(1, round(original_h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        if not detect_rgb.flags.c_contiguous:
            detect_rgb = np.ascontiguousarray(detect_rgb)

        # LandmarkDetector.detect() converts BGR back to RGB and expands every
        # landmark. Bypass those costs while keeping identical MediaPipe input.
        wrapper = self._detector
        if wrapper.backend == "tasks":
            from prepare_geometry import mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=detect_rgb)
            result = wrapper.detector.detect(mp_image)
            landmarks = result.face_landmarks[0] if result.face_landmarks else None
        else:
            result = wrapper.detector.process(detect_rgb)
            landmarks = (
                result.multi_face_landmarks[0].landmark
                if result.multi_face_landmarks else None
            )

        points = None
        if landmarks is not None:
            if len(landmarks) > int(GEOMETRY_LANDMARK_INDICES.max()):
                points = np.zeros(
                    (GEOMETRY_LANDMARK_DENSE_COUNT, 2), dtype=np.float32
                )
                for idx in GEOMETRY_LANDMARK_INDICES:
                    point = landmarks[int(idx)]
                    points[int(idx), 0] = float(point.x) * float(original_w)
                    points[int(idx), 1] = float(point.y) * float(original_h)
            else:
                # Compatibility fallback for older 468-point FaceMesh models.
                points = np.asarray(
                    [
                        (float(point.x) * float(original_w),
                         float(point.y) * float(original_h))
                        for point in landmarks
                    ],
                    dtype=np.float32,
                )

        # build_prior_and_meta reads only the image dimensions, not its colours.
        prior_np, meta = self._build_prior_and_meta(rgb, points, self._dargs)
        prior = Image.fromarray(prior_np)
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
        "patch_backend": str(args.patch_backend),
        "amp": str(args.amp),
        "grid_solver": str(args.grid_solver),
        "geometry_detect_max_side": int(args.geometry_detect_max_side),
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


def load_or_build_geometry(
    glare_path: Path,
    image: Image.Image,
    args,
    geometry_builder: LazyGeometryBuilder,
    input_signature: Optional[dict] = None,
):
    sample_dir = glare_path.parent
    prior_path = sample_dir / 'eye_prior.png'
    meta_path = sample_dir / 'face_meta.json'

    if args.use_existing_geometry:
        prior, meta, reason = validate_cached_geometry(
            glare_path,
            image,
            prior_path,
            meta_path,
            actual_signature=input_signature,
            allow_legacy_v1=not bool(getattr(args, "strict_geometry_cache", False)),
        )
        if prior is not None and meta is not None:
            return prior, meta, "cached"
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
    return prior, meta, "detected"


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
    input_signature: Optional[dict] = None
    cpu_prepare_ms: float = 0.0


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


def _load_rgb_and_signature(path: Path) -> tuple[Image.Image, dict]:
    """Read compressed bytes once, then hash and decode from the same buffer."""
    encoded = Path(path).read_bytes()
    signature = {
        "algorithm": "sha256",
        "digest": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
    }
    with Image.open(BytesIO(encoded)) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB").copy()
    return image, signature


def prepare_one(
    record: InputRecord,
    args,
    saved,
    cfg,
    model,
    geometry_builders: ThreadLocalGeometryBuilders,
) -> PreparedSample:
    prepare_started = time.perf_counter()
    # The former path opened the file for PIL and then read it again for SHA256.
    # Decode and hash the same compressed byte buffer to halve source-file I/O.
    image, input_signature = _load_rgb_and_signature(record.glare_path)
    prior, meta, geometry_source = load_or_build_geometry(
        record.glare_path,
        image,
        args,
        geometry_builders.get(),
        input_signature=input_signature,
    )
    if not bool(meta.get("success")):
        record_warning(
            args,
            "geometry_fallback",
            record.glare_path,
            f"eye detection/meta not fully successful: "
            f"{meta.get('fail_reason', 'unknown')}; using fallback geometry",
        )

    grid = (
        build_adaptive_grid(prior, meta, image.size, cfg, solver=args.grid_solver)
        if model.use_adaptive_grid else build_uniform_grid(cfg)
    )
    leaves_obj = list(grid.leaves)
    leaves = np.asarray(
        [[int(v.x), int(v.y), int(v.size)] for v in leaves_obj], dtype=np.int64
    )
    logical_prior = np.asarray(
        logical_canvas(prior, cfg.canvas_size, is_mask=True), dtype=np.uint8
    )
    region = _region_probs_from_logical_u8(logical_prior, leaves)
    rgb_square, crop_box = _square_edge_rgb_u8(image)
    prior_square = _square_zero_u8(prior)
    side = int(rgb_square.shape[0])

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

    canonical_patches = None
    if args.patch_backend == "pil":
        canonical_patches = (
            extract_canonical_patches(
                image, leaves_obj, cfg.canvas_size, int(saved.canonical_size)
            ).numpy().astype(np.float32, copy=False)
        )

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
        patch_boxes=_precompute_patch_boxes(leaves, side, cfg.canvas_size),
        input_signature=input_signature,
        cpu_prepare_ms=(time.perf_counter() - prepare_started) * 1000.0,
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


def gpu_extract_canonical_patches(
    native_rgb: torch.Tensor,
    patch_boxes: Sequence[np.ndarray],
    canonical_size: int,
) -> torch.Tensor:
    """Group equal crop shapes into batched bicubic GPU resizes."""
    batch = int(native_rgb.shape[0])
    tokens = int(patch_boxes[0].shape[0])
    output = torch.empty(
        (batch * tokens, 3, canonical_size, canonical_size),
        device=native_rgb.device,
        dtype=native_rgb.dtype,
    )
    groups: dict[tuple[int, int], list[tuple[int, int, int, int, int, int]]] = {}
    for batch_index, boxes in enumerate(patch_boxes):
        for token_index, row in enumerate(boxes):
            x1, y1, x2, y2 = map(int, row)
            groups.setdefault((y2 - y1, x2 - x1), []).append(
                (batch_index, token_index, x1, y1, x2, y2)
            )
    for rows in groups.values():
        crops = torch.stack(
            [native_rgb[b, :, y1:y2, x1:x2] for b, _, x1, y1, x2, y2 in rows]
        )
        resized = F.interpolate(
            crops, size=(canonical_size, canonical_size), mode="bicubic",
            align_corners=False, antialias=True,
        )
        indices = torch.tensor(
            [b * tokens + t for b, t, *_ in rows],
            device=native_rgb.device, dtype=torch.long,
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


def one_step_x0_batch_same_seed(model, glare, leaves, region_probs, seed: int):
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
    )
    return x0.reshape(batch, tokens, *glare.shape[2:])


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


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
        return [resized[index] for index in range(resized.shape[0])]

    result = []
    for index, sample in enumerate(samples):
        mode = sample.lens_interpolation
        kwargs = {"align_corners": False} if mode == "bilinear" else {}
        result.append(F.interpolate(
            lens_logical[index:index + 1], size=(sample.side, sample.side),
            mode=mode, **kwargs,
        )[0].float().clamp(0, 1))
    return result


def decode_from_rendered(
    model, clean_native, delta_native, native_glare, lens_native, eye_prior_native,
):
    """Exact current renderer policy after the combined linear PoU pass."""
    residual_candidate = native_glare.float() + delta_native.float()
    lens_native = lens_native.float().clamp(0, 1)
    signed_correction = base._blur_native_rgb(
        delta_native.float(), int(model.reconstruction_blur_kernel)
    )
    correction_magnitude = signed_correction.abs().mean(dim=0, keepdim=True)
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
        eye_mask = base._smoothstep01(base._blur_native_scalar(
            eye_mask, int(model.eye_core_feather_kernel)
        ).clamp(0, 1))
        iris_mask = base._smoothstep01(base._blur_native_scalar(
            iris_mask, int(model.eye_core_feather_kernel)
        ).clamp(0, 1))
    beta_inside = torch.full_like(lens_native, float(model.native_hf_skin))
    beta_inside = beta_inside * (1 - eye_mask) + float(model.native_hf_eye) * eye_mask
    beta_inside = beta_inside * (1 - iris_mask) + float(model.native_hf_iris) * iris_mask
    beta_inside = beta_inside * (
        1 - float(model.correction_hf_suppression) * reconstruction_need
    )
    rgb01 = ((native_glare.float() + 1) * .5).clamp(0, 1)
    luminance = rgb01.mean(dim=0, keepdim=True)
    chroma = rgb01.amax(dim=0, keepdim=True) - rgb01.amin(dim=0, keepdim=True)
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
        low_clean_alpha, native_hf_weight = model.learned_frequency_gate(
            native_glare.float(), clean_native.float(), delta_native.float(),
            correction_magnitude, eye_mask, iris_mask, lens_native,
            low_clean_alpha, native_hf_weight,
            correction_normalizer=float(model.reconstruction_high),
        )
    kernel = int(model.texture_lowpass_kernel)
    clean_low = base._blur_native_rgb(clean_native.float(), kernel)
    residual_low = base._blur_native_rgb(residual_candidate, kernel)
    native_low = base._blur_native_rgb(native_glare.float(), kernel)
    clean_high = clean_native.float() - clean_low
    residual_high = residual_candidate - residual_low
    native_high = native_glare.float() - native_low
    photographic_high = (1 - lens_native) * residual_high + lens_native * native_high
    rendered = (
        low_clean_alpha * clean_low + (1 - low_clean_alpha) * residual_low
        + native_hf_weight * photographic_high
        + (1 - native_hf_weight) * clean_high
    ).clamp(-1, 1)
    # Do not copy the old fast script's omission: outside support remains exact input.
    return ((1 - lens_native) * native_glare.float() + lens_native * rendered).clamp(-1, 1)


def render_decode_batch(model, x0, glare_patches, native_rgb, native_prior, lens, samples):
    leaves = [sample.leaves_cpu for sample in samples]
    sides = [sample.side for sample in samples]
    lenses = resize_lenses(lens, samples)
    if not model.use_frequency_renderer:
        rendered_delta = base.render_continuous_field_batch(
            x0.float() - glare_patches.float(), leaves, sides,
            model.canvas_size, model.halo_ratio,
        )
        predictions = []
        for index, sample in enumerate(samples):
            glare = native_rgb[index, :, :sample.side, :sample.side].float()
            rendered = (glare + rendered_delta[index]).clamp(-1, 1)
            predictions.append(
                ((1 - lenses[index]) * glare + lenses[index] * rendered).clamp(-1, 1)
            )
        return predictions

    # PoU rendering is linear: R(x0) and R(x0-glare) can share one 6-channel pass.
    fields = torch.cat([x0.float(), (x0 - glare_patches).float()], dim=2)
    rendered = base.render_continuous_field_batch(
        fields, leaves, sides, model.canvas_size, model.halo_ratio,
    )
    predictions = []
    for index, sample in enumerate(samples):
        side = sample.side
        predictions.append(decode_from_rendered(
            model, rendered[index][:3], rendered[index][3:],
            native_rgb[index, :, :side, :side], lenses[index],
            (
                native_prior[index, :, :side, :side]
                if model.use_semantic_conditioning else None
            ),
        ))
    return predictions


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


def tensors_to_pils(crops: Sequence[torch.Tensor], device: torch.device):
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
        torch.cuda.synchronize(device)
    else:
        host_arrays = [value.cpu() for value in device_arrays]
    return [Image.fromarray(value.numpy()) for value in host_arrays]


def infer_batch_once(samples, model, saved, device, args):
    pin = device.type == "cuda"
    rgb_cpu, prior_cpu, leaves_cpu, region_cpu, lens_cpu, patches_cpu = pack_batch_cpu(
        samples, pin_memory=pin
    )
    non_blocking = device.type == "cuda"
    rgb_u8 = rgb_cpu.to(device, non_blocking=non_blocking)
    prior_u8 = prior_cpu.to(device, non_blocking=non_blocking)
    leaves = leaves_cpu.to(device, non_blocking=non_blocking)
    region = region_cpu.to(device, non_blocking=non_blocking)
    lens = lens_cpu.to(device, non_blocking=non_blocking)
    # Match pil_to_tensor's operation order in the legacy single-image path.
    native_rgb = rgb_u8.float().div_(127.5).sub_(1.0)
    patches = (
        patches_cpu.to(device, non_blocking=non_blocking)
        if patches_cpu is not None else gpu_extract_canonical_patches(
            native_rgb, [sample.patch_boxes for sample in samples],
            int(saved.canonical_size),
        )
    )
    with torch.inference_mode(), autocast_context(device, args.amp):
        x0 = one_step_x0_batch_same_seed(model, patches, leaves, region, args.seed)
    with torch.inference_mode():
        if args.pmf_only:
            predictions = assemble_raw_pmf_batch(model, x0, samples)
        else:
            predictions = render_decode_batch(
                model, x0, patches, native_rgb, prior_u8, lens, samples
            )
    crops = []
    for prediction, sample in zip(predictions, samples):
        left, top, right, bottom = sample.crop_box
        crops.append(prediction[:, top:bottom, left:right])
    return tensors_to_pils(crops, device)


def infer_batch_adaptive(samples, model, saved, device, args):
    try:
        return infer_batch_once(samples, model, saved, device, args)
    except torch.cuda.OutOfMemoryError:
        if len(samples) == 1:
            raise
        torch.cuda.empty_cache()
        middle = len(samples) // 2
        print(f"[OOM] batch={len(samples)} -> retry {middle}+{len(samples)-middle}", flush=True)
        return (
            infer_batch_adaptive(samples[:middle], model, saved, device, args)
            + infer_batch_adaptive(samples[middle:], model, saved, device, args)
        )


def save_prediction(
    sample: PreparedSample,
    image: Image.Image,
    run_signature: dict,
    png_compress_level: int,
):
    save_started = time.perf_counter()
    output_path = sample.record.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".png":
        # Compression level changes file size only; decoded pixels stay lossless.
        image.save(output_path, compress_level=int(png_compress_level))
    else:
        image.save(output_path)
    write_prediction_manifest(
        sample.record.glare_path,
        output_path,
        run_signature,
        input_signature=sample.input_signature,
    )
    return (time.perf_counter() - save_started) * 1000.0


def make_output_path(input_root: Path, glare_path: Path, output_root_or_file: Path, output_name: str) -> Path:
    if input_root.is_file():
        return output_root_or_file
    rel_dir = glare_path.parent.relative_to(input_root)
    return output_root_or_file / rel_dir / output_name


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input', required=True,
                   help='Input glare image path OR a directory to recursively scan.')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True,
                   help='Output image path (single-image mode) OR output root directory (batch mode).')

    p.add_argument('--pattern', default='glare.png',
                   help='Filename pattern used when --input is a directory.')
    p.add_argument('--output_name', default='pred.png',
                   help='Output filename used in batch mode for each sample folder.')
    p.add_argument('--skip_existing', action='store_true',
                   help='Skip samples whose output file already exists.')

    p.add_argument('--use_existing_geometry', action='store_true', default=True,
                   help='Reuse eye_prior.png and face_meta.json if present.')
    p.add_argument('--force_redetect', action='store_true',
                   help='Ignore existing eye_prior.png / face_meta.json and redetect geometry.')
    p.add_argument('--strict_geometry_cache', action='store_true',
                   help='Require current v2 geometry plus matching source SHA256; otherwise compatible v1 caches are accepted.')
    p.add_argument('--no_detection_fallback', action='store_true',
                   help='If cached geometry is missing/invalid, fail instead of running detector.')

    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--backend', choices=('auto', 'tasks', 'legacy'), default='auto')
    p.add_argument('--model_asset_path', default=str(Path(__file__).parent/'assets'/'face_landmarker.task'))
    p.add_argument('--min_detection_confidence', type=float, default=.5)
    p.add_argument('--min_presence_confidence', type=float, default=.5)
    p.add_argument('--min_tracking_confidence', type=float, default=.5)
    p.add_argument('--geometry_detect_max_side',type=int,default=768,
                   help='Downscale only online landmark detection to this maximum side; 0 preserves full-resolution detection.')

    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pmf_only', action='store_true',
                   help='save raw PMF x0 after linear PoU assembly; bypass all post-render decoding')
    p.add_argument('--disable_learned_renderer', action='store_true',
                   help='keep frequency decomposition but use handcrafted gates only')
    p.add_argument('--disable_frequency_renderer', action='store_true',
                   help='bypass the entire low/high-frequency renderer')
    p.add_argument('--max_items', type=int, default=0,
                   help='For debugging: limit number of images in batch mode. 0 means no limit.')
    speed=p.add_argument_group('high-throughput inference')
    speed.add_argument('--batch_size',type=int,default=8,
                       help='Batched PMF/renderer size; CUDA OOM automatically splits the batch.')
    speed.add_argument('--preprocess_workers',type=int,
                       default=max(1,min(4,os.cpu_count() or 1)),
                       help='Parallel image/geometry/grid preparation workers.')
    speed.add_argument('--prefetch_batches',type=int,default=3,
                       help='Number of batches prepared ahead of GPU execution.')
    speed.add_argument('--save_workers',type=int,default=4,
                       help='Asynchronous PNG/manifest writer threads.')
    speed.add_argument('--png_compress_level',type=int,choices=range(0,10),default=1,
                       metavar='0..9',
                       help='PNG compression level; pixels remain lossless, lower is faster/larger.')
    speed.add_argument('--patch_backend',choices=('gpu','pil'),default='gpu',
                       help='gpu is fast bicubic+antialias extraction; pil exactly matches the legacy preprocessing.')
    speed.add_argument('--amp',choices=('off','bf16','fp16'),default='bf16',
                       help='Mixed precision for PMF only; renderer always remains FP32.')
    speed.add_argument('--grid_solver',choices=('fast','verify','dp'),default='fast',
                       help='Exact heap grid solver, regression verifier, or legacy dynamic programming.')
    speed.add_argument('--progress_interval',type=int,default=100,
                       help='Print one progress line after roughly this many processed images; 0 prints only the final summary.')
    speed.add_argument('--verbose_warnings',action='store_true',
                       help='Print every per-image geometry warning instead of only a final grouped summary.')
    lens=p.add_argument_group('renderer lens-support overrides')
    lens.add_argument('--renderer_lens_expand_x',type=float,default=None)
    lens.add_argument('--renderer_lens_expand_y',type=float,default=None)
    lens.add_argument('--renderer_lens_min_height_ratio',type=float,default=None)
    lens.add_argument('--renderer_lens_feather',type=float,default=None)
    return p.parse_args()


def main():
    args = parse_args()
    reset_warning_summary()
    try:
        config.validate_inference_args(args)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"inference configuration error: {error}") from error
    if args.force_redetect:
        args.use_existing_geometry = False

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

    model, saved = load_model(args.checkpoint, device)
    renderer_overrides=apply_renderer_geometry_overrides(model,args)
    if renderer_overrides:
        print(f"[INFO] Renderer lens overrides: {renderer_overrides}",flush=True)
    if args.disable_learned_renderer:
        model.use_learned_frequency_renderer = False
    if args.disable_frequency_renderer:
        model.use_frequency_renderer = False
    cfg = config.make_grid_config(saved)
    run_signature=prediction_run_signature(Path(args.checkpoint),model,args)

    glare_files = discover_inputs(input_path, args.pattern, args.max_items)

    is_batch = input_path.is_dir()
    total = len(glare_files)
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
        f"[FAST] batch={args.batch_size} preprocess_workers={args.preprocess_workers} "
        f"prefetch_batches={args.prefetch_batches} patch={args.patch_backend} "
        f"amp={args.amp} grid_solver={args.grid_solver} "
        f"geometry_max_side={args.geometry_detect_max_side} "
        f"png_level={args.png_compress_level}",flush=True,
    )

    records=[]
    for index,glare_path in enumerate(glare_files,start=1):
        out_path=make_output_path(input_path,glare_path,output_path,args.output_name)
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

    geometry_builders=ThreadLocalGeometryBuilders(args)
    preprocess_executor=ThreadPoolExecutor(max_workers=args.preprocess_workers)
    save_executor=ThreadPoolExecutor(max_workers=args.save_workers)
    save_futures=[]
    cpu_prepare_total_ms=0.0
    cpu_prepare_images=0
    pipeline_total_ms=0.0
    pipeline_images=0
    save_total_ms=0.0
    save_images=0
    last_progress=0
    window_size=max(args.batch_size,args.batch_size*args.prefetch_batches)
    windows=[records[i:i+window_size] for i in range(0,len(records),window_size)]

    def submit_window(window):
        return [preprocess_executor.submit(
            prepare_one,record,args,saved,cfg,model,geometry_builders
        ) for record in window]

    started=time.perf_counter()
    try:
        current=submit_window(windows[0]) if windows else []
        for window_index in range(len(windows)):
            prepared=[]
            for future,record in zip(current,windows[window_index]):
                try:
                    sample=future.result()
                    prepared.append(sample)
                    cpu_prepare_total_ms+=float(sample.cpu_prepare_ms)
                    cpu_prepare_images+=1
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
                try:
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    pipeline_started=time.perf_counter()
                    images=infer_batch_adaptive(batch,model,saved,device,args)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    pipeline_total_ms+=(time.perf_counter()-pipeline_started)*1000.0
                    pipeline_images+=len(batch)
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
        for future in save_futures:
            try:
                save_total_ms+=float(future.result())
                save_images+=1
            except Exception as error:
                failed+=1; done=max(0,done-1)
                print(f"[FAIL] asynchronous save: {error}",flush=True)
    finally:
        preprocess_executor.shutdown(wait=True,cancel_futures=True)
        geometry_builders.close_all()
        save_executor.shutdown(wait=True,cancel_futures=True)

    print("=" * 72, flush=True)
    elapsed=time.perf_counter()-started
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
    print(
        f"[TIMING] average_async_save_ms_per_image={save_mean_ms:.3f}",
        flush=True,
    )
    print(
        "[TIMING] component latencies overlap; "
        "average_end_to_end_ms_per_image is the authoritative wall-clock average.",
        flush=True,
    )
    print_warning_summary()
    print("=" * 72, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
