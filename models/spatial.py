from __future__ import annotations

import json
import hashlib
import math
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

LANDMARK_SCHEMA = "lrda_8ch_v1"
GEOMETRY_VERSION = "landmark_detail_adapter_v2"


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


def _head_mask(
    metadata: Mapping[str, object], original_size: Tuple[int, int], cfg: AdaptiveGridConfig
) -> Tuple[np.ndarray, str]:
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
    """Estimate a glasses/lens support from eye contours only.

    This is intentionally geometric rather than a glare detector.  It is used to
    allocate p4/p8 capacity across the whole likely lens region so thin window/grid
    reflections do not land in coarse p16/p32 leaves.  If detailed landmarks are
    unavailable, the existing periocular prior is used as a conservative fallback.
    """
    groups = metadata.get("landmark_groups_normalized", {})
    if not isinstance(groups, Mapping):
        return fallback_periocular.astype(np.uint8, copy=True)
    out = np.zeros_like(fallback_periocular, dtype=np.uint8)
    made = False
    for key in ("left_eye_contour", "right_eye_contour"):
        contour = groups.get(key)
        if not isinstance(contour, (list, tuple)) or len(contour) < 3:
            continue
        pts = []
        for point in contour:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                pts.append(_normalized_point_to_logical(point, original_size, cfg.canvas_size))
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
        # Eye contours are very flat; real spectacle lenses are not.  Keep a
        # conservative minimum aspect ratio so the band covers reflection lines
        # above/below the visible eye without expanding to the whole face.
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
    """Build a soft, rotated rounded lens support for the *renderer only*.

    This deliberately does not participate in the cached quadtree solve.  V29.4.3.1
    keeps cache-version 4 / V29.4 leaves fully compatible and corrects only the
    inference reconstruction policy.  Eye contours are converted to the logical
    square, aligned by their principal axis, expanded to spectacle-lens scale, then
    rasterized as a superellipse with a narrow feather.  No clean image or
    clean-glare difference is consulted.

    Returns a float32 HxW map in [0,1], or ``None`` when eye contours are absent.
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

    groups = metadata.get("landmark_groups_normalized", {})
    if not isinstance(groups, Mapping):
        return None

    yy, xx = np.mgrid[0:canvas, 0:canvas].astype(np.float32)
    out = np.zeros((canvas, canvas), dtype=np.float32)
    made = False

    for key in ("left_eye_contour", "right_eye_contour"):
        contour = groups.get(key)
        if not isinstance(contour, (list, tuple)) or len(contour) < 3:
            continue
        pts = []
        for point in contour:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                pts.append(_normalized_point_to_logical(point, original_size, canvas))
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
        # Keep a stable left-to-right orientation; sign is otherwise arbitrary.
        if axis_x[0] < 0:
            axis_x = -axis_x
        axis_y = np.array([-axis_x[1], axis_x[0]], dtype=np.float32)

        proj_x = centered @ axis_x
        proj_y = centered @ axis_y
        x_min, x_max = float(proj_x.min()), float(proj_x.max())
        y_min, y_max = float(proj_y.min()), float(proj_y.max())
        span_x = max(1.0, x_max - x_min)
        span_y = max(1.0, y_max - y_min)

        # Recenter on the projected bounding box instead of the point mean so
        # asymmetric eyelid samples do not bias the spectacle support.
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
        radius = (np.abs(u) ** power + np.abs(v) ** power) ** (1.0 / power)

        # Approximate signed distance in logical pixels.  The superellipse is
        # exactly 1 inside, then fades only across a narrow renderer feather.
        distance = (radius - 1.0) * min(half_w, half_h)
        if float(feather) <= 0:
            alpha = (distance <= 0).astype(np.float32)
        else:
            t = np.clip(distance / float(feather), 0.0, 1.0)
            smooth = t * t * (3.0 - 2.0 * t)
            alpha = np.where(distance <= 0.0, 1.0, 1.0 - smooth).astype(np.float32)
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

    groups = metadata.get("landmark_groups_normalized", {})
    if not isinstance(groups, Mapping):
        groups = {}
    hard_anchor = np.zeros_like(prior, dtype=np.uint8)
    anchor_points: List[Tuple[float, float]] = []
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
                _normalized_point_to_logical(point, original_size, cfg.canvas_size)
            )
    # Sparse samples from upper/lower eye contour preserve eyelid geometry
    # without forcing the complete eyelid band to the finest scale.
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
