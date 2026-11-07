from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from PIL import Image

from .spatial import AdaptiveGridConfig, AdaptiveLeaf, GridBuildResult, validate_grid

CACHE_VERSION = 4


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def grid_config_payload(cfg: AdaptiveGridConfig) -> Dict[str, object]:
    return asdict(cfg)


def grid_config_hash(cfg: AdaptiveGridConfig) -> str:
    return hashlib.sha256(_stable_json(grid_config_payload(cfg)).encode("utf-8")).hexdigest()


def _file_signature(path: Optional[Path]) -> Dict[str, object]:
    if path is None or not path.is_file():
        return {"exists": False}
    st = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"exists": True, "size": int(st.st_size), "sha256": h.hexdigest()}


def sample_input_signature(
    glare_path: Path,
    prior_path: Optional[Path],
    meta_path: Optional[Path],
    *,
    original_size: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    # Glare pixels do not affect the cached geometry.  Only image size and
    # inference-time geometry files are hashed.
    if original_size is None:
        with Image.open(glare_path) as image:
            original_size = image.size
    return {
        "original_size": [int(original_size[0]), int(original_size[1])],
        "prior": _file_signature(prior_path),
        "meta": _file_signature(meta_path),
    }


def sample_cache_path(root: Path, split: str, sample_name: str) -> Path:
    return Path(root) / str(split) / f"{sample_name}.json"


def grid_to_record(grid: GridBuildResult) -> Dict[str, object]:
    return {
        "leaves": [[int(v.x), int(v.y), int(v.size)] for v in grid.leaves],
        "geometry_success": bool(grid.geometry_success),
        "geometry_source": str(grid.geometry_source),
        "constraint_mode": str(grid.constraint_mode),
        "dp_total_cost": float(grid.dp_total_cost),
        "patch4_count": int(grid.patch4_count),
        "patch8_count": int(grid.patch8_count),
        "patch16_count": int(grid.patch16_count),
        "patch32_count": int(grid.patch32_count),
        "correction_token_count": int(grid.correction_token_count),
        "anchor_token_count": int(grid.anchor_token_count),
        "lens_token_count": int(grid.lens_token_count),
        "hard_constraints_relaxed": bool(grid.hard_constraints_relaxed),
    }


def make_sample_record(*, split: str, sample_name: str, original_size: Sequence[int],
                       input_signature: Mapping[str, object], cfg: AdaptiveGridConfig,
                       grid: GridBuildResult) -> Dict[str, object]:
    return {
        "cache_version": CACHE_VERSION,
        "split": str(split),
        "sample_name": str(sample_name),
        "original_size": [int(original_size[0]), int(original_size[1])],
        "grid_config_hash": grid_config_hash(cfg),
        "input_signature": dict(input_signature),
        "grid": grid_to_record(grid),
    }


def save_record(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_stable_json(record), encoding="utf-8")
    tmp.replace(path)


def load_record(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_record(record: Mapping[str, object], *, split: str, sample_name: str,
                    input_signature: Mapping[str, object], cfg: AdaptiveGridConfig) -> list[str]:
    reasons: list[str] = []
    version = int(record.get("cache_version", -1))
    # V29.4 changes grid semantics again: a geometry-derived spectacle lens band
    # receives dedicated p4/p8 capacity. Reject older caches so stale leaves cannot
    # silently bypass the high-frequency glare fix.
    if version != CACHE_VERSION:
        reasons.append("cache_version")
    if str(record.get("split", "")) != str(split):
        reasons.append("split")
    if str(record.get("sample_name", "")) != str(sample_name):
        reasons.append("sample_name")
    if version == CACHE_VERSION and str(record.get("grid_config_hash", "")) != grid_config_hash(cfg):
        reasons.append("grid_config")
    if record.get("input_signature") != dict(input_signature):
        reasons.append("input_signature")
    value = record.get("grid")
    if not isinstance(value, Mapping) or not isinstance(value.get("leaves"), list):
        reasons.append("grid")
        return reasons
    try:
        leaves = [AdaptiveLeaf(int(x), int(y), int(size)) for x, y, size in value["leaves"]]
        validate_grid(leaves, cfg)
    except Exception:
        reasons.append("grid.leaves_invalid")
    return reasons


def grid_from_record(record: Mapping[str, object], cfg: AdaptiveGridConfig) -> GridBuildResult:
    value = record["grid"]
    leaves = [AdaptiveLeaf(int(x), int(y), int(size), "cache") for x, y, size in value["leaves"]]
    validate_grid(leaves, cfg)
    return GridBuildResult(
        leaves=leaves,
        geometry_success=bool(value.get("geometry_success", False)),
        geometry_source=str(value.get("geometry_source", "cache")),
        target_token_count=len(leaves),
        dp_total_cost=float(value.get("dp_total_cost", 0.0)),
        patch4_count=int(value.get("patch4_count", sum(v.size == cfg.patch4 for v in leaves))),
        patch8_count=int(value.get("patch8_count", sum(v.size == cfg.patch8 for v in leaves))),
        patch16_count=int(value.get("patch16_count", sum(v.size == cfg.patch16 for v in leaves))),
        patch32_count=int(value.get("patch32_count", sum(v.size == cfg.patch32 for v in leaves))),
        eye_token_count=int(value.get("correction_token_count", 0)),
        hard_constraints_relaxed=bool(value.get("hard_constraints_relaxed", False)),
        internal_splits=(len(leaves) - (cfg.canvas_size // cfg.patch32) ** 2) // 3,
        correction_token_count=int(value.get("correction_token_count", 0)),
        anchor_token_count=int(value.get("anchor_token_count", 0)),
        lens_token_count=int(value.get("lens_token_count", 0)),
        constraint_mode=str(value.get("constraint_mode", "cache")),
    )
