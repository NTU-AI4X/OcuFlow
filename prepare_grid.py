#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from models.grid_cache import (
    CACHE_VERSION, grid_config_hash, grid_config_payload, load_record,
    make_sample_record, sample_cache_path, sample_input_signature, save_record,
    validate_record,
)
from models.spatial import AdaptiveGridConfig, build_adaptive_grid, load_metadata


def make_config(args: argparse.Namespace) -> AdaptiveGridConfig:
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


def sample_folders(root: Path, split: str) -> list[Path]:
    d = root / split
    if not d.is_dir():
        return []
    return [x for x in sorted(d.iterdir()) if x.is_dir() and (x / "glare.png").is_file() and (x / "clean.png").is_file()]


def build_one(job):
    data_root_s, cache_root_s, split, name, cfg_payload = job
    data_root, cache_root = Path(data_root_s), Path(cache_root_s)
    cfg = AdaptiveGridConfig(**cfg_payload)
    folder = data_root / split / name
    glare_path = folder / "glare.png"
    prior_path = folder / "eye_prior.png"
    meta_path = folder / "face_meta.json"
    prior_opt = prior_path if prior_path.is_file() else None
    meta_opt = meta_path if meta_path.is_file() else None
    with Image.open(glare_path) as im:
        original_size = tuple(map(int, im.size))
    signature = sample_input_signature(glare_path, prior_opt, meta_opt, original_size=original_size)
    path = sample_cache_path(cache_root, split, name)
    if path.is_file():
        try:
            rec = load_record(path)
            validation_errors = validate_record(
                rec, split=split, sample_name=name, input_signature=signature, cfg=cfg
            )
            if not validation_errors:
                g = rec["grid"]
                return (name, g["patch4_count"], g["patch8_count"], g["patch16_count"],
                        g["patch32_count"], g.get("anchor_token_count", 0), g.get("lens_token_count", 0),
                        g.get("constraint_mode", "cache"), True)
        except Exception:
            pass
    if prior_opt:
        with Image.open(prior_opt) as im:
            prior = im.convert("L").copy()
    else:
        prior = Image.new("L", original_size, 0)
    grid = build_adaptive_grid(prior, load_metadata(meta_opt), original_size, cfg)
    rec = make_sample_record(split=split, sample_name=name, original_size=original_size,
                             input_signature=signature, cfg=cfg, grid=grid)
    save_record(path, rec)
    return (name, grid.patch4_count, grid.patch8_count, grid.patch16_count,
            grid.patch32_count, grid.anchor_token_count, grid.lens_token_count, grid.constraint_mode, False)


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute exact-budget adaptive grids with eye anchors and lens-band p4/p8 capacity.")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "test_sys", "test_real"])
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    p.add_argument("--canvas_size", type=int, default=256)
    p.add_argument("--base_patch_size", type=int, default=16)
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
    args = p.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.hard_anchor_radius < 0:
        raise ValueError("--hard_anchor_radius must be >= 0")
    data_root = Path(args.data_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = make_config(args)
    jobs = []
    split_counts = {}
    payload = grid_config_payload(cfg)
    for split in args.splits:
        folders = sample_folders(data_root, split)
        if not folders:
            print(f"[grid-cache] skip split={split}: missing/empty", flush=True)
            continue
        split_counts[split] = len(folders)
        jobs.extend((str(data_root), str(output), split, f.name, payload) for f in folders)
    if not jobs:
        raise RuntimeError("no paired samples found")
    totals = [0, 0, 0, 0]
    anchor_tokens_total = 0
    lens_tokens_total = 0
    modes = {}
    reused = 0
    done = 0
    def consume(res):
        nonlocal reused, done, anchor_tokens_total, lens_tokens_total
        _, a, b, c, d, anchor_tokens, lens_tokens, mode, was_reused = res
        vals = [a, b, c, d]
        for i, v in enumerate(vals): totals[i] += int(v)
        modes[mode] = modes.get(mode, 0) + 1
        anchor_tokens_total += int(anchor_tokens)
        lens_tokens_total += int(lens_tokens)
        reused += int(was_reused)
        done += 1
        if done % 250 == 0 or done == len(jobs):
            print(f"[grid-cache] {done}/{len(jobs)}", flush=True)
    if args.workers <= 1:
        for res in map(build_one, jobs): consume(res)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(build_one, j) for j in jobs]
            for f in as_completed(futures): consume(f.result())
    n = float(len(jobs))
    manifest = {
        "cache_version": CACHE_VERSION,
        "type": "v29.4_lens_band_hard_anchor_token_density_exact_budget_quadtree",
        "grid_config": payload,
        "grid_config_hash": grid_config_hash(cfg),
        "splits": split_counts,
        "samples": len(jobs),
        "reused_records": reused,
        "constraint_modes": modes,
        "average_patch_counts": {k: totals[i] / n for i, k in enumerate(("p4", "p8", "p16", "p32"))},
        "average_anchor_token_count": anchor_tokens_total / n,
        "average_lens_token_count": lens_tokens_total / n,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[grid-cache] complete samples={len(jobs)} avg p4/p8/p16/p32="
          f"{totals[0]/n:.1f}/{totals[1]/n:.1f}/{totals[2]/n:.1f}/{totals[3]/n:.1f} "
          f"avg_anchor_tokens={anchor_tokens_total/n:.1f} avg_lens_tokens={lens_tokens_total/n:.1f} modes={modes}", flush=True)
    if anchor_tokens_total == 0:
        print(
            "[grid-cache][WARNING] no structural hard anchors were found in any sample. "
            "Check/regenerate face_meta.json with prepare_geometry.py; otherwise geometry anchors cannot activate.",
            flush=True,
        )
    if lens_tokens_total == 0:
        print(
            "[grid-cache][WARNING] no lens-band tokens were found in any sample. "
            "Check landmark_groups_normalized eye contours in face_meta.json.",
            flush=True,
        )


if __name__ == "__main__":
    main()
