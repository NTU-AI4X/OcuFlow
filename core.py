#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from models.grid_cache import (
    grid_from_record, load_record, sample_cache_path, sample_input_signature,
    validate_record,
)
from models.pmf import pmfDiT_B_16
from models.spatial import (
    GEOMETRY_VERSION, LANDMARK_SCHEMA, AdaptiveGridConfig, AdaptiveLeaf,
    build_geometry_maps, build_renderer_lens_alpha, build_uniform_grid,
    extract_canonical_patches, logical_canvas, load_metadata, pil_to_tensor,
    source_signature, square_edge_canvas, square_zero_canvas, tensor_to_pil,
)


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


def flip_leaves(leaves: Sequence[AdaptiveLeaf], canvas_size: int) -> list[AdaptiveLeaf]:
    out = [AdaptiveLeaf(canvas_size - v.x - v.size, v.y, v.size, "flipped_cache") for v in leaves]
    out.sort(key=lambda v: (v.y, v.x, v.size))
    return out


def region_probs_from_prior(prior: Image.Image, leaves: Sequence[AdaptiveLeaf], canvas_size: int) -> torch.Tensor:
    arr = np.asarray(logical_canvas(prior, canvas_size, is_mask=True), dtype=np.uint8)
    bg = (arr < 64).astype(np.float32)
    peri = ((arr >= 64) & (arr < 128)).astype(np.float32)
    eye = ((arr >= 128) & (arr < 192)).astype(np.float32)
    iris = (arr >= 192).astype(np.float32)
    maps = np.stack([bg, peri, eye, iris], axis=-1)
    rows = []
    for leaf in leaves:
        patch = maps[leaf.y:leaf.y + leaf.size, leaf.x:leaf.x + leaf.size]
        p = patch.mean(axis=(0, 1)) if patch.size else np.array([1, 0, 0, 0], dtype=np.float32)
        total = float(p.sum())
        if total <= 0:
            p = np.array([1, 0, 0, 0], dtype=np.float32)
        else:
            p = p / total
        rows.append(p)
    return torch.from_numpy(np.stack(rows, axis=0)).float()



class BalancedDomainBatchSampler(Sampler[list[int]]):
    """Yield fixed-ratio synthetic/real batches with per-domain reshuffling.

    One logical epoch is long enough to visit every sample from the larger
    domain. The smaller domain is independently reshuffled whenever exhausted,
    which performs true oversampling without cloning dataset records.
    """

    def __init__(self, dataset, batch_size: int, real_fraction: float = 0.5,
                 synthetic_prefix: str = "syn_", real_prefix: str = "pub_",
                 seed: int = 2026) -> None:
        self.batch_size = int(batch_size)
        self.real_fraction = float(real_fraction)
        self.synthetic_prefix = str(synthetic_prefix)
        self.real_prefix = str(real_prefix)
        self.seed = int(seed)
        self.epoch = 0
        self.batch_cursor = 0
        if self.batch_size < 2:
            raise ValueError("balanced domain batches require batch_size >= 2")
        if not 0.0 < self.real_fraction < 1.0:
            raise ValueError("real_fraction must be strictly between 0 and 1")

        self.real_per_batch = max(1, min(
            self.batch_size - 1, int(round(self.batch_size * self.real_fraction))
        ))
        self.synthetic_per_batch = self.batch_size - self.real_per_batch
        self.synthetic_indices = []
        self.real_indices = []
        unknown = []
        for index, record in enumerate(dataset.samples):
            name = str(record.get("name", ""))
            if name.startswith(self.synthetic_prefix):
                self.synthetic_indices.append(index)
            elif name.startswith(self.real_prefix):
                self.real_indices.append(index)
            else:
                unknown.append(name)
        if unknown:
            raise ValueError(
                f"balanced sampler found {len(unknown)} samples with unknown domain prefix: "
                f"{unknown[:10]}"
            )
        if not self.synthetic_indices or not self.real_indices:
            raise ValueError(
                f"balanced sampler requires both domains, got "
                f"synthetic={len(self.synthetic_indices)}, real={len(self.real_indices)}"
            )
        self.num_batches = max(
            math.ceil(len(self.synthetic_indices) / self.synthetic_per_batch),
            math.ceil(len(self.real_indices) / self.real_per_batch),
        )

    @staticmethod
    def _cycle(indices: list[int], generator: torch.Generator):
        while True:
            order = torch.randperm(len(indices), generator=generator).tolist()
            for position in order:
                yield indices[position]

    @staticmethod
    def _augmentation_seed(seed: int, epoch: int, batch: int, slot: int) -> int:
        # Stable integer mixing: unlike Python hash(), this is process-independent.
        value = int(seed) & ((1 << 64) - 1)
        for item in (epoch, batch, slot):
            value ^= (int(item) + 0x9E3779B97F4A7C15 + (value << 6) + (value >> 2))
            value &= (1 << 64) - 1
        return value & ((1 << 63) - 1)

    def set_position(self, epoch: int, batch_cursor: int) -> None:
        epoch = int(epoch); batch_cursor = int(batch_cursor)
        if epoch < 0 or not 0 <= batch_cursor < self.num_batches:
            raise ValueError(
                f"invalid sampler position epoch={epoch} batch_cursor={batch_cursor}; "
                f"num_batches={self.num_batches}"
            )
        self.epoch = epoch
        self.batch_cursor = batch_cursor

    def state_dict(self) -> dict:
        return {
            "kind": type(self).__name__, "seed": self.seed,
            "epoch": self.epoch, "batch_cursor": self.batch_cursor,
            "num_batches": self.num_batches,
        }

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        synthetic = self._cycle(self.synthetic_indices, generator)
        real = self._cycle(self.real_indices, generator)
        for batch_index in range(self.num_batches):
            batch = (
                [next(synthetic) for _ in range(self.synthetic_per_batch)]
                + [next(real) for _ in range(self.real_per_batch)]
            )
            order = torch.randperm(len(batch), generator=generator).tolist()
            if batch_index < self.batch_cursor:
                continue
            yield [
                (
                    batch[position],
                    self._augmentation_seed(self.seed, self.epoch, batch_index, slot),
                )
                for slot, position in enumerate(order)
            ]

    def __len__(self) -> int:
        return self.num_batches


class ResumableRandomBatchSampler(Sampler[list[int]]):
    """Deterministic natural-frequency batches with an explicit resume cursor."""

    def __init__(self, dataset, batch_size: int, *, seed: int, drop_last: bool = True) -> None:
        self.dataset_size = len(dataset)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.num_batches = (
            self.dataset_size // self.batch_size if self.drop_last
            else math.ceil(self.dataset_size / self.batch_size)
        )
        if self.num_batches <= 0:
            raise ValueError("dataset is too small for the requested natural batch")
        self.epoch = 0
        self.batch_cursor = 0

    def set_position(self, epoch: int, batch_cursor: int) -> None:
        epoch = int(epoch); batch_cursor = int(batch_cursor)
        if epoch < 0 or not 0 <= batch_cursor < self.num_batches:
            raise ValueError(
                f"invalid sampler position epoch={epoch} batch_cursor={batch_cursor}; "
                f"num_batches={self.num_batches}"
            )
        self.epoch = epoch
        self.batch_cursor = batch_cursor

    def state_dict(self) -> dict:
        return {
            "kind": type(self).__name__, "seed": self.seed,
            "epoch": self.epoch, "batch_cursor": self.batch_cursor,
            "num_batches": self.num_batches,
        }

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.dataset_size, generator=generator).tolist()
        for batch_index in range(self.num_batches):
            start = batch_index * self.batch_size
            batch = order[start:start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                break
            if batch_index < self.batch_cursor:
                continue
            yield [
                (
                    index,
                    BalancedDomainBatchSampler._augmentation_seed(
                        self.seed, self.epoch, batch_index, slot
                    ),
                )
                for slot, index in enumerate(batch)
            ]

    def __len__(self) -> int:
        return self.num_batches


class ReglassDataset(Dataset):
    def __init__(self, root: str, split: str, cache_root: str, cfg: AdaptiveGridConfig,
                 canonical_size: int = 24, augment: bool = False, identity_prob: float = 0.0,
                 sample_names: Sequence[str] = (), exclude_names: Sequence[str] = (),
                 need_native: bool = False, need_mask: bool = False,
                 uniform_grid: bool = False) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.cache_root = Path(cache_root)
        self.cfg = cfg
        self.canonical_size = int(canonical_size)
        if self.canonical_size <= 0:
            raise ValueError("canonical_size must be positive")
        self.augment = bool(augment)
        self.identity_prob = float(identity_prob) if augment else 0.0
        # Full-native tensors are needed only for epoch visualization / sample mode.
        # Training runs purely in adaptive-token space and skips this expensive CPU
        # conversion and GPU transfer path entirely.
        self.need_native = bool(need_native)
        self.need_mask = bool(need_mask)
        if self.need_mask and not self.need_native:
            raise ValueError("need_mask=True requires need_native=True")
        self.uniform_grid = bool(uniform_grid)
        uniform = build_uniform_grid(cfg) if self.uniform_grid else None
        allowed = set(sample_names)
        excluded = set(exclude_names)
        split_dir = self.root / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(split_dir)
        self.samples = []
        modes: Dict[str, int] = {}
        counts = {4: 0, 8: 0, 16: 0, 32: 0}
        anchor_tokens_total = 0
        lens_tokens_total = 0
        for folder in sorted(split_dir.iterdir()):
            if (
                not folder.is_dir() or folder.name in excluded
                or (allowed and folder.name not in allowed)
            ):
                continue
            clean = folder / "clean.png"
            glare = folder / "glare.png"
            if not clean.is_file() or not glare.is_file():
                continue
            prior = folder / "eye_prior.png"
            meta = folder / "face_meta.json"
            mask = folder / "mask.png"
            prior_opt = prior if prior.is_file() else None
            meta_opt = meta if meta.is_file() else None
            with Image.open(glare) as im:
                original_size = tuple(map(int, im.size))
            if self.need_native:
                metadata = load_metadata(meta_opt)
                geometry_errors = []
                if not prior_opt:
                    geometry_errors.append("missing eye_prior.png")
                if not meta_opt:
                    geometry_errors.append("missing face_meta.json")
                else:
                    if metadata.get("geometry_version") != GEOMETRY_VERSION:
                        geometry_errors.append("geometry_version")
                    if metadata.get("landmark_schema") != LANDMARK_SCHEMA:
                        geometry_errors.append("landmark_schema")
                    if [metadata.get("source_width"), metadata.get("source_height")] != list(original_size):
                        geometry_errors.append("source_size")
                    if metadata.get("source_signature") != source_signature(glare):
                        geometry_errors.append("source_signature")
                if geometry_errors:
                    raise RuntimeError(
                        f"invalid native geometry for {folder.name}: {geometry_errors}. "
                        "Run prepare_geometry.py and prepare_grid.py before three-stage training."
                    )
            if uniform is not None:
                grid = uniform
            else:
                signature = sample_input_signature(glare, prior_opt, meta_opt, original_size=original_size)
                cache_path = sample_cache_path(self.cache_root, self.split, folder.name)
                if not cache_path.is_file():
                    raise FileNotFoundError(
                        f"missing non-uniform grid cache: {cache_path}. Run prepare_grid.py first."
                    )
                rec = load_record(cache_path)
                reasons = validate_record(rec, split=self.split, sample_name=folder.name,
                                          input_signature=signature, cfg=self.cfg)
                if reasons:
                    raise RuntimeError(
                        f"stale/invalid grid cache for {folder.name}: {reasons}. "
                        "V29.4 requires cache_version=4 because lens-band constraints now affect the solver; "
                        "regenerate with this project's prepare_grid.py."
                    )
                grid = grid_from_record(rec, self.cfg)
            if len(grid.leaves) != self.cfg.target_tokens:
                raise RuntimeError(f"{folder.name}: expected {self.cfg.target_tokens} leaves, got {len(grid.leaves)}")
            modes[grid.constraint_mode] = modes.get(grid.constraint_mode, 0) + 1
            anchor_tokens_total += int(grid.anchor_token_count)
            lens_tokens_total += int(getattr(grid, "lens_token_count", 0))
            for leaf in grid.leaves:
                counts[leaf.size] = counts.get(leaf.size, 0) + 1
            self.samples.append({
                "name": folder.name, "clean": clean, "glare": glare,
                "prior": prior_opt, "meta": meta_opt,
                "mask": mask if mask.is_file() else None,
                "original_size": original_size, "grid": grid,
            })
        if not self.samples:
            raise RuntimeError(f"no paired samples under {split_dir}")
        n = float(len(self.samples))
        print(
            f"[dataset] split={self.split} samples={len(self.samples)} exact_tokens={cfg.target_tokens} "
            f"canonical={self.canonical_size} grid_base={cfg.base_patch_size} "
            f"avg_p4/8/16/32={counts.get(4,0)/n:.1f}/{counts.get(8,0)/n:.1f}/"
            f"{counts.get(16,0)/n:.1f}/{counts.get(32,0)/n:.1f} "
            f"avg_anchor_tokens={anchor_tokens_total/n:.1f} avg_lens_tokens={lens_tokens_total/n:.1f} "
            f"modes={modes} cache={'BYPASSED' if self.uniform_grid else self.cache_root}",
            flush=True,
        )
        if anchor_tokens_total == 0 and not self.uniform_grid:
            print(
                f"[dataset][WARNING] split={self.split} has zero cached structural-anchor tokens. "
                "Regenerate face geometry/cache if landmark_groups_normalized should be available.",
                flush=True,
            )
        if lens_tokens_total == 0 and not self.uniform_grid:
            print(
                f"[dataset][WARNING] split={self.split} has zero cached lens-band tokens. "
                "Check face_meta.json eye contours and regenerate the V29.4 cache.",
                flush=True,
            )
        self.name_to_index = {str(s["name"]): i for i, s in enumerate(self.samples)}

    def __len__(self) -> int:
        return len(self.samples)

    def index_for_name(self, name: str) -> int:
        if name not in self.name_to_index:
            raise KeyError(f"sample {name!r} not found in split={self.split}")
        return self.name_to_index[name]

    def __getitem__(self, index: int) -> Dict[str, object]:
        augmentation_seed = None
        if isinstance(index, (tuple, list)):
            if len(index) != 2:
                raise ValueError(f"dataset request must be index or (index, seed), got {index!r}")
            index, augmentation_seed = int(index[0]), int(index[1])
        rng = random if augmentation_seed is None else random.Random(augmentation_seed)
        item = self.samples[index]
        clean = ImageOps.exif_transpose(Image.open(item["clean"])).convert("RGB")
        glare = ImageOps.exif_transpose(Image.open(item["glare"])).convert("RGB")
        if clean.size != glare.size:
            raise ValueError(f"unaligned pair {item['name']}: {clean.size} != {glare.size}")
        prior = Image.open(item["prior"]).convert("L") if item["prior"] else Image.new("L", clean.size, 0)
        mask = None
        if self.need_mask:
            if not item["mask"]:
                raise FileNotFoundError(f"missing mask.png for validation sample {item['name']}")
            mask = Image.open(item["mask"]).convert("L")
            if mask.size != clean.size:
                mask = mask.resize(clean.size, Image.Resampling.NEAREST)
        if prior.size != clean.size:
            prior = prior.resize(clean.size, Image.Resampling.NEAREST)
        leaves = list(item["grid"].leaves)

        if self.augment and rng.random() < 0.5:
            clean = clean.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            glare = glare.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            prior = prior.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            leaves = flip_leaves(leaves, self.cfg.canvas_size)
        if self.augment:
            brightness = rng.uniform(0.95, 1.05)
            contrast = rng.uniform(0.95, 1.05)
            clean = ImageEnhance.Contrast(ImageEnhance.Brightness(clean).enhance(brightness)).enhance(contrast)
            glare = ImageEnhance.Contrast(ImageEnhance.Brightness(glare).enhance(brightness)).enhance(contrast)
            if self.identity_prob > 0 and rng.random() < self.identity_prob:
                glare = clean.copy()

        clean_patches = extract_canonical_patches(clean, leaves, self.cfg.canvas_size, self.canonical_size)
        glare_patches = extract_canonical_patches(glare, leaves, self.cfg.canvas_size, self.canonical_size)
        region_probs = region_probs_from_prior(prior, leaves, self.cfg.canvas_size)

        sample = {
            "name": str(item["name"]),
            "clean_patches": clean_patches,
            "glare_patches": glare_patches,
            "leaves": torch.tensor([[v.x, v.y, v.size] for v in leaves], dtype=torch.long),
            "region_probs": region_probs,
            "constraint_mode": str(item["grid"].constraint_mode),
        }
        if not self.need_native:
            return sample

        clean_square, crop_box = square_edge_canvas(clean)
        glare_square, crop_box2 = square_edge_canvas(glare)
        prior_square, _ = square_zero_canvas(prior)
        mask_square = square_zero_canvas(mask)[0] if mask is not None else None
        if crop_box != crop_box2 or clean_square.size != glare_square.size:
            raise AssertionError("square alignment mismatch")
        side = clean_square.size[0]
        metadata = load_metadata(item.get("meta"))
        geometry_maps = build_geometry_maps(prior, metadata, clean.size, self.cfg)
        lens_mask_logical = geometry_maps["lens_band"].astype(np.float32)
        lens_logical = Image.fromarray((lens_mask_logical * 255).astype(np.uint8))
        lens_square = lens_logical.resize((side, side), Image.Resampling.NEAREST)
        sample.update({
            # Native tensors are visualization/inference-only in V29.4.4.
            "leaves_cpu": [(int(v.x), int(v.y), int(v.size)) for v in leaves],
            "native_clean": pil_to_tensor(clean_square),
            "native_glare": pil_to_tensor(glare_square),
            "native_prior_u8": torch.from_numpy(np.asarray(prior_square, dtype=np.uint8).copy())[None],
            "native_lens_u8": torch.from_numpy(np.asarray(lens_square, dtype=np.uint8).copy())[None],
            "original_size": tuple(map(int, clean.size)),
            "crop_box": tuple(map(int, crop_box)),
            "native_side": int(side),
            "metadata": metadata,
        })
        if mask_square is not None:
            sample["native_mask_u8"] = torch.from_numpy(
                np.asarray(mask_square, dtype=np.uint8).copy()
            )[None]
        return sample


def collate_batch(samples):
    """Pack the fixed 256-token PMF tensors into a real batch.

    PMF-only training does not collate native-resolution images. If a caller requested
    ``need_native=True`` (gate/joint training, visualization, or inference), the ragged
    native fields are retained as Python lists without padding.
    """
    if not samples:
        raise RuntimeError("empty batch")
    fixed = ("clean_patches", "glare_patches", "leaves", "region_probs")
    batch = {key: torch.stack([sample[key] for sample in samples], dim=0) for key in fixed}
    batch["name"] = [sample["name"] for sample in samples]
    batch["constraint_mode"] = [sample["constraint_mode"] for sample in samples]
    optional = (
        "leaves_cpu", "native_clean", "native_glare", "native_prior_u8", "native_lens_u8",
        "native_mask_u8", "original_size", "crop_box", "native_side", "metadata",
    )
    for key in optional:
        if key in samples[0]:
            batch[key] = [sample[key] for sample in samples]
    return batch


def move_sample(sample: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out = {}
    for k, v in sample.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        elif isinstance(value, list) and value and torch.is_tensor(value[0]):
            out[key] = [v.to(device, non_blocking=True) for v in value]
        else:
            out[key] = value
    return out

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

def _unwrap_state(payload: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must be a mapping")
    for key in ("ema_inference", "ema", "model", "state_dict", "model_state_dict", "net", "params"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    if any(torch.is_tensor(v) for v in payload.values()):
        return payload  # type: ignore[return-value]
    raise TypeError("checkpoint has no recognizable state_dict")


def _key_candidates(raw_key: str) -> list[str]:
    out = []
    def add(k):
        if k and k not in out: out.append(k)
    add(raw_key)
    k = raw_key
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model.", "net."):
            if k.startswith(prefix):
                k = k[len(prefix):]; add(k); changed = True
    add("backbone." + k)
    if k.startswith("backbone."):
        add(k[len("backbone."):])
    return out



def _square_side_from_channel_flat(length: int, channels: int = 3) -> Optional[int]:
    if int(length) <= 0 or int(length) % int(channels):
        return None
    area = int(length) // int(channels)
    side = int(round(math.sqrt(area)))
    return side if side * side == area else None


def _migrate_canonical_tensor(key: str, source: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
    """Resize only the tensors whose spatial shape changes from c16 -> c24/c32.

    Grid geometry stays based on patch_size=16.  This migration exists solely so a
    trained V29.4 checkpoint can initialize the higher-bandwidth canonical raster:

    * patch-embed kernels are bicubically resized with DC-gain compensation;
    * direct U/V final maps are bicubically resized in output space, making the new
      head start close to an upsampled copy of the old direct prediction.

    Everything else must still match shape exactly.
    """
    name = str(key)
    src = source.detach().float().cpu()
    tgt_shape = tuple(target.shape)

    if name.endswith("x_embedder.proj1.weight"):
        if src.ndim != 4 or target.ndim != 4 or tuple(src.shape[:2]) != tuple(target.shape[:2]):
            return None
        sh, sw = map(int, src.shape[-2:])
        th, tw = map(int, target.shape[-2:])
        if sh == th and sw == tw:
            return source.to(dtype=target.dtype)
        flat = src.reshape(-1, 1, sh, sw)
        resized = F.interpolate(flat, size=(th, tw), mode="bicubic", align_corners=False)
        # Preserve response to a spatially constant patch after increasing kernel area.
        resized.mul_(float(sh * sw) / float(th * tw))
        return resized.reshape(tgt_shape).to(dtype=target.dtype)

    if name.endswith("_final_layer.linear._flax_linear.weight"):
        if src.ndim != 2 or target.ndim != 2 or int(src.shape[1]) != int(target.shape[1]):
            return None
        sp = _square_side_from_channel_flat(int(src.shape[0]))
        tp = _square_side_from_channel_flat(int(target.shape[0]))
        if sp is None or tp is None:
            return None
        hidden = int(src.shape[1])
        channels = 3
        # FinalLayer is decoded as:
        #   linear(...).reshape(T, P, P, C).permute(0, 3, 1, 2)
        # so the flattened output-row layout is H-W-C (RGB interleaved per pixel),
        # NOT C-H-W. Preserve that exact layout while resizing each RGB/hidden
        # coefficient map independently in spatial output coordinates.
        maps = (
            src.reshape(sp, sp, channels, hidden)
            .permute(2, 3, 0, 1)
            .reshape(channels * hidden, 1, sp, sp)
        )
        maps = F.interpolate(maps, size=(tp, tp), mode="bicubic", align_corners=False)
        out = (
            maps.reshape(channels, hidden, tp, tp)
            .permute(2, 3, 0, 1)
            .reshape(tp * tp * channels, hidden)
        )
        return out.to(dtype=target.dtype)

    if name.endswith("_final_layer.linear._flax_linear.bias"):
        if src.ndim != 1 or target.ndim != 1:
            return None
        sp = _square_side_from_channel_flat(int(src.numel()))
        tp = _square_side_from_channel_flat(int(target.numel()))
        if sp is None or tp is None:
            return None
        channels = 3
        # Bias follows the same H-W-C flattened layout as the weight rows.
        maps = src.reshape(sp, sp, channels).permute(2, 0, 1).unsqueeze(0)
        maps = F.interpolate(maps, size=(tp, tp), mode="bicubic", align_corners=False)
        out = maps[0].permute(1, 2, 0).reshape(tp * tp * channels)
        return out.to(dtype=target.dtype)

    return None

def warmstart(path: str, model: nn.Module, *, payload: object = None) -> None:
    if payload is None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    source = _unwrap_state(payload)
    current = model.state_dict()
    compatible = {}
    migrated = []
    for rk, value in source.items():
        if not torch.is_tensor(value):
            continue
        for cand in _key_candidates(str(rk)):
            if cand not in current:
                continue
            target = current[cand]
            if tuple(value.shape) == tuple(target.shape):
                compatible[cand] = value.to(dtype=target.dtype)
                break
            converted = _migrate_canonical_tensor(cand, value, target)
            if converted is not None and tuple(converted.shape) == tuple(target.shape):
                compatible[cand] = converted
                migrated.append(cand)
                break
    if not compatible:
        raise RuntimeError(f"{path}: zero compatible tensors")
    model.load_state_dict(compatible, strict=False)
    local = [k for k in current if k not in compatible]
    matched_v = sum(1 for k in compatible if _is_v_key(k))
    total_v = sum(1 for k in current if _is_v_key(k))
    print(
        f"[init] {path}: matched={len(compatible)}/{len(current)} local_init={len(local)} "
        f"v_matched={matched_v}/{total_v} canonical_migrated={len(migrated)}",
        flush=True,
    )
    if migrated:
        print(f"[init] canonical migration={migrated}", flush=True)
    print(f"[init] local examples={local[:12]}", flush=True)


def unwrap_official_state(payload: object) -> Dict[str, torch.Tensor]:
    state = payload
    if isinstance(payload, Mapping):
        for candidate in ("state_dict", "model", "net"):
            value = payload.get(candidate)
            if isinstance(value, Mapping):
                state = value
                break
    if not isinstance(state, Mapping):
        raise TypeError("official checkpoint does not contain a state_dict")
    output: Dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model.", "net.", "backbone."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        output[key] = value
    return output


def load_official_pmf_weights(
    path: str, model: "MaskFreeResidualPMF", *, payload: object = None
) -> None:
    """Map an official pMF-B/16 checkpoint into the adaptive V29 backbone.

    The adaptive position/scale parameters and eye-prior projection intentionally
    remain at their local zero initialization.
    """
    backbone = model.backbone
    if not (
        int(backbone.input_size) == 256
        and int(backbone.patch_size) == 16
        and int(backbone.hidden_size) == 768
        and int(backbone.depth) == 16
        and int(backbone.num_heads) == 12
    ):
        raise ValueError(
            "official pMF initialization requires canvas_size=256 and grid base_patch_size=16; "
            "canonical_size may differ and is migrated by spatial interpolation"
        )
    if payload is None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    official = unwrap_official_state(payload)
    current = backbone.state_dict()
    transferable: Dict[str, torch.Tensor] = {}

    migrated = []
    for key, value in official.items():
        if key not in current:
            continue
        target = current[key]
        if tuple(value.shape) == tuple(target.shape):
            transferable[key] = value.to(dtype=target.dtype)
            continue
        converted = _migrate_canonical_tensor(key, value, target)
        if converted is not None and tuple(converted.shape) == tuple(target.shape):
            transferable[key] = converted
            migrated.append(key)

    source_class = official.get("class_tokens")
    if (
        source_class is not None
        and source_class.ndim == 3
        and source_class.shape[1] >= backbone.num_class_tokens
    ):
        transferable["class_tokens"] = source_class[:, : backbone.num_class_tokens].to(
            current["class_tokens"].dtype
        )

    class_key = "y_embedder.embedding_table._flax_embedding.weight"
    if class_key in official and class_key in current:
        target = current[class_key].clone()
        target[:] = official[class_key][-1].to(target.dtype)
        transferable[class_key] = target

    source_pos = official.get("pos_embed")
    target_pos = current.get("pos_embed")
    if source_pos is not None and target_pos is not None:
        image_count = int(target_pos.shape[1] - backbone.prefix_tokens)
        source_prefix = int(source_pos.shape[1] - image_count)
        time_start = source_prefix - int(backbone.num_time_tokens)
        if source_prefix >= 0 and time_start >= 0:
            converted = torch.cat(
                [
                    source_pos[:, : backbone.num_class_tokens],
                    source_pos[:, time_start : time_start + backbone.num_time_tokens],
                    source_pos[:, source_prefix:],
                ],
                dim=1,
            )
            if tuple(converted.shape) == tuple(target_pos.shape):
                transferable["pos_embed"] = converted.to(target_pos.dtype)

    required = (
        "x_embedder.proj1.weight",
        "x_embedder.proj2.weight",
        "shared_blocks.0.attn.q_proj._flax_linear.weight",
        "u_heads.0.attn.q_proj._flax_linear.weight",
        "u_final_layer.linear._flax_linear.weight",
        "pos_embed",
    )
    missing = [key for key in required if key not in transferable]
    if missing:
        raise RuntimeError(f"official checkpoint mapping failed; missing required tensors: {missing}")

    result = backbone.load_state_dict(transferable, strict=False)
    loaded_v = sum(1 for k in transferable if k.startswith("v_heads.") or k.startswith("v_final_layer."))
    total_v = sum(1 for k in current if k.startswith("v_heads.") or k.startswith("v_final_layer."))
    print(
        f"[official-init] loaded={len(transferable)}/{len(current)} "
        f"v_loaded={loaded_v}/{total_v} canonical_migrated={len(migrated)} "
        f"local_init={len(result.missing_keys)}; "
        "adaptive/eye-prior parameters keep local zero init",
        flush=True,
    )


def load_initial_weights(path: str, model: "MaskFreeResidualPMF") -> str:
    """Load either an official pMF checkpoint or a local compatible checkpoint.

    Official releases are bare ``net.*`` state dictionaries and need special
    class-token, label-embedding and positional-embedding migration. Local training
    checkpoints carry an ``architecture`` field and continue to use warm-start.
    The payload is read once so a multi-hundred-MB official checkpoint is not loaded
    twice merely to identify its format.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_state = payload
    if isinstance(payload, Mapping):
        for candidate in ("state_dict", "model", "net"):
            value = payload.get(candidate)
            if isinstance(value, Mapping):
                raw_state = value
                break
    raw_keys = [str(key) for key in raw_state] if isinstance(raw_state, Mapping) else []
    normalized_keys = set(unwrap_official_state(payload)) if isinstance(raw_state, Mapping) else set()
    official = (
        isinstance(payload, Mapping)
        and "architecture" not in payload
        and "x_embedder.proj1.weight" in normalized_keys
        and "pos_embed" in normalized_keys
        and "class_tokens" in normalized_keys
        and not any("conditioner." in key or "learned_frequency_gate." in key for key in raw_keys)
    )
    if official:
        load_official_pmf_weights(path, model, payload=payload)
        return "official-pMF"
    warmstart(path, model, payload=payload)
    return "compatible-checkpoint"


def _is_v_key(key: str) -> bool:
    return (
        key.startswith("backbone.v_heads.")
        or key.startswith("backbone.v_final_layer.")
        or key.startswith("v_heads.")
        or key.startswith("v_final_layer.")
    )


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

def resolve_device(args: argparse.Namespace, *, train_mode: bool) -> torch.device:
    """Resolve CPU/CUDA explicitly and never silently run a training job on CPU."""
    requested = str(getattr(args, "device", "") or "").strip()
    if requested:
        device = torch.device(requested)
        if device.type not in ("cuda", "cpu"):
            raise ValueError(f"unsupported --device {requested!r}; use cuda[:index] or cpu")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {requested!r} was requested, but torch.cuda.is_available() is False. "
                "Check NVIDIA driver/container visibility before running PMF."
            )
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if train_mode:
        raise RuntimeError(
            "CUDA is unavailable. V29.4.4 refuses to silently fall back to CPU training. "
            "Fix NVIDIA/CUDA visibility, or pass --device cpu explicitly only for debugging."
        )
    return torch.device("cpu")


def validate_precision_for_device(args: argparse.Namespace, device: torch.device) -> None:
    precision = str(args.precision)
    if precision not in ("bf16", "fp32"):
        raise ValueError("V29.4.4 supports only --precision bf16 or fp32")
    if device.type == "cuda" and precision == "bf16":
        with torch.cuda.device(device):
            bf16_ok = bool(torch.cuda.is_bf16_supported())
        if not bf16_ok:
            raise RuntimeError(
                f"--precision bf16 was requested, but {device} does not report BF16 support; "
                "use --precision fp32."
            )
    if device.type == "cpu" and precision != "fp32":
        raise ValueError("CPU training/debug requires --precision fp32; BF16 autocast is CUDA-only here")


def autocast_ctx(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision != "bf16":
        raise ValueError(f"unsupported precision={precision!r}; V29.4.4 supports only bf16/fp32")
    return torch.autocast("cuda", dtype=torch.bfloat16)


def crop_original(t: torch.Tensor, box: Sequence[int]) -> torch.Tensor:
    x1, y1, x2, y2 = map(int, box)
    return t[..., y1:y2, x1:x2]


def grid_overlay(glare_square: torch.Tensor, leaves: torch.Tensor, canvas_size: int) -> Image.Image:
    image = tensor_to_pil(glare_square).convert("RGB")
    draw = ImageDraw.Draw(image)
    side = image.width
    colors = {4: (0, 255, 255), 8: (0, 220, 80), 16: (255, 170, 0), 32: (255, 0, 220)}
    for x, y, size in leaves.detach().cpu().tolist():
        x1 = int(round(x * side / canvas_size)); y1 = int(round(y * side / canvas_size))
        x2 = int(round((x+size) * side / canvas_size)); y2 = int(round((y+size) * side / canvas_size))
        draw.rectangle((x1, y1, max(x1, x2-1), max(y1, y2-1)), outline=colors.get(int(size), (255,255,255)), width=1)
    draw.rectangle((5,5,190,65), fill=(0,0,0))
    y = 8
    for size in (4,8,16,32):
        draw.line((10,y+6,28,y+6), fill=colors[size], width=3)
        draw.text((34,y), f"p{size}", fill=(255,255,255)); y += 14
    return image


def labeled_tile(image: Image.Image, label: str, tile: int = 320) -> Image.Image:
    im = image.convert("RGB")
    im.thumbnail((tile, tile-24), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (tile, tile), "white")
    x = (tile-im.width)//2; y = 24 + (tile-24-im.height)//2
    canvas.paste(im, (x,y))
    ImageDraw.Draw(canvas).text((6,5), label, fill="black")
    return canvas
