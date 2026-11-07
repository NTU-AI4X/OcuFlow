#!/usr/bin/env python3
"""Generate glare-derived eye priors and ROI metadata for mask-free ReGlass.

The script intentionally reads ``glare.png`` rather than ``clean.png`` so the
same geometry is available during training and deployment.  It supports the
current MediaPipe Tasks Face Landmarker API and a legacy FaceMesh fallback.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Sequence

# MediaPipe may import matplotlib even for --help. Put its config in a writable
# temporary directory unless the user already selected MPLCONFIGDIR.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "reglass_mplconfig"))

import cv2
import mediapipe as mp
import numpy as np
from models.spatial import GEOMETRY_VERSION, LANDMARK_SCHEMA, source_signature


# MediaPipe anatomical left/right naming follows the official FaceMesh
# connection sets.  In an unmirrored frontal image the anatomical left eye is
# normally on the image-right side.
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_BROW = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
RIGHT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Inner means nose-side; outer means temple-side.
LEFT_INNER_CORNER = 362
LEFT_OUTER_CORNER = 263
RIGHT_INNER_CORNER = 133
RIGHT_OUTER_CORNER = 33
def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def valid_indices(indices: Sequence[int], count: int) -> list[int]:
    return [index for index in indices if 0 <= index < count]


def points_for(points: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    valid = valid_indices(indices, len(points))
    return points[valid] if valid else np.empty((0, 2), dtype=np.float32)


def sort_clockwise(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angle = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angle)]


def draw_polygon(prior: np.ndarray, points: np.ndarray, indices: Sequence[int], value: int) -> bool:
    selected = points_for(points, indices)
    if len(selected) < 3:
        return False
    polygon = sort_clockwise(selected).round().astype(np.int32)
    cv2.fillPoly(prior, [polygon], int(value))
    return True


def draw_iris(prior: np.ndarray, points: np.ndarray, indices: Sequence[int], value: int, dilation_ratio: float) -> bool:
    selected = points_for(points, indices)
    if len(selected) < 3:
        return False
    center = selected.mean(axis=0)
    radius = max(float(np.linalg.norm(selected - center[None], axis=1).mean()), 2.0)
    radius *= max(1.0, 1.0 + float(dilation_ratio))
    cv2.circle(
        prior,
        tuple(np.round(center).astype(np.int32).tolist()),
        max(1, int(round(radius))),
        int(value),
        thickness=-1,
    )
    return True


def clipped_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, int(math.floor(x1))))
    top = max(0, min(height - 1, int(math.floor(y1))))
    right = max(left + 1, min(width, int(math.ceil(x2))))
    bottom = max(top + 1, min(height, int(math.ceil(y2))))
    return left, top, right, bottom


def periocular_box(
    points: np.ndarray,
    eye_indices: Sequence[int],
    brow_indices: Sequence[int],
    width: int,
    height: int,
    horizontal_ratio: float,
    vertical_ratio: float,
) -> Optional[tuple[int, int, int, int]]:
    selected = np.concatenate([points_for(points, eye_indices), points_for(points, brow_indices)], axis=0)
    if len(selected) < 3:
        return None
    x1, y1 = selected.min(axis=0)
    x2, y2 = selected.max(axis=0)
    span_x = max(1.0, float(x2 - x1))
    span_y = max(1.0, float(y2 - y1))
    return clipped_box(
        x1 - horizontal_ratio * span_x,
        y1 - vertical_ratio * span_y,
        x2 + horizontal_ratio * span_x,
        y2 + 0.7 * vertical_ratio * span_y,
        width,
        height,
    )


def eye_center(points: np.ndarray, indices: Sequence[int]) -> Optional[np.ndarray]:
    selected = points_for(points, indices)
    return selected.mean(axis=0) if len(selected) else None


def normalized_group(points: np.ndarray, indices: Sequence[int], width: int, height: int) -> list[list[float]]:
    selected = points_for(points, indices)
    return [[float(x) / width, float(y) / height] for x, y in selected]


def normalized_point(points: np.ndarray, index: int, width: int, height: int) -> Optional[list[float]]:
    if not 0 <= int(index) < len(points):
        return None
    x, y = points[int(index)]
    return [float(x) / width, float(y) / height]


def normalized_center(points: np.ndarray, indices: Sequence[int], width: int, height: int) -> Optional[list[float]]:
    center = eye_center(points, indices)
    if center is None:
        return None
    return [float(center[0]) / width, float(center[1]) / height]


def landmark_groups_normalized(points: np.ndarray, width: int, height: int) -> dict:
    """Return the fixed eight-channel LRDA semantic landmark schema."""
    return {
        "left_eye_contour": normalized_group(points, LEFT_EYE, width, height),
        "right_eye_contour": normalized_group(points, RIGHT_EYE, width, height),
        "left_iris_center": normalized_center(points, LEFT_IRIS, width, height),
        "right_iris_center": normalized_center(points, RIGHT_IRIS, width, height),
        "left_inner_corner": normalized_point(points, LEFT_INNER_CORNER, width, height),
        "left_outer_corner": normalized_point(points, LEFT_OUTER_CORNER, width, height),
        "right_inner_corner": normalized_point(points, RIGHT_INNER_CORNER, width, height),
        "right_outer_corner": normalized_point(points, RIGHT_OUTER_CORNER, width, height),
    }


def build_prior_and_meta(image_bgr: np.ndarray, points: Optional[np.ndarray], args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    height, width = image_bgr.shape[:2]
    prior = np.zeros((height, width), dtype=np.uint8)
    meta = {
        "success": False,
        "fail_reason": "face_landmarker_failed",
        "source_image": "glare.png",
        "source_width": width,
        "source_height": height,
        "roi_normalized": None,
        "roi_pixels": None,
        "eye_line_angle_degrees": 0.0,
        "has_eye": False,
        "has_iris": False,
        "prior_valid": False,
        "geometry_version": GEOMETRY_VERSION,
        "landmark_schema": LANDMARK_SCHEMA,
        "landmark_groups_normalized": None,
    }
    if points is None:
        return prior, meta

    periocular_boxes = []
    for eye_indices, brow_indices in ((LEFT_EYE, LEFT_BROW), (RIGHT_EYE, RIGHT_BROW)):
        box = periocular_box(
            points,
            eye_indices,
            brow_indices,
            width,
            height,
            args.periocular_horizontal_ratio,
            args.periocular_vertical_ratio,
        )
        if box is not None:
            left, top, right, bottom = box
            periocular_boxes.append(box)
            prior[top:bottom, left:right] = np.maximum(prior[top:bottom, left:right], 64)
            meta["has_eye"] = True

    left_eye = draw_polygon(prior, points, LEFT_EYE, 128)
    right_eye = draw_polygon(prior, points, RIGHT_EYE, 128)
    left_iris = draw_iris(prior, points, LEFT_IRIS, 192, args.iris_dilation_ratio)
    right_iris = draw_iris(prior, points, RIGHT_IRIS, 192, args.iris_dilation_ratio)
    meta["has_eye"] = bool(meta["has_eye"] or left_eye or right_eye)
    meta["has_iris"] = bool(left_iris and right_iris)

    if len(periocular_boxes) != 2:
        meta["fail_reason"] = "periocular_roi_failed"
        return prior, meta
    left = min(box[0] for box in periocular_boxes)
    top = min(box[1] for box in periocular_boxes)
    right = max(box[2] for box in periocular_boxes)
    bottom = max(box[3] for box in periocular_boxes)
    left_center = eye_center(points, LEFT_EYE)
    right_center = eye_center(points, RIGHT_EYE)
    if left_center is None or right_center is None:
        meta["fail_reason"] = "eye_center_failed"
        return prior, meta
    # Measure the line from image-left (anatomical right) to image-right
    # (anatomical left), yielding an angle near zero for an upright face.
    vector = left_center - right_center
    angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    meta.update(
        {
            "success": True,
            "fail_reason": "",
            "roi_pixels": [left, top, right, bottom],
            "roi_normalized": [left / width, top / height, right / width, bottom / height],
            "eye_line_angle_degrees": angle,
            "roi_definition": "exact_union_of_yellow_periocular_boxes",
            "geometry_version": GEOMETRY_VERSION,
            "landmark_schema": LANDMARK_SCHEMA,
            "landmark_groups_normalized": landmark_groups_normalized(points, width, height),
            "prior_valid": bool(meta["has_eye"]),
            "landmark_count": int(len(points)),
        }
    )
    return prior, meta


class LandmarkDetector:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.backend = args.backend
        self.detector = None
        if self.backend == "auto":
            self.backend = "tasks" if args.model_asset_path and Path(args.model_asset_path).is_file() else "legacy"
        if self.backend == "tasks":
            if not args.model_asset_path or not Path(args.model_asset_path).is_file():
                raise FileNotFoundError("--model_asset_path is required for --backend tasks")
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=args.model_asset_path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=args.min_detection_confidence,
                min_face_presence_confidence=args.min_presence_confidence,
                min_tracking_confidence=args.min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        elif self.backend == "legacy":
            solutions = getattr(mp, "solutions", None)
            if solutions is None or not hasattr(solutions, "face_mesh"):
                raise RuntimeError(
                    "This MediaPipe build does not expose mp.solutions.face_mesh. "
                    "Use --backend tasks with --model_asset_path /path/to/face_landmarker.task."
                )
            self.detector = solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=args.min_detection_confidence,
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()

    def detect(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "tasks":
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(image)
            if not result.face_landmarks:
                return None
            landmarks = result.face_landmarks[0]
        else:
            result = self.detector.process(rgb)
            if not result.multi_face_landmarks:
                return None
            landmarks = result.multi_face_landmarks[0].landmark
        return np.asarray([[point.x * width, point.y * height] for point in landmarks], dtype=np.float32)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def debug_visual(image_bgr: np.ndarray, prior: np.ndarray, meta: dict) -> np.ndarray:
    overlay = image_bgr.copy()
    colors = np.zeros_like(image_bgr)
    colors[prior >= 64] = (0, 255, 255)
    colors[prior >= 128] = (0, 128, 255)
    colors[prior >= 192] = (0, 0, 255)
    valid = prior > 0
    if np.any(valid):
        mixed = cv2.addWeighted(image_bgr, 0.65, colors, 0.35, 0.0)
        overlay[valid] = mixed[valid]
    roi = meta.get("roi_pixels")
    if roi:
        left, top, right, bottom = map(int, roi)
        cv2.rectangle(overlay, (left, top), (right - 1, bottom - 1), (0, 255, 0), 2)
    prior_bgr = cv2.cvtColor(prior, cv2.COLOR_GRAY2BGR)
    return np.concatenate([image_bgr, prior_bgr, overlay], axis=1)


def iter_samples(root: Path, splits: Sequence[str], max_samples: int) -> Iterable[tuple[str, Path]]:
    count = 0
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            print(f"[warn] missing split: {split_dir}", flush=True)
            continue
        for folder in sorted(split_dir.iterdir()):
            if folder.is_dir() and (folder / "glare.png").is_file():
                yield split, folder
                count += 1
                if max_samples > 0 and count >= max_samples:
                    return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "test_real", "test_syn"])
    parser.add_argument("--backend", choices=("auto", "tasks", "legacy"), default="auto")
    parser.add_argument("--model_asset_path", default="", help="MediaPipe face_landmarker.task model for Tasks backend.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--save_debug_num", type=int, default=30)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_presence_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    parser.add_argument("--periocular_horizontal_ratio", type=float, default=0.22)
    parser.add_argument("--periocular_vertical_ratio", type=float, default=0.35)
    parser.add_argument("--iris_dilation_ratio", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)
    debug_dir = root / "face_geometry_debug"
    manifest_path = root / "face_geometry_manifest.csv"
    summary_path = root / "face_geometry_summary.json"
    samples = list(iter_samples(root, args.splits, args.max_samples))
    rows: list[dict] = []
    debug_saved = 0
    print(f"[start] samples={len(samples)} backend={args.backend}", flush=True)

    with LandmarkDetector(args) as detector:
        print(f"[detector] resolved_backend={detector.backend}", flush=True)
        for index, (split, folder) in enumerate(samples, start=1):
            glare_path = folder / "glare.png"
            prior_path = folder / "eye_prior.png"
            meta_path = folder / "face_meta.json"
            signature = source_signature(glare_path)
            if prior_path.exists() and meta_path.exists() and not args.overwrite:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if (
                    meta.get("geometry_version") == GEOMETRY_VERSION
                    and meta.get("landmark_schema") == LANDMARK_SCHEMA
                    and meta.get("source_signature") == signature
                ):
                    source = folder.name.split("_", 1)[0].lower() if "_" in folder.name else "other"
                    row = {"split": split, "source": source, "sample_id": folder.name, **meta, "skipped": True}
                    rows.append(row)
                    continue
                print(f"[stale] {split}/{folder.name}: geometry signature/version changed; rebuilding", flush=True)
            image = imread_unicode(glare_path)
            if image is None:
                meta = {
                    "success": False, "fail_reason": "glare_read_failed",
                    "source_image": "glare.png", "geometry_version": GEOMETRY_VERSION,
                    "landmark_schema": LANDMARK_SCHEMA,
                }
                prior = np.zeros((1, 1), dtype=np.uint8)
            else:
                points = detector.detect(image)
                prior, meta = build_prior_and_meta(image, points, args)
                if not imwrite_unicode(prior_path, prior):
                    meta.update(success=False, fail_reason="prior_write_failed")
            meta["backend"] = detector.backend
            meta["source_signature"] = signature
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            source = folder.name.split("_", 1)[0].lower() if "_" in folder.name else "other"
            row = {"split": split, "source": source, "sample_id": folder.name, **meta, "skipped": False}
            rows.append(row)
            if image is not None and debug_saved < args.save_debug_num:
                if imwrite_unicode(debug_dir / f"{split}_{folder.name}.png", debug_visual(image, prior, meta)):
                    debug_saved += 1
            if index == 1 or index % 100 == 0 or index == len(samples):
                success = sum(bool(item.get("success")) for item in rows)
                print(f"[progress] {index}/{len(samples)} success={success} failed={len(rows)-success}", flush=True)

    structured_fields = {"roi_normalized", "roi_pixels", "landmark_groups_normalized", "source_signature"}
    fieldnames = sorted({key for row in rows for key in row.keys() if key not in structured_fields})
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames + ["roi_normalized", "roi_pixels", "landmark_groups_normalized"],
        )
        writer.writeheader()
        for row in rows:
            serial = dict(row)
            serial["roi_normalized"] = json.dumps(row.get("roi_normalized"), ensure_ascii=False)
            serial["roi_pixels"] = json.dumps(row.get("roi_pixels"), ensure_ascii=False)
            serial["landmark_groups_normalized"] = json.dumps(
                row.get("landmark_groups_normalized"), ensure_ascii=False
            )
            writer.writerow({key: serial.get(key, "") for key in writer.fieldnames})
    split_summary = {}
    source_summary = {}
    for row in rows:
        split_name = str(row.get("split", "unknown"))
        source_name = str(row.get("source", "other"))
        split_summary.setdefault(split_name, {"total": 0, "success": 0, "failed": 0})
        source_summary.setdefault(source_name, {"total": 0, "success": 0, "failed": 0})
        for bucket in (split_summary[split_name], source_summary[source_name]):
            bucket["total"] += 1
            if bool(row.get("success")):
                bucket["success"] += 1
            else:
                bucket["failed"] += 1
    summary = {
        "dataset_root": str(root),
        "total": len(rows),
        "success": sum(bool(row.get("success")) for row in rows),
        "failed": sum(not bool(row.get("success")) for row in rows),
        "has_iris": sum(bool(row.get("has_iris")) for row in rows),
        "by_split": split_summary,
        "by_source": source_summary,
        "debug_saved": debug_saved,
        "source_image": "glare.png",
        "gray_values": {"0": "background", "64": "periocular", "128": "eye_region", "192": "iris_pupil"},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] manifest={manifest_path}", flush=True)
    print(f"[done] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
