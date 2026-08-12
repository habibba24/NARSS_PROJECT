"""Grow archive-2's sparse algae point-labels into weak segmentation masks.

archive-2/combined_annotations_remapped.csv has one row per labeled point:
Name, Row, Column, Label (Row = y, Column = x, verified against image
dimensions). No boxes/masks exist in the source data, so for each point
labeled as an algae class we flood-fill a small neighborhood around it
(bounded by a radius, so it can't leak across the whole image) to get an
approximate local blob, then union all blobs for an image into one mask.

Class choice: only true algae-overgrowth types (macroalgae, algae,
green_fleshy_algae) are treated as the damage-relevant "algae" class.
crustose_coralline_algae and turf are deliberately excluded — CCA is
generally a natural/beneficial reef component, not a stress indicator,
so including it would mislabel large amounts of healthy substrate as
"damage".

Outputs, for each processed image:
- data/processed/algae_masks/masks/<stem>.png       (binary mask, for pixel IoU/Dice eval)
- data/processed/algae_masks/labels/<split>/<stem>.txt  (YOLO-seg polygon format, for training)
- data/processed/algae_masks/images/<split>/<stem>.txt  (path to source image, no copy - archive-2 stays put)
"""
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "archive-2" / "combined_annotations_remapped.csv"
IMAGE_DIR = REPO_ROOT / "archive-2" / "images" / "images"
OUT_DIR = REPO_ROOT / "data" / "processed" / "algae_masks"
ALGAE_LABELS = {"macroalgae", "algae", "green_fleshy_algae"}
VAL_FRACTION = 0.2
SEED = 42
FLOOD_TOLERANCE = 10
MIN_CONTOUR_AREA = 20


def find_image_file(name: str) -> Path | None:
    p = IMAGE_DIR / name
    if p.exists():
        return p
    stem = Path(name).stem
    matches = list(IMAGE_DIR.glob(f"{stem}.*"))
    return matches[0] if matches else None


def grow_mask(image_bgr: np.ndarray, points_xy: list[tuple[int, int]]) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    radius = int(np.clip(min(w, h) * 0.03, 15, 60))
    full_mask = np.zeros((h, w), dtype=np.uint8)

    for x, y in points_xy:
        x0, x1 = max(0, x - radius), min(w, x + radius)
        y0, y1 = max(0, y - radius), min(h, y + radius)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = image_bgr[y0:y1, x0:x1].copy()
        seed = (int(np.clip(x - x0, 0, roi.shape[1] - 1)), int(np.clip(y - y0, 0, roi.shape[0] - 1)))
        ff_mask = np.zeros((roi.shape[0] + 2, roi.shape[1] + 2), dtype=np.uint8)
        try:
            cv2.floodFill(
                roi, ff_mask, seed, newVal=(255, 255, 255),
                loDiff=(FLOOD_TOLERANCE,) * 3, upDiff=(FLOOD_TOLERANCE,) * 3,
                flags=8 | cv2.FLOODFILL_FIXED_RANGE,
            )
        except cv2.error:
            continue
        blob = ff_mask[1:-1, 1:-1]
        full_mask[y0:y1, x0:x1] = np.maximum(full_mask[y0:y1, x0:x1], blob * 255)

    return full_mask


def mask_to_yolo_seg_lines(mask: np.ndarray, class_idx: int = 0) -> list[str]:
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        pts = contour.reshape(-1, 2).astype(np.float64)
        pts[:, 0] /= w
        pts[:, 1] /= h
        coord_str = " ".join(f"{v:.6f}" for v in pts.flatten())
        lines.append(f"{class_idx} {coord_str}")
    return lines


def main():
    random.seed(SEED)
    df = pd.read_csv(CSV_PATH)
    algae_df = df[df["Label"].isin(ALGAE_LABELS)]
    image_names = sorted(algae_df["Name"].unique())
    print(f"{len(image_names)} images have algae-class points ({len(algae_df)} points total)")

    random.shuffle(image_names)
    n_val = max(1, int(len(image_names) * VAL_FRACTION))
    splits = {"val": image_names[:n_val], "train": image_names[n_val:]}

    (OUT_DIR / "masks").mkdir(parents=True, exist_ok=True)
    written = 0
    for split, names in splits.items():
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)

        for name in names:
            image_path = find_image_file(name)
            if image_path is None:
                print(f"WARN: image not found for {name}, skipping")
                continue
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"WARN: failed to read {image_path}, skipping")
                continue

            points = algae_df[algae_df["Name"] == name][["Column", "Row"]].to_records(index=False)
            points_xy = [(int(c), int(r)) for c, r in points]

            mask = grow_mask(image_bgr, points_xy)
            stem = image_path.stem
            cv2.imwrite(str(OUT_DIR / "masks" / f"{stem}.png"), mask)

            lines = mask_to_yolo_seg_lines(mask)
            (OUT_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n" if lines else "")
            # symlink rather than copy - archive-2 images stay in place, zero extra disk cost
            link_path = OUT_DIR / "images" / split / image_path.name
            if not link_path.exists():
                link_path.symlink_to(image_path)
            written += 1

        print(f"{split}: processed {len(names)} images")

    data_yaml = OUT_DIR / "data.yaml"
    data_yaml.write_text(
        f"path: {OUT_DIR}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['algae']\n"
    )
    print(f"Wrote {written} masks total, and {data_yaml}")


if __name__ == "__main__":
    main()
