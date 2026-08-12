"""Convert coral_soft's bbox JSON annotations into a YOLO-format dataset.

coral_soft/annotations/*.json each contain: [{"image": "<name>.JPG",
"annotations": [{"label": str, "coordinates": {x, y, width, height}}]}]
where x,y are the box CENTER in pixels (verified against source images,
not top-left) and width/height are pixel box size. An image can carry
boxes for multiple coral classes.

Image files live under coral_soft/image/<Class>/<name>.JPG but are
inconsistently encoded (JPEG/PNG/MPO all under a .JPG extension), so
this script re-encodes every referenced image to real JPEG in the
output directory rather than copying bytes as-is.
"""
import json
import random
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = REPO_ROOT / "coral_soft" / "annotations"
IMAGE_DIR = REPO_ROOT / "coral_soft" / "image"
OUT_DIR = REPO_ROOT / "data" / "processed" / "coral_soft"
VAL_FRACTION = 0.2
SEED = 42


def find_image_file(image_name: str) -> Path | None:
    matches = list(IMAGE_DIR.glob(f"*/{image_name}"))
    if not matches:
        # extension case mismatch (e.g. .jpg vs .JPG) - fall back to stem search
        stem = Path(image_name).stem
        matches = [p for p in IMAGE_DIR.glob(f"*/{stem}.*")]
    return matches[0] if matches else None


def load_annotations():
    records = []
    labels = set()
    for json_path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        with open(json_path) as f:
            entries = json.load(f)
        for entry in entries:
            image_name = entry["image"]
            boxes = entry["annotations"]
            image_path = find_image_file(image_name)
            if image_path is None:
                print(f"WARN: no image file found for {image_name} (from {json_path.name}), skipping")
                continue
            records.append({"image_path": image_path, "boxes": boxes})
            for box in boxes:
                labels.add(box["label"])
    return records, sorted(labels)


def convert_box_to_yolo(box, img_w, img_h):
    c = box["coordinates"]
    cx, cy, w, h = c["x"], c["y"], c["width"], c["height"]
    # clip to image bounds defensively
    cx = min(max(cx, 0), img_w)
    cy = min(max(cy, 0), img_h)
    w = min(w, img_w)
    h = min(h, img_h)
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def main():
    random.seed(SEED)
    records, class_names = load_annotations()
    class_index = {name: i for i, name in enumerate(class_names)}
    print(f"Found {len(records)} images, {len(class_names)} classes: {class_names}")

    random.shuffle(records)
    n_val = max(1, int(len(records) * VAL_FRACTION))
    splits = {"val": records[:n_val], "train": records[n_val:]}

    for split, split_records in splits.items():
        img_out = OUT_DIR / "images" / split
        lbl_out = OUT_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for record in split_records:
            image_path = record["image_path"]
            try:
                im = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"WARN: failed to open {image_path}: {e}, skipping")
                continue
            img_w, img_h = im.size

            stem = image_path.stem
            out_image_path = img_out / f"{stem}.jpg"
            im.save(out_image_path, "JPEG", quality=90)

            lines = []
            for box in record["boxes"]:
                cx, cy, w, h = convert_box_to_yolo(box, img_w, img_h)
                cls_idx = class_index[box["label"]]
                lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")

        print(f"{split}: wrote {len(split_records)} images")

    data_yaml = OUT_DIR / "data.yaml"
    yaml_lines = [
        f"path: {OUT_DIR}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
    ]
    data_yaml.write_text("\n".join(yaml_lines) + "\n")
    print(f"Wrote {data_yaml}")


if __name__ == "__main__":
    main()
