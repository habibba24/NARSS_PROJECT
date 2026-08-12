"""Stage 2a: algae-overgrowth segmenter.

Trains a YOLO segmentation model on the weak masks grown from archive-2's
algae/macroalgae/green_fleshy_algae points (see grow_algae_masks.py).
Single class ("algae") - the point was never genus-level precision here,
it's "is this pixel overgrown with algae."
"""
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "data" / "processed" / "algae_masks" / "data.yaml"
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints" / "algae_segmenter"
BASE_MODEL = "yolo11n-seg.pt"


def train(epochs: int = 30, imgsz: int = 640, device: str | None = None):
    model = YOLO(BASE_MODEL)
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=str(CHECKPOINT_DIR.parent),
        name=CHECKPOINT_DIR.name,
        exist_ok=True,
        patience=10,
    )
    return results


def load_best() -> YOLO:
    best_path = CHECKPOINT_DIR / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"No trained checkpoint at {best_path} - run train() first")
    return YOLO(str(best_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    train(epochs=args.epochs, imgsz=args.imgsz, device=args.device)
