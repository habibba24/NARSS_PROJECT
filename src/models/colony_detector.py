"""Stage 1: coral colony detector.

Fine-tunes a YOLO detector on coral_soft's genus-labeled boxes. Its job
is purely to localize "this region is coral tissue" so the downstream
algae/bleaching/disease stages only ever look inside real coral, not
sand/background/glare. Genus identity itself isn't the point of this
model (that's what coral_soft was built for, but here we fold all 6
classes into localizing coral-vs-not) - kept multi-class anyway since
it's free from the same training run and may be useful later.
"""
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "data" / "processed" / "coral_soft" / "data.yaml"
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints" / "colony_detector"
BASE_MODEL = "yolo11n.pt"


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
