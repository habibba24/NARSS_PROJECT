"""Export each trained stage as its own standalone ONNX graph, with the
underwater color-cast correction embedded inside the graph (via
PreprocessedDetector - see pipeline/coral_damage_model.py for why this
is done per-stage rather than as one monolithic graph for the whole
pipeline).

Each exported .onnx file takes a [1,3,640,640] float tensor in [0,255]
(RGB, already letterbox-resized to 640x640 - see
pipeline/preprocessing.letterbox_resize_numpy for the exact resize the
client must replicate before calling the model) and returns raw model
output; NMS/box-decoding for detection models happens client-side using
the same logic ultralytics' own ONNX/TFJS export helpers document.

bleaching_module.onnx is exported directly (it's already pure torch, no
YOLO backbone) so it's runnable via onnxruntime-web/mobile exactly like
the trained stages - the user's deployment target has no Python backend,
so every stage needs to be an ONNX artifact, trained or not.
"""
from pathlib import Path

import torch

from models import colony_detector, algae_segmenter
from models.bleaching_module import BleachingModule
from pipeline.coral_damage_model import PreprocessedDetector

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = REPO_ROOT / "exports"
INPUT_SIZE = 640
ONNX_OPSET = 17


def export_yolo_stage(load_fn, name: str, num_boxes_dummy: bool = True):
    yolo = load_fn()
    wrapped = PreprocessedDetector(yolo.model).eval()
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)

    out_path = EXPORT_DIR / f"{name}.onnx"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        input_names=["raw_rgb_0_255"],
        output_names=["output"],
        opset_version=ONNX_OPSET,
        dynamic_axes={"raw_rgb_0_255": {0: "batch"}, "output": {0: "batch"}},
        # torch 2.9's default dynamo/torch.export-based exporter chokes on
        # ultralytics' dynamic-shape box-decode (.item() calls trip up its
        # unbacked-symint tracing). The legacy TorchScript-tracing exporter
        # - what ultralytics' own .export(format="onnx") uses internally -
        # handles this model correctly.
        dynamo=False,
    )
    print(f"Exported {out_path}")


def export_bleaching_module():
    module = BleachingModule().eval()
    dummy_rgb = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    dummy_genus_idx = torch.zeros(1, dtype=torch.long)

    out_path = EXPORT_DIR / "bleaching_module.onnx"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    class _Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, rgb, genus_idx):
            out = self.m(rgb, genus_idx)
            return out["paling_score"], out["mask"].float()

    torch.onnx.export(
        _Wrapped(module),
        (dummy_rgb, dummy_genus_idx),
        str(out_path),
        input_names=["rgb_0_255", "genus_idx"],
        output_names=["paling_score", "mask"],
        opset_version=ONNX_OPSET,
        dynamic_axes={"rgb_0_255": {0: "batch"}, "genus_idx": {0: "batch"}},
        dynamo=False,
    )
    print(f"Exported {out_path}")


def main():
    exported_any = False
    try:
        export_yolo_stage(colony_detector.load_best, "colony_detector")
        exported_any = True
    except FileNotFoundError as e:
        print(f"Skipping colony_detector export: {e}")

    try:
        export_yolo_stage(algae_segmenter.load_best, "algae_segmenter")
        exported_any = True
    except FileNotFoundError as e:
        print(f"Skipping algae_segmenter export: {e}")

    try:
        export_bleaching_module()
        exported_any = True
    except (FileNotFoundError, ValueError) as e:
        print(f"Skipping bleaching_module export: {e}")

    if not exported_any:
        print("Nothing exported - train at least one stage first.")


if __name__ == "__main__":
    main()
