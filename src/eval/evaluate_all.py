"""One summary table for the three trained/deterministic components' accuracy
(colony detector, algae segmenter, bleaching module) - see the plan's
Evaluation & accuracy metrics section for why each needs its own metric
rather than a single "accuracy %". Each block is independent and skips
cleanly (printing why) if its checkpoint/data isn't available yet, so
this can be run at any point during the build instead of only at the end.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
results_table = []  # (component, metric, value, n)


def eval_colony_detector():
    from models import colony_detector

    try:
        model = colony_detector.load_best()
    except FileNotFoundError as e:
        print(f"[colony_detector] skipped: {e}")
        return
    metrics = model.val(data=str(colony_detector.DATA_YAML), verbose=False)
    results_table.append(("colony_detector", "mAP50", metrics.box.map50, metrics.box.nc))
    results_table.append(("colony_detector", "mAP50-95", metrics.box.map, metrics.box.nc))
    results_table.append(("colony_detector", "recall(mean)", metrics.box.mr, metrics.box.nc))


def eval_algae_segmenter():
    from models import algae_segmenter

    try:
        model = algae_segmenter.load_best()
    except FileNotFoundError as e:
        print(f"[algae_segmenter] skipped: {e}")
        return
    metrics = model.val(data=str(algae_segmenter.DATA_YAML), verbose=False)
    # mask mAP is IoU-thresholded (0.5:0.95) - the closest built-in analogue
    # to the plan's "pixel IoU/Dice" for a trained segmenter.
    results_table.append(("algae_segmenter", "mask_mAP50", metrics.seg.map50, 1))
    results_table.append(("algae_segmenter", "mask_mAP50-95", metrics.seg.map, 1))
    results_table.append(("algae_segmenter", "mask_recall", metrics.seg.mr, 1))


def eval_bleaching_module():
    import csv

    import torch
    from PIL import Image

    from models import colony_detector
    from models.bleaching_module import BleachingModule
    from pipeline.coral_damage_model import CoralDamagePipeline

    manifest_path = REPO_ROOT / "data" / "processed" / "bleaching_reference" / "kaggle_calibration_manifest.csv"
    if not manifest_path.exists():
        print("[bleaching_module] skipped: no calibration manifest (Kaggle set not prepared)")
        return
    try:
        pipeline = CoralDamagePipeline()
    except FileNotFoundError as e:
        print(f"[bleaching_module] skipped: {e}")
        return

    tp = fp = tn = fn = 0
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            findings = pipeline.run(row["image_path"])
            predicted_bleached = any(f_.damage_type == "bleaching" for f_ in findings)
            actual_bleached = row["label"] == "bleached"
            if predicted_bleached and actual_bleached:
                tp += 1
            elif predicted_bleached and not actual_bleached:
                fp += 1
            elif not predicted_bleached and actual_bleached:
                fn += 1
            else:
                tn += 1

    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    results_table.append(("bleaching_module", "precision", precision, n))
    results_table.append(("bleaching_module", "recall", recall, n))
    results_table.append(("bleaching_module", "F1", f1, n))


def main():
    eval_colony_detector()
    eval_algae_segmenter()
    eval_bleaching_module()

    print("\n=== Summary ===")
    print(f"{'component':20s} {'metric':40s} {'value':>8s} {'N':>6s}")
    for component, metric, value, n in results_table:
        print(f"{component:20s} {metric:40s} {value:8.3f} {n:6d}")

    if not results_table:
        print("Nothing to evaluate yet - train at least one component first.")


if __name__ == "__main__":
    main()
