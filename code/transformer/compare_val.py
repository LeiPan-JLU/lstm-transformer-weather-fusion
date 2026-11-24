"""Print a Markdown comparison of explicitly selected validation predictions (degrees C)."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import PREDS_DIR, SEQ_OUT
from common.metrics import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, nargs="+", help="*_val.npy files in display order")
    parser.add_argument("--truth", type=Path, default=PREDS_DIR / "y_val.npy",
                        help="ground truth (default: outputs/v3/preds/y_val.npy)")
    args = parser.parse_args()
    if args.truth.name.lower().endswith("_test.npy"):
        parser.error(f"--truth: {args.truth} is a test file; use validation ground truth")
    truth_path = args.truth.resolve()
    truth = np.load(truth_path)
    if truth.ndim != 2 or truth.shape[1] != SEQ_OUT or len(truth) == 0:
        parser.error(f"{args.truth}: expected [N, {SEQ_OUT}] with N > 0, got {truth.shape}")
    horizons = (1, 6, 12, 24)
    headers = ["Prediction", "MAE", "RMSE", *(f"h{h} MAE" for h in horizons)]
    rows = [headers, ["---"] * len(headers)]
    for path in args.predictions:
        if not path.name.endswith("_val.npy"):
            parser.error(f"{path}: expected a *_val.npy prediction")
        if path.name.lower() == "y_val.npy" or path.resolve() == truth_path:
            parser.error(f"{path}: ground truth cannot be used as a prediction")
        pred = np.load(path)
        if pred.shape != truth.shape:
            parser.error(f"{path}: expected {truth.shape} to match {args.truth}, got {pred.shape}")
        metrics = evaluate(pred, truth)
        values = [metrics["MAE"], metrics["RMSE"],
                  *(evaluate(pred[:, h-1], truth[:, h-1])["MAE"] for h in horizons)]
        rows.append([path.as_posix(), *(f"{value:.4f}" for value in values)])
    for row in rows:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
