#!/usr/bin/env python3
# modules/model_training/evaluate.py
"""
Evaluate trained model on the test split.

Reads:
  - outputs/datasets/<exercise>/test/{X.npy,y.npy}
  - outputs/models/<exercise>_model.h5

Writes:
  - outputs/models/<exercise>_metrics.json

Usage:
  python -m modules.model_training.evaluate --exercise squat
"""

import argparse
import json
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from tensorflow import keras

from modules.common.paths import get_dataset_dir, get_models_dir

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate trained model on test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--exercise", type=str, required=True)
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for sigmoid output")
    return p.parse_args()

def main():
    args = parse_args()
    exercise = args.exercise.strip().lower()

    ds = get_dataset_dir(exercise)
    X_te = np.load(ds / "test" / "X.npy")
    y_te = np.load(ds / "test" / "y.npy")

    model_path = get_models_dir(exercise) / f"{exercise}_model.h5"
    model = keras.models.load_model(model_path)

    probs = model.predict(X_te, verbose=0).reshape(-1)
    preds = (probs >= args.threshold).astype(int)

    acc = float(accuracy_score(y_te, preds))
    try:
        auc = float(roc_auc_score(y_te, probs))
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_te, preds).tolist()
    report = classification_report(y_te, preds, output_dict=True)

    metrics = {
        "accuracy": acc,
        "auc": auc,
        "threshold": args.threshold,
        "confusion_matrix": cm,
        "classification_report": report
    }

    out_path = get_models_dir(exercise) / f"{exercise}_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[RESULT] accuracy={acc:.4f} auc={auc:.4f}")
    print(f"[SUCCESS] Metrics saved to: {out_path}")

if __name__ == "__main__":
    main()
