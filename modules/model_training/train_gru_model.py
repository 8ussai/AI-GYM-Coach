#!/usr/bin/env python3
# modules/model_training/train_gru_model.py
"""
Train a GRU model per exercise on the per-rep sequences.

Reads:
  - outputs/datasets/<exercise>/{train,val}/X.npy, y.npy
  - outputs/models/<exercise>_meta.json  (optional for feature info)

Writes:
  - outputs/models/<exercise>_model.h5
  - outputs/models/<exercise>_history.json

Usage:
  python -m modules.model_training.train_gru_model --exercise squat --epochs 30 --batch-size 64 --use-class-weights --verbose
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np

from modules.common.paths import get_dataset_dir, get_models_dir

# Silence TF logs a bit (optional)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# If you want deterministic cuDNN, you can set seeds outside TF too.

from tensorflow import keras
from tensorflow.keras import layers


def parse_args():
    p = argparse.ArgumentParser(
        description="Train GRU model for an exercise",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--exercise", type=str, required=True, help="Exercise key (e.g., squat)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--gru-units", type=int, default=64)
    p.add_argument("--use-class-weights", action="store_true", help="Compute class weights from train labels")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_data(exercise: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    ds = get_dataset_dir(exercise)
    X_tr = np.load(ds / "train" / "X.npy")
    y_tr = np.load(ds / "train" / "y.npy")
    X_va = np.load(ds / "val" / "X.npy")
    y_va = np.load(ds / "val" / "y.npy")
    return (X_tr, y_tr), (X_va, y_va)


def build_model(input_shape, units: int, dropout: float, lr: float) -> keras.Model:
    """
    input_shape: (T, F)
    """
    inp = layers.Input(shape=input_shape, name="input_seq")
    x = layers.GRU(units, return_sequences=True, name="gru_1")(inp)
    x = layers.Dropout(dropout, name="drop_1")(x)
    x = layers.GRU(units, name="gru_2")(x)
    x = layers.Dropout(dropout, name="drop_2")(x)
    x = layers.Dense(64, activation="relu", name="fc_1")(x)
    x = layers.Dropout(dropout, name="drop_3")(x)
    out = layers.Dense(1, activation="sigmoid", name="out")(x)

    model = keras.Model(inp, out, name="gru_rep_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Simple inverse-frequency class weights for binary labels {0,1}.
    """
    from collections import Counter
    c = Counter(y.tolist())
    total = float(sum(c.values()))
    n_classes = max(1, len(c))
    weights = {cls: total / (n_classes * cnt) for cls, cnt in c.items() if cnt > 0}
    # Ensure both classes exist in the dict
    if 0 not in weights:
        weights[0] = 1.0
    if 1 not in weights:
        weights[1] = 1.0
    return {int(k): float(v) for k, v in weights.items()}


def main():
    args = parse_args()
    exercise = args.exercise.strip().lower()

    (X_tr, y_tr), (X_va, y_va) = load_data(exercise)
    input_shape = X_tr.shape[1:]  # (T, F)

    if args.verbose:
        print("======== Train GRU ========")
        print(f"exercise       : {exercise}")
        print(f"X_tr shape     : {X_tr.shape}, y_tr={y_tr.shape} (pos={int((y_tr==1).sum())}, neg={int((y_tr==0).sum())})")
        print(f"X_va shape     : {X_va.shape}, y_va={y_va.shape} (pos={int((y_va==1).sum())}, neg={int((y_va==0).sum())})")
        print(f"units/dropout  : {args.gru_units}/{args.dropout}")
        print(f"epochs/lr      : {args.epochs}/{args.lr}")
        print(f"class weights  : {args.use_class_weights}")
        print("===========================")

    model = build_model(input_shape, units=args.gru_units, dropout=args.dropout, lr=args.lr)

    models_dir = get_models_dir(exercise)
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / f"{exercise}_model.h5"

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        ),
    ]

    class_weight = None
    if args.use_class_weights:
        class_weight = compute_class_weights(y_tr)
        if args.verbose:
            print(f"[INFO] Class weights: {class_weight}")

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    # Save best model already handled by ModelCheckpoint
    # Save history
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (models_dir / f"{exercise}_history.json").write_text(json.dumps(hist, indent=2))

    # Final log
    print(f"[SUCCESS] Best model saved to: {ckpt_path}")
    print(f"[SUCCESS] Training history saved to: {models_dir / f'{exercise}_history.json'}")

if __name__ == "__main__":
    main()
