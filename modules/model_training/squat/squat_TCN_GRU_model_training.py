from modules.common.paths import MODELS_DIR, get_dataset_dir

import numpy as np
import tensorflow as tf
import json

from tensorflow.keras import layers as L
from pathlib import Path
from collections import Counter


def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_metadata(meta_path: Path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cfg = meta.get("config", {})
    stats = meta.get("dataset_stats", {})
    norm = meta.get("normalization_params", {})
    label_map = meta.get("label_mapping", {})

    out = {
        "seq_len": int(stats.get("sequence_length", 40)),
        "num_features": int(stats.get("num_features", 7)),
        "num_classes": int(stats.get("num_classes", 5)),
        "feature_names": meta.get("feature_names", cfg.get("feature_cols", [])),
        "mean": np.array(norm.get("mean", []), dtype=np.float32),
        "std":  np.array(norm.get("std",  []), dtype=np.float32),
        "label_map": {int(k): v for k, v in label_map.items()}
    }
    return out

def compute_class_weights(y):
    classes, counts = np.unique(y.astype(np.int64), return_counts=True)
    total = y.shape[0]
    return {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}

def tcn_block(x, filters=64, kernel_size=3, dilation_rate=1, dropout=0.2, norm="layer"):
    Norm = L.LayerNormalization if norm == "layer" else L.BatchNormalization

    y = L.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate, activation=None)(x)
    y = Norm()(y)
    y = L.Activation("relu")(y)
    y = L.SpatialDropout1D(dropout)(y)
    y = L.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate, activation=None)(y)
    y = Norm()(y)
    y = L.Activation("relu")(y)
    y = L.SpatialDropout1D(dropout)(y)

    in_ch = int(x.shape[-1])
    if in_ch != filters:
        shortcut = L.Conv1D(filters, 1, padding="same")(x)
    else:
        shortcut = x

    return L.Activation("relu")(L.Add()([y, shortcut]))

def build_model(input_length, n_features, n_classes):
    inp = L.Input(shape=(input_length, n_features))
    x = inp
    x = tcn_block(x, filters=64, kernel_size=3, dilation_rate=1, dropout=0.2, norm="layer")
    x = tcn_block(x, filters=64, kernel_size=3, dilation_rate=2, dropout=0.2, norm="layer")
    x = tcn_block(x, filters=64, kernel_size=3, dilation_rate=4, dropout=0.2, norm="layer")
    x = tcn_block(x, filters=64, kernel_size=3, dilation_rate=8, dropout=0.2, norm="layer")
    x = L.GRU(64, return_sequences=False)(x)
    x = L.Dropout(0.3)(x)
    out = L.Dense(n_classes, activation="softmax",dtype="float32")(x)
    model = tf.keras.Model(inp, out)
    return model

def main():

    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    ensure_dirs(MODELS_DIR / "squat")

    meta = load_metadata(get_dataset_dir("squat") / "squat_metadata.json")
    seq_len      = meta["seq_len"]
    num_features = meta["num_features"]
    num_classes  = meta["num_classes"]

    X_train = np.load(get_dataset_dir("squat") / "X_train.npy")
    y_train = np.load(get_dataset_dir("squat") / "y_train.npy")
    X_val   = np.load(get_dataset_dir("squat") / "X_val.npy")
    y_val   = np.load(get_dataset_dir("squat") / "y_val.npy")
    X_test  = np.load(get_dataset_dir("squat") / "X_test.npy")
    y_test  = np.load(get_dataset_dir("squat") / "y_test.npy")

    model = build_model(seq_len, num_features, num_classes)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    ckpt_path = MODELS_DIR / "squat" / "squat_tcn_gru.keras"
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_accuracy", mode="max",
                                           save_best_only=True, save_weights_only=False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
    ]

    class_weights = compute_class_weights(y_train)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    best = tf.keras.models.load_model(ckpt_path)
    test_loss, test_acc = best.evaluate(X_test, y_test, verbose=0)

    report = {
        "label_map": meta["label_map"],
        "class_weights": class_weights,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "test": {"loss": float(test_loss), "accuracy": float(test_acc), "support": Counter([int(t) for t in y_test])}
    }
    with open(MODELS_DIR / "squat" / "squat_training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()