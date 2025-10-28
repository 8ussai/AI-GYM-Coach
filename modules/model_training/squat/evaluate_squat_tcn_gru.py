from modules.common.paths import MODELS_DIR, get_dataset_dir

import numpy as np
import json
import tensorflow as tf

from pathlib import Path
from typing import Dict, Optional
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
import matplotlib.pyplot as plt

def load_label_map(meta_path: Path) -> Dict[int, str]:
    if not meta_path.exists():
        return {0:"Correct",1:"asymmetry",2:"back_rounding",3:"low_depth",4:"stance_width"}
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    m = meta.get("label_mapping") or {}

    return {int(k): v for k, v in m.items()}

def plot_confusion_matrix(cm: np.ndarray, classes: Dict[int,str], out_path: Path, normalize=True):
    plt.figure(figsize=(7,6))
    if normalize:
        with np.errstate(invalid='ignore'):
            cmn = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cmn = np.nan_to_num(cmn)
        data = cmn
        title = "Confusion Matrix (normalized)"
    else:
        data = cm
        title = "Confusion Matrix (counts)"
    plt.imshow(data, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, [classes[i] for i in range(len(classes))], rotation=30, ha='right')
    plt.yticks(tick_marks, [classes[i] for i in range(len(classes))])
    # write values
    thresh = data.max() / 2. if data.size else 0.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            plt.text(j, i, txt, ha="center", va="center",
                     color="white" if val > thresh else "black", fontsize=9)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_roc_curves(y_true, y_prob, classes: Dict[int,str], out_path: Path):
    n_classes = y_prob.shape[1]
    y_bin = np.zeros_like(y_prob)

    for i, c in enumerate(y_true):
        y_bin[i, c] = 1

    plt.figure(figsize=(7,6))
    macro_aucs = []

    for c in range(n_classes):
        if y_bin[:, c].sum() == 0:
            continue 
        
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        macro_aucs.append(roc_auc)
        plt.plot(fpr, tpr, label=f"{classes.get(c,str(c))} (AUC={roc_auc:.3f})")

    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    plt.plot(fpr_micro, tpr_micro, linestyle="--", label=f"micro-average (AUC={auc_micro:.3f})")
    plt.plot([0,1],[0,1], linestyle=':')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    if macro_aucs:
        plt.title(f"ROC Curves | macro-AUC≈{np.mean(macro_aucs):.3f}")
    else:
        plt.title("ROC Curves")

    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_pr_curves(y_true, y_prob, classes: Dict[int,str], out_path: Path):
    n_classes = y_prob.shape[1]
    y_bin = np.zeros_like(y_prob)
    for i, c in enumerate(y_true):
        y_bin[i, c] = 1

    plt.figure(figsize=(7,6))
    aps = []

    for c in range(n_classes):
        if y_bin[:, c].sum() == 0:
            continue

        precision, recall, _ = precision_recall_curve(y_bin[:, c], y_prob[:, c])
        ap = average_precision_score(y_bin[:, c], y_prob[:, c])
        aps.append(ap)
        plt.plot(recall, precision, label=f"{classes.get(c,str(c))} (AP={ap:.3f})")

    precision_micro, recall_micro, _ = precision_recall_curve(y_bin.ravel(), y_prob.ravel())
    ap_micro = average_precision_score(y_bin.ravel(), y_prob.ravel())
    plt.plot(recall_micro, precision_micro, linestyle="--", label=f"micro-average (AP={ap_micro:.3f})")

    if aps:
        plt.title(f"Precision-Recall | macro-AP≈{np.mean(aps):.3f}")
    else:
        plt.title("Precision-Recall")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_prob_hist(y_prob, out_path: Path):
    conf = np.max(y_prob, axis=1)

    plt.figure(figsize=(6,4))
    plt.hist(conf, bins=20)
    plt.title("Confidence histogram (max softmax)")
    plt.xlabel("confidence")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_learning_curves(report_path: Path, out_path: Path):
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    hist = rep.get("history") or {}
    acc = hist.get("accuracy") or []
    val_acc = hist.get("val_accuracy") or []
    loss = hist.get("loss") or []
    val_loss = hist.get("val_loss") or []
    lr = hist.get("learning_rate") or []
    epochs = range(1, max(len(acc), len(loss)) + 1)
    fig = plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)

    if acc: 
        plt.plot(epochs, acc, label="train")

    if val_acc: 
        plt.plot(range(1, len(val_acc)+1), val_acc, label="val")

    plt.title("Accuracy"); plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend()
    plt.subplot(1,2,2)

    if loss: 
        plt.plot(epochs, loss, label="train")

    if val_loss: 
        plt.plot(range(1, len(val_loss)+1), val_loss, label="val")

    if lr:
        ax2 = plt.gca().twinx()
        ax2.plot(range(1, len(lr)+1), lr, linestyle='--', alpha=0.5)
        ax2.set_ylabel("learning rate")

    plt.title("Loss"); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True

def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass

    ds_dir = get_dataset_dir("squat")

    model_path = MODELS_DIR / "squat" / "squat_tcn_gru.keras"

    meta_path = ds_dir / "squat_metadata.json"

    label_map = load_label_map(meta_path)

    class_names = {i: label_map.get(i, str(i)) for i in range(len(label_map) or 5)}

    X_test = np.load(ds_dir / "X_test.npy")
    y_test = np.load(ds_dir / "y_test.npy")

    n_classes = len(class_names)

    best = tf.keras.models.load_model(model_path)

    test_loss, test_acc = best.evaluate(X_test, y_test, verbose=0)

    y_prob = best.predict(X_test, batch_size=64, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    rep_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))

    eval_dir = MODELS_DIR / "squat" / "eval"

    summary = {
        "model_path": str(model_path),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "classification_report": rep_dict,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(X_test.shape[0]),
        "n_classes": n_classes,
        "class_names": class_names
    }

    with open(eval_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_confusion_matrix(cm, class_names, eval_dir / "confusion_matrix_norm.png", normalize=True)
    plot_confusion_matrix(cm, class_names, eval_dir / "confusion_matrix_counts.png", normalize=False)
    plot_roc_curves(y_test, y_prob, class_names, eval_dir / "roc_curves.png")
    plot_pr_curves(y_test, y_prob, class_names, eval_dir / "pr_curves.png")
    plot_prob_hist(y_prob, eval_dir / "prob_hist.png")
    plot_learning_curves(MODELS_DIR / "squat" / "squat_training_report.json", eval_dir / "learning_curves.png")

if __name__ == "__main__":
    main()
