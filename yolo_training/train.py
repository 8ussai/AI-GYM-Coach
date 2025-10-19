#!/usr/bin/env python3
"""
Minimal YOLO training launcher.

It trains a YOLO model with your dataset and copies the best weight to:
  yolo_training/runs/best.pt

Usage examples:
  python yolo_training/train.py --data yolo_training/data.yaml
  python yolo_training/train.py --data yolo_training/data.yaml --model yolov8n.pt --epochs 100 --imgsz 736
  python yolo_training/train.py --data yolo_training/data.yaml --task detect
  python yolo_training/train.py --data yolo_training/data.yaml --task segment --model yolov8n-seg.pt
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO and export best.pt to yolo_training/runs/best.pt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, default="yolo_training/data.yaml",
        help="Path to data.yaml describing train/val sets and nc/names"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Base model weights (.pt) to start from"
    )
    parser.add_argument(
        "--task", type=str, choices=["detect", "segment"], default="detect",
        help="YOLO task type (detection or segmentation)"
    )
    parser.add_argument(
        "--epochs", type=int, default=80, help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Training image size"
    )
    parser.add_argument(
        "--batch", type=int, default=-1, help="Auto-batch if -1, else set a value"
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="CUDA device id (e.g. '0') or 'cpu'"
    )
    parser.add_argument(
        "--project", type=str, default="yolo_training/runs/train",
        help="Project directory for YOLO runs (exp, exp2, ... will be created here)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Run name (default: auto 'exp', 'exp2', ...)"
    )
    parser.add_argument(
        "--exist-ok", action="store_true",
        help="Allow existing project/name without incrementing"
    )
    return parser.parse_args()


def latest_exp_dir(project_dir: Path) -> Path | None:
    if not project_dir.exists():
        return None
    exps = [p for p in project_dir.iterdir() if p.is_dir() and p.name.startswith("exp")]
    if not exps:
        return None
    exps.sort(key=lambda p: p.stat().st_mtime)
    return exps[-1]


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parents[1]  # .../yolo_training
    project_dir = (base_dir / args.project).resolve() if not str(args.project).startswith(str(base_dir)) else Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    data_path = (base_dir / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] data.yaml not found: {data_path}")
        raise SystemExit(1)

    # Print config for sanity
    print("======== YOLO Training Config ========")
    print(f"time          : {datetime.now().isoformat(timespec='seconds')}")
    print(f"task          : {args.task}")
    print(f"data          : {data_path}")
    print(f"model         : {args.model}")
    print(f"epochs        : {args.epochs}")
    print(f"imgsz         : {args.imgsz}")
    print(f"batch         : {args.batch}")
    print(f"device        : {args.device or '(auto)'}")
    print(f"project dir   : {project_dir}")
    print(f"name          : {args.name or '(auto exp/exp2...)'}")
    print("======================================")

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[ERROR] Failed to import ultralytics. Install it via: pip install ultralytics")
        raise

    # Load model
    print("[INFO] Loading base model...")
    model = YOLO(args.model)

    # Choose task-specific overrides if needed
    overrides = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_dir),
        name=args.name,
        exist_ok=args.exist_ok,
        # You can add more overrides here (lr0, patience, optimizer, etc.)
    )

    print("[INFO] Starting training...")
    # For ultralytics>=8, .train returns a Results object but we rely on filesystem for robustness.
    model.train(**overrides)

    # Resolve latest exp directory after training
    run_dir = latest_exp_dir(project_dir)
    if not run_dir:
        print("[ERROR] Could not locate the YOLO run directory.")
        raise SystemExit(2)

    best_src = run_dir / "weights" / "best.pt"
    if not best_src.exists():
        print(f"[ERROR] best.pt not found at: {best_src}")
        raise SystemExit(3)

    # Copy to a stable path: yolo_training/runs/best.pt
    stable_runs_dir = base_dir / "runs"
    stable_runs_dir.mkdir(parents=True, exist_ok=True)
    best_dst = stable_runs_dir / "best.pt"
    shutil.copy2(best_src, best_dst)

    print("[SUCCESS] Training finished.")
    print(f"[INFO] Latest run directory : {run_dir}")
    print(f"[INFO] Best weight (source) : {best_src}")
    print(f"[INFO] Best weight (copied) : {best_dst}")
    print("[INFO] You can now reference this weight in your code.")


if __name__ == "__main__":
    main()
