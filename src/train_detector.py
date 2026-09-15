"""
train_detector.py
-----------------
Fine-tune YOLO to detect drones. "drone" is NOT a COCO class, so you must train
on a drone dataset. The easiest source is a Roboflow "drone detection" dataset
(or Kaggle), which downloads in YOLO format with a data.yaml.

On your M3 the GPU is used via device="mps". yolov8n (nano) trains comfortably;
for a quick demo even 30-50 epochs on a few thousand images is enough.

Usage:
  1) Get a dataset (one option):
       pip install roboflow
       # then in Python, with a free Roboflow account:
       #   from roboflow import Roboflow
       #   rf = Roboflow(api_key="YOUR_KEY")
       #   ds = rf.workspace("...").project("drone-detection-...").version(1).download("yolov8")
       # this creates a folder with data.yaml
  2) Train:
       python train_detector.py --data /path/to/data.yaml --epochs 50

Output weights land in runs/detect/train*/weights/best.pt  -> pass that to detect_track.py
"""

import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to dataset data.yaml")
    ap.add_argument("--base", default="yolov8n.pt",
                    help="base weights to fine-tune from (downloaded automatically)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="train image size; raise (e.g. 1280) for tiny/distant drones")
    ap.add_argument("--device", default="mps", help="mps / cpu / 0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="runs/detect",
                    help="where Ultralytics should save training runs")
    ap.add_argument("--name", default="train",
                    help="training run name inside --project")
    args = ap.parse_args()

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        project=args.project,
        name=args.name,
        # small-object friendly augmentation
        mosaic=1.0, scale=0.5, fliplr=0.5,
        patience=20,
    )
    # quick validation summary (mAP etc.)
    metrics = model.val()
    print("\nValidation metrics:")
    print(f"  mAP50    : {metrics.box.map50:.3f}")
    print(f"  mAP50-95 : {metrics.box.map:.3f}")
    print("Best weights: runs/detect/train*/weights/best.pt")


if __name__ == "__main__":
    main()
