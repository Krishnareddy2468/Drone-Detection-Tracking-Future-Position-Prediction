# Demo Guide (Runbook)

How to run the software demo end to end. Run every command **from the repo root** so
the relative `data/` and `models/` paths resolve.

> **Before you start:** videos, the YOLO dataset and the base YOLO weights are *not*
> tracked in git (they are hundreds of megabytes). See
> [Getting the data and weights](../README.md#getting-the-data-and-weights) in the
> README first. Steps 0 and 5 (`--synthetic`) need no data at all.

For the *why* behind each stage see [02_Project_Report.md](02_Project_Report.md); for
measured numbers see [03_Benchmarks_and_RealTime.md](03_Benchmarks_and_RealTime.md).

---

## Setup (Apple Silicon / M3)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon the GPU is used automatically via PyTorch's **MPS** backend
(`--device mps`). There is no CUDA on Mac — that is expected. On an NVIDIA box pass
`--device 0`; on anything else `--device cpu`. Inference is fast; training a nano model
is fine for a demo.

All commands below assume that venv is activated (`source .venv/bin/activate`).

---

## Step 0 — sanity check (no data needed)

```bash
python src/lstm_predict.py
```

Trains the LSTM on synthetic trajectories and prints a held-out ADE (~4.5 px). Confirms
PyTorch + MPS + the learned predictor all work.

## Step 1 — get a drone detector

`drone` is **not** a COCO class, so stock weights won't find drones. Fine-tune on a
drone dataset in YOLO format (see the README for where to get one):

```bash
python src/train_detector.py \
    --data "data/datasets/drone detection.yolov11/data_local.yaml" \
    --epochs 50 --device mps
# best weights -> runs/detect/train*/weights/best.pt
```

Tip: for tiny/distant drones, train at higher resolution (`--imgsz 1280`). Use
`data_local.yaml` (absolute paths), not the shipped `data.yaml` — see Troubleshooting.

## Step 2 — collect trajectories (to train the LSTM on real motion)

```bash
python src/detect_track.py --weights runs/detect/train/weights/best.pt \
    --source data/videos/ --collect tracks.json --device mps
```

## Step 3 — train the LSTM on the harvested tracks

```bash
python -c "import sys; sys.path.insert(0,'src'); import json, lstm_predict as L; L.train(json.load(open('tracks.json')))"
# -> lstm_traj.pt
```

(Few tracks? Start from the synthetic model in Step 0 and fine-tune, or lower `K`/`H`.)

## Step 4 — full demo

```bash
python src/detect_track.py --weights runs/detect/train/weights/best.pt \
    --source data/videos/Drone3.mp4 --lstm models/lstm_traj.pt --save out.mp4 --device mps
```

**What to look for in `out.mp4` / the console:**

- Each drone: a green box + track ID.
- A **cyan** path = Kalman (CA) forecast; a **magenta** path = LSTM forecast, projecting
  ahead of the target.
- Top-left HUD: the legend **and** a live `"… ms/frame | … FPS"` readout.
- Console: `[ALERT] confirmed track id=…` once a track persists ≥5 frames (the
  confirmation gate), and a final `[perf] … FPS` whole-run summary.

> Quick smoke test without a trained detector: substitute `--weights models/yolov8n.pt`.
> Boxes will be sparse (stock weights don't know "drone"), but the pipeline, HUD and
> alerts all run — this is exactly how the real-time numbers in
> [§4 of the benchmarks doc](03_Benchmarks_and_RealTime.md#4-real-time-throughput) were captured.

## Step 5 — benchmark

```bash
# synthetic, no data needed (writes JSON + Markdown table):
python src/benchmark.py --synthetic --H 15
# on your harvested tracks:
python src/benchmark.py --tracks tracks.json --H 15
# detection mAP (needs a trained model for non-trivial numbers):
python src/benchmark.py --weights runs/detect/train/weights/best.pt \
    --detect-data "data/datasets/drone detection.yolov11/data_local.yaml" --device mps
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `val: ... images not found` during mAP | The shipped `data.yaml` uses relative `../valid/images`, which misresolves from the repo root. Use **`data_local.yaml`** (absolute `path:`), or run with the dataset dir as the working directory. |
| Red squiggles on `from ultralytics import YOLO` in the editor | Wrong interpreter selected — point VS Code at `./.venv/bin/python` (Cmd+Shift+P → *Python: Select Interpreter*). Not a runtime error. |
| `[warn] LSTM checkpoint not found … skipping LSTM` | Pass a valid `--lstm path.pt`; only the cyan Kalman path draws without it. |
| LSTM trains poorly / refuses | Needs trajectories ≥ `K+H` long; `--collect` keeps tracks ≥25 frames. With few real tracks, start from the synthetic model. |
| Slow / falls back to CPU | MPS unavailable in the venv; the device picker drops to CPU automatically (`pick_device`). Throughput drops accordingly. |
| `FileNotFoundError` on `data/...` | Those assets are not in git. See [Getting the data and weights](../README.md#getting-the-data-and-weights). |
| `detect-segment mixed dataset` warning | The Roboflow export ships segments + boxes; Ultralytics uses boxes only. Harmless for detection. |

---

## Files at a glance

| Path | Role |
|---|---|
| `src/detect_track.py` | Main pipeline: detect → track → Kalman+LSTM → annotated video, HUD, alerts |
| `src/kalman.py` | Constant-acceleration Kalman filter (smoothing + H-step forecast) |
| `src/lstm_predict.py` | LSTM trajectory model: train / infer / synthetic smoke test |
| `src/train_detector.py` | Fine-tune YOLO on the drone dataset |
| `src/benchmark.py` | ADE/FDE/RMSE prediction benchmark + detection mAP; emits Markdown tables |
| `models/lstm_traj.pt`, `models/lstm_traj_free.pt` | Trained LSTM checkpoints (tracked — ~80 KB each) |
| `models/yolov8n.pt` | Base YOLO weights — *not tracked*; ultralytics downloads it on first use |
| `data/videos/` | Demo clips — *not tracked*; bring your own (see README) |
| `data/datasets/drone detection.yolov11/` | YOLO-format dataset (+ `data_local.yaml`) — *not tracked* |
| `runs/`, `outputs/` | Generated by training / demo runs — *not tracked* |
