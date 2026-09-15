# Drone Detection, Tracking & Future-Position Prediction

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8-00b0ff.svg)](https://docs.ultralytics.com/)

A runnable counter-UAS perception pipeline: **YOLO detection → ByteTrack tracking →
hybrid Kalman + LSTM trajectory prediction**, drawn frame-by-frame onto a video with a
live latency HUD and track-confirmation alerts.

```
input video ──► YOLO.track (ByteTrack) ──► per-track centre history
                                              │
                                  ┌───────────┴───────────┐
                                  ▼                       ▼
                          Kalman (CA) future      LSTM future path
                          path  (cyan)            (magenta + uncertainty rings)
                                  └───────────┬───────────┘
                                              ▼
                                  annotated video / out.mp4
```

The design rationale, the Kalman maths, the measured benchmarks and the honest
limitations all live in [docs/](docs/) — see [Documentation](#documentation).

---

## Quick start

No dataset or video needed for this — it trains the predictor on synthetic
trajectories and prints a held-out error:

```bash
git clone <this-repo-url> drone_system && cd drone_system
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/lstm_predict.py        # smoke test  -> held-out ADE in pixels
python src/benchmark.py --synthetic --H 15   # full comparison table
```

Expected benchmark output (synthetic data, horizon = 15 frames, pixels, lower is
better):

```
Predictor                ADE      FDE     RMSE    MR@20       N
---------------------------------------------------------------
Constant-Velocity       9.84    21.07    19.30    23.2%     800
Kalman (CA)            20.34    50.22    29.96    87.2%     800
LSTM (direct)           5.34    11.37     8.34    12.0%     800
LSTM (residual)         5.15    10.83     7.88    11.1%     800
```

For the full end-to-end run on real video, follow the
[Demo Guide](docs/04_Demo_Guide.md).

---

## Getting the data and weights

Videos, the YOLO training dataset and the fine-tuned detector weights are **not tracked
in git** — together they are several hundred megabytes. Everything the code needs to
regenerate them is here:

| Asset | Where it goes | How to get it |
|---|---|---|
| Base YOLO weights (`yolov8n.pt`, ~6 MB) | `models/` or CWD | Downloaded automatically by Ultralytics on first use, or [from the Ultralytics release page](https://github.com/ultralytics/assets/releases) |
| Drone-detection dataset (YOLO format, with `data.yaml`) | `data/datasets/<name>/` | A free [Roboflow](https://universe.roboflow.com/search?q=drone+detection) or Kaggle "drone detection" export. `drone` is not a COCO class, so a detector must be fine-tuned. |
| Test clips | `data/videos/` | Any drone footage. The docs were written against four clips (`Drone3.mp4`, `Drone 2.mp4`, `DroneVideo1.mp4`, `Birds.mp4`). |
| Fine-tuned detector (`best.pt`) | `runs/detect/train*/weights/` | Produced by `src/train_detector.py` (Step 1 of the demo guide) |

Two trained LSTM checkpoints **are** tracked (`models/lstm_traj.pt`,
`models/lstm_traj_free.pt`, ~80 KB each), so the prediction half of the pipeline runs
immediately after clone.

If a Roboflow export's `data.yaml` uses relative paths that misresolve from the repo
root, write a sibling `data_local.yaml` with an absolute `path:` — see the demo guide's
troubleshooting table.

---

## Project structure

```
drone_system/
├── src/                  # all Python source (flat — the modules import each other)
│   ├── kalman.py         # constant-acceleration Kalman filter
│   ├── lstm_predict.py   # LSTM trajectory model: train / infer / synthetic test
│   ├── detect_track.py   # main pipeline: YOLO+ByteTrack -> Kalman & LSTM -> video
│   ├── train_detector.py # fine-tune YOLO on a drone dataset
│   └── benchmark.py      # ADE/FDE/RMSE vs baselines; detection mAP
├── models/               # tracked LSTM checkpoints (YOLO weights are downloaded)
├── docs/                 # architecture, report, benchmarks, runbook
├── data/                 # videos + datasets            (git-ignored, see above)
├── runs/                 # Ultralytics training output   (git-ignored)
├── outputs/              # annotated videos & plots      (git-ignored)
└── requirements.txt
```

| File | Role |
|------|------|
| [src/kalman.py](src/kalman.py) | Constant-acceleration Kalman filter — smoothing plus short-horizon extrapolation. The physics half of the hybrid. |
| [src/lstm_predict.py](src/lstm_predict.py) | LSTM over displacement deltas. Two modes: `direct` (predict future deltas) and `residual` (predict the correction on a constant-velocity prior). Stores a per-step error envelope for the uncertainty rings. |
| [src/detect_track.py](src/detect_track.py) | The pipeline: detection, ByteTrack association, per-track motion features, both forecasts, behaviour classification (HOVER / LOITER / APPROACH / DEPART / TRANSIT), alerts, and the latency HUD. |
| [src/train_detector.py](src/train_detector.py) | YOLO fine-tuning wrapper with small-object-friendly augmentation. |
| [src/benchmark.py](src/benchmark.py) | Held-out comparison of CV / Kalman / LSTM at a given horizon; emits JSON **and** a Markdown table for the docs. Also computes detection mAP. |

---

## Documentation

| Doc | Contents |
|-----|----------|
| [01_System_Architecture_and_Flowcharts.md](docs/01_System_Architecture_and_Flowcharts.md) | System architecture + 9 Mermaid flowcharts (context, pipeline, detection, tracking, lifecycle, prediction, sensor-sync, deployment, latency budget) + traceability table |
| [02_Project_Report.md](docs/02_Project_Report.md) | Model justification, Kalman math, IMM+LSTM, edge cases, metrics, real-time, limitations, references |
| [03_Benchmarks_and_RealTime.md](docs/03_Benchmarks_and_RealTime.md) | Measured results: prediction (ADE/FDE/RMSE @ H=5/10/15), detection mAP, latency/FPS |
| [04_Demo_Guide.md](docs/04_Demo_Guide.md) | Step-by-step runbook + troubleshooting |
| [05_Architecture_Walkthrough_Script.md](docs/05_Architecture_Walkthrough_Script.md) | Narration script for walking through the architecture |

---

## Results at a glance

Measured on an Apple M3 (MPS backend). Full detail and the exact commands are in
[03_Benchmarks_and_RealTime.md](docs/03_Benchmarks_and_RealTime.md).

**Prediction** — the LSTM beats both baselines at every horizon, and the gap widens as
the horizon grows (H=15, synthetic, pixels):

| Predictor | ADE | FDE | RMSE | MR@20 |
|---|---|---|---|---|
| Constant-velocity | 9.84 | 21.07 | 19.30 | 23.2% |
| Kalman (CA) | 20.34 | 50.22 | 29.96 | 87.2% |
| LSTM (direct) | 5.34 | 11.37 | 8.34 | 12.0% |
| **LSTM (residual)** | **5.15** | **10.83** | **7.88** | **11.1%** |

**Real-time** — 20.9 ms/frame end to end, 47.9 FPS on an M3, comfortably above the
25 FPS the input clips run at.

Two caveats stated up front rather than buried:

- The constant-acceleration Kalman **over-extrapolates** on highly oscillatory synthetic
  paths (it compounds instantaneous acceleration), which is why it trails plain
  constant-velocity here. On real drone tracks, which are smoother over short horizons,
  it is far more competitive — raise process noise `q` or drop to a CV model.
- Tracking metrics (MOTA / IDF1 / HOTA) are **not** reported. They need MOT-format
  ground-truth tracks, which drone-detection datasets don't ship. Stating that beats
  fabricating numbers.

---

## How this maps to the design docs

- **Detection / small targets** — YOLO at higher `--imgsz`; the report adds tiling/SAHI,
  a P2 head and IR fusion as the production path.
- **Tracking** — `tracker="bytetrack.yaml"`. ByteTrack associates low-confidence
  detections, which is exactly why it suits distant drones.
- **Motion features** — per-track centre history plus Kalman state (position, velocity,
  acceleration, speed).
- **Prediction** — `kalman.py` is the physics prior; `lstm_predict.py` is the learned
  non-linear part. The report's "residual on top of physics" variant is `mode="residual"`.
- **Edge cases** — high speed: `KalmanCA.speed` widens the search gate; sudden motion:
  raise process noise `q`; false positives: ByteTrack's track confirmation requires a
  detection to persist before it becomes a stable ID.

## Known limitations

- Works in **pixel** coordinates, not metric world coordinates. Converting to metres
  needs camera calibration and altitude (covered in the report).
- The LSTM is the basic variant (future deltas, single-shot). The report describes the
  fuller Seq2Seq + attention + calibrated-uncertainty model as the upgrade path.
- Vision-only. A real ~1 km counter-UAS system fuses EO/IR with radar and RF; this
  demonstrates the ML pipeline, not a fielded system.
# Drone-Detection-Tracking-Future-Position-Prediction
