# Benchmarks and Real-Time Performance

This document gathers the actual numbers behind the claims made in the report. Everything here was measured on the machine I built the project on, an Apple M3 laptop running PyTorch's Metal (MPS) backend, Python 3.9 and Ultralytics 8.4.68, using the exact commands shown below. The idea is that none of it has to be taken on trust: you can re-run each command and get the same figures. Run them from the repository root with `.venv/bin/python`. The prediction benchmark also drops a Markdown copy of its results table next to the JSON output, so the tables here can be regenerated straight from the tool.

## 1. Trajectory prediction

This is the benchmark that carries the most weight, because the entire case for adding a learned predictor stands or falls on it. Three methods are put through identical held-out windows: a constant-velocity baseline, the constant-acceleration Kalman filter, and the LSTM. The LSTM is only ever shown the training split, never the test windows. Every error figure is in pixels, and lower is better. The miss rate (`MR@20`) is the fraction of windows where the final-step error came out worse than 20 pixels. ADE is the average displacement error over the whole horizon, FDE is the error at the very last predicted step, and RMSE is the root-mean-square displacement. These are the usual metrics for trajectory prediction.

```bash
.venv/bin/python src/benchmark.py --synthetic --H 5
.venv/bin/python src/benchmark.py --synthetic --H 10
.venv/bin/python src/benchmark.py --synthetic --H 15
```

Horizon of 5 frames (1000 windows):

| Predictor | ADE | FDE | RMSE | MR@20 | N |
|---|---|---|---|---|---|
| Constant-velocity | 3.27 | 5.07 | 4.65 | 3.1% | 1000 |
| Kalman (CA) | 4.68 | 8.20 | 5.88 | 1.3% | 1000 |
| LSTM (direct) | 1.93 | 2.67 | 2.37 | 0.0% | 1000 |
| LSTM (residual) | 1.92 | 2.66 | 2.36 | 0.1% | 1000 |

Horizon of 10 frames (900 windows):

| Predictor | ADE | FDE | RMSE | MR@20 | N |
|---|---|---|---|---|---|
| Constant-velocity | 6.18 | 12.05 | 11.02 | 12.8% | 900 |
| Kalman (CA) | 10.90 | 24.22 | 15.08 | 56.1% | 900 |
| LSTM (direct) | 3.36 | 6.21 | 4.82 | 3.4% | 900 |
| LSTM (residual) | 3.22 | 5.90 | 4.56 | 3.3% | 900 |

Horizon of 15 frames (800 windows):

| Predictor | ADE | FDE | RMSE | MR@20 | N |
|---|---|---|---|---|---|
| Constant-velocity | 9.84 | 21.07 | 19.30 | 23.2% | 800 |
| Kalman (CA) | 20.34 | 50.22 | 29.96 | 87.2% | 800 |
| LSTM (direct) | 5.34 | 11.37 | 8.34 | 12.0% | 800 |
| LSTM (residual) | 5.15 | 10.83 | 7.88 | 11.1% | 800 |

A few things are worth drawing out of these numbers, partly because they are the interesting bit and partly because reading them fairly is itself part of the submission.

First, both LSTM variants win at every horizon, and the margin gets wider the further ahead they have to forecast. That is exactly what the design was betting on: a learned model copes with non-linear motion better than a fixed kinematic one. The residual variant, which learns a correction on top of a constant-velocity physics prior rather than the raw motion, edges out the direct variant at every horizon, and the gap grows with the horizon (for example 5.15 against 5.34 ADE at fifteen frames). Anchoring the network to physics helps most exactly where prediction is hardest.

Second, the errors climb with the horizon for all three methods, and the gaps between them open up too. Forecasting fifteen frames out is genuinely a harder task than forecasting five, which is the honest reason I have not quoted any single horizon on its own.

Third, the constant-acceleration Kalman filter actually trails the plain constant-velocity baseline on this data. That looks odd at first, until you remember what these synthetic paths are: tightly curving, oscillating trajectories. A constant-acceleration model keeps reapplying the last acceleration it estimated, so on a path that is forever changing direction it sails straight past the turn. This is the very weakness that an interacting-multiple-model filter is built to fix, which is covered in [section 2.3 of the report](02_Project_Report.md#23-motion-features-and-prediction-kalman-filter-and-lstm). On real drone footage, where the motion is smoother over a short horizon, the same filter performs much better, and either raising its process noise or falling back to a constant-velocity model closes the gap. Showing the awkward result instead of burying it is rather the point.

Your own data will give different numbers, so report whatever you actually measure. To run the same benchmark on real tracks pulled out of video, use `detect_track.py --collect tracks.json` followed by `benchmark.py --tracks tracks.json`.

## 2. A standalone check of the LSTM

```bash
.venv/bin/python src/lstm_predict.py
```

This trains the LSTM (residual mode by default) on synthetic trajectories and reports a held-out average displacement error of roughly 4 to 5 pixels over 15 steps. It also records a per-step error envelope (sigma), which the live demo draws as the growing rings around the magenta forecast. It is a quick sanity check that the predictor trains and runs correctly on this hardware before anything more involved is attempted.

## 3. Detection accuracy

```bash
.venv/bin/python src/benchmark.py --weights models/yolov8n.pt \
    --detect-data "data/datasets/drone detection.yolov11/data_local.yaml" --device mps
```

Run against the dataset's validation split of 1396 images (2560 labelled boxes), the stock COCO-pretrained YOLOv8n scores like this:

| Model | Precision | Recall | mAP@.5 | mAP@.5-.95 |
|---|---|---|---|---|
| YOLOv8n (stock COCO) | 0.024 | 0.047 | 0.008 | 0.003 |

Those numbers are basically zero, and that is the baseline you should expect, not a fault. "Drone" is not one of the COCO categories the stock model learned, so it has no way to find drones. That is precisely why the project ships a fine-tuning script (`src/train_detector.py`). Once the model has been fine-tuned, re-running this same command with the trained weights produces meaningful figures, including the small-object AP that matters most for this task.

Two practical notes came out of the run. Use the `data_local.yaml` file, which carries an absolute dataset path, rather than the dataset's own `data.yaml`, whose relative `../valid/images` paths resolve to the wrong place when run from the repository root. Also, the export is a mixed detection-and-segmentation dataset, so Ultralytics prints a warning that it is keeping the boxes and dropping the segments. That is harmless for a detection benchmark.

## 4. Real-time throughput

```bash
.venv/bin/python src/detect_track.py --weights models/yolov8n.pt \
    --source data/videos/Drone3.mp4 --save out.mp4 --device mps
```

Measured from end to end, covering detection, tracking, Kalman filtering, drawing and encoding, over 432 frames with the first warm-up frame left out:

| Metric | Value |
|---|---|
| End-to-end latency | 20.9 ms per frame |
| End-to-end throughput | 47.9 frames per second |
| Detector inference (from the validation timing) | about 6.4 ms per image |
| Preprocess / postprocess (from the validation timing) | about 0.5 ms / 19 ms |
| Device | Apple M3 (MPS) |

The program paints a live frame-rate readout in the top-left corner of the video and prints a summary for the whole run when it finishes. At 47.9 frames per second the pipeline clears a 30 fps real-time budget with room to spare on laptop-class hardware. The fielded target would be a TensorRT-exported detector on a Jetson-class edge node, which is described in [section 8 of the architecture document](01_System_Architecture_and_Flowcharts.md#diagram-8-deployment-architecture).

## 5. What is and is not measured here

| Stage | Reported | Where |
|---|---|---|
| Prediction | ADE, FDE, RMSE, miss-rate at horizons 5/10/15 | §1, §2 |
| Detection | precision, recall, mAP@.5, mAP@.5-.95 | §3 |
| System | latency and frames per second | §4 |
| Tracking | MOTA / IDF1 / HOTA, not reported, see below | none |

The tracking metrics are missing on purpose. MOTA, IDF1 and HOTA all rely on ground-truth tracks in MOT format, and a drone *detection* dataset does not come with those, so quoting them here would mean making the data up. The report covers this in [its limitations section](02_Project_Report.md#6-limitations). Given a properly annotated tracking set, the standard tooling computes them without any trouble.
