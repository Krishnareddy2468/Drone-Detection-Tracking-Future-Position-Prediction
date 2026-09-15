"""
detect_track.py
---------------
End-to-end demo: Detection -> Tracking -> Future-position prediction, drawn on a
video. Implements stages 3-7 of the report's flowchart.

  Detection + Tracking : Ultralytics YOLO `.track()` (ByteTrack under the hood)
  Motion features       : per-track centre history + Kalman state
  Prediction            : Kalman (always) + LSTM (if a trained model is given)
  Output                : annotated video with current boxes/IDs and predicted paths

Typical workflow on your M3:
  1) Collect trajectories from one or more videos:
        python detect_track.py --weights drone.pt --source clips/ --collect tracks.json
  2) Train the LSTM on them:
        python -c "import json,lstm_predict as L; L.train(json.load(open('tracks.json')))"
  3) Run the full demo with both predictors:
        python detect_track.py --weights drone.pt --source clip.mp4 \
               --lstm lstm_traj.pt --save out.mp4

Notes:
  * --device mps uses the M3 GPU. Falls back to cpu automatically if unavailable.
  * "drone" is not a COCO class, so --weights must point at a drone-trained model
    (see train_detector.py / README). A stock yolov8n.pt will run but won't find drones.
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

from kalman import KalmanCA

# LSTM is optional; only imported/used if --lstm is provided.
try:
    from lstm_predict import LSTMPredictor
except Exception:
    LSTMPredictor = None


def pick_device(requested):
    if requested:
        return requested
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def draw_path(frame, pts, color, thickness=2):
    """Draw a predicted path as a fading polyline of dots/segments."""
    pts = [(int(x), int(y)) for x, y in pts]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)
    for p in pts[::3]:
        cv2.circle(frame, p, 2, color, -1, cv2.LINE_AA)


def draw_band(frame, pts, sigmas, color):
    """Draw a growing uncertainty envelope as rings around each predicted point.
    sigmas: per-step error (px) from the model; radius grows with the horizon."""
    for (x, y), s in zip(pts, sigmas):
        r = int(max(2, min(s, 80)))
        cv2.circle(frame, (int(x), int(y)), r, color, 1, cv2.LINE_AA)


def behaviour_of(speed, areas, hover_count,
                 hover_speed=1.5, loiter_frames=30, grow=1.20, shrink=0.83):
    """Classify a track's behaviour from its speed and bounding-box area trend.
    Returns (label, new_hover_count)."""
    if speed < hover_speed:
        hover_count += 1
        return ("LOITER" if hover_count >= loiter_frames else "HOVER"), hover_count
    hover_count = 0
    if len(areas) >= 8:
        older = sum(list(areas)[:4]) / 4.0
        recent = sum(list(areas)[-4:]) / 4.0
        if older > 0 and recent > grow * older:
            return "APPROACH", hover_count
        if older > 0 and recent < shrink * older:
            return "DEPART", hover_count
    return "TRANSIT", hover_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="drone-trained YOLO weights (.pt)")
    ap.add_argument("--source", required=True, help="video file, folder, or stream URL")
    ap.add_argument("--lstm", default=None, help="trained LSTM checkpoint (.pt)")
    ap.add_argument("--save", default=None, help="output annotated video path")
    ap.add_argument("--collect", default=None, help="JSON path to dump trajectories instead of predicting")
    ap.add_argument("--device", default=None, help="mps / cpu / 0 (auto if omitted)")
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    ap.add_argument("--horizon", type=int, default=15, help="frames to predict ahead")
    ap.add_argument("--hist", type=int, default=30, help="max centre-history kept per track")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[info] device = {device}")

    model = YOLO(args.weights)

    # Per-track state
    history = defaultdict(lambda: deque(maxlen=max(args.hist, args.horizon + 12)))
    kfs = {}
    collected = defaultdict(list)   # full trajectories for --collect mode
    track_cls = defaultdict(Counter)               # per-track class votes
    area_hist = defaultdict(lambda: deque(maxlen=12))  # per-track box-area trend
    hover_count = defaultdict(int)                 # consecutive low-speed frames
    behaviour = {}                                 # last behaviour label per track
    names = model.names                            # class-index -> name map

    lstm = None
    if args.lstm and not args.collect:
        if LSTMPredictor is None:
            print("[warn] torch/lstm_predict unavailable; skipping LSTM")
        elif not os.path.exists(args.lstm):
            print(f"[warn] LSTM checkpoint not found: {args.lstm}; skipping LSTM")
        else:
            lstm = LSTMPredictor(args.lstm)
            print(f"[info] LSTM loaded (K={lstm.K}, H={lstm.H})")

    writer = None

    # --- real-time instrumentation (rubric: latency / FPS under load) ---
    frame_times = deque(maxlen=30)   # rolling per-frame wall time (s)
    alerted = set()                  # track ids we have already alerted on
    t_last = time.perf_counter()
    total_time, n_frames, first_dt = 0.0, 0, 0.0  # whole-run accumulators

    # Stream results frame-by-frame. ByteTrack is selected via tracker yaml.
    results = model.track(source=args.source, stream=True, persist=True,
                          tracker="bytetrack.yaml", conf=args.conf,
                          device=device, verbose=False)

    for res in results:
        # time spans the previous body + this frame's detect+track step
        t_now = time.perf_counter()
        dt = t_now - t_last
        t_last = t_now
        if n_frames == 0:
            first_dt = dt        # warm-up frame: model load + first inference
        frame_times.append(dt)
        total_time += dt
        n_frames += 1

        frame = res.orig_img.copy()
        boxes = res.boxes

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            clss = (boxes.cls.int().cpu().tolist()
                    if boxes.cls is not None else [0] * len(ids))

            for tid, (x1, y1, x2, y2), cidx in zip(ids, xyxy, clss):
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                # --- motion features: smooth the centre with a Kalman filter ---
                if tid not in kfs:
                    kfs[tid] = KalmanCA(q=1.0, r=4.0)
                sx, sy = kfs[tid].update((cx, cy))
                history[tid].append((sx, sy))
                if args.collect is not None:
                    collected[tid].append((float(cx), float(cy)))

                if args.collect is not None:
                    continue  # collecting only; skip drawing predictions

                # --- classification: vote the detected class over the track ---
                track_cls[tid][cidx] += 1
                maj = track_cls[tid].most_common(1)[0][0]
                try:
                    cls_name = names[maj]
                except Exception:
                    cls_name = str(maj)

                # --- behaviour from speed + box-area trend ---
                area_hist[tid].append(float((x2 - x1) * (y2 - y1)))
                beh, hover_count[tid] = behaviour_of(
                    kfs[tid].speed, area_hist[tid], hover_count[tid])
                if behaviour.get(tid) != beh:
                    print(f"[BEHAVIOUR] track id={tid} ({cls_name}): {beh}")
                    behaviour[tid] = beh

                # --- track confirmation -> alert (defence-system behaviour) ---
                # Fire once per track after it has persisted a few frames, the
                # same temporal-persistence gate the report describes.
                if tid not in alerted and len(history[tid]) >= 5:
                    alerted.add(tid)
                    print(f"[ALERT] confirmed track id={tid} class={cls_name} "
                          f"speed={kfs[tid].speed:.1f}px/f at ({cx:.0f},{cy:.0f})")

                # --- draw current detection: class + id, behaviour below ---
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name} {tid}", (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, beh, (int(x1), int(y2) + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # --- Kalman future path (cyan) ---
                kpath = kfs[tid].predict_future(args.horizon)
                if kpath:
                    draw_path(frame, [(sx, sy)] + kpath, (255, 255, 0), 2)

                # --- LSTM future path (magenta) + uncertainty band ---
                if lstm is not None:
                    lpath = lstm.predict(list(history[tid]))
                    if lpath:
                        draw_path(frame, [(sx, sy)] + lpath, (255, 0, 255), 2)
                        if getattr(lstm, "sigma", None):
                            draw_band(frame, lpath, lstm.sigma, (200, 80, 200))

        # legend + real-time HUD (latency / FPS)
        if args.collect is None:
            cv2.putText(frame, "cyan = Kalman   magenta = LSTM   rings = uncertainty",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if frame_times:
                avg = sum(frame_times) / len(frame_times)
                fps = 1.0 / avg if avg > 0 else 0.0
                cv2.putText(frame, f"{avg * 1000:.0f} ms/frame | {fps:.1f} FPS",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

        # output
        if args.save and writer is None and args.collect is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                     25, (w, h))
        if writer is not None:
            writer.write(frame)

    if writer is not None:
        writer.release()
        print(f"[done] saved annotated video -> {args.save}")

    # real-time summary (drop the first frame: it carries model/warm-up cost)
    if n_frames > 1:
        avg = (total_time - first_dt) / (n_frames - 1)
        print(f"[perf] {n_frames} frames | {avg * 1000:.1f} ms/frame "
              f"| {1.0 / avg if avg > 0 else 0.0:.1f} FPS (device={device})")

    if args.collect is not None:
        # keep only tracks long enough to be useful for training
        traj = [v for v in collected.values() if len(v) >= 25]
        with open(args.collect, "w") as f:
            json.dump(traj, f)
        print(f"[done] saved {len(traj)} trajectories -> {args.collect}")


if __name__ == "__main__":
    main()
