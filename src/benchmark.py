"""
benchmark.py
------------
Quantitative evaluation of the TRAJECTORY-PREDICTION module, which is the part we
can benchmark rigorously on our own harvested tracks. It compares three
predictors on identical held-out windows:

    1. Constant-Velocity (CV)  -- naive physics baseline
    2. Kalman (CA)             -- the physics prior in our hybrid
    3. LSTM                    -- the learned model

Metrics (standard in trajectory-prediction literature):
    ADE  : Average Displacement Error  (mean L2 over the horizon)        [px]
    FDE  : Final  Displacement Error   (L2 at the last predicted step)   [px]
    RMSE : root mean squared displacement                                [px]
    MR@T : Miss Rate -- fraction of windows whose FDE exceeds T px

Lower is better for all. The point of the table is the *comparison*: the LSTM
should beat CV, and beat or match Kalman, especially on curved/accelerating
motion -- which is exactly the report's argument for the hybrid design.

Usage:
    # on real harvested tracks (from detect_track.py --collect tracks.json):
    python benchmark.py --tracks tracks.json
    # quick run on synthetic data (no data needed):
    python benchmark.py --synthetic

Also includes a detection-mAP helper (uses a dataset's val split):
    python benchmark.py --detect-data /path/to/data.yaml --weights best.pt
"""

import argparse
import json
import numpy as np

import lstm_predict as L
from kalman import KalmanCA


# --------------------------------------------------------------------------- #
# Predictors -> each returns H future absolute positions given history p[:t]
# --------------------------------------------------------------------------- #
def predict_cv(history, H, m=3):
    """Constant velocity from the average of the last m steps."""
    p = np.asarray(history, dtype=float)
    if len(p) < 2:
        return None
    v = np.mean(np.diff(p[-(m + 1):], axis=0), axis=0)
    last = p[-1]
    return np.array([last + v * (i + 1) for i in range(H)])


def predict_kalman(history, H, warmup=25):
    p = np.asarray(history, dtype=float)
    kf = KalmanCA(q=1.0, r=4.0)
    for (x, y) in p[-warmup:]:
        kf.update((x, y))
    fut = kf.predict_future(H)
    return np.array(fut) if fut else None


def predict_lstm(model, history, H):
    fut = model.predict(list(history))
    return np.array(fut) if len(fut) == H else None


# --------------------------------------------------------------------------- #
# Metric computation over all valid windows in all trajectories
# --------------------------------------------------------------------------- #
def evaluate(trajectories, predictors, H=15, K=10, stride=5, miss_thresh=20.0):
    """
    predictors: dict name -> callable(history_positions, H) -> (H,2) or None
    Returns dict name -> {ADE, FDE, RMSE, MR, n}
    """
    acc = {name: {"ade": [], "fde": [], "sq": [], "miss": []} for name in predictors}

    for traj in trajectories:
        p = np.asarray(traj, dtype=float)
        L_ = len(p)
        # anchor t: need K+1 history before, H ground-truth after
        for t in range(max(K + 1, 26), L_ - H + 1, stride):
            hist = p[:t]
            gt = p[t:t + H]
            for name, fn in predictors.items():
                pred = fn(hist, H)
                if pred is None or len(pred) != H:
                    continue
                d = np.linalg.norm(pred - gt, axis=1)   # per-step L2
                acc[name]["ade"].append(d.mean())
                acc[name]["fde"].append(d[-1])
                acc[name]["sq"].append((d ** 2).mean())
                acc[name]["miss"].append(1.0 if d[-1] > miss_thresh else 0.0)

    out = {}
    for name, a in acc.items():
        if not a["ade"]:
            continue
        out[name] = {
            "ADE": float(np.mean(a["ade"])),
            "FDE": float(np.mean(a["fde"])),
            "RMSE": float(np.sqrt(np.mean(a["sq"]))),
            "MR": float(np.mean(a["miss"])),
            "n": len(a["ade"]),
        }
    return out


def print_table(results, miss_thresh):
    order = ["Constant-Velocity", "Kalman (CA)", "LSTM (direct)", "LSTM (residual)"]
    names = [n for n in order if n in results] + \
            [n for n in results if n not in order]
    w = max(len(n) for n in names) + 2
    head = f"{'Predictor':<{w}} {'ADE':>8} {'FDE':>8} {'RMSE':>8} {'MR@'+str(int(miss_thresh)):>8} {'N':>7}"
    print("\n" + head)
    print("-" * len(head))
    for n in names:
        r = results[n]
        print(f"{n:<{w}} {r['ADE']:>8.2f} {r['FDE']:>8.2f} {r['RMSE']:>8.2f} "
              f"{r['MR']*100:>7.1f}% {r['n']:>7d}")
    print("\n(units: pixels; MR = miss rate, fraction of windows with FDE > "
          f"{int(miss_thresh)} px; lower is better)")


def markdown_table(results, miss_thresh, H=None):
    """Render the same results as a GitHub-flavoured Markdown table (for the docs)."""
    order = ["Constant-Velocity", "Kalman (CA)", "LSTM (direct)", "LSTM (residual)"]
    names = [n for n in order if n in results] + \
            [n for n in results if n not in order]
    mr = f"MR@{int(miss_thresh)}"
    lines = []
    if H is not None:
        lines.append(f"**Prediction horizon H = {H} frames** (units: pixels; lower is better)\n")
    lines.append(f"| Predictor | ADE | FDE | RMSE | {mr} | N |")
    lines.append("|---|---|---|---|---|---|")
    for n in names:
        r = results[n]
        lines.append(f"| {n} | {r['ADE']:.2f} | {r['FDE']:.2f} | {r['RMSE']:.2f} "
                     f"| {r['MR']*100:.1f}% | {r['n']} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Detection mAP (uses the dataset's own val split)
# --------------------------------------------------------------------------- #
def detection_map(weights, data_yaml, device="mps", imgsz=640):
    from ultralytics import YOLO
    model = YOLO(weights)
    m = model.val(data=data_yaml, device=device, imgsz=imgsz, verbose=False)
    print("\nDetection metrics (val split):")
    print(f"  Precision : {float(m.box.mp):.3f}")
    print(f"  Recall    : {float(m.box.mr):.3f}")
    print(f"  mAP@.5    : {float(m.box.map50):.3f}")
    print(f"  mAP@.5-.95: {float(m.box.map):.3f}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=None, help="harvested trajectories JSON")
    ap.add_argument("--synthetic", action="store_true", help="benchmark on synthetic data")
    ap.add_argument("--H", type=int, default=15, help="prediction horizon (frames)")
    ap.add_argument("--K", type=int, default=10, help="LSTM input window")
    ap.add_argument("--miss", type=float, default=20.0, help="miss-rate threshold (px)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="benchmark_results.json")
    # detection
    ap.add_argument("--detect-data", default=None, help="data.yaml for detection mAP")
    ap.add_argument("--weights", default=None, help="YOLO weights for detection mAP")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    if args.detect_data and args.weights:
        detection_map(args.weights, args.detect_data, device=args.device)

    if args.tracks:
        data = [np.asarray(t, float) for t in json.load(open(args.tracks))]
        print(f"[info] loaded {len(data)} trajectories from {args.tracks}")
    elif args.synthetic:
        data = L.synthetic_trajectories(n=500, length=80, seed=7)
        print(f"[info] generated {len(data)} synthetic trajectories")
    else:
        print("Provide --tracks tracks.json or --synthetic (or detection args). "
              "Nothing to benchmark.")
        return

    # train/test split -- LSTM only ever sees the train split
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(data))
    cut = int(0.8 * len(data))
    train_data = [data[i] for i in idx[:cut]]
    test_data = [data[i] for i in idx[cut:]]
    print(f"[info] train={len(train_data)}  test={len(test_data)}")

    print("[info] training LSTM on the train split (direct + residual)...")
    L.train(train_data, K=args.K, H=args.H, epochs=args.epochs,
            save_path="lstm_bench_direct.pt", mode="direct", verbose=False)
    L.train(train_data, K=args.K, H=args.H, epochs=args.epochs,
            save_path="lstm_bench_resid.pt", mode="residual", verbose=False)
    lstm_d = L.LSTMPredictor("lstm_bench_direct.pt")
    lstm_r = L.LSTMPredictor("lstm_bench_resid.pt")

    predictors = {
        "Constant-Velocity": lambda h, H: predict_cv(h, H),
        "Kalman (CA)":       lambda h, H: predict_kalman(h, H),
        "LSTM (direct)":     lambda h, H: predict_lstm(lstm_d, h, H),
        "LSTM (residual)":   lambda h, H: predict_lstm(lstm_r, h, H),
    }

    print("[info] evaluating on the held-out test split...")
    results = evaluate(test_data, predictors, H=args.H, K=args.K, miss_thresh=args.miss)
    print_table(results, args.miss)

    json.dump(results, open(args.out, "w"), indent=2)
    md_path = args.out.rsplit(".", 1)[0] + ".md"
    with open(md_path, "w") as f:
        f.write(markdown_table(results, args.miss, H=args.H))
    print(f"\n[done] saved -> {args.out} and {md_path}")


if __name__ == "__main__":
    main()
