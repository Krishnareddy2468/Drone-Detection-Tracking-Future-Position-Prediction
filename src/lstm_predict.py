"""
lstm_predict.py
---------------
The learned half of the hybrid predictor. A small LSTM that, given the last K
displacement steps of a drone, predicts the next H steps. We work in DISPLACEMENT
(delta) space rather than absolute pixels so the model is translation-invariant
and generalises across the frame.

Two prediction modes:
    "direct"   : the LSTM outputs the future deltas directly (the basic model).
    "residual" : the LSTM outputs the CORRECTION on top of a constant-velocity
                 physics prior, i.e. future = (CV prior) + (LSTM residual).
                 This is the residual-on-physics design from the report: the
                 prior answers "where simple physics says it goes" and the
                 network only learns the deviation, which is easier to fit and
                 keeps the forecast physically anchored.

Each trained model also stores a per-step error envelope (`sigma`, one value per
horizon step in pixels) measured on a held-out split, so the demo can draw a
growing uncertainty band around the forecast.

Pipeline role:
    tracker -> per-track position history -> (this) LSTM -> future path (+ sigma)

Runs on Apple Silicon via device="mps".
"""

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- #
# Device helper (M3 -> mps, else cpu)
# ----------------------------------------------------------------------------- #
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------- #
# Model
# ----------------------------------------------------------------------------- #
class TrajLSTM(nn.Module):
    def __init__(self, hidden=64, layers=1, horizon=15):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden,
                            num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, horizon * 2)

    def forward(self, x):                     # x: (B, K, 2) input deltas
        _, (h, _) = self.lstm(x)              # h: (layers, B, hidden)
        out = self.head(h[-1])                # (B, H*2)
        return out.view(-1, self.horizon, 2)  # (B, H, 2) predicted (deltas or residual)


# ----------------------------------------------------------------------------- #
# Data preparation
# ----------------------------------------------------------------------------- #
def trajectories_to_deltas(traj):
    """traj: list/array of (x, y) -> array of (dx, dy) of length len-1."""
    traj = np.asarray(traj, dtype=float)
    return np.diff(traj, axis=0)


def make_windows(trajectories, K=10, H=15):
    """
    trajectories: list of [(x,y), ...] sequences (one per tracked drone).
    Returns X (N,K,2) input deltas and Y (N,H,2) target future deltas.
    """
    X, Y = [], []
    for traj in trajectories:
        d = trajectories_to_deltas(traj)
        if len(d) < K + H:
            continue
        for i in range(len(d) - K - H + 1):
            X.append(d[i:i + K])
            Y.append(d[i + K:i + K + H])
    if not X:
        return np.zeros((0, K, 2)), np.zeros((0, H, 2))
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def _cv_prior(input_deltas, H, m=3):
    """Constant-velocity prior: mean of the last m input deltas, repeated H times.
    input_deltas: (..., K, 2). Returns (..., H, 2)."""
    k = input_deltas.shape[-2]
    step = input_deltas[..., -min(m, k):, :].mean(axis=-2, keepdims=True)  # (...,1,2)
    return np.repeat(step, H, axis=-2)                                     # (...,H,2)


# ----------------------------------------------------------------------------- #
# Training
# ----------------------------------------------------------------------------- #
def train(trajectories, K=10, H=15, hidden=64, epochs=60, lr=1e-3,
          batch=64, save_path="lstm_traj.pt", mode="residual",
          val_frac=0.1, verbose=True):
    device = get_device()
    X, Y = make_windows(trajectories, K, H)
    if len(X) == 0:
        raise RuntimeError("Not enough trajectory data to build training windows. "
                           "Collect more tracks (longer videos) or lower K/H.")

    # Target: either the future deltas (direct) or the residual over a
    # constant-velocity physics prior (residual mode).
    cv = _cv_prior(X, H)                          # (N,H,2)
    target = (Y - cv) if mode == "residual" else Y
    target = target.astype(np.float32)

    scale_x = float(np.std(X) + 1e-6)             # input normalisation
    scale_y = float(np.std(target) + 1e-6)        # output normalisation

    Xn = torch.tensor(X / scale_x, device=device)
    Tn = torch.tensor(target / scale_y, device=device)

    n = len(Xn)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    cut = int((1 - val_frac) * n) if n > 20 else n
    tr = perm[:cut]

    model = TrajLSTM(hidden=hidden, horizon=H).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()   # robust (Huber) -- less sensitive to outliers

    tr_t = torch.tensor(tr, device=device)
    for ep in range(epochs):
        model.train()
        sub = tr_t[torch.randperm(len(tr_t), device=device)]
        total = 0.0
        for i in range(0, len(sub), batch):
            idx = sub[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xn[idx]), Tn[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  loss {total / max(len(sub),1):.5f}")

    # --- per-step uncertainty envelope (sigma) from the held-out split ---
    eval_idx = perm[cut:] if cut < n else perm
    model.eval()
    with torch.no_grad():
        out = model(Xn[eval_idx]).cpu().numpy() * scale_y      # (M,H,2) target space
    pred_deltas = (cv[eval_idx] + out) if mode == "residual" else out
    pred_pos = np.cumsum(pred_deltas, axis=1)
    true_pos = np.cumsum(Y[eval_idx], axis=1)
    err = np.linalg.norm(pred_pos - true_pos, axis=2)          # (M,H)
    sigma = np.sqrt((err ** 2).mean(axis=0)).astype(float).tolist()

    torch.save({"state": model.state_dict(), "K": K, "H": H, "hidden": hidden,
                "scale_x": scale_x, "scale_y": scale_y, "scale": scale_x,
                "mode": mode, "sigma": sigma}, save_path)
    if verbose:
        print(f"saved -> {save_path}  (mode={mode}, {n} windows, device={device})")
        print(f"  held-out per-step error sigma[1,{H//2},{H}] = "
              f"{sigma[0]:.1f}, {sigma[H//2]:.1f}, {sigma[-1]:.1f} px")
    return save_path


# ----------------------------------------------------------------------------- #
# Inference wrapper
# ----------------------------------------------------------------------------- #
class LSTMPredictor:
    def __init__(self, path="lstm_traj.pt"):
        self.device = get_device()
        ckpt = torch.load(path, map_location=self.device)
        self.K, self.H = ckpt["K"], ckpt["H"]
        self.mode = ckpt.get("mode", "direct")
        self.scale_x = ckpt.get("scale_x", ckpt.get("scale"))
        self.scale_y = ckpt.get("scale_y", ckpt.get("scale"))
        self.sigma = ckpt.get("sigma", None)     # list of H floats (px) or None
        self.model = TrajLSTM(hidden=ckpt["hidden"], horizon=self.H).to(self.device)
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, recent_positions):
        """
        recent_positions: list of the last >=K+1 (x,y) centres of one track.
        Returns a list of H future absolute (x, y) positions.
        """
        pos = np.asarray(recent_positions, dtype=float)
        if len(pos) < self.K + 1:
            return []
        d = np.diff(pos, axis=0)[-self.K:]                  # last K deltas
        x = torch.tensor((d / self.scale_x)[None].astype(np.float32),
                         device=self.device)
        out = self.model(x)[0].cpu().numpy() * self.scale_y  # (H,2)
        if self.mode == "residual":
            cv_step = d[-min(3, len(d)):].mean(axis=0)       # (2,) CV prior step
            pred_deltas = cv_step[None, :] + out             # prior + residual
        else:
            pred_deltas = out
        future = pos[-1] + np.cumsum(pred_deltas, axis=0)    # back to absolute
        return [tuple(p) for p in future]


# ----------------------------------------------------------------------------- #
# Synthetic data (for a smoke test / when you have no tracks yet)
# ----------------------------------------------------------------------------- #
def synthetic_trajectories(n=400, length=80, seed=0):
    """Mix of straight, curved, and accelerating paths with noise."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        t = np.arange(length)
        x0, y0 = rng.uniform(0, 200, 2)
        vx, vy = rng.uniform(-3, 3, 2)
        kind = rng.integers(0, 3)
        if kind == 0:                       # straight
            xs, ys = x0 + vx * t, y0 + vy * t
        elif kind == 1:                     # curved (sinusoidal)
            a, w = rng.uniform(20, 60), rng.uniform(0.05, 0.2)
            xs, ys = x0 + vx * t, y0 + a * np.sin(w * t)
        else:                               # accelerating
            ax, ay = rng.uniform(-0.1, 0.1, 2)
            xs = x0 + vx * t + 0.5 * ax * t * t
            ys = y0 + vy * t + 0.5 * ay * t * t
        xs += rng.normal(0, 0.8, length)
        ys += rng.normal(0, 0.8, length)
        out.append(np.stack([xs, ys], axis=1))
    return out


if __name__ == "__main__":
    # Smoke test: train on synthetic data and report error on held-out tracks.
    print("Generating synthetic trajectories...")
    data = synthetic_trajectories(n=400)
    train_data, test_data = data[:350], data[350:]
    print("Training LSTM (residual mode)...")
    train(train_data, K=10, H=15, epochs=60, mode="residual", save_path="lstm_traj.pt")

    pred = LSTMPredictor("lstm_traj.pt")
    errs = []
    for traj in test_data:
        hist = traj[:30]
        fut_true = traj[30:30 + pred.H]
        fut_pred = pred.predict(hist)
        if len(fut_pred) == len(fut_true):
            e = np.mean(np.linalg.norm(np.array(fut_pred) - fut_true, axis=1))
            errs.append(e)
    print(f"\nHeld-out ADE (avg displacement error): {np.mean(errs):.2f} px "
          f"over {pred.H} steps, {len(errs)} test tracks")
