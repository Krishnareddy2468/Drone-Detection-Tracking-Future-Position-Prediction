"""
kalman.py
---------
A 2-D constant-acceleration (CA) Kalman filter used to (a) smooth the noisy
per-frame centre positions coming out of the tracker and (b) extrapolate the
drone's position H frames into the future.

State vector x = [px, py, vx, vy, ax, ay]^T   (position, velocity, acceleration)
Measurement  z = [px, py]^T                   (the detected box centre)

This is the "physics-based" half of the hybrid predictor described in the
report. It is fast, needs no training data, and is accurate for short horizons
and smooth motion. The LSTM handles the longer / non-linear part.
"""

import numpy as np


class KalmanCA:
    def __init__(self, dt=1.0, q=1.0, r=4.0):
        """
        dt : time step between frames (1.0 = "per frame")
        q  : process-noise scale (higher -> trusts the model less, reacts faster
             to manoeuvres). Tie this to your "sudden motion" handling.
        r  : measurement-noise scale (higher -> trusts detections less, smoother)
        """
        self.dt = dt
        self.initialized = False

        # State transition F (constant acceleration kinematics)
        self.F = np.array([
            [1, 0, dt, 0, 0.5 * dt * dt, 0],
            [0, 1, 0, dt, 0, 0.5 * dt * dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix H (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], dtype=float)

        self.Q = np.eye(6) * q          # process noise covariance
        self.R = np.eye(2) * r          # measurement noise covariance
        self.P = np.eye(6) * 1000.0     # initial state covariance (very uncertain)
        self.x = np.zeros((6, 1))

    def update(self, z):
        """Feed one new measured centre z = (px, py). Returns smoothed (px, py)."""
        z = np.array(z, dtype=float).reshape(2, 1)

        if not self.initialized:
            self.x[0, 0], self.x[1, 0] = z[0, 0], z[1, 0]
            self.initialized = True
            return float(self.x[0, 0]), float(self.x[1, 0])

        # ---- Predict ----
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # ---- Update ----
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)          # Kalman gain
        y = z - self.H @ self.x                            # innovation
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        return float(self.x[0, 0]), float(self.x[1, 0])

    def predict_future(self, horizon=15):
        """
        Roll the model forward `horizon` steps WITHOUT new measurements.
        Returns a list of (px, py) future positions = the predicted path.
        """
        if not self.initialized:
            return []
        x = self.x.copy()
        out = []
        for _ in range(horizon):
            x = self.F @ x
            out.append((float(x[0, 0]), float(x[1, 0])))
        return out

    @property
    def speed(self):
        """Current estimated speed (px/frame) — useful for adaptive gating."""
        if not self.initialized:
            return 0.0
        return float(np.hypot(self.x[2, 0], self.x[3, 0]))
