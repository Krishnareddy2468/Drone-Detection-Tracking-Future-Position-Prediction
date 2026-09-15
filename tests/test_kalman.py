"""Tests for the constant-acceleration Kalman filter."""

import numpy as np
import pytest

from kalman import KalmanCA


def test_first_update_seeds_state_from_measurement():
    kf = KalmanCA()
    assert kf.predict_future(5) == []          # nothing to extrapolate yet
    assert kf.update((100.0, 50.0)) == (100.0, 50.0)
    assert kf.initialized


def test_tracks_constant_velocity_motion():
    """Fed a clean straight line, the filter should lock onto it and extrapolate."""
    kf = KalmanCA(q=1.0, r=1.0)
    vx, vy = 3.0, -2.0
    for t in range(40):
        kf.update((10.0 + vx * t, 200.0 + vy * t))

    assert kf.speed == pytest.approx(np.hypot(vx, vy), abs=0.3)

    H = 10
    future = kf.predict_future(H)
    assert len(future) == H
    for step, (px, py) in enumerate(future, start=1):
        assert px == pytest.approx(10.0 + vx * (39 + step), abs=2.0)
        assert py == pytest.approx(200.0 + vy * (39 + step), abs=2.0)


def test_smooths_measurement_noise():
    """The smoothed output should sit closer to the truth than the raw measurement."""
    rng = np.random.default_rng(0)
    kf = KalmanCA(q=0.01, r=25.0)
    truth, smoothed_err, raw_err = None, [], []
    for t in range(60):
        truth = (5.0 * t, 5.0 * t)
        noisy = (truth[0] + rng.normal(0, 5), truth[1] + rng.normal(0, 5))
        sx, sy = kf.update(noisy)
        if t > 20:                              # let the filter converge first
            smoothed_err.append(np.hypot(sx - truth[0], sy - truth[1]))
            raw_err.append(np.hypot(noisy[0] - truth[0], noisy[1] - truth[1]))

    assert np.mean(smoothed_err) < np.mean(raw_err)


def test_covariance_stays_symmetric_and_finite():
    kf = KalmanCA()
    for t in range(30):
        kf.update((t, t * t * 0.1))
    assert np.all(np.isfinite(kf.P))
    assert np.allclose(kf.P, kf.P.T, atol=1e-6)
