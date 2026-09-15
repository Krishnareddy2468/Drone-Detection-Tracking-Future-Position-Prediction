"""Tests for the LSTM trajectory predictor and its data plumbing."""

import numpy as np
import pytest

import lstm_predict as L


def test_trajectories_to_deltas():
    traj = [(0.0, 0.0), (1.0, 2.0), (3.0, 5.0)]
    d = L.trajectories_to_deltas(traj)
    assert d.shape == (2, 2)
    np.testing.assert_allclose(d, [[1.0, 2.0], [2.0, 3.0]])


def test_make_windows_shapes_and_alignment():
    K, H = 4, 3
    traj = [np.stack([np.arange(20.0), np.arange(20.0) * 2], axis=1)]
    X, Y = L.make_windows(traj, K=K, H=H)
    # 19 deltas -> 19 - K - H + 1 windows
    assert X.shape == (19 - K - H + 1, K, 2)
    assert Y.shape == (19 - K - H + 1, H, 2)
    # constant-velocity input -> every delta is the same step
    np.testing.assert_allclose(X[0], np.tile([1.0, 2.0], (K, 1)))


def test_make_windows_skips_short_trajectories():
    X, Y = L.make_windows([np.zeros((5, 2))], K=10, H=15)
    assert X.shape == (0, 10, 2)
    assert Y.shape == (0, 15, 2)


def test_cv_prior_repeats_the_mean_step():
    deltas = np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]])  # (1,4,2)
    prior = L._cv_prior(deltas, H=3, m=3)
    assert prior.shape == (1, 3, 2)
    np.testing.assert_allclose(prior[0], np.tile([3.0, 3.0], (3, 1)))      # mean of last 3


def test_synthetic_trajectories_are_reproducible():
    a = L.synthetic_trajectories(n=5, length=30, seed=7)
    b = L.synthetic_trajectories(n=5, length=30, seed=7)
    assert len(a) == 5 and a[0].shape == (30, 2)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y)


def test_train_raises_without_enough_data(tmp_path):
    with pytest.raises(RuntimeError, match="Not enough trajectory data"):
        L.train([np.zeros((5, 2))], K=10, H=15,
                save_path=str(tmp_path / "unused.pt"), verbose=False)


@pytest.mark.parametrize("mode", ["direct", "residual"])
def test_train_then_predict_roundtrip(tmp_path, mode):
    """A short training run must produce a loadable checkpoint that forecasts H steps."""
    K, H = 5, 6
    data = L.synthetic_trajectories(n=40, length=40, seed=1)
    ckpt = tmp_path / f"lstm_{mode}.pt"
    L.train(data, K=K, H=H, hidden=16, epochs=3, mode=mode,
            save_path=str(ckpt), verbose=False)
    assert ckpt.exists()

    pred = L.LSTMPredictor(str(ckpt))
    assert (pred.K, pred.H, pred.mode) == (K, H, mode)
    assert len(pred.sigma) == H

    out = pred.predict(data[0][:20])
    assert len(out) == H
    assert np.all(np.isfinite(np.asarray(out)))

    # Too little history -> no forecast rather than a crash.
    assert pred.predict(data[0][:K]) == []
