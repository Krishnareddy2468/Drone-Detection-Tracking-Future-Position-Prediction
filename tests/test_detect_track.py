"""Tests for the pure helpers in the pipeline module (no video or weights needed)."""

from collections import deque

import pytest

detect_track = pytest.importorskip("detect_track",
                                   reason="needs ultralytics/opencv installed")


def _areas(values):
    d = deque(maxlen=12)
    d.extend(values)
    return d


def test_hover_becomes_loiter_after_enough_slow_frames():
    label, count = detect_track.behaviour_of(0.2, _areas([]), hover_count=0)
    assert (label, count) == ("HOVER", 1)

    label, count = detect_track.behaviour_of(0.2, _areas([]), hover_count=29,
                                             loiter_frames=30)
    assert (label, count) == ("LOITER", 30)


def test_moving_target_resets_the_hover_counter():
    label, count = detect_track.behaviour_of(9.0, _areas([]), hover_count=25)
    assert (label, count) == ("TRANSIT", 0)


def test_growing_box_reads_as_approach():
    # first four frames average 100, last four average 200 -> > 1.20x growth
    label, _ = detect_track.behaviour_of(9.0, _areas([100] * 4 + [200] * 4), 0)
    assert label == "APPROACH"


def test_shrinking_box_reads_as_depart():
    label, _ = detect_track.behaviour_of(9.0, _areas([200] * 4 + [100] * 4), 0)
    assert label == "DEPART"


def test_too_little_area_history_stays_transit():
    label, _ = detect_track.behaviour_of(9.0, _areas([100, 400]), 0)
    assert label == "TRANSIT"


def test_pick_device_honours_an_explicit_request():
    assert detect_track.pick_device("cpu") == "cpu"
    assert detect_track.pick_device("mps") == "mps"


def test_pick_device_autodetects_something_valid():
    assert detect_track.pick_device(None) in {"mps", "cuda", "cpu", "0"}
