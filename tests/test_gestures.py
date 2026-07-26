import math
import pytest

from src.gesture_detector import GestureDetector, distance, normalized_distance, eye_aspect_ratio


def test_distance():
    assert distance((0, 0), (3, 4)) == 5


def test_normalized_distance_is_scale_invariant():
    a = normalized_distance((0, 0), (2, 0), 10)
    b = normalized_distance((0, 0), (4, 0), 20)
    assert a == pytest.approx(b)


def test_normalized_distance_handles_zero_scale():
    assert math.isfinite(normalized_distance((0, 0), (1, 0), 0))


def test_eye_aspect_ratio():
    eye = [(0, 0), (1, -1), (3, -1), (4, 0), (3, 1), (1, 1)]
    assert eye_aspect_ratio(eye) == pytest.approx(0.5)


def test_eye_aspect_ratio_requires_six_points():
    with pytest.raises(ValueError):
        eye_aspect_ratio([(0, 0)])


def test_cooldown_debounces_triggers():
    detector = GestureDetector()
    assert detector._fire("left_wink", 10.0)
    assert not detector._fire("left_wink", 10.2)
    assert detector._fire("left_wink", 11.1)


def test_threshold_override():
    detector = GestureDetector({"mouth": .2})
    assert detector.thresholds["mouth"] == .2
    assert "wink" in detector.thresholds
