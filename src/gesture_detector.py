"""Normalized facial measurements, smoothing, and debounced gesture detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
import time
from typing import Mapping, Sequence

Point = tuple[float, float]


def distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def normalized_distance(a: Point, b: Point, scale: float) -> float:
    """Return a scale-independent distance, safely handling bad face scales."""
    return distance(a, b) / max(float(scale), 1e-6)


def eye_aspect_ratio(points: Sequence[Point]) -> float:
    """Six points ordered outer, upper-outer, upper-inner, inner, lower-inner, lower-outer."""
    if len(points) != 6:
        raise ValueError("EAR requires six points")
    width = distance(points[0], points[3])
    return (distance(points[1], points[5]) + distance(points[2], points[4])) / max(2 * width, 1e-6)


@dataclass
class GestureState:
    mouth: float = 0.0
    smile: float = 0.0
    left_brow: float = 0.0
    right_brow: float = 0.0
    left_ear: float = 0.3
    right_ear: float = 0.3
    cheek_puff: float = 0.0
    face_scale: float = 0.0
    confidence: float = 0.0
    active: dict[str, bool] = field(default_factory=dict)
    triggers: list[str] = field(default_factory=list)


DEFAULT_THRESHOLDS = {
    "mouth": 0.035,
    "smile": 0.43,
    "left_brow": 0.105,
    "right_brow": 0.105,
    "wink": 0.19,
    "cheek_puff": 0.055,
}


class GestureDetector:
    """Converts MediaPipe landmarks to expressive, normalized control signals."""

    # Eye rings are arranged to match eye_aspect_ratio's expected ordering.
    LEFT_EYE = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE = (362, 385, 387, 263, 373, 380)

    def __init__(self, thresholds: Mapping[str, float] | None = None, smoothing: float = 0.32):
        self.thresholds = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
        self.smoothing = smoothing
        self.values: dict[str, float] = {}
        self.last_trigger: dict[str, float] = {}
        self.closed_since: float | None = None
        self.cooldowns = {"left_wink": 1.0, "right_wink": 1.0, "both_closed": 1.5, "cheek_puff": 2.0}

    def _ema(self, name: str, value: float) -> float:
        old = self.values.get(name, value)
        value = old + self.smoothing * (value - old)
        self.values[name] = value
        return value

    def _fire(self, name: str, now: float) -> bool:
        if now - self.last_trigger.get(name, -999.0) >= self.cooldowns[name]:
            self.last_trigger[name] = now
            return True
        return False

    def measure(self, lm: Sequence[Point], confidence: float = 1.0, now: float | None = None) -> GestureState:
        if len(lm) < 468:
            return GestureState(confidence=0.0)
        now = time.monotonic() if now is None else now
        face_width = distance(lm[234], lm[454])
        mouth = self._ema("mouth", normalized_distance(lm[13], lm[14], face_width))
        mouth_width = normalized_distance(lm[61], lm[291], face_width)
        corner_height = ((lm[61][1] + lm[291][1]) * 0.5 - (lm[13][1] + lm[14][1]) * 0.5) / max(face_width, 1e-6)
        smile = self._ema("smile", mouth_width + max(0.0, corner_height) * 1.8)
        left_brow = self._ema("left_brow", normalized_distance(lm[105], lm[159], face_width))
        right_brow = self._ema("right_brow", normalized_distance(lm[334], lm[386], face_width))
        left_ear = self._ema("left_ear", eye_aspect_ratio([lm[i] for i in self.LEFT_EYE]))
        right_ear = self._ema("right_ear", eye_aspect_ratio([lm[i] for i in self.RIGHT_EYE]))
        # Experimental proxy: cheeks widen while lips remain mostly closed.
        cheek_width = normalized_distance(lm[50], lm[280], face_width)
        cheek_puff = self._ema("cheek_puff", max(0.0, cheek_width - 0.54) + max(0.0, 0.035 - mouth))

        t = self.thresholds
        active = {
            "mouth": mouth > t["mouth"],
            "smile": smile > t["smile"],
            "left_brow": left_brow > t["left_brow"],
            "right_brow": right_brow > t["right_brow"],
            "turbo": left_brow > t["left_brow"] and right_brow > t["right_brow"],
            "cheek_puff": cheek_puff > t["cheek_puff"],
        }
        triggers: list[str] = []
        reliable = confidence >= 0.5
        left_closed, right_closed = left_ear < t["wink"], right_ear < t["wink"]
        if reliable and left_closed and not right_closed and self._fire("left_wink", now):
            triggers.append("left_wink")
        if reliable and right_closed and not left_closed and self._fire("right_wink", now):
            triggers.append("right_wink")
        if left_closed and right_closed:
            self.closed_since = self.closed_since or now
            if reliable and now - self.closed_since > 0.22 and self._fire("both_closed", now):
                triggers.append("both_closed")
        else:
            self.closed_since = None
        if reliable and active["cheek_puff"] and self._fire("cheek_puff", now):
            triggers.append("cheek_puff")
        return GestureState(mouth, smile, left_brow, right_brow, left_ear, right_ear,
                            cheek_puff, face_width, confidence, active, triggers)

