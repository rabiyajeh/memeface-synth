"""MediaPipe hand tracking and scale-independent hand gesture recognition."""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Sequence

import cv2
import mediapipe as mp

from src.gesture_detector import distance, normalized_distance, Point


@dataclass
class HandState:
    landmarks: list[Point] = field(default_factory=list)
    handedness: str = ""
    confidence: float = 0.0
    cursor: Point = (0.5, 0.5)
    pinch: float = 1.0
    palm_open: bool = False
    fist: bool = False
    peace: bool = False
    swipe: str = ""


class HandTracker:
    """Tracks up to two hands and exposes game-friendly gesture states."""

    def __init__(self, max_hands=2, detection_confidence=.55, tracking_confidence=.55):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=0,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.previous_x: dict[str, tuple[float, float]] = {}
        self.last_swipe: dict[str, float] = {}

    @staticmethod
    def recognize(points: Sequence[Point], handedness="", confidence=1.0,
                  previous: tuple[float, float] | None = None, now: float | None = None) -> HandState:
        if len(points) < 21:
            return HandState(confidence=0.0)
        now = time.monotonic() if now is None else now
        palm_scale = max(distance(points[0], points[9]), 1e-6)
        pinch = normalized_distance(points[4], points[8], palm_scale)
        # Finger extension is based on fingertip-to-wrist vs PIP-to-wrist, so it
        # works at any rotation better than a simple y-coordinate comparison.
        extended = []
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            extended.append(distance(points[tip], points[0]) > distance(points[pip], points[0]) * 1.16)
        open_count = sum(extended)
        thumb_open = distance(points[4], points[5]) > palm_scale * .55
        palm_open = open_count >= 4 and thumb_open
        fist = open_count == 0 and pinch > .30
        peace = extended[0] and extended[1] and not extended[2] and not extended[3]
        swipe = ""
        if previous:
            old_x, old_time = previous
            dt = max(now - old_time, 1e-3)
            velocity = (points[9][0] - old_x) / dt
            if abs(velocity) > 1.45:
                swipe = "right" if velocity > 0 else "left"
        return HandState(list(points), handedness, confidence, points[8], pinch,
                         palm_open, fist, peace, swipe)

    def process(self, frame) -> list[HandState]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        states = []
        if not result.multi_hand_landmarks:
            return states
        now = time.monotonic()
        handedness_list = result.multi_handedness or []
        for index, hand in enumerate(result.multi_hand_landmarks):
            points = [(p.x, p.y) for p in hand.landmark]
            label, score = f"Hand {index+1}", 1.0
            if index < len(handedness_list):
                classification = handedness_list[index].classification[0]
                label, score = classification.label, classification.score
            state = self.recognize(points, label, score, self.previous_x.get(label), now)
            if state.swipe and now - self.last_swipe.get(label, -999) < .7:
                state.swipe = ""
            elif state.swipe:
                self.last_swipe[label] = now
            self.previous_x[label] = (points[9][0], now)
            states.append(state)
        return states

    def close(self):
        self.hands.close()
