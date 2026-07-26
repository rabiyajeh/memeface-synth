"""Webcam and MediaPipe Face Mesh adapter."""
from __future__ import annotations

import cv2
import mediapipe as mp


class FaceTracker:
    def __init__(self, camera_index: int = 0, width: int = 960, height: int = 540):
        self.camera_index, self.width, self.height = camera_index, width, height
        self.capture: cv2.VideoCapture | None = None
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.55, min_tracking_confidence=0.55)

    def start(self) -> bool:
        self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(self.camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.capture.isOpened()

    def read(self):
        if not self.capture or not self.capture.isOpened():
            return False, None, None, 0.0
        ok, frame = self.capture.read()
        if not ok:
            return False, None, None, 0.0
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return True, frame, None, 0.0
        raw = result.multi_face_landmarks[0].landmark
        points = [(p.x, p.y) for p in raw]
        # MediaPipe does not expose per-frame confidence; validity/in-frame ratio is a useful proxy.
        confidence = sum(0 <= x <= 1 and 0 <= y <= 1 for x, y in points[:468]) / 468
        return True, frame, points, confidence

    def stop(self):
        if self.capture:
            self.capture.release()
            self.capture = None

    def close(self):
        self.stop()
        self.mesh.close()

