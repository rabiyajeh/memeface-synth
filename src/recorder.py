"""Synchronized processed-video and generated-audio recording."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import queue
import threading
import wave
import cv2
import numpy as np


class Recorder:
    def __init__(self, folder="recordings", fps=30, sample_rate=44100):
        self.folder, self.fps = Path(folder), fps
        self.sample_rate = sample_rate
        self.writer = None
        self.path = None
        self.audio_path = None
        self.audio_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=128)
        self.audio_thread: threading.Thread | None = None
        self.dropped_audio_blocks = 0

    @property
    def recording(self): return self.writer is not None

    def start(self, frame):
        self.folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.folder / f"memeface_{stamp}.mp4"
        self.audio_path = self.folder / f"memeface_{stamp}.wav"
        h, w = frame.shape[:2]
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if not self.writer.isOpened():
            self.writer = None
            return False
        self.dropped_audio_blocks = 0
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()
        return True

    def write(self, frame):
        if self.writer: self.writer.write(frame)

    def push_audio(self, samples: np.ndarray):
        """Queue audio without ever blocking the real-time sound callback."""
        if not self.recording:
            return
        try:
            self.audio_queue.put_nowait(samples)
        except queue.Full:
            self.dropped_audio_blocks += 1

    def _audio_worker(self):
        assert self.audio_path is not None
        with wave.open(str(self.audio_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            while True:
                block = self.audio_queue.get()
                if block is None:
                    break
                pcm = (np.clip(block, -1, 1) * 32767).astype("<i2")
                wav.writeframes(pcm.tobytes())

    def stop(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        if self.audio_thread:
            self.audio_queue.put(None)
            self.audio_thread.join(timeout=3)
            self.audio_thread = None
