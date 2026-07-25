"""MemeFace Synth — turn facial expressions into audio, chaos, and a mini-game."""
from __future__ import annotations
import json
from pathlib import Path
import time
import cv2
import numpy as np

from src.audio_engine import AudioEngine, PRESETS
from src.calibration import Calibrator
from src.face_tracker import FaceTracker
from src.game_modes import ModeController
from src.gesture_detector import GestureDetector, GestureState
from src.hand_tracker import HandTracker
from src.recorder import Recorder
from src.ui import GameUI
from src.visual_effects import VisualEffects

ROOT = Path(__file__).resolve().parent


class MemeFaceApp:
    def __init__(self):
        settings = json.loads((ROOT/"config/settings.json").read_text(encoding="utf-8"))
        calibration_path = ROOT/settings["calibration_file"]
        self.tracker = FaceTracker(settings["camera_index"], settings["width"], settings["height"])
        self.detector = GestureDetector(Calibrator.load(calibration_path), settings["smoothing"])
        self.calibrator = Calibrator(calibration_path)
        self.hand_tracker = HandTracker()
        self.audio = AudioEngine()
        self.audio.volume = settings["master_volume"]
        self.effects = VisualEffects()
        self.effects.intensity = settings["effect_intensity"]
        self.mode, self.ui = ModeController(), GameUI()
        self.recorder = Recorder(ROOT/"recordings", settings["recording_fps"])
        self.mesh, self.running = settings["face_mesh"], True
        self.sensitivity = settings["gesture_sensitivity"]
        self.base_thresholds = dict(self.detector.thresholds)
        self.camera_on = False
        self.fps, self.last_frame = 0.0, time.perf_counter()
        self.window = "MemeFace Synth — AI Puppet Instrument"

    def toggle_camera(self):
        if self.camera_on:
            self.tracker.stop(); self.camera_on = False
        else:
            self.camera_on = self.tracker.start()

    def action(self, action, frame=None):
        if action == "camera": self.toggle_camera()
        elif action == "mode": self.mode.cycle()
        elif action == "preset": self.audio.preset = (self.audio.preset+1)%len(PRESETS)
        elif action == "calibrate": self.calibrator.start()
        elif action == "mesh": self.mesh = not self.mesh
        elif action == "audio": self.audio.muted = not self.audio.muted
        elif action == "record" and frame is not None:
            if self.recorder.recording:
                self.audio.set_record_sink(None)
                self.recorder.stop()
            elif self.recorder.start(frame):
                self.audio.set_record_sink(self.recorder.push_audio)
        elif action == "replay" and self.mode.index == 1:
            self.mode.game.start()
        elif action == "replay_hand" and self.mode.index == 3:
            self.mode.hand_game.start()
        elif action == "help": self.ui.help = not self.ui.help
        elif action == "quit": self.running = False

    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.ui.mouse)
        cv2.createTrackbar("Volume", self.window, int(self.audio.volume*100), 100, lambda _x: None)
        cv2.createTrackbar("Effects", self.window, int(self.effects.intensity*100), 200, lambda _x: None)
        cv2.createTrackbar("Sensitivity", self.window, int(self.sensitivity*100), 200, lambda _x: None)
        self.toggle_camera()
        audio_ok = self.audio.start()
        if not audio_ok:
            print(f"Audio disabled: {self.audio.last_error or 'sounddevice is unavailable'}")
        if Calibrator.load(self.calibrator.path) is None:
            self.calibrator.start()
        try:
            while self.running:
                self.audio.volume = cv2.getTrackbarPos("Volume", self.window) / 100
                self.effects.intensity = cv2.getTrackbarPos("Effects", self.window) / 100
                self.sensitivity = max(.25, cv2.getTrackbarPos("Sensitivity", self.window) / 100)
                for name, value in self.base_thresholds.items():
                    # Higher sensitivity makes activation easier. EAR is inverted:
                    # a larger threshold recognizes a less-complete blink.
                    self.detector.thresholds[name] = (value*self.sensitivity if name == "wink"
                                                      else value/self.sensitivity)
                ok, frame, lm, confidence = self.tracker.read() if self.camera_on else (False,None,None,0)
                if frame is None:
                    frame = np.zeros((540,960,3),np.uint8)
                    cv2.putText(frame,"CAMERA UNAVAILABLE - click START/STOP to retry",(300,270),
                                cv2.FONT_HERSHEY_SIMPLEX,.7,(0,130,255),2)
                state = self.detector.measure(lm,confidence) if lm else GestureState()
                hand_states = self.hand_tracker.process(frame) if self.camera_on else []
                if confidence >= .5:
                    self.audio.set_controls(state)
                    new_thresholds = self.calibrator.update(state)
                    if new_thresholds:
                        self.base_thresholds.update(new_thresholds)
                        self.detector.thresholds.update(new_thresholds)
                    mouth_anchor=(int(lm[13][0]*frame.shape[1]),int(lm[13][1]*frame.shape[0]))
                    for trigger in state.triggers:
                        if trigger == "both_closed": self.audio.muted = not self.audio.muted
                        else: self.audio.trigger(trigger)
                        self.effects.trigger(trigger,mouth_anchor)
                game_events = self.mode.update(state, hand_states)
                for event in game_events:
                    target = self.mode.hand_game.target
                    anchor = ((int(target.x*frame.shape[1]), int(target.y*frame.shape[0]))
                              if target else (frame.shape[1]//2, frame.shape[0]//2))
                    if event in ("hand_collect", "hand_swipe", "hand_shield"):
                        self.audio.trigger("left_wink")
                    elif event in ("hand_bomb", "hand_miss"):
                        self.audio.trigger("cheek_puff")
                    self.effects.trigger(event, anchor)
                frame = self.effects.render(frame,lm,state,self.mesh)
                frame = self.effects.render_hands(frame, hand_states)
                now=time.perf_counter(); dt=now-self.last_frame; self.last_frame=now
                if dt>0: self.fps=.9*self.fps+.1/dt
                frame=self.ui.draw(frame,state,self.audio.preset_name,self.mode,self.fps,self.recorder.recording,
                                   self.calibrator.prompt,not self.audio.muted and self.audio.enabled,
                                   self.detector.thresholds)
                action=self.ui.pop_action()
                if action: self.action(action,frame)
                if self.recorder.recording: self.recorder.write(frame)
                cv2.imshow(self.window,frame)
                key=cv2.waitKey(1)&0xFF
                if ord("1")<=key<=ord("5"): self.audio.preset=key-ord("1")
                elif key in (ord("q"),27): self.running=False
                elif key==ord("m"): self.action("audio")
                elif key==ord("f"): self.action("mesh")
                elif key==ord("c"): self.action("calibrate")
                elif key==ord("r"): self.action("record",frame)
                elif key==ord("g"): self.action("mode")
                elif key==ord("h"): self.action("help")
                elif key==ord("t") and self.mode.index==2:
                    self.mode.training_index=(self.mode.training_index+1)%len(self.mode.TRAIN)
        finally:
            self.audio.set_record_sink(None)
            self.recorder.stop(); self.tracker.close(); self.hand_tracker.close()
            self.audio.close(); cv2.destroyAllWindows()


if __name__ == "__main__":
    MemeFaceApp().run()
