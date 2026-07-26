"""Non-blocking five-stage personal calibration and JSON persistence."""
from __future__ import annotations
import json
from pathlib import Path
import time


class Calibrator:
    STAGES = [
        ("Neutral face", 1.8), ("Open your mouth", 0.8), ("Big smile!", 0.8),
        ("Raise both eyebrows", 0.8), ("Wink either eye", 0.8),
    ]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.active = False
        self.stage = 0
        self.stage_started = 0.0
        self.samples: list[list[dict[str, float]]] = []

    def start(self):
        self.active, self.stage, self.stage_started = True, 0, time.monotonic()
        self.samples = [[] for _ in self.STAGES]

    def update(self, state) -> dict[str, float] | None:
        if not self.active or state.confidence < 0.5:
            return None
        keys = ("mouth", "smile", "left_brow", "right_brow", "left_ear", "right_ear", "cheek_puff")
        self.samples[self.stage].append({k: getattr(state, k) for k in keys})
        if time.monotonic() - self.stage_started < self.STAGES[self.stage][1]:
            return None
        self.stage += 1
        self.stage_started = time.monotonic()
        if self.stage < len(self.STAGES):
            return None
        self.active = False
        return self.finish()

    def finish(self) -> dict[str, float]:
        avg = lambda stage, key, default: (sum(x[key] for x in self.samples[stage]) / len(self.samples[stage])
                                            if self.samples[stage] else default)
        neutral_mouth, open_mouth = avg(0, "mouth", .015), avg(1, "mouth", .06)
        neutral_smile, big_smile = avg(0, "smile", .36), avg(2, "smile", .48)
        lb0, rb0 = avg(0, "left_brow", .08), avg(0, "right_brow", .08)
        lb1, rb1 = avg(3, "left_brow", .12), avg(3, "right_brow", .12)
        wink_ear = min(avg(4, "left_ear", .16), avg(4, "right_ear", .16))
        thresholds = {
            "mouth": (neutral_mouth + open_mouth) / 2,
            "smile": (neutral_smile + big_smile) / 2,
            "left_brow": (lb0 + lb1) / 2,
            "right_brow": (rb0 + rb1) / 2,
            "wink": min(.24, wink_ear + .025),
            "cheek_puff": .055,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"thresholds": thresholds}, indent=2), encoding="utf-8")
        return thresholds

    @property
    def prompt(self) -> str:
        if not self.active:
            return ""
        label, duration = self.STAGES[self.stage]
        left = max(0.0, duration - (time.monotonic() - self.stage_started))
        return f"CALIBRATION: {label}  {left:.1f}s"

    @staticmethod
    def load(path: str | Path) -> dict[str, float] | None:
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))["thresholds"]
        except (OSError, KeyError, ValueError, TypeError):
            return None
