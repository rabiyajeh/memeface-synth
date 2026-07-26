"""Low-latency procedural synth and original gesture sound effects."""
from __future__ import annotations
import threading
import numpy as np
from collections.abc import Callable

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

PRESETS = [
    {"name": "Cyber Bass", "wave": "saw", "base": 55, "drive": 1.6},
    {"name": "Alien Voice", "wave": "formant", "base": 82, "drive": 1.2},
    {"name": "Robot Squeak", "wave": "square", "base": 150, "drive": 1.0},
    {"name": "Meme Horn", "wave": "horn", "base": 98, "drive": 1.4},
    {"name": "Haunted Synth", "wave": "sine", "base": 65, "drive": 2.0},
]


class AudioEngine:
    def __init__(self, sample_rate=44100, blocksize=512):
        self.sr, self.blocksize = sample_rate, blocksize
        self.enabled, self.muted, self.volume = True, False, .35
        self.preset = 0
        self.targets = {"mouth": .02, "brow": .08, "mod": .08, "smile": 0}
        self.params = dict(self.targets)
        self.phase = self.lfo_phase = 0.0
        self.filter_state = 0.0
        self.voices: list[np.ndarray] = []
        self.lock = threading.Lock()
        self.stream = None
        self.last_error = ""
        self.record_sink: Callable[[np.ndarray], None] | None = None

    @property
    def preset_name(self):
        return PRESETS[self.preset]["name"]

    def start(self) -> bool:
        if sd is None:
            self.enabled = False
            return False
        try:
            self.stream = sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32",
                                          blocksize=self.blocksize, latency="low", callback=self._callback)
            self.stream.start()
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.enabled = False
            return False

    def set_record_sink(self, sink: Callable[[np.ndarray], None] | None):
        """Attach a non-blocking consumer for final mono audio blocks."""
        self.record_sink = sink

    def set_controls(self, state):
        self.targets["mouth"] = state.mouth
        self.targets["brow"] = state.left_brow
        self.targets["mod"] = state.right_brow
        self.targets["smile"] = state.smile

    def trigger(self, name: str):
        n = int(self.sr * (1.0 if name == "cheek_puff" else .55))
        t = np.arange(n, dtype=np.float32) / self.sr
        if name == "left_wink":  # Original rising brass/noise chirp.
            sig = np.sin(2*np.pi*(190*t + 330*t*t)) + .25*np.sin(2*np.pi*380*t)
        elif name == "right_wink":  # Procedural comedic impact.
            sig = np.sin(2*np.pi*(95*t - 45*t*t)) + .5*np.random.default_rng().normal(0, 1, n)
        else:  # Deep explosion.
            sig = np.sin(2*np.pi*(72*t - 24*t*t)) + .35*np.random.default_rng().normal(0, 1, n)
        sig = (.5 * sig * np.exp(-t * (3 if name == "cheek_puff" else 6))).astype(np.float32)
        with self.lock:
            self.voices.append(sig)

    def _oscillator(self, phase, wave):
        if wave == "saw":
            return 2 * (phase - np.floor(phase + .5))
        if wave == "square":
            return np.where(np.sin(2*np.pi*phase) >= 0, 1., -1.)
        if wave == "horn":
            return np.sin(2*np.pi*phase) + .45*np.sin(4*np.pi*phase) + .2*np.sin(6*np.pi*phase)
        if wave == "formant":
            return np.sin(2*np.pi*phase) * (.6 + .4*np.sin(10*np.pi*phase))
        return np.sin(2*np.pi*phase)

    def _callback(self, outdata, frames, _time, status):
        for key in self.params:
            self.params[key] += .08 * (self.targets[key] - self.params[key])
        p = PRESETS[self.preset]
        brow = np.clip((self.params["brow"] - .06) * 12, 0, 1)
        mod = np.clip((self.params["mod"] - .06) * 15, 0, 1)
        freq = p["base"] * (1 + 1.3*brow)
        idx = np.arange(frames)
        lfo = np.sin(self.lfo_phase + 2*np.pi*5.5*idx/self.sr) * mod * .035
        increments = freq * (1 + lfo) / self.sr
        phases = self.phase + np.cumsum(increments)
        signal = self._oscillator(phases, p["wave"])
        # A smile fades in a consonant fifth, giving an immediate harmony layer.
        smile_mix = np.clip((self.params["smile"] - .38) * 6, 0, .35)
        signal = signal + smile_mix * self._oscillator(phases * 1.5, p["wave"])
        self.phase = phases[-1] % 1
        self.lfo_phase = (self.lfo_phase + 2*np.pi*5.5*frames/self.sr) % (2*np.pi)
        cutoff = np.clip(100 + self.params["mouth"] * 16000, 100, 9000)
        alpha = 1 - np.exp(-2*np.pi*cutoff/self.sr)
        filtered = np.empty(frames, dtype=np.float32)
        z = self.filter_state
        for i in range(frames):
            z += alpha * (signal[i] - z)
            filtered[i] = z
        self.filter_state = float(z)
        signal = np.tanh(filtered * p["drive"]) * (.15 + np.clip(self.params["mouth"]*5, 0, .5))
        with self.lock:
            remaining = []
            for voice in self.voices:
                count = min(frames, len(voice))
                signal[:count] += voice[:count]
                if count < len(voice):
                    remaining.append(voice[count:])
            self.voices = remaining
        gain = 0 if self.muted or not self.enabled else self.volume
        rendered = np.clip(signal * gain, -1, 1).astype(np.float32, copy=False)
        outdata[:, 0] = rendered
        if self.record_sink is not None:
            # The sink only queues a copy. Disk I/O happens on a recorder thread.
            self.record_sink(rendered.copy())

    def close(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
