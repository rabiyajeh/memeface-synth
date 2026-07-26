import wave
import numpy as np

from src.recorder import Recorder


def test_audio_worker_writes_pcm_wav(tmp_path):
    recorder = Recorder(tmp_path, sample_rate=8000)
    recorder.audio_path = tmp_path / "test.wav"
    recorder.writer = object()  # Recording state; video methods are not used here.
    recorder.audio_thread = __import__("threading").Thread(target=recorder._audio_worker)
    recorder.audio_thread.start()
    recorder.push_audio(np.array([0.0, .5, -.5], dtype=np.float32))
    recorder.audio_queue.put(None)
    recorder.audio_thread.join(timeout=2)
    recorder.audio_thread = None
    recorder.writer = None
    with wave.open(str(recorder.audio_path), "rb") as wav:
        assert wav.getframerate() == 8000
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 3
