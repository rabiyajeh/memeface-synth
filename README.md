# MemeFace Synth — The AI Puppet Instrument

MemeFace Synth is a playful local webcam instrument. Open your jaw for a wah-wah
filter, raise eyebrows to bend and wobble pitch, wink for procedural comic sounds,
smile for rainbow chaos, and puff your cheeks for a bass explosion.

All webcam processing happens locally. No images, landmarks, recordings, or audio
are uploaded.

## Requirements

- Python 3.11 (recommended; 64-bit)
- Windows 10/11, macOS, or Linux
- A webcam
- A working audio output device and drivers (no microphone is required)

## Install and run

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

On macOS/Linux, activate with `source .venv/bin/activate`. macOS may require camera
permission for Terminal. Linux users may need PortAudio (`libportaudio2`) and a
Video4Linux-compatible camera.

The first launch starts a five-second calibration. Follow the on-screen prompts:
neutral, mouth open, smile, eyebrows raised, and wink. Personalized thresholds are
saved to `config/calibration.json`.

## Controls

| Control | Action |
|---|---|
| Jaw open/close | Filter cutoff, loudness, mouth laser |
| Left eyebrow | Pitch |
| Right eyebrow | Vibrato/distortion |
| Both eyebrows | Turbo red-eye effect |
| Left wink | Original procedural horn |
| Right wink | Original procedural impact |
| Both eyes closed briefly | Mute/unmute |
| Smile | Rainbow particles, harmony/chaos |
| Puff cheeks | Bass explosion, shake, shockwave |

Keyboard shortcuts: `1`–`5` select a preset, `M` mute, `F` Face Mesh, `C`
recalibrate, `R` record, `G` change mode, `H` help, `T` advance the training
gesture, and `Q`/`Esc` quit. The matching on-screen controls are clickable.
The Volume, Effects, and Sensitivity trackbars under the video update live.

Presets are Cyber Bass, Alien Voice, Robot Squeak, Meme Horn, and Haunted Synth.

## Modes

- **Free Play** is the full facial instrument.
- **Stay Calm** runs for 60 seconds. Mouth opening, smiles, eyebrow movement, and
  sudden expression changes reduce 100 calmness points. It reports score, reaction
  count, and best calm streak. Cycle away and back with `G` to replay.
- **Gesture Training** displays a live value and instructions. Press `T` to test
  the next gesture. Adjust `gesture_sensitivity` or generated calibration values in
  `config/settings.json` / `config/calibration.json`.
- **Hand Mayhem** is a 45-second arcade round. Your index fingertip is the cursor:
  pinch ordinary orbs, touch yellow bonus orbs with an open palm, and use a fist
  shield (a second hand works best) when dealing with red bombs. Fast swipes earn
  small style bonuses. The round tracks score, lives, combo, collected targets, and
  best combo; click Replay on the results panel to try again.

Cheek puffing is experimental because Face Mesh has no direct pressure measurement.
The detector combines cheek width with closed-lip shape. Its fallback is deliberately
puffing while keeping lips closed; tune `cheek_puff` in `config/calibration.json`
(lower is more sensitive).

## Recording

Press `R` to save two synchronized timestamped files under `recordings/`: processed
MP4 video and the generated synth/effect audio as PCM WAV. Audio is queued from the
real-time callback and written on a background thread so disk access cannot stall
sound generation. Merge the pair with:

```powershell
ffmpeg -i recordings\memeface_TIMESTAMP.mp4 -i recordings\memeface_TIMESTAMP.wav -c:v copy -c:a aac -shortest recordings\memeface_TIMESTAMP_merged.mp4
```

## Configuration and performance

Edit `config/settings.json` for camera index, dimensions, smoothing, master volume,
effect intensity, and target recording FPS. The 960×540 default is intended to hold
30 FPS on typical laptops. If it does not, use 640×480, disable Face Mesh (`F`), and
close other camera apps.

## Troubleshooting

- **Camera unavailable:** close Zoom/Teams/browser tabs, change `camera_index` to
  `1`, and click Start/Stop to retry.
- **No audio:** confirm the default output device, install/update PortAudio drivers,
  and restart. The visual app keeps running if audio initialization fails.
- **MediaPipe install fails:** use 64-bit CPython 3.11 and upgrade pip. Avoid the
  Microsoft Store's aliased Python executable.
- **Winks repeat or fail:** recalibrate in even lighting; increase `wink` slightly
  if closing is missed, or decrease it if false triggers occur.
- **Face not detected:** face the camera, improve front lighting, and keep the full
  face in frame. Discrete triggers pause automatically at low confidence.
- **MP4 will not play:** install an OpenCV build with MP4 support or change the
  codec/container in `src/recorder.py`.

## Extending the app

To add a gesture, calculate a normalized measurement in `GestureDetector.measure`,
smooth it through `_ema`, add a threshold and active state, then map its trigger in
`main.py`. Keep discrete actions behind `_fire` cooldowns.

To add a preset, append a dictionary to `PRESETS` in `src/audio_engine.py`; add a
waveform branch in `_oscillator` when needed. Audio callbacks must remain free of
file I/O and blocking work.

To add a visual, add persistent state and a `trigger` case to `VisualEffects`, then
draw and expire it in `render`. Anchor positions should come from normalized
landmarks and sizes should derive from `state.face_scale`.

Hand gestures live in `src/hand_tracker.py`. Add a scale-independent measurement to
`HandTracker.recognize`, expose it through `HandState`, and consume it in
`HandMayhemGame.update`. Hand processing uses MediaPipe's lightweight complexity-0
model and remains local.

## Tests

```powershell
pytest -q
```

Tests cover Euclidean and scale-normalized distance math, eye-aspect ratio,
threshold overrides, zero-scale safety, and cooldown debouncing.
