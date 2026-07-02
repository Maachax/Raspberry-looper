# Raspberry Looper

A DIY guitar loop pedal on Raspberry Pi with a web UI. Designed to live
permanently next to the amp: power on the Pi and it is ready to record —
no keyboard, no SSH ("lamp mode").

Signal chain: Guitar → Audio interface → Amp (direct monitor) + Pi
(records and plays back loops through the same interface).

## Features

- **Multi-layer looping** — record a master loop, overdub unlimited layers
- **Sections launcher** — group loops into launchable sections (verse,
  chorus, …); launching swaps the playing set at the next loop boundary
- **Audio FX** — per-loop effect chains (baked, via pedalboard), per-section
  chain overrides, and a live master-bus reverb; rotary-knob UI
- **Scale visualizer** — fretboard display with interval labels, formula and
  explanation for the detected/selected scale or mode
- **Automatic quantization** — recordings snap to the nearest bar/beat
- **Tap tempo & metronome** — count-in and click track during recording
- **Visual waveform trimming** — trim editor with beat-grid overlay
- **Sessions** — save and reload full sessions (all layers + metadata)
- **Per-layer control** — volume and mute per layer, master volume
- **Export** — mixed MP3/WAV (bus reverb baked in) or individual stems
- **Web interface** — control from any device on the network (phone-first)

## Requirements

- Raspberry Pi 5 (8 GB) or any Linux/macOS machine with Python 3.11+
- Audio interface with input and output channels
- Python packages: see `requirements.txt`
  (`pedalboard` is optional — loops play dry without it;
  `librosa` is optional — enables tempo/scale detection)

## Setup

```bash
git clone https://github.com/Maachax/Raspberry-looper.git
cd Raspberry-looper
python3 -m venv .
./bin/pip install -r requirements.txt
```

## Running

```bash
./bin/python main.py
```

- First run: pick your audio device from the list — the choice is
  remembered (by name) in `_config.json`.
- Later runs use the saved device automatically; `--pick` re-opens the
  picker.
- `--headless` never prompts: saved device, else first valid device, else
  exit 1 (this is what systemd runs).

Open the printed URL (e.g. `http://<pi-ip>:5000`) from any browser on the
network.

## Autostart at boot (lamp mode)

```bash
./tools/install_service.sh
```

Installs and starts a systemd unit that runs `main.py --headless` and
restarts on failure — so if the audio interface isn't up yet at boot, it
keeps retrying every 5 s until it is. Check on it with:

```bash
systemctl status looper
journalctl -u looper -f
```

The unit assumes the repo lives at `/home/max/looper` with its venv at
`./bin/python` — edit `tools/looper.service` if your paths differ.

## Tests

```bash
./bin/python -m pytest tests/ -q
```
