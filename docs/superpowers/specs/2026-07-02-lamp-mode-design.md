# Lamp Mode: Headless Autostart + Housekeeping — Design

**Date:** 2026-07-02
**Status:** Approved

## Goal

The looper should behave like a lamp: power the Pi on, and within a minute the
web UI is reachable and ready to record — no keyboard, no SSH, no prompts.
Alongside this, remove dead legacy code and fix repo hygiene issues.

## Problem

`main.py` blocks on an interactive `input()` prompt to select the audio
device, so it cannot start under systemd. There is no service unit. The
device is addressed by sounddevice index, which shifts between boots.

Housekeeping: `looper_web.py` is a 5,131-line dead monolith (superseded by
`main.py` + `audio.py` + `routes.py` + `effects.py`); the README still names
it as the entry point; `.gitignore` begins with a venv-generated `*` line
that ignores every new file (requiring `git add -f`).

## Design

### 1. Device memory (`_config.json`)

A JSON file at the repo root:

```json
{"device_name": "USB Audio CODEC"}
```

- Written whenever a device is chosen interactively.
- **Resolution order at startup:**
  1. Saved `device_name` matched against current input+output devices.
  2. Fallback: first valid input+output device.
  3. No valid device → print error, `exit(1)` (systemd retries).
- Matching is by device **name**, not index — ALSA indices shift between
  boots; names are stable.
- Resolution is a pure function (`resolve_device(devices, saved_name) ->
  index | None`) so it is unit-testable without hardware.

### 2. Startup flow (`main.py`)

| Invocation | Behaviour |
|---|---|
| `main.py --headless` | Never prompts. Resolve per order above; `exit(1)` if no device. Used by systemd. |
| `main.py` | If saved device resolves, use it and print "using saved device X (run with --pick to change)". Otherwise show today's picker, then save the choice. |
| `main.py --pick` | Force the interactive picker even when a saved device exists; save the new choice. |

### 3. systemd unit + installer

`tools/looper.service`:

```ini
[Unit]
Description=Guitar Looper
After=sound.target network.target

[Service]
Type=simple
User=max
WorkingDirectory=/home/max/looper
ExecStart=/home/max/looper/bin/python main.py --headless
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

- `Restart=on-failure` + `RestartSec=5`: a missing audio interface at boot
  (exit 1) or a mid-jam crash both self-recover; `systemctl stop` (SIGTERM,
  clean exit) stays stopped.
- `tools/install_service.sh`: copies the unit to `/etc/systemd/system/`,
  runs `systemctl daemon-reload`, `systemctl enable --now looper`. Needs
  sudo; idempotent (safe to re-run after edits).
- The install script lives in `tools/`, not `bin/` — `bin/` is the venv.

### 4. Housekeeping

- **Delete `looper_web.py`.** Dead code; nothing imports it.
- **Fix `.gitignore`.** Remove the leading `*` (venv-generated,
  ignores everything). Explicitly ignore venv artefacts (`bin/`, `lib/`,
  `lib64/`, `include/`, `share/`, `pyvenv.cfg`), `_sessions/`, and
  `_config.json`.
- **Rewrite README.** Entry point is `main.py`; document current features
  (sections launcher, FX chains + bus reverb, scale visualizer, sessions,
  trim editor, tap tempo/metronome) and the autostart setup
  (`tools/install_service.sh`).

## Error handling

- No valid audio device (headless): error to stdout (journal), exit 1.
- Saved device name no longer present: fall back to first valid device and
  log which one was used; do not overwrite the saved name (the preferred
  interface may just be unplugged).
- Corrupt/missing `_config.json`: treat as no saved device.

## Testing

- Unit tests for `resolve_device`: saved name found; saved name missing →
  fallback; no valid devices → None; corrupt config handled by caller.
- Manual verification on the Pi: run installer, reboot, confirm web UI
  reachable and looper functional; unplug interface, reboot, confirm the
  service keeps retrying and recovers when replugged.

## Out of scope

- Autosave/quicksave of sessions (next feature, builds on this).
- Graceful SIGTERM state preservation (covered by autosave work).
- WiFi hotspot, MIDI.
