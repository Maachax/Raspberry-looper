# MIDI Control Surface (Akai MPK Mini mk3) — Design

**Date:** 2026-07-02
**Status:** Approved

## Goal

Everything Max can do in the web UI today — record, launch sections, mix,
create sections, add and tune FX — controllable from the MPK Mini mk3, with
the web UI as a glance-only display. No mouse, no keyboard, no phone taps
mid-jam.

## Hardware facts (measured on Max's unit, 2026-07-02)

Captured via `amidi -d`; these are the unit's *current program*, not
datasheet assumptions.

| Control | Message | Channel | Numbers |
|---|---|---|---|
| Pads bank A, top row L→R | Program Change | 10 | 4, 5, 6, 7 |
| Pads bank A, bottom row L→R | Program Change | 10 | 0, 1, 2, 3 |
| Pads bank B, top row L→R | Program Change | 10 | 12, 13, 14, 15 |
| Pads bank B, bottom row L→R | Program Change | 10 | 8, 9, 10, 11 |
| Knobs K1–K8 | Control Change | 1 | CC 70–77, absolute 0–127 |
| Keybed (25 keys, default octave) | Note On/Off | 1 | 48–72 |

Notes:
- "Pad N" in this spec means **reading order** (top-left = pad 1), which is
  how Max reaches for them — not Akai's printed numbering.
- Pads send PC because the unit's PROG CHANGE pad-mode is engaged. The
  binding layer is message-type agnostic, so this is fine; if the pad mode
  ever changes to notes, MIDI-learn re-binds without code changes.
- The OCT+/− buttons shift keybed note numbers. Key commands assume the
  default octave (lowest key = note 48). Documented in the cheat sheet;
  the MPK's octave LEDs show when it is centred.

## Architecture

### `midi_control.py` — new module

`MidiController(looper, notify)` — one daemon thread.

- **Library:** `mido` + `python-rtmidi` (prebuilt Pi wheels on piwheels).
  Added to `requirements.txt`.
- **Hot-plug:** when no port is open, poll every 2 s for an input port whose
  name contains `"MPK mini"` (configurable). On read error (unplug), close,
  mark disconnected, resume polling. Works with the always-on systemd
  service; no restart needed.
- **Trigger normalization:** every incoming message becomes a trigger key:
  `('pc', channel, program)`, `('cc', channel, cc_number)`, or
  `('note', channel, note)` (Note On with velocity > 0 only; Note Off
  ignored). CC messages carry their 0–127 value alongside.
- **Dispatch:** trigger key → action lookup in the active mode's binding
  map, then a handler calls the same `WebLooper` methods the socket
  commands use. Brief lock, no disk I/O — nothing new for the audio
  callback.
- **UI sync:** after any state-changing action, call `notify()` (provided
  by routes.py; broadcasts `update` with `get_state()`), debounced to at
  most 10/s so knob sweeps don't flood clients.

### Modes

Server-side state machine in `MidiController`: `PLAY` (default),
`SECTION_EDIT`, `FX_EDIT`. Also selection state: `selected_loop`
(defaults to the most recently recorded loop), `selected_fx_slot`,
`editing_section`. Entering one edit mode exits the other; the edit-mode
pad toggles its own mode off.

`get_state()` gains a `midi` block:

```json
"midi": {"connected": true, "mode": "fx_edit", "selected_loop": 2,
         "selected_fx_slot": 1, "editing_section": null,
         "learn": null}
```

### Bindings & MIDI-learn

- All default bindings live in one `DEFAULT_BINDINGS` structure:
  per-mode maps `{mode: {trigger_key: action_id}}` plus a `global` map
  (actions that work in every mode: record, tap tempo, mode toggles,
  save session).
- User overrides persist under `"midi"` in `_config.json`; merged over
  defaults at load; corrupt/missing config falls back to defaults.
- **Learn:** the web UI MIDI tab lists every action with its current
  binding and a LEARN button. LEARN arms the server
  (`midi.learn = action_id`); the next incoming trigger binds and
  persists (10 s timeout disarms). Any action can be re-bound, including
  key commands.

## Control layout (defaults)

### Pads bank A — sections (identical position in every mode)

- PLAY: launch section 1–8 (Nth section in list order; missing → no-op).
- SECTION_EDIT: select section 1–8 as the one being edited.

### Pads bank B — actions (global, work in every mode)

| Pad (reading order) | Action |
|---|---|
| B1 | Record / overdub toggle (mirrors the UI record button state machine) |
| B2 | Tap tempo |
| B3 | Create section from currently playing loops (auto-named) |
| B4 | Save session (auto-named with timestamp) |
| B5 | Toggle SECTION_EDIT mode |
| B6 | Toggle FX_EDIT mode |
| B7 | Mute/unmute selected loop |
| B8 | Delete selected loop (double-tap within 1 s) |

### Keybed

**Lower octave, notes 48–59 (chromatic, left to right):**
- PLAY / FX_EDIT: select loop 1–12.
- SECTION_EDIT: toggle loop 1–12 membership in the edited section.

**Upper octave, notes 60–72:**
- PLAY: unbound (available to learn).
- FX_EDIT:

| Note | Action |
|---|---|
| 60–64 | Add effect to selected loop's chain: reverb, delay, chorus, distortion, filter |
| 65 | Select previous chain slot |
| 67 | Select next chain slot |
| 69 | Toggle selected effect enabled/bypassed |
| 71 | Remove selected effect (double-tap within 1 s) |
| 72 | Exit FX_EDIT |
| 66, 68, 70 | Reserved (unbound) |

- SECTION_EDIT:

| Note | Action |
|---|---|
| 71 | Delete edited section (double-tap within 1 s) |
| 72 | Exit SECTION_EDIT |
| 60–70 | Reserved (unbound) |

### Knobs (absolute; CC value scaled linearly into the target range)

- PLAY / SECTION_EDIT: K1–K8 = volume of loops 1–8.
- FX_EDIT: K1–K3 = the selected effect's schema params in order
  (numeric: scaled min→max; the filter's LP/HP enum: value < 64 = LP,
  ≥ 64 = HP). K4–K6 unbound. K7–K8 = bus reverb `room_size` / `wet`.

## Web UI (display-only feedback)

- **Mode banner** in the top bar: PLAY (green) / SECTION EDIT: <name>
  (orange) / FX EDIT: <loop> · <effect> (purple). Driven by the `midi`
  state block.
- **Selection highlight** on the selected loop row and, in FX_EDIT, the
  selected chain slot in the existing FX panel.
- **MIDI tab** in the side panel: connected indicator, action list with
  bindings and LEARN buttons.

## Error handling

- Port read error / device unplugged → disconnected state, poll to
  reconnect; UI LED reflects it.
- Action on a missing target (loop/section/slot index out of range) →
  silent no-op.
- Unknown action id in config → ignored with a log line.
- Malformed `"midi"` config block → defaults.
- Learn armed but nothing received in 10 s → disarm, UI updates.

## Phasing (each phase ships working software)

**Phase 1 — foundation + PLAY:** `midi_control.py` (thread, hot-plug,
triggers, dispatch, config, learn backend), bank A → launch sections,
K1–K8 → loop volumes, B1 record toggle, B2 tap tempo, `midi` state block,
UI: connected LED + MIDI learn tab.

**Phase 2 — the modal layer:** SECTION_EDIT and FX_EDIT modes, selection
model, keybed commands, contextual knobs, B3–B8 actions, mode banner and
selection highlights, double-tap confirms, printable cheat sheet
(`docs/midi-cheatsheet.md`).

## Testing

- Unit tests (no hardware): trigger normalization from raw mido messages,
  per-mode dispatch against a fake looper, learn arm/bind/timeout,
  double-tap confirm windows, knob scaling (numeric + enum), membership
  toggle, config merge and corrupt-config fallback.
- Live verification per phase with the real MPK: pads launch, knobs mix,
  record toggles; then full modal walkthrough (create section, add and
  tune an effect, delete a loop) hands-free.

## Out of scope

- Driving MPK pad LEDs (mk3 LEDs are not host-controllable).
- Text entry (renaming) — stays on-screen.
- Trim editing from the controller.
- Footswitch support (future; the trigger/binding model already fits it).
