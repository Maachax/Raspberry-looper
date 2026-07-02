# Trim Precision, Quieter Defaults, Pad 7 Cleanup — Design

Three small tweaks after live verification of screen-driven knob layers.

## 1. Trim handles: thin cut line, fat grip

**Problem:** The handle bars (10px desktop, 28px mobile) are too thick to
read: the start handle's *left* edge is the cut point while the end handle
is offset so the point lands near its middle — you can't tell which edge of
the bar is the actual trim point.

**Design:** Each handle is a 2px vertical line at the exact cut ratio; the
line IS the trim point, identically for both handles.

- `.trim-handle`: 2px wide, `var(--accent)`, full height.
- Touch/click target: a transparent `::after` zone ~44px wide centered on
  the line (desktop keeps a smaller ~20px zone). The old mobile-only fat
  rules are removed.
- A small grip tab (rounded rectangle, accent color) sits at the top of the
  line so it stays grabbable/visible over the waveform; start tab hangs
  right of the line, end tab hangs left, so tabs don't collide at close
  range.
- `app.js` positions both handles centered on their ratio:
  `left: calc(<ratio>% - 1px)` for both start and end (drop the `- 12px`
  end offset).
- The dark `.trim-overlay-*` regions already end at the exact ratios;
  unchanged.

## 2. Snap-to-beat and recording quantize OFF by default

- `audio.py`: `self.quantize_enabled = False` (was True). The UI checkbox
  `quantizeToggle` syncs from server state; remove its hardcoded `checked`
  attribute in `index.html` so it doesn't flash on before first sync.
- `index.html`: remove `checked` from `snapToBeat` (pure client-side).
- Both remain toggleable in the UI exactly as before.

## 3. Bank B pad 7: remove Mute selected

- Delete `pc:9:10 → mute_selected` from `DEFAULT_BINDINGS['global']` in
  `midi_control.py`. Pad 7 does nothing by default; `mute_selected` stays
  in the action catalog for MIDI-learn.
- No saved-bindings migration needed: `_config.json` does not exist on the
  device.
- Update `docs/midi-cheatsheet.md` Bank B row (pad 7 = "–").

## Testing

- pytest: update/add tests for the pad-7 default and the quantize default;
  full suite stays green.
- Trim handle rendering: `node --check` + live check on the phone (Max).
