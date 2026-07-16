# Session Tweaks — Delete Loop, Input-Safe Shortcuts, Trim Reset, 24-Fret Board

**Date:** 2026-07-16
**Status:** Approved

Four independent fixes from live-session feedback.

## 1. Delete a loop from the UI

`deleteLayer()` (app.js) and `AudioEngine.delete_layer()` (audio.py) already exist
and handle section remapping; nothing in the UI calls them.

- Add a ✕ Delete button in each layer's **expanded panel**, in the row next to
  MUTE, for non-master layers only (`layer.id !== 0`).
- Button calls the existing `deleteLayer(id)` which confirms and sends
  `delete_layer`.
- Master layer gets no delete button; clearing everything remains the Clear
  button's job.

## 2. Keyboard shortcuts must not fire while typing

The global `keydown` listener (app.js ~line 1155) fires T (tap tempo),
D (detect tempo), and Space (transport) unconditionally, which makes it
impossible to type "t", "d", or space into the session-name input.

- At the top of the handler: if `e.target` is an `<input>`, `<textarea>`,
  `<select>`, or has `isContentEditable`, return immediately.

## 3. Trim Reset restores the original take

Today `resetTrim()` only resets the on-screen handles; `apply_trim()` slices
the master buffer destructively, so after Apply the original audio is gone and
Reset appears broken.

### Backend (audio.py)

- `apply_trim()`: before replacing the master buffer, if no backup exists yet,
  store a copy of the current master buffer (and its length) as
  `self._pre_trim_backup`. Successive trims keep the **first** backup, so Reset
  always returns to the full original take. `auto_trim_silence()` routes
  through `apply_trim()` and needs no change.
- New `reset_trim()` method: only valid when a backup exists and trimming would
  be allowed (single layer, playing). Restores the backup as the master layer
  (same swap pattern as `apply_trim`: new `LoopLayer(0, "Master", buf)`, update
  `master_length`, reset `master_position`), then clears the backup.
- Backup is discarded when: an overdub is committed, `clear_all` runs, or a new
  master recording starts. In-memory only — never saved into sessions.
- State dict: the existing `trim` block gains `can_reset: bool` (backup exists
  and trim is allowed).

### Frontend (app.js) / socket (routes.py)

- New socket command `reset_trim`.
- The existing ↺ Reset button keeps resetting the handles; additionally, when
  `serverState.trim.can_reset` is true it sends `reset_trim` and re-fetches the
  waveform (same 100 ms delayed `get_waveform` as Apply).
- Trim editor accessibility is unchanged: already locked out once overdubs
  exist (`can_trim` requires exactly one layer).

## 4. Fretboard — all 24 frets, two rows

`renderFretboard()` hardcodes `FRETS = 12`. Extend to 24 frets in a two-row
layout; always 24 (superset of 22-fret guitars, splits evenly).

- One SVG, two stacked boards. Row A = open strings + frets 0–12 with note
  dots on 0–12 (exactly as today). Row B = frets 13–24: its leftmost fret line
  is fret 12 (the shared visual anchor, labeled "12"), with note dots on
  13–24 only — fret-12 notes are not duplicated.
- Position markers: row A keeps 3, 5, 7, 9 single dots and 12 double dots;
  row B gets 15, 17, 19, 21 single dots and 24 double dots.
- Same dot radius, string spacing, and label sizes as today; the panel simply
  grows taller by one board height plus a small row gap. No scrolling.
- Both 6-string and 8-string modes render both rows.

## Non-goals

- No session-format or schema changes (trim backup is volatile by design).
- No undo history for trims — Reset is a single jump back to the original.
- No new dependencies.

## Testing

- New backend tests: trim backup lifecycle — backup created on first trim,
  preserved across successive trims, `reset_trim` restores original length and
  content, backup cleared after overdub commit / clear_all / new recording,
  `can_reset` reported correctly.
- Existing test suite must stay green.
- UI (delete button, typing in inputs, fretboard rows) verified live on the Pi.
