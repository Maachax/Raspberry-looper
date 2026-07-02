# Mobile Companion UI — Design

**Date:** 2026-07-02
**Status:** Approved

## Goal

Make the web UI fully usable on a phone so no PC is needed when the rig
moves rooms. Division of labour (Max's split): **phone = manage + display**
(create sections, add FX, name things, sessions, glanceable status),
**MPK = perform** (launch, select, tweak). The phone is the MPK's screen.

## Approach

One responsive page (no separate mobile app or route). A `max-width: 740px`
breakpoint swaps the chrome; desktop above the breakpoint is untouched.
Same `app.js`, same socket state, same commands — zero feature drift.

## Design

### 1. Navigation (≤740px)

- Side icon strip hidden; fixed **bottom tab bar**:
  `◉ HOME · ▦ SECTIONS · ≈ FX · ♩ SCALE · ⋯ MORE`
- MORE is a simple stacked view containing: meters/boost, MIDI bindings
  (learn), sessions (save/load/delete).
- Side panels render as full-screen views between topbar and tab bar.
- All touch targets ≥44px.

### 2. HOME — status display (default view)

Top to bottom:
- Transport state + BPM, large; beat-position bar.
- The MIDI **mode banner scaled up** (readable from ~1m): PLAY hidden,
  SECTION EDIT orange + section name, FX EDIT purple + loop · effect,
  confirm hint appended.
- Loop list (existing rows, larger): color, name, volume bar, mute state,
  MPK selection outline; tap name → rename prompt; trim button opens the
  touch trim view.
- Two big buttons above the tab bar: **⏺ REC** (same state machine as
  spacebar) and **SAVE** (opens the sessions view with the name input
  focused).

### 3. Section naming (small backend addition)

Sections currently have no name. Add:
- `WebLooper.rename_section(section_id, name) -> bool`; `name` stored in
  the section dict, persisted in session meta, survives load (missing name
  in old sessions → unnamed, falls back to `#id` in UI).
- Socket command `rename_section {section_id, name}`.
- Shown in: launcher buttons, MIDI mode banner (already reads `sec.name`),
  phone HOME banner.
- Tap section name in the sections tab → rename prompt (works on desktop
  too).

### 4. Touch trim view

- Opens from a loop row; full-screen between topbar and tabs.
- Full-width waveform canvas, drag handles enlarged to ≥44px hit areas,
  APPLY / CANCEL buttons full-width at the bottom.
- Same `get_waveform` / `apply_trim` commands as desktop.

### 5. Appliance touches

- **Web-app manifest** (`static/manifest.json`: name, icons, standalone
  display, dark theme colors) + `<link rel="manifest">` + apple-touch-icon
  meta so "Add to Home Screen" gives a chromeless app.
- **Screen Wake Lock**: request `navigator.wakeLock('screen')` on first
  user interaction, re-acquire on `visibilitychange`; silently skip where
  unsupported. The phone must not sleep while acting as the MPK's display.

### 6. What stays desktop-shaped

Above 740px nothing changes: two-column DAW layout, side strip, hover
interactions. The scale fretboard renders as-is on mobile inside its tab
(horizontally scrollable if needed).

## Error handling

- Wake lock rejection/unsupported → ignore (no UI error).
- Rename with empty string → keep previous name (backend refuses).
- Rename of missing section id → False, no crash.

## Testing

- Backend: unit tests for `rename_section` (rename, empty-name refused,
  missing id, persistence round-trip through save/load_session).
- Frontend: `node --check`; layout and touch verified live by Max on his
  phone (final walkthrough: navigate all tabs, rename a section, create a
  section, add + tune an effect, trim a loop, save/load a session, banner
  readable, wake lock keeps screen on).

## Out of scope

- Service worker / offline (server is on the LAN; offline is meaningless).
- Native app wrappers.
- Desktop layout changes.
