# UI Rework Design — DAW Layout

**Date:** 2026-05-28  
**Branch:** ui-rework  
**Status:** Approved for implementation

---

## Goal

Replace the current "plain web" full-width layout with a dense DAW-style UI. Key complaints with the current state: rows take full screen width, components feel generic, overall feel is unpolished.

---

## Visual Direction

**Style:** Flat + accent borders. Zero `border-radius` everywhere. No shadows or gradients. Buttons are flat fills (active) or transparent with a border (inactive). Sliders are 2px tracks with a rectangular thumb (2×8px).

**Font:** Monospace for all numbers (BPM, time, dB, multipliers). System sans-serif for labels.

**Palette A — Indigo/Teal:**

| Role | Hex |
|---|---|
| Base background | `#0d0f14` |
| Surface (panels, rows) | `#111318` |
| Border / divider | `#1a1f2e` |
| Border dim | `#252a36` |
| Text primary | `#dde4f0` |
| Text muted | `#4a5a6e` |
| Primary accent (TEMPO, progress) | `#5865f2` |
| Scale panel accent | `#4fd1c5` |
| Layers panel accent | `#ed8936` |
| Active / playing green | `#2bca6e` |
| REC red | `#e8424a` |
| Layer color 1 | `#667eea` |
| Layer color 2 | `#38a169` |

Section header left-border accents: blue for TEMPO, orange for LAYERS, teal for SCALE panel.

---

## Layout

### Performance Mode (default)

Fixed height, no scroll. Two areas side by side:

```
┌─────────────────────────────────────┐
│  TOPBAR: ● PLAYING | A min · 120 BPM  EDIT ▼ │
├─────────────────────┬───────────────┤
│  Left column        │  Side panel   │
│  - TEMPO            │  (icon strip  │
│  - POSITION         │   or expanded │
│  - LAYERS           │   view)       │
├─────────────────────┴───────────────┤
│  Footer: IN ▓▓▓▓░░ -8dB | BOOST ─●─ 1.4× │
│  [● REC]  [+ OVD]  [✕]             │
└─────────────────────────────────────┘
```

**Topbar:** Single row. Left: status LED + state label + `|` divider + `A min · 120 BPM`. Right: `EDIT ▼` button. Bottom border is 2px `#5865f2`.

**Left column (~44% width):**
- TEMPO panel: large monospace BPM number, `BPM · 4/4` label
- POSITION panel: 3px progress bar with beat tick marks, time labels
- LAYERS panel: fills remaining height. Each layer row has a 2px left-border in the layer's colour, inline volume bar, volume number.

**Side panel (right):**
- Default state: a 28px-wide icon strip (`#0a0c10` background). Three icon slots: ♩ (Scale), FX (future), ✦ (Art/future).
- Expanded state: icon strip stays visible on the left edge of the panel; content area opens to the right.
- Tapping an active icon collapses the panel back to strip.
- When collapsed, left column fills the full remaining width.

**Footer:**
- Row 1: `IN` label + input level bar (green) + dB value + `|` divider + `BOOST` label + boost slider + multiplier value
- Row 2: transport buttons — `● REC` (flex:3, red fill), `+ OVD` (flex:2, outline blue), `✕` (flex:1, dim outline)

### Edit Mode

Accessed via `EDIT ▼` in the topbar. The two-column performance block stays pinned at top. Additional edit panels scroll below it. Edit panels (to be designed in detail when implemented): loop trim, per-layer volume/mute, export.

---

## Side Panel Views

### SCALE view (implemented now)

Content when ♩ icon is active:
- Panel header: `SCALE` label (teal accent border), 6str/8str toggle buttons (right-aligned)
- Current scale row: root note (bold) + scale type chip
- Root selector: grid of note buttons, active root highlighted with filled background (`#f59e42`)
- Fretboard: fills remaining height. Only renders SVG when panel is open.

### FX, Art (future stubs)

Icon slots are present in the strip but show a dim `border: 1px solid #1a1f2e` inactive state. Tapping them does nothing until implemented.

---

## Component Rules

**Panel headers:** `background: #111318`, `padding: 4px 10px`, `border-bottom: 1px solid #1a1f2e`, `border-left: 2px solid <accent>`. Label in accent colour, `font-size: 8px`, `letter-spacing: 1.2px`, `font-weight: 700`, all caps.

**Layer rows:** `background: #111318`, `border: 1px solid #1a1f2e`, `border-left: 2px solid <layer-colour>`, `padding: 4px 6px`. Inline volume bar: `width: 36px`, `height: 2px`.

**Buttons (inactive):** `background: transparent`, `border: 1px solid <colour>`, flat, no shadow.  
**Buttons (active/filled):** solid background fill, no border.

**Sliders:** `height: 2px` track, thumb is `width: 2px; height: 8px` rectangle (`#dde4f0`).

**All border-radius:** 0 everywhere except the status LED dot (50%).

---

## What Changes in the Existing Code

| File | Change |
|---|---|
| `static/style.css` | Full rewrite of layout, panels, buttons, sliders |
| `templates/index.html` | Restructure into topbar / two-column body / footer markup |
| `static/app.js` | Side panel open/close toggle; fretboard render gated on panel open state |

The backend (Flask, SocketIO, audio) is untouched.

---

## Out of Scope

- Edit mode panel content (trim, export) — stub the section, detail later
- FX and Art side panel views — icon stubs only
- Animation / transitions between panel states
