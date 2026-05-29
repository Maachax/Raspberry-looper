# Scale & Mode Interval-Learning Visualization — Design

**Date:** 2026-05-29
**Status:** Approved (design), pending spec review
**Area:** Looper web UI — Scale side panel (`templates/index.html`, `static/app.js`, `static/style.css`)

## Goal

Turn the existing scale-panel fretboard from a "where are the notes" diagram into a
**learning tool that teaches the intervallic identity of each scale/mode**. Max wants to
internalize the unique traits that define each scale — e.g. that Dorian is a minor scale
with a raised 6th, that the ♭5 is the blues "blue note" — by *seeing* them on the
fretboard.

Two capabilities, both purely visual / client-side:

1. **Interval identity** — every scale tone on the fretboard is labeled with its scale
   degree (note name + interval, e.g. `F♯` / `6`), so the player reads intervals directly.
2. **Defining-note spotlight** — the characteristic note(s) that give each scale its
   flavor are visually highlighted, with a plain-English explanation of *why* that note
   matters.

**Out of scope** (explicitly declined during brainstorming): audio playback / ear
training, and side-by-side mode comparison views. Keep the door open for them later but
do not build them now.

## Current state (what exists)

- `static/app.js` `renderFretboard()` draws an SVG fretboard into `#fretboard`. Each scale
  tone is a small dot (`r=6.5`): root in orange `#ed8936`, other tones in teal `#4fd1c5`.
  Only the root dot shows a note-name label.
- Scale data lives in `SCALE_INTERVALS` (14 scales) and `SCALE_LABELS` in `app.js`.
- 6/8-string toggle (`guitarMode`), root buttons, scale-type `<select>`, and a server-backed
  `DETECT SCALE` feature already work. `set_scale` is emitted to the server (used for future
  music-generator sync) — unchanged by this work.
- Panel markup: `templates/index.html` lines ~88–126 (`#panelScale`).

## Visual design (approved via mockups)

The chosen fretboard label style is **note name + interval** ("Style B"): each scale-tone
dot shows the note letter on top and the interval shorthand below.

The Scale panel gains three new elements stacked around the existing fretboard:

```
[ A ]  [ Dorian ▾ ]                 ← existing header row
INTERVAL FORMULA
[R][2][♭3][4][5][6*][♭7]            ← NEW: formula strip, defining degree(s) gold
┌──────────────────────────────┐
│  fretboard (note + interval,  │   ← existing fretboard, restyled
│  defining notes glow gold)    │
└──────────────────────────────┘
● Root  ● Scale tone  ● Defining   ← NEW: legend
┌ WHAT DEFINES DORIAN ───────────┐
│ Minor scale with a raised 6th. │   ← NEW: explanation box
│ That natural 6 (F♯) is what…   │
└────────────────────────────────┘
[ DETECT SCALE ]                    ← existing, stays below
```

### Color / treatment

- Root dot: orange `#ed8936` (unchanged).
- Ordinary scale tone: teal `#4fd1c5` (unchanged).
- **Defining note: gold `#f6c453`**, with a thin gold ring and a soft SVG glow
  (`feGaussianBlur` filter) so it reads as "special" at a glance. Same gold is used for the
  highlighted pill(s) in the formula strip and the explanation box's accent border.

### Dot / label sizing

Adding a two-line label (note + interval) requires larger dots than today's `r=6.5`. The
fretboard must enlarge dots to ~`r=9–10` and increase string spacing so labels are legible.
On 8-string (8 rows) this makes the SVG taller than 6-string; the `.fretboard-container`
should size to its SVG (SVG keeps `width:100%`, height follows `viewBox`), and the panel body
scrolls vertically if the total panel exceeds the viewport. Note-name accidentals render with
the music sharp glyph `♯` (not ASCII `#`).

## Interval labeling

A single chromatic interval-label map (semitones-from-root → shorthand), using flat
spelling, drives both the dots and the formula strip:

| semitones | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|-----------|---|---|---|---|---|---|---|---|---|---|----|----|
| label     | R | ♭2| 2 | ♭3| 3 | 4 | ♭5| 5 | ♭6| 6 | ♭7 | 7  |

This is intentionally simple (flats throughout, no enharmonic spelling per-scale). It
correctly communicates the *quality* Max cares about (minor 3rd = ♭3, major 7th = 7) without
the complexity of key-correct note spelling. `♭5` covers the Lydian `#4` / blues blue-note
case; this is called out in the explanation text where it matters.

## Defining notes & explanations (content)

A new data table `SCALE_INFO` keyed by scale type. Each entry has:

- `characteristic`: array of intervals (semitones from root) to highlight gold. May be
  empty (symmetric scales), one, or two entries.
- `title`: heading for the explanation box, e.g. `"WHAT DEFINES DORIAN"`.
- `text`: 1–2 sentence plain-English description of the defining trait and its sound.

Curated values (author may refine wording during implementation; the `characteristic`
intervals are fixed by this spec):

| scale | characteristic | one-line gist (text expands on this) |
|-------|----------------|--------------------------------------|
| major (Ionian) | 3 (M3), 11 (M7) | Bright/resolved; major 3rd = happy, major 7th pulls home. |
| minor (Aeolian) | 3 (♭3) | The ♭3 is the sad/dark core of every minor sound. |
| dorian | 9 (M6) | Minor with a raised 6th — bright, hopeful, "Santana" color. |
| phrygian | 1 (♭2) | Minor with a flat 2nd — dark, Spanish/metal tension. |
| lydian | 6 (#4) | Major with a raised 4th — dreamy, floating, "film score" lift. |
| mixolydian | 10 (♭7) | Major with a flat 7th — bluesy, dominant, rock/funk. |
| locrian | 1 (♭2), 6 (♭5) | Flat 2 **and** flat 5 — unstable, no solid home. |
| harmonic_minor | 11 (M7) | Minor with a raised 7th — the ♭6→7 leap gives the exotic/classical color. |
| melodic_minor | 9 (M6), 11 (M7) | Minor scale with a major top (natural 6 & 7) — jazz minor. |
| pent_major | 3 (M3) | Major scale with the two tension notes (4th, 7th) removed — safe over anything. |
| pent_minor | 3 (♭3) | Minor 3rd + no 2nd/6th — the universal rock/blues box. |
| blues | 6 (♭5) | Minor pentatonic + the ♭5 "blue note" — that flat-five is the whole sound. |
| diminished | 6 (♭5) | Symmetric whole-half pattern — tense, used over dim7/dominant. |
| whole_tone | 6 (#4) | All whole steps, no half steps — dreamlike, no pull home (augmented). |

For symmetric scales (diminished, whole_tone) and the pentatonics, the highlight is a
*representative* tension/identity note and the `text` carries the real explanation (the
pattern / what's missing), since "one defining note" is a weaker concept there.

## Implementation outline

All changes are client-side (HTML/CSS/JS); no Python/server changes.

1. **`static/app.js`**
   - Add `INTERVAL_LABELS` map (table above) and `SCALE_INFO` table (content above).
   - Rewrite `renderFretboard()`:
     - larger dots (`r≈9–10`), wider string spacing;
     - two-line label per dot (note `♯`-spelled + interval) for every scale tone, not just root;
     - gold fill + ring + glow for `characteristic` intervals.
   - Add `renderScaleInfo()` (or fold into the scale-change handlers): populates the formula
     strip and explanation box; called from `setScaleRoot`, `setScaleType`,
     `syncScaleFromServer`, `applyScaleCandidate`, and init.
2. **`templates/index.html`** — inside `#panelScale`, add: formula-strip container (above the
   fretboard), legend row, and explanation box (below the fretboard, above DETECT SCALE).
3. **`static/style.css`** — styles for `.interval-formula`, `.pill` / `.pill.char`,
   `.fretboard-legend`, `.scale-explain`; ensure panel body scrolls if content overflows.

## Testing / verification

This is a visual feature with no automated UI tests in the project. Verification is manual,
in the running app:

- Each of the 14 scales renders without overlap on both 6- and 8-string; labels legible.
- Spot-check intervals against the table: A Dorian shows F♯ as `6` gold; A minor shows C as
  `♭3`; A Mixolydian shows G as `♭7` gold; A blues shows D♯ as `♭5` gold.
- Formula strip degrees match the dots; defining pill(s) gold and match `characteristic`.
- Changing root/scale (and applying a detected candidate) updates dots, formula, and
  explanation together.
- Pure JS interval-mapping helpers (label-for-semitone, build-formula-for-scale) are simple
  and self-contained; add lightweight unit coverage if a JS test path exists, otherwise rely
  on the manual spot-checks above.

## Open questions / future hooks

- Explanation wording is curated here; fine to refine during build.
- Future (not now): tap a formula pill or dot to hear the interval; a compare-to-parent
  overlay. The `SCALE_INFO` structure leaves room for both.
