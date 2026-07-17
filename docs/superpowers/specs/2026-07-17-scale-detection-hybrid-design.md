# Scale Detection v2: Better Analysis + Fretboard Note-Picker (Hybrid)

**Date:** 2026-07-17
**Status:** Approved

## Problem

Scale detection (`detect_scale()` in audio.py) is mostly inaccurate. Two distinct failure modes were identified:

1. **Wrong notes entirely** — `chroma_stft` on the raw loop picks up guitar harmonics and pick attack, polluting the pitch-class profile.
2. **Right notes, wrong root/mode** — modes of the same parent scale (A minor / C major / D dorian…) share identical pitch-class sets; the current 2× root weighting on full-band chroma is too weak a signal to pick the tonal center.

A third structural flaw: the scoring only rewards in-scale energy and never penalizes out-of-scale energy, so wrong scales still score highly.

## Solution Overview

Two complementary pieces that combine into a hybrid flow:

- **Better backend analysis** — cleaner chroma, penalized scoring, bass-weighted root detection.
- **Fretboard note-picker** — the user taps the notes they actually played; those act as a hard filter over the candidates.

Picked notes filter (scale must contain all of them); audio score ranks the survivors. Nothing picked = pure auto-detect. Notes picked with no loop recorded = pure theory matching.

## Part 1: Backend Analysis (audio.py)

Changes inside `detect_scale()`:

1. **Harmonic separation.** Run `librosa.effects.harmonic(y)` on the loop before any chroma work. Removes percussive/attack energy.
2. **`chroma_cqt` instead of `chroma_stft`.** Constant-Q chroma resolves low guitar registers much better. Keep the existing onset-strength weighting of frames.
3. **Penalized template scoring.** Replace the current reward-only score with:
   `score = in_scale_energy − PENALTY × out_of_scale_energy`, normalized by scale size as today. `PENALTY` is a tunable constant (start ~0.7; tune against test fixtures).
4. **Bass-weighted root.** Compute a second chroma vector restricted to the low register (low CQT octaves, roughly below ~200 Hz). Use this bass chroma — not the full-band chroma — for the root-emphasis term. The looped root almost always dominates the bass.

Return shape is unchanged: `{'success': bool, 'candidates': [{root, scale_type, score}]}` — top 5, scores normalized to best = 100.

### New parameter: pitch-class filter

`detect_scale(selected_notes: list[str] | None = None)`:

- `selected_notes` is a list of note names (e.g. `['A', 'C', 'E']`).
- When provided, candidates are filtered to scales whose pitch-class set contains **all** selected notes; ranking of survivors uses the audio score.
- When provided but **no loop is recorded**, skip audio analysis entirely and rank matching scales by fewest extra notes (smallest scale that contains the selection wins). `success = True` in this path.
- When `None`/empty and no loop: current error behavior ("No audio recorded").
- Edge case: selection matches no template (e.g. contains a cluster like C, C#, D, D#, E, F, F#) → `success = True` with empty candidates list; frontend shows "no matching scale".

## Part 2: Frontend Note-Picker (static/app.js, templates/index.html)

- **"PICK NOTES" toggle button** in the scale side panel, near the detect button.
- When active, `renderFretboard()` additionally renders every fret position as a faint clickable dot labeled with its note name. Positions belonging to picked pitch classes render highlighted (visually distinct from scale-note dots).
- Clicking any position toggles its **pitch class** globally — tapping one A lights every A on the board. State is a client-side `Set` of pitch classes.
- A **count + clear** control: "3 notes picked ✕". Clearing empties the set and re-renders.
- Picker state persists while the panel is open; it does not need to survive page reload.
- The existing detect button becomes the single entry point for matching: it sends `{selected_notes: [...]}` with the `detect_scale` socket event. Button is enabled when a loop is playing **or** at least one note is picked (updates the current `state !== 'playing'` guard).
- Results render as the existing candidate chips; tapping a chip applies root + scale type as today.

## Part 3: Wiring (routes.py)

- The `detect_scale` socket handler passes `selected_notes` from the payload through to `AudioEngine.detect_scale()`.

## Testing (tests/test_scale_detection.py)

- **Filter logic (pure, no audio):** selections that match one scale, several scales, and no scale; ranking by fewest extra notes in the no-audio path.
- **Synthetic audio fixtures:** generate sine-mix loops of known scales (root emphasized in a low octave) and assert the correct root+scale appears in the top candidates; assert an out-of-scale-heavy signal does not rank a wrong scale first.
- **Hybrid:** synthetic ambiguous audio (A minor pitch set) + picked note constraint → picked-consistent candidate ranks first.

## Out of Scope

- Per-position (octave-aware) picking — pitch classes only.
- Persisting picked notes across sessions.
- Changing the fretboard's normal scale-visualization rendering.
