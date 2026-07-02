# Scale Detection Design

**Date:** 2026-05-03  
**Status:** Approved

## Overview

Add a manual "Detect Scale" feature that analyzes the recorded loop's audio, scores it against 168 scale candidates (14 types × 12 roots), and presents a ranked shortlist the user can click to apply to the fretboard visualizer.

Because a melody/chord progression can fit multiple scales, the result is always a ranked list — not a single answer. The user picks the one that matches their intent.

## Backend — `audio.py`

Add `detect_scale()` method to `WebLooper`, following the same pattern as `detect_tempo()`:

1. Copy master loop audio under lock, process outside lock
2. Compute onset envelope: `librosa.onset.onset_strength(y, sr)`
3. Compute chroma: `librosa.feature.chroma_stft(y, sr)`, weight each frame by its onset strength, average into a 12-element chroma vector
4. Score all 168 candidates (14 scale types × 12 roots) by dot-product of the chroma vector against each scale's binary pitch-class template
5. Normalize scores to percentages, sort descending, return top 5

**Scale types (14):**
- Diatonic modes: Major (Ionian), Dorian, Phrygian, Lydian, Mixolydian, Natural Minor (Aeolian), Locrian
- Minor variants: Harmonic Minor, Melodic Minor
- Pentatonics: Major Pentatonic, Minor Pentatonic
- Blues (minor pentatonic + ♭5)
- Symmetric: Diminished (whole-half), Whole Tone

**Return shape:**
```json
{
  "success": true,
  "candidates": [
    {"root": "A", "scale_type": "minor", "score": 91},
    {"root": "C", "scale_type": "major", "score": 91},
    ...
  ]
}
```

Falls back gracefully if librosa is unavailable (same guard as `detect_tempo`).

## Socket — `routes.py`

Add one handler mirroring `detect_tempo`:

```
client emits 'detect_scale'
→ server calls looper.detect_scale()
→ server emits 'scale_detected' with result dict
```

No new HTTP routes.

## UI — `app.js` / `index.html`

**Trigger:** "Detect Scale" button placed near the scale root/type picker in the scale visualizer panel.

**Loading state:** Button gets a `.detecting` class (spinner) while waiting, same pattern as Detect Tempo button.

**Results:** On `scale_detected` event, render a row of clickable chips below the fretboard:
- Each chip: `A minor 91%` — styled to match existing UI palette
- Clicking a chip fires `set_scale` command → fretboard updates → chips clear

**Stale result clearing:** Candidate chips are cleared when:
- User manually changes root or scale type
- User records a new loop (`clear_all` or new master recording starts)

## Constraints

- Detection is manual only — no auto-trigger on loop stop
- Runs outside the audio callback thread (no real-time impact)
- RPi-friendly: chroma + onset weighting is fast (~100–200ms for a typical loop)
- Requires librosa (already a project dependency)
