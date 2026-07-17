# Scale Detection v2 + Fretboard Note-Picker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make scale detection accurate (harmonic separation, CQT chroma, penalized scoring, bass-weighted root) and add a fretboard note-picker whose picked notes hard-filter the detection candidates.

**Architecture:** Two pure, testable scoring functions (`score_scale_templates`, `match_scales_by_notes`) live at module level in `audio.py`; `WebLooper.detect_scale(selected_notes=None)` orchestrates them across three paths (audio-only, hybrid, theory-only). The frontend adds a PICK NOTES mode to the existing SVG fretboard using event delegation on `data-pc` attributes, and sends picked note names with the existing `detect_scale` socket event.

**Tech Stack:** Python 3 / Flask-SocketIO / librosa / numpy (backend), vanilla JS + inline SVG (frontend), pytest (tests).

**Spec:** `docs/superpowers/specs/2026-07-17-scale-detection-hybrid-design.md`

## Global Constraints

- Return shape of `detect_scale()` is unchanged: `{'success': bool, 'error'?: str, 'candidates': [{root, scale_type, score}]}` — max 5 candidates, scores normalized so top = 100.
- Theory-only path (notes picked, no loop) must work without librosa.
- Note names use the existing `NOTE_NAMES` spelling (`C`, `C#`, `D`, … sharps only).
- Existing tests in `tests/test_scale_detection.py` must keep passing (they may be reorganized but not weakened).
- Commit after each task (user preference).
- Run tests with the project venv: `cd /home/max/looper && ./bin/python -m pytest` (the repo root is a venv; `bin/python` is its interpreter).

---

### Task 1: Pure theory matcher `match_scales_by_notes`

**Files:**
- Modify: `audio.py` (add module-level function near `SCALE_TEMPLATES`, line ~34)
- Test: `tests/test_scale_detection.py`

**Interfaces:**
- Consumes: `SCALE_TEMPLATES`, `NOTE_NAMES` (existing module constants in `audio.py`).
- Produces: `match_scales_by_notes(selected_pcs: set[int]) -> list[dict]` — ALL matching candidates (not capped), sorted best-first, each `{'root': str, 'scale_type': str, '_score': float}`. Score = coverage ratio (`len(selected)/scale_size`) + 0.5 bonus if the root is a selected note. Task 3 consumes this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scale_detection.py`:

```python
from audio import match_scales_by_notes, SCALE_TEMPLATES, NOTE_NAMES


def _pcs(candidate):
    """Pitch-class set of a candidate dict."""
    root = NOTE_NAMES.index(candidate['root'])
    return {(root + iv) % 12 for iv in SCALE_TEMPLATES[candidate['scale_type']]}


def test_match_scales_requires_all_selected_notes():
    """Every returned scale must contain all selected pitch classes."""
    selected = {9, 0, 4}  # A, C, E
    results = match_scales_by_notes(selected)
    assert len(results) > 0
    for c in results:
        assert selected <= _pcs(c)


def test_match_scales_prefers_small_scales_with_selected_root():
    """Top result should be a pentatonic rooted on a selected note (A/C/E)."""
    selected = {9, 0, 4}  # A, C, E
    results = match_scales_by_notes(selected)
    top = results[0]
    assert NOTE_NAMES.index(top['root']) in selected
    assert len(SCALE_TEMPLATES[top['scale_type']]) == 5  # pentatonic beats 7-note


def test_match_scales_root_bonus_beats_coverage_tiebreak():
    """With one selected note, scales rooted on it must outrank same-size scales that merely contain it."""
    results = match_scales_by_notes({9})  # just A
    top = results[0]
    assert top['root'] == 'A'


def test_match_scales_chromatic_cluster_matches_nothing():
    """7 consecutive semitones fit no template."""
    results = match_scales_by_notes({0, 1, 2, 3, 4, 5, 6})
    assert results == []


def test_match_scales_sorted_descending():
    results = match_scales_by_notes({9, 0, 4})
    scores = [c['_score'] for c in results]
    assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/max/looper && ./bin/python -m pytest tests/test_scale_detection.py -v -k match_scales`
Expected: FAIL — `ImportError: cannot import name 'match_scales_by_notes'`

- [ ] **Step 3: Implement `match_scales_by_notes`**

In `audio.py`, directly after the `SCALE_TEMPLATES` dict (after line 34), add:

```python
def match_scales_by_notes(selected_pcs: set) -> list:
    """
    Pure theory matching: return every root+scale whose pitch-class set contains
    ALL of selected_pcs, best-first. Score favors tighter scales (fewer extra
    notes) and roots that are themselves selected.
    """
    candidates = []
    for root_idx, root_name in enumerate(NOTE_NAMES):
        for scale_type, intervals in SCALE_TEMPLATES.items():
            pcs = {(root_idx + iv) % 12 for iv in intervals}
            if not selected_pcs <= pcs:
                continue
            coverage = len(selected_pcs) / len(pcs)
            root_bonus = 0.5 if root_idx in selected_pcs else 0.0
            candidates.append({'root': root_name, 'scale_type': scale_type,
                               '_score': coverage + root_bonus})
    candidates.sort(key=lambda c: c['_score'], reverse=True)
    return candidates
```

Note: `NOTE_NAMES` is defined at line 17, *above* `SCALE_TEMPLATES` — no forward-reference issue.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/max/looper && ./bin/python -m pytest tests/test_scale_detection.py -v`
Expected: all PASS (new + 4 pre-existing).

- [ ] **Step 5: Commit**

```bash
git add audio.py tests/test_scale_detection.py
git commit -m "feat(audio): pure theory scale matcher for picked notes"
```

---

### Task 2: Pure scorer `score_scale_templates` (penalty + bass root)

**Files:**
- Modify: `audio.py` (constants + function next to `match_scales_by_notes`)
- Test: `tests/test_scale_detection.py`

**Interfaces:**
- Consumes: `SCALE_TEMPLATES`, `NOTE_NAMES`.
- Produces: `score_scale_templates(chroma_norm, bass_norm) -> list[dict]` — all 168 candidates sorted best-first, same dict shape as Task 1 (`root`, `scale_type`, `_score`). Both args are length-12 numpy arrays summing to 1 (pitch-class order C=0…B=11). Constants `SCALE_OUT_PENALTY = 0.7`, `ROOT_BASS_WEIGHT = 0.12`. Task 3 consumes this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scale_detection.py`:

```python
from audio import score_scale_templates


def _uniform_chroma(pcs):
    """Length-12 chroma with equal energy on the given pitch classes, sum=1."""
    v = np.zeros(12)
    for pc in pcs:
        v[pc] = 1.0
    return v / v.sum()


A_MINOR_PCS = [9, 11, 0, 2, 4, 5, 7]  # A B C D E F G


def test_score_templates_bass_disambiguates_relative_modes():
    """Same 7 notes, bass says A -> A minor must beat C major and D dorian."""
    chroma = _uniform_chroma(A_MINOR_PCS)
    bass = np.zeros(12)
    bass[9] = 1.0  # all bass energy on A
    results = score_scale_templates(chroma, bass)
    by_key = {(c['root'], c['scale_type']): c['_score'] for c in results}
    assert by_key[('A', 'minor')] > by_key[('C', 'major')]
    assert by_key[('A', 'minor')] > by_key[('D', 'dorian')]
    assert results[0]['root'] == 'A'


def test_score_templates_out_of_scale_energy_penalized():
    """Adding out-of-scale energy must lower a scale's score."""
    clean = _uniform_chroma(A_MINOR_PCS)
    polluted = np.array(clean)
    polluted[1] += 0.3  # C# does not belong to A minor
    polluted = polluted / polluted.sum()
    bass = _uniform_chroma([9])

    def score_of(results, root, stype):
        return next(c['_score'] for c in results
                    if c['root'] == root and c['scale_type'] == stype)

    s_clean = score_of(score_scale_templates(clean, bass), 'A', 'minor')
    s_polluted = score_of(score_scale_templates(polluted, bass), 'A', 'minor')
    assert s_polluted < s_clean


def test_score_templates_size_normalization():
    """Playing only pentatonic notes: the pentatonic must beat the full scale."""
    a_pent_minor = [9, 0, 2, 4, 7]  # A C D E G
    chroma = _uniform_chroma(a_pent_minor)
    bass = _uniform_chroma([9])
    results = score_scale_templates(chroma, bass)
    by_key = {(c['root'], c['scale_type']): c['_score'] for c in results}
    assert by_key[('A', 'pent_minor')] > by_key[('A', 'minor')]


def test_score_templates_returns_all_168_sorted():
    chroma = _uniform_chroma(A_MINOR_PCS)
    bass = _uniform_chroma([9])
    results = score_scale_templates(chroma, bass)
    assert len(results) == 12 * len(SCALE_TEMPLATES)
    scores = [c['_score'] for c in results]
    assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/max/looper && ./bin/python -m pytest tests/test_scale_detection.py -v -k score_templates`
Expected: FAIL — `ImportError: cannot import name 'score_scale_templates'`

- [ ] **Step 3: Implement `score_scale_templates`**

In `audio.py`, directly after `match_scales_by_notes`, add:

```python
# Scoring tunables: penalty for energy on out-of-scale notes, and how much
# the (bass-weighted) root emphasis contributes on top of the template fit.
SCALE_OUT_PENALTY = 0.7
ROOT_BASS_WEIGHT = 0.12


def score_scale_templates(chroma_norm, bass_norm) -> list:
    """
    Score all 12 roots x all templates against a chroma profile.
    Both inputs are length-12 arrays summing to 1 (C=0 .. B=11).
    Fit rewards in-scale energy, penalizes out-of-scale energy, and is
    normalized by scale size; the root term blends bass-register and
    full-band energy at the root to pick the tonal center.
    """
    candidates = []
    for root_idx, root_name in enumerate(NOTE_NAMES):
        for scale_type, intervals in SCALE_TEMPLATES.items():
            pcs = [(root_idx + iv) % 12 for iv in intervals]
            in_e = float(sum(chroma_norm[pc] for pc in pcs))
            out_e = 1.0 - in_e
            base = (in_e - SCALE_OUT_PENALTY * out_e) / len(intervals)
            root_term = 0.5 * float(bass_norm[root_idx]) + 0.5 * float(chroma_norm[root_idx])
            candidates.append({'root': root_name, 'scale_type': scale_type,
                               '_score': base + ROOT_BASS_WEIGHT * root_term})
    candidates.sort(key=lambda c: c['_score'], reverse=True)
    return candidates
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/max/looper && ./bin/python -m pytest tests/test_scale_detection.py -v`
Expected: all PASS. If `test_score_templates_bass_disambiguates_relative_modes` fails, raise `ROOT_BASS_WEIGHT` in small steps (0.12 → 0.15 → 0.2) until it passes while `test_score_templates_size_normalization` still passes — do not weaken the tests.

- [ ] **Step 5: Commit**

```bash
git add audio.py tests/test_scale_detection.py
git commit -m "feat(audio): penalized template scoring with bass-weighted root"
```

---

### Task 3: Rework `detect_scale()` — new pipeline, hybrid filter, theory path, socket wiring

**Files:**
- Modify: `audio.py:1215-1273` (`WebLooper.detect_scale`)
- Modify: `routes.py:173-178` (`handle_detect_scale`)
- Test: `tests/test_scale_detection.py`

**Interfaces:**
- Consumes: `match_scales_by_notes` (Task 1), `score_scale_templates` (Task 2), existing `LIBROSA_AVAILABLE`, `SAMPLE_RATE`, `self.lock`, `self.layers`, `self.master_length`.
- Produces: `WebLooper.detect_scale(selected_notes: list[str] | None = None) -> dict` with the unchanged return shape (Global Constraints). Socket event `detect_scale` now accepts an optional payload `{'selected_notes': ['A', 'C', ...]}`. Frontend (Task 5) relies on: theory path works with no loop; `success=True` with empty `candidates` when the filter eliminates everything.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scale_detection.py`:

```python
def _make_looper_with_signal(signal):
    looper = WebLooper()
    looper.layers = [LoopLayer(0, "Master", signal)]
    looper.master_length = len(signal)
    return looper


def _a_minor_with_bass_drone(seconds=2.0):
    """All 7 A-natural-minor tones in octave 4 plus a strong A2 (110 Hz) drone."""
    n = int(SAMPLE_RATE * seconds)
    t = np.linspace(0, seconds, n, dtype=np.float64)
    # A4 B4 C5 D5 E5 F5 G5
    freqs = [440.0, 493.88, 523.25, 587.33, 659.25, 698.46, 783.99]
    sig = sum(0.1 * np.sin(2 * np.pi * f * t) for f in freqs)
    sig += 0.5 * np.sin(2 * np.pi * 110.0 * t)  # A2 bass drone
    return sig.astype(np.float32)


def test_detect_scale_bass_picks_a_minor_over_relatives():
    """A minor notes + A bass drone: top candidate is root A, minor family."""
    looper = _make_looper_with_signal(_a_minor_with_bass_drone())
    result = looper.detect_scale()
    assert result['success'] is True
    assert result['candidates'][0]['root'] == 'A'
    assert result['candidates'][0]['scale_type'] in ('minor', 'pent_minor')


def test_detect_scale_selected_notes_filter_candidates():
    """Every returned candidate must contain all selected notes."""
    looper = _make_looper_with_signal(_a_minor_with_bass_drone())
    result = looper.detect_scale(selected_notes=['C#'])
    assert result['success'] is True
    cs_pc = NOTE_NAMES.index('C#')
    for c in result['candidates']:
        assert cs_pc in _pcs(c)


def test_detect_scale_theory_mode_without_loop():
    """Notes picked but nothing recorded: pure theory matching, no librosa needed."""
    looper = WebLooper()
    result = looper.detect_scale(selected_notes=['A', 'C', 'E'])
    assert result['success'] is True
    assert 0 < len(result['candidates']) <= 5
    assert result['candidates'][0]['score'] == 100
    selected = {NOTE_NAMES.index(x) for x in ['A', 'C', 'E']}
    for c in result['candidates']:
        assert selected <= _pcs(c)


def test_detect_scale_filter_can_empty_the_list():
    """A selection no scale contains: success with empty candidates."""
    looper = _make_looper_with_signal(_a_minor_with_bass_drone())
    result = looper.detect_scale(
        selected_notes=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#'])
    assert result['success'] is True
    assert result['candidates'] == []


def test_detect_scale_rejects_unknown_note_name():
    looper = WebLooper()
    result = looper.detect_scale(selected_notes=['H'])
    assert result['success'] is False
    assert result['candidates'] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/max/looper && ./bin/python -m pytest tests/test_scale_detection.py -v -k "bass_picks or selected_notes or theory_mode or filter_can or rejects_unknown"`
Expected: FAIL — `TypeError: detect_scale() got an unexpected keyword argument 'selected_notes'` (and wrong-root failures for the bass test).

- [ ] **Step 3: Rewrite `detect_scale()`**

Replace the whole method at `audio.py:1215-1273` with:

```python
    def detect_scale(self, selected_notes=None) -> dict:
        """
        Detect scale from the master loop (harmonic-separated CQT chroma,
        penalized template scoring, bass-weighted root). Optional
        selected_notes (note names the user picked) hard-filter the
        candidates; with notes but no loop, falls back to pure theory
        matching. Returns dict with success flag and top 5 candidates.
        """
        selected_pcs = set()
        if selected_notes:
            try:
                selected_pcs = {NOTE_NAMES.index(n) for n in selected_notes}
            except ValueError:
                return {'success': False, 'error': 'Unknown note name',
                        'candidates': []}

        with self.lock:
            has_audio = len(self.layers) > 0 and self.master_length > 0
            audio = (self.layers[0].buffer[:self.master_length].copy()
                     if has_audio else None)

        if not has_audio:
            if selected_pcs:
                candidates = match_scales_by_notes(selected_pcs)
                return {'success': True,
                        'candidates': _top_scale_candidates(candidates)}
            return {'success': False, 'error': 'No audio recorded',
                    'candidates': []}

        if not LIBROSA_AVAILABLE:
            return {'success': False, 'error': 'librosa not installed',
                    'candidates': []}

        try:
            audio_64 = audio.astype(np.float64)
            harmonic = librosa.effects.harmonic(audio_64)

            # Full-band chroma from the harmonic part, onset-weighted from
            # the original signal (harmonic separation flattens attacks).
            onset_env = librosa.onset.onset_strength(y=audio_64, sr=SAMPLE_RATE)
            chroma = librosa.feature.chroma_cqt(y=harmonic, sr=SAMPLE_RATE)

            n_frames = min(len(onset_env), chroma.shape[1])
            weights = onset_env[:n_frames]
            weighted_sum = weights.sum()
            if weighted_sum > 0:
                chroma_vec = (chroma[:, :n_frames] * weights).sum(axis=1) / weighted_sum
            else:
                chroma_vec = chroma.mean(axis=1)

            total = chroma_vec.sum()
            if total <= 0:
                return {'success': False, 'error': 'No pitched content detected',
                        'candidates': []}
            chroma_norm = chroma_vec / total

            # Bass-register profile (E1-E3) via raw CQT magnitudes folded to
            # pitch classes — the looped tonic usually dominates the bass.
            bass_cqt = np.abs(librosa.cqt(harmonic, sr=SAMPLE_RATE,
                                          fmin=librosa.note_to_hz('E1'),
                                          n_bins=24))
            bass_vec = np.zeros(12)
            for i in range(24):
                bass_vec[(4 + i) % 12] += bass_cqt[i].mean()  # bin 0 = E
            bass_total = bass_vec.sum()
            bass_norm = bass_vec / bass_total if bass_total > 1e-9 else chroma_norm

            candidates = score_scale_templates(chroma_norm, bass_norm)
            if selected_pcs:
                candidates = [
                    c for c in candidates
                    if selected_pcs <= {
                        (NOTE_NAMES.index(c['root']) + iv) % 12
                        for iv in SCALE_TEMPLATES[c['scale_type']]}
                ]
                if not candidates:
                    return {'success': True, 'candidates': []}

            result_candidates = _top_scale_candidates(candidates)
            print(f"✓ Scale detected: {result_candidates[0]['root']} "
                  f"{result_candidates[0]['scale_type']} "
                  f"({result_candidates[0]['score']}%)")
            return {'success': True, 'candidates': result_candidates}

        except Exception as e:
            print(f"✗ Scale detection failed: {e}")
            return {'success': False, 'error': str(e), 'candidates': []}
```

Then add the shared top-5 helper at module level in `audio.py`, after `score_scale_templates`:

```python
def _top_scale_candidates(candidates: list, limit: int = 5) -> list:
    """Cap to the top candidates and normalize scores so the best is 100."""
    top = candidates[:limit]
    if not top:
        return []
    best = top[0]['_score'] if top[0]['_score'] > 0 else 1.0
    return [{'root': c['root'], 'scale_type': c['scale_type'],
             'score': round(max(c['_score'], 0.0) / best * 100)}
            for c in top]
```

- [ ] **Step 4: Update the socket handler**

Replace `routes.py:173-178` with:

```python
@socketio.on('detect_scale')
def handle_detect_scale(data=None):
    """Detect scale from recorded loop and/or picked notes, send result to client."""
    if looper:
        selected = (data or {}).get('selected_notes')
        result = looper.detect_scale(selected_notes=selected)
        emit('scale_detected', result)
```

- [ ] **Step 5: Run the full test file**

Run: `cd /home/max/looper && ./bin/python -m pytest tests/test_scale_detection.py -v`
Expected: all PASS — including the 4 pre-existing tests (A-major-chord top root is still A: with no bass content the bass CQT total falls under the threshold and root weighting falls back to full-band chroma, where A dominates). If `test_detect_scale_bass_picks_a_minor_over_relatives` fails on `scale_type`, print the top-5 for debugging — root 'A' is the hard requirement; if root is wrong, tune `ROOT_BASS_WEIGHT` upward as described in Task 2 Step 4.

- [ ] **Step 6: Run the whole suite to catch regressions**

Run: `cd /home/max/looper && ./bin/python -m pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add audio.py routes.py tests/test_scale_detection.py
git commit -m "feat(audio): rework scale detection - harmonic cqt chroma, bass root, hybrid note filter"
```

---

### Task 4: Frontend PICK NOTES mode on the fretboard

**Files:**
- Modify: `templates/index.html` (scale panel, around line 144)
- Modify: `static/app.js` (`renderFretboard` at ~255, new state + handlers near scale state at ~115)
- Modify: `static/style.css` (after `.scale-candidate-chip` rules, ~line 1379)

**Interfaces:**
- Consumes: existing `renderFretboard()`, `SCALE_NOTES`, `OPEN_STRINGS_6/8`, `guitarMode`.
- Produces: globals `pickMode: boolean`, `pickedNotes: Set<number>` (pitch classes 0-11, C=0); functions `togglePickMode()`, `clearPickedNotes()`, `updatePickedUI()`. Task 5 reads `pickedNotes` and calls `updatePickedUI()`'s conventions. No backend interaction in this task.

- [ ] **Step 1: Add the HTML row**

In `templates/index.html`, insert between the `.scale-explain` div (ends line 143) and the detect button (line 144):

```html
                <div class="pick-notes-row">
                    <button class="pick-notes-btn" id="pickNotesBtn"
                            onclick="togglePickMode()">🎯 PICK NOTES</button>
                    <span class="picked-count" id="pickedCount"></span>
                    <button class="clear-picked-btn" id="clearPickedBtn"
                            onclick="clearPickedNotes()" style="display:none">✕ CLEAR</button>
                </div>
```

- [ ] **Step 2: Add the CSS**

In `static/style.css`, after the `.scale-candidate-chip .chip-score` rule (line 1379), add:

```css
.pick-notes-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
}
.pick-notes-btn {
    padding: 6px 12px;
    background: var(--surface);
    border: 1px solid var(--border-dim);
    border-radius: 0;
    color: var(--text-muted);
    font-size: 0.78em;
    font-weight: 600;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.15s;
}
.pick-notes-btn:hover { color: var(--text); border-color: var(--accent); }
.pick-notes-btn.active { color: #4f9cf1; border-color: #4f9cf1; }
.picked-count { font-size: 0.75em; color: var(--text-muted); }
.clear-picked-btn {
    padding: 4px 8px;
    background: none;
    border: 1px solid var(--border-dim);
    border-radius: 0;
    color: var(--text-muted);
    font-size: 0.7em;
    cursor: pointer;
}
.clear-picked-btn:hover { color: var(--text); border-color: var(--accent); }
```

- [ ] **Step 3: Add state and handlers in app.js**

In `static/app.js`, after `let scaleType = 'minor';` (line 116), add:

```js
        // Note-picker state: pitch classes (0-11, C=0) the user tapped
        let pickMode = false;
        let pickedNotes = new Set();

        function togglePickMode() {
            pickMode = !pickMode;
            document.getElementById('pickNotesBtn').classList.toggle('active', pickMode);
            renderFretboard();
            updatePickedUI();
        }

        function clearPickedNotes() {
            pickedNotes.clear();
            renderFretboard();
            updatePickedUI();
        }

        function togglePickedNote(pc) {
            if (pickedNotes.has(pc)) pickedNotes.delete(pc);
            else pickedNotes.add(pc);
            renderFretboard();
            updatePickedUI();
        }

        function updatePickedUI() {
            const count = document.getElementById('pickedCount');
            const clearBtn = document.getElementById('clearPickedBtn');
            const n = pickedNotes.size;
            count.textContent = n === 0 ? '' : `${n} note${n > 1 ? 's' : ''} picked`;
            clearBtn.style.display = n > 0 ? '' : 'none';
        }
```

- [ ] **Step 4: Make the fretboard render pick mode**

In `renderFretboard()`'s note-dot loop (`static/app.js`, the `for (let f = fStart; ...)` block at lines 316-331), replace the loop body with:

```js
                    for (let f = fStart; f <= fretHi; f++) {
                        const noteIdx = (OPEN_STRINGS[s] + f) % 12;
                        const interval = (noteIdx - rootIdx + 12) % 12;
                        const inScale = intervals.has(interval);
                        const isPicked = pickedNotes.has(noteIdx);
                        if (!pickMode && !inScale) continue;
                        const isRoot = interval === 0;
                        const isChar = charSet.has(interval);
                        const x = noteX(f);
                        const noteName = SCALE_NOTES[noteIdx].replace('#', '♯');
                        let dot = '';
                        if (inScale) {
                            const fill = isRoot ? '#ed8936' : (isChar ? '#f6c453' : '#4fd1c5');
                            if (isChar) {
                                dot += `<circle cx="${x}" cy="${y}" r="${DOT_R + 2.5}" fill="none" stroke="#f6c453" stroke-width="1.4" opacity="0.9"/>`;
                            }
                            dot += `<circle cx="${x}" cy="${y}" r="${DOT_R}" fill="${fill}"${isChar ? ' filter="url(#goldGlow)"' : ''} opacity="0.95"/>`;
                            dot += `<text x="${x}" y="${y - 1.5}" text-anchor="middle" font-size="8" font-weight="bold" fill="#15202b">${noteName}</text>`;
                            dot += `<text x="${x}" y="${y + 7}" text-anchor="middle" font-size="6" fill="#15202b" opacity="0.82">${INTERVAL_LABELS[interval]}</text>`;
                        } else {
                            // Pick mode only: faint dot for out-of-scale positions
                            dot += `<circle cx="${x}" cy="${y}" r="${DOT_R}" fill="#2d3748" opacity="0.5"/>`;
                            dot += `<text x="${x}" y="${y + 2.5}" text-anchor="middle" font-size="8" fill="#8b96a5" opacity="0.7">${noteName}</text>`;
                        }
                        if (isPicked) {
                            dot += `<circle cx="${x}" cy="${y}" r="${DOT_R + 3}" fill="none" stroke="#4f9cf1" stroke-width="2"/>`;
                        }
                        if (pickMode) {
                            out += `<g data-pc="${noteIdx}" style="cursor:pointer">${dot}</g>`;
                        } else {
                            out += dot;
                        }
                    }
```

(The `const y = stringY(s);` line above the loop stays as is.)

- [ ] **Step 5: Add the delegated click listener**

In `static/app.js`, find the init section at the bottom where `setGuitarMode(guitarMode);` is called (line ~2141) and add just before it:

```js
        document.getElementById('fretboard').addEventListener('click', (e) => {
            if (!pickMode) return;
            const g = e.target.closest('[data-pc]');
            if (!g) return;
            togglePickedNote(parseInt(g.dataset.pc, 10));
        });
```

- [ ] **Step 6: Manually verify**

Start the app (`cd /home/max/looper && ./bin/python main.py`, open `http://<pi>:5000`), open the SCALE panel:
- Click PICK NOTES → button highlights; every fret position now shows a dot (scale notes colored, others faint gray).
- Tap any A → every A on the board gets a blue ring; count reads "1 note picked"; CLEAR appears.
- Tap the same A again → rings disappear, count empties.
- Tap 3 different notes → "3 notes picked".
- CLEAR → all rings gone.
- Toggle PICK NOTES off → fretboard returns to normal scale view (no faint dots), rings hidden.
- Change scale root/type while picking → picked rings survive re-render.

- [ ] **Step 7: Commit**

```bash
git add templates/index.html static/app.js static/style.css
git commit -m "feat(ui): pick-notes mode on the fretboard"
```

---

### Task 5: Frontend hybrid detect wiring

**Files:**
- Modify: `static/app.js` (`detectScale` at ~180, `handleScaleDetected` at ~194, detect-button enablement at ~1848; `updatePickedUI` from Task 4)
- Modify: `static/style.css` (one rule for the no-match message)

**Interfaces:**
- Consumes: `pickedNotes`, `updatePickedUI()` (Task 4); socket event `detect_scale` accepting `{selected_notes}` (Task 3); `serverState`, `isDetectingScale`, `SCALE_NOTES` (existing globals).
- Produces: user-facing hybrid flow — detect works with a loop, with picked notes, or both.

- [ ] **Step 1: Extract a single detect-button enablement helper**

In `static/app.js`, add after `updatePickedUI()` (Task 4):

```js
        function updateDetectScaleBtn() {
            const btn = document.getElementById('detectScaleBtn');
            if (!btn) return;
            const canDetect = serverState.state === 'playing' || pickedNotes.size > 0;
            btn.disabled = !canDetect || isDetectingScale;
        }
```

Replace the existing block at lines 1848-1851:

```js
            const btnDetectScale = document.getElementById('detectScaleBtn');
            if (btnDetectScale) {
                btnDetectScale.disabled = (state !== 'playing' || isDetectingScale);
            }
```

with:

```js
            updateDetectScaleBtn();
```

And append `updateDetectScaleBtn();` as the last line of `updatePickedUI()` so picking notes enables the button immediately.

- [ ] **Step 2: Send picked notes with the detect request**

Replace `detectScale()` (lines 180-192) with:

```js
        function detectScale() {
            if (isDetectingScale) return;
            if (serverState.state !== 'playing' && pickedNotes.size === 0) {
                alert('Record a loop or pick some notes first');
                return;
            }
            isDetectingScale = true;
            const btn = document.getElementById('detectScaleBtn');
            btn.classList.add('detecting');
            btn.textContent = '🔍 DETECTING...';
            document.getElementById('scaleCandidates').classList.remove('visible');
            socket.emit('detect_scale', {
                selected_notes: [...pickedNotes].map(pc => SCALE_NOTES[pc])
            });
        }
```

- [ ] **Step 3: Handle the empty-after-filter result**

In `handleScaleDetected()` (lines 194-216), replace the failure guard:

```js
            if (!result.success || !result.candidates || result.candidates.length === 0) {
                alert(result.error || 'Scale detection failed');
                return;
            }
```

with:

```js
            if (!result.success) {
                alert(result.error || 'Scale detection failed');
                return;
            }
            const container = document.getElementById('scaleCandidates');
            if (result.candidates.length === 0) {
                container.innerHTML = '<div class="no-match-msg">No scale matches those notes</div>';
                container.classList.add('visible');
                return;
            }
```

and delete the now-duplicate `const container = ...` line just below (line 205).

Add to `static/style.css` next to the chip rules:

```css
.no-match-msg { font-size: 0.78em; color: var(--text-muted); padding: 5px 0; }
```

- [ ] **Step 4: Run the backend suite (unchanged but cheap insurance)**

Run: `cd /home/max/looper && ./bin/python -m pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 5: Manually verify the hybrid flow end-to-end**

With the app running:
1. No loop, no picked notes → DETECT SCALE disabled.
2. No loop, pick A + C + E → button enables; detect → chips appear (theory matches, pentatonics rooted on picked notes first); tap a chip → fretboard applies that scale.
3. Record a loop in a known key, no picked notes → detect → candidates should now be plausibly right (better than before).
4. Same loop, pick 2-3 notes you actually played → detect → list narrows to scales containing them.
5. Same loop, pick 7 chromatic notes (C through F#) → detect → "No scale matches those notes".

- [ ] **Step 6: Commit**

```bash
git add static/app.js static/style.css
git commit -m "feat(ui): hybrid scale detect - picked notes filter candidates"
```
