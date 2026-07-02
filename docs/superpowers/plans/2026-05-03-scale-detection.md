# Scale Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a manual "Detect Scale" feature that analyzes the recorded loop with onset-weighted chroma, scores 168 candidates (14 scale types × 12 roots), and shows a ranked clickable chip list that updates the fretboard on selection.

**Architecture:** `detect_scale()` in `audio.py` uses librosa chroma + onset strength (same lock/copy pattern as `detect_tempo`). A new socket event `detect_scale` → `scale_detected` carries the result to the frontend. The UI adds a button + chips container in the scale section; clicking a chip fires the existing `set_scale` command.

**Tech Stack:** Python (librosa, numpy), Flask-SocketIO, vanilla JS, SVG fretboard (existing)

---

## File Map

| File | Change |
|------|--------|
| `audio.py` | Add `SCALE_TEMPLATES`, `NOTE_NAMES` (module-level); add `detect_scale()` to `WebLooper` |
| `routes.py` | Add `handle_detect_scale` socket handler |
| `static/app.js` | Expand `SCALE_INTERVALS` (8→14 types); add `SCALE_LABELS`; add `detectScale`, `handleScaleDetected`, `applyScaleCandidate`, `clearScaleCandidates`; wire `scale_detected` event; clear chips on state change and manual root/type change |
| `templates/index.html` | Add 6 new `<option>` tags to scale select; add detect-scale button + candidates container in scale section |
| `static/style.css` | Add `.detect-scale-btn`, `.scale-candidates`, `.scale-candidate-chip` styles |
| `tests/test_scale_detection.py` | Unit test for `detect_scale()` |

---

## Task 1: Backend — scale templates and `detect_scale()`

**Files:**
- Modify: `audio.py` (after the librosa import block ~line 24; new method after `detect_tempo` ~line 998)
- Create: `tests/test_scale_detection.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scale_detection.py`:

```python
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from audio import WebLooper, LoopLayer, SAMPLE_RATE


def test_detect_scale_returns_five_candidates():
    """detect_scale() should return 5 ranked candidates for any audio."""
    looper = WebLooper()
    n = int(SAMPLE_RATE * 2.0)
    t = np.linspace(0, 2.0, n, dtype=np.float32)
    # A major chord: A4 + C#5 + E5
    signal = (
        0.4 * np.sin(2 * np.pi * 440.0 * t) +
        0.3 * np.sin(2 * np.pi * 554.37 * t) +
        0.2 * np.sin(2 * np.pi * 659.25 * t)
    )
    looper.layers = [LoopLayer(0, "Master", signal)]
    looper.master_length = n

    result = looper.detect_scale()

    assert result['success'] is True
    assert len(result['candidates']) == 5
    assert result['candidates'][0]['score'] == 100  # top is always 100
    for c in result['candidates']:
        assert 0 <= c['score'] <= 100
        assert c['root'] in ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        assert 'scale_type' in c


def test_detect_scale_top_root_is_a_for_a_major_chord():
    """A major chord should rank A as the top root."""
    looper = WebLooper()
    n = int(SAMPLE_RATE * 2.0)
    t = np.linspace(0, 2.0, n, dtype=np.float32)
    signal = (
        0.4 * np.sin(2 * np.pi * 440.0 * t) +
        0.3 * np.sin(2 * np.pi * 554.37 * t) +
        0.2 * np.sin(2 * np.pi * 659.25 * t)
    )
    looper.layers = [LoopLayer(0, "Master", signal)]
    looper.master_length = n

    result = looper.detect_scale()

    assert result['candidates'][0]['root'] == 'A'


def test_detect_scale_fails_gracefully_with_no_audio():
    """detect_scale() should return success=False when no audio is loaded."""
    looper = WebLooper()
    result = looper.detect_scale()
    assert result['success'] is False
    assert result['candidates'] == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/max/looper && python -m pytest tests/test_scale_detection.py -v
```

Expected: `AttributeError: 'WebLooper' object has no attribute 'detect_scale'`

- [ ] **Step 3: Add `SCALE_TEMPLATES` and `NOTE_NAMES` to `audio.py`**

Insert after line 24 (after the `from config import ...` block), before the librosa try/except:

```python
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

SCALE_TEMPLATES = {
    'major':          [0, 2, 4, 5, 7, 9, 11],
    'dorian':         [0, 2, 3, 5, 7, 9, 10],
    'phrygian':       [0, 1, 3, 5, 7, 8, 10],
    'lydian':         [0, 2, 4, 6, 7, 9, 11],
    'mixolydian':     [0, 2, 4, 5, 7, 9, 10],
    'minor':          [0, 2, 3, 5, 7, 8, 10],
    'locrian':        [0, 1, 3, 5, 6, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor':  [0, 2, 3, 5, 7, 9, 11],
    'pent_major':     [0, 2, 4, 7, 9],
    'pent_minor':     [0, 3, 5, 7, 10],
    'blues':          [0, 3, 5, 6, 7, 10],
    'diminished':     [0, 2, 3, 5, 6, 8, 9, 11],
    'whole_tone':     [0, 2, 4, 6, 8, 10],
}
```

- [ ] **Step 4: Add `detect_scale()` method to `WebLooper`**

Insert after the closing of `detect_tempo()` (~line 998), before the `# EXPORT FUNCTIONS` section:

```python
def detect_scale(self) -> dict:
    """
    Detect scale from the master loop using onset-weighted chroma analysis.
    Returns dict with success flag and top 5 scale candidates ranked by fit.
    """
    if not LIBROSA_AVAILABLE:
        return {'success': False, 'error': 'librosa not installed', 'candidates': []}

    with self.lock:
        if len(self.layers) == 0 or self.master_length == 0:
            return {'success': False, 'error': 'No audio recorded', 'candidates': []}
        audio = self.layers[0].buffer[:self.master_length].copy()

    try:
        audio_64 = audio.astype(np.float64)

        onset_env = librosa.onset.onset_strength(y=audio_64, sr=SAMPLE_RATE)
        chroma = librosa.feature.chroma_stft(y=audio_64, sr=SAMPLE_RATE)

        # Weight each chroma frame by its onset strength, then average
        n_frames = min(len(onset_env), chroma.shape[1])
        weights = onset_env[:n_frames]
        weighted_sum = weights.sum()
        if weighted_sum > 0:
            chroma_vec = (chroma[:, :n_frames] * weights).sum(axis=1) / weighted_sum
        else:
            chroma_vec = chroma.mean(axis=1)

        total = chroma_vec.sum()
        if total <= 0:
            return {'success': False, 'error': 'No pitched content detected', 'candidates': []}
        chroma_norm = chroma_vec / total

        # Score all 14 × 12 = 168 candidates
        candidates = []
        for root_idx, root_name in enumerate(NOTE_NAMES):
            for scale_type, intervals in SCALE_TEMPLATES.items():
                pitch_classes = [(root_idx + iv) % 12 for iv in intervals]
                raw = sum(float(chroma_norm[pc]) for pc in pitch_classes)
                # Adjust for scale size so pentatonic vs heptatonic are comparable
                adjusted = raw / (len(intervals) / 12)
                candidates.append({'root': root_name, 'scale_type': scale_type, '_score': adjusted})

        candidates.sort(key=lambda c: c['_score'], reverse=True)
        top5 = candidates[:5]

        best = top5[0]['_score'] if top5[0]['_score'] > 0 else 1.0
        result_candidates = [
            {'root': c['root'], 'scale_type': c['scale_type'], 'score': round(c['_score'] / best * 100)}
            for c in top5
        ]

        print(f"✓ Scale detected: {result_candidates[0]['root']} {result_candidates[0]['scale_type']} ({result_candidates[0]['score']}%)")
        return {'success': True, 'candidates': result_candidates}

    except Exception as e:
        print(f"✗ Scale detection failed: {e}")
        return {'success': False, 'error': str(e), 'candidates': []}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/max/looper && python -m pytest tests/test_scale_detection.py -v
```

Expected: all 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add -f tests/test_scale_detection.py audio.py
git commit -m "feat: add detect_scale() with onset-weighted chroma and 14 scale templates"
```

---

## Task 2: Socket handler in `routes.py`

**Files:**
- Modify: `routes.py` (after `handle_detect_tempo` at line ~167)

- [ ] **Step 1: Add the socket handler**

Insert after the `handle_detect_tempo` function (after line 167):

```python
@socketio.on('detect_scale')
def handle_detect_scale():
    """Detect scale from recorded loop and send result to client."""
    if looper:
        result = looper.detect_scale()
        emit('scale_detected', result)
```

- [ ] **Step 2: Verify the handler works**

Start the server and open the browser console. With a loop playing, run:
```javascript
socket.emit('detect_scale')
// Then listen: socket.on('scale_detected', console.log)
```

Expected: a JSON object with `success: true` and `candidates: [...]` in the console.

- [ ] **Step 3: Commit**

```bash
git add routes.py
git commit -m "feat: add detect_scale socket handler"
```

---

## Task 3: Expand scale data in frontend

**Files:**
- Modify: `static/app.js` lines 44–53 (SCALE_INTERVALS dict)
- Modify: `templates/index.html` lines 121–130 (scale type select options)

- [ ] **Step 1: Expand `SCALE_INTERVALS` in `app.js` and add `SCALE_LABELS`**

Replace lines 44–53 in `static/app.js`:

```javascript
        const SCALE_INTERVALS = {
            'minor':          [0, 2, 3, 5, 7, 8, 10],
            'major':          [0, 2, 4, 5, 7, 9, 11],
            'dorian':         [0, 2, 3, 5, 7, 9, 10],
            'phrygian':       [0, 1, 3, 5, 7, 8, 10],
            'lydian':         [0, 2, 4, 6, 7, 9, 11],
            'mixolydian':     [0, 2, 4, 5, 7, 9, 10],
            'locrian':        [0, 1, 3, 5, 6, 8, 10],
            'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
            'melodic_minor':  [0, 2, 3, 5, 7, 9, 11],
            'pent_major':     [0, 2, 4, 7, 9],
            'pent_minor':     [0, 3, 5, 7, 10],
            'blues':          [0, 3, 5, 6, 7, 10],
            'diminished':     [0, 2, 3, 5, 6, 8, 9, 11],
            'whole_tone':     [0, 2, 4, 6, 8, 10],
        };
        const SCALE_LABELS = {
            'minor':          'Minor',
            'major':          'Major',
            'dorian':         'Dorian',
            'phrygian':       'Phrygian',
            'lydian':         'Lydian',
            'mixolydian':     'Mixolydian',
            'locrian':        'Locrian',
            'harmonic_minor': 'Harm. Minor',
            'melodic_minor':  'Mel. Minor',
            'pent_major':     'Pent. Major',
            'pent_minor':     'Pent. Minor',
            'blues':          'Blues',
            'diminished':     'Diminished',
            'whole_tone':     'Whole Tone',
        };
```

- [ ] **Step 2: Add 6 new options to the scale type select in `index.html`**

Replace lines 121–130 in `templates/index.html`:

```html
                <select class="scale-type-select" id="scaleTypeSelect" onchange="setScaleType(this.value)">
                    <option value="minor">Natural Minor</option>
                    <option value="major">Major</option>
                    <option value="dorian">Dorian</option>
                    <option value="phrygian">Phrygian</option>
                    <option value="lydian">Lydian</option>
                    <option value="mixolydian">Mixolydian</option>
                    <option value="locrian">Locrian</option>
                    <option value="harmonic_minor">Harm. Minor</option>
                    <option value="melodic_minor">Mel. Minor</option>
                    <option value="pent_minor">Pent. Minor</option>
                    <option value="pent_major">Pent. Major</option>
                    <option value="blues">Blues</option>
                    <option value="diminished">Diminished</option>
                    <option value="whole_tone">Whole Tone</option>
                </select>
```

- [ ] **Step 3: Verify fretboard still renders for all new scale types**

Open the browser, select each new scale type from the dropdown, confirm dots appear on the fretboard. No console errors.

- [ ] **Step 4: Commit**

```bash
git add static/app.js templates/index.html
git commit -m "feat: expand scale types to 14 in frontend (add lydian, locrian, harmonic/melodic minor, diminished, whole tone)"
```

---

## Task 4: Detect Scale UI — button, chips, JS logic, CSS

**Files:**
- Modify: `templates/index.html` (scale section, after fretboard container)
- Modify: `static/app.js` (new functions + socket event + state-change clearing)
- Modify: `static/style.css` (new CSS rules at end of file)

- [ ] **Step 1: Add button and candidates container to `index.html`**

Replace the scale section (lines 117–136) with:

```html
        <!-- Scale visualizer -->
        <div class="section scale-section">
            <div class="scale-header">
                <span class="section-label">Scale</span>
                <select class="scale-type-select" id="scaleTypeSelect" onchange="setScaleType(this.value)">
                    <option value="minor">Natural Minor</option>
                    <option value="major">Major</option>
                    <option value="dorian">Dorian</option>
                    <option value="phrygian">Phrygian</option>
                    <option value="lydian">Lydian</option>
                    <option value="mixolydian">Mixolydian</option>
                    <option value="locrian">Locrian</option>
                    <option value="harmonic_minor">Harm. Minor</option>
                    <option value="melodic_minor">Mel. Minor</option>
                    <option value="pent_minor">Pent. Minor</option>
                    <option value="pent_major">Pent. Major</option>
                    <option value="blues">Blues</option>
                    <option value="diminished">Diminished</option>
                    <option value="whole_tone">Whole Tone</option>
                </select>
            </div>
            <div class="scale-root-row" id="scaleRootRow"></div>
            <div class="fretboard-container">
                <div id="fretboard"></div>
            </div>
            <button class="detect-scale-btn" id="detectScaleBtn" onclick="detectScale()" disabled>
                🔍 DETECT SCALE
            </button>
            <div class="scale-candidates" id="scaleCandidates"></div>
        </div>
```

(This also incorporates the Task 3 select changes, so Task 3 Step 2 can be skipped if this task runs after Task 3.)

- [ ] **Step 2: Add scale detection JS functions to `app.js`**

Insert after the closing of the `syncScaleFromServer` function (after line 96, before the `renderFretboard` function):

```javascript
        // =================================================================
        // SCALE DETECTION
        // =================================================================

        let isDetectingScale = false;

        function detectScale() {
            if (isDetectingScale) return;
            if (serverState.state !== 'playing') {
                alert('Record a loop first before detecting scale');
                return;
            }
            isDetectingScale = true;
            const btn = document.getElementById('detectScaleBtn');
            btn.classList.add('detecting');
            btn.textContent = '🔍 DETECTING...';
            document.getElementById('scaleCandidates').classList.remove('visible');
            socket.emit('detect_scale');
        }

        function handleScaleDetected(result) {
            isDetectingScale = false;
            const btn = document.getElementById('detectScaleBtn');
            btn.classList.remove('detecting');
            btn.textContent = '🔍 DETECT SCALE';

            if (!result.success || !result.candidates || result.candidates.length === 0) {
                alert(result.error || 'Scale detection failed');
                return;
            }

            const container = document.getElementById('scaleCandidates');
            container.innerHTML = result.candidates.map(c => {
                const label = SCALE_LABELS[c.scale_type] || c.scale_type;
                return `<button class="scale-candidate-chip"
                    onclick="applyScaleCandidate('${c.root}', '${c.scale_type}')">
                    ${c.root} ${label} <span class="chip-score">${c.score}%</span>
                </button>`;
            }).join('');
            container.classList.add('visible');
        }

        function applyScaleCandidate(root, type) {
            scaleRoot = root;
            scaleType = type;
            updateScaleRootButtons();
            const sel = document.getElementById('scaleTypeSelect');
            if (sel) sel.value = scaleType;
            renderFretboard();
            sendCommand('set_scale', { root: scaleRoot, scale_type: scaleType });
            clearScaleCandidates();
        }

        function clearScaleCandidates() {
            const el = document.getElementById('scaleCandidates');
            if (el) el.classList.remove('visible');
        }
```

- [ ] **Step 3: Clear candidates when root/type is changed manually**

In `setScaleRoot` (line 72–77), add `clearScaleCandidates();` before `sendCommand`:

```javascript
        function setScaleRoot(note) {
            scaleRoot = note;
            updateScaleRootButtons();
            renderFretboard();
            clearScaleCandidates();
            sendCommand('set_scale', { root: scaleRoot, scale_type: scaleType });
        }
```

In `setScaleType` (lines 79–83), add `clearScaleCandidates();` before `sendCommand`:

```javascript
        function setScaleType(type) {
            scaleType = type;
            renderFretboard();
            clearScaleCandidates();
            sendCommand('set_scale', { root: scaleRoot, scale_type: scaleType });
        }
```

- [ ] **Step 4: Wire `scale_detected` socket event and enable/disable button**

In the `connect()` function, after the `tempo_detected` handler (line ~412):

```javascript
            socket.on('scale_detected', (result) => {
                handleScaleDetected(result);
            });
```

In the `updateUI` section that handles `detectTempoBtn` (around line 1309), add the detect scale button:

```javascript
            // DETECT SCALE button - only enabled when playing
            const btnDetectScale = document.getElementById('detectScaleBtn');
            if (btnDetectScale) {
                btnDetectScale.disabled = (state !== 'playing' || isDetectingScale);
            }
```

- [ ] **Step 5: Clear candidates when loop is cleared or a new recording starts**

In the `socket.on('update', ...)` handler (line 379), add stale-clearing before `updateUI()`:

```javascript
            socket.on('update', (data) => {
                const prevState = serverState.state;
                serverState = data;
                if (prevState === 'playing' &&
                    (data.state === 'idle' || data.state === 'recording_master')) {
                    clearScaleCandidates();
                }
                updateUI();
            });
```

- [ ] **Step 6: Add CSS**

Append to `static/style.css`:

```css
/* Scale detection */
.detect-scale-btn {
    width: 100%;
    margin-top: 10px;
    padding: 9px;
    background: var(--surface-2);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius);
    color: var(--text-dim);
    font-size: 0.82em;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.15s;
}
.detect-scale-btn:not(:disabled):hover { color: var(--text); border-color: var(--accent); }
.detect-scale-btn:disabled { opacity: 0.4; cursor: default; }
.detect-scale-btn.detecting { color: var(--armed); border-color: var(--armed); }

.scale-candidates {
    display: none;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
}
.scale-candidates.visible { display: flex; }

.scale-candidate-chip {
    padding: 5px 10px;
    background: var(--surface-2);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-sm);
    color: var(--text-dim);
    font-size: 0.78em;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    gap: 5px;
}
.scale-candidate-chip:hover { border-color: var(--accent); color: var(--text); }
.scale-candidate-chip .chip-score { color: var(--text-muted); font-size: 0.88em; }
```

- [ ] **Step 7: End-to-end smoke test**

1. Start the server: `python main.py`
2. Open browser, record a short loop (a few notes or a chord)
3. Click "🔍 DETECT SCALE" — button should show "🔍 DETECTING..." briefly
4. 5 chips appear, e.g. "A Minor 100%", "C Major 91%", etc.
5. Click one chip — fretboard updates, chips disappear
6. Try the select dropdown — all 14 scales render dots on the fretboard
7. Clear all loops — chips are gone if they were visible

- [ ] **Step 8: Commit**

```bash
git add templates/index.html static/app.js static/style.css
git commit -m "feat: add Detect Scale UI — button, ranked chips, fretboard update on selection"
```
