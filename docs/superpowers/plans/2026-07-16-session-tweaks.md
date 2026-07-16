# Session Tweaks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Four live-session fixes: a delete button for loops, keyboard shortcuts that don't fire while typing, a trim Reset that restores the original take, and a 24-fret two-row fretboard.

**Architecture:** All four are independent. Task 1 adds a volatile pre-trim backup to `WebLooper` (audio.py) with a `reset_trim()` method, TDD'd in a new test file. Task 2 wires it through the socket command dispatch (routes.py) and the trim editor UI (app.js). Tasks 3–5 are frontend-only changes to app.js (+ one CSS rule).

**Tech Stack:** Python (numpy, Flask-SocketIO), vanilla JS single-file frontend, pytest.

**Spec:** `docs/superpowers/specs/2026-07-16-session-tweaks-design.md`

## Global Constraints

- No new dependencies.
- No session-format changes — the trim backup is in-memory only, never written to disk.
- Commit after each task (user rule).
- Run the full suite (`python -m pytest tests/ -q`) before each commit; it must stay green.

---

### Task 1: Backend — pre-trim backup and `reset_trim()`

**Files:**
- Modify: `audio.py` (WebLooper `__init__` ~line 134; `_finalize_overdub` ~line 364; `start_recording` ~line 451; `clear_all` ~line 598; `load_session` ~line 829; `apply_trim` ~line 996; `get_state` snapshot ~line 1577 and trim block ~line 1646)
- Test: `tests/test_trim_reset.py` (new)

**Interfaces:**
- Produces: `WebLooper.reset_trim() -> bool`; `WebLooper._pre_trim_backup: np.ndarray | None`; `get_state()['trim']['can_reset']: bool` (true only when a backup exists AND `can_trim` conditions hold).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trim_reset.py`:

```python
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from audio import WebLooper, LoopLayer, LooperState, SAMPLE_RATE


def _looper_with_master(seconds=2.0):
    """A playing looper with a single master layer of ramp audio."""
    looper = WebLooper()
    n = int(seconds * SAMPLE_RATE)
    buf = np.linspace(0, 1, n, dtype=np.float32)
    looper.layers = [LoopLayer(0, "Master", buf)]
    looper.master_length = n
    looper.state = LooperState.PLAYING
    return looper


def test_first_trim_creates_backup_of_original():
    looper = _looper_with_master(2.0)
    original = looper.layers[0].buffer.copy()
    assert looper.apply_trim(0.5, 1.5) is True
    assert looper._pre_trim_backup is not None
    assert np.array_equal(looper._pre_trim_backup, original)


def test_successive_trims_keep_first_backup():
    looper = _looper_with_master(2.0)
    original = looper.layers[0].buffer.copy()
    looper.apply_trim(0.5, 1.5)
    looper.apply_trim(0.25, 0.75)   # relative to the already-trimmed loop
    assert np.array_equal(looper._pre_trim_backup, original)


def test_reset_trim_restores_original_and_clears_backup():
    looper = _looper_with_master(2.0)
    original = looper.layers[0].buffer.copy()
    looper.apply_trim(0.5, 1.5)
    assert looper.reset_trim() is True
    assert looper.master_length == len(original)
    assert np.array_equal(looper.layers[0].buffer, original)
    assert looper.master_position == 0
    assert looper._pre_trim_backup is None


def test_reset_trim_without_backup_fails():
    looper = _looper_with_master(2.0)
    assert looper.reset_trim() is False


def test_reset_trim_with_overdubs_fails():
    looper = _looper_with_master(2.0)
    looper.apply_trim(0.5, 1.5)
    looper.layers.append(LoopLayer(1, "Overdub 1",
                                   np.zeros(looper.master_length, dtype=np.float32)))
    assert looper.reset_trim() is False


def test_overdub_commit_discards_backup():
    looper = _looper_with_master(2.0)
    looper.apply_trim(0.5, 1.5)
    looper.recording_buffer = np.zeros(looper.max_samples, dtype=np.float32)
    looper._finalize_overdub()
    assert looper._pre_trim_backup is None


def test_clear_all_discards_backup():
    looper = _looper_with_master(2.0)
    looper.apply_trim(0.5, 1.5)
    looper.clear_all()
    assert looper._pre_trim_backup is None


def test_new_recording_discards_backup():
    looper = _looper_with_master(2.0)
    looper.apply_trim(0.5, 1.5)
    looper.state = LooperState.IDLE
    looper.start_recording()
    assert looper._pre_trim_backup is None


def test_state_reports_can_reset():
    looper = _looper_with_master(2.0)
    assert looper.get_state()['trim']['can_reset'] is False
    looper.apply_trim(0.5, 1.5)
    assert looper.get_state()['trim']['can_reset'] is True
    looper.reset_trim()
    assert looper.get_state()['trim']['can_reset'] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trim_reset.py -v`
Expected: FAIL — `AttributeError: 'WebLooper' object has no attribute '_pre_trim_backup'` / `has no attribute 'reset_trim'` / `KeyError: 'can_reset'`.

- [ ] **Step 3: Implement**

In `WebLooper.__init__` (near the other state fields, after `self.layers`):

```python
self._pre_trim_backup = None  # master audio before first trim; volatile, never saved
```

In `apply_trim()`, right after `old_buffer = self.layers[0].buffer` (~line 1029):

```python
if self._pre_trim_backup is None:
    self._pre_trim_backup = old_buffer[:self.master_length].copy()
```

New method after `can_trim()` (~line 1047):

```python
def reset_trim(self) -> bool:
    """Restore master to its pre-trim original (undoes all trims)."""
    with self.lock:
        if self._pre_trim_backup is None:
            print("✗ Cannot reset trim: nothing to restore")
            return False
        if len(self.layers) != 1:
            print("✗ Cannot reset trim: overdubs exist")
            return False
        buf = self._pre_trim_backup
        self.layers[0] = LoopLayer(0, "Master", buf)
        self.master_length = len(buf)
        self.master_position = 0
        self._pre_trim_backup = None
        print(f"✓ Trim reset: restored original ({len(buf) / SAMPLE_RATE:.2f}s)")
        return True
```

Discard the backup at the three lifecycle points plus session load — add `self._pre_trim_backup = None` inside each (all already hold or are under the lock):
- `_finalize_overdub()` — after `self.layers.append(layer)`
- `start_recording()` — next to `self.recording_position = 0`
- `clear_all()` — next to `self.master_position = 0`
- `load_session()` — next to `self.master_position = 0` in the `with self.lock:` block

In `get_state()`: capture under the lock (next to `num_layers = len(self.layers)` ~line 1577):

```python
has_trim_backup = self._pre_trim_backup is not None
```

and in the returned `trim` block (~line 1646):

```python
'trim': {
    'can_trim': can_trim,
    'can_reset': can_trim and has_trim_backup,
    'reason': '' if can_trim else ('Add overdubs disabled trimming' if num_layers > 1 else ''),
},
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trim_reset.py -v` → all PASS.
Run: `python -m pytest tests/ -q` → whole suite green.

- [ ] **Step 5: Commit**

```bash
git add audio.py tests/test_trim_reset.py
git commit -m "feat(audio): keep pre-trim backup; reset_trim restores original take"
```

---

### Task 2: Wire `reset_trim` through socket and trim editor UI

**Files:**
- Modify: `routes.py` (command dispatch, after the `auto_trim_silence` branch ~line 227)
- Modify: `static/app.js` (default `serverState.trim` ~line 22; `resetTrim()` ~line 1380)

**Interfaces:**
- Consumes: `WebLooper.reset_trim()` and `trim.can_reset` from Task 1.
- Produces: socket command `reset_trim` (no payload).

- [ ] **Step 1: Add the route branch**

In `routes.py`, after the `auto_trim_silence` branch:

```python
elif command == 'reset_trim':
    looper.reset_trim()
```

- [ ] **Step 2: Update the frontend**

In `static/app.js` line ~22, extend the default trim state:

```javascript
trim: {
    can_trim: false,
    can_reset: false,
},
```

(match the existing object shape — only add `can_reset: false`).

Replace `resetTrim()` (~line 1380):

```javascript
function resetTrim() {
    if (serverState.trim?.can_reset) {
        sendCommand('reset_trim');
        // Server restores the original; re-fetch so handles + waveform follow
        setTimeout(() => socket.emit('get_waveform'), 100);
        return;
    }
    trimStart = 0;
    trimEnd = originalDuration;
    updateTrimUI();
}
```

(The `waveform` socket handler already resets `originalDuration`, `trimStart`, `trimEnd` from the fresh server state, same as after Apply.)

- [ ] **Step 3: Verify manually**

Run: `python -m pytest tests/ -q` → green (no behavior change expected in suite).
Live check on the Pi (or note for later): record loop → trim → ↺ Reset restores full take; before any trim, ↺ Reset still just re-spreads the handles.

- [ ] **Step 4: Commit**

```bash
git add routes.py static/app.js
git commit -m "feat(ui): trim Reset restores the original take via reset_trim"
```

---

### Task 3: Keyboard shortcuts must not fire while typing

**Files:**
- Modify: `static/app.js` (global `keydown` listener ~line 1155)

**Interfaces:** none.

- [ ] **Step 1: Add the guard**

At the top of the `document.addEventListener('keydown', (e) => {` handler, before the KeyT check:

```javascript
// Never steal keys from text entry (session name, trim inputs, ...)
const t = e.target;
if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' ||
          t.tagName === 'SELECT' || t.isContentEditable)) {
    return;
}
```

- [ ] **Step 2: Verify manually**

Open the app, focus the session-name input, type "tada d t " — text appears, no tap-tempo/detect/transport fires. Click outside the input; T/D/Space work again.

- [ ] **Step 3: Commit**

```bash
git add static/app.js
git commit -m "fix(ui): don't fire T/D/Space shortcuts while typing in inputs"
```

---

### Task 4: Delete button on non-master loops

**Files:**
- Modify: `static/app.js` (layer expanded panel ~line 1856–1866)
- Modify: `static/style.css` (one rule, near `.btn-mute` ~line 648)

**Interfaces:**
- Consumes: existing `deleteLayer(layerId)` (app.js ~line 713, confirms then sends `delete_layer`).

- [ ] **Step 1: Add the button**

In the layer expanded panel's volume row (app.js ~line 1856), after the MUTE button:

```javascript
${layer.id !== 0 ? `
    <button class="btn btn-small btn-delete-layer"
            onclick="deleteLayer(${layer.id})">✕</button>
` : ''}
```

(Template-literal context: this sits inside the existing backtick string built in `serverState.layers.map(layer => ...)`; interpolate as shown.)

- [ ] **Step 2: Style it**

In `style.css`, after the `.btn-mute` rules (~line 653):

```css
.btn-delete-layer { color: var(--text-muted); }
.btn-delete-layer:hover { color: var(--rec); border-color: var(--rec); }
```

- [ ] **Step 3: Verify manually**

Record a master + one overdub → expand the overdub row (edit mode) → ✕ shows, confirm dialog appears, layer deletes and sections remap. Master row shows no ✕.

- [ ] **Step 4: Commit**

```bash
git add static/app.js static/style.css
git commit -m "feat(ui): delete button on non-master loops"
```

---

### Task 5: Fretboard — 24 frets in two rows

**Files:**
- Modify: `static/app.js` (`renderFretboard()` ~line 254–325)

**Interfaces:** none (self-contained render function; called from existing sites).

- [ ] **Step 1: Rewrite `renderFretboard()`**

Replace the body of `renderFretboard()` with a two-row version. Keep the existing signature, guard, and constants; extract the per-row drawing into a local helper:

```javascript
function renderFretboard() {
    if (activeSidePanel !== 'scale') return;
    renderScaleInfo();
    const rootIdx = SCALE_NOTES.indexOf(scaleRoot);
    const intervals = new Set(SCALE_INTERVALS[scaleType] || []);
    const charSet = new Set((SCALE_INFO[scaleType] || {}).characteristic || []);

    const OPEN_STRINGS = guitarMode === '8string' ? OPEN_STRINGS_8 : OPEN_STRINGS_6;
    const STRINGS = OPEN_STRINGS.length;

    const W = 640;
    const padL = 42, padR = 12, padT = 18, padB = 22;
    const STRING_SPACING = 23; // px between strings (room for 2-line labels)
    const DOT_R = 10;
    const boardH = padT + padB + STRING_SPACING * (STRINGS - 1);
    const ROW_GAP = 8;
    const H = boardH * 2 + ROW_GAP;

    // One row of the board: frets (fretLo..fretHi], plus open strings on row A
    function drawRow(yTop, fretLo, fretHi, withOpen, singleMarkers, doubleMarker) {
        const nFrets = fretHi - fretLo;
        const fretW = (W - padL - padR) / nFrets;
        const openX = padL - fretW * 0.45; // leaves room for the gold ring on open-string notes
        const fretX = f => padL + (f - fretLo) * fretW;
        const noteX = f => f === 0 ? openX : padL + (f - fretLo - 0.5) * fretW;
        const stringY = s => yTop + padT + (STRINGS - 1 - s) * STRING_SPACING; // s=0 = lowest string = bottom
        let out = '';

        // String lines
        for (let s = 0; s < STRINGS; s++) {
            const y = stringY(s);
            const sw = 0.7 + s * 0.32;
            const x1 = withOpen ? openX - 4 : fretX(fretLo);
            out += `<line x1="${x1}" y1="${y}" x2="${fretX(fretHi)}" y2="${y}" stroke="#3a4557" stroke-width="${sw}"/>`;
        }

        // Fret lines (nut / row-anchor thicker)
        for (let f = fretLo; f <= fretHi; f++) {
            const x = fretX(f);
            const isAnchor = f === fretLo;
            out += `<line x1="${x}" y1="${yTop + padT}" x2="${x}" y2="${yTop + padT + (STRINGS-1)*STRING_SPACING}" stroke="${isAnchor ? '#6b7280' : '#1e2533'}" stroke-width="${isAnchor ? 3 : 1.5}"/>`;
        }

        // Row B: label its anchor fret ("12")
        if (!withOpen) {
            out += `<text x="${fretX(fretLo)}" y="${yTop + padT - 6}" text-anchor="middle" font-size="8" fill="#6b7280">${fretLo}</text>`;
        }

        // Position markers below strings
        const markerY = yTop + padT + (STRINGS - 1) * STRING_SPACING + 14;
        for (const mf of singleMarkers) {
            out += `<circle cx="${padL + (mf - fretLo - 0.5) * fretW}" cy="${markerY}" r="3.5" fill="#2d3748"/>`;
        }
        const xd = padL + (doubleMarker - fretLo - 0.5) * fretW;
        out += `<circle cx="${xd - 5}" cy="${markerY}" r="3" fill="#2d3748"/>`;
        out += `<circle cx="${xd + 5}" cy="${markerY}" r="3" fill="#2d3748"/>`;

        // Note dots — every scale tone labeled note-name + interval; defining notes gold
        const fStart = withOpen ? 0 : fretLo + 1;
        for (let s = 0; s < STRINGS; s++) {
            const y = stringY(s);
            for (let f = fStart; f <= fretHi; f++) {
                const noteIdx = (OPEN_STRINGS[s] + f) % 12;
                const interval = (noteIdx - rootIdx + 12) % 12;
                if (!intervals.has(interval)) continue;
                const isRoot = interval === 0;
                const isChar = charSet.has(interval);
                const x = noteX(f);
                const fill = isRoot ? '#ed8936' : (isChar ? '#f6c453' : '#4fd1c5');
                if (isChar) {
                    out += `<circle cx="${x}" cy="${y}" r="${DOT_R + 2.5}" fill="none" stroke="#f6c453" stroke-width="1.4" opacity="0.9"/>`;
                }
                out += `<circle cx="${x}" cy="${y}" r="${DOT_R}" fill="${fill}"${isChar ? ' filter="url(#goldGlow)"' : ''} opacity="0.95"/>`;
                const noteName = SCALE_NOTES[noteIdx].replace('#', '♯');
                out += `<text x="${x}" y="${y - 1.5}" text-anchor="middle" font-size="8" font-weight="bold" fill="#15202b">${noteName}</text>`;
                out += `<text x="${x}" y="${y + 7}" text-anchor="middle" font-size="6" fill="#15202b" opacity="0.82">${INTERVAL_LABELS[interval]}</text>`;
            }
        }
        return out;
    }

    let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" width="100%" style="display:block">`;
    svg += `<defs><filter id="goldGlow"><feGaussianBlur stdDeviation="2.6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
    svg += drawRow(0, 0, 12, true, [3, 5, 7, 9], 12);                 // open + frets 0–12
    svg += drawRow(boardH + ROW_GAP, 12, 24, false, [15, 17, 19, 21], 24); // frets 13–24
    svg += `</svg>`;
    document.getElementById('fretboard').innerHTML = svg;
}
```

Notes for the implementer:
- Row A is pixel-identical to today's board (same 12-fret geometry, nut at fret 0, double marker at 12).
- Row B reuses the same fret width; its left edge is the fret-12 line drawn thick with an "12" label above; dots start at fret 13 (fret-12 notes are not duplicated).
- `drawRow` closes over `rootIdx`, `intervals`, `charSet`, `OPEN_STRINGS`, `STRINGS`, and the layout constants.

- [ ] **Step 2: Verify manually**

Run: `node --check static/app.js` (syntax check only). Expected: no output.
Live: open the scale panel — two boards render; C major dots on both rows; 8-string mode shows both rows with 8 strings; markers at 3/5/7/9, 12·12, 15/17/19/21, 24·24.

- [ ] **Step 3: Commit**

```bash
git add static/app.js
git commit -m "feat(ui): fretboard shows all 24 frets in two rows"
```

---

## Final verification

- [ ] `python -m pytest tests/ -q` — full suite green.
- [ ] Live smoke test on the Pi: record → trim → Reset restores; type "t"/"d"/space in session name; delete an overdub; view 24-fret board.
- [ ] Mark plan tasks done; update spec status if desired.
