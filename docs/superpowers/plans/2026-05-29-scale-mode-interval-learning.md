# Scale & Mode Interval-Learning Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Looper Scale side-panel so the fretboard teaches the intervallic identity of each scale/mode — every note labeled with its interval, the defining note(s) spotlighted in gold, plus an interval-formula strip and a plain-English "what defines this scale" explanation.

**Architecture:** Pure client-side change. Two new data tables (`INTERVAL_LABELS`, `SCALE_INFO`) drive a rewritten `renderFretboard()` (note+interval labels, gold characteristic notes) and a new `renderScaleInfo()` (formula strip + explanation box). New DOM lives inside the existing `#panelScale`; new CSS styles it. No Python/server changes — `set_scale` and `DETECT SCALE` are untouched.

**Tech Stack:** Vanilla JS (inline SVG), HTML, CSS. Verification via Node (data consistency) + manual browser spot-checks (no JS test framework in this project).

---

## File Structure

- `static/app.js` — add `INTERVAL_LABELS` + `SCALE_INFO` tables near the existing `SCALE_INTERVALS`/`SCALE_LABELS` (~line 46–77); rewrite `renderFretboard()` (~line 198–261); add `renderScaleInfo()`.
- `templates/index.html` — add formula-strip, legend, and explanation DOM inside `#panelScale` (~line 88–126).
- `static/style.css` — add styles for `.interval-formula`, `.pill`, `.fretboard-legend`, `.scale-explain`; ensure the panel scrolls if content overflows.

The data tables and the two render functions are the units: tables = content/source-of-truth, `renderFretboard` = the neck diagram, `renderScaleInfo` = the textual teaching. `renderFretboard` calls `renderScaleInfo` so there is one entry point and all existing call sites keep working unchanged.

---

## Task 1: Add interval-label and scale-info data tables

**Files:**
- Modify: `static/app.js` (insert after `SCALE_LABELS`, ~line 77)
- Test: `/tmp/check_scale_data.js` (throwaway Node consistency guard)

- [ ] **Step 1: Write the consistency check (fails first)**

Create `/tmp/check_scale_data.js`. It embeds the *intended* data and asserts it is internally consistent. Run it before editing `app.js` to confirm the assertions are real (they will pass against the embedded copy — its job is to lock the content contract you then paste into `app.js`).

```js
// Mirrors the tables being added to static/app.js — content-correctness guard.
const SCALE_INTERVALS = {
  minor:[0,2,3,5,7,8,10], major:[0,2,4,5,7,9,11], dorian:[0,2,3,5,7,9,10],
  phrygian:[0,1,3,5,7,8,10], lydian:[0,2,4,6,7,9,11], mixolydian:[0,2,4,5,7,9,10],
  locrian:[0,1,3,5,6,8,10], harmonic_minor:[0,2,3,5,7,8,11], melodic_minor:[0,2,3,5,7,9,11],
  pent_major:[0,2,4,7,9], pent_minor:[0,3,5,7,10], blues:[0,3,5,6,7,10],
  diminished:[0,2,3,5,6,8,9,11], whole_tone:[0,2,4,6,8,10],
};
const INTERVAL_LABELS = {0:'R',1:'♭2',2:'2',3:'♭3',4:'3',5:'4',6:'♭5',7:'5',8:'♭6',9:'6',10:'♭7',11:'7'};
const SCALE_INFO = {
  major:{characteristic:[4,11]}, minor:{characteristic:[3]}, dorian:{characteristic:[9]},
  phrygian:{characteristic:[1]}, lydian:{characteristic:[6]}, mixolydian:{characteristic:[10]},
  locrian:{characteristic:[1,6]}, harmonic_minor:{characteristic:[11]}, melodic_minor:{characteristic:[9,11]},
  pent_major:{characteristic:[3]}, pent_minor:{characteristic:[3]}, blues:{characteristic:[6]},
  diminished:{characteristic:[6]}, whole_tone:{characteristic:[6]},
};
let ok = true;
for (let i=0;i<12;i++) if (!(i in INTERVAL_LABELS)) { console.error('missing label',i); ok=false; }
for (const k of Object.keys(SCALE_INTERVALS)) {
  if (!SCALE_INFO[k]) { console.error('SCALE_INFO missing',k); ok=false; continue; }
  for (const c of SCALE_INFO[k].characteristic)
    if (!SCALE_INTERVALS[k].includes(c)) { console.error('char not in scale',k,c); ok=false; }
}
console.log(ok ? 'OK: scale data consistent' : 'FAIL');
process.exit(ok?0:1);
```

- [ ] **Step 2: Run it to verify it passes against the embedded copy**

Run: `node /tmp/check_scale_data.js`
Expected: `OK: scale data consistent`

- [ ] **Step 3: Add the tables to `static/app.js`**

Insert immediately after the `SCALE_LABELS` object (after its closing `};`, ~line 77):

```js
        const INTERVAL_LABELS = {
            0: 'R', 1: '♭2', 2: '2', 3: '♭3', 4: '3', 5: '4',
            6: '♭5', 7: '5', 8: '♭6', 9: '6', 10: '♭7', 11: '7',
        };
        // characteristic = interval(s) (semitones from root) that define the scale's flavor.
        const SCALE_INFO = {
            'major':          { characteristic: [4, 11], title: 'WHAT DEFINES MAJOR',
                text: "The bright, resolved sound. The major 3rd (3) makes it happy and the major 7th (7) leans strongly back to the root." },
            'minor':          { characteristic: [3], title: 'WHAT DEFINES NATURAL MINOR',
                text: "The minor 3rd (♭3) is the dark, sad core of every minor sound; the ♭6 and ♭7 deepen the melancholy." },
            'dorian':         { characteristic: [9], title: 'WHAT DEFINES DORIAN',
                text: "A minor scale with a raised 6th. That natural 6 is the note your ear latches onto — it gives Dorian its bright, hopeful, “Santana” color instead of plain-sad minor." },
            'phrygian':       { characteristic: [1], title: 'WHAT DEFINES PHRYGIAN',
                text: "A minor scale with a flat 2nd. That ♭2 sitting right above the root is the tension — dark, Spanish, a metal favorite." },
            'lydian':         { characteristic: [6], title: 'WHAT DEFINES LYDIAN',
                text: "A major scale with a raised 4th (♯4, shown here as ♭5). That floating ♯4 gives Lydian its dreamy, weightless, film-score lift." },
            'mixolydian':     { characteristic: [10], title: 'WHAT DEFINES MIXOLYDIAN',
                text: "A major scale with a flat 7th. The ♭7 keeps it bright but bluesy — the classic dominant, rock and funk sound." },
            'locrian':        { characteristic: [1, 6], title: 'WHAT DEFINES LOCRIAN',
                text: "Flat 2nd AND flat 5th. With its 5th lowered there is no stable home — tense, and rarely used as a key center." },
            'harmonic_minor': { characteristic: [11], title: 'WHAT DEFINES HARMONIC MINOR',
                text: "A minor scale with a raised 7th. The big jump from ♭6 up to the natural 7 gives that exotic, classical/neoclassical color." },
            'melodic_minor':  { characteristic: [9, 11], title: 'WHAT DEFINES MELODIC MINOR',
                text: "A minor scale with a major top — natural 6 and natural 7. Minor on the bottom, major up top: the “jazz minor” sound." },
            'pent_major':     { characteristic: [4], title: 'WHAT DEFINES MAJOR PENTATONIC',
                text: "The major scale with its two tension notes (the 4th and 7th) removed. No half-steps means nothing clashes — it sits safely over almost anything." },
            'pent_minor':     { characteristic: [3], title: 'WHAT DEFINES MINOR PENTATONIC',
                text: "The minor 3rd plus no 2nd or 6th — five notes, no clashes. The universal rock and blues “box”." },
            'blues':          { characteristic: [6], title: 'WHAT DEFINES THE BLUES SCALE',
                text: "Minor pentatonic plus one extra note: the ♭5 “blue note”. That flat-five passing tone is the entire sound." },
            'diminished':     { characteristic: [6], title: 'WHAT DEFINES DIMINISHED',
                text: "A symmetric whole-step / half-step pattern that repeats every minor 3rd. Tense and unstable — used over diminished and dominant chords." },
            'whole_tone':     { characteristic: [6], title: 'WHAT DEFINES WHOLE TONE',
                text: "Every step is a whole tone — no half-steps at all. With no leading tone there is no pull home, giving a dreamy, augmented, floating sound." },
        };
```

- [ ] **Step 4: Re-run the consistency guard**

Run: `node /tmp/check_scale_data.js`
Expected: `OK: scale data consistent` (the `characteristic` arrays above match the embedded copy in the check).

- [ ] **Step 5: Commit**

```bash
git add -f static/app.js
git commit -m "feat(scale): add interval-label and scale-info data tables

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Add formula strip, legend, and explanation DOM

**Files:**
- Modify: `templates/index.html` (inside `#panelScale`, ~line 119–122)

- [ ] **Step 1: Add the formula strip before the fretboard**

Find (around line 119):

```html
                <div class="scale-root-row" id="scaleRootRow"></div>
                <div class="fretboard-container">
                    <div id="fretboard"></div>
                </div>
```

Replace with:

```html
                <div class="scale-root-row" id="scaleRootRow"></div>
                <div class="interval-formula-label">INTERVAL FORMULA</div>
                <div class="interval-formula" id="intervalFormula"></div>
                <div class="fretboard-container">
                    <div id="fretboard"></div>
                </div>
                <div class="fretboard-legend">
                    <span><i class="leg-dot leg-root"></i>Root</span>
                    <span><i class="leg-dot leg-tone"></i>Scale tone</span>
                    <span><i class="leg-dot leg-char"></i>Defining note</span>
                </div>
                <div class="scale-explain">
                    <div class="scale-explain-title" id="scaleExplainTitle"></div>
                    <div class="scale-explain-text" id="scaleExplainText"></div>
                </div>
```

- [ ] **Step 2: Verify the file is well-formed**

Run: `python3 -c "from html.parser import HTMLParser; HTMLParser().feed(open('templates/index.html').read()); print('parsed OK')"`
Expected: `parsed OK`

- [ ] **Step 3: Commit**

```bash
git add -f templates/index.html
git commit -m "feat(scale): add formula-strip, legend, explanation DOM to scale panel

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Add CSS for the new elements

**Files:**
- Modify: `static/style.css` (append near the existing scale-panel styles; end of file is fine)

- [ ] **Step 1: Append the styles**

Add to `static/style.css`:

```css
/* ---- Scale learning: formula strip, legend, explanation ---- */
.interval-formula-label {
    font-size: 11px; letter-spacing: .06em; color: #6b7a90; margin: 6px 0 4px;
}
.interval-formula {
    display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 10px;
}
.interval-formula .pill {
    background: #202a3a; border: 1px solid #34425a; color: #aeb9c9;
    border-radius: 6px; padding: 5px 0; min-width: 30px; text-align: center;
    font-family: monospace; font-size: 14px; font-weight: 600;
}
.interval-formula .pill-root { color: #ed8936; border-color: #ed8936; }
.interval-formula .pill-char {
    color: #f6c453; border-color: #f6c453;
    background: rgba(246,196,83,.14); box-shadow: 0 0 8px rgba(246,196,83,.30);
}
.fretboard-legend {
    display: flex; gap: 14px; margin: 8px 2px 0; font-size: 11px; color: #8595ab;
}
.fretboard-legend i.leg-dot {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    vertical-align: middle; margin-right: 4px;
}
.leg-root { background: #ed8936; }
.leg-tone { background: #4fd1c5; }
.leg-char { background: #f6c453; box-shadow: 0 0 6px #f6c453; }
.scale-explain {
    margin-top: 12px; background: #1b2433; border-left: 3px solid #f6c453;
    border-radius: 0 6px 6px 0; padding: 10px 12px;
}
.scale-explain-title {
    color: #f6c453; font-size: 12px; font-weight: 700; letter-spacing: .04em;
    margin-bottom: 3px;
}
.scale-explain-text { color: #c4cfde; font-size: 13px; line-height: 1.5; }
```

- [ ] **Step 2: Ensure the scale panel scrolls if content overflows**

The taller (8-string) fretboard plus the new boxes can exceed the viewport. Confirm the panel body already scrolls: search for the rule that sets `overflow` on `.side-panel`.

Run: `grep -n "side-panel" static/style.css | head`

If no `.side-panel` rule sets `overflow-y`, append this fallback:

```css
.side-panel.active { overflow-y: auto; }
```

(If an existing rule already gives the panel/`#sideContent` `overflow-y: auto`, skip this — do not add a duplicate.)

- [ ] **Step 3: Commit**

```bash
git add -f static/style.css
git commit -m "style(scale): styles for interval formula, legend, explanation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Add `renderScaleInfo()` and wire it into `renderFretboard()`

**Files:**
- Modify: `static/app.js` (add `renderScaleInfo`; edit top of `renderFretboard`, ~line 198)

- [ ] **Step 1: Add `renderScaleInfo()` immediately before `renderFretboard()`**

Insert this function just above the line `function renderFretboard() {` (~line 198):

```js
        function renderScaleInfo() {
            const info = SCALE_INFO[scaleType] || { characteristic: [], title: '', text: '' };
            const charSet = new Set(info.characteristic || []);
            const intervals = SCALE_INTERVALS[scaleType] || [];

            const formulaEl = document.getElementById('intervalFormula');
            if (formulaEl) {
                formulaEl.innerHTML = intervals.map(iv => {
                    const cls = iv === 0 ? 'pill pill-root'
                              : charSet.has(iv) ? 'pill pill-char'
                              : 'pill';
                    return `<span class="${cls}">${INTERVAL_LABELS[iv]}</span>`;
                }).join('');
            }

            const titleEl = document.getElementById('scaleExplainTitle');
            const textEl = document.getElementById('scaleExplainText');
            if (titleEl) titleEl.textContent = info.title || '';
            if (textEl) textEl.textContent = info.text || '';
        }
```

- [ ] **Step 2: Call it from `renderFretboard()`**

Find the start of `renderFretboard` (~line 198):

```js
        function renderFretboard() {
            if (activeSidePanel !== 'scale') return;
            const rootIdx = SCALE_NOTES.indexOf(scaleRoot);
```

Replace with (adds the `renderScaleInfo()` call after the guard):

```js
        function renderFretboard() {
            if (activeSidePanel !== 'scale') return;
            renderScaleInfo();
            const rootIdx = SCALE_NOTES.indexOf(scaleRoot);
```

- [ ] **Step 3: Syntax-check the JS**

Run: `node --check static/app.js`
Expected: no output, exit 0 (syntax OK).

- [ ] **Step 4: Commit**

```bash
git add -f static/app.js
git commit -m "feat(scale): render interval formula and explanation on scale change

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Rewrite `renderFretboard()` to label every note and spotlight defining notes

**Files:**
- Modify: `static/app.js` (`renderFretboard` body, ~line 198–261)

- [ ] **Step 1: Replace the whole `renderFretboard` function**

Replace the entire current `renderFretboard` (from `function renderFretboard() {` through its closing `}` that ends with `document.getElementById('fretboard').innerHTML = svg;`) with:

```js
        function renderFretboard() {
            if (activeSidePanel !== 'scale') return;
            renderScaleInfo();
            const rootIdx = SCALE_NOTES.indexOf(scaleRoot);
            const intervals = new Set(SCALE_INTERVALS[scaleType] || []);
            const charSet = new Set((SCALE_INFO[scaleType] || {}).characteristic || []);

            const OPEN_STRINGS = guitarMode === '8string' ? OPEN_STRINGS_8 : OPEN_STRINGS_6;
            const STRINGS = OPEN_STRINGS.length;

            const W = 640;
            const padL = 34, padR = 12, padT = 18, padB = 22;
            const FRETS = 12;
            const STRING_SPACING = 23; // px between strings (room for 2-line labels)
            const H = padT + padB + STRING_SPACING * (STRINGS - 1);
            const fretW = (W - padL - padR) / FRETS;
            const DOT_R = 10;
            const openX = padL - fretW * 0.55;

            const fretX = f => padL + f * fretW;
            const noteX = f => f === 0 ? openX : padL + (f - 0.5) * fretW;
            const stringY = s => padT + (STRINGS - 1 - s) * STRING_SPACING; // s=0 = lowest string = bottom

            let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" width="100%" style="display:block">`;
            svg += `<defs><filter id="goldGlow"><feGaussianBlur stdDeviation="2.6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;

            // String lines
            for (let s = 0; s < STRINGS; s++) {
                const y = stringY(s);
                const sw = 0.7 + s * 0.32;
                svg += `<line x1="${openX - 4}" y1="${y}" x2="${fretX(FRETS)}" y2="${y}" stroke="#3a4557" stroke-width="${sw}"/>`;
            }

            // Fret lines (nut thicker)
            for (let f = 0; f <= FRETS; f++) {
                const x = fretX(f);
                svg += `<line x1="${x}" y1="${padT}" x2="${x}" y2="${padT + (STRINGS-1)*STRING_SPACING}" stroke="${f === 0 ? '#6b7280' : '#1e2533'}" stroke-width="${f === 0 ? 3 : 1.5}"/>`;
            }

            // Position markers below strings
            const markerY = padT + (STRINGS - 1) * STRING_SPACING + 14;
            for (const mf of [3, 5, 7, 9]) {
                svg += `<circle cx="${padL + (mf - 0.5) * fretW}" cy="${markerY}" r="3.5" fill="#2d3748"/>`;
            }
            const x12 = padL + 11.5 * fretW;
            svg += `<circle cx="${x12 - 5}" cy="${markerY}" r="3" fill="#2d3748"/>`;
            svg += `<circle cx="${x12 + 5}" cy="${markerY}" r="3" fill="#2d3748"/>`;

            // Note dots — every scale tone labeled note-name + interval; defining notes gold
            for (let s = 0; s < STRINGS; s++) {
                const y = stringY(s);
                for (let f = 0; f <= FRETS; f++) {
                    const noteIdx = (OPEN_STRINGS[s] + f) % 12;
                    const interval = (noteIdx - rootIdx + 12) % 12;
                    if (!intervals.has(interval)) continue;
                    const isRoot = interval === 0;
                    const isChar = charSet.has(interval);
                    const x = noteX(f);
                    const fill = isRoot ? '#ed8936' : (isChar ? '#f6c453' : '#4fd1c5');
                    if (isChar) {
                        svg += `<circle cx="${x}" cy="${y}" r="${DOT_R + 2.5}" fill="none" stroke="#f6c453" stroke-width="1.4" opacity="0.9"/>`;
                    }
                    svg += `<circle cx="${x}" cy="${y}" r="${DOT_R}" fill="${fill}"${isChar ? ' filter="url(#goldGlow)"' : ''} opacity="0.95"/>`;
                    const noteName = SCALE_NOTES[noteIdx].replace('#', '♯');
                    svg += `<text x="${x}" y="${y - 1.5}" text-anchor="middle" font-size="8" font-weight="bold" fill="#15202b">${noteName}</text>`;
                    svg += `<text x="${x}" y="${y + 7}" text-anchor="middle" font-size="6" fill="#15202b" opacity="0.82">${INTERVAL_LABELS[interval]}</text>`;
                }
            }

            svg += `</svg>`;
            document.getElementById('fretboard').innerHTML = svg;
        }
```

- [ ] **Step 2: Syntax-check the JS**

Run: `node --check static/app.js`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add -f static/app.js
git commit -m "feat(scale): label every fretboard note with interval + spotlight defining notes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Manual verification across scales and tunings

**Files:** none (verification only)

This project has no automated UI tests; this task is the real correctness gate. Use the project's run path to launch the web UI (see README / `main.py`).

- [ ] **Step 1: Launch the app and open the Scale panel**

Start the looper web server (per project README, e.g. `python3 main.py` or the documented run command) and open the UI in a browser. Click the Scale side-panel icon (♩).

- [ ] **Step 2: Verify interval labels and defining-note spotlight**

With root = A, step through scales and confirm against the table below. Each scale-tone dot shows note name (top) + interval (bottom); the listed defining note(s) are gold with a ring/glow and the matching formula pill(s) are gold.

| Scale | Defining note shown gold | Sample dot to check |
|-------|--------------------------|---------------------|
| Dorian | the `6` (F♯) | F♯ labeled `6`, gold |
| Natural Minor | the `♭3` (C) | C labeled `♭3` |
| Mixolydian | the `♭7` (G) | G labeled `♭7`, gold |
| Lydian | the `♭5`/♯4 (D♯) | D♯ labeled `♭5`, gold |
| Blues | the `♭5` (D♯/E♭) | the blue note labeled `♭5`, gold |
| Locrian | `♭2` (B♭) and `♭5` (E♭) | both gold |

Expected: labels and gold highlights match; root dot is orange and labeled `R`.

- [ ] **Step 3: Verify the formula strip and explanation update together**

Change scale type and root via the panel controls. Expected: the INTERVAL FORMULA pills, the fretboard, and the WHAT DEFINES… box all update in sync; the explanation text matches the selected scale.

- [ ] **Step 4: Verify both tunings render cleanly**

Toggle 6 ↔ 8 string. Expected: dots/labels are legible and non-overlapping in both; if the 8-string panel is taller than the viewport, the panel scrolls.

- [ ] **Step 5: Verify DETECT SCALE still works**

Record a loop, run DETECT SCALE, apply a candidate. Expected: applying a candidate updates fretboard + formula + explanation together (no regression).

- [ ] **Step 6: Final commit (if any tweaks were needed)**

If Steps 1–5 required small fixes, commit them:

```bash
git add -f static/app.js static/style.css templates/index.html
git commit -m "fix(scale): adjustments from manual verification

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

If no fixes were needed, there is nothing to commit — note that verification passed.

---

## Notes for the implementer

- **`git add -f` is required** for every commit: this repo's `.gitignore` starts with `*`, so all tracked files are force-added by convention. Match it.
- **Unicode glyphs** (`♭`, `♯`, `♩`, curly quotes) are intentional — keep them as UTF-8, do not escape to entities in JS strings.
- **No server/Python changes.** If you find yourself editing `audio.py`, `routes.py`, or `looper_web.py`, stop — that is out of scope.
- The throwaway `/tmp/check_scale_data.js` is a content guard, not a project test; do not commit it.
