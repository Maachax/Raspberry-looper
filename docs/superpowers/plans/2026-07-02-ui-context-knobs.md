# Screen-Driven Knob Layers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In PLAY mode the MPK knobs follow the screen — FX panel → edit the shown effect, trim editor → move the handles (A key applies), otherwise volumes.

**Architecture:** Client computes a context (`home`/`fx`/`trim`) from its view state and sends `ui_context` on change; `MidiController` intercepts knob CCs in PLAY mode per context, reusing the idle-commit param editor (now parameterized by target) and adding a trim-ratio preview pushed through the `midi` state block.

**Tech Stack:** existing midi_control.py / app.js / routes.py, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-02-ui-context-knobs-design.md`.
- Knob CCs are `cc:0:70`–`cc:0:77`. In `fx`/`trim` contexts ALL eight are swallowed (no volume fall-through); explicit `section_edit`/`fx_edit` modes ignore context entirely.
- Trim: ratios clamped to `end - start ≥ 0.02`; apply = note:0:69 only in trim context; duration = `looper.master_length / SAMPLE_RATE` (import `SAMPLE_RATE` from config).
- `set_loop_chain` audio-safety rule unchanged (0.3 s idle-commit).
- Tests: `./bin/python -m pytest` — currently 112 passing.
- Commits end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

### Task 1: Controller — ui_context state, param-editor targeting, context dispatch, trim preview

**Files:**
- Modify: `midi_control.py`
- Modify: `audio.py` (default `midi` block gains `ui_context`/`trim_preview`)
- Modify: `routes.py` (command `ui_context`)
- Test: `tests/test_midi_control.py`

**Interfaces:**
- Produces:
  - `set_ui_context(data: dict)` — normalizes `context` (unknown → `home`), stores `ui_fx_loop`/`ui_fx_slot`, clears `trim_preview` on leaving `trim`, notifies.
  - `_edit_param(param_index, cc_value, loop_idx=None, slot=None)` — existing behavior when defaults used.
  - `_handle_context_trigger(trigger, value) -> bool` — True if consumed.
  - `status()` gains `ui_context` (str) and `trim_preview` (`{'start','end'}` or None).
  - Socket command `ui_context {context, fx_loop, fx_slot}`.

- [x] **Step 1: Failing tests** — append to `tests/test_midi_control.py`; add to `FakeLooper.__init__`: `self.master_length = 44100 * 4` and `self.trims = []`; add method `def apply_trim(self, s, e): self.trims.append((s, e))`.

```python
def test_ui_context_normalization(tmp_path):
    _, ctl, _ = make_controller(tmp_path)
    ctl.set_ui_context({'context': 'fx', 'fx_loop': 1, 'fx_slot': 0})
    assert ctl.status()['ui_context'] == 'fx'
    ctl.set_ui_context({'context': 'warp'})
    assert ctl.status()['ui_context'] == 'home'


def test_fx_context_knobs_edit_screen_target_not_volumes(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    looper.layers[1].fx_chain = [{'type': 'distortion',
                                  'params': {'drive_db': 18.0}, 'enabled': True}]
    ctl.set_ui_context({'context': 'fx', 'fx_loop': 1, 'fx_slot': 0})
    ctl.handle_trigger('cc:0:70', 127)      # K1 -> param, NOT loop 1 volume
    ctl.handle_trigger('cc:0:73', 64)       # K4 -> swallowed, NOT loop 4 volume
    assert looper.volumes == {}
    ctl.flush_params()
    assert looper.layers[1].fx_chain[0]['params']['drive_db'] == 40.0


def test_explicit_mode_overrides_context(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    looper.layers[2].fx_chain = [{'type': 'distortion',
                                  'params': {'drive_db': 18.0}, 'enabled': True}]
    ctl.set_ui_context({'context': 'fx', 'fx_loop': 1, 'fx_slot': 0})
    ctl.selected_loop = 2
    ctl.set_mode('fx_edit')                 # explicit mode wins over context
    ctl.handle_trigger('cc:0:70', 0)
    ctl.flush_params()
    assert looper.layers[2].fx_chain[0]['params']['drive_db'] == 0.0
    assert looper.layers[1].fx_chain == []


def test_trim_context_knobs_and_apply(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl.set_ui_context({'context': 'trim'})
    ctl.handle_trigger('cc:0:70', 0)        # start -> 0.0
    ctl.handle_trigger('cc:0:71', 64)       # end -> 64/127
    tp = ctl.status()['trim_preview']
    assert tp['start'] == 0.0 and abs(tp['end'] - 64 / 127) < 1e-9
    ctl.handle_trigger('cc:0:70', 127)      # start clamps below end
    assert ctl.trim_preview['start'] == ctl.trim_preview['end'] - 0.02
    ctl.handle_trigger('note:0:69', 64)     # A key applies
    s, e = looper.trims[-1]
    assert abs(e - (64 / 127) * 4.0) < 1e-6
    assert ctl.trim_preview is None


def test_leaving_trim_context_clears_preview(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl.set_ui_context({'context': 'trim'})
    ctl.handle_trigger('cc:0:70', 30)
    ctl.set_ui_context({'context': 'home'})
    assert ctl.trim_preview is None
    ctl.handle_trigger('note:0:69', 64)     # apply outside context: no-op
    assert looper.trims == []
    ctl.handle_trigger('cc:0:70', 127)      # knobs are volumes again
    assert looper.volumes[0] == 1.0
```

- [x] **Step 2: Verify fail** — `./bin/python -m pytest tests/test_midi_control.py -q` → new tests FAIL (`AttributeError: set_ui_context`).

- [x] **Step 3: Implement** in `midi_control.py`:

Top imports: `from config import CONFIG_PATH, SAMPLE_RATE`.

`__init__` additions:

```python
        self.ui_context = 'home'     # what the browser is showing
        self.ui_fx_loop = None
        self.ui_fx_slot = 0
        self.trim_preview = None     # {'start': r, 'end': r} ratios, or None
```

New methods:

```python
    # ------------------------------------------------------------ ui context
    def set_ui_context(self, data):
        ctx = data.get('context')
        if ctx not in ('home', 'fx', 'trim'):
            ctx = 'home'
        if ctx != 'trim':
            self.trim_preview = None
        self.ui_context = ctx
        self.ui_fx_loop = data.get('fx_loop')
        self.ui_fx_slot = data.get('fx_slot') or 0
        self.notify()

    _KNOB_CCS = {f'cc:0:{70 + i}' for i in range(8)}

    def _handle_context_trigger(self, trigger, value):
        """Screen-driven knob layer (PLAY mode only). True = consumed."""
        if trigger in self._KNOB_CCS:
            if value is None:
                return True
            k = int(trigger.rsplit(':', 1)[1]) - 70          # knob 0..7
            if self.ui_context == 'fx':
                if k <= 2 and self.ui_fx_loop is not None:
                    self._edit_param(k, value, loop_idx=self.ui_fx_loop,
                                     slot=self.ui_fx_slot)
                elif k == 6:
                    self._edit_bus('room_size', value)
                elif k == 7:
                    self._edit_bus('wet', value)
            elif self.ui_context == 'trim' and k in (0, 1):
                self._edit_trim(k, value)
            return True                     # knobs never fall through
        if self.ui_context == 'trim' and trigger == 'note:0:69':
            self._apply_trim_preview()
            return True
        return False

    def _edit_trim(self, knob, value):
        if self.trim_preview is None:
            self.trim_preview = {'start': 0.0, 'end': 1.0}
        r = value / 127.0
        if knob == 0:
            self.trim_preview['start'] = min(r, self.trim_preview['end'] - 0.02)
        else:
            self.trim_preview['end'] = max(r, self.trim_preview['start'] + 0.02)

    def _apply_trim_preview(self):
        if self.trim_preview is None or not getattr(self.looper, 'master_length', 0):
            return
        duration = self.looper.master_length / SAMPLE_RATE
        self.looper.apply_trim(self.trim_preview['start'] * duration,
                               self.trim_preview['end'] * duration)
        self.trim_preview = None
```

`handle_trigger`: after the learn check, before the binding lookups, insert:

```python
        if self.mode == 'play' and self.ui_context != 'home':
            if self._handle_context_trigger(trigger, value):
                self._notify_debounced()
                return
```

`_edit_param` becomes target-parameterized (drop the `_fx_chain()` call):

```python
    def _edit_param(self, param_index, cc_value, loop_idx=None, slot=None):
        """Buffer a param edit; commit after 0.3s idle (set_loop_chain re-bakes
        under looper.lock, so it must never run per CC tick)."""
        if slot is None:
            slot = self.selected_fx_slot
        if self._param_pending is None:
            if loop_idx is None:
                loop_idx = self.effective_loop()
            if loop_idx is None or not (0 <= loop_idx < len(self.looper.layers)):
                return
            chain = copy.deepcopy(self.looper.layers[loop_idx].fx_chain)
            if slot >= len(chain):
                return
            self._param_pending = (loop_idx, chain)
        idx, chain = self._param_pending
        if slot >= len(chain):
            return
        effect = chain[slot]
        schema = EFFECT_SCHEMAS.get(effect['type'], [])
        if param_index >= len(schema):
            return
        param = schema[param_index]
        effect['params'][param['name']] = scale_param(param, cc_value)
        if self._param_timer is not None:
            self._param_timer.cancel()
        self._param_timer = threading.Timer(0.3, self.flush_params)
        self._param_timer.daemon = True
        self._param_timer.start()
```

`status()` gains:

```python
            'ui_context': self.ui_context,
            'trim_preview': self.trim_preview,
```

`audio.py` no-controller default gains `'ui_context': 'home', 'trim_preview': None` (update `test_get_state_has_midi_block` accordingly).

`routes.py`, in the command handler:

```python
    elif command == 'ui_context':
        if midi is not None:
            midi.set_ui_context(data)
```

- [x] **Step 4:** `./bin/python -m pytest tests/ -q` → 117 passed.
- [x] **Step 5:** Commit `feat(midi): screen-driven knob layers — ui context, fx targeting, trim preview`.

---

### Task 2: Client — report context, trim handles follow preview, knob banner

**Files:**
- Modify: `static/app.js`, `static/style.css`

**Interfaces:**
- Consumes: `serverState.midi.ui_context`, `.trim_preview`; client vars `activeSidePanel`, `trimEditorExpanded`, `fxLoopId`, `fxActiveIdx`.

- [x] **Step 1: reportUiContext** — add near the mobile navigation block:

```javascript
        let _lastUiContext = '';
        function reportUiContext() {
            let ctx = { context: 'home' };
            if (document.body.classList.contains('edit-mode') && trimEditorExpanded) {
                ctx = { context: 'trim' };
            } else if (activeSidePanel === 'fx') {
                ctx = { context: 'fx', fx_loop: fxLoopId, fx_slot: fxActiveIdx };
            }
            const j = JSON.stringify(ctx);
            if (j === _lastUiContext) return;
            _lastUiContext = j;
            sendCommand('ui_context', ctx);
        }
```

Call `reportUiContext();` at the end of: `setSidePanel` (both branches, after the class sync lines), `toggleViewMode`, `toggleTrimEditor`, and `renderFx` (covers chip clicks and loop-select changes; the change-guard makes repeat calls free).

- [x] **Step 2: Trim handles follow the preview** — in the update handler after `renderMidiBanner();`, add `applyTrimPreviewFromMidi();`, implemented next to the trim editor code (find the client's trim variables — `trimStart`/`trimEnd` in seconds and its redraw function — and set them from the ratios):

```javascript
        function applyTrimPreviewFromMidi() {
            const tp = serverState.midi && serverState.midi.trim_preview;
            if (!tp || !trimEditorExpanded) return;
            const dur = originalDuration || serverState.master_duration || 0;
            if (!dur) return;
            trimStart = tp.start * dur;
            trimEnd = tp.end * dur;
            updateTrimUI();
        }
```

(If the redraw function is named differently — check `toggleTrimEditor`'s body — use that name; it must reposition handles/overlays from `trimStart`/`trimEnd`.)

- [x] **Step 3: Knob banner** — in `renderMidiBanner`, the `play` branch currently hides the banner; replace that branch:

```javascript
            if (midi.mode === 'play') {
                if (midi.ui_context && midi.ui_context !== 'home') {
                    el.style.display = '';
                    el.className = 'midi-mode-banner banner-knobs';
                    el.textContent = midi.ui_context === 'trim'
                        ? 'KNOBS → TRIM  (A key applies)'
                        : 'KNOBS → FX';
                } else {
                    el.style.display = 'none';
                }
            } else {
```

CSS: `.midi-mode-banner.banner-knobs { background: #4fd1c5; color: #1a1a1a; }`

- [x] **Step 4:** `node --check static/app.js` → OK; suite still 117; commit `feat(ui): report ui context; trim handles + banner follow MPK knobs`.

---

### Task 3: Live verification (Max)

- [ ] Restart service; hard-refresh phone.
- [ ] HOME: knobs = volumes. FX tab: K1–K3 shape the shown effect, K7/K8 bus; tap another chip → knobs retarget. Open trim: K1/K2 move handles live on screen, A key applies. Close trim → volumes again. B6 FX EDIT mode still overrides regardless of screen.
