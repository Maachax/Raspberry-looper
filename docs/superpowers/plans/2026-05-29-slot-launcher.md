# Slot Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the looper's "scenes" feature with an Ableton-Session-style **slot launcher**: a side-panel tab of slots, each holding a drag-and-droppable set of loops with its own launch button that swaps the playing loops at the next loop start.

**Architecture:** Add a `slots` list to `WebLooper` (each slot = `{'id', 'loop_ids'}`), reuse the existing quantized `pending_*`-at-loop-restart mechanism for launching, expose slots over the existing socket/`get_state` channel, and build a new `#panelSlots` side panel with HTML5 drag-and-drop. Build additively first (slots alongside scenes so the app keeps running), then remove scenes + collapse-on-silence in the final task.

**Tech Stack:** Python (`audio.py` real-time looper, Flask-SocketIO in `routes.py`), vanilla JS (`static/app.js`), HTML (`templates/index.html`), CSS (`static/style.css`). Tests via `pytest` (see `tests/test_scale_detection.py` for the existing style — construct `WebLooper()` directly, no audio device needed).

**Key existing facts the engineer must know:**
- `LoopLayer` fields: `id`, `name`, `color`, `volume`, `is_playing`, `buffer`, `length`. Serialized by `to_dict()` with keys `id, name, duration, volume, is_playing, color`.
- **Layer ids are positional and get renumbered on delete.** `delete_layer` (`audio.py:499`) does `del self.layers[layer_id]` then re-assigns `layer.id = i` for all remaining layers. Any stored `loop_ids` must be remapped on delete (drop the deleted id; decrement ids above it).
- The audio callback applies a queued change on loop wrap (`audio.py:310-312`): `if loop_restarted and self.pending_scene is not None ...`.
- `_apply_scene` (`audio.py:570`) does NOT take the lock — it's called from inside code that already holds `self.lock`. New `_apply_slot` must follow the same rule.
- Frontend sends commands via `sendCommand(command, data)` → `socket.emit('command', {command, ...data})` (`app.js:594`). Server dispatches in `routes.py:184 handle_command` and broadcasts `looper.get_state()`.
- Side panels: a `.side-icon` button calls `setSidePanel(name)` (`app.js:1674`), which toggles `#sideBtn<Name>` and `#panel<Name>` `.active` classes. Panels live in `#sideContent` (`index.html:86`).

---

## Task 1: Slot data model + methods (`audio.py`)

Add slot state and methods **alongside** the existing scenes (scenes are removed in Task 7). `_apply_slot` must not acquire the lock.

**Files:**
- Modify: `audio.py` — `__init__` (after the Scenes block, ~line 176) and a new methods block (after `clear_all`, ~line 541)
- Test: `tests/test_slots.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_slots.py`:

```python
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from audio import WebLooper, LoopLayer, LooperState, SAMPLE_RATE


def _looper_with_layers(n):
    """A looper with n one-second layers (ids 0..n-1), playing."""
    looper = WebLooper()
    looper.layers = [LoopLayer(i, f"L{i}", np.zeros(SAMPLE_RATE, dtype=np.float32))
                     for i in range(n)]
    looper.master_length = SAMPLE_RATE
    looper.state = LooperState.PLAYING
    return looper


def test_add_slot_appends_empty_slot_with_unique_id():
    looper = WebLooper()
    s1 = looper.add_slot()
    s2 = looper.add_slot()
    assert s1['loop_ids'] == []
    assert s1['id'] != s2['id']
    assert looper.slots == [s1, s2]


def test_set_slot_loops_keeps_only_existing_layer_ids():
    looper = _looper_with_layers(3)          # valid ids: 0,1,2
    slot = looper.add_slot()
    looper.set_slot_loops(slot['id'], [0, 2, 99])
    assert slot['loop_ids'] == [0, 2]        # 99 dropped


def test_delete_slot_removes_it_and_clears_active():
    looper = _looper_with_layers(2)
    slot = looper.add_slot()
    looper._apply_slot(slot)                 # marks it active
    assert looper.active_slot_id == slot['id']
    assert looper.delete_slot(slot['id']) is True
    assert looper.slots == []
    assert looper.active_slot_id is None


def test_apply_slot_turns_on_only_member_loops():
    looper = _looper_with_layers(3)
    slot = looper.add_slot()
    looper.set_slot_loops(slot['id'], [0, 2])
    looper._apply_slot(slot)
    assert [l.is_playing for l in looper.layers] == [True, False, True]
    assert looper.active_slot_id == slot['id']


def test_apply_empty_slot_mutes_everything():
    looper = _looper_with_layers(2)
    slot = looper.add_slot()
    looper._apply_slot(slot)
    assert [l.is_playing for l in looper.layers] == [False, False]


def test_launch_slot_while_playing_is_quantized_not_immediate():
    looper = _looper_with_layers(2)          # state=PLAYING, master_length>0
    slot = looper.add_slot()
    looper.set_slot_loops(slot['id'], [1])
    looper.launch_slot(slot['id'], quantized=True)
    assert looper.pending_slot is slot       # queued, not applied yet
    assert looper.layers[0].is_playing is True   # unchanged until loop wrap


def test_launch_slot_applies_immediately_when_not_playing():
    looper = _looper_with_layers(2)
    looper.state = LooperState.IDLE
    slot = looper.add_slot()
    looper.set_slot_loops(slot['id'], [1])
    looper.launch_slot(slot['id'], quantized=True)
    assert looper.pending_slot is None
    assert [l.is_playing for l in looper.layers] == [False, True]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_slots.py -v`
Expected: FAIL — `AttributeError: 'WebLooper' object has no attribute 'add_slot'` (and `slots`).

- [ ] **Step 3: Add slot state in `__init__`**

In `audio.py`, immediately after the Scenes block (after line 176, the `self.pending_scene = None` line), add:

```python
        # Slots (launchable loop sets — replaces scenes)
        self.slots = []            # list of {'id': int, 'loop_ids': [int, ...]}
        self._next_slot_id = 1
        self.pending_slot = None   # slot to apply at next loop restart
        self.active_slot_id = None # id of the slot whose set is currently playing
```

- [ ] **Step 4: Add slot methods**

In `audio.py`, after `clear_all` (after line 541), add:

```python
    # -------------------------------------------------------------------------
    # SLOT LAUNCHER
    # -------------------------------------------------------------------------

    def add_slot(self) -> dict:
        """Append a new empty slot and return it."""
        with self.lock:
            slot = {'id': self._next_slot_id, 'loop_ids': []}
            self._next_slot_id += 1
            self.slots.append(slot)
            return slot

    def delete_slot(self, slot_id: int) -> bool:
        """Delete a slot. Clears pending/active references to it."""
        with self.lock:
            before = len(self.slots)
            self.slots = [s for s in self.slots if s['id'] != slot_id]
            if self.pending_slot and self.pending_slot['id'] == slot_id:
                self.pending_slot = None
            if self.active_slot_id == slot_id:
                self.active_slot_id = None
            return len(self.slots) < before

    def set_slot_loops(self, slot_id: int, loop_ids: list) -> bool:
        """Replace a slot's loop set, keeping only ids of layers that exist."""
        with self.lock:
            slot = next((s for s in self.slots if s['id'] == slot_id), None)
            if slot is None:
                return False
            valid = {layer.id for layer in self.layers}
            slot['loop_ids'] = [int(i) for i in loop_ids if int(i) in valid]
            return True

    def _apply_slot(self, slot: dict):
        """Turn on only the slot's member loops. Must be called holding self.lock."""
        ids = set(slot['loop_ids'])
        for layer in self.layers:
            layer.is_playing = layer.id in ids
        self.active_slot_id = slot['id']

    def launch_slot(self, slot_id: int, quantized: bool = True) -> bool:
        """Launch a slot: queue for next loop restart if playing, else apply now."""
        with self.lock:
            slot = next((s for s in self.slots if s['id'] == slot_id), None)
            if slot is None:
                return False
            if quantized and self.state == LooperState.PLAYING and self.master_length > 0:
                self.pending_slot = slot
            else:
                self._apply_slot(slot)
            return True
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_slots.py -v`
Expected: PASS (7 passed).

- [ ] **Step 6: Commit**

```bash
git add -f audio.py tests/test_slots.py
git commit -m "feat(slots): slot data model and launch methods"
```

---

## Task 2: Quantized launch in the audio callback + prune slots on layer delete/clear (`audio.py`)

Apply `pending_slot` on loop wrap, and keep `loop_ids` correct when layers are deleted (ids renumber!) or cleared.

**Files:**
- Modify: `audio.py` — callback (~line 312), `delete_layer` (~line 508), `clear_all` (~line 539)
- Test: `tests/test_slots.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_slots.py`:

```python
def test_delete_layer_remaps_slot_loop_ids():
    looper = _looper_with_layers(3)          # ids 0,1,2
    slot = looper.add_slot()
    looper.set_slot_loops(slot['id'], [0, 2])
    looper.delete_layer(1)                   # id 2 becomes id 1 after renumber
    assert slot['loop_ids'] == [0, 1]

def test_delete_layer_drops_its_own_id_from_slots():
    looper = _looper_with_layers(3)
    slot = looper.add_slot()
    looper.set_slot_loops(slot['id'], [1, 2])
    looper.delete_layer(1)                   # remove id 1; id 2 -> id 1
    assert slot['loop_ids'] == [1]

def test_clear_all_empties_slots_and_resets():
    looper = _looper_with_layers(2)
    looper.add_slot()
    looper.active_slot_id = 1
    looper.pending_slot = looper.slots[0]
    looper.clear_all()
    assert looper.slots == []
    assert looper.active_slot_id is None
    assert looper.pending_slot is None
    assert looper._next_slot_id == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_slots.py -k "remaps or drops or clear_all" -v`
Expected: FAIL — `delete_layer` doesn't touch slots yet (`assert [0,2] == [0,1]`), `clear_all` leaves slots populated.

- [ ] **Step 3: Apply pending_slot in the callback**

In `audio.py`, find (~line 310):

```python
                        # Apply pending scene at loop restart (PLAYING state only)
                        if loop_restarted and self.pending_scene is not None and self.state == LooperState.PLAYING:
                            self._apply_scene(self.pending_scene)
                            self.pending_scene = None
```

Add immediately after it:

```python
                        # Apply pending slot at loop restart (PLAYING state only)
                        if loop_restarted and self.pending_slot is not None and self.state == LooperState.PLAYING:
                            self._apply_slot(self.pending_slot)
                            self.pending_slot = None
```

- [ ] **Step 4: Remap slots in `delete_layer`**

In `audio.py` `delete_layer`, after the renumber loop (after line 512, the `layer.id = i` loop) and before the `print(...)`, add:

```python
            # Remap slots: drop the deleted id; shift higher ids down by one
            for slot in self.slots:
                slot['loop_ids'] = [i - 1 if i > layer_id else i
                                    for i in slot['loop_ids'] if i != layer_id]
```

- [ ] **Step 5: Reset slots in `clear_all`**

In `audio.py` `clear_all`, after `self.state = LooperState.IDLE` (line 539) and before the `print`, add:

```python
            self.slots = []
            self._next_slot_id = 1
            self.pending_slot = None
            self.active_slot_id = None
```

- [ ] **Step 6: Run the full slot test file to verify pass**

Run: `python -m pytest tests/test_slots.py -v`
Expected: PASS (10 passed).

- [ ] **Step 7: Commit**

```bash
git add -f audio.py tests/test_slots.py
git commit -m "feat(slots): quantized launch on loop wrap; prune slots on delete/clear"
```

---

## Task 3: Expose slots in `get_state` (`audio.py`)

**Files:**
- Modify: `audio.py` — `get_state` snapshot (~line 1435) and return dict (~line 1517)
- Test: `tests/test_slots.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_slots.py`:

```python
def test_get_state_includes_slots_block():
    looper = _looper_with_layers(2)
    slot = looper.add_slot()
    looper.set_slot_loops(slot['id'], [0])
    looper._apply_slot(slot)
    state = looper.get_state()
    assert state['slots']['list'] == [{'id': slot['id'], 'loop_ids': [0]}]
    assert state['slots']['active_id'] == slot['id']
    assert state['slots']['pending_id'] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_slots.py::test_get_state_includes_slots_block -v`
Expected: FAIL — `KeyError: 'slots'`.

- [ ] **Step 3: Snapshot slots under the lock**

In `get_state`, after the scenes/collapse snapshot lines (after line 1440, `collapse_timeout = self.collapse_timeout`), add:

```python
            slots_data = [{'id': s['id'], 'loop_ids': list(s['loop_ids'])} for s in self.slots]
            pending_slot_id = self.pending_slot['id'] if self.pending_slot else None
            active_slot_id = self.active_slot_id
```

- [ ] **Step 4: Add the slots block to the return dict**

In the `return { ... }` of `get_state`, after the `'collapse': { ... },` block (before `'scale':`), add:

```python
            'slots': {
                'list': slots_data,
                'pending_id': pending_slot_id,
                'active_id': active_slot_id,
            },
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_slots.py -v`
Expected: PASS (11 passed).

- [ ] **Step 6: Commit**

```bash
git add -f audio.py tests/test_slots.py
git commit -m "feat(slots): expose slots in get_state"
```

---

## Task 4: Session persistence + scene→slot migration (`audio.py`)

Persist slots in sessions, and convert old scene-based sessions to slots on load via a testable static helper.

**Files:**
- Modify: `audio.py` — `save_session` meta (~line 671), `load_session` (~line 729), add `_slots_from_meta` static method
- Test: `tests/test_slots.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_slots.py`:

```python
def test_slots_from_meta_reads_slots_when_present():
    meta = {'slots': [{'id': 1, 'loop_ids': [0, 2]}, {'id': 5, 'loop_ids': []}],
            'next_slot_id': 6}
    slots, next_id = WebLooper._slots_from_meta(meta)
    assert slots == [{'id': 1, 'loop_ids': [0, 2]}, {'id': 5, 'loop_ids': []}]
    assert next_id == 6

def test_slots_from_meta_migrates_old_scenes():
    # Old session: scenes dict keyed by str id, each with layer_states
    meta = {'scenes': {
        '1': {'id': 1, 'name': 'A', 'layer_states': [
            {'id': 0, 'is_playing': True, 'volume': 1.0},
            {'id': 1, 'is_playing': False, 'volume': 1.0}]},
        '2': {'id': 2, 'name': 'B', 'layer_states': [
            {'id': 0, 'is_playing': True, 'volume': 1.0},
            {'id': 1, 'is_playing': True, 'volume': 1.0}]},
    }}
    slots, next_id = WebLooper._slots_from_meta(meta)
    assert slots == [{'id': 1, 'loop_ids': [0]}, {'id': 2, 'loop_ids': [0, 1]}]
    assert next_id == 3

def test_slots_from_meta_empty_when_neither_present():
    slots, next_id = WebLooper._slots_from_meta({})
    assert slots == []
    assert next_id == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_slots.py -k slots_from_meta -v`
Expected: FAIL — `AttributeError: type object 'WebLooper' has no attribute '_slots_from_meta'`.

- [ ] **Step 3: Add the `_slots_from_meta` static method**

In `audio.py`, add near the other session helpers (e.g. just before `load_session`, ~line 688):

```python
    @staticmethod
    def _slots_from_meta(meta: dict) -> tuple:
        """Return (slots, next_slot_id) from session meta. Migrates old 'scenes'."""
        if 'slots' in meta:
            slots = [{'id': int(s['id']), 'loop_ids': [int(i) for i in s['loop_ids']]}
                     for s in meta['slots']]
            next_id = meta.get('next_slot_id',
                               max([s['id'] for s in slots], default=0) + 1)
            return slots, next_id
        # Migrate legacy scenes: one slot per scene, ids = its playing layers
        slots = []
        for scene in meta.get('scenes', {}).values():
            active = [st['id'] for st in scene.get('layer_states', [])
                      if st.get('is_playing')]
            slots.append({'id': len(slots) + 1, 'loop_ids': active})
        return slots, len(slots) + 1
```

- [ ] **Step 4: Write slots in `save_session`**

In `save_session`'s `meta = { ... }` dict, replace these two lines (currently ~671-672):

```python
                'scenes': {str(k): v for k, v in self.scenes.items()},
                'next_scene_id': self._next_scene_id,
```

with:

```python
                'slots': [{'id': s['id'], 'loop_ids': list(s['loop_ids'])} for s in self.slots],
                'next_slot_id': self._next_slot_id,
```

- [ ] **Step 5: Load slots in `load_session`**

In `load_session`, replace these four lines (currently ~729-732):

```python
            raw_scenes = meta.get('scenes', {})
            self.scenes = {int(k): v for k, v in raw_scenes.items()}
            self._next_scene_id = meta.get('next_scene_id', len(self.scenes) + 1)
            self.pending_scene = None
```

with:

```python
            self.slots, self._next_slot_id = self._slots_from_meta(meta)
            self.pending_slot = None
            self.active_slot_id = None
```

- [ ] **Step 6: Run to verify pass**

Run: `python -m pytest tests/test_slots.py -v`
Expected: PASS (14 passed).

- [ ] **Step 7: Commit**

```bash
git add -f audio.py tests/test_slots.py
git commit -m "feat(slots): persist slots in sessions; migrate legacy scenes"
```

---

## Task 5: Socket commands for slots (`routes.py`)

**Files:**
- Modify: `routes.py` — `handle_command` (~line 250, near the existing scene commands)

- [ ] **Step 1: Add the slot command handlers**

In `routes.py` `handle_command`, after the `set_scale` branch (~line 251) and before the final broadcast, add:

```python
    elif command == 'add_slot':
        looper.add_slot()
    elif command == 'delete_slot':
        looper.delete_slot(data.get('slot_id'))
    elif command == 'set_slot_loops':
        looper.set_slot_loops(data.get('slot_id'), data.get('loop_ids', []))
    elif command == 'launch_slot':
        looper.launch_slot(data.get('slot_id'), data.get('quantized', True))
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "import routes; print('ok')"`
Expected: prints `ok` (no syntax/import error).

- [ ] **Step 3: Commit**

```bash
git add -f routes.py
git commit -m "feat(slots): socket commands add/delete/set-loops/launch slot"
```

---

## Task 6: Slots side-panel UI (`index.html`, `style.css`, `app.js`)

Add the ▦ side-strip tab, the `#panelSlots` markup, styles, and the render + drag-and-drop + launch logic. No JS unit harness exists in this repo, so verification is `node --check` + manual.

**Files:**
- Modify: `templates/index.html` — side strip (~line 80) and `#sideContent` (after `#panelMeters`, ~line 168)
- Modify: `static/style.css` — append a Slots panel section
- Modify: `static/app.js` — add `renderSlots` + handlers; call from update handler (~line 1588) and `setSidePanel` (~line 1697)

- [ ] **Step 1: Add the side-strip button**

In `templates/index.html`, in `.side-strip`, replace the first stub button line:

```html
            <button class="side-icon side-icon-stub" title="FX (coming soon)">FX</button>
```

with:

```html
            <button class="side-icon" id="sideBtnSlots" data-panel="slots"
                    onclick="setSidePanel('slots')" title="Slots / Launcher">▦</button>
```

- [ ] **Step 2: Add the `#panelSlots` markup**

In `templates/index.html`, immediately after the closing `</div>` of `#panelMeters` (the METERS panel, ~line 168) and still inside `#sideContent`, add:

```html
            <!-- SLOTS panel -->
            <div class="side-panel" id="panelSlots">
                <div class="panel-header accent-primary"><span>SLOTS</span></div>
                <div class="slots-list" id="slotsList"></div>
                <button class="add-slot-btn" id="addSlotBtn" onclick="addSlot()">＋ add slot</button>
                <div class="slot-palette-label">LOOPS — drag into a slot</div>
                <div class="slot-palette" id="slotPalette"></div>
            </div>
```

- [ ] **Step 3: Add the styles**

Append to `static/style.css`:

```css
/* ── SLOTS panel ── */
.slots-list { display: flex; flex-direction: column; gap: 6px; padding: 8px; }
.slot-row {
    display: flex; align-items: center; gap: 8px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 7px; padding: 6px 8px; min-height: 40px;
}
.slot-row.drop-hover { border-color: #7cf; }
.slot-row.active { border-color: #7cf; box-shadow: inset 0 0 0 1px #7cf; }
.slot-row.pending { border-style: dashed; border-color: #7cf; }
.slot-launch {
    width: 28px; height: 28px; flex-shrink: 0; border-radius: 50%;
    border: none; cursor: pointer; background: #2f3340; color: #7cf; font-size: 12px;
}
.slot-row.active .slot-launch { background: #7cf; color: #111; }
.slot-loops { display: flex; flex-wrap: wrap; gap: 5px; flex: 1; }
.slot-chip {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; background: #2f3340; border-radius: 5px; padding: 4px 7px;
}
.slot-chip .chip-dot { width: 9px; height: 9px; border-radius: 50%; }
.slot-chip .chip-x { cursor: pointer; color: var(--text-muted); margin-left: 2px; }
.slot-empty { color: var(--text-muted); font-style: italic; font-size: 10px; }
.slot-delete { background: none; border: none; color: var(--text-muted); cursor: pointer; }
.add-slot-btn {
    margin: 4px 8px; padding: 6px; width: calc(100% - 16px);
    background: none; border: 1px dashed var(--border); border-radius: 7px;
    color: #7cf; cursor: pointer; font-size: 11px;
}
.slot-palette-label {
    font-size: 9px; text-transform: uppercase; letter-spacing: .5px;
    color: var(--text-muted); padding: 8px 8px 4px;
}
.slot-palette { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 8px 10px; }
.palette-chip {
    display: inline-flex; align-items: center; gap: 5px; cursor: grab;
    font-size: 11px; background: #2f3340; border: 1px dashed var(--border);
    border-radius: 5px; padding: 4px 7px;
}
.palette-chip .chip-dot { width: 9px; height: 9px; border-radius: 50%; }
```

- [ ] **Step 4: Add the JS — render, drag-drop, launch**

In `static/app.js`, add these functions (e.g. right after `setLayerColor`, ~line 723):

```javascript
        // =================================================================
        // SLOTS
        // =================================================================

        function addSlot() { sendCommand('add_slot'); }
        function deleteSlot(slotId) { sendCommand('delete_slot', { slot_id: slotId }); }
        function launchSlot(slotId) {
            const quantized = serverState.state === 'playing';
            sendCommand('launch_slot', { slot_id: slotId, quantized });
        }
        function addLoopToSlot(slotId, loopId) {
            const slot = (serverState.slots?.list || []).find(s => s.id === slotId);
            if (!slot || slot.loop_ids.includes(loopId)) return;
            sendCommand('set_slot_loops', { slot_id: slotId, loop_ids: [...slot.loop_ids, loopId] });
        }
        function removeLoopFromSlot(slotId, loopId) {
            const slot = (serverState.slots?.list || []).find(s => s.id === slotId);
            if (!slot) return;
            sendCommand('set_slot_loops', { slot_id: slotId, loop_ids: slot.loop_ids.filter(i => i !== loopId) });
        }
        function slotDragOver(ev) { ev.preventDefault(); ev.currentTarget.classList.add('drop-hover'); }
        function slotDragLeave(ev) { ev.currentTarget.classList.remove('drop-hover'); }
        function slotDrop(ev, slotId) {
            ev.preventDefault();
            ev.currentTarget.classList.remove('drop-hover');
            const loopId = parseInt(ev.dataTransfer.getData('text/plain'), 10);
            if (!Number.isNaN(loopId)) addLoopToSlot(slotId, loopId);
        }

        let _lastSlotsJson = '';
        function renderSlots() {
            const json = JSON.stringify(serverState.slots) + JSON.stringify(
                (serverState.layers || []).map(l => [l.id, l.name, l.color]));
            if (json === _lastSlotsJson) return;
            _lastSlotsJson = json;

            const layers = serverState.layers || [];
            const byId = Object.fromEntries(layers.map(l => [l.id, l]));
            const slots = serverState.slots || { list: [], active_id: null, pending_id: null };

            const list = document.getElementById('slotsList');
            if (!list) return;
            list.innerHTML = slots.list.map(slot => {
                const cls = (slot.id === slots.active_id ? ' active' : '')
                          + (slot.id === slots.pending_id ? ' pending' : '');
                const chips = slot.loop_ids.length === 0
                    ? '<span class="slot-empty">drop loops here…</span>'
                    : slot.loop_ids.map(id => {
                        const l = byId[id]; if (!l) return '';
                        return `<span class="slot-chip">
                            <span class="chip-dot" style="background:${l.color}"></span>${l.name}
                            <span class="chip-x" onclick="removeLoopFromSlot(${slot.id}, ${id})">✕</span>
                        </span>`;
                      }).join('');
                return `<div class="slot-row${cls}" ondragover="slotDragOver(event)"
                            ondragleave="slotDragLeave(event)" ondrop="slotDrop(event, ${slot.id})">
                    <button class="slot-launch" onclick="launchSlot(${slot.id})">▶</button>
                    <div class="slot-loops">${chips}</div>
                    <button class="slot-delete" onclick="deleteSlot(${slot.id})">✕</button>
                </div>`;
            }).join('');

            const palette = document.getElementById('slotPalette');
            palette.innerHTML = layers.map(l => `
                <span class="palette-chip" draggable="true"
                      ondragstart="event.dataTransfer.setData('text/plain', '${l.id}')">
                    <span class="chip-dot" style="background:${l.color}"></span>${l.name}
                </span>`).join('');
        }
```

- [ ] **Step 5: Call `renderSlots` from the update handler**

In `static/app.js`, find the `// --- Scenes ---` block (~line 1587-1588):

```javascript
            // --- Scenes ---
            renderScenes();
```

Add immediately after it:

```javascript
            // --- Slots ---
            renderSlots();
```

- [ ] **Step 6: Render palette when the Slots panel opens**

In `static/app.js` `setSidePanel`, find the last line before the closing brace (~line 1697):

```javascript
            if (name === 'scale') renderFretboard();
```

Add after it:

```javascript
            if (name === 'slots') renderSlots();
```

- [ ] **Step 7: Syntax-check the JS**

Run: `node --check static/app.js && echo JS_OK`
Expected: prints `JS_OK`. (If `node` is unavailable, skip — Step 8 covers it.)

- [ ] **Step 8: Manual verification**

Run the app (`python main.py`, pick the audio device), open the web UI, record a master loop + an overdub so there are ≥2 loops. Then:
- Click the ▦ side icon → the SLOTS panel opens; the loop palette shows your loops.
- Click `＋ add slot` → an empty slot row appears ("drop loops here…").
- Drag a palette chip onto the slot → the loop chip appears in the slot.
- Click a chip's ✕ → it leaves the slot.
- Click a slot's ▶ while playing → on the next loop wrap, only that slot's loops are audible and the row highlights as active.
- Click the slot's row ✕ → the slot is removed.

- [ ] **Step 9: Commit**

```bash
git add -f templates/index.html static/style.css static/app.js
git commit -m "feat(slots): side-panel launcher UI with drag-and-drop"
```

---

## Task 7: Remove scenes + collapse-on-silence (`audio.py`, `routes.py`, `index.html`, `app.js`, `style.css`)

Now that slots fully cover the use case, delete the scenes feature and the reactive collapse-on-silence. Do this as one atomic change so no dangling references remain.

**Files:**
- Modify: `audio.py` — `__init__` Scenes + collapse blocks, scene/collapse methods, callback collapse block, `get_state` scenes/collapse snapshot + return blocks
- Modify: `routes.py` — scene/collapse command branches
- Modify: `templates/index.html` — SCENES edit section + collapse controls
- Modify: `static/app.js` — scene/collapse functions + the `renderScenes()` call
- Modify: `static/style.css` — scene/collapse CSS (optional cleanup)

- [ ] **Step 1: Remove scene + collapse state from `__init__`**

In `audio.py` `__init__`, delete the entire `# Scenes` block (lines ~173-176: `self.scenes`, `self._next_scene_id`, `self.pending_scene`) and the entire `# Reactive scene collapse` block (lines ~178-184: `collapse_enabled`, `collapse_scene_id`, `collapse_timeout`, `collapse_threshold`, `_silence_frames`, `_collapse_triggered`). Leave the new Slots block intact.

- [ ] **Step 2: Remove the collapse block from the callback**

In `audio.py` `audio_callback`, delete the `# REACTIVE SCENE COLLAPSE` block (lines ~240-254, the `if (self.collapse_enabled ...)` through the `else: self._silence_frames = 0 / self._collapse_triggered = False`).

- [ ] **Step 3: Remove the scene/collapse application from the callback**

In `audio.py` `audio_callback`, delete the pending-scene apply block (~line 310-312):

```python
                        # Apply pending scene at loop restart (PLAYING state only)
                        if loop_restarted and self.pending_scene is not None and self.state == LooperState.PLAYING:
                            self._apply_scene(self.pending_scene)
                            self.pending_scene = None
```

(Keep the pending-**slot** block added in Task 2.)

- [ ] **Step 4: Delete the scene/collapse methods**

In `audio.py`, delete these methods entirely: `save_scene`, `_apply_scene`, `load_scene`, `delete_scene`, `rename_scene`, `set_collapse_scene`, `set_collapse_enabled` (the `SCENE MANAGEMENT` block, ~lines 543-635). Keep the `SLOT LAUNCHER` block.

- [ ] **Step 5: Remove scenes/collapse from `get_state`**

In `audio.py` `get_state`, delete the snapshot lines:

```python
            scenes_data = list(self.scenes.values())
            pending_scene_id = self.pending_scene['id'] if self.pending_scene else None
            collapse_enabled = self.collapse_enabled
            collapse_scene_id = self.collapse_scene_id
            collapse_timeout = self.collapse_timeout
```

and delete the two return blocks:

```python
            'scenes': {
                'list': scenes_data,
                'pending_id': pending_scene_id,
            },
            'collapse': {
                'enabled': collapse_enabled,
                'scene_id': collapse_scene_id,
                'timeout': collapse_timeout,
            },
```

(Keep the `'slots'` block added in Task 3.)

- [ ] **Step 6: Remove the `pending_scene = None` in `load_session`**

If any `self.pending_scene = None` line remains (e.g. ~line 732 was already replaced in Task 4 — verify none remains anywhere). Search: `grep -n "pending_scene\|self.scenes\|collapse" audio.py` should return **nothing**.

- [ ] **Step 7: Remove scene/collapse command branches in `routes.py`**

In `routes.py` `handle_command`, delete the branches for `save_scene`, `load_scene`, `delete_scene`, `rename_scene`, `set_collapse_scene`, `set_collapse_enabled` (lines ~223-234).

- [ ] **Step 8: Remove the SCENES section in `index.html`**

In `templates/index.html`, delete the entire Scenes edit section (lines ~243-267): the `<!-- Scenes -->` block containing `#sceneNameInput`, `#scenesList`, and the `.collapse-controls` div (`#collapseToggle`, `#collapseTimeout`, `#collapseHint`).

- [ ] **Step 9: Remove scene/collapse JS in `app.js`**

In `static/app.js`, delete the functions `saveScene`, `loadScene`, `deleteScene`, `renderScenes`, `updateCollapseControls`, `setCollapseScene`, `setCollapseEnabled`, `setCollapseTimeout` (the `// SCENES` block, ~lines 727-810), the `let _lastScenesJson = '';` declaration (~line 345), and the `// --- Scenes ---` / `renderScenes();` call (~line 1587-1588).

- [ ] **Step 10: (Optional) Remove dead scene/collapse CSS**

In `static/style.css`, remove rules only used by the deleted markup (e.g. `.scene-item`, `.scenes-empty`, `.btn-load-scene`, `.btn-delete-scene`, `.btn-idle-scene`, `.scene-pending-badge`, `.collapse-controls`, `.collapse-row`, `.collapse-timeout-input`, `.collapse-hint`, `.save-scene-row` if not reused by sessions). Leave `.scene-name-input` if still referenced by the session save row (`#sessionNameInput` uses it — keep it).

- [ ] **Step 11: Verify no dangling references**

Run:
```bash
grep -rn "scene\|collapse\|pending_scene" audio.py routes.py
grep -rn "renderScenes\|saveScene\|loadScene\|deleteScene\|setCollapse\|_lastScenesJson\|collapseToggle" static/app.js templates/index.html
```
Expected: both return **nothing** (or only unrelated matches like the kept `#sessionNameInput`'s `scene-name-input` class). Then:
```bash
python -c "import routes; print('import ok')"
node --check static/app.js && echo JS_OK
python -m pytest tests/ -v
```
Expected: `import ok`, `JS_OK`, all tests pass.

- [ ] **Step 12: Manual smoke test**

Run the app, record a couple of loops, confirm the Slots panel still works end-to-end (add/drag/launch/delete) and that the old SCENES UI is gone with no console errors. Save a session and reload it — slots persist.

- [ ] **Step 13: Commit**

```bash
git add -f audio.py routes.py templates/index.html static/app.js static/style.css
git commit -m "refactor: remove scenes and collapse-on-silence (replaced by slots)"
```

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** slot model (T1), quantized launch + delete/clear pruning (T2), state exposure (T3), session persistence + legacy migration (T4), socket commands (T5), drag-and-drop UI (T6), scenes + collapse removal (T7). On/off-only and no-auto-advance are honored (no per-slot volume field; no sequencing logic).
- **Layer-id renumbering** is the subtle correctness risk — covered explicitly in T2 Step 4 with a test.
- **Ordering** is additive-then-remove so the app stays runnable and tests stay green at every commit; scenes are only deleted in T7 after slots fully replace them.
