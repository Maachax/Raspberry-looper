# Slot Launcher — Design

**Date:** 2026-05-29
**Status:** Approved for planning

## Goal

Add an Ableton-Session-style **slot launcher** to the looper: a side-panel tab holding
a list of *slots*. Each slot is a drag-and-droppable set of loops with its own launch
button. Hitting a slot's launch button swaps the set of playing loops at the next loop
start. The launched slot keeps looping until another slot is launched.

This replaces the existing **scenes** feature, which modeled the same idea (a saved
combination of active layers) but with a clunkier save/name/recall UI. Slots are the
better expression of it: direct, visual, drag-to-build, launch-in-place.

## Concepts

- **Loop / layer** — unchanged. A recorded `LoopLayer` (`id`, `name`, `color`, `volume`,
  `is_playing`, buffer). All layers share the master loop length.
- **Slot** — an ordered entry holding a *set of loop ids*. Launching a slot sets
  `is_playing = (layer.id in slot.loop_ids)` for every layer: loops in the slot turn on,
  everything else turns off. An **empty slot** is valid and means silence (a breakdown /
  stop). Slots carry **no per-slot volume** — each loop plays at its current volume.

## Behavior

- **Launch** — clicking a slot's ▸ button:
  - If playing and `master_length > 0`: schedule the slot to apply at the next loop
    restart (quantized), reusing the existing `pending_*` mechanism.
  - Otherwise: apply immediately.
- **Looping** — launching only changes which loops are on; the master loop keeps running.
  The launched slot's set persists until another slot is launched. No auto-advance, no
  sequence (deliberately out of scope — see below).
- **Active highlight** — the panel highlights the slot whose loop set matches the current
  `is_playing` state (`active_slot_id`). Manually toggling a layer in the LAYERS panel
  that breaks the match clears the highlight; it does not modify the slot.
- **Empty-on-launch** — launching an empty slot mutes all loops (valid silence state).
- **New loop while a slot is active** — a freshly recorded layer defaults to playing and
  is not part of any slot; the highlight clears until a slot is launched again.

## UI

A third icon in the existing `.side-strip` (after ♩ Scale, dB Meters), e.g. ▦, toggling a
new `.side-panel#panelSlots`. The panel is the same vertical-scroll column as the Scale
panel, so slots stack **vertically**; each slot is a horizontal row:

```
SLOTS
┌──────────────────────────────────────┐
│ ▸  [Drums][Bass]                      │
│ ▸  [Drums][Bass][Lead]   ← playing    │
│ ▸  [Lead]                             │
│ ▸  drop loops here…                   │
│ ＋ add slot                           │
├──────────────────────────────────────┤
│ Loops — drag into a slot              │
│ [Drums] [Bass] [Lead]                 │
└──────────────────────────────────────┘
```

- **Slot row** — ▸ launch button (glows when this slot is the active one) + the slot's
  loops as colored chips. Each chip has a small × to remove it from the slot.
- **＋ add slot** — appends a new empty slot. A slot delete control (e.g. row hover ✕)
  removes it.
- **Loop palette** — all current loops as draggable chips (color + name). Drag a chip
  onto a slot row to add that loop to the slot. Dragging a loop already in the slot is a
  no-op.
- Slot reordering is **not** included (future nicety).

## Data Model & Backend Changes (`audio.py`)

Replace the scenes fields/methods with slots:

- **State (`__init__`)**: remove `scenes`, `_next_scene_id`, `pending_scene`, and all
  `collapse_*` / `_silence_frames` / `_collapse_triggered` fields. Add:
  - `self.slots: list[dict]` — each `{'id': int, 'loop_ids': [int, ...]}`
  - `self._next_slot_id: int = 1`
  - `self.pending_slot: dict | None = None`
  - `self.active_slot_id: int | None = None`
- **Methods** (mirror the old scene methods):
  - `add_slot() -> dict` — append empty slot, return it.
  - `delete_slot(slot_id)` — remove slot; clear `pending_slot`/`active_slot_id` if it
    referenced this slot.
  - `set_slot_loops(slot_id, loop_ids)` — replace a slot's loop set (used by add/remove
    chip; frontend sends the resulting list). Drops ids that no longer exist.
  - `launch_slot(slot_id, quantized=True)` — schedule via `pending_slot` or apply now.
  - `_apply_slot(slot)` — set every layer's `is_playing`, set `active_slot_id`.
- **Audio callback**: remove the collapse block (lines ~242–254). Where `pending_scene`
  was applied on `loop_restarted` (~310–312), apply `pending_slot` instead.
- **`delete_layer` / `clear_all`**: prune the deleted layer id from every slot
  (`clear_all` empties `slots` too).
- **`get_state`**: replace the `'scenes'` and `'collapse'` blocks with a `'slots'` block:
  `{'list': [{'id','loop_ids'}, ...], 'pending_id': ..., 'active_id': ...}`.
- **Session persistence**:
  - `save_session`: write `'slots'` + `'next_slot_id'` instead of `'scenes'`/`'next_scene_id'`.
  - `load_session`: read `'slots'`. **Backward compat** — if an old session has `'scenes'`
    but no `'slots'`, convert each scene to a slot using the ids where `is_playing` is true
    in its `layer_states`. Reset `pending_slot`/`active_slot_id`.

## Routes / Sockets (`routes.py`)

- Remove command handlers: `save_scene`, `load_scene`, `delete_scene`, `rename_scene`,
  `set_collapse_scene`, `set_collapse_enabled`.
- Add: `add_slot`, `delete_slot`, `set_slot_loops` (args `slot_id`, `loop_ids`),
  `launch_slot` (args `slot_id`, `quantized`). Each broadcasts updated state as today.

## Frontend (`templates/index.html`, `static/app.js`, `static/style.css`)

- Add the ▦ side-strip button + `#panelSlots` markup; register it in `setSidePanel`.
- Remove the SCENES edit-panel section and collapse controls (index.html ~243–266) and
  their JS (`saveScene`, `loadScene`, `deleteScene`, `renameScene`, `setCollapseEnabled`,
  `setCollapseTimeout`) and CSS.
- Render slots from `state.slots`: one row per slot (launch button + chips + remove ×),
  ＋ add slot, and the loop palette from `state.layers`.
- Implement HTML5 drag-and-drop: palette chip = draggable; slot row = drop target →
  emit `set_slot_loops` with the new id list. Chip × → `set_slot_loops` without that id.
- Launch button → `launch_slot`. Highlight the row whose `id === state.slots.active_id`.

## Out of Scope (future)

- **Auto-advancing sequences** (the original `1,2,1,2` "structure" idea) — slots are
  manually launched only.
- **Per-slot volume / mix snapshots** — on/off only.
- **Slot names and reordering.**
- **Collapse-on-silence** — removed entirely; can return later pointed at a slot.

## Testing

- Unit (`audio.py`): add/delete slot; `set_slot_loops` prunes stale ids; `launch_slot`
  immediate vs. quantized (`pending_slot` applied on loop restart); `_apply_slot` sets
  `is_playing` correctly incl. empty slot = all off; `delete_layer`/`clear_all` prune
  slots; session round-trip with slots; old-session `scenes`→`slots` migration.
- Manual: drag loops into slots, launch, confirm switch lands on the loop boundary,
  confirm highlight tracks active slot and clears on manual layer toggle.
