# Screen-Driven Knob Layers (UI Context) — Design

**Date:** 2026-07-02
**Status:** Approved

## Goal

In PLAY mode, the MPK knobs follow the screen: FX panel open → knobs shape
the effect being looked at; trim editor open → knobs move the trim handles;
anything else → loop volumes. Explicit MPK modes (SECTION_EDIT / FX_EDIT)
still override.

## Design

### Client → server: `ui_context`

The browser computes its context whenever navigation/panel/trim state
changes and sends the socket command `ui_context`:

```json
{"context": "fx", "fx_loop": 2, "fx_slot": 0}
```

- `context`: `"trim"` (trim editor expanded, in edit view) beats
  `"fx"` (FX side panel active) beats `"home"` (everything else —
  sections/scale/meters don't change knobs).
- Sent only on change; routed to `MidiController.set_ui_context(ctx)`.
- Multiple clients: last writer wins (single-user home setup).

### Knob resolution (PLAY mode only)

Dispatch order for CC triggers while `mode == 'play'`:

| context | K1–K3 (CC 70–72) | K7/K8 (CC 76/77) | other CCs |
|---|---|---|---|
| `home` | loop volumes 1–3 (bindings, unchanged) | loop volumes 7–8 | volumes |
| `fx` | params 1–3 of (`fx_loop`, `fx_slot`) — same 300 ms idle-commit | bus room / wet | unbound |
| `trim` | K1 = start ratio, K2 = end ratio (CC/127) | unbound | unbound |

`section_edit` / `fx_edit` modes ignore `ui_context` entirely (explicit
mode wins); exiting returns knobs to screen-driven.

### Trim preview + apply

- Knob turns update `trim_preview = {start, end}` ratios on the controller
  (start clamped ≥0, end ≤1, `end - start ≥ 0.02`), pushed to clients via
  the `midi` state block; the open trim editor moves its handles to match
  (preview only, no audio change).
- Apply: **note 69 (A key)** while context is `trim` → server calls
  `apply_trim(start_ratio * duration, end_ratio * duration)` using the
  master loop duration, then clears the preview. The phone's APPLY button
  keeps working independently.
- Preview starts unset (`None`); the first knob turn sets it (absolute
  knobs jump the handle to the knob position — accepted).
- Leaving the trim context clears the preview without applying.

### Status / banner

`status()` gains `'ui_context': 'home'|'fx'|'trim'` and
`'trim_preview': {start, end} | null`. `renderMidiBanner` shows a teal
"KNOBS → FX …" / "KNOBS → TRIM" strip when PLAY mode has a non-home
context (edit-mode banners unchanged and take precedence).

## Error handling

- Context `fx` with missing loop/slot (deleted meanwhile) → knob no-op.
- Trim apply with no preview or no master loop → no-op.
- Unknown context string → treated as `home`.

## Testing

- Unit: `set_ui_context` normalization; CC rerouting per context; volumes
  untouched in `home`; explicit modes override context; trim ratio
  clamping; apply computes seconds from ratios and clears preview; leaving
  context clears preview.
- Live: Max on phone + MPK (FX knob follows panel selection; trim handles
  follow K1/K2; A key applies; volumes normal on HOME).

## Out of scope

- Per-client contexts (single user).
- Encoder-relative knob modes.
