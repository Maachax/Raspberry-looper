# Audio Effects — Design

**Date:** 2026-05-29
**Status:** Approved for planning

## Goal

Add audio effects (reverb, delay, chorus, distortion, low/high-pass filter) to the looper.
Each loop has an effect chain; effects are **baked** into the loop's audio (rendered once,
played back cheaply), with a single **live bus reverb** on the master mix. Effect chains can
differ **per section** via overrides, matching how the user performs (e.g. Section 1: loop1 =
echo+delay, loop2 = distortion, reverb on all; Section 2: loop1 = delay only, loop2 =
distortion with different params). Effects are included automatically in audio export.

## Key Decisions (from brainstorming)

- **Scope:** effects apply to recorded **loop playback** only. Live-input effects are a future
  phase and out of scope here.
- **Routing model (C):** each loop has a **default chain**; a **section may override** a
  loop's chain. Resolution = section override if present, else loop default.
- **Processing = baked:** render a loop's dry audio through its chain into a **wet** buffer
  that plays back. Re-render on edit. Playback cost stays as low as today.
- **Bus reverb is live:** the section/master reverb runs once on the summed mix in the audio
  callback — a single constant-cost instance with a natural open tail. (Baking it per-loop
  would chop the tail to the loop length.) This is the only live-DSP element.
- **DSP backend = `pedalboard`** (Spotify's open-source, JUCE-backed, NumPy-native library;
  runs fully offline; GPLv3 — fine for personal use). Gated by an early feasibility benchmark
  on the Pi, with a hand-rolled NumPy/SciPy fallback if it won't install/perform.
- **v1 palette:** Reverb, Delay, Chorus, Distortion, Filter (LP/HP).

## Architecture

```
dry buffer ──render(chain)──▶ wet buffer ─┐   (per loop, baked, cached)
dry buffer ──render(chain)──▶ wet buffer ─┤─▶ sum ─▶ LIVE bus reverb ─▶ master vol ─▶ out
                                          ┘        (one pedalboard instance, per-block)
```

- **`pedalboard`** provides both the offline render (`Pedalboard(chain).process(audio, sr)`)
  and the live bus reverb (a persistent `Pedalboard([Reverb(...)])` processed per audio block
  with streaming state retained).
- The existing per-sample mix loop in `audio_callback` is unchanged except it reads each
  layer's **wet** buffer; after summing, the block is run through the live bus reverb before
  master volume.

## Data Model

- **`Effect`** = `{'type': str, 'params': {name: value}, 'enabled': bool}`.
  Types: `reverb`, `delay`, `chorus`, `distortion`, `filter`. A **chain** is an ordered list
  of effects.
- **`effects.py` `EFFECT_SCHEMAS`** — for each type, an ordered list of params. A param is
  either **numeric** `{name, min, max, default, unit}` or **enum**
  `{name, options: [...], default}` (e.g. distortion: `drive_db` 0–40; delay: `time_s`,
  `feedback` 0–0.95, `mix` 0–1; filter: `mode` enum LP/HP, `cutoff_hz`, `resonance`; reverb:
  `room_size`, `damping`, `wet`; chorus: `rate_hz`, `depth`, `mix`). Drives both the live
  effect construction and the UI controls (slider for numeric, toggle/select for enum).
- **`LoopLayer`** gains:
  - `dry: np.ndarray` — the original recorded audio (never mutated by effects).
  - `fx_chain: list[Effect]` — the loop's default chain.
  - existing `buffer` becomes the **wet** buffer that the callback plays (equals `dry` when
    the resolved chain is empty).
- **Section** (`{id, loop_ids}`) gains:
  - `fx_overrides: {loop_id: chain}` — per-loop chain overrides for this section.
  - `bus: Effect | None` — optional per-section bus reverb override.
- **WebLooper** gains:
  - `wet_cache: {(dry_hash, chain_hash): np.ndarray}` — rendered wet buffers, so identical
    chains never re-render and section switches just swap buffers.
  - `bus_reverb` — the live `pedalboard.Pedalboard` instance + its current params.
  - `master_bus: Effect | None` — the global default bus reverb (used when a section has no
    `bus` override).

## Behaviour

- **Chain resolution** (`effects.resolve_chain(loop, section)`): returns the section override
  for that loop if present, else the loop's `fx_chain`.
- **Bake** (`effects.render_wet(dry, chain, sr)`): tile `dry` 3× → process through the chain →
  return the **last** cycle (`[2·len : 3·len]`). This bakes the tail bleeding from the
  previous repeat so the loop seams seamlessly. Empty/all-disabled chain → return `dry`
  unchanged. Memoised via `wet_cache`.
- **Applying a section** (`_apply_section`, extended): for every layer set
  `is_playing = id in loop_ids`; set each layer's wet `buffer` to the cached/rendered wet for
  its resolved chain; reconfigure the live `bus_reverb` from the section `bus` (else
  `master_bus`). When no section is active (manual play), loops use their default-chain wet.
- **Editing effects** re-bakes the affected loop's wet buffer (debounced ~150 ms). The new wet
  buffer is swapped in at the **next loop restart** (reusing the existing quantized
  `pending_*`-at-loop-wrap mechanism) to avoid mid-cycle clicks — consistent with the
  "near-instant, not a continuous knob sweep" experience agreed in brainstorming.
- **Record/trim** updates `dry`; the wet buffer is re-baked from the new dry.
- **Bus reverb** processes the summed master block every callback with streaming state
  retained, giving a continuous open tail across loop cycles.

## UI (FX side-panel tab `≈`) — validated in brainstorming

New side-strip icon `≈` opening `#panelFx`, same vertical panel pattern as Scale/Sections:

1. **Loop selector** — which loop's chain you're editing.
2. **Scope toggle** — `Default ⟷ Section: <active>`. When the active section overrides this
   loop, a banner shows "Overrides the loop default in this section · ↺ revert".
3. **Chain** — effect chips (drag to reorder, tap to select, `＋ add` from the palette,
   remove). Order = processing order.
4. **Selected effect params** — sliders generated from `EFFECT_SCHEMAS`.
5. **Bus** — the section/master reverb toggle + params ("reverb on all").

`get_state` exposes per-loop `fx_chain`, per-section `fx_overrides`, the resolved bus, and
`EFFECT_SCHEMAS` so the frontend can render chips and knobs generically.

## Socket Commands (`routes.py`)

- `fx_set_loop_chain` (loop_id, chain) — set a loop's default chain.
- `fx_set_section_override` (section_id, loop_id, chain) — set/replace a section override.
- `fx_clear_section_override` (section_id, loop_id) — revert to the loop default.
- `fx_set_bus` (section_id|null, effect|null) — set the per-section bus, or the master bus when
  section_id is null.

Each triggers the appropriate re-bake and broadcasts state. (UI add/remove/reorder/param edits
all resolve to a `fx_set_loop_chain` / `fx_set_section_override` with the new chain.)

## Export

The mixed export already sums the layers' (now wet) buffers, so insert effects are included
automatically. After mixing, apply the current resolved bus reverb offline over the mixed
buffer (a fresh `pedalboard` reverb with the active bus params, tail included). Export reflects
the currently active section.

## Feasibility Gate (first implementation task)

Before building anything else: install `pedalboard` on the Pi and benchmark
(a) `render_wet` for an ~8 s mono loop through a 3-effect chain — must be well under ~100 ms so
edits feel instant; (b) the live bus reverb sustained in the audio callback — confirm
`callback_time` stays within budget and `dropout_count` does not climb. If `pedalboard` will
not install or hold real-time, fall back to hand-rolled NumPy/SciPy effects (delay = circular
buffer, filter = `scipy.signal` biquad, distortion = waveshaping, chorus = modulated delay,
reverb = Schroeder/Freeverb-style), keeping the same `effects.py` interface.

## File Structure

- **`effects.py` (new):** `EFFECT_SCHEMAS`, `make_pedalboard(chain)`, `render_wet(dry, chain, sr)`,
  `resolve_chain(loop_chain, section_override)`, `chain_hash(chain)`, `make_bus(reverb_params)`.
  Pure, dependency-light, unit-testable without an audio device.
- **`audio.py`:** `LoopLayer` gains `dry`/`fx_chain`; `WebLooper` gains `wet_cache`,
  `bus_reverb`, `master_bus`, the FX setters, re-bake logic, extended `_apply_section`,
  callback bus processing, and extended `get_state` + session persistence of chains/overrides/bus.
- **`routes.py`:** the four FX commands.
- **Frontend:** `#panelFx` markup, `renderFx()` + handlers in `app.js`, CSS.

## Testing

- **`effects.py` unit:** `resolve_chain` (override vs default vs empty); `chain_hash` stability
  & sensitivity; `render_wet` — empty chain returns dry unchanged, output length == loop length,
  **seam continuity** (last sample → first sample step is small for a tail effect), filter
  attenuates highs, distortion raises RMS / adds harmonics, bypass(disabled)==dry.
- **`audio.py` unit:** wet_cache hit on identical chain; `_apply_section` sets wet buffers +
  `is_playing` from resolved chains; editing a chain re-bakes; record/trim re-bakes from dry;
  session round-trip preserves chains/overrides/bus.
- **Manual (Pi):** feasibility benchmark; no dropouts with live bus reverb; UI scope
  toggle/override banner/revert; export contains effects.

## Out of Scope (future)

- Live-input (monitoring) effects.
- Parameter automation/LFO over time; sidechain; per-section bus effects other than reverb.
- A saveable user effect-preset library.
