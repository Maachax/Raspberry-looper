# Audio Effects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add baked per-loop audio effects (reverb, delay, chorus, distortion, filter) with per-section chain overrides and one live master-bus reverb, included automatically in export.

**Architecture:** A new pure module `effects.py` defines the effect schemas, builds `pedalboard` chains, and renders a loop's dry audio into a "wet" buffer (tiled 3× and last-cycle extracted so delay/reverb tails wrap seamlessly). `audio.py`'s `LoopLayer` keeps an untouched `dry` buffer plus the played `buffer` (wet); the audio callback mixes wet buffers and runs the summed block through one persistent `pedalboard` reverb (the live bus). Sections may override a loop's chain; resolution is "section override else loop default" (model C). Effects are non-destructive (always re-baked from `dry`) and swapped in at the next loop boundary.

**Tech Stack:** Python, NumPy, `pedalboard` (Spotify, JUCE-backed, NumPy-native, GPLv3), Flask-SocketIO, vanilla JS. Tests via `pytest` (`tests/`, construct `WebLooper()` directly — see `tests/test_sections.py`).

**Key existing facts:**
- `LoopLayer.__init__(self, layer_id, name, buffer)` sets `id, name, buffer, length, volume=1.0, is_playing=True, color`. `to_dict()` returns `id, name, duration, volume, is_playing, color`.
- The master layer is created in `stop_recording` (`audio.py:412`) and re-created in `apply_trim` (`audio.py:868`); overdub layers in `_finalize_overdub` (`audio.py:357`). All do `LoopLayer(id, name, buffer)`.
- The audio callback mixes per sample into `loop_output` (a `frames`-length np array) then `output += loop_output * self.master_volume` (`audio.py:277-278`). The mix reads `layer.buffer`.
- `_apply_section(section)` (`audio.py:561`) sets `is_playing` per `loop_ids` and `active_section_id`; it's called within the lock (callback applies `pending_section` at loop wrap).
- Sections are `{'id', 'loop_ids'}`; persisted in `save_session` meta (`audio.py:614`) and rebuilt by `_sections_from_meta` (`audio.py:632`). `get_state` has a `'sections'` block.
- `delete_layer` (`audio.py:475`) renumbers layer ids and prunes section `loop_ids`. `clear_all` (`audio.py:514`) empties layers/sections.
- `.gitignore` has a blanket `*`: **new files need `git add -f`.**
- `pedalboard.Pedalboard` is callable: `board(samples_1d_float32, sample_rate, reset=False) -> np.ndarray`. Effects used: `Reverb(room_size, damping, wet_level, dry_level)`, `Delay(delay_seconds, feedback, mix)`, `Chorus(rate_hz, depth, mix)`, `Distortion(drive_db)`, `LadderFilter(mode, cutoff_hz, resonance)` with `LadderFilter.Mode.LPF12 / HPF12`.

---

## Task 1: pedalboard feasibility gate (spike)

Fail fast: confirm `pedalboard` installs and performs on the Pi before building on it. Not TDD — a measured spike that produces a reusable benchmark script.

**Files:**
- Create: `tools/fx_bench.py`

- [ ] **Step 1: Install pedalboard**

Run: `pip install pedalboard --break-system-packages`
Expected: installs a wheel (aarch64 manylinux) without compiling. If it fails to install, STOP and report — the fallback is hand-rolled effects (out of scope for this task; escalate to re-plan `effects.py` internals).

- [ ] **Step 2: Write the benchmark script**

Create `tools/fx_bench.py`:

```python
"""Feasibility benchmark for pedalboard on this machine.

Renders a representative loop through a 3-effect chain and measures per-render
time, then measures per-block time for a live reverb. Run on the Pi:
    python tools/fx_bench.py
"""
import time
import numpy as np
from pedalboard import Pedalboard, Reverb, Delay, Distortion

SR = 44100
BLOCK = 256

def main():
    dry = np.random.randn(SR * 8).astype(np.float32) * 0.2   # 8s mono
    chain = Pedalboard([Distortion(drive_db=12), Delay(delay_seconds=0.25, feedback=0.4, mix=0.4), Reverb(room_size=0.5)])

    # Offline render (tiled x3, like render_wet)
    tiled = np.tile(dry, 3)
    t0 = time.perf_counter()
    for _ in range(5):
        chain(tiled, SR, reset=True)
    render_ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"render_wet (8s x3, 3 fx): {render_ms:.1f} ms  (target: < 100 ms)")

    # Live bus reverb per block
    bus = Pedalboard([Reverb(room_size=0.6)])
    block = np.random.randn(BLOCK).astype(np.float32) * 0.2
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        bus(block, SR, reset=False)
    per_block_ms = (time.perf_counter() - t0) / n * 1000
    budget_ms = BLOCK / SR * 1000
    print(f"live bus reverb / block: {per_block_ms:.3f} ms  (block budget: {budget_ms:.2f} ms)")
    print("PASS" if render_ms < 100 and per_block_ms < budget_ms * 0.5 else "REVIEW")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the benchmark on the Pi**

Run: `python tools/fx_bench.py`
Expected: prints render and per-block timings and `PASS`. Acceptance: render < 100 ms, live bus per-block < 50% of the ~5.8 ms block budget. If `REVIEW`/over budget, STOP and report numbers — we either lower quality/limit bus, or fall back to hand-rolled effects.

- [ ] **Step 4: Commit**

```bash
git add -f tools/fx_bench.py
git commit -m "chore(fx): pedalboard feasibility benchmark"
```

---

## Task 2: effects.py — schemas, defaults, resolution, hashing (pure, TDD)

**Files:**
- Create: `effects.py`
- Test: `tests/test_effects.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_effects.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import effects


def test_schemas_cover_all_five_types():
    assert set(effects.EFFECT_SCHEMAS) == {'reverb', 'delay', 'chorus', 'distortion', 'filter'}
    for params in effects.EFFECT_SCHEMAS.values():
        assert isinstance(params, list) and params
        for p in params:
            assert 'name' in p and 'default' in p
            assert ('min' in p and 'max' in p) or ('options' in p)  # numeric or enum


def test_default_effect_uses_schema_defaults():
    e = effects.default_effect('delay')
    assert e['type'] == 'delay'
    assert e['enabled'] is True
    for p in effects.EFFECT_SCHEMAS['delay']:
        assert e['params'][p['name']] == p['default']


def test_default_effect_unknown_type_raises():
    try:
        effects.default_effect('nope')
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_resolve_chain_prefers_section_override():
    loop_chain = [effects.default_effect('reverb')]
    override = [effects.default_effect('delay')]
    assert effects.resolve_chain(loop_chain, override) == override
    assert effects.resolve_chain(loop_chain, None) == loop_chain


def test_chain_hash_stable_and_sensitive():
    a = [effects.default_effect('delay')]
    b = [effects.default_effect('delay')]
    assert effects.chain_hash(a) == effects.chain_hash(b)
    b[0]['params']['mix'] = 0.99
    assert effects.chain_hash(a) != effects.chain_hash(b)
    assert effects.chain_hash([]) == effects.chain_hash([])
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_effects.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'effects'`.

- [ ] **Step 3: Implement schemas + helpers**

Create `effects.py`:

```python
"""Audio effect definitions, chain resolution, and offline rendering.

Pure and dependency-light apart from `pedalboard` (imported lazily inside the
render/build functions so schema/resolution helpers work without it).
"""
import hashlib
import json
import numpy as np

# Each param is numeric {name, min, max, default, unit} or enum {name, options, default}.
EFFECT_SCHEMAS = {
    'reverb': [
        {'name': 'room_size', 'min': 0.0, 'max': 1.0, 'default': 0.5, 'unit': ''},
        {'name': 'damping',   'min': 0.0, 'max': 1.0, 'default': 0.5, 'unit': ''},
        {'name': 'wet',       'min': 0.0, 'max': 1.0, 'default': 0.3, 'unit': ''},
    ],
    'delay': [
        {'name': 'time_s',   'min': 0.01, 'max': 2.0, 'default': 0.25, 'unit': 's'},
        {'name': 'feedback', 'min': 0.0,  'max': 0.95, 'default': 0.35, 'unit': ''},
        {'name': 'mix',      'min': 0.0,  'max': 1.0,  'default': 0.4, 'unit': ''},
    ],
    'chorus': [
        {'name': 'rate_hz', 'min': 0.1, 'max': 8.0, 'default': 1.0, 'unit': 'Hz'},
        {'name': 'depth',   'min': 0.0, 'max': 1.0, 'default': 0.25, 'unit': ''},
        {'name': 'mix',     'min': 0.0, 'max': 1.0, 'default': 0.5, 'unit': ''},
    ],
    'distortion': [
        {'name': 'drive_db', 'min': 0.0, 'max': 40.0, 'default': 18.0, 'unit': 'dB'},
    ],
    'filter': [
        {'name': 'mode', 'options': ['LP', 'HP'], 'default': 'LP'},
        {'name': 'cutoff_hz', 'min': 50.0, 'max': 18000.0, 'default': 2000.0, 'unit': 'Hz'},
        {'name': 'resonance', 'min': 0.0, 'max': 1.0, 'default': 0.2, 'unit': ''},
    ],
}


def default_effect(effect_type: str) -> dict:
    """A new effect dict with schema-default params."""
    if effect_type not in EFFECT_SCHEMAS:
        raise ValueError(f"unknown effect type: {effect_type}")
    params = {p['name']: p['default'] for p in EFFECT_SCHEMAS[effect_type]}
    return {'type': effect_type, 'params': params, 'enabled': True}


def resolve_chain(loop_chain: list, section_override) -> list:
    """Section override wins if present (not None), else the loop's default chain."""
    return section_override if section_override is not None else loop_chain


def chain_hash(chain: list) -> str:
    """Stable hash of a chain's audible content (order, types, params, enabled)."""
    payload = [
        {'type': e['type'], 'params': e.get('params', {}), 'enabled': e.get('enabled', True)}
        for e in (chain or [])
    ]
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_effects.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add -f effects.py tests/test_effects.py
git commit -m "feat(fx): effect schemas, defaults, chain resolution and hashing"
```

---

## Task 3: effects.py — build pedalboard chains and render wet buffers (TDD)

**Files:**
- Modify: `effects.py`
- Test: `tests/test_effects.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_effects.py`:

```python
import numpy as np
SR = 44100


def _tone(freq, secs=1.0, amp=0.3):
    t = np.linspace(0, secs, int(SR * secs), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_render_wet_empty_chain_returns_dry_copy():
    dry = _tone(440)
    wet = effects.render_wet(dry, [], SR)
    assert np.array_equal(wet, dry)
    assert wet is not dry            # a copy, not the same object


def test_render_wet_preserves_length():
    dry = _tone(440, 0.5)
    e = effects.default_effect('reverb')
    wet = effects.render_wet(dry, [e], SR)
    assert len(wet) == len(dry)


def test_render_wet_disabled_effect_is_bypassed():
    dry = _tone(440, 0.5)
    e = effects.default_effect('distortion'); e['enabled'] = False
    wet = effects.render_wet(dry, [e], SR)
    assert np.array_equal(wet, dry)


def test_lowpass_filter_attenuates_high_frequency():
    dry = _tone(8000, 0.5)
    f = effects.default_effect('filter')
    f['params'].update({'mode': 'LP', 'cutoff_hz': 500.0})
    wet = effects.render_wet(dry, [f], SR)
    assert float(np.sqrt(np.mean(wet**2))) < float(np.sqrt(np.mean(dry**2))) * 0.6


def test_make_bus_reverb_processes_block_without_error():
    bus = effects.make_bus_reverb({'room_size': 0.6, 'damping': 0.5, 'wet': 0.3})
    block = _tone(440, 0.01)
    out = bus(block, SR, reset=False)
    assert out.shape == block.shape
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_effects.py -k "render_wet or bus" -v`
Expected: FAIL — `AttributeError: module 'effects' has no attribute 'render_wet'`.

- [ ] **Step 3: Implement chain building + rendering**

Append to `effects.py`:

```python
def _make_plugin(effect: dict):
    """Map one effect dict to a pedalboard plugin instance."""
    import pedalboard as pb
    t, p = effect['type'], effect.get('params', {})
    if t == 'reverb':
        return pb.Reverb(room_size=p['room_size'], damping=p['damping'],
                         wet_level=p['wet'], dry_level=1.0 - p['wet'])
    if t == 'delay':
        return pb.Delay(delay_seconds=p['time_s'], feedback=p['feedback'], mix=p['mix'])
    if t == 'chorus':
        return pb.Chorus(rate_hz=p['rate_hz'], depth=p['depth'], mix=p['mix'])
    if t == 'distortion':
        return pb.Distortion(drive_db=p['drive_db'])
    if t == 'filter':
        mode = pb.LadderFilter.Mode.LPF12 if p['mode'] == 'LP' else pb.LadderFilter.Mode.HPF12
        return pb.LadderFilter(mode=mode, cutoff_hz=p['cutoff_hz'], resonance=p['resonance'])
    raise ValueError(f"unknown effect type: {t}")


def make_pedalboard(chain: list):
    """Build a pedalboard.Pedalboard from the enabled effects in order."""
    import pedalboard as pb
    return pb.Pedalboard([_make_plugin(e) for e in chain if e.get('enabled', True)])


def render_wet(dry: np.ndarray, chain: list, sample_rate: int) -> np.ndarray:
    """Render dry through the chain, baking wrapped tails so the loop still seams.

    Tiles dry 3x, processes, and returns the final cycle. Empty/all-disabled
    chain returns an untouched copy of dry.
    """
    active = [e for e in (chain or []) if e.get('enabled', True)]
    if not active or len(dry) == 0:
        return dry.copy()
    board = make_pedalboard(active)
    n = len(dry)
    tiled = np.tile(dry.astype(np.float32), 3)
    processed = np.asarray(board(tiled, sample_rate, reset=True), dtype=np.float32)
    return processed[2 * n:3 * n].copy()


def make_bus_reverb(params: dict):
    """A persistent single-Reverb pedalboard for the live master bus."""
    import pedalboard as pb
    return pb.Pedalboard([pb.Reverb(room_size=params['room_size'], damping=params['damping'],
                                    wet_level=params['wet'], dry_level=1.0 - params['wet'])])
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_effects.py -v`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add -f effects.py tests/test_effects.py
git commit -m "feat(fx): build pedalboard chains and render wet buffers with tail-wrap"
```

---

## Task 4: LoopLayer dry buffer + per-loop chain + bake pipeline (`audio.py`, TDD)

Give each layer an untouched `dry` and a default `fx_chain`; derive the played `buffer` (wet) from dry. Add a wet cache and a setter that re-bakes.

**Files:**
- Modify: `audio.py` — `LoopLayer.__init__`/`to_dict`, `stop_recording`, `_finalize_overdub`, `apply_trim`, `__init__` (cache), new methods; `clear_all`
- Test: `tests/test_effects_engine.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_effects_engine.py`:

```python
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from audio import WebLooper, LoopLayer, LooperState, SAMPLE_RATE
import effects


def _tone(freq, secs=0.5, amp=0.3):
    t = np.linspace(0, secs, int(SAMPLE_RATE * secs), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_new_layer_dry_equals_buffer_and_empty_chain():
    layer = LoopLayer(0, "Master", _tone(440))
    assert np.array_equal(layer.dry, layer.buffer)
    assert layer.fx_chain == []


def test_set_loop_chain_rebakes_wet_from_dry():
    looper = WebLooper()
    looper.layers = [LoopLayer(0, "Master", _tone(8000))]
    looper.master_length = looper.layers[0].length
    looper.state = LooperState.PLAYING
    f = effects.default_effect('filter'); f['params'].update({'mode': 'LP', 'cutoff_hz': 500.0})
    looper.set_loop_chain(0, [f])
    layer = looper.layers[0]
    assert layer.fx_chain[0]['type'] == 'filter'
    # dry preserved; wet differs and is quieter (high tone filtered)
    assert np.array_equal(layer.dry, _tone(8000))
    assert float(np.sqrt(np.mean(layer.buffer**2))) < float(np.sqrt(np.mean(layer.dry**2))) * 0.7


def test_set_empty_chain_restores_dry():
    looper = WebLooper()
    looper.layers = [LoopLayer(0, "Master", _tone(440))]
    looper.master_length = looper.layers[0].length
    looper.set_loop_chain(0, [effects.default_effect('distortion')])
    looper.set_loop_chain(0, [])
    assert np.array_equal(looper.layers[0].buffer, looper.layers[0].dry)


def test_wet_cache_reused_for_identical_chain():
    looper = WebLooper()
    looper.layers = [LoopLayer(0, "Master", _tone(440))]
    looper.master_length = looper.layers[0].length
    chain = [effects.default_effect('reverb')]
    looper.set_loop_chain(0, chain)
    first = looper.layers[0].buffer
    looper.set_loop_chain(0, [effects.default_effect('reverb')])  # identical content
    assert looper.layers[0].buffer is first   # served from cache, same object
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_effects_engine.py -v`
Expected: FAIL — `AttributeError: 'LoopLayer' object has no attribute 'dry'`.

- [ ] **Step 3: Add `dry` + `fx_chain` to `LoopLayer`**

In `audio.py` `LoopLayer.__init__` (after `self.color = ...`, ~line 98), add:

```python
        self.dry = buffer            # untouched source audio
        self.fx_chain = []           # this layer's default effect chain
```

In `LoopLayer.to_dict` return dict, add a `fx_chain` entry:

```python
            'fx_chain': self.fx_chain,
```

- [ ] **Step 4: Add the wet cache in `WebLooper.__init__`**

In `WebLooper.__init__`, just after the Slots/Sections block (after `self.active_section_id = None`), add:

```python
        # Effects
        self.wet_cache = {}          # (dry_hash, chain_hash) -> wet np.ndarray
        self.master_bus = None       # default bus reverb effect dict (None = off)
        self.bus_reverb = None       # live pedalboard.Pedalboard for the master bus
```

- [ ] **Step 5: Add the bake helpers + `set_loop_chain`**

In `audio.py`, add after `_finalize_overdub` (or near the layer methods, before SLOT/SECTION section):

```python
    def _wet_for(self, dry, chain):
        """Return the rendered wet buffer for dry+chain, memoised by content hash."""
        import effects
        key = (hash(dry.tobytes()), effects.chain_hash(chain))
        cached = self.wet_cache.get(key)
        if cached is None:
            cached = effects.render_wet(dry, chain, SAMPLE_RATE)
            self.wet_cache[key] = cached
        return cached

    def _rebake_layer(self, layer, chain):
        """Set a layer's played buffer to the wet render of its dry through chain."""
        layer.buffer = self._wet_for(layer.dry, chain)
        layer.length = len(layer.buffer)

    def set_loop_chain(self, layer_id: int, chain: list) -> bool:
        """Set a layer's default effect chain and re-bake its played buffer."""
        with self.lock:
            if not (0 <= layer_id < len(self.layers)):
                return False
            layer = self.layers[layer_id]
            layer.fx_chain = chain
            self._rebake_layer(layer, chain)
            return True
```

- [ ] **Step 6: Run to verify they pass**

Run: `python -m pytest tests/test_effects_engine.py -v`
Expected: PASS (4 passed).

- [ ] **Step 7: Commit**

```bash
git add -f audio.py tests/test_effects_engine.py
git commit -m "feat(fx): dry buffer + per-loop chain + memoised wet bake"
```

---

## Task 5: Persist chains + expose FX in get_state (`audio.py`, TDD)

**Files:**
- Modify: `audio.py` — `save_session` meta, `load_session` layer rebuild, `get_state`
- Test: `tests/test_effects_engine.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_effects_engine.py`:

```python
def test_get_state_exposes_chain_and_schemas():
    looper = WebLooper()
    looper.layers = [LoopLayer(0, "Master", _tone(440))]
    looper.master_length = looper.layers[0].length
    looper.set_loop_chain(0, [effects.default_effect('delay')])
    st = looper.get_state()
    assert st['layers'][0]['fx_chain'][0]['type'] == 'delay'
    assert set(st['fx']['schemas']) == set(effects.EFFECT_SCHEMAS)


def test_session_roundtrip_preserves_chain(tmp_path, monkeypatch):
    import audio
    monkeypatch.setattr(audio, 'SESSIONS_DIR', tmp_path)
    looper = WebLooper()
    looper.layers = [LoopLayer(0, "Master", _tone(440))]
    looper.master_length = looper.layers[0].length
    looper.state = LooperState.PLAYING
    looper.set_loop_chain(0, [effects.default_effect('chorus')])
    res = looper.save_session("fxtest")
    assert res['success']
    looper2 = WebLooper()
    assert looper2.load_session(res['session_id'])['success']
    assert looper2.layers[0].fx_chain[0]['type'] == 'chorus'
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_effects_engine.py -k "get_state or roundtrip" -v`
Expected: FAIL — `KeyError: 'fx'` / chain not restored.

- [ ] **Step 3: Persist `fx_chain` in `save_session`**

In `save_session` meta `'layers'` list comprehension (`audio.py:604-613`), add `fx_chain` to each layer dict:

```python
                        'is_playing': l.is_playing,
                        'fx_chain': l.fx_chain,
```

- [ ] **Step 4: Restore `fx_chain` in `load_session`**

In `load_session`, where each `LoopLayer` is rebuilt from `layer_meta` (search for `layer.is_playing = layer_meta.get('is_playing', True)`), add after it:

```python
                layer.fx_chain = layer_meta.get('fx_chain', []) or []
                self._rebake_layer(layer, layer.fx_chain)
```

- [ ] **Step 5: Add the `fx` block to `get_state`**

In `get_state`, in the returned dict after the `'sections'` block, add:

```python
            'fx': {
                'schemas': __import__('effects').EFFECT_SCHEMAS,
                'master_bus': self.master_bus,
            },
```

(`fx_chain` per layer is already included via `LoopLayer.to_dict`.)

- [ ] **Step 6: Run to verify they pass**

Run: `python -m pytest tests/test_effects_engine.py -v`
Expected: PASS (6 passed).

- [ ] **Step 7: Commit**

```bash
git add -f audio.py tests/test_effects_engine.py
git commit -m "feat(fx): persist loop chains; expose fx schemas in get_state"
```

---

## Task 6: Per-section chain overrides (`audio.py`, TDD)

Sections may override a loop's chain; applying a section sets each layer's wet buffer from the resolved chain.

**Files:**
- Modify: `audio.py` — `_apply_section`, new `set_section_override`/`clear_section_override`, `_sections_from_meta` (carry `fx_overrides`), `save_session` (sections already serialised — include overrides)
- Test: `tests/test_effects_engine.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_effects_engine.py`:

```python
def _two_layer_looper():
    looper = WebLooper()
    looper.layers = [LoopLayer(0, "Master", _tone(440)), LoopLayer(1, "L1", _tone(8000))]
    looper.master_length = looper.layers[0].length
    looper.state = LooperState.PLAYING
    return looper


def test_apply_section_uses_override_else_default():
    looper = _two_layer_looper()
    # loop1 default = distortion; section overrides loop1 to a lowpass
    looper.set_loop_chain(1, [effects.default_effect('distortion')])
    sec = looper.add_section()
    looper.set_section_loops(sec['id'], [0, 1])
    f = effects.default_effect('filter'); f['params'].update({'mode': 'LP', 'cutoff_hz': 500.0})
    looper.set_section_override(sec['id'], 1, [f])
    looper._apply_section(sec)
    # layer1 wet should equal the override (lowpassed dry), not the default (distorted)
    expected = looper._wet_for(looper.layers[1].dry, [f])
    assert np.array_equal(looper.layers[1].buffer, expected)


def test_clear_section_override_reverts_to_default():
    looper = _two_layer_looper()
    sec = looper.add_section(); looper.set_section_loops(sec['id'], [0, 1])
    looper.set_section_override(sec['id'], 1, [effects.default_effect('reverb')])
    assert sec['fx_overrides'].get(1) is not None
    looper.clear_section_override(sec['id'], 1)
    assert 1 not in sec['fx_overrides']
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_effects_engine.py -k "override" -v`
Expected: FAIL — `AttributeError: 'WebLooper' object has no attribute 'set_section_override'`.

- [ ] **Step 3: Ensure sections always have `fx_overrides`**

In `add_section` (`audio.py:532`), include the field:

```python
            section = {'id': self._next_section_id, 'loop_ids': [], 'fx_overrides': {}}
```

In `_sections_from_meta`, when reading stored sections, carry overrides (keys come back as strings from JSON — coerce to int). Replace the `sections.append({...})` for the stored-sections branch with:

```python
                sections.append({
                    'id': int(s.get('id', len(sections) + 1)),
                    'loop_ids': [int(i) for i in (s.get('loop_ids') or [])],
                    'fx_overrides': {int(k): v for k, v in (s.get('fx_overrides') or {}).items()},
                })
```

And in the legacy-scenes branch append `'fx_overrides': {}` too:

```python
            sections.append({'id': len(sections) + 1, 'loop_ids': active, 'fx_overrides': {}})
```

- [ ] **Step 4: Serialise overrides in `save_session`**

In `save_session` meta, replace the `'sections'` line (`audio.py:614`) with:

```python
                'sections': [{'id': s['id'], 'loop_ids': list(s['loop_ids']),
                              'fx_overrides': s.get('fx_overrides', {})} for s in self.sections],
```

- [ ] **Step 5: Resolve chains in `_apply_section` + add the setters**

Replace `_apply_section` (`audio.py:561-566`) with:

```python
    def _apply_section(self, section: dict):
        """Activate the section's loops and bake each to its resolved chain. Holds lock."""
        import effects
        ids = set(section['loop_ids'])
        overrides = section.get('fx_overrides', {})
        for layer in self.layers:
            layer.is_playing = layer.id in ids
            chain = effects.resolve_chain(layer.fx_chain, overrides.get(layer.id))
            self._rebake_layer(layer, chain)
        self.active_section_id = section['id']
```

Add after `set_section_loops` (`audio.py:551`):

```python
    def set_section_override(self, section_id: int, loop_id: int, chain: list) -> bool:
        """Override a loop's chain for one section."""
        with self.lock:
            section = next((s for s in self.sections if s['id'] == section_id), None)
            if section is None:
                return False
            section.setdefault('fx_overrides', {})[loop_id] = chain
            if self.active_section_id == section_id:
                layer = next((l for l in self.layers if l.id == loop_id), None)
                if layer is not None:
                    self._rebake_layer(layer, chain)
            return True

    def clear_section_override(self, section_id: int, loop_id: int) -> bool:
        """Revert a section override back to the loop default."""
        with self.lock:
            section = next((s for s in self.sections if s['id'] == section_id), None)
            if section is None:
                return False
            section.get('fx_overrides', {}).pop(loop_id, None)
            if self.active_section_id == section_id:
                layer = next((l for l in self.layers if l.id == loop_id), None)
                if layer is not None:
                    self._rebake_layer(layer, layer.fx_chain)
            return True
```

- [ ] **Step 6: Run to verify they pass**

Run: `python -m pytest tests/test_effects_engine.py -v`
Expected: PASS (8 passed).

- [ ] **Step 7: Commit**

```bash
git add -f audio.py tests/test_effects_engine.py
git commit -m "feat(fx): per-section chain overrides resolved on section apply"
```

---

## Task 7: Live master-bus reverb (`audio.py`)

Add a single live reverb on the summed mix, configured from the master bus (or section bus override), processed each callback block.

**Files:**
- Modify: `audio.py` — `__init__` already has `master_bus`/`bus_reverb`; add `set_bus`, `_refresh_bus_reverb`; `_apply_section` bus selection; callback block processing; `clear_all` reset
- Test: `tests/test_effects_engine.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_effects_engine.py`:

```python
def test_set_bus_builds_live_reverb_and_clear_removes_it():
    looper = WebLooper()
    looper.set_bus(None, effects.default_effect('reverb'))   # None section = master bus
    assert looper.master_bus is not None
    assert looper.bus_reverb is not None
    block = _tone(440, 0.01)
    out = looper.bus_reverb(block, SAMPLE_RATE, reset=False)
    assert out.shape == block.shape
    looper.set_bus(None, None)
    assert looper.master_bus is None
    assert looper.bus_reverb is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_effects_engine.py -k "set_bus" -v`
Expected: FAIL — `AttributeError: 'WebLooper' object has no attribute 'set_bus'`.

- [ ] **Step 3: Add bus setter + refresh helper**

In `audio.py`, add near the FX methods:

```python
    def _refresh_bus_reverb(self, effect):
        """(Re)build the live bus reverb pedalboard from an effect dict (or clear it)."""
        import effects as fx
        if effect and effect.get('enabled', True) and effect.get('type') == 'reverb':
            self.bus_reverb = fx.make_bus_reverb(effect['params'])
        else:
            self.bus_reverb = None

    def set_bus(self, section_id, effect) -> bool:
        """Set the master bus (section_id None) or a section's bus override; rebuild live reverb."""
        with self.lock:
            if section_id is None:
                self.master_bus = effect
            else:
                section = next((s for s in self.sections if s['id'] == section_id), None)
                if section is None:
                    return False
                section['bus'] = effect
            # Refresh the live instance for the currently effective bus
            active = next((s for s in self.sections if s['id'] == self.active_section_id), None)
            effective = (active.get('bus') if active and active.get('bus') is not None
                         else self.master_bus)
            self._refresh_bus_reverb(effective)
            return True
```

- [ ] **Step 4: Select the bus on section apply**

At the end of `_apply_section` (after the loop), add:

```python
        effective_bus = section.get('bus') if section.get('bus') is not None else self.master_bus
        self._refresh_bus_reverb(effective_bus)
```

- [ ] **Step 5: Process the bus in the callback**

In `audio_callback`, replace the line `output += loop_output * self.master_volume` (`audio.py:278`) with:

```python
                        if self.bus_reverb is not None:
                            loop_output = np.asarray(
                                self.bus_reverb(loop_output, SAMPLE_RATE, reset=False),
                                dtype=np.float32)
                        output += loop_output * self.master_volume
```

- [ ] **Step 6: Reset bus in `clear_all`**

In `clear_all` (`audio.py:514`), where section state is reset, also add:

```python
            self.master_bus = None
            self.bus_reverb = None
            self.wet_cache = {}
```

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_effects_engine.py -v`
Expected: PASS (9 passed).

- [ ] **Step 8: Commit**

```bash
git add -f audio.py tests/test_effects_engine.py
git commit -m "feat(fx): live master-bus reverb processed in the audio callback"
```

---

## Task 8: Include effects in export (`audio.py`)

Wet buffers already carry insert FX (export mixes `layer.buffer`). Apply the effective bus reverb offline over the mixed result.

**Files:**
- Modify: `audio.py` — `export_mixed`
- Test: `tests/test_effects_engine.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_effects_engine.py`:

```python
def test_apply_bus_offline_changes_mix_when_bus_set():
    looper = WebLooper()
    looper.set_bus(None, effects.default_effect('reverb'))
    mix = _tone(440, 0.3)
    out = looper._apply_bus_offline(mix)
    assert out.shape == mix.shape
    assert not np.array_equal(out, mix)          # reverb altered it
    looper.set_bus(None, None)
    assert np.array_equal(looper._apply_bus_offline(mix), mix)  # no bus = unchanged
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_effects_engine.py -k "bus_offline" -v`
Expected: FAIL — `AttributeError: ... no attribute '_apply_bus_offline'`.

- [ ] **Step 3: Add the offline bus helper and call it in export**

In `audio.py`, add near the FX methods:

```python
    def _apply_bus_offline(self, mixed: np.ndarray) -> np.ndarray:
        """Apply the currently effective bus reverb to a finished mix (for export)."""
        import effects as fx
        active = next((s for s in self.sections if s['id'] == self.active_section_id), None)
        effective = (active.get('bus') if active and active.get('bus') is not None
                     else self.master_bus)
        if not (effective and effective.get('enabled', True) and effective.get('type') == 'reverb'):
            return mixed
        board = fx.make_bus_reverb(effective['params'])
        return np.asarray(board(mixed.astype(np.float32), SAMPLE_RATE, reset=True), dtype=np.float32)
```

In `export_mixed`, the layers are summed into a local `mixed` and then master volume is applied. Find:

```python
            # Apply master volume
            mixed *= master_vol
```

and insert the bus reverb immediately *before* it (so the bus processes the summed mix, matching live playback order — bus then master volume):

```python
            # Apply bus reverb (matches live playback) before master volume
            mixed = self._apply_bus_offline(mixed)
            # Apply master volume
            mixed *= master_vol
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_effects_engine.py -v`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add -f audio.py tests/test_effects_engine.py
git commit -m "feat(fx): bake bus reverb into mixed export"
```

---

## Task 9: Socket commands (`routes.py`)

**Files:**
- Modify: `routes.py` — `handle_command`

- [ ] **Step 1: Add FX command branches**

In `routes.py` `handle_command`, after the section command branches (`launch_section`) and before the final broadcast, add:

```python
    elif command == 'fx_set_loop_chain':
        looper.set_loop_chain(data.get('layer_id'), data.get('chain', []))
    elif command == 'fx_set_section_override':
        looper.set_section_override(data.get('section_id'), data.get('layer_id'), data.get('chain', []))
    elif command == 'fx_clear_section_override':
        looper.clear_section_override(data.get('section_id'), data.get('layer_id'))
    elif command == 'fx_set_bus':
        looper.set_bus(data.get('section_id'), data.get('effect'))
```

- [ ] **Step 2: Verify the module imports**

Run: `python -c "import routes; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add -f routes.py
git commit -m "feat(fx): socket commands for loop chains, section overrides, and bus"
```

---

## Task 10: FX side-panel tab UI (`index.html`, `style.css`, `app.js`)

Build the `≈` tab validated in brainstorming: loop selector, Default/Section scope toggle + override banner, chain chips (add/remove/reorder), per-effect param controls, bus reverb. No JS unit harness — verify with `node --check` + manual.

**Files:**
- Modify: `templates/index.html` (side strip + `#panelFx`), `static/style.css` (append), `static/app.js` (render + handlers; call from update handler and `setSidePanel`)

- [ ] **Step 1: Add the side-strip button**

In `templates/index.html` `.side-strip`, replace the remaining stub:

```html
            <button class="side-icon side-icon-stub" title="Art (coming soon)">✦</button>
```

with:

```html
            <button class="side-icon" id="sideBtnFx" data-panel="fx"
                    onclick="setSidePanel('fx')" title="Effects">≈</button>
```

- [ ] **Step 2: Add the `#panelFx` markup**

In `templates/index.html`, immediately after the `#panelSections` panel's closing `</div>` (inside `#sideContent`), add:

```html
            <!-- FX panel -->
            <div class="side-panel" id="panelFx">
                <div class="panel-header accent-primary"><span>FX</span></div>
                <div class="fx-pick">
                    <label class="fx-pick-label">LOOP</label>
                    <select id="fxLoopSelect" class="fx-select" onchange="fxOnLoopChange()"></select>
                </div>
                <div class="fx-pick">
                    <label class="fx-pick-label">SCOPE</label>
                    <div class="fx-scope" id="fxScope"></div>
                </div>
                <div class="fx-override-banner" id="fxOverrideBanner" style="display:none">
                    <span>⤷ Overrides the loop default here</span>
                    <button class="fx-revert" onclick="fxRevertOverride()">↺ revert</button>
                </div>
                <div class="fx-chain" id="fxChain"></div>
                <div class="fx-params" id="fxParams"></div>
                <div class="fx-bus">
                    <div class="fx-bus-head">BUS — reverb (all loops)</div>
                    <div id="fxBus"></div>
                </div>
            </div>
```

- [ ] **Step 3: Append CSS**

Append to `static/style.css`:

```css
/* ── FX panel ── */
.fx-pick { display:flex; align-items:center; gap:8px; padding:7px 10px; border-bottom:1px solid var(--border); }
.fx-pick-label { font-size:9px; letter-spacing:.5px; color:var(--text-muted); width:42px; }
.fx-select { flex:1; background:var(--surface); border:1px solid var(--border); color:var(--text); font-size:11px; padding:4px 6px; border-radius:5px; }
.fx-scope { display:flex; border:1px solid var(--border); border-radius:6px; overflow:hidden; }
.fx-scope button { flex:1; background:var(--surface); border:none; color:var(--text-muted); font-size:11px; padding:5px 8px; cursor:pointer; }
.fx-scope button.on { background:#7cf; color:#111; }
.fx-override-banner { display:flex; justify-content:space-between; align-items:center; gap:8px; margin:7px 10px; padding:5px 8px; font-size:10px; color:#cbb6f0; background:#2a2433; border:1px solid #5a4a7a; border-radius:6px; }
.fx-revert { background:none; border:none; color:#cbb6f0; cursor:pointer; font-size:10px; }
.fx-chain { display:flex; flex-wrap:wrap; gap:6px; padding:9px 10px; border-bottom:1px solid var(--border); align-items:center; }
.fx-chip { display:inline-flex; align-items:center; gap:6px; background:#2a2433; border:1px solid #5a4a7a; border-radius:6px; padding:5px 8px; color:#cbb6f0; font-size:11px; cursor:grab; }
.fx-chip.active { background:#5a4a7a; color:#fff; }
.fx-chip .fx-x { cursor:pointer; }
.fx-add { border:1px dashed var(--border); border-radius:6px; padding:5px 8px; color:#7cf; font-size:11px; background:none; cursor:pointer; }
.fx-params { padding:9px 10px; }
.fx-knob { display:flex; align-items:center; gap:8px; margin-bottom:7px; }
.fx-knob label { width:64px; font-size:10px; color:var(--text-muted); }
.fx-knob input[type=range] { flex:1; }
.fx-knob .fx-val { width:48px; text-align:right; font-size:10px; color:var(--text); }
.fx-bus { padding:9px 10px; border-top:1px solid var(--border); }
.fx-bus-head { font-size:9px; letter-spacing:.5px; color:var(--text-muted); margin-bottom:6px; }
```

- [ ] **Step 4: Add the JS (state + render + handlers)**

In `static/app.js`, add after the SECTIONS block:

```javascript
        // =================================================================
        // FX
        // =================================================================
        let fxLoopId = 0;        // which loop's chain we're editing
        let fxScopeSection = false;  // false = loop default, true = active section override
        let fxActiveIdx = 0;     // selected effect index in the chain

        function fxActiveSection() {
            const sid = serverState.sections?.active_id;
            return (serverState.sections?.list || []).find(s => s.id === sid) || null;
        }
        function fxCurrentChain() {
            const layer = (serverState.layers || []).find(l => l.id === fxLoopId);
            const base = (layer && layer.fx_chain) ? layer.fx_chain : [];
            if (fxScopeSection) {
                const sec = fxActiveSection();
                const ov = sec && sec.fx_overrides ? sec.fx_overrides[fxLoopId] : undefined;
                return ov !== undefined && ov !== null ? ov : base;
            }
            return base;
        }
        function fxCommitChain(chain) {
            if (fxScopeSection) {
                const sec = fxActiveSection();
                if (sec) sendCommand('fx_set_section_override', { section_id: sec.id, layer_id: fxLoopId, chain });
            } else {
                sendCommand('fx_set_loop_chain', { layer_id: fxLoopId, chain });
            }
        }
        function fxOnLoopChange() { fxLoopId = parseInt(document.getElementById('fxLoopSelect').value, 10) || 0; fxActiveIdx = 0; renderFx(); }
        function fxSetScope(isSection) { fxScopeSection = isSection; fxActiveIdx = 0; renderFx(); }
        function fxRevertOverride() {
            const sec = fxActiveSection();
            if (sec) sendCommand('fx_clear_section_override', { section_id: sec.id, layer_id: fxLoopId });
        }
        function fxAddEffect() {
            const type = prompt('Add effect: reverb, delay, chorus, distortion, filter');
            if (!type || !serverState.fx?.schemas[type]) return;
            const params = {}; serverState.fx.schemas[type].forEach(p => params[p.name] = p.default);
            fxCommitChain([...fxCurrentChain(), { type, params, enabled: true }]);
        }
        function fxRemoveEffect(idx) { const c = fxCurrentChain().slice(); c.splice(idx, 1); fxActiveIdx = 0; fxCommitChain(c); }
        function fxMoveEffect(idx, dir) {
            const c = fxCurrentChain().slice(); const j = idx + dir;
            if (j < 0 || j >= c.length) return;
            [c[idx], c[j]] = [c[j], c[idx]]; fxActiveIdx = j; fxCommitChain(c);
        }
        function fxSetParam(idx, name, value) {
            const c = fxCurrentChain().map(e => ({ ...e, params: { ...e.params } }));
            c[idx].params[name] = value; fxCommitChain(c);
        }
        function fxSetBusParam(name, value) {
            const cur = serverState.fx?.master_bus
                || { type: 'reverb', params: Object.fromEntries((serverState.fx?.schemas?.reverb || []).map(p => [p.name, p.default])), enabled: true };
            const effect = { ...cur, params: { ...cur.params, [name]: value } };
            sendCommand('fx_set_bus', { section_id: null, effect });
        }
        function fxToggleBus(on) {
            if (!on) { sendCommand('fx_set_bus', { section_id: null, effect: null }); return; }
            fxSetBusParam('room_size', (serverState.fx?.schemas?.reverb?.[0]?.default) ?? 0.5);
        }

        function renderFx() {
            const panel = document.getElementById('panelFx');
            if (!panel || !panel.classList.contains('active')) return;
            const layers = serverState.layers || [];
            if (!layers.some(l => l.id === fxLoopId)) fxLoopId = layers.length ? layers[0].id : 0;

            // Loop selector
            document.getElementById('fxLoopSelect').innerHTML = layers.map(l =>
                `<option value="${l.id}" ${l.id === fxLoopId ? 'selected' : ''}>${l.name}</option>`).join('');

            // Scope toggle
            const sec = fxActiveSection();
            document.getElementById('fxScope').innerHTML =
                `<button class="${!fxScopeSection ? 'on' : ''}" onclick="fxSetScope(false)">Default</button>
                 <button class="${fxScopeSection ? 'on' : ''}" onclick="fxSetScope(true)" ${sec ? '' : 'disabled'}>Section${sec ? ': ' + (sec.name || sec.id) : ''}</button>`;

            // Override banner
            const isOverriding = fxScopeSection && sec && sec.fx_overrides && sec.fx_overrides[fxLoopId] != null;
            document.getElementById('fxOverrideBanner').style.display = isOverriding ? 'flex' : 'none';

            // Chain chips
            const chain = fxCurrentChain();
            if (fxActiveIdx >= chain.length) fxActiveIdx = Math.max(0, chain.length - 1);
            document.getElementById('fxChain').innerHTML = chain.map((e, i) => `
                <span class="fx-chip ${i === fxActiveIdx ? 'active' : ''}" onclick="fxActiveIdx=${i};renderFx()">
                    <span onclick="event.stopPropagation();fxMoveEffect(${i},-1)">‹</span>${e.type}
                    <span onclick="event.stopPropagation();fxMoveEffect(${i},1)">›</span>
                    <span class="fx-x" onclick="event.stopPropagation();fxRemoveEffect(${i})">✕</span>
                </span>`).join('') + `<button class="fx-add" onclick="fxAddEffect()">＋ add</button>`;

            // Params for selected effect
            const schemas = serverState.fx?.schemas || {};
            const sel = chain[fxActiveIdx];
            document.getElementById('fxParams').innerHTML = (sel && schemas[sel.type]) ? schemas[sel.type].map(p => {
                const v = sel.params[p.name];
                if (p.options) {
                    return `<div class="fx-knob"><label>${p.name}</label>
                        <select onchange="fxSetParam(${fxActiveIdx},'${p.name}',this.value)">
                          ${p.options.map(o => `<option ${o === v ? 'selected' : ''}>${o}</option>`).join('')}
                        </select></div>`;
                }
                return `<div class="fx-knob"><label>${p.name}</label>
                    <input type="range" min="${p.min}" max="${p.max}" step="${(p.max - p.min) / 100}" value="${v}"
                           oninput="this.nextElementSibling.textContent=(+this.value).toFixed(2)"
                           onchange="fxSetParam(${fxActiveIdx},'${p.name}',parseFloat(this.value))">
                    <span class="fx-val">${(+v).toFixed(2)}</span></div>`;
            }).join('') : '<div style="font-size:10px;color:var(--text-muted)">No effect selected</div>';

            // Bus
            const bus = serverState.fx?.master_bus;
            const busOn = !!bus;
            const rp = (schemas.reverb || []);
            document.getElementById('fxBus').innerHTML =
                `<div class="fx-knob"><label>enabled</label>
                   <input type="checkbox" ${busOn ? 'checked' : ''} onchange="fxToggleBus(this.checked)"></div>` +
                (busOn ? rp.map(p => {
                    const v = bus.params[p.name];
                    return `<div class="fx-knob"><label>${p.name}</label>
                        <input type="range" min="${p.min}" max="${p.max}" step="${(p.max - p.min) / 100}" value="${v}"
                               oninput="this.nextElementSibling.textContent=(+this.value).toFixed(2)"
                               onchange="fxSetBusParam('${p.name}',parseFloat(this.value))">
                        <span class="fx-val">${(+v).toFixed(2)}</span></div>`;
                }).join('') : '');
        }
```

- [ ] **Step 5: Wire `renderFx` into the update handler and `setSidePanel`**

In `static/app.js`, after the `// --- Sections ---` / `renderSections();` call in the update handler, add:

```javascript
            // --- FX ---
            renderFx();
```

In `setSidePanel`, after `if (name === 'sections') renderSections();`, add:

```javascript
            if (name === 'fx') renderFx();
```

- [ ] **Step 6: Syntax check + manual verification**

Run: `node --check static/app.js && echo JS_OK` (skip if node unavailable).
Manual (on the Pi, after recording ≥1 loop): open the `≈` FX tab → pick a loop → `＋ add` an effect (e.g. distortion) → hear it after the next loop cycle → tweak a slider → reorder/remove → launch a section, flip Scope to "Section", add a different effect → override banner appears → `↺ revert` restores default → enable Bus reverb → hear it on the whole mix → export and confirm effects are present.

- [ ] **Step 7: Commit**

```bash
git add -f templates/index.html static/style.css static/app.js
git commit -m "feat(fx): FX side-panel tab with chains, section scope, and bus"
```

---

## Self-Review Notes

- **Spec coverage:** feasibility gate (T1); effects.py schemas/resolution/hash (T2) + build/render (T3); dry/wet + per-loop chain + bake/cache (T4); persistence + get_state schemas (T5); per-section overrides (T6); live bus reverb (T7); export bus (T8); socket commands (T9); FX tab UI (T10). Baked inserts, live bus, model-C resolution, tail-wrap, non-destructive dry, export — all covered.
- **Type consistency:** `Effect = {type, params, enabled}`; chain = `list[Effect]`; `set_loop_chain`/`set_section_override`/`clear_section_override`/`set_bus`/`_apply_bus_offline`/`_rebake_layer`/`_wet_for`; `effects.render_wet/make_pedalboard/make_bus_reverb/resolve_chain/chain_hash/default_effect/EFFECT_SCHEMAS`; state keys `layers[].fx_chain`, `fx.schemas`, `fx.master_bus`, `sections[].fx_overrides` — consistent across tasks.
- **Risks:** pedalboard ARM install/perf (T1 gate, hand-rolled fallback); buffer swap on edit happens via re-bake (applied immediately here — if mid-cycle clicks are heard, defer the swap to the next loop wrap using the existing `pending_section` mechanism); reconfiguring `bus_reverb` mid-stream may click (acceptable; could crossfade later); `export_mixed` mixed-buffer variable name must be confirmed when editing T8.
