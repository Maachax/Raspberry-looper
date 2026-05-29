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
