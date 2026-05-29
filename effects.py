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
