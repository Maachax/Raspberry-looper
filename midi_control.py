"""MPK Mini MIDI control surface: triggers, bindings, dispatch, hot-plug."""
import json
import threading
import time

from config import CONFIG_PATH


def normalize(msg):
    """mido Message -> (trigger_str, value) or None for non-control events."""
    if msg.type == 'program_change':
        return f"pc:{msg.channel}:{msg.program}", None
    if msg.type == 'control_change':
        return f"cc:{msg.channel}:{msg.control}", msg.value
    if msg.type == 'note_on' and msg.velocity > 0:
        return f"note:{msg.channel}:{msg.note}", msg.velocity
    return None


# action_id -> (mode, human label). Mode 'global' works in every mode.
ACTIONS = {
    'record_toggle': ('global', 'Record / Overdub'),
    'tap_tempo': ('global', 'Tap tempo'),
    **{f'launch_section_{i}': ('play', f'Launch section {i}') for i in range(1, 9)},
    **{f'loop_volume_{i}': ('play', f'Loop {i} volume') for i in range(1, 9)},
}

_PAD_A_READING_ORDER = [4, 5, 6, 7, 0, 1, 2, 3]  # PC numbers, top-left -> bottom-right

DEFAULT_BINDINGS = {
    'global': {
        'pc:9:12': 'record_toggle',   # bank B pad 1 (top-left)
        'pc:9:13': 'tap_tempo',       # bank B pad 2
    },
    'play': {
        **{f'pc:9:{p}': f'launch_section_{i + 1}'
           for i, p in enumerate(_PAD_A_READING_ORDER)},
        **{f'cc:0:{70 + i}': f'loop_volume_{i + 1}' for i in range(8)},
    },
}


def load_bindings(path=None):
    """Saved bindings from config if valid, else a deep copy of defaults."""
    path = path or CONFIG_PATH
    try:
        saved = json.loads(path.read_text())['midi']['bindings']
        if (isinstance(saved, dict)
                and saved
                and all(isinstance(m, dict) for m in saved.values())):
            return {mode: {t: a for t, a in m.items() if a in ACTIONS}
                    for mode, m in saved.items()}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {mode: dict(m) for mode, m in DEFAULT_BINDINGS.items()}


def save_bindings(bindings, path=None):
    """Persist bindings under the 'midi' config key, preserving other keys."""
    path = path or CONFIG_PATH
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data.setdefault('midi', {})['bindings'] = bindings
    path.write_text(json.dumps(data, indent=2))
