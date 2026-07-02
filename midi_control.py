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
