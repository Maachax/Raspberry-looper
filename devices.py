"""Audio device persistence and name-based resolution for headless startup."""
import json

from config import CONFIG_PATH


def load_saved_device_name(path=None):
    """Return the remembered device name, or None if unset/corrupt."""
    path = path or CONFIG_PATH
    try:
        data = json.loads(path.read_text())
        return data.get('device_name') if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def save_device_name(name, path=None):
    """Remember the chosen device name, preserving other config keys."""
    path = path or CONFIG_PATH
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data['device_name'] = name
    path.write_text(json.dumps(data, indent=2))


def _is_valid(dev):
    return dev['max_input_channels'] > 0 and dev['max_output_channels'] > 0


def find_device_by_name(devices, name):
    """Index of the input+output device with this exact name, else None."""
    if name is None:
        return None
    for i, dev in enumerate(devices):
        if dev['name'] == name and _is_valid(dev):
            return i
    return None


def first_valid_device(devices):
    """Index of the first device with both input and output channels."""
    for i, dev in enumerate(devices):
        if _is_valid(dev):
            return i
    return None


def resolve_device(devices, saved_name):
    """Headless resolution: saved name match, else first valid, else None.

    Never persists the fallback — the preferred interface may just be
    unplugged right now.
    """
    idx = find_device_by_name(devices, saved_name)
    if idx is not None:
        return idx
    return first_valid_device(devices)
