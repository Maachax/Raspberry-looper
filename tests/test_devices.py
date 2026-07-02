import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from devices import (load_saved_device_name, save_device_name,
                     find_device_by_name, resolve_device)


def dev(name, ins=1, outs=2):
    return {'name': name, 'max_input_channels': ins, 'max_output_channels': outs}


DEVICES = [
    dev('bcm2835 Headphones', ins=0),        # output-only, never valid
    dev('USB Audio CODEC'),                   # first valid
    dev('Scarlett 2i2 USB'),                  # second valid
]


def test_find_device_by_name_returns_index():
    assert find_device_by_name(DEVICES, 'Scarlett 2i2 USB') == 2


def test_find_device_by_name_missing_returns_none():
    assert find_device_by_name(DEVICES, 'Unplugged Interface') is None


def test_find_device_by_name_none_returns_none():
    assert find_device_by_name(DEVICES, None) is None


def test_find_device_by_name_ignores_invalid_devices():
    # name matches but device has no inputs -> not usable
    assert find_device_by_name(DEVICES, 'bcm2835 Headphones') is None


def test_resolve_prefers_saved_name():
    assert resolve_device(DEVICES, 'Scarlett 2i2 USB') == 2


def test_resolve_falls_back_to_first_valid():
    assert resolve_device(DEVICES, 'Unplugged Interface') == 1
    assert resolve_device(DEVICES, None) == 1


def test_resolve_no_valid_devices_returns_none():
    assert resolve_device([dev('hdmi', ins=0)], 'anything') is None


def test_save_and_load_roundtrip(tmp_path):
    cfg = tmp_path / '_config.json'
    save_device_name('USB Audio CODEC', path=cfg)
    assert load_saved_device_name(path=cfg) == 'USB Audio CODEC'


def test_load_missing_file_returns_none(tmp_path):
    assert load_saved_device_name(path=tmp_path / 'nope.json') is None


def test_load_corrupt_file_returns_none(tmp_path):
    cfg = tmp_path / '_config.json'
    cfg.write_text('{not json')
    assert load_saved_device_name(path=cfg) is None


def test_save_preserves_other_keys(tmp_path):
    cfg = tmp_path / '_config.json'
    cfg.write_text(json.dumps({'other': 42}))
    save_device_name('X', path=cfg)
    data = json.loads(cfg.read_text())
    assert data == {'other': 42, 'device_name': 'X'}
