import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import mido

from midi_control import normalize


def test_normalize_program_change():
    msg = mido.Message('program_change', channel=9, program=4)
    assert normalize(msg) == ('pc:9:4', None)


def test_normalize_control_change():
    msg = mido.Message('control_change', channel=0, control=70, value=99)
    assert normalize(msg) == ('cc:0:70', 99)


def test_normalize_note_on():
    msg = mido.Message('note_on', channel=0, note=48, velocity=64)
    assert normalize(msg) == ('note:0:48', 64)


def test_normalize_ignores_note_off_and_zero_velocity():
    assert normalize(mido.Message('note_off', channel=0, note=48)) is None
    assert normalize(mido.Message('note_on', channel=0, note=48, velocity=0)) is None


def test_normalize_ignores_other_messages():
    assert normalize(mido.Message('pitchwheel', channel=0, pitch=100)) is None


from midi_control import (ACTIONS, DEFAULT_BINDINGS,
                          load_bindings, save_bindings)


def test_default_bindings_pads_reading_order():
    play = DEFAULT_BINDINGS['play']
    # top row of bank A (PC 4-7) = sections 1-4, bottom row (PC 0-3) = 5-8
    assert play['pc:9:4'] == 'launch_section_1'
    assert play['pc:9:7'] == 'launch_section_4'
    assert play['pc:9:0'] == 'launch_section_5'
    assert play['pc:9:3'] == 'launch_section_8'


def test_default_bindings_knobs_and_globals():
    assert DEFAULT_BINDINGS['play']['cc:0:70'] == 'loop_volume_1'
    assert DEFAULT_BINDINGS['play']['cc:0:77'] == 'loop_volume_8'
    assert DEFAULT_BINDINGS['global']['pc:9:12'] == 'record_toggle'
    assert DEFAULT_BINDINGS['global']['pc:9:13'] == 'tap_tempo'


def test_every_default_binding_targets_a_known_action():
    for mode_map in DEFAULT_BINDINGS.values():
        for action in mode_map.values():
            assert action in ACTIONS


def test_load_bindings_missing_file_returns_defaults(tmp_path):
    b = load_bindings(path=tmp_path / 'nope.json')
    assert b == DEFAULT_BINDINGS
    assert b is not DEFAULT_BINDINGS  # deep copy, safe to mutate


def test_load_bindings_corrupt_returns_defaults(tmp_path):
    cfg = tmp_path / '_config.json'
    cfg.write_text('{broken')
    assert load_bindings(path=cfg) == DEFAULT_BINDINGS
    cfg.write_text(json.dumps({'midi': {'bindings': 'nonsense'}}))
    assert load_bindings(path=cfg) == DEFAULT_BINDINGS


def test_save_and_load_roundtrip_preserves_other_keys(tmp_path):
    cfg = tmp_path / '_config.json'
    cfg.write_text(json.dumps({'device_name': 'USB Audio CODEC'}))
    custom = {m: dict(v) for m, v in DEFAULT_BINDINGS.items()}
    custom['global']['note:0:60'] = 'record_toggle'
    save_bindings(custom, path=cfg)
    assert load_bindings(path=cfg)['global']['note:0:60'] == 'record_toggle'
    assert json.loads(cfg.read_text())['device_name'] == 'USB Audio CODEC'


def test_load_bindings_drops_unknown_actions(tmp_path):
    cfg = tmp_path / '_config.json'
    bad = {m: dict(v) for m, v in DEFAULT_BINDINGS.items()}
    bad['play']['cc:0:99'] = 'fly_to_the_moon'
    cfg.write_text(json.dumps({'midi': {'bindings': bad}}))
    loaded = load_bindings(path=cfg)
    assert 'cc:0:99' not in loaded['play']
