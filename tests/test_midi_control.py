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


from midi_control import MidiController


class FakeLooper:
    def __init__(self):
        self.sections = [{'id': 11}, {'id': 22}]
        self.launched = []
        self.volumes = {}
        self.bpms = []
        self.calls = []
        self._state_value = 'idle'

    @property
    def state(self):
        class S: value = self._state_value
        return S

    def launch_section(self, section_id):
        self.launched.append(section_id)

    def set_layer_volume(self, idx, vol):
        self.volumes[idx] = vol

    def set_bpm(self, bpm):
        self.bpms.append(bpm)

    def start_recording(self): self.calls.append('start_recording')
    def stop_recording(self): self.calls.append('stop_recording')
    def arm_overdub(self): self.calls.append('arm_overdub')
    def cancel_overdub(self): self.calls.append('cancel_overdub')


def make_controller(tmp_path):
    looper = FakeLooper()
    notes = []
    ctl = MidiController(looper, notify=lambda: notes.append(1),
                         _config_path=tmp_path / '_config.json')
    return looper, ctl, notes


def test_pad_launches_nth_section(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl.handle_trigger('pc:9:4', None)   # top-left pad = section 1
    ctl.handle_trigger('pc:9:5', None)   # section 2
    assert looper.launched == [11, 22]


def test_pad_for_missing_section_is_noop(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl.handle_trigger('pc:9:3', None)   # section 8, only 2 exist
    assert looper.launched == []


def test_knob_sets_loop_volume_scaled(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl.handle_trigger('cc:0:70', 127)   # K1 -> loop 1 (index 0) full
    ctl.handle_trigger('cc:0:77', 0)     # K8 -> loop 8 (index 7) silent
    assert looper.volumes[0] == 1.0
    assert looper.volumes[7] == 0.0


def test_record_toggle_follows_state_machine(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    for state, expected in [('idle', 'start_recording'),
                            ('recording_master', 'stop_recording'),
                            ('playing', 'arm_overdub'),
                            ('overdub_armed', 'cancel_overdub')]:
        looper._state_value = state
        ctl.handle_trigger('pc:9:12', None)
        assert looper.calls[-1] == expected
    looper._state_value = 'recording_overdub'
    n = len(looper.calls)
    ctl.handle_trigger('pc:9:12', None)   # finalizes at loop wrap on its own
    assert len(looper.calls) == n


def test_tap_tempo_averages_intervals(tmp_path, monkeypatch):
    # settable fake clock: handle_trigger reads monotonic() more than once
    # per trigger (tap + notify debounce), so an iterator would exhaust
    looper, ctl, _ = make_controller(tmp_path)
    t = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: t[0])
    for now in (0.0, 0.5, 1.0):          # two 0.5s gaps = 120 BPM
        t[0] = now
        ctl.handle_trigger('pc:9:13', None)
    assert looper.bpms[-1] == 120.0


def test_tap_tempo_resets_after_2s_gap(tmp_path, monkeypatch):
    looper, ctl, _ = make_controller(tmp_path)
    t = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: t[0])
    for now in (0.0, 5.0, 5.5):          # 5s gap resets, then one 0.5s gap
        t[0] = now
        ctl.handle_trigger('pc:9:13', None)
    assert looper.bpms == [120.0]        # only the post-reset pair counts


def test_unbound_trigger_is_noop(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl.handle_trigger('note:0:99', 64)
    assert looper.launched == [] and looper.calls == []


def test_actions_notify_ui(tmp_path):
    looper, ctl, notes = make_controller(tmp_path)
    ctl.handle_trigger('pc:9:4', None)
    assert notes  # at least one notify fired
