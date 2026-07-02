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
        self.sections = [{'id': 11, 'loop_ids': []}, {'id': 22, 'loop_ids': []}]
        self.launched = []
        self.volumes = {}
        self.bpms = []
        self.calls = []
        self._state_value = 'idle'

        class L:
            def __init__(self, i, name):
                self.id, self.name = i, name
                self.is_playing = True
                self.fx_chain = []
        self.layers = [L(0, 'Master'), L(1, 'Loop 1'), L(2, 'Loop 2')]
        self.active_section_id = None
        self.master_bus = None

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

    def add_section(self):
        s = {'id': 100 + len(self.sections), 'loop_ids': [], 'fx_overrides': {}}
        self.sections.append(s)
        return s

    def set_section_loops(self, sid, ids):
        next(s for s in self.sections if s['id'] == sid)['loop_ids'] = list(ids)

    def delete_section(self, sid):
        self.sections = [s for s in self.sections if s['id'] != sid]

    def save_session(self, name):
        self.calls.append(f'save_session:{name}')
        return {'success': True, 'session_id': 'x', 'name': name or 'auto'}

    def toggle_layer(self, idx): self.calls.append(f'toggle_layer:{idx}')

    def delete_layer(self, idx):
        if idx <= 0 or idx >= len(self.layers):
            return False
        del self.layers[idx]
        return True

    def set_loop_chain(self, idx, chain):
        self.layers[idx].fx_chain = chain
        self.calls.append(f'set_loop_chain:{idx}')
        return True


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


def test_learn_binds_and_persists(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl.arm_learn('record_toggle')
    ctl.handle_trigger('note:0:60', 64)          # captured, not dispatched
    assert looper.calls == []
    assert ctl.learn is None
    assert ctl.bindings['global']['note:0:60'] == 'record_toggle'
    assert 'pc:9:12' not in ctl.bindings['global']   # old trigger removed
    # persisted: a fresh controller on the same config sees it
    _, ctl2, _ = make_controller(tmp_path)
    assert ctl2.bindings['global']['note:0:60'] == 'record_toggle'


def test_learn_unknown_action_ignored(tmp_path):
    _, ctl, _ = make_controller(tmp_path)
    ctl.arm_learn('warp_drive')
    assert ctl.learn is None


def test_learn_timeout_disarms(tmp_path, monkeypatch):
    _, ctl, notes = make_controller(tmp_path)
    t = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: t[0])
    ctl.arm_learn('tap_tempo')
    t[0] = 5.0
    ctl.check_learn_timeout()
    assert ctl.learn == 'tap_tempo'              # still armed at 5s
    t[0] = 11.0
    ctl.check_learn_timeout()
    assert ctl.learn is None                     # disarmed after 10s


def test_status_lists_actions_with_triggers(tmp_path):
    _, ctl, _ = make_controller(tmp_path)
    st = ctl.status()
    assert st['connected'] is False and st['mode'] == 'play'
    by_id = {a['id']: a for a in st['actions']}
    assert by_id['record_toggle']['trigger'] == 'pc:9:12'
    assert by_id['loop_volume_1']['trigger'] == 'cc:0:70'
    assert by_id['launch_section_1']['label'] == 'Launch section 1'


def test_match_port_case_insensitive_substring(tmp_path):
    _, ctl, _ = make_controller(tmp_path)
    names = ['Midi Through:Midi Through Port-0 14:0',
             'MPK mini 3:MPK mini 3 MIDI 1 28:0']
    assert ctl._match_port(names) == 'MPK mini 3:MPK mini 3 MIDI 1 28:0'
    assert ctl._match_port(['Midi Through 14:0']) is None
    assert ctl._match_port([]) is None


def test_on_message_routes_through_normalize(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl._on_message(mido.Message('program_change', channel=9, program=4))
    assert looper.launched == [11]
    ctl._on_message(mido.Message('note_off', channel=0, note=48))  # ignored
    assert looper.launched == [11]


def test_get_state_has_midi_block(tmp_path):
    from audio import WebLooper
    looper = WebLooper()
    state = looper.get_state()
    assert state['midi'] == {'connected': False, 'mode': 'play',
                             'learn': None, 'actions': [],
                             'selected_loop': None, 'selected_fx_slot': 0,
                             'editing_section': None, 'confirm': None}
    ctl = MidiController(looper, notify=lambda: None,
                         _config_path=tmp_path / '_config.json')
    looper.midi_status = ctl.status
    assert looper.get_state()['midi']['mode'] == 'play'
    assert looper.get_state()['midi']['actions']  # real action list now


def test_effective_loop_defaults_to_last(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    assert ctl.effective_loop() == 2          # last of 3 layers
    ctl.selected_loop = 1
    assert ctl.effective_loop() == 1
    ctl.selected_loop = 99                     # stale selection
    assert ctl.effective_loop() == 2
    looper.layers.clear()
    assert ctl.effective_loop() is None


def test_set_mode_transitions(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    looper.active_section_id = 22
    ctl.set_mode('section_edit')
    assert ctl.mode == 'section_edit'
    assert ctl.editing_section == 22           # seeded from active section
    ctl.set_mode('fx_edit')                    # entering one exits the other
    assert ctl.mode == 'fx_edit'
    assert ctl.selected_fx_slot == 0
    ctl.set_mode('play')
    assert ctl.mode == 'play'


def test_status_has_phase2_keys(tmp_path):
    _, ctl, _ = make_controller(tmp_path)
    st = ctl.status()
    assert st['selected_loop'] == 2            # effective
    assert st['selected_fx_slot'] == 0
    assert st['editing_section'] is None
    assert st['confirm'] is None


def test_phase2_default_bindings():
    g = DEFAULT_BINDINGS['global']
    assert g['pc:9:14'] == 'create_section'        # bank B pad 3
    assert g['pc:9:15'] == 'save_session'          # bank B pad 4
    assert g['pc:9:8'] == 'toggle_section_edit'    # bank B pad 5
    assert g['pc:9:9'] == 'toggle_fx_edit'         # bank B pad 6
    assert g['pc:9:10'] == 'mute_selected'         # bank B pad 7
    assert g['pc:9:11'] == 'delete_selected'       # bank B pad 8
    assert g['note:0:72'] == 'exit_mode'           # top key everywhere
    sel = DEFAULT_BINDINGS['select']
    assert sel['note:0:48'] == 'select_loop_1'
    assert sel['note:0:59'] == 'select_loop_12'
    se = DEFAULT_BINDINGS['section_edit']
    assert se['pc:9:4'] == 'edit_section_1'        # bank A, reading order
    assert se['note:0:48'] == 'toggle_member_1'
    assert se['note:0:71'] == 'delete_section'
    fx = DEFAULT_BINDINGS['fx_edit']
    assert fx['note:0:60'] == 'fx_add_reverb'
    assert fx['note:0:64'] == 'fx_add_filter'
    assert fx['note:0:65'] == 'fx_prev_slot'
    assert fx['note:0:67'] == 'fx_next_slot'
    assert fx['note:0:69'] == 'fx_toggle_enabled'
    assert fx['note:0:71'] == 'fx_remove'
    assert fx['cc:0:70'] == 'fx_param_1'
    assert fx['cc:0:72'] == 'fx_param_3'
    assert fx['cc:0:76'] == 'bus_room'
    assert fx['cc:0:77'] == 'bus_wet'


def test_dispatch_order_mode_over_select(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    # in play, note 48 selects loop 1
    ctl.handle_trigger('note:0:48', 64)
    assert ctl.selected_loop == 0
    # in fx_edit, cc 70 must NOT reach loop volumes (fx map wins over play map)
    ctl.set_mode('fx_edit')
    ctl.handle_trigger('cc:0:70', 127)
    assert 0 not in looper.volumes
    # selection keys still work in fx_edit via the select map
    ctl.handle_trigger('note:0:49', 64)
    assert ctl.selected_loop == 1
    # in section_edit, selection keys do NOT select (they toggle membership)
    ctl.set_mode('section_edit')
    ctl.handle_trigger('note:0:50', 64)
    assert ctl.selected_loop == 1              # unchanged


def test_knob_k4_unbound_in_fx_edit(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl.set_mode('fx_edit')
    ctl.handle_trigger('cc:0:73', 127)         # K4: reserved in fx_edit
    assert looper.volumes == {}


def test_create_section_from_playing_loops(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    looper.layers[1].is_playing = False
    ctl.handle_trigger('pc:9:14', None)
    new = looper.sections[-1]
    assert new['loop_ids'] == [0, 2]           # only the playing loops


def test_create_section_noop_when_nothing_playing(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    for l in looper.layers:
        l.is_playing = False
    n = len(looper.sections)
    ctl.handle_trigger('pc:9:14', None)
    assert len(looper.sections) == n


def test_save_session_fires_callback(tmp_path):
    looper = FakeLooper()
    saved = []
    ctl = MidiController(looper, notify=lambda: None,
                         on_session_saved=lambda: saved.append(1),
                         _config_path=tmp_path / '_config.json')
    ctl.handle_trigger('pc:9:15', None)
    assert looper.calls == ['save_session:']
    assert saved == [1]


def test_mute_selected(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl.selected_loop = 1
    ctl.handle_trigger('pc:9:10', None)
    assert looper.calls[-1] == 'toggle_layer:1'


def test_delete_selected_needs_double_tap(tmp_path, monkeypatch):
    looper, ctl, _ = make_controller(tmp_path)
    t = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: t[0])
    ctl.selected_loop = 2
    ctl.handle_trigger('pc:9:11', None)        # first tap: arms only
    assert len(looper.layers) == 3
    assert ctl.status()['confirm'] == 'delete_selected'
    t[0] = 0.5
    ctl.handle_trigger('pc:9:11', None)        # second tap inside 1s: fires
    assert len(looper.layers) == 2
    assert ctl.selected_loop is None           # selection cleared


def test_double_tap_expires_and_other_action_disarms(tmp_path, monkeypatch):
    looper, ctl, _ = make_controller(tmp_path)
    t = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: t[0])
    ctl.handle_trigger('pc:9:11', None)
    t[0] = 2.0                                  # window expired
    ctl.handle_trigger('pc:9:11', None)         # re-arms, doesn't fire
    assert len(looper.layers) == 3
    ctl.handle_trigger('pc:9:10', None)         # different action disarms
    assert ctl.status()['confirm'] is None
    t[0] = 2.2
    ctl.handle_trigger('pc:9:11', None)         # arms fresh again
    assert len(looper.layers) == 3


def test_section_edit_pad_picks_section(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl.set_mode('section_edit')
    ctl.handle_trigger('pc:9:5', None)          # bank A pad 2 = 2nd section
    assert ctl.editing_section == 22


def test_membership_toggle_adds_and_removes(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    looper.sections[0]['loop_ids'] = [0]
    ctl.set_mode('section_edit')
    ctl.handle_trigger('pc:9:4', None)          # edit section id 11
    ctl.handle_trigger('note:0:49', 64)         # toggle loop 2 (idx 1) -> add
    assert looper.sections[0]['loop_ids'] == [0, 1]
    ctl.handle_trigger('note:0:48', 64)         # toggle loop 1 (idx 0) -> remove
    assert looper.sections[0]['loop_ids'] == [1]
    ctl.handle_trigger('note:0:55', 64)         # loop 8 doesn't exist -> no-op
    assert looper.sections[0]['loop_ids'] == [1]


def test_membership_noop_without_edited_section(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl.set_mode('section_edit')                # no active section seeded
    assert ctl.editing_section is None
    ctl.handle_trigger('note:0:48', 64)
    assert looper.sections[0]['loop_ids'] == []


def test_delete_section_double_tap_and_exit(tmp_path, monkeypatch):
    looper, ctl, _ = make_controller(tmp_path)
    t = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: t[0])
    ctl.set_mode('section_edit')
    ctl.handle_trigger('pc:9:4', None)
    ctl.handle_trigger('note:0:71', 64)         # arm
    assert len(looper.sections) == 2
    t[0] = 0.4
    ctl.handle_trigger('note:0:71', 64)         # confirm
    assert len(looper.sections) == 1
    assert ctl.editing_section is None
    ctl.handle_trigger('note:0:72', 64)         # exit key
    assert ctl.mode == 'play'


def test_fx_add_appends_default_effect_and_selects_it(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl.selected_loop = 1
    ctl.set_mode('fx_edit')
    ctl.handle_trigger('note:0:60', 64)         # add reverb
    ctl.handle_trigger('note:0:61', 64)         # add delay
    chain = looper.layers[1].fx_chain
    assert [e['type'] for e in chain] == ['reverb', 'delay']
    assert chain[0]['enabled'] is True
    assert ctl.selected_fx_slot == 1            # follows the newest effect


def test_fx_slot_stepping_clamps(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl.selected_loop = 1
    ctl.set_mode('fx_edit')
    ctl.handle_trigger('note:0:60', 64)
    ctl.handle_trigger('note:0:61', 64)
    ctl.handle_trigger('note:0:65', 64)         # prev -> 0
    assert ctl.selected_fx_slot == 0
    ctl.handle_trigger('note:0:65', 64)         # prev at 0 stays 0
    assert ctl.selected_fx_slot == 0
    ctl.handle_trigger('note:0:67', 64)         # next -> 1
    ctl.handle_trigger('note:0:67', 64)         # next at end stays
    assert ctl.selected_fx_slot == 1


def test_fx_toggle_enabled(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl.selected_loop = 1
    ctl.set_mode('fx_edit')
    ctl.handle_trigger('note:0:60', 64)
    ctl.handle_trigger('note:0:69', 64)
    assert looper.layers[1].fx_chain[0]['enabled'] is False
    ctl.handle_trigger('note:0:69', 64)
    assert looper.layers[1].fx_chain[0]['enabled'] is True


def test_fx_remove_double_tap(tmp_path, monkeypatch):
    looper, ctl, _ = make_controller(tmp_path)
    t = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: t[0])
    ctl.selected_loop = 1
    ctl.set_mode('fx_edit')
    ctl.handle_trigger('note:0:60', 64)
    ctl.handle_trigger('note:0:71', 64)         # arm
    assert len(looper.layers[1].fx_chain) == 1
    t[0] = 0.3
    ctl.handle_trigger('note:0:71', 64)         # confirm
    assert looper.layers[1].fx_chain == []
    assert ctl.selected_fx_slot == 0


def test_fx_keys_noop_with_no_layers(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    looper.layers.clear()
    ctl.set_mode('fx_edit')
    ctl.handle_trigger('note:0:60', 64)         # must not raise
    assert looper.calls == []


from midi_control import scale_param


def test_scale_param_numeric_and_enum():
    num = {'name': 'drive_db', 'min': 0.0, 'max': 40.0, 'default': 18.0}
    assert scale_param(num, 0) == 0.0
    assert scale_param(num, 127) == 40.0
    assert abs(scale_param(num, 64) - 40.0 * 64 / 127) < 1e-9
    enum = {'name': 'mode', 'options': ['LP', 'HP'], 'default': 'LP'}
    assert scale_param(enum, 0) == 'LP'
    assert scale_param(enum, 63) == 'LP'
    assert scale_param(enum, 64) == 'HP'
    assert scale_param(enum, 127) == 'HP'


def test_fx_param_knob_commits_after_idle_not_per_tick(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl.selected_loop = 1
    ctl.set_mode('fx_edit')
    ctl.handle_trigger('note:0:63', 64)          # add distortion (1 bake)
    bakes = looper.calls.count('set_loop_chain:1')
    for v in range(10, 120, 10):                 # a knob sweep
        ctl.handle_trigger('cc:0:70', v)
    # audio-safety: the sweep itself must not re-bake
    assert looper.calls.count('set_loop_chain:1') == bakes
    ctl.flush_params()                           # idle commit (timer fires this in prod)
    assert looper.calls.count('set_loop_chain:1') == bakes + 1
    assert abs(looper.layers[1].fx_chain[0]['params']['drive_db']
               - 40.0 * 110 / 127) < 1e-9


def test_fx_param_knob_ignores_missing_param(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    ctl.selected_loop = 1
    ctl.set_mode('fx_edit')
    ctl.handle_trigger('note:0:63', 64)          # distortion has 1 param
    ctl.handle_trigger('cc:0:72', 64)            # fx_param_3 -> no such param
    ctl.flush_params()
    assert looper.layers[1].fx_chain[0]['params'] == {'drive_db': 18.0}


def test_bus_knobs_apply_immediately_and_autocreate(tmp_path):
    looper, ctl, _ = make_controller(tmp_path)
    looper.master_bus = None
    applied = []
    looper.set_bus = lambda sid, eff: applied.append((sid, eff)) or True
    ctl.set_mode('fx_edit')
    ctl.handle_trigger('cc:0:77', 127)           # bus wet full
    assert applied[-1][0] is None
    assert applied[-1][1]['type'] == 'reverb'
    assert applied[-1][1]['params']['wet'] == 1.0
