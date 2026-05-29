import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from audio import WebLooper, LoopLayer, LooperState, SAMPLE_RATE


def _looper_with_layers(n):
    """A looper with n one-second layers (ids 0..n-1), playing."""
    looper = WebLooper()
    looper.layers = [LoopLayer(i, f"L{i}", np.zeros(SAMPLE_RATE, dtype=np.float32))
                     for i in range(n)]
    looper.master_length = SAMPLE_RATE
    looper.state = LooperState.PLAYING
    return looper


def test_add_slot_appends_empty_slot_with_unique_id():
    looper = WebLooper()
    s1 = looper.add_slot()
    s2 = looper.add_slot()
    assert s1['loop_ids'] == []
    assert s1['id'] != s2['id']
    assert looper.slots == [s1, s2]


def test_set_slot_loops_keeps_only_existing_layer_ids():
    looper = _looper_with_layers(3)          # valid ids: 0,1,2
    slot = looper.add_slot()
    looper.set_slot_loops(slot['id'], [0, 2, 99])
    assert slot['loop_ids'] == [0, 2]        # 99 dropped


def test_delete_slot_removes_it_and_clears_active():
    looper = _looper_with_layers(2)
    slot = looper.add_slot()
    looper._apply_slot(slot)                 # marks it active
    assert looper.active_slot_id == slot['id']
    assert looper.delete_slot(slot['id']) is True
    assert looper.slots == []
    assert looper.active_slot_id is None


def test_apply_slot_turns_on_only_member_loops():
    looper = _looper_with_layers(3)
    slot = looper.add_slot()
    looper.set_slot_loops(slot['id'], [0, 2])
    looper._apply_slot(slot)
    assert [l.is_playing for l in looper.layers] == [True, False, True]
    assert looper.active_slot_id == slot['id']


def test_apply_empty_slot_mutes_everything():
    looper = _looper_with_layers(2)
    slot = looper.add_slot()
    looper._apply_slot(slot)
    assert [l.is_playing for l in looper.layers] == [False, False]


def test_launch_slot_while_playing_is_quantized_not_immediate():
    looper = _looper_with_layers(2)          # state=PLAYING, master_length>0
    slot = looper.add_slot()
    looper.set_slot_loops(slot['id'], [1])
    looper.launch_slot(slot['id'], quantized=True)
    assert looper.pending_slot is slot       # queued, not applied yet
    assert looper.layers[0].is_playing is True   # unchanged until loop wrap


def test_launch_slot_applies_immediately_when_not_playing():
    looper = _looper_with_layers(2)
    looper.state = LooperState.IDLE
    slot = looper.add_slot()
    looper.set_slot_loops(slot['id'], [1])
    looper.launch_slot(slot['id'], quantized=True)
    assert looper.pending_slot is None
    assert [l.is_playing for l in looper.layers] == [False, True]


def test_delete_layer_remaps_slot_loop_ids():
    looper = _looper_with_layers(3)          # ids 0,1,2
    slot = looper.add_slot()
    looper.set_slot_loops(slot['id'], [0, 2])
    looper.delete_layer(1)                   # id 2 becomes id 1 after renumber
    assert slot['loop_ids'] == [0, 1]

def test_delete_layer_drops_its_own_id_from_slots():
    looper = _looper_with_layers(3)
    slot = looper.add_slot()
    looper.set_slot_loops(slot['id'], [1, 2])
    looper.delete_layer(1)                   # remove id 1; id 2 -> id 1
    assert slot['loop_ids'] == [1]

def test_clear_all_empties_slots_and_resets():
    looper = _looper_with_layers(2)
    looper.add_slot()
    looper.active_slot_id = 1
    looper.pending_slot = looper.slots[0]
    looper.clear_all()
    assert looper.slots == []
    assert looper.active_slot_id is None
    assert looper.pending_slot is None
    assert looper._next_slot_id == 1


def test_get_state_includes_slots_block():
    looper = _looper_with_layers(2)
    slot = looper.add_slot()
    looper.set_slot_loops(slot['id'], [0])
    looper._apply_slot(slot)
    state = looper.get_state()
    assert state['slots']['list'] == [{'id': slot['id'], 'loop_ids': [0]}]
    assert state['slots']['active_id'] == slot['id']
    assert state['slots']['pending_id'] is None

def test_slots_from_meta_reads_slots_when_present():
    meta = {'slots': [{'id': 1, 'loop_ids': [0, 2]}, {'id': 5, 'loop_ids': []}],
            'next_slot_id': 6}
    slots, next_id = WebLooper._slots_from_meta(meta)
    assert slots == [{'id': 1, 'loop_ids': [0, 2]}, {'id': 5, 'loop_ids': []}]
    assert next_id == 6

def test_slots_from_meta_migrates_old_scenes():
    meta = {'scenes': {
        '1': {'id': 1, 'name': 'A', 'layer_states': [
            {'id': 0, 'is_playing': True, 'volume': 1.0},
            {'id': 1, 'is_playing': False, 'volume': 1.0}]},
        '2': {'id': 2, 'name': 'B', 'layer_states': [
            {'id': 0, 'is_playing': True, 'volume': 1.0},
            {'id': 1, 'is_playing': True, 'volume': 1.0}]},
    }}
    slots, next_id = WebLooper._slots_from_meta(meta)
    assert slots == [{'id': 1, 'loop_ids': [0]}, {'id': 2, 'loop_ids': [0, 1]}]
    assert next_id == 3

def test_slots_from_meta_empty_when_neither_present():
    slots, next_id = WebLooper._slots_from_meta({})
    assert slots == []
    assert next_id == 1

def test_slots_from_meta_tolerates_null_and_missing_fields():
    assert WebLooper._slots_from_meta({'slots': None}) == ([], 1)
    slots, next_id = WebLooper._slots_from_meta({'slots': [{'id': 3}], 'next_slot_id': 4})
    assert slots == [{'id': 3, 'loop_ids': []}]
    assert next_id == 4

def test_slots_from_meta_guards_stale_next_id():
    slots, next_id = WebLooper._slots_from_meta(
        {'slots': [{'id': 5, 'loop_ids': [0]}], 'next_slot_id': 2})
    assert next_id == 6

def test_slots_from_meta_skips_non_dict_scene_entries():
    meta = {'scenes': {'1': 'garbage',
                       '2': {'layer_states': [{'id': 0, 'is_playing': True}]}}}
    slots, next_id = WebLooper._slots_from_meta(meta)
    assert slots == [{'id': 1, 'loop_ids': [0]}]
    assert next_id == 2
