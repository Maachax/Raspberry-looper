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


def test_add_section_appends_empty_section_with_unique_id():
    looper = WebLooper()
    s1 = looper.add_section()
    s2 = looper.add_section()
    assert s1['loop_ids'] == []
    assert s1['id'] != s2['id']
    assert looper.sections == [s1, s2]


def test_set_section_loops_keeps_only_existing_layer_ids():
    looper = _looper_with_layers(3)          # valid ids: 0,1,2
    section = looper.add_section()
    looper.set_section_loops(section['id'], [0, 2, 99])
    assert section['loop_ids'] == [0, 2]        # 99 dropped


def test_delete_section_removes_it_and_clears_active():
    looper = _looper_with_layers(2)
    section = looper.add_section()
    looper._apply_section(section)                 # marks it active
    assert looper.active_section_id == section['id']
    assert looper.delete_section(section['id']) is True
    assert looper.sections == []
    assert looper.active_section_id is None


def test_apply_section_turns_on_only_member_loops():
    looper = _looper_with_layers(3)
    section = looper.add_section()
    looper.set_section_loops(section['id'], [0, 2])
    looper._apply_section(section)
    assert [l.is_playing for l in looper.layers] == [True, False, True]
    assert looper.active_section_id == section['id']


def test_apply_empty_section_mutes_everything():
    looper = _looper_with_layers(2)
    section = looper.add_section()
    looper._apply_section(section)
    assert [l.is_playing for l in looper.layers] == [False, False]


def test_launch_section_while_playing_is_quantized_not_immediate():
    looper = _looper_with_layers(2)          # state=PLAYING, master_length>0
    section = looper.add_section()
    looper.set_section_loops(section['id'], [1])
    looper.launch_section(section['id'], quantized=True)
    assert looper.pending_section is section       # queued, not applied yet
    assert looper.layers[0].is_playing is True   # unchanged until loop wrap


def test_launch_section_applies_immediately_when_not_playing():
    looper = _looper_with_layers(2)
    looper.state = LooperState.IDLE
    section = looper.add_section()
    looper.set_section_loops(section['id'], [1])
    looper.launch_section(section['id'], quantized=True)
    assert looper.pending_section is None
    assert [l.is_playing for l in looper.layers] == [False, True]


def test_delete_layer_remaps_section_loop_ids():
    looper = _looper_with_layers(3)          # ids 0,1,2
    section = looper.add_section()
    looper.set_section_loops(section['id'], [0, 2])
    looper.delete_layer(1)                   # id 2 becomes id 1 after renumber
    assert section['loop_ids'] == [0, 1]

def test_delete_layer_drops_its_own_id_from_sections():
    looper = _looper_with_layers(3)
    section = looper.add_section()
    looper.set_section_loops(section['id'], [1, 2])
    looper.delete_layer(1)                   # remove id 1; id 2 -> id 1
    assert section['loop_ids'] == [1]

def test_clear_all_empties_sections_and_resets():
    looper = _looper_with_layers(2)
    looper.add_section()
    looper.active_section_id = 1
    looper.pending_section = looper.sections[0]
    looper.clear_all()
    assert looper.sections == []
    assert looper.active_section_id is None
    assert looper.pending_section is None
    assert looper._next_section_id == 1


def test_get_state_includes_sections_block():
    looper = _looper_with_layers(2)
    section = looper.add_section()
    looper.set_section_loops(section['id'], [0])
    looper._apply_section(section)
    state = looper.get_state()
    assert state['sections']['list'] == [{'id': section['id'], 'loop_ids': [0], 'fx_overrides': {}}]
    assert state['sections']['active_id'] == section['id']
    assert state['sections']['pending_id'] is None

def test_sections_from_meta_reads_sections_when_present():
    meta = {'sections': [{'id': 1, 'loop_ids': [0, 2]}, {'id': 5, 'loop_ids': []}],
            'next_section_id': 6}
    sections, next_id = WebLooper._sections_from_meta(meta)
    assert sections == [{'id': 1, 'loop_ids': [0, 2], 'fx_overrides': {}}, {'id': 5, 'loop_ids': [], 'fx_overrides': {}}]
    assert next_id == 6

def test_sections_from_meta_migrates_old_scenes():
    meta = {'scenes': {
        '1': {'id': 1, 'name': 'A', 'layer_states': [
            {'id': 0, 'is_playing': True, 'volume': 1.0},
            {'id': 1, 'is_playing': False, 'volume': 1.0}]},
        '2': {'id': 2, 'name': 'B', 'layer_states': [
            {'id': 0, 'is_playing': True, 'volume': 1.0},
            {'id': 1, 'is_playing': True, 'volume': 1.0}]},
    }}
    sections, next_id = WebLooper._sections_from_meta(meta)
    assert sections == [{'id': 1, 'loop_ids': [0], 'fx_overrides': {}}, {'id': 2, 'loop_ids': [0, 1], 'fx_overrides': {}}]
    assert next_id == 3

def test_sections_from_meta_empty_when_neither_present():
    sections, next_id = WebLooper._sections_from_meta({})
    assert sections == []
    assert next_id == 1

def test_sections_from_meta_tolerates_null_and_missing_fields():
    assert WebLooper._sections_from_meta({'sections': None}) == ([], 1)
    sections, next_id = WebLooper._sections_from_meta({'sections': [{'id': 3}], 'next_section_id': 4})
    assert sections == [{'id': 3, 'loop_ids': [], 'fx_overrides': {}}]
    assert next_id == 4

def test_sections_from_meta_guards_stale_next_id():
    sections, next_id = WebLooper._sections_from_meta(
        {'sections': [{'id': 5, 'loop_ids': [0]}], 'next_section_id': 2})
    assert next_id == 6

def test_sections_from_meta_skips_non_dict_scene_entries():
    meta = {'scenes': {'1': 'garbage',
                       '2': {'layer_states': [{'id': 0, 'is_playing': True}]}}}
    sections, next_id = WebLooper._sections_from_meta(meta)
    assert sections == [{'id': 1, 'loop_ids': [0], 'fx_overrides': {}}]
    assert next_id == 2


def test_rename_section():
    looper = WebLooper()
    s = looper.add_section()
    assert s['name'] == ''
    assert looper.rename_section(s['id'], '  Verse ') is True
    assert looper.sections[0]['name'] == 'Verse'
    assert looper.get_state()['sections']['list'][0]['name'] == 'Verse'


def test_rename_section_empty_refused():
    looper = WebLooper()
    s = looper.add_section()
    looper.rename_section(s['id'], 'Chorus')
    assert looper.rename_section(s['id'], '   ') is False
    assert looper.sections[0]['name'] == 'Chorus'


def test_rename_section_missing_id():
    looper = WebLooper()
    assert looper.rename_section(999, 'X') is False


def test_sections_from_meta_preserves_name():
    meta = {'sections': [{'id': 1, 'loop_ids': [0], 'name': 'Verse'}],
            'next_section_id': 2}
    sections, _ = WebLooper._sections_from_meta(meta)
    assert sections[0]['name'] == 'Verse'


def test_sections_from_meta_old_format_defaults_empty_name():
    meta = {'sections': [{'id': 1, 'loop_ids': [0]}], 'next_section_id': 2}
    sections, _ = WebLooper._sections_from_meta(meta)
    assert sections[0]['name'] == ''
