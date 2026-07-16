import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from audio import WebLooper, LoopLayer, LooperState, SAMPLE_RATE


def _looper_with_master(seconds=2.0):
    """A playing looper with a single master layer of ramp audio."""
    looper = WebLooper()
    n = int(seconds * SAMPLE_RATE)
    buf = np.linspace(0, 1, n, dtype=np.float32)
    looper.layers = [LoopLayer(0, "Master", buf)]
    looper.master_length = n
    looper.state = LooperState.PLAYING
    return looper


def test_first_trim_creates_backup_of_original():
    looper = _looper_with_master(2.0)
    original = looper.layers[0].buffer.copy()
    assert looper.apply_trim(0.5, 1.5) is True
    assert looper._pre_trim_backup is not None
    assert np.array_equal(looper._pre_trim_backup, original)


def test_successive_trims_keep_first_backup():
    looper = _looper_with_master(2.0)
    original = looper.layers[0].buffer.copy()
    looper.apply_trim(0.5, 1.5)
    looper.apply_trim(0.25, 0.75)   # relative to the already-trimmed loop
    assert np.array_equal(looper._pre_trim_backup, original)


def test_reset_trim_restores_original_and_clears_backup():
    looper = _looper_with_master(2.0)
    original = looper.layers[0].buffer.copy()
    looper.apply_trim(0.5, 1.5)
    assert looper.reset_trim() is True
    assert looper.master_length == len(original)
    assert np.array_equal(looper.layers[0].buffer, original)
    assert looper.master_position == 0
    assert looper._pre_trim_backup is None


def test_reset_trim_without_backup_fails():
    looper = _looper_with_master(2.0)
    assert looper.reset_trim() is False


def test_reset_trim_with_overdubs_fails():
    looper = _looper_with_master(2.0)
    looper.apply_trim(0.5, 1.5)
    looper.layers.append(LoopLayer(1, "Overdub 1",
                                   np.zeros(looper.master_length, dtype=np.float32)))
    assert looper.reset_trim() is False


def test_overdub_commit_discards_backup():
    looper = _looper_with_master(2.0)
    looper.apply_trim(0.5, 1.5)
    looper.recording_buffer = np.zeros(looper.max_samples, dtype=np.float32)
    looper._finalize_overdub()
    assert looper._pre_trim_backup is None


def test_clear_all_discards_backup():
    looper = _looper_with_master(2.0)
    looper.apply_trim(0.5, 1.5)
    looper.clear_all()
    assert looper._pre_trim_backup is None


def test_new_recording_discards_backup():
    looper = _looper_with_master(2.0)
    looper.apply_trim(0.5, 1.5)
    looper.state = LooperState.IDLE
    looper.start_recording()
    assert looper._pre_trim_backup is None


def test_state_reports_can_reset():
    looper = _looper_with_master(2.0)
    assert looper.get_state()['trim']['can_reset'] is False
    looper.apply_trim(0.5, 1.5)
    assert looper.get_state()['trim']['can_reset'] is True
    looper.reset_trim()
    assert looper.get_state()['trim']['can_reset'] is False
