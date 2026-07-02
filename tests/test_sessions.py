import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path

from audio import WebLooper
from config import SESSIONS_DIR


def test_delete_session_rejects_path_traversal(tmp_path):
    # a real directory one level above the sessions dir — a traversal id
    # would rmtree it if the guard is missing
    probe = SESSIONS_DIR.parent / '__traversal_probe__'
    probe.mkdir(exist_ok=True)
    try:
        looper = WebLooper()
        result = looper.delete_session('../__traversal_probe__')
        assert result['success'] is False
        assert probe.exists(), "traversal id escaped the sessions dir and deleted outside it"
    finally:
        if probe.exists():
            probe.rmdir()


def test_load_session_rejects_path_traversal():
    looper = WebLooper()
    result = looper.load_session('../..')
    assert result['success'] is False


def test_delete_session_missing_id_fails_cleanly():
    looper = WebLooper()
    result = looper.delete_session('no-such-session-xyz')
    assert result['success'] is False
