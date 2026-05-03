import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from audio import WebLooper, LoopLayer, SAMPLE_RATE


def test_detect_scale_returns_five_candidates():
    """detect_scale() should return 5 ranked candidates for any audio."""
    looper = WebLooper()
    n = int(SAMPLE_RATE * 2.0)
    t = np.linspace(0, 2.0, n, dtype=np.float32)
    # A major chord: A4 + C#5 + E5
    signal = (
        0.4 * np.sin(2 * np.pi * 440.0 * t) +
        0.3 * np.sin(2 * np.pi * 554.37 * t) +
        0.2 * np.sin(2 * np.pi * 659.25 * t)
    )
    looper.layers = [LoopLayer(0, "Master", signal)]
    looper.master_length = n

    result = looper.detect_scale()

    assert result['success'] is True
    assert len(result['candidates']) == 5
    assert result['candidates'][0]['score'] == 100  # top is always 100
    for c in result['candidates']:
        assert 0 <= c['score'] <= 100
        assert c['root'] in ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        assert 'scale_type' in c


def test_detect_scale_top_root_is_a_for_a_major_chord():
    """A major chord should rank A as the top root."""
    looper = WebLooper()
    n = int(SAMPLE_RATE * 2.0)
    t = np.linspace(0, 2.0, n, dtype=np.float32)
    signal = (
        0.4 * np.sin(2 * np.pi * 440.0 * t) +
        0.3 * np.sin(2 * np.pi * 554.37 * t) +
        0.2 * np.sin(2 * np.pi * 659.25 * t)
    )
    looper.layers = [LoopLayer(0, "Master", signal)]
    looper.master_length = n

    result = looper.detect_scale()

    assert result['candidates'][0]['root'] == 'A'


def test_detect_scale_fails_gracefully_with_no_audio():
    """detect_scale() should return success=False when no audio is loaded."""
    looper = WebLooper()
    result = looper.detect_scale()
    assert result['success'] is False
    assert result['candidates'] == []


def test_detect_scale_fails_gracefully_with_silence():
    """Silent buffer should return success=False."""
    looper = WebLooper()
    n = int(SAMPLE_RATE * 2.0)
    looper.layers = [LoopLayer(0, "Master", np.zeros(n, dtype=np.float32))]
    looper.master_length = n
    result = looper.detect_scale()
    assert result['success'] is False
    assert result['candidates'] == []
