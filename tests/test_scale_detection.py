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


from audio import match_scales_by_notes, SCALE_TEMPLATES, NOTE_NAMES


def _pcs(candidate):
    """Pitch-class set of a candidate dict."""
    root = NOTE_NAMES.index(candidate['root'])
    return {(root + iv) % 12 for iv in SCALE_TEMPLATES[candidate['scale_type']]}


def test_match_scales_requires_all_selected_notes():
    """Every returned scale must contain all selected pitch classes."""
    selected = {9, 0, 4}  # A, C, E
    results = match_scales_by_notes(selected)
    assert len(results) > 0
    for c in results:
        assert selected <= _pcs(c)


def test_match_scales_prefers_small_scales_with_selected_root():
    """Top result should be a pentatonic rooted on a selected note (A/C/E)."""
    selected = {9, 0, 4}  # A, C, E
    results = match_scales_by_notes(selected)
    top = results[0]
    assert NOTE_NAMES.index(top['root']) in selected
    assert len(SCALE_TEMPLATES[top['scale_type']]) == 5  # pentatonic beats 7-note


def test_match_scales_root_bonus_beats_coverage_tiebreak():
    """With one selected note, scales rooted on it must outrank same-size scales that merely contain it."""
    results = match_scales_by_notes({9})  # just A
    top = results[0]
    assert top['root'] == 'A'


def test_match_scales_chromatic_cluster_matches_nothing():
    """7 consecutive semitones fit no template."""
    results = match_scales_by_notes({0, 1, 2, 3, 4, 5, 6})
    assert results == []


def test_match_scales_sorted_descending():
    results = match_scales_by_notes({9, 0, 4})
    scores = [c['_score'] for c in results]
    assert scores == sorted(scores, reverse=True)
