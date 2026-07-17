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


from audio import score_scale_templates


def _uniform_chroma(pcs):
    """Length-12 chroma with equal energy on the given pitch classes, sum=1."""
    v = np.zeros(12)
    for pc in pcs:
        v[pc] = 1.0
    return v / v.sum()


A_MINOR_PCS = [9, 11, 0, 2, 4, 5, 7]  # A B C D E F G


def test_score_templates_bass_disambiguates_relative_modes():
    """Same 7 notes, bass says A -> A minor must beat C major and D dorian."""
    chroma = _uniform_chroma(A_MINOR_PCS)
    bass = np.zeros(12)
    bass[9] = 1.0  # all bass energy on A
    results = score_scale_templates(chroma, bass)
    by_key = {(c['root'], c['scale_type']): c['_score'] for c in results}
    assert by_key[('A', 'minor')] > by_key[('C', 'major')]
    assert by_key[('A', 'minor')] > by_key[('D', 'dorian')]
    assert results[0]['root'] == 'A'


def test_score_templates_out_of_scale_energy_penalized():
    """Adding out-of-scale energy must lower a scale's score."""
    clean = _uniform_chroma(A_MINOR_PCS)
    polluted = np.array(clean)
    polluted[1] += 0.3  # C# does not belong to A minor
    polluted = polluted / polluted.sum()
    bass = _uniform_chroma([9])

    def score_of(results, root, stype):
        return next(c['_score'] for c in results
                    if c['root'] == root and c['scale_type'] == stype)

    s_clean = score_of(score_scale_templates(clean, bass), 'A', 'minor')
    s_polluted = score_of(score_scale_templates(polluted, bass), 'A', 'minor')
    assert s_polluted < s_clean


def test_score_templates_size_normalization():
    """Playing only pentatonic notes: the pentatonic must beat the full scale."""
    a_pent_minor = [9, 0, 2, 4, 7]  # A C D E G
    chroma = _uniform_chroma(a_pent_minor)
    bass = _uniform_chroma([9])
    results = score_scale_templates(chroma, bass)
    by_key = {(c['root'], c['scale_type']): c['_score'] for c in results}
    assert by_key[('A', 'pent_minor')] > by_key[('A', 'minor')]


def test_score_templates_returns_all_168_sorted():
    chroma = _uniform_chroma(A_MINOR_PCS)
    bass = _uniform_chroma([9])
    results = score_scale_templates(chroma, bass)
    assert len(results) == 12 * len(SCALE_TEMPLATES)
    scores = [c['_score'] for c in results]
    assert scores == sorted(scores, reverse=True)


def _make_looper_with_signal(signal):
    looper = WebLooper()
    looper.layers = [LoopLayer(0, "Master", signal)]
    looper.master_length = len(signal)
    return looper


def _a_minor_with_bass_drone(seconds=2.0):
    """All 7 A-natural-minor tones in octave 4 plus a strong A2 (110 Hz) drone."""
    n = int(SAMPLE_RATE * seconds)
    t = np.linspace(0, seconds, n, dtype=np.float64)
    # A4 B4 C5 D5 E5 F5 G5
    freqs = [440.0, 493.88, 523.25, 587.33, 659.25, 698.46, 783.99]
    sig = sum(0.1 * np.sin(2 * np.pi * f * t) for f in freqs)
    sig += 0.5 * np.sin(2 * np.pi * 110.0 * t)  # A2 bass drone
    return sig.astype(np.float32)


def test_detect_scale_bass_picks_a_minor_over_relatives():
    """A minor notes + A bass drone: top candidate is root A, minor family."""
    looper = _make_looper_with_signal(_a_minor_with_bass_drone())
    result = looper.detect_scale()
    assert result['success'] is True
    assert result['candidates'][0]['root'] == 'A'
    assert result['candidates'][0]['scale_type'] in ('minor', 'pent_minor')


def test_detect_scale_selected_notes_filter_candidates():
    """Every returned candidate must contain all selected notes."""
    looper = _make_looper_with_signal(_a_minor_with_bass_drone())
    result = looper.detect_scale(selected_notes=['C#'])
    assert result['success'] is True
    cs_pc = NOTE_NAMES.index('C#')
    for c in result['candidates']:
        assert cs_pc in _pcs(c)


def test_detect_scale_theory_mode_without_loop():
    """Notes picked but nothing recorded: pure theory matching, no librosa needed."""
    looper = WebLooper()
    result = looper.detect_scale(selected_notes=['A', 'C', 'E'])
    assert result['success'] is True
    assert 0 < len(result['candidates']) <= 5
    assert result['candidates'][0]['score'] == 100
    selected = {NOTE_NAMES.index(x) for x in ['A', 'C', 'E']}
    for c in result['candidates']:
        assert selected <= _pcs(c)


def test_detect_scale_filter_can_empty_the_list():
    """A selection no scale contains: success with empty candidates."""
    looper = _make_looper_with_signal(_a_minor_with_bass_drone())
    result = looper.detect_scale(
        selected_notes=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#'])
    assert result['success'] is True
    assert result['candidates'] == []


def test_detect_scale_rejects_unknown_note_name():
    looper = WebLooper()
    result = looper.detect_scale(selected_notes=['H'])
    assert result['success'] is False
    assert result['candidates'] == []
