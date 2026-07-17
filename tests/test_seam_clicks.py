"""Loop-seam click prevention: fx bake and export must not expose the seam step.

A live take rarely ends at the amplitude it started at. Playback hides that
step with an 8ms edge fade, but (bug) the fx bake tiled the raw dry — so a
delay echoed the seam click into the loop body — and exports skipped the
fade entirely.
"""
import sys, os
import io
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from audio import WebLooper, LoopLayer, LooperState, SAMPLE_RATE
import effects


def _seamed_dry(secs=1.0, start=0.5):
    """A take whose start (0.5) doesn't match its end (0.0) — a seam step."""
    n = int(SAMPLE_RATE * secs)
    return np.linspace(start, 0.0, n, endpoint=False).astype(np.float32)


def _wav_samples(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes)) as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0


def test_render_wet_does_not_echo_seam_click():
    dry = _seamed_dry()
    delay = effects.default_effect('delay')
    delay['params'].update({'time_s': 0.25, 'feedback': 0.3, 'mix': 0.5})
    wet = effects.render_wet(dry, [delay], SAMPLE_RATE)

    # Without the fix the delay replays the 0.5 seam step at t=0.25s as a
    # one-sample jump of ~0.25. Program material here moves ~1e-5/sample.
    i = int(0.25 * SAMPLE_RATE)
    echo_jump = np.abs(np.diff(wet[i - 100:i + 100])).max()
    assert echo_jump < 0.01, f"seam click echoed into loop body (jump={echo_jump:.3f})"


def test_export_mixed_fades_overdub_seam():
    looper = WebLooper()
    master = LoopLayer(0, "Master", np.zeros(SAMPLE_RATE, dtype=np.float32))
    over = LoopLayer(1, "Over", np.full(SAMPLE_RATE, 0.5, dtype=np.float32))
    looper.layers = [master, over]
    looper.master_length = SAMPLE_RATE
    looper.master_volume = 1.0

    wav_bytes, _, _ = looper.export_mixed('wav')
    assert wav_bytes is not None
    samples = _wav_samples(wav_bytes)

    # Overdub layers must be edge-faded like live playback: silent at the
    # seam, full level in the body.
    assert abs(samples[0]) < 0.01, "export should fade in overdub at loop start"
    assert abs(samples[-1]) < 0.01, "export should fade out overdub at loop end"
    assert abs(samples[len(samples) // 2]) > 0.45, "body level should be untouched"


def test_export_layer_fades_overdub_seam():
    looper = WebLooper()
    master = LoopLayer(0, "Master", np.zeros(SAMPLE_RATE, dtype=np.float32))
    over = LoopLayer(1, "Over", np.full(SAMPLE_RATE, 0.5, dtype=np.float32))
    looper.layers = [master, over]
    looper.master_length = SAMPLE_RATE

    wav_bytes, _, _ = looper.export_layer(1, 'wav')
    assert wav_bytes is not None
    samples = _wav_samples(wav_bytes)

    assert abs(samples[0]) < 0.01, "single-layer export should fade in at loop start"
    assert abs(samples[-1]) < 0.01, "single-layer export should fade out at loop end"
    assert abs(samples[len(samples) // 2]) > 0.45, "body level should be untouched"


def test_save_load_roundtrip_does_not_restack_fx():
    # save_session must store the dry take: load_session re-bakes the fx
    # chain, so saving the wet buffer applies the effect twice.
    n = SAMPLE_RATE
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    dry = (0.3 * np.sin(2 * np.pi * 220 * t) * np.linspace(1, 0, n)).astype(np.float32)

    looper = WebLooper()
    looper.layers = [LoopLayer(0, "Master", dry.copy())]
    looper.master_length = n
    delay = effects.default_effect('delay')
    delay['params'].update({'time_s': 0.1, 'feedback': 0.3, 'mix': 0.5})
    looper.set_loop_chain(0, [delay])
    wet_before = looper.layers[0].buffer.copy()

    res = looper.save_session('pytest-roundtrip')
    assert res['success']
    try:
        loaded = WebLooper()
        assert loaded.load_session(res['session_id'])['success']
        assert np.allclose(loaded.layers[0].dry, dry, atol=1e-6), \
            "saved session lost the dry take"
        assert np.allclose(loaded.layers[0].buffer, wet_before, atol=1e-4), \
            "loaded wet differs from saved wet (fx applied twice?)"
    finally:
        looper.delete_session(res['session_id'])


def test_export_mixed_leaves_master_layer_unfaded():
    looper = WebLooper()
    master = LoopLayer(0, "Master", np.full(SAMPLE_RATE, 0.3, dtype=np.float32))
    looper.layers = [master]
    looper.master_length = SAMPLE_RATE
    looper.master_volume = 1.0

    wav_bytes, _, _ = looper.export_mixed('wav')
    assert wav_bytes is not None
    samples = _wav_samples(wav_bytes)

    # Live playback applies no seam fade to the master loop; export matches.
    assert abs(samples[0] - 0.3) < 0.01
    assert abs(samples[-1] - 0.3) < 0.01
