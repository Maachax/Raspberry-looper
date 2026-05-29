import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from audio import WebLooper, LoopLayer, LooperState, SAMPLE_RATE
import effects


def _tone(freq, secs=0.5, amp=0.3):
    t = np.linspace(0, secs, int(SAMPLE_RATE * secs), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_new_layer_dry_equals_buffer_and_empty_chain():
    layer = LoopLayer(0, "Master", _tone(440))
    assert np.array_equal(layer.dry, layer.buffer)
    assert layer.fx_chain == []


def test_set_loop_chain_rebakes_wet_from_dry():
    looper = WebLooper()
    looper.layers = [LoopLayer(0, "Master", _tone(8000))]
    looper.master_length = looper.layers[0].length
    looper.state = LooperState.PLAYING
    f = effects.default_effect('filter'); f['params'].update({'mode': 'LP', 'cutoff_hz': 500.0})
    looper.set_loop_chain(0, [f])
    layer = looper.layers[0]
    assert layer.fx_chain[0]['type'] == 'filter'
    assert np.array_equal(layer.dry, _tone(8000))
    assert float(np.sqrt(np.mean(layer.buffer**2))) < float(np.sqrt(np.mean(layer.dry**2))) * 0.7


def test_set_empty_chain_restores_dry():
    looper = WebLooper()
    looper.layers = [LoopLayer(0, "Master", _tone(440))]
    looper.master_length = looper.layers[0].length
    looper.set_loop_chain(0, [effects.default_effect('distortion')])
    looper.set_loop_chain(0, [])
    assert np.array_equal(looper.layers[0].buffer, looper.layers[0].dry)


def test_wet_cache_reused_for_identical_chain():
    looper = WebLooper()
    looper.layers = [LoopLayer(0, "Master", _tone(440))]
    looper.master_length = looper.layers[0].length
    chain = [effects.default_effect('reverb')]
    looper.set_loop_chain(0, chain)
    first = looper.layers[0].buffer
    looper.set_loop_chain(0, [effects.default_effect('reverb')])  # identical content
    assert looper.layers[0].buffer is first   # served from cache, same object
