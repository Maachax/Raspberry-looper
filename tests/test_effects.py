import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import effects


def test_schemas_cover_all_five_types():
    assert set(effects.EFFECT_SCHEMAS) == {'reverb', 'delay', 'chorus', 'distortion', 'filter'}
    for params in effects.EFFECT_SCHEMAS.values():
        assert isinstance(params, list) and params
        for p in params:
            assert 'name' in p and 'default' in p
            assert ('min' in p and 'max' in p) or ('options' in p)  # numeric or enum


def test_default_effect_uses_schema_defaults():
    e = effects.default_effect('delay')
    assert e['type'] == 'delay'
    assert e['enabled'] is True
    for p in effects.EFFECT_SCHEMAS['delay']:
        assert e['params'][p['name']] == p['default']


def test_default_effect_unknown_type_raises():
    try:
        effects.default_effect('nope')
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_resolve_chain_prefers_section_override():
    loop_chain = [effects.default_effect('reverb')]
    override = [effects.default_effect('delay')]
    assert effects.resolve_chain(loop_chain, override) == override
    assert effects.resolve_chain(loop_chain, None) == loop_chain


def test_chain_hash_stable_and_sensitive():
    a = [effects.default_effect('delay')]
    b = [effects.default_effect('delay')]
    assert effects.chain_hash(a) == effects.chain_hash(b)
    b[0]['params']['mix'] = 0.99
    assert effects.chain_hash(a) != effects.chain_hash(b)
    assert effects.chain_hash([]) == effects.chain_hash([])


import numpy as np
SR = 44100


def _tone(freq, secs=1.0, amp=0.3):
    t = np.linspace(0, secs, int(SR * secs), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_render_wet_empty_chain_returns_dry_copy():
    dry = _tone(440)
    wet = effects.render_wet(dry, [], SR)
    assert np.array_equal(wet, dry)
    assert wet is not dry            # a copy, not the same object


def test_render_wet_preserves_length():
    dry = _tone(440, 0.5)
    e = effects.default_effect('reverb')
    wet = effects.render_wet(dry, [e], SR)
    assert len(wet) == len(dry)


def test_render_wet_disabled_effect_is_bypassed():
    dry = _tone(440, 0.5)
    e = effects.default_effect('distortion'); e['enabled'] = False
    wet = effects.render_wet(dry, [e], SR)
    assert np.array_equal(wet, dry)


def test_lowpass_filter_attenuates_high_frequency():
    dry = _tone(8000, 0.5)
    f = effects.default_effect('filter')
    f['params'].update({'mode': 'LP', 'cutoff_hz': 500.0})
    wet = effects.render_wet(dry, [f], SR)
    assert float(np.sqrt(np.mean(wet**2))) < float(np.sqrt(np.mean(dry**2))) * 0.6


def test_make_bus_reverb_processes_block_without_error():
    bus = effects.make_bus_reverb({'room_size': 0.6, 'damping': 0.5, 'wet': 0.3})
    block = _tone(440, 0.01)
    out = bus(block, SR, reset=False)
    assert out.shape == block.shape
