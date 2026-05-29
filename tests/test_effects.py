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
