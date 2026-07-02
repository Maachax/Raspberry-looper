import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import mido

from midi_control import normalize


def test_normalize_program_change():
    msg = mido.Message('program_change', channel=9, program=4)
    assert normalize(msg) == ('pc:9:4', None)


def test_normalize_control_change():
    msg = mido.Message('control_change', channel=0, control=70, value=99)
    assert normalize(msg) == ('cc:0:70', 99)


def test_normalize_note_on():
    msg = mido.Message('note_on', channel=0, note=48, velocity=64)
    assert normalize(msg) == ('note:0:48', 64)


def test_normalize_ignores_note_off_and_zero_velocity():
    assert normalize(mido.Message('note_off', channel=0, note=48)) is None
    assert normalize(mido.Message('note_on', channel=0, note=48, velocity=0)) is None


def test_normalize_ignores_other_messages():
    assert normalize(mido.Message('pitchwheel', channel=0, pitch=100)) is None
