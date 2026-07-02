"""MPK Mini MIDI control surface: triggers, bindings, dispatch, hot-plug."""
import json
import threading
import time

from config import CONFIG_PATH


def normalize(msg):
    """mido Message -> (trigger_str, value) or None for non-control events."""
    if msg.type == 'program_change':
        return f"pc:{msg.channel}:{msg.program}", None
    if msg.type == 'control_change':
        return f"cc:{msg.channel}:{msg.control}", msg.value
    if msg.type == 'note_on' and msg.velocity > 0:
        return f"note:{msg.channel}:{msg.note}", msg.velocity
    return None


# action_id -> (mode, human label). 'global' works everywhere; 'select' is a
# shared map consulted in play and fx_edit (loop cursor keys).
ACTIONS = {
    'record_toggle': ('global', 'Record / Overdub'),
    'tap_tempo': ('global', 'Tap tempo'),
    'create_section': ('global', 'Create section from playing'),
    'save_session': ('global', 'Save session'),
    'toggle_section_edit': ('global', 'Section edit mode'),
    'toggle_fx_edit': ('global', 'FX edit mode'),
    'mute_selected': ('global', 'Mute selected loop'),
    'delete_selected': ('global', 'Delete selected loop (double-tap)'),
    'exit_mode': ('global', 'Exit edit mode'),
    **{f'launch_section_{i}': ('play', f'Launch section {i}') for i in range(1, 9)},
    **{f'loop_volume_{i}': ('play', f'Loop {i} volume') for i in range(1, 9)},
    **{f'select_loop_{i}': ('select', f'Select loop {i}') for i in range(1, 13)},
    **{f'edit_section_{i}': ('section_edit', f'Edit section {i}') for i in range(1, 9)},
    **{f'toggle_member_{i}': ('section_edit', f'Toggle loop {i} in section') for i in range(1, 13)},
    'delete_section': ('section_edit', 'Delete edited section (double-tap)'),
    'fx_add_reverb': ('fx_edit', 'Add reverb'),
    'fx_add_delay': ('fx_edit', 'Add delay'),
    'fx_add_chorus': ('fx_edit', 'Add chorus'),
    'fx_add_distortion': ('fx_edit', 'Add distortion'),
    'fx_add_filter': ('fx_edit', 'Add filter'),
    'fx_prev_slot': ('fx_edit', 'Previous FX slot'),
    'fx_next_slot': ('fx_edit', 'Next FX slot'),
    'fx_toggle_enabled': ('fx_edit', 'Toggle FX on/off'),
    'fx_remove': ('fx_edit', 'Remove FX (double-tap)'),
    **{f'fx_param_{i}': ('fx_edit', f'FX param {i}') for i in range(1, 4)},
    'bus_room': ('fx_edit', 'Bus reverb room'),
    'bus_wet': ('fx_edit', 'Bus reverb wet'),
}

_PAD_A_READING_ORDER = [4, 5, 6, 7, 0, 1, 2, 3]   # PC numbers, top-left -> bottom-right
_PAD_B_READING_ORDER = [12, 13, 14, 15, 8, 9, 10, 11]
_FX_ADD_KEYS = ['reverb', 'delay', 'chorus', 'distortion', 'filter']  # notes 60-64

DEFAULT_BINDINGS = {
    'global': {
        f'pc:9:{_PAD_B_READING_ORDER[0]}': 'record_toggle',
        f'pc:9:{_PAD_B_READING_ORDER[1]}': 'tap_tempo',
        f'pc:9:{_PAD_B_READING_ORDER[2]}': 'create_section',
        f'pc:9:{_PAD_B_READING_ORDER[3]}': 'save_session',
        f'pc:9:{_PAD_B_READING_ORDER[4]}': 'toggle_section_edit',
        f'pc:9:{_PAD_B_READING_ORDER[5]}': 'toggle_fx_edit',
        f'pc:9:{_PAD_B_READING_ORDER[6]}': 'mute_selected',
        f'pc:9:{_PAD_B_READING_ORDER[7]}': 'delete_selected',
        'note:0:72': 'exit_mode',
    },
    'play': {
        **{f'pc:9:{p}': f'launch_section_{i + 1}'
           for i, p in enumerate(_PAD_A_READING_ORDER)},
        **{f'cc:0:{70 + i}': f'loop_volume_{i + 1}' for i in range(8)},
    },
    'select': {
        **{f'note:0:{48 + i}': f'select_loop_{i + 1}' for i in range(12)},
    },
    'section_edit': {
        **{f'pc:9:{p}': f'edit_section_{i + 1}'
           for i, p in enumerate(_PAD_A_READING_ORDER)},
        **{f'note:0:{48 + i}': f'toggle_member_{i + 1}' for i in range(12)},
        'note:0:71': 'delete_section',
    },
    'fx_edit': {
        **{f'note:0:{60 + i}': f'fx_add_{name}' for i, name in enumerate(_FX_ADD_KEYS)},
        'note:0:65': 'fx_prev_slot',
        'note:0:67': 'fx_next_slot',
        'note:0:69': 'fx_toggle_enabled',
        'note:0:71': 'fx_remove',
        **{f'cc:0:{70 + i}': f'fx_param_{i + 1}' for i in range(3)},
        'cc:0:76': 'bus_room',
        'cc:0:77': 'bus_wet',
    },
}


def load_bindings(path=None):
    """Saved bindings from config if valid, else a deep copy of defaults."""
    path = path or CONFIG_PATH
    try:
        saved = json.loads(path.read_text())['midi']['bindings']
        if (isinstance(saved, dict)
                and saved
                and all(isinstance(m, dict) for m in saved.values())):
            return {mode: {t: a for t, a in m.items() if a in ACTIONS}
                    for mode, m in saved.items()}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {mode: dict(m) for mode, m in DEFAULT_BINDINGS.items()}


def save_bindings(bindings, path=None):
    """Persist bindings under the 'midi' config key, preserving other keys."""
    path = path or CONFIG_PATH
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data.setdefault('midi', {})['bindings'] = bindings
    path.write_text(json.dumps(data, indent=2))


class MidiController:
    """Dispatches normalized MIDI triggers to looper actions. One instance,
    one manager thread (started by start(); tests call handle_trigger directly)."""

    def __init__(self, looper, notify, port_match='MPK mini',
                 on_session_saved=None, _config_path=None):
        self.looper = looper
        self.notify = notify
        self.on_session_saved = on_session_saved
        self.port_match = port_match
        self._config_path = _config_path
        self.bindings = load_bindings(path=_config_path)
        self.connected = False
        self.mode = 'play'
        self.learn = None            # action_id currently being learned
        self._learn_armed_at = 0.0
        self._taps = []
        self.selected_loop = None    # loop index cursor; None = follow last
        self.selected_fx_slot = 0
        self.editing_section = None  # section id being edited in section_edit
        self._confirm = None         # (action_id, armed_at) double-tap state
        self._last_notify = 0.0
        self._port = None
        self._stop = threading.Event()

    # ------------------------------------------------------------ selection
    def effective_loop(self):
        """Selected loop index; defaults to the most recent layer."""
        n = len(self.looper.layers)
        if n == 0:
            return None
        if self.selected_loop is None or not (0 <= self.selected_loop < n):
            return n - 1
        return self.selected_loop

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'section_edit':
            self.editing_section = self.looper.active_section_id
        elif mode == 'fx_edit':
            self.selected_fx_slot = 0
        self.notify()

    # ------------------------------------------------------------ dispatch
    def handle_trigger(self, trigger, value):
        if self.learn is not None:
            self._bind_learned(trigger)
            return
        action = self.bindings.get('global', {}).get(trigger)
        if action is None:
            action = self.bindings.get(self.mode, {}).get(trigger)
        if action is None and self.mode in ('play', 'fx_edit'):
            action = self.bindings.get('select', {}).get(trigger)
        if action is None:
            return
        self._run_action(action, value)
        self._notify_debounced()

    _CONFIRM_ACTIONS = {'delete_selected', 'delete_section', 'fx_remove'}

    def _confirmed(self, action):
        """Double-tap guard: True only on the second tap within 1s."""
        now = time.monotonic()
        if (self._confirm and self._confirm[0] == action
                and now - self._confirm[1] <= 1.0):
            self._confirm = None
            return True
        self._confirm = (action, now)
        return False

    def _run_action(self, action, value):
        if self._confirm and action != self._confirm[0]:
            self._confirm = None            # any other action disarms
        if action in self._CONFIRM_ACTIONS and not self._confirmed(action):
            return
        if action == 'create_section':
            playing = [l.id for l in self.looper.layers if l.is_playing]
            if playing:
                section = self.looper.add_section()
                self.looper.set_section_loops(section['id'], playing)
            return
        if action == 'save_session':
            self.looper.save_session('')
            if self.on_session_saved:
                self.on_session_saved()
            return
        if action == 'mute_selected':
            idx = self.effective_loop()
            if idx is not None:
                self.looper.toggle_layer(idx)
            return
        if action == 'delete_selected':
            idx = self.effective_loop()
            if idx is not None and self.looper.delete_layer(idx):
                self.selected_loop = None
            return
        if action.startswith('select_loop_'):
            idx = int(action.rsplit('_', 1)[1]) - 1
            if idx < len(self.looper.layers):
                self.selected_loop = idx
            return
        if action == 'exit_mode':
            self.set_mode('play')
            return
        if action == 'toggle_section_edit':
            self.set_mode('play' if self.mode == 'section_edit' else 'section_edit')
            return
        if action == 'toggle_fx_edit':
            self.set_mode('play' if self.mode == 'fx_edit' else 'fx_edit')
            return
        if action == 'record_toggle':
            self._record_toggle()
        elif action == 'tap_tempo':
            self._tap()
        elif action.startswith('launch_section_'):
            idx = int(action.rsplit('_', 1)[1]) - 1
            sections = self.looper.sections
            if idx < len(sections):
                self.looper.launch_section(sections[idx]['id'])
        elif action.startswith('loop_volume_'):
            if value is not None:
                idx = int(action.rsplit('_', 1)[1]) - 1
                self.looper.set_layer_volume(idx, value / 127.0)

    def _record_toggle(self):
        s = self.looper.state.value
        if s == 'idle':
            self.looper.start_recording()
        elif s == 'recording_master':
            self.looper.stop_recording()
        elif s == 'playing':
            self.looper.arm_overdub()
        elif s == 'overdub_armed':
            self.looper.cancel_overdub()
        # recording_overdub: finalizes at loop wrap on its own

    def _tap(self):
        now = time.monotonic()
        if self._taps and now - self._taps[-1] > 2.0:
            self._taps = []
        self._taps.append(now)
        if len(self._taps) >= 2:
            gaps = [b - a for a, b in zip(self._taps, self._taps[1:])]
            self.looper.set_bpm(60.0 / (sum(gaps) / len(gaps)))

    # ------------------------------------------------------------ UI sync
    def _notify_debounced(self):
        """At most ~10 notifies/s; the UI's periodic poll covers the tail."""
        now = time.monotonic()
        if now - self._last_notify >= 0.1:
            self._last_notify = now
            self.notify()

    # ------------------------------------------------------------ learn
    def arm_learn(self, action_id):
        if action_id not in ACTIONS:
            return
        self.learn = action_id
        self._learn_armed_at = time.monotonic()
        self.notify()

    def _bind_learned(self, trigger):
        action = self.learn
        mode = ACTIONS[action][0]
        mode_map = self.bindings.setdefault(mode, {})
        for t in [t for t, a in mode_map.items() if a == action]:
            del mode_map[t]
        mode_map[trigger] = action
        save_bindings(self.bindings, path=self._config_path)
        self.learn = None
        self.notify()

    def check_learn_timeout(self):
        if self.learn is not None and time.monotonic() - self._learn_armed_at > 10.0:
            self.learn = None
            self.notify()

    # ------------------------------------------------------------ status
    def status(self):
        trigger_of = {}
        for mode_map in self.bindings.values():
            for trig, act in mode_map.items():
                trigger_of[act] = trig
        return {
            'connected': self.connected,
            'mode': self.mode,
            'learn': self.learn,
            'selected_loop': self.effective_loop(),
            'selected_fx_slot': self.selected_fx_slot,
            'editing_section': self.editing_section,
            'confirm': self._confirm[0] if self._confirm else None,
            'actions': [{'id': a, 'label': label, 'trigger': trigger_of.get(a)}
                        for a, (_mode, label) in ACTIONS.items()],
        }

    # ------------------------------------------------------------ hot-plug
    def _match_port(self, names):
        want = self.port_match.lower()
        return next((n for n in names if want in n.lower()), None)

    def _on_message(self, msg):
        normalized = normalize(msg)
        if normalized:
            self.handle_trigger(*normalized)

    def start(self):
        threading.Thread(target=self._manager_loop, daemon=True,
                         name='midi-manager').start()

    def stop(self):
        self._stop.set()

    def _manager_loop(self):
        import mido
        while not self._stop.is_set():
            self.check_learn_timeout()
            try:
                names = mido.get_input_names()
            except Exception:
                names = []
            match = self._match_port(names)
            if self._port is None and match:
                try:
                    self._port = mido.open_input(match, callback=self._on_message)
                    self.connected = True
                    print(f"🎹 MIDI connected: {match}")
                    self.notify()
                except Exception as e:
                    print(f"🎹 MIDI open failed: {e}")
                    self._port = None
            elif self._port is not None and match is None:
                try:
                    self._port.close()
                except Exception:
                    pass
                self._port = None
                self.connected = False
                print("🎹 MIDI disconnected")
                self.notify()
            self._stop.wait(2.0)
