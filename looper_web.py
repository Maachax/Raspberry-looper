#!/usr/bin/env python3
"""
Guitar Looper - Clean Implementation
Flask + WebSockets for real-time control

Features:
    - Tap Tempo
    - Metronome during countdown
    - Quantized loop length
    - Visual beat grid
    - MP3/WAV Export

State Machine:
    IDLE → RECORDING_MASTER → PLAYING ⇄ OVERDUB_ARMED → RECORDING_OVERDUB → PLAYING
"""

import json
import sounddevice as sd
import numpy as np
import threading
import time
from enum import Enum
from io import BytesIO
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
from flask import Flask, render_template_string, request, Response, send_file
from flask_socketio import SocketIO, emit

# Optional: librosa for tempo detection
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠ librosa not installed - tempo detection disabled")
    print("  Install with: pip install librosa")

# Optional: pydub for audio export
try:
    from pydub import AudioSegment
    from pydub.utils import which
    PYDUB_AVAILABLE = True
    FFMPEG_AVAILABLE = which("ffmpeg") is not None
    if not FFMPEG_AVAILABLE:
        print("⚠ ffmpeg not found - MP3 export disabled")
        print("  Install with: sudo apt install ffmpeg (Linux)")
        print("              or: brew install ffmpeg (macOS)")
except ImportError:
    PYDUB_AVAILABLE = False
    FFMPEG_AVAILABLE = False
    print("⚠ pydub not installed - audio export disabled")
    print("  Install with: pip install pydub")

# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_RATE = 44100
BLOCKSIZE = 256
CHANNELS = 1
MAX_LOOP_SECONDS = 120

# Export settings
EXPORT_MP3_BITRATE = "192k"
EXPORT_WAV_SAMPLE_WIDTH = 2  # 16-bit

# Sessions directory (relative to this script)
SESSIONS_DIR = Path(__file__).parent / '_sessions'
SESSIONS_DIR.mkdir(exist_ok=True)

# Layer color palette (cycles by layer id)
LAYER_COLORS = [
    '#667eea',  # Purple
    '#38a169',  # Green
    '#ed8936',  # Orange
    '#e53e3e',  # Red
    '#319795',  # Teal
    '#d53f8c',  # Pink
    '#d69e2e',  # Yellow
    '#3182ce',  # Blue
]

# =============================================================================
# STATE MACHINE
# =============================================================================

class LooperState(Enum):
    IDLE = "idle"                           # No master loop yet
    RECORDING_MASTER = "recording_master"   # Recording first loop
    PLAYING = "playing"                     # Loops playing, ready for overdub
    OVERDUB_ARMED = "overdub_armed"         # Waiting for loop restart
    RECORDING_OVERDUB = "recording_overdub" # Recording overdub layer

# =============================================================================
# LOOP LAYER
# =============================================================================

class LoopLayer:
    """A single loop layer with its audio buffer."""
    
    def __init__(self, layer_id: int, name: str, buffer: np.ndarray):
        self.id = layer_id
        self.name = name
        self.buffer = buffer
        self.length = len(buffer)
        self.volume = 1.0
        self.is_playing = True
        self.color = LAYER_COLORS[layer_id % len(LAYER_COLORS)]

    def get_sample_at(self, position: int) -> float:
        """Get sample value at given position (with volume applied)."""
        if not self.is_playing or position >= self.length:
            return 0.0
        return self.buffer[position] * self.volume
    
    def to_dict(self) -> dict:
        """Serialize layer state for web client."""
        return {
            'id': self.id,
            'name': self.name,
            'duration': self.length / SAMPLE_RATE,
            'volume': self.volume,
            'is_playing': self.is_playing,
            'color': self.color,
        }

# =============================================================================
# WEB LOOPER
# =============================================================================

class WebLooper:
    """
    Main looper engine with web interface support.
    
    Audio flow:
        Input → [Recording buffer] → Output mix
                                   ↓
                            [Layer buffers] → Output mix
    """
    
    def __init__(self, device=None):
        self.device = device
        self.max_samples = SAMPLE_RATE * MAX_LOOP_SECONDS
        
        # State
        self.state = LooperState.IDLE
        self.layers: list[LoopLayer] = []
        
        # Master loop timing
        self.master_length = 0          # Length in samples
        self.master_position = 0        # Current playback position
        
        # Recording buffers
        self.recording_buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.recording_position = 0
        
        # Tempo settings
        self.bpm = 120.0                # Beats per minute
        self.beats_per_bar = 4          # Time signature (beats per bar)
        self.quantize_enabled = True    # Auto-snap to nearest bar
        
        # Master volume (0.0 to 1.0)
        self.master_volume = 0.8        # Default 80% - leaves headroom
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Audio stream
        self.stream = None
        
        # Stats
        self.callback_time = 0
        self.dropout_count = 0

        # Input level metering
        self.input_level = 0.0       # Smoothed RMS (0–1)
        self.input_peak = 0.0        # Peak hold (0–1)
        self._peak_hold_frames = 0   # Callback frames since last peak reset

        # Scenes
        self.scenes = {}           # {scene_id: scene_dict}
        self._next_scene_id = 1
        self.pending_scene = None  # Scene to apply at next loop restart

        # Reactive scene collapse
        self.collapse_enabled = False
        self.collapse_scene_id = None   # int scene id to switch to on silence
        self.collapse_timeout = 4.0     # seconds of silence before collapsing
        self.collapse_threshold = 0.01  # RMS threshold (≈ -40dB)
        self._silence_frames = 0        # consecutive silent callback frames
        self._collapse_triggered = False  # True after collapse, reset when playing again

        print("✓ Looper initialized")
    
    # -------------------------------------------------------------------------
    # AUDIO CALLBACK (Real-time thread)
    # -------------------------------------------------------------------------
    
    def audio_callback(self, indata, outdata, frames, time_info, status):
        """
        Real-time audio callback. Called by sounddevice.
        
        CRITICAL: This runs in a separate thread. Keep lock time minimal!
        """
        start = time.perf_counter()
        
        if status:
            self.dropout_count += 1
        
        # Input samples (mono) - make a copy to avoid issues
        if indata.ndim > 1:
            input_samples = indata[:, 0].copy()
        else:
            input_samples = indata.flatten().copy()
        
        # Output buffer - start with pass-through (ALWAYS hear yourself)
        output = input_samples.copy()
        
        # Try to acquire lock without blocking for too long
        acquired = self.lock.acquire(blocking=True, timeout=0.005)  # 5ms timeout
        
        if acquired:
            try:
                # -------------------------------------------------------------
                # INPUT LEVEL METERING (always, every callback)
                # -------------------------------------------------------------
                rms = float(np.sqrt(np.mean(input_samples ** 2)))
                # Fast attack, slow decay
                if rms > self.input_level:
                    self.input_level = self.input_level * 0.4 + rms * 0.6
                else:
                    self.input_level = self.input_level * 0.93 + rms * 0.07
                # Peak hold: ~1.5s before decaying
                if rms >= self.input_peak:
                    self.input_peak = rms
                    self._peak_hold_frames = 0
                else:
                    self._peak_hold_frames += 1
                    if self._peak_hold_frames > 258:  # ~1.5s at 172 callbacks/sec
                        self.input_peak = max(self.input_peak * 0.994, rms)

                # -------------------------------------------------------------
                # REACTIVE SCENE COLLAPSE
                # -------------------------------------------------------------
                if (self.collapse_enabled
                        and self.collapse_scene_id is not None
                        and self.collapse_scene_id in self.scenes
                        and self.state == LooperState.PLAYING):
                    if rms < self.collapse_threshold:
                        self._silence_frames += frames
                        silence_secs = self._silence_frames / SAMPLE_RATE
                        if silence_secs >= self.collapse_timeout and not self._collapse_triggered:
                            self.pending_scene = self.scenes[self.collapse_scene_id]
                            self._collapse_triggered = True
                    else:
                        self._silence_frames = 0
                        self._collapse_triggered = False

                # -------------------------------------------------------------
                # STATE: RECORDING_MASTER
                # -------------------------------------------------------------
                if self.state == LooperState.RECORDING_MASTER:
                    # Record input to buffer
                    end_pos = self.recording_position + frames
                    if end_pos <= self.max_samples:
                        self.recording_buffer[self.recording_position:end_pos] = input_samples
                        self.recording_position = end_pos
                
                # -------------------------------------------------------------
                # STATE: PLAYING / OVERDUB_ARMED / RECORDING_OVERDUB
                # -------------------------------------------------------------
                elif self.state in (LooperState.PLAYING, 
                                    LooperState.OVERDUB_ARMED, 
                                    LooperState.RECORDING_OVERDUB):
                    
                    if self.master_length > 0:
                        # Crossfade samples for smooth loop transition
                        xfade_samples = min(int(0.008 * SAMPLE_RATE), self.master_length // 8)  # 8ms
                        
                        # Mix all layers at current position into separate buffer
                        loop_output = np.zeros(frames, dtype=np.float32)
                        
                        for layer in self.layers:
                            if layer.is_playing and layer.length > 0:
                                for i in range(frames):
                                    pos = (self.master_position + i) % self.master_length
                                    
                                    if pos < layer.length:
                                        sample = layer.buffer[pos] * layer.volume
                                        
                                        # Apply crossfade near loop boundary (only for overdubs)
                                        if layer.id > 0 and xfade_samples > 0:
                                            if pos < xfade_samples:
                                                # Near start: fade in
                                                fade = pos / xfade_samples
                                                sample *= fade
                                            elif pos >= layer.length - xfade_samples:
                                                # Near end: fade out
                                                fade = (layer.length - pos) / xfade_samples
                                                sample *= fade
                                        
                                        loop_output[i] += sample
                        
                        # Apply master volume to loop output and add to pass-through
                        output += loop_output * self.master_volume
                        
                        # Check for loop restart (position wrapping)
                        old_position = self.master_position
                        self.master_position = (self.master_position + frames) % self.master_length
                        loop_restarted = self.master_position < old_position

                        # Apply pending scene at loop restart (PLAYING state only)
                        if loop_restarted and self.pending_scene is not None and self.state == LooperState.PLAYING:
                            self._apply_scene(self.pending_scene)
                            self.pending_scene = None

                        # Handle OVERDUB_ARMED → RECORDING_OVERDUB transition
                        if self.state == LooperState.OVERDUB_ARMED and loop_restarted:
                            self.state = LooperState.RECORDING_OVERDUB
                            self.recording_buffer = np.zeros(self.master_length, dtype=np.float32)
                            self.recording_position = 0
                            
                            # Record only the samples that belong to the NEW loop cycle
                            # (the ones after position wrapped to 0)
                            for i in range(frames):
                                pos = (old_position + i) % self.master_length
                                # Only record if we've wrapped past the boundary
                                if old_position + i >= self.master_length:
                                    self.recording_buffer[pos] = input_samples[i]
                            self.recording_position = self.master_position
                        
                        # Handle RECORDING_OVERDUB (normal recording, after start)
                        elif self.state == LooperState.RECORDING_OVERDUB:
                            if loop_restarted:
                                # Loop completed - finalize
                                # First, record the remaining samples up to the boundary
                                for i in range(frames):
                                    pos = (old_position + i) % self.master_length
                                    # Only record up to the wrap point
                                    if old_position + i < self.master_length:
                                        self.recording_buffer[pos] += input_samples[i]
                                
                                self._finalize_overdub()
                            else:
                                # Normal recording
                                for i in range(frames):
                                    pos = (old_position + i) % self.master_length
                                    self.recording_buffer[pos] += input_samples[i]
                                
                                self.recording_position += frames
                
                # -------------------------------------------------------------
                # Soft limiting to prevent clipping
                # -------------------------------------------------------------
                max_val = np.abs(output).max()
                if max_val > 0.95:
                    output *= (0.95 / max_val)
            
            finally:
                self.lock.release()
        else:
            # Lock not acquired - just output pass-through (already in output)
            self.dropout_count += 1
        
        # Write to ALL output channels (handle mono and stereo)
        if outdata.ndim > 1:
            for ch in range(outdata.shape[1]):
                outdata[:, ch] = output
        else:
            outdata[:] = output
        
        self.callback_time = time.perf_counter() - start
    
    def _finalize_overdub(self):
        """Create new layer from recorded overdub. Called within lock."""
        layer_id = len(self.layers)
        name = f"Overdub {layer_id}"
        
        # Copy buffer - crossfade is applied at playback time
        buffer = self.recording_buffer[:self.master_length].copy()
        
        layer = LoopLayer(layer_id, name, buffer)
        self.layers.append(layer)
        
        self.state = LooperState.PLAYING
        print(f"✓ {name} recorded ({self.master_length / SAMPLE_RATE:.1f}s)")
    
    # -------------------------------------------------------------------------
    # COMMANDS (Called from web interface)
    # -------------------------------------------------------------------------
    
    def start_recording(self) -> bool:
        """Start recording master loop. Returns success."""
        with self.lock:
            if self.state != LooperState.IDLE:
                print("✗ Cannot start recording: not in IDLE state")
                return False
            
            self.recording_buffer = np.zeros(self.max_samples, dtype=np.float32)
            self.recording_position = 0
            self.state = LooperState.RECORDING_MASTER
            
            print("● Recording master loop...")
            return True
    
    def stop_recording(self) -> bool:
        """Stop recording master loop. Returns success."""
        with self.lock:
            if self.state != LooperState.RECORDING_MASTER:
                print("✗ Cannot stop recording: not recording master")
                return False
            
            recorded_length = self.recording_position
            recorded_duration = recorded_length / SAMPLE_RATE
            
            # Quantize to nearest bar if enabled
            if self.quantize_enabled and self.bpm > 0:
                beat_duration = 60.0 / self.bpm  # seconds per beat
                bar_duration = beat_duration * self.beats_per_bar  # seconds per bar
                
                # Calculate nearest number of bars
                num_bars = round(recorded_duration / bar_duration)
                num_bars = max(1, num_bars)  # At least 1 bar
                
                quantized_duration = num_bars * bar_duration
                quantized_length = int(quantized_duration * SAMPLE_RATE)
                
                print(f"  Recorded: {recorded_duration:.2f}s → Quantized: {quantized_duration:.2f}s ({num_bars} bars)")
                
                self.master_length = quantized_length
            else:
                self.master_length = recorded_length
            
            # Create master layer buffer (no modification - crossfade handled at playback)
            buffer = self.recording_buffer[:self.master_length].copy()
            
            layer = LoopLayer(0, "Master", buffer)
            self.layers = [layer]
            
            self.master_position = 0
            self.state = LooperState.PLAYING
            
            duration = self.master_length / SAMPLE_RATE
            print(f"✓ Master loop recorded: {duration:.2f}s")
            return True
    
    def arm_overdub(self) -> bool:
        """Arm overdub recording (waits for loop start). Returns success."""
        with self.lock:
            if self.state != LooperState.PLAYING:
                print(f"✗ Cannot arm overdub: state is {self.state.value}")
                return False
            
            self.state = LooperState.OVERDUB_ARMED
            print("● Overdub armed, waiting for loop start...")
            return True
    
    def cancel_overdub(self) -> bool:
        """Cancel armed overdub. Returns success."""
        with self.lock:
            if self.state != LooperState.OVERDUB_ARMED:
                return False
            
            self.state = LooperState.PLAYING
            print("✓ Overdub cancelled")
            return True
    
    def set_layer_volume(self, layer_id: int, volume: float) -> bool:
        """Set volume for a layer. Returns success."""
        with self.lock:
            if 0 <= layer_id < len(self.layers):
                self.layers[layer_id].volume = max(0.0, min(1.0, volume))
                return True
            return False
    
    def set_master_volume(self, volume: float) -> bool:
        """Set master output volume (0.0 to 1.0). Returns success."""
        with self.lock:
            self.master_volume = max(0.0, min(1.0, volume))
            print(f"✓ Master volume: {int(self.master_volume * 100)}%")
            return True
    
    def toggle_layer(self, layer_id: int) -> bool:
        """Toggle play/pause for a layer. Returns success."""
        with self.lock:
            if 0 <= layer_id < len(self.layers):
                self.layers[layer_id].is_playing = not self.layers[layer_id].is_playing
                status = "playing" if self.layers[layer_id].is_playing else "muted"
                print(f"✓ {self.layers[layer_id].name} → {status}")
                return True
            return False
    
    def delete_layer(self, layer_id: int) -> bool:
        """Delete a layer (cannot delete master). Returns success."""
        with self.lock:
            # Cannot delete master (layer 0)
            if layer_id <= 0 or layer_id >= len(self.layers):
                print(f"✗ Cannot delete layer {layer_id}")
                return False
            
            name = self.layers[layer_id].name
            del self.layers[layer_id]
            
            # Renumber remaining layers
            for i, layer in enumerate(self.layers):
                layer.id = i
            
            print(f"✓ Deleted {name}")
            return True

    def rename_layer(self, layer_id: int, name: str) -> bool:
        """Rename a layer."""
        with self.lock:
            if 0 <= layer_id < len(self.layers):
                self.layers[layer_id].name = name.strip() or self.layers[layer_id].name
                return True
            return False

    def set_layer_color(self, layer_id: int, color: str) -> bool:
        """Set color for a layer."""
        with self.lock:
            if 0 <= layer_id < len(self.layers):
                self.layers[layer_id].color = color
                return True
            return False

    def clear_all(self) -> bool:
        """Clear all loops and reset to IDLE state."""
        with self.lock:
            self.layers = []
            self.master_length = 0
            self.master_position = 0
            self.state = LooperState.IDLE
            print("✓ All loops cleared")
            return True

    # -------------------------------------------------------------------------
    # SCENE MANAGEMENT
    # -------------------------------------------------------------------------

    def save_scene(self, name: str) -> dict:
        """Save current layer state as a named scene."""
        with self.lock:
            if not self.layers:
                return {'success': False, 'error': 'No layers to save'}

            scene_id = self._next_scene_id
            self._next_scene_id += 1

            layer_states = [
                {'id': layer.id, 'is_playing': layer.is_playing, 'volume': layer.volume}
                for layer in self.layers
            ]

            scene = {
                'id': scene_id,
                'name': name.strip() or f'Scene {scene_id}',
                'layer_states': layer_states,
            }
            self.scenes[scene_id] = scene
            print(f"✓ Scene saved: '{scene['name']}' ({len(layer_states)} layers)")
            return {'success': True, 'scene': scene}

    def _apply_scene(self, scene: dict):
        """Apply a scene's layer states. Must be called within lock."""
        layer_map = {layer.id: layer for layer in self.layers}
        for state in scene['layer_states']:
            layer = layer_map.get(state['id'])
            if layer:
                layer.is_playing = state['is_playing']
                layer.volume = state['volume']
        print(f"✓ Scene applied: '{scene['name']}'")

    def load_scene(self, scene_id: int, quantized: bool = True) -> bool:
        """Load a scene. If quantized=True and playing, waits for next loop restart."""
        with self.lock:
            scene = self.scenes.get(scene_id)
            if not scene:
                return False

            if quantized and self.state == LooperState.PLAYING and self.master_length > 0:
                self.pending_scene = scene
                print(f"✓ Scene '{scene['name']}' scheduled for next loop restart")
            else:
                self._apply_scene(scene)
            return True

    def delete_scene(self, scene_id: int) -> bool:
        """Delete a scene."""
        with self.lock:
            if scene_id not in self.scenes:
                return False
            name = self.scenes[scene_id]['name']
            del self.scenes[scene_id]
            if self.pending_scene and self.pending_scene['id'] == scene_id:
                self.pending_scene = None
            print(f"✓ Scene deleted: '{name}'")
            return True

    def rename_scene(self, scene_id: int, name: str) -> bool:
        """Rename a scene."""
        with self.lock:
            if scene_id not in self.scenes:
                return False
            self.scenes[scene_id]['name'] = name.strip() or self.scenes[scene_id]['name']
            return True

    def set_collapse_scene(self, scene_id) -> bool:
        """Designate a scene as the idle/collapse target. Pass None to clear."""
        with self.lock:
            if scene_id is not None and scene_id not in self.scenes:
                return False
            self.collapse_scene_id = scene_id
            self._silence_frames = 0
            self._collapse_triggered = False
            status = f"scene {scene_id}" if scene_id else "cleared"
            print(f"✓ Collapse scene {status}")
            return True

    def set_collapse_enabled(self, enabled: bool, timeout: float = None) -> bool:
        """Enable/disable reactive scene collapse, optionally update timeout."""
        with self.lock:
            self.collapse_enabled = enabled
            if timeout is not None:
                self.collapse_timeout = max(1.0, float(timeout))
            self._silence_frames = 0
            self._collapse_triggered = False
            print(f"✓ Collapse {'enabled' if enabled else 'disabled'} (timeout: {self.collapse_timeout}s)")
            return True

    # -------------------------------------------------------------------------
    # SESSION PERSISTENCE
    # -------------------------------------------------------------------------

    def save_session(self, name: str) -> dict:
        """Save current session (all loops + metadata) to disk."""
        with self.lock:
            if not self.layers or self.master_length == 0:
                return {'success': False, 'error': 'Nothing to save'}
            if self.state in (LooperState.RECORDING_MASTER, LooperState.RECORDING_OVERDUB):
                return {'success': False, 'error': 'Cannot save while recording'}

            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            safe_name = ''.join(c if c.isalnum() or c in '-_ ' else '' for c in name.strip())
            folder_name = f"{timestamp}_{safe_name}" if safe_name else timestamp
            session_dir = SESSIONS_DIR / folder_name

            meta = {
                'name': name.strip() or timestamp,
                'created_at': datetime.now().isoformat(),
                'bpm': self.bpm,
                'beats_per_bar': self.beats_per_bar,
                'master_volume': self.master_volume,
                'master_length': self.master_length,
                'layers': [
                    {
                        'id': l.id,
                        'name': l.name,
                        'color': l.color,
                        'volume': l.volume,
                        'is_playing': l.is_playing,
                    }
                    for l in self.layers
                ],
                'scenes': {str(k): v for k, v in self.scenes.items()},
                'next_scene_id': self._next_scene_id,
            }
            buffers = [(l.id, l.buffer[:l.length].copy()) for l in self.layers]

        # Write to disk without lock
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
            for layer_id, buf in buffers:
                np.save(str(session_dir / f'layer_{layer_id}.npy'), buf)
            print(f"✓ Session saved: {folder_name} ({len(buffers)} layers)")
            return {'success': True, 'session_id': folder_name, 'name': meta['name']}
        except Exception as e:
            print(f"✗ Session save failed: {e}")
            return {'success': False, 'error': str(e)}

    def load_session(self, session_id: str) -> dict:
        """Load a session from disk, replacing current state."""
        session_dir = SESSIONS_DIR / session_id

        if not session_dir.exists():
            return {'success': False, 'error': 'Session not found'}

        try:
            meta = json.loads((session_dir / 'meta.json').read_text())

            buffers = {}
            for layer_meta in meta['layers']:
                layer_id = layer_meta['id']
                npy_path = session_dir / f'layer_{layer_id}.npy'
                if npy_path.exists():
                    buffers[layer_id] = np.load(str(npy_path))

            if 0 not in buffers:
                return {'success': False, 'error': 'Master layer audio missing'}

        except Exception as e:
            return {'success': False, 'error': f'Failed to read session: {e}'}

        with self.lock:
            self.layers = []
            for layer_meta in meta['layers']:
                lid = layer_meta['id']
                if lid not in buffers:
                    continue
                layer = LoopLayer(lid, layer_meta['name'], buffers[lid])
                layer.color = layer_meta.get('color', LAYER_COLORS[lid % len(LAYER_COLORS)])
                layer.volume = layer_meta.get('volume', 1.0)
                layer.is_playing = layer_meta.get('is_playing', True)
                self.layers.append(layer)

            self.master_length = meta.get('master_length', len(buffers[0]))
            self.master_position = 0
            self.bpm = meta.get('bpm', 120.0)
            self.beats_per_bar = meta.get('beats_per_bar', 4)
            self.master_volume = meta.get('master_volume', 0.8)

            raw_scenes = meta.get('scenes', {})
            self.scenes = {int(k): v for k, v in raw_scenes.items()}
            self._next_scene_id = meta.get('next_scene_id', len(self.scenes) + 1)
            self.pending_scene = None

            self.state = LooperState.PLAYING
            print(f"✓ Session loaded: {meta['name']} ({len(self.layers)} layers)")

        return {'success': True, 'name': meta['name']}

    def delete_session(self, session_id: str) -> dict:
        """Delete a session from disk."""
        session_dir = SESSIONS_DIR / session_id
        if not session_dir.exists():
            return {'success': False, 'error': 'Session not found'}
        try:
            import shutil
            shutil.rmtree(session_dir)
            print(f"✓ Session deleted: {session_id}")
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    @staticmethod
    def list_sessions() -> list:
        """Return list of saved sessions sorted by most recent first."""
        sessions = []
        for session_dir in sorted(SESSIONS_DIR.iterdir(), reverse=True):
            if not session_dir.is_dir():
                continue
            meta_path = session_dir / 'meta.json'
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
                sessions.append({
                    'id': session_dir.name,
                    'name': meta.get('name', session_dir.name),
                    'created_at': meta.get('created_at', ''),
                    'bpm': meta.get('bpm', 0),
                    'layer_count': len(meta.get('layers', [])),
                    'scene_count': len(meta.get('scenes', {})),
                })
            except Exception:
                continue
        return sessions

    def set_bpm(self, bpm: float) -> bool:
        """Set tempo in BPM."""
        with self.lock:
            if 30 <= bpm <= 300:
                self.bpm = bpm
                print(f"✓ BPM set to {bpm:.1f}")
                return True
            return False
    
    def set_beats_per_bar(self, beats: int) -> bool:
        """Set time signature (beats per bar)."""
        with self.lock:
            if 1 <= beats <= 12:
                self.beats_per_bar = beats
                print(f"✓ Time signature: {beats}/4")
                return True
            return False
    
    def set_quantize(self, enabled: bool) -> bool:
        """Enable/disable quantization."""
        with self.lock:
            self.quantize_enabled = enabled
            status = "enabled" if enabled else "disabled"
            print(f"✓ Quantization {status}")
            return True
    
    # -------------------------------------------------------------------------
    # WAVEFORM & TRIM
    # -------------------------------------------------------------------------
    
    def get_waveform(self, num_points: int = 600) -> list:
        """
        Generate waveform data for visualization.
        Returns list of {min, max} values for each segment.
        
        IMPORTANT: Minimizes lock time by copying buffer first.
        """
        # Quick copy while holding lock
        with self.lock:
            if len(self.layers) == 0:
                return []
            
            buffer = self.layers[0].buffer.copy()  # Copy!
            length = self.layers[0].length
        
        # Process WITHOUT lock
        if length == 0:
            return []
        
        # Calculate samples per point
        samples_per_point = length / num_points
        waveform = []
        
        for i in range(num_points):
            start_idx = int(i * samples_per_point)
            end_idx = int((i + 1) * samples_per_point)
            end_idx = min(end_idx, length)
            
            if start_idx >= end_idx:
                waveform.append({'min': 0, 'max': 0})
                continue
            
            segment = buffer[start_idx:end_idx]
            waveform.append({
                'min': float(np.min(segment)),
                'max': float(np.max(segment))
            })
        
        return waveform
    
    def apply_trim(self, start_time: float, end_time: float) -> bool:
        """
        Apply trim to master loop.
        Only allowed when no overdubs exist.
        Bypasses quantization.
        """
        with self.lock:
            # Only allow trimming master when no overdubs
            if len(self.layers) != 1:
                print("✗ Cannot trim: overdubs exist")
                return False
            
            if self.state not in (LooperState.PLAYING, LooperState.IDLE):
                print("✗ Cannot trim: invalid state")
                return False
            
            # Convert times to samples
            start_sample = int(start_time * SAMPLE_RATE)
            end_sample = int(end_time * SAMPLE_RATE)
            
            # Validate
            if start_sample < 0:
                start_sample = 0
            if end_sample > self.master_length:
                end_sample = self.master_length
            if start_sample >= end_sample:
                print("✗ Invalid trim range")
                return False
            
            # Calculate new length
            new_length = end_sample - start_sample
            
            # Create new trimmed buffer (no modification - crossfade handled at playback)
            old_buffer = self.layers[0].buffer
            new_buffer = old_buffer[start_sample:end_sample].copy()
            
            # Update master layer
            self.layers[0] = LoopLayer(0, "Master", new_buffer)
            self.master_length = new_length
            self.master_position = 0  # Reset position
            
            duration = new_length / SAMPLE_RATE
            print(f"✓ Trim applied: {start_time:.2f}s - {end_time:.2f}s = {duration:.2f}s")
            return True
    
    def can_trim(self) -> bool:
        """Check if trimming is currently allowed."""
        with self.lock:
            return (len(self.layers) == 1 and
                    self.state in (LooperState.PLAYING,) and
                    self.master_length > 0)

    def auto_trim_silence(self, threshold_db: float = -40.0) -> bool:
        """
        Trim silence from start and end of master loop.
        Scans for first/last window exceeding threshold_db.
        Only allowed when no overdubs exist.
        """
        with self.lock:
            if len(self.layers) != 1 or self.master_length == 0:
                print("✗ Auto-trim: no master loop or overdubs exist")
                return False
            if self.state not in (LooperState.PLAYING,):
                print("✗ Auto-trim: invalid state")
                return False
            buffer = self.layers[0].buffer[:self.master_length].copy()

        threshold_linear = 10 ** (threshold_db / 20.0)
        window = max(int(0.005 * SAMPLE_RATE), 1)  # 5ms window
        abs_buf = np.abs(buffer)

        start_sample = 0
        for i in range(0, len(buffer) - window, window):
            if np.sqrt(np.mean(abs_buf[i:i + window] ** 2)) > threshold_linear:
                start_sample = max(0, i - window)
                break

        end_sample = len(buffer)
        for i in range(len(buffer) - window, 0, -window):
            if np.sqrt(np.mean(abs_buf[i:i + window] ** 2)) > threshold_linear:
                end_sample = min(len(buffer), i + 2 * window)
                break

        if start_sample >= end_sample:
            print("✗ Auto-trim: entire buffer below threshold — nothing to trim")
            return False

        start_time = start_sample / SAMPLE_RATE
        end_time = end_sample / SAMPLE_RATE
        print(f"  Auto-trim detected: {start_time:.3f}s → {end_time:.3f}s")
        return self.apply_trim(start_time, end_time)

    def detect_tempo(self) -> dict:
        """
        Detect tempo from the master loop using librosa.
        Returns dict with bpm, confidence, and beats array.
        """
        if not LIBROSA_AVAILABLE:
            return {
                'success': False,
                'error': 'librosa not installed',
                'bpm': 0,
                'confidence': 0
            }
        
        # Quick copy of audio data while holding lock
        with self.lock:
            if len(self.layers) == 0 or self.master_length == 0:
                return {
                    'success': False,
                    'error': 'No audio recorded',
                    'bpm': 0,
                    'confidence': 0
                }
            
            audio = self.layers[0].buffer[:self.master_length].copy()
        
        # Process WITHOUT lock (this takes time)
        try:
            # Convert to float64 for librosa
            audio_64 = audio.astype(np.float64)
            
            # Detect tempo using librosa's beat tracker
            # This returns tempo estimate and beat frames
            tempo, beat_frames = librosa.beat.beat_track(
                y=audio_64, 
                sr=SAMPLE_RATE,
                start_bpm=120,  # Initial guess
                units='frames'
            )
            
            # Handle both old and new librosa versions
            # New versions return an array, old versions return a scalar
            if hasattr(tempo, '__len__'):
                bpm = float(tempo[0]) if len(tempo) > 0 else 0
            else:
                bpm = float(tempo)
            
            # Calculate confidence based on beat regularity
            if len(beat_frames) >= 2:
                # Convert frames to times
                beat_times = librosa.frames_to_time(beat_frames, sr=SAMPLE_RATE)
                
                # Calculate intervals between beats
                intervals = np.diff(beat_times)
                
                if len(intervals) > 0:
                    # Confidence = how regular the intervals are (low std = high confidence)
                    mean_interval = np.mean(intervals)
                    std_interval = np.std(intervals)
                    
                    # Coefficient of variation (lower = more regular)
                    if mean_interval > 0:
                        cv = std_interval / mean_interval
                        # Convert to confidence percentage (cv of 0 = 100%, cv of 0.5 = 0%)
                        confidence = max(0, min(100, (1 - cv * 2) * 100))
                    else:
                        confidence = 0
                else:
                    confidence = 50  # Not enough data
            else:
                confidence = 30  # Very few beats detected
            
            # Round BPM to nearest integer
            bpm = round(bpm)
            
            # Sanity check
            if bpm < 30 or bpm > 300:
                return {
                    'success': False,
                    'error': f'Detected tempo ({bpm}) out of range',
                    'bpm': 0,
                    'confidence': 0
                }
            
            print(f"✓ Tempo detected: {bpm} BPM (confidence: {confidence:.0f}%)")
            
            return {
                'success': True,
                'bpm': bpm,
                'confidence': round(confidence),
                'beat_count': len(beat_frames)
            }
            
        except Exception as e:
            print(f"✗ Tempo detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'bpm': 0,
                'confidence': 0
            }
    
    # -------------------------------------------------------------------------
    # EXPORT FUNCTIONS
    # -------------------------------------------------------------------------
    
    def _generate_filename(self, suffix: str, fmt: str) -> str:
        """
        Generate filename with timestamp.
        Format: Loop_YYYY-MM-DD_HH-MM_{suffix}.{format}
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        return f"Loop_{timestamp}_{suffix}.{fmt}"
    
    def _normalize(self, samples: np.ndarray, target_db: float = -1.0) -> np.ndarray:
        """
        Normalize audio to target peak level.
        
        Args:
            samples: Audio samples (float32, -1 to 1)
            target_db: Target peak level in dB (default -1dB)
        
        Returns:
            Normalized samples
        """
        peak = np.abs(samples).max()
        if peak == 0:
            return samples
        
        # Convert dB to linear: -1dB ≈ 0.891
        target_linear = 10 ** (target_db / 20)
        
        # Only attenuate if peak exceeds target (don't amplify quiet signals)
        if peak > target_linear:
            return samples * (target_linear / peak)
        
        return samples
    
    def _numpy_to_audiosegment(self, samples: np.ndarray) -> 'AudioSegment':
        """
        Convert numpy float32 array to pydub AudioSegment.
        
        Args:
            samples: Audio samples (float32, -1 to 1)
        
        Returns:
            pydub AudioSegment
        """
        # Normalize to prevent clipping
        normalized = self._normalize(samples)
        
        # Convert float32 [-1, 1] → int16 [-32768, 32767]
        int16_samples = (normalized * 32767).astype(np.int16)
        
        # Create AudioSegment from raw bytes
        audio_segment = AudioSegment(
            data=int16_samples.tobytes(),
            sample_width=2,  # 16-bit = 2 bytes
            frame_rate=SAMPLE_RATE,
            channels=CHANNELS
        )
        
        return audio_segment
    
    def get_export_info(self) -> dict:
        """
        Get export availability and estimated file sizes.
        """
        with self.lock:
            can_export = (len(self.layers) > 0 and 
                         self.master_length > 0 and
                         self.state not in (LooperState.RECORDING_MASTER, 
                                           LooperState.RECORDING_OVERDUB))
            
            duration = self.master_length / SAMPLE_RATE if self.master_length > 0 else 0
            num_layers = len(self.layers)
        
        # Estimate file sizes
        # MP3: ~192kbps = 24KB/sec
        # WAV: 44100 * 2 bytes * 1 channel = 88.2KB/sec
        mp3_size = int(duration * 24000) if duration > 0 else 0
        wav_size = int(duration * 88200) if duration > 0 else 0
        
        return {
            'can_export': can_export,
            'pydub_available': PYDUB_AVAILABLE,
            'ffmpeg_available': FFMPEG_AVAILABLE,
            'duration': duration,
            'num_layers': num_layers,
            'estimates': {
                'mp3_mixed': mp3_size,
                'wav_mixed': wav_size,
                'mp3_layers_zip': mp3_size * num_layers,
                'wav_layers_zip': wav_size * num_layers,
            }
        }
    
    def export_mixed(self, fmt: str = "mp3") -> tuple:
        """
        Export all layers mixed together, respecting volume settings.
        
        Args:
            fmt: "mp3" or "wav"
        
        Returns:
            (audio_bytes, filename, content_type) or (None, error_message, None)
        """
        if not PYDUB_AVAILABLE:
            return (None, "pydub not installed", None)
        
        if fmt == "mp3" and not FFMPEG_AVAILABLE:
            return (None, "ffmpeg not installed (required for MP3)", None)
        
        # Copy layer data while holding lock
        with self.lock:
            if len(self.layers) == 0 or self.master_length == 0:
                return (None, "No audio recorded", None)
            
            if self.state in (LooperState.RECORDING_MASTER, LooperState.RECORDING_OVERDUB):
                return (None, "Cannot export while recording", None)
            
            # Copy all layer buffers and volumes
            layer_data = []
            for layer in self.layers:
                layer_data.append({
                    'buffer': layer.buffer[:self.master_length].copy(),
                    'volume': layer.volume,
                    'is_playing': layer.is_playing
                })
            master_length = self.master_length
            master_vol = self.master_volume
        
        # Process WITHOUT lock
        try:
            # Mix all layers
            mixed = np.zeros(master_length, dtype=np.float32)
            for layer in layer_data:
                if layer['is_playing']:
                    mixed += layer['buffer'] * layer['volume']
            
            # Apply master volume
            mixed *= master_vol
            
            # Convert to AudioSegment
            audio_segment = self._numpy_to_audiosegment(mixed)
            
            # Add metadata
            audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE)
            
            # Export to bytes
            buffer = BytesIO()
            
            if fmt == "mp3":
                audio_segment.export(
                    buffer, 
                    format="mp3",
                    bitrate=EXPORT_MP3_BITRATE,
                    tags={'title': self._generate_filename('mixed', 'mp3').replace('.mp3', '')}
                )
                content_type = "audio/mpeg"
            else:  # wav
                audio_segment.export(buffer, format="wav")
                content_type = "audio/wav"
            
            buffer.seek(0)
            filename = self._generate_filename("mixed", fmt)
            
            print(f"✓ Exported mixed: {filename} ({len(buffer.getvalue())} bytes)")
            
            return (buffer.getvalue(), filename, content_type)
            
        except Exception as e:
            print(f"✗ Export failed: {e}")
            return (None, str(e), None)
    
    def export_layer(self, layer_id: int, fmt: str = "mp3") -> tuple:
        """
        Export a single layer with its volume applied.
        
        Args:
            layer_id: Index of layer to export
            fmt: "mp3" or "wav"
        
        Returns:
            (audio_bytes, filename, content_type) or (None, error_message, None)
        """
        if not PYDUB_AVAILABLE:
            return (None, "pydub not installed", None)
        
        if fmt == "mp3" and not FFMPEG_AVAILABLE:
            return (None, "ffmpeg not installed (required for MP3)", None)
        
        # Copy layer data while holding lock
        with self.lock:
            if layer_id < 0 or layer_id >= len(self.layers):
                return (None, f"Invalid layer ID: {layer_id}", None)
            
            if self.state in (LooperState.RECORDING_MASTER, LooperState.RECORDING_OVERDUB):
                return (None, "Cannot export while recording", None)
            
            layer = self.layers[layer_id]
            buffer = layer.buffer[:layer.length].copy()
            volume = layer.volume
            name = layer.name.replace(" ", "_").lower()
        
        # Process WITHOUT lock
        try:
            # Apply volume
            samples = buffer * volume
            
            # Convert to AudioSegment
            audio_segment = self._numpy_to_audiosegment(samples)
            
            # Export to bytes
            output_buffer = BytesIO()
            
            if fmt == "mp3":
                audio_segment.export(
                    output_buffer, 
                    format="mp3",
                    bitrate=EXPORT_MP3_BITRATE
                )
                content_type = "audio/mpeg"
            else:  # wav
                audio_segment.export(output_buffer, format="wav")
                content_type = "audio/wav"
            
            output_buffer.seek(0)
            filename = self._generate_filename(name, fmt)
            
            print(f"✓ Exported layer {layer_id}: {filename}")
            
            return (output_buffer.getvalue(), filename, content_type)
            
        except Exception as e:
            print(f"✗ Export layer failed: {e}")
            return (None, str(e), None)
    
    def export_all_layers(self, fmt: str = "mp3") -> tuple:
        """
        Export all layers as separate files in a ZIP archive.
        
        Args:
            fmt: "mp3" or "wav"
        
        Returns:
            (zip_bytes, filename, content_type) or (None, error_message, None)
        """
        if not PYDUB_AVAILABLE:
            return (None, "pydub not installed", None)
        
        if fmt == "mp3" and not FFMPEG_AVAILABLE:
            return (None, "ffmpeg not installed (required for MP3)", None)
        
        # Copy layer data while holding lock
        with self.lock:
            if len(self.layers) == 0:
                return (None, "No audio recorded", None)
            
            if self.state in (LooperState.RECORDING_MASTER, LooperState.RECORDING_OVERDUB):
                return (None, "Cannot export while recording", None)
            
            # Copy all layer data
            layer_data = []
            for layer in self.layers:
                layer_data.append({
                    'id': layer.id,
                    'name': layer.name,
                    'buffer': layer.buffer[:layer.length].copy(),
                    'volume': layer.volume
                })
        
        # Process WITHOUT lock
        try:
            # Create ZIP file in memory
            zip_buffer = BytesIO()
            
            with ZipFile(zip_buffer, 'w') as zip_file:
                for layer in layer_data:
                    # Apply volume
                    samples = layer['buffer'] * layer['volume']
                    
                    # Convert to AudioSegment
                    audio_segment = self._numpy_to_audiosegment(samples)
                    
                    # Export to bytes
                    audio_buffer = BytesIO()
                    if fmt == "mp3":
                        audio_segment.export(audio_buffer, format="mp3", bitrate=EXPORT_MP3_BITRATE)
                    else:
                        audio_segment.export(audio_buffer, format="wav")
                    
                    # Add to ZIP
                    layer_name = layer['name'].replace(" ", "_").lower()
                    zip_file.writestr(f"{layer_name}.{fmt}", audio_buffer.getvalue())
            
            zip_buffer.seek(0)
            filename = self._generate_filename("layers", "zip")
            
            print(f"✓ Exported {len(layer_data)} layers to {filename}")
            
            return (zip_buffer.getvalue(), filename, "application/zip")
            
        except Exception as e:
            print(f"✗ Export layers failed: {e}")
            return (None, str(e), None)
    
    # -------------------------------------------------------------------------
    # STATE QUERY
    # -------------------------------------------------------------------------
    
    def get_state(self) -> dict:
        """Get current state for web client. Minimizes lock time."""
        # Quick snapshot while holding lock
        with self.lock:
            state = self.state.value
            master_length = self.master_length
            master_position = self.master_position
            recording_position = self.recording_position
            current_state = self.state
            bpm = self.bpm
            beats_per_bar = self.beats_per_bar
            quantize_enabled = self.quantize_enabled
            master_volume = self.master_volume
            callback_time = self.callback_time
            dropout_count = self.dropout_count
            layers_data = [layer.to_dict() for layer in self.layers]
            num_layers = len(self.layers)
            input_level = self.input_level
            input_peak = self.input_peak
            scenes_data = list(self.scenes.values())
            pending_scene_id = self.pending_scene['id'] if self.pending_scene else None
            collapse_enabled = self.collapse_enabled
            collapse_scene_id = self.collapse_scene_id
            collapse_timeout = self.collapse_timeout
        
        # Compute derived values WITHOUT lock
        position_ratio = 0.0
        current_time = 0.0
        master_duration = master_length / SAMPLE_RATE if master_length > 0 else 0
        
        if current_state == LooperState.RECORDING_MASTER:
            current_time = recording_position / SAMPLE_RATE
        elif master_length > 0:
            position_ratio = master_position / master_length
            current_time = master_position / SAMPLE_RATE
        
        # Calculate beat positions (no lock needed)
        beat_positions = []
        if bpm > 0 and master_duration > 0:
            beat_duration = 60.0 / bpm
            beat_time = 0.0
            beat_index = 0
            while beat_time < master_duration:
                beat_positions.append({
                    'time': beat_time,
                    'ratio': beat_time / master_duration,
                    'is_downbeat': (beat_index % beats_per_bar) == 0
                })
                beat_time += beat_duration
                beat_index += 1
        
        # Check if trim is available
        can_trim = (num_layers == 1 and 
                   current_state == LooperState.PLAYING and
                   master_length > 0)
        
        # Check if export is available
        can_export = (num_layers > 0 and 
                     master_length > 0 and
                     current_state not in (LooperState.RECORDING_MASTER, 
                                          LooperState.RECORDING_OVERDUB))
        
        return {
            'state': state,
            'master_duration': master_duration,
            'master_volume': master_volume,
            'position_ratio': position_ratio,
            'current_time': current_time,
            'recording_time': recording_position / SAMPLE_RATE if current_state == LooperState.RECORDING_MASTER else 0,
            'layers': layers_data,
            'tempo': {
                'bpm': bpm,
                'beats_per_bar': beats_per_bar,
                'quantize_enabled': quantize_enabled,
                'beat_positions': beat_positions,
            },
            'trim': {
                'can_trim': can_trim,
                'reason': '' if can_trim else ('Add overdubs disabled trimming' if num_layers > 1 else ''),
            },
            'export': {
                'can_export': can_export,
                'pydub_available': PYDUB_AVAILABLE,
                'ffmpeg_available': FFMPEG_AVAILABLE,
            },
            'stats': {
                'callback_time_ms': callback_time * 1000,
                'latency_ms': (BLOCKSIZE / SAMPLE_RATE) * 1000,
                'dropout_count': dropout_count,
            },
            'input_level': input_level,
            'input_peak': input_peak,
            'scenes': {
                'list': scenes_data,
                'pending_id': pending_scene_id,
            },
            'collapse': {
                'enabled': collapse_enabled,
                'scene_id': collapse_scene_id,
                'timeout': collapse_timeout,
            },
        }
    
    # -------------------------------------------------------------------------
    # AUDIO STREAM
    # -------------------------------------------------------------------------
    
    def start_stream(self):
        """Start audio stream."""
        try:
            # Query device info for debugging
            if self.device is not None:
                dev_info = sd.query_devices(self.device)
                print(f"  Device: {dev_info['name']}")
                print(f"  Input channels: {dev_info['max_input_channels']}")
                print(f"  Output channels: {dev_info['max_output_channels']}")
            
            self.stream = sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCKSIZE,
                device=self.device,
                channels=CHANNELS,
                callback=self.audio_callback,
                latency='low'
            )
            self.stream.start()
            
            actual_latency = self.stream.latency
            print(f"✓ Audio stream started")
            print(f"  Sample rate: {SAMPLE_RATE} Hz")
            print(f"  Block size: {BLOCKSIZE} samples")
            print(f"  Latency: {actual_latency[0]*1000:.1f}ms in / {actual_latency[1]*1000:.1f}ms out")
        except Exception as e:
            print(f"✗ Failed to start audio stream: {e}")
            raise
    
    def stop_stream(self):
        """Stop audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("✓ Audio stream stopped")

# =============================================================================
# FLASK APPLICATION
# =============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'looper_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global looper instance
looper: WebLooper = None

# =============================================================================
# HTML TEMPLATE
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>🎸 Guitar Looper</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
            color: #eee;
        }
        
        .container {
            max-width: 700px;
            margin: 0 auto;
        }
        
        /* Header */
        .header {
            text-align: center;
            padding: 20px 0;
        }
        
        .header h1 {
            font-size: 2em;
            margin-bottom: 10px;
            color: #fff;
        }
        
        .status-badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .status-idle { background: #4a5568; }
        .status-recording { background: #e53e3e; animation: pulse 1s infinite; }
        .status-playing { background: #38a169; }
        .status-armed { background: #d69e2e; animation: pulse 1s infinite; }
        .status-countdown { background: #805ad5; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        /* Countdown overlay */
        .countdown-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.85);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        
        .countdown-number {
            font-size: 12rem;
            font-weight: bold;
            color: #fff;
            animation: countdownPop 0.3s ease-out;
        }
        
        .countdown-go {
            font-size: 6rem;
            color: #38a169;
        }
        
        .countdown-bpm {
            font-size: 1.5rem;
            color: #a0aec0;
            margin-top: 20px;
        }
        
        @keyframes countdownPop {
            0% { transform: scale(0.5); opacity: 0; }
            50% { transform: scale(1.1); opacity: 1; }
            100% { transform: scale(1); opacity: 1; }
        }
        
        /* Tempo Section */
        .tempo-section {
            background: #2d3748;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
        }
        
        .tempo-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .tempo-title {
            font-size: 1em;
            color: #a0aec0;
            font-weight: bold;
        }
        
        .tempo-controls {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .bpm-display {
            background: #1a202c;
            padding: 15px 25px;
            border-radius: 10px;
            text-align: center;
            min-width: 120px;
        }
        
        .bpm-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            font-family: "SF Mono", Monaco, monospace;
        }
        
        .bpm-label {
            font-size: 0.8em;
            color: #718096;
            text-transform: uppercase;
        }
        
        .tap-tempo-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 20px 30px;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.15s;
            text-transform: uppercase;
        }
        
        .tap-tempo-btn:hover {
            background: #5a67d8;
            transform: scale(1.02);
        }
        
        .tap-tempo-btn:active {
            transform: scale(0.98);
            background: #4c51bf;
        }
        
        .tap-tempo-btn.tapping {
            animation: tapPulse 0.15s ease-out;
        }
        
        @keyframes tapPulse {
            0% { transform: scale(1); }
            50% { transform: scale(0.95); background: #4c51bf; }
            100% { transform: scale(1); }
        }
        
        .detect-tempo-btn {
            background: #38a169;
            color: white;
            border: none;
            padding: 15px 20px;
            border-radius: 10px;
            font-size: 0.9em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.15s;
            text-transform: uppercase;
        }
        
        .detect-tempo-btn:hover:not(:disabled) {
            background: #2f855a;
            transform: scale(1.02);
        }
        
        .detect-tempo-btn:active:not(:disabled) {
            transform: scale(0.98);
        }
        
        .detect-tempo-btn:disabled {
            background: #4a5568;
            cursor: not-allowed;
            opacity: 0.6;
        }
        
        .detect-tempo-btn.detecting {
            animation: pulse 1s infinite;
        }
        
        .tempo-detect-result {
            background: #1a202c;
            border-radius: 10px;
            padding: 12px 15px;
            margin-top: 15px;
            display: none;
        }
        
        .tempo-detect-result.visible {
            display: block;
        }
        
        .tempo-detect-result.success {
            border-left: 4px solid #38a169;
        }
        
        .tempo-detect-result.error {
            border-left: 4px solid #e53e3e;
        }
        
        .detect-result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .detect-result-bpm {
            font-size: 1.4em;
            font-weight: bold;
            color: #38a169;
        }
        
        .detect-result-confidence {
            font-size: 0.85em;
            color: #a0aec0;
        }
        
        .confidence-bar {
            height: 4px;
            background: #2d3748;
            border-radius: 2px;
            overflow: hidden;
            margin: 8px 0;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #e53e3e 0%, #d69e2e 50%, #38a169 100%);
            border-radius: 2px;
            transition: width 0.3s;
        }
        
        .detect-result-actions {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        
        .btn-apply-tempo {
            background: #38a169;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 0.85em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.15s;
        }
        
        .btn-apply-tempo:hover {
            background: #2f855a;
        }
        
        .btn-dismiss {
            background: transparent;
            color: #718096;
            border: 1px solid #4a5568;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 0.85em;
            cursor: pointer;
            transition: all 0.15s;
        }
        
        .btn-dismiss:hover {
            background: #2d3748;
        }
        
        .tempo-options {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .tempo-option {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .tempo-option label {
            color: #a0aec0;
            font-size: 0.9em;
        }
        
        .tempo-option select,
        .tempo-option input[type="number"] {
            background: #1a202c;
            border: 1px solid #4a5568;
            color: #fff;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.9em;
        }
        
        .tempo-option input[type="number"] {
            width: 80px;
        }
        
        .toggle-switch {
            position: relative;
            width: 50px;
            height: 26px;
        }
        
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #4a5568;
            transition: 0.3s;
            border-radius: 26px;
        }
        
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 20px;
            width: 20px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: 0.3s;
            border-radius: 50%;
        }
        
        .toggle-switch input:checked + .toggle-slider {
            background-color: #667eea;
        }
        
        .toggle-switch input:checked + .toggle-slider:before {
            transform: translateX(24px);
        }
        
        /* Master Volume Section */
        .master-volume-section {
            background: #2d3748;
            border-radius: 15px;
            padding: 15px 25px;
            margin: 15px 0;
            display: flex;
            align-items: center;
            gap: 20px;
        }
        
        .master-volume-label {
            color: #a0aec0;
            font-weight: bold;
            font-size: 1em;
            white-space: nowrap;
        }
        
        .master-volume-slider-container {
            flex: 1;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .master-volume-slider {
            flex: 1;
            height: 8px;
            border-radius: 4px;
            background: #1a202c;
            -webkit-appearance: none;
            cursor: pointer;
        }
        
        .master-volume-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }
        
        .master-volume-slider::-webkit-slider-thumb:hover {
            transform: scale(1.1);
        }
        
        .master-volume-value {
            min-width: 55px;
            text-align: right;
            color: #fff;
            font-weight: bold;
            font-size: 1.1em;
            font-family: "SF Mono", Monaco, monospace;
        }
        
        /* Progress Section */
        .progress-section {
            background: #2d3748;
            border-radius: 15px;
            padding: 25px;
            margin: 15px 0;
        }
        
        .time-display {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            font-family: "SF Mono", Monaco, monospace;
            margin-bottom: 15px;
            color: #fff;
        }
        
        .time-display .separator {
            color: #718096;
            margin: 0 5px;
        }
        
        .time-display .total {
            color: #a0aec0;
        }
        
        .progress-bar-container {
            background: #1a202c;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
            position: relative;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            transition: width 0.05s linear;
            position: relative;
            z-index: 2;
        }
        
        .progress-bar.recording {
            background: linear-gradient(90deg, #e53e3e 0%, #c53030 100%);
        }
        
        /* Beat grid overlay */
        .beat-grid {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 1;
            pointer-events: none;
        }
        
        .beat-marker {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 2px;
            background: rgba(255, 255, 255, 0.2);
        }
        
        .beat-marker.downbeat {
            width: 3px;
            background: rgba(255, 255, 255, 0.4);
        }
        
        /* Controls */
        .controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin: 20px 0;
        }
        
        .btn {
            padding: 18px 35px;
            border: none;
            border-radius: 12px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        }
        
        .btn:active:not(:disabled) {
            transform: translateY(0);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .btn-rec {
            background: #e53e3e;
            color: white;
        }
        
        .btn-rec.recording {
            animation: pulse 1s infinite;
            box-shadow: 0 0 20px rgba(229, 62, 62, 0.5);
        }
        
        .btn-overdub {
            background: #d69e2e;
            color: white;
        }
        
        .btn-overdub.armed {
            animation: pulse 1s infinite;
            box-shadow: 0 0 20px rgba(214, 158, 46, 0.5);
        }
        
        .btn-clear {
            background: #4a5568;
            color: white;
        }
        
        /* Keyboard hint */
        .keyboard-hint {
            text-align: center;
            color: #718096;
            font-size: 0.85em;
            margin-bottom: 15px;
        }
        
        .keyboard-hint kbd {
            background: #4a5568;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: monospace;
            margin: 0 2px;
        }
        
        /* Layers List */
        .layers-section {
            background: #2d3748;
            border-radius: 15px;
            padding: 20px;
            margin-top: 15px;
        }
        
        .layers-title {
            font-size: 1em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #a0aec0;
        }
        
        .layer {
            background: #1a202c;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
        }
        
        .layer:last-child {
            margin-bottom: 0;
        }
        
        .layer-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        
        .layer-name {
            font-weight: bold;
            font-size: 1.1em;
        }

        .layer-name.master {
            color: #667eea;
        }

        .layer-name-btn {
            background: none;
            border: none;
            font-weight: bold;
            font-size: 1.1em;
            cursor: pointer;
            padding: 0;
        }

        .layer-name-btn.master {
            color: #667eea;
        }

        .color-swatches {
            display: flex;
            gap: 6px;
            margin-top: 6px;
            flex-wrap: wrap;
        }

        .color-swatch {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid transparent;
            transition: border-color 0.15s;
        }

        .color-swatch.active,
        .color-swatch:hover {
            border-color: #fff;
        }

        .layer-status {
            font-size: 1.2em;
        }
        
        .layer-controls {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .btn-small {
            padding: 8px 15px;
            font-size: 0.85em;
            border-radius: 8px;
        }
        
        .btn-mute {
            background: #3182ce;
            color: white;
        }
        
        .btn-mute.muted {
            background: #2c5282;
        }
        
        .btn-delete {
            background: #e53e3e;
            color: white;
        }
        
        .volume-control {
            display: flex;
            align-items: center;
            gap: 10px;
            flex: 1;
        }
        
        .volume-control span {
            font-size: 1.2em;
        }
        
        .volume-slider {
            flex: 1;
            height: 6px;
            border-radius: 3px;
            background: #4a5568;
            -webkit-appearance: none;
            cursor: pointer;
        }
        
        .volume-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
        }
        
        .volume-value {
            min-width: 45px;
            text-align: right;
            color: #a0aec0;
        }
        
        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: #718096;
        }
        
        .empty-state-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }
        
        /* Connection status */
        .connection-status {
            position: fixed;
            top: 15px;
            right: 15px;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .connected { background: #38a169; color: white; }
        .disconnected { background: #e53e3e; color: white; }
        
        /* Stats */
        .stats {
            text-align: center;
            margin-top: 15px;
            color: #4a5568;
            font-size: 0.8em;
        }

        /* Input Level Meter */
        .level-meter-section {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 0;
            margin: 5px 0 10px;
        }

        .level-meter-label {
            font-size: 0.75em;
            color: #718096;
            width: 36px;
            flex-shrink: 0;
        }

        .clip-indicator {
            color: #e53e3e;
            font-weight: bold;
            font-size: 0.85em;
        }

        .level-meter-bar-bg {
            flex: 1;
            height: 8px;
            background: #2d3748;
            border-radius: 4px;
            position: relative;
            overflow: visible;
        }

        .level-meter-bar {
            height: 100%;
            background: linear-gradient(to right, #38a169, #d69e2e, #e53e3e);
            border-radius: 4px;
            width: 0%;
            transition: width 0.05s linear;
        }

        .level-meter-peak {
            position: absolute;
            top: -2px;
            width: 3px;
            height: 12px;
            background: #fff;
            border-radius: 1px;
            left: 0%;
            transition: left 0.1s linear;
        }

        .level-meter-db {
            font-size: 0.75em;
            color: #718096;
            width: 44px;
            text-align: right;
            flex-shrink: 0;
        }

        /* Scenes */
        .scenes-section {
            background: #2d3748;
            border-radius: 15px;
            padding: 15px;
            margin: 15px 0;
        }

        .scenes-title {
            font-size: 0.85em;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 10px;
        }

        .save-scene-row {
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
        }

        .scene-name-input {
            flex: 1;
            background: #1a202c;
            border: 1px solid #4a5568;
            border-radius: 8px;
            color: #e2e8f0;
            padding: 6px 10px;
            font-size: 0.9em;
        }

        .scene-name-input:focus {
            outline: none;
            border-color: #667eea;
        }

        .btn-save-scene {
            background: #667eea;
            color: #fff;
            font-size: 0.8em;
            padding: 6px 12px;
            white-space: nowrap;
        }

        .scene-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 0;
            border-bottom: 1px solid #4a5568;
        }

        .scene-item:last-child {
            border-bottom: none;
        }

        .scene-item.pending {
            background: rgba(102, 126, 234, 0.08);
            border-radius: 6px;
            padding: 8px 6px;
        }

        .scene-name-display {
            flex: 1;
            font-size: 0.95em;
            color: #e2e8f0;
        }

        .scene-pending-badge {
            font-size: 0.7em;
            background: #2c3a6e;
            color: #667eea;
            border-radius: 4px;
            padding: 2px 6px;
        }

        .scene-layer-count {
            font-size: 0.75em;
            color: #718096;
        }

        .btn-load-scene {
            background: #2d4a3e;
            color: #38a169;
            border: 1px solid #38a169;
            font-size: 0.75em;
            padding: 4px 10px;
        }

        .btn-delete-scene {
            background: none;
            color: #fc8181;
            border: 1px solid #fc8181;
            font-size: 0.75em;
            padding: 4px 8px;
        }

        .btn-idle-scene {
            background: none;
            border: 1px solid #4a5568;
            color: #718096;
            font-size: 0.8em;
            padding: 4px 8px;
        }

        .btn-idle-scene:hover { color: #38a169; border-color: #38a169; }

        .scene-item.collapse-scene {
            border-left: 2px solid #38a169;
            padding-left: 8px;
        }

        .scenes-empty {
            color: #4a5568;
            font-size: 0.85em;
            padding: 8px 0;
        }

        .collapse-controls {
            margin-top: 12px;
            padding-top: 10px;
            border-top: 1px solid #4a5568;
        }

        .collapse-row {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }

        .collapse-label {
            font-size: 0.85em;
            color: #a0aec0;
        }

        .collapse-timeout-input {
            width: 48px;
            background: #1a202c;
            border: 1px solid #4a5568;
            border-radius: 6px;
            color: #e2e8f0;
            padding: 3px 6px;
            font-size: 0.85em;
            text-align: center;
        }

        .collapse-hint {
            font-size: 0.75em;
            color: #718096;
            margin-top: 4px;
        }

        /* Sessions */
        .sessions-section {
            background: #2d3748;
            border-radius: 15px;
            padding: 15px;
            margin: 15px 0;
        }

        .sessions-title {
            font-size: 0.85em;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 10px;
        }

        .session-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 0;
            border-bottom: 1px solid #4a5568;
        }

        .session-item:last-child {
            border-bottom: none;
        }

        .session-info {
            flex: 1;
        }

        .session-name {
            font-size: 0.95em;
            color: #e2e8f0;
        }

        .session-meta {
            font-size: 0.75em;
            color: #718096;
            margin-top: 2px;
        }

        .btn-load-session {
            background: #2d4a3e;
            color: #38a169;
            border: 1px solid #38a169;
            font-size: 0.75em;
            padding: 4px 10px;
        }

        .btn-delete-session {
            background: none;
            color: #fc8181;
            border: 1px solid #fc8181;
            font-size: 0.75em;
            padding: 4px 8px;
        }

        .sessions-empty {
            color: #4a5568;
            font-size: 0.85em;
            padding: 8px 0;
        }

        /* Trim Editor */
        .trim-section {
            background: #2d3748;
            border-radius: 15px;
            margin: 15px 0;
            overflow: hidden;
        }
        
        .trim-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            cursor: pointer;
            user-select: none;
            transition: background 0.2s;
        }
        
        .trim-header:hover {
            background: rgba(255,255,255,0.05);
        }
        
        .trim-title {
            font-size: 1em;
            color: #a0aec0;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .trim-toggle-icon {
            transition: transform 0.3s;
        }
        
        .trim-section.expanded .trim-toggle-icon {
            transform: rotate(180deg);
        }
        
        .trim-disabled-badge {
            font-size: 0.8em;
            color: #718096;
            background: #1a202c;
            padding: 4px 10px;
            border-radius: 10px;
        }
        
        .trim-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }
        
        .trim-section.expanded .trim-content {
            max-height: 400px;
        }
        
        .trim-inner {
            padding: 0 20px 20px 20px;
        }
        
        /* Waveform container */
        .waveform-container {
            position: relative;
            background: #1a202c;
            border-radius: 10px;
            height: 120px;
            margin-bottom: 15px;
            overflow: hidden;
        }
        
        .waveform-canvas {
            width: 100%;
            height: 100%;
        }
        
        /* Trim overlay regions (grayed out) */
        .trim-overlay-start,
        .trim-overlay-end {
            position: absolute;
            top: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.6);
            pointer-events: none;
        }
        
        .trim-overlay-start {
            left: 0;
        }
        
        .trim-overlay-end {
            right: 0;
        }
        
        /* Trim handles */
        .trim-handle {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 12px;
            background: #667eea;
            cursor: ew-resize;
            z-index: 10;
            transition: background 0.2s;
        }
        
        .trim-handle:hover {
            background: #5a67d8;
        }
        
        .trim-handle::after {
            content: '⋮⋮';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 10px;
            letter-spacing: -2px;
        }
        
        .trim-handle-start {
            border-radius: 4px 0 0 4px;
        }
        
        .trim-handle-end {
            border-radius: 0 4px 4px 0;
        }
        
        /* Playhead on waveform */
        .waveform-playhead {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #f6e05e;
            pointer-events: none;
            z-index: 5;
        }
        
        /* Beat markers on waveform */
        .waveform-beat-marker {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 1px;
            background: rgba(255, 255, 255, 0.15);
            pointer-events: none;
        }
        
        .waveform-beat-marker.downbeat {
            width: 2px;
            background: rgba(255, 255, 255, 0.3);
        }
        
        /* Trim controls */
        .trim-controls {
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }
        
        .trim-time-input {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .trim-time-input label {
            color: #a0aec0;
            font-size: 0.9em;
        }
        
        .trim-time-input input {
            background: #1a202c;
            border: 1px solid #4a5568;
            color: #fff;
            padding: 8px 12px;
            border-radius: 6px;
            width: 90px;
            font-family: "SF Mono", Monaco, monospace;
        }
        
        .trim-duration {
            color: #667eea;
            font-weight: bold;
            font-family: "SF Mono", Monaco, monospace;
        }
        
        .trim-actions {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }
        
        .btn-trim {
            background: #667eea;
            color: white;
        }
        
        .btn-trim-reset {
            background: #4a5568;
            color: white;
        }
        
        .trim-snap-option {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #a0aec0;
            font-size: 0.9em;
        }
        
        /* Export Section */
        .export-section {
            background: #2d3748;
            border-radius: 15px;
            margin: 15px 0;
            overflow: hidden;
        }
        
        .export-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
        }
        
        .export-title {
            font-size: 1em;
            color: #a0aec0;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .export-disabled-badge {
            font-size: 0.8em;
            color: #718096;
            background: #1a202c;
            padding: 4px 10px;
            border-radius: 10px;
        }
        
        .export-header-controls {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .export-format-select {
            background: #1a202c;
            border: 1px solid #4a5568;
            color: #fff;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.9em;
            cursor: pointer;
        }
        
        .export-mode-select {
            background: #1a202c;
            border: 1px solid #4a5568;
            color: #fff;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.9em;
            cursor: pointer;
        }
        
        .btn-export {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 25px;
            font-size: 1em;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-export:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn-export:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            background: #4a5568;
        }
        
        .btn-export.loading {
            pointer-events: none;
        }
        
        .export-warning {
            background: rgba(214, 158, 46, 0.2);
            border-left: 4px solid #d69e2e;
            padding: 10px 15px;
            margin: 0 20px 15px 20px;
            border-radius: 0 8px 8px 0;
            font-size: 0.85em;
            color: #d69e2e;
        }
        
        .export-info {
            padding: 10px 20px 15px 20px;
            color: #718096;
            font-size: 0.85em;
            text-align: center;
        }

        /* -----------------------------------------------------------------
         * TWO-VIEW MODE: PERFORMANCE vs EDIT
         * ----------------------------------------------------------------- */

        .header-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
        }

        .view-toggle-btn {
            background: transparent;
            border: 1px solid #4a5568;
            color: #a0aec0;
            padding: 6px 18px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
            letter-spacing: 0.5px;
        }

        .view-toggle-btn:hover {
            border-color: #667eea;
            color: #667eea;
            background: rgba(102, 126, 234, 0.1);
        }

        /* Hide edit-only elements in performance mode */
        .performance-mode .edit-only {
            display: none !important;
        }

        /* Bigger main buttons in performance mode */
        .performance-mode .btn-rec,
        .performance-mode .btn-overdub,
        .performance-mode .btn-clear {
            padding: 28px 40px;
            font-size: 1.3em;
        }
    </style>
</head>
<body>
    <div class="connection-status disconnected" id="connectionStatus">● Disconnected</div>
    
    <!-- Countdown Overlay -->
    <div class="countdown-overlay" id="countdownOverlay" style="display: none;">
        <div class="countdown-number" id="countdownNumber">3</div>
        <div class="countdown-bpm" id="countdownBpm"></div>
    </div>
    
    <div class="container">
        <div class="header">
            <h1>🎸 Guitar Looper</h1>
            <div class="header-row">
                <div class="status-badge status-idle" id="statusBadge">Ready</div>
                <button class="view-toggle-btn" id="viewToggleBtn" onclick="toggleViewMode()">✏️ Edit</button>
            </div>
        </div>
        
        <!-- Tempo Section -->
        <div class="tempo-section">
            <div class="tempo-header">
                <span class="tempo-title">⏱️ Tempo</span>
                <div class="tempo-option">
                    <label>Metronome</label>
                    <label class="toggle-switch">
                        <input type="checkbox" id="metronomeToggle" checked>
                        <span class="toggle-slider"></span>
                    </label>
                </div>
            </div>
            
            <div class="tempo-controls">
                <div class="bpm-display">
                    <div class="bpm-value" id="bpmValue">120</div>
                    <div class="bpm-label">BPM</div>
                </div>
                
                <button class="tap-tempo-btn" id="tapTempoBtn" onclick="handleTap()">
                    TAP<br>TEMPO
                </button>
                
                <button class="detect-tempo-btn edit-only" id="detectTempoBtn" onclick="detectTempo()" disabled>
                    🔍 DETECT<br>TEMPO
                </button>

                <div class="tempo-options edit-only">
                    <div class="tempo-option">
                        <label>BPM:</label>
                        <input type="number" id="bpmInput" min="30" max="300" value="120" 
                               onchange="setBpmManual(this.value)">
                    </div>
                    
                    <div class="tempo-option">
                        <label>Time:</label>
                        <select id="beatsPerBar" onchange="setBeatsPerBar(this.value)">
                            <option value="3">3/4</option>
                            <option value="4" selected>4/4</option>
                            <option value="6">6/8</option>
                        </select>
                    </div>
                    
                    <div class="tempo-option">
                        <label>Quantize:</label>
                        <label class="toggle-switch">
                            <input type="checkbox" id="quantizeToggle" checked
                                   onchange="setQuantize(this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="tempo-option">
                        <label>Pre-roll</label>
                        <label class="toggle-switch">
                            <input type="checkbox" id="prerollToggle" checked>
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                </div>
            </div>
            
            <!-- Tempo Detection Result -->
            <div class="tempo-detect-result edit-only" id="tempoDetectResult">
                <div class="detect-result-header">
                    <span class="detect-result-bpm" id="detectedBpm">-- BPM</span>
                    <span class="detect-result-confidence" id="detectedConfidence">Confidence: --%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidenceFill" style="width: 0%"></div>
                </div>
                <div class="detect-result-actions">
                    <button class="btn-apply-tempo" onclick="applyDetectedTempo()">✓ Apply</button>
                    <button class="btn-dismiss" onclick="dismissDetection()">Dismiss</button>
                </div>
            </div>
        </div>
        
        <!-- Master Volume Section -->
        <div class="master-volume-section">
            <span class="master-volume-label">🔊 Master Volume</span>
            <div class="master-volume-slider-container">
                <input type="range" 
                       class="master-volume-slider" 
                       id="masterVolumeSlider"
                       min="0" max="1" step="0.01" value="0.8"
                       oninput="setMasterVolume(this.value)">
                <span class="master-volume-value" id="masterVolumeValue">80%</span>
            </div>
        </div>
        
        <!-- Progress Section -->
        <div class="progress-section">
            <div class="time-display" id="timeDisplay">
                <span class="current">0.0</span><span class="separator">/</span><span class="total">0.0</span><span class="unit">s</span>
            </div>
            <div class="progress-bar-container">
                <div class="beat-grid" id="beatGrid"></div>
                <div class="progress-bar" id="progressBar" style="width: 0%"></div>
            </div>
        </div>
        
        <!-- Trim Editor Section -->
        <div class="trim-section edit-only" id="trimSection">
            <div class="trim-header" onclick="toggleTrimEditor()">
                <span class="trim-title">
                    <span>✂️ Trim Editor</span>
                    <span class="trim-disabled-badge" id="trimDisabledBadge" style="display: none;">
                        Disabled (overdubs recorded)
                    </span>
                </span>
                <span class="trim-toggle-icon">▼</span>
            </div>
            <div class="trim-content">
                <div class="trim-inner">
                    <div class="waveform-container" id="waveformContainer">
                        <canvas class="waveform-canvas" id="waveformCanvas"></canvas>
                        <div class="trim-overlay-start" id="trimOverlayStart"></div>
                        <div class="trim-overlay-end" id="trimOverlayEnd"></div>
                        <div class="trim-handle trim-handle-start" id="trimHandleStart"></div>
                        <div class="trim-handle trim-handle-end" id="trimHandleEnd"></div>
                        <div class="waveform-playhead" id="waveformPlayhead"></div>
                        <div id="waveformBeatMarkers"></div>
                    </div>
                    
                    <div class="trim-controls">
                        <div class="trim-time-input">
                            <label>Start:</label>
                            <input type="number" id="trimStartInput" min="0" step="0.01" value="0.00"
                                   onchange="updateTrimFromInputs()">
                            <span>s</span>
                        </div>
                        
                        <div class="trim-time-input">
                            <label>End:</label>
                            <input type="number" id="trimEndInput" min="0" step="0.01" value="0.00"
                                   onchange="updateTrimFromInputs()">
                            <span>s</span>
                        </div>
                        
                        <div class="trim-time-input">
                            <label>Duration:</label>
                            <span class="trim-duration" id="trimDuration">0.00s</span>
                        </div>
                        
                        <div class="trim-snap-option">
                            <input type="checkbox" id="snapToBeat" checked>
                            <label for="snapToBeat">Snap to beats</label>
                        </div>
                    </div>
                    
                    <div class="trim-actions">
                        <button class="btn btn-small btn-auto-trim" onclick="autoTrimSilence()">
                            ✂ Auto-trim silence
                        </button>
                        <button class="btn btn-small btn-trim-reset" onclick="resetTrim()">
                            ↺ Reset
                        </button>
                        <button class="btn btn-small btn-trim" onclick="applyTrim()">
                            ✓ Apply Trim
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Export Section -->
        <div class="export-section edit-only" id="exportSection">
            <div class="export-header">
                <span class="export-title">
                    <span>📥 Export</span>
                    <span class="export-disabled-badge" id="exportDisabledBadge" style="display: none;">
                        Record a loop first
                    </span>
                </span>
                <div class="export-header-controls">
                    <select class="export-mode-select" id="exportModeSelect" onchange="updateExportPreview()">
                        <option value="mixed">🎵 Mixed</option>
                        <option value="separate">📦 Layers (ZIP)</option>
                    </select>
                    <select class="export-format-select" id="exportFormatSelect" onchange="updateExportPreview()">
                        <option value="mp3">MP3</option>
                        <option value="wav">WAV</option>
                    </select>
                    <button class="btn-export" id="exportBtn" onclick="downloadExport()" disabled>
                        <span id="exportBtnText">⬇️ DOWNLOAD</span>
                    </button>
                </div>
            </div>
            <!-- Warning if ffmpeg not available -->
            <div class="export-warning" id="exportWarning" style="display: none;">
                ⚠️ ffmpeg not installed - MP3 export unavailable. WAV export still works.
            </div>
            <div class="export-info" id="exportInfo">Duration: 0.0s | Est. size: ~0 KB</div>
        </div>
        
        <div class="keyboard-hint edit-only">
            <kbd>SPACE</kbd> Record / Stop / Overdub &nbsp;|&nbsp; <kbd>T</kbd> Tap tempo &nbsp;|&nbsp; <kbd>D</kbd> Detect tempo
        </div>
        
        <div class="controls">
            <button class="btn btn-rec" id="btnRec" onclick="handleRec()">● REC</button>
            <button class="btn btn-overdub" id="btnOverdub" onclick="handleOverdub()" disabled>+ OVERDUB</button>
            <button class="btn btn-clear" onclick="handleClear()">CLEAR</button>
        </div>

        <div class="level-meter-section">
            <div class="level-meter-label">IN <span class="clip-indicator" id="clipIndicator"></span></div>
            <div class="level-meter-bar-bg">
                <div class="level-meter-bar" id="levelMeterBar"></div>
                <div class="level-meter-peak" id="levelMeterPeak"></div>
            </div>
            <div class="level-meter-db" id="levelMeterDb">-∞</div>
        </div>

        <div class="layers-section">
            <div class="layers-title">Layers</div>
            <div id="layersList">
                <div class="empty-state">
                    <div class="empty-state-icon">🎵</div>
                    <p>No loops recorded yet</p>
                </div>
            </div>
        </div>

        <div class="scenes-section">
            <div class="scenes-title">Scenes</div>
            <div class="save-scene-row edit-only">
                <input type="text" id="sceneNameInput" class="scene-name-input"
                       placeholder="Scene name..."
                       onkeydown="if(event.key==='Enter') saveScene()" />
                <button class="btn btn-save-scene" onclick="saveScene()">SAVE SCENE</button>
            </div>
            <div id="scenesList">
                <div class="scenes-empty">No scenes saved yet</div>
            </div>
            <div class="collapse-controls edit-only">
                <div class="collapse-row">
                    <label class="toggle-switch">
                        <input type="checkbox" id="collapseToggle"
                               onchange="setCollapseEnabled(this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                    <span class="collapse-label">Auto-collapse on silence after</span>
                    <input type="number" id="collapseTimeout" class="collapse-timeout-input"
                           min="1" max="30" step="1" value="4"
                           onchange="setCollapseTimeout(this.value)">
                    <span class="collapse-label">s</span>
                </div>
                <div class="collapse-hint" id="collapseHint"></div>
            </div>
        </div>

        <div class="sessions-section edit-only">
            <div class="sessions-title">Sessions</div>
            <div class="save-scene-row">
                <input type="text" id="sessionNameInput" class="scene-name-input"
                       placeholder="Session name..."
                       onkeydown="if(event.key==='Enter') saveSession()" />
                <button class="btn btn-save-scene" onclick="saveSession()">SAVE</button>
            </div>
            <div id="sessionsList">
                <div class="sessions-empty">No sessions saved yet</div>
            </div>
        </div>

        <div class="stats edit-only" id="stats"></div>
    </div>

    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        // =================================================================
        // STATE
        // =================================================================
        
        let socket = null;
        let serverState = {
            state: 'idle',
            master_duration: 0,
            master_volume: 0.8,
            position_ratio: 0,
            current_time: 0,
            recording_time: 0,
            layers: [],
            tempo: {
                bpm: 120,
                beats_per_bar: 4,
                quantize_enabled: true,
                beat_positions: []
            },
            trim: {
                can_trim: false,
                reason: ''
            },
            export: {
                can_export: false,
                pydub_available: false,
                ffmpeg_available: false
            }
        };
        
        // Layer color palette (must match Python LAYER_COLORS)
        const LAYER_COLORS = [
            '#667eea', '#38a169', '#ed8936', '#e53e3e',
            '#319795', '#d53f8c', '#d69e2e', '#3182ce'
        ];

        // Local tempo state
        let localBpm = 120;
        let tapTimes = [];
        const TAP_TIMEOUT = 2000; // Reset taps after 2 seconds of inactivity
        
        // Trim editor state
        let waveformData = [];
        let trimStart = 0;      // in seconds
        let trimEnd = 0;        // in seconds
        let originalDuration = 0;
        let isDragging = null;  // 'start', 'end', or null
        let trimEditorExpanded = false;
        
        // Export state
        let exportSectionExpanded = false;
        
        // =================================================================
        // WEB AUDIO - METRONOME
        // =================================================================
        
        let audioContext = null;
        
        function initAudio() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            return audioContext;
        }
        
        function playClick(isDownbeat = false) {
            if (!document.getElementById('metronomeToggle').checked) return;
            
            try {
                const ctx = initAudio();
                const oscillator = ctx.createOscillator();
                const gainNode = ctx.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(ctx.destination);
                
                // Higher pitch for downbeat
                oscillator.frequency.value = isDownbeat ? 1000 : 800;
                oscillator.type = 'sine';
                
                // Quick envelope
                const now = ctx.currentTime;
                gainNode.gain.setValueAtTime(0.3, now);
                gainNode.gain.exponentialRampToValueAtTime(0.001, now + 0.1);
                
                oscillator.start(now);
                oscillator.stop(now + 0.1);
            } catch (e) {
                console.warn('Audio click failed:', e);
            }
        }
        
        // =================================================================
        // TAP TEMPO
        // =================================================================
        
        function handleTap() {
            const now = Date.now();
            
            // Reset if too much time has passed
            if (tapTimes.length > 0 && (now - tapTimes[tapTimes.length - 1]) > TAP_TIMEOUT) {
                tapTimes = [];
            }
            
            tapTimes.push(now);
            
            // Visual feedback
            const btn = document.getElementById('tapTempoBtn');
            btn.classList.remove('tapping');
            void btn.offsetWidth; // Force reflow
            btn.classList.add('tapping');
            
            // Play click for feedback
            playClick(tapTimes.length === 1);
            
            // Calculate BPM after at least 2 taps
            if (tapTimes.length >= 2) {
                const intervals = [];
                for (let i = 1; i < tapTimes.length; i++) {
                    intervals.push(tapTimes[i] - tapTimes[i - 1]);
                }
                
                // Average interval in ms
                const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
                
                // Convert to BPM
                localBpm = Math.round(60000 / avgInterval);
                localBpm = Math.max(30, Math.min(300, localBpm)); // Clamp
                
                // Update display
                document.getElementById('bpmValue').textContent = localBpm;
                document.getElementById('bpmInput').value = localBpm;
                
                // Send to server (after 4+ taps for stability)
                if (tapTimes.length >= 4) {
                    sendCommand('set_bpm', { bpm: localBpm });
                }
            }
            
            // Keep only last 8 taps
            if (tapTimes.length > 8) {
                tapTimes.shift();
            }
        }
        
        function setBpmManual(value) {
            localBpm = parseInt(value) || 120;
            localBpm = Math.max(30, Math.min(300, localBpm));
            document.getElementById('bpmValue').textContent = localBpm;
            document.getElementById('bpmInput').value = localBpm;
            sendCommand('set_bpm', { bpm: localBpm });
        }
        
        function setBeatsPerBar(value) {
            sendCommand('set_beats_per_bar', { beats: parseInt(value) });
        }
        
        function setQuantize(enabled) {
            sendCommand('set_quantize', { enabled: enabled });
        }
        
        // =================================================================
        // TEMPO DETECTION
        // =================================================================
        
        let detectedTempoValue = 0;
        let isDetecting = false;
        
        function detectTempo() {
            if (isDetecting) return;
            if (serverState.state !== 'playing') {
                alert('Record a loop first before detecting tempo');
                return;
            }
            
            isDetecting = true;
            const btn = document.getElementById('detectTempoBtn');
            btn.classList.add('detecting');
            btn.innerHTML = '🔍 DETECTING...';
            
            // Hide previous result
            document.getElementById('tempoDetectResult').classList.remove('visible', 'success', 'error');
            
            // Request tempo detection from server
            socket.emit('detect_tempo');
        }
        
        function handleTempoDetected(result) {
            isDetecting = false;
            const btn = document.getElementById('detectTempoBtn');
            btn.classList.remove('detecting');
            btn.innerHTML = '🔍 DETECT<br>TEMPO';
            
            const resultDiv = document.getElementById('tempoDetectResult');
            
            if (result.success) {
                detectedTempoValue = result.bpm;
                
                document.getElementById('detectedBpm').textContent = `${result.bpm} BPM`;
                document.getElementById('detectedConfidence').textContent = `Confidence: ${result.confidence}%`;
                document.getElementById('confidenceFill').style.width = `${result.confidence}%`;
                
                resultDiv.classList.remove('error');
                resultDiv.classList.add('visible', 'success');
            } else {
                document.getElementById('detectedBpm').textContent = 'Detection failed';
                document.getElementById('detectedConfidence').textContent = result.error || 'Unknown error';
                document.getElementById('confidenceFill').style.width = '0%';
                
                resultDiv.classList.remove('success');
                resultDiv.classList.add('visible', 'error');
            }
        }
        
        function applyDetectedTempo() {
            if (detectedTempoValue > 0) {
                localBpm = detectedTempoValue;
                document.getElementById('bpmValue').textContent = localBpm;
                document.getElementById('bpmInput').value = localBpm;
                sendCommand('set_bpm', { bpm: localBpm });
                
                // Hide the result after applying
                dismissDetection();
            }
        }
        
        function dismissDetection() {
            document.getElementById('tempoDetectResult').classList.remove('visible', 'success', 'error');
        }
        
        // =================================================================
        // SOCKET CONNECTION
        // =================================================================
        
        function connect() {
            socket = io();
            
            socket.on('connect', () => {
                console.log('✓ Connected');
                document.getElementById('connectionStatus').textContent = '● Connected';
                document.getElementById('connectionStatus').className = 'connection-status connected';
                // Initialize audio context on first interaction
                document.addEventListener('click', () => initAudio(), { once: true });
                // Fetch saved sessions
                socket.emit('list_sessions');
            });
            
            socket.on('disconnect', () => {
                console.log('✗ Disconnected');
                document.getElementById('connectionStatus').textContent = '● Disconnected';
                document.getElementById('connectionStatus').className = 'connection-status disconnected';
            });
            
            socket.on('update', (data) => {
                serverState = data;
                updateUI();
            });

            socket.on('sessions_list', (data) => {
                sessionsList = data.sessions || [];
                renderSessions();
            });

            socket.on('session_saved', (result) => {
                if (!result.success) alert('Save failed: ' + result.error);
            });

            socket.on('session_loaded', (result) => {
                if (!result.success) alert('Load failed: ' + result.error);
            });

            // Handle waveform data from server
            socket.on('waveform', (data) => {
                waveformData = data.data || [];
                originalDuration = serverState.master_duration;
                trimStart = 0;
                trimEnd = originalDuration;
                
                renderWaveform();
                renderBeatMarkersOnWaveform();
                updateTrimUI();
            });
            
            // Handle tempo detection result
            socket.on('tempo_detected', (result) => {
                handleTempoDetected(result);
            });
        }
        
        function sendCommand(command, data = {}) {
            console.log('→', command, data);
            socket.emit('command', { command, ...data });
        }
        
        // =================================================================
        // COMMAND HANDLERS
        // =================================================================
        
        let isCountingDown = false;
        
        async function countdown(beats) {
            const overlay = document.getElementById('countdownOverlay');
            const numberEl = document.getElementById('countdownNumber');
            const bpmEl = document.getElementById('countdownBpm');
            
            overlay.style.display = 'flex';
            isCountingDown = true;
            
            // Disable REC button during countdown
            document.getElementById('btnRec').disabled = true;
            
            // Update status badge
            const badge = document.getElementById('statusBadge');
            badge.textContent = 'Get ready...';
            badge.className = 'status-badge status-countdown';
            
            // Show BPM
            bpmEl.textContent = `${localBpm} BPM`;
            
            // Calculate beat duration
            const beatDuration = 60000 / localBpm; // ms per beat
            
            // Count down beats (typically 4 beats = 1 bar)
            for (let i = beats; i > 0; i--) {
                numberEl.textContent = i;
                numberEl.className = 'countdown-number';
                
                // Play metronome click
                playClick(i === beats); // Downbeat on first
                
                // Force re-animation
                void numberEl.offsetWidth;
                numberEl.className = 'countdown-number';
                
                await new Promise(resolve => setTimeout(resolve, beatDuration));
            }
            
            // Show "GO!" — fire recording on the beat, then hide overlay
            numberEl.textContent = 'GO!';
            numberEl.className = 'countdown-number countdown-go';
            playClick(true);
            sendCommand('start_recording');  // On the beat, not 300ms late

            await new Promise(resolve => setTimeout(resolve, 300));

            overlay.style.display = 'none';
            isCountingDown = false;
            document.getElementById('btnRec').disabled = false;
        }

        async function handleRec() {
            if (isCountingDown) return;

            if (serverState.state === 'idle') {
                initAudio();
                const preroll = document.getElementById('prerollToggle').checked;
                if (preroll) {
                    const beatsPerBar = parseInt(document.getElementById('beatsPerBar').value) || 4;
                    await countdown(beatsPerBar);
                    // countdown sends start_recording itself
                } else {
                    sendCommand('start_recording');
                }
            } else if (serverState.state === 'recording_master') {
                sendCommand('stop_recording');
            }
        }
        
        function handleOverdub() {
            if (serverState.state === 'playing') {
                sendCommand('arm_overdub');
            } else if (serverState.state === 'overdub_armed') {
                sendCommand('cancel_overdub');
            }
        }
        
        function handleClear() {
            if (serverState.layers.length > 0) {
                if (confirm('Clear all loops?')) {
                    sendCommand('clear_all');
                }
            }
        }
        
        function toggleLayer(layerId) {
            sendCommand('toggle_layer', { layer_id: layerId });
        }
        
        function setVolume(layerId, volume) {
            sendCommand('set_volume', { layer_id: layerId, volume: parseFloat(volume) });
        }
        
        function setMasterVolume(volume) {
            const vol = parseFloat(volume);
            document.getElementById('masterVolumeValue').textContent = `${Math.round(vol * 100)}%`;
            sendCommand('set_master_volume', { volume: vol });
        }
        
        function deleteLayer(layerId) {
            if (confirm('Delete this layer?')) {
                sendCommand('delete_layer', { layer_id: layerId });
            }
        }

        function renameLayer(layerId, currentName) {
            const name = prompt('Rename layer:', currentName);
            if (name !== null && name.trim() !== '') {
                sendCommand('rename_layer', { layer_id: layerId, name: name.trim() });
            }
        }

        function setLayerColor(layerId, color) {
            sendCommand('set_layer_color', { layer_id: layerId, color });
        }

        // =================================================================
        // SCENES
        // =================================================================

        function saveScene() {
            const input = document.getElementById('sceneNameInput');
            const name = input.value.trim();
            sendCommand('save_scene', { name });
            input.value = '';
        }

        function loadScene(sceneId) {
            const isPlaying = serverState.state === 'playing';
            sendCommand('load_scene', { scene_id: sceneId, quantized: isPlaying });
        }

        function deleteScene(sceneId) {
            if (confirm('Delete this scene?')) {
                sendCommand('delete_scene', { scene_id: sceneId });
            }
        }

        function renderScenes() {
            const scenesList = document.getElementById('scenesList');
            if (!serverState.scenes || serverState.scenes.list.length === 0) {
                scenesList.innerHTML = '<div class="scenes-empty">No scenes saved yet</div>';
                updateCollapseControls();
                return;
            }

            const pendingId = serverState.scenes.pending_id;
            const collapseId = serverState.collapse?.scene_id ?? null;
            scenesList.innerHTML = serverState.scenes.list.map(scene => {
                const isPending = scene.id === pendingId;
                const isCollapse = scene.id === collapseId;
                const layerCount = scene.layer_states.length;
                const activeCount = scene.layer_states.filter(l => l.is_playing).length;
                return `
                    <div class="scene-item ${isPending ? 'pending' : ''} ${isCollapse ? 'collapse-scene' : ''}">
                        <div class="scene-name-display">${scene.name}</div>
                        <span class="scene-layer-count">${activeCount}/${layerCount}</span>
                        ${isPending ? '<span class="scene-pending-badge">queued</span>' : ''}
                        ${isCollapse ? '<span class="scene-pending-badge" style="background:#2d4a3e;color:#38a169">idle</span>' : ''}
                        <button class="btn btn-idle-scene" title="${isCollapse ? 'Clear idle scene' : 'Set as idle scene'}"
                                onclick="setCollapseScene(${isCollapse ? 'null' : scene.id})">
                            ${isCollapse ? '★' : '☆'}
                        </button>
                        <button class="btn btn-load-scene" onclick="loadScene(${scene.id})">LOAD</button>
                        <button class="btn btn-delete-scene" onclick="deleteScene(${scene.id})">✕</button>
                    </div>
                `;
            }).join('');
            updateCollapseControls();
        }

        function updateCollapseControls() {
            const collapse = serverState.collapse || {};
            const toggle = document.getElementById('collapseToggle');
            const timeoutEl = document.getElementById('collapseTimeout');
            if (toggle) toggle.checked = collapse.enabled || false;
            if (timeoutEl) timeoutEl.value = collapse.timeout || 4;
            const hasScene = collapse.scene_id !== null && collapse.scene_id !== undefined;
            const hint = document.getElementById('collapseHint');
            if (hint) hint.textContent = hasScene ? '' : 'Star a scene above to enable';
        }

        function setCollapseScene(sceneId) {
            sendCommand('set_collapse_scene', { scene_id: sceneId });
        }

        function setCollapseEnabled(enabled) {
            const timeout = parseFloat(document.getElementById('collapseTimeout').value) || 4;
            sendCommand('set_collapse_enabled', { enabled, timeout });
        }

        function setCollapseTimeout(timeout) {
            const enabled = document.getElementById('collapseToggle').checked;
            sendCommand('set_collapse_enabled', { enabled, timeout: parseFloat(timeout) });
        }

        // =================================================================
        // SESSIONS
        // =================================================================

        let sessionsList = [];

        function saveSession() {
            const input = document.getElementById('sessionNameInput');
            const name = input.value.trim();
            sendCommand('save_session', { name });
            input.value = '';
        }

        function loadSession(sessionId, sessionName) {
            const hasLoops = serverState.layers.length > 0;
            const msg = hasLoops
                ? `Load "${sessionName}"? Current loops will be replaced.`
                : `Load "${sessionName}"?`;
            if (!confirm(msg)) return;
            sendCommand('load_session', { session_id: sessionId });
        }

        function deleteSession(sessionId, sessionName) {
            if (!confirm(`Delete session "${sessionName}"? This cannot be undone.`)) return;
            sendCommand('delete_session', { session_id: sessionId });
        }

        function renderSessions() {
            const el = document.getElementById('sessionsList');
            if (sessionsList.length === 0) {
                el.innerHTML = '<div class="sessions-empty">No sessions saved yet</div>';
                return;
            }
            el.innerHTML = sessionsList.map(s => {
                const date = s.created_at ? new Date(s.created_at).toLocaleDateString() : '';
                const bpm = s.bpm ? `${Math.round(s.bpm)} BPM · ` : '';
                const meta = `${bpm}${s.layer_count} loop${s.layer_count !== 1 ? 's' : ''}${s.scene_count ? ` · ${s.scene_count} scenes` : ''} · ${date}`;
                return `
                    <div class="session-item">
                        <div class="session-info">
                            <div class="session-name">${s.name}</div>
                            <div class="session-meta">${meta}</div>
                        </div>
                        <button class="btn btn-load-session"
                                onclick="loadSession('${s.id}', ${JSON.stringify(s.name)})">LOAD</button>
                        <button class="btn btn-delete-session"
                                onclick="deleteSession('${s.id}', ${JSON.stringify(s.name)})">✕</button>
                    </div>
                `;
            }).join('');
        }

        // =================================================================
        // KEYBOARD HANDLING
        // =================================================================
        
        document.addEventListener('keydown', (e) => {
            // TAP TEMPO: T key
            if (e.code === 'KeyT' && !e.repeat) {
                e.preventDefault();
                handleTap();
                return;
            }
            
            // DETECT TEMPO: D key
            if (e.code === 'KeyD' && !e.repeat) {
                e.preventDefault();
                if (serverState.state === 'playing' && !isDetecting) {
                    detectTempo();
                }
                return;
            }
            
            // SPACE: context-dependent action
            if (e.code === 'Space') {
                e.preventDefault();
                
                // Ignore during countdown
                if (isCountingDown) return;
                
                switch (serverState.state) {
                    case 'idle':
                        handleRec();
                        break;
                    
                    case 'recording_master':
                        sendCommand('stop_recording');
                        break;
                    
                    case 'playing':
                        handleOverdub();
                        break;
                    
                    case 'overdub_armed':
                        sendCommand('cancel_overdub');
                        break;
                }
            }
        });
        
        // =================================================================
        // TRIM EDITOR
        // =================================================================
        
        function toggleTrimEditor() {
            const section = document.getElementById('trimSection');
            trimEditorExpanded = !trimEditorExpanded;
            
            if (trimEditorExpanded) {
                section.classList.add('expanded');
                // Request waveform data when expanding
                if (serverState.trim?.can_trim) {
                    socket.emit('get_waveform');
                }
            } else {
                section.classList.remove('expanded');
            }
        }
        
        function initTrimEditor() {
            const container = document.getElementById('waveformContainer');
            const handleStart = document.getElementById('trimHandleStart');
            const handleEnd = document.getElementById('trimHandleEnd');
            
            // Mouse events for dragging handles
            handleStart.addEventListener('mousedown', (e) => {
                e.preventDefault();
                isDragging = 'start';
            });
            
            handleEnd.addEventListener('mousedown', (e) => {
                e.preventDefault();
                isDragging = 'end';
            });
            
            document.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                
                const rect = container.getBoundingClientRect();
                let ratio = (e.clientX - rect.left) / rect.width;
                ratio = Math.max(0, Math.min(1, ratio));
                
                let time = ratio * originalDuration;
                
                // Snap to beat if enabled
                if (document.getElementById('snapToBeat').checked) {
                    time = snapToNearestBeat(time);
                }
                
                if (isDragging === 'start') {
                    trimStart = Math.min(time, trimEnd - 0.1);
                    trimStart = Math.max(0, trimStart);
                } else if (isDragging === 'end') {
                    trimEnd = Math.max(time, trimStart + 0.1);
                    trimEnd = Math.min(originalDuration, trimEnd);
                }
                
                updateTrimUI();
            });
            
            document.addEventListener('mouseup', () => {
                isDragging = null;
            });
            
            // Touch events for mobile
            handleStart.addEventListener('touchstart', (e) => {
                e.preventDefault();
                isDragging = 'start';
            });
            
            handleEnd.addEventListener('touchstart', (e) => {
                e.preventDefault();
                isDragging = 'end';
            });
            
            document.addEventListener('touchmove', (e) => {
                if (!isDragging) return;
                
                const touch = e.touches[0];
                const container = document.getElementById('waveformContainer');
                const rect = container.getBoundingClientRect();
                let ratio = (touch.clientX - rect.left) / rect.width;
                ratio = Math.max(0, Math.min(1, ratio));
                
                let time = ratio * originalDuration;
                
                if (document.getElementById('snapToBeat').checked) {
                    time = snapToNearestBeat(time);
                }
                
                if (isDragging === 'start') {
                    trimStart = Math.min(time, trimEnd - 0.1);
                    trimStart = Math.max(0, trimStart);
                } else if (isDragging === 'end') {
                    trimEnd = Math.max(time, trimStart + 0.1);
                    trimEnd = Math.min(originalDuration, trimEnd);
                }
                
                updateTrimUI();
            });
            
            document.addEventListener('touchend', () => {
                isDragging = null;
            });
        }
        
        function snapToNearestBeat(time) {
            const beats = serverState.tempo?.beat_positions || [];
            if (beats.length === 0) return time;
            
            let nearestBeat = beats[0].time;
            let minDist = Math.abs(time - nearestBeat);
            
            for (const beat of beats) {
                const dist = Math.abs(time - beat.time);
                if (dist < minDist) {
                    minDist = dist;
                    nearestBeat = beat.time;
                }
            }
            
            // Also check end of loop
            const endDist = Math.abs(time - originalDuration);
            if (endDist < minDist) {
                nearestBeat = originalDuration;
            }
            
            // Only snap if within threshold (10% of beat duration)
            const beatDuration = 60 / localBpm;
            if (minDist < beatDuration * 0.25) {
                return nearestBeat;
            }
            
            return time;
        }
        
        function updateTrimUI() {
            if (originalDuration <= 0) return;
            
            const startRatio = trimStart / originalDuration;
            const endRatio = trimEnd / originalDuration;
            
            // Update overlays
            document.getElementById('trimOverlayStart').style.width = `${startRatio * 100}%`;
            document.getElementById('trimOverlayEnd').style.width = `${(1 - endRatio) * 100}%`;
            
            // Update handles
            document.getElementById('trimHandleStart').style.left = `${startRatio * 100}%`;
            document.getElementById('trimHandleEnd').style.left = `calc(${endRatio * 100}% - 12px)`;
            
            // Update inputs
            document.getElementById('trimStartInput').value = trimStart.toFixed(2);
            document.getElementById('trimEndInput').value = trimEnd.toFixed(2);
            
            // Update duration
            const duration = trimEnd - trimStart;
            document.getElementById('trimDuration').textContent = `${duration.toFixed(2)}s`;
        }
        
        function updateTrimFromInputs() {
            const startInput = parseFloat(document.getElementById('trimStartInput').value) || 0;
            const endInput = parseFloat(document.getElementById('trimEndInput').value) || originalDuration;
            
            trimStart = Math.max(0, Math.min(startInput, originalDuration));
            trimEnd = Math.max(trimStart + 0.1, Math.min(endInput, originalDuration));
            
            updateTrimUI();
        }
        
        function resetTrim() {
            trimStart = 0;
            trimEnd = originalDuration;
            updateTrimUI();
        }

        function autoTrimSilence() {
            if (!serverState.trim?.can_trim) return;
            sendCommand('auto_trim_silence');
        }

        function applyTrim() {
            if (!serverState.trim?.can_trim) {
                alert('Cannot trim: overdubs have been recorded');
                return;
            }
            
            if (trimStart === 0 && trimEnd === originalDuration) {
                alert('No trimming needed - boundaries unchanged');
                return;
            }
            
            const duration = trimEnd - trimStart;
            if (!confirm(`Apply trim? Loop will be ${duration.toFixed(2)}s (from ${trimStart.toFixed(2)}s to ${trimEnd.toFixed(2)}s)`)) {
                return;
            }
            
            sendCommand('apply_trim', { start: trimStart, end: trimEnd });
            
            // Reset trim state after applying
            setTimeout(() => {
                socket.emit('get_waveform');
            }, 100);
        }
        
        function renderWaveform() {
            const canvas = document.getElementById('waveformCanvas');
            const ctx = canvas.getContext('2d');
            const container = document.getElementById('waveformContainer');
            
            // Set canvas size
            canvas.width = container.clientWidth * 2;  // 2x for retina
            canvas.height = container.clientHeight * 2;
            
            const width = canvas.width;
            const height = canvas.height;
            const centerY = height / 2;
            
            // Clear
            ctx.fillStyle = '#1a202c';
            ctx.fillRect(0, 0, width, height);
            
            if (waveformData.length === 0) {
                ctx.fillStyle = '#4a5568';
                ctx.font = '24px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('No waveform data', width / 2, height / 2);
                return;
            }
            
            // Draw center line
            ctx.strokeStyle = '#2d3748';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, centerY);
            ctx.lineTo(width, centerY);
            ctx.stroke();
            
            // Draw waveform
            const barWidth = width / waveformData.length;
            
            ctx.fillStyle = '#667eea';
            
            for (let i = 0; i < waveformData.length; i++) {
                const point = waveformData[i];
                const x = i * barWidth;
                
                // Scale amplitude (typical audio is -1 to 1, but might be quieter)
                const maxAmp = Math.max(Math.abs(point.min), Math.abs(point.max));
                const scaledMax = point.max * height * 0.45;
                const scaledMin = point.min * height * 0.45;
                
                const barHeight = scaledMax - scaledMin;
                const barY = centerY - scaledMax;
                
                ctx.fillRect(x, barY, Math.max(1, barWidth - 1), Math.max(1, barHeight));
            }
        }
        
        function updateWaveformPlayhead() {
            if (originalDuration <= 0) return;
            
            const playhead = document.getElementById('waveformPlayhead');
            const ratio = serverState.position_ratio || 0;
            playhead.style.left = `${ratio * 100}%`;
        }
        
        function renderBeatMarkersOnWaveform() {
            const container = document.getElementById('waveformBeatMarkers');
            const beats = serverState.tempo?.beat_positions || [];
            
            if (beats.length === 0 || originalDuration <= 0) {
                container.innerHTML = '';
                return;
            }
            
            container.innerHTML = beats.map(beat => `
                <div class="waveform-beat-marker ${beat.is_downbeat ? 'downbeat' : ''}"
                     style="left: ${beat.ratio * 100}%"></div>
            `).join('');
        }
        
        // Initialize trim editor on page load
        document.addEventListener('DOMContentLoaded', initTrimEditor);
        
        // =================================================================
        // EXPORT
        // =================================================================
        
        function getExportMode() {
            return document.getElementById('exportModeSelect')?.value || 'mixed';
        }
        
        function getExportFormat() {
            return document.getElementById('exportFormatSelect')?.value || 'mp3';
        }
        
        function updateExportPreview() {
            const mode = getExportMode();
            const format = getExportFormat();
            const duration = serverState.master_duration || 0;
            const numLayers = serverState.layers?.length || 0;
            
            let estSize;
            if (mode === 'mixed') {
                if (format === 'mp3') {
                    estSize = Math.round(duration * 24); // ~24KB/sec at 192kbps
                } else {
                    estSize = Math.round(duration * 88.2); // ~88.2KB/sec for 16-bit WAV
                }
            } else {
                if (format === 'mp3') {
                    estSize = Math.round(duration * 24 * numLayers);
                } else {
                    estSize = Math.round(duration * 88.2 * numLayers);
                }
            }
            
            // Format size
            let sizeStr;
            if (estSize < 1024) {
                sizeStr = `${estSize} KB`;
            } else {
                sizeStr = `${(estSize / 1024).toFixed(1)} MB`;
            }
            
            const modeStr = mode === 'mixed' ? 'mixed' : `${numLayers} layers`;
            document.getElementById('exportInfo').textContent = 
                `Duration: ${duration.toFixed(1)}s | ${modeStr} | Est. size: ~${sizeStr}`;
        }
        
        async function downloadExport() {
            const btn = document.getElementById('exportBtn');
            const btnText = document.getElementById('exportBtnText');
            
            if (btn.disabled) return;
            
            const mode = getExportMode();
            const format = getExportFormat();
            
            // Check if MP3 is available
            if (format === 'mp3' && !serverState.export?.ffmpeg_available) {
                alert('MP3 export is not available. Please install ffmpeg or use WAV format.');
                return;
            }
            
            // Show loading state
            btn.disabled = true;
            btn.classList.add('loading');
            btnText.textContent = '⏳ Exporting...';
            
            try {
                let url;
                if (mode === 'mixed') {
                    url = `/export/mixed/${format}`;
                } else {
                    url = `/export/all-layers/${format}`;
                }
                
                const response = await fetch(url);
                
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(errorText || `Export failed: ${response.statusText}`);
                }
                
                // Get filename from Content-Disposition header
                const contentDisposition = response.headers.get('Content-Disposition');
                let filename = 'export';
                if (contentDisposition) {
                    const match = contentDisposition.match(/filename="?([^"]+)"?/);
                    if (match) {
                        filename = match[1];
                    }
                }
                
                // Download the file
                const blob = await response.blob();
                const downloadUrl = URL.createObjectURL(blob);
                
                const a = document.createElement('a');
                a.href = downloadUrl;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                
                URL.revokeObjectURL(downloadUrl);
                
                console.log(`✓ Downloaded: ${filename}`);
                
            } catch (error) {
                console.error('Export failed:', error);
                alert(`Export failed: ${error.message}`);
            } finally {
                // Reset button state
                btn.disabled = false;
                btn.classList.remove('loading');
                btnText.textContent = '⬇️ DOWNLOAD';
                updateExportUI();
            }
        }
        
        function updateExportUI() {
            const badge = document.getElementById('exportDisabledBadge');
            const btn = document.getElementById('exportBtn');
            const warning = document.getElementById('exportWarning');
            const formatSelect = document.getElementById('exportFormatSelect');
            
            const canExport = serverState.export?.can_export;
            const pydubAvailable = serverState.export?.pydub_available;
            const ffmpegAvailable = serverState.export?.ffmpeg_available;
            
            // Show/hide warning and handle MP3 availability
            if (pydubAvailable && !ffmpegAvailable) {
                warning.style.display = 'block';
                // Disable MP3 option
                const mp3Option = formatSelect.querySelector('option[value="mp3"]');
                if (mp3Option) mp3Option.disabled = true;
                // Switch to WAV if MP3 was selected
                if (getExportFormat() === 'mp3') {
                    formatSelect.value = 'wav';
                }
            } else {
                warning.style.display = 'none';
                const mp3Option = formatSelect.querySelector('option[value="mp3"]');
                if (mp3Option) mp3Option.disabled = false;
            }
            
            // Update badge
            if (!pydubAvailable) {
                badge.style.display = 'inline';
                badge.textContent = 'pydub not installed';
            } else if (!canExport) {
                badge.style.display = 'inline';
                badge.textContent = serverState.state === 'idle' ? 'Record a loop first' : 'Cannot export while recording';
            } else {
                badge.style.display = 'none';
            }
            
            // Update button
            btn.disabled = !canExport || !pydubAvailable;
            
            // Update preview
            updateExportPreview();
        }
        
        // =================================================================
        // UI UPDATE
        // =================================================================
        
        function formatTime(seconds) {
            return seconds.toFixed(1);
        }
        
        function updateBeatGrid() {
            const beatGrid = document.getElementById('beatGrid');
            const beats = serverState.tempo?.beat_positions || [];
            
            if (beats.length === 0) {
                beatGrid.innerHTML = '';
                return;
            }
            
            beatGrid.innerHTML = beats.map(beat => `
                <div class="beat-marker ${beat.is_downbeat ? 'downbeat' : ''}" 
                     style="left: ${beat.ratio * 100}%"></div>
            `).join('');
        }
        
        function updateUI() {
            const state = serverState.state;
            
            // --- Status Badge ---
            const badge = document.getElementById('statusBadge');
            badge.className = 'status-badge';
            
            switch (state) {
                case 'idle':
                    badge.textContent = 'Ready';
                    badge.classList.add('status-idle');
                    break;
                case 'recording_master':
                    badge.textContent = '● Recording';
                    badge.classList.add('status-recording');
                    break;
                case 'playing':
                    badge.textContent = '▶ Playing';
                    badge.classList.add('status-playing');
                    break;
                case 'overdub_armed':
                    badge.textContent = 'Waiting for loop start...';
                    badge.classList.add('status-armed');
                    break;
                case 'recording_overdub':
                    badge.textContent = '● Recording Overdub';
                    badge.classList.add('status-recording');
                    break;
            }
            
            // --- Tempo display (sync from server) ---
            if (serverState.tempo) {
                const serverBpm = serverState.tempo.bpm;
                if (Math.abs(serverBpm - localBpm) > 1) {
                    localBpm = serverBpm;
                    document.getElementById('bpmValue').textContent = Math.round(localBpm);
                    document.getElementById('bpmInput').value = Math.round(localBpm);
                }
                document.getElementById('quantizeToggle').checked = serverState.tempo.quantize_enabled;
            }
            
            // --- Master Volume (sync from server) ---
            const masterVolumeSlider = document.getElementById('masterVolumeSlider');
            const masterVolumeValue = document.getElementById('masterVolumeValue');
            if (serverState.master_volume !== undefined) {
                // Only update if not currently being dragged
                if (document.activeElement !== masterVolumeSlider) {
                    masterVolumeSlider.value = serverState.master_volume;
                    masterVolumeValue.textContent = `${Math.round(serverState.master_volume * 100)}%`;
                }
            }
            
            // --- Time Display ---
            const timeDisplay = document.getElementById('timeDisplay');
            let currentTime, totalTime;
            
            if (state === 'recording_master') {
                currentTime = serverState.recording_time;
                totalTime = currentTime;
                timeDisplay.innerHTML = `<span class="current">${formatTime(currentTime)}</span><span class="unit">s</span>`;
            } else {
                currentTime = serverState.current_time;
                totalTime = serverState.master_duration;
                timeDisplay.innerHTML = `<span class="current">${formatTime(currentTime)}</span><span class="separator"> / </span><span class="total">${formatTime(totalTime)}</span><span class="unit">s</span>`;
            }
            
            // --- Progress Bar ---
            const progressBar = document.getElementById('progressBar');
            progressBar.className = 'progress-bar';
            
            if (state === 'recording_master') {
                progressBar.classList.add('recording');
                progressBar.style.width = '100%';
            } else if (serverState.master_duration > 0) {
                progressBar.style.width = `${serverState.position_ratio * 100}%`;
            } else {
                progressBar.style.width = '0%';
            }
            
            // --- Beat Grid ---
            updateBeatGrid();
            
            // --- Buttons ---
            const btnRec = document.getElementById('btnRec');
            const btnOverdub = document.getElementById('btnOverdub');
            
            // REC button
            btnRec.className = 'btn btn-rec';
            if (state === 'recording_master') {
                btnRec.textContent = '⏹ STOP';
                btnRec.classList.add('recording');
            } else {
                btnRec.textContent = '● REC';
            }
            btnRec.disabled = (state !== 'idle' && state !== 'recording_master');
            
            // OVERDUB button
            btnOverdub.className = 'btn btn-overdub';
            if (state === 'overdub_armed') {
                btnOverdub.textContent = '✕ CANCEL';
                btnOverdub.classList.add('armed');
                btnOverdub.disabled = false;
            } else if (state === 'recording_overdub') {
                btnOverdub.textContent = '● RECORDING...';
                btnOverdub.classList.add('armed');
                btnOverdub.disabled = true;
            } else if (state === 'playing') {
                btnOverdub.textContent = '+ OVERDUB';
                btnOverdub.disabled = false;
            } else {
                btnOverdub.textContent = '+ OVERDUB';
                btnOverdub.disabled = true;
            }
            
            // DETECT TEMPO button - only enabled when playing
            const btnDetect = document.getElementById('detectTempoBtn');
            if (btnDetect) {
                btnDetect.disabled = (state !== 'playing' || isDetecting);
            }
            
            // --- Layers List ---
            const layersList = document.getElementById('layersList');
            
            if (serverState.layers.length === 0) {
                layersList.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">🎵</div>
                        <p>No loops recorded yet</p>
                    </div>
                `;
            } else {
                layersList.innerHTML = serverState.layers.map(layer => `
                    <div class="layer" style="border-left-color: ${layer.color}">
                        <div class="layer-header">
                            <button class="layer-name-btn ${layer.id === 0 ? 'master' : ''}"
                                    style="color: ${layer.color}"
                                    onclick="renameLayer(${layer.id}, '${layer.name.replace(/'/g, "\\'")}')">
                                ${layer.name}
                            </button>
                            <span class="layer-status">${layer.is_playing ? '▶️' : '⏸️'}</span>
                        </div>
                        <div class="color-swatches edit-only">
                            ${LAYER_COLORS.map(c => `
                                <div class="color-swatch ${c === layer.color ? 'active' : ''}"
                                     style="background: ${c}"
                                     onclick="setLayerColor(${layer.id}, '${c}')">
                                </div>
                            `).join('')}
                        </div>
                        <div class="layer-controls" style="margin-top: 10px">
                            <button class="btn btn-small btn-mute ${!layer.is_playing ? 'muted' : ''}"
                                    onclick="toggleLayer(${layer.id})">
                                ${layer.is_playing ? 'MUTE' : 'UNMUTE'}
                            </button>
                            <div class="volume-control">
                                <span>🔊</span>
                                <input type="range"
                                       class="volume-slider"
                                       min="0" max="1" step="0.01"
                                       value="${layer.volume}"
                                       oninput="setVolume(${layer.id}, this.value)">
                                <span class="volume-value">${Math.round(layer.volume * 100)}%</span>
                            </div>
                            ${layer.id > 0 ? `
                                <button class="btn btn-small btn-delete" onclick="deleteLayer(${layer.id})">
                                    ✕
                                </button>
                            ` : ''}
                        </div>
                    </div>
                `).join('');
            }
            
            // --- Input Level Meter ---
            if (serverState.input_level !== undefined) {
                const level = serverState.input_level;
                const peak = serverState.input_peak;

                // Square root scaling for perceptual response
                const levelPct = Math.min(100, Math.sqrt(level) * 140);
                const peakPct = Math.min(100, Math.sqrt(peak) * 140);

                document.getElementById('levelMeterBar').style.width = levelPct + '%';
                document.getElementById('levelMeterPeak').style.left = peakPct + '%';

                const db = level > 0.00001 ? (20 * Math.log10(level)).toFixed(1) : '-∞';
                document.getElementById('levelMeterDb').textContent = db + (db !== '-∞' ? 'dB' : '');

                const clipEl = document.getElementById('clipIndicator');
                clipEl.textContent = peak > 0.95 ? 'CLIP' : '';
            }

            // --- Scenes ---
            renderScenes();

            // --- Stats ---
            if (serverState.stats) {
                const callbackMs = serverState.stats.callback_time_ms;
                const latencyMs = serverState.stats.latency_ms;
                const cpuPercent = (callbackMs / latencyMs * 100).toFixed(0);
                document.getElementById('stats').textContent = 
                    `Latency: ${latencyMs.toFixed(1)}ms | CPU: ${cpuPercent}% | Dropouts: ${serverState.stats.dropout_count}`;
            }
            
            // --- Trim Editor ---
            const trimSection = document.getElementById('trimSection');
            const trimDisabledBadge = document.getElementById('trimDisabledBadge');
            
            if (serverState.trim?.can_trim) {
                trimSection.style.opacity = '1';
                trimSection.style.pointerEvents = 'auto';
                trimDisabledBadge.style.display = 'none';
                
                // Update original duration if changed
                if (originalDuration !== serverState.master_duration && serverState.master_duration > 0) {
                    originalDuration = serverState.master_duration;
                    trimEnd = originalDuration;
                    updateTrimUI();
                    
                    // Request new waveform if editor is expanded
                    if (trimEditorExpanded) {
                        socket.emit('get_waveform');
                    }
                }
            } else if (serverState.layers && serverState.layers.length > 1) {
                // Overdubs recorded - disable trim
                trimSection.style.opacity = '0.6';
                trimSection.style.pointerEvents = 'none';
                trimDisabledBadge.style.display = 'inline';
                trimDisabledBadge.textContent = 'Disabled (overdubs recorded)';
            } else if (serverState.state === 'idle') {
                // No recording yet
                trimSection.style.opacity = '0.6';
                trimSection.style.pointerEvents = 'none';
                trimDisabledBadge.style.display = 'inline';
                trimDisabledBadge.textContent = 'Record a loop first';
            }
            
            // Update waveform playhead
            if (trimEditorExpanded && serverState.master_duration > 0) {
                updateWaveformPlayhead();
            }
            
            // --- Export ---
            updateExportUI();
        }
        
        // =================================================================
        // VIEW MODE
        // =================================================================

        let performanceMode = localStorage.getItem('looper_view') !== 'edit';

        function toggleViewMode() {
            performanceMode = !performanceMode;
            applyViewMode();
            localStorage.setItem('looper_view', performanceMode ? 'perform' : 'edit');
        }

        function applyViewMode() {
            const btn = document.getElementById('viewToggleBtn');
            if (performanceMode) {
                document.body.classList.add('performance-mode');
                btn.textContent = '✏️ Edit';
            } else {
                document.body.classList.remove('performance-mode');
                btn.textContent = '▶ Perform';
            }
        }

        applyViewMode();

        // =================================================================
        // INITIALIZATION
        // =================================================================

        connect();
        
        // Poll for UI updates (progress bar + level meter)
        setInterval(() => {
            socket.emit('get_state');
        }, 100);  // 100ms = 10 updates/sec
    </script>
</body>
</html>
"""

# =============================================================================
# ROUTES & SOCKET HANDLERS
# =============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


# -----------------------------------------------------------------------------
# Export Routes
# -----------------------------------------------------------------------------

@app.route('/export/mixed/<fmt>')
def export_mixed_route(fmt):
    """Export mixed audio (all layers combined)."""
    if fmt not in ('mp3', 'wav'):
        return "Invalid format. Use 'mp3' or 'wav'", 400
    
    if not looper:
        return "Looper not initialized", 500
    
    audio_bytes, filename_or_error, content_type = looper.export_mixed(fmt)
    
    if audio_bytes is None:
        return filename_or_error, 500
    
    return Response(
        audio_bytes,
        mimetype=content_type,
        headers={
            'Content-Disposition': f'attachment; filename="{filename_or_error}"',
            'Content-Length': len(audio_bytes)
        }
    )


@app.route('/export/layer/<int:layer_id>/<fmt>')
def export_layer_route(layer_id, fmt):
    """Export a single layer."""
    if fmt not in ('mp3', 'wav'):
        return "Invalid format. Use 'mp3' or 'wav'", 400
    
    if not looper:
        return "Looper not initialized", 500
    
    audio_bytes, filename_or_error, content_type = looper.export_layer(layer_id, fmt)
    
    if audio_bytes is None:
        return filename_or_error, 500
    
    return Response(
        audio_bytes,
        mimetype=content_type,
        headers={
            'Content-Disposition': f'attachment; filename="{filename_or_error}"',
            'Content-Length': len(audio_bytes)
        }
    )


@app.route('/export/all-layers/<fmt>')
def export_all_layers_route(fmt):
    """Export all layers as separate files in a ZIP archive."""
    if fmt not in ('mp3', 'wav'):
        return "Invalid format. Use 'mp3' or 'wav'", 400
    
    if not looper:
        return "Looper not initialized", 500
    
    zip_bytes, filename_or_error, content_type = looper.export_all_layers(fmt)
    
    if zip_bytes is None:
        return filename_or_error, 500
    
    return Response(
        zip_bytes,
        mimetype=content_type,
        headers={
            'Content-Disposition': f'attachment; filename="{filename_or_error}"',
            'Content-Length': len(zip_bytes)
        }
    )


# -----------------------------------------------------------------------------
# Socket Handlers
# -----------------------------------------------------------------------------

@socketio.on('connect')
def handle_connect():
    print(f"✓ Client connected: {request.sid}")
    if looper:
        emit('update', looper.get_state())


@socketio.on('disconnect')
def handle_disconnect():
    print(f"✗ Client disconnected: {request.sid}")


@socketio.on('get_state')
def handle_get_state():
    """Send current state to requesting client."""
    if looper:
        emit('update', looper.get_state())


@socketio.on('get_waveform')
def handle_get_waveform():
    """Send waveform data to requesting client."""
    if looper:
        waveform = looper.get_waveform(600)
        emit('waveform', {'data': waveform})


@socketio.on('detect_tempo')
def handle_detect_tempo():
    """Detect tempo from recorded loop and send result to client."""
    if looper:
        result = looper.detect_tempo()
        emit('tempo_detected', result)


@socketio.on('list_sessions')
def handle_list_sessions():
    emit('sessions_list', {'sessions': WebLooper.list_sessions()})


@socketio.on('command')
def handle_command(data):
    """Handle commands from web client."""
    command = data.get('command')
    print(f"← Command: {command}")
    
    if command == 'start_recording':
        looper.start_recording()
    elif command == 'stop_recording':
        looper.stop_recording()
    elif command == 'arm_overdub':
        looper.arm_overdub()
    elif command == 'cancel_overdub':
        looper.cancel_overdub()
    elif command == 'toggle_layer':
        looper.toggle_layer(data.get('layer_id', 0))
    elif command == 'set_volume':
        looper.set_layer_volume(data.get('layer_id', 0), data.get('volume', 1.0))
    elif command == 'set_master_volume':
        looper.set_master_volume(data.get('volume', 0.8))
    elif command == 'delete_layer':
        looper.delete_layer(data.get('layer_id', 0))
    elif command == 'clear_all':
        looper.clear_all()
    elif command == 'set_bpm':
        looper.set_bpm(data.get('bpm', 120.0))
    elif command == 'set_beats_per_bar':
        looper.set_beats_per_bar(data.get('beats', 4))
    elif command == 'set_quantize':
        looper.set_quantize(data.get('enabled', True))
    elif command == 'apply_trim':
        looper.apply_trim(data.get('start', 0.0), data.get('end', 0.0))
    elif command == 'auto_trim_silence':
        looper.auto_trim_silence()
    elif command == 'rename_layer':
        looper.rename_layer(data.get('layer_id', 0), data.get('name', ''))
    elif command == 'set_layer_color':
        looper.set_layer_color(data.get('layer_id', 0), data.get('color', '#667eea'))
    elif command == 'save_scene':
        looper.save_scene(data.get('name', ''))
    elif command == 'load_scene':
        looper.load_scene(data.get('scene_id'), data.get('quantized', True))
    elif command == 'delete_scene':
        looper.delete_scene(data.get('scene_id'))
    elif command == 'rename_scene':
        looper.rename_scene(data.get('scene_id'), data.get('name', ''))
    elif command == 'set_collapse_scene':
        looper.set_collapse_scene(data.get('scene_id'))
    elif command == 'set_collapse_enabled':
        looper.set_collapse_enabled(data.get('enabled', False), data.get('timeout'))
    elif command == 'save_session':
        result = looper.save_session(data.get('name', ''))
        emit('session_saved', result)
        emit('sessions_list', {'sessions': WebLooper.list_sessions()}, broadcast=True)
        return  # broadcast already done
    elif command == 'load_session':
        result = looper.load_session(data.get('session_id', ''))
        if result['success']:
            emit('update', looper.get_state(), broadcast=True)
        emit('session_loaded', result)
        return
    elif command == 'delete_session':
        looper.delete_session(data.get('session_id', ''))
        emit('sessions_list', {'sessions': WebLooper.list_sessions()}, broadcast=True)
        return

    # Broadcast updated state to all clients
    emit('update', looper.get_state(), broadcast=True)

# =============================================================================
# UTILITIES
# =============================================================================

def get_local_ip():
    """Get local IP address for network access."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def list_audio_devices():
    """List available audio devices."""
    print("\n🔊 Available audio devices:\n")
    devices = sd.query_devices()
    
    valid_devices = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0 and dev['max_output_channels'] > 0:
            valid_devices.append(i)
            print(f"  [{i}] {dev['name']}")
    
    return valid_devices

# =============================================================================
# MAIN
# =============================================================================

def main():
    global looper
    
    print("=" * 60)
    print("🎸 GUITAR LOOPER")
    print("=" * 60)
    
    # Show export capability status
    print("\n📦 Export capabilities:")
    if PYDUB_AVAILABLE:
        print("  ✓ pydub installed")
        if FFMPEG_AVAILABLE:
            print("  ✓ ffmpeg available - MP3 export enabled")
        else:
            print("  ⚠ ffmpeg not found - MP3 export disabled, WAV only")
    else:
        print("  ⚠ pydub not installed - export disabled")
    
    # List and select audio device
    valid_devices = list_audio_devices()
    
    if not valid_devices:
        print("\n❌ No suitable audio device found!")
        print("   Please connect an audio interface with input and output.")
        return
    
    device_choice = input("\nDevice number [Enter for default]: ").strip()
    device = None
    
    if device_choice:
        try:
            device = int(device_choice)
            if device not in valid_devices:
                print(f"⚠ Device {device} may not support input+output")
        except ValueError:
            pass
    
    # Create and start looper
    looper = WebLooper(device=device)
    looper.start_stream()
    
    # Display access URL
    ip = get_local_ip()
    print("\n" + "=" * 60)
    print("🌐 WEB INTERFACE")
    print("=" * 60)
    print(f"\n   Local:   http://localhost:5000")
    print(f"   Network: http://{ip}:5000")
    print("\n" + "=" * 60)
    print("\nPress Ctrl+C to stop the server\n")
    
    # Start Flask server
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n\n✓ Shutting down...")
    finally:
        looper.stop_stream()


if __name__ == "__main__":
    # Check dependencies
    try:
        from flask_socketio import SocketIO
    except ImportError:
        print("Installing dependencies...")
        import subprocess
        subprocess.run([
            "pip3", "install", 
            "flask-socketio", "simple-websocket",
            "--break-system-packages"
        ])
        print("\n✓ Dependencies installed. Please restart the script.\n")
        exit(0)
    
    # Check optional dependencies
    if not LIBROSA_AVAILABLE:
        print("\n💡 Tip: Install librosa for tempo detection:")
        print("   pip install librosa\n")
    
    if not PYDUB_AVAILABLE:
        print("\n💡 Tip: Install pydub for audio export:")
        print("   pip install pydub\n")
    elif not FFMPEG_AVAILABLE:
        print("\n💡 Tip: Install ffmpeg for MP3 export:")
        print("   sudo apt install ffmpeg (Linux)")
        print("   brew install ffmpeg (macOS)\n")
    
    main()