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

from config import (
    SAMPLE_RATE, BLOCKSIZE, CHANNELS, MAX_LOOP_SECONDS,
    EXPORT_MP3_BITRATE, EXPORT_WAV_SAMPLE_WIDTH, SESSIONS_DIR, LAYER_COLORS,
)

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

SCALE_TEMPLATES = {
    'major':          [0, 2, 4, 5, 7, 9, 11],
    'dorian':         [0, 2, 3, 5, 7, 9, 10],
    'phrygian':       [0, 1, 3, 5, 7, 8, 10],
    'lydian':         [0, 2, 4, 6, 7, 9, 11],
    'mixolydian':     [0, 2, 4, 5, 7, 9, 10],
    'minor':          [0, 2, 3, 5, 7, 8, 10],
    'locrian':        [0, 1, 3, 5, 6, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor':  [0, 2, 3, 5, 7, 9, 11],
    'pent_major':     [0, 2, 4, 7, 9],
    'pent_minor':     [0, 3, 5, 7, 10],
    'blues':          [0, 3, 5, 6, 7, 10],
    'diminished':     [0, 2, 3, 5, 6, 8, 9, 11],
    'whole_tone':     [0, 2, 4, 6, 8, 10],
}

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

        # Scale visualizer
        self.scale_root = 'A'
        self.scale_type = 'minor'

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

    def set_scale(self, root: str, scale_type: str) -> bool:
        """Set scale root and type for the visualizer."""
        with self.lock:
            self.scale_root = root
            self.scale_type = scale_type
        print(f"✓ Scale: {root} {scale_type}")
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

    def detect_scale(self) -> dict:
        """
        Detect scale from the master loop using onset-weighted chroma analysis.
        Returns dict with success flag and top 5 scale candidates ranked by fit.
        """
        if not LIBROSA_AVAILABLE:
            return {'success': False, 'error': 'librosa not installed', 'candidates': []}

        with self.lock:
            if len(self.layers) == 0 or self.master_length == 0:
                return {'success': False, 'error': 'No audio recorded', 'candidates': []}
            audio = self.layers[0].buffer[:self.master_length].copy()

        try:
            audio_64 = audio.astype(np.float64)

            onset_env = librosa.onset.onset_strength(y=audio_64, sr=SAMPLE_RATE)
            chroma = librosa.feature.chroma_stft(y=audio_64, sr=SAMPLE_RATE)

            # Weight each chroma frame by its onset strength, then average
            n_frames = min(len(onset_env), chroma.shape[1])
            weights = onset_env[:n_frames]
            weighted_sum = weights.sum()
            if weighted_sum > 0:
                chroma_vec = (chroma[:, :n_frames] * weights).sum(axis=1) / weighted_sum
            else:
                chroma_vec = chroma.mean(axis=1)

            total = chroma_vec.sum()
            if total <= 0:
                return {'success': False, 'error': 'No pitched content detected', 'candidates': []}
            chroma_norm = chroma_vec / total

            # Score all 14 × 12 = 168 candidates
            candidates = []
            for root_idx, root_name in enumerate(NOTE_NAMES):
                for scale_type, intervals in SCALE_TEMPLATES.items():
                    pitch_classes = [(root_idx + iv) % 12 for iv in intervals]
                    # Give extra weight to the root note in the scale, then normalize by scale size
                    score_val = (float(chroma_norm[root_idx]) * 2.0
                                 + sum(float(chroma_norm[pc]) for pc in pitch_classes if pc != root_idx))
                    adjusted = score_val / (len(intervals) + 1)
                    candidates.append({'root': root_name, 'scale_type': scale_type, '_score': adjusted})

            candidates.sort(key=lambda c: c['_score'], reverse=True)
            top5 = candidates[:5]

            best = top5[0]['_score'] if top5[0]['_score'] > 0 else 1.0
            result_candidates = [
                {'root': c['root'], 'scale_type': c['scale_type'], 'score': round(c['_score'] / best * 100)}
                for c in top5
            ]

            print(f"✓ Scale detected: {result_candidates[0]['root']} {result_candidates[0]['scale_type']} ({result_candidates[0]['score']}%)")
            return {'success': True, 'candidates': result_candidates}

        except Exception as e:
            print(f"✗ Scale detection failed: {e}")
            return {'success': False, 'error': str(e), 'candidates': []}

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
            scale_root = self.scale_root
            scale_type = self.scale_type
        
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
            'scale': {
                'root': scale_root,
                'scale_type': scale_type,
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
