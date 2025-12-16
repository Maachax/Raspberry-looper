#!/usr/bin/env python3
"""
Guitar Looper - Clean Implementation
Flask + WebSockets for real-time control

Features:
    - Tap Tempo
    - Metronome during countdown
    - Quantized loop length
    - Visual beat grid

State Machine:
    IDLE → RECORDING_MASTER → PLAYING ⇄ OVERDUB_ARMED → RECORDING_OVERDUB → PLAYING
"""

import sounddevice as sd
import numpy as np
import threading
import time
from enum import Enum
from flask import Flask, render_template_string, request
from flask_socketio import SocketIO, emit

# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_RATE = 44100
BLOCKSIZE = 256
CHANNELS = 1
MAX_LOOP_SECONDS = 120

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
        
        # Thread safety
        self.lock = threading.Lock()

        # Audio stream
        self.stream = None

        # Stats
        self.callback_time = 0
        self.dropout_count = 0

        # Undo stack (store deleted layers)
        self.deleted_layers_stack = []  # Stack of (layer, original_position) tuples

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
                        # Mix all layers at current position (vectorized for speed)
                        for layer in self.layers:
                            if layer.is_playing and layer.length > 0:
                                for i in range(frames):
                                    pos = (self.master_position + i) % self.master_length
                                    if pos < layer.length:
                                        output[i] += layer.buffer[pos] * layer.volume
                        
                        # Check for loop restart (position wrapping)
                        old_position = self.master_position
                        self.master_position = (self.master_position + frames) % self.master_length
                        loop_restarted = self.master_position < old_position
                        
                        # Handle OVERDUB_ARMED → RECORDING_OVERDUB transition
                        if self.state == LooperState.OVERDUB_ARMED and loop_restarted:
                            self.state = LooperState.RECORDING_OVERDUB
                            self.recording_buffer = np.zeros(self.master_length, dtype=np.float32)
                            self.recording_position = 0
                        
                        # Handle RECORDING_OVERDUB
                        if self.state == LooperState.RECORDING_OVERDUB:
                            # Record input
                            for i in range(frames):
                                pos = (old_position + i) % self.master_length
                                if pos < self.master_length:
                                    self.recording_buffer[pos] += input_samples[i]
                            
                            # Check if overdub complete (loop restarted)
                            if loop_restarted and self.recording_position > 0:
                                self._finalize_overdub()
                            
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
        
        # Write to output (handle both 1D and 2D arrays)
        if outdata.ndim > 1:
            outdata[:, 0] = output
        else:
            outdata[:] = output
        
        self.callback_time = time.perf_counter() - start
    
    def _finalize_overdub(self):
        """Create new layer from recorded overdub. Called within lock."""
        layer_id = len(self.layers)
        name = f"Overdub {layer_id}"
        
        # Copy buffer (important: don't keep reference to recording_buffer)
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
                
                # Adjust buffer
                if quantized_length <= recorded_length:
                    # Trim: apply short fade-out at end to avoid click
                    fade_samples = min(int(0.01 * SAMPLE_RATE), quantized_length // 4)
                    fade = np.linspace(1.0, 0.0, fade_samples)
                    self.recording_buffer[quantized_length - fade_samples:quantized_length] *= fade
                else:
                    # Extend with silence (loop was too short)
                    pass  # Buffer is already zeros beyond recorded_length
                
                self.master_length = quantized_length
            else:
                self.master_length = recorded_length
            
            # Create master layer
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

            # Store deleted layer for undo (keep only last 5 deletions)
            deleted_layer = self.layers[layer_id]
            self.deleted_layers_stack.append((deleted_layer, layer_id))
            if len(self.deleted_layers_stack) > 5:
                self.deleted_layers_stack.pop(0)

            name = self.layers[layer_id].name
            del self.layers[layer_id]

            # Renumber remaining layers
            for i, layer in enumerate(self.layers):
                layer.id = i

            print(f"✓ Deleted {name}")
            return True

    def undo_delete(self) -> bool:
        """Undo the last layer deletion. Returns success."""
        with self.lock:
            if not self.deleted_layers_stack:
                print("✗ Nothing to undo")
                return False

            # Get last deleted layer
            deleted_layer, original_position = self.deleted_layers_stack.pop()

            # Add layer back at the end (simpler than trying to restore exact position)
            new_id = len(self.layers)
            deleted_layer.id = new_id
            self.layers.append(deleted_layer)

            print(f"✓ Restored {deleted_layer.name}")
            return True

    def can_undo(self) -> bool:
        """Check if there are any deletions to undo."""
        with self.lock:
            return len(self.deleted_layers_stack) > 0

    def clear_all(self) -> bool:
        """Clear all loops and reset to IDLE state."""
        with self.lock:
            self.layers = []
            self.master_length = 0
            self.master_position = 0
            self.state = LooperState.IDLE
            self.deleted_layers_stack = []  # Clear undo stack
            print("✓ All loops cleared")
            return True
    
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
            
            # Create new trimmed buffer
            old_buffer = self.layers[0].buffer
            new_buffer = old_buffer[start_sample:end_sample].copy()
            
            # Apply short fade in/out to avoid clicks
            fade_samples = min(int(0.005 * SAMPLE_RATE), new_length // 4)  # 5ms fade
            if fade_samples > 0:
                fade_in = np.linspace(0.0, 1.0, fade_samples)
                fade_out = np.linspace(1.0, 0.0, fade_samples)
                new_buffer[:fade_samples] *= fade_in
                new_buffer[-fade_samples:] *= fade_out
            
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
            callback_time = self.callback_time
            dropout_count = self.dropout_count
            layers_data = [layer.to_dict() for layer in self.layers]
            num_layers = len(self.layers)
            can_undo_delete = len(self.deleted_layers_stack) > 0
        
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
        
        return {
            'state': state,
            'master_duration': master_duration,
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
            'undo': {
                'can_undo': can_undo_delete,
            },
            'stats': {
                'callback_time_ms': callback_time * 1000,
                'latency_ms': (BLOCKSIZE / SAMPLE_RATE) * 1000,
                'dropout_count': dropout_count,
            }
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

        .btn-undo {
            background: #805ad5;
            color: white;
        }

        .btn-undo:disabled {
            background: #553c9a;
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
            <div class="status-badge status-idle" id="statusBadge">Ready</div>
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
                
                <div class="tempo-options">
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
                </div>
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
        <div class="trim-section" id="trimSection">
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
        
        <div class="keyboard-hint">
            <kbd>SPACE</kbd> Record / Stop / Overdub &nbsp;|&nbsp; <kbd>T</kbd> Tap tempo &nbsp;|&nbsp; <kbd>Ctrl+Z</kbd> Undo
        </div>
        
        <div class="controls">
            <button class="btn btn-rec" id="btnRec" onclick="handleRec()">● REC</button>
            <button class="btn btn-overdub" id="btnOverdub" onclick="handleOverdub()" disabled>+ OVERDUB</button>
            <button class="btn btn-undo" id="btnUndo" onclick="handleUndo()" disabled>↶ UNDO</button>
            <button class="btn btn-clear" onclick="handleClear()">CLEAR</button>
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
        
        <div class="stats" id="stats"></div>
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
            undo: {
                can_undo: false
            }
        };
        
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
            
            // Show "GO!" with downbeat click
            numberEl.textContent = 'GO!';
            numberEl.className = 'countdown-number countdown-go';
            playClick(true);
            
            await new Promise(resolve => setTimeout(resolve, 300));
            
            overlay.style.display = 'none';
            isCountingDown = false;
            document.getElementById('btnRec').disabled = false;
        }
        
        async function handleRec() {
            if (isCountingDown) return;
            
            if (serverState.state === 'idle') {
                // Initialize audio context (required for metronome)
                initAudio();
                
                // Get beats per bar for countdown
                const beatsPerBar = parseInt(document.getElementById('beatsPerBar').value) || 4;
                
                // Start countdown (1 bar), then record
                await countdown(beatsPerBar);
                sendCommand('start_recording');
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

        function handleUndo() {
            sendCommand('undo_delete');
        }

        function toggleLayer(layerId) {
            sendCommand('toggle_layer', { layer_id: layerId });
        }
        
        function setVolume(layerId, volume) {
            sendCommand('set_volume', { layer_id: layerId, volume: parseFloat(volume) });
        }
        
        function deleteLayer(layerId) {
            if (confirm('Delete this layer?')) {
                sendCommand('delete_layer', { layer_id: layerId });
            }
        }
        
        // =================================================================
        // KEYBOARD HANDLING
        // =================================================================
        
        document.addEventListener('keydown', (e) => {
            // UNDO: Ctrl+Z or Cmd+Z
            if ((e.ctrlKey || e.metaKey) && e.code === 'KeyZ' && !e.repeat) {
                e.preventDefault();
                if (serverState.undo?.can_undo) {
                    handleUndo();
                }
                return;
            }

            // TAP TEMPO: T key
            if (e.code === 'KeyT' && !e.repeat) {
                e.preventDefault();
                handleTap();
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
            const btnUndo = document.getElementById('btnUndo');

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

            // UNDO button
            btnUndo.disabled = !(serverState.undo?.can_undo);

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
                    <div class="layer">
                        <div class="layer-header">
                            <span class="layer-name ${layer.id === 0 ? 'master' : ''}">${layer.name}</span>
                            <span class="layer-status">${layer.is_playing ? '▶️' : '⏸️'}</span>
                        </div>
                        <div class="layer-controls">
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
        }
        
        // =================================================================
        // INITIALIZATION
        // =================================================================
        
        connect();
        
        // Poll for UI updates (smoother progress bar)
        setInterval(() => {
            if (serverState.state !== 'idle') {
                socket.emit('get_state');
            }
        }, 100);  // 100ms = 10 updates/sec (was 50ms = 20/sec)
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
    elif command == 'delete_layer':
        looper.delete_layer(data.get('layer_id', 0))
    elif command == 'undo_delete':
        looper.undo_delete()
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
    
    main()