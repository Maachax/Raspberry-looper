# Raspberry Looper

A web-based guitar looper pedal simulator with real-time audio processing, designed for Raspberry Pi and other Linux/macOS/Windows systems.

## Features

- **Multi-layer looping** - Record a master loop and add unlimited overdub layers
- **Automatic quantization** - Recordings snap to the nearest bar/beat for perfect timing
- **Tap tempo** - Set the tempo by tapping with automatic BPM detection
- **Metronome** - In-browser count-in and click track during recording
- **Individual layer control** - Adjust volume and mute/unmute each layer independently
- **Visual waveform trimming** - Interactive trim editor with beat grid overlay
- **Real-time monitoring** - Track latency, CPU usage, and audio dropouts
- **Web interface** - Access from any device on your network via browser
- **Keyboard shortcuts** - Space for record/overdub, T for tap tempo

## Requirements

### System Requirements
- Python 3.6+
- Audio device with input (microphone/guitar) and output (speakers/headphones)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Python Dependencies
- `flask` - Web framework
- `flask-socketio` - WebSocket support for real-time communication
- `simple-websocket` - WebSocket protocol handling
- `sounddevice` - Low-latency audio I/O
- `numpy` - Audio signal processing

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Maachax/Raspberry-looper.git
cd Raspberry-looper

# Run the application (dependencies auto-install if missing)
python3 looper_web.py
```

### Manual Installation

```bash
pip3 install flask flask-socketio simple-websocket sounddevice numpy
```

## Usage

1. **Start the server:**
   ```bash
   python3 looper_web.py
   ```

2. **Select your audio device** from the displayed list (or press Enter for default)

3. **Open the web interface** using the displayed URL:
   - Local: `http://localhost:5000`
   - Network: `http://<your-ip>:5000`

4. **Start looping!**

### Basic Workflow

```
┌─────────┐      ┌───────────────────┐      ┌─────────┐
│  IDLE   │─────▶│  RECORDING MASTER │─────▶│ PLAYING │
└─────────┘      └───────────────────┘      └────┬────┘
                                                  │
                                                  ▼
                 ┌───────────────────┐      ┌─────────────┐
                 │ RECORDING OVERDUB │◀─────│ OVERDUB ARM │
                 └─────────┬─────────┘      └─────────────┘
                           │
                           ▼
                      ┌─────────┐
                      │ PLAYING │ (with new layer)
                      └─────────┘
```

### Controls

| Action | Button | Keyboard |
|--------|--------|----------|
| Start/Stop Recording | Record button | Space |
| Arm Overdub | Overdub button | Space (while playing) |
| Tap Tempo | Tap Tempo button | T |
| Adjust BPM | BPM input field | - |
| Change Time Signature | 3/4, 4/4, 6/8 buttons | - |
| Toggle Quantization | Quantize checkbox | - |
| Toggle Metronome | Metronome checkbox | - |

### Layer Management

- **Master layer** (Layer 0): The foundational loop, cannot be deleted
- **Overdub layers**: Can be muted, volume-adjusted, or deleted
- **Trim editor**: Available only when master layer is the sole layer

## Configuration

Audio settings can be adjusted in `looper_web.py`:

```python
SAMPLE_RATE = 44100    # Audio sample rate (Hz)
BLOCKSIZE = 256        # Buffer size (~5.8ms latency)
CHANNELS = 1           # Mono recording/playback
MAX_LOOP_SECONDS = 120 # Maximum loop duration (seconds)
```

### Web UI Settings

- **BPM**: 30-300 beats per minute
- **Time Signature**: 3/4, 4/4, or 6/8
- **Quantization**: Enable/disable beat-snapping
- **Metronome**: Enable/disable click track

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web Browser                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │              HTML/CSS/JavaScript UI              │    │
│  │    - Waveform visualization                      │    │
│  │    - Beat grid overlay                           │    │
│  │    - Interactive trim editor                     │    │
│  └──────────────────────┬──────────────────────────┘    │
└─────────────────────────┼───────────────────────────────┘
                          │ WebSocket (Socket.IO)
┌─────────────────────────┼───────────────────────────────┐
│            Flask Server │ (looper_web.py)               │
│  ┌──────────────────────┴──────────────────────────┐    │
│  │              Flask-SocketIO                      │    │
│  │         Real-time bi-directional comm            │    │
│  └──────────────────────┬──────────────────────────┘    │
│  ┌──────────────────────┴──────────────────────────┐    │
│  │              WebLooper Engine                    │    │
│  │    - State machine management                    │    │
│  │    - Layer mixing                                │    │
│  │    - Quantization logic                          │    │
│  └──────────────────────┬──────────────────────────┘    │
│  ┌──────────────────────┴──────────────────────────┐    │
│  │           Audio Callback (Thread)                │    │
│  │    - Real-time audio processing                  │    │
│  │    - Thread-safe state access                    │    │
│  │    - Soft limiting                               │    │
│  └──────────────────────┬──────────────────────────┘    │
└─────────────────────────┼───────────────────────────────┘
                          │ sounddevice
                          ▼
                   ┌─────────────┐
                   │ Audio Device │
                   │  (USB/Built-in)│
                   └─────────────┘
```

## Technical Details

### Low-Latency Design
- 256-sample block size provides ~5.8ms latency at 44.1kHz
- Thread-safe audio callback with 5ms lock timeout
- Graceful fallback to pass-through on lock failure
- Soft limiting prevents audio clipping

### Performance Monitoring
The web UI displays real-time statistics:
- **Callback time**: Audio processing duration per block
- **Max callback**: Peak processing time
- **CPU usage**: Audio thread CPU utilization
- **Dropouts**: Count of audio underruns

## Raspberry Pi Setup

For best results on Raspberry Pi:

1. Use a USB audio interface for lower latency
2. Raspberry Pi 4+ recommended for stable performance
3. Consider using a real-time kernel for minimal audio dropouts

## Troubleshooting

### No audio devices found
- Ensure your audio device is connected and recognized by the OS
- On Linux, check with `arecord -l` and `aplay -l`

### High latency or dropouts
- Try a smaller `BLOCKSIZE` value (128)
- Close other CPU-intensive applications
- Use a USB audio interface instead of built-in audio

### Web interface not loading
- Check that port 5000 is not blocked by firewall
- Verify the server started without errors
- Try accessing via `localhost` instead of IP address

## License

This project is open source.

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.
