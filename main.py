#!/usr/bin/env python3
import sys
from pathlib import Path

import sounddevice as sd

from config import SAMPLE_RATE
from audio import WebLooper, PYDUB_AVAILABLE, FFMPEG_AVAILABLE, LIBROSA_AVAILABLE
import routes
from routes import app, socketio


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


def main():
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
    routes.looper = WebLooper(device=device)
    routes.looper.start_stream()

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
        routes.looper.stop_stream()


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
