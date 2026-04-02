from flask import Flask, render_template, request, Response, send_file, jsonify
from flask_socketio import SocketIO, emit

from audio import WebLooper, PYDUB_AVAILABLE, FFMPEG_AVAILABLE

app = Flask(__name__)
app.config['SECRET_KEY'] = 'looper_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global looper instance (set by main.py)
looper: WebLooper = None

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')


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
# Scale API (for music generator integration)
# -----------------------------------------------------------------------------

@app.route('/api/scale', methods=['GET'])
def get_scale_api():
    """Return current scale + BPM — used by music generator to sync."""
    if not looper:
        return jsonify({'error': 'not ready'}), 503
    return jsonify({
        'root': looper.scale_root,
        'scale': looper.scale_type,
        'bpm': looper.bpm,
    })


@app.route('/api/scale', methods=['POST'])
def set_scale_api():
    """Set scale root/type from music generator (and optionally BPM)."""
    if not looper:
        return jsonify({'error': 'not ready'}), 503
    data = request.json or {}
    if 'root' in data or 'scale' in data:
        looper.set_scale(
            data.get('root', looper.scale_root),
            data.get('scale', looper.scale_type),
        )
    if 'bpm' in data:
        looper.set_bpm(float(data['bpm']))
    socketio.emit('update', looper.get_state())
    return jsonify({'success': True})


# =============================================================================
# SOCKET HANDLERS
# =============================================================================

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
    elif command == 'set_scale':
        looper.set_scale(data.get('root', 'A'), data.get('scale_type', 'minor'))

    # Broadcast updated state to all clients
    emit('update', looper.get_state(), broadcast=True)
