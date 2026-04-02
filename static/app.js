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

        // =================================================================
        // SCALE VISUALIZER
        // =================================================================

        const SCALE_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
        // Open string notes in semitones from C: Low E, A, D, G, B, High e
        const OPEN_STRINGS = [4, 9, 2, 7, 11, 4];
        const SCALE_INTERVALS = {
            'minor':      [0, 2, 3, 5, 7, 8, 10],
            'major':      [0, 2, 4, 5, 7, 9, 11],
            'pent_minor': [0, 3, 5, 7, 10],
            'pent_major': [0, 2, 4, 7, 9],
            'blues':      [0, 3, 5, 6, 7, 10],
            'dorian':     [0, 2, 3, 5, 7, 9, 10],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10],
            'phrygian':   [0, 1, 3, 5, 7, 8, 10],
        };

        let scaleRoot = 'A';
        let scaleType = 'minor';

        function initScaleRootButtons() {
            const row = document.getElementById('scaleRootRow');
            row.innerHTML = SCALE_NOTES.map(n =>
                `<button class="scale-root-btn${n === scaleRoot ? ' active' : ''}" ` +
                `data-note="${n}" onclick="setScaleRoot('${n}')">${n}</button>`
            ).join('');
        }

        function updateScaleRootButtons() {
            document.querySelectorAll('.scale-root-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.note === scaleRoot);
            });
        }

        function setScaleRoot(note) {
            scaleRoot = note;
            updateScaleRootButtons();
            renderFretboard();
            sendCommand('set_scale', { root: scaleRoot, scale_type: scaleType });
        }

        function setScaleType(type) {
            scaleType = type;
            renderFretboard();
            sendCommand('set_scale', { root: scaleRoot, scale_type: scaleType });
        }

        function syncScaleFromServer(scaleData) {
            if (!scaleData) return;
            const newRoot = scaleData.root || 'A';
            const newType = scaleData.scale_type || 'minor';
            if (newRoot === scaleRoot && newType === scaleType) return;
            scaleRoot = newRoot;
            scaleType = newType;
            updateScaleRootButtons();
            const sel = document.getElementById('scaleTypeSelect');
            if (sel) sel.value = scaleType;
            renderFretboard();
        }

        function renderFretboard() {
            const rootIdx = SCALE_NOTES.indexOf(scaleRoot);
            const intervals = new Set(SCALE_INTERVALS[scaleType] || []);

            const W = 640, H = 112;
            const padL = 32, padR = 10, padT = 12, padB = 22;
            const FRETS = 12, STRINGS = 6;
            const fretW = (W - padL - padR) / FRETS;
            const stringH = (H - padT - padB) / (STRINGS - 1);
            const DOT_R = 6.5;
            const openX = padL - fretW * 0.58;

            const fretX = f => padL + f * fretW;
            const noteX = f => f === 0 ? openX : padL + (f - 0.5) * fretW;
            const stringY = s => padT + (STRINGS - 1 - s) * stringH; // s=0 = low E = bottom row

            let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" width="100%" style="display:block">`;

            // String lines (low E thickest at bottom)
            for (let s = 0; s < STRINGS; s++) {
                const y = stringY(s);
                const sw = 0.7 + s * 0.32;
                svg += `<line x1="${openX - 4}" y1="${y}" x2="${fretX(FRETS)}" y2="${y}" stroke="#3a4557" stroke-width="${sw}"/>`;
            }

            // Fret lines (nut thicker)
            for (let f = 0; f <= FRETS; f++) {
                const x = fretX(f);
                svg += `<line x1="${x}" y1="${padT}" x2="${x}" y2="${padT + (STRINGS-1)*stringH}" stroke="${f === 0 ? '#6b7280' : '#1e2533'}" stroke-width="${f === 0 ? 3 : 1.5}"/>`;
            }

            // Position markers below strings
            const markerY = padT + (STRINGS - 1) * stringH + 13;
            for (const mf of [3, 5, 7, 9]) {
                svg += `<circle cx="${padL + (mf - 0.5) * fretW}" cy="${markerY}" r="3.5" fill="#2d3748"/>`;
            }
            const x12 = padL + 11.5 * fretW;
            svg += `<circle cx="${x12 - 5}" cy="${markerY}" r="3" fill="#2d3748"/>`;
            svg += `<circle cx="${x12 + 5}" cy="${markerY}" r="3" fill="#2d3748"/>`;

            // Note dots
            for (let s = 0; s < STRINGS; s++) {
                const y = stringY(s);
                for (let f = 0; f <= FRETS; f++) {
                    const noteIdx = (OPEN_STRINGS[s] + f) % 12;
                    const interval = (noteIdx - rootIdx + 12) % 12;
                    if (!intervals.has(interval)) continue;
                    const isRoot = interval === 0;
                    const x = noteX(f);
                    svg += `<circle cx="${x}" cy="${y}" r="${DOT_R}" fill="${isRoot ? '#ed8936' : '#4fd1c5'}" opacity="0.92"/>`;
                    if (isRoot) {
                        svg += `<text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="central" font-size="7.5" fill="#1a1a2e" font-weight="bold">${SCALE_NOTES[noteIdx]}</text>`;
                    }
                }
            }

            svg += `</svg>`;
            document.getElementById('fretboard').innerHTML = svg;
        }

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

            // --- Scale (sync from external source like music generator) ---
            syncScaleFromServer(serverState.scale);
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
                btn.textContent = 'Edit';
            } else {
                document.body.classList.remove('performance-mode');
                btn.textContent = 'Perform';
            }
        }

        applyViewMode();

        // =================================================================
        // INITIALIZATION
        // =================================================================

        connect();
        initScaleRootButtons();
        renderFretboard();

        // Poll for UI updates (progress bar + level meter)
        setInterval(() => {
            socket.emit('get_state');
        }, 100);  // 100ms = 10 updates/sec
