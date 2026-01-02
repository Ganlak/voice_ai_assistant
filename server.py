import asyncio
import json
import base64
import os
from fastapi import FastAPI, WebSocket, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse, Connect, Say
import uvicorn
from typing import Dict, List
import uuid

from src.config.settings import SERVER_HOST, SERVER_PORT, TWILIO_PHONE_NUMBER
from src.agent.graph import run_agent, run_agent_with_history
from src.voice.handlers import simple_voice_handler
from src.services.tts_service import tts_service
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="PSR Voice AI Assistant",
    description="Voice AI Assistant for WellStreet Urgent Care",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ============================================================
# SESSION STORAGE - Maintains conversation history
# ============================================================

conversation_sessions: Dict[str, List] = {}


def get_or_create_session(session_id: str) -> List:
    """Get or create a conversation session."""
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = []
        logger.info(f"Created new session: {session_id}")
    return conversation_sessions[session_id]


def clear_session(session_id: str):
    """Clear a conversation session."""
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
        logger.info(f"Cleared session: {session_id}")


@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "PSR Voice AI Assistant",
        "version": "2.0.0"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ============================================================
# TWILIO VOICE CALL HANDLERS
# ============================================================

@app.post("/twilio/voice")
async def twilio_voice_webhook(request: Request):
    """
    Twilio webhook - Called when someone dials the phone number.
    """
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "unknown")
    from_number = form_data.get("From", "unknown")
    
    logger.info(f"📞 Incoming call: {call_sid} from {from_number}")
    
    response = VoiceResponse()
    
    host = request.headers.get("host", "localhost:8000")
    ws_protocol = "wss" if "railway" in host or "https" in str(request.url) else "ws"
    ws_url = f"{ws_protocol}://{host}/twilio/media-stream"
    
    logger.info(f"Connecting to WebSocket: {ws_url}")
    
    connect = Connect()
    connect.stream(url=ws_url)
    response.append(connect)
    
    return Response(content=str(response), media_type="application/xml")


@app.websocket("/twilio/media-stream")
async def twilio_media_stream(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.
    """
    await websocket.accept()
    logger.info("🎙️ Twilio media stream connected")
    
    stream_sid = None
    call_sid = None
    
    try:
        async for message in websocket.iter_text():
            data = json.loads(message)
            event = data.get("event")
            
            if event == "connected":
                logger.info("Twilio connected event received")
                
            elif event == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"]["callSid"]
                logger.info(f"📞 Stream started: {stream_sid}")
                
            elif event == "media":
                # Audio from caller - would go to Deepgram STT
                pass
                
            elif event == "stop":
                logger.info(f"📞 Stream stopped: {stream_sid}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info(f"📞 Call ended: {call_sid}")


# ============================================================
# API ENDPOINTS
# ============================================================

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """Text chat endpoint with session support."""
    data = await request.json()
    user_input = data.get("message", "")
    session_id = data.get("session_id", "default")
    
    if not user_input:
        return {"error": "No message provided"}
    
    # Get conversation history
    history = get_or_create_session(session_id)
    
    # Run agent with history
    response, updated_history = await asyncio.to_thread(
        run_agent_with_history,
        user_input,
        history
    )
    
    # Update session
    conversation_sessions[session_id] = updated_history
    
    return {
        "message": user_input,
        "response": response,
        "session_id": session_id,
        "history_length": len(updated_history)
    }


@app.post("/api/voice-response")
async def voice_response_endpoint(request: Request):
    """Get text and audio response with session support."""
    data = await request.json()
    user_input = data.get("message", "")
    session_id = data.get("session_id", "default")
    
    if not user_input:
        return {"error": "No message provided"}
    
    # Get conversation history
    history = get_or_create_session(session_id)
    
    # Run agent with history
    text_response, updated_history = await asyncio.to_thread(
        run_agent_with_history,
        user_input,
        history
    )
    
    # Update session
    conversation_sessions[session_id] = updated_history
    
    logger.info(f"Session {session_id}: {len(updated_history)} messages")
    
    try:
        audio_bytes = tts_service.synthesize(text_response)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return {
            "text": text_response,
            "audio": audio_base64,
            "audio_format": "mp3",
            "session_id": session_id,
            "history_length": len(updated_history)
        }
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        return {
            "text": text_response,
            "audio": None,
            "error": str(e),
            "session_id": session_id
        }


@app.post("/api/new-session")
async def new_session_endpoint():
    """Create a new conversation session."""
    session_id = str(uuid.uuid4())[:8]
    conversation_sessions[session_id] = []
    logger.info(f"New session created: {session_id}")
    return {"session_id": session_id}


@app.post("/api/clear-session")
async def clear_session_endpoint(request: Request):
    """Clear a conversation session."""
    data = await request.json()
    session_id = data.get("session_id", "default")
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


# ============================================================
# DEMO UI - PSR VOICE AI ASSISTANT
# ============================================================

@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """PSR Voice AI Assistant Demo with Microphone Support and Session Memory."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PSR Voice AI Assistant - WellStreet Urgent Care</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0a192f 0%, #112240 100%);
            min-height: 100vh;
            color: #e6f1ff;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        
        header {
            text-align: center;
            padding: 40px 0;
            border-bottom: 1px solid rgba(100, 255, 218, 0.1);
            margin-bottom: 40px;
        }
        
        .logo {
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #64ffda, #00bcd4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            color: #8892b0;
            margin-top: 10px;
            font-size: 1.2rem;
        }
        
        .phone-banner {
            background: linear-gradient(135deg, rgba(100, 255, 218, 0.1), rgba(0, 188, 212, 0.1));
            border: 1px solid rgba(100, 255, 218, 0.2);
            border-radius: 16px;
            padding: 30px;
            text-align: center;
            margin-bottom: 40px;
        }
        
        .phone-label {
            font-size: 1rem;
            color: #8892b0;
            margin-bottom: 10px;
        }
        
        .phone-number {
            font-size: 2.5rem;
            font-weight: 700;
            color: #64ffda;
            letter-spacing: 2px;
        }
        
        .phone-status {
            margin-top: 15px;
            font-size: 1rem;
            color: #00e676;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            background: #00e676;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.2); }
        }
        
        .session-info {
            margin-top: 15px;
            font-size: 0.85rem;
            color: #64ffda;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }
        
        .session-badge {
            background: rgba(100, 255, 218, 0.1);
            padding: 5px 12px;
            border-radius: 20px;
            border: 1px solid rgba(100, 255, 218, 0.3);
        }
        
        .btn-new-call {
            background: rgba(255, 107, 107, 0.2);
            border: 1px solid rgba(255, 107, 107, 0.5);
            color: #ff6b6b;
            padding: 5px 15px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.3s;
        }
        
        .btn-new-call:hover {
            background: rgba(255, 107, 107, 0.3);
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        .panel {
            background: rgba(17, 34, 64, 0.8);
            border: 1px solid rgba(100, 255, 218, 0.1);
            border-radius: 16px;
            padding: 25px;
        }
        
        .panel-title {
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #64ffda;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .conversation-area {
            background: #0a192f;
            border-radius: 12px;
            padding: 20px;
            min-height: 400px;
            max-height: 500px;
            overflow-y: auto;
        }
        
        .message {
            margin: 15px 0;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }
        
        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
        }
        
        .caller-avatar { background: linear-gradient(135deg, #667eea, #764ba2); }
        .sarah-avatar { background: linear-gradient(135deg, #64ffda, #00bcd4); }
        
        .message-sender {
            font-weight: 600;
            font-size: 0.95rem;
        }
        
        .caller-name { color: #667eea; }
        .sarah-name { color: #64ffda; }
        
        .message-time {
            font-size: 0.8rem;
            color: #8892b0;
            margin-left: auto;
        }
        
        .message-content {
            padding: 15px 18px;
            border-radius: 12px;
            margin-left: 50px;
            line-height: 1.6;
            font-size: 1rem;
        }
        
        .message.caller .message-content {
            background: rgba(102, 126, 234, 0.1);
            border-left: 3px solid #667eea;
        }
        
        .message.sarah .message-content {
            background: rgba(100, 255, 218, 0.1);
            border-left: 3px solid #64ffda;
        }
        
        .test-section {
            margin-top: 25px;
            padding-top: 20px;
            border-top: 1px solid rgba(100, 255, 218, 0.1);
        }
        
        .test-label {
            font-size: 0.85rem;
            color: #8892b0;
            margin-bottom: 10px;
        }
        
        .input-area {
            display: flex;
            gap: 12px;
            align-items: center;
        }
        
        .voice-input {
            flex: 1;
            padding: 15px 20px;
            border: 1px solid rgba(100, 255, 218, 0.2);
            border-radius: 12px;
            background: #0a192f;
            color: #e6f1ff;
            font-size: 1rem;
        }
        
        .voice-input:focus {
            outline: none;
            border-color: #64ffda;
        }
        
        .voice-input.recording {
            border-color: #ff6b6b;
            background: rgba(255, 107, 107, 0.1);
        }
        
        .btn-mic {
            width: 55px;
            height: 55px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }
        
        .btn-mic:hover {
            transform: scale(1.1);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn-mic.recording {
            background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
            animation: pulse-mic 1s infinite;
        }
        
        @keyframes pulse-mic {
            0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 107, 107, 0.4); }
            50% { transform: scale(1.05); box-shadow: 0 0 0 15px rgba(255, 107, 107, 0); }
        }
        
        .btn-mic:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-send {
            padding: 15px 30px;
            background: linear-gradient(135deg, #64ffda, #00bcd4);
            color: #0a192f;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-send:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(100, 255, 218, 0.3);
        }
        
        .btn-send:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .recording-indicator {
            display: none;
            align-items: center;
            gap: 10px;
            padding: 10px 15px;
            background: rgba(255, 107, 107, 0.1);
            border: 1px solid rgba(255, 107, 107, 0.3);
            border-radius: 10px;
            margin-top: 15px;
            color: #ff6b6b;
            font-size: 0.9rem;
        }
        
        .recording-indicator.active {
            display: flex;
        }
        
        .recording-dot {
            width: 10px;
            height: 10px;
            background: #ff6b6b;
            border-radius: 50%;
            animation: blink 0.5s infinite;
        }
        
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        
        .audio-section {
            margin-top: 20px;
            background: #0a192f;
            border-radius: 12px;
            padding: 15px;
        }
        
        .audio-label {
            font-size: 0.85rem;
            color: #64ffda;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .speaking-dot {
            width: 8px;
            height: 8px;
            background: #00e676;
            border-radius: 50%;
            display: none;
            animation: blink 0.5s infinite;
        }
        
        .speaking-dot.active {
            display: inline-block;
        }
        
        #audioPlayer {
            width: 100%;
            height: 45px;
        }
        
        .pipeline-step {
            display: flex;
            align-items: center;
            padding: 15px;
            background: #0a192f;
            border-radius: 10px;
            margin-bottom: 12px;
            opacity: 0.4;
            transition: all 0.3s;
        }
        
        .pipeline-step.active {
            opacity: 1;
            border: 1px solid rgba(100, 255, 218, 0.3);
        }
        
        .pipeline-step.completed {
            opacity: 1;
        }
        
        .step-icon {
            width: 45px;
            height: 45px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            font-size: 1.3rem;
            background: rgba(100, 255, 218, 0.05);
        }
        
        .pipeline-step.completed .step-icon {
            background: rgba(100, 255, 218, 0.2);
        }
        
        .step-info h4 {
            font-size: 1rem;
            margin-bottom: 3px;
        }
        
        .step-info p {
            font-size: 0.8rem;
            color: #8892b0;
        }
        
        .step-status {
            margin-left: auto;
            font-size: 1.3rem;
        }
        
        .activity-log {
            background: #0a192f;
            border-radius: 12px;
            padding: 15px;
            margin-top: 20px;
            max-height: 200px;
            overflow-y: auto;
        }
        
        .log-entry {
            font-family: 'Courier New', monospace;
            font-size: 0.8rem;
            color: #64ffda;
            margin: 8px 0;
            padding: 8px 10px;
            background: rgba(100, 255, 218, 0.05);
            border-radius: 6px;
            border-left: 2px solid #64ffda;
        }
        
        .log-entry.caller { border-left-color: #667eea; }
        .log-entry.agent { border-left-color: #00e676; }
        .log-entry.tool { border-left-color: #ffd700; }
        
        .mic-hint {
            font-size: 0.75rem;
            color: #8892b0;
            margin-top: 10px;
            text-align: center;
        }
        
        .mic-not-supported {
            color: #ff6b6b;
            font-size: 0.8rem;
            margin-top: 10px;
            display: none;
        }
        
        @media (max-width: 900px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">PSR Voice AI Assistant</div>
            <div class="subtitle">WellStreet Urgent Care</div>
        </header>
        
        <div class="phone-banner">
            <div class="phone-label">📞 Call This Number to Talk to Sarah</div>
            <div class="phone-number">""" + (TWILIO_PHONE_NUMBER or "+1 (361) 204-2794") + """</div>
            <div class="phone-status">
                <span class="status-dot"></span>
                <span>System Active - Ready for Calls</span>
            </div>
            <div class="session-info">
                <span class="session-badge">Session: <span id="sessionId">Loading...</span></span>
                <span class="session-badge">Messages: <span id="messageCount">0</span></span>
                <button class="btn-new-call" onclick="startNewCall()">🔄 New Call</button>
            </div>
        </div>
        
        <div class="dashboard">
            <!-- Conversation Panel -->
            <div class="panel">
                <div class="panel-title">
                    <span>💬</span> Live Conversation
                </div>
                
                <div class="conversation-area" id="conversationArea">
                    <div class="message sarah">
                        <div class="message-header">
                            <div class="message-avatar sarah-avatar">👩</div>
                            <span class="message-sender sarah-name">Sarah</span>
                            <span class="message-time">Ready</span>
                        </div>
                        <div class="message-content">
                            Thank you for calling WellStreet Urgent Care. This is Sarah. How may I help you today?
                        </div>
                    </div>
                </div>
                
                <div class="test-section">
                    <div class="test-label">🎤 Speak or type to talk to Sarah</div>
                    <div class="input-area">
                        <button class="btn-mic" id="btnMic" onclick="toggleRecording()" title="Click to speak">
                            🎤
                        </button>
                        <input type="text" class="voice-input" id="voiceInput" 
                               placeholder="Type your message or click the mic to speak..." 
                               onkeypress="if(event.key==='Enter')sendMessage()">
                        <button class="btn-send" id="btnSend" onclick="sendMessage()">
                            Send
                        </button>
                    </div>
                    
                    <div class="recording-indicator" id="recordingIndicator">
                        <span class="recording-dot"></span>
                        <span id="recordingStatus">Listening... Speak now</span>
                    </div>
                    
                    <div class="mic-hint" id="micHint">
                        💡 Click the microphone button and speak, or type your message
                    </div>
                    
                    <div class="mic-not-supported" id="micNotSupported">
                        ⚠️ Speech recognition is not supported in your browser. Please use Chrome or Edge.
                    </div>
                </div>
                
                <div class="audio-section">
                    <div class="audio-label">
                        <span class="speaking-dot" id="speakingDot"></span>
                        <span id="audioLabel">🔊 Sarah's Voice Response</span>
                    </div>
                    <audio id="audioPlayer" controls></audio>
                </div>
            </div>
            
            <!-- Pipeline Panel -->
            <div class="panel">
                <div class="panel-title">
                    <span>⚙️</span> AI Processing Pipeline
                </div>
                
                <div class="pipeline-step" id="step1">
                    <div class="step-icon">🎤</div>
                    <div class="step-info">
                        <h4>Speech Recognition</h4>
                        <p>Browser Web Speech API / Deepgram</p>
                    </div>
                    <div class="step-status" id="status1">⏳</div>
                </div>
                
                <div class="pipeline-step" id="step2">
                    <div class="step-icon">🧠</div>
                    <div class="step-info">
                        <h4>LangGraph Agent</h4>
                        <p>Azure OpenAI GPT-4o-mini</p>
                    </div>
                    <div class="step-status" id="status2">⏳</div>
                </div>
                
                <div class="pipeline-step" id="step3">
                    <div class="step-icon">🔍</div>
                    <div class="step-info">
                        <h4>SOP Search</h4>
                        <p>FAISS Vector Database</p>
                    </div>
                    <div class="step-status" id="status3">⏳</div>
                </div>
                
                <div class="pipeline-step" id="step4">
                    <div class="step-icon">💬</div>
                    <div class="step-info">
                        <h4>Response Generation</h4>
                        <p>Voice-Optimized Output</p>
                    </div>
                    <div class="step-status" id="status4">⏳</div>
                </div>
                
                <div class="pipeline-step" id="step5">
                    <div class="step-icon">🔊</div>
                    <div class="step-info">
                        <h4>Voice Synthesis</h4>
                        <p>Cartesia Sonic TTS</p>
                    </div>
                    <div class="step-status" id="status5">⏳</div>
                </div>
                
                <div class="activity-log" id="activityLog">
                    <div class="log-entry agent">[System] PSR Voice AI Assistant ready...</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let isProcessing = false;
        let isRecording = false;
        let recognition = null;
        let sessionId = null;
        let messageCount = 0;
        
        const audioPlayer = document.getElementById('audioPlayer');
        const speakingDot = document.getElementById('speakingDot');
        const btnMic = document.getElementById('btnMic');
        const voiceInput = document.getElementById('voiceInput');
        const recordingIndicator = document.getElementById('recordingIndicator');
        const recordingStatus = document.getElementById('recordingStatus');
        const micHint = document.getElementById('micHint');
        const micNotSupported = document.getElementById('micNotSupported');
        
        // Initialize session on page load
        async function initSession() {
            try {
                const response = await fetch('/api/new-session', { method: 'POST' });
                const data = await response.json();
                sessionId = data.session_id;
                document.getElementById('sessionId').textContent = sessionId;
                addLog('New call session started: ' + sessionId, 'agent');
            } catch (error) {
                sessionId = 'default';
                document.getElementById('sessionId').textContent = 'default';
            }
        }
        
        // Start a new call (clear session)
        async function startNewCall() {
            try {
                await fetch('/api/clear-session', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: sessionId})
                });
            } catch (e) {}
            
            // Clear UI
            const area = document.getElementById('conversationArea');
            area.innerHTML = `
                <div class="message sarah">
                    <div class="message-header">
                        <div class="message-avatar sarah-avatar">👩</div>
                        <span class="message-sender sarah-name">Sarah</span>
                        <span class="message-time">Ready</span>
                    </div>
                    <div class="message-content">
                        Thank you for calling WellStreet Urgent Care. This is Sarah. How may I help you today?
                    </div>
                </div>
            `;
            
            document.getElementById('activityLog').innerHTML = '';
            messageCount = 0;
            document.getElementById('messageCount').textContent = '0';
            
            // Create new session
            await initSession();
            addLog('Call restarted - new session', 'agent');
        }
        
        // Initialize on load
        initSession();
        
        // Initialize Speech Recognition
        function initSpeechRecognition() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            
            if (!SpeechRecognition) {
                micNotSupported.style.display = 'block';
                micHint.style.display = 'none';
                btnMic.disabled = true;
                btnMic.style.opacity = '0.3';
                return false;
            }
            
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            
            recognition.onstart = function() {
                isRecording = true;
                btnMic.classList.add('recording');
                btnMic.innerHTML = '🛑';
                voiceInput.classList.add('recording');
                recordingIndicator.classList.add('active');
                recordingStatus.textContent = 'Listening... Speak now';
                addLog('Microphone activated', 'agent');
            };
            
            recognition.onresult = function(event) {
                let interimTranscript = '';
                let finalTranscript = '';
                
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        finalTranscript += transcript;
                    } else {
                        interimTranscript += transcript;
                    }
                }
                
                if (interimTranscript) {
                    voiceInput.value = interimTranscript;
                    recordingStatus.textContent = 'Hearing: ' + interimTranscript;
                }
                
                if (finalTranscript) {
                    voiceInput.value = finalTranscript;
                    recordingStatus.textContent = 'Got it!';
                    addLog('Speech recognized: "' + finalTranscript + '"', 'caller');
                    
                    setTimeout(() => {
                        if (finalTranscript.trim()) {
                            sendMessage();
                        }
                    }, 500);
                }
            };
            
            recognition.onerror = function(event) {
                console.error('Speech recognition error:', event.error);
                addLog('Speech error: ' + event.error, 'caller');
                
                if (event.error === 'no-speech') {
                    recordingStatus.textContent = 'No speech detected. Try again.';
                } else if (event.error === 'audio-capture') {
                    recordingStatus.textContent = 'No microphone found.';
                } else if (event.error === 'not-allowed') {
                    recordingStatus.textContent = 'Microphone access denied.';
                }
                
                stopRecording();
            };
            
            recognition.onend = function() {
                stopRecording();
            };
            
            return true;
        }
        
        function toggleRecording() {
            if (isProcessing) return;
            
            if (isRecording) {
                recognition.stop();
            } else {
                if (!recognition) {
                    if (!initSpeechRecognition()) return;
                }
                voiceInput.value = '';
                recognition.start();
            }
        }
        
        function stopRecording() {
            isRecording = false;
            btnMic.classList.remove('recording');
            btnMic.innerHTML = '🎤';
            voiceInput.classList.remove('recording');
            recordingIndicator.classList.remove('active');
        }
        
        // Initialize speech recognition on load
        initSpeechRecognition();
        
        audioPlayer.onplay = () => speakingDot.classList.add('active');
        audioPlayer.onended = () => speakingDot.classList.remove('active');
        audioPlayer.onpause = () => speakingDot.classList.remove('active');
        
        async function sendMessage() {
            if (isProcessing) return;
            
            const input = document.getElementById('voiceInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            isProcessing = true;
            input.value = '';
            document.getElementById('btnSend').disabled = true;
            btnMic.disabled = true;
            
            addMessage('caller', message);
            addLog(`Caller: "${message}"`, 'caller');
            
            await processPipeline(message);
            
            isProcessing = false;
            document.getElementById('btnSend').disabled = false;
            btnMic.disabled = false;
        }
        
        async function processPipeline(message) {
            resetPipeline();
            
            await activateStep(1, 300);
            addLog('Input received', 'agent');
            
            await activateStep(2, 400);
            addLog('Sarah is thinking...', 'agent');
            
            await activateStep(3, 300);
            addLog('Searching SOP documents...', 'tool');
            
            activateStepPartial(4);
            
            try {
                const response = await fetch('/api/voice-response', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: message,
                        session_id: sessionId
                    })
                });
                
                const data = await response.json();
                
                completeStep(4);
                addLog('Response ready', 'agent');
                
                // Update message count
                if (data.history_length) {
                    messageCount = data.history_length;
                    document.getElementById('messageCount').textContent = messageCount;
                }
                
                await activateStep(5, 200);
                addLog('Voice generated', 'agent');
                
                addMessage('sarah', data.text);
                
                if (data.audio) {
                    audioPlayer.src = `data:audio/mp3;base64,${data.audio}`;
                    audioPlayer.play().catch(e => console.log('Audio play failed:', e));
                }
                
            } catch (error) {
                addLog('Error: ' + error.message, 'caller');
                addMessage('sarah', "I apologize, but I'm having trouble right now. Could you please try again?");
            }
        }
        
        function resetPipeline() {
            for (let i = 1; i <= 5; i++) {
                document.getElementById(`step${i}`).classList.remove('active', 'completed');
                document.getElementById(`status${i}`).textContent = '⏳';
            }
        }
        
        async function activateStep(step, delay) {
            document.getElementById(`step${step}`).classList.add('active');
            document.getElementById(`status${step}`).textContent = '🔄';
            await new Promise(r => setTimeout(r, delay));
            completeStep(step);
        }
        
        function activateStepPartial(step) {
            document.getElementById(`step${step}`).classList.add('active');
            document.getElementById(`status${step}`).textContent = '🔄';
        }
        
        function completeStep(step) {
            document.getElementById(`step${step}`).classList.remove('active');
            document.getElementById(`step${step}`).classList.add('completed');
            document.getElementById(`status${step}`).textContent = '✅';
        }
        
        function addMessage(type, text) {
            const area = document.getElementById('conversationArea');
            const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
            
            const isSarah = type === 'sarah';
            const avatar = isSarah ? '👩' : '👤';
            const name = isSarah ? 'Sarah' : 'Caller';
            
            const html = `
                <div class="message ${type}">
                    <div class="message-header">
                        <div class="message-avatar ${type}-avatar">${avatar}</div>
                        <span class="message-sender ${type}-name">${name}</span>
                        <span class="message-time">${time}</span>
                    </div>
                    <div class="message-content">${text}</div>
                </div>
            `;
            
            area.innerHTML += html;
            area.scrollTop = area.scrollHeight;
        }
        
        function addLog(message, type) {
            const log = document.getElementById('activityLog');
            const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});
            log.innerHTML += `<div class="log-entry ${type}">[${time}] ${message}</div>`;
            log.scrollTop = log.scrollHeight;
        }
    </script>
</body>
</html>
"""


# ============================================================
# SERVER STARTUP
# ============================================================

def start_server():
    """Start the FastAPI server."""
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting PSR Voice AI Assistant on 0.0.0.0:{port}")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()