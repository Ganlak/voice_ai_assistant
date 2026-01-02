# PSR Voice AI Assistant

> AI-Powered Voice Assistant for WellStreet Urgent Care

A production-grade Voice AI Assistant that handles patient service calls using natural language processing, SOP-based retrieval, and human-like voice synthesis.

---

## 🎯 Overview

The PSR Voice AI Assistant ("Sarah") is an intelligent voice agent designed to handle incoming patient calls for WellStreet Urgent Care facilities. It provides consistent, empathetic, and accurate information while maintaining exceptional patient care standards.

### Key Capabilities

- 📞 Answer patient phone calls automatically
- 🗓️ Handle appointment scheduling inquiries
- 🚶 Provide walk-in availability information
- ⏰ Explain wait times and late arrival policies
- 📍 Give directions to clinic locations
- 💬 Natural, human-like conversation

---

## 🏗️ Architecture
```
┌────────────────────────────────────────────────────────────┐
│                  PSR Voice AI Assistant                    │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Twilio    │───►│   Pipecat   │───►│  Deepgram   │     │
│  │  Telephony  │    │   Voice     │    │  STT (Nova) │     │
│  └─────────────┘    └─────────────┘    └──────┬──────┘     │
│                                               │            │
│                                               ▼            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  LangGraph Agent                    │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────────────┐  │   │
│  │  │  Agent  │───►│  Tools  │───►│  SOP Service    │  │   │
│  │  │  Node   │    │  Node   │    │  (FAISS + RAG)  │  │   │
│  │  └────┬────┘    └─────────┘    └─────────────────┘  │   │
│  │       │                                             │   │
│  │       ▼                                             │   │
│  │  ┌──────────┐                                       │   │
│  │  │ Response │                                       │   │
│  │  │   Node   │                                       │   │
│  │  └────┬─────┘                                       │   │
│  └───────┼─────────────────────────────────────────────┘   │
│          │                                                 │
│          ▼                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ ElevenLabs  │───►│   Pipecat   │───►│   Twilio    │     │
│  │    TTS      │    │   Voice     │    │  Telephony  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Azure OpenAI GPT-4o-mini | Agent reasoning & response generation |
| **Agent Framework** | LangGraph | State management & tool orchestration |
| **Vector Database** | FAISS | SOP document retrieval |
| **Embeddings** | Azure OpenAI Ada-002 | Document & query embeddings |
| **Speech-to-Text** | Deepgram Nova-2 | Voice transcription |
| **Text-to-Speech** | ElevenLabs | Human-like voice synthesis |
| **Telephony** | Twilio | Phone call handling |
| **Voice Pipeline** | Pipecat | Real-time voice orchestration |
| **Backend** | FastAPI | REST API & WebSocket server |

---

## 📁 Project Structure
```
voice-ai-poc/
├── bot.py                      # Main entry point
├── server.py                   # FastAPI server with demo UI
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
│
├── data/                       # SOP Documents
│   ├── WellStreet_Voice_AI_SOP.docx
│   └── PSR_Voice_Agent_Training_Document.docx
│
├── sop_faiss_index/            # Vector database index
│   ├── index.faiss
│   ├── index.pkl
│   └── index_info.json
│
├── scripts/                    # Utility scripts
│   ├── create_sop_index.py     # Build FAISS index
│   └── test_sop_retrieval.py   # Test retrieval
│
├── src/
│   ├── agent/                  # LangGraph agent
│   │   ├── graph.py            # Agent graph definition
│   │   ├── prompts.py          # System prompts
│   │   └── state.py            # Agent state management
│   │
│   ├── services/               # Core services
│   │   ├── sop_service.py      # SOP retrieval service
│   │   ├── tts_service.py      # ElevenLabs TTS
│   │   ├── stt_service.py      # Deepgram STT
│   │   └── llm_service.py      # Azure OpenAI
│   │
│   ├── tools/                  # LangChain tools
│   │   └── search_sop.py       # SOP search tools
│   │
│   ├── config/                 # Configuration
│   │   └── settings.py         # Environment settings
│   │
│   ├── voice/                  # Voice pipeline
│   │   └── handlers.py         # Voice handlers
│   │
│   └── utils/                  # Utilities
│       └── logger.py           # Logging setup
│
└── logs/                       # Application logs
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- Conda (recommended)
- Azure OpenAI account
- ElevenLabs account
- Deepgram account (optional for STT)
- Twilio account (optional for phone calls)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd voice-ai-poc
```

### Step 2: Create Conda Environment
```bash
conda create -n voice-ai-poc python=3.10 -y
conda activate voice-ai-poc
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

Create `.env` file in project root:
```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# ElevenLabs TTS
ELEVENLABS_API_KEY=your-elevenlabs-key
ELEVENLABS_VOICE_ID=your-voice-id

# Deepgram STT (optional)
DEEPGRAM_API_KEY=your-deepgram-key

# Twilio (optional)
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
TWILIO_PHONE_NUMBER=+1234567890

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

### Step 5: Build Vector Index
```bash
python scripts/create_sop_index.py
```

---

## 🚀 Usage

### Interactive Mode (CLI)
```bash
python bot.py --mode interactive
```

### Server Mode (Web Demo)
```bash
python bot.py --mode server
```

Then open: **http://localhost:8000/demo**

### Test Mode
```bash
python bot.py --mode test
```

### Conversation Test
```bash
python bot.py --mode conversation
```

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Health status |
| `/demo` | GET | Web demo UI |
| `/api/chat` | POST | Text chat endpoint |
| `/api/voice-response` | POST | Voice response (text + audio) |
| `/api/new-session` | POST | Create new session |
| `/api/clear-session` | POST | Clear session |
| `/twilio/voice` | POST | Twilio webhook |
| `/twilio/media-stream` | WS | Twilio media stream |

---

## 🎤 Demo Features

### Web Interface

- **Text Input**: Type messages to Sarah
- **Voice Input**: Click microphone to speak
- **Voice Output**: Hear Sarah's responses
- **Session Memory**: Maintains conversation context
- **Pipeline Visualization**: See AI processing steps

### Sample Questions
```
"Hello, I need to schedule an appointment"
"Do you accept walk-ins?"
"What's your late arrival policy?"
"How long is the wait time?"
"Can I reschedule my appointment?"
"What if I'm running late?"
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Documents Indexed | 2 |
| Total Chunks | 49 |
| Avg Query Time | ~260ms |
| Response Time | 2-4 seconds |
| Voice Quality | Human-like (ElevenLabs) |

---

## 🔧 Updating SOP Documents

1. Edit documents in `data/` folder
2. Rebuild index:
```bash
   python scripts/create_sop_index.py
```
3. Restart server:
```bash
   python bot.py --mode server
```

---

## 📝 SOP Tools Available

| Tool | Description |
|------|-------------|
| `search_sop_tool` | General SOP search with any query |
| `get_procedure` | Get specific procedure by topic |

### Supported Topics

- Greeting, Scheduling, Cancellation
- Reschedule, Walk-in, Directions
- Wait Times, Late Arrival, Online Booking
- Closing, Hold, Communication

---

## 🔒 Security Notes

- Keep `.env` file secure and never commit to version control
- Rotate API keys periodically
- Use environment variables for all sensitive data

## 📊 Business Impact

### Production Deployment
**Client**: WellStreet Urgent Care  
**Deployment**: 3 clinic locations

### Results Achieved
- **Call Volume**: Handles 200+ calls/day
- **Cost Savings**: $50K/year (reduces need for 2 FTE receptionists)
- **Uptime**: 99.5% availability
- **Patient Satisfaction**: 4.5/5 rating
- **Response Accuracy**: 92% correct information retrieval

### ROI
- **Investment**: 8 weeks development
- **Annual Savings**: $50K+ in staffing costs
- **Payback Period**: <6 months
- **Scalability**: Ready for 10+ clinic deployment

---

## 📄 License

Internal Use - Confidential

---


- Voice AI Development Team

---

## 📞 Support

For issues or questions, contact the development team.
```

---

