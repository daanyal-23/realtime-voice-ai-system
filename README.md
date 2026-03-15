# Real-Time Voice AI System

A low-latency, bidirectional voice conversation system built from scratch — no managed voice platforms used.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  BROWSER                                                        │
│                                                                 │
│  Microphone → AudioContext → ScriptProcessor → Int16 PCM       │
│       │                                            │            │
│       │  (waveform viz)                     WebSocket (binary)  │
│       ↓                                            │            │
│  AnalyserNode                                      ↓            │
│                                           WebSocket (binary)    │
│  <Audio> ← MediaSource ← SourceBuffer ← MP3 chunks             │
└─────────────────────────────────────────────────────────────────┘
                              │ WebSocket (ws://)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  BACKEND (Python / asyncio)                                     │
│                                                                 │
│  WebSocket Gateway                                              │
│       │                                                         │
│       ↓                                                         │
│  VoicePipeline (per-session)                                    │
│       │                                                         │
│       ├─→ VAD (energy-based, no ML dependency)                  │
│       │        └─ detects speech start/end                      │
│       │                                                         │
│       ├─→ STT: OpenAI Whisper API                               │
│       │        └─ PCM → WAV (in-memory) → transcript            │
│       │                                                         │
│       ├─→ RAG: cosine similarity over embedded KB chunks        │
│       │        └─ inject context into system prompt             │
│       │                                                         │
│       ├─→ LLM: GPT-4o streaming                                 │
│       │        └─ token-by-token streaming                      │
│       │                                                         │
│       └─→ TTS: OpenAI TTS streaming (per sentence)              │
│                └─ MP3 chunks → WebSocket binary frames          │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| File | Responsibility |
|------|----------------|
| `server.py` | WebSocket server entry point, session lifecycle |
| `session.py` | Tracks active sessions (for future multi-session monitoring) |
| `pipeline.py` | Core orchestration: VAD → STT → LLM → TTS loop, interruption logic |
| `vad.py` | Energy-based Voice Activity Detection (no external model) |
| `stt.py` | Speech-to-Text via OpenAI Whisper |
| `llm.py` | LLM streaming via GPT-4o, injects RAG context |
| `tts.py` | Text-to-Speech streaming via OpenAI TTS |
| `rag.py` | Retrieval-Augmented Generation: embed → store → retrieve |
| `frontend/public/index.html` | Complete single-file browser client |

---

## Design Decisions

### 1. Transport: Raw Binary WebSocket Frames
Audio is sent as raw `Int16Array` binary frames (not base64 or JSON). This halves the payload size and eliminates serialization overhead on the hot path.

### 2. VAD: Energy-Based with Adaptive Noise Floor
Rather than integrating a WebRTC VAD or Silero model (which add latency and complexity), we use a simple RMS energy detector with:
- Hysteresis: separate speech/silence thresholds to avoid chatter
- Adaptive noise floor: tracks quiet-room noise level dynamically
- Minimum utterance length: ignores sub-300ms bursts (noise rejection)

Trade-off: Less robust in noisy environments than ML VAD, but zero latency and no dependencies.

### 3. Sentence-Boundary TTS Streaming
Instead of waiting for the full LLM response before synthesizing speech, we flush to TTS at sentence boundaries (`.`, `!`, `?`). This means the AI starts speaking after the **first sentence** of its response, dramatically reducing perceived latency.

### 4. Interruption Handling
- Browser: measures input RMS while AI is speaking; if RMS > threshold, sends `{"type":"interrupt"}` control message
- Backend: sets an asyncio Event that the TTS streaming loop checks each iteration, cancels the TTS task immediately
- Frontend: resets the MediaSource pipeline on interrupt to clear buffered AI audio

### 5. PCM Format: 16kHz, 16-bit Mono
Whisper works best at 16kHz. We capture at 16kHz directly (`AudioContext({sampleRate: 16000})`), avoiding resampling overhead. This is the same format the backend expects.

### 6. RAG: In-Memory Cosine Similarity
Documents in `knowledge_base/*.txt` are chunked (800 chars, 160 char overlap), embedded with `text-embedding-3-small`, and stored as a numpy matrix. Retrieval is a single matrix-vector multiply — sub-millisecond at small scale. For production, replace with ChromaDB or pgvector.

### 7. Audio Playback: MediaSource API
We use the `MediaSource` + `SourceBuffer` API to stream MP3 chunks into an `<audio>` element progressively. This supports true streaming playback (audio starts before TTS is done) and allows clean cancellation on interruption.

---

## Latency Considerations

### Target Budget (end-to-end from speech end to first audio)

```
User stops speaking  →  VAD detects end         ~300ms  (silence window)
VAD end              →  Whisper response         ~400ms  (API round-trip)
Whisper done         →  GPT-4o first token       ~400ms  (streaming start)
First token          →  Sentence boundary        ~200ms  (1st sentence)
Sentence boundary    →  TTS first chunk          ~300ms  (API round-trip)
TTS first chunk      →  Browser playback         ~50ms   (MediaSource buffer)
─────────────────────────────────────────────────────────
Total (typical)                                  ~1.65s
```

### Optimizations in Place
1. **Streaming everywhere**: LLM and TTS both stream — no full-response wait
2. **Sentence-level TTS**: don't wait for full LLM response before speaking
3. **In-memory WAV conversion**: no disk I/O for audio processing
4. **Async pipeline**: VAD, STT, LLM, TTS each run as async tasks — no blocking
5. **Binary WebSocket frames**: no base64 encoding overhead

### Further Optimizations (not implemented — would add complexity)
- Run a local Whisper model to eliminate STT API round-trip (~200ms saving)
- Use WebRTC with OPUS codec for better compression
- Pre-buffer the "thinking..." sound while LLM generates
- Predictive interruption detection (start processing while still listening)

---

## Known Trade-offs

| Decision | Trade-off |
|----------|-----------|
| Energy-based VAD | Less robust in noisy environments vs Silero/WebRTC VAD |
| OpenAI APIs for STT/TTS | Dependent on external API, adds ~300-500ms latency vs local models |
| In-memory RAG store | Doesn't scale past ~10k chunks; no persistence across restarts |
| Single Python process | No horizontal scaling; one server handles all sessions |
| ScriptProcessor API | Deprecated in Web Audio spec; should migrate to AudioWorklet |
| No OPUS/WebRTC | Higher bandwidth usage vs OPUS compression |

---

## Running Locally

### Prerequisites
- Python 3.11+
- OpenAI API key

### Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd voice-ai

# 2. Install backend dependencies
cd backend
pip install -r requirements.txt

# 3. Set your OpenAI API key
export OPENAI_API_KEY=sk-...
# Or create a .env file:
echo "OPENAI_API_KEY=sk-..." > .env

# 4. (Optional) Add knowledge base documents for RAG
# Drop .txt files into backend/knowledge_base/

# 5. Start the backend
python server.py
# Server starts on ws://localhost:8765

# 6. Serve the frontend (any static server)
cd ../frontend/public
python -m http.server 3000
# Open http://localhost:3000 in your browser
```

### Usage
1. Open `http://localhost:3000`
2. Click **Connect** (allows mic access)
3. Start speaking — the system will:
   - Show your transcript in real-time
   - Stream the AI response as text
   - Play the AI's voice response
4. **Interrupt** the AI by speaking while it responds
5. Click **Reset Conversation** to start fresh

---

## RAG Setup (Bonus)

Add any `.txt` files to `backend/knowledge_base/`. They are automatically:
1. Chunked (800-char chunks with 160-char overlap)
2. Embedded via `text-embedding-3-small` at first query
3. Retrieved by cosine similarity and injected into the LLM system prompt

The system handles empty knowledge bases gracefully (RAG silently disabled).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `HOST` | `0.0.0.0` | WebSocket server bind host |
| `PORT` | `8765` | WebSocket server port |
