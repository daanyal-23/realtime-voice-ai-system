# Real-Time Voice AI System

## Overview

This project implements a real-time browser-based voice AI system that supports
low-latency bidirectional speech conversations.

Users speak through the browser microphone and receive streaming AI responses
that can be interrupted at any time. The system implements the complete voice
pipeline from scratch:

```
Audio Capture → Voice Activity Detection → Speech-to-Text → LLM → Text-to-Speech
```

The system supports real-time interruption ("barge-in"): users can speak while
the AI is responding, instantly stopping playback and starting a new turn.

The focus of this project is **systems architecture, streaming design, and
latency management**, rather than UI complexity.

This system is implemented entirely from scratch without using managed voice
platforms such as LiveKit, Pipecat, Daily.co, Retell AI, VAPI, or similar
abstractions.

---

## Demo

A short demo video showing the system in action is included with the submission.

Features demonstrated:
- Real-time voice conversation
- Streaming LLM responses
- Sentence-level TTS playback
- Mid-response interruption ("barge-in")
- RAG-powered knowledge base responses
- Latency monitoring in the UI

---

## Requirements

- Python 3.9+
- Google Chrome (for microphone access)
- Internet connection (for Groq STT and LLM APIs)

---

## Quick Start

### 1. Get a free Groq API key
Sign up at **https://console.groq.com** → API Keys → Create Key.
Groq is free, no credit card required.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API key
Create a `.env` file inside your project folder:
```
GROQ_API_KEY=gsk_your_key_here
```

### 4. Start the backend
```bash
python server.py
```
You should see:
```
Warming up STT model...
STT warmup complete
Warming up RAG...
RAG warmup complete
Starting Voice AI WebSocket server on ws://0.0.0.0:8765
Server ready. Waiting for connections...
```

### 5. Serve the frontend
Open a second terminal in the same folder:
```bash
python -m http.server 3000
```
Then open **http://localhost:3000** in Chrome.

- Click **Connect**
- Allow microphone access when prompted
- Start speaking

---

## Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| Transport | WebSocket — raw binary PCM frames | Free |
| VAD | Custom energy-based detector | Free |
| STT | Groq `whisper-large-v3-turbo` | Free tier |
| LLM | Groq `llama-3.3-70b-versatile` | Free tier |
| TTS | `pyttsx3` — OS native engine | Free, offline |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers | Free, local |
| RAG Store | In-memory numpy cosine similarity | Free |

---

## Architecture

```
BROWSER
  Mic → AudioContext (16kHz) → PCM frames → WebSocket

BACKEND
  WebSocket Server (server.py)
      ↓
  VoicePipeline — one instance per session (pipeline.py)
      ├─ VAD    — energy-based speech detection (vad.py)
      ├─ STT    — Groq Whisper (stt.py)
      ├─ RAG    — cosine similarity over KB chunks (rag.py)
      ├─ LLM    — Groq Llama 3.3 70B streaming (llm.py)
      └─ TTS    — pyttsx3 local synthesis (tts.py)

BROWSER
  WebSocket binary (WAV) → decodeAudioData → AudioBufferSourceNode
  Interrupt support:
  AudioBufferSourceNode.stop() — hardware-level instant playback stop
```

---

## Design Decisions

### 1. Raw binary WebSocket frames
Audio is transmitted as raw `Int16Array` binary — no base64, no JSON wrapper.
This halves bandwidth and removes all serialization overhead on the hot path.

### 2. Energy-based VAD with adaptive noise floor
A simple RMS energy detector with hysteresis. The noise floor adapts dynamically
to the room's background level so the threshold is always calibrated to the
environment. No ML model needed — zero added latency, zero dependencies.

Trade-off: less robust in very noisy environments compared to Silero or WebRTC VAD.

### 3. Sentence-boundary TTS flushing
LLM tokens are buffered until a sentence boundary is detected (`.` `!` `?`), then
the complete sentence is synthesized and sent immediately. This means the AI starts
speaking after its **first sentence** — not after the full response is generated.
This is the single biggest latency reduction in the pipeline.

### 4. Web Audio API for playback
Audio is played via `AudioBufferSourceNode` rather than the `<audio>` element.
`AudioBufferSourceNode.stop()` is a hardware-level instruction that cuts playback
instantly — unlike `<audio>.pause()` which leaves decoded audio buffered internally.
This enables true barge-in interruption even after the last sentence has been sent.

### 5. Interruption handling
The browser measures microphone RMS energy every 256ms. The interrupt condition
checks `isAISpeaking || isPlaying` — this catches both cases:
- User speaks while LLM is still generating
- User speaks while audio is still playing after generation completes

On interrupt:
- `AudioBufferSourceNode.stop()` cuts playback immediately on the client
- `{"type":"interrupt"}` is sent to the backend
- Backend sets an `asyncio.Event` that stops the LLM token loop
- Any queued TTS sentences are skipped
- System returns to LISTENING state

### 6. RAG with local embeddings
Documents in `knowledge_base/*.txt` are chunked (800 chars, 160 char overlap),
embedded with `all-MiniLM-L6-v2` (runs locally, ~80MB), and stored as a numpy
matrix. Retrieval is a single dot product — sub-millisecond at demo scale.
Context is injected into the LLM system prompt on every turn.

### 7. Server warmup
On startup, a silent audio clip is sent to Groq Whisper and the RAG index is
built before any client connects. This eliminates cold-start delays on the
first user query.

### 8. pyttsx3 for TTS
Uses the OS-native speech engine (Windows SAPI5 / macOS NSSpeechSynthesizer).
No network calls, no API key, no model download. ~150-300ms per sentence.
Works out of the box on Windows (SAPI5) and macOS (NSSpeechSynthesizer).
Linux systems may require `espeak`: `sudo apt install espeak`

---

## Latency Budget

```
User stops speaking  →  VAD silence window         ~160ms
VAD end              →  Groq Whisper response       ~600ms
Whisper done         →  Groq LLM first token        ~800ms
LLM first token      →  First sentence boundary     ~200ms
Sentence ready       →  pyttsx3 synthesis           ~200ms
WAV bytes            →  Browser playback starts     ~50ms
──────────────────────────────────────────────────────────
Typical end-to-end                                  ~2.0s
```

**Observed latency (running from India, Groq servers in US):**
```
STT:             765ms
LLM First Token: 968ms
End-to-End:      2599ms
```

The remaining latency is network-bound — geographic distance from India to
Groq's US infrastructure adds round-trip overhead to both STT and LLM stages.
All architectural optimizations are in place. On infrastructure co-located
with Groq's servers, end-to-end latency would be under 1 second.

---

## Known Trade-offs

| Decision | Trade-off |
|----------|-----------|
| Energy-based VAD | Less robust in noisy rooms vs Silero/WebRTC VAD |
| Full-utterance STT | ~300ms extra vs streaming STT — simpler, more accurate |
| Sentence-level TTS | Brief gap between sentences vs truly continuous speech |
| pyttsx3 | OS-native voice quality vs neural TTS (edge-tts, OpenAI TTS) |
| In-memory RAG | No persistence across restarts, max ~10k chunks |
| `ScriptProcessor` | Deprecated Web Audio API — production should use `AudioWorklet` |
| Groq free tier | Rate limits apply under heavy load |

---

## File Structure

```
SNIPERTHINK/
├── server.py           — WebSocket gateway, session lifecycle, warmup
├── pipeline.py         — Core orchestration: VAD → STT → LLM → TTS
├── vad.py              — Voice Activity Detection (energy-based)
├── stt.py              — Speech-to-Text via Groq Whisper
├── llm.py              — LLM streaming via Groq Llama 3.3 70B
├── tts.py              — Text-to-Speech via pyttsx3 (local)
├── rag.py              — Retrieval-Augmented Generation (bonus)
├── session.py          — Session management
├── requirements.txt    — Python dependencies
├── .env                — API keys (not committed to Git)
├── knowledge_base/
│   └── sniperthink.txt   — Knowledge base for RAG
└── index.html          — Browser client (single file)
```

---

## RAG — Knowledge Base Setup

Drop any `.txt` files into the `knowledge_base/` folder.
On server start they are automatically:
1. Chunked into 800-char overlapping segments
2. Embedded locally via `all-MiniLM-L6-v2`
3. Retrieved by cosine similarity on each user query
4. Injected into the LLM system prompt as context

The `sentence-transformers` model (~80MB) downloads on first run and is
cached permanently after that.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `GROQ_API_KEY not set` | Check your `.env` file is in the project folder |
| No microphone / MIC ERROR | Use Chrome at `http://localhost:3000` (not IPv6) |
| WebSocket connection refused | Make sure `python server.py` is running first |
| No audio output | Check backend logs for pyttsx3 errors |
| First query slow | Normal — `sentence-transformers` downloads on first run only |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Free at console.groq.com |
| `HOST` | No (default: `0.0.0.0`) | WebSocket bind host |
| `PORT` | No (default: `8765`) | WebSocket port |

---

## Future Improvements

While the current system focuses on architectural clarity and low latency,
several improvements could be explored in a production setting:

- Replace `ScriptProcessor` with **AudioWorklet** for lower audio latency
- Add **streaming STT** instead of full-utterance transcription
- Use **neural TTS** (OpenAI TTS / Coqui / ElevenLabs) for more natural speech
- Deploy the backend **closer to inference infrastructure** to reduce network latency
- Persist RAG embeddings in a **vector database** (FAISS / Qdrant / Pinecone)
- Add **WebRTC transport** for large-scale real-time deployments

---

## Author

**Syed Daanyal Pasha**

This project was developed as part of a technical evaluation for a
Real-Time Voice AI Engineering role, with the goal of demonstrating
system design, streaming architectures, and latency optimization.

---

## License

This project is provided for evaluation and educational purposes.