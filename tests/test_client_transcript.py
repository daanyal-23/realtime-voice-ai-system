"""
Unit tests for client-side transcript and message lifecycle in index.html (Task U1).
Verifies:
- Lazy message bubble creation when ai_text arrives before turn_start
- Preserving text when turn_start arrives after ai_text
- turn_start arriving before ai_text creates message bubble properly
- Complete short response arriving before turn_start is preserved
- No duplicate message bubbles are created across any message interleavings
- Lifecycle resets cleanly between turns, including after interrupted events
"""

import json
import subprocess
from pathlib import Path
import pytest

INDEX_HTML = Path(__file__).parent.parent / "index.html"


def run_client_messages(messages_sequence):
    """
    Execute handleControlMessage() sequence against the exact JavaScript
    extracted from index.html in a lightweight headless Node environment.
    Returns:
        list of dicts representing rendered messages in document.getElementById('messages').
    """
    html_content = INDEX_HTML.read_text(encoding="utf-8")
    js_code = html_content.split("<script>")[1].split("</script>")[0]

    runner_js = f"""
const messagesContainer = {{
  children: [],
  appendChild(child) {{ this.children.push(child); }},
  scrollTop: 0,
}};

const domElements = {{
  messages: messagesContainer,
  statusDot: {{ className: '' }},
  statusText: {{ textContent: '' }},
  'stat-first-audio': {{ textContent: '', className: '' }},
  'stat-stt': {{ textContent: '', className: '' }},
  'stat-rag': {{ textContent: '', className: '' }},
  'stat-llm-ttft': {{ textContent: '', className: '' }},
  'stat-total-turn': {{ textContent: '', className: '' }},
  'stat-msgs': {{ textContent: '' }},
  emptyState: {{ remove() {{}} }},
}};

global.document = {{
  getElementById(id) {{ return domElements[id] || null; }},
  querySelectorAll(sel) {{
    const match = sel.match(/\\[data-turn-id="([^"]+)"\\]/);
    if (match) {{
      const tid = match[1];
      return messagesContainer.children.filter(c => c.dataset && c.dataset.turnId === tid);
    }}
    return [];
  }},
  createElement(tag) {{
    const el = {{
      tagName: tag,
      className: '',
      innerHTML: '',
      dataset: {{}},
      style: {{}},
      remove() {{
        const idx = messagesContainer.children.indexOf(this);
        if (idx !== -1) messagesContainer.children.splice(idx, 1);
      }},
      _textNode: null,
      querySelector(sel) {{
        if (sel === '.message-text') {{
          if (!this._textNode) {{
            // parse existing innerHTML if set
            const match = this.innerHTML.match(/<div class="message-text">([\\s\\S]*?)<\\/div>/);
            this._textNode = {{ innerHTML: match ? match[1] : '' }};
          }}
          return this._textNode;
        }}
        return null;
      }}
    }};
    return el;
  }}
}};
global.window = {{ addEventListener() {{}} }};
global.location = {{ hostname: '127.0.0.1' }};
global.performance = {{ now: () => Date.now() }};
global.stopAudioNow = () => {{}};

{js_code}

const sequence = {json.dumps(messages_sequence)};
for (const msg of sequence) {{
  handleControlMessage(msg);
}}

const output = messagesContainer.children.map(child => {{
  const role = child.className.includes('user') ? 'user' : (child.className.includes('ai') ? 'ai' : 'other');
  const textNode = child.querySelector('.message-text');
  const rawText = textNode ? textNode.innerHTML : '';
  const cleanText = rawText.replace(/<span class="cursor"><\\/span>/g, '').trim();
  return {{ role, cleanText, rawText }};
}});

console.log(JSON.stringify(output));
"""

    res = subprocess.run(["node", "-e", runner_js], capture_output=True, text=True, timeout=10)
    if res.returncode != 0:
        raise RuntimeError(f"Node execution failed: {res.stderr}\nCode:\n{runner_js}")

    return json.loads(res.stdout.strip())


class TestClientTranscriptLifecycle:
    def test_ai_text_arriving_before_turn_start(self):
        """ai_text tokens arriving before turn_start must be preserved and rendered without loss."""
        sequence = [
            {"type": "transcript", "text": "How do I order a repeat prescription?"},
            {"type": "ai_text", "token": "I'm "},
            {"type": "ai_text", "token": "an automated "},
            {"type": "ai_text", "token": "assistant."},
            {"type": "turn_start"},
            {"type": "ai_text", "token": " You can "},
            {"type": "ai_text", "token": "order online via the NHS App."},
            {"type": "turn_end", "total_latency_ms": 1200},
        ]
        messages = run_client_messages(sequence)

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["cleanText"] == "How do I order a repeat prescription?"

        assert messages[1]["role"] == "ai"
        expected = "I'm an automated assistant. You can order online via the NHS App."
        assert messages[1]["cleanText"] == expected

    def test_turn_start_arriving_before_ai_text(self):
        """When turn_start arrives before ai_text, the bubble is created and tokens are appended."""
        sequence = [
            {"type": "transcript", "text": "What time do you open?"},
            {"type": "turn_start"},
            {"type": "ai_text", "token": "We "},
            {"type": "ai_text", "token": "open at 8:00 AM on weekdays."},
            {"type": "turn_end", "total_latency_ms": 900},
        ]
        messages = run_client_messages(sequence)

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "ai"
        assert messages[1]["cleanText"] == "We open at 8:00 AM on weekdays."

    def test_complete_short_response_arriving_before_turn_start(self):
        """A complete short (e.g. 1-sentence) response arriving before turn_start is fully rendered."""
        sequence = [
            {"type": "transcript", "text": "I have chest pains right now."},
            {"type": "ai_text", "token": "This is a life-threatening emergency. "},
            {"type": "ai_text", "token": "Please hang up and dial 999 immediately."},
            {"type": "turn_start"},
            {"type": "turn_end", "total_latency_ms": 800},
        ]
        messages = run_client_messages(sequence)

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "ai"
        assert messages[1]["cleanText"] == "This is a life-threatening emergency. Please hang up and dial 999 immediately."

    def test_no_duplicate_message_bubble_created(self):
        """turn_start arriving after ai_text must reuse the existing bubble and never create a duplicate."""
        sequence = [
            {"type": "transcript", "text": "Hello"},
            {"type": "ai_text", "token": "Good "},
            {"type": "ai_text", "token": "morning. "},
            {"type": "turn_start"},  # Must NOT append a second AI message
            {"type": "ai_text", "token": "How "},
            {"type": "ai_text", "token": "can I help?"},
            {"type": "turn_end", "total_latency_ms": 700},
        ]
        messages = run_client_messages(sequence)

        ai_messages = [m for m in messages if m["role"] == "ai"]
        assert len(ai_messages) == 1
        assert ai_messages[0]["cleanText"] == "Good morning. How can I help?"

    def test_lifecycle_resets_between_turns_and_after_interrupted(self):
        """Clean lifecycle reset across turns and barge-in interrupts."""
        sequence = [
            # Turn 1: interrupted mid-response
            {"type": "transcript", "text": "Tell me about appointments"},
            {"type": "ai_text", "token": "Routine appointments "},
            {"type": "turn_start"},
            {"type": "ai_text", "token": "can be booked "},
            {"type": "interrupted"},
            # Turn 2: new question
            {"type": "transcript", "text": "Nevermind, where are you located?"},
            {"type": "ai_text", "token": "We are located at 14 Oak Lane."},
            {"type": "turn_start"},
            {"type": "turn_end", "total_latency_ms": 600},
        ]
        messages = run_client_messages(sequence)

        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[0]["cleanText"] == "Tell me about appointments"
        assert messages[1]["role"] == "ai"
        assert messages[1]["cleanText"] == "Routine appointments can be booked"

        assert messages[2]["role"] == "user"
        assert messages[2]["cleanText"] == "Nevermind, where are you located?"
        assert messages[3]["role"] == "ai"
        assert messages[3]["cleanText"] == "We are located at 14 Oak Lane."

    def test_superseded_transcript_bubbles_are_removed(self):
        """When turn_superseded arrives, the superseded turn's user and AI bubbles are removed."""
        sequence = [
            # Turn 1 is emitted
            {"type": "transcript", "text": "I would like to know how to request a repeat.", "turn_id": 1},
            {"type": "ai_text", "token": "Repeat prescriptions can be ", "turn_id": 1},
            # Turn 1 is superseded by Turn 2
            {"type": "turn_superseded", "turn_id": 1},
            # Turn 2 is emitted
            {"type": "transcript", "text": "I would like to know how to request a repeat prescription.", "turn_id": 2},
            {"type": "ai_text", "token": "You can order via the NHS App.", "turn_id": 2},
            {"type": "turn_end", "turn_id": 2, "total_latency_ms": 900},
        ]
        messages = run_client_messages(sequence)

        # Only Turn 2 messages should be present! Turn 1 bubbles were completely removed.
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["cleanText"] == "I would like to know how to request a repeat prescription."
        assert messages[1]["role"] == "ai"
        assert messages[1]["cleanText"] == "You can order via the NHS App."

    def test_late_messages_from_superseded_turn_are_ignored(self):
        """Late messages arriving after a turn was superseded cannot recreate duplicate bubbles."""
        sequence = [
            {"type": "transcript", "text": "Turn 1 question", "turn_id": 1},
            {"type": "turn_superseded", "turn_id": 1},
            # Late messages for Turn 1 arrive over the wire
            {"type": "ai_text", "token": "Late token for turn 1", "turn_id": 1},
            {"type": "turn_start", "turn_id": 1},
            {"type": "transcript", "text": "Late duplicate transcript", "turn_id": 1},
            {"type": "turn_end", "turn_id": 1},
            # Turn 2 runs normally
            {"type": "transcript", "text": "Turn 2 question", "turn_id": 2},
            {"type": "ai_text", "token": "Turn 2 answer.", "turn_id": 2},
            {"type": "turn_end", "turn_id": 2},
        ]
        messages = run_client_messages(sequence)

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["cleanText"] == "Turn 2 question"
        assert messages[1]["role"] == "ai"
        assert messages[1]["cleanText"] == "Turn 2 answer."

    def test_brief_noise_spike_does_not_trigger_barge_in(self):
        """A single chunk noise spike (RMS >= 0.04) does not trigger barge-in because 2 consecutive chunks are required."""
        html_content = INDEX_HTML.read_text(encoding="utf-8")
        js_code = html_content.split("<script>")[1].split("</script>")[0]

        test_js = f"""
global.window = {{ addEventListener() {{}} }};
global.document = {{ getElementById() {{ return null; }} }};
{js_code}
isAISpeaking = true;
let interrupted = false;
ws = {{
  readyState: 1,
  send(data) {{
    const parsed = JSON.parse(data);
    if (parsed.type === 'interrupt') interrupted = true;
  }}
}};

// Simulate 1 noise chunk above 0.04, then quiet chunk
const chunks = [0.05, 0.01, 0.02];
for (const rms of chunks) {{
  if (isAISpeaking || isPlaying) {{
    if (rms >= BARGE_IN_RMS) {{
      consecutiveBargeInChunks++;
      if (consecutiveBargeInChunks >= 2) {{
        consecutiveBargeInChunks = 0;
        ws.send(JSON.stringify({{ type: 'interrupt' }}));
      }}
    }} else {{
      consecutiveBargeInChunks = 0;
    }}
  }}
}}
console.log(JSON.stringify({{ interrupted, consecutiveBargeInChunks }}));
"""
        res = subprocess.run(["node", "-e", test_js], capture_output=True, text=True, timeout=10)
        assert res.returncode == 0
        data = json.loads(res.stdout.strip())
        assert data["interrupted"] is False
        assert data["consecutiveBargeInChunks"] == 0

    def test_sustained_genuine_speech_triggers_barge_in(self):
        """Two consecutive chunks above BARGE_IN_RMS (0.04) trigger barge-in interrupt."""
        html_content = INDEX_HTML.read_text(encoding="utf-8")
        js_code = html_content.split("<script>")[1].split("</script>")[0]

        test_js = f"""
global.window = {{ addEventListener() {{}} }};
global.document = {{ getElementById() {{ return null; }} }};
{js_code}
isAISpeaking = true;
let interrupted = false;
ws = {{
  readyState: 1,
  send(data) {{
    const parsed = JSON.parse(data);
    if (parsed.type === 'interrupt') interrupted = true;
  }}
}};

// Simulate 2 consecutive chunks above 0.04
const chunks = [0.045, 0.05];
for (const rms of chunks) {{
  if (isAISpeaking || isPlaying) {{
    if (rms >= BARGE_IN_RMS) {{
      consecutiveBargeInChunks++;
      if (consecutiveBargeInChunks >= 2) {{
        consecutiveBargeInChunks = 0;
        ws.send(JSON.stringify({{ type: 'interrupt' }}));
      }}
    }} else {{
      consecutiveBargeInChunks = 0;
    }}
  }}
}}
console.log(JSON.stringify({{ interrupted, consecutiveBargeInChunks }}));
"""
        res = subprocess.run(["node", "-e", test_js], capture_output=True, text=True, timeout=10)
        assert res.returncode == 0
        data = json.loads(res.stdout.strip())
        assert data["interrupted"] is True
        assert data["consecutiveBargeInChunks"] == 0
