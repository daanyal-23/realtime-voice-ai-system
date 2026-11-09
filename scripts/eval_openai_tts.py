"""
OpenAI Neural TTS (gpt-4o-mini-tts) vs SAPI5 Hazel Live Evaluation Script.
Synthesizes benchmark GP receptionist demo sentences using:
- SAPI5 Hazel (local Windows pyttsx3)
- OpenAI gpt-4o-mini-tts with voice 'cedar' (via httpx)
- OpenAI gpt-4o-mini-tts with voice 'marin' (via httpx)

Saves output WAV files to bakeoff/ and measures actual generation latency.
Run with:
    python scripts/eval_openai_tts.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import httpx
from tts import TTSSession, _tts_worker_script, RATE, VOLUME

BAKEOFF_DIR = PROJECT_ROOT / "bakeoff"
BAKEOFF_DIR.mkdir(exist_ok=True)

BENCHMARK_SENTENCES = [
    {
        "id": "s1_greeting",
        "text": "Good morning, Oak Lane Medical Practice, how may I help you today?",
    },
    {
        "id": "s2_prescription",
        "text": "Repeat prescriptions take two working days to process and send to your nominated pharmacy.",
    },
    {
        "id": "s3_emergency",
        "text": "If this is a life-threatening medical emergency, please hang up and dial 999 immediately.",
    },
]

INSTRUCTIONS = (
    "Speak in a warm, welcoming, professional British English GP receptionist tone. "
    "Maintain a clear, calm, and natural conversational pace."
)


async def synthesize_sapi5_hazel(text: str, out_path: Path) -> float:
    """Synthesize using SAPI5 Hazel via TTSSession."""
    session = TTSSession(session_id="eval_sapi5", voice_name="Hazel", provider="sapi5")
    t0 = time.perf_counter()
    chunks = []
    async for chunk in session.stream_tts_audio(text):
        chunks.append(chunk)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    await session.close()

    if chunks:
        out_path.write_bytes(chunks[0])
    return elapsed_ms


async def synthesize_openai(voice: str, text: str, out_path: Path, api_key: str) -> float:
    """Synthesize using OpenAI gpt-4o-mini-tts with British conversational instructions."""
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini-tts",
        "voice": voice,
        "input": text,
        "instructions": INSTRUCTIONS,
        "response_format": "wav",
    }

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text[:120]}")
        wav_bytes = resp.content
    elapsed_ms = (time.perf_counter() - t0) * 1000

    out_path.write_bytes(wav_bytes)
    return elapsed_ms


async def run_evaluation():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    has_key = bool(api_key)

    print("=" * 75)
    print("TTS Evaluation: SAPI5 Hazel vs OpenAI Neural (gpt-4o-mini-tts)")
    print("=" * 75)
    print(f"OPENAI_API_KEY present: {has_key}")
    if not has_key:
        print("Note: To benchmark OpenAI live, add OPENAI_API_KEY to your .env file.")
    print("-" * 75)

    results = {}

    for item in BENCHMARK_SENTENCES:
        sid = item["id"]
        text = item["text"]
        print(f"\n[Phrase: {sid}] '{text}'")

        # 1. SAPI5 Hazel
        hazel_file = BAKEOFF_DIR / f"sapi5_hazel_{sid}.wav"
        try:
            t_hazel = await synthesize_sapi5_hazel(text, hazel_file)
            print(f"  - SAPI5 Hazel:         {t_hazel:6.1f} ms -> {hazel_file.name} ({hazel_file.stat().st_size} bytes)")
            results.setdefault("sapi5_hazel", []).append(t_hazel)
        except Exception as e:
            print(f"  - SAPI5 Hazel:         FAILED ({e})")

        # 2. OpenAI Cedar
        if has_key:
            cedar_file = BAKEOFF_DIR / f"openai_cedar_{sid}.wav"
            try:
                t_cedar = await synthesize_openai("cedar", text, cedar_file, api_key)
                print(f"  - OpenAI Cedar:        {t_cedar:6.1f} ms -> {cedar_file.name} ({cedar_file.stat().st_size} bytes)")
                results.setdefault("openai_cedar", []).append(t_cedar)
            except Exception as e:
                print(f"  - OpenAI Cedar:        FAILED ({e})")

        # 3. OpenAI Marin
        if has_key:
            marin_file = BAKEOFF_DIR / f"openai_marin_{sid}.wav"
            try:
                t_marin = await synthesize_openai("marin", text, marin_file, api_key)
                print(f"  - OpenAI Marin:        {t_marin:6.1f} ms -> {marin_file.name} ({marin_file.stat().st_size} bytes)")
                results.setdefault("openai_marin", []).append(t_marin)
            except Exception as e:
                print(f"  - OpenAI Marin:        FAILED ({e})")

    print("\n" + "=" * 75)
    print("Summary Latency Comparison:")
    print("-" * 75)
    for provider, times in results.items():
        avg = sum(times) / len(times)
        print(f"  {provider:<16}: avg {avg:6.1f} ms  (min {min(times):.1f} ms, max {max(times):.1f} ms)")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(run_evaluation())

