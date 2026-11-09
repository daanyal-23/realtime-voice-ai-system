"""
Evaluation script to benchmark OpenAI gpt-4o-mini-tts (Marin) across speed settings:
- Speed 1.0 (baseline with old instructions)
- Speed 1.0 (with new concise brisk instructions)
- Speed 1.15 (with new concise brisk instructions)
- Speed 1.25 (with new concise brisk instructions)

Measures:
- Synthesis latency (ms)
- WAV audio duration (seconds) via wave module
- File size (bytes)
- Effective speaking rate (words per minute)
"""

import asyncio
import os
import sys
import time
import wave
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import httpx

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

OLD_INSTRUCTIONS = (
    "Speak in a warm, welcoming, professional British English GP receptionist tone. "
    "Maintain a clear, calm, and natural conversational pace."
)

NEW_INSTRUCTIONS = (
    "Speak in a warm, professional British English receptionist voice. "
    "Use a natural, confident, moderately brisk conversational pace. "
    "Avoid unnecessarily long pauses. Keep pronunciation clear and reassuring."
)


async def synthesize(voice: str, text: str, instructions: str, speed: float, out_path: Path, api_key: str):
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini-tts",
        "voice": voice,
        "input": text,
        "instructions": instructions,
        "response_format": "wav",
        "speed": speed,
    }

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text[:120]}")
        wav_bytes = resp.content
    latency_ms = (time.perf_counter() - t0) * 1000

    out_path.write_bytes(wav_bytes)

    # Read WAV metadata
    with wave.open(str(out_path), "rb") as wf:
        n_frames = wf.getnframes()
        framerate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        duration_s = n_frames / float(framerate)

    word_count = len(text.split())
    wpm = (word_count / duration_s) * 60.0 if duration_s > 0 else 0

    return {
        "latency_ms": latency_ms,
        "duration_s": duration_s,
        "framerate": framerate,
        "channels": n_channels,
        "sampwidth": sampwidth,
        "size_bytes": len(wav_bytes),
        "wpm": wpm,
    }


async def run_benchmark():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set in environment or .env")
        sys.exit(1)

    configurations = [
        ("old_inst_1.0", OLD_INSTRUCTIONS, 1.0),
        ("new_inst_1.0", NEW_INSTRUCTIONS, 1.0),
        ("new_inst_1.15", NEW_INSTRUCTIONS, 1.15),
        ("new_inst_1.25", NEW_INSTRUCTIONS, 1.25),
    ]

    print("=" * 90)
    print("OPENAI MARIN TTS CONTROLLED SPEED & INSTRUCTION EVALUATION")
    print("=" * 90)

    results = []

    for item in BENCHMARK_SENTENCES:
        s_id = item["id"]
        text = item["text"]
        print(f"\n--- Sentence: {s_id} ({len(text.split())} words) ---")
        print(f"Text: \"{text}\"")

        for cfg_name, instructions, speed in configurations:
            out_file = BAKEOFF_DIR / f"marin_{s_id}_{cfg_name}.wav"
            try:
                data = await synthesize("marin", text, instructions, speed, out_file, api_key)
                data.update({
                    "sentence_id": s_id,
                    "cfg_name": cfg_name,
                    "speed": speed,
                    "instructions_type": "old" if instructions == OLD_INSTRUCTIONS else "new",
                    "file": str(out_file.name),
                })
                results.append(data)
                print(
                    f"[{cfg_name:14s}] Latency: {data['latency_ms']:6.0f}ms | "
                    f"Duration: {data['duration_s']:5.2f}s | "
                    f"Rate: {data['framerate']}Hz | "
                    f"WPM: {data['wpm']:5.1f} | "
                    f"Size: {data['size_bytes']:6d} B"
                )
            except Exception as e:
                print(f"[{cfg_name:14s}] FAILED: {e}")

    # Summary Table by Configuration
    print("\n" + "=" * 90)
    print("AVERAGE SUMMARY BY CONFIGURATION")
    print("=" * 90)
    print(f"{'Configuration':<16} | {'Avg Latency (ms)':<16} | {'Avg Audio Duration (s)':<22} | {'Avg WPM':<10}")
    print("-" * 90)

    for cfg_name, _, speed in configurations:
        cfg_results = [r for r in results if r["cfg_name"] == cfg_name]
        if cfg_results:
            avg_lat = sum(r["latency_ms"] for r in cfg_results) / len(cfg_results)
            avg_dur = sum(r["duration_s"] for r in cfg_results) / len(cfg_results)
            avg_wpm = sum(r["wpm"] for r in cfg_results) / len(cfg_results)
            print(f"{cfg_name:<16} | {avg_lat:<16.0f} | {avg_dur:<22.2f} | {avg_wpm:<10.1f}")

    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(run_benchmark())

