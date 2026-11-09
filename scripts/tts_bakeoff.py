"""
TTS Bake-off Evaluation Script (Task T1)
Compares synthesis latency and audio output across available TTS engines:
- SAPI5 local voices (David, Hazel, Zira via pyttsx3)
- edge-tts (British English neural voices, if installed)
- OpenAI TTS (tts-1 / gpt-4o-mini-tts via httpx, if OPENAI_API_KEY is configured)
- Piper (if installed)

Outputs all synthesized WAV files into bakeoff/ for human listening evaluation.
Run with: python scripts/tts_bakeoff.py
"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

BAKEOFF_DIR = PROJECT_ROOT / "bakeoff"

# Three identical GP-receptionist demo sentences
BENCHMARK_SENTENCES = [
    {
        "id": "s1",
        "label": "Greeting",
        "text": "Good morning, Oak Lane Medical Practice, how may I help you today?",
    },
    {
        "id": "s2",
        "label": "Repeat Prescription",
        "text": "Repeat prescriptions take two working days to process and send to your nominated pharmacy.",
    },
    {
        "id": "s3",
        "label": "Emergency Warning",
        "text": "If this is a life-threatening medical emergency, please hang up and dial 999 immediately.",
    },
]


def synthesize_sapi5(voice_id: str, text: str, output_path: Path, rate: int = 175, volume: float = 1.0) -> float:
    """Synthesize text using pyttsx3 via an isolated worker subprocess."""
    safe_text = text.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    safe_voice = voice_id.replace("\\", "\\\\").replace("'", "\\'")
    safe_path = str(output_path).replace("\\", "\\\\")

    worker_code = f"""
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', {rate})
engine.setProperty('volume', {volume})
engine.setProperty('voice', '{safe_voice}')
engine.save_to_file('{safe_text}', '{safe_path}')
engine.runAndWait()
engine.stop()
"""
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", worker_code],
        capture_output=True,
        text=True,
        timeout=15,
    )
    duration_ms = (time.perf_counter() - t0) * 1000

    if proc.returncode != 0:
        raise RuntimeError(f"pyttsx3 failed (exit {proc.returncode}): {proc.stderr.strip()}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Output file not generated or empty: {output_path}")

    return duration_ms


async def synthesize_edgetts(voice: str, text: str, output_path: Path) -> float:
    """Synthesize text using edge-tts (if importable)."""
    import edge_tts
    t0 = time.perf_counter()
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))
    duration_ms = (time.perf_counter() - t0) * 1000

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"edge-tts output file not generated or empty: {output_path}")

    return duration_ms


async def synthesize_openai(api_key: str, voice: str, text: str, output_path: Path) -> float:
    """Synthesize text using OpenAI TTS via httpx."""
    import httpx
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": "wav",
    }

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI TTS API error {resp.status_code}: {resp.text}")
        output_path.write_bytes(resp.content)
    duration_ms = (time.perf_counter() - t0) * 1000

    return duration_ms


def get_installed_sapi5_voices() -> List[Tuple[str, str, str]]:
    """Return installed SAPI5 voices as (key, name, id)."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        results = []
        for v in voices:
            name_lower = v.name.lower()
            if "david" in name_lower:
                results.append(("sapi5_david", "SAPI5 - Microsoft David (US)", v.id))
            elif "hazel" in name_lower:
                results.append(("sapi5_hazel", "SAPI5 - Microsoft Hazel (UK)", v.id))
            elif "zira" in name_lower:
                results.append(("sapi5_zira", "SAPI5 - Microsoft Zira (US)", v.id))
            else:
                clean_name = v.name.split("-")[0].strip()
                results.append((f"sapi5_{clean_name.lower()}", f"SAPI5 - {v.name}", v.id))
        return results
    except Exception as e:
        print(f"Warning: Failed to inspect SAPI5 voices: {e}")
        return []


async def run_bakeoff():
    print("=" * 80)
    print("TTS Provider & Voice Bake-off Evaluation (Task T1)")
    print("=" * 80)
    print(f"Destination folder: {BAKEOFF_DIR.resolve()}\n")

    BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)

    results_table: List[Dict] = []
    skipped_providers: List[Tuple[str, str]] = []

    # 1. SAPI5 Voices (David, Hazel, Zira)
    sapi5_voices = get_installed_sapi5_voices()
    if sapi5_voices:
        print(f"Discovered {len(sapi5_voices)} SAPI5 voice(s):")
        for key, name, vid in sapi5_voices:
            print(f"  • {name}")
    else:
        skipped_providers.append(("SAPI5", "No SAPI5 voices detected or pyttsx3 initialization failed"))

    for key, name, vid in sapi5_voices:
        print(f"\n[Benchmarking] {name}...")
        timings = {}
        files = {}
        error_msg = None

        for s in BENCHMARK_SENTENCES:
            sid = s["id"]
            out_file = BAKEOFF_DIR / f"{key}_{sid}.wav"
            try:
                ms = synthesize_sapi5(vid, s["text"], out_file)
                timings[sid] = ms
                files[sid] = (out_file.name, out_file.stat().st_size)
                print(f"  Sentence {sid.upper()} ({s['label']}): {ms:.1f} ms -> {out_file.name} ({out_file.stat().st_size:,} bytes)")
            except Exception as e:
                error_msg = str(e)
                print(f"  Sentence {sid.upper()} FAILED: {e}")
                break

        if not error_msg and len(timings) == 3:
            avg_ms = sum(timings.values()) / 3
            results_table.append({
                "provider": name,
                "s1": timings["s1"],
                "s2": timings["s2"],
                "s3": timings["s3"],
                "avg": avg_ms,
                "files": [files[s["id"]][0] for s in BENCHMARK_SENTENCES],
            })
        else:
            skipped_providers.append((name, f"Synthesis error: {error_msg}"))

    # 2. edge-tts
    edge_tts_available = False
    try:
        import edge_tts
        edge_tts_available = True
    except ImportError:
        skipped_providers.append(("edge-tts", "Package 'edge-tts' is not installed in the environment"))

    if edge_tts_available:
        # Benchmark British English voices
        candidate_voices = [
            ("edgetts_sonia", "edge-tts - Sonia Neural (en-GB)", "en-GB-SoniaNeural"),
            ("edgetts_ryan", "edge-tts - Ryan Neural (en-GB)", "en-GB-RyanNeural"),
            ("edgetts_libby", "edge-tts - Libby Neural (en-GB)", "en-GB-LibbyNeural"),
        ]
        for key, name, voice_id in candidate_voices:
            print(f"\n[Benchmarking] {name}...")
            timings = {}
            files = {}
            error_msg = None

            for s in BENCHMARK_SENTENCES:
                sid = s["id"]
                out_file = BAKEOFF_DIR / f"{key}_{sid}.wav"
                try:
                    ms = await synthesize_edgetts(voice_id, s["text"], out_file)
                    timings[sid] = ms
                    files[sid] = (out_file.name, out_file.stat().st_size)
                    print(f"  Sentence {sid.upper()} ({s['label']}): {ms:.1f} ms -> {out_file.name} ({out_file.stat().st_size:,} bytes)")
                except Exception as e:
                    error_msg = str(e)
                    print(f"  Sentence {sid.upper()} FAILED: {e}")
                    break

            if not error_msg and len(timings) == 3:
                avg_ms = sum(timings.values()) / 3
                results_table.append({
                    "provider": name,
                    "s1": timings["s1"],
                    "s2": timings["s2"],
                    "s3": timings["s3"],
                    "avg": avg_ms,
                    "files": [files[s["id"]][0] for s in BENCHMARK_SENTENCES],
                })
            else:
                skipped_providers.append((name, f"Synthesis error: {error_msg}"))

    # 3. OpenAI TTS (via httpx)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        skipped_providers.append(("OpenAI TTS", "OPENAI_API_KEY is not configured in environment or .env"))
    else:
        try:
            import httpx
            print(f"\n[Benchmarking] OpenAI TTS (tts-1, alloy)...")
            timings = {}
            files = {}
            error_msg = None

            for s in BENCHMARK_SENTENCES:
                sid = s["id"]
                out_file = BAKEOFF_DIR / f"openai_alloy_{sid}.wav"
                try:
                    ms = await synthesize_openai(openai_key, "alloy", s["text"], out_file)
                    timings[sid] = ms
                    files[sid] = (out_file.name, out_file.stat().st_size)
                    print(f"  Sentence {sid.upper()} ({s['label']}): {ms:.1f} ms -> {out_file.name} ({out_file.stat().st_size:,} bytes)")
                except Exception as e:
                    error_msg = str(e)
                    print(f"  Sentence {sid.upper()} FAILED: {e}")
                    break

            if not error_msg and len(timings) == 3:
                avg_ms = sum(timings.values()) / 3
                results_table.append({
                    "provider": "OpenAI TTS (tts-1 alloy)",
                    "s1": timings["s1"],
                    "s2": timings["s2"],
                    "s3": timings["s3"],
                    "avg": avg_ms,
                    "files": [files[s["id"]][0] for s in BENCHMARK_SENTENCES],
                })
            else:
                skipped_providers.append(("OpenAI TTS", f"Synthesis error: {error_msg}"))
        except ImportError:
            skipped_providers.append(("OpenAI TTS", "Package 'httpx' is not installed in the environment"))

    # 4. Piper
    piper_available = False
    try:
        import piper
        piper_available = True
    except ImportError:
        skipped_providers.append(("Piper", "Package 'piper' / 'piper-tts' is not installed in the environment"))

    # Summary Output
    print("\n" + "=" * 80)
    print("TTS BAKE-OFF SYNTHESIS TIMINGS & RESULTS SUMMARY")
    print("=" * 80)

    if results_table:
        header = f"{'Provider / Voice':<36} | {'S1 (ms)':<9} | {'S2 (ms)':<9} | {'S3 (ms)':<9} | {'Avg (ms)':<9}"
        print(header)
        print("-" * len(header))
        for r in results_table:
            print(f"{r['provider']:<36} | {r['s1']:>7.1f}ms | {r['s2']:>7.1f}ms | {r['s3']:>7.1f}ms | {r['avg']:>7.1f}ms")
    else:
        print("No TTS providers successfully completed the benchmark.")

    print("\n" + "-" * 80)
    print("UNAVAILABLE / SKIPPED PROVIDERS:")
    for prov, reason in skipped_providers:
        print(f"  • {prov:<18} : {reason}")

    print("\n" + "-" * 80)
    print(f"AUDIO FILES SAVED TO: {BAKEOFF_DIR.resolve()}")
    saved_files = sorted(list(BAKEOFF_DIR.glob("*.wav")))
    for f in saved_files:
        print(f"  • {f.name:<25} ({f.stat().st_size:,} bytes)")

    print("\n" + "=" * 80)
    print("HUMAN EVALUATION STATUS:")
    print("Listening quality comparison and final voice selection: PENDING HUMAN (Task H2/H5)")
    print("Target first-sentence latency: <= 1000 ms")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_bakeoff())

