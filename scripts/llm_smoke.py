"""
Smoke benchmark script for Groq GPT-OSS models.
Evaluates first-content-token latency across typical GP-receptionist queries.
"""

import asyncio
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from groq import AsyncGroq

from config import (
    LLM_FALLBACK_MODEL,
    LLM_MAX_COMPLETION_TOKENS,
    LLM_MODEL,
    LLM_REASONING_EFFORT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_S,
)

load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

# 3 GP-receptionist questions
GP_QUESTIONS = [
    "Hello, I'd like to book an appointment with Dr. Smith for tomorrow morning if possible.",
    "Can I get a repeat prescription for my blood pressure medication, or do I need to see the doctor first?",
    "What are your clinic opening hours this Thursday, and is parking available on site?",
]

SYSTEM_PROMPT = """You are a helpful, professional GP clinic receptionist.
Keep your response concise, polite, and natural for voice conversation. Speak in plain sentences."""

REPETITIONS = 5


async def benchmark_single_run(
    client: AsyncGroq,
    model: str,
    question: str,
) -> Dict[str, Optional[float]]:
    """
    Execute a single streaming completion and record first-content-token latency.

    Returns dict with:
        'latency_ms': float or None
        'error': str or None
        'no_content': bool
        'token_count': int
    """
    t0 = time.perf_counter()
    first_latency_ms: Optional[float] = None
    token_count = 0
    content_chunks: List[str] = []

    create_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "stream": True,
        "temperature": LLM_TEMPERATURE,
        "max_completion_tokens": LLM_MAX_COMPLETION_TOKENS,
    }

    if "gpt-oss" in model.lower():
        create_kwargs["reasoning_effort"] = LLM_REASONING_EFFORT

    try:
        async with asyncio.timeout(LLM_TIMEOUT_S):
            stream = await client.chat.completions.create(**create_kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    if first_latency_ms is None:
                        first_latency_ms = (time.perf_counter() - t0) * 1000
                    token_count += 1
                    content_chunks.append(content)

        if token_count == 0:
            return {
                "latency_ms": None,
                "error": None,
                "no_content": True,
                "token_count": 0,
            }

        return {
            "latency_ms": first_latency_ms,
            "error": None,
            "no_content": False,
            "token_count": token_count,
        }

    except TimeoutError:
        return {
            "latency_ms": None,
            "error": f"Timeout after {LLM_TIMEOUT_S}s",
            "no_content": token_count == 0,
            "token_count": token_count,
        }
    except Exception as e:
        # Sanitize error message to prevent accidental secret leakage
        err_msg = f"{type(e).__name__}: {str(e)[:120]}"
        return {
            "latency_ms": None,
            "error": err_msg,
            "no_content": token_count == 0,
            "token_count": token_count,
        }


async def benchmark_model(client: AsyncGroq, model: str) -> Dict:
    """Benchmark all GP questions with repetitions for a single model."""
    print(f"\n=======================================================")
    print(f"Benchmarking Model: {model}")
    print(f"Configuration: reasoning_effort={LLM_REASONING_EFFORT}, max_completion_tokens={LLM_MAX_COMPLETION_TOKENS}")
    print(f"Total runs: {len(GP_QUESTIONS)} questions × {REPETITIONS} repeats = {len(GP_QUESTIONS) * REPETITIONS} calls")
    print(f"=======================================================")

    latencies: List[float] = []
    error_count = 0
    no_content_count = 0

    for q_idx, question in enumerate(GP_QUESTIONS, 1):
        print(f"\n[Question {q_idx}/3] \"{question[:50]}...\"")
        for rep in range(1, REPETITIONS + 1):
            res = await benchmark_single_run(client, model, question)

            if res["error"]:
                error_count += 1
                print(f"  Rep {rep}/{REPETITIONS}: ERROR - {res['error']}")
            elif res["no_content"]:
                no_content_count += 1
                print(f"  Rep {rep}/{REPETITIONS}: NO CONTENT (0 tokens)")
            else:
                lat = res["latency_ms"]
                latencies.append(lat)
                print(f"  Rep {rep}/{REPETITIONS}: First token = {lat:.1f}ms ({res['token_count']} tokens)")

            # Slight pause between requests to respect rate limits
            await asyncio.sleep(0.5)

    p50 = statistics.median(latencies) if latencies else None
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (max(latencies) if latencies else None)
    max_lat = max(latencies) if latencies else None
    min_lat = min(latencies) if latencies else None
    mean_lat = statistics.mean(latencies) if latencies else None

    return {
        "model": model,
        "total_runs": len(GP_QUESTIONS) * REPETITIONS,
        "successful_runs": len(latencies),
        "error_count": error_count,
        "no_content_count": no_content_count,
        "p50_ms": p50,
        "max_ms": max_lat,
        "min_ms": min_lat,
        "mean_ms": mean_lat,
        "latencies": latencies,
    }


async def main():
    print("Initializing Groq Async Client for Benchmarking...")
    client = AsyncGroq()

    models = [LLM_MODEL, LLM_FALLBACK_MODEL]
    results = []

    for model in models:
        stats = await benchmark_model(client, model)
        results.append(stats)

    print("\n\n" + "=" * 70)
    print("BENCHMARK SUMMARY REPORT")
    print("=" * 70)
    print(f"{'Model':<25} | {'Runs':<5} | {'p50 (ms)':<10} | {'Max (ms)':<10} | {'Errors':<7} | {'No-Content':<10}")
    print("-" * 75)

    for r in results:
        p50_str = f"{r['p50_ms']:.1f}" if r["p50_ms"] is not None else "N/A"
        max_str = f"{r['max_ms']:.1f}" if r["max_ms"] is not None else "N/A"
        print(
            f"{r['model']:<25} | "
            f"{r['total_runs']:<5} | "
            f"{p50_str:<10} | "
            f"{max_str:<10} | "
            f"{r['error_count']:<7} | "
            f"{r['no_content_count']:<10}"
        )

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
