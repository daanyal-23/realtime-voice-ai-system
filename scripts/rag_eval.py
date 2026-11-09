"""
RAG Evaluation Script for Fictional GP Knowledge Base
Evaluates retrieval accuracy against 12 in-KB questions and checks score boundaries for 4 out-of-KB questions.
Run with: python scripts/rag_eval.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    from rag import init_rag, _store, _get_embedder, SIMILARITY_THRESHOLD
except ImportError as e:
    print(f"Error importing RAG module: {e}")
    sys.exit(1)


IN_KB_CASES = [
    {
        "question": "Where is Oak Lane Medical Practice located and what is your phone number?",
        "expected": "practice_details.txt",
    },
    {
        "question": "What are your opening hours on weekdays and when do lines open?",
        "expected": "opening_hours.txt",
    },
    {
        "question": "How do I book an appointment with a GP or practice nurse?",
        "expected": "appointments.txt",
    },
    {
        "question": "How can I order a repeat prescription and how long does it take?",
        "expected": "repeat_prescriptions.txt",
    },
    {
        "question": "I need to cancel or reschedule my appointment for tomorrow.",
        "expected": "cancellations.txt",
    },
    {
        "question": "When can I call to get my blood test results?",
        "expected": "test_results.txt",
    },
    {
        "question": "How do I register as a new patient at this practice?",
        "expected": "new_patients.txt",
    },
    {
        "question": "What should I do if the surgery is closed and I need out-of-hours advice?",
        "expected": "out_of_hours.txt",
    },
    {
        "question": "My father is having severe chest pains and trouble breathing right now.",
        "expected": "emergencies.txt",
    },
    {
        "question": "What can this automated voice assistant do and what are its limitations?",
        "expected": "assistant_capabilities.txt",
    },
    {
        "question": "Can a doctor visit me at home if I am housebound and too unwell to travel?",
        "expected": "home_visits.txt",
    },
    {
        "question": "Do you offer travel vaccinations, flu jabs, and chronic disease clinics?",
        "expected": "clinics_and_services.txt",
    },
]

OUT_OF_KB_CASES = [
    "How do I change the oil and spark plugs in my car?",
    "What is the capital city of Australia?",
    "Can you give me a recipe for chocolate sourdough bread?",
    "What was the football score between Arsenal and Chelsea yesterday?",
]


async def evaluate():
    print("=" * 70)
    print("GP Knowledge Base RAG Retrieval Evaluation")
    print("=" * 70)

    print("\n[1] Initializing RAG vector index...")
    await init_rag()

    if not _store.is_ready:
        print("ERROR: Vector store failed to initialize!")
        sys.exit(1)

    print(f"Indexed {_store.embeddings.shape[0]} chunks from knowledge base.")
    embedder = _get_embedder()

    # Evaluate In-KB questions
    print("\n[2] Evaluating 12 In-KB Questions:")
    print("-" * 70)

    hits = 0
    misses = []

    for i, case in enumerate(IN_KB_CASES, 1):
        q = case["question"]
        expected = case["expected"]

        # Search top 3
        results = await _store.search(q, top_k=3)

        if results:
            top_chunk, top_score, top_source = results[0]
            is_hit = (top_source == expected)
        else:
            # Check raw max if below threshold
            loop = asyncio.get_event_loop()
            q_emb = await loop.run_in_executor(
                None, lambda: embedder.encode([q], normalize_embeddings=True)[0]
            )
            sims = _store.embeddings @ q_emb
            best_idx = int(np.argmax(sims))
            top_score = float(sims[best_idx])
            top_source = _store.sources[best_idx]
            is_hit = False

        status = "HIT" if is_hit else "MISS"
        if is_hit:
            hits += 1
        else:
            misses.append({
                "question": q,
                "expected": expected,
                "got": top_source,
                "score": top_score,
            })

        print(f"[{i:02d}] {status} | Score: {top_score:.4f}")
        print(f"     Query:    \"{q}\"")
        print(f"     Expected: {expected}")
        print(f"     Retrieved: {top_source}")
        print()

    # Evaluate Out-of-KB questions
    print("\n[3] Evaluating 4 Out-of-KB Questions (Score Boundaries):")
    print("-" * 70)

    loop = asyncio.get_event_loop()
    for i, q in enumerate(OUT_OF_KB_CASES, 1):
        results = await _store.search(q, top_k=1)
        q_emb = await loop.run_in_executor(
            None, lambda: embedder.encode([q], normalize_embeddings=True)[0]
        )
        sims = _store.embeddings @ q_emb
        best_idx = int(np.argmax(sims))
        raw_top_score = float(sims[best_idx])
        nearest_doc = _store.sources[best_idx]

        in_results = len(results) > 0
        status_label = "REJECTED (< 0.3)" if not in_results else "PASSED THRESHOLD"

        print(f"[{i:02d}] {status_label} | Raw Score: {raw_top_score:.4f} | Nearest: {nearest_doc}")
        print(f"     Query: \"{q}\"")
        print()

    # Summary
    print("=" * 70)
    print(f"RESULTS SUMMARY:")
    print(f"In-KB Retrieval Hits: {hits} / 12 ({hits / 12 * 100:.1f}%)")
    print(f"Target: >= 10 / 12 (>= 83.3%)")

    if misses:
        print(f"\nMissed Questions ({len(misses)}):")
        for m in misses:
            print(f" - Query: \"{m['question']}\"")
            print(f"   Expected: {m['expected']} | Got: {m['got']} (score: {m['score']:.4f})")
    else:
        print("\nNo misses! Perfect 12/12 retrieval accuracy.")

    print("=" * 70)

    if hits >= 10:
        print("VERIFICATION: PASS")
        sys.exit(0)
    else:
        print("VERIFICATION: FAIL (Hits < 10)")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(evaluate())
