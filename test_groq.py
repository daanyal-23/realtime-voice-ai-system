# test_groq.py
import asyncio, time, os
from dotenv import load_dotenv
load_dotenv()
from groq import AsyncGroq

async def test():
    client = AsyncGroq()
    t = time.perf_counter()
    stream = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":"Say hello in one word"}],
        stream=True, max_tokens=10
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(f"First token: {(time.perf_counter()-t)*1000:.0f}ms")
            break

asyncio.run(test())