"""Learn the real service. Writes probe.jsonl. Modes:
  seq    : N sequential tiny requests (latency distribution, rate-limit headers)
  conc   : K at once (concurrency/in-flight behaviour)
  ramp   : fire R requests as fast as possible until 429s (finds RPM; records retry-after, body, headers)
  recover: after a 429, poll every 2s until a request succeeds (time until it recovers)
Each request spends real quota (your daily limit is shared!). Start small: seq 20, conc 10, ramp 45."""
import asyncio, json, os, sys, time
import llm
PROMPT = [{"role": "user", "content": "Explain self-attention in one paragraph (about 80 words)."}]
async def one(tag, stream=True):
    t = time.time()
    try:
        r = await llm.chat(PROMPT, max_tokens=120, stream=stream, timeout=60)
        rec = dict(tag=tag, t=t, ok=True, ttft=r["ttft_ms"], total=r["total_ms"], tin=r["tokens_in"], tout=r["tokens_out"],
                   hdr={k: v for k, v in r["headers"].items() if "ratelimit" in k or k in ("retry-after", "x-groq-region")})
    except llm.LLMError as e:
        rec = dict(tag=tag, t=t, ok=False, kind=e.kind, status=e.status, retry_after=e.retry_after, body=e.body[:400],
                   hdr={k: v for k, v in e.headers.items() if "ratelimit" in k or k == "retry-after"})
    open("probe.jsonl", "a").write(json.dumps(rec) + "\n"); return rec
async def main():
    mode, n = sys.argv[1], int(sys.argv[2])
    if mode == "seq":
        for i in range(n): await one("seq"); await asyncio.sleep(2.5)
    elif mode == "conc": await asyncio.gather(*(one(f"conc{n}") for _ in range(n)))
    elif mode == "ramp": rs = await asyncio.gather(*(one("ramp") for _ in range(n))); print("429s:", sum(not r["ok"] for r in rs), "of", n)
    elif mode == "recover":
        t0 = time.time()
        while time.time() - t0 < 180:
            r = await one("recover")
            if r["ok"]: print("recovered after", round(time.time() - t0, 1), "s"); break
            await asyncio.sleep(2)
asyncio.run(main())
