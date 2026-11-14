"""Stand-in model service (OpenAI-compatible). Run: uvicorn standin:app --port 8001
Knobs (POST /admin/config): latency lognormal, per-token ms, load slowdown, failures, slow tail,
rpm/tpm/rpd limits (429 + retry-after + x-ratelimit headers), max concurrency, streaming."""
import asyncio, json, math, random, re, time, collections
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

CFG = dict(ttft_mu=math.log(350), ttft_sigma=0.5, per_tok_ms=8.0, load_k=0.08, fail_rate=0.0,
           slow_rate=0.02, slow_mult=8.0, rpm=30, tpm=6000, rpd=14400, max_conc=0, out_tokens=90)
try: CFG.update(json.load(open("standin_config.json")))   # written by calibrate.py
except Exception: pass
S = dict(reqs=collections.deque(), toks=collections.deque(), day=0, inflight=0, peak=0, n429=0)
app = FastAPI()
etok = lambda t: max(1, len(t) // 4)

@app.post("/admin/config")
async def cfg(r: Request):
    CFG.update(await r.json())
    if (await r.json()).get("reset"): S.update(reqs=collections.deque(), toks=collections.deque(), day=0, n429=0, peak=0)
    return CFG
@app.get("/admin/stats")
async def stats(): return {"day": S["day"], "inflight": S["inflight"], "peak": S["peak"], "n429": S["n429"], "cfg": CFG}

def make_reply(msgs, max_tokens):
    sys_ = msgs[0]["content"]
    user = msgs[-1]["content"]

    # Understand stage: produce a search query.
    if "Reply JSON only" in sys_:
        q = user.split("QUESTION:")[-1].strip()
        return json.dumps({"query": q, "in_scope": True})

    # Draft stage: inspect the numbered evidence passages.
    if "Answer ONLY from the numbered passages" in sys_:
        passages = re.findall(
            r"\[S(\d+)\]\s*\([^\n]*\)\n(.*?)(?=\n\[S\d+\]|\n\nQUESTION:|$)",
            user,
            re.S
        )

        question = user.split("QUESTION:")[-1].strip().lower()

        if not passages:
            return "NOT_IN_PAPERS"

        # Only return passages that contain meaningful question terms.
        stopwords = {
            "what", "does", "have", "the", "is", "are", "was",
            "were", "how", "many", "what", "which", "for",
            "and", "in", "of", "to", "a", "an", "use"
        }

        terms = set(re.findall(r"[a-z0-9_.-]+", question))
        terms -= stopwords

        scored = []
        for sid, text in passages:
            words = set(re.findall(r"[a-z0-9_.-]+", text.lower()))
            score = len(terms & words)

            if score:
                scored.append((score, sid, text))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            return "NOT_IN_PAPERS"

        output = []
        for _, sid, text in scored[:2]:
            excerpt = " ".join(text.split()[:35])
            output.append(f"The relevant passage states: \"{excerpt}\" [S{sid}].")

        return " ".join(output)

    return "{}"
@app.post("/chat/completions")
async def chat(r: Request):
    b = await r.json(); now = time.time()
    tin = sum(etok(m["content"]) for m in b["messages"]); mt = b.get("max_tokens", 300)
    for dq in (S["reqs"], S["toks"]):
        while dq and now - dq[0][0] > 60: dq.popleft()
    used_t = sum(x[1] for x in S["toks"])
    def hdrs(): return {"x-ratelimit-limit-requests": str(CFG["rpd"]), "x-ratelimit-remaining-requests": str(max(0, CFG["rpd"] - S["day"])),
                        "x-ratelimit-limit-tokens": str(CFG["tpm"]), "x-ratelimit-remaining-tokens": str(max(0, CFG["tpm"] - used_t))}
    def limited(msg, wait):
        S["n429"] += 1
        return JSONResponse({"error": {"message": msg, "type": "rate_limit_exceeded"}}, 429,
                            {**hdrs(), "retry-after": str(max(1, math.ceil(wait)))})
    if S["day"] >= CFG["rpd"]: return limited("daily request limit reached", 3600)
    if len(S["reqs"]) >= CFG["rpm"]: return limited("requests per minute limit", 60 - (now - S["reqs"][0][0]))
    if used_t + tin + mt > CFG["tpm"] and S["toks"]: return limited("tokens per minute limit", 60 - (now - S["toks"][0][0]))
    if CFG["max_conc"] and S["inflight"] >= CFG["max_conc"]: return limited("too many concurrent requests", 2)
    S["reqs"].append((now, 1)); S["toks"].append((now, tin + mt)); S["day"] += 1
    S["inflight"] += 1; S["peak"] = max(S["peak"], S["inflight"])
    slow = 1 + CFG["load_k"] * (S["inflight"] - 1)
    if random.random() < CFG["slow_rate"]: slow *= CFG["slow_mult"]
    ttft = random.lognormvariate(CFG["ttft_mu"], CFG["ttft_sigma"]) * slow
    try:
        await asyncio.sleep(ttft / 1000)
        if random.random() < CFG["fail_rate"]:
            return JSONResponse({"error": {"message": "internal"}}, 500)
        text = make_reply(b["messages"], mt)
        words = text.split(" "); toks_out = min(mt, max(1, int(len(words) * 1.3)))
        if not b.get("stream"):
            await asyncio.sleep(toks_out * CFG["per_tok_ms"] * slow / 1000)
            return JSONResponse({"choices": [{"message": {"content": text}}],
                                 "usage": {"prompt_tokens": tin, "completion_tokens": toks_out}}, headers=hdrs())
        async def gen():
            try:
                for i, w in enumerate(words):
                    await asyncio.sleep(1.3 * CFG["per_tok_ms"] * slow / 1000)
                    yield "data: " + json.dumps({"choices": [{"delta": {"content": w + (" " if i < len(words) - 1 else "")}}]}) + "\n\n"
                yield "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": tin, "completion_tokens": toks_out}}) + "\n\ndata: [DONE]\n\n"
            finally:
                S["inflight"] -= 1
        S["inflight"] += 1  # compensated by gen()'s finally; outer finally decrements the original
        return StreamingResponse(gen(), media_type="text/event-stream", headers=hdrs())
    finally:
        S["inflight"] -= 1
