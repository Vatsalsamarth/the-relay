"""python loadtest.py --scenario burst --source standin --n 200 [--rate 60] [--duration 60] [--setting s2_med_k4_rr] [--set-standin '{"rpm":15}']
Appends to results.csv and answers.jsonl. A person 'gives up' if no first token by patience_ms (status timeout)."""
import argparse, asyncio, csv, json, os, random, time
import httpx, llm, pipeline, retrieval
from pipeline import Cfg, Governor

COLS = "request_id,scenario,source,start_time,first_token_ms,total_ms,status,retries,tokens_in,tokens_out,retrieval_setting,answer_grade,mode".split(",")

def arrivals(kind, n, rate, duration):
    if kind == "steady": return sorted(random.uniform(0, n / rate * 60) for _ in range(n))               # ~uniform at `rate`/min
    if kind == "burst":  return sorted([random.uniform(0, 3) for _ in range(int(n * .8))] + [random.uniform(3, 30) for _ in range(n - int(n * .8))])
    if kind == "overload": return sorted(random.uniform(0, duration) for _ in range(n))                   # n >> capacity over `duration`
    raise SystemExit("kind: steady|burst|overload")

async def main():
    a = argparse.ArgumentParser()
    a.add_argument("--scenario", required=True); a.add_argument("--kind", default=None)
    a.add_argument("--source", required=True, choices=["real", "standin"]); a.add_argument("--n", type=int, default=60)
    a.add_argument("--rate", type=float, default=20); a.add_argument("--duration", type=float, default=60)
    a.add_argument("--setting", default="s2_med_k4_rr"); a.add_argument("--day-used", type=int, default=0)
    a.add_argument("--cfg", default="{}", help="JSON overrides for pipeline.Cfg"); a.add_argument("--set-standin", default=None)
    a.add_argument("--questions", default="questions.json")
    o = a.parse_args()
    if o.source == "standin":
        base = os.environ.get("LLM_BASE", "http://127.0.0.1:8001"); llm.BASE = base
        if o.set_standin: httpx.post(base + "/admin/config", json={**json.loads(o.set_standin), "reset": True})
    cfg = Cfg(setting=o.setting, **json.loads(o.cfg)); g = Governor(cfg, o.day_used)
    index = retrieval.Index(setting=o.setting)
    qs = json.load(open(o.questions)); kind = o.kind or o.scenario.split("_")[0]
    offs = arrivals(kind if kind in ("steady", "burst", "overload") else "steady", o.n, o.rate, o.duration)
    t_run = time.time(); rows = []; answers = []
    async def one(i, off):
        await asyncio.sleep(off); q = qs[i % len(qs)]; st = time.time()
        r = await pipeline.handle(f"{o.scenario}-{i}", q["q"], g, index, cfg)
        ft = r["first_token_ms"]; given_up = ft is None or ft > cfg.patience_ms
        row = dict(request_id=f"{o.scenario}-{i}", scenario=o.scenario, source=o.source, start_time=round(st - t_run, 3),
                   first_token_ms=round(ft, 1) if ft else "", total_ms=round(r["total_ms"], 1),
                   status="timeout" if given_up else r["status"], retries=r["retries"], tokens_in=r["tokens_in"], tokens_out=r["tokens_out"],
                   retrieval_setting=o.setting, answer_grade="", mode=r["mode"])
        rows.append(row); answers.append({"request_id": row["request_id"], "qid": q["id"], "answer": r["answer"], "refused": r["refused"],
                                          "wait_ms": r["wait_ms"], "work_ms": r["work_ms"], "retry_ms": r["retry_ms"], "events": r["events"], "peak_queue": g.peak_queue})
    await asyncio.gather(*(one(i, off) for i, off in enumerate(offs)))
    dur = time.time() - t_run
    new = not os.path.exists("results.csv")
    with open("results.csv", "a", newline="") as f:
        w = csv.DictWriter(f, COLS); new and w.writeheader(); w.writerows(rows)
    with open("answers.jsonl", "a") as f:
        for x in answers: f.write(json.dumps(x) + "\n")
    json.dump({"scenario": o.scenario, "duration_s": dur, "peak_queue": g.peak_queue, "n": o.n, "offered_per_min": o.n / (max(offs) or 1) * 60,
               "day_used_end": g.day_used}, open(f"meta_{o.scenario}.json", "w"))
    print(o.scenario, "done", round(dur, 1), "s; peak queue", g.peak_queue)

asyncio.run(main())
