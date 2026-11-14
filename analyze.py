"""Analyze system performance and answer quality separately."""
import csv
import glob
import json
import collections
import numpy as np

rows = list(csv.DictReader(open("results.csv")))

if not rows:
    raise SystemExit("No results found.")

by = collections.defaultdict(list)
for r in rows:
    by[r["scenario"]].append(r)

def pct(xs, p):
    return round(float(np.percentile(xs, p)), 1) if xs else "n/a"

print("\n========== TABLE 2: SYSTEM PERFORMANCE ==========")

for sc, rs in by.items():
    meta = json.load(open(f"meta_{sc}.json")) if glob.glob(f"meta_{sc}.json") else {}

    n = len(rs)
    counts = collections.Counter(r["status"] for r in rs)
    ft = [float(r["first_token_ms"]) for r in rs if r["first_token_ms"]]
    tt = [float(r["total_ms"]) for r in rs if r["total_ms"]]

    print(f"\nScenario: {sc}")
    print(f"Requests: {n} | Duration: {meta.get('duration_s', 'n/a')} sec")
    print(f"Offered/min: {meta.get('offered_per_min', 'n/a')}")
    print(f"First-token latency p50/p95/p99: {pct(ft,50)}/{pct(ft,95)}/{pct(ft,99)} ms")
    print(f"Total latency p50/p95/p99: {pct(tt,50)}/{pct(tt,95)}/{pct(tt,99)} ms")
    print(f"Success: {100*counts['ok']/n:.1f}%")
    print(f"Rate-limited: {100*counts['rate_limited']/n:.1f}%")
    print(f"Timeout: {100*counts['timeout']/n:.1f}%")
    print(f"Errors: {100*counts['error']/n:.1f}%")
    print(f"Peak queue: {meta.get('peak_queue', 'n/a')}")

print("\n========== TABLE 3: ANSWER QUALITY ==========")

for sc, rs in by.items():
    graded = [r for r in rs if r.get("answer_type")]
    if not graded:
        continue

    types = collections.Counter(r["answer_type"] for r in graded)
    generated = [r for r in graded if r["answer_type"] == "generated"]
    fallback = [r for r in graded if r["answer_type"] == "fallback"]
    refusals = [r for r in graded if r["answer_type"] == "refusal"]

    correct = sum(r["answer_grade"] == "correct" for r in generated)
    fallback_hits = sum(r["retrieval_hit"] == "true" for r in fallback)

    print(f"\nScenario: {sc}")
    print(f"Graded requests: {len(graded)}")
    print(f"Generated answers: {len(generated)} ({100*len(generated)/len(graded):.1f}%)")

    if generated:
        print(f"Generated-answer accuracy: {100*correct/len(generated):.1f}%")
    else:
        print("Generated-answer accuracy: n/a")

    print(f"Fallbacks: {len(fallback)} ({100*len(fallback)/len(graded):.1f}%)")

    if fallback:
        print(f"Fallback retrieval coverage: {100*fallback_hits/len(fallback):.1f}%")

    print(f"Refusals: {len(refusals)}")
    print("Grade breakdown:", dict(collections.Counter(
        r["answer_grade"] for r in graded
    )))

print("\nNote: Generated-answer accuracy excludes extraction fallbacks.")
print("Fallback retrieval coverage measures keyword presence in fallback text.")
