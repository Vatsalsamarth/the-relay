import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("dev_runs/final_submission")
CSV_PATH = BASE / "final_results.csv"
JSONL_PATH = BASE / "final_answers.jsonl"
OUT_PATH = BASE / "tables.md"

with CSV_PATH.open(newline="") as f:
    rows = list(csv.DictReader(f))

answers = {}
with JSONL_PATH.open() as f:
    for line in f:
        r = json.loads(line)
        answers[str(r["request_id"])] = r

def percentile(values, p):
    values = sorted(float(x) for x in values if x not in ("", None))
    if not values:
        return "n/a"
    pos = (len(values) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    value = values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return f"{value:.1f}"

def fmt(values):
    return " / ".join(percentile(values, p) for p in (0.50, 0.95, 0.99))

def group_rows(key):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    return groups

lines = [
    "# Final benchmark results",
    "",
    "Generated from `final_results.csv` and `final_answers.jsonl`.",
    "Latency values are in milliseconds. Percentiles use linear interpolation.",
    "",
    "## Table 1. Latency by scenario and source",
    "",
    "| Scenario | Source | Requests | First-token p50 / p95 / p99 | Total p50 / p95 / p99 |",
    "|---|---|---:|---:|---:|",
]

for (scenario, source), group in group_rows(
    lambda r: (r["scenario"], r["source"])
).items():
    lines.append(
        f"| {scenario} | {source} | {len(group)} "
        f"| {fmt([r['first_token_ms'] for r in group])} "
        f"| {fmt([r['total_ms'] for r in group])} |"
    )

load_scenarios = [
    "steady_v2", "burst_v2", "overload_v2",
    "half_rpm_v2", "daily_exhausted_v2"
]

lines += [
    "",
    "## Table 2. Load-test outcomes and execution modes",
    "",
    "`rate_limited` is the pipeline CSV status; it is not equivalent to an HTTP 429.",
    "Provider 429s below are counted separately from event logs.",
    "",
    "| Scenario | Requests | OK | rate_limited status | Timeout | Error | Full | Lean | Extract | Predicted-wait shed events | Provider 429 events |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for scenario in load_scenarios:
    group = [r for r in rows if r["scenario"] == scenario]
    status = Counter(r["status"] for r in group)
    modes = Counter()
    events = Counter()
    matched = 0

    for row in group:
        answer = answers.get(str(row["request_id"]))
        if answer is None:
            continue
        matched += 1
        modes[answer.get("mode", "missing")] += 1
        for event in answer.get("events", []):
            if event:
                events[str(event[0])] += 1

    provider_429 = sum(
        count for name, count in events.items()
        if name.startswith("llm_rate_limited:429")
    )

    lines.append(
        f"| {scenario} | {len(group)} "
        f"| {status['ok']} | {status['rate_limited']} "
        f"| {status['timeout']} | {status['error']} "
        f"| {modes['full']} | {modes['lean']} | {modes['extract']} "
        f"| {events['shed_predicted_wait']} | {provider_429} |"
    )

    if matched != len(group):
        lines.append(
            f"\n> **Data check:** {scenario}: matched {matched}/{len(group)} "
            "CSV requests to answer-log records."
        )

lines += [
    "",
    "## Table 3. Retrieval answer grades",
    "",
    "| Configuration | Source | Correct | Wrong | Correct refusals | Not evaluated | Fallback |",
    "|---|---|---:|---:|---:|---:|---:|",
]

retrieval = [
    r for r in rows if r["scenario"].startswith("retrieval_")
]

for (scenario, source), group in group_rows(
    lambda r: (r["scenario"], r["source"])
).items():
    if not scenario.startswith("retrieval_"):
        continue
    grades = Counter(r["answer_grade"] for r in group)
    lines.append(
        f"| {scenario} | {source} | {grades['correct']} "
        f"| {grades['wrong']} | {grades['refused_correctly']} "
        f"| {grades['not_evaluated']} | {grades['fallback']} |"
    )

lines += [
    "",
    "## Interpretation notes",
    "",
    "- Real-provider retrieval results and stand-in results are reported separately.",
    "- Latency percentiles include all recorded attempts, not only successful generated answers.",
    "- Stand-in load tests are simulations, not measurements of Groq capacity.",
    "- `rate_limited` is a pipeline status and may include load shedding or fallback.",
    "- Explicit provider HTTP 429 events are counted from event logs separately.",
    "- `extract` mode returns retrieved passages rather than a fully generated answer.",
    "- `not_evaluated` grades are not counted as correct or incorrect.",
    "- No API cost estimate is included because pricing assumptions have not been verified.",
    "",
]

OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Generated: {OUT_PATH}")
print(f"CSV records: {len(rows)}")
print(f"Answer-log records: {len(answers)}")
print(f"Retrieval records: {len(retrieval)}")
