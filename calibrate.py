
"""Calibrate the stand-in simulator using valid probe measurements."""

import json
import math
import statistics as st
from pathlib import Path

# Read existing probe data without deleting anything.
probe_file = Path("probe.jsonl")

if not probe_file.exists():
    raise SystemExit("ERROR: probe.jsonl was not found.")

with probe_file.open() as f:
    recs = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

# Keep only successful records with valid TTFT measurements.
valid = [
    r for r in recs
    if r.get("ok") is True
    and isinstance(r.get("ttft"), (int, float))
    and r["ttft"] > 0
    and isinstance(r.get("total"), (int, float))
    and isinstance(r.get("tout"), (int, float))
]

# Prefer sequential measurements for latency calibration.
seq = [
    r for r in valid
    if r.get("tag") == "seq"
]

ok = seq if seq else valid

if len(ok) < 2:
    raise SystemExit(
        f"ERROR: Only {len(ok)} valid TTFT measurements found.\n"
        "Run: python probe_groq.py seq 20\n"
        "Then run calibration again."
    )

# Fit a lognormal distribution to TTFT.
log_ttft = [
    math.log(r["ttft"])
    for r in ok
]

# Estimate per-token generation time.
decode = [
    (r["total"] - r["ttft"]) / r["tout"]
    for r in ok
    if r["tout"] > 5 and r["total"] >= r["ttft"]
]

if not decode:
    raise SystemExit(
        "ERROR: Not enough valid measurements to estimate decode speed."
    )

cfg = {
    "ttft_mu": st.mean(log_ttft),
    "ttft_sigma": st.pstdev(log_ttft),
    "per_tok_ms": st.median(decode)
}

# Copy observed rate limits from successful responses.
# Groq headers: requests = requests per day (RPD);
# tokens = tokens per minute (TPM).
# RPM and TPD are not inferred from these two headers.
lim = next(
    (r["hdr"] for r in ok if r.get("hdr")),
    {}
)

for header, name in (
    ("x-ratelimit-limit-requests", "rpd"),
    ("x-ratelimit-limit-tokens", "tpm")
):
    if header in lim:
        cfg[name] = int(float(lim[header]))

# Save simulator configuration.
with open("standin_config.json", "w") as f:
    json.dump(cfg, f, indent=2)

print("\nCALIBRATION SUCCESSFUL")
print("----------------------")
print("Valid measurements:", len(ok))
print("TTFT mean (log):", round(cfg["ttft_mu"], 4))
print("TTFT sigma:", round(cfg["ttft_sigma"], 4))
print("Per-token time (ms):", round(cfg["per_tok_ms"], 4))
print("\nGenerated standin_config.json")
print("\nTODO: Set rpm, max_conc, load_k, slow_rate and fail_rate manually.")
