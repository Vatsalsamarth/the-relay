# Final benchmark results

Generated from `final_results.csv` and `final_answers.jsonl`.
Latency values are in milliseconds. Percentiles use linear interpolation.

## Table 1. Latency by scenario and source

| Scenario | Source | Requests | First-token p50 / p95 / p99 | Total p50 / p95 / p99 |
|---|---|---:|---:|---:|
| retrieval_s1_small_k3 | standin | 18 | 729.2 / 2107.3 / 6822.6 | 791.5 / 2170.2 / 6885.2 |
| retrieval_s2_med_k4_rr | standin | 18 | 4833.1 / 11820.1 / 11823.5 | 4896.0 / 11820.1 / 11823.5 |
| retrieval_s3_large_k6 | standin | 18 | 11811.3 / 11818.8 / 11819.9 | 11811.4 / 11818.9 / 11820.0 |
| retrieval_s1_small_k3 | real | 18 | 964.9 / 1189.9 / 1459.9 | 965.4 / 1204.1 / 1462.7 |
| retrieval_s2_med_k4_rr | real | 18 | 1014.1 / 26526.1 / 27533.2 | 1015.4 / 26526.8 / 27536.9 |
| retrieval_s3_large_k6 | real | 18 | 21391.7 / 71854.1 / 71888.6 | 21392.8 / 71854.2 / 71888.6 |
| steady_v2 | standin | 60 | 13.2 / 984.2 / 1436.3 | 13.2 / 1037.9 / 1498.5 |
| burst_v2 | standin | 60 | 6.5 / 434.6 / 7640.0 | 6.5 / 434.6 / 7676.7 |
| overload_v2 | standin | 120 | 11.9 / 41.6 / 1691.2 | 11.9 / 41.6 / 1753.3 |
| half_rpm_v2 | standin | 60 | 13.2 / 1948.2 / 7653.9 | 13.2 / 2019.3 / 7956.6 |
| daily_exhausted_v2 | standin | 18 | 12.4 / 17.6 / 19.3 | 12.4 / 17.6 / 19.3 |

## Table 2. Load-test outcomes and execution modes

`rate_limited` is the pipeline CSV status; it is not equivalent to an HTTP 429.
Provider 429s below are counted separately from event logs.

| Scenario | Requests | OK | rate_limited status | Timeout | Error | Full | Lean | Extract | Predicted-wait shed events | Provider 429 events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| steady_v2 | 60 | 12 | 48 | 0 | 0 | 9 | 3 | 48 | 48 | 0 |
| burst_v2 | 60 | 1 | 59 | 0 | 0 | 0 | 1 | 59 | 58 | 3 |
| overload_v2 | 120 | 4 | 115 | 1 | 0 | 2 | 2 | 116 | 115 | 1 |
| half_rpm_v2 | 60 | 11 | 49 | 0 | 0 | 4 | 7 | 49 | 48 | 1 |
| daily_exhausted_v2 | 18 | 0 | 18 | 0 | 0 | 0 | 0 | 18 | 0 | 0 |

## Table 3. Retrieval answer grades

| Configuration | Source | Correct | Wrong | Correct refusals | Not evaluated | Fallback |
|---|---|---:|---:|---:|---:|---:|
| retrieval_s1_small_k3 | standin | 8 | 5 | 5 | 0 | 0 |
| retrieval_s2_med_k4_rr | standin | 5 | 3 | 3 | 7 | 0 |
| retrieval_s3_large_k6 | standin | 2 | 1 | 0 | 12 | 3 |
| retrieval_s1_small_k3 | real | 8 | 5 | 5 | 0 | 0 |
| retrieval_s2_med_k4_rr | real | 9 | 4 | 5 | 0 | 0 |
| retrieval_s3_large_k6 | real | 4 | 2 | 4 | 8 | 0 |

## Interpretation notes

- Real-provider retrieval results and stand-in results are reported separately.
- Latency percentiles include all recorded attempts, not only successful generated answers.
- Stand-in load tests are simulations, not measurements of Groq capacity.
- `rate_limited` is a pipeline status and may include load shedding or fallback.
- Explicit provider HTTP 429 events are counted from event logs separately.
- `extract` mode returns retrieved passages rather than a fully generated answer.
- `not_evaluated` grades are not counted as correct or incorrect.
- No API cost estimate is included because pricing assumptions have not been verified.
