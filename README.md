# The Relay

**A retrieval-augmented research question-answering pipeline with citation verification, evidence-based refusal, rate-aware execution and load testing.**

The Relay investigates how reliably an LLM answers questions from technical research papers and how the system behaves under latency, rate limits and capacity pressure.

## 1. Project Overview

The system answers questions using exactly three specified paper versions:

- *Attention Is All You Need* — arXiv:1706.03762, v7
- *Moshi* — arXiv:2410.00037, v1
- *Dynamic Chunking* — arXiv:2507.07955, v2

The pipeline retrieves relevant passages, expands context, evaluates answerability, generates answers and verifies citations. It also supports refusal and lower-cost fallback behaviour.

## 2. Architecture

1. Question understanding.
2. BM25 document retrieval.
3. Neighbouring-passage expansion.
4. Evidence and answerability evaluation.
5. LLM-based answer generation.
6. Citation and source-text verification.
7. Streaming and passage-only fallback.

The pipeline also includes admission control, request deadlines, retry controls and a daily usage budget.

## 3. Technology Stack

- Python
- FastAPI and Uvicorn
- Groq API
- `openai/gpt-oss-20b`
- BM25 retrieval
- Custom reranking
- Pandas
- JSONL and CSV experiment logging

## 4. Retrieval Design

The retrieval implementation uses page-bounded chunks, configurable chunk sizes and overlap, BM25 ranking, neighbouring-passage expansion and an optional hand-tuned reranker.

| Setting | Chunk words | Overlap | Retrieved pieces | Reranking |
|---|---:|---:|---:|---|
| Small | 80 | 20 | 3 | No |
| Medium | 160 | 40 | 4 | Yes |
| Large | 300 | 60 | 6 | No |

The reranker uses lexical and domain-specific signals. It is hand-tuned rather than trained as a semantic ranking model.

## 5. Groq Integration and Calibration

The real-service probes used `openai/gpt-oss-20b`.

Observed response headers included:

- Request limit: 1,000 (RPD)
- Token limit: 8,000 (TPM)
- HTTP 429 responses during a concurrent burst
- `Retry-After: 2` on rate-limit responses

The 45-request concurrent burst produced 31 successful responses and 14 HTTP 429 responses. Ten simultaneous requests also succeeded in a separate test. Maximum concurrency was not established.

Published Developer-plan documentation listed 30 RPM and 200,000 TPD for the model. These values were not independently verified as the limits of the account used for testing.

Calibration used 20 valid sequential latency samples:

- Mean log TTFT: approximately 5.9118
- Log TTFT standard deviation: approximately 0.2575
- Estimated per-token generation time: approximately 0.9692 ms

## 6. Stand-in Service

A local FastAPI stand-in simulates service behaviour under controlled conditions. It supports configurable latency, request and token limits, load-related slowdown, simulated failures and quota exhaustion.

The stand-in was calibrated using real-service latency observations. Its configured limits and behaviour are experimental parameters, not an exact reproduction of Groq infrastructure.

Run it:

```bash
uvicorn standin:app --host 127.0.0.1 --port 8001
```

Inspect statistics:

```bash
curl http://127.0.0.1:8001/admin/stats
```

## 7. Load Testing

The load-test implementation evaluates steady traffic, bursts, sustained overload, reduced request capacity and daily quota exhaustion.

The five principal stand-in scenarios were:

| Scenario | Requests | Offered rate/min | Pipeline success | Timeout |
|---|---:|---:|---:|---:|
| Steady | 60 | 20.55 | 21.7% | 78.3% |
| Burst | 60 | 121.68 | 56.7% | 43.3% |
| Sustained overload | 120 | 122.38 | 40.0% | 60.0% |
| Half RPM | 60 | 20.12 | 26.7% | 73.3% |
| Daily quota exhausted | 18 | 5.05 | 100%* | 0% |

*All requests in the daily-exhausted scenario returned passage-only fallbacks. None produced a generated LLM answer.*

Latency percentiles use successful requests only. Pipeline TTFT is the time until the first verified answer text is emitted, not raw provider TTFT.

## 8. Retrieval Evaluation

The controlled retrieval experiment used the same 18-question benchmark for each configuration.

| Setting | Correct and cited | Wrong | Correct refusals |
|---|---:|---:|---:|
| Small, K=3 | 5.6% | 94.4% | 0/5 |
| Medium, K=4 + reranking | 11.1% | 50.0% | 0/5 |
| Large, K=6 | 0.0% | 38.9% | 0/5 |

Wrong-answer percentages use all 18 questions as the denominator. Timed-out and unevaluated requests are reported separately.

The best configuration achieved 2 correct-and-cited answers out of 18. Increasing retrieval size did not automatically improve answer quality and increased token consumption and timeout rates.

The automated grader checks expected keywords and citation presence. It does not perform complete semantic verification or independently establish that every citation supports its claim.

## 9. Rate Limits, Retries and Fallbacks

The pipeline incorporates:

- Admission control and waiting deadlines.
- A hard request deadline.
- A bounded retry budget.
- Rate-limit cooling behaviour.
- A daily usage budget.
- Lower-cost execution modes.
- Passage-only fallback when normal generation cannot proceed.

The main five load scenarios recorded zero retries per request. An earlier stress record captured an empty-response recovery path, but the experiments do not establish a reliable retry success rate.

## 10. Repository Structure

```text
the-relay/
├── data/
│   ├── attention_v7.pdf
│   ├── dchunk_v2.pdf
│   ├── moshi_v1.pdf
│   └── pages.json
├── .env.example
├── .gitignore
├── README.md
├── analyze.py
├── calibrate.py
├── fetch_papers.py
├── grade.py
├── llm.py
├── loadtest.py
├── pipeline.py
├── probe_groq.py
├── questions.json
├── retrieval.py
├── standin.py
├── standin_config.json
└── results.csv
```

## 11. Setup and Execution

Clone the repository:

```bash
git clone https://github.com/Vatsalsamarth/the-relay.git
cd the-relay
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies required by the project.

Create a local `.env` file using `.env.example`:

```env
GROQ_API_KEY=your_groq_api_key
LLM_MODEL=openai/gpt-oss-20b
```

Never commit the actual `.env` file or API credentials.

Run the stand-in:

```bash
uvicorn standin:app --host 127.0.0.1 --port 8001
```

A benchmark can be run using the load-test script and its supported arguments:

```bash
python loadtest.py --scenario pipeline_v9_final --kind steady --source real --n 18 --rate 2 --questions questions.json
```

## 12. Experiment Data

`results.csv` contains raw request-level experiment records with these columns:

```text
request_id,scenario,source,start_time,first_token_ms,total_ms,status,retries,tokens_in,tokens_out,retrieval_setting,answer_grade
```

The file contains 594 experiment records after removing six smoke-test records. It includes the principal load scenarios, controlled retrieval comparisons and earlier exploratory experiments.

The raw records allow aggregate metrics to be recalculated independently.

## 13. Limitations and Future Work

- The benchmark contains only 18 questions.
- BM25 and hand-tuned reranking do not guarantee semantic relevance.
- Automated grading is based on keyword matching and citation presence.
- Citation validation does not establish complete semantic support.
- Account-specific Groq RPM and TPD limits were not independently verified.
- Maximum provider concurrency and exact recovery time were not measured.
- The stand-in cannot perfectly reproduce real provider infrastructure.
- Cost per 1,000 generated answers was not measured.

Future improvements include semantic retrieval, stronger citation-to-claim verification, a more representative benchmark, better refusal evaluation and additional controlled experiments.

## 14. Conclusion

The Relay demonstrates a multi-stage research QA pipeline combined with retrieval evaluation, real-service probing, a calibrated stand-in and load testing.

The experiments show why answer quality, retrieval size, latency, capacity management and fallback behaviour must be evaluated together. A successful pipeline response is not necessarily a correct generated answer, and more retrieved context does not necessarily produce better results.
