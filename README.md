
# The Relay

A retrieval-augmented question-answering pipeline that answers questions using technical research papers, with citation verification and evidence-based refusal.

## Overview

The pipeline consists of:

1. Question understanding
2. Document retrieval using BM25
3. Custom retrieval reranking
4. Context expansion
5. LLM-based answer generation
6. Citation verification
7. Refusal handling for unsupported questions

## Technology Stack

- Python
- FastAPI
- Uvicorn
- Groq API
- openai/gpt-oss-20b
- BM25 retrieval
- Custom reranking

## Project Structure

```text
the-relay/
├── data/
├── .env
├── .env.example
├── .gitignore
├── analyze.py
├── calibrate.py
├── fetch_papers.py
├── grade.py
├── llm.py
├── loadtest.py
├── pipeline.py
├── probe_groq.py
├── questions.json
├── README.md
├── retrieval.py
├── standin.py
└── standin_config.json
```

## Configuration

Create a `.env` file using `.env.example` as a template.

```env
GROQ_API_KEY=your_groq_api_key
LLM_MODEL=openai/gpt-oss-20b
```

Never commit your actual API key.

## Setup

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install dependencies using your project's dependency file, if provided.

## Running the Stand-in Server

```bash
uvicorn standin:app --host 127.0.0.1 --port 8001
```

Check server statistics:

```bash
curl http://127.0.0.1:8001/admin/stats
```

## Running the Benchmark

```bash
python loadtest.py --scenario pipeline_v9_final --kind steady --source real --n 18 --rate 2 --questions questions.json
```

## Calibration

Recorded calibration results:

- Valid measurements: 20
- Mean log TTFT: 5.9118
- TTFT sigma: 0.2575
- Per-token generation time: approximately 0.9692 ms

Observed provider limits during probing:

- RPM: 30
- TPM: 8,000

## Final Benchmark Results

| Metric | Result |
|---|---:|
| Total questions | 18 |
| Answerable questions answered | 13/13 |
| Unanswerable questions refused | 5/5 |
| Total benchmark duration | 525.9 seconds |
| Peak queue depth | 1 |

These results represent the tested 18-question dataset, not a universal accuracy guarantee.

## Citation Verification

The pipeline checks that:

- Citation IDs exist in retrieved passages.
- Quoted text matches normalized source text.
- Unverified trailing text is discarded when verified answer content has already been emitted.

These checks do not independently establish semantic support for every paraphrased claim.

## Limitations

- Results depend on document coverage and retrieval quality.
- LLM output may vary between runs.
- External API latency and rate limits affect performance.
- The benchmark contains a limited set of 18 questions.
- Citation validation is not full semantic verification.

## Conclusion

The Relay combines document retrieval, custom reranking, LLM-based answer generation and citation validation.

In the final real-model benchmark, it answered all 13 answerable questions and refused all five unanswerable questions.
