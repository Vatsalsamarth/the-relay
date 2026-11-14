
"""Thin async client for Groq and OpenAI-compatible endpoints."""

import os
import time
import json
import asyncio
import httpx
from pathlib import Path


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

env_file = Path(__file__).resolve().parent / ".env"

if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()

        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)

            os.environ.setdefault(
                name.strip(),
                value.strip().strip('"').strip("'")
            )


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

BASE = os.environ.get(
    "LLM_BASE",
    "https://api.groq.com/openai/v1"
)

KEY = os.environ.get("GROQ_API_KEY", "")

MODEL = os.environ.get(
    "LLM_MODEL",
    "openai/gpt-oss-20b"
)

_client = None


# --------------------------------------------------
# 3. HTTP CLIENT
# --------------------------------------------------

def client():
    global _client

    if _client is None:
        if not KEY:
            raise LLMError(
                "configuration_error",
                body="GROQ_API_KEY is missing from .env"
            )

        _client = httpx.AsyncClient(
            base_url=BASE,
            headers={
                "Authorization": f"Bearer {KEY}"
            },
            timeout=None,
            limits=httpx.Limits(
                max_connections=300
            )
        )

    return _client


# --------------------------------------------------
# 4. CUSTOM ERRORS
# --------------------------------------------------

class LLMError(Exception):

    def __init__(
        self,
        kind,
        status=None,
        retry_after=None,
        headers=None,
        body=""
    ):
        super().__init__(
            f"{kind} {status} {body[:120]}"
        )

        self.kind = kind
        self.status = status
        self.retry_after = retry_after
        self.headers = headers or {}
        self.body = body


class Abort(Exception):
    """Raised by an on_token callback to stop a stream early."""


# --------------------------------------------------
# 5. TOKEN ESTIMATION
# --------------------------------------------------

def est_tokens(text):
    return max(1, len(text) // 4)


# --------------------------------------------------
# 6. CHAT COMPLETION
# --------------------------------------------------

async def chat(
    messages,
    max_tokens=300,
    stream=False,
    timeout=30.0,
    on_token=None,
    temperature=0.1
):

    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,

        # Groq GPT-OSS settings
        "reasoning_effort": "low",
        "include_reasoning": False
    }

    t0 = time.perf_counter()

    text = []
    ttft = None
    usage = None
    hdr = {}
    finish_reason = None

    try:

        async with asyncio.timeout(timeout):

            async with client().stream(
                "POST",
                "/chat/completions",
                json=body
            ) as r:

                hdr = {
                    k.lower(): v
                    for k, v in r.headers.items()
                }

                # ----------------------------------
                # HANDLE HTTP ERRORS
                # ----------------------------------

                if r.status_code != 200:

                    error_body = (
                        await r.aread()
                    ).decode(errors="ignore")

                    retry_after = hdr.get("retry-after")

                    try:
                        retry_after = (
                            float(retry_after)
                            if retry_after
                            else None
                        )
                    except ValueError:
                        retry_after = None

                    raise LLMError(
                        "rate_limited"
                        if r.status_code == 429
                        else "error",
                        r.status_code,
                        retry_after,
                        hdr,
                        error_body
                    )

                # ----------------------------------
                # STREAMING RESPONSE
                # ----------------------------------

                if stream:

                    async for line in r.aiter_lines():

                        if not line.startswith("data:"):
                            continue

                        data = line[5:].strip()

                        if data == "[DONE]":
                            break

                        if not data:
                            continue

                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        # Capture token usage
                        chunk_usage = (
                            chunk.get("usage")
                            or (chunk.get("x_groq") or {}).get("usage")
                        )

                        if chunk_usage:
                            usage = chunk_usage

                        choices = chunk.get("choices", [])

                        for choice in choices:

                            delta = choice.get("delta") or {}

                            # Record finish reason
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]

                            # Extract visible response text
                            token = delta.get("content")

                            if isinstance(token, str) and token:

                                # Measure time to first visible token
                                if ttft is None:
                                    ttft = (
                                        time.perf_counter() - t0
                                    ) * 1000

                                text.append(token)

                                # Optional streaming callback
                                if on_token:

                                    result = on_token(token)

                                    if asyncio.iscoroutine(result):
                                        await result

                    # Do not report an empty response as success
                    if not "".join(text).strip():

                        raise LLMError(
                            "empty_response",
                            headers=hdr,
                            body=(
                                "No visible text received. "
                                f"Finish reason: {finish_reason}. "
                                f"Usage: {usage}"
                            )
                        )

                # ----------------------------------
                # NON-STREAMING RESPONSE
                # ----------------------------------

                else:

                    response_data = json.loads(
                        await r.aread()
                    )

                    usage = response_data.get("usage")

                    choices = response_data.get("choices", [])

                    if not choices:
                        raise LLMError(
                            "empty_response",
                            headers=hdr,
                            body="No choices returned by the API"
                        )

                    message = choices[0].get("message") or {}

                    content = message.get("content")

                    if not isinstance(content, str) or not content.strip():

                        raise LLMError(
                            "empty_response",
                            headers=hdr,
                            body="No visible message content returned"
                        )

                    text.append(content)

                    ttft = (
                        time.perf_counter() - t0
                    ) * 1000

    # ------------------------------------------
    # ERROR HANDLING
    # ------------------------------------------

    except TimeoutError:

        raise LLMError("timeout")

    except httpx.HTTPError as error:

        raise LLMError(
            "error",
            body=str(error)
        )

    # ------------------------------------------
    # 7. FINAL RESPONSE
    # ------------------------------------------

    output = "".join(text)

    tokens_in = (
        (usage or {}).get("prompt_tokens")
        or sum(
            est_tokens(message["content"])
            for message in messages
        )
    )

    tokens_out = (
        (usage or {}).get("completion_tokens")
        or est_tokens(output)
    )

    return {
        "text": output,
        "ttft_ms": ttft,
        "total_ms": (
            time.perf_counter() - t0
        ) * 1000,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "headers": hdr
    }
