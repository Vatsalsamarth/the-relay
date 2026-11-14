"""The Relay pipeline:
understand(LLM) -> retrieve(local) -> selective_context_expansion
-> draft(LLM, streamed, sentence-gated) -> check(loop).

Governor = admission/limits/daily budget.
Modes: full -> lean (skip understand call) -> extract (no model).
"""

import asyncio
import json
import random
import re
import time

from collections import deque
from dataclasses import dataclass

import llm
import retrieval

from llm import LLMError, Abort, est_tokens


@dataclass
class Cfg:
    max_inflight: int = 6
    rpm: int = 30
    tpm: int = 8000
    rpd: int = 14400

    safety: float = 0.85
    patience_ms: float = 20000

    max_attempts: int = 3
    retry_budget: float = 0.2

    lean_pressure: float = 0.6
    extract_pressure: float = 0.95
    daily_reserve: int = 100

    refuse_score: float = 2.0
    setting: str = "s2_med_k4_rr"


class Giveup(Exception):
    pass


class Governor:

    def __init__(self, cfg, day_used=0):
        self.cfg = cfg
        self.sem = asyncio.Semaphore(cfg.max_inflight)

        self.reqs = deque()
        self.toks = deque()

        self.queue = 0
        self.peak_queue = 0

        self.n_req = 0
        self.n_retry = 0

        self.cool_until = 0.0
        self.day_used = day_used

    def _prune(self, now):
        for dq in (self.reqs, self.toks):
            while dq and now - dq[0][0] > 60:
                dq.popleft()

    def pressure(self):
        now = time.time()
        self._prune(now)

        c = self.cfg

        return max(
            len(self.reqs) / max(1, c.rpm * c.safety),
            sum(t for _, t in self.toks)
            / max(1, c.tpm * c.safety),
            (
                c.max_inflight - self.sem._value + self.queue
            ) / max(1, c.max_inflight) / 2,
            1.0 if now < self.cool_until else 0
        )

    def day_left(self):
        return self.cfg.rpd - self.day_used

    def mode(self):

        if self.day_left() <= self.cfg.daily_reserve:
            return "extract"

        p = self.pressure()

        return (
            "extract" if p >= self.cfg.extract_pressure * 2
            else "lean" if p >= self.cfg.lean_pressure
            else "full"
        )

    def cool(self, secs):
        self.cool_until = max(
            self.cool_until,
            time.time() + secs
        )

    def settle(self, reservation, actual_tokens):
        """Replace estimated TPM usage with actual usage."""
        if reservation is not None:
            reservation[1] = max(1, int(actual_tokens))

    async def acquire(self, tokens, deadline):

        c = self.cfg
        t0 = time.time()

        self.queue += 1
        self.peak_queue = max(self.peak_queue, self.queue)

        acquired = False
        reserved = False

        try:

            # Obtain concurrency capacity first.
            remaining = deadline - time.time()

            if remaining <= 0:
                raise Giveup("no slot before deadline")

            try:
                await asyncio.wait_for(
                    self.sem.acquire(),
                    remaining
                )
                acquired = True

            except asyncio.TimeoutError:
                raise Giveup("no slot before deadline")

            # Wait for rolling RPM/TPM capacity.
            while True:

                now = time.time()
                self._prune(now)

                if (
                    now >= self.cool_until
                    and len(self.reqs) < c.rpm * c.safety
                    and sum(t for _, t in self.toks)
                    + tokens <= c.tpm * c.safety
                ):
                    break

                if now + 0.25 > deadline:
                    raise Giveup("no capacity before deadline")

                await asyncio.sleep(0.25)

            # Reserve capacity only after all admission checks pass.
            self.reqs.append((time.time(), 1))

            # Mutable entry lets settle() update the token estimate.
            token_entry = [time.time(), max(1, int(tokens))]
            self.toks.append(token_entry)

            self.day_used += 1
            reserved = True

            return (time.time() - t0) * 1000, token_entry

        finally:

            self.queue -= 1

            # Never leak a semaphore slot if admission fails.
            if acquired and not reserved:
                self.sem.release()


class Ctx:

    def __init__(self, rid):

        self.rid = rid
        self.t0 = time.perf_counter()

        self.wait = 0.0
        self.work = 0.0
        self.retry = 0.0

        self.retries = 0

        self.tin = 0
        self.tout = 0

        self.ft = None
        self.events = []

    def now(self):
        return (time.perf_counter() - self.t0) * 1000

    def ev(self, name):
        self.events.append(
            (name, round(self.now(), 1))
        )


async def call_llm(
    g,
    ctx,
    messages,
    deadline,
    acquire_deadline=None,
    **kw
):
    """One logical model call with retries."""

    c = g.cfg

    for attempt in range(c.max_attempts):

        est = (
            sum(
                est_tokens(m["content"])
                for m in messages
            )
            + kw.get("max_tokens", 300)
        )

        wait_started = time.perf_counter()

        try:
            wait_ms, reservation = await g.acquire(
                est,
                min(
                    deadline,
                    acquire_deadline or deadline
                )
            )
        except Giveup:
            ctx.wait += (
                time.perf_counter() - wait_started
            ) * 1000
            raise

        ctx.wait += wait_ms

        t = time.perf_counter()
        error = None
        result = None

        try:

            try:
                result = await llm.chat(
                    messages,
                    timeout=max(
                        1.0,
                        deadline - time.time()
                    ),
                    **kw
                )

            except LLMError as e:
                error = e

            if error is None:

                actual = (
                    result.get("tokens_in", 0)
                    + result.get("tokens_out", 0)
                )

                g.settle(reservation, actual)

                ctx.work += (
                    time.perf_counter() - t
                ) * 1000

                ctx.tin += result.get("tokens_in", 0)
                ctx.tout += result.get("tokens_out", 0)

        finally:
            # Release the concurrency slot on success,
            # LLM errors, streaming aborts and other exceptions.
            g.sem.release()

        if error is None:
            return result

        e = error

        ctx.retry += (
            time.perf_counter() - t
        ) * 1000

        ctx.ev(f"llm_{e.kind}")

        if e.kind == "rate_limited":
            g.cool(
                min(e.retry_after or 2, 30)
            )

        ok = (
            e.kind in ("rate_limited", "error")
            and attempt + 1 < c.max_attempts
            and g.n_retry <= c.retry_budget * g.n_req + 5
        )

        delay = max(
            e.retry_after or 0,
            random.uniform(
                0,
                min(8, 0.5 * 2 ** attempt)
            )
        )

        if (
            e.kind == "timeout"
            or not ok
            or time.time() + delay > deadline
        ):
            raise e

        g.n_retry += 1
        ctx.retries += 1

        ctx.retry += delay * 1000

        await asyncio.sleep(delay)


UNDERSTAND = (
    'Rewrite the question as a short search query for a library of 3 research papers '
    '(Transformer, Moshi speech model, Dynamic Chunking/H-Net). '
    'Preserve important technical terms and named entities. '
    'Reply JSON only: {"query": str}. '
    'Never refuse a question or decide whether it is in scope. '
    'Always provide the best search query possible.'
)


DRAFT = (
    "Answer ONLY from the numbered passages. Cite every sentence with [S#]. "
    "Quote sparingly (short exact quotes in double quotes). "
    "Be concise (max 4 sentences). "
    "If the passages do not answer the question, reply exactly NOT_IN_PAPERS."
)


def sentence_ok(s, by_id):
    ids = re.findall(r"\[S(\d+)\]", s)

    if not ids or any(int(i) not in by_id for i in ids):
        return False

    for q in re.findall(r'"([^"]{12,})"', s):
        if retrieval.norm(q) not in " ".join(
            retrieval.norm(by_id[int(i)]["text"])
            for i in ids
        ):
            return False

    return True

def extract_answer(hits, why):

    lines = [
        f"[{why}] Showing the most relevant passages instead of a written answer:"
    ]

    for _, c in hits[:2]:

        lines.append(
            f"- ({c['title']}, p.{c['page']}) "
            f"{' '.join(c['text'].split()[:60])}..."
        )

    return "\n".join(lines)


def expand_context(index, hits):
    """
    Preserve the reranked order from Index.search().
    Insert the first chunk of selected pages next to their
    retrieved chunks so table headers remain close to evidence.
    """

    if not hits:
        return hits

    # Select the three highest-ranked distinct pages.
    selected_pages = []
    seen_pages = set()

    for _, chunk in hits:

        key = (
            chunk["paper"],
            chunk["page"]
        )

        if key not in seen_pages:
            selected_pages.append(key)
            seen_pages.add(key)

        if len(selected_pages) == 2:
            break

    selected_page_set = set(selected_pages)

    # Find the first chunk on every page.
    first_by_page = {}

    for chunk in index.chunks:

        key = (
            chunk["paper"],
            chunk["page"]
        )

        if key not in first_by_page:
            first_by_page[key] = chunk

    expanded = []
    seen_ids = set()
    inserted_pages = set()

    for score, chunk in hits:

        key = (
            chunk["paper"],
            chunk["page"]
        )

        # Insert the page's first chunk immediately before
        # its first retrieved chunk.
        if (
            key in selected_page_set
            and key not in inserted_pages
        ):

            first_chunk = first_by_page.get(key)

            if (
                first_chunk is not None
                and first_chunk["id"] not in seen_ids
            ):

                expanded.append(
                    (score, first_chunk)
                )

                seen_ids.add(first_chunk["id"])

            inserted_pages.add(key)

        if chunk["id"] not in seen_ids:

            expanded.append(
                (score, chunk)
            )

            seen_ids.add(chunk["id"])

    return expanded


async def handle(rid, question, g, index, cfg):

    ctx = Ctx(rid)

    c = cfg
    start = time.time()

    g.n_req += 1

    hard = (
        start + c.patience_ms * 2.5 / 1000
    )

    res = dict(
        status="ok",
        mode=g.mode(),
        answer="",
        refused=False
    )

    mode = res["mode"]
    emitted = []

    def emit(txt):

        if ctx.ft is None:
            ctx.ft = ctx.now()

        emitted.append(txt)

    hits = []

    try:

        query = question

        # --------------------------------------------------
        # STEP 1: UNDERSTAND THE QUESTION
        # --------------------------------------------------

        if mode == "full":

            ctx.ev("understand")

            try:

                r = await call_llm(
                    g,
                    ctx,
                    [
                        {
                            "role": "system",
                            "content": UNDERSTAND
                        },
                        {
                            "role": "user",
                            "content": "QUESTION: " + question
                        }
                    ],
                    start + c.patience_ms * 0.4 / 1000,
                    max_tokens=60
                )

                match = re.search(
                    r"\{.*\}",
                    r["text"],
                    re.S
                )

                if not match:
                    raise ValueError("No JSON returned")

                j = json.loads(match.group(0))

                query = j.get("query") or question

            except (
                LLMError,
                Giveup,
                AttributeError,
                ValueError,
                TypeError
            ):

                ctx.ev("understand_skipped")

                query = question

        # --------------------------------------------------
        # STEP 2: INITIAL RETRIEVAL
        # --------------------------------------------------

        ctx.ev("retrieve")

        # Keep the original wording as well as the rewritten query.
        # This prevents useful terms from disappearing during rewriting.
        search_query = (
            question
            if query.strip().lower() == question.strip().lower()
            else question + " " + query
        )

        hits = index.search(search_query)

        # --------------------------------------------------
        # STEP 3: SELECTIVE CONTEXT EXPANSION
        # --------------------------------------------------

        if hits:

            ctx.ev("tool_expand")

            hits = expand_context(
                index,
                hits
            )

        # --------------------------------------------------
        # STEP 4: REFUSAL CHECK
        # --------------------------------------------------

        if (
            not hits
            or hits[0][0] < c.refuse_score
        ):

            emit(
                "The papers don't say "
                "(no passage matched closely enough)."
            )

            res["refused"] = True

            return finish(
                ctx,
                res,
                emitted
            )

        by_id = {
            h[1]["id"]: h[1]
            for h in hits
        }

        # --------------------------------------------------
        # STEP 5: EXTRACT MODE UNDER HIGH LOAD
        # --------------------------------------------------

        if (
            mode == "extract"
            or time.time() > start + c.patience_ms * 0.5 / 1000
        ):

            res["mode"] = "extract"

            emit(
                extract_answer(
                    hits,
                    "high load"
                )
            )

            return finish(
                ctx,
                res,
                emitted
            )

        # --------------------------------------------------
        # STEP 6: BUILD LLM CONTEXT
        # --------------------------------------------------

        ctx_txt = "\n".join(
            f"[S{h[1]['id']}] "
            f"({h[1]['title']}, p.{h[1]['page']})\n"
            f"{h[1]['text']}"
            for h in hits
        )

        # --------------------------------------------------
        # STEP 7: DRAFT AND VERIFY
        # --------------------------------------------------

        for attempt in range(2):

            pend = {"s": ""}
            bad = {"v": False}

            async def on_token(tok):

                pend["s"] += tok

                if (
                    not emitted
                    and pend["s"].lstrip().startswith("NOT_IN")
                ):
                    return

                m = re.search(
                    r"(.+?\[S\d+\][.!?]?)(\s|$)",
                    pend["s"],
                    re.S
                )

                while (
                    m
                    and m.group(0).strip().endswith(
                        ("]", ".", "!", "?")
                    )
                    and len(pend["s"].split()) > 3
                ):

                    sent = m.group(1)

                    if sentence_ok(sent, by_id):

                        emit(sent + " ")

                        pend["s"] = pend["s"][m.end():]

                    else:

                        bad["v"] = True

                        raise Abort()

                    m = re.search(
                        r"(.+?\[S\d+\][.!?]?)(\s|$)",
                        pend["s"],
                        re.S
                    )

            ctx.ev("draft")

            try:

                r = await call_llm(
                    g,
                    ctx,
                    [
                        {
                            "role": "system",
                            "content": DRAFT + (
                                " Every sentence MUST end with "
                                "a valid [S#] citation."
                                if attempt else ""
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"{ctx_txt}\n\n"
                                f"QUESTION: {question}"
                            )
                        }
                    ],
                    hard,
                    acquire_deadline=(
                        start + c.patience_ms * 0.6 / 1000
                    ),
                    max_tokens=140,
                    stream=True,
                    on_token=on_token
                )

            except Abort:
                r = None

            if (
                r
                and r["text"].strip().startswith("NOT_IN_PAPERS")
            ):

                emit("The papers don't say.")

                res["refused"] = True

                break

            if r and pend["s"].strip():

                if sentence_ok(
                    pend["s"].strip(),
                    by_id
                ):

                    emit(
                        pend["s"].strip()
                    )

                else:

                    if not emitted:
                        bad["v"] = True

            if not bad["v"]:
                break

            ctx.ev("check_failed")

            if emitted:

                emit(
                    "\n[Rest of answer withheld: "
                    "could not verify it against the papers.]"
                )

                break

        else:

            emit(
                extract_answer(
                    hits,
                    "could not verify a written answer"
                )
            )

    except (LLMError, Giveup) as e:

        kind = (
            "admission_timeout"
            if isinstance(e, Giveup)
            else getattr(e, "kind", "error")
        )

        ctx.ev("fallback_" + kind)

        res["status"] = (
            "rate_limited" if kind == "rate_limited"
            else "admission_timeout" if kind == "admission_timeout"
            else "timeout" if kind == "timeout"
            else "error"
        )

        if not emitted:

            res["mode"] = "extract"

            emit(
                extract_answer(
                    hits,
                    "model unavailable"
                )
            )

    return finish(
        ctx,
        res,
        emitted
    )


def finish(ctx, res, emitted):

    total = ctx.now()

    if (
        ctx.ft is None
        and res["status"] == "ok"
    ):
        res["status"] = "timeout"

    res.update(
        answer="".join(emitted).strip(),
        first_token_ms=ctx.ft,
        total_ms=total,
        retries=ctx.retries,
        tokens_in=ctx.tin,
        tokens_out=ctx.tout,
        wait_ms=ctx.wait,
        work_ms=ctx.work,
        retry_ms=ctx.retry,
        events=ctx.events
    )

    return res
