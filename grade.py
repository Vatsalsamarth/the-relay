
"""Grade generated answers against the expected answers in questions.json.

Usage:
    python grade.py
    python grade.py --show
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

QUESTIONS_FILE = BASE_DIR / "questions.json"
ANSWERS_FILE = BASE_DIR / "answers.jsonl"
RESULTS_FILE = BASE_DIR / "results.csv"

# --------------------------------------------------
# LOAD QUESTIONS
# --------------------------------------------------

with open(QUESTIONS_FILE, encoding="utf-8") as f:
    questions = {
        q["id"]: q
        for q in json.load(f)
    }

# --------------------------------------------------
# LOAD GENERATED ANSWERS
# --------------------------------------------------

answers = {}

with open(ANSWERS_FILE, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        data = json.loads(line)
        answers[data["request_id"]] = data

# --------------------------------------------------
# REFUSAL DETECTION
# --------------------------------------------------

REFUSAL_PHRASES = (
    "papers don't say",
    "papers don't cover",
    "papers don’t say",
    "the papers don't say",
    "the papers don’t say",
    "can't answer it",
    "cannot answer",
    "not enough information",
)

def is_refusal(answer, refused_flag):
    """Detect explicit refusal responses."""

    text = answer.lower()

    return (
        bool(refused_flag)
        or any(phrase in text for phrase in REFUSAL_PHRASES)
    )


# --------------------------------------------------
# ANSWER KEYWORD VALIDATION
# --------------------------------------------------

def keyword_match(answer, keys):
    """
    Every key group must have at least one matching alternative.

    Example:
        [["512"], ["base transformer"]]

    Both groups must match.

    Empty keys are NOT considered a successful match.
    """

    if not keys:
        return False

    text = answer.lower()

    return all(
        any(
            alternative.lower() in text
            for alternative in group
        )
        for group in keys
    )


# --------------------------------------------------
# CITATION VALIDATION
# --------------------------------------------------

def has_citation(answer):
    """Check for at least one citation such as [S37]."""

    return bool(
        re.search(r"\[S\d+\]", answer, re.IGNORECASE)
    )


# --------------------------------------------------
# LOAD BENCHMARK RESULTS
# --------------------------------------------------

with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    raise SystemExit("No benchmark results found.")

# --------------------------------------------------
# GRADE EACH RESULT
# --------------------------------------------------

show_details = "--show" in sys.argv

for row in rows:

    request_id = row.get("request_id")
    answer_data = answers.get(request_id)

    if not answer_data:
        row["answer_type"] = "missing"
        row["answer_grade"] = "missing"
        row["retrieval_hit"] = ""
        continue

    qid = answer_data.get("qid")

    if qid not in questions:
        row["answer_type"] = "unknown_question"
        row["answer_grade"] = "not_graded"
        row["retrieval_hit"] = ""
        continue

    question = questions[qid]

    answer = answer_data.get("answer", "")
    text = answer.lower()

    answerable = question["answerable"]
    keys = question.get("keys", [])

    fallback = (
        "showing the most relevant passages" in text
        or "[model unavailable]" in text
    )

    refused = is_refusal(
        answer,
        answer_data.get("refused", False)
    )

    hit = keyword_match(answer, keys)

    cited = has_citation(answer)

    row["retrieval_hit"] = str(hit).lower()

    # ----------------------------------------------
    # CLASSIFY ANSWER
    # ----------------------------------------------

    if fallback:

        answer_type = "fallback"
        answer_grade = "fallback"

    elif refused:

        answer_type = "refusal"

        if answerable:
            answer_grade = "refused_wrongly"
        else:
            answer_grade = "refused_correctly"

    else:

        answer_type = "generated"

        if answerable:

            if hit and cited:
                answer_grade = "correct"
            else:
                answer_grade = "wrong"

        else:

            # Answerable=False means the system should
            # refuse rather than invent an answer.
            answer_grade = "wrong"

    row["answer_type"] = answer_type
    row["answer_grade"] = answer_grade

    # ----------------------------------------------
    # OPTIONAL DETAILED OUTPUT
    # ----------------------------------------------

    if show_details:

        print(
            request_id,
            "| Question:", qid,
            "| Type:", answer_type,
            "| Grade:", answer_grade,
            "| Keyword hit:", hit,
            "| Citation:", cited,
        )

        print("Answer:", answer[:250].replace("\n", " "))
        print("-" * 80)


# --------------------------------------------------
# SAVE UPDATED RESULTS
# --------------------------------------------------

fields = list(rows[0].keys())

for field in (
    "answer_type",
    "answer_grade",
    "retrieval_hit",
):
    if field not in fields:
        fields.append(field)

with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(rows)


# --------------------------------------------------
# PRINT SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("BENCHMARK GRADING SUMMARY")
print("=" * 55)

grades = Counter(
    row.get("answer_grade", "")
    for row in rows
    if row.get("answer_grade")
)

answer_types = Counter(
    row.get("answer_type", "")
    for row in rows
    if row.get("answer_type")
)

print("\nAnswer grades:")

for grade, count in sorted(grades.items()):
    print(f"  {grade:<22} {count}")

print("\nAnswer types:")

for answer_type, count in sorted(answer_types.items()):
    print(f"  {answer_type:<22} {count}")

graded = sum(
    count for grade, count in grades.items()
    if grade != "missing" and grade != "not_graded"
)

correct = grades["correct"] + grades["refused_correctly"]

if graded:
    accuracy = correct / graded * 100
    print(f"\nOverall grading success: {accuracy:.2f}%")
else:
    print("\nNo graded answers available.")

print(f"Total benchmark rows: {len(rows)}")

print("\nUpdated results.csv")
print("=" * 55)
