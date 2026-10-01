"""Prototype a generated retrieval index and measure its ceiling.

The document-side answer to the register mismatch. Rather than rewriting the
question toward the documents (T001), this writes an additional
retrieval-oriented representation of each *document* and embeds that beside
the existing vectors, so a natural question can land near it.

Two forms are generated, because they attack the mismatch differently:

* `summary` — a descriptive paragraph. Normalizes register.
* `questions` — the questions the node answers, one embedded row each. This
  goes further: it makes the indexed text question-shaped, which is exactly
  why transcripts already retrieve well while authored notes do not.

**This is a throwaway.** Its only product is a number telling T010 whether a
generated index is worth building at corpus scale. It writes into a scratch
copy of the database, never the live one, and generates only for the nodes the
ground-truth set names — roughly 39 of 3910, because the point is the ceiling
rather than the corpus.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import struct
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: Bumped when a prompt changes, so a run is attributable to the text that
#: produced it. Version 2 prepends `/no_think`: the local Qwen3 build is a
#: reasoning model, and under version 1 it spent its entire token budget on a
#: thinking block and returned empty content -- `finish_reason: length` with
#: 2000 reasoning tokens and 0 characters of answer. LM Studio keeps that
#: deliberation in `reasoning_content`, so the failure looks like an empty
#: reply rather than an exhausted budget.
PROMPT_VERSION = 2

#: Qwen3's convention for suppressing the thinking block. Prepended to every
#: prompt: without it this model does not answer at all at any budget worth
#: paying, and with it a summary completes in about 90s rather than never.
NO_THINK = "/no_think\n"

FORMS = ("summary", "questions")

PROMPTS = {
    "summary": (
        "Write one paragraph describing what this document is about, in the third "
        "person, as a reference work would describe it. Name the specific subjects, "
        "people, systems and decisions it covers. Do not editorialise, do not "
        "preface, and reply with the paragraph alone.\n\n{document}"
    ),
    "questions": (
        "Write five questions this document answers, one per line, as a person would "
        "naturally ask them in the first person where that fits. Be specific to the "
        "document's actual content. No numbering, no preamble, questions alone.\n\n"
        "{document}"
    ),
}

#: Reasoning models emit a thinking block before their answer.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

#: List markers a model uses even when told not to.
_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


class GenerateError(RuntimeError):
    """The prototype could not produce or store a generated representation."""


def expected_node_ids(question_set: dict[str, Any], corpus: str = "kb") -> list[str]:
    """Every node id the question set names for `corpus`, in first-seen order.

    The table is `[[question]]`, so the parsed key is `question`. Getting this
    wrong is silent: the selector returns nothing, generation writes nothing,
    and the measurement reports the control's numbers under the prototype's
    name.

    Filtered by corpus because the set now names mail messages by `Message-ID`
    as well. Those are not rows in `kb.db`, and generating for them would mean
    `kb get` on an identifier the database has never held.
    """
    seen: list[str] = []
    questions = question_set.get("question", [])
    if not isinstance(questions, list):
        return seen
    for question in questions:
        for expectation in question.get("expect", []) if isinstance(question, dict) else []:
            if not isinstance(expectation, dict):
                continue
            if expectation.get("corpus", "kb") != corpus:
                continue
            node_id = expectation.get("node_id")
            if isinstance(node_id, str) and node_id not in seen:
                seen.append(node_id)
    return seen


def split_generated(generated: str, form: str) -> list[str]:
    """Split a model's output into the rows that will be embedded.

    A summary is one row. Questions are one row each: averaging five questions
    into a single vector would blur the distinctions that make the form worth
    trying at all.

    Raises:
        GenerateError: If `form` is not a known generation form.
    """
    if form not in FORMS:
        raise GenerateError(f"unknown generation form {form!r}; expected one of {list(FORMS)}")
    text = _THINK_RE.sub("", generated).strip()
    if form == "summary":
        collapsed = " ".join(text.split())
        return [collapsed] if collapsed else []
    rows = []
    for line in text.splitlines():
        stripped = " ".join(_MARKER_RE.sub("", line).split())
        if stripped:
            rows.append(stripped)
    return rows


def encode_vector(vector: Sequence[float]) -> bytes:
    """Encode a vector the way kb stores it: little-endian f32."""
    return struct.pack(f"<{len(vector)}f", *vector)


def decode_vector(blob: bytes) -> list[float]:
    """Decode a stored vector, for tests and inspection."""
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def complete_with_retry(base_url: str, model: str, prompt: str, budget: int) -> str:
    """Ask for a completion, retrying once with a larger budget.

    The local model is a reasoning model and `/no_think` suppresses its
    thinking block on most inputs but not all: on a long document it deliberates
    anyway and exhausts the budget without answering. One retry at triple the
    budget converts that from a failed batch into a slower node.

    Raises:
        GenerateError: If the second attempt also returns nothing.
    """
    try:
        return complete(base_url, model, prompt, max_tokens=budget)
    except GenerateError as first:
        if "deliberating" not in str(first):
            raise
        return complete(base_url, model, prompt, max_tokens=budget * 3)


def complete(
    base_url: str, model: str, prompt: str, timeout: float = 900.0, max_tokens: int = 3000
) -> str:
    """Ask the local chat model for one completion.

    Raises:
        GenerateError: If the endpoint fails or answers in an unexpected shape.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        choice = body["choices"][0]
        content = str(choice["message"].get("content") or "")
        if not content:
            # Distinguish an exhausted budget from a model with nothing to
            # say: the first is a configuration problem and the second is a
            # finding, and they look identical from the content field alone.
            reasoning = choice["message"].get("reasoning_content") or ""
            raise GenerateError(
                f"model {model!r} returned no content "
                f"(finish_reason={choice.get('finish_reason')!r}, "
                f"{len(reasoning)} characters of reasoning): it spent its budget "
                "deliberating rather than answering"
            )
        return content
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
        raise GenerateError(f"generation model {model!r} failed: {exc}") from exc


def embed(base_url: str, model: str, text: str, timeout: float = 300.0) -> list[float]:
    """Embed `text` as a document.

    No prefix is applied: kb's document prefix for this model is empty, and
    the query prefix belongs to queries. Prefixing here would embed the
    generated text under a convention no document uses.

    Raises:
        GenerateError: If the endpoint fails or answers in an unexpected shape.
    """
    payload = json.dumps({"model": model, "input": text}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return [float(v) for v in body["data"][0]["embedding"]]
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
        raise GenerateError(f"embedding model {model!r} failed: {exc}") from exc


def read_node(binary: str, db: Path, node_id: str) -> str:
    """Read a node's org text from the scratch database.

    Raises:
        GenerateError: If the node cannot be read.
    """
    completed = subprocess.run(
        [binary, "--db", str(db), "get", node_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise GenerateError(f"`kb get {node_id}` failed: {completed.stderr.strip()}")
    return completed.stdout


def store_rows(db: Path, node_id: str, model: str, vectors: list[list[float]]) -> int:
    """Append generated vectors as extra chunks of `node_id`.

    kb scores a node by the maximum cosine over its chunks, so an extra chunk
    is exactly how an alternative representation becomes retrievable without
    displacing the original text's vectors.

    Returns:
        The number of rows written.
    """
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(chunk_ix), -1), "
            "(SELECT updated_at FROM nodes WHERE id = ?1) "
            "FROM embeddings WHERE node_id = ?1 AND model = ?2",
            (node_id, model),
        ).fetchone()
        next_ix = int(row[0]) + 1
        updated_at = row[1] or ""
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings "
            "(node_id, model, chunk_ix, embedding, source_updated_at) VALUES (?, ?, ?, ?, ?)",
            [
                (node_id, model, next_ix + offset, encode_vector(vector), updated_at)
                for offset, vector in enumerate(vectors)
            ],
        )
    return len(vectors)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate an index form for the ground-truth nodes and store it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--form", choices=FORMS, required=True)
    parser.add_argument("--db", type=Path, required=True, help="Scratch database to write into.")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--generation-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--kb-binary", default="kb")
    parser.add_argument("--out", type=Path, required=True, help="Where to write generated text.")
    args = parser.parse_args(argv)

    import tomllib

    with args.questions.open("rb") as handle:
        question_set = tomllib.load(handle)
    node_ids = expected_node_ids(question_set)
    print(f"generating {args.form} for {len(node_ids)} ground-truth nodes", flush=True)

    # Written after every node rather than at the end: an hour-long batch that
    # loses everything to a failure on node 26 is a batch nobody runs twice.
    # Nodes already recorded are skipped, so a rerun resumes rather than
    # duplicating rows.
    record: dict[str, Any] = {
        "form": args.form,
        "prompt_version": PROMPT_VERSION,
        "prompt": PROMPTS[args.form],
        "generation_model": args.generation_model,
        "embedding_model": args.embedding_model,
        "nodes": {},
    }
    if args.out.exists():
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        if previous.get("prompt_version") == PROMPT_VERSION and previous.get("form") == args.form:
            record = previous
            print(f"resuming: {len(record['nodes'])} node(s) already generated", flush=True)
    written = 0
    try:
        for index, node_id in enumerate(node_ids, start=1):
            if node_id in record["nodes"]:
                continue
            document = read_node(args.kb_binary, args.db, node_id)
            generated = complete_with_retry(
                args.base_url,
                args.generation_model,
                NO_THINK + PROMPTS[args.form].format(document=document[:6000]),
                3000,
            )
            rows = split_generated(generated, args.form)
            if not rows:
                raise GenerateError(f"{node_id}: generation produced no usable text")
            vectors = [embed(args.base_url, args.embedding_model, row) for row in rows]
            written += store_rows(args.db, node_id, args.embedding_model, vectors)
            record["nodes"][node_id] = rows
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"  [{index}/{len(node_ids)}] {node_id}: {len(rows)} row(s)", flush=True)
    except GenerateError as exc:
        print(f"generate_index: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {written} embedding rows; generated text in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
