#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Measure kb retrieval quality against a ground-truth question set.

Runs every question in `resources/eval/retrieval-questions.toml` through the
real `kb` binary with `--json` and reports, per question and in aggregate:
recall@5, recall@10, MRR, the rank and similarity of the best-ranked expected
hit, and the number of results carrying a non-null similarity.

That last figure is the **vector candidate count**, and it is a first-class
output rather than a derived one. It is what revealed that a query could produce
two vector candidates out of 3874 embedded nodes while every rank-based metric
merely looked mediocre. Because it is read off `--json` output it is capped by
`--limit`, so the limit is fixed at 100 by default — matching the shipped
`VECTOR_CANDIDATES` cut in `src/storage.rs` — and a count equal to the limit is
reported as saturated rather than as an exact figure. T031 can raise the
experimental cut while the 100-result output ceiling remains fixed; that arm
measures which records enter fusion, not an exact count above the ceiling.

Design notes:

* **Purity.** Everything from `parse_question_set` down to `aggregate` is pure
  and deterministic: no subprocess, no filesystem, no endpoint. The imperative
  shell is confined to the bottom of this file. This is `REPO_INVARIANTS.md`
  ENG-008, and it is what lets the arithmetic be tested with fixture data and no
  live corpus.
* **Read-only.** The harness invokes only `kb search` and `kb get`, and opens
  the derived index itself read-only, so a write is impossible at the SQLite
  layer rather than merely unintended.
* **Environmental faults are loud** (ENG-004). `kb search` degrades to
  keyword-only ranking and still exits 0 when the embedding endpoint is
  unreachable, announcing it on stderr. Reporting the resulting low score as a
  measurement would silently corrupt every comparison this harness exists to
  support, so that stderr note is escalated to a fatal error naming the
  endpoint. A missing index and an unresolvable expected node id are checked
  before any question runs, for the same reason.
* **Not a CI gate.** It requires a populated index and a reachable embedding
  endpoint, neither of which exists in CI. `mise run eval:retrieval` is
  operator-invoked tooling.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import query_rewrite
import rerank

# `VECTOR_CANDIDATES` in `src/storage.rs`: the default vector candidate list is
# truncated to this before fusion. T031 can name a wider experimental cut, but
# the harness still reports only the first 100 output records.
VECTOR_CANDIDATES = 100
DENSE_POOLING_CHOICES = ("max", "mean-top-three", "length-normalized-max")
DENSE_POOLING_DEFAULT = "mean-top-three"

ENV_BASE_URL = "KB_EMBEDDING_BASE_URL"
ENV_MODEL = "KB_EMBEDDING_MODEL"
ENV_QUERY_PREFIX = "KB_EMBEDDING_QUERY_PREFIX"
ENV_MIN_SIMILARITY = "KB_EMBEDDING_MIN_SIMILARITY"
ENV_INDEX_PATH = "KB_INDEX_PATH"
#: The binary's own reranking configuration. Distinct from `--rerank-url`,
#: which is the harness-side stage T019 measured: when these are set the
#: reordering happens inside `kb search` itself, and the run record has to say
#: so or a reranked run is indistinguishable from a fused one.
ENV_RERANK_BASE_URL = "KB_RERANK_BASE_URL"
ENV_RERANK_TOP_K = "KB_RERANK_TOP_K"

# `resolve_search_vector` in `src/cli_main.rs` emits this on stderr and then
# exits 0 when the query could not be embedded. Matching on it is what turns a
# silent degradation into a fatal one here.
DEGRADED_MARKER = "keyword-only ranking"


class EvalError(Exception):
    """The evaluation cannot produce a trustworthy measurement.

    Raised for a malformed question set and for every environmental fault — an
    absent database, an unreachable endpoint, an expected node id that no longer
    resolves. Never raised for a genuinely poor retrieval result, which is a
    measurement rather than an error.
    """


# ── Pure core: model ───────────────────────────────────────────────────────


#: The corpus a record belongs to when nothing says otherwise. Every one of
#: the 42 original questions is written as a bare `node_id`, and every
#: committed run record was measured against them, so this default is what
#: keeps those comparable rather than a convenience.
DEFAULT_CORPUS = "kb"

#: The corpus vocabulary. Fixed here rather than by the registry, because
#: T013 runs before T012 and the question set is the control for every
#: retrieval claim in the plan: a registry that invented its own identifiers
#: would silently break the join between an index row and an `expect` entry.
#: T012 conforms to this list; a mismatch is a defect in T012.
KNOWN_CORPORA = ("kb", "mail")


@dataclass(frozen=True, slots=True)
class Expectation:
    """One acceptable answer for a question.

    Attributes:
        node_id: The node that answers the question, or None when the
            expectation is expressed only as a title pattern.
        title: The node's title as of authoring. Never matched on; compared
            only so that a corpus change is visible rather than silent.
        title_regex: A pattern matched against a result's title, so an
            expectation can survive a re-import that changes node ids. Stored
            titles are truncated at 80 characters, so a pattern must anchor at
            the start of the title rather than assume a full one.
    """

    node_id: str | None = None
    title: str | None = None
    title_regex: str | None = None
    corpus: str = DEFAULT_CORPUS


@dataclass(frozen=True, slots=True)
class Question:
    """One ground-truth question and the answers that count as hits."""

    id: str
    query: str
    population: str
    expect: tuple[Expectation, ...]
    note: str | None = None


@dataclass(frozen=True, slots=True)
class QuestionSet:
    """A parsed question set together with the corpus it was authored against."""

    schema_version: int
    corpus: dict[str, object]
    questions: tuple[Question, ...]


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One result from `kb search --json`.

    Attributes:
        node_id: The node's id.
        title: The node's title, as stored — truncated at 80 characters.
        similarity: The cosine of the node's best-matching chunk against the
            query, or None when the hit came from the keyword side alone. None
            means no vector score was computed, not that the node scored zero.
    """

    node_id: str
    title: str
    similarity: float | None
    corpus: str = DEFAULT_CORPUS


@dataclass(frozen=True, slots=True)
class QuestionResult:
    """The scored outcome of running one question."""

    question_id: str
    population: str
    corpus: str
    query: str
    issued_query: str | None
    rank: int | None
    hit_similarity: float | None
    hit_node_id: str | None
    max_similarity: float | None
    candidate_count: int
    candidates_saturated: bool
    result_count: int
    #: Per signal, what it returned and where it put the expected answer.
    #: Empty unless `--explain-signals` asked for the attribution pass, which
    #: costs a second query per question.
    signals: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def reciprocal_rank(self) -> float:
        """The contribution this question makes to MRR."""
        return 0.0 if self.rank is None else 1.0 / self.rank

    def recall_at(self, k: int) -> bool:
        """Whether an expected answer appeared within the first `k` results."""
        return self.rank is not None and self.rank <= k


@dataclass(frozen=True, slots=True)
class Aggregate:
    """Aggregate metrics over a set of question results."""

    question_count: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    mean_candidate_count: float
    median_candidate_count: float
    min_candidate_count: int
    max_candidate_count: int
    saturated_questions: int
    mean_hit_similarity: float | None
    misses: tuple[str, ...] = field(default=())


# ── Pure core: parsing ─────────────────────────────────────────────────────


def _require(condition: bool, message: str) -> None:
    """Raise `EvalError` with `message` unless `condition` holds."""
    if not condition:
        raise EvalError(message)


def parse_expectation(raw: object, question_id: str) -> Expectation:
    """Parse one expectation entry from the question set.

    Args:
        raw: The decoded TOML inline table.
        question_id: Used only to locate a fault in the error message.

    Returns:
        The parsed expectation.

    Raises:
        EvalError: If the entry is not a table, or names neither a `node_id`
            nor a `title_regex` and so could never match anything.
    """
    _require(isinstance(raw, dict), f"{question_id}: each expect entry must be a table")
    assert isinstance(raw, dict)
    node_id = raw.get("node_id")
    title = raw.get("title")
    title_regex = raw.get("title_regex")
    _require(
        node_id is not None or title_regex is not None,
        f"{question_id}: an expect entry names neither node_id nor title_regex",
    )
    _require(
        node_id is None or isinstance(node_id, str),
        f"{question_id}: node_id must be a string",
    )
    _require(title is None or isinstance(title, str), f"{question_id}: title must be a string")
    _require(
        title_regex is None or isinstance(title_regex, str),
        f"{question_id}: title_regex must be a string",
    )
    if isinstance(title_regex, str):
        try:
            re.compile(title_regex)
        except re.error as exc:
            raise EvalError(f"{question_id}: title_regex is not a valid regex: {exc}") from exc
    corpus = raw.get("corpus", DEFAULT_CORPUS)
    _require(
        isinstance(corpus, str) and corpus in KNOWN_CORPORA,
        f"{question_id}: corpus {corpus!r} is not one of {list(KNOWN_CORPORA)}; "
        "the vocabulary is fixed by the question set and the registry conforms to it, "
        "so a typo here would name a corpus nothing indexes",
    )
    assert isinstance(corpus, str)
    return Expectation(
        node_id=node_id if isinstance(node_id, str) else None,
        title=title if isinstance(title, str) else None,
        title_regex=title_regex if isinstance(title_regex, str) else None,
        corpus=corpus,
    )


def parse_question(raw: object) -> Question:
    """Parse one `[[question]]` table.

    Args:
        raw: The decoded TOML table.

    Returns:
        The parsed question.

    Raises:
        EvalError: If a required field is missing or has the wrong type, or the
            question carries no expectations.
    """
    _require(isinstance(raw, dict), "each question must be a table")
    assert isinstance(raw, dict)
    qid = raw.get("id")
    _require(isinstance(qid, str) and qid != "", "a question is missing a string id")
    assert isinstance(qid, str)
    query = raw.get("query")
    _require(isinstance(query, str) and query.strip() != "", f"{qid}: query must be non-empty")
    population = raw.get("population")
    _require(
        isinstance(population, str) and population != "",
        f"{qid}: population must be a string",
    )
    expect_raw = raw.get("expect")
    _require(
        isinstance(expect_raw, list) and len(expect_raw) > 0,
        f"{qid}: expect must be non-empty",
    )
    assert isinstance(expect_raw, list)
    note = raw.get("note")
    _require(note is None or isinstance(note, str), f"{qid}: note must be a string")
    assert isinstance(query, str)
    assert isinstance(population, str)
    return Question(
        id=qid,
        query=query,
        population=population,
        expect=tuple(parse_expectation(e, qid) for e in expect_raw),
        note=note if isinstance(note, str) else None,
    )


def parse_question_set(data: dict[str, object]) -> QuestionSet:
    """Parse a decoded question-set document.

    Args:
        data: The result of `tomllib.loads` over the question file.

    Returns:
        The parsed question set.

    Raises:
        EvalError: If the schema version is unsupported, the question list is
            absent or empty, or two questions share an id.
    """
    version = data.get("schema_version")
    _require(version == 1, f"unsupported schema_version {version!r}; this harness understands 1")
    corpus = data.get("corpus", {})
    _require(isinstance(corpus, dict), "corpus must be a table")
    assert isinstance(corpus, dict)
    raw_questions = data.get("question")
    _require(
        isinstance(raw_questions, list) and len(raw_questions) > 0,
        "the question set contains no [[question]] entries",
    )
    assert isinstance(raw_questions, list)
    questions = tuple(parse_question(q) for q in raw_questions)
    ids = [q.id for q in questions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    _require(not duplicates, f"duplicate question ids: {', '.join(duplicates)}")
    return QuestionSet(schema_version=1, corpus=corpus, questions=questions)


def parse_search_payload(raw: str) -> tuple[SearchHit, ...]:
    """Parse the JSON envelope `kb search --json` writes to stdout.

    Args:
        raw: The captured stdout.

    Returns:
        The hits in fused rank order; position in the tuple is the rank.

    Raises:
        EvalError: If the payload is not JSON, does not report `ok`, or holds a
            malformed entry. A malformed payload is an environmental fault, not
            a zero-scoring query.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvalError(f"kb search did not emit JSON: {exc}; got {raw[:200]!r}") from exc
    _require(isinstance(payload, dict), "kb search payload is not a JSON object")
    assert isinstance(payload, dict)
    _require(payload.get("ok") is True, f"kb search reported failure: {payload!r}")
    data = payload.get("data")
    _require(isinstance(data, list), "kb search payload has no data array")
    assert isinstance(data, list)
    return parse_hits(data)


def parse_hits(data: list[object]) -> tuple[SearchHit, ...]:
    """Read a list of result objects into hits, in the order given.

    Shared by the CLI's `--json` envelope and the MCP `search` tool's payload,
    which carry the same four fields per hit because they are produced by the
    same ranking (T030).

    Raises:
        EvalError: If an entry is malformed. That is an environmental fault,
            not a zero-scoring query.
    """
    hits: list[SearchHit] = []
    for entry in data:
        _require(isinstance(entry, dict), "a kb search result is not an object")
        assert isinstance(entry, dict)
        node_id = entry.get("id")
        title = entry.get("title")
        similarity = entry.get("similarity")
        # A hit says which corpus it came from. Records written before the
        # payload carried the field are read as the kb corpus, which is what
        # they were: `kb search` searched nothing else until T026.
        corpus = entry.get("corpus", DEFAULT_CORPUS)
        _require(isinstance(node_id, str), "a kb search result has no string id")
        _require(isinstance(title, str), "a kb search result has no string title")
        _require(
            similarity is None or isinstance(similarity, int | float),
            "a kb search result has a non-numeric similarity",
        )
        _require(isinstance(corpus, str), "a kb search result has a non-string corpus")
        assert isinstance(node_id, str)
        assert isinstance(title, str)
        assert isinstance(corpus, str)
        hits.append(
            SearchHit(
                node_id=node_id,
                title=title,
                similarity=None if similarity is None else float(similarity),
                corpus=corpus,
            )
        )
    return tuple(hits)


def parse_tool_payload(raw: str) -> tuple[SearchHit, ...]:
    """Parse what the MCP `search` tool returns.

    The tool answers with `{"count", "results"}` and, where a configured stage
    could not run, a `degraded` array. That array is the tool's equivalent of
    the CLI's note on stderr and is escalated the same way: a degraded ranking
    recorded as a measurement silently corrupts every comparison this harness
    exists to support.

    Raises:
        EvalError: If the payload is not JSON, is malformed, or reports a
            degraded ranking.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvalError(f"the search tool did not emit JSON: {exc}; got {raw[:200]!r}") from exc
    _require(isinstance(payload, dict), "the search tool's payload is not a JSON object")
    assert isinstance(payload, dict)
    degraded = payload.get("degraded")
    if degraded:
        raise EvalError(
            f"the search tool reported a degraded ranking: {degraded}. "
            "Refusing to record a degraded run as a measurement."
        )
    results = payload.get("results")
    _require(isinstance(results, list), "the search tool's payload has no results array")
    assert isinstance(results, list)
    return parse_hits(results)


# ── Pure core: scoring ─────────────────────────────────────────────────────


def expectation_matches(expectation: Expectation, hit: SearchHit) -> bool:
    """Whether `hit` satisfies `expectation`.

    A node id matches by equality; a title regex matches by `re.search` against
    the result's title. `title` is never consulted — it is a drift cross-check,
    not a matching key.
    """
    # The corpus is part of the key, not a label on it. A Message-ID is not a
    # UUID and nothing prevents an id colliding across corpora, so matching on
    # the id alone would eventually score a mail hit as a kb answer.
    if expectation.corpus != hit.corpus:
        return False
    if expectation.node_id is not None and expectation.node_id == hit.node_id:
        return True
    if expectation.title_regex is None:
        return False
    return re.search(expectation.title_regex, hit.title) is not None


def question_corpus(question: Question) -> str:
    """Which corpus a question's answers live in.

    `mixed` when its expectations span several. Cross-corpus questions are
    coming — "what did we decide about X", answered across a note and a mail
    thread — and folding them into one corpus's figures would misattribute the
    result to whichever expectation happened to be written first.
    """
    corpora = {e.corpus for e in question.expect}
    if len(corpora) == 1:
        return next(iter(corpora))
    return "mixed"


def measured_corpora(questions: Sequence[Question]) -> tuple[str, ...]:
    """Every corpus the selected questions expect an answer from, sorted.

    Distinct from `question_corpus`, which reports "mixed" for a question whose
    expectations span several: nothing is stored in a corpus called "mixed",
    and the environment block has to name the real corpora whose size a later
    comparison will want to check.
    """
    return tuple(sorted({e.corpus for q in questions for e in q.expect}))


def first_expected_rank(question: Question, hits: Sequence[SearchHit]) -> int | None:
    """The 1-indexed rank of the best-ranked expected answer, or None.

    A question's expectations are alternatives, not a set that must all be
    found: several nodes may each answer the question, and retrieving any one of
    them is a hit.
    """
    for index, hit in enumerate(hits, start=1):
        if any(expectation_matches(e, hit) for e in question.expect):
            return index
    return None


def vector_candidate_count(hits: Sequence[SearchHit]) -> int:
    """How many results carry a non-null similarity.

    This is the number of vector candidates that cleared the similarity floor
    and survived into the fused ranking. A null similarity means the hit came
    from the keyword side alone.
    """
    return sum(1 for hit in hits if hit.similarity is not None)


def max_similarity(hits: Sequence[SearchHit]) -> float | None:
    """The highest similarity over all results, or None if none carry one.

    Reported alongside the rank-based metrics because the defect this harness
    was built to detect is visible as a collapse in the absolute similarity
    scale, which no rank-based metric can see.
    """
    scores = [hit.similarity for hit in hits if hit.similarity is not None]
    return max(scores) if scores else None


def score_question(
    question: Question,
    hits: Sequence[SearchHit],
    limit: int,
    issued_query: str | None = None,
) -> QuestionResult:
    """Score one question against the results it produced.

    Args:
        question: The question, carrying its acceptable answers.
        hits: The results in fused rank order.
        limit: The `--limit` the search ran at, used to detect saturation.
        issued_query: The text actually searched for, when it differs from the
            question because a rewrite strategy was in force. Recorded so a run
            can be read without rerunning the model that produced it.

    Returns:
        The scored result.
    """
    rank = first_expected_rank(question, hits)
    hit = hits[rank - 1] if rank is not None else None
    count = vector_candidate_count(hits)
    return QuestionResult(
        question_id=question.id,
        population=question.population,
        corpus=question_corpus(question),
        query=question.query,
        issued_query=issued_query,
        rank=rank,
        hit_similarity=None if hit is None else hit.similarity,
        hit_node_id=None if hit is None else hit.node_id,
        max_similarity=max_similarity(hits),
        candidate_count=count,
        candidates_saturated=count >= limit,
        result_count=len(hits),
    )


def aggregate(results: Sequence[QuestionResult]) -> Aggregate:
    """Aggregate per-question results.

    Recall@k is the fraction of questions for which an expected answer appeared
    within the first k results. MRR averages the reciprocal of the best expected
    rank, contributing 0 for a question with no expected answer in the results.
    Mean hit similarity averages only over questions whose expected answer both
    appeared and carried a vector score, so a question retrieved by keyword
    alone does not enter as a zero.

    Raises:
        EvalError: If `results` is empty, which would otherwise report a
            division by zero as a metric.
    """
    _require(len(results) > 0, "cannot aggregate an empty result set")
    n = len(results)
    counts = [r.candidate_count for r in results]
    hit_scores = [r.hit_similarity for r in results if r.hit_similarity is not None]
    return Aggregate(
        question_count=n,
        recall_at_5=sum(1 for r in results if r.recall_at(5)) / n,
        recall_at_10=sum(1 for r in results if r.recall_at(10)) / n,
        mrr=sum(r.reciprocal_rank for r in results) / n,
        mean_candidate_count=statistics.fmean(counts),
        median_candidate_count=statistics.median(counts),
        min_candidate_count=min(counts),
        max_candidate_count=max(counts),
        saturated_questions=sum(1 for r in results if r.candidates_saturated),
        mean_hit_similarity=statistics.fmean(hit_scores) if hit_scores else None,
        misses=tuple(r.question_id for r in results if r.rank is None),
    )


def aggregate_by_corpus(results: Sequence[QuestionResult]) -> dict[str, Aggregate]:
    """Aggregate separately per corpus.

    There is no reason to expect mail to retrieve like transcripts and every
    reason to expect its own register problem: mail is correspondence, and the
    operator's queries will not be.
    """
    corpora = sorted({r.corpus for r in results})
    return {c: aggregate([r for r in results if r.corpus == c]) for c in corpora}


def aggregate_by_population(results: Sequence[QuestionResult]) -> dict[str, Aggregate]:
    """Aggregate separately per population.

    A set that samples several populations can improve on average while getting
    worse on the one that matters; this is what makes that visible.
    """
    populations = sorted({r.population for r in results})
    return {p: aggregate([r for r in results if r.population == p]) for p in populations}


# ── Pure core: rendering ───────────────────────────────────────────────────


def _fmt_opt_float(value: float | None, places: int = 3) -> str:
    """Render an optional float, showing an absent value as a dash."""
    return "-" if value is None else f"{value:.{places}f}"


def render_table(results: Sequence[QuestionResult], limit: int) -> str:
    """Render the per-question results as a fixed-width table."""
    header = (
        f"{'question':<8} {'population':<21} {'rank':>5} {'sim':>7} {'max sim':>8} "
        f"{'cands':>7}  query"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        rank = "MISS" if r.rank is None else str(r.rank)
        cands = f"{r.candidate_count}*" if r.candidates_saturated else str(r.candidate_count)
        lines.append(
            f"{r.question_id:<8} {r.population:<21} {rank:>5} "
            f"{_fmt_opt_float(r.hit_similarity):>7} {_fmt_opt_float(r.max_similarity):>8} "
            f"{cands:>7}  {r.query[:60]}"
        )
    lines.append("")
    lines.append(f"* candidate count saturated at --limit {limit}; the true count may be higher.")
    return "\n".join(lines)


def render_summary(
    overall: Aggregate,
    by_population: dict[str, Aggregate],
    by_corpus: dict[str, Aggregate] | None = None,
) -> str:
    """Render the aggregate metrics as fixed-width tables.

    Broken down per corpus as well as per population, because a corpus can
    drag the average in either direction: adding mail enlarges every candidate
    pool, so a gain on mail and a loss on kb can net out to no visible change
    while both are real.
    """
    header = (
        f"{'population':<21} {'n':>4} {'recall@5':>9} {'recall@10':>10} {'MRR':>7} "
        f"{'mean cands':>11} {'mean sim':>9}"
    )
    lines = [header, "-" * len(header)]

    def row(name: str, a: Aggregate) -> str:
        return (
            f"{name:<21} {a.question_count:>4} {a.recall_at_5:>9.3f} {a.recall_at_10:>10.3f} "
            f"{a.mrr:>7.3f} {a.mean_candidate_count:>11.1f} "
            f"{_fmt_opt_float(a.mean_hit_similarity):>9}"
        )

    lines.extend(row(name, a) for name, a in by_population.items())
    lines.append("-" * len(header))
    lines.append(row("ALL", overall))

    # Suppressed while one corpus holds everything: a table with a single row
    # identical to ALL is noise, and this becomes informative when mail lands.
    if by_corpus and len(by_corpus) > 1:
        lines.append("")
        lines.append(
            f"{'corpus':<21} {'n':>4} {'recall@5':>9} {'recall@10':>10} {'MRR':>7} "
            f"{'mean cands':>11} {'mean sim':>9}"
        )
        lines.append("-" * len(header))
        lines.extend(row(name, a) for name, a in by_corpus.items())
    lines.append("")
    lines.append(
        f"candidate count: min {overall.min_candidate_count}, "
        f"median {overall.median_candidate_count:.0f}, max {overall.max_candidate_count}, "
        f"saturated {overall.saturated_questions}/{overall.question_count}"
    )
    if overall.misses:
        lines.append(f"missed entirely: {', '.join(overall.misses)}")
    return "\n".join(lines)


def _aggregate_json(a: Aggregate) -> dict[str, object]:
    """Render an aggregate as a JSON-serializable mapping."""
    return {
        "question_count": a.question_count,
        "recall_at_5": a.recall_at_5,
        "recall_at_10": a.recall_at_10,
        "mrr": a.mrr,
        "mean_candidate_count": a.mean_candidate_count,
        "median_candidate_count": a.median_candidate_count,
        "min_candidate_count": a.min_candidate_count,
        "max_candidate_count": a.max_candidate_count,
        "saturated_questions": a.saturated_questions,
        "mean_hit_similarity": a.mean_hit_similarity,
        "misses": list(a.misses),
    }


def build_record(
    label: str,
    environment: dict[str, object],
    limit: int,
    results: Sequence[QuestionResult],
) -> dict[str, object]:
    """Assemble the machine-readable run record.

    The record names the model, endpoint, corpus counts, and date alongside the
    metrics, so a later comparison cannot silently compare runs taken against
    different corpora or different endpoints.
    """
    overall = aggregate(results)
    return {
        "label": label,
        "limit": limit,
        "environment": environment,
        "aggregate": _aggregate_json(overall),
        "by_population": {
            name: _aggregate_json(a) for name, a in aggregate_by_population(results).items()
        },
        "by_corpus": {name: _aggregate_json(a) for name, a in aggregate_by_corpus(results).items()},
        "questions": [
            {
                "id": r.question_id,
                "population": r.population,
                "corpus": r.corpus,
                "query": r.query,
                "issued_query": r.issued_query,
                "rank": r.rank,
                "hit_node_id": r.hit_node_id,
                "hit_similarity": r.hit_similarity,
                "max_similarity": r.max_similarity,
                "candidate_count": r.candidate_count,
                "candidates_saturated": r.candidates_saturated,
                "result_count": r.result_count,
                "signals": r.signals,
            }
            for r in results
        ],
    }


# ── Imperative shell ───────────────────────────────────────────────────────


def resolve_kb_binary(explicit: str | None) -> Path:
    """Locate the `kb` binary to measure.

    Args:
        explicit: A path from `--kb-binary` or `KB_BINARY`, if given.

    Returns:
        The resolved path.

    Raises:
        EvalError: If the named binary does not exist, or none is on PATH.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise EvalError(f"kb binary not found at {path}")
        return path.resolve()
    found = shutil.which("kb")
    if found is None:
        raise EvalError("no `kb` binary on PATH; pass --kb-binary or set KB_BINARY")
    return Path(found).resolve()


def resolve_index_path(env: dict[str, str]) -> Path:
    """Resolve the derived index path exactly as the binary does.

    Raises:
        EvalError: If the resolved path does not exist. A missing index must
            fail loudly rather than measure an empty corpus as a zero score.
    """
    raw = env.get(ENV_INDEX_PATH)
    path = Path(raw).expanduser() if raw else Path.home() / ".local/share/kb/index.db"
    if not path.exists():
        raise EvalError(f"kb index not found at {path}; set {ENV_INDEX_PATH} or rebuild it")
    return path


def read_corpus_counts(index_path: Path, corpus: str) -> dict[str, object]:
    """Read the size of one corpus from the derived index, opened read-only.

    `mode=ro` rather than `immutable=1`: the index is a live SQLite database in
    write-ahead-log mode, and `immutable=1` tells SQLite the file cannot change
    and so the log can be ignored — which reports the corpus as it stood before
    the last embedding run committed. Read-only is the invariant this harness
    needs; immutability is a claim about the file that is not true.

    Vectors are keyed by content rather than by record, so a vector belongs to
    a corpus only by way of the passage whose span contains it, which is what
    the join expresses.

    Raises:
        EvalError: If the index cannot be read.
    """
    try:
        connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise EvalError(f"cannot open {index_path} read-only: {exc}") from exc
    try:
        records = connection.execute(
            "select count(*) from records where corpus = ?", (corpus,)
        ).fetchone()[0]
        embedded_query = """
            select count(distinct p.record_id), count(*)
            from embeddings e
            join passages p
              on p.stream_hash = e.stream_hash
             and e.span_start >= p.span_start
             and e.span_start + e.span_len <= p.span_start + p.span_len
            join records r on r.record_id = p.record_id
            where r.corpus = ?
        """
        embedded, chunks = connection.execute(embedded_query, (corpus,)).fetchone()
    except sqlite3.Error as exc:
        raise EvalError(f"cannot read corpus counts from {index_path}: {exc}") from exc
    finally:
        connection.close()
    return {"records": records, "embedded_records": embedded, "chunk_embeddings": chunks}


def check_embedding_configured(env: dict[str, str]) -> None:
    """Fail loudly when the environment disables embedding entirely.

    Without both variables the binary ranks by keyword alone and says so on
    stderr. Catching it here names the missing variable instead of reporting the
    first question's degraded result.

    Raises:
        EvalError: If either required variable is absent or empty.
    """
    for name in (ENV_BASE_URL, ENV_MODEL):
        if not env.get(name):
            raise EvalError(
                f"{name} is unset or empty, so `kb search` would rank by keyword alone; "
                "the measurement would be meaningless"
            )


def run_kb_search(
    binary: Path,
    query: str,
    limit: int,
    env: dict[str, str],
    corpus: str = "kb",
    dense_pooling: str = DENSE_POOLING_DEFAULT,
    vector_candidates: int = VECTOR_CANDIDATES,
) -> tuple[SearchHit, ...]:
    """Run one query through the real binary and parse its results.

    `corpus` is dispatched to `kb search --corpus`, because a search names one
    corpus: since T026 the planner selects its signals by corpus, and a mail
    question asked without the flag is answered from the kb corpus and scores a
    miss for a reason that has nothing to do with retrieval.

    Raises:
        EvalError: If the binary fails, or announces on stderr that it fell back
            to keyword-only ranking. That fallback is exactly the environmental
            fault a low score would otherwise be mistaken for, so it is fatal
            and the message names the endpoint.
    """
    command = [
        str(binary),
        "search",
        query,
        "--limit",
        str(limit),
        "--corpus",
        corpus,
        "--dense-pooling",
        dense_pooling,
        "--vector-candidates",
        str(vector_candidates),
        "--json",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        raise EvalError(
            f"`kb search` exited {completed.returncode} for {query!r}: "
            f"{stderr or completed.stdout[:200]}"
        )
    if DEGRADED_MARKER in stderr:
        raise EvalError(
            f"`kb search` fell back to keyword-only ranking for {query!r}: {stderr}\n"
            f"The embedding endpoint at {env.get(ENV_BASE_URL)!r} serving "
            f"{env.get(ENV_MODEL)!r} is not answering. Refusing to record a "
            "degraded run as a measurement."
        )
    if stderr:
        raise EvalError(f"`kb search` wrote to stderr for {query!r}: {stderr}")
    return parse_search_payload(completed.stdout)


class McpSession:
    """A running `kb-mcp`, answering searches over stdio.

    The point of measuring through the tool rather than the CLI is that it is
    the surface an agent actually has. Since T030 both run
    `kb::search::resolve`, so the two should measure the same; this is what
    makes "should" checkable rather than asserted.

    Framing is newline-delimited JSON-RPC, which is what MCP's stdio transport
    specifies. One process serves every question, because process startup is
    not what is being measured.
    """

    def __init__(self, binary: Path, env: dict[str, str]) -> None:
        """Start the server and complete the initialize handshake.

        Raises:
            EvalError: If the binary is missing or the handshake fails.
        """
        if not binary.is_file():
            raise EvalError(f"kb-mcp binary not found at {binary}")
        self._next_id = 1
        self._process = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        self._request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "eval_retrieval", "version": "0"},
            },
        )
        self._notify("notifications/initialized", {})

    def _write(self, message: dict[str, object]) -> None:
        """Send one JSON-RPC message.

        Raises:
            EvalError: If the server has closed its input.
        """
        if self._process.stdin is None:
            raise EvalError("the MCP server has no stdin")
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()

    def _notify(self, method: str, params: dict[str, object]) -> None:
        """Send a notification, which has no response."""
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict[str, object]) -> dict[str, object]:
        """Send a request and read its response.

        Raises:
            EvalError: If the server closes or answers unparsably.
        """
        request_id = self._next_id
        self._next_id += 1
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        if self._process.stdout is None:
            raise EvalError("the MCP server has no stdout")
        while True:
            line = self._process.stdout.readline()
            if not line:
                raise EvalError(f"the MCP server closed while answering {method}")
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalError(f"the MCP server sent unparsable JSON: {exc}") from exc
            # Notifications carry no id and are not answers to anything.
            if isinstance(parsed, dict) and parsed.get("id") == request_id:
                return parsed

    def search(self, query: str, limit: int, corpus: str) -> tuple[SearchHit, ...]:
        """Run one query through the `search` tool and parse its results.

        Raises:
            EvalError: If the call fails or the payload is degraded.
        """
        response = self._request(
            "tools/call",
            {
                "name": "search",
                "arguments": {"query": query, "limit": limit, "corpus": corpus},
            },
        )
        result = response.get("result")
        if not isinstance(result, dict):
            raise EvalError(f"the search tool returned no result: {response!r}")
        content = result.get("content")
        if not isinstance(content, list) or not content:
            raise EvalError(f"the search tool returned no content: {result!r}")
        first = content[0]
        text = first.get("text") if isinstance(first, dict) else None
        if not isinstance(text, str):
            raise EvalError(f"the search tool returned no text content: {result!r}")
        if result.get("isError") is True:
            raise EvalError(f"the search tool failed for {query!r}: {text}")
        return parse_tool_payload(text)

    def close(self) -> None:
        """Close the session, which is how a stdio server is told to stop."""
        if self._process.stdin is not None:
            self._process.stdin.close()
        self._process.wait(timeout=30)


def run_kb_explain(
    binary: Path,
    query: str,
    limit: int,
    env: dict[str, str],
    corpus: str,
    dense_pooling: str = DENSE_POOLING_DEFAULT,
    vector_candidates: int = VECTOR_CANDIDATES,
) -> dict[str, object]:
    """What each signal contributed to one query.

    Attribution is what the planner exists for, so a cross-corpus measurement
    that reported only fused figures would leave the interesting question —
    which half of mail retrieval is doing the work — exactly as unanswerable as
    it was before the planner was built.

    Raises:
        EvalError: If the binary fails or its trace cannot be parsed.
    """
    command = [
        str(binary),
        "search",
        query,
        "--limit",
        str(limit),
        "--corpus",
        corpus,
        "--dense-pooling",
        dense_pooling,
        "--vector-candidates",
        str(vector_candidates),
        "--explain",
        "--json",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    if completed.returncode != 0:
        raise EvalError(
            f"`kb search --explain` exited {completed.returncode} for {query!r}: "
            f"{completed.stderr.strip() or completed.stdout[:200]}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise EvalError(f"`kb search --explain` returned unparsable JSON: {exc}") from exc
    trace = payload.get("data")
    if not isinstance(trace, dict):
        raise EvalError(f"`kb search --explain` returned no trace: {completed.stdout[:200]}")
    return trace


def signal_contributions(
    question: Question, trace: dict[str, object]
) -> dict[str, dict[str, object]]:
    """Report each signal's candidate count and the expected answer's rank in it.

    A signal that returned nothing and a signal that returned a hundred
    candidates without the answer among them are different failures, and only
    the first is a reason to change the query rather than the ranking.
    """
    wanted = {e.node_id for e in question.expect if e.node_id}
    found: dict[str, dict[str, object]] = {}
    signals = trace.get("signals")
    if not isinstance(signals, list):
        return found
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        candidates = signal.get("candidates")
        candidates = candidates if isinstance(candidates, list) else []
        rank = next(
            (i + 1 for i, candidate in enumerate(candidates) if candidate in wanted),
            None,
        )
        found[str(signal.get("name"))] = {
            "count": len(candidates),
            "expected_rank": rank,
            "error": signal.get("error"),
        }
    return found


def ids_to_verify(
    questions: Sequence[Question], corpora: Sequence[str] = ("kb",)
) -> list[tuple[str, str]]:
    """The `(question id, node id)` pairs `kb get` can actually resolve.

    Only corpora kb has ingested. Mail questions are authored *before* mail is
    ingested — that ordering is what forces "does mail answer things the rest
    of the corpus cannot" to be asked while the answer can still change the
    plan — so their ids do not resolve yet, and asking would turn authoring
    ahead of ingest into a fatal error.
    """
    pairs = []
    for question in questions:
        for expectation in question.expect:
            if expectation.node_id is None or expectation.corpus not in corpora:
                continue
            pairs.append((question.id, expectation.node_id))
    return pairs


def select_corpora(
    questions: Sequence[Question], corpora: Sequence[str] | None
) -> tuple[Question, ...]:
    """The questions belonging to `corpora`, or all of them when None.

    A corpus that is not yet indexed scores every one of its questions a miss,
    and recording that as a measurement would misreport an ingest that has not
    happened as a retrieval failure.
    """
    if corpora is None:
        return tuple(questions)
    return tuple(q for q in questions if question_corpus(q) in corpora)


def verify_expected_nodes(binary: Path, questions: Sequence[Question], env: dict[str, str]) -> None:
    """Confirm every expected node id still resolves before measuring.

    An expected id that no longer resolves turns that question into a guaranteed
    miss, which would read as a retrieval regression rather than as a stale
    question set.

    Raises:
        EvalError: Naming every question whose expected node is unresolvable.
    """
    unresolved: list[str] = []
    for question_id, node_id in ids_to_verify(questions):
        completed = subprocess.run(
            [str(binary), "get", node_id],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            unresolved.append(f"{question_id} -> {node_id}")
    if unresolved:
        raise EvalError(
            "expected nodes no longer resolve; the question set is stale relative to "
            "the corpus:\n  " + "\n  ".join(unresolved)
        )


def build_environment(args: argparse.Namespace) -> dict[str, str]:
    """Assemble the child environment, applying any variant overrides.

    Variants are applied here rather than by editing source, which is what lets
    a prefix or floor be measured without changing a default.
    """
    env = dict(os.environ)
    if args.query_prefix is not None:
        env[ENV_QUERY_PREFIX] = args.query_prefix
    if args.min_similarity is not None:
        env[ENV_MIN_SIMILARITY] = str(args.min_similarity)
    return env


def describe_source_revision(repo_root: Path) -> dict[str, object]:
    """Record the repository revision the run was taken at.

    The binary's path alone does not say what is in it. A comparison between a
    baseline and a post-change run is only meaningful if the revision differs by
    exactly the intended change, so the revision and whether the tree was dirty
    are recorded rather than assumed. Absence of git is not a fault — the
    harness measures a binary, not a checkout — so this degrades to nulls.
    """

    def capture(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    head = capture("rev-parse", "HEAD")
    status = capture("status", "--porcelain")
    return {
        "head": head,
        "dirty": None if status is None else status != "",
        "branch": capture("rev-parse", "--abbrev-ref", "HEAD"),
    }


def describe_environment(
    binary: Path,
    index_path: Path,
    env: dict[str, str],
    counts: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Record everything a later comparison needs to know the runs are alike."""
    return {
        "date": datetime.now(UTC).isoformat(),
        "kb_binary": str(binary),
        "repository": describe_source_revision(Path(__file__).resolve().parent.parent),
        "index": str(index_path),
        "endpoint": env.get(ENV_BASE_URL),
        "model": env.get(ENV_MODEL),
        "query_prefix": env.get(ENV_QUERY_PREFIX),
        "query_prefix_source": "override" if ENV_QUERY_PREFIX in env else "binary default",
        "min_similarity": env.get(ENV_MIN_SIMILARITY),
        "min_similarity_source": "override" if ENV_MIN_SIMILARITY in env else "binary default",
        "binary_rerank_base_url": env.get(ENV_RERANK_BASE_URL),
        "binary_rerank_top_k": env.get(ENV_RERANK_TOP_K),
        "corpus": counts,
    }


def load_question_set(path: Path) -> QuestionSet:
    """Read and parse the question set.

    Raises:
        EvalError: If the file is absent or is not valid TOML.
    """
    if not path.is_file():
        raise EvalError(f"question set not found at {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise EvalError(f"{path} is not valid TOML: {exc}") from exc
    return parse_question_set(data)


def apply_rerank(
    binary: Path,
    reranker: rerank.RerankClient,
    query: str,
    hits: Sequence[SearchHit],
    top_k: int,
    env: dict[str, str],
) -> tuple[SearchHit, ...]:
    """Rescore the top `top_k` hits with the cross-encoder and reorder them.

    The *original* question is what the reranker is given, never a rewritten
    one: seeing query and document together is precisely how a cross-encoder
    is supposed to bridge a register difference, so rewriting first would
    measure the two strategies through each other.

    Raises:
        EvalError: If a candidate's text cannot be read or the reranker fails.
            Either would otherwise silently degrade the run to plain retrieval.
    """
    window = list(hits[:top_k])
    if not window:
        return tuple(hits)
    documents = [read_node_text(binary, hit.node_id, env) for hit in window]
    try:
        scores = reranker.score(query, documents)
    except rerank.RerankError as exc:
        raise EvalError(str(exc)) from exc
    by_id = {hit.node_id: hit for hit in hits}
    ordered_ids = rerank.reorder(
        [hit.node_id for hit in hits],
        {hit.node_id: score for hit, score in zip(window, scores, strict=True)},
        top_k,
    )
    return tuple(by_id[node_id] for node_id in ordered_ids)


def read_node_text(binary: Path, node_id: str, env: dict[str, str]) -> str:
    """Read a node's text, truncated to what the reranker is shown.

    Raises:
        EvalError: If the node cannot be read, since reranking a candidate
            against an empty document would score it as irrelevant and quietly
            demote it.
    """
    completed = subprocess.run(
        [str(binary), "get", node_id],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise EvalError(f"`kb get {node_id}` failed while reranking: {completed.stderr.strip()}")
    return rerank.readable_text(completed.stdout)[: rerank.DOCUMENT_CHARS]


def build_rewrite_client(
    args: argparse.Namespace, env: dict[str, str]
) -> query_rewrite.Client | None:
    """Build the rewriting client a strategy needs, or None when it needs none.

    Raises:
        EvalError: If a model-backed strategy was asked for without naming a
            model. Guessing one would put an unrecorded variable into a
            measurement.
    """
    strategy = query_rewrite.STRATEGIES[args.rewrite]
    if not strategy.needs_model:
        return None
    if not args.rewrite_model:
        raise EvalError(
            f"--rewrite {args.rewrite} needs a local model; pass --rewrite-model "
            "or set KB_REWRITE_MODEL"
        )
    base_url = env.get(ENV_BASE_URL, "")
    if not base_url:
        raise EvalError(f"{ENV_BASE_URL} is unset, so the rewriting model has no endpoint")
    return query_rewrite.LocalChatClient(base_url=base_url, model=args.rewrite_model)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        prog="eval_retrieval.py",
        description="Measure kb retrieval quality against a ground-truth question set.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=repo_root / "resources/eval/retrieval-questions.toml",
        help="Path to the ground-truth question set.",
    )
    parser.add_argument(
        "--label",
        default="unlabelled",
        help="Name for this run; used in the output and as the record filename.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=VECTOR_CANDIDATES,
        help=(
            f"--limit passed to `kb search`. Must be at least {VECTOR_CANDIDATES}, "
            "or the vector candidate count saturates inside the range being compared."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "resources/eval/runs",
        help="Directory the JSON run record is written to.",
    )
    parser.add_argument(
        "--kb-binary",
        default=os.environ.get("KB_BINARY"),
        help="The kb binary to measure. Defaults to $KB_BINARY, then `kb` on PATH.",
    )
    parser.add_argument(
        "--query-prefix",
        default=None,
        help=(
            f"Override {ENV_QUERY_PREFIX} for this run. Pass an empty string to "
            "measure the no-prefix variant."
        ),
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help=f"Override {ENV_MIN_SIMILARITY} for this run, for a floor sweep.",
    )
    parser.add_argument(
        "--dense-pooling",
        choices=DENSE_POOLING_CHOICES,
        default=DENSE_POOLING_DEFAULT,
        help=(
            "How the binary combines passage similarities into one record score. "
            "The default is T031's measured mean of the best three passages."
        ),
    )
    parser.add_argument(
        "--vector-candidates",
        type=int,
        default=VECTOR_CANDIDATES,
        help=(
            "How many dense records enter fusion. The default is the shipped cut; "
            "T031's wider-budget arm uses 400."
        ),
    )
    parser.add_argument(
        "--rewrite",
        default="none",
        choices=sorted(query_rewrite.STRATEGIES),
        help=(
            "Rewrite each question into descriptive register before searching. "
            "`template` is deterministic and model-free; `paraphrase` and `hyde` "
            "call a local model. No kb default is touched either way."
        ),
    )
    parser.add_argument(
        "--rewrite-model",
        default=os.environ.get("KB_REWRITE_MODEL"),
        help="Local chat model used by --rewrite paraphrase|hyde.",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        default=None,
        help=(
            "Measure only questions whose answers live in this corpus; repeatable. "
            "Defaults to every corpus. A corpus kb has not ingested scores every "
            "question a miss, which is an ingest state rather than a measurement."
        ),
    )
    parser.add_argument(
        "--rerank-url",
        default=os.environ.get("KB_RERANK_URL"),
        help=(
            "Base URL of a local reranking server exposing /v1/rerank. When set, the "
            "top --rerank-top-k fused candidates are rescored by the cross-encoder "
            "and reordered. No kb default is touched."
        ),
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=20,
        help="How many fused candidates the reranker rescores.",
    )
    parser.add_argument(
        "--explain-signals",
        action="store_true",
        help=(
            "Record what each retrieval signal contributed per question. Costs a second "
            "query per question and is what makes a fused figure attributable."
        ),
    )
    parser.add_argument(
        "--via",
        choices=("cli", "mcp"),
        default="cli",
        help=(
            "Which surface to measure: `kb search --json`, or the MCP `search` tool over "
            "stdio. The two share one resolution since T030, so a difference between them "
            "is a defect rather than a variant."
        ),
    )
    parser.add_argument(
        "--mcp-binary",
        default=None,
        help="Path to `kb-mcp` for --via mcp. Defaults to the one beside the kb binary.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Print the tables but write no JSON record.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation and report.

    Returns:
        0 on a completed measurement, 1 on any environmental fault. A poor
        retrieval score is a successful measurement and returns 0.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.limit < VECTOR_CANDIDATES:
            raise EvalError(
                f"--limit {args.limit} is below VECTOR_CANDIDATES ({VECTOR_CANDIDATES}); "
                "the vector candidate count would saturate and the comparison would be void"
            )
        if args.vector_candidates < 1:
            raise EvalError("--vector-candidates must be at least 1")
        if args.via == "mcp" and (
            args.dense_pooling != DENSE_POOLING_DEFAULT
            or args.vector_candidates != VECTOR_CANDIDATES
        ):
            raise EvalError(
                "T031 variants are CLI-only measurement controls; --via mcp serves the "
                "shipped defaults"
            )
        question_set = load_question_set(args.questions)
        selected = select_corpora(question_set.questions, args.corpus)
        if not selected:
            raise EvalError(f"--corpus {args.corpus} selected no questions")
        env = build_environment(args)
        check_embedding_configured(env)
        binary = resolve_kb_binary(args.kb_binary)
        index_path = resolve_index_path(env)
        counts = {
            corpus: read_corpus_counts(index_path, corpus) for corpus in measured_corpora(selected)
        }
        verify_expected_nodes(binary, selected, env)

        client = build_rewrite_client(args, env)
        reranker = rerank.RerankClient(base_url=args.rerank_url) if args.rerank_url else None
        session = None
        if args.via == "mcp":
            if args.explain_signals:
                raise EvalError("--explain-signals needs the CLI's trace; drop --via mcp")
            mcp_binary = (
                Path(args.mcp_binary).expanduser() if args.mcp_binary else binary.parent / "kb-mcp"
            )
            session = McpSession(mcp_binary, env)
        results = []
        rerank_seconds: list[float] = []
        reachable = 0
        for q in selected:
            try:
                issued = query_rewrite.rewrite(q.query, args.rewrite, client)
            except query_rewrite.RewriteError as exc:
                # A failed rewrite is an environmental fault, not a poor
                # score: recording the un-rewritten question under the
                # strategy's name would report the control's number as the
                # strategy's.
                raise EvalError(f"{q.id}: {exc}") from exc
            corpus = question_corpus(q)
            hits = (
                session.search(issued, args.limit, corpus)
                if session is not None
                else run_kb_search(
                    binary,
                    issued,
                    args.limit,
                    env,
                    corpus,
                    args.dense_pooling,
                    args.vector_candidates,
                )
            )
            if reranker is not None:
                # Whether the answer was inside the window at all is recorded
                # before reordering: a reranker cannot promote what it never
                # saw, and counting those as its failures would blame it for
                # the retriever's miss.
                if rerank.window_contains_answer(first_expected_rank(q, hits), args.rerank_top_k):
                    reachable += 1
                started = time.monotonic()
                hits = apply_rerank(binary, reranker, q.query, hits, args.rerank_top_k, env)
                rerank_seconds.append(time.monotonic() - started)
            scored = score_question(q, hits, args.limit, None if issued == q.query else issued)
            if args.explain_signals:
                trace = run_kb_explain(
                    binary,
                    issued,
                    args.limit,
                    env,
                    corpus,
                    args.dense_pooling,
                    args.vector_candidates,
                )
                scored = dataclasses.replace(scored, signals=signal_contributions(q, trace))
            results.append(scored)
        if session is not None:
            session.close()
        environment = describe_environment(binary, index_path, env, counts)
        environment["via"] = args.via
        environment["dense_pooling"] = args.dense_pooling
        environment["vector_candidates"] = args.vector_candidates
        environment["dense_pooling_fit"] = (
            "per-query exact expected maximum under independent draws from the "
            "corpus-wide empirical passage-score distribution"
            if args.dense_pooling == "length-normalized-max"
            else None
        )
        environment["rewrite"] = args.rewrite
        environment["rewrite_model"] = args.rewrite_model if client is not None else None
        environment["rewrite_prompt_version"] = (
            query_rewrite.PROMPT_VERSION if client is not None else None
        )
        environment["rerank_url"] = args.rerank_url
        environment["rerank_top_k"] = args.rerank_top_k if reranker is not None else None
        environment["rerank_added_seconds_per_query"] = (
            round(statistics.mean(rerank_seconds), 3) if rerank_seconds else None
        )
        environment["rerank_answer_in_window"] = reachable if reranker is not None else None
        record = build_record(args.label, environment, args.limit, results)
    except EvalError as exc:
        print(f"eval_retrieval: {exc}", file=sys.stderr)
        return 1

    # The record is written before anything is rendered. A measurement that has
    # already run is not something to lose to a formatting bug in the report of
    # it — which is exactly how the first t029 run was thrown away.
    destination = None
    if not args.no_record:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir / f"{args.label}.json"
        destination.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(f"label: {args.label}")
    print(f"model: {environment['model']}   endpoint: {environment['endpoint']}")
    for corpus, sizes in counts.items():
        print(
            f"corpus {corpus}: {sizes['records']} records, "
            f"{sizes['embedded_records']} embedded, {sizes['chunk_embeddings']} vectors"
        )
    print(f"query prefix: {environment['query_prefix']!r} ({environment['query_prefix_source']})")
    print(
        f"min similarity: {environment['min_similarity']!r} "
        f"({environment['min_similarity_source']})"
    )
    print()
    print(render_table(results, args.limit))
    print()
    print(
        render_summary(
            aggregate(results),
            aggregate_by_population(results),
            aggregate_by_corpus(results),
        )
    )

    if destination is not None:
        print()
        print(f"record: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
