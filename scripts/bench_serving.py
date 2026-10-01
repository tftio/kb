"""Measure the served index backend under the T025 workload.

The question T025 asks is whether the derived SQLite index serves the
deployment's workload, and the answer has to come from the served surface
under concurrency and writes rather than from a single `kb search` timed at a
terminal. This harness fixes the workload the plan names and runs it:

* the committed retrieval questions, at one, four and eight concurrent
  clients, over the MCP HTTP transport `kb-mcp --http` serves;
* under three conditions — idle; while one ingest worker drains a queue of
  submissions the same server accepted; and while `kb reindex` rebuilds the
  index underneath it;
* three complete passes of each;
* in two server variants — **backend**, with no reranker configured, so the
  latency is the index's and the embedding endpoint's alone; and **served**,
  the ordinary reranked path — because a cross-encoder call dominates a
  reranked search and would otherwise hide or manufacture an index result.

Everything the decision turns on is recorded rather than summarized away:
p50, p95 and maximum latency per cell, every error classified as a database
error or not, whether each writer completed, queue progress, the per-stage
timings the tool reports (`stages_ms`), resident memory sampled through the
pass, disk, and record and vector counts before and after. The threshold
verdict is a pure function of the cell summaries so that it is reproducible
from the record alone.

The measurement runs against a **snapshot** of the live store and index
copied into a working directory, never against the live corpus: the ingest
and reindex conditions write, and the ingested records are synthetic.
`kb-mcp`, `kb queue` and `kb reindex` are all pointed at the snapshot by path,
so nothing here needs a modified `HOME`.

Like `eval_retrieval.py`, this is operator-invoked measurement and not a CI
gate: it needs a populated index, a reachable embedding endpoint and, for the
served variant, a reranker. The pure core is covered by
`tests/test_bench_serving.py` without any of those.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import eval_retrieval as ev

# ── Constants ──────────────────────────────────────────────────────────────

#: The two server variants, in the order they run. `backend` is the index
#: path alone; `served` is what an agent actually gets.
VARIANTS: tuple[str, ...] = ("backend", "served")

#: The three conditions, in the order they run within a variant.
CONDITIONS: tuple[str, ...] = ("idle", "ingest", "reindex")

#: Concurrent search clients, as the plan fixes them.
DEFAULT_CLIENTS: tuple[int, ...] = (1, 4, 8)

#: Complete passes per cell.
DEFAULT_PASSES = 3

#: Submissions enqueued before each ingest pass. Enough that the worker is
#: writing for most of a reranked pass at one client; few enough that the
#: snapshot grows by a small fraction over the whole matrix.
DEFAULT_SUBMISSIONS = 20

#: Results asked for per search; the default `kb search` uses.
SEARCH_LIMIT = 10

#: The thresholds of the 2026-08-22 decision.
THRESHOLD_LOADED_RATIO = 2.0
THRESHOLD_DENSE_P95_MS = 300.0

#: T016's projections, carried into the record beside the served count.
T016_PROJECTIONS = {
    "100000_vectors_ms": 335.7,
    "459436_vectors_ms": 2080.0,
    "microseconds_per_vector": 3.36,
}

#: The stages the dense threshold is read from, one per corpus.
DENSE_STAGES: tuple[str, ...] = ("kb-dense", "mail-dense")

#: Substrings that mark an error as the database's rather than the tool's.
DATABASE_ERROR_MARKERS: tuple[str, ...] = (
    "database is locked",
    "sqlite_busy",
    "sqlite",
    "disk i/o error",
    "database disk image is malformed",
    "no such table",
    "database schema has changed",
    "readonly database",
)

ENV_TOKEN = "KB_MCP_TOKEN"
ENV_RERANK_BASE_URL = "KB_RERANK_BASE_URL"
ENV_RERANK_TOP_K = "KB_RERANK_TOP_K"


class BenchError(Exception):
    """A failure of the harness itself, reported and fatal."""


# ── Pure core: the matrix ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Cell:
    """One run of every question: a variant, a condition, a client count, a pass."""

    variant: str
    condition: str
    clients: int
    pass_number: int

    @property
    def key(self) -> str:
        """A stable name for the cell, used in the record and in logs."""
        return f"{self.variant}/{self.condition}/c{self.clients}/p{self.pass_number}"


def matrix(
    variants: Sequence[str] = VARIANTS,
    conditions: Sequence[str] = CONDITIONS,
    clients: Sequence[int] = DEFAULT_CLIENTS,
    passes: int = DEFAULT_PASSES,
) -> tuple[Cell, ...]:
    """Enumerate the cells in the order they run.

    Variant is outermost because a variant is a server; condition next, so the
    writer workload changes least often; then client count; then pass. Three
    passes of one cell run back to back so they see the same corpus size,
    which keeps "pass" a repeat rather than a drift.

    Raises:
        BenchError: If any dimension is empty, a name is unknown, or a client
            count is not positive — an empty matrix would write a record that
            looks like a measurement of nothing.
    """
    if not variants or not conditions or not clients or passes < 1:
        raise BenchError("the matrix needs at least one variant, condition, client count and pass")
    for unknown in sorted(set(variants) - set(VARIANTS)):
        raise BenchError(f"unknown variant {unknown!r}; choose from {VARIANTS}")
    for unknown in sorted(set(conditions) - set(CONDITIONS)):
        raise BenchError(f"unknown condition {unknown!r}; choose from {CONDITIONS}")
    if any(c < 1 for c in clients):
        raise BenchError("every client count must be positive")
    return tuple(
        Cell(v, cond, c, p)
        for v in variants
        for cond in conditions
        for c in clients
        for p in range(1, passes + 1)
    )


# ── Pure core: statistics ─────────────────────────────────────────────────


def percentile(values: Sequence[float], p: float) -> float:
    """The `p`-th percentile by linear interpolation between order statistics.

    `p` is in [0, 100]. Interpolation rather than nearest rank because the
    samples per cell are few (one per question) and nearest rank would make
    p95 jump between two adjacent questions' latencies.

    Raises:
        BenchError: If there are no values; a percentile of nothing is not zero.
    """
    if not values:
        raise BenchError("cannot take a percentile of no values")
    if not 0.0 <= p <= 100.0:
        raise BenchError(f"percentile {p} is outside [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (p / 100.0)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """p50, p95, maximum and mean of a sample, in the sample's unit."""

    n: int
    p50: float | None
    p95: float | None
    max: float | None
    mean: float | None

    def to_json(self) -> dict[str, object]:
        """The JSON form."""
        return asdict(self)


def summarize(values: Sequence[float]) -> LatencyStats:
    """Summarize a latency sample; an empty sample has a count and no figures."""
    if not values:
        return LatencyStats(n=0, p50=None, p95=None, max=None, mean=None)
    return LatencyStats(
        n=len(values),
        p50=percentile(values, 50),
        p95=percentile(values, 95),
        max=max(values),
        mean=statistics.fmean(values),
    )


# ── Pure core: errors ─────────────────────────────────────────────────────


def classify_error(text: str) -> str:
    """Name the kind of an error: `database`, `transport` or `tool`.

    A database error is what the backend decision counts; a transport error
    is the harness's connection failing; everything else is the tool
    reporting a failure in its own terms. The classification is by substring
    against what SQLite and rusqlite actually say, because the tool reports
    failures as readable text rather than codes.
    """
    lowered = text.lower()
    if any(marker in lowered for marker in DATABASE_ERROR_MARKERS):
        return "database"
    if lowered.startswith("transport:"):
        return "transport"
    return "tool"


# ── Pure core: the served address ─────────────────────────────────────────


def parse_announced_address(line: str) -> str:
    """Read `host:port` out of `kb-mcp`'s "listening on http://host:port/mcp".

    Raises:
        BenchError: If the line is not the announcement, so a server that
            failed to bind is reported as such rather than as a connection
            refused later.
    """
    found = re.search(r"http://([^/\s]+)/mcp", line)
    if found is None:
        raise BenchError(f"kb-mcp did not announce an address: {line.strip()!r}")
    return found.group(1)


# ── Pure core: synthetic submissions ──────────────────────────────────────

_ID_PROPERTY = re.compile(r"^(\s*:ID:\s*)\S+\s*$", re.MULTILINE)


def synthesize_submission(document: str, new_id: str, corpus: str = "kb") -> dict[str, str]:
    """A submission carrying `document` under a fresh id.

    The document's own `:ID:` property is rewritten to `new_id` where it has
    one, so the header the worker derives and the body agree; a document with
    no drawer is posted as it is. The text is otherwise untouched, because the
    point of ingesting real documents rather than lorem ipsum is that the
    writer's cost — passages, vectors, links — is the real corpus's cost.
    """
    rewritten = _ID_PROPERTY.sub(lambda m: f"{m.group(1)}{new_id}", document, count=1)
    return {"id": new_id, "corpus": corpus, "document": rewritten}


def assemble_document(title: str, bodies: Sequence[str], record_id: str) -> str:
    """Reconstitute an org document from a record's cached passage bodies.

    The index caches each passage's text (T006's deliberate denormalization),
    so a document of the record's real length and vocabulary can be rebuilt
    from the index alone without reading the store. The result is not the
    stored bytes — headings below the top level are not recovered — but it is
    the same text the record's vectors were computed from, which is what makes
    its embedding cost representative.
    """
    joined = "\n\n".join(body.strip() for body in bodies if body.strip())
    return f"#+title: {title}\n\n* {title}\n:PROPERTIES:\n:ID: {record_id}\n:END:\n\n{joined}\n"


# ── Pure core: outcomes, summaries and the verdict ────────────────────────


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """One search through the served surface, as observed by the client."""

    question_id: str
    latency_ms: float
    ok: bool
    error: str | None
    error_kind: str | None
    stages_ms: dict[str, float]
    rank: int | None
    result_count: int


@dataclass(frozen=True, slots=True)
class WriterReport:
    """What a writer running beside a pass did, and whether it finished."""

    kind: str
    started: bool
    completed: bool
    exit_code: int | None
    elapsed_s: float | None
    runs: int
    detail: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ErrorEntry:
    """One failed query, with the error classified."""

    question: str
    kind: str | None
    message: str | None


@dataclass(frozen=True, slots=True)
class CellSummary:
    """Everything the record keeps for one cell."""

    cell: Cell
    queries: int
    wall_s: float
    latency: LatencyStats
    latencies_ms: tuple[float, ...]
    stages: dict[str, LatencyStats]
    errors: tuple[ErrorEntry, ...]
    ranks: dict[str, int | None]
    recall_at_5: float | None
    writer: WriterReport | None
    rss_kib_max: int | None
    rss_kib_end: int | None

    @property
    def database_errors(self) -> int:
        """How many of the errors were the database's."""
        return sum(1 for e in self.errors if e.kind == "database")

    def to_json(self) -> dict[str, object]:
        """The JSON form, flat enough to read in the run record."""
        return {
            "cell": self.cell.key,
            "variant": self.cell.variant,
            "condition": self.cell.condition,
            "clients": self.cell.clients,
            "pass": self.cell.pass_number,
            "queries": self.queries,
            "wall_s": self.wall_s,
            "latency_ms": self.latency.to_json(),
            "latencies_ms": list(self.latencies_ms),
            "stages_ms": {name: stats.to_json() for name, stats in sorted(self.stages.items())},
            "errors": [asdict(e) for e in self.errors],
            "database_errors": self.database_errors,
            "ranks": dict(sorted(self.ranks.items())),
            "recall_at_5": self.recall_at_5,
            "writer": None if self.writer is None else asdict(self.writer),
            "rss_kib_max": self.rss_kib_max,
            "rss_kib_end": self.rss_kib_end,
        }


def summarize_cell(
    cell: Cell,
    outcomes: Sequence[QueryOutcome],
    writer: WriterReport | None,
    rss_kib_max: int | None,
    rss_kib_end: int | None,
    wall_s: float,
) -> CellSummary:
    """Summarize one cell.

    Latency statistics are over every query, failed or not — a query that
    failed fast would otherwise make a failing cell look quick. Ranks are
    kept per question so a recall change between cells names what moved. Stage
    statistics are over the queries that reported each stage. Errors are
    listed in full, because the decision rule says "no database errors" and a
    count alone would not let a reader check the classification.
    """
    latencies = tuple(o.latency_ms for o in outcomes)
    stages: dict[str, list[float]] = {}
    for outcome in outcomes:
        for name, ms in outcome.stages_ms.items():
            stages.setdefault(name, []).append(ms)
    errors = tuple(
        ErrorEntry(question=o.question_id, kind=o.error_kind, message=o.error)
        for o in outcomes
        if not o.ok
    )
    ranked = [o.rank for o in outcomes if o.ok]
    recall_at_5 = (
        sum(1 for r in ranked if r is not None and r <= 5) / len(ranked) if ranked else None
    )
    # Per-question ranks, so a recall change between cells can be traced to
    # the questions that moved rather than read as a number.
    ranks = {o.question_id: o.rank for o in outcomes if o.ok}
    return CellSummary(
        cell=cell,
        queries=len(outcomes),
        wall_s=wall_s,
        latency=summarize(latencies),
        latencies_ms=latencies,
        stages={name: summarize(values) for name, values in stages.items()},
        errors=errors,
        ranks=ranks,
        recall_at_5=recall_at_5,
        writer=writer,
        rss_kib_max=rss_kib_max,
        rss_kib_end=rss_kib_end,
    )


def _worst_p95(cells: Iterable[CellSummary]) -> float | None:
    """The worst p95 across cells, or None when none has one."""
    values = [c.latency.p95 for c in cells if c.latency.p95 is not None]
    return max(values) if values else None


@dataclass(frozen=True, slots=True)
class Verdict:
    """The threshold decision, with the figure each test was decided on."""

    sqlite_passes: bool
    tests: dict[str, dict[str, object]]

    def to_json(self) -> dict[str, object]:
        """The JSON form."""
        return {"sqlite_passes": self.sqlite_passes, "tests": self.tests}


def verdict(cells: Sequence[CellSummary]) -> Verdict:
    """Apply the 2026-08-22 thresholds to a set of cell summaries.

    SQLite passes when: no cell reports a database error; every writer that
    was started completed; for each variant and client count, the worst p95
    under a writing condition is at most twice the idle p95; and the dense
    stage's p95 over the backend variant stays below 300ms. Each test is
    reported separately with the figure it was decided on, because a verdict
    without its figures is an opinion.
    """
    database_errors = sum(c.database_errors for c in cells)

    writers_started = sum(1 for c in cells if c.writer is not None and c.writer.started)
    writers_completed = sum(
        1 for c in cells if c.writer is not None and c.writer.started and c.writer.completed
    )

    ratios: dict[str, float] = {}
    for variant in sorted({c.cell.variant for c in cells}):
        for clients in sorted({c.cell.clients for c in cells if c.cell.variant == variant}):
            same = [c for c in cells if c.cell.variant == variant and c.cell.clients == clients]
            idle = _worst_p95(c for c in same if c.cell.condition == "idle")
            loaded = _worst_p95(c for c in same if c.cell.condition != "idle")
            if idle is not None and loaded is not None and idle > 0:
                ratios[f"{variant}/c{clients}"] = loaded / idle
    worst_ratio = max(ratios.values()) if ratios else None

    dense = [
        stats.p95
        for c in cells
        if c.cell.variant == "backend"
        for name, stats in c.stages.items()
        if name in DENSE_STAGES and stats.p95 is not None
    ]
    dense_p95 = max(dense) if dense else None

    tests: dict[str, dict[str, object]] = {
        "no_database_errors": {"value": database_errors, "passed": database_errors == 0},
        "writers_completed": {
            "started": writers_started,
            "completed": writers_completed,
            "passed": writers_started == writers_completed,
        },
        "loaded_p95_within_twice_idle": {
            "ratios": ratios,
            "worst": worst_ratio,
            "threshold": THRESHOLD_LOADED_RATIO,
            "passed": worst_ratio is not None and worst_ratio <= THRESHOLD_LOADED_RATIO,
        },
        "dense_p95_below_threshold": {
            "worst_p95_ms": dense_p95,
            "threshold_ms": THRESHOLD_DENSE_P95_MS,
            "passed": dense_p95 is not None and dense_p95 < THRESHOLD_DENSE_P95_MS,
        },
    }
    return Verdict(
        sqlite_passes=all(bool(t["passed"]) for t in tests.values()),
        tests=tests,
    )


def render_report(cells: Sequence[CellSummary], result: Verdict) -> str:
    """A table of the cells and the verdict, for the terminal."""

    def figure(value: float | None) -> str:
        return "-" if value is None else f"{value:.0f}"

    lines = [
        f"{'cell':<28} {'p50':>8} {'p95':>8} {'max':>8} {'errs':>5} {'db':>3} "
        f"{'writer':>7} {'r@5':>5}"
    ]
    for cell in cells:
        writer = "-" if cell.writer is None else ("done" if cell.writer.completed else "FAIL")
        recall = "-" if cell.recall_at_5 is None else f"{cell.recall_at_5:.3f}"
        lines.append(
            f"{cell.cell.key:<28} {figure(cell.latency.p50):>8} {figure(cell.latency.p95):>8} "
            f"{figure(cell.latency.max):>8} {len(cell.errors):>5} {cell.database_errors:>3} "
            f"{writer:>7} {recall:>5}"
        )
    lines.append("")
    lines.append(f"SQLite passes: {result.sqlite_passes}")
    for name, test in result.tests.items():
        lines.append(f"  {name}: {json.dumps(test, default=str)}")
    return "\n".join(lines)


# ── Shell: the served MCP client ───────────────────────────────────────────


class McpHttpClient:
    """One client connection to `kb-mcp --http`, speaking JSON-RPC over POST.

    The transport is stateless (T021), so there is no session to keep; a
    connection is kept alive because reconnecting per request would measure
    the TCP handshake rather than the index.
    """

    def __init__(self, address: str, token: str) -> None:
        """Connect to `address` (`host:port`) and complete the handshake."""
        self._address = address
        self._token = token
        self._next_id = 1
        self._connection = self._connect()
        self._request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "bench_serving", "version": "0"},
            },
        )

    def _connect(self) -> http.client.HTTPConnection:
        host, _, port = self._address.rpartition(":")
        return http.client.HTTPConnection(host, int(port), timeout=600)

    def _post(self, path: str, body: str, accept: str | None) -> tuple[int, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._token}",
        }
        if accept is not None:
            headers["Accept"] = accept
        try:
            self._connection.request("POST", path, body=body, headers=headers)
            response = self._connection.getresponse()
            raw = response.read().decode("utf-8", errors="replace")
        except (OSError, http.client.HTTPException) as exc:
            self._connection.close()
            self._connection = self._connect()
            raise BenchError(f"transport: {exc}") from exc
        return response.status, raw

    def _request(self, method: str, params: dict[str, object]) -> dict[str, object]:
        request_id = self._next_id
        self._next_id += 1
        body = json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        status, raw = self._post("/mcp", body, "application/json, text/event-stream")
        if status != 200:
            raise BenchError(f"transport: HTTP {status}: {raw[:200]}")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BenchError(f"transport: unparsable response: {raw[:200]}") from exc
        if not isinstance(parsed, dict):
            raise BenchError(f"transport: not a JSON-RPC response: {raw[:200]}")
        return parsed

    def search(self, query: str, limit: int, corpus: str) -> dict[str, object]:
        """Call the search tool and return its payload.

        Raises:
            BenchError: For a transport failure (prefixed `transport:`), a
                JSON-RPC error, or a tool-level failure, carrying the tool's
                own words so the error can be classified.
        """
        response = self._request(
            "tools/call",
            {"name": "search", "arguments": {"query": query, "limit": limit, "corpus": corpus}},
        )
        if "error" in response:
            raise BenchError(f"jsonrpc: {json.dumps(response['error'])}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise BenchError(f"transport: no result in {json.dumps(response)[:200]}")
        content = result.get("content")
        if not isinstance(content, list) or not content:
            raise BenchError("transport: the search tool returned no content")
        first = content[0]
        text = first.get("text") if isinstance(first, dict) else None
        if not isinstance(text, str):
            raise BenchError("transport: the search tool returned no text content")
        if result.get("isError") is True:
            raise BenchError(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BenchError(f"transport: unparsable tool payload: {text[:200]}") from exc
        if not isinstance(payload, dict):
            raise BenchError(f"transport: tool payload is not an object: {text[:200]}")
        return payload

    def post_ingest(self, submission: dict[str, str]) -> str:
        """POST one submission to `/ingest`, returning the acknowledged body.

        Raises:
            BenchError: If the server does not answer 202.
        """
        status, raw = self._post("/ingest", json.dumps(submission), None)
        if status != 202:
            raise BenchError(f"ingest refused: HTTP {status}: {raw[:200]}")
        return raw

    def close(self) -> None:
        """Drop the connection."""
        self._connection.close()


# ── Shell: the server ─────────────────────────────────────────────────────


class Server:
    """A running `kb-mcp --http`, with its stderr collected and RSS readable."""

    def __init__(
        self,
        binary: Path,
        env: dict[str, str],
        store: Path,
        index: Path,
        queue: Path,
        cwd: Path,
    ) -> None:
        """Start the server on a free loopback port and wait for its address."""
        self._process = subprocess.Popen(
            [
                str(binary),
                "--http",
                "127.0.0.1:0",
                "--store",
                str(store),
                "--index",
                str(index),
                "--queue",
                str(queue),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=cwd,
        )
        if self._process.stderr is None:
            raise BenchError("kb-mcp has no stderr")
        first = self._process.stderr.readline()
        if not first:
            self._process.wait()
            raise BenchError(
                f"kb-mcp exited before announcing an address ({self._process.returncode})"
            )
        self.address = parse_announced_address(first)
        self.stderr_lines: list[str] = []
        self._reader = threading.Thread(target=self._collect_stderr, daemon=True)
        self._reader.start()

    def _collect_stderr(self) -> None:
        if self._process.stderr is None:
            return
        for line in self._process.stderr:
            self.stderr_lines.append(line.rstrip("\n"))

    @property
    def pid(self) -> int:
        """The server's process id, for sampling."""
        return self._process.pid

    def rss_kib(self) -> int | None:
        """Resident set size in KiB, via `ps`, or None if it cannot be read."""
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(self.pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        text = completed.stdout.strip()
        return int(text) if completed.returncode == 0 and text.isdigit() else None

    def stop(self) -> int | None:
        """Stop the server as a supervisor would, and return its exit code."""
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._reader.join(timeout=5)
        return self._process.returncode


class RssSampler:
    """Sample a server's RSS on a thread for the life of a pass."""

    def __init__(self, server: Server, interval_s: float = 0.5) -> None:
        """Start sampling `server` every `interval_s` seconds."""
        self._server = server
        self._interval = interval_s
        self._stop = threading.Event()
        self.samples: list[int] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._server.rss_kib()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self._interval)

    def finish(self) -> tuple[int | None, int | None]:
        """Stop sampling and return (max, last)."""
        self._stop.set()
        self._thread.join(timeout=5)
        last = self._server.rss_kib()
        if last is not None:
            self.samples.append(last)
        return (max(self.samples) if self.samples else None, last)


# ── Shell: the snapshot and its paths ─────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Paths:
    """Where the snapshot lives."""

    root: Path
    store: Path
    index: Path
    queue: Path


def snapshot(source: Path, workdir: Path) -> Paths:
    """Copy the live store and index under `workdir`, or reuse a copy already there.

    The index is copied with its write-ahead log and shared-memory files when
    they exist, so the copy is the committed state plus whatever the log
    holds; copying the main file alone would silently drop the last commits.
    The store is a bare git repository and copies as a directory.

    Raises:
        BenchError: If the source index or store is missing.
    """
    store_src = source / "store"
    index_src = source / "index.db"
    if not index_src.is_file():
        raise BenchError(f"no index at {index_src}")
    if not store_src.is_dir():
        raise BenchError(f"no store at {store_src}")
    root = workdir / "snapshot"
    paths = Paths(root=root, store=root / "store", index=root / "index.db", queue=root / "queue")
    if paths.index.exists() and paths.store.exists():
        paths.queue.mkdir(parents=True, exist_ok=True)
        return paths
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(store_src, paths.store, dirs_exist_ok=True)
    shutil.copy2(index_src, paths.index)
    for suffix in ("-wal", "-shm"):
        companion = index_src.with_name(index_src.name + suffix)
        if companion.exists():
            shutil.copy2(companion, paths.index.with_name(paths.index.name + suffix))
    paths.queue.mkdir(parents=True, exist_ok=True)
    return paths


def read_counts(index: Path) -> dict[str, object]:
    """Records, passages and vectors per corpus, read-only."""
    counts: dict[str, object] = {}
    connection = sqlite3.connect(f"file:{index}?mode=ro", uri=True)
    try:
        corpora = [
            str(row[0])
            for row in connection.execute("select distinct corpus from records order by 1")
        ]
        counts["vectors_total"] = connection.execute("select count(*) from embeddings").fetchone()[
            0
        ]
        counts["passages_total"] = connection.execute("select count(*) from passages").fetchone()[0]
    finally:
        connection.close()
    for corpus in corpora:
        counts[corpus] = ev.read_corpus_counts(index, corpus)
    return counts


def disk_usage(paths: Paths) -> dict[str, int]:
    """Bytes on disk for the index (with its log), the store and the queue."""

    def tree(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

    companions = (paths.index.with_name(paths.index.name + s) for s in ("-wal", "-shm"))
    index_bytes = sum(p.stat().st_size for p in (paths.index, *companions) if p.exists())
    return {
        "index_bytes": index_bytes,
        "store_bytes": tree(paths.store),
        "queue_bytes": tree(paths.queue),
    }


def sample_documents(index: Path, count: int, seed: int) -> list[tuple[str, str]]:
    """`count` (title, document) pairs reconstituted from kb records in the index.

    Deterministic for a seed — records are chosen by a fixed stride over the
    sorted id list — so a run can be repeated against the same snapshot with
    the same submissions.

    The submissions are therefore duplicates of real records under new ids,
    and a duplicate competes with its original in the ranking: a cell run
    after an ingest pass can rank a question's expected record below its
    copy. That is a property of this workload, not of the backend, and the
    per-question ranks in the record are what let it be read as such.

    Raises:
        BenchError: If the snapshot holds no kb records.
    """
    connection = sqlite3.connect(f"file:{index}?mode=ro", uri=True)
    try:
        ids = [
            str(row[0])
            for row in connection.execute(
                "select record_id from records where corpus = 'kb' order by record_id"
            )
        ]
        if not ids:
            raise BenchError("the snapshot holds no kb records to synthesize submissions from")
        chosen = [ids[(seed * 7919 + i * 104729) % len(ids)] for i in range(count)]
        documents: list[tuple[str, str]] = []
        for record_id in chosen:
            title_row = connection.execute(
                "select title from records where record_id = ?", (record_id,)
            ).fetchone()
            bodies = [
                str(row[0])
                for row in connection.execute(
                    "select body from passages where record_id = ? order by span_start",
                    (record_id,),
                )
            ]
            title = str(title_row[0]) if title_row and title_row[0] else record_id
            documents.append((title, assemble_document(title, bodies, record_id)))
    finally:
        connection.close()
    return documents


# ── Shell: writers ────────────────────────────────────────────────────────


def _json_envelope_data(stdout: str) -> object:
    """The `data` of a kb JSON envelope, or the whole thing, or the raw text."""
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout.strip()[-2000:]
    if isinstance(envelope, dict):
        return envelope.get("data", envelope)
    return envelope


class DrainWorker:
    """`kb queue drain` run once beside a pass."""

    def __init__(self, kb: Path, env: dict[str, str], paths: Paths, cwd: Path) -> None:
        """Start the worker."""
        self._started = time.perf_counter()
        self._process = subprocess.Popen(
            [
                str(kb),
                "queue",
                "--queue",
                str(paths.queue),
                "--store",
                str(paths.store),
                "--index",
                str(paths.index),
                "drain",
                "--json",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=cwd,
        )

    def finish(self, timeout_s: float) -> WriterReport:
        """Wait for the worker and report what it did."""
        try:
            stdout, stderr = self._process.communicate(timeout=timeout_s)
            completed = self._process.returncode == 0
            code: int | None = self._process.returncode
        except subprocess.TimeoutExpired:
            self._process.kill()
            stdout, stderr = self._process.communicate()
            completed = False
            code = None
        detail: dict[str, object] = {
            "stderr": stderr.strip()[-2000:],
            "report": _json_envelope_data(stdout),
        }
        return WriterReport(
            kind="ingest",
            started=True,
            completed=completed,
            exit_code=code,
            elapsed_s=time.perf_counter() - self._started,
            runs=1,
            detail=detail,
        )


class ReindexLoop:
    """`kb reindex` run repeatedly on a thread until told to stop.

    A full rebuild of this corpus is seconds, and a pass is minutes, so one
    rebuild would leave most of the pass idle. The loop keeps a rebuild in
    flight for the whole pass and records every run; "writer completes" is
    then every run exiting zero, with the last one allowed to finish after the
    pass ends rather than being killed mid-transaction.
    """

    def __init__(self, kb: Path, env: dict[str, str], paths: Paths, cwd: Path) -> None:
        """Start the loop."""
        self._kb = kb
        self._env = env
        self._paths = paths
        self._cwd = cwd
        self._stop = threading.Event()
        self.runs: list[dict[str, object]] = []
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            began = time.perf_counter()
            completed = subprocess.run(
                [
                    str(self._kb),
                    "reindex",
                    "--store",
                    str(self._paths.store),
                    "--index",
                    str(self._paths.index),
                ],
                capture_output=True,
                text=True,
                check=False,
                env=self._env,
                cwd=self._cwd,
            )
            self.runs.append(
                {
                    "elapsed_s": time.perf_counter() - began,
                    "exit_code": completed.returncode,
                    "stdout": completed.stdout.strip()[-500:],
                    "stderr": completed.stderr.strip()[-500:],
                }
            )

    def finish(self, timeout_s: float) -> WriterReport:
        """Let the in-flight rebuild finish, then report every run."""
        self._stop.set()
        self._thread.join(timeout=timeout_s)
        alive = self._thread.is_alive()
        codes = [run["exit_code"] for run in self.runs]
        completed = not alive and bool(codes) and all(code == 0 for code in codes)
        last = codes[-1] if codes and not alive else None
        return WriterReport(
            kind="reindex",
            started=True,
            completed=completed,
            exit_code=last if isinstance(last, int) else None,
            elapsed_s=time.perf_counter() - self._started,
            runs=len(self.runs),
            detail={"runs": self.runs, "timed_out": alive},
        )


# ── Shell: running a cell ─────────────────────────────────────────────────


def _stages_of(payload: dict[str, object]) -> dict[str, float]:
    raw = payload.get("stages_ms")
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items() if isinstance(v, int | float)}


def run_queries(
    server: Server,
    token: str,
    questions: Sequence[ev.Question],
    clients: int,
) -> list[QueryOutcome]:
    """Run every question once through `clients` concurrent connections.

    Questions are dealt round-robin to the clients so each client's share is
    fixed and the cell's wall clock is the slowest client's. A client that
    fails a query records the failure and moves on; the pass does not stop,
    because a failing cell is a result.
    """
    outcomes: list[QueryOutcome] = []
    lock = threading.Lock()

    def one(client: McpHttpClient, question: ev.Question) -> QueryOutcome:
        corpus = ev.question_corpus(question)
        began = time.perf_counter()
        try:
            payload = client.search(question.query, SEARCH_LIMIT, corpus)
            latency = (time.perf_counter() - began) * 1000.0
            results = payload.get("results")
            hits = ev.parse_hits(list(results) if isinstance(results, list) else [])
            return QueryOutcome(
                question_id=question.id,
                latency_ms=latency,
                ok=True,
                error=None,
                error_kind=None,
                stages_ms=_stages_of(payload),
                rank=ev.first_expected_rank(question, hits),
                result_count=len(hits),
            )
        except (BenchError, ev.EvalError) as exc:
            latency = (time.perf_counter() - began) * 1000.0
            message = str(exc)
            return QueryOutcome(
                question_id=question.id,
                latency_ms=latency,
                ok=False,
                error=message,
                error_kind=classify_error(message),
                stages_ms={},
                rank=None,
                result_count=0,
            )

    def worker(share: Sequence[ev.Question]) -> None:
        client = McpHttpClient(server.address, token)
        try:
            for question in share:
                outcome = one(client, question)
                with lock:
                    outcomes.append(outcome)
        finally:
            client.close()

    shares = [list(questions[i::clients]) for i in range(clients)]
    threads = [threading.Thread(target=worker, args=(share,)) for share in shares if share]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outcomes


def enqueue_submissions(
    server: Server,
    token: str,
    documents: Sequence[tuple[str, str]],
) -> list[str]:
    """Post one submission per document to the served `/ingest`, returning the ids."""
    client = McpHttpClient(server.address, token)
    ids: list[str] = []
    try:
        for _, document in documents:
            new_id = str(uuid.uuid4())
            client.post_ingest(synthesize_submission(document, new_id))
            ids.append(new_id)
    finally:
        client.close()
    return ids


def queue_status(kb: Path, env: dict[str, str], paths: Paths, cwd: Path) -> object:
    """`kb queue status --json` against the snapshot's queue."""
    completed = subprocess.run(
        [str(kb), "queue", "--queue", str(paths.queue), "status", "--json"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=cwd,
    )
    if completed.returncode != 0:
        return {"exit_code": completed.returncode, "stderr": completed.stderr.strip()[-500:]}
    return _json_envelope_data(completed.stdout)


@dataclass(frozen=True, slots=True)
class Tooling:
    """The binaries, environment and working directory a run uses."""

    kb: Path
    kb_mcp: Path
    env: dict[str, str]
    token: str
    cwd: Path
    submissions: int
    seed: int
    writer_timeout_s: float


def run_cell(
    cell: Cell,
    server: Server,
    tooling: Tooling,
    paths: Paths,
    questions: Sequence[ev.Question],
) -> CellSummary:
    """Run one cell under its condition and summarize it."""
    writer: DrainWorker | ReindexLoop | None = None
    before: dict[str, object] = {}
    if cell.condition == "ingest":
        documents = sample_documents(
            paths.index, tooling.submissions, tooling.seed + cell.pass_number
        )
        enqueued = enqueue_submissions(server, tooling.token, documents)
        before = {
            "enqueued": len(enqueued),
            "queue": queue_status(tooling.kb, tooling.env, paths, tooling.cwd),
        }
        writer = DrainWorker(tooling.kb, tooling.env, paths, tooling.cwd)
    elif cell.condition == "reindex":
        writer = ReindexLoop(tooling.kb, tooling.env, paths, tooling.cwd)

    sampler = RssSampler(server)
    began = time.perf_counter()
    outcomes = run_queries(server, tooling.token, questions, cell.clients)
    wall = time.perf_counter() - began
    rss_max, rss_end = sampler.finish()

    report: WriterReport | None = None
    if writer is not None:
        report = writer.finish(tooling.writer_timeout_s)
        if cell.condition == "ingest":
            after = queue_status(tooling.kb, tooling.env, paths, tooling.cwd)
            report = WriterReport(
                kind=report.kind,
                started=report.started,
                completed=report.completed,
                exit_code=report.exit_code,
                elapsed_s=report.elapsed_s,
                runs=report.runs,
                detail={**report.detail, "queue_before": before, "queue_after": after},
            )
    return summarize_cell(cell, outcomes, report, rss_max, rss_end, wall)


# ── Shell: the run ────────────────────────────────────────────────────────


def describe_host() -> dict[str, object]:
    """What the measurement ran on, as far as the platform will say."""

    def sysctl(name: str) -> str | None:
        completed = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    memory: str | None = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory = line.split(":", 1)[1].strip()
                break
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "memory": sysctl("hw.memsize") or memory,
    }


def variant_env(base: dict[str, str], variant: str, rerank_url: str | None) -> dict[str, str]:
    """The server environment for a variant.

    `backend` unsets the reranker so the measured latency is the index's;
    `served` names it. Both strip the path variables so the command-line paths
    are the only ones in force.
    """
    env = dict(base)
    for name in ("KB_QUEUE_PATH", "KB_INDEX_PATH", "KB_STORE_PATH", "KB_DB_PATH"):
        env.pop(name, None)
    if variant == "backend":
        env.pop(ENV_RERANK_BASE_URL, None)
    elif rerank_url:
        env[ENV_RERANK_BASE_URL] = rerank_url
    return env


def build_parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(
        description="Measure the served SQLite index under T025's concurrency, ingest and "
        "reindex matrix.",
    )
    parser.add_argument(
        "--workdir", type=Path, required=True, help="Where the snapshot and queue live."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / ".local/share/kb",
        help="The live kb data directory to snapshot (default: ~/.local/share/kb).",
    )
    parser.add_argument("--kb-binary", default=os.environ.get("KB_BINARY"), help="Path to kb.")
    parser.add_argument(
        "--kb-mcp-binary", default=os.environ.get("KB_MCP_BINARY"), help="Path to kb-mcp."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("resources/eval/retrieval-questions.toml"),
        help="The ground-truth question set.",
    )
    parser.add_argument("--corpus", action="append", help="Measure only this corpus; repeatable.")
    parser.add_argument(
        "--variant", action="append", choices=VARIANTS, help="Run only this variant."
    )
    parser.add_argument(
        "--condition", action="append", choices=CONDITIONS, help="Run only this condition."
    )
    parser.add_argument("--clients", type=int, action="append", help="A client count; repeatable.")
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES, help="Passes per cell.")
    parser.add_argument(
        "--submissions",
        type=int,
        default=DEFAULT_SUBMISSIONS,
        help="Submissions per ingest pass.",
    )
    parser.add_argument(
        "--limit-questions", type=int, help="Use only the first N questions (smoke runs)."
    )
    parser.add_argument(
        "--rerank-url", default=os.environ.get(ENV_RERANK_BASE_URL), help="Reranker base URL."
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed for submission sampling.")
    parser.add_argument(
        "--writer-timeout", type=float, default=1800.0, help="Seconds to wait for a writer."
    )
    parser.add_argument(
        "--venue", default="laptop", help="Where the run was taken, for the record."
    )
    parser.add_argument("--label", default="t025-sqlite", help="Run label; names the record file.")
    parser.add_argument(
        "--out", type=Path, help="Record path (default resources/eval/runs/<label>.json)."
    )
    return parser


def resolve_binary(explicit: str | None, name: str) -> Path:
    """Find a binary by explicit path or on PATH."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise BenchError(f"{name} not found at {path}")
        return path.resolve()
    found = shutil.which(name)
    if found is None:
        raise BenchError(f"no `{name}` on PATH; pass --{name}-binary")
    return Path(found).resolve()


def _stop_server(server: Server, variant: str | None) -> dict[str, object]:
    """Stop a variant's server and keep what it said."""
    code = server.stop()
    return {"variant": variant, "exit_code": code, "stderr": server.stderr_lines[-50:]}


def _run_matrix(
    cells: Sequence[Cell],
    questions: Sequence[ev.Question],
    tooling: Tooling,
    paths: Paths,
    base_env: dict[str, str],
    rerank_url: str | None,
) -> tuple[list[CellSummary], list[dict[str, object]]]:
    """Run every cell, starting one server per variant."""
    results: list[CellSummary] = []
    server_reports: list[dict[str, object]] = []
    current_variant: str | None = None
    server: Server | None = None
    try:
        for cell in cells:
            if server is None or cell.variant != current_variant:
                if server is not None:
                    server_reports.append(_stop_server(server, current_variant))
                env = variant_env(base_env, cell.variant, rerank_url)
                server = Server(
                    tooling.kb_mcp, env, paths.store, paths.index, paths.queue, tooling.cwd
                )
                current_variant = cell.variant
                print(f"bench_serving: {cell.variant} server at {server.address}")
            summary = run_cell(cell, server, tooling, paths, questions)
            results.append(summary)
            stats = summary.latency
            print(
                f"bench_serving: {cell.key}: p50 {stats.p50 or 0:.0f}ms p95 {stats.p95 or 0:.0f}ms "
                f"max {stats.max or 0:.0f}ms errors {len(summary.errors)} "
                f"wall {summary.wall_s:.1f}s"
            )
    finally:
        if server is not None:
            server_reports.append(_stop_server(server, current_variant))
    return results, server_reports


def main(argv: Sequence[str] | None = None) -> int:
    """Run the matrix and write the record."""
    args = build_parser().parse_args(argv)
    try:
        kb = resolve_binary(args.kb_binary, "kb")
        kb_mcp = resolve_binary(args.kb_mcp_binary, "kb-mcp")
        base_env = dict(os.environ)
        ev.check_embedding_configured(base_env)
        token = base_env.get(ENV_TOKEN) or uuid.uuid4().hex
        base_env[ENV_TOKEN] = token
        variants = tuple(args.variant) if args.variant else VARIANTS
        if "served" in variants and not args.rerank_url:
            raise BenchError(
                "the served variant needs a reranker; pass --rerank-url or set "
                f"{ENV_RERANK_BASE_URL}, or run --variant backend alone"
            )
        cells = matrix(
            variants,
            tuple(args.condition) if args.condition else CONDITIONS,
            tuple(args.clients) if args.clients else DEFAULT_CLIENTS,
            args.passes,
        )
        question_set = ev.load_question_set(args.questions)
        questions = list(question_set.questions)
        if args.corpus:
            wanted = set(args.corpus)
            questions = [q for q in questions if ev.question_corpus(q) in wanted]
        if args.limit_questions:
            questions = questions[: args.limit_questions]
        if not questions:
            raise BenchError("no questions selected")

        cwd = Path(__file__).resolve().parent.parent
        paths = snapshot(args.source, args.workdir)
        tooling = Tooling(
            kb=kb,
            kb_mcp=kb_mcp,
            env=variant_env(base_env, "backend", None),
            token=token,
            cwd=cwd,
            submissions=args.submissions,
            seed=args.seed,
            writer_timeout_s=args.writer_timeout,
        )
        counts_before = read_counts(paths.index)
        disk_before = disk_usage(paths)
        started = datetime.now(UTC)
        print(
            f"bench_serving: {len(cells)} cells over {len(questions)} questions; "
            f"snapshot at {paths.root}"
        )
        results, server_reports = _run_matrix(
            cells, questions, tooling, paths, base_env, args.rerank_url
        )
        result = verdict(results)
        record: dict[str, object] = {
            "label": args.label,
            "venue": args.venue,
            "started": started.isoformat(),
            "finished": datetime.now(UTC).isoformat(),
            "host": describe_host(),
            "repository": ev.describe_source_revision(cwd),
            "binaries": {"kb": str(kb), "kb_mcp": str(kb_mcp)},
            "snapshot": {"source": str(args.source), "root": str(paths.root)},
            "endpoint": base_env.get(ev.ENV_BASE_URL),
            "model": base_env.get(ev.ENV_MODEL),
            "rerank_url": args.rerank_url if "served" in variants else None,
            "rerank_top_k": base_env.get(ENV_RERANK_TOP_K),
            "questions": {"count": len(questions), "path": str(args.questions)},
            "matrix": {
                "variants": list(variants),
                "conditions": sorted({c.condition for c in cells}, key=CONDITIONS.index),
                "clients": sorted({c.clients for c in cells}),
                "passes": args.passes,
                "submissions_per_ingest_pass": args.submissions,
                "search_limit": SEARCH_LIMIT,
            },
            "thresholds": {
                "loaded_p95_over_idle_p95_max": THRESHOLD_LOADED_RATIO,
                "dense_p95_ms_max": THRESHOLD_DENSE_P95_MS,
            },
            "t016_projections": T016_PROJECTIONS,
            "counts_before": counts_before,
            "counts_after": read_counts(paths.index),
            "disk_before": disk_before,
            "disk_after": disk_usage(paths),
            "servers": server_reports,
            "cells": [c.to_json() for c in results],
            "verdict": result.to_json(),
        }
        out = args.out or (cwd / "resources" / "eval" / "runs" / f"{args.label}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(render_report(results, result))
        print(f"bench_serving: record written to {out}")
    except (BenchError, ev.EvalError) as exc:
        print(f"bench_serving: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
