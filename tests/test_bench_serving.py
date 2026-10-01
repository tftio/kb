"""Tests for the serving-matrix harness's pure core.

Everything here runs without a server, an index, an embedding endpoint or a
reranker: the matrix, the statistics, the error classification, the synthetic
submissions, the cell summaries and the threshold verdict are all pure
functions, and the verdict in particular has to be checkable from the record
alone.
"""

from __future__ import annotations

import bench_serving as bs
import pytest

# ── The matrix ─────────────────────────────────────────────────────────────


def test_matrix_runs_variant_outermost_and_pass_innermost():
    cells = bs.matrix(("backend", "served"), ("idle", "ingest"), (1, 4), passes=2)
    assert len(cells) == 2 * 2 * 2 * 2
    keys = [c.key for c in cells]
    assert keys[:4] == [
        "backend/idle/c1/p1",
        "backend/idle/c1/p2",
        "backend/idle/c4/p1",
        "backend/idle/c4/p2",
    ]
    assert keys[-1] == "served/ingest/c4/p2"


def test_matrix_defaults_are_the_plans():
    cells = bs.matrix()
    assert len(cells) == 2 * 3 * 3 * 3
    assert {c.clients for c in cells} == {1, 4, 8}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variants": ()},
        {"conditions": ()},
        {"clients": ()},
        {"passes": 0},
        {"variants": ("postgres",)},
        {"conditions": ("sleeping",)},
        {"clients": (0,)},
    ],
)
def test_matrix_refuses_an_empty_or_unknown_dimension(kwargs):
    with pytest.raises(bs.BenchError):
        bs.matrix(**kwargs)


# ── Statistics ─────────────────────────────────────────────────────────────


def test_percentile_interpolates_between_order_statistics():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert bs.percentile(values, 0) == 10.0
    assert bs.percentile(values, 50) == 30.0
    assert bs.percentile(values, 100) == 50.0
    assert bs.percentile(values, 95) == pytest.approx(48.0)
    assert bs.percentile([7.0], 95) == 7.0


def test_percentile_of_nothing_is_an_error_not_zero():
    with pytest.raises(bs.BenchError):
        bs.percentile([], 50)
    with pytest.raises(bs.BenchError):
        bs.percentile([1.0], 101)


def test_summarize_reports_a_count_and_no_figures_for_an_empty_sample():
    empty = bs.summarize([])
    assert empty.n == 0
    assert empty.p95 is None
    full = bs.summarize([3.0, 1.0, 2.0])
    assert (full.n, full.p50, full.max) == (3, 2.0, 3.0)
    assert full.mean == pytest.approx(2.0)


# ── Errors ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("database is locked", "database"),
        ("SQLITE_BUSY: the index is in use", "database"),
        ("transport: HTTP 500: oops", "transport"),
        ("no node with id abc", "tool"),
        ("keyword-only ranking; endpoint refused", "tool"),
    ],
)
def test_errors_are_classified_by_what_sqlite_says(text, kind):
    assert bs.classify_error(text) == kind


# ── The announced address ─────────────────────────────────────────────────


def test_the_announced_address_is_read_from_the_server_line():
    assert bs.parse_announced_address("kb-mcp: listening on http://127.0.0.1:54321/mcp\n") == (
        "127.0.0.1:54321"
    )


def test_a_line_that_is_not_the_announcement_is_a_fault():
    with pytest.raises(bs.BenchError):
        bs.parse_announced_address("kb-mcp: KB_MCP_TOKEN is unset")


# ── Synthetic submissions ─────────────────────────────────────────────────


def test_a_submission_carries_the_document_under_a_fresh_id():
    document = "* Note\n:PROPERTIES:\n:ID: old-id\n:END:\n\nbody text\n"
    submission = bs.synthesize_submission(document, "new-id")
    assert submission["id"] == "new-id"
    assert submission["corpus"] == "kb"
    assert ":ID: new-id" in submission["document"]
    assert "old-id" not in submission["document"]
    assert "body text" in submission["document"]


def test_a_document_with_no_drawer_is_posted_as_it_is():
    document = "* Note\n\nbody text\n"
    assert bs.synthesize_submission(document, "x")["document"] == document


def test_assembled_documents_carry_the_title_the_id_and_every_passage():
    document = bs.assemble_document("Ownership", ["first body", "", "second body"], "rec-1")
    assert document.startswith("#+title: Ownership\n\n* Ownership\n")
    assert ":ID: rec-1" in document
    assert "first body\n\nsecond body" in document
    assert bs.synthesize_submission(document, "fresh")["document"].count("fresh") == 1


# ── Summaries ─────────────────────────────────────────────────────────────


def _outcome(
    question: str,
    latency: float,
    *,
    ok: bool = True,
    error: str | None = None,
    stages: dict[str, float] | None = None,
    rank: int | None = 1,
) -> bs.QueryOutcome:
    return bs.QueryOutcome(
        question_id=question,
        latency_ms=latency,
        ok=ok,
        error=error,
        error_kind=None if error is None else bs.classify_error(error),
        stages_ms=stages or {},
        rank=rank if ok else None,
        result_count=10 if ok else 0,
    )


def _cell(
    variant: str,
    condition: str,
    clients: int,
    latencies: list[float],
    *,
    dense_p95: float | None = None,
    writer: bs.WriterReport | None = None,
    errors: list[str] | None = None,
    pass_number: int = 1,
) -> bs.CellSummary:
    outcomes = [
        _outcome(
            f"Q{i:03}",
            latency,
            stages={"kb-dense": dense_p95} if dense_p95 is not None else None,
        )
        for i, latency in enumerate(latencies)
    ]
    outcomes += [_outcome(f"E{i}", 1.0, ok=False, error=e) for i, e in enumerate(errors or [])]
    return bs.summarize_cell(
        bs.Cell(variant, condition, clients, pass_number),
        outcomes,
        writer,
        rss_kib_max=1000,
        rss_kib_end=900,
        wall_s=1.0,
    )


def test_a_cell_summary_counts_database_errors_and_recall():
    summary = _cell(
        "backend",
        "idle",
        1,
        [10.0, 20.0],
        errors=["database is locked", "no node with id z"],
    )
    assert summary.queries == 4
    assert summary.database_errors == 1
    assert len(summary.errors) == 2
    # Failed queries count in the latency sample — a fast failure must not
    # make a failing cell look quick — but not in recall.
    assert summary.latency.n == 4
    assert summary.recall_at_5 == 1.0
    as_json = summary.to_json()
    assert as_json["cell"] == "backend/idle/c1/p1"
    assert as_json["database_errors"] == 1
    assert as_json["ranks"] == {"Q000": 1, "Q001": 1}, "failed queries carry no rank"


def test_stage_statistics_cover_only_queries_that_reported_the_stage():
    outcomes = [
        _outcome("Q1", 5.0, stages={"kb-dense": 40.0}),
        _outcome("Q2", 6.0, stages={}),
    ]
    summary = bs.summarize_cell(bs.Cell("backend", "idle", 1, 1), outcomes, None, None, None, 0.1)
    assert summary.stages["kb-dense"].n == 1
    assert summary.stages["kb-dense"].p95 == 40.0


# ── The verdict ───────────────────────────────────────────────────────────


def _done(kind: str = "ingest") -> bs.WriterReport:
    return bs.WriterReport(kind, True, True, 0, 1.0, 1)


def _failed(kind: str = "reindex") -> bs.WriterReport:
    return bs.WriterReport(kind, True, False, 1, 1.0, 1)


def _ratios(result: bs.Verdict) -> dict[str, float]:
    ratios = result.tests["loaded_p95_within_twice_idle"]["ratios"]
    assert isinstance(ratios, dict)
    return {str(k): float(v) for k, v in ratios.items()}


def _passing_matrix() -> list[bs.CellSummary]:
    return [
        _cell("backend", "idle", 1, [100.0, 110.0], dense_p95=40.0),
        _cell("backend", "ingest", 1, [120.0, 150.0], dense_p95=45.0, writer=_done()),
        _cell("backend", "reindex", 1, [130.0, 180.0], dense_p95=50.0, writer=_done("reindex")),
        _cell("served", "idle", 1, [1500.0, 1600.0]),
        _cell("served", "ingest", 1, [1700.0, 1900.0], writer=_done()),
        _cell("served", "reindex", 1, [1600.0, 2000.0], writer=_done("reindex")),
    ]


def test_sqlite_passes_when_every_threshold_holds():
    result = bs.verdict(_passing_matrix())
    assert result.sqlite_passes is True
    tests = result.tests
    assert tests["no_database_errors"]["value"] == 0
    assert tests["writers_completed"] == {"started": 4, "completed": 4, "passed": True}
    assert _ratios(result)["backend/c1"] == pytest.approx(177.5 / 109.5)
    assert tests["dense_p95_below_threshold"]["worst_p95_ms"] == 50.0


def test_a_database_error_fails_the_verdict():
    cells = _passing_matrix()
    cells[1] = _cell("backend", "ingest", 1, [120.0], writer=_done(), errors=["SQLITE_BUSY"])
    result = bs.verdict(cells)
    assert result.sqlite_passes is False
    assert result.tests["no_database_errors"]["passed"] is False


def test_a_writer_that_did_not_complete_fails_the_verdict():
    cells = _passing_matrix()
    cells[2] = _cell("backend", "reindex", 1, [130.0], dense_p95=50.0, writer=_failed())
    result = bs.verdict(cells)
    assert result.sqlite_passes is False
    assert result.tests["writers_completed"]["completed"] == 3


def test_loaded_latency_beyond_twice_idle_fails_the_verdict():
    cells = _passing_matrix()
    cells[4] = _cell("served", "ingest", 1, [4000.0, 4100.0], writer=_done())
    result = bs.verdict(cells)
    assert result.sqlite_passes is False
    ratios = _ratios(result)
    assert ratios["served/c1"] > 2.0
    assert ratios["backend/c1"] < 2.0


def test_a_slow_dense_stage_fails_the_verdict_even_when_latency_is_fine():
    cells = _passing_matrix()
    cells[0] = _cell("backend", "idle", 1, [100.0, 110.0], dense_p95=350.0)
    result = bs.verdict(cells)
    assert result.sqlite_passes is False
    assert result.tests["dense_p95_below_threshold"]["worst_p95_ms"] == 350.0


def test_the_dense_threshold_reads_the_backend_variant_only():
    # A served cell's dense figure, inflated by whatever else the reranker's
    # host was doing, must not decide the backend test.
    cells = _passing_matrix()
    cells[3] = _cell("served", "idle", 1, [1500.0], dense_p95=900.0)
    assert bs.verdict(cells).tests["dense_p95_below_threshold"]["worst_p95_ms"] == 50.0


def test_a_verdict_with_no_measurements_does_not_pass():
    result = bs.verdict([])
    assert result.sqlite_passes is False


# ── The report and the environment ────────────────────────────────────────


def test_the_report_names_every_cell_and_the_verdict():
    cells = _passing_matrix()
    text = bs.render_report(cells, bs.verdict(cells))
    assert "backend/idle/c1/p1" in text
    assert "served/reindex/c1/p1" in text
    assert "SQLite passes: True" in text
    assert "dense_p95_below_threshold" in text


def test_the_backend_variant_runs_without_a_reranker_and_the_served_one_with():
    base = {
        "KB_RERANK_BASE_URL": "http://old",
        "KB_INDEX_PATH": "/elsewhere",
        "KB_EMBEDDING_MODEL": "m",
    }
    backend = bs.variant_env(base, "backend", "http://reranker")
    served = bs.variant_env(base, "served", "http://reranker")
    assert "KB_RERANK_BASE_URL" not in backend
    assert served["KB_RERANK_BASE_URL"] == "http://reranker"
    for env in (backend, served):
        assert "KB_INDEX_PATH" not in env
        assert env["KB_EMBEDDING_MODEL"] == "m"
    assert base["KB_RERANK_BASE_URL"] == "http://old", "the base environment was mutated"
