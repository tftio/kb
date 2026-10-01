"""Tests for the retrieval harness's pure scoring core.

The harness is made importable by `conftest.py`, which puts `scripts/` on the
path. Every test here runs against fixture data with no database, no embedding
endpoint, and no `kb` binary — which is the point of keeping the scoring
separate from the subprocess shell.
"""

from __future__ import annotations

import json
import sqlite3

import eval_retrieval as ev
import pytest


def _hit(node_id: str, title: str = "t", similarity: float | None = None) -> ev.SearchHit:
    """Build a SearchHit with defaults for the fields not under test."""
    return ev.SearchHit(node_id=node_id, title=title, similarity=similarity)


def _question(
    *expect: ev.Expectation,
    qid: str = "Q001",
    population: str = "authored",
) -> ev.Question:
    """Build a Question with defaults for the fields not under test."""
    return ev.Question(id=qid, query="q", population=population, expect=expect)


# ── Matching ───────────────────────────────────────────────────────────────


def test_node_id_matches_by_equality():
    expectation = ev.Expectation(node_id="a", title="recorded title")
    assert ev.expectation_matches(expectation, _hit("a"))
    assert not ev.expectation_matches(expectation, _hit("b"))


def test_recorded_title_is_never_matched_on():
    # `title` records drift, it does not select. A hit whose title matches but
    # whose id does not is not a hit.
    expectation = ev.Expectation(node_id="a", title="Project overview")
    assert not ev.expectation_matches(expectation, _hit("b", title="Project overview"))


def test_title_regex_matches_a_truncated_title():
    # Stored titles are truncated at 80 characters, so a pattern anchored at the
    # start must still match when the tail is gone.
    expectation = ev.Expectation(title_regex="^ADR-006: Replace the session cache")
    truncated = "ADR-006: Replace the session cache with a write-through store for every tenant C"
    assert ev.expectation_matches(expectation, _hit("x", title=truncated))


def test_either_expectation_form_suffices():
    expectation = ev.Expectation(node_id="a", title_regex="^nothing")
    assert ev.expectation_matches(expectation, _hit("a", title="unrelated"))
    assert ev.expectation_matches(expectation, _hit("z", title="nothing here"))


# ── Rank extraction ────────────────────────────────────────────────────────


def test_rank_is_one_indexed_and_takes_the_best_ranked_expectation():
    question = _question(ev.Expectation(node_id="c"), ev.Expectation(node_id="b"))
    hits = [_hit("a"), _hit("b"), _hit("c")]
    assert ev.first_expected_rank(question, hits) == 2


def test_rank_is_none_when_no_expectation_appears():
    question = _question(ev.Expectation(node_id="z"))
    assert ev.first_expected_rank(question, [_hit("a"), _hit("b")]) is None


def test_rank_is_none_for_empty_results():
    assert ev.first_expected_rank(_question(ev.Expectation(node_id="a")), []) is None


# ── Vector candidate count ─────────────────────────────────────────────────


def test_candidate_count_counts_only_non_null_similarities():
    hits = [_hit("a", similarity=0.7), _hit("b"), _hit("c", similarity=0.41)]
    assert ev.vector_candidate_count(hits) == 2


def test_a_zero_similarity_still_counts_as_a_candidate():
    # A null similarity means no vector score was computed; 0.0 means one was.
    # Conflating them would understate the candidate count for the exact
    # configuration this harness exists to detect.
    assert ev.vector_candidate_count([_hit("a", similarity=0.0)]) == 1


def test_candidate_count_is_zero_when_every_hit_is_keyword_only():
    assert ev.vector_candidate_count([_hit("a"), _hit("b")]) == 0


def test_max_similarity_ignores_nulls_and_is_none_when_all_are_null():
    mixed = [_hit("a", similarity=0.4), _hit("b"), _hit("c", similarity=0.6)]
    assert ev.max_similarity(mixed) == 0.6
    assert ev.max_similarity([_hit("a"), _hit("b")]) is None


# ── Per-question scoring ───────────────────────────────────────────────────


def test_score_question_reports_the_matched_hit_not_the_first_hit():
    question = _question(ev.Expectation(node_id="b"))
    hits = [_hit("a", similarity=0.9), _hit("b", similarity=0.5)]
    result = ev.score_question(question, hits, limit=100)
    assert result.rank == 2
    assert result.hit_node_id == "b"
    assert result.hit_similarity == 0.5
    assert result.max_similarity == 0.9


def test_score_question_marks_saturation_at_the_limit():
    hits = [_hit(str(i), similarity=0.5) for i in range(100)]
    result = ev.score_question(_question(ev.Expectation(node_id="0")), hits, limit=100)
    assert result.candidate_count == 100
    assert result.candidates_saturated


def test_score_question_below_the_limit_is_not_saturated():
    hits = [_hit(str(i), similarity=0.5) for i in range(99)]
    result = ev.score_question(_question(ev.Expectation(node_id="0")), hits, limit=100)
    assert not result.candidates_saturated


def test_a_missed_question_carries_no_rank_or_similarity():
    result = ev.score_question(_question(ev.Expectation(node_id="z")), [_hit("a")], limit=100)
    assert result.rank is None
    assert result.hit_similarity is None
    assert result.hit_node_id is None
    assert result.reciprocal_rank == 0.0
    assert not result.recall_at(10)


# ── Recall and MRR ─────────────────────────────────────────────────────────


def _result(
    qid: str,
    rank: int | None,
    *,
    population: str = "p",
    candidates: int = 10,
    corpus: str = "kb",
):
    """Build a QuestionResult with only the fields the aggregate consumes."""
    return ev.QuestionResult(
        question_id=qid,
        population=population,
        corpus=corpus,
        query="q",
        issued_query=None,
        rank=rank,
        hit_similarity=None if rank is None else 0.5,
        hit_node_id=None if rank is None else "n",
        max_similarity=0.6,
        candidate_count=candidates,
        candidates_saturated=False,
        result_count=20,
    )


@pytest.mark.parametrize(
    ("rank", "at_5", "at_10"),
    [(1, True, True), (5, True, True), (6, False, True), (10, False, True), (11, False, False)],
)
def test_recall_at_k_is_inclusive_of_k(rank: int, at_5: bool, at_10: bool):
    result = _result("Q", rank)
    assert result.recall_at(5) is at_5
    assert result.recall_at(10) is at_10


def test_reciprocal_rank_is_the_reciprocal_of_the_rank():
    assert _result("Q", 1).reciprocal_rank == 1.0
    assert _result("Q", 4).reciprocal_rank == 0.25


def test_aggregate_computes_recall_mrr_and_candidate_statistics():
    results = [
        _result("Q1", 1, candidates=2),
        _result("Q2", 6, candidates=14),
        _result("Q3", None, candidates=0),
        _result("Q4", 2, candidates=8),
    ]
    a = ev.aggregate(results)
    assert a.question_count == 4
    assert a.recall_at_5 == 0.5
    assert a.recall_at_10 == 0.75
    assert a.mrr == pytest.approx((1.0 + 1 / 6 + 0.0 + 0.5) / 4)
    assert a.mean_candidate_count == 6.0
    assert a.min_candidate_count == 0
    assert a.max_candidate_count == 14
    assert a.misses == ("Q3",)


def test_a_miss_contributes_zero_to_mrr_rather_than_being_dropped():
    both = ev.aggregate([_result("Q1", 1), _result("Q2", None)])
    assert both.mrr == 0.5


def test_mean_hit_similarity_skips_keyword_only_hits_rather_than_scoring_them_zero():
    keyword_only = ev.QuestionResult(
        question_id="Q2",
        population="p",
        corpus="kb",
        query="q",
        issued_query=None,
        rank=3,
        hit_similarity=None,
        hit_node_id="n",
        max_similarity=None,
        candidate_count=0,
        candidates_saturated=False,
        result_count=5,
    )
    a = ev.aggregate([_result("Q1", 1), keyword_only])
    assert a.mean_hit_similarity == 0.5


def test_aggregate_refuses_an_empty_result_set():
    with pytest.raises(ev.EvalError, match="empty result set"):
        ev.aggregate([])


def test_aggregate_by_population_separates_the_populations():
    results = [
        _result("Q1", 1, population="authored"),
        _result("Q2", None, population="authored"),
        _result("Q3", 1, population="lobsters"),
    ]
    by_population = ev.aggregate_by_population(results)
    assert set(by_population) == {"authored", "lobsters"}
    assert by_population["authored"].recall_at_5 == 0.5
    assert by_population["lobsters"].recall_at_5 == 1.0


# ── Payload parsing ────────────────────────────────────────────────────────


def test_parse_search_payload_preserves_rank_order_and_null_similarity():
    raw = json.dumps(
        {
            "command": "search",
            "ok": True,
            "data": [
                {"id": "a", "title": "A", "similarity": 0.51},
                {"id": "b", "title": "B", "similarity": None},
            ],
        }
    )
    hits = ev.parse_search_payload(raw)
    assert [h.node_id for h in hits] == ["a", "b"]
    assert hits[0].similarity == 0.51
    assert hits[1].similarity is None


def test_parse_search_payload_accepts_an_empty_result_set():
    assert ev.parse_search_payload(json.dumps({"ok": True, "data": []})) == ()


def test_parse_search_payload_rejects_non_json():
    with pytest.raises(ev.EvalError, match="did not emit JSON"):
        ev.parse_search_payload("kb: no such command")


def test_parse_search_payload_rejects_a_failure_envelope():
    with pytest.raises(ev.EvalError, match="reported failure"):
        ev.parse_search_payload(json.dumps({"ok": False, "error": "boom"}))


def test_parse_tool_payload_reads_the_tools_results_in_rank_order():
    raw = json.dumps(
        {
            "count": 2,
            "results": [
                {"id": "a", "title": "A", "similarity": 0.51, "corpus": "kb"},
                {"id": "b", "title": "B", "similarity": None, "corpus": "kb"},
            ],
        }
    )
    hits = ev.parse_tool_payload(raw)
    assert [h.node_id for h in hits] == ["a", "b"]
    assert hits[0].similarity == 0.51
    assert hits[1].similarity is None


def test_parse_tool_payload_carries_the_corpus_each_hit_came_from():
    raw = json.dumps(
        {
            "count": 1,
            "results": [{"id": "<m@x>", "title": "Subject", "similarity": 0.4, "corpus": "mail"}],
        }
    )
    assert ev.parse_tool_payload(raw)[0].corpus == "mail"


def test_parse_tool_payload_refuses_a_degraded_ranking():
    """The tool's `degraded` array is the CLI's stderr note, and is as fatal.

    A run recorded while the reranker was unreachable is not a measurement of
    the ranking; it is a measurement of the outage.
    """
    raw = json.dumps(
        {
            "count": 1,
            "results": [{"id": "a", "title": "A", "similarity": None, "corpus": "kb"}],
            "degraded": ["fused ranking; connection refused"],
        }
    )
    with pytest.raises(ev.EvalError, match="degraded"):
        ev.parse_tool_payload(raw)


def test_parse_tool_payload_rejects_a_payload_without_results():
    with pytest.raises(ev.EvalError, match="no results array"):
        ev.parse_tool_payload(json.dumps({"count": 0}))


def test_parse_search_payload_rejects_a_malformed_entry():
    with pytest.raises(ev.EvalError, match="no string id"):
        ev.parse_search_payload(json.dumps({"ok": True, "data": [{"title": "A"}]}))


# ── Question-set parsing ───────────────────────────────────────────────────


def _document(**overrides: object) -> dict[str, object]:
    """A minimal valid question-set document, with overrides applied."""
    return {
        "schema_version": 1,
        "corpus": {"node_count": 1},
        "question": [
            {
                "id": "Q001",
                "query": "a question",
                "population": "authored",
                "expect": [{"node_id": "a", "title": "A"}],
            }
        ],
    } | overrides


def test_parse_question_set_reads_a_well_formed_document():
    parsed = ev.parse_question_set(_document())
    assert parsed.schema_version == 1
    assert len(parsed.questions) == 1
    assert parsed.questions[0].expect[0].node_id == "a"


def test_parse_question_set_rejects_an_unknown_schema_version():
    with pytest.raises(ev.EvalError, match="unsupported schema_version"):
        ev.parse_question_set(_document(schema_version=2))


def test_parse_question_set_rejects_duplicate_question_ids():
    question = {
        "id": "Q001",
        "query": "q",
        "population": "p",
        "expect": [{"node_id": "a"}],
    }
    with pytest.raises(ev.EvalError, match="duplicate question ids: Q001"):
        ev.parse_question_set(_document(question=[question, dict(question)]))


def test_parse_question_set_rejects_a_question_with_no_expectations():
    with pytest.raises(ev.EvalError, match="expect must be non-empty"):
        ev.parse_question_set(
            _document(question=[{"id": "Q1", "query": "q", "population": "p", "expect": []}])
        )


def test_parse_question_set_rejects_an_expectation_that_could_never_match():
    with pytest.raises(ev.EvalError, match="neither node_id nor title_regex"):
        ev.parse_question_set(
            _document(
                question=[{"id": "Q1", "query": "q", "population": "p", "expect": [{"title": "A"}]}]
            )
        )


def test_parse_question_set_rejects_an_invalid_title_regex():
    with pytest.raises(ev.EvalError, match="not a valid regex"):
        ev.parse_question_set(
            _document(
                question=[
                    {
                        "id": "Q1",
                        "query": "q",
                        "population": "p",
                        "expect": [{"title_regex": "([unclosed"}],
                    }
                ]
            )
        )


def test_parse_question_set_rejects_a_blank_query():
    with pytest.raises(ev.EvalError, match="query must be non-empty"):
        ev.parse_question_set(
            _document(
                question=[
                    {"id": "Q1", "query": "   ", "population": "p", "expect": [{"node_id": "a"}]}
                ]
            )
        )


def test_parse_question_set_rejects_a_document_with_no_questions():
    with pytest.raises(ev.EvalError, match="no \\[\\[question\\]\\] entries"):
        ev.parse_question_set(_document(question=[]))


# ── The committed question set ─────────────────────────────────────────────


def test_the_committed_question_set_parses_and_meets_its_own_acceptance_bar():
    # T001's acceptance checks, as a regression test: the file that every later
    # measurement in this plan is keyed to must stay parseable and stay sampled
    # across populations. Node-id resolution is not checked here — that needs a
    # live corpus and is enforced by the harness at run time.
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    path = root / "resources/eval/retrieval-questions.toml"
    import tomllib

    parsed = ev.parse_question_set(tomllib.loads(path.read_text(encoding="utf-8")))
    assert len(parsed.questions) >= 30
    authored = [q for q in parsed.questions if q.population == "authored"]
    assert len(authored) >= 5
    assert len({q.population for q in parsed.questions}) >= 3


# ── Record assembly ────────────────────────────────────────────────────────


def test_build_record_is_json_serializable_and_names_the_environment():
    # Round-tripped through JSON rather than inspected directly, because being
    # serializable is the property that matters: a record that cannot be written
    # cannot be compared against by a later run.
    environment: dict[str, object] = {
        "model": "m",
        "endpoint": "http://e/v1",
        "corpus": {"nodes": 3886},
    }
    built = ev.build_record("baseline", environment, 100, [_result("Q1", 1)])
    record = json.loads(json.dumps(built))
    assert record["label"] == "baseline"
    assert record["limit"] == 100
    assert record["environment"]["model"] == "m"
    assert record["environment"]["corpus"]["nodes"] == 3886
    assert record["aggregate"]["recall_at_5"] == 1.0
    assert record["by_population"]["p"]["question_count"] == 1
    assert record["questions"][0]["id"] == "Q1"


def test_render_table_flags_saturated_counts():
    saturated = ev.QuestionResult(
        question_id="Q1",
        population="authored",
        corpus="kb",
        query="q",
        issued_query=None,
        rank=1,
        hit_similarity=0.5,
        hit_node_id="n",
        max_similarity=0.5,
        candidate_count=100,
        candidates_saturated=True,
        result_count=100,
    )
    table = ev.render_table([saturated], limit=100)
    assert "100*" in table
    assert "saturated" in table


def test_render_summary_lists_misses():
    summary = ev.render_summary(
        ev.aggregate([_result("Q1", 1), _result("Q2", None)]),
        ev.aggregate_by_population([_result("Q1", 1), _result("Q2", None)]),
    )
    assert "missed entirely: Q2" in summary


# ── T013: addressing records by corpus ─────────────────────────────────────


def test_a_bare_node_id_means_the_kb_corpus():
    """Moving this default would invalidate all 42 questions and every run."""
    expectation = ev.Expectation(node_id="a")

    assert expectation.corpus == ev.DEFAULT_CORPUS == "kb"


def test_a_hit_without_a_corpus_is_assumed_to_be_kb():
    """Until T012 makes search report one, every hit is a kb hit."""
    assert ev.SearchHit(node_id="a", title="t", similarity=None).corpus == "kb"


def test_matching_requires_the_corpus_to_agree():
    """Matching on the id alone would score a mail hit as a kb answer."""
    mail_expectation = ev.Expectation(node_id="x", corpus="mail")

    assert not ev.expectation_matches(mail_expectation, _hit("x")), (
        "a mail expectation must not be satisfied by a kb hit with the same id"
    )


def test_a_kb_expectation_does_not_match_a_mail_hit():
    expectation = ev.Expectation(node_id="x")
    mail_hit = ev.SearchHit(node_id="x", title="t", similarity=None, corpus="mail")

    assert not ev.expectation_matches(expectation, mail_hit)


def test_a_mail_expectation_matches_a_mail_hit():
    expectation = ev.Expectation(node_id="<m1@example.com>", corpus="mail")
    mail_hit = ev.SearchHit(node_id="<m1@example.com>", title="t", similarity=None, corpus="mail")

    assert ev.expectation_matches(expectation, mail_hit)


def test_a_title_regex_still_matches_within_a_corpus():
    """The regex escape hatch survives re-import, but not across corpora."""
    expectation = ev.Expectation(title_regex="^Voice", corpus="mail")

    assert not ev.expectation_matches(expectation, _hit("a", title="Voice Profile"))
    assert ev.expectation_matches(
        expectation,
        ev.SearchHit(node_id="a", title="Voice Profile", similarity=None, corpus="mail"),
    )


def test_an_unknown_corpus_is_refused_at_the_boundary():
    """A typo must fail rather than name a corpus nothing indexes."""
    with pytest.raises(ev.EvalError, match="corpus"):
        ev.parse_question(
            {
                "id": "Q1",
                "query": "q",
                "population": "authored",
                "expect": [{"node_id": "a", "corpus": "maail"}],
            }
        )


def test_a_question_takes_the_corpus_of_its_expectations():
    question = _question(ev.Expectation(node_id="a", corpus="mail"))

    assert ev.question_corpus(question) == "mail"


def test_a_question_expecting_two_corpora_is_reported_as_mixed():
    """Folding a cross-corpus question into one corpus misattributes it."""
    question = _question(
        ev.Expectation(node_id="a"),
        ev.Expectation(node_id="<m@x>", corpus="mail"),
    )

    assert ev.question_corpus(question) == "mixed"


def test_aggregates_are_reported_per_corpus():
    results = [
        _result("Q1", 1, corpus="kb"),
        _result("Q2", None, corpus="mail"),
    ]

    by_corpus = ev.aggregate_by_corpus(results)

    assert sorted(by_corpus) == ["kb", "mail"]
    assert by_corpus["kb"].recall_at_5 == 1.0
    assert by_corpus["mail"].recall_at_5 == 0.0


def test_the_summary_breaks_down_per_corpus_once_there_is_more_than_one():
    """A single-corpus table repeats ALL verbatim, so it is suppressed."""
    kb_only = [_result("Q1", 1, corpus="kb")]
    mixed = [_result("Q1", 1, corpus="kb"), _result("Q2", None, corpus="mail")]

    single = ev.render_summary(
        ev.aggregate(kb_only), ev.aggregate_by_population(kb_only), ev.aggregate_by_corpus(kb_only)
    )
    both = ev.render_summary(
        ev.aggregate(mixed), ev.aggregate_by_population(mixed), ev.aggregate_by_corpus(mixed)
    )

    assert "corpus" not in single
    assert "corpus" in both
    assert "mail" in both


def test_only_ingested_corpora_are_checked_for_staleness():
    """Ids from a corpus kb has not ingested cannot resolve, so are not asked."""
    mail_q = _question(ev.Expectation(node_id="<m@x>", corpus="mail"), qid="MQ1")
    kb_q = _question(ev.Expectation(node_id="n1"), qid="Q1")

    assert ev.ids_to_verify([kb_q, mail_q], corpora=("kb",)) == [("Q1", "n1")]


def test_selecting_a_corpus_filters_the_questions_run():
    """Running an unindexed corpus records guaranteed misses as measurement."""
    mail_q = _question(ev.Expectation(node_id="<m@x>", corpus="mail"), qid="MQ1")
    kb_q = _question(ev.Expectation(node_id="n1"), qid="Q1")

    assert [q.id for q in ev.select_corpora([kb_q, mail_q], None)] == ["Q1", "MQ1"]
    assert [q.id for q in ev.select_corpora([kb_q, mail_q], ("kb",))] == ["Q1"]
    assert [q.id for q in ev.select_corpora([kb_q, mail_q], ("mail",))] == ["MQ1"]


# ── The index the corpus counts are read from ──────────────────────────────


def _index(path, *, records, spans, model="m") -> None:
    """Write a fixture index holding `records` records and `spans` vectors.

    Only the three tables the harness reads are created. The shape is the
    index's, not the retired database's: vectors are keyed by content, so the
    corpus a vector belongs to is reachable only through the passage whose
    span contains it.
    """
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        create table records (record_id text primary key, corpus text not null);
        create table passages (
            passage_id integer primary key, record_id text not null,
            stream_hash text not null, span_start integer not null,
            span_len integer not null
        );
        create table embeddings (
            stream_hash text not null, span_start integer not null,
            span_len integer not null, model text not null, vector blob not null,
            primary key (stream_hash, span_start, span_len, model)
        );
        """
    )
    for record, corpus in records:
        connection.execute("insert into records values (?, ?)", (record, corpus))
        connection.execute(
            "insert into passages (record_id, stream_hash, span_start, span_len) values (?,?,?,?)",
            (record, f"h-{record}", 0, 1000),
        )
    for record, start, length in spans:
        connection.execute(
            "insert into embeddings values (?,?,?,?,?)",
            (f"h-{record}", start, length, model, b"\x00"),
        )
    connection.commit()
    connection.close()


def test_corpus_counts_are_read_from_the_index_for_one_corpus(tmp_path):
    # The mail record and its vector must not be counted into the kb figures:
    # a run's environment block is what a later comparison uses to decide two
    # runs measured the same corpus.
    path = tmp_path / "index.db"
    _index(
        path,
        records=[("a", "kb"), ("b", "kb"), ("m1", "mail")],
        spans=[("a", 0, 500), ("a", 500, 500), ("m1", 0, 500)],
    )
    assert ev.read_corpus_counts(path, "kb") == {
        "records": 2,
        "embedded_records": 1,
        "chunk_embeddings": 2,
    }


def test_corpus_counts_see_writes_a_concurrent_process_has_committed(tmp_path):
    # Read-only, not immutable: `immutable=1` tells SQLite to ignore the
    # write-ahead log, which on a live index silently reports the corpus as it
    # was before the last embedding run committed.
    path = tmp_path / "index.db"
    _index(path, records=[("a", "kb")], spans=[])
    connection = sqlite3.connect(path)
    connection.execute("pragma journal_mode=wal")
    connection.execute("insert into records values ('b', 'kb')")
    connection.commit()
    try:
        assert ev.read_corpus_counts(path, "kb")["records"] == 2
    finally:
        connection.close()


def test_an_unreadable_index_is_an_environmental_fault(tmp_path):
    path = tmp_path / "index.db"
    path.write_text("not a database")
    with pytest.raises(ev.EvalError, match="corpus counts"):
        ev.read_corpus_counts(path, "kb")


def test_the_index_path_comes_from_the_environment(tmp_path):
    path = tmp_path / "index.db"
    path.write_text("")
    assert ev.resolve_index_path({ev.ENV_INDEX_PATH: str(path)}) == path


def test_a_missing_index_is_refused_before_any_question_runs(tmp_path):
    absent = tmp_path / "gone.db"
    with pytest.raises(ev.EvalError, match="index not found"):
        ev.resolve_index_path({ev.ENV_INDEX_PATH: str(absent)})


def test_the_measured_corpora_are_the_ones_the_expectations_name():
    # Not `question_corpus`, which collapses a cross-corpus question to
    # "mixed": the environment block has to record the size of every corpus the
    # run actually searched, and "mixed" is not a corpus anything is stored in.
    kb_only = _question(ev.Expectation(node_id="a", corpus="kb"))
    crossing = _question(
        ev.Expectation(node_id="b", corpus="kb"),
        ev.Expectation(node_id="c", corpus="mail"),
        qid="Q002",
    )
    assert ev.measured_corpora([kb_only]) == ("kb",)
    assert ev.measured_corpora([kb_only, crossing]) == ("kb", "mail")


def test_t031_measurement_controls_default_to_the_decided_shipped_ranking():
    args = ev.build_parser().parse_args([])
    assert args.dense_pooling == "mean-top-three"
    assert args.vector_candidates == ev.VECTOR_CANDIDATES


def test_t031_measurement_controls_accept_each_experimental_arm():
    parser = ev.build_parser()
    assert parser.parse_args(["--dense-pooling", "mean-top-three"]).dense_pooling == (
        "mean-top-three"
    )
    normalized = parser.parse_args(
        ["--dense-pooling", "length-normalized-max", "--vector-candidates", "400"]
    )
    assert normalized.dense_pooling == "length-normalized-max"
    assert normalized.vector_candidates == 400
