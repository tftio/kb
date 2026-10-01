"""Tests for the pure half of cross-encoder reranking.

A reranker can only reorder what retrieval surfaced, so the arithmetic of
*which* candidates enter the window, and what happens to the ones outside it,
decides what the measurement means. That is what these cover; the model itself
is exercised against the live endpoint, not here.
"""

from __future__ import annotations

import pytest
import rerank as rr


def _hits(*ids: str) -> list[str]:
    return list(ids)


def test_the_window_is_reordered_by_score():
    reordered = rr.reorder(_hits("a", "b", "c"), {"a": 0.1, "b": 0.9, "c": 0.5}, top_k=3)
    assert reordered == ["b", "c", "a"]


def test_candidates_outside_the_window_keep_their_order_and_stay_below():
    """The reranker never saw them, so it can have no opinion about them."""
    reordered = rr.reorder(_hits("a", "b", "c", "d"), {"a": 0.1, "b": 0.9}, top_k=2)
    assert reordered == ["b", "a", "c", "d"]


def test_a_shorter_candidate_list_than_the_window_is_fine():
    assert rr.reorder(_hits("a"), {"a": 0.4}, top_k=20) == ["a"]


def test_an_empty_candidate_list_reranks_to_nothing():
    assert rr.reorder([], {}, top_k=20) == []


def test_ties_preserve_the_retrieval_order():
    """An unseparated pair leaves the fused ranking's judgement standing."""
    assert rr.reorder(_hits("a", "b", "c"), {"a": 0.5, "b": 0.5, "c": 0.5}, top_k=3) == [
        "a",
        "b",
        "c",
    ]


def test_a_missing_score_is_refused():
    """Scoring fewer documents than were sent would silently drop candidates."""
    with pytest.raises(rr.RerankError, match="score"):
        rr.reorder(_hits("a", "b"), {"a": 0.5}, top_k=2)


# ── Whether reranking could possibly have helped ───────────────────────────


def test_an_answer_inside_the_window_is_reachable():
    assert rr.window_contains_answer(rank=3, top_k=20) is True
    assert rr.window_contains_answer(rank=20, top_k=20) is True


def test_an_answer_below_the_window_is_out_of_reach():
    """A loss here is the retriever's miss, not the reranker's failure."""
    assert rr.window_contains_answer(rank=21, top_k=20) is False
    assert rr.window_contains_answer(rank=None, top_k=20) is False


# ── What the reranker is shown ─────────────────────────────────────────────


def test_the_metadata_drawer_is_not_shown_to_the_reranker():
    """An identical drawer on every record distinguishes no candidate."""
    text = (
        ":PROPERTIES:\n:ID: a471a048\n:CREATED: 2026-08-12T22:22:30.153Z\n"
        ":UPDATED: 2026-08-12T22:22:30.153Z\n:END:\n* The heading\n\nThe body.\n"
    )
    assert rr.readable_text(text) == "* The heading\n\nThe body.\n"


def test_a_document_without_a_drawer_is_left_alone():
    assert rr.readable_text("* Heading\n\nBody.\n") == "* Heading\n\nBody.\n"


def test_an_unterminated_drawer_is_left_alone():
    """Dropping everything after a malformed opener would blank it."""
    text = ":PROPERTIES:\n:ID: x\n* Heading\n\nBody.\n"
    assert rr.readable_text(text) == text
