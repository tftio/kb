"""Tests for the pure half of query rewriting.

The measured defect is a register mismatch: an authored biography node is
absent from the top 100 for "what is my background and what do I work on" and
ranks first at cosine 0.563 for "<author> biography core facts". These tests cover the
deterministic template that attacks that mismatch without a model, so the
strategy can be reasoned about rather than only observed.
"""

from __future__ import annotations

import pytest
import query_rewrite as qr

# ── The template ───────────────────────────────────────────────────────────


def test_the_interrogative_opening_is_removed():
    assert qr.rewrite_template("what did we decide about the store?") == "op decide about the store"


def test_first_person_becomes_the_operator():
    assert qr.rewrite_template("what is my background") == "op's background"
    assert qr.rewrite_template("how do I run the harness") == "op run the harness"


def test_the_motivating_question_becomes_the_register_that_retrieves_it():
    """The one case the whole strategy is extrapolated from."""
    rewritten = qr.rewrite_template("what is my background and what do I work on")
    assert "op" in rewritten
    assert "what" not in rewritten
    assert "?" not in rewritten


def test_a_question_already_in_descriptive_register_is_left_alone():
    """A statement is not a question; rewriting it would be damage."""
    assert qr.rewrite_template("storage invariants for the content-addressed store") == (
        "storage invariants for the content-addressed store"
    )


def test_rewriting_is_deterministic():
    question = "why did we choose git over a hand-rolled store?"
    assert qr.rewrite_template(question) == qr.rewrite_template(question)


def test_whitespace_is_collapsed():
    assert qr.rewrite_template("what   is  my   background") == "op's background"


def test_an_empty_question_survives_rewriting():
    """A degenerate input must not become something that searches for nothing."""
    assert qr.rewrite_template("") == ""
    assert qr.rewrite_template("?") == ""


# ── Strategy dispatch ──────────────────────────────────────────────────────


def test_the_none_strategy_is_the_identity():
    """The control. If this ever altered a query the comparison would be void."""
    assert qr.STRATEGIES["none"].needs_model is False
    assert qr.rewrite("what is my background", "none", client=None) == "what is my background"


def test_the_template_strategy_needs_no_model():
    assert qr.STRATEGIES["template"].needs_model is False
    assert qr.rewrite("what is my background", "template", client=None) == "op's background"


def test_a_model_strategy_without_a_client_is_refused():
    """A silent fallback would report the control's numbers under another name."""
    with pytest.raises(qr.RewriteError, match="paraphrase"):
        qr.rewrite("what is my background", "paraphrase", client=None)


def test_an_unknown_strategy_is_refused():
    with pytest.raises(qr.RewriteError, match="nonsense"):
        qr.rewrite("q", "nonsense", client=None)


# ── Model-backed strategies, against a stub client ─────────────────────────


class _StubClient:
    """A client that records its prompts and returns canned completions."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_a_paraphrase_uses_the_model_output():
    client = _StubClient("  op's professional background and current work.  ")
    assert qr.rewrite("what is my background", "paraphrase", client=client) == (
        "op's professional background and current work."
    )
    assert "what is my background" in client.prompts[0]


def test_a_hyde_answer_uses_the_model_output():
    client = _StubClient("op works on knowledge retrieval systems in Rust.")
    assert qr.rewrite("what do I work on", "hyde", client=client) == (
        "op works on knowledge retrieval systems in Rust."
    )


def test_a_model_that_answers_with_nothing_is_refused():
    """An empty rewrite would score zero and read as a failed strategy."""
    client = _StubClient("   ")
    with pytest.raises(qr.RewriteError, match="empty"):
        qr.rewrite("what is my background", "paraphrase", client=client)


def test_a_model_that_wraps_its_answer_in_quotes_is_unwrapped():
    """Quoted output would otherwise embed the quotation marks as content."""
    client = _StubClient('"op\'s background"')
    assert qr.rewrite("q", "paraphrase", client=client) == "op's background"


def test_a_model_that_prefixes_a_label_is_unwrapped():
    client = _StubClient("Rewritten query: op's background")
    assert qr.rewrite("q", "paraphrase", client=client) == "op's background"


def test_reasoning_blocks_are_stripped():
    """A reasoning block would otherwise be embedded as though it were the query."""
    client = _StubClient("<think>The user wants...</think>\nop's background")
    assert qr.rewrite("q", "paraphrase", client=client) == "op's background"


def test_quotation_marks_are_stripped_from_a_generated_passage():
    """Quotes are formatting, and an unbalanced one breaks kb's escaping."""
    client = _StubClient('JFB uses a "Stratocaster" and an amp')
    assert qr.rewrite("q", "hyde", client=client) == "JFB uses a Stratocaster and an amp"


def test_a_degenerate_repetition_loop_is_refused():
    """Searching a model's loop would score its failure as the strategy's."""
    loop = "Need plausible. " + 'no. "Boss Catalinbread" ' * 40
    client = _StubClient(loop)
    with pytest.raises(qr.RewriteError, match="repetit"):
        qr.rewrite("what music do I listen to", "hyde", client=client)


def test_ordinary_repetition_is_not_mistaken_for_a_loop():
    """A passage may legitimately repeat a subject's name several times."""
    client = _StubClient("op writes in org-mode. op stores notes in kb. op searches them daily.")
    assert "org-mode" in qr.rewrite("q", "hyde", client=client)
