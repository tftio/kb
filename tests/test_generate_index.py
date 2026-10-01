"""Tests for the pure core of the generated-index prototype.

The prototype's product is a number, not code that survives — T010 is where
an index-generation strategy is actually built. What is worth testing here is
the part whose silent failure would corrupt that number: which nodes are
generated for, and how generated text becomes retrievable rows.
"""

from __future__ import annotations

import generate_index as gi
import pytest


def test_only_nodes_the_question_set_names_are_selected():
    """The ceiling is measurable on the ground-truth nodes alone."""
    questions = {
        "question": [
            {"id": "Q1", "expect": [{"node_id": "a"}, {"node_id": "b"}]},
            {"id": "Q2", "expect": [{"node_id": "a"}]},
            {"id": "Q3", "expect": [{"title": "no id here"}]},
        ]
    }
    assert gi.expected_node_ids(questions) == ["a", "b"]


def test_a_question_set_naming_nothing_selects_nothing():
    assert gi.expected_node_ids({"question": []}) == []


def test_the_committed_question_set_selects_its_nodes():
    """Against the real file, not a fixture of the shape I assumed it had.

    The first version of this selector read a `questions` key that the TOML
    does not have. Every unit test passed, generation silently produced
    nothing, and the run would have reported the control's numbers as the
    prototype's. A fixture cannot catch that; only the real file can.
    """
    import tomllib
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "resources/eval/retrieval-questions.toml"
    with path.open("rb") as handle:
        question_set = tomllib.load(handle)

    ids = gi.expected_node_ids(question_set)

    assert len(ids) == 29, f"expected the 29 kb-corpus ground-truth nodes, got {len(ids)}"
    assert all(not i.startswith("<") for i in ids), "a mail Message-ID leaked into the kb set"
    assert all(isinstance(i, str) and i for i in ids)


def test_the_summary_form_yields_one_row():
    assert gi.split_generated("A single descriptive paragraph.", "summary") == [
        "A single descriptive paragraph."
    ]


def test_the_questions_form_yields_a_row_per_question():
    """Averaging five questions into one vector would blur the distinctions."""
    generated = "What is X?\nHow does Y work?\n- Why Z?\n"
    assert gi.split_generated(generated, "questions") == [
        "What is X?",
        "How does Y work?",
        "Why Z?",
    ]


def test_list_markers_and_numbering_are_stripped():
    generated = "1. What is X?\n2) How does Y work?\n* Why Z?"
    assert gi.split_generated(generated, "questions") == [
        "What is X?",
        "How does Y work?",
        "Why Z?",
    ]


def test_blank_generated_text_yields_no_rows():
    assert gi.split_generated("   \n  \n", "questions") == []


def test_an_unknown_form_is_refused():
    with pytest.raises(gi.GenerateError, match="nonsense"):
        gi.split_generated("text", "nonsense")


def test_a_vector_is_encoded_as_little_endian_f32():
    """A mismatch would produce vectors silently wrong rather than absent."""
    blob = gi.encode_vector([1.0, -2.0])
    assert len(blob) == 8
    assert gi.decode_vector(blob) == pytest.approx([1.0, -2.0])
