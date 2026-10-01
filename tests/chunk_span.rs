//! Splitting a long passage into embeddable spans.
//!
//! The embeddings table is keyed by `(stream_hash, span_start, span_len,
//! model)`, so it already addresses any region of a stream. A message too
//! long for the model's window therefore needs no new concept — only smaller
//! spans — and the corpus contains a 268,000-character one.

use kb::embed_text::chunk_span;

#[test]
fn a_short_passage_is_one_span() {
    assert_eq!(chunk_span("short enough", 100, 6000), vec![(100, 12)]);
}

/// Spans have to tile the passage exactly: a gap loses text from retrieval
/// and an overlap ranks the same words twice.
#[test]
fn the_spans_tile_the_passage_without_gap_or_overlap() {
    let text = "line of text\n".repeat(400);

    let spans = chunk_span(&text, 50, 1000);

    assert!(spans.len() > 1);
    let mut expected = 50;
    for (start, len) in &spans {
        assert_eq!(*start, expected, "spans did not tile: {spans:?}");
        expected = start + len;
    }
    assert_eq!(expected, 50 + text.len());
    assert!(spans.iter().all(|(_, len)| *len <= 1000));
}

/// A chunk that ends mid-sentence embeds worse than one that ends at a line,
/// so a break in the last quarter of the budget is taken over the ceiling.
#[test]
fn a_break_near_the_budget_is_preferred_to_a_hard_cut() {
    let text = format!("{}\n{}", "a".repeat(90), "b".repeat(200));

    let spans = chunk_span(&text, 0, 100);

    assert_eq!(
        spans.first().map(|(_, len)| *len),
        Some(91),
        "the cut ignored the line break: {spans:?}"
    );
}

/// A long run with no break in range still has to yield something: the
/// corpus contains a 268,000-character message.
#[test]
fn an_unbroken_run_is_cut_at_the_budget() {
    let text = "x".repeat(2500);

    let spans = chunk_span(&text, 0, 1000);

    assert_eq!(spans, vec![(0, 1000), (1000, 1000), (2000, 500)]);
}

/// Byte offsets address the normalized stream, and an offset that is not a
/// character boundary would make the slice unreadable.
#[test]
fn a_cut_never_lands_inside_a_character() {
    let text = "é".repeat(500);

    let spans = chunk_span(&text, 0, 101);

    for (start, len) in &spans {
        assert!(text.is_char_boundary(start.saturating_sub(0)), "{spans:?}");
        assert!(
            text.get(*start..start + len).is_some(),
            "span {start}+{len} is not a valid slice: {spans:?}"
        );
    }
}

/// Degenerate budgets must not loop or drop the passage.
#[test]
fn a_zero_budget_yields_the_whole_passage_rather_than_nothing() {
    assert_eq!(chunk_span("text", 7, 0), vec![(7, 4)]);
    assert_eq!(chunk_span("", 7, 100), vec![(7, 0)]);
}
