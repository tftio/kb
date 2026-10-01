//! Composition of embeddable text from a knowledge-base document.
//!
//! An embedding endpoint takes text, not an AST, so every write path that
//! stores a vector must first flatten a [`Document`] into the string the
//! model sees. That flattening is a domain decision — which parts of the
//! document represent it for retrieval — and it is shared by every caller
//! that embeds, so it lives here rather than being inlined at a call site.
//!
//! The module is deliberately pure: it performs no I/O and knows nothing
//! about endpoints, models, or storage. That keeps it testable without a
//! daemon and confines the imperative shell to the callers
//! (`REPO_INVARIANTS.md` ENG-008).
//!
//! Dependency direction is `embed_text` → `storage` → `embedding`. Placing
//! these functions in `embedding` instead would require importing `storage`
//! for the title and body extractors while `storage` already imports
//! `embedding` for `decode_embedding`, forming a module cycle.

use tftio_org::ast::{Block, Document};

use crate::storage;

/// Compose the text embedded to represent `doc`.
///
/// The title is prepended to the body text on its own line. The title is a
/// node's most concentrated description of itself, and a body whose opening
/// is a bare heading such as `Gotchas` is close to meaningless without it,
/// so a vector computed from the body alone loses the node's subject.
///
/// **Verbatim blocks are excluded.** The body comes from
/// [`storage::blocks_prose_text`] rather than
/// [`storage::extract_body_text`], so a src or example block contributes
/// nothing to the vector. The node keeps the block and the FTS index keeps
/// indexing it; only the embedding narrows to prose. See
/// [`storage::blocks_prose_text`] for why. A node whose body is entirely
/// verbatim therefore embeds as its title alone, which is a weak
/// representation but an honest one — and still enough to be retrieved by
/// subject.
///
/// The result is the whole document. Callers embedding a long document are
/// responsible for splitting it to fit the model's context window — see
/// [`chunk_document`] — because an endpoint handed more text than it
/// accepts truncates silently.
#[must_use]
pub fn embeddable_text(doc: &Document) -> String {
    let title = storage::extract_title(doc);
    let body_text = storage::blocks_prose_text(&doc.blocks);
    format!("{title}\n{body_text}")
}

/// Divisor giving the trailing context repeated at the head of every chunk
/// after the first: one tenth of the chunk budget.
///
/// A split lands wherever a chunk's budget runs out, which for a long
/// unbroken passage is an arbitrary point mid-argument. Repeating the tail
/// of the previous chunk means a fact spanning the seam is intact in the
/// later chunk rather than halved in both.
///
/// A *fraction* rather than a fixed character count, because how much
/// overlap is enough depends entirely on the chunk size. The same 256
/// characters that comfortably bridge a seam between two-page chunks are a
/// rounding error between twenty-page ones, where they would leave the
/// mechanism nominally present and practically inert.
const OVERLAP_DIVISOR: usize = 10;

/// Split `doc` into chunks of at most `max_chars` characters, each carrying
/// the node's title.
///
/// A document that fits in `max_chars` yields exactly one chunk, and that
/// chunk is [`embeddable_text`] verbatim — the composition the stored
/// corpus was built under. Only an overflowing document takes the splitting
/// path below, which is why the common case (the great majority of nodes)
/// is unaffected by any of it.
///
/// **Splitting.** The document is flattened into *units*: a heading
/// contributes its title as one unit followed by the units of its children,
/// and every other block contributes its rendered text. Units are packed
/// greedily into chunks, so a break always lands on a block boundary —
/// most often between paragraphs, and at a section boundary whenever a
/// heading's title unit begins a chunk. A single unit too large for the
/// budget, such as a single unbroken paragraph running to many pages, is
/// split at whitespace where possible and hard-split where not, because a
/// chunk that exceeded the window would be silently truncated by the
/// endpoint and the excess lost without any error.
///
/// **Title prefix.** Every chunk begins with the node title, because a
/// chunk whose leading heading is `Gotchas` says nothing about what it is a
/// gotcha concerning. The title is charged against every chunk's budget.
///
/// **A known asymmetry.** Heading titles appear as units on the splitting
/// path but not in `embeddable_text`, whose body comes from
/// [`storage::blocks_prose_text`] and has never included them. A short node
/// is therefore embedded without its inner headings while a long one is
/// embedded with them. Correcting this means changing the shared block
/// walker, which also feeds the FTS body index, so it is a
/// retrieval-quality decision to be measured rather than a side effect of
/// chunking. Verbatim-block exclusion is deliberately *not* an instance of
/// this asymmetry: both paths render through `blocks_prose_text` and drop
/// the same blocks.
///
/// The function is pure and deterministic: identical input yields identical
/// output, and it performs no I/O.
///
/// A `max_chars` of 0, or one too small to hold the title and a single
/// character, yields one chunk truncated to `max_chars` — the only result
/// that both honours the ceiling and reports something rather than nothing.
#[must_use]
pub fn chunk_document(doc: &Document, max_chars: usize) -> Vec<String> {
    let whole = embeddable_text(doc);
    if count_chars(&whole) <= max_chars {
        return vec![whole];
    }

    let prefix = format!("{}\n", storage::extract_title(doc));
    let prefix_len = count_chars(&prefix);

    // The title is repeated on every chunk, so it is charged against every
    // chunk's budget. When it alone fills the window there is no room for
    // content, and a single truncated chunk is the only output that both
    // honours the ceiling and says something.
    let Some(budget) = max_chars.checked_sub(prefix_len).filter(|b| *b > 0) else {
        return vec![take_chars(&whole, max_chars)];
    };

    // Overlap is charged against every chunk after the first, so unit
    // splitting targets the smaller budget and any chunk fits either way.
    let overlap = budget / OVERLAP_DIVISOR;
    let body_budget = budget.saturating_sub(overlap).max(1);

    let mut units = Vec::new();
    collect_units(&doc.blocks, body_budget, &mut units);
    if units.is_empty() {
        return vec![take_chars(&whole, max_chars)];
    }

    let packed = pack_units(&units, budget, body_budget);
    prefix_chunks(&packed, &prefix, overlap)
}

/// Flatten `blocks` into text units no larger than `max_unit` characters.
///
/// A heading contributes its own title before its children's units, so a
/// section boundary is always an available break point for the packer. Any
/// other block contributes the text of that block alone.
///
/// Verbatim blocks are excluded here exactly as they are in
/// [`embeddable_text`], by rendering each block through
/// [`storage::blocks_prose_text`]: such a block renders empty and
/// [`push_unit`] drops it. Using the same renderer on both paths is what
/// keeps a node's representation independent of its length — were only one
/// path to exclude them, whether a src block reached the model would depend
/// on whether the node happened to exceed the chunk budget.
fn collect_units(blocks: &[Block], max_unit: usize, out: &mut Vec<String>) {
    for block in blocks {
        if let Block::Heading {
            title, children, ..
        } = block
        {
            push_unit(&title.0, max_unit, out);
            collect_units(children, max_unit, out);
        } else {
            push_unit(
                &storage::blocks_prose_text(std::slice::from_ref(block)),
                max_unit,
                out,
            );
        }
    }
}

/// Append `text` as one unit, subdividing it if it exceeds `max_unit`.
/// Whitespace-only text contributes nothing.
fn push_unit(text: &str, max_unit: usize, out: &mut Vec<String>) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    if count_chars(text) <= max_unit {
        out.push(text.to_string());
        return;
    }
    split_oversized(text, max_unit, out);
}

/// Split a single oversized unit into pieces of at most `max_unit`
/// characters, preferring the last whitespace inside each window so a word
/// is not cut in half. Text with no whitespace at all is hard-split, which
/// is the only way to bound a single enormous token.
fn split_oversized(text: &str, max_unit: usize, out: &mut Vec<String>) {
    let mut rest = text;
    loop {
        let trimmed = rest.trim_start();
        if count_chars(trimmed) <= max_unit {
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
            return;
        }
        let window: String = trimmed.chars().take(max_unit).collect();
        // `rfind` returns a byte offset inside `window`, and a cut at 0
        // would make no progress, so fall back to the whole window.
        let cut = window
            .rfind(char::is_whitespace)
            .filter(|i| *i > 0)
            .unwrap_or(window.len());
        let (head, _) = window.split_at(cut);
        let head_trimmed = head.trim_end();
        if !head_trimmed.is_empty() {
            out.push(head_trimmed.to_string());
        }
        rest = trimmed.get(head.len()..).unwrap_or("");
    }
}

/// Pack units greedily into chunk bodies, joining them with newlines.
///
/// The first chunk may use the full `budget`; later chunks reserve room for
/// the overlap they will be given, so `body_budget` bounds them.
fn pack_units(units: &[String], budget: usize, body_budget: usize) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    for unit in units {
        let limit = if chunks.is_empty() {
            budget
        } else {
            body_budget
        };
        if !current.is_empty() {
            if count_chars(&current) + 1 + count_chars(unit) > limit {
                chunks.push(std::mem::take(&mut current));
            } else {
                current.push('\n');
            }
        }
        current.push_str(unit);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Prepend the title to every chunk body, and the tail of the previous body
/// to every body after the first.
fn prefix_chunks(bodies: &[String], prefix: &str, overlap: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(bodies.len());
    let mut previous: Option<&String> = None;
    for body in bodies {
        let chunk = match previous {
            // One character of the overlap allowance pays for the newline
            // separating it from the body, so the total stays within budget.
            Some(prev) if overlap > 1 => {
                let tail = tail_chars(prev, overlap - 1);
                if tail.is_empty() {
                    format!("{prefix}{body}")
                } else {
                    format!("{prefix}{tail}\n{body}")
                }
            }
            _ => format!("{prefix}{body}"),
        };
        out.push(chunk);
        previous = Some(body);
    }
    out
}

fn count_chars(s: &str) -> usize {
    s.chars().count()
}

fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// The last `n` characters of `s`, trimmed so the overlap does not begin
/// mid-word where a word boundary is available.
fn tail_chars(s: &str, n: usize) -> String {
    let total = count_chars(s);
    let skip = total.saturating_sub(n);
    let tail: String = s.chars().skip(skip).collect();
    if skip == 0 {
        return tail;
    }
    match tail.find(char::is_whitespace) {
        Some(i) => tail.get(i..).unwrap_or(&tail).trim_start().to_string(),
        None => tail,
    }
}

/// Split one passage's span into sub-spans no longer than `max_bytes`.
///
/// Spans rather than strings, because the embeddings table is keyed by
/// `(stream_hash, span_start, span_len, model)` and therefore already
/// addresses any region of a stream. A long message needs no new concept to
/// be embeddable — only smaller spans.
///
/// Splits at line boundaries where one is available in the last quarter of
/// the budget, so a chunk ends at a sentence or a quoted line rather than
/// mid-word. A run with no line break in range is cut at the budget: a
/// 268,000-character message with no newlines exists in the corpus and has to
/// yield something.
///
/// Byte offsets throughout, matching how spans address the normalized stream,
/// and never splitting inside a multi-byte character.
#[must_use]
pub fn chunk_span(text: &str, span_start: usize, max_bytes: usize) -> Vec<(usize, usize)> {
    if max_bytes == 0 || text.is_empty() {
        return vec![(span_start, text.len())];
    }
    if text.len() <= max_bytes {
        return vec![(span_start, text.len())];
    }
    let mut spans = Vec::new();
    let mut at = 0_usize;
    while at < text.len() {
        let remaining = text.len().saturating_sub(at);
        if remaining <= max_bytes {
            spans.push((span_start.saturating_add(at), remaining));
            break;
        }
        let ceiling = at.saturating_add(max_bytes);
        let floor = at.saturating_add(max_bytes.saturating_sub(max_bytes / 4));
        let cut = text
            .get(floor..ceiling)
            .and_then(|window| window.rfind('\n'))
            .map_or(ceiling, |at_break| {
                floor.saturating_add(at_break).saturating_add(1)
            });
        // Never split inside a character: an offset that is not a boundary
        // would make the slice unreadable and the span meaningless.
        let mut cut = cut.min(text.len());
        while cut > at && !text.is_char_boundary(cut) {
            cut = cut.saturating_sub(1);
        }
        if cut <= at {
            cut = ceiling.min(text.len());
            while cut < text.len() && !text.is_char_boundary(cut) {
                cut = cut.saturating_add(1);
            }
        }
        spans.push((span_start.saturating_add(at), cut.saturating_sub(at)));
        at = cut;
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::embeddable_text;
    use tftio_org::ast::{Block, Document, Inline, Title};

    /// A document with one heading and one paragraph beneath it.
    fn doc_with_heading(title: &str, body: &str) -> Document {
        Document {
            blocks: vec![Block::Heading {
                level: 1,
                title: Title(title.to_string()),
                tags: vec![],
                children: vec![Block::Paragraph {
                    inlines: vec![Inline::Plain(body.to_string())],
                }],
            }],
        }
    }

    #[test]
    fn title_precedes_body_on_its_own_line() {
        let doc = doc_with_heading("Retrieval design", "Hybrid fusion at k=60.");
        let text = embeddable_text(&doc);

        let (first, rest) = text
            .split_once('\n')
            .unwrap_or_else(|| panic!("composition must contain a newline: {text:?}"));
        assert_eq!(first, "Retrieval design");
        assert!(
            rest.contains("Hybrid fusion at k=60."),
            "body text must follow the title, got {rest:?}"
        );
    }

    /// The composition must match what `api::compute_doc_embedding` produced
    /// before the port, or every vector written after it silently describes
    /// a different string than the vectors written before it.
    ///
    /// The body extractor named here is `blocks_prose_text` rather than
    /// `extract_body_text`, which is the 2026-07-29 verbatim-block change;
    /// the two agree on any document without a src or example block, as
    /// this one is.
    #[test]
    fn composition_is_title_newline_body() {
        let doc = doc_with_heading("Subject", "Body sentence.");
        let expected = format!(
            "{}\n{}",
            crate::storage::extract_title(&doc),
            crate::storage::blocks_prose_text(&doc.blocks)
        );
        assert_eq!(embeddable_text(&doc), expected);
        assert_eq!(
            crate::storage::extract_body_text(&doc),
            crate::storage::blocks_prose_text(&doc.blocks),
            "the two extractors must agree on prose-only documents"
        );
    }

    /// A document with a heading, a paragraph, a src block, and a second
    /// paragraph — the shape of one exchange in an imported session
    /// transcript.
    fn doc_with_src_block(title: &str, before: &str, code: &str, after: &str) -> Document {
        Document {
            blocks: vec![Block::Heading {
                level: 1,
                title: Title(title.to_string()),
                tags: vec![],
                children: vec![
                    Block::Paragraph {
                        inlines: vec![Inline::Plain(before.to_string())],
                    },
                    Block::SrcBlock {
                        language: "text".into(),
                        content: code.to_string(),
                    },
                    Block::Paragraph {
                        inlines: vec![Inline::Plain(after.to_string())],
                    },
                ],
            }],
        }
    }

    /// The corpus is mostly session transcripts whose bulk is serialized
    /// tool calls. Those are kept and keyword-indexed, never embedded.
    #[test]
    fn a_src_blocks_content_is_not_embedded_but_its_surrounding_prose_is() {
        let doc = doc_with_src_block(
            "Session",
            "Here is what I ran.",
            "tool_use: Bash {\"command\": \"ls /etc\"}",
            "That listing shows the config.",
        );
        let text = embeddable_text(&doc);
        assert!(
            !text.contains("tool_use"),
            "verbatim content leaked into the embedded text: {text:?}"
        );
        assert!(text.contains("Here is what I ran."), "got {text:?}");
        assert!(
            text.contains("That listing shows the config."),
            "got {text:?}"
        );
    }

    /// The companion half of the rule: nothing is lost. `blocks_text` still
    /// carries the block, so the FTS body index still finds the command.
    #[test]
    fn the_same_src_block_is_still_present_for_keyword_indexing() {
        let doc = doc_with_src_block(
            "Session",
            "Here is what I ran.",
            "tool_use: Bash {\"command\": \"ls /etc\"}",
            "That listing shows the config.",
        );
        let indexed = crate::storage::blocks_text(&doc.blocks);
        assert!(
            indexed.contains("tool_use") && indexed.contains("ls /etc"),
            "the FTS path must still see verbatim content, got {indexed:?}"
        );
    }

    /// A node that is nothing but a code dump still has a subject, and the
    /// title alone is a weak representation but a usable one. The failure
    /// this guards against is emitting no chunk at all, which would leave
    /// the node unreachable by vector search entirely.
    #[test]
    fn a_document_of_only_verbatim_content_still_embeds_its_title() {
        let doc = Document {
            blocks: vec![Block::Heading {
                level: 1,
                title: Title("Deploy log".into()),
                tags: vec![],
                children: vec![Block::SrcBlock {
                    language: "text".into(),
                    content: "x".repeat(50_000),
                }],
            }],
        };
        let chunks = chunk_document(&doc, 6_000);
        assert_eq!(chunks.len(), 1, "got {} chunks", chunks.len());
        let only = chunks.first().map(String::as_str).unwrap_or_default();
        assert!(only.contains("Deploy log"), "got {only:?}");
        assert!(!only.contains('x'), "verbatim content leaked: {only:?}");
    }

    /// The two paths must drop the same blocks. Were only `embeddable_text`
    /// to exclude them, whether a src block reached the model would depend
    /// on whether the node happened to exceed the chunk budget — a node's
    /// representation would vary with its length.
    #[test]
    fn a_large_src_block_changes_neither_the_chunks_nor_their_number() {
        let sections: Vec<(String, String)> = (0..12)
            .map(|i| {
                (
                    format!("Section {i}"),
                    format!("Prose paragraph {i}. ").repeat(40),
                )
            })
            .collect();
        let refs: Vec<(&str, &str)> = sections
            .iter()
            .map(|(h, b)| (h.as_str(), b.as_str()))
            .collect();

        let without = doc_with_sections("Long note", &refs);
        let mut with = doc_with_sections("Long note", &refs);
        with.blocks.push(Block::SrcBlock {
            language: "text".into(),
            content: "tool_result: ".to_string() + &"noise ".repeat(20_000),
        });

        let a = chunk_document(&without, 2_000);
        let b = chunk_document(&with, 2_000);
        assert!(
            a.len() > 1,
            "the fixture must exercise splitting, got {a:?}"
        );
        assert_eq!(a, b, "a verbatim block must not affect chunking at all");
    }

    /// A document with no heading takes its title from its first paragraph
    /// (`storage::find_first_title`), so the composition repeats that text:
    /// once as the title line, once as the body. This is the behaviour the
    /// stored corpus was embedded under and is asserted here to pin it, not
    /// because the duplication is desirable.
    #[test]
    fn headingless_document_repeats_its_first_paragraph() {
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![Inline::Plain("Body without any heading.".into())],
            }],
        };
        assert_eq!(
            embeddable_text(&doc),
            "Body without any heading.\nBody without any heading."
        );
    }

    #[test]
    fn empty_document_yields_a_title_line_and_nothing_else() {
        let doc = Document { blocks: vec![] };
        assert_eq!(embeddable_text(&doc), "(untitled)\n");
    }

    #[test]
    fn composition_is_deterministic() {
        let doc = doc_with_heading("Stable", "Same every time.");
        assert_eq!(embeddable_text(&doc), embeddable_text(&doc));
    }

    // ── chunk_document ────────────────────────────────────────────────

    use super::{chunk_document, count_chars};
    use tftio_org::ast::Tag;

    /// A document whose first heading is the node title, followed by
    /// `sections` further top-level headings each holding one paragraph.
    fn doc_with_sections(title: &str, sections: &[(&str, &str)]) -> Document {
        let mut blocks = vec![Block::Heading {
            level: 1,
            title: Title(title.to_string()),
            tags: vec![Tag("test".into())],
            children: vec![],
        }];
        for (heading, body) in sections {
            blocks.push(Block::Heading {
                level: 2,
                title: Title((*heading).to_string()),
                tags: vec![],
                children: vec![Block::Paragraph {
                    inlines: vec![Inline::Plain((*body).to_string())],
                }],
            });
        }
        Document { blocks }
    }

    fn paragraph(text: &str) -> Block {
        Block::Paragraph {
            inlines: vec![Inline::Plain(text.to_string())],
        }
    }

    #[test]
    fn a_document_that_fits_yields_exactly_one_chunk() {
        let doc = doc_with_heading("Short note", "A single sentence.");
        let chunks = chunk_document(&doc, 4096);
        assert_eq!(chunks.len(), 1);
    }

    /// The single-chunk case must be byte-identical to `embeddable_text`.
    /// If it were not, the same node would embed differently depending on
    /// which write path stored it.
    #[test]
    fn the_single_chunk_case_is_embeddable_text_verbatim() {
        let doc = doc_with_heading("Short note", "A single sentence.");
        assert_eq!(chunk_document(&doc, 4096), vec![embeddable_text(&doc)]);
    }

    #[test]
    fn every_chunk_begins_with_the_node_title() {
        let doc = doc_with_sections(
            "Retrieval design",
            &[
                (
                    "Fusion",
                    &"reciprocal rank fusion at k equals sixty ".repeat(20),
                ),
                ("Chunking", &"split on heading boundaries first ".repeat(20)),
                (
                    "Storage",
                    &"packed little endian float thirty two ".repeat(20),
                ),
            ],
        );
        let chunks = chunk_document(&doc, 300);
        assert!(
            chunks.len() > 1,
            "fixture must overflow, got {}",
            chunks.len()
        );
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.starts_with("Retrieval design\n"),
                "chunk {i} does not begin with the title: {chunk:?}"
            );
        }
    }

    #[test]
    fn no_chunk_exceeds_max_chars() {
        let doc = doc_with_sections(
            "Budgeted",
            &[
                ("Alpha", &"alpha content ".repeat(50)),
                ("Beta", &"beta content ".repeat(50)),
            ],
        );
        for max in [64_usize, 100, 256, 512, 1024, 4096] {
            for (i, chunk) in chunk_document(&doc, max).iter().enumerate() {
                assert!(
                    count_chars(chunk) <= max,
                    "max={max}: chunk {i} is {} chars",
                    count_chars(chunk)
                );
            }
        }
    }

    #[test]
    fn an_oversized_single_section_is_split_at_whitespace() {
        // One section far larger than the budget: the split must fall
        // inside it, and must not cut a word in half.
        let doc = doc_with_sections("Long", &[("Body", &"lorem ipsum dolor ".repeat(200))]);
        let chunks = chunk_document(&doc, 200);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            for word in chunk.split_whitespace() {
                assert!(
                    ["Long", "Body", "lorem", "ipsum", "dolor"].contains(&word),
                    "a word was cut in half: {word:?}"
                );
            }
        }
    }

    #[test]
    fn a_unit_with_no_whitespace_is_hard_split_rather_than_dropped() {
        // A single enormous token — a base64 blob, a minified line — has
        // no whitespace to split on. Bounding the chunk matters more than
        // keeping the token intact, because the endpoint would truncate
        // the excess silently.
        let doc = Document {
            blocks: vec![
                Block::Heading {
                    level: 1,
                    title: Title("Blob".into()),
                    tags: vec![],
                    children: vec![],
                },
                paragraph(&"x".repeat(1000)),
            ],
        };
        let chunks = chunk_document(&doc, 120);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(count_chars(chunk) <= 120);
        }
        let recovered: usize = chunks.iter().map(|c| c.matches('x').count()).sum();
        assert!(
            recovered >= 1000,
            "every x must survive somewhere, got {recovered}"
        );
    }

    #[test]
    fn chunking_loses_no_non_whitespace_content() {
        let doc = doc_with_sections(
            "Coverage",
            &[
                ("One", "alpha beta gamma delta"),
                ("Two", "epsilon zeta eta theta"),
                ("Three", "iota kappa lambda mu"),
            ],
        );
        let joined = chunk_document(&doc, 90).join(" ");
        for word in [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
            "lambda", "mu",
        ] {
            assert!(joined.contains(word), "{word} was lost");
        }
    }

    #[test]
    fn adjacent_chunks_overlap() {
        // The tail of each chunk's body reappears at the head of the next,
        // so a fact spanning the seam survives intact on one side.
        let doc = doc_with_sections(
            "Overlap",
            &[("Body", &"one two three four five ".repeat(80))],
        );
        let chunks = chunk_document(&doc, 1200);
        assert!(chunks.len() > 1, "fixture must overflow");
        let first = chunks.first().expect("at least two chunks");
        let second = chunks.get(1).expect("at least two chunks");
        let tail: String = first
            .chars()
            .skip(count_chars(first).saturating_sub(40))
            .collect();
        assert!(
            second.contains(tail.trim()),
            "chunk 2 must repeat the tail of chunk 1\ntail: {tail:?}\nnext: {second:?}"
        );
    }

    #[test]
    fn an_empty_document_yields_one_chunk() {
        let chunks = chunk_document(&Document { blocks: vec![] }, 128);
        assert_eq!(chunks, vec!["(untitled)\n".to_string()]);
    }

    #[test]
    fn a_document_with_no_headings_still_chunks() {
        let doc = Document {
            blocks: vec![
                paragraph(&"first paragraph text ".repeat(30)),
                paragraph(&"second paragraph text ".repeat(30)),
            ],
        };
        let chunks = chunk_document(&doc, 200);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(count_chars(chunk) <= 200);
        }
    }

    #[test]
    fn a_budget_too_small_for_the_title_yields_one_truncated_chunk() {
        // Degenerate, but it must bound the output rather than loop or
        // emit a chunk larger than the caller's ceiling.
        let doc = doc_with_heading("A rather long node title indeed", "body");
        let chunks = chunk_document(&doc, 8);
        assert_eq!(chunks.len(), 1);
        assert_eq!(count_chars(chunks.first().expect("one chunk")), 8);
    }

    #[test]
    fn a_zero_budget_yields_one_empty_chunk() {
        let doc = doc_with_heading("Title", "body");
        assert_eq!(chunk_document(&doc, 0), vec![String::new()]);
    }

    #[test]
    fn chunking_is_deterministic() {
        let doc = doc_with_sections(
            "Stable",
            &[
                ("A", &"repeated content ".repeat(40)),
                ("B", &"more content ".repeat(40)),
            ],
        );
        assert_eq!(chunk_document(&doc, 256), chunk_document(&doc, 256));
    }

    /// Build a document from arbitrary section texts, so the property
    /// below sees empty sections, whitespace-only sections, single
    /// enormous tokens, and multi-byte characters.
    fn arb_document() -> impl proptest::strategy::Strategy<Value = Document> {
        use proptest::prelude::{Just, prop_oneof};
        use proptest::strategy::Strategy;

        let word = prop_oneof![
            Just("alpha".to_string()),
            Just("  ".to_string()),
            Just(String::new()),
            Just("ünïcøde".to_string()),
            Just("x".repeat(300)),
            Just("a b c d e f g".to_string()),
        ];
        proptest::collection::vec((word.clone(), word), 0..=8).prop_map(|sections| {
            let mut blocks = vec![Block::Heading {
                level: 1,
                title: Title("Property".into()),
                tags: vec![],
                children: vec![],
            }];
            for (heading, body) in sections {
                blocks.push(Block::Heading {
                    level: 2,
                    title: Title(heading),
                    tags: vec![],
                    children: vec![Block::Paragraph {
                        inlines: vec![Inline::Plain(body)],
                    }],
                });
            }
            Document { blocks }
        })
    }

    proptest::proptest! {
        /// The ceiling is the whole point: a chunk over the window is
        /// truncated by the endpoint without an error, so the excess is
        /// lost silently. It must hold for every document and every
        /// budget, including the degenerate ones.
        #[test]
        fn no_emitted_chunk_ever_exceeds_max_chars(
            doc in arb_document(),
            max in 0_usize..=512,
        ) {
            for chunk in chunk_document(&doc, max) {
                proptest::prop_assert!(
                    count_chars(&chunk) <= max,
                    "chunk of {} chars exceeds max {max}",
                    count_chars(&chunk)
                );
            }
        }

        /// Chunking must terminate and say something. An empty result
        /// would mean a node silently carried no vector at all.
        #[test]
        fn chunking_always_produces_at_least_one_chunk(
            doc in arb_document(),
            max in 0_usize..=512,
        ) {
            proptest::prop_assert!(!chunk_document(&doc, max).is_empty());
        }
    }

    #[test]
    fn section_headings_are_available_break_points() {
        // A heading contributes its own unit, so its title is present in
        // the chunk stream rather than being dropped as
        // `extract_body_text` drops it.
        let doc = doc_with_sections(
            "Doc",
            &[
                ("Decision", &"we chose the local daemon ".repeat(30)),
                ("Consequences", &"the daemon is load bearing ".repeat(30)),
            ],
        );
        let joined = chunk_document(&doc, 300).join("\n");
        assert!(joined.contains("Decision"));
        assert!(joined.contains("Consequences"));
    }
}
