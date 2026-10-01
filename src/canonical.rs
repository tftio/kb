//! Canonical-form mapping over [`Document`].
//!
//! The parser is a function from bytes to a single deterministic AST. Several
//! distinct ASTs can render to byte-identical text, so the parser collapses
//! each equivalence class onto one representative. [`canonicalize`] computes
//! that representative directly from any input AST.
//!
//! Ported from Haskell `KB/AST/Canonical.hs`.

use tftio_org::ast::{Block, Document, Inline, ListItem, LogEntry, TableCell, Tag, Title};

/// Reduce a [`Document`] to its canonical form.
#[must_use]
pub fn canonicalize(doc: &Document) -> Document {
    Document {
        blocks: doc.blocks.iter().flat_map(canonicalize_block).collect(),
    }
}

/// Block canonicalization. Returns a list because some inputs canonicalize
/// to nothing (empty paragraphs) or to several siblings (heading children
/// flatten to top-level blocks when not recognized as children by the parser).
fn canonicalize_block(block: &Block) -> Vec<Block> {
    match block {
        Block::Heading {
            level,
            title,
            tags,
            children,
        } => {
            let stripped = title.0.trim();
            let (title_no_tags, extracted_tags) = split_title_and_tags(stripped);
            let mut all_tags = tags.clone();
            all_tags.extend(extracted_tags);
            let canonical_children: Vec<Block> =
                children.iter().flat_map(canonicalize_block).collect();
            vec![Block::Heading {
                level: (*level).max(1),
                title: Title(title_no_tags),
                tags: all_tags,
                children: canonical_children,
            }]
        }
        Block::Paragraph { inlines } => {
            let cs = canonicalize_inlines(inlines);
            if cs.is_empty() {
                vec![]
            } else {
                vec![Block::Paragraph { inlines: cs }]
            }
        }
        Block::SrcBlock { language, content } => {
            vec![Block::SrcBlock {
                language: language.trim().to_string(),
                content: canonicalize_src_body(content),
            }]
        }
        Block::ExampleBlock { content } => {
            // A non-empty body always ends in `\n`, matching the parser.
            let content = if content.is_empty() || content.ends_with('\n') {
                content.clone()
            } else {
                format!("{content}\n")
            };
            vec![Block::ExampleBlock { content }]
        }
        Block::PropertyDrawer { entries } => {
            vec![Block::PropertyDrawer {
                entries: entries.clone(),
            }]
        }
        Block::Planning { entries } => {
            if entries.is_empty() {
                vec![]
            } else {
                vec![Block::Planning {
                    entries: entries.clone(),
                }]
            }
        }
        Block::LogbookDrawer { entries } => {
            vec![Block::LogbookDrawer {
                entries: entries
                    .iter()
                    .map(|e| LogEntry {
                        timestamp: e.timestamp.clone(),
                        note: e.note.clone(),
                    })
                    .collect(),
            }]
        }
        Block::List { list_type, items } => {
            vec![Block::List {
                list_type: list_type.clone(),
                items: items.iter().map(canonicalize_list_item).collect(),
            }]
        }
        Block::QuoteBlock { children } => {
            vec![Block::QuoteBlock {
                children: children.iter().flat_map(canonicalize_block).collect(),
            }]
        }
        Block::Comment { text } => {
            vec![Block::Comment { text: text.clone() }]
        }
        Block::Keyword { name, value } => {
            vec![Block::Keyword {
                name: name.clone(),
                value: value.clone(),
            }]
        }
        Block::BlankLine => vec![Block::BlankLine],
        Block::HorizontalRule => vec![Block::HorizontalRule],
        Block::Table { rows } => {
            vec![Block::Table {
                rows: rows
                    .iter()
                    .map(|row| row.iter().map(canonicalize_table_cell).collect())
                    .collect(),
            }]
        }
    }
}

fn canonicalize_table_cell(cell: &TableCell) -> TableCell {
    TableCell {
        inlines: canonicalize_inlines(&cell.inlines),
    }
}

fn canonicalize_list_item(item: &ListItem) -> ListItem {
    ListItem {
        content: item.content.iter().flat_map(canonicalize_block).collect(),
        checkbox: item.checkbox.clone(),
    }
}

/// Inline canonicalization: drop empty Plain runs, merge adjacent Plain runs,
/// recurse into Bold/Italic (dropping them when content vanishes).
fn canonicalize_inlines(inlines: &[Inline]) -> Vec<Inline> {
    merge_plains(
        &inlines
            .iter()
            .flat_map(canonicalize_inline)
            .collect::<Vec<_>>(),
    )
}

/// Canonicalize a single inline element. Empty-markup inputs are translated to
/// the literal-character runs the parser produces from the generator's output.
fn canonicalize_inline(inline: &Inline) -> Vec<Inline> {
    match inline {
        Inline::Plain(t) => {
            if t.is_empty() {
                vec![]
            } else {
                vec![Inline::Plain(t.clone())]
            }
        }
        Inline::Bold(is) => {
            let cs = canonicalize_inlines(is);
            if cs.is_empty() {
                vec![Inline::Plain("**".into())]
            } else {
                vec![Inline::Bold(cs)]
            }
        }
        Inline::Italic(is) => {
            let cs = canonicalize_inlines(is);
            if cs.is_empty() {
                vec![Inline::Plain("//".into())]
            } else {
                vec![Inline::Italic(cs)]
            }
        }
        Inline::Strikethrough(is) => {
            let cs = canonicalize_inlines(is);
            if cs.is_empty() {
                vec![Inline::Plain("++".into())]
            } else {
                vec![Inline::Strikethrough(cs)]
            }
        }
        Inline::InlineCode(t) => {
            if t.is_empty() {
                vec![Inline::Plain("==".into())]
            } else {
                vec![Inline::InlineCode(t.clone())]
            }
        }
        Inline::Verbatim(t) => {
            if t.is_empty() {
                vec![Inline::Plain("~~".into())]
            } else {
                vec![Inline::Verbatim(t.clone())]
            }
        }
        Inline::LineBreak => vec![Inline::LineBreak],
        Inline::Link {
            target,
            description: None,
        } => {
            if target.is_empty() {
                vec![Inline::Plain("[[]]".into())]
            } else {
                vec![Inline::Link {
                    target: target.clone(),
                    description: None,
                }]
            }
        }
        Inline::Link {
            target,
            description: Some(desc),
        } => {
            if target.is_empty() {
                vec![Inline::Plain(format!("[[][{desc}]]"))]
            } else if desc.is_empty() {
                vec![Inline::Plain(format!("[[{target}][]]"))]
            } else {
                vec![Inline::Link {
                    target: target.clone(),
                    description: Some(desc.clone()),
                }]
            }
        }
    }
}

/// Merge adjacent Plain values.
fn merge_plains(inlines: &[Inline]) -> Vec<Inline> {
    let mut result: Vec<Inline> = Vec::new();
    for inline in inlines {
        match (result.last_mut(), inline) {
            (Some(Inline::Plain(a)), Inline::Plain(b)) => {
                a.push_str(b);
            }
            _ => {
                result.push(inline.clone());
            }
        }
    }
    result
}

/// Strip a trailing `:tag1:tag2:` block off a heading title.
///
/// A tag block is recognized only when the line contains at least two `:`
/// characters and the text before the block is empty or ends in whitespace.
#[must_use]
pub fn split_title_and_tags(line: &str) -> (String, Vec<Tag>) {
    if line.matches(':').count() < 2 {
        return (line.to_string(), vec![]);
    }

    let segments: Vec<&str> = line.split(':').collect();
    // Look for trailing :: (ends with colon)
    if line.ends_with(':') {
        // The last segment is empty (trailing colon)
        let rev_segs: Vec<&str> = segments.iter().rev().copied().collect();
        // Skip the empty trailing segment. `line` ends with ':' and contains
        // at least two ':', so `segments` (and `rev_segs`) has at least three
        // elements — `rev_segs[1..]` is always in bounds; the fallback empty
        // slice below only guards the unreachable short-input case.
        let after_trailing = rev_segs.get(1..).unwrap_or_default();
        if let Some(tag_end) = find_trailing_tags(after_trailing) {
            let tag_count = tag_end;
            // `tag_count <= rev_segs.len() - 1` (it counts tags found after the
            // trailing empty segment), so `title_seg_count` is in `0..segments.len()`.
            let title_seg_count = segments.len() - 1 - tag_count;
            let title_before = segments
                .get(..title_seg_count)
                .unwrap_or_default()
                .join(":");
            let tags: Vec<Tag> = segments
                .get(title_seg_count..segments.len() - 1)
                .unwrap_or_default()
                .iter()
                .map(|t| Tag((*t).to_string()))
                .collect();
            let ends_with_space = title_before
                .chars()
                .last()
                .is_some_and(|c| c == ' ' || c == '\t');
            if title_before.is_empty() || ends_with_space {
                return (title_before.trim_end().to_string(), tags);
            }
        }
    }
    (line.to_string(), vec![])
}

/// Mirror of Haskell `KB.AST.Canonical.isValidTag`: alphanumerics plus
/// `_`, `-`, `@`, `#`, `%`. Widened from the original `[A-Za-z0-9_]`
/// in T11-01.
fn is_valid_tag_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '-' | '@' | '#' | '%')
}

fn find_trailing_tags(rev_segs: &[&str]) -> Option<usize> {
    let mut tag_count = 0;
    for seg in rev_segs {
        if seg.is_empty() || !seg.chars().all(is_valid_tag_char) {
            break;
        }
        tag_count += 1;
    }
    if tag_count > 0 { Some(tag_count) } else { None }
}

/// Canonicalize a src-block body: non-empty bodies always end in `\n`,
/// empty body becomes `"\n"`.
fn canonicalize_src_body(body: &str) -> String {
    if body.is_empty() || body == "\n" {
        "\n".into()
    } else if body.ends_with('\n') {
        body.to_string()
    } else {
        format!("{body}\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalize_idempotent() {
        let doc = Document {
            blocks: vec![
                Block::Heading {
                    level: 1,
                    title: Title("  Hello  ".into()),
                    tags: vec![],
                    children: vec![],
                },
                Block::Paragraph {
                    inlines: vec![Inline::Plain("a".into()), Inline::Plain("b".into())],
                },
            ],
        };
        let c1 = canonicalize(&doc);
        let c2 = canonicalize(&c1);
        assert_eq!(c1, c2, "canonicalize must be idempotent");
    }

    #[test]
    fn empty_bold_becomes_plain_stars() {
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![Inline::Bold(vec![])],
            }],
        };
        let c = canonicalize(&doc);
        assert_eq!(
            c.blocks[0],
            Block::Paragraph {
                inlines: vec![Inline::Plain("**".into())]
            }
        );
    }

    #[test]
    fn empty_italic_becomes_plain_slashes() {
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![Inline::Italic(vec![])],
            }],
        };
        let c = canonicalize(&doc);
        assert_eq!(
            c.blocks[0],
            Block::Paragraph {
                inlines: vec![Inline::Plain("//".into())]
            }
        );
    }

    #[test]
    fn empty_inline_code_becomes_plain_equals() {
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![Inline::InlineCode(String::new())],
            }],
        };
        let c = canonicalize(&doc);
        assert_eq!(
            c.blocks[0],
            Block::Paragraph {
                inlines: vec![Inline::Plain("==".into())]
            }
        );
    }

    #[test]
    fn empty_verbatim_becomes_plain_tildes() {
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![Inline::Verbatim(String::new())],
            }],
        };
        let c = canonicalize(&doc);
        assert_eq!(
            c.blocks[0],
            Block::Paragraph {
                inlines: vec![Inline::Plain("~~".into())]
            }
        );
    }

    #[test]
    fn empty_link_becomes_plain_brackets() {
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![Inline::Link {
                    target: String::new(),
                    description: None,
                }],
            }],
        };
        let c = canonicalize(&doc);
        assert_eq!(
            c.blocks[0],
            Block::Paragraph {
                inlines: vec![Inline::Plain("[[]]".into())]
            }
        );
    }

    #[test]
    fn heading_title_trimmed() {
        let doc = Document {
            blocks: vec![Block::Heading {
                level: 1,
                title: Title("  Hello World  ".into()),
                tags: vec![],
                children: vec![],
            }],
        };
        let c = canonicalize(&doc);
        if let Block::Heading { title, .. } = &c.blocks[0] {
            assert_eq!(title.0, "Hello World");
        } else {
            panic!("expected heading");
        }
    }

    #[test]
    fn tag_charclass_extended_chars() {
        // Mirror of Haskell isValidTag (post-T11-01): alnum + _-@#%
        let cases = [
            ("Title :claude-memory:", "Title", vec!["claude-memory"]),
            ("Title :a@b:", "Title", vec!["a@b"]),
            ("Title :c#d:", "Title", vec!["c#d"]),
            ("Title :e%f:", "Title", vec!["e%f"]),
            (
                "Title :foo_bar:baz-qux:",
                "Title",
                vec!["foo_bar", "baz-qux"],
            ),
        ];
        for (line, expected_title, expected_tags) in cases {
            let (t, tags) = split_title_and_tags(line);
            assert_eq!(t, expected_title, "title mismatch for {line:?}");
            let got: Vec<&str> = tags.iter().map(|x| x.0.as_str()).collect();
            assert_eq!(got, expected_tags, "tags mismatch for {line:?}");
        }
    }

    #[test]
    fn tag_charclass_rejects_invalid() {
        // A space inside a "tag" segment must terminate tag detection.
        let (t, tags) = split_title_and_tags("Title :has space:");
        assert_eq!(t, "Title :has space:");
        assert!(tags.is_empty());
    }

    #[test]
    fn heading_with_trailing_tags() {
        let doc = Document {
            blocks: vec![Block::Heading {
                level: 1,
                title: Title("My Title :tag1:tag2:".into()),
                tags: vec![],
                children: vec![],
            }],
        };
        let c = canonicalize(&doc);
        if let Block::Heading { title, tags, .. } = &c.blocks[0] {
            assert_eq!(title.0, "My Title");
            assert_eq!(tags.len(), 2);
            assert_eq!(tags[0].0, "tag1");
            assert_eq!(tags[1].0, "tag2");
        } else {
            panic!("expected heading");
        }
    }

    #[test]
    fn src_body_always_ends_with_newline() {
        let doc = Document {
            blocks: vec![Block::SrcBlock {
                language: "rust".into(),
                content: "no newline".into(),
            }],
        };
        let c = canonicalize(&doc);
        if let Block::SrcBlock { content, .. } = &c.blocks[0] {
            assert!(content.ends_with('\n'));
        } else {
            panic!("expected src block");
        }
    }

    #[test]
    fn empty_paragraph_dropped() {
        let doc = Document {
            blocks: vec![
                Block::Heading {
                    level: 1,
                    title: Title("Title".into()),
                    tags: vec![],
                    children: vec![],
                },
                Block::Paragraph { inlines: vec![] },
            ],
        };
        let c = canonicalize(&doc);
        assert_eq!(c.blocks.len(), 1); // empty paragraph dropped
    }

    #[test]
    fn plain_merging() {
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![
                    Inline::Plain("a".into()),
                    Inline::Plain("b".into()),
                    Inline::Plain("c".into()),
                ],
            }],
        };
        let c = canonicalize(&doc);
        if let Block::Paragraph { inlines } = &c.blocks[0] {
            assert_eq!(inlines.len(), 1);
            assert_eq!(inlines[0], Inline::Plain("abc".into()));
        } else {
            panic!("expected paragraph");
        }
    }
}
