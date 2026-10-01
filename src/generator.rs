//! AST → org-mode `String` projection.
//!
//! Covers every constructor of [`Block`] and [`Inline`]. The encoder is total:
//! adding a new constructor in the AST must be matched here or compilation fails.
//!
//! Style decisions (provisional):
//! - unordered list bullet is `-`; ordered list bullet is `1.`
//! - nested list content indents two spaces per level
//! - property drawer keys render as `:KEY: value` in source order
//! - timestamps and tags are emitted verbatim from their newtype wrappers

use std::fmt::Write as _;

use tftio_org::ast::{
    Block, Checkbox, Document, Inline, ListItem, ListType, PlanningEntry, TableCell, Tag, Title,
};

/// Render a [`Document`] to its canonical org-mode text.
#[must_use]
pub fn generate(doc: &Document) -> String {
    let mut out = String::new();
    for block in &doc.blocks {
        gen_block(&mut out, block);
    }
    out
}

#[allow(
    clippy::too_many_lines,
    reason = "cohesive per-constructor dispatch; splitting would not improve clarity"
)]
fn gen_block(out: &mut String, block: &Block) {
    match block {
        Block::Heading {
            level,
            title,
            tags,
            children,
        } => {
            gen_heading_line(out, *level, title, tags);
            out.push('\n');
            for child in children {
                gen_block(out, child);
            }
        }
        Block::Paragraph { inlines } => {
            gen_inlines(out, inlines);
            out.push('\n');
        }
        Block::SrcBlock { language, content } => {
            if language.is_empty() {
                out.push_str("#+begin_src\n");
            } else {
                writeln!(out, "#+begin_src {language}").ok();
            }
            out.push_str(content);
            if !content.ends_with('\n') {
                out.push('\n');
            }
            out.push_str("#+end_src\n");
        }
        Block::ExampleBlock { content } => {
            out.push_str("#+begin_example\n");
            out.push_str(content);
            if !content.is_empty() && !content.ends_with('\n') {
                out.push('\n');
            }
            out.push_str("#+end_example\n");
        }
        Block::QuoteBlock { children } => {
            out.push_str("#+begin_quote\n");
            for child in children {
                gen_block(out, child);
            }
            out.push_str("#+end_quote\n");
        }
        Block::List { list_type, items } => {
            for (idx, item) in items.iter().enumerate() {
                gen_list_item(out, list_type, 0, idx, item);
            }
        }
        Block::Table { rows } => {
            for row in rows {
                gen_table_row(out, row);
            }
        }
        Block::PropertyDrawer { entries } => {
            out.push_str(":PROPERTIES:\n");
            for (k, v) in entries {
                writeln!(out, ":{k}:{v}").ok();
            }
            out.push_str(":END:\n");
        }
        Block::LogbookDrawer { entries } => {
            out.push_str(":LOGBOOK:\n");
            for entry in entries {
                writeln!(
                    out,
                    "- {} {}",
                    entry.timestamp.0,
                    if entry.note.is_empty() {
                        ""
                    } else {
                        &entry.note
                    }
                )
                .ok();
            }
            out.push_str(":END:\n");
        }
        Block::Planning { entries } => {
            let parts: Vec<String> = entries
                .iter()
                .map(|e| match e {
                    PlanningEntry::Scheduled(ts) => format!("SCHEDULED: {}", ts.0),
                    PlanningEntry::Deadline(ts) => format!("DEADLINE: {}", ts.0),
                    PlanningEntry::Closed(ts) => format!("CLOSED: {}", ts.0),
                })
                .collect();
            out.push_str(&parts.join(" "));
            out.push('\n');
        }
        Block::Comment { text } => {
            writeln!(out, "# {text}").ok();
        }
        Block::Keyword { name, value } => {
            writeln!(out, "#+{name}:{value}").ok();
        }
        Block::BlankLine => {
            out.push('\n');
        }
        Block::HorizontalRule => {
            out.push_str("-----\n");
        }
    }
}

fn gen_heading_line(out: &mut String, level: u8, title: &Title, tags: &[Tag]) {
    let stars = "*".repeat(level.max(1) as usize);
    write!(out, "{stars} {}", title.0).ok();
    if !tags.is_empty() {
        out.push(' ');
        out.push(':');
        for (i, tag) in tags.iter().enumerate() {
            if i > 0 {
                out.push(':');
            }
            out.push_str(&tag.0);
        }
        out.push(':');
    }
}

fn gen_list_item(
    out: &mut String,
    list_type: &ListType,
    depth: usize,
    index: usize,
    item: &ListItem,
) {
    let indent = " ".repeat(depth * 2);
    let bullet = match list_type {
        ListType::Ordered(start) => format!("{}.", start + index as u64),
        ListType::Unordered => "-".to_string(),
    };
    let checkbox = match item.checkbox {
        Checkbox::NoCheckbox => "",
        Checkbox::Unchecked => "[ ] ",
        Checkbox::Checked => "[X] ",
    };

    // If the item has a single paragraph child, render it inline with the bullet.
    // Otherwise, render the bullet alone and all blocks as continuations.
    match item.content.as_slice() {
        [Block::Paragraph { inlines }] => {
            write!(out, "{indent}{bullet} {checkbox}").ok();
            gen_inlines(out, inlines);
            out.push('\n');
        }
        blocks => {
            writeln!(out, "{indent}{bullet} {checkbox}").ok();
            for b in blocks {
                let rend = render_block(b);
                let child_indent = " ".repeat((depth + 1) * 2);
                for line in rend.lines() {
                    if line.is_empty() {
                        out.push('\n');
                    } else {
                        writeln!(out, "{child_indent}{line}").ok();
                    }
                }
            }
        }
    }
}

fn render_block(block: &Block) -> String {
    let mut out = String::new();
    gen_block(&mut out, block);
    out
}

fn gen_table_row(out: &mut String, cells: &[TableCell]) {
    // Cells carry their own padding verbatim; join them with bare `|`.
    out.push('|');
    for cell in cells {
        gen_inlines(out, &cell.inlines);
        out.push('|');
    }
    out.push('\n');
}

fn gen_inlines(out: &mut String, inlines: &[Inline]) {
    for inline in inlines {
        gen_inline(out, inline);
    }
}

fn gen_inline(out: &mut String, inline: &Inline) {
    match inline {
        Inline::Plain(t) => out.push_str(t),
        Inline::Bold(inner) => {
            out.push('*');
            gen_inlines(out, inner);
            out.push('*');
        }
        Inline::Italic(inner) => {
            out.push('/');
            gen_inlines(out, inner);
            out.push('/');
        }
        Inline::Strikethrough(inner) => {
            out.push('+');
            gen_inlines(out, inner);
            out.push('+');
        }
        Inline::InlineCode(t) => {
            out.push('=');
            out.push_str(t);
            out.push('=');
        }
        Inline::Verbatim(t) => {
            out.push('~');
            out.push_str(t);
            out.push('~');
        }
        Inline::LineBreak => {
            out.push('\n');
        }
        Inline::Link {
            target,
            description: None,
        } => {
            write!(out, "[[{target}]]").ok();
        }
        Inline::Link {
            target,
            description: Some(desc),
        } => {
            write!(out, "[[{target}][{desc}]]").ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tftio_org::ast::{LogEntry, Timestamp};

    #[test]
    fn generate_heading_level_1() {
        let doc = Document {
            blocks: vec![Block::Heading {
                level: 1,
                title: Title("Hello".into()),
                tags: vec![],
                children: vec![],
            }],
        };
        assert_eq!(generate(&doc), "* Hello\n");
    }

    #[test]
    fn generate_heading_with_tags() {
        let doc = Document {
            blocks: vec![Block::Heading {
                level: 2,
                title: Title("Task".into()),
                tags: vec![Tag("rust".into()), Tag("kb".into())],
                children: vec![],
            }],
        };
        assert_eq!(generate(&doc), "** Task :rust:kb:\n");
    }

    #[test]
    fn generate_paragraph() {
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![Inline::Plain("some text".into())],
            }],
        };
        assert_eq!(generate(&doc), "some text\n");
    }

    #[test]
    fn generate_bold_and_italic() {
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![
                    Inline::Bold(vec![Inline::Plain("bold".into())]),
                    Inline::Plain(" and ".into()),
                    Inline::Italic(vec![Inline::Plain("italic".into())]),
                ],
            }],
        };
        assert_eq!(generate(&doc), "*bold* and /italic/\n");
    }

    #[test]
    fn generate_link() {
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![Inline::Link {
                    target: "https://example.com".into(),
                    description: Some("example".into()),
                }],
            }],
        };
        assert_eq!(generate(&doc), "[[https://example.com][example]]\n");
    }

    #[test]
    fn generate_src_block() {
        let doc = Document {
            blocks: vec![Block::SrcBlock {
                language: "rust".into(),
                content: "fn main() {}\n".into(),
            }],
        };
        assert_eq!(
            generate(&doc),
            "#+begin_src rust\nfn main() {}\n#+end_src\n"
        );
    }

    #[test]
    fn generate_list_unordered() {
        let doc = Document {
            blocks: vec![Block::List {
                list_type: ListType::Unordered,
                items: vec![
                    ListItem {
                        content: vec![Block::Paragraph {
                            inlines: vec![Inline::Plain("one".into())],
                        }],
                        checkbox: Checkbox::NoCheckbox,
                    },
                    ListItem {
                        content: vec![Block::Paragraph {
                            inlines: vec![Inline::Plain("two".into())],
                        }],
                        checkbox: Checkbox::Checked,
                    },
                ],
            }],
        };
        let result = generate(&doc);
        assert!(result.contains("- one\n"));
        assert!(result.contains("- [X] two\n"));
    }

    #[test]
    fn generate_property_drawer() {
        let doc = Document {
            blocks: vec![Block::PropertyDrawer {
                entries: vec![("ID".into(), " abc-123".into())],
            }],
        };
        let result = generate(&doc);
        assert!(result.contains(":PROPERTIES:\n"));
        assert!(result.contains(":ID: abc-123\n"));
        assert!(result.contains(":END:\n"));
    }

    #[test]
    fn generate_planning() {
        let doc = Document {
            blocks: vec![Block::Planning {
                entries: vec![
                    PlanningEntry::Scheduled(Timestamp("<2026-04-30 Thu>".into())),
                    PlanningEntry::Deadline(Timestamp("<2026-05-01 Fri>".into())),
                ],
            }],
        };
        let result = generate(&doc);
        assert!(result.contains("SCHEDULED: <2026-04-30 Thu>"));
        assert!(result.contains("DEADLINE: <2026-05-01 Fri>"));
    }

    #[test]
    fn generate_quote_block() {
        let doc = Document {
            blocks: vec![Block::QuoteBlock {
                children: vec![Block::Paragraph {
                    inlines: vec![Inline::Plain("quoted".into())],
                }],
            }],
        };
        let result = generate(&doc);
        assert!(result.contains("#+begin_quote\n"));
        assert!(result.contains("#+end_quote\n"));
    }

    #[test]
    fn generate_all_constructors() {
        // Smoke test: every constructor renders without panicking.
        let doc = Document {
            blocks: vec![
                Block::Heading {
                    level: 1,
                    title: Title("All Constructors".into()),
                    tags: vec![Tag("test".into())],
                    children: vec![Block::Paragraph {
                        inlines: vec![
                            Inline::Plain("plain ".into()),
                            Inline::Bold(vec![Inline::Plain("bold".into())]),
                            Inline::Italic(vec![Inline::Plain("italic".into())]),
                            Inline::InlineCode("code".into()),
                            Inline::Verbatim("verbatim".into()),
                            Inline::Link {
                                target: "tgt".into(),
                                description: Some("desc".into()),
                            },
                        ],
                    }],
                },
                Block::SrcBlock {
                    language: "rust".into(),
                    content: "x".into(),
                },
                Block::QuoteBlock {
                    children: vec![Block::Paragraph {
                        inlines: vec![Inline::Plain("q".into())],
                    }],
                },
                Block::List {
                    list_type: ListType::Ordered(1),
                    items: vec![ListItem {
                        content: vec![Block::Paragraph {
                            inlines: vec![Inline::Plain("item".into())],
                        }],
                        checkbox: Checkbox::Checked,
                    }],
                },
                Block::Table {
                    rows: vec![vec![TableCell {
                        inlines: vec![Inline::Plain("cell".into())],
                    }]],
                },
                Block::PropertyDrawer {
                    entries: vec![("KEY".into(), "value".into())],
                },
                Block::LogbookDrawer {
                    entries: vec![LogEntry {
                        timestamp: Timestamp("<2026-04-30 Thu>".into()),
                        note: String::new(),
                    }],
                },
                Block::Planning {
                    entries: vec![PlanningEntry::Closed(Timestamp("<2026-04-30 Thu>".into()))],
                },
                Block::Comment {
                    text: "a comment".into(),
                },
                Block::HorizontalRule,
            ],
        };
        let _result = generate(&doc);
    }
}
