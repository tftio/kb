//! S-expression encoding for [`Document`] matching Haskell `KB/AST/Sexp.hs`.
//!
//! Format: `(kb-doc 1 block…)`. A future migration would bump the version
//! number and add a decoder for the new shape, or keep accepting older
//! versions in parallel.
//!
//! ## Format reference
//!
//! ```text
//! (kb-doc 1 block …)
//!
//! block ::= (heading int str (tags str …) (children block …))
//!         | (paragraph inline …)
//!         | (src-block str-lang str-body)
//!         | (example-block str-body)
//!         | (quote-block block …)
//!         | (list (ordered int)|unordered list-item …)   ; bare `ordered` accepted on decode (legacy, = start 1)
//!         | (table row …)
//!         | (property-drawer (prop str-key str-val) …)
//!         | (logbook-drawer (log str-ts str-note) …)
//!         | (planning (scheduled str-ts) | (deadline str-ts) | (closed str-ts) …)
//!         | (comment str)
//!         | (keyword str-name str-value)
//!         | (blank)
//!         | (rule)
//!
//! list-item ::= (item no-checkbox|unchecked|checked block …)
//! row       ::= (row (cell inline …) …)
//! inline    ::= (plain str)
//!             | (bold inline …)
//!             | (italic inline …)
//!             | (code str)
//!             | (verbatim str)
//!             | (break)
//!             | (link str-target [str-desc])
//! ```

use std::fmt::Write as _;

use tftio_org::ast::{
    Block, Checkbox, Document, Inline, ListItem, ListType, LogEntry, PlanningEntry, TableCell, Tag,
    Timestamp, Title,
};

/// Why a decode failed.
#[derive(Debug, Clone)]
pub enum SexpError {
    /// The s-expression was malformed (unbalanced parens, bad escape).
    Parse(String),
    /// The s-expression parsed but did not match an AST shape.
    Malformed(String),
}

impl std::fmt::Display for SexpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(msg) => write!(f, "parse error: {msg}"),
            Self::Malformed(msg) => write!(f, "malformed: {msg}"),
        }
    }
}

impl std::error::Error for SexpError {}

/// Render a [`Document`] as the canonical s-expression text.
#[must_use]
pub fn encode_document(doc: &Document) -> String {
    let mut out = String::from("(kb-doc 1");
    for block in &doc.blocks {
        out.push(' ');
        encode_block(&mut out, block);
    }
    out.push(')');
    out
}

/// Parse and decode an s-expression string into a [`Document`].
///
/// # Errors
///
/// Returns [`SexpError`] on parse or structural problems.
pub fn decode_document(input: &str) -> Result<Document, SexpError> {
    let sx = parse_sexp(input)?;
    decode_doc(&sx)
}

// ── Internal s-expression representation ────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Sexp {
    Sym(String),
    Str(String),
    Num(i64),
    Lst(Vec<Self>),
}

// ── Encoding ────────────────────────────────────────────────────────────

#[allow(
    clippy::too_many_lines,
    reason = "cohesive per-constructor s-expression encoder; splitting would not improve clarity"
)]
fn encode_block(out: &mut String, block: &Block) {
    match block {
        Block::Heading {
            level,
            title,
            tags,
            children,
        } => {
            write!(out, "(heading {level} ").ok();
            encode_str(out, &title.0);
            out.push_str(" (tags");
            for tag in tags {
                out.push(' ');
                out.push_str(&tag.0);
            }
            out.push_str(") (children");
            for child in children {
                out.push(' ');
                encode_block(out, child);
            }
            out.push(')');
            out.push(')');
        }
        Block::Paragraph { inlines } => {
            out.push_str("(paragraph");
            for inline in inlines {
                out.push(' ');
                encode_inline(out, inline);
            }
            out.push(')');
        }
        Block::SrcBlock { language, content } => {
            out.push_str("(src-block ");
            encode_str(out, language);
            out.push(' ');
            encode_str(out, content);
            out.push(')');
        }
        Block::ExampleBlock { content } => {
            out.push_str("(example-block ");
            encode_str(out, content);
            out.push(')');
        }
        Block::QuoteBlock { children } => {
            out.push_str("(quote-block");
            for child in children {
                out.push(' ');
                encode_block(out, child);
            }
            out.push(')');
        }
        Block::List { list_type, items } => {
            out.push_str("(list ");
            match list_type {
                ListType::Ordered(start) => {
                    write!(out, "(ordered {start})").ok();
                }
                ListType::Unordered => out.push_str("unordered"),
            }
            for item in items {
                out.push(' ');
                encode_list_item(out, item);
            }
            out.push(')');
        }
        Block::Table { rows } => {
            out.push_str("(table");
            for row in rows {
                out.push_str(" (row");
                for cell in row {
                    out.push_str(" (cell");
                    for inline in &cell.inlines {
                        out.push(' ');
                        encode_inline(out, inline);
                    }
                    out.push(')');
                }
                out.push(')');
            }
            out.push(')');
        }
        Block::PropertyDrawer { entries } => {
            out.push_str("(property-drawer");
            for (k, v) in entries {
                out.push_str(" (prop ");
                encode_str(out, k);
                out.push(' ');
                encode_str(out, v);
                out.push(')');
            }
            out.push(')');
        }
        Block::LogbookDrawer { entries } => {
            out.push_str("(logbook-drawer");
            for entry in entries {
                out.push_str(" (log ");
                encode_str(out, &entry.timestamp.0);
                out.push(' ');
                encode_str(out, &entry.note);
                out.push(')');
            }
            out.push(')');
        }
        Block::Planning { entries } => {
            out.push_str("(planning");
            for entry in entries {
                out.push(' ');
                encode_planning_entry(out, entry);
            }
            out.push(')');
        }
        Block::Comment { text } => {
            out.push_str("(comment ");
            encode_str(out, text);
            out.push(')');
        }
        Block::Keyword { name, value } => {
            out.push_str("(keyword ");
            encode_str(out, name);
            out.push(' ');
            encode_str(out, value);
            out.push(')');
        }
        Block::BlankLine => {
            out.push_str("(blank)");
        }
        Block::HorizontalRule => {
            out.push_str("(rule)");
        }
    }
}

fn encode_inline(out: &mut String, inline: &Inline) {
    match inline {
        Inline::Plain(t) => {
            out.push_str("(plain ");
            encode_str(out, t);
            out.push(')');
        }
        Inline::Bold(inner) => {
            out.push_str("(bold");
            for i in inner {
                out.push(' ');
                encode_inline(out, i);
            }
            out.push(')');
        }
        Inline::Italic(inner) => {
            out.push_str("(italic");
            for i in inner {
                out.push(' ');
                encode_inline(out, i);
            }
            out.push(')');
        }
        Inline::Strikethrough(inner) => {
            out.push_str("(strikethrough");
            for i in inner {
                out.push(' ');
                encode_inline(out, i);
            }
            out.push(')');
        }
        Inline::InlineCode(t) => {
            out.push_str("(code ");
            encode_str(out, t);
            out.push(')');
        }
        Inline::Verbatim(t) => {
            out.push_str("(verbatim ");
            encode_str(out, t);
            out.push(')');
        }
        Inline::LineBreak => {
            out.push_str("(break)");
        }
        Inline::Link {
            target,
            description,
        } => {
            out.push_str("(link ");
            encode_str(out, target);
            if let Some(desc) = description {
                out.push(' ');
                encode_str(out, desc);
            }
            out.push(')');
        }
    }
}

fn encode_list_item(out: &mut String, item: &ListItem) {
    out.push_str("(item ");
    match item.checkbox {
        Checkbox::NoCheckbox => out.push_str("no-checkbox"),
        Checkbox::Unchecked => out.push_str("unchecked"),
        Checkbox::Checked => out.push_str("checked"),
    }
    for block in &item.content {
        out.push(' ');
        encode_block(out, block);
    }
    out.push(')');
}

fn encode_planning_entry(out: &mut String, entry: &PlanningEntry) {
    match entry {
        PlanningEntry::Scheduled(ts) => {
            out.push_str("(scheduled ");
            encode_str(out, &ts.0);
            out.push(')');
        }
        PlanningEntry::Deadline(ts) => {
            out.push_str("(deadline ");
            encode_str(out, &ts.0);
            out.push(')');
        }
        PlanningEntry::Closed(ts) => {
            out.push_str("(closed ");
            encode_str(out, &ts.0);
            out.push(')');
        }
    }
}

fn encode_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
}

// ── Parsing ─────────────────────────────────────────────────────────────

fn parse_sexp(input: &str) -> Result<Sexp, SexpError> {
    let chars: Vec<char> = input.chars().collect();
    let mut pos = 0;
    skip_whitespace(&chars, &mut pos);
    let sx = parse_one(&chars, &mut pos)?;
    skip_whitespace(&chars, &mut pos);
    if pos < chars.len() {
        return Err(SexpError::Parse(format!(
            "trailing input at position {pos}"
        )));
    }
    Ok(sx)
}

fn parse_one(chars: &[char], pos: &mut usize) -> Result<Sexp, SexpError> {
    skip_whitespace(chars, pos);
    // The guard above guarantees `*pos < chars.len()`, so this is always Some;
    // `None` reproduces the end-of-input error path.
    let Some(&first) = chars.get(*pos) else {
        return Err(SexpError::Parse("unexpected end of input".into()));
    };
    match first {
        '(' => {
            *pos += 1;
            let mut items = Vec::new();
            loop {
                skip_whitespace(chars, pos);
                if *pos >= chars.len() {
                    return Err(SexpError::Parse("unterminated list".into()));
                }
                if chars.get(*pos) == Some(&')') {
                    *pos += 1;
                    return Ok(Sexp::Lst(items));
                }
                items.push(parse_one(chars, pos)?);
            }
        }
        ')' => Err(SexpError::Parse("unexpected closing paren".into())),
        '"' => {
            *pos += 1;
            let mut s = String::new();
            loop {
                let Some(&c) = chars.get(*pos) else {
                    return Err(SexpError::Parse("unterminated string".into()));
                };
                match c {
                    '"' => {
                        *pos += 1;
                        return Ok(Sexp::Str(s));
                    }
                    '\\' => {
                        *pos += 1;
                        let Some(&esc) = chars.get(*pos) else {
                            return Err(SexpError::Parse("unterminated escape".into()));
                        };
                        match esc {
                            '"' => s.push('"'),
                            '\\' => s.push('\\'),
                            'n' => s.push('\n'),
                            'r' => s.push('\r'),
                            't' => s.push('\t'),
                            c => return Err(SexpError::Parse(format!("unknown escape: \\{c}"))),
                        }
                        *pos += 1;
                    }
                    c => {
                        s.push(c);
                        *pos += 1;
                    }
                }
            }
        }
        '-' | '0'..='9' => {
            let start = *pos;
            if chars.get(*pos) == Some(&'-') {
                *pos += 1;
            }
            while *pos < chars.len() && chars.get(*pos).is_some_and(char::is_ascii_digit) {
                *pos += 1;
            }
            let num_str: String = chars.get(start..*pos).unwrap_or_default().iter().collect();
            Ok(Sexp::Num(num_str.parse().map_err(
                |e: std::num::ParseIntError| SexpError::Parse(e.to_string()),
            )?))
        }
        _ => {
            let start = *pos;
            while *pos < chars.len()
                && chars
                    .get(*pos)
                    .is_some_and(|c| !c.is_whitespace() && *c != '(' && *c != ')')
            {
                *pos += 1;
            }
            let sym: String = chars.get(start..*pos).unwrap_or_default().iter().collect();
            Ok(Sexp::Sym(sym))
        }
    }
}

fn skip_whitespace(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars.get(*pos).is_some_and(|c| c.is_whitespace()) {
        *pos += 1;
    }
}

// ── Decoding (Sexp → AST) ───────────────────────────────────────────────

fn decode_doc(sx: &Sexp) -> Result<Document, SexpError> {
    match sx {
        Sexp::Lst(items) if !items.is_empty() => match items.first() {
            Some(Sexp::Sym(s)) if s == "kb-doc" => {
                let Some(version) = items.get(1) else {
                    return Err(SexpError::Malformed("expected (kb-doc <version> …)".into()));
                };
                match version {
                    Sexp::Num(1) => {
                        let blocks: Result<Vec<Block>, _> = items
                            .get(2..)
                            .unwrap_or(&[])
                            .iter()
                            .map(decode_block)
                            .collect();
                        Ok(Document { blocks: blocks? })
                    }
                    Sexp::Num(v) => Err(SexpError::Malformed(format!(
                        "unsupported kb-doc version {v}"
                    ))),
                    _ => Err(SexpError::Malformed(
                        "kb-doc version must be a number".into(),
                    )),
                }
            }
            _ => Err(SexpError::Malformed("expected (kb-doc <version> …)".into())),
        },
        _ => Err(SexpError::Malformed("expected (kb-doc <version> …)".into())),
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "cohesive per-tag block decoder; splitting would not improve clarity"
)]
fn decode_block(sx: &Sexp) -> Result<Block, SexpError> {
    match sx {
        Sexp::Lst(items) if !items.is_empty() => match items.first() {
            Some(Sexp::Sym(tag)) => match tag.as_str() {
                "heading" => {
                    if items.len() < 5 {
                        return malformed("heading needs level, title, tags, children");
                    }
                    let level = u8::try_from(
                        decode_num(nth(items, 1, "heading needs level")?)?
                            .min(255)
                            .clamp(1, 255),
                    )
                    .unwrap_or(1);
                    let title = Title(decode_str(nth(items, 2, "heading needs title")?)?);
                    let tags = decode_tag_list(nth(items, 3, "heading needs tags")?)?;
                    let children = decode_child_list(nth(items, 4, "heading needs children")?)?;
                    Ok(Block::Heading {
                        level,
                        title,
                        tags,
                        children,
                    })
                }
                "paragraph" => {
                    let inlines: Result<Vec<Inline>, _> =
                        rest(items, 1).iter().map(decode_inline).collect();
                    Ok(Block::Paragraph { inlines: inlines? })
                }
                "src-block" => {
                    if items.len() < 3 {
                        return malformed("src-block needs language and body");
                    }
                    Ok(Block::SrcBlock {
                        language: decode_str(nth(items, 1, "src-block needs language")?)?,
                        content: decode_str(nth(items, 2, "src-block needs body")?)?,
                    })
                }
                "example-block" => {
                    if items.len() < 2 {
                        return malformed("example-block needs body");
                    }
                    Ok(Block::ExampleBlock {
                        content: decode_str(nth(items, 1, "example-block needs body")?)?,
                    })
                }
                "quote-block" => {
                    let children: Result<Vec<Block>, _> =
                        rest(items, 1).iter().map(decode_block).collect();
                    Ok(Block::QuoteBlock {
                        children: children?,
                    })
                }
                "list" => {
                    if items.len() < 3 {
                        return malformed("list needs list-type and items");
                    }
                    let list_type = decode_list_type(nth(items, 1, "list needs list-type")?)?;
                    let list_items: Result<Vec<ListItem>, _> =
                        rest(items, 2).iter().map(decode_list_item).collect();
                    Ok(Block::List {
                        list_type,
                        items: list_items?,
                    })
                }
                "table" => {
                    let rows: Result<Vec<Vec<TableCell>>, _> =
                        rest(items, 1).iter().map(decode_row).collect();
                    Ok(Block::Table { rows: rows? })
                }
                "property-drawer" => {
                    let entries: Result<Vec<(String, String)>, _> =
                        rest(items, 1).iter().map(decode_prop).collect();
                    Ok(Block::PropertyDrawer { entries: entries? })
                }
                "logbook-drawer" => {
                    let entries: Result<Vec<LogEntry>, _> =
                        rest(items, 1).iter().map(decode_log).collect();
                    Ok(Block::LogbookDrawer { entries: entries? })
                }
                "planning" => {
                    let entries: Result<Vec<PlanningEntry>, _> =
                        rest(items, 1).iter().map(decode_planning_entry).collect();
                    Ok(Block::Planning { entries: entries? })
                }
                "comment" => {
                    if items.len() < 2 {
                        return malformed("comment needs text");
                    }
                    Ok(Block::Comment {
                        text: decode_str(nth(items, 1, "comment needs text")?)?,
                    })
                }
                "keyword" => {
                    if items.len() < 3 {
                        return malformed("keyword needs name and value");
                    }
                    Ok(Block::Keyword {
                        name: decode_str(nth(items, 1, "keyword needs name")?)?,
                        value: decode_str(nth(items, 2, "keyword needs value")?)?,
                    })
                }
                "blank" => Ok(Block::BlankLine),
                "rule" => Ok(Block::HorizontalRule),
                _ => malformed(&format!("unrecognized block tag: {tag}")),
            },
            _ => malformed("block must start with a symbol"),
        },
        _ => malformed("expected list for block"),
    }
}

fn decode_inline(sx: &Sexp) -> Result<Inline, SexpError> {
    match sx {
        Sexp::Lst(items) if !items.is_empty() => match items.first() {
            Some(Sexp::Sym(tag)) => match tag.as_str() {
                "plain" => {
                    if items.len() < 2 {
                        return malformed("plain needs text");
                    }
                    Ok(Inline::Plain(decode_str(nth(
                        items,
                        1,
                        "plain needs text",
                    )?)?))
                }
                "bold" => {
                    let inner: Result<Vec<Inline>, _> =
                        rest(items, 1).iter().map(decode_inline).collect();
                    Ok(Inline::Bold(inner?))
                }
                "italic" => {
                    let inner: Result<Vec<Inline>, _> =
                        rest(items, 1).iter().map(decode_inline).collect();
                    Ok(Inline::Italic(inner?))
                }
                "strikethrough" => {
                    let inner: Result<Vec<Inline>, _> =
                        rest(items, 1).iter().map(decode_inline).collect();
                    Ok(Inline::Strikethrough(inner?))
                }
                "code" => {
                    if items.len() < 2 {
                        return malformed("code needs text");
                    }
                    Ok(Inline::InlineCode(decode_str(nth(
                        items,
                        1,
                        "code needs text",
                    )?)?))
                }
                "verbatim" => {
                    if items.len() < 2 {
                        return malformed("verbatim needs text");
                    }
                    Ok(Inline::Verbatim(decode_str(nth(
                        items,
                        1,
                        "verbatim needs text",
                    )?)?))
                }
                "break" => Ok(Inline::LineBreak),
                "link" => {
                    if items.len() < 2 {
                        return malformed("link needs target");
                    }
                    let target = decode_str(nth(items, 1, "link needs target")?)?;
                    let description = if items.len() > 2 {
                        Some(decode_str(nth(items, 2, "link description")?)?)
                    } else {
                        None
                    };
                    Ok(Inline::Link {
                        target,
                        description,
                    })
                }
                _ => malformed(&format!("unrecognized inline tag: {tag}")),
            },
            _ => malformed("inline must start with a symbol"),
        },
        _ => malformed("expected list for inline"),
    }
}

fn decode_tag_list(sx: &Sexp) -> Result<Vec<Tag>, SexpError> {
    match sx {
        Sexp::Lst(items) => {
            if !matches!(items.first(), Some(Sexp::Sym(s)) if s == "tags") {
                return malformed("expected (tags …)");
            }
            rest(items, 1)
                .iter()
                .map(|sx| match sx {
                    Sexp::Sym(t) => Ok(Tag(t.clone())),
                    _ => malformed("tag must be a symbol"),
                })
                .collect()
        }
        _ => malformed("expected list for tags"),
    }
}

fn decode_child_list(sx: &Sexp) -> Result<Vec<Block>, SexpError> {
    match sx {
        Sexp::Lst(items) => {
            if !matches!(items.first(), Some(Sexp::Sym(s)) if s == "children") {
                return malformed("expected (children …)");
            }
            rest(items, 1).iter().map(decode_block).collect()
        }
        _ => malformed("expected list for children"),
    }
}

fn decode_list_type(sx: &Sexp) -> Result<ListType, SexpError> {
    match sx {
        Sexp::Sym(s) if s == "unordered" => Ok(ListType::Unordered),
        // Blobs written before ordered lists carried a start number encode
        // the list type as a bare `ordered` symbol; org's default start is 1.
        // Accepted on decode only — the encoder always writes `(ordered N)`.
        Sexp::Sym(s) if s == "ordered" => Ok(ListType::Ordered(1)),
        Sexp::Lst(items)
            if items.len() == 2
                && matches!(items.first(), Some(Sexp::Sym(s)) if s == "ordered") =>
        {
            let start = u64::try_from(decode_num(nth(items, 1, "ordered needs start")?)?)
                .map_err(|_| SexpError::Malformed("ordered start must be non-negative".into()))?;
            Ok(ListType::Ordered(start))
        }
        _ => malformed("expected (ordered N) or unordered"),
    }
}

fn decode_list_item(sx: &Sexp) -> Result<ListItem, SexpError> {
    match sx {
        Sexp::Lst(items) if items.len() >= 2 => match items.first() {
            Some(Sexp::Sym(s)) if s == "item" => {
                let checkbox = decode_checkbox(nth(items, 1, "item needs checkbox")?)?;
                let content: Result<Vec<Block>, _> =
                    rest(items, 2).iter().map(decode_block).collect();
                Ok(ListItem {
                    content: content?,
                    checkbox,
                })
            }
            _ => malformed("expected (item …)"),
        },
        _ => malformed("expected (item …)"),
    }
}

fn decode_checkbox(sx: &Sexp) -> Result<Checkbox, SexpError> {
    match sx {
        Sexp::Sym(s) if s == "no-checkbox" => Ok(Checkbox::NoCheckbox),
        Sexp::Sym(s) if s == "unchecked" => Ok(Checkbox::Unchecked),
        Sexp::Sym(s) if s == "checked" => Ok(Checkbox::Checked),
        _ => malformed("expected checkbox state"),
    }
}

fn decode_row(sx: &Sexp) -> Result<Vec<TableCell>, SexpError> {
    match sx {
        Sexp::Lst(items)
            if !items.is_empty() && matches!(items.first(), Some(Sexp::Sym(s)) if s == "row") =>
        {
            rest(items, 1).iter().map(decode_cell).collect()
        }
        _ => malformed("expected (row …)"),
    }
}

fn decode_cell(sx: &Sexp) -> Result<TableCell, SexpError> {
    match sx {
        Sexp::Lst(items)
            if !items.is_empty() && matches!(items.first(), Some(Sexp::Sym(s)) if s == "cell") =>
        {
            let inlines: Result<Vec<Inline>, _> =
                rest(items, 1).iter().map(decode_inline).collect();
            Ok(TableCell { inlines: inlines? })
        }
        _ => malformed("expected (cell …)"),
    }
}

fn decode_prop(sx: &Sexp) -> Result<(String, String), SexpError> {
    match sx {
        Sexp::Lst(items)
            if items.len() >= 3 && matches!(items.first(), Some(Sexp::Sym(s)) if s == "prop") =>
        {
            Ok((
                decode_str(nth(items, 1, "prop needs key")?)?,
                decode_str(nth(items, 2, "prop needs val")?)?,
            ))
        }
        _ => malformed("expected (prop key val)"),
    }
}

fn decode_log(sx: &Sexp) -> Result<LogEntry, SexpError> {
    match sx {
        Sexp::Lst(items)
            if items.len() >= 3 && matches!(items.first(), Some(Sexp::Sym(s)) if s == "log") =>
        {
            Ok(LogEntry {
                timestamp: Timestamp(decode_str(nth(items, 1, "log needs ts")?)?),
                note: decode_str(nth(items, 2, "log needs note")?)?,
            })
        }
        _ => malformed("expected (log ts note)"),
    }
}

fn decode_planning_entry(sx: &Sexp) -> Result<PlanningEntry, SexpError> {
    match sx {
        Sexp::Lst(items) if items.len() >= 2 => match items.first() {
            Some(Sexp::Sym(s)) if s == "scheduled" => Ok(PlanningEntry::Scheduled(Timestamp(
                decode_str(nth(items, 1, "scheduled needs ts")?)?,
            ))),
            Some(Sexp::Sym(s)) if s == "deadline" => Ok(PlanningEntry::Deadline(Timestamp(
                decode_str(nth(items, 1, "deadline needs ts")?)?,
            ))),
            Some(Sexp::Sym(s)) if s == "closed" => Ok(PlanningEntry::Closed(Timestamp(
                decode_str(nth(items, 1, "closed needs ts")?)?,
            ))),
            _ => malformed("expected planning entry type"),
        },
        _ => malformed("expected planning entry list"),
    }
}

fn decode_num(sx: &Sexp) -> Result<i64, SexpError> {
    match sx {
        Sexp::Num(n) => Ok(*n),
        _ => malformed("expected number"),
    }
}

fn decode_str(sx: &Sexp) -> Result<String, SexpError> {
    match sx {
        Sexp::Str(s) => Ok(s.clone()),
        _ => malformed("expected string"),
    }
}

fn malformed<T>(msg: &str) -> Result<T, SexpError> {
    Err(SexpError::Malformed(msg.to_string()))
}

/// Index into a decoded list, reporting a malformed error when the index is
/// out of range. Callers verify arity first, so the error path is defensive.
fn nth<'a>(items: &'a [Sexp], n: usize, msg: &str) -> Result<&'a Sexp, SexpError> {
    items
        .get(n)
        .ok_or_else(|| SexpError::Malformed(msg.to_string()))
}

/// The tail of a decoded list from index `n`, empty when out of range.
fn rest(items: &[Sexp], n: usize) -> &[Sexp] {
    items.get(n..).unwrap_or(&[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_simple_document() {
        let doc = Document {
            blocks: vec![
                Block::Heading {
                    level: 1,
                    title: Title("Hello".into()),
                    tags: vec![Tag("test".into())],
                    children: vec![Block::Paragraph {
                        inlines: vec![
                            Inline::Plain("some ".into()),
                            Inline::Bold(vec![Inline::Plain("bold".into())]),
                            Inline::Plain(" text".into()),
                        ],
                    }],
                },
                Block::SrcBlock {
                    language: "rust".into(),
                    content: "fn main() {}\n".into(),
                },
            ],
        };
        let encoded = encode_document(&doc);
        let decoded = decode_document(&encoded).unwrap();
        assert_eq!(doc, decoded);
    }

    #[test]
    fn roundtrip_all_constructors() {
        let doc = Document {
            blocks: vec![Block::Heading {
                level: 2,
                title: Title("All".into()),
                tags: vec![Tag("a".into()), Tag("b".into())],
                children: vec![
                    Block::Paragraph {
                        inlines: vec![
                            Inline::Plain("p".into()),
                            Inline::Bold(vec![Inline::Plain("b".into())]),
                            Inline::Italic(vec![Inline::Plain("i".into())]),
                            Inline::InlineCode("c".into()),
                            Inline::Verbatim("v".into()),
                            Inline::Link {
                                target: "t".into(),
                                description: Some("d".into()),
                            },
                        ],
                    },
                    Block::SrcBlock {
                        language: "rs".into(),
                        content: "x".into(),
                    },
                    Block::QuoteBlock {
                        children: vec![Block::Paragraph {
                            inlines: vec![Inline::Plain("q".into())],
                        }],
                    },
                    Block::List {
                        list_type: ListType::Unordered,
                        items: vec![ListItem {
                            content: vec![Block::Paragraph {
                                inlines: vec![Inline::Plain("one".into())],
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
                        entries: vec![("ID".into(), "123".into())],
                    },
                    Block::LogbookDrawer {
                        entries: vec![LogEntry {
                            timestamp: Timestamp("<2026-05-01 Fri>".into()),
                            note: String::new(),
                        }],
                    },
                    Block::Planning {
                        entries: vec![
                            PlanningEntry::Scheduled(Timestamp("<2026-05-01 Fri>".into())),
                            PlanningEntry::Deadline(Timestamp("<2026-05-15 Fri>".into())),
                        ],
                    },
                    Block::Comment {
                        text: "note".into(),
                    },
                    Block::HorizontalRule,
                ],
            }],
        };
        let encoded = encode_document(&doc);
        let decoded = decode_document(&encoded).unwrap();
        assert_eq!(doc, decoded);
    }

    #[test]
    fn decode_kbdoc_version_check() {
        let result = decode_document("(kb-doc 2 foo)");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("unsupported kb-doc version")
        );
    }

    #[test]
    fn decode_legacy_bare_ordered_list() {
        let legacy = "(kb-doc 1 (list ordered (item no-checkbox (paragraph (plain \"x\")))))";
        let doc = decode_document(legacy).unwrap();
        assert_eq!(
            doc.blocks,
            vec![Block::List {
                list_type: ListType::Ordered(1),
                items: vec![ListItem {
                    content: vec![Block::Paragraph {
                        inlines: vec![Inline::Plain("x".into())],
                    }],
                    checkbox: Checkbox::NoCheckbox,
                }],
            }]
        );
        // Re-encoding normalizes to the current form.
        let reencoded = encode_document(&doc);
        assert!(reencoded.contains("(list (ordered 1)"));
    }

    #[test]
    fn encode_empty_document() {
        let doc = Document { blocks: vec![] };
        let encoded = encode_document(&doc);
        assert_eq!(encoded, "(kb-doc 1)");
        let decoded = decode_document(&encoded).unwrap();
        assert_eq!(doc, decoded);
    }

    /// Every refusal the reader can make, in one table.
    ///
    /// This is the reader for the frozen archive: what it decodes is the only
    /// copy of a node written before the store existed. A decoder that
    /// guesses at malformed input is worse than one that stops, because the
    /// guess is silently wrong and the stop is loud. Each case below is a
    /// shape somebody could actually produce — a truncated blob, a hand
    /// edit, a version from a schema that no longer exists.
    #[test]
    fn malformed_input_is_refused_with_the_reason() {
        for (input, expected) in [
            ("", "unexpected end of input"),
            ("(kb-doc 1) trailing", "trailing input"),
            ("(kb-doc 1", "unterminated list"),
            (")", "unexpected closing paren"),
            ("(kb-doc 1 (paragraph (plain \"x", "unterminated string"),
            ("(kb-doc 1 (paragraph (plain \"x\\", "unterminated escape"),
            ("(kb-doc 1 (paragraph (plain \"\\q\")))", "unknown escape"),
            ("(hello 1)", "expected (kb-doc"),
            ("(kb-doc not-a-number)", "version must be a number"),
            ("symbol", "expected (kb-doc"),
            ("(kb-doc 1 (heading 1))", "heading needs level"),
            ("(kb-doc 1 (src-block))", "src-block needs language"),
            ("(kb-doc 1 (example-block))", "example-block needs body"),
            ("(kb-doc 1 (list))", "list needs list-type"),
            ("(kb-doc 1 (comment))", "comment needs text"),
            ("(kb-doc 1 (keyword name))", "keyword needs name and value"),
            ("(kb-doc 1 (nonesuch))", "unrecognized block tag"),
            ("(kb-doc 1 (\"quoted\"))", "block must start with a symbol"),
            ("(kb-doc 1 symbol)", "expected list for block"),
            ("(kb-doc 1 (paragraph (plain)))", "plain needs text"),
            ("(kb-doc 1 (paragraph (code)))", "code needs text"),
            ("(kb-doc 1 (paragraph (verbatim)))", "verbatim needs text"),
            ("(kb-doc 1 (paragraph (link)))", "link needs target"),
            (
                "(kb-doc 1 (paragraph (nonesuch)))",
                "unrecognized inline tag",
            ),
            (
                "(kb-doc 1 (paragraph (\"quoted\")))",
                "inline must start with a symbol",
            ),
            ("(kb-doc 1 (paragraph symbol))", "expected list for inline"),
        ] {
            let outcome = decode_document(input);
            let message = match &outcome {
                Err(e) => e.to_string(),
                Ok(doc) => format!("<decoded {} blocks>", doc.blocks.len()),
            };
            assert!(
                message.contains(expected),
                "decoding {input:?} should have said {expected:?}, said {message:?}"
            );
        }
    }

    /// Whitespace that is not a space survives the round trip, because a
    /// tab in a code block is content and a decoder that ate it would be
    /// lossy in exactly the place people notice.
    #[test]
    fn control_whitespace_survives_encoding() {
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![Inline::Plain("a\tb\rc".into())],
            }],
        };
        let encoded = encode_document(&doc);
        assert!(
            encoded.contains("\\t") && encoded.contains("\\r"),
            "{encoded}"
        );
        assert_eq!(decode_document(&encoded).unwrap(), doc);
    }

    /// Strikethrough and line breaks are inline kinds like any other; they
    /// were simply absent from the fixtures the round-trip property
    /// happened to generate.
    #[test]
    fn strikethrough_and_breaks_round_trip() {
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![
                    Inline::Strikethrough(vec![Inline::Plain("struck".into())]),
                    Inline::LineBreak,
                ],
            }],
        };
        let encoded = encode_document(&doc);
        assert!(encoded.contains("(strikethrough"), "{encoded}");
        assert!(encoded.contains("(break)"), "{encoded}");
        assert_eq!(decode_document(&encoded).unwrap(), doc);
    }
}
