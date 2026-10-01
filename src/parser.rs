//! Org-mode text → AST parser using hand-written recursive descent.
//!
//! Parses org syntax into the canonical [`Document`] AST. Covers every
//! constructor that the generator produces so round-trip is well-defined.

use tftio_org::ast::{
    Block, Checkbox, Document, Inline, ListItem, ListType, LogEntry, PlanningEntry, TableCell, Tag,
    Timestamp, Title,
};

/// Parse error with position context.
#[derive(Debug, Clone)]
pub struct ParseError {
    /// Human-readable description of the parse failure.
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parse an org-mode document string into a [`Document`].
///
/// # Errors
///
/// Returns `ParseError` if the input cannot be parsed.
pub fn parse_document(input: &str) -> Result<Document, ParseError> {
    parse_document_with_residue(input).map(|(doc, _residue)| doc)
}

/// Parse an org-mode document, also returning the content of input lines
/// that no block claimed.
///
/// A file whose residue is empty lost no content during parsing, even if
/// it does not round-trip byte-for-byte: every source line is represented
/// in some block. Blank lines are never residue. Lines inside quote
/// blocks are not tracked.
///
/// # Errors
///
/// Returns `ParseError` if the input cannot be parsed.
pub fn parse_document_with_residue(input: &str) -> Result<(Document, Vec<String>), ParseError> {
    let lines: Vec<&str> = input.lines().collect();
    let mut residue = Vec::new();
    let (blocks, _) = parse_blocks(&lines, 0, &mut residue)?;
    Ok((Document { blocks }, residue))
}

type ParseResult<T> = Result<(T, usize), ParseError>;

/// Parse a sequence of blocks starting at `pos`. Returns parsed blocks and new position.
///
/// Unrecognized non-blank lines are appended to `residue`.
#[allow(
    clippy::unnecessary_wraps,
    reason = "mirrors the fallible `ParseResult` shape of the sibling `try_parse_*` combinators for uniform composition"
)]
fn parse_blocks(lines: &[&str], pos: usize, residue: &mut Vec<String>) -> ParseResult<Vec<Block>> {
    let mut blocks = Vec::new();
    let mut i = pos;
    while i < lines.len() {
        let Some(&line) = lines.get(i) else { break };
        if line.is_empty() {
            // Blank lines are represented explicitly for faithful spacing.
            blocks.push(Block::BlankLine);
            i += 1;
            continue;
        }

        if let Some(Ok((block, next))) =
            try_parse_heading(lines, i, residue).or_else(|| try_parse_non_heading_block(lines, i))
        {
            blocks.push(block);
            i = next;
        } else {
            // Unrecognized line — record as residue (dropped content).
            if !line.is_empty() {
                residue.push(line.to_string());
            }
            i += 1;
        }
    }
    Ok((blocks, i))
}

/// Every block kind but a heading, tried in the same order `parse_blocks`
/// tries them after its own heading check.
///
/// Pulled out so `try_parse_heading`'s own child-collection loop can call
/// exactly this chain rather than maintain a second, shorter list of its
/// own. It used to: that second list carried only `try_parse_property_drawer`
/// through `try_parse_table` plus a bare `try_parse_paragraph` fallback,
/// omitting `try_parse_planning`, `try_parse_comment`, `try_parse_keyword`
/// and `try_parse_horizontal_rule` entirely. A `# comment`, a `#+keyword:`,
/// a `SCHEDULED:`/`DEADLINE:`/`CLOSED:` line, or a `-----` rule appearing as
/// a heading's direct child matched none of the seven child parsers and
/// also failed `try_parse_paragraph` (`is_paragraph_line` excludes exactly
/// those shapes, correctly, since each has its own block type) — so it fell
/// through to the child loop's own residue path and was silently dropped.
/// This surfaced as one of the false positives in the T017 rehearsal
/// (`PLAN-20260923-project-identity`): a transcript whose prose quoted a
/// shell comment (`# ...`) inside a fenced code block, at a heading's top
/// level, lost that line on every `kb create`/`kb update`, so the imported
/// text could never again compare equal to what was stored.
fn try_parse_non_heading_block(lines: &[&str], pos: usize) -> Option<ParseResult<Block>> {
    let line = *lines.get(pos)?;
    try_parse_property_drawer(lines, pos)
        .or_else(|| try_parse_logbook_drawer(lines, pos))
        .or_else(|| try_parse_src_block(lines, pos))
        .or_else(|| try_parse_example_block(lines, pos))
        .or_else(|| try_parse_quote_block(lines, pos))
        .or_else(|| try_parse_list(lines, pos))
        .or_else(|| try_parse_table(lines, pos))
        .or_else(|| try_parse_planning(pos, line))
        .or_else(|| try_parse_comment(pos, line))
        .or_else(|| try_parse_keyword(pos, line))
        .or_else(|| try_parse_horizontal_rule(pos, line))
        .or_else(|| try_parse_paragraph(lines, pos))
}

fn try_parse_heading(
    lines: &[&str],
    pos: usize,
    residue: &mut Vec<String>,
) -> Option<ParseResult<Block>> {
    let line = *lines.get(pos)?;
    if !line.starts_with('*') {
        return None;
    }

    let raw_level = line.chars().take_while(|c| *c == '*').count();
    // Require at least one space after stars for a valid heading
    if raw_level >= line.len() || !line[raw_level..].starts_with(' ') {
        return None;
    }
    let level: u8 = u8::try_from(raw_level.min(255)).unwrap_or(u8::MAX);
    let rest = line[raw_level..].trim();

    // Parse tags at end: "Title :tag1:tag2:"
    let (title_str, tags) = rest.rfind(" :").map_or((rest, vec![]), |tag_start| {
        let tag_part = &rest[tag_start + 1..];
        if tag_part.starts_with(':') && tag_part.ends_with(':') && tag_part.len() > 2 {
            let title = rest[..tag_start].trim();
            let tags: Vec<Tag> = tag_part[1..tag_part.len() - 1]
                .split(':')
                .filter(|t| !t.is_empty())
                .map(|t| Tag(t.to_string()))
                .collect();
            (title, tags)
        } else {
            (rest, vec![])
        }
    });

    let title = Title(title_str.to_string());

    // Collect children (blocks at higher indentation level)
    let mut children = Vec::new();
    let mut next = pos + 1;
    while next < lines.len() && !lines.get(next).is_some_and(|l| l.starts_with('*')) {
        // Gather child blocks that start on this or later lines up to the
        // next blank-line-separated block or next heading.
        let Some(&line) = lines.get(next) else { break };
        if line.is_empty() {
            children.push(Block::BlankLine);
            next += 1;
            continue;
        }
        // Try to parse the next item as a child block, through the same
        // dispatch chain `parse_blocks` uses for every non-heading block
        // (see `try_parse_non_heading_block`'s doc comment for the defect
        // a second, shorter list of parsers here used to cause).
        if let Some(Ok((child_block, new_pos))) = try_parse_non_heading_block(lines, next) {
            children.push(child_block);
            next = new_pos;
        } else {
            // Unrecognized child line — record as residue.
            if let Some(&child) = lines.get(next)
                && !child.is_empty()
            {
                residue.push(child.to_string());
            }
            next += 1;
        }
    }

    Some(Ok((
        Block::Heading {
            level,
            title,
            tags,
            children,
        },
        next,
    )))
}

/// Parse a `:PROPERTIES:` drawer, at column 0 only.
///
/// Worg's Org Syntax §2.3 does not exclude drawers from being indented, so
/// org itself permits `:PROPERTIES:` under a heading with leading
/// whitespace. This parser requires column 0 anyway, for the same reason
/// and by the same precedent as `try_parse_comment`/`try_parse_horizontal_rule`
/// (T017) and the block-opener fixes right above (T019, pattern B): none of
/// `Block::PropertyDrawer`, `Block::LogbookDrawer` carry a position or
/// indentation field the generator could use to restore an indent, so a
/// drawer recognized while indented always regenerates at column 0 — silent
/// content distortion, not loss, but still not byte-stable. Requiring exact
/// column-0 `":PROPERTIES:"`/`":LOGBOOK:"` keeps an indented look-alike
/// (e.g. a drawer nested inside an indented example in a quoted
/// transcript) as ordinary paragraph text instead, preserving the bytes
/// exactly. A `:PROPERTIES:` drawer directly under a heading at column 0 is
/// unaffected — that is exactly the shape the generator itself always
/// produces. Minimized from `codex-*` rehearsal records (T018/T019).
fn try_parse_property_drawer(lines: &[&str], pos: usize) -> Option<ParseResult<Block>> {
    if *lines.get(pos)? != ":PROPERTIES:" {
        return None;
    }
    let mut entries = Vec::new();
    let mut i = pos + 1;
    while i < lines.len() {
        let Some(line) = lines.get(i).map(|l| l.trim()) else {
            break;
        };
        if line == ":END:" {
            return Some(Ok((Block::PropertyDrawer { entries }, i + 1)));
        }
        if let Some(stripped) = line.strip_prefix(':')
            && let Some(colon_pos) = stripped.find(':')
        {
            let key = &stripped[..colon_pos];
            // Value kept verbatim (leading padding included) so aligned
            // drawers round-trip.
            let value = &stripped[colon_pos + 1..];
            entries.push((key.to_string(), value.to_string()));
        }
        i += 1;
    }
    Some(Ok((Block::PropertyDrawer { entries }, i)))
}

/// Parse a `:LOGBOOK:` drawer, at column 0 only. See
/// `try_parse_property_drawer`'s doc comment for why.
fn try_parse_logbook_drawer(lines: &[&str], pos: usize) -> Option<ParseResult<Block>> {
    if *lines.get(pos)? != ":LOGBOOK:" {
        return None;
    }
    let mut entries = Vec::new();
    let mut i = pos + 1;
    while i < lines.len() {
        let Some(line) = lines.get(i).map(|l| l.trim()) else {
            break;
        };
        if line == ":END:" {
            return Some(Ok((Block::LogbookDrawer { entries }, i + 1)));
        }
        // Parse "- <timestamp> note"
        if let Some(rest) = line.strip_prefix("- ")
            && rest.starts_with('<')
            && let Some(close) = rest.find('>')
        {
            let ts = &rest[..=close];
            let note = rest[close + 1..].trim();
            entries.push(LogEntry {
                timestamp: Timestamp(ts.to_string()),
                note: note.to_string(),
            });
        }
        i += 1;
    }
    Some(Ok((Block::LogbookDrawer { entries }, i)))
}

/// Parse a `#+begin_src ... #+end_src` block, whose opener starts at
/// column 0 only.
///
/// Worg's Org Syntax §2.3 does not exclude greater blocks from being
/// indented, so org itself permits an indented `#+begin_src`. This parser
/// requires column 0 anyway, by the same precedent as
/// `try_parse_comment`/`try_parse_horizontal_rule` (T017): `Block::SrcBlock`
/// carries only `language` and `content`, no indentation field the
/// generator could use to restore an original indent, so an opener
/// recognized while indented always regenerates its `#+begin_src`/
/// `#+end_src` lines at column 0 — the leading space kb#35's rehearsal
/// found being stripped from every `#+name:`-annotated noweb block quoted
/// in a Codex transcript (pattern A, T019,
/// `PLAN-20260923-project-identity`), even though the sibling `#+name:`
/// keyword line right above it already survived (see
/// `try_parse_keyword`/`is_paragraph_line`'s existing column-0 requirement
/// for keywords, T017). Requiring column 0 here too keeps an indented
/// look-alike as ordinary paragraph text, preserving the bytes exactly.
fn try_parse_src_block(lines: &[&str], pos: usize) -> Option<ParseResult<Block>> {
    let line = *lines.get(pos)?;
    if !line.starts_with("#+begin_src") {
        return None;
    }
    let language = line.strip_prefix("#+begin_src")?.trim().to_string();

    let mut i = pos + 1;
    let mut content = String::new();
    while i < lines.len() {
        let Some(&cur) = lines.get(i) else { break };
        if cur.trim() == "#+end_src" {
            // Canonical form: non-empty src bodies always end with newline
            if !content.is_empty() && !content.ends_with('\n') {
                content.push('\n');
            }
            return Some(Ok((Block::SrcBlock { language, content }, i + 1)));
        }
        if !content.is_empty() {
            content.push('\n');
        }
        content.push_str(cur);
        i += 1;
    }
    // No end marker found — treat rest as content
    Some(Ok((Block::SrcBlock { language, content }, i)))
}

/// Parse a `#+begin_example ... #+end_example` block, at column 0 only.
/// See `try_parse_src_block`'s doc comment for why.
fn try_parse_example_block(lines: &[&str], pos: usize) -> Option<ParseResult<Block>> {
    if *lines.get(pos)? != "#+begin_example" {
        return None;
    }
    let mut i = pos + 1;
    let mut content = String::new();
    while i < lines.len() {
        let Some(&cur) = lines.get(i) else { break };
        if cur.trim() == "#+end_example" {
            if !content.is_empty() && !content.ends_with('\n') {
                content.push('\n');
            }
            return Some(Ok((Block::ExampleBlock { content }, i + 1)));
        }
        if !content.is_empty() {
            content.push('\n');
        }
        content.push_str(cur);
        i += 1;
    }
    // No end marker — treat the rest as content.
    Some(Ok((Block::ExampleBlock { content }, i)))
}

/// Parse a `#+begin_quote ... #+end_quote` block, at column 0 only. See
/// `try_parse_src_block`'s doc comment for why.
fn try_parse_quote_block(lines: &[&str], pos: usize) -> Option<ParseResult<Block>> {
    if *lines.get(pos)? != "#+begin_quote" {
        return None;
    }
    let mut i = pos + 1;
    let mut child_lines = Vec::new();
    while i < lines.len() {
        let Some(&cur) = lines.get(i) else { break };
        if cur.trim() == "#+end_quote" {
            // Quote-block interiors are not residue-tracked.
            let (children, _) = parse_blocks(&child_lines, 0, &mut Vec::new()).ok()?;
            return Some(Ok((Block::QuoteBlock { children }, i + 1)));
        }
        child_lines.push(cur);
        i += 1;
    }
    None
}

fn try_parse_list(lines: &[&str], pos: usize) -> Option<ParseResult<Block>> {
    let (list_type, first_checkbox, first_rest) = match_bullet(lines.get(pos)?)?;

    let mut items: Vec<ListItem> = Vec::new();
    let mut cur_checkbox = first_checkbox;
    let mut cur_inlines = parse_inlines(first_rest);
    let mut i = pos + 1;
    // The most recently consumed ordinal, for an `Ordered` list only —
    // tracked so a later restart (pattern C, T019,
    // `PLAN-20260923-project-identity`) can be told apart from a genuine
    // next sibling item. See the restart check below for why.
    let mut last_ordinal = match list_type {
        ListType::Ordered(n) => Some(n),
        ListType::Unordered => None,
    };

    while i < lines.len() {
        let Some(&line) = lines.get(i) else { break };
        if line.is_empty() {
            // A blank line ends the list, unless it is followed (after any
            // further blank lines) by a line that is genuinely a
            // continuation of the current item — indented, and not itself
            // some other kind of structure. See
            // `is_list_continuation_line`'s doc comment for exactly what
            // qualifies and why; a first attempt at this bridged too
            // eagerly and is documented there as a cautionary example.
            let continues = lines
                .get(i + 1..)
                .and_then(|rest| rest.iter().find(|l| !l.is_empty()))
                .is_some_and(|next| is_list_continuation_line(next));
            if !continues {
                break;
            }
            // One `LineBreak` for the blank line itself; the indented-line
            // branch below contributes the second, exactly as it does
            // between two ordinary (non-blank-separated) continuation
            // lines.
            cur_inlines.push(Inline::LineBreak);
            i += 1;
            continue;
        }
        if let Some((lt, checkbox, rest)) = match_bullet(line)
            && std::mem::discriminant(&lt) == std::mem::discriminant(&list_type)
        {
            // A column-0 bullet of the *same* kind (ordered vs. unordered)
            // starts the next sibling item. `ListType` (`tftio_org::ast`) is
            // one value per list, not per item — "so a list that starts at
            // `2.` round-trips faithfully" — which means a list cannot mix
            // marker kinds in the AST at all. Requiring the match here, and
            // otherwise falling through to end the list (below), is what
            // keeps that constraint from silently coercing a `-` item that
            // happens to follow an `N.` item into a renumbered `N.` item:
            // before this check, `- ` and `N. ` lines were swallowed into
            // one list under whichever kind the first item had, and the
            // generator re-emitted every item — content unchanged, marker
            // kind wrong — as that one kind. A mismatched marker now ends
            // this list without consuming the line, so the outer dispatch
            // (`parse_blocks`/`try_parse_non_heading_block`) starts a fresh
            // list of the other kind right there instead. Regression: T017
            // (`PLAN-20260923-project-identity`), found via a real
            // rehearsal transcript whose reply numbered its first bullet
            // and dashed the rest.
            //
            // A same-kind `Ordered` bullet whose own number is strictly
            // *less than* the previous item's is a restart, not a
            // continuation, and ends this list the same way a mismatched
            // marker kind does — without consuming the line — rather than
            // joining it. `ListType::Ordered` carries one start value per
            // *list*, not per item (see the comment above), so the
            // generator renumbers every item of a list sequentially from
            // that one value on every render; folding a fresh `1.`/`2.`
            // list that happens to follow another straight into the
            // running list would keep it as one list and renumber the
            // restarted items past the first list's end (`6.`, `7.`, … in
            // the reproduction that motivated this, T019 pattern C,
            // `PLAN-20260923-project-identity`) — a distortion, not
            // content loss, but not byte-stable either.
            //
            // Equal is deliberately *not* a restart, unlike a strictly
            // lower number: `ordered_list_two_items` (`tests/parser.rs`,
            // ported from the Haskell suite) already establishes, as
            // existing committed behavior, that `"1. one\n1. two\n"` is
            // one two-item list — the common hand-written org convention
            // of numbering every item `1.` and relying on export to
            // renumber sequentially (Worg's Org Syntax §4.5 lists ordered
            // items only as "a numeral followed by either a period or a
            // right parenthesis", with no requirement that the numerals
            // increase). Ending the list on a repeat as well as a
            // decrease would silently split every hand-numbered `1.`-only
            // list into as many one-item lists as it has items — a
            // regression this task's own invariant forbids. Two adjacent
            // lists that are genuinely one continuous or one
            // repeated-numeral sequence are unaffected: this only ends
            // the list when the number goes strictly backward.
            if let ListType::Ordered(new_ordinal) = lt
                && last_ordinal.is_some_and(|last| new_ordinal < last)
            {
                break;
            }
            if let ListType::Ordered(new_ordinal) = lt {
                last_ordinal = Some(new_ordinal);
            }
            items.push(ListItem {
                content: vec![Block::Paragraph {
                    inlines: std::mem::take(&mut cur_inlines),
                }],
                checkbox: cur_checkbox,
            });
            cur_checkbox = checkbox;
            cur_inlines = parse_inlines(rest);
            i += 1;
        } else if match_bullet(line).is_some() {
            // A column-0 bullet of a *different* kind ends this list; it is
            // not consumed, so it starts its own list on the next pass.
            break;
        } else if line.starts_with(' ') || line.starts_with('\t') {
            // Indented continuation of the current item — including
            // nested sub-bullets, kept verbatim as continuation text.
            cur_inlines.push(Inline::LineBreak);
            cur_inlines.extend(parse_inlines(line));
            i += 1;
        } else {
            // A column-0 non-bullet line ends the list.
            break;
        }
    }
    items.push(ListItem {
        content: vec![Block::Paragraph {
            inlines: cur_inlines,
        }],
        checkbox: cur_checkbox,
    });
    Some(Ok((Block::List { list_type, items }, i)))
}

/// If `line` starts (column 0) with a list bullet, return the list type,
/// checkbox state, and the content after the bullet and checkbox.
fn match_bullet(line: &str) -> Option<(ListType, Checkbox, &str)> {
    if let Some(rest) = line.strip_prefix("- ") {
        let (checkbox, rest) = strip_checkbox(rest);
        return Some((ListType::Unordered, checkbox, rest));
    }
    // Ordered: `N. ` for one or more digits.
    let digits = line.chars().take_while(char::is_ascii_digit).count();
    if digits > 0
        && let Some(rest) = line[digits..].strip_prefix(". ")
    {
        let ordinal: u64 = line[..digits].parse().unwrap_or(1);
        let (checkbox, rest) = strip_checkbox(rest);
        return Some((ListType::Ordered(ordinal), checkbox, rest));
    }
    None
}

/// Whether an indented line following a blank line, inside a list, is
/// genuinely that list item's continuation and should bridge the blank
/// line rather than end the list.
///
/// Real model replies routinely put a bullet's prose, a blank line, then an
/// indented table or paragraph under the same item — org's own "loose
/// list" allows exactly this, and ending the list here hands the indented
/// content to whichever top-level block parser recognizes it instead
/// (`try_parse_table` for an indented table, say), which carries no
/// indentation of its own and silently loses it on regeneration. So an
/// indented, structurally inert line — ordinary prose, or a table row —
/// should bridge the blank line and stay part of the item.
///
/// A first version of this fix (T017, `PLAN-20260923-project-identity`)
/// bridged whenever the next line was merely indented, with no further
/// check, and that was too eager: `escape_org`
/// (`kb_import.model`) indents a literal `*`/`#+` at the
/// start of a line by exactly one column to defuse it, and a reply that
/// closed one numbered list, added a blank line, then opened
/// `**Suggested order:**` (escaped to ` **Suggested order:**`, since it
/// starts with `*`) followed by a *fresh* `1. ...` list, had that escaped
/// bold line bridged into the prior list's last item — swallowing it as
/// plain continuation text — and then, because the following `1.` line
/// carries the same `Ordered` discriminant as the prior list, it was taken
/// for that list's next sibling item rather than the start of a new one,
/// renumbering it far past `1`. Evidence: `cc-7b992f21...`, where a fresh
/// `1. Clear items...` list came back numbered from `18.`.
///
/// So this refuses to bridge into anything that represents real,
/// deliberate structure rather than filler prose or a table: a list item
/// of its own (`match_bullet`), a heading, a `#+keyword:` line or a
/// `#+begin_...` block opener, a comment, a drawer, a planning line, a
/// horizontal rule — and, independently of whether this parser would ever
/// have recognized any of those forms, the `escape_org` shape itself
/// (exactly one leading space before a literal `*` or `#+`), because that
/// shape means "the importer defanged a structural assertion here", not
/// "here is some table filler that happens to be indented". Prose and
/// table rows alike are unaffected and still bridge.
fn is_list_continuation_line(line: &str) -> bool {
    if !(line.starts_with(' ') || line.starts_with('\t')) {
        return false;
    }
    // The `escape_org` shape: exactly one leading space (not two or more,
    // which is ordinary indentation, not escaping) in front of a literal
    // `*` or `#+`.
    if let Some(rest) = line.strip_prefix(' ')
        && !rest.starts_with(' ')
        && !rest.starts_with('\t')
        && (rest.starts_with('*') || rest.starts_with("#+"))
    {
        return false;
    }
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    let stars = trimmed.chars().take_while(|c| *c == '*').count();
    if stars > 0 && trimmed[stars..].starts_with(' ') {
        return false; // A heading, de-indented.
    }
    if match_bullet(trimmed).is_some() {
        return false; // A list item of its own.
    }
    if trimmed.starts_with("#+")
        || trimmed.starts_with("# ")
        || trimmed.starts_with(":PROPERTIES:")
        || trimmed.starts_with(":LOGBOOK:")
        || trimmed.starts_with("SCHEDULED:")
        || trimmed.starts_with("DEADLINE:")
        || trimmed.starts_with("CLOSED:")
        || trimmed == "-----"
    {
        return false; // A keyword, block opener, comment, drawer, planning
        // line, or horizontal rule, de-indented.
    }
    true
}

/// Strip a leading `[ ] ` / `[X] ` checkbox marker, if present.
fn strip_checkbox(s: &str) -> (Checkbox, &str) {
    s.strip_prefix("[X] ").map_or_else(
        || {
            s.strip_prefix("[ ] ")
                .map_or((Checkbox::NoCheckbox, s), |r| (Checkbox::Unchecked, r))
        },
        |r| (Checkbox::Checked, r),
    )
}

/// Parse a table (a run of `| ... |` rows), each row at column 0 only.
///
/// Worg's Org Syntax §2.3 does not exclude tables from being indented, so
/// org itself permits an indented table. This parser requires column 0
/// anyway, by the same precedent as `try_parse_comment` (T017) and the
/// block-opener fixes above (T019, pattern A): `Block::Table` carries only
/// `rows`, no indentation, so a table recognized while indented always
/// regenerates flush left — silently losing its indent, pattern B (T019,
/// `PLAN-20260923-project-identity`). Requiring column 0 keeps an indented
/// look-alike as ordinary paragraph text, preserving the bytes exactly. A
/// row that is itself indented (even mid-table) ends the table here rather
/// than being folded in and losing its indent; it is not consumed, so the
/// outer dispatch tries it as a fresh block (typically paragraph text) at
/// that position.
fn try_parse_table(lines: &[&str], pos: usize) -> Option<ParseResult<Block>> {
    let raw_first = *lines.get(pos)?;
    if raw_first.starts_with(' ') || raw_first.starts_with('\t') {
        return None;
    }
    let first = raw_first.trim();
    if !first.starts_with('|') || !first.ends_with('|') {
        return None;
    }
    let mut rows = Vec::new();
    let mut i = pos;
    while i < lines.len() {
        let Some(&raw_line) = lines.get(i) else {
            break;
        };
        // A blank line ends the table.
        if raw_line.is_empty() {
            break;
        }
        // An indented row ends the table without being consumed (see the
        // doc comment above).
        if raw_line.starts_with(' ') || raw_line.starts_with('\t') {
            break;
        }
        let line = raw_line.trim();
        if line.len() < 2 || !line.starts_with('|') || !line.ends_with('|') {
            break;
        }
        // Cells are kept verbatim, padding included — column alignment
        // and separator rows (`|---+---|`) round-trip as cell content.
        let cells: Vec<TableCell> = line[1..line.len() - 1]
            .split('|')
            .map(|c| TableCell {
                inlines: parse_inlines(c),
            })
            .collect();
        rows.push(cells);
        i += 1;
    }
    if rows.is_empty() {
        return None;
    }
    Some(Ok((Block::Table { rows }, i)))
}

fn try_parse_planning(pos: usize, line: &str) -> Option<ParseResult<Block>> {
    let trimmed = line.trim();
    if !trimmed.starts_with("SCHEDULED: ")
        && !trimmed.starts_with("DEADLINE: ")
        && !trimmed.starts_with("CLOSED: ")
    {
        return None;
    }

    let mut entries = Vec::new();
    // Scan the line for keyword + timestamp pairs
    let mut remaining = trimmed;
    while !remaining.is_empty() {
        if let Some(rest) = remaining.strip_prefix("SCHEDULED: ")
            && let Some((ts, after)) = extract_timestamp(rest)
        {
            entries.push(PlanningEntry::Scheduled(Timestamp(ts)));
            remaining = after;
            continue;
        }
        if let Some(rest) = remaining.strip_prefix("DEADLINE: ")
            && let Some((ts, after)) = extract_timestamp(rest)
        {
            entries.push(PlanningEntry::Deadline(Timestamp(ts)));
            remaining = after;
            continue;
        }
        if let Some(rest) = remaining.strip_prefix("CLOSED: ")
            && let Some((ts, after)) = extract_timestamp(rest)
        {
            entries.push(PlanningEntry::Closed(Timestamp(ts)));
            remaining = after;
            continue;
        }
        break;
    }

    if entries.is_empty() {
        return None;
    }
    Some(Ok((Block::Planning { entries }, pos + 1)))
}

/// Extract a timestamp like `<2026-04-30 Thu>` from the start of `s`.
/// Returns the timestamp string and the remaining text.
fn extract_timestamp(s: &str) -> Option<(String, &str)> {
    let s = s.trim();
    if !s.starts_with('<') {
        return None;
    }
    let close = s.find('>')?;
    let ts = s[..=close].to_string();
    Some((ts, s[close + 1..].trim()))
}

/// Parse a `# comment` line, at column 0 only.
///
/// Real org syntax does *not* require this (Worg's Org Syntax §4.3.5: "A
/// comment line starts with a hash character (`#`) and either a whitespace
/// character or the immediate end of the line"; §2.3 lists which elements
/// cannot be indented — headings, inlinetasks, footnote definitions, diary
/// sexps — and comments are not among them, so org itself allows an
/// indented `# comment`). This parser deviates from that on purpose: like
/// `try_parse_keyword` before it (T017, `PLAN-20260923-project-identity`,
/// the first false-positive cause fixed), `Block::Comment` carries only its
/// text, no position, so a comment recognized while indented cannot be
/// regenerated with that indentation restored — the generator always emits
/// `# {text}` at column 0. Before this fix, an indented `# ...` line (a
/// real Python/shell comment quoted inside a fenced code block in a
/// transcript, say) was recognized as a `Comment` and silently lost its
/// leading whitespace on every `kb create`/`kb update`. Requiring column 0
/// here, matching what `is_paragraph_line` already requires for a heading
/// or a keyword, keeps an indented `#` line as ordinary paragraph text —
/// content preserved exactly, at the cost of not modeling it as a comment.
/// The alternative, carrying indentation on `Block::Comment` itself, would
/// need a field this repository's `tftio_org` dependency does not have and
/// cannot add without a published version bump, which is out of scope
/// here. Minimized from real rehearsal transcripts (`cc-b90c37fc...`,
/// `cc-c58df4cd...`).
fn try_parse_comment(pos: usize, line: &str) -> Option<ParseResult<Block>> {
    line.strip_prefix("# ").map(|text| {
        Ok((
            Block::Comment {
                text: text.to_string(),
            },
            pos + 1,
        ))
    })
}

/// Parse a `#+NAME: value` keyword line.
///
/// `name` is the run of non-`:`, non-whitespace characters after `#+`;
/// the character immediately after must be `:`. `value` is the verbatim
/// remainder after that `:`, leading space included. Block delimiters
/// such as `#+begin_src` have no `:` after the name and fall through.
fn try_parse_keyword(pos: usize, line: &str) -> Option<ParseResult<Block>> {
    let rest = line.strip_prefix("#+")?;
    let name_len = rest
        .find(|c: char| c == ':' || c.is_whitespace())
        .unwrap_or(rest.len());
    if name_len == 0 || rest.as_bytes().get(name_len) != Some(&b':') {
        return None;
    }
    let name = rest[..name_len].to_string();
    let value = rest[name_len + 1..].to_string();
    Some(Ok((Block::Keyword { name, value }, pos + 1)))
}

/// Parse a `-----` horizontal rule line, at column 0 only.
///
/// Worg's Org Syntax §4.3.7 defines a horizontal rule only as "a line
/// consisting of at least five consecutive hyphens"; §2.3's indentation
/// rule excludes headings, inlinetasks, footnote definitions, and diary
/// sexps from being indented, and a horizontal rule is not on that list —
/// so org itself permits an indented one. This parser requires column 0
/// anyway, for the same reason and by the same precedent as
/// `try_parse_comment` right above: `Block::HorizontalRule` is a bare unit
/// variant carrying no data at all, so there is no way to record an
/// original indent for the generator (`"-----\n"`, always at column 0) to
/// restore. An indented `-----` (an ASCII-table rule quoted inside a
/// transcript, say) used to be recognized as a real horizontal rule and
/// lose its indentation on every write; requiring column 0 keeps it as
/// ordinary paragraph text instead, preserving the bytes exactly.
/// Minimized from a real rehearsal transcript (`cc-89c9fe92...`).
fn try_parse_horizontal_rule(pos: usize, line: &str) -> Option<ParseResult<Block>> {
    if line == "-----" {
        Some(Ok((Block::HorizontalRule, pos + 1)))
    } else {
        None
    }
}

fn try_parse_paragraph(lines: &[&str], pos: usize) -> Option<ParseResult<Block>> {
    if !is_paragraph_line(lines.get(pos)?) {
        return None;
    }
    // Consume consecutive paragraph lines into one block, joining them
    // with explicit `LineBreak`s so the source wrapping round-trips.
    let mut inlines = Vec::new();
    let mut i = pos;
    while let Some(&line) = lines.get(i).filter(|l| is_paragraph_line(l)) {
        if i > pos {
            inlines.push(Inline::LineBreak);
        }
        inlines.extend(parse_inlines(line));
        i += 1;
    }
    Some(Ok((Block::Paragraph { inlines }, i)))
}

/// Whether `line` can appear as paragraph content: neither blank nor the
/// start of any other block kind.
fn is_paragraph_line(line: &str) -> bool {
    if line.is_empty() {
        return false;
    }
    let trimmed = line.trim();
    // Heading: one or more `*` followed by a space, at column 0 exactly as
    // `try_parse_heading` requires (`line.starts_with('*')`, no trim). An
    // indented look-alike — the `escape_org` shape the transcript importer
    // (`kb_import.model`) uses to defuse a literal `*`/`#+` at
    // the start of imported prose — is not a heading anywhere else in this
    // parser and must stay paragraph text here too, or it is silently
    // dropped as residue instead of round-tripping (see the regression test
    // built from a real transcript shape in `tests/`).
    let stars = line.chars().take_while(|c| *c == '*').count();
    if stars > 0 && line[stars..].starts_with(' ') {
        return false;
    }
    // The three block openers this parser actually recognizes
    // (`try_parse_src_block`, `try_parse_example_block`,
    // `try_parse_quote_block`), matched exactly as they match themselves and,
    // since T019 (pattern A, `PLAN-20260923-project-identity`), at column 0
    // only — `#+begin_src` as a prefix of the raw, untrimmed line (a
    // language name may follow), the other two as exact untrimmed lines. See
    // `try_parse_src_block`'s doc comment for why column 0: none of the
    // three carries an indentation field the generator could restore, so an
    // indented opener must stay paragraph text instead, like the
    // heading/keyword checks above. A fourth org block keyword this parser
    // does not implement — `#+begin_export`, `#+begin_signature`, and the
    // like are common in real prose that discusses or quotes org/
    // message-signature snippets — is not one of these three, so excluding
    // it here as if it were still a block opener left it matching no
    // `try_parse_*` at all; it fell through to residue and was silently
    // dropped (T017, a fourth false-positive cause found via the same
    // rehearsal).
    // Comment (`# `), horizontal rule (`-----`), `:PROPERTIES:`/`:LOGBOOK:`
    // drawers, and table rows (`| ... |`), all at column 0 exactly as their
    // `try_parse_*` counterparts now require (see their doc comments: org
    // itself allows each of these indented, but none of `Block::Comment`,
    // `Block::HorizontalRule`, `Block::PropertyDrawer`, `Block::LogbookDrawer`,
    // `Block::Table` can carry an indent for the generator to restore). An
    // indented look-alike of any of these is not real structure anywhere
    // else in this parser and must stay paragraph text here too. The table
    // check also requires at least two characters (`trimmed.len() >= 2`),
    // matching `try_parse_table`'s own minimum: a line holding only
    // whitespace and a lone `|` — a gutter column in a quoted compiler
    // diagnostic, say — is not a one-cell table row (`try_parse_table`
    // already declines it, since a real row needs an opening and a closing
    // pipe) and must not be excluded from paragraph text either, or it
    // matches no `try_parse_*` at all and is silently dropped as residue
    // (pattern D, T019).
    if line.starts_with("# ") || line == "-----" {
        return false;
    }
    if line.starts_with(":PROPERTIES:")
        || line.starts_with(":LOGBOOK:")
        || line.starts_with("#+begin_src")
        || line == "#+begin_example"
        || line == "#+begin_quote"
        || trimmed.starts_with("SCHEDULED:")
        || trimmed.starts_with("DEADLINE:")
        || trimmed.starts_with("CLOSED:")
        || (line.starts_with('|') && line.ends_with('|') && trimmed.len() >= 2)
    {
        return false;
    }
    // A column-0 list bullet starts a list, not a paragraph. An indented
    // bullet has no column-0 list to join, so it stays paragraph text.
    if match_bullet(line).is_some() {
        return false;
    }
    // Keyword line `#+name:`, at column 0 exactly as `try_parse_keyword`
    // requires it (called with the raw, untrimmed line). An indented
    // look-alike is not a keyword line anywhere else in this parser; see the
    // heading check above for why this must match `line`, not `trimmed`.
    if let Some(rest) = line.strip_prefix("#+") {
        let name_len = rest
            .find(|c: char| c == ':' || c.is_whitespace())
            .unwrap_or(rest.len());
        if name_len > 0 && rest.as_bytes().get(name_len) == Some(&b':') {
            return false;
        }
    }
    true
}

/// Parse inline formatting from a string.
fn parse_inlines(input: &str) -> Vec<Inline> {
    let mut inlines = Vec::new();
    let mut pos = 0;
    let chars: Vec<char> = input.chars().collect();

    // `slice(a..b)` collects an in-range char range to a String. Every call
    // site below derives its bounds from `find_closing` / `next_marker_or_end`
    // / the loop guard, so the range is always valid; an empty fallback would
    // only ever appear on a logic bug.
    let slice =
        |a: usize, b: usize| -> String { chars.get(a..b).unwrap_or_default().iter().collect() };
    let slice_from = |a: usize| -> String { chars.get(a..).unwrap_or_default().iter().collect() };

    while pos < chars.len() {
        let Some(&c) = chars.get(pos) else { break };
        match c {
            '*' => {
                if let Some(end) = find_closing(&chars, pos + 1, '*') {
                    let inner = slice(pos + 1, end);
                    inlines.push(Inline::Bold(parse_inlines(&inner)));
                    pos = end + 1;
                } else {
                    // Treat as literal
                    if let Some(end) = next_marker_or_end(&chars, pos) {
                        inlines.push(Inline::Plain(slice(pos, end)));
                        pos = end;
                    } else {
                        inlines.push(Inline::Plain(slice_from(pos)));
                        pos = chars.len();
                    }
                }
            }
            '/' => {
                if let Some(end) = find_closing(&chars, pos + 1, '/') {
                    let inner = slice(pos + 1, end);
                    inlines.push(Inline::Italic(parse_inlines(&inner)));
                    pos = end + 1;
                } else if let Some(end) = next_marker_or_end(&chars, pos) {
                    inlines.push(Inline::Plain(slice(pos, end)));
                    pos = end;
                } else {
                    inlines.push(Inline::Plain(slice_from(pos)));
                    pos = chars.len();
                }
            }
            '+' => {
                if let Some(end) = find_closing(&chars, pos + 1, '+') {
                    let inner = slice(pos + 1, end);
                    inlines.push(Inline::Strikethrough(parse_inlines(&inner)));
                    pos = end + 1;
                } else if let Some(end) = next_marker_or_end(&chars, pos) {
                    inlines.push(Inline::Plain(slice(pos, end)));
                    pos = end;
                } else {
                    inlines.push(Inline::Plain(slice_from(pos)));
                    pos = chars.len();
                }
            }
            '=' => {
                if let Some(end) = find_closing(&chars, pos + 1, '=') {
                    let code = slice(pos + 1, end);
                    inlines.push(Inline::InlineCode(code));
                    pos = end + 1;
                } else if let Some(end) = next_marker_or_end(&chars, pos) {
                    inlines.push(Inline::Plain(slice(pos, end)));
                    pos = end;
                } else {
                    inlines.push(Inline::Plain(slice_from(pos)));
                    pos = chars.len();
                }
            }
            '~' => {
                if let Some(end) = find_closing(&chars, pos + 1, '~') {
                    let verb = slice(pos + 1, end);
                    inlines.push(Inline::Verbatim(verb));
                    pos = end + 1;
                } else if let Some(end) = next_marker_or_end(&chars, pos) {
                    inlines.push(Inline::Plain(slice(pos, end)));
                    pos = end;
                } else {
                    inlines.push(Inline::Plain(slice_from(pos)));
                    pos = chars.len();
                }
            }
            '[' => {
                let (inline, next) = consume_bracket(&chars, pos);
                inlines.push(inline);
                pos = next;
            }
            _ => {
                if let Some(end) = next_marker_or_end(&chars, pos) {
                    inlines.push(Inline::Plain(slice(pos, end)));
                    pos = end;
                } else {
                    inlines.push(Inline::Plain(slice_from(pos)));
                    pos = chars.len();
                }
            }
        }
    }

    // Merge adjacent Plain inlines
    merge_adjacent_plain(&mut inlines);
    inlines
}

/// Consume a `[`-run at `pos`: an org `[[target]]` / `[[target][desc]]`
/// link, or — when the brackets do not form a well-shaped link — a plain
/// literal run up to the next inline marker. Returns the inline to emit
/// and the position after it. Never drops trailing text.
fn consume_bracket(chars: &[char], pos: usize) -> (Inline, usize) {
    // All ranges below are derived from `position(|c| c == ']')` matches or
    // the loop-verified `pos`, so they are always in bounds; an empty
    // fallback would only surface on a logic bug.
    let collect =
        |a: usize, b: usize| -> String { chars.get(a..b).unwrap_or_default().iter().collect() };
    let collect_from = |a: usize| -> String { chars.get(a..).unwrap_or_default().iter().collect() };
    let plain_to = |end: usize| Inline::Plain(collect(pos, end));
    let plain_rest = || Inline::Plain(collect_from(pos));
    let literal = || {
        next_marker_or_end(chars, pos)
            .map_or_else(|| (plain_rest(), chars.len()), |end| (plain_to(end), end))
    };

    // Not a `[[…` link opener — single bracket, literal.
    if pos + 1 >= chars.len() || chars.get(pos + 1) != Some(&'[') {
        return literal();
    }
    let start = pos + 2;
    let Some(bracket_end) = chars
        .get(start..)
        .and_then(|rest| rest.iter().position(|&c| c == ']'))
        .map(|p| start + p)
    else {
        return (plain_rest(), chars.len());
    };

    if bracket_end + 1 < chars.len() && chars.get(bracket_end + 1) == Some(&']') {
        // [[target]]
        let target = collect(start, bracket_end);
        return (
            Inline::Link {
                target,
                description: None,
            },
            bracket_end + 2,
        );
    }
    if bracket_end + 1 >= chars.len() || chars.get(bracket_end + 1) != Some(&'[') {
        // `[[…]` followed by something other than `]` or `[` — literal.
        return literal();
    }
    // [[target][description]]
    let target: String = collect(start, bracket_end);
    let desc_start = bracket_end + 2;
    match chars
        .get(desc_start..)
        .and_then(|rest| rest.iter().position(|&c| c == ']'))
        .map(|p| desc_start + p)
    {
        Some(desc_end) if desc_end + 1 < chars.len() && chars.get(desc_end + 1) == Some(&']') => {
            let description = collect(desc_start, desc_end);
            (
                Inline::Link {
                    target,
                    description: Some(description),
                },
                desc_end + 2,
            )
        }
        // Malformed `[[target][…` — literal up to the unmatched `]`.
        Some(desc_end) => (plain_to(desc_end + 1), desc_end + 1),
        None => (plain_rest(), chars.len()),
    }
}

fn find_closing(chars: &[char], start: usize, marker: char) -> Option<usize> {
    for i in start..chars.len() {
        let Some(&c) = chars.get(i) else { break };
        if c == marker && (i + 1 == chars.len() || chars.get(i + 1) != Some(&marker)) {
            return Some(i);
        }
    }
    None
}

fn next_marker_or_end(chars: &[char], pos: usize) -> Option<usize> {
    for i in pos..chars.len() {
        let Some(&c) = chars.get(i) else { break };
        if c == '*' || c == '/' || c == '+' || c == '=' || c == '~' || c == '[' {
            if i == pos {
                // Find the next different char
                continue;
            }
            return Some(i);
        }
        if c == '[' && i + 1 < chars.len() && chars.get(i + 1) == Some(&'[') {
            if i == pos {
                continue;
            }
            return Some(i);
        }
    }
    None
}

fn merge_adjacent_plain(inlines: &mut Vec<Inline>) {
    let mut i = 0;
    while i + 1 < inlines.len() {
        if let (Some(Inline::Plain(a)), Some(Inline::Plain(b))) =
            (inlines.get(i), inlines.get(i + 1))
        {
            let merged = format!("{a}{b}");
            if let Some(slot) = inlines.get_mut(i) {
                *slot = Inline::Plain(merged);
            }
            inlines.remove(i + 1);
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_heading_level_1() {
        let doc = parse_document("* Hello\n").unwrap();
        assert_eq!(
            doc.blocks[0],
            Block::Heading {
                level: 1,
                title: Title("Hello".into()),
                tags: vec![],
                children: vec![]
            }
        );
    }

    #[test]
    fn parse_heading_with_tags() {
        let doc = parse_document("** Task :rust:kb:\n").unwrap();
        if let Block::Heading {
            level,
            title,
            tags,
            children: _,
        } = &doc.blocks[0]
        {
            assert_eq!(*level, 2);
            assert_eq!(title.0, "Task");
            assert_eq!(tags.len(), 2);
            assert_eq!(tags[0].0, "rust");
            assert_eq!(tags[1].0, "kb");
        } else {
            panic!("expected heading");
        }
    }

    #[test]
    fn parse_paragraph() {
        let doc = parse_document("some text\n").unwrap();
        assert_eq!(
            doc.blocks[0],
            Block::Paragraph {
                inlines: vec![Inline::Plain("some text".into())]
            }
        );
    }

    /// The transcript importer's `escape_org` (`kb_import.model`)
    /// defuses a literal `*`/`#+` at the start of an imported line by
    /// indenting it one column, on the theory that only column 0 is
    /// structural in org. `is_paragraph_line` used to disagree: it trimmed
    /// the line before testing for a heading star or a keyword prefix, so an
    /// indented look-alike still failed the paragraph test even though
    /// neither `try_parse_heading` nor `try_parse_keyword` would ever have
    /// matched it (both require column 0, untrimmed). With no other block
    /// parser claiming it either, the line vanished into
    /// `parse_document_with_residue`'s residue — silent content loss on
    /// every transcript containing exactly the shape the importer's own
    /// escaping was meant to protect. This is a minimized reproduction of a
    /// real Claude Code transcript's shape (T017; see
    /// `tests/import_escaped_lines.rs`).
    #[test]
    fn escaped_heading_and_keyword_lines_survive_as_paragraph_text() {
        let input = "* Human\nline one\n * escaped star line\nline three\n \
                      #+escaped keyword line\nline five\n";
        let (doc, residue) = parse_document_with_residue(input).unwrap();
        assert!(
            residue.is_empty(),
            "expected no dropped content, got residue: {residue:?}"
        );
        let Block::Heading { children, .. } = &doc.blocks[0] else {
            panic!("expected a heading");
        };
        let rendered = crate::generator::generate(&doc);
        assert!(
            rendered.contains(" * escaped star line"),
            "escaped heading-star line was lost on round-trip: {rendered:?}"
        );
        assert!(
            rendered.contains(" #+escaped keyword line"),
            "escaped keyword-looking line was lost on round-trip: {rendered:?}"
        );
        assert!(
            !children.is_empty(),
            "the escaped lines should have joined the paragraph under the heading"
        );
    }

    /// A second, independent false-positive cause found in the T017
    /// rehearsal: `try_parse_heading`'s child-collection loop used to try
    /// only seven block kinds (`try_parse_non_heading_block`'s doc comment
    /// lists them) before falling back to a bare `try_parse_paragraph`,
    /// omitting `try_parse_comment`, `try_parse_keyword`, `try_parse_planning`
    /// and `try_parse_horizontal_rule` — every one of which `is_paragraph_line`
    /// *correctly* excludes from paragraph content, since each has its own
    /// block type. A `# shell comment` line quoted inside a transcript's
    /// fenced code block, appearing as a heading's direct child, therefore
    /// matched no child parser and failed the paragraph fallback too, and
    /// was silently dropped into the child loop's own residue path — while
    /// the exact same line at the top level, or inside a further-nested
    /// block, round-tripped correctly. This reproduces a real Claude Code
    /// transcript's shape (minimized, not quoted).
    #[test]
    fn a_comment_line_as_a_headings_direct_child_survives() {
        let input = "* Human\nThe exact commands:\n\n```bash\n# here, on the laptop\nyadm push origin main\n```\n";
        let (doc, residue) = parse_document_with_residue(input).unwrap();
        assert!(
            residue.is_empty(),
            "expected no dropped content, got residue: {residue:?}"
        );
        let rendered = crate::generator::generate(&doc);
        assert!(
            rendered.contains("# here, on the laptop"),
            "a comment line under a heading was lost on round-trip: {rendered:?}"
        );
    }

    /// A third, independent false-positive cause found in the T017
    /// rehearsal: `try_parse_list` used to swallow *any* column-0 bullet
    /// line into the running list regardless of its marker kind, discarding
    /// each later item's own parsed `ListType` and keeping only the first
    /// item's. Since `tftio_org::ast::ListType` is one value per list (an
    /// `Ordered(start)` list renumbers every item from `start` on
    /// generation), a transcript reply that opened with `6. ...` and
    /// continued with plain `- ...` bullets got every `-` item silently
    /// renumbered into `7.`, `8.`, … on the very first `kb create` —
    /// content preserved, marker kind wrong, and permanently disagreeing
    /// with every future render of the same reply. Minimized from a real
    /// rehearsal transcript (`cc-0d9071e4...`).
    #[test]
    fn a_dash_item_after_a_numbered_item_keeps_its_own_marker() {
        let input = "* Human\n6. Item six\n- item seven\n- item eight\n";
        let (doc, residue) = parse_document_with_residue(input).unwrap();
        assert!(
            residue.is_empty(),
            "expected no dropped content: {residue:?}"
        );
        let rendered = crate::generator::generate(&doc);
        assert_eq!(rendered, input, "a `-` item must not be renumbered as `N.`");
    }

    /// A fourth, independent false-positive cause found in the T017
    /// rehearsal: `is_paragraph_line` excluded *any* `#+begin_...` line as
    /// if it opened one of the three block kinds this parser implements
    /// (`try_parse_src_block`, `try_parse_example_block`,
    /// `try_parse_quote_block`). A quoted org snippet using a block keyword
    /// this parser does not implement — `#+begin_export`,
    /// `#+begin_signature`, real org keywords this parser simply has no
    /// constructor for — matched none of the three, so it fell through to
    /// residue exactly like the first two causes: excluded from paragraph
    /// content by a check broader than what any real parser accepts, then
    /// claimed by nothing. Minimized from a real rehearsal transcript
    /// (`cc-b0f241f4...`) whose prose quoted an Emacs `org-msg` signature
    /// variable containing nested `#+begin_export`/`#+begin_signature`
    /// blocks.
    #[test]
    fn an_unimplemented_block_keyword_survives_as_paragraph_text() {
        let input = "* Human\n #+begin_signature\n #+begin_export html\n text\n #+end_export\n #+end_signature\n";
        let (doc, residue) = parse_document_with_residue(input).unwrap();
        assert!(
            residue.is_empty(),
            "expected no dropped content: {residue:?}"
        );
        let rendered = crate::generator::generate(&doc);
        assert_eq!(rendered, input);
    }

    /// A fifth, independent false-positive cause found in the T017
    /// rehearsal (redesigned after a first attempt regressed; see
    /// `is_list_continuation_line`'s doc comment for the full account): a
    /// blank line unconditionally ended a list, even when the next
    /// non-blank line was a genuine continuation of the current item —
    /// indented, structurally inert prose or a table, the shape real model
    /// output produces routinely. Ending the list handed that indented
    /// content to whichever top-level block parser recognized it instead
    /// — here, `try_parse_table`, which parses it correctly but carries no
    /// indentation in `Block::Table`, so the table's original indent was
    /// silently lost on regeneration. Minimized from a real rehearsal
    /// transcript (`cc-ec002615...`): a reply's bullet was followed, across
    /// one blank line, by an indented markdown table.
    #[test]
    fn a_list_item_continues_across_a_blank_line_into_an_indented_table() {
        let input = "* Human\n- some prose here.\n\n  | A | B |\n  |---|---|\n  | 1 | 2 |\n";
        let (doc, residue) = parse_document_with_residue(input).unwrap();
        assert!(
            residue.is_empty(),
            "expected no dropped content: {residue:?}"
        );
        let rendered = crate::generator::generate(&doc);
        assert_eq!(
            rendered, input,
            "the table's original indentation must survive as list-item continuation text"
        );
    }

    /// The regression a first version of the fix above caused, now guarded
    /// directly: `escape_org` (`kb_import.model`) indents a
    /// literal `*`/`#+` at the start of a line by exactly one column, so a
    /// reply that closed a numbered list, added a blank line, then opened
    /// `**Suggested order:**` (escaped to a one-space indent, since it
    /// starts with `*`) followed by a *fresh* `1. ...` list must not have
    /// that escaped bold line bridged into the prior list — which would
    /// then take the following `1.` line for that list's next sibling item
    /// (sharing its `Ordered` discriminant) rather than the start of a new
    /// list, renumbering it far past `1`. Minimized from a real rehearsal
    /// transcript (`cc-7b992f21...`), where a fresh `1. Clear items...`
    /// list came back numbered from `18.`.
    #[test]
    fn an_escaped_bold_line_after_a_list_does_not_bridge_into_a_fresh_list() {
        let input = "* Human\n16. item sixteen\n17. item seventeen, blocked by items 6 and 7.\n\n \
                      **Suggested order:**\n1. Clear items 1-5\n2. Decide item 6\n";
        let (doc, residue) = parse_document_with_residue(input).unwrap();
        assert!(
            residue.is_empty(),
            "expected no dropped content: {residue:?}"
        );
        let rendered = crate::generator::generate(&doc);
        assert_eq!(
            rendered, input,
            "a fresh list after an escaped bold line must keep its own numbering, \
             not continue the prior list's"
        );
    }

    /// A sixth, independent false-positive cause, found by a final
    /// rehearsal verification after the five above were fixed: an indented
    /// `# comment` line (real Python/shell prose quoted inside a
    /// transcript's fenced code block) was recognized as a `Block::Comment`
    /// — real org syntax permits this (Worg's Org Syntax §2.3 excludes only
    /// headings, inlinetasks, footnote definitions, and diary sexps from
    /// being indented; comments are not on that list) — but
    /// `Block::Comment` carries only its text, no position, so the
    /// generator always re-emitted it at column 0, losing the original
    /// indentation on every write. `try_parse_comment` (see its doc
    /// comment for the full account, including why this parser deviates
    /// from org here on purpose) now requires column 0, matching this
    /// parser's existing precedent for headings and keywords; an indented
    /// `#` line stays ordinary paragraph text and round-trips exactly.
    /// Minimized from real rehearsal transcripts (`cc-b90c37fc...`,
    /// `cc-c58df4cd...`).
    #[test]
    fn an_indented_comment_line_survives_as_paragraph_text() {
        let input =
            "* Human\n```python\ndef f():\n    # a comment about pg_temp_N\n    return 1\n```\n";
        let (doc, residue) = parse_document_with_residue(input).unwrap();
        assert!(
            residue.is_empty(),
            "expected no dropped content: {residue:?}"
        );
        let rendered = crate::generator::generate(&doc);
        assert_eq!(
            rendered, input,
            "an indented `#` line's original indentation must survive exactly"
        );
    }

    /// The horizontal-rule half of the same false-positive cause:
    /// `Block::HorizontalRule` is a bare unit variant carrying no data at
    /// all, so an indented `-----` (an ASCII-table rule quoted inside a
    /// transcript) recognized as a real horizontal rule — again permitted
    /// by real org syntax per Worg §2.3, and again requiring column 0 here
    /// for the same reason as the comment case above — had no way to
    /// regenerate with its indentation restored.
    /// `try_parse_horizontal_rule` now requires column 0. Minimized from a
    /// real rehearsal transcript (`cc-89c9fe92...`).
    #[test]
    fn an_indented_horizontal_rule_survives_as_paragraph_text() {
        let input = "* Human\nElastic IPs released       16   x $3.65     58\n                                         -----\n    total: 100\n";
        let (doc, residue) = parse_document_with_residue(input).unwrap();
        assert!(
            residue.is_empty(),
            "expected no dropped content: {residue:?}"
        );
        let rendered = crate::generator::generate(&doc);
        assert_eq!(
            rendered, input,
            "an indented `-----` line's original indentation must survive exactly"
        );
    }

    #[test]
    fn parse_bold() {
        let doc = parse_document("*bold*\n").unwrap();
        if let Block::Paragraph { inlines } = &doc.blocks[0] {
            assert_eq!(inlines.len(), 1);
            assert_eq!(inlines[0], Inline::Bold(vec![Inline::Plain("bold".into())]));
        } else {
            panic!("expected paragraph");
        }
    }

    #[test]
    fn parse_italic() {
        let doc = parse_document("/italic/\n").unwrap();
        if let Block::Paragraph { inlines } = &doc.blocks[0] {
            assert_eq!(
                inlines[0],
                Inline::Italic(vec![Inline::Plain("italic".into())])
            );
        } else {
            panic!("expected paragraph");
        }
    }

    #[test]
    fn parse_strikethrough() {
        let doc = parse_document("+struck+\n").unwrap();
        if let Block::Paragraph { inlines } = &doc.blocks[0] {
            assert_eq!(
                inlines[0],
                Inline::Strikethrough(vec![Inline::Plain("struck".into())])
            );
        } else {
            panic!("expected paragraph");
        }
    }

    #[test]
    fn parse_link_no_description() {
        let doc = parse_document("[[https://example.com]]\n").unwrap();
        if let Block::Paragraph { inlines } = &doc.blocks[0] {
            assert_eq!(
                inlines[0],
                Inline::Link {
                    target: "https://example.com".into(),
                    description: None,
                }
            );
        } else {
            panic!("expected paragraph");
        }
    }

    #[test]
    fn parse_link_with_description() {
        let doc = parse_document("[[https://example.com][example]]\n").unwrap();
        if let Block::Paragraph { inlines } = &doc.blocks[0] {
            assert_eq!(
                inlines[0],
                Inline::Link {
                    target: "https://example.com".into(),
                    description: Some("example".into()),
                }
            );
        } else {
            panic!("expected paragraph");
        }
    }

    #[test]
    fn parse_src_block() {
        let input = "#+begin_src rust\nfn main() {}\n#+end_src\n";
        let doc = parse_document(input).unwrap();
        if let Block::SrcBlock { language, content } = &doc.blocks[0] {
            assert_eq!(language, "rust");
            assert_eq!(content, "fn main() {}\n");
        } else {
            panic!("expected src block");
        }
    }

    #[test]
    fn parse_example_block() {
        let input = "#+begin_example\n$ ls\nfoo\n#+end_example\n";
        let doc = parse_document(input).unwrap();
        assert_eq!(
            doc.blocks[0],
            Block::ExampleBlock {
                content: "$ ls\nfoo\n".into(),
            }
        );
    }

    #[test]
    fn parse_property_drawer() {
        let input = ":PROPERTIES:\n:ID: abc-123\n:END:\n";
        let doc = parse_document(input).unwrap();
        if let Block::PropertyDrawer { entries } = &doc.blocks[0] {
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].0, "ID");
            // Value kept verbatim, including the space after the key colon.
            assert_eq!(entries[0].1, " abc-123");
        } else {
            panic!("expected property drawer");
        }
    }

    #[test]
    fn parse_list_unordered() {
        let input = "- one\n- two\n";
        let doc = parse_document(input).unwrap();
        if let Block::List { list_type, items } = &doc.blocks[0] {
            assert_eq!(*list_type, ListType::Unordered);
            assert_eq!(items.len(), 2);
        } else {
            panic!("expected list");
        }
    }

    #[test]
    fn parse_comment() {
        let doc = parse_document("# a comment\n").unwrap();
        assert_eq!(
            doc.blocks[0],
            Block::Comment {
                text: "a comment".into()
            }
        );
    }

    #[test]
    fn parse_horizontal_rule() {
        let doc = parse_document("-----\n").unwrap();
        assert_eq!(doc.blocks[0], Block::HorizontalRule);
    }

    #[test]
    fn parse_single_bracket_run_keeps_trailing_text() {
        // A `[...]` that is not a `[[link]]` must not drop the rest of
        // the line.
        let doc = parse_document("see [his] notes here\n").unwrap();
        assert_eq!(
            doc.blocks[0],
            Block::Paragraph {
                inlines: vec![Inline::Plain("see [his] notes here".into())],
            }
        );
    }

    #[test]
    fn residue_empty_when_every_line_claimed() {
        let input = "* Heading\n\nA paragraph.\n";
        let (_, residue) = parse_document_with_residue(input).unwrap();
        assert!(
            residue.is_empty(),
            "no line should be unclaimed: {residue:?}"
        );
    }

    #[test]
    fn an_unrecognized_block_marker_is_paragraph_text_not_residue() {
        // `#+begin_verse` opens no block kb models (src, quote, and example
        // are the only three; see `try_parse_non_heading_block`'s doc
        // comment). It used to be recorded as residue (dropped) by the same
        // defect `an_unimplemented_block_keyword_survives_as_paragraph_text`
        // regression-tests directly: `is_paragraph_line` excluded any
        // `#+begin_...` line as if every one of them opened a real block.
        // T017 narrowed that exclusion to the three kinds this parser
        // actually implements, so an unrecognized one is now ordinary
        // paragraph text, round-tripping like any other line, rather than
        // silently dropped content.
        let (doc, residue) = parse_document_with_residue("#+begin_verse\n").unwrap();
        assert!(
            residue.is_empty(),
            "must not be dropped as residue: {residue:?}"
        );
        assert_eq!(crate::generator::generate(&doc), "#+begin_verse\n");
    }

    #[test]
    fn an_unrecognized_block_marker_under_a_heading_is_paragraph_text_not_residue() {
        let input = "* Heading\n#+begin_verse\n";
        let (doc, residue) = parse_document_with_residue(input).unwrap();
        assert!(
            residue.is_empty(),
            "must not be dropped as residue: {residue:?}"
        );
        assert_eq!(crate::generator::generate(&doc), input);
    }

    #[test]
    fn residue_excludes_blank_lines() {
        let (_, residue) = parse_document_with_residue("para\n\n\n").unwrap();
        assert!(
            residue.is_empty(),
            "blank lines are not residue: {residue:?}"
        );
    }

    /// T019 pattern A (`PLAN-20260923-project-identity`): an indented
    /// `#+begin_src`/`#+end_src` pair lost its leading space on every
    /// round trip, even though the `#+name:` keyword line right above it
    /// already survived (T017 already requires column 0 for keywords).
    /// See `try_parse_src_block`'s doc comment for the cause and why this
    /// parser deliberately does not model an indented src block as
    /// `Block::SrcBlock`. Minimized from the T018 rehearsal's 17 remaining
    /// Codex differences.
    #[test]
    fn an_indented_src_block_survives_as_paragraph_text() {
        let input = "* Heading\n\n #+name: mysrc\n #+begin_src org :noweb yes\nexample content\n #+end_src\n";
        let (doc, residue) = parse_document_with_residue(input).unwrap();
        assert!(
            residue.is_empty(),
            "expected no dropped content: {residue:?}"
        );
        let rendered = crate::generator::generate(&doc);
        assert_eq!(
            rendered, input,
            "an indented #+begin_src/#+end_src pair must keep its indentation"
        );
    }

    /// T019 pattern B (`PLAN-20260923-project-identity`), table half: an
    /// indented table lost its indentation on every round trip, because
    /// `try_parse_table` used to recognize a table's opening `|` after
    /// trimming. See `try_parse_table`'s doc comment.
    #[test]
    fn an_indented_table_survives_as_paragraph_text() {
        let input = "* Heading\n\n | Field | Type |\n |---|---|\n | id | UUID |\n";
        let (doc, residue) = parse_document_with_residue(input).unwrap();
        assert!(
            residue.is_empty(),
            "expected no dropped content: {residue:?}"
        );
        let rendered = crate::generator::generate(&doc);
        assert_eq!(
            rendered, input,
            "an indented table must keep its indentation"
        );
    }

    /// T019 pattern B (`PLAN-20260923-project-identity`), drawer half: an
    /// indented `:PROPERTIES:` drawer lost its indentation on every round
    /// trip. See `try_parse_property_drawer`'s doc comment. A
    /// `:PROPERTIES:` drawer directly under a heading at column 0 is
    /// unaffected — the sibling test `parse_property_drawer` above already
    /// covers that shape.
    #[test]
    fn an_indented_property_drawer_survives_as_paragraph_text() {
        let input = "*** Note: Example\n :PROPERTIES:\n :CUSTOM_ID: n-example\n :END:\n";
        let (doc, residue) = parse_document_with_residue(input).unwrap();
        assert!(
            residue.is_empty(),
            "expected no dropped content: {residue:?}"
        );
        let rendered = crate::generator::generate(&doc);
        assert_eq!(
            rendered, input,
            "an indented :PROPERTIES: drawer must keep its indentation"
        );
    }

    /// A `:PROPERTIES:` drawer directly under a heading at column 0 still
    /// parses as a real property drawer (T019's invariant that only an
    /// indented look-alike changes).
    #[test]
    fn a_column_zero_property_drawer_under_a_heading_still_parses_as_a_drawer() {
        let input = "*** Note: Example\n:PROPERTIES:\n:CUSTOM_ID: n-example\n:END:\n";
        let doc = parse_document(input).unwrap();
        let Block::Heading { children, .. } = &doc.blocks[0] else {
            panic!("expected a heading");
        };
        assert!(
            matches!(children.first(), Some(Block::PropertyDrawer { .. })),
            "expected a real PropertyDrawer child, got {children:?}"
        );
        assert_eq!(crate::generator::generate(&doc), input);
    }

    /// T019 pattern C (`PLAN-20260923-project-identity`): two adjacent
    /// numbered lists with no blank-line separator merged into one, and
    /// the second — which restarted at `1.` — was renumbered `6.`, `7.`,
    /// continuing the first list's count, because `try_parse_list` only
    /// checked marker *kind* (T017 fix 3), not whether the number itself
    /// went backward. A restart (a number not strictly greater than the
    /// previous item's) now ends the list without consuming the line, so
    /// the second list starts fresh with its own `Ordered(1)` start value.
    /// See the restart check in `try_parse_list` for why this is the only
    /// reading that can round-trip: org renumbers ordered lists on export
    /// regardless of their source numbering, so a bare integer alone
    /// cannot distinguish "next item" from "new list" except by comparing
    /// it against the previous one. Minimized from the T018 rehearsal's 17
    /// remaining Codex differences.
    #[test]
    fn adjacent_numbered_lists_stay_separate_when_the_second_restarts() {
        let input = "* Heading\n\n1. first rule\n2. second rule\n3. third rule\n\
                      4. fourth rule\n5. fifth rule\n1. sixth rule restated as one\n\
                      2. seventh rule restated as two\n";
        let (doc, residue) = parse_document_with_residue(input).unwrap();
        assert!(
            residue.is_empty(),
            "expected no dropped content: {residue:?}"
        );
        let Block::Heading { children, .. } = &doc.blocks[0] else {
            panic!("expected a heading");
        };
        let lists: Vec<&Block> = children
            .iter()
            .filter(|b| matches!(b, Block::List { .. }))
            .collect();
        assert_eq!(
            lists.len(),
            2,
            "a restarted number must start a new list, got {lists:?}"
        );
        let rendered = crate::generator::generate(&doc);
        assert_eq!(
            rendered, input,
            "a restarted numbered list must not be renumbered as a continuation \
             of the prior one"
        );
    }

    /// An ordinary sequential ordered list (T017 fix 3's own shape, and
    /// the common case) is unaffected by the pattern-C restart check:
    /// every item's number is strictly greater than the last, so the list
    /// stays one list.
    #[test]
    fn an_ordinary_sequential_ordered_list_is_not_split() {
        let input = "1. one\n2. two\n3. three\n";
        let doc = parse_document(input).unwrap();
        assert_eq!(doc.blocks.len(), 1, "expected a single list block");
        assert!(matches!(doc.blocks[0], Block::List { .. }));
        assert_eq!(crate::generator::generate(&doc), input);
    }

    /// T019 pattern D (`PLAN-20260923-project-identity`): a line holding
    /// only whitespace and a lone `|` — a gutter column in a quoted
    /// compiler diagnostic — matched neither `try_parse_table` (which
    /// already declines a one-character row) nor `is_paragraph_line`
    /// (which excluded any trimmed `|...|` shape, including this
    /// single-character one, from paragraph content), so it fell through
    /// to residue and was silently dropped. See `is_paragraph_line`'s doc
    /// comment. Minimized from the T018 rehearsal's 17 remaining Codex
    /// differences (a `rustc`-style diagnostic quoted in a transcript).
    #[test]
    fn a_gutter_only_pipe_line_survives_as_paragraph_text() {
        let input = "* Heading\n\n   --> src/lib.rs:10:4\n    |\n10  | fn example() -> ()\n    \
                      |    ^^^^^^^\n    |\n    = note: unused\n";
        let (doc, residue) = parse_document_with_residue(input).unwrap();
        assert!(
            residue.is_empty(),
            "expected no dropped content: {residue:?}"
        );
        let rendered = crate::generator::generate(&doc);
        assert_eq!(
            rendered, input,
            "a gutter-only `|` line must survive as paragraph text"
        );
    }
}
