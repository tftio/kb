//! SQLite-backed persistence for kb.
//!
//! The schema mirrors the Haskell `KB.Storage` bootstrap. Connection
//! pragmas run on every connection; bootstrap DDL initializes a fresh
//! database. AST blobs use s-expression encoding via [`crate::sexp`]
//! matching the Haskell on-disk format.
//!
//! ## Design decisions
//!
//! AST encoding — `nodes.ast_blob` stores the AST as an s-expression via
//! `crate::sexp`. Versioned with a `(kb-doc 1 …)` wrapper.
//!
//! Tags — normalized into `node_tags` (one row per (node, tag)).
//!
//! Links — one physical `links` table holds both id-links and
//! name-links, discriminated by `link_type` ('id' | 'name'). `source_id`
//! is a FOREIGN KEY into `nodes(id)` ON DELETE CASCADE. `target_id` is
//! a FOREIGN KEY into `nodes(id)` ON DELETE SET NULL — name-link rows
//! demote to broken (`target_id` NULL) when their target is removed; the
//! `delete_node` codepath manually removes id-link rows pointing at the
//! deleted node before the cascade so the CHECK constraint (id-link
//! rows require NOT NULL `target_id`) is never violated. A schema-level
//! CHECK enforces the per-link_type column contract:
//!   id  -> `target_id` NOT NULL, `target_slug` NULL
//!   name -> `target_slug` NOT NULL (`target_id` may be NULL = broken)
//!
//! [Audit log] `audit_log.node_id` is plain `TEXT` with no foreign key,
//! so a deleted node's history survives. Every mutation
//! (`insert_node`/`update_node`/`delete_node`) writes one audit row in
//! the same transaction as the data change.
//!
//! [Schema version] Tracked via `PRAGMA user_version`. Bootstrap stamps
//! it to [`CURRENT_SCHEMA_VERSION`].
//!
//! [Title/tag derivation] `insert_node` / `update_node` take a `Document`
//! and derive the row's title and tags from its content. Title comes from
//! the first heading, or first paragraph text truncated to 80 chars,
//! or "(untitled)".

use crate::error::KbError;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};

use rusqlite::{Connection, params};

use crate::sexp;
use tftio_org::ast::{Block, Document, Inline, NodeId, Tag, Title};

// ── Schema ──────────────────────────────────────────────────────────────

/// The schema the superseded database was last written under.
///
/// Kept as a record of what this module reads rather than as something it
/// creates: nothing writes that database any more (T029). Its four versions
/// were the s-expression re-encoding of every `ast_blob`, the re-derivation
/// of `node_tags` after [`normalize_tag`] gained case-boundary splitting,
/// and the materialized `nodes.name_slug` that made name-link resolution an
/// indexed lookup. The migrations that produced them went with the writer.
pub const CURRENT_SCHEMA_VERSION: u32 = 4;

/// Open a database connection with pragmas applied.
///
/// # Errors
///
/// Returns `rusqlite::Error` on connection failure.
pub fn open_db(path: &str) -> Result<Connection, KbError> {
    // Best-effort: ensure the parent directory exists so the default
    // `$HOME/.local/share/kb/` location works on first run. A genuine
    // failure surfaces from `Connection::open` below.
    if let Some(parent) = std::path::Path::new(path).parent()
        && !parent.as_os_str().is_empty()
    {
        let _ = std::fs::create_dir_all(parent);
    }
    let conn = Connection::open(path)?;
    conn.execute_batch("PRAGMA foreign_keys = ON;")?;
    Ok(conn)
}

/// Default kb database path: `$HOME/.local/share/kb/kb.db`.
///
/// Falls back to `kb.db` in the current directory when `$HOME` is unset.
#[must_use]
pub fn default_db_path() -> std::path::PathBuf {
    dirs::home_dir().map_or_else(
        || std::path::PathBuf::from("kb.db"),
        |home| home.join(".local/share/kb/kb.db"),
    )
}

// ── Row types ───────────────────────────────────────────────────────────

/// A row from the `nodes` table.
#[derive(Debug, Clone)]
pub struct NodeRow {
    /// Stable node identifier.
    pub id: NodeId,
    /// Derived node title.
    pub title: Title,
    /// Serialized S-expression AST blob.
    pub ast_blob: String,
    /// RFC 3339 creation timestamp.
    pub created_at: String,
    /// RFC 3339 last-update timestamp.
    pub updated_at: String,
}

/// A row from the unified `links` table.
///
/// `target_id` is `Some` for every resolved row (id-links by
/// construction, name-links once their slug is matched);
/// `target_slug` is `Some` for every name-link row and `None` for
/// id-link rows.
/// Discriminator for a row in the unified `links` table: an id-link
/// (UUID `target_id`) or a name-link (bracketed `target_slug`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LinkType {
    /// A link to a node by its UUID (`target_id`).
    Id,
    /// A link to a node by bracketed slug (`target_slug`).
    Name,
}

impl LinkType {
    /// The on-disk / wire token for this link type (`"id"` or `"name"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Id => "id",
            Self::Name => "name",
        }
    }
}

impl std::fmt::Display for LinkType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl rusqlite::types::FromSql for LinkType {
    fn column_result(value: rusqlite::types::ValueRef<'_>) -> rusqlite::types::FromSqlResult<Self> {
        match value.as_str()? {
            "id" => Ok(Self::Id),
            "name" => Ok(Self::Name),
            // The schema CHECK restricts this column to 'id' | 'name', so any
            // other value can only come from external corruption.
            _ => Err(rusqlite::types::FromSqlError::InvalidType),
        }
    }
}

// `NodeId`, `Title`, and `Tag` are newtypes owned by `tftio_org`, so the orphan
// rule forbids implementing rusqlite's `FromSql` for them here. Read the
// underlying `String` column and wrap it at each call site instead.

/// A row from the unified `links` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkRow {
    /// Identifier of the node the link originates from.
    pub source_id: String,
    /// Whether the link resolves by id or by name.
    pub link_type: LinkType,
    /// Resolved target node id, when the link points to a known node.
    pub target_id: Option<String>,
    /// Raw target slug for an unresolved name link.
    pub target_slug: Option<String>,
}

/// A row from the `audit_log` table.
#[derive(Debug, Clone)]
pub struct AuditRow {
    /// Auto-increment audit row identifier.
    pub id: i64,
    /// Identifier of the affected node.
    pub node_id: String,
    /// Operation recorded (`create`, `update`, or `delete`).
    pub operation: String,
    /// Prior AST blob, when the operation replaced existing content.
    pub old_blob: Option<String>,
    /// New AST blob, when the operation stored content.
    pub new_blob: Option<String>,
    /// RFC 3339 timestamp of the operation.
    pub timestamp: String,
}

/// Graph neighborhood of a node: outgoing and incoming edges as
/// `(other-node-id, link_type)` pairs. Order is unspecified.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Neighborhood {
    /// Edges leaving this node, as `(target-id, link_type)` pairs.
    pub outgoing: Vec<(NodeId, LinkType)>,
    /// Edges entering this node, as `(source-id, link_type)` pairs.
    pub incoming: Vec<(NodeId, LinkType)>,
}

// ── Embedding write helper ──────────────────────────────────────────────

// ── Operations ──────────────────────────────────────────────────────────

/// Decode a stored `ast_blob`, attributing failure to the owning node.
fn decode_blob(node_id: &str, blob: &str) -> Result<Document, KbError> {
    sexp::decode_document(blob).map_err(|e| KbError::CorruptAstBlob {
        node_id: node_id.to_string(),
        reason: e.to_string(),
    })
}

/// Outcome of a [`crate::write::merge_tag`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagMerge {
    /// Normalized form of the tag that was merged away.
    pub from: String,
    /// Normalized form of the tag it was merged into.
    pub to: String,
    /// Nodes whose stored document was rewritten, in id order.
    pub rewritten: Vec<String>,
}

/// Why a [`crate::write::merge_tag`] call was refused before touching any record.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TagMergeRefusal {
    /// A tag argument normalized to the empty string.
    #[error("{side} tag {raw:?} normalizes to the empty string")]
    EmptyTag {
        /// Which argument was at fault.
        side: &'static str,
        /// The argument as supplied.
        raw: String,
    },
    /// Both arguments normalize to the same tag, so there is nothing to do.
    #[error("{from:?} and {to:?} both normalize to {normalized:?}; nothing to merge")]
    SameTag {
        /// The `from` argument as supplied.
        from: String,
        /// The `to` argument as supplied.
        to: String,
        /// Their shared normal form.
        normalized: String,
    },
}

/// Rewrite every tag token whose normal form is `from` to `to`, in place.
///
/// Mirrors [`extract_tags`] exactly - heading tags, `#+filetags:` keyword
/// values, and recursion through heading children and quote blocks - so a
/// tag the extractor can see is a tag the merge can rewrite. Returns
/// whether anything changed.
pub fn rewrite_tag_in_blocks(blocks: &mut [Block], from: &str, to: &str) -> bool {
    let mut changed = false;
    for block in blocks {
        match block {
            Block::Heading { tags, children, .. } => {
                for tag in tags.iter_mut() {
                    if normalize_tag(&tag.0) == from {
                        tag.0 = to.to_string();
                        changed = true;
                    }
                }
                changed |= rewrite_tag_in_blocks(children, from, to);
            }
            Block::QuoteBlock { children } => {
                changed |= rewrite_tag_in_blocks(children, from, to);
            }
            Block::Keyword { name, value } if name.eq_ignore_ascii_case("filetags") => {
                let rewritten = rewrite_filetags_value(value, from, to);
                if rewritten != *value {
                    *value = rewritten;
                    changed = true;
                }
            }
            _ => {}
        }
    }
    changed
}

/// Rewrite matching tokens inside a `#+filetags:` value, preserving the
/// colon delimiters and the surrounding whitespace org-roam writes.
fn rewrite_filetags_value(value: &str, from: &str, to: &str) -> String {
    let leading: String = value.chars().take_while(|c| c.is_whitespace()).collect();
    let trailing: String = value
        .chars()
        .rev()
        .take_while(|c| c.is_whitespace())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let core = value.trim();
    if core.is_empty() {
        return value.to_string();
    }

    let rewritten: Vec<String> = core
        .split(':')
        .map(|token| {
            if !token.is_empty() && normalize_tag(token) == from {
                to.to_string()
            } else {
                token.to_string()
            }
        })
        .collect();

    format!("{leading}{}{trailing}", rewritten.join(":"))
}

/// Outcome of a single-record tag edit ([`crate::write::add_tags`],
/// [`crate::write::remove_tags`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagEdit {
    /// Id of the node that was edited.
    pub id: String,
    /// Tags actually added or removed, normalized. Empty means the request
    /// was already satisfied and no write was performed.
    pub changed: Vec<String>,
    /// The node's tag set after the edit, normalized.
    pub tags: Vec<String>,
}

/// Why a single-node tag edit was refused before touching the node.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TagEditRefusal {
    /// A tag argument normalized to the empty string.
    #[error("tag {raw:?} normalizes to the empty string")]
    EmptyTag {
        /// The argument as supplied.
        raw: String,
    },
}

/// Normalize tag arguments, refusing any that normalize away entirely.
///
/// Duplicates within one call collapse, so `kb tags add n1 rust Rust`
/// reports one addition rather than two.
#[allow(
    clippy::missing_errors_doc,
    reason = "documented above: an argument that normalizes away is refused"
)]
pub fn normalize_arguments(tags: &[String]) -> Result<Vec<String>, KbError> {
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(tags.len());
    for raw in tags {
        let norm = normalize_tag(raw);
        if norm.is_empty() {
            return Err(KbError::TagEdit(TagEditRefusal::EmptyTag {
                raw: raw.clone(),
            }));
        }
        if seen.insert(norm.clone()) {
            out.push(norm);
        }
    }
    Ok(out)
}

/// The document's tag set as plain normalized strings, in first-seen order.
#[must_use]
pub fn tag_names(doc: &Document) -> Vec<String> {
    extract_tags(doc).into_iter().map(|t| t.0).collect()
}

/// Place already-normalized tags into a document, in place.
///
/// The single placement rule for the whole crate, shared by
/// [`crate::write::add_tags`]
/// and by the CLI's `--tag` handling on create and update. In order: an
/// existing `#+filetags:` keyword is extended, preserving its delimiter
/// style; otherwise the first top-level heading's tag list is extended;
/// otherwise a `#+filetags:` keyword is inserted at the top.
///
/// **It never invents a heading.** The `--tag` path used to prepend an
/// empty level-1 heading purely to carry tags when the document had none,
/// which put a bare `* ` line into the majority of the corpus, since the
/// documented capture shape - a `#+title:` line, a `#+filetags:` line,
/// then prose - contains no heading. `#+filetags:` is
/// org's document-level tag mechanism and needs no heading, so the last
/// branch above tags such a document without altering its structure.
///
/// Tags are expected already normalized by [`normalize_tag`]; a raw
/// spelling containing a colon would split a `#+filetags:` value into two
/// tokens.
pub fn place_tags(doc: &mut Document, additions: &[String]) {
    for block in &mut doc.blocks {
        if let Block::Keyword { name, value } = block
            && name.eq_ignore_ascii_case("filetags")
        {
            *value = extend_filetags_value(value, additions);
            return;
        }
    }

    for block in &mut doc.blocks {
        if let Block::Heading { tags, .. } = block {
            tags.extend(additions.iter().map(|t| Tag(t.clone())));
            return;
        }
    }

    doc.blocks.insert(
        0,
        Block::Keyword {
            name: "filetags".to_string(),
            // The leading space is org's convention and what org-roam
            // writes. The generator concatenates the keyword value
            // verbatim, so without it the line renders `#+filetags::a:`.
            value: format!(" :{}:", additions.join(":")),
        },
    );
}

/// Append tokens to a `#+filetags:` value in the shape it was found in.
///
/// org-roam writes the value colon-wrapped (`:a:b:`) and the keyword value
/// is stored verbatim, so both the wrapping and the surrounding whitespace
/// are preserved rather than normalized to one house style.
fn extend_filetags_value(value: &str, additions: &[String]) -> String {
    let leading: String = value.chars().take_while(|c| c.is_whitespace()).collect();
    let trailing: String = value
        .chars()
        .rev()
        .take_while(|c| c.is_whitespace())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let core = value.trim();
    let wrapped = core.is_empty() || core.starts_with(':');

    let mut tokens: Vec<String> = parse_filetags(core).collect();
    tokens.extend(additions.iter().cloned());
    let joined = tokens.join(":");

    if wrapped {
        format!("{leading}:{joined}:{trailing}")
    } else {
        format!("{leading}{joined}{trailing}")
    }
}

/// Strip every occurrence of `doomed` from a block list, in place.
///
/// Mirrors [`extract_tags`] exactly - heading tags, recursion through
/// heading children and quote blocks, and `#+filetags:` values - so a tag
/// the extractor can see is a tag this can remove. A `#+filetags:` keyword
/// emptied by the strip is dropped from the list.
#[allow(
    clippy::implicit_hasher,
    reason = "the caller is this crate, which uses the default hasher throughout"
)]
pub fn remove_tags_in_blocks(blocks: &mut Vec<Block>, doomed: &HashSet<String>) {
    let mut emptied: Vec<usize> = Vec::new();

    for (index, block) in blocks.iter_mut().enumerate() {
        match block {
            Block::Heading { tags, children, .. } => {
                tags.retain(|tag| !doomed.contains(&normalize_tag(&tag.0)));
                remove_tags_in_blocks(children, doomed);
            }
            Block::QuoteBlock { children } => remove_tags_in_blocks(children, doomed),
            Block::Keyword { name, value } if name.eq_ignore_ascii_case("filetags") => {
                let stripped = strip_filetags_value(value, doomed);
                if stripped != *value {
                    if parse_filetags(&stripped).next().is_none() {
                        emptied.push(index);
                    }
                    *value = stripped;
                }
            }
            _ => {}
        }
    }

    for index in emptied.into_iter().rev() {
        blocks.remove(index);
    }
}

/// Drop matching tokens from a `#+filetags:` value, preserving the colon
/// delimiters and surrounding whitespace of what remains.
fn strip_filetags_value(value: &str, doomed: &HashSet<String>) -> String {
    let leading: String = value.chars().take_while(|c| c.is_whitespace()).collect();
    let trailing: String = value
        .chars()
        .rev()
        .take_while(|c| c.is_whitespace())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let core = value.trim();
    if core.is_empty() {
        return value.to_string();
    }
    let wrapped = core.starts_with(':');

    let kept: Vec<String> = parse_filetags(core)
        .filter(|token| !doomed.contains(&normalize_tag(token)))
        .collect();
    let joined = kept.join(":");

    if joined.is_empty() {
        format!("{leading}{trailing}")
    } else if wrapped {
        format!("{leading}:{joined}:{trailing}")
    } else {
        format!("{leading}{joined}{trailing}")
    }
}

/// Paginated full enumeration. Returns `(NodeId, Title)` pairs ordered
/// by `updated_at` DESC. `limit = 0` returns an empty vec.
///
/// # Errors
///
/// Returns `rusqlite::Error` on database failure.
pub fn list_all_nodes(
    conn: &Connection,
    limit: usize,
    offset: usize,
) -> Result<Vec<(NodeId, Title)>, rusqlite::Error> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut stmt =
        conn.prepare("SELECT id, title FROM nodes ORDER BY updated_at DESC LIMIT ?1 OFFSET ?2")?;
    let rows = stmt.query_map(
        params![
            i64::try_from(limit).unwrap_or(i64::MAX),
            i64::try_from(offset).unwrap_or(i64::MAX)
        ],
        |row| Ok((NodeId(row.get(0)?), Title(row.get(1)?))),
    )?;
    rows.collect()
}

/// Full node data: row fields plus the decoded document and tags.
#[derive(Debug, Clone)]
pub struct NodeFullData {
    /// Derived node title.
    pub title: String,
    /// Tags attached to the node.
    pub tags: Vec<Tag>,
    /// Decoded document AST.
    pub document: Document,
    /// RFC 3339 creation timestamp.
    pub created_at: String,
    /// RFC 3339 last-update timestamp.
    pub updated_at: String,
}

/// Get full node data by ID, joining `node_tags` and decoding the sexp blob.
///
/// # Errors
///
/// Returns [`KbError::Database`] on database failure, or
/// [`KbError::CorruptAstBlob`] when the stored blob fails to decode.
/// The tags the superseded database filed against one node.
fn get_node_tags(conn: &Connection, node_id: &str) -> Result<Vec<Tag>, rusqlite::Error> {
    let mut stmt = conn.prepare("SELECT tag FROM node_tags WHERE node_id = ?1")?;
    let rows = stmt.query_map(params![node_id], |row| row.get::<_, String>(0))?;
    rows.map(|r| r.map(Tag)).collect()
}

/// One node of the superseded database, with its document and tags.
///
/// What `kb export` reads to carry the frozen archive into the store.
///
/// # Errors
///
/// [`KbError`] if the row cannot be read or its stored document cannot be
/// decoded.
pub fn get_node_full(conn: &Connection, id: &str) -> Result<Option<NodeFullData>, KbError> {
    let mut stmt =
        conn.prepare("SELECT title, ast_blob, created_at, updated_at FROM nodes WHERE id = ?1")?;
    let mut rows = stmt.query_map(params![id], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;
    match rows.next() {
        Some(Ok((title, blob, created_at, updated_at))) => {
            let doc = decode_blob(id, &blob)?;
            let tags = get_node_tags(conn, id)?;
            Ok(Some(NodeFullData {
                title,
                tags,
                document: doc,
                created_at,
                updated_at,
            }))
        }
        Some(Err(e)) => Err(KbError::Database(e)),
        None => Ok(None),
    }
}

/// How a search string is turned into an FTS5 `MATCH` expression.
///
/// The two modes exist because the forgiving default and a genuine FTS5
/// expression are mutually exclusive: quoting every token is what makes
/// arbitrary user text safe to pass to `SQLite`, and it is also what makes
/// `OR`, `NOT`, `NEAR`, phrases, and column filters inert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatchMode {
    /// Prefix-matched conjunctive keyword search: every whitespace-separated
    /// token is quoted, given a `*` prefix suffix, and joined with `AND`.
    /// Every token must appear; each matches as a prefix. No FTS5 operator
    /// survives this rewriting, so no user input can be a syntax error.
    #[default]
    Keywords,
    /// The query reaches FTS5 verbatim, so the full expression grammar is
    /// available — and a malformed expression is a reportable error rather
    /// than an empty result set.
    Fts5,
}

/// Build the FTS5 `MATCH` expression for `query` under `mode`.
///
/// Pure; `query` is assumed already trimmed and non-empty (callers short
/// circuit a blank query before reaching here).
#[must_use]
pub fn build_fts_query(query: &str, mode: MatchMode) -> String {
    match mode {
        MatchMode::Keywords => query
            .split_whitespace()
            // Two hazards, both of which let a token escape the quoting
            // that is supposed to contain it. A double quote closes the
            // phrase early, so everything after it reaches FTS5 as
            // expression syntax -- which is how `quote " then rust: language`
            // came back as `no such column: rust`; FTS5 escapes it by
            // doubling. A control character, NUL above all, terminates the
            // string inside SQLite and swallows the closing quote, so it is
            // dropped: no such character is part of any indexed token, and a
            // query is not made worse by removing what could never match.
            .map(|t| {
                let cleaned: String = t.chars().filter(|c| !c.is_control()).collect();
                cleaned.replace('"', "\"\"")
            })
            .filter(|t| !t.is_empty())
            .map(|t| format!("\"{t}\"*"))
            .collect::<Vec<_>>()
            .join(" AND "),
        MatchMode::Fts5 => query.to_string(),
    }
}

// ── Hybrid search ──────────────────────────────────────────────────────

/// Cosine similarity between two same-length vectors. Returns `0.0` if
/// the lengths differ or either vector has zero norm. Mirrors the
/// Haskell `cosineSimilarity`.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (na * nb)
}

/// Reciprocal Rank Fusion. Each input list ranks an item by 1-indexed
/// position; a node's score is `sum(1 / (k + rank))` across all lists
/// in which it appears. Returns nodes sorted by descending score.
///
/// Tie-breaking is deterministic: equal scores are returned in
/// ascending lexicographic order of the underlying node id.
#[must_use]
pub fn reciprocal_rank_fusion(k: usize, lists: &[Vec<NodeId>]) -> Vec<NodeId> {
    let mut scores: BTreeMap<String, f64> = BTreeMap::new();
    for list in lists {
        for (rank0, NodeId(id)) in list.iter().enumerate() {
            let rank = rank0 + 1;
            #[allow(
                clippy::cast_precision_loss,
                reason = "(k + rank) is a small positive rank index; the f64 cast for RRF scoring loses no significant precision"
            )]
            let contribution = 1.0_f64 / (k + rank) as f64;
            *scores.entry(id.clone()).or_insert(0.0) += contribution;
        }
    }
    let mut entries: Vec<(String, f64)> = scores.into_iter().collect();
    // BTreeMap iter is ascending by key; stable sort by score descending
    // preserves ascending key order on ties.
    entries.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    entries.into_iter().map(|(s, _)| NodeId(s)).collect()
}

/// How many nodes the vector side contributes to the fusion.
///
/// [`crate::index::Index::rank_by_embedding`] ranks *every* record holding a
/// vector for the model,
/// because cosine similarity is defined for all of them — there is no
/// equivalent of "no match". Fusing that list whole makes the result set the
/// entire corpus for any query at all, which is what `kb search` did until
/// this constant existed: 1831 of 1831 nodes returned for a query matching
/// nothing by keyword.
///
/// The cut is by **rank**, not by a similarity floor. A cosine threshold has
/// no stable meaning across models — the value separating signal from noise
/// under one embedding model is not the value that does so under another,
/// and nothing in the schema would catch a threshold silently becoming wrong
/// when the model changed. A rank cut carries no such model-dependence.
///
/// 100 is comfortably wider than any sensible page of results, so the cut
/// removes only candidates whose fused contribution was already negligible:
/// at rank 100 a node contributes 1/160 of a first-place vote.
pub const VECTOR_CANDIDATES: usize = 100;

// ── sqlite-vec extension loader ────────────────────────────────────────

/// Why loading the `sqlite-vec` dynamic extension failed.
///
/// Both variants leave the connection usable for normal queries; a
/// caller can branch on whether enabling extension loading failed
/// versus the extension file itself failing to load.
#[derive(Debug, thiserror::Error)]
pub enum VecExtensionError {
    /// Enabling extension loading (the `LoadExtensionGuard`) failed.
    #[error("enable_load_extension failed: {0}")]
    EnableLoad(rusqlite::Error),

    /// `load_extension` failed (file not found, ABI mismatch, &c.).
    #[error("load_extension failed: {0}")]
    LoadExtension(rusqlite::Error),
}

// ── Links (unified id + name graph) ────────────────────────────────────

/// A single link reference extracted from a document body.
///
/// The relinker walks the document once and emits one `LinkRef` per
/// outgoing edge — both id-link UUID targets and name-link bracket
/// slugs come out of the same walk so the body is never traversed
/// twice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkRef {
    /// `[[id:UUID]]` — the target is a node id.
    Id(String),
    /// `[[slug]]` — the target is a `#+name:` slug.
    Name(String),
}

/// A node's full link neighborhood under the unified graph.
///
/// Carries every column of every row — including broken name-links
/// that have no resolved `target_id`. Used by `kb links` so the verb
/// can surface `[[slug]] -> (broken)` rows alongside resolved edges.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LinkNeighborhood {
    /// Outgoing edges from this node. Each row carries `source_id =
    /// node_id`. Broken name-links (`target_id` NULL) ARE included.
    pub outgoing: Vec<LinkRow>,
    /// Incoming edges into this node. Each row has `target_id =
    /// node_id`; broken rows by definition have no incoming side.
    pub incoming: Vec<LinkRow>,
}

/// One hub entry: a node id, its title, and its total in-degree
/// across both link types in the unified graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubEntry {
    /// Identifier of the hub node.
    pub id: String,
    /// Derived title of the hub node.
    pub title: String,
    /// Total resolved in-degree across both link types.
    pub in_degree: i64,
}

// ── AST → row-field projection ─────────────────────────────────────────

/// Derive the `nodes.title` from an AST: a `#+title:` keyword line wins;
/// else the first heading title; else the first paragraph's plain text;
/// truncated to 80 chars; else "(untitled)".
#[must_use]
pub fn extract_title(doc: &Document) -> String {
    title_keyword(&doc.blocks)
        .or_else(|| find_first_title(&doc.blocks))
        .map_or_else(|| "(untitled)".into(), |t| truncate80(t.trim()))
}

/// The value of the first non-empty `#+title:` keyword line, if any.
fn title_keyword(blocks: &[Block]) -> Option<String> {
    blocks.iter().find_map(|block| match block {
        Block::Keyword { name, value } if name.eq_ignore_ascii_case("title") => {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        _ => None,
    })
}

fn find_first_title(blocks: &[Block]) -> Option<String> {
    for block in blocks {
        match block {
            Block::Heading {
                title, children, ..
            } => {
                if !title.0.trim().is_empty() {
                    return Some(title.0.clone());
                }
                if let Some(t) = find_first_title(children) {
                    return Some(t);
                }
            }
            Block::Paragraph { inlines } => {
                let text = inline_text(inlines);
                if !text.trim().is_empty() {
                    return Some(text);
                }
            }
            Block::QuoteBlock { children } => {
                if let Some(t) = find_first_title(children) {
                    return Some(t);
                }
            }
            _ => {}
        }
    }
    None
}

fn truncate80(s: &str) -> String {
    if s.chars().count() <= 80 {
        s.to_string()
    } else {
        s.chars().take(80).collect()
    }
}

/// Union of every tag in the document — heading tags plus any
/// `#+filetags:` keyword lines — normalized, deduplicated, in
/// first-seen order. See [`normalize_tag`] for the normal form.
#[must_use]
pub fn extract_tags(doc: &Document) -> Vec<Tag> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    collect_tags(&doc.blocks, &mut seen, &mut result);
    result
}

/// Normalize a tag to lowercase kebab-case.
///
/// Alphanumeric characters are lowercased; every run of other characters
/// (spaces, `_`, punctuation) collapses to a single `-`, with leading and
/// trailing separators trimmed. A case boundary is also treated as a
/// separator, so `silentCritic` -> `silent-critic` and `HTTPServer` ->
/// `http-server`. Applied symmetrically on the write path
/// ([`extract_tags`]) and the query path ([`crate::index::Index::records_tagged`])
/// so lookups
/// match regardless of how the caller cased or spaced the tag.
///
/// **What this does not do.** A spelling that carries no boundary at all
/// still collapses: `CICD` and `cicd` both normalize to `cicd`, which is a
/// different tag from the `ci-cd` produced by `CI/CD` or `CI CD`. No rule
/// here can recover a word break that was never written. Callers should
/// consult `kb tags` for the existing vocabulary rather than rely on
/// normalization to reconcile spellings for them.
#[must_use]
pub fn normalize_tag(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut pending_sep = false;
    let mut prev: Option<char> = None;
    let mut chars = raw.chars().peekable();

    while let Some(ch) = chars.next() {
        if !ch.is_alphanumeric() {
            pending_sep = true;
            prev = Some(ch);
            continue;
        }

        // A case boundary marks a word break the writer expressed through
        // capitalization rather than punctuation: the lowercase-or-digit to
        // uppercase transition of `silentCritic`, and the last uppercase of
        // a run that starts a new word, as the `S` in `HTTPServer`.
        if ch.is_uppercase()
            && let Some(before) = prev
        {
            let starts_word = before.is_lowercase() || before.is_numeric();
            let ends_acronym =
                before.is_uppercase() && chars.peek().is_some_and(|next| next.is_lowercase());
            if starts_word || ends_acronym {
                pending_sep = true;
            }
        }

        if pending_sep && !out.is_empty() {
            out.push('-');
        }
        pending_sep = false;
        out.extend(ch.to_lowercase());
        prev = Some(ch);
    }

    out
}

/// Parse an org `#+filetags:` value into bare tag names.
///
/// org-roam writes file tags colon-delimited (`:a:b:c:`); the keyword
/// value is verbatim, so leading/trailing padding and the surrounding
/// colons are stripped here.
fn parse_filetags(value: &str) -> impl Iterator<Item = String> + '_ {
    value
        .split(':')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Normalize `raw`, then push it onto `out` if non-empty and unseen.
fn push_tag(raw: &str, seen: &mut HashSet<String>, out: &mut Vec<Tag>) {
    let tag = normalize_tag(raw);
    if !tag.is_empty() && seen.insert(tag.clone()) {
        out.push(Tag(tag));
    }
}

fn collect_tags(blocks: &[Block], seen: &mut HashSet<String>, out: &mut Vec<Tag>) {
    for block in blocks {
        match block {
            Block::Heading { tags, children, .. } => {
                for tag in tags {
                    push_tag(&tag.0, seen, out);
                }
                collect_tags(children, seen, out);
            }
            Block::QuoteBlock { children } => {
                collect_tags(children, seen, out);
            }
            Block::Keyword { name, value } if name.eq_ignore_ascii_case("filetags") => {
                for tag in parse_filetags(value) {
                    push_tag(&tag, seen, out);
                }
            }
            _ => {}
        }
    }
}

/// Extract all plain text from a document for FTS5 body indexing.
#[must_use]
pub fn extract_body_text(doc: &Document) -> String {
    blocks_text(&doc.blocks)
}

/// [`extract_body_text`] over an arbitrary slice of blocks.
///
/// Exposed for callers that need text at a finer grain than a whole
/// document — the chunker renders one block at a time so it can split a
/// long node at block boundaries. Borrowing a slice rather than taking a
/// `Document` is what lets those callers avoid cloning every block of a
/// multi-megabyte node just to reuse this walker.
///
/// Heading titles are **not** included: this walks a heading's children and
/// skips its own title, which is the behaviour FTS body indexing has always
/// had and which callers wanting the title must supply themselves.
#[must_use]
pub fn blocks_text(blocks: &[Block]) -> String {
    walk_text(blocks, Verbatim::Include)
}

/// [`blocks_text`] with the content of verbatim blocks — `SrcBlock` and
/// `ExampleBlock` — left out.
///
/// This is the text a node is *embedded* from, and the difference from
/// [`blocks_text`] is a retrieval decision rather than a formatting one.
/// Most of this corpus by volume is imported agent-session transcripts
/// whose bulk is serialized tool calls and their output: command
/// invocations, directory listings, file dumps. That content is worth
/// keeping and worth finding by keyword, but as a vector it is close to
/// meaningless, and because a node scores the maximum over its chunks one
/// stray slab of it can carry a whole transcript above a note that answers
/// the query.
///
/// Nothing is discarded. `blocks_text` still feeds the FTS body index, so
/// every command and path remains searchable by keyword — which is the
/// better instrument for an exact identifier in any case.
#[must_use]
pub fn blocks_prose_text(blocks: &[Block]) -> String {
    walk_text(blocks, Verbatim::Exclude)
}

/// Whether a text walk descends into the content of verbatim blocks.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Verbatim {
    Include,
    Exclude,
}

fn walk_text(blocks: &[Block], verbatim: Verbatim) -> String {
    let mut texts = Vec::new();
    collect_texts(blocks, verbatim, &mut texts);
    texts
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn collect_texts(blocks: &[Block], verbatim: Verbatim, out: &mut Vec<String>) {
    for block in blocks {
        match block {
            Block::Heading { children, .. } | Block::QuoteBlock { children } => {
                collect_texts(children, verbatim, out);
            }
            Block::Paragraph { inlines } => {
                out.push(inline_text(inlines));
            }
            Block::SrcBlock { content, .. } | Block::ExampleBlock { content } => {
                if verbatim == Verbatim::Include {
                    out.push(content.clone());
                }
            }
            Block::List { items, .. } => {
                for item in items {
                    collect_texts(&item.content, verbatim, out);
                }
            }
            Block::Table { rows } => {
                for row in rows {
                    for cell in row {
                        out.push(inline_text(&cell.inlines));
                    }
                }
            }
            Block::PropertyDrawer { entries } => {
                for (_, v) in entries {
                    out.push(v.clone());
                }
            }
            Block::LogbookDrawer { entries } => {
                for entry in entries {
                    out.push(entry.note.clone());
                }
            }
            Block::Comment { text } => {
                out.push(text.clone());
            }
            Block::Keyword { value, .. } => {
                out.push(value.clone());
            }
            Block::Planning { .. } | Block::BlankLine | Block::HorizontalRule => {}
        }
    }
}

fn inline_text(inlines: &[Inline]) -> String {
    let mut parts = Vec::new();
    for inline in inlines {
        match inline {
            Inline::Plain(t) | Inline::InlineCode(t) | Inline::Verbatim(t) => parts.push(t.clone()),
            Inline::Bold(is) | Inline::Italic(is) | Inline::Strikethrough(is) => {
                parts.push(inline_text(is));
            }
            Inline::Link {
                description: Some(d),
                ..
            } => parts.push(d.clone()),
            Inline::Link { target, .. } => parts.push(target.clone()),
            Inline::LineBreak => parts.push(" ".to_string()),
        }
    }
    parts.join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── build_fts_query ────────────────────────────────────────────────

    /// The keyword rewriting is what makes the default mode forgiving and
    /// also what makes it inert to FTS5 operators. Pin it exactly: this is
    /// the string every pre-existing search test depends on.
    #[test]
    fn keyword_mode_quotes_and_prefixes_every_token() {
        assert_eq!(
            build_fts_query("rust ownership", MatchMode::Keywords),
            "\"rust\"* AND \"ownership\"*"
        );
        assert_eq!(build_fts_query("kube", MatchMode::Keywords), "\"kube\"*");
        // Interior runs of whitespace collapse; split_whitespace yields no
        // empty tokens, so no empty quoted term is ever emitted.
        assert_eq!(
            build_fts_query("a \t b\nc", MatchMode::Keywords),
            "\"a\"* AND \"b\"* AND \"c\"*"
        );
    }

    /// Keyword mode neutralizes operators by construction: this is the
    /// defect reported in kb#3, pinned as intended behavior of the default.
    #[test]
    fn keyword_mode_neutralizes_fts5_operators() {
        assert_eq!(
            build_fts_query("rust OR embedding", MatchMode::Keywords),
            "\"rust\"* AND \"OR\"* AND \"embedding\"*"
        );
        assert_eq!(
            build_fts_query("title:Alpha", MatchMode::Keywords),
            "\"title:Alpha\"*"
        );
    }

    #[test]
    fn fts5_mode_passes_the_expression_through_byte_for_byte() {
        for expr in [
            "rust OR embedding",
            "title:Alpha",
            "\"exact phrase\"",
            "rust NOT ownership",
            "kube*",
            "one",
        ] {
            assert_eq!(build_fts_query(expr, MatchMode::Fts5), expr);
        }
    }

    #[test]
    fn keywords_is_the_default_mode() {
        assert_eq!(MatchMode::default(), MatchMode::Keywords);
    }

    // ── Basic ops (unchanged behaviour) ────────────────────────────────

    #[test]
    fn title_from_first_heading() {
        let doc = Document {
            blocks: vec![Block::Heading {
                level: 1,
                title: Title("My Note".into()),
                tags: vec![],
                children: vec![],
            }],
        };
        assert_eq!(extract_title(&doc), "My Note");
    }

    #[test]
    fn title_from_paragraph_fallback() {
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![Inline::Plain("First paragraph text".into())],
            }],
        };
        assert_eq!(extract_title(&doc), "First paragraph text");
    }

    #[test]
    fn title_untitled_when_empty() {
        let doc = Document { blocks: vec![] };
        assert_eq!(extract_title(&doc), "(untitled)");
    }

    #[test]
    fn title_truncated_at_80() {
        let long = "a".repeat(100);
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![Inline::Plain(long)],
            }],
        };
        let title = extract_title(&doc);
        assert_eq!(title.chars().count(), 80);
    }

    #[test]
    fn extract_tags_collects_all() {
        let doc = Document {
            blocks: vec![
                Block::Heading {
                    level: 1,
                    title: Title("A".into()),
                    tags: vec![Tag("rust".into()), Tag("kb".into())],
                    children: vec![Block::Heading {
                        level: 2,
                        title: Title("B".into()),
                        tags: vec![Tag("testing".into())],
                        children: vec![],
                    }],
                },
                Block::Heading {
                    level: 1,
                    title: Title("C".into()),
                    tags: vec![Tag("rust".into())], // duplicate
                    children: vec![],
                },
            ],
        };
        let tags = extract_tags(&doc);
        assert_eq!(tags.len(), 3);
        assert_eq!(tags[0].0, "rust");
        assert_eq!(tags[1].0, "kb");
        assert_eq!(tags[2].0, "testing");
    }

    #[test]
    fn extract_tags_includes_filetags_keyword() {
        // org-roam document shape: a `#+filetags:` line, no heading tags.
        let doc = Document {
            blocks: vec![
                Block::Keyword {
                    name: "filetags".into(),
                    value: " :design:claude-memory:project:adr:".into(),
                },
                Block::Heading {
                    level: 1,
                    title: Title("Decision".into()),
                    tags: vec![Tag("design".into())], // duplicate of a filetag
                    children: vec![],
                },
            ],
        };
        let extracted = extract_tags(&doc);
        let tags: Vec<&str> = extracted.iter().map(|t| t.0.as_str()).collect();
        assert_eq!(tags, ["design", "claude-memory", "project", "adr"]);
    }

    #[test]
    fn normalize_tag_lowercase_kebab() {
        assert_eq!(normalize_tag("Rust"), "rust");
        assert_eq!(normalize_tag("Claude Memory"), "claude-memory");
        assert_eq!(normalize_tag("claude_memory"), "claude-memory");
        assert_eq!(normalize_tag("silent-critic"), "silent-critic");
        // Behavior change: a case boundary is a word break.
        // This previously yielded "silentcritic".
        assert_eq!(normalize_tag("silentCritic"), "silent-critic");
        assert_eq!(normalize_tag("  spaced  tag  "), "spaced-tag");
        assert_eq!(normalize_tag("+++"), "");
    }

    #[test]
    fn case_boundaries_are_word_breaks() {
        assert_eq!(normalize_tag("silentCritic"), "silent-critic");
        assert_eq!(normalize_tag("SilentCritic"), "silent-critic");
        assert_eq!(normalize_tag("HTTPServer"), "http-server");
        assert_eq!(normalize_tag("parseHTTPResponse"), "parse-http-response");
        assert_eq!(normalize_tag("kVN"), "k-vn");
        assert_eq!(normalize_tag("v2Migration"), "v2-migration");
    }

    #[test]
    fn a_spelling_with_no_boundary_still_collapses() {
        // The documented limit, and the reason the fragmentation observed in
        // the live corpus is not fixed by this rule: nothing marks the word
        // break, so nothing can recover it.
        assert_eq!(normalize_tag("CICD"), "cicd");
        assert_eq!(normalize_tag("cicd"), "cicd");
        assert_ne!(normalize_tag("CICD"), normalize_tag("CI/CD"));
        assert_eq!(normalize_tag("CI/CD"), "ci-cd");
    }

    #[test]
    fn already_normal_tags_are_unchanged() {
        for tag in [
            "rust",
            "ci-cd",
            "agent-summary",
            "claude-memory",
            "book-club",
            "v2",
        ] {
            assert_eq!(
                normalize_tag(tag),
                tag,
                "{tag} should be its own normal form"
            );
        }
    }

    #[test]
    fn extract_tags_normalizes_heading_tags() {
        let doc = Document {
            blocks: vec![Block::Heading {
                level: 1,
                title: Title("H".into()),
                tags: vec![Tag("Rust".into()), Tag("rust".into())],
                children: vec![],
            }],
        };
        let tags = extract_tags(&doc);
        // Both collapse to one normalized tag.
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].0, "rust");
    }

    #[test]
    fn extract_title_prefers_title_keyword() {
        let doc = Document {
            blocks: vec![
                Block::Keyword {
                    name: "title".into(),
                    value: " The Real Title".into(),
                },
                Block::Heading {
                    level: 1,
                    title: Title("First Heading".into()),
                    tags: vec![],
                    children: vec![],
                },
            ],
        };
        assert_eq!(extract_title(&doc), "The Real Title");
    }

    #[test]
    fn extract_title_falls_back_to_heading_without_keyword() {
        let doc = Document {
            blocks: vec![Block::Heading {
                level: 1,
                title: Title("First Heading".into()),
                tags: vec![],
                children: vec![],
            }],
        };
        assert_eq!(extract_title(&doc), "First Heading");
    }

    // ── Schema parity ──────────────────────────────────────────────────

    #[test]
    fn schema_version_constant_is_four() {
        assert_eq!(CURRENT_SCHEMA_VERSION, 4);
    }

    // ── merge_tag ──────────────────────────────────────────────────────

    // ── add_tags / remove_tags ─────────────────────────────────────────

    // ── Audit log writers ──────────────────────────────────────────────

    // ── relink_one ─────────────────────────────────────────────────────

    // ── relink_all ─────────────────────────────────────────────────────

    // ── Neighborhood ───────────────────────────────────────────────────

    // ── list_all_nodes ─────────────────────────────────────────────────

    // ── Write-path relinks ────────────────────────────────────────────

    // ── Embeddings: schema parity ─────────────────────────────────────

    // ── Embeddings: the chunking / staleness migration ────────────────

    // ── cosine_similarity ────────────────────────────────────────────

    #[test]
    fn cosine_similarity_identical_vectors_is_one() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let s = cosine_similarity(&v, &v);
        assert!((s - 1.0).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors_is_zero() {
        let s = cosine_similarity(&[1.0_f32, 0.0], &[0.0, 1.0]);
        assert!(s.abs() < 1e-6, "got {s}");
    }

    #[test]
    fn cosine_similarity_opposite_vectors_is_negative_one() {
        let s = cosine_similarity(&[1.0_f32, 0.0], &[-1.0, 0.0]);
        assert!((s + 1.0).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn cosine_similarity_length_mismatch_returns_zero() {
        let s = cosine_similarity(&[1.0_f32, 2.0], &[1.0, 2.0, 3.0]);
        assert!(s.to_bits() == 0.0_f32.to_bits(), "got {s}");
    }

    #[test]
    fn cosine_similarity_zero_norm_returns_zero() {
        let s = cosine_similarity(&[0.0_f32, 0.0, 0.0], &[1.0, 2.0, 3.0]);
        assert!(s.to_bits() == 0.0_f32.to_bits(), "got {s}");
        let s = cosine_similarity(&[1.0_f32, 2.0], &[0.0, 0.0]);
        assert!(s.to_bits() == 0.0_f32.to_bits(), "got {s}");
    }

    #[test]
    fn cosine_similarity_empty_vectors_is_zero() {
        // Empty vectors have norm 0 -> 0.
        let s = cosine_similarity(&[], &[]);
        assert!(s.to_bits() == 0.0_f32.to_bits(), "got {s}");
    }

    // ── reciprocal_rank_fusion ───────────────────────────────────────

    #[test]
    fn rrf_empty_lists_yields_empty() {
        let out = reciprocal_rank_fusion(60, &[]);
        assert!(out.is_empty());
        let out = reciprocal_rank_fusion(60, &[Vec::<NodeId>::new(), Vec::<NodeId>::new()]);
        assert!(out.is_empty());
    }

    #[test]
    fn rrf_single_list_preserves_order() {
        let l = vec![NodeId("a".into()), NodeId("b".into()), NodeId("c".into())];
        let out = reciprocal_rank_fusion(60, std::slice::from_ref(&l));
        assert_eq!(out, l);
    }

    #[test]
    fn rrf_two_lists_combine_scores() {
        let l1 = vec![NodeId("a".into()), NodeId("b".into())];
        let l2 = vec![NodeId("b".into()), NodeId("a".into())];
        // Both nodes appear in both lists. With k=60:
        //   a: 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.0325224
        //   b: 1/61 + 1/62 = same
        // Tie-broken by ascending id -> a, b.
        let out = reciprocal_rank_fusion(60, &[l1, l2]);
        assert_eq!(out, vec![NodeId("a".into()), NodeId("b".into())]);
    }

    #[test]
    fn rrf_node_only_in_one_list_ranks_lower() {
        // a in list 1 only; b in both lists.
        let l1 = vec![NodeId("a".into()), NodeId("b".into())];
        let l2 = vec![NodeId("b".into())];
        // a: 1/61 = 0.01639
        // b: 1/62 + 1/61 = 0.01613 + 0.01639 = 0.03252
        // -> b first, a second.
        let out = reciprocal_rank_fusion(60, &[l1, l2]);
        assert_eq!(out, vec![NodeId("b".into()), NodeId("a".into())]);
    }

    #[test]
    fn rrf_ties_break_by_ascending_id() {
        let l1 = vec![NodeId("z".into()), NodeId("a".into()), NodeId("m".into())];
        let l2 = vec![NodeId("z".into()), NodeId("a".into()), NodeId("m".into())];
        // All three appear at identical ranks in both lists -> equal scores.
        // Tie-break by ascending id -> a, m, z (within each rank tier).
        // But ranks differ: z@1, a@2, m@3 in both. Scores:
        //   z: 2/61
        //   a: 2/62
        //   m: 2/63
        // Distinct, so order is z, a, m.
        let out = reciprocal_rank_fusion(60, &[l1, l2]);
        assert_eq!(
            out,
            vec![NodeId("z".into()), NodeId("a".into()), NodeId("m".into())]
        );
    }

    #[test]
    fn rrf_truly_tied_scores_break_by_ascending_id() {
        // Force a real tie: same node at rank 1 in two lists has same score.
        // To engineer two distinct nodes with truly equal scores, give each
        // one the same rank in some list.
        let l1 = vec![NodeId("z".into())]; // z@1 -> 1/61
        let l2 = vec![NodeId("a".into())]; // a@1 -> 1/61
        let out = reciprocal_rank_fusion(60, &[l1, l2]);
        // Equal score -> ascending id -> a, z.
        assert_eq!(out, vec![NodeId("a".into()), NodeId("z".into())]);
    }

    // ── try_load_vec_extension ───────────────────────────────────────

    // ── Embedding write path (precomputed vector) ───────────────────

    // ── search_hybrid ────────────────────────────────────────────────

    // ── embedding_disabled (default behaviour with no precomputed embedding) ──

    // ── Unified links table: id-links + name-links coexist ────────────

    // ── The document helpers the retired backend left behind ───────────

    /// A heading nested inside a quote block is still the document's title.
    /// Titles come from wherever the first one is, because an author who
    /// wraps their opening in a quote has not thereby made the note
    /// nameless.
    #[test]
    fn a_title_is_found_however_deeply_it_is_nested() {
        let doc = Document {
            blocks: vec![Block::QuoteBlock {
                children: vec![Block::Heading {
                    level: 2,
                    title: Title("Nested".into()),
                    tags: vec![],
                    children: vec![],
                }],
            }],
        };
        assert_eq!(extract_title(&doc), "Nested");
    }

    /// A heading with no title text of its own falls through to whatever
    /// names the document next, rather than naming it with an empty string.
    #[test]
    fn an_empty_heading_does_not_name_a_document() {
        let doc = Document {
            blocks: vec![Block::Heading {
                level: 1,
                title: Title("   ".into()),
                tags: vec![],
                children: vec![Block::Heading {
                    level: 2,
                    title: Title("Inner".into()),
                    tags: vec![],
                    children: vec![],
                }],
            }],
        };
        assert_eq!(extract_title(&doc), "Inner");
    }

    /// Titles are bounded at eighty characters, because they are rendered in
    /// result lists and a paragraph-long one would push every other column
    /// off the line.
    #[test]
    fn a_long_title_is_truncated() {
        let long = "x".repeat(200);
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![Inline::Plain(long)],
            }],
        };
        let title = extract_title(&doc);
        assert_eq!(title.chars().count(), 80, "{title}");
    }

    /// Prose text skips what is not prose. A code block is content a reader
    /// wants and an embedding does not: averaging a vector over a shell
    /// transcript buries whatever the surrounding paragraph said.
    #[test]
    fn prose_text_leaves_out_verbatim_blocks() {
        let doc = Document {
            blocks: vec![
                Block::Paragraph {
                    inlines: vec![Inline::Plain("the sentence".into())],
                },
                Block::SrcBlock {
                    language: "sh".into(),
                    content: "rm -rf /".into(),
                },
            ],
        };
        let prose = blocks_prose_text(&doc.blocks);
        assert!(prose.contains("the sentence"), "{prose}");
        assert!(!prose.contains("rm -rf"), "{prose}");
        // The full walk keeps it: the difference between the two is the
        // whole reason both exist.
        assert!(blocks_text(&doc.blocks).contains("rm -rf"));
    }

    /// Inline markup contributes its text and not its punctuation, so a
    /// bolded word is one word to a reader and to a query.
    #[test]
    fn inline_markup_contributes_its_text() {
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![
                    Inline::Plain("plain ".into()),
                    Inline::Bold(vec![Inline::Plain("bold".into())]),
                    Inline::Plain(" and ".into()),
                    Inline::InlineCode("code".into()),
                ],
            }],
        };
        let text = blocks_text(&doc.blocks);
        assert!(text.contains("plain"), "{text}");
        assert!(text.contains("bold"), "{text}");
        assert!(text.contains("code"), "{text}");
        assert!(!text.contains("*bold*"), "markup leaked: {text}");
    }

    /// The body text is everything but the title, which is reported
    /// separately; repeating it would weight it twice in anything that
    /// reads both.
    #[test]
    fn body_text_is_the_document_under_its_heading() {
        let doc = Document {
            blocks: vec![Block::Heading {
                level: 1,
                title: Title("Heading".into()),
                tags: vec![],
                children: vec![Block::Paragraph {
                    inlines: vec![Inline::Plain("the body".into())],
                }],
            }],
        };
        let body = extract_body_text(&doc);
        assert!(body.contains("the body"), "{body}");
    }

    /// A tag rewrite reaches inside a quote block, because the extractor
    /// does: a tag the corpus can see is a tag a merge has to be able to
    /// change.
    #[test]
    fn a_tag_inside_a_quote_block_is_rewritten() {
        let mut doc = Document {
            blocks: vec![Block::QuoteBlock {
                children: vec![Block::Heading {
                    level: 1,
                    title: Title("Quoted".into()),
                    tags: vec![Tag("bookclub".into())],
                    children: vec![],
                }],
            }],
        };
        assert!(rewrite_tag_in_blocks(
            &mut doc.blocks,
            "bookclub",
            "book-club"
        ));
        assert_eq!(tag_names(&doc), vec!["book-club".to_owned()]);
    }

    /// Removal reaches as deep as rewriting does, and a `#+filetags:` line
    /// left holding nothing is dropped rather than left as an empty stub.
    #[test]
    fn removing_the_last_filetag_drops_the_keyword() {
        let mut doc = Document {
            blocks: vec![
                Block::Keyword {
                    name: "filetags".into(),
                    value: ":only:".into(),
                },
                Block::Paragraph {
                    inlines: vec![Inline::Plain("body".into())],
                },
            ],
        };
        let doomed: HashSet<String> = std::iter::once("only".to_owned()).collect();
        remove_tags_in_blocks(&mut doc.blocks, &doomed);
        assert!(tag_names(&doc).is_empty(), "{:?}", tag_names(&doc));
        assert!(
            !doc.blocks
                .iter()
                .any(|b| matches!(b, Block::Keyword { name, .. } if name == "filetags")),
            "an empty filetags keyword was left behind"
        );
    }

    /// The link type names itself for display and reads back from the
    /// database it is stored in; an unknown spelling is refused rather than
    /// guessed at.
    #[test]
    fn a_link_type_round_trips_through_its_name() {
        use rusqlite::types::{FromSql, ValueRef};
        assert_eq!(LinkType::Id.as_str(), "id");
        assert_eq!(LinkType::Name.as_str(), "name");
        assert_eq!(format!("{}", LinkType::Name), "name");
        assert!(matches!(
            LinkType::column_result(ValueRef::Text(b"id")),
            Ok(LinkType::Id)
        ));
        assert!(LinkType::column_result(ValueRef::Text(b"sideways")).is_err());
    }

    /// Every block kind a document can hold contributes its text, because a
    /// list item, a table cell, a drawer value and a comment are all things
    /// somebody wrote and might search for.
    #[test]
    fn every_block_kind_contributes_its_text() {
        use tftio_org::ast::{Checkbox, ListItem, ListType, LogEntry, TableCell, Timestamp};
        let doc = Document {
            blocks: vec![
                Block::List {
                    list_type: ListType::Unordered,
                    items: vec![ListItem {
                        content: vec![Block::Paragraph {
                            inlines: vec![Inline::Plain("an item".into())],
                        }],
                        checkbox: Checkbox::NoCheckbox,
                    }],
                },
                Block::Table {
                    rows: vec![vec![TableCell {
                        inlines: vec![Inline::Plain("a cell".into())],
                    }]],
                },
                Block::PropertyDrawer {
                    entries: vec![("CUSTOM".into(), "a property".into())],
                },
                Block::LogbookDrawer {
                    entries: vec![LogEntry {
                        timestamp: Timestamp("2026-08-21".into()),
                        note: "a log note".into(),
                    }],
                },
                Block::Comment {
                    text: "a comment".into(),
                },
            ],
        };
        let text = blocks_text(&doc.blocks);
        for expected in ["an item", "a cell", "a property", "a log note", "a comment"] {
            assert!(text.contains(expected), "{expected} missing from {text}");
        }
    }

    /// A link contributes what it was written to say. Its target is an
    /// address rather than prose, and indexing it would match a query on
    /// the letters of a URL.
    #[test]
    fn a_link_contributes_its_description_and_not_its_target() {
        let doc = Document {
            blocks: vec![Block::Paragraph {
                inlines: vec![
                    Inline::Link {
                        target: "https://example.invalid/xyzzy".into(),
                        description: Some("the description".into()),
                    },
                    Inline::LineBreak,
                    Inline::Plain("after".into()),
                ],
            }],
        };
        let text = blocks_text(&doc.blocks);
        assert!(text.contains("the description"), "{text}");
        assert!(!text.contains("xyzzy"), "the target leaked: {text}");
        assert!(text.contains("after"), "{text}");
    }
}
