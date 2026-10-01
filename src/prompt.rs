//! `kb prompt` — `MiniJinja` template runner over the local kb corpus.
//!
//! kb assembles; the user executes. This module:
//!
//! - Resolves template names against the user override directory
//!   (`$XDG_CONFIG_HOME/kb/prompts/<name>.j2`, `$HOME/.config` fallback)
//!   and the built-in templates embedded under `crates/kb/templates/`.
//! - Exposes a documented query surface to `MiniJinja` contexts —
//!   `recent`, `orphans`, `hubs`, `tag_frequency` as eagerly-computed
//!   values plus `by_tag`, `search`, `all_nodes`, `get`,
//!   `link_distance`, `links` as callable functions.
//! - Renders templates in `MiniJinja`'s sandboxed
//!   [`UndefinedBehavior::Strict`] mode and returns the rendered string.
//!
//! Per-node values carry a summary projection (`node.body`) defined as
//! title + the first paragraph's plain text, capped at
//! [`SUMMARY_BODY_CHAR_BUDGET`] characters. The full body remains
//! available via `node.body_full` for templates that explicitly opt in.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::path::PathBuf;

use minijinja::value::Value;
use minijinja::{Environment, Error as JinjaError, ErrorKind, UndefinedBehavior};
use serde::Serialize;
use thiserror::Error;

use crate::index::{Index, IndexError};
use crate::storage;
use tftio_org::ast::{Block, Document, Inline};

/// Failure modes of template resolution and rendering.
#[derive(Debug, Error)]
pub enum PromptError {
    /// No template with the requested name exists (neither a user
    /// override nor a built-in).
    #[error("no template named {0}")]
    TemplateNotFound(String),

    /// A user-override template file could not be read.
    #[error("failed to read user template {path}: {source}")]
    ReadTemplate {
        /// Path of the template file that failed to read.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// A query backing the template context failed.
    #[error("query layer failed: {0}")]
    Index(#[from] IndexError),

    /// A stored record could not be read back as a document.
    #[error("unreadable record: {0}")]
    Record(String),

    /// `MiniJinja` failed to parse, load, or render the template.
    #[error("template error: {0}")]
    Render(#[from] JinjaError),
}

/// Character budget for the summary projection (`node.body`). A
/// conservative window so contexts stay short by default. Templates
/// that want more reach for `node.body_full`.
pub const SUMMARY_BODY_CHAR_BUDGET: usize = 500;

/// Default cap on the number of "recent" nodes eagerly bound into the
/// template context.
pub const DEFAULT_RECENT_LIMIT: usize = 50;

/// Default cap on the number of hubs eagerly bound into the context.
pub const DEFAULT_HUBS_LIMIT: usize = 20;

/// Cap on `all_nodes`. Full enumeration of a personal corpus is fine; an
/// unbounded one should not be able to exhaust the renderer.
const ALL_NODES_LIMIT: usize = 10_000;

/// Built-in templates embedded at compile time.
///
/// Each entry is `(name, body)` where `name` is the bare template name
/// (no extension). The user-override layer in
/// [`resolve_template_source`] takes precedence over this list.
const BUILTIN_TEMPLATES: &[(&str, &str)] = &[(
    "cold-start-audit",
    include_str!("../templates/cold-start-audit.j2"),
)];

/// File extension used for templates on disk.
const TEMPLATE_EXT: &str = "j2";

// ── Public entry points ────────────────────────────────────────────────

/// A summary record describing one available template — built-in or
/// user-supplied. Used by `kb prompt list`.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct TemplateSummary {
    /// Bare template name (no extension).
    pub name: String,
    /// `"builtin"` when the template ships embedded with the crate;
    /// `"user"` when resolved from the override directory.
    pub source: String,
    /// Absolute path on disk when `source = "user"`; `None` for
    /// built-ins.
    pub path: Option<PathBuf>,
}

/// List every template available to `kb prompt`, merging built-ins with
/// user-override files. User overrides win on name collision and
/// `source` then reports `"user"`.
#[must_use]
pub fn list_templates() -> Vec<TemplateSummary> {
    list_templates_in(user_template_dir().as_deref())
}

/// `list_templates`, parameterized over the override directory. Passing
/// `None` means "no user overrides considered" — only built-ins are
/// returned. Tests use this directly with a tempdir to avoid mutating
/// process-global env.
#[must_use]
fn list_templates_in(override_dir: Option<&std::path::Path>) -> Vec<TemplateSummary> {
    let mut by_name: BTreeMap<String, TemplateSummary> = BTreeMap::new();
    for (name, _) in BUILTIN_TEMPLATES {
        by_name.insert(
            (*name).to_string(),
            TemplateSummary {
                name: (*name).to_string(),
                source: "builtin".to_string(),
                path: None,
            },
        );
    }
    if let Some(dir) = override_dir
        && let Ok(entries) = std::fs::read_dir(dir)
    {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some(TEMPLATE_EXT) {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            by_name.insert(
                stem.to_string(),
                TemplateSummary {
                    name: stem.to_string(),
                    source: "user".to_string(),
                    path: Some(path),
                },
            );
        }
    }
    by_name.into_values().collect()
}

/// Resolved template source: the raw text plus where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTemplate {
    /// Template name without extension.
    pub name: String,
    /// Where the template came from: `"builtin"` or `"user"`.
    pub source: String,
    /// Filesystem path for a user override, if any.
    pub path: Option<PathBuf>,
    /// Raw `MiniJinja` template text.
    pub body: String,
}

/// Resolve a template by name: user override wins, then built-in.
///
/// # Errors
///
/// Returns [`PromptError::TemplateNotFound`] if no template with that
/// name exists, or [`PromptError::ReadTemplate`] if a matching user
/// override file cannot be read.
pub fn resolve_template_source(name: &str) -> Result<ResolvedTemplate, PromptError> {
    resolve_template_source_in(name, user_template_dir().as_deref())
}

/// `resolve_template_source`, parameterized over the override directory.
/// Passing `None` means "no user overrides considered" — only built-ins
/// are searched. Tests use this directly with a tempdir to avoid
/// mutating process-global env.
fn resolve_template_source_in(
    name: &str,
    override_dir: Option<&std::path::Path>,
) -> Result<ResolvedTemplate, PromptError> {
    if let Some(dir) = override_dir {
        let path = dir.join(format!("{name}.{TEMPLATE_EXT}"));
        if path.is_file() {
            let body =
                std::fs::read_to_string(&path).map_err(|source| PromptError::ReadTemplate {
                    path: path.clone(),
                    source,
                })?;
            return Ok(ResolvedTemplate {
                name: name.to_string(),
                source: "user".to_string(),
                path: Some(path),
                body,
            });
        }
    }
    for (n, body) in BUILTIN_TEMPLATES {
        if *n == name {
            return Ok(ResolvedTemplate {
                name: name.to_string(),
                source: "builtin".to_string(),
                path: None,
                body: (*body).to_string(),
            });
        }
    }
    Err(PromptError::TemplateNotFound(name.to_string()))
}

/// Render the named template against the live kb corpus.
///
/// # Errors
///
/// Returns [`PromptError`] when the template cannot be resolved, a stored
/// document fails to decode, the SQL query layer fails, or `MiniJinja`
/// reports a parse/load/render error.
pub fn render_prompt(index: &Index, name: &str) -> Result<String, PromptError> {
    let resolved = resolve_template_source(name)?;
    let context = build_context(index)?;
    let mut env = build_env();
    env.add_template_owned(resolved.name.clone(), resolved.body.clone())?;
    let tmpl = env.get_template(&resolved.name)?;
    Ok(tmpl.render(context)?)
}

// ── User override directory ────────────────────────────────────────────

/// `$XDG_CONFIG_HOME/kb/prompts` (with `$HOME/.config/kb/prompts`
/// fallback). Returns `None` if neither variable is set.
#[must_use]
#[allow(
    clippy::disallowed_methods,
    reason = "XDG_CONFIG_HOME/HOME locate the optional user template override dir; sanctioned bootstrap locators (REPO_INVARIANTS.md #5b)"
)]
pub fn user_template_dir() -> Option<PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME")
        && !xdg.is_empty()
    {
        return Some(PathBuf::from(xdg).join("kb").join("prompts"));
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.is_empty()
    {
        return Some(
            PathBuf::from(home)
                .join(".config")
                .join("kb")
                .join("prompts"),
        );
    }
    dirs::config_dir().map(|d| d.join("kb").join("prompts"))
}

// ── MiniJinja environment ──────────────────────────────────────────────

fn build_env() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_undefined_behavior(UndefinedBehavior::Strict);
    env
}

// ── Context construction ───────────────────────────────────────────────

/// The eagerly-bound corpus snapshot exposed as the template context.
///
/// Function bindings (`by_tag`, `search`, `all_nodes`, `get`,
/// `link_distance`, `links`) are attached as callables so a template
/// can pull additional node sets at render time without paying for them
/// when unused.
fn build_context(index: &Index) -> Result<Value, PromptError> {
    let recent_rows = index.recent(crate::search::CORPUS_KB, DEFAULT_RECENT_LIMIT)?;
    let orphan_rows = index.orphans(crate::search::CORPUS_KB)?;
    let hub_rows = index.hubs(crate::search::CORPUS_KB, DEFAULT_HUBS_LIMIT)?;
    let tag_freq = tag_frequency(index)?;

    let recent: Vec<Value> = recent_rows
        .iter()
        .map(|(id, _)| node_value(index, id))
        .collect::<Result<_, _>>()?;
    let orphans: Vec<Value> = orphan_rows
        .iter()
        .map(|(id, _)| node_value(index, id))
        .collect::<Result<_, _>>()?;
    let hubs: Vec<Value> = hub_rows
        .iter()
        .map(|h| {
            Value::from_serialize(serde_json::json!({
                "id": h.id,
                "title": h.title,
                "in_degree": h.in_degree,
            }))
        })
        .collect();
    let tag_frequency_val: Vec<Value> = tag_freq
        .iter()
        .map(|(tag, count)| {
            Value::from_serialize(serde_json::json!({
                "tag": tag,
                "count": count,
            }))
        })
        .collect();

    let path = index_path_of(index);
    let by_tag = make_by_tag(path.clone());
    let search_fn = make_search(path.clone());
    let all_fn = make_all_nodes(path.clone());
    let get_fn = make_get(path.clone());
    let links_fn = make_links(path.clone());
    let link_distance_fn = make_link_distance(path);

    let mut ctx: BTreeMap<&'static str, Value> = BTreeMap::new();
    ctx.insert("recent", Value::from(recent));
    ctx.insert("orphans", Value::from(orphans));
    ctx.insert("hubs", Value::from(hubs));
    ctx.insert("tag_frequency", Value::from(tag_frequency_val));
    ctx.insert("by_tag", by_tag);
    ctx.insert("search", search_fn);
    ctx.insert("all_nodes", all_fn);
    ctx.insert("get", get_fn);
    ctx.insert("links", links_fn);
    ctx.insert("link_distance", link_distance_fn);

    Ok(Value::from_serialize(&ctx))
}

/// Best-effort recovery of the index's path so the template callbacks can
/// reopen their own handles. `MiniJinja` callables run in `'static` closures,
/// which precludes borrowing the one the caller has.
fn index_path_of(index: &Index) -> String {
    index
        .path()
        .unwrap_or_else(|| crate::index::default_index_path().display().to_string())
}

// ── Node projection ────────────────────────────────────────────────────

/// Project one record into the `MiniJinja` value shape:
/// `{id, title, tags, body, body_full, created_at, updated_at}`.
///
/// Read from the derived index (T029). The tags come from the record's own
/// tag rows rather than from the text, because normalizing drops the
/// `#+filetags:` keyword the tags may have been written as — the record
/// header is where a tag is a fact rather than a rendering.
fn node_value(index: &Index, id: &str) -> Result<Value, PromptError> {
    let row = index.record(id)?;
    let text = index.record_text(id)?;
    let doc = crate::parser::parse_document(&text)
        .map_err(|e| PromptError::Record(format!("{id}: {e}")))?;
    let body_full = full_body(&doc);
    let body = summary_body(row.title.as_str(), &doc);
    let tags = index.tags_of(id)?;
    Ok(Value::from_serialize(serde_json::json!({
        "id": row.record_id,
        "title": row.title,
        "tags": tags,
        "body": body,
        "body_full": body_full,
        "created_at": row.created,
        "updated_at": row.updated,
    })))
}

/// Summary projection: title + first paragraph's plain text, capped at
/// [`SUMMARY_BODY_CHAR_BUDGET`] characters. Heuristic chosen for
/// context-window discipline — see crate-level docs.
#[must_use]
pub fn summary_body(title: &str, doc: &Document) -> String {
    let first_para = first_paragraph_text(&doc.blocks).unwrap_or_default();
    let mut out = String::new();
    if !title.trim().is_empty() {
        out.push_str(title);
    }
    if !first_para.trim().is_empty() {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(&first_para);
    }
    truncate_chars(&out, SUMMARY_BODY_CHAR_BUDGET)
}

/// The document's full body text, uncapped.
///
/// `node.body_full` exists precisely for templates that have decided they
/// want everything; capping it here would make the opt-out an opt-out from
/// one cap into another.
fn full_body(doc: &Document) -> String {
    storage::extract_body_text(doc)
}

fn first_paragraph_text(blocks: &[Block]) -> Option<String> {
    for block in blocks {
        match block {
            Block::Paragraph { inlines } => {
                let text = inline_plain(inlines);
                if !text.trim().is_empty() {
                    return Some(text);
                }
            }
            Block::Heading { children, .. } | Block::QuoteBlock { children } => {
                if let Some(t) = first_paragraph_text(children) {
                    return Some(t);
                }
            }
            _ => {}
        }
    }
    None
}

fn inline_plain(inlines: &[Inline]) -> String {
    let mut out = String::new();
    for inl in inlines {
        match inl {
            Inline::Plain(s) | Inline::InlineCode(s) | Inline::Verbatim(s) => out.push_str(s),
            Inline::Bold(xs) | Inline::Italic(xs) | Inline::Strikethrough(xs) => {
                out.push_str(&inline_plain(xs));
            }
            Inline::LineBreak => out.push('\n'),
            Inline::Link {
                target,
                description,
            } => {
                if let Some(desc) = description {
                    out.push_str(desc);
                } else {
                    out.push_str(target);
                }
            }
        }
    }
    out
}

fn truncate_chars(s: &str, cap: usize) -> String {
    if s.chars().count() <= cap {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(cap).collect();
        format!("{truncated}…")
    }
}

// ── Tag frequency ──────────────────────────────────────────────────────

/// Corpus-level tag frequency table: `(tag, count)` pairs sorted by
/// count desc then tag asc.
///
/// Delegates to [`Index::tag_counts`] so this template binding and the
/// `kb tags` verb cannot drift apart.
///
/// # Errors
///
/// [`IndexError`] if the index cannot be read.
pub fn tag_frequency(index: &Index) -> Result<Vec<(String, i64)>, IndexError> {
    index.tag_counts()
}

// ── Callable bindings ──────────────────────────────────────────────────

/// Open a fresh handle on the index inside a callable closure.
/// `MiniJinja` callables are `Send + Sync + 'static` so they cannot borrow
/// the caller's; we trade the open cost for clean lifetimes.
fn open_index(path: &str) -> Result<Index, JinjaError> {
    Index::open(std::path::Path::new(path)).map_err(|e| {
        JinjaError::new(
            ErrorKind::InvalidOperation,
            format!("kb: failed to open the index {path}: {e}"),
        )
    })
}

/// Project one record inside a callable closure, lifting a read failure
/// into a `MiniJinja` error so it surfaces at render time.
fn node_value_in_callable(index: &Index, id: &str) -> Result<Value, JinjaError> {
    node_value(index, id).map_err(|e| JinjaError::new(ErrorKind::InvalidOperation, e.to_string()))
}

/// Project a list of records inside a callable closure.
fn node_values_in_callable(index: &Index, ids: &[String]) -> Result<Vec<Value>, JinjaError> {
    ids.iter()
        .map(|id| node_value_in_callable(index, id))
        .collect()
}

fn make_by_tag(path: String) -> Value {
    Value::from_function(move |tag: &str| -> Result<Value, JinjaError> {
        let index = open_index(&path)?;
        let rows = index
            .records_tagged(&storage::normalize_tag(tag))
            .map_err(|e| {
                JinjaError::new(ErrorKind::InvalidOperation, format!("by_tag failed: {e}"))
            })?;
        let ids: Vec<String> = rows.into_iter().map(|(id, _)| id).collect();
        Ok(Value::from(node_values_in_callable(&index, &ids)?))
    })
}

fn make_search(path: String) -> Value {
    Value::from_function(move |query: &str| -> Result<Value, JinjaError> {
        let index = open_index(&path)?;
        let expression = storage::build_fts_query(query.trim(), storage::MatchMode::Keywords);
        if expression.is_empty() {
            return Ok(Value::from(Vec::<Value>::new()));
        }
        let ids = index
            .search_text_in(crate::search::CORPUS_KB, &expression, None, None)
            .map_err(|e| {
                JinjaError::new(ErrorKind::InvalidOperation, format!("search failed: {e}"))
            })?;
        Ok(Value::from(node_values_in_callable(&index, &ids)?))
    })
}

fn make_all_nodes(path: String) -> Value {
    Value::from_function(move || -> Result<Value, JinjaError> {
        let index = open_index(&path)?;
        // Cap at a generous bound — full enumeration on a personal kb
        // is fine, but unbounded queries on shared kbs should not
        // accidentally OOM the renderer.
        let mut ids = index.records_in(crate::search::CORPUS_KB).map_err(|e| {
            JinjaError::new(
                ErrorKind::InvalidOperation,
                format!("all_nodes failed: {e}"),
            )
        })?;
        ids.truncate(ALL_NODES_LIMIT);
        Ok(Value::from(node_values_in_callable(&index, &ids)?))
    })
}

fn make_get(path: String) -> Value {
    Value::from_function(move |id: &str| -> Result<Value, JinjaError> {
        let index = open_index(&path)?;
        if index.record(id).is_err() {
            return Ok(Value::from(()));
        }
        node_value_in_callable(&index, id)
    })
}

fn make_links(path: String) -> Value {
    Value::from_function(move |id: &str| -> Result<Value, JinjaError> {
        let index = open_index(&path)?;
        let nb = index.links_of(id).map_err(|e| {
            JinjaError::new(ErrorKind::InvalidOperation, format!("links failed: {e}"))
        })?;
        let to_row = |r: &crate::index::IndexLinkRow| {
            serde_json::json!({
                "source_id": r.source_id,
                "link_type": r.link_type,
                "target_id": r.target_id,
                "target_slug": r.target_slug,
            })
        };
        Ok(Value::from_serialize(serde_json::json!({
            "outgoing": nb.outgoing.iter().map(to_row).collect::<Vec<_>>(),
            "incoming": nb.incoming.iter().map(to_row).collect::<Vec<_>>(),
        })))
    })
}

fn make_link_distance(path: String) -> Value {
    Value::from_function(
        move |id: &str, max_depth: i64| -> Result<Value, JinjaError> {
            let index = open_index(&path)?;
            let max = usize::try_from(max_depth.max(0)).unwrap_or(0);
            let ids = bfs_link_distance(&index, id, max).map_err(|e| {
                JinjaError::new(
                    ErrorKind::InvalidOperation,
                    format!("link_distance failed: {e}"),
                )
            })?;
            let reachable: Vec<String> = ids
                .into_iter()
                .filter(|nid| index.record(nid).is_ok())
                .collect();
            Ok(Value::from(node_values_in_callable(&index, &reachable)?))
        },
    )
}

/// Breadth-first walk of the unified link graph from `start` outward up
/// to `max_depth` hops, returning every reachable node id including
/// `start`.
fn bfs_link_distance(
    index: &Index,
    start: &str,
    max_depth: usize,
) -> Result<Vec<String>, IndexError> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();
    let mut order: Vec<String> = Vec::new();
    queue.push_back((start.to_string(), 0));
    visited.insert(start.to_string());
    while let Some((node, depth)) = queue.pop_front() {
        order.push(node.clone());
        if depth >= max_depth {
            continue;
        }
        let nb = index.links_of(&node)?;
        for row in nb.outgoing.iter().chain(nb.incoming.iter()) {
            let candidates = [row.target_id.as_ref(), Some(&row.source_id)];
            for c in candidates.into_iter().flatten() {
                if c == &node {
                    continue;
                }
                if visited.insert(c.clone()) {
                    queue.push_back((c.clone(), depth + 1));
                }
            }
        }
    }
    Ok(order)
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;
    use minijinja::value::ValueKind;

    /// A store and the index derived from it, both under one temporary
    /// directory. The callables reopen the index by path, so it has to be a
    /// real file rather than an in-memory handle.
    fn workspace() -> (tempfile::TempDir, crate::store::GitBlobStore, Index) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = crate::store::GitBlobStore::open_or_init(&dir.path().join("store"))
            .expect("open store");
        let index = Index::open_for_rebuild(&dir.path().join("index.db")).expect("open index");
        (dir, store, index)
    }

    fn insert(store: &crate::store::GitBlobStore, index: &Index, id: &str, body: &str) {
        let doc = parser::parse_document(body).expect("parse");
        let options = crate::write::WriteOptions::note(id).expect("options");
        crate::write::put_record(store, index, id, &doc, &options).expect("write record");
    }

    /// Anchors criterion 09fa365d: the cold-start-audit template ships
    /// embedded with the crate.
    #[test]
    fn cold_start_audit_template_ships() {
        let resolved =
            resolve_template_source("cold-start-audit").expect("cold-start-audit template missing");
        assert_eq!(resolved.source, "builtin");
        assert!(resolved.path.is_none());
        assert!(
            resolved.body.contains("Cold-Start Audit"),
            "template body should contain its title heading: {}",
            resolved.body
        );
        let summary = list_templates();
        assert!(
            summary.iter().any(|t| t.name == "cold-start-audit"),
            "list_templates must include cold-start-audit: {summary:?}"
        );
    }

    /// Anchors criterion ed2ea536: rendering produces text that
    /// references the corpus.
    #[test]
    fn prompt_renders_to_stdout() {
        let (_dir, store, index) = workspace();
        insert(
            &store,
            &index,
            "n1",
            "#+title: Hello World\n* heading\n\nA short paragraph here.\n",
        );
        insert(&store, &index, "n2", "* Another\n\nMore text.\n");
        let out = render_prompt(&index, "cold-start-audit").expect("render");
        assert!(out.contains("Cold-Start Audit"), "expected heading: {out}");
        assert!(out.contains("Hello World"), "expected n1 title: {out}");
    }

    /// Anchors criterion da830e76: node.body defaults to the summary
    /// projection, capped at `SUMMARY_BODY_CHAR_BUDGET`, and `node.body_full`
    /// returns the full text.
    #[test]
    fn summary_projection_default() {
        let long_para: String = "x".repeat(SUMMARY_BODY_CHAR_BUDGET * 3);
        let body = format!("#+title: Big\n\n{long_para}\n");
        let doc = parser::parse_document(&body).expect("parse");
        let summary = summary_body("Big", &doc);
        assert!(
            summary.chars().count() <= SUMMARY_BODY_CHAR_BUDGET + 1,
            "summary should be capped to ~{SUMMARY_BODY_CHAR_BUDGET} chars, got {}",
            summary.chars().count()
        );
        assert!(summary.starts_with("Big"));

        // Render a tiny template that exercises both body and body_full.
        let (_dir, store, index) = workspace();
        let options = crate::write::WriteOptions::note("n1").expect("options");
        crate::write::put_record(&store, &index, "n1", &doc, &options).expect("write record");
        let ctx = build_context(&index).expect("ctx");
        let mut env = build_env();
        env.add_template(
            "t",
            "{{ recent[0].body }}|{{ recent[0].body_full | length }}",
        )
        .expect("compile");
        let rendered = env.get_template("t").unwrap().render(ctx).expect("render");
        let parts: Vec<&str> = rendered.split('|').collect();
        assert_eq!(parts.len(), 2, "two fields rendered: {rendered}");
        assert!(
            parts[0].chars().count() <= SUMMARY_BODY_CHAR_BUDGET + 1,
            "body summary length: {}",
            parts[0].chars().count()
        );
        let full_len: usize = parts[1].parse().expect("body_full length parses");
        assert!(
            full_len > SUMMARY_BODY_CHAR_BUDGET,
            "body_full must exceed summary cap: {full_len}"
        );
    }

    /// Anchors criterion 9820d90d: the documented query surface
    /// (`recent`, `orphans`, `hubs`, `tag_frequency`, `by_tag`,
    /// `search`, `all_nodes`, `get`, `links`, `link_distance`) is
    /// callable from templates and returns sensible shapes.
    #[test]
    fn template_query_surface() {
        let (_dir, store, index) = workspace();
        insert(
            &store,
            &index,
            "n1",
            "#+title: Alpha\n#+filetags: :rust:\n* h\n\nAlpha body.\n",
        );
        insert(
            &store,
            &index,
            "n2",
            "#+title: Beta\n#+filetags: :rust:\n* h\n\nBeta body links to [[id:n1]].\n",
        );

        let mut env = build_env();
        let ctx = build_context(&index).expect("ctx");
        let template = r#"
recent={{ recent | length }}
orphans={{ orphans | length }}
hubs={{ hubs | length }}
tags={{ tag_frequency | length }}
bytag={{ by_tag("rust") | length }}
search={{ search("Alpha") | length }}
all={{ all_nodes() | length }}
get={{ get("n1").title }}
links_out={{ links("n2").outgoing | length }}
distance={{ link_distance("n1", 2) | length }}
"#;
        env.add_template("t", template).expect("compile");
        let out = env.get_template("t").unwrap().render(ctx).expect("render");
        assert!(out.contains("recent=2"), "{out}");
        assert!(out.contains("tags=1"), "{out}");
        assert!(out.contains("bytag=2"), "{out}");
        assert!(out.contains("search=1"), "{out}");
        assert!(out.contains("all=2"), "{out}");
        assert!(out.contains("get=Alpha"), "{out}");
        assert!(
            out.contains("distance=2") || out.contains("distance=1"),
            "{out}"
        );
    }

    /// The edges of the callable surface: a name nothing carries, a query
    /// with no searchable content, and an identifier no record has. Each is
    /// an answer rather than a failure, because a template that asks about
    /// something absent should render, not abort.
    #[test]
    fn the_callables_answer_for_things_that_are_not_there() {
        let (_dir, store, index) = workspace();
        insert(&store, &index, "n1", "#+title: Alpha\n* h\n\nAlpha body.\n");
        let ctx = build_context(&index).expect("ctx");
        let mut env = build_env();
        env.add_template(
            "t",
            "{{ by_tag('nobody') | length }}|{{ search('   ') | length }}|\
             {{ get('no-such') is none }}|{{ links('n1').outgoing | length }}|\
             {{ link_distance('no-such', 1) | length }}",
        )
        .expect("compile");
        let out = env.get_template("t").unwrap().render(ctx).expect("render");
        assert_eq!(out, "0|0|True|0|0", "{out}");
    }

    /// Inline markup reaches the summary as its text. A template rendering
    /// `node.body` shows what the sentence said, not how it was marked up.
    #[test]
    fn the_summary_flattens_inline_markup() {
        let doc = crate::parser::parse_document(
            "#+title: T\n* h\n\nplain *bold* /italic/ +struck+ =code=.\n",
        )
        .expect("parse");
        let summary = summary_body("T", &doc);
        for word in ["plain", "bold", "italic", "struck", "code"] {
            assert!(summary.contains(word), "{word} missing from {summary}");
        }
        assert!(!summary.contains("*bold*"), "markup leaked: {summary}");
    }

    /// Anchors criterion bb10df24: user override directory wins over
    /// built-in templates.
    #[test]
    fn user_override_resolution() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let prompts_dir = tmp.path().join("kb").join("prompts");
        std::fs::create_dir_all(&prompts_dir).expect("mkdir");
        let user_path = prompts_dir.join("cold-start-audit.j2");
        std::fs::write(&user_path, "USER OVERRIDE BODY").expect("write");

        let resolved =
            resolve_template_source_in("cold-start-audit", Some(&prompts_dir)).expect("resolve");
        assert_eq!(resolved.source, "user", "user override should win");
        assert_eq!(resolved.body, "USER OVERRIDE BODY");
        assert_eq!(resolved.path.as_deref(), Some(user_path.as_path()));

        let summaries = list_templates_in(Some(&prompts_dir));
        let cs = summaries
            .iter()
            .find(|t| t.name == "cold-start-audit")
            .expect("present");
        assert_eq!(cs.source, "user");

        // And a brand-new user template appears alongside built-ins.
        std::fs::write(prompts_dir.join("custom.j2"), "x").expect("write");
        let summaries2 = list_templates_in(Some(&prompts_dir));
        assert!(
            summaries2
                .iter()
                .any(|t| t.name == "custom" && t.source == "user")
        );

        // With `None`, only built-ins are visible — verifies the
        // parameter is load-bearing, not a no-op.
        let builtins_only = list_templates_in(None);
        assert!(
            builtins_only.iter().all(|t| t.source == "builtin"),
            "no override dir → no user-source rows"
        );
    }

    /// `MiniJinja` `Value` keeps its shape across the boundary — guard
    /// the public projection contract.
    #[test]
    fn node_value_has_expected_keys() {
        let (_dir, store, index) = workspace();
        insert(&store, &index, "n1", "#+title: T\n* h\n\nbody.\n");
        let v = node_value(&index, "n1").expect("project record");
        assert_eq!(v.kind(), ValueKind::Map);
        for key in [
            "id",
            "title",
            "tags",
            "body",
            "body_full",
            "created_at",
            "updated_at",
        ] {
            assert!(
                v.get_attr(key).is_ok(),
                "missing key {key} in node value: {v:?}"
            );
        }
    }
}
