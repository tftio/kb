//! The agent-facing tool surface, independent of the protocol carrying it.
//!
//! The first four tools, tiered by token cost: `search` returns identifiers and titles,
//! `context` returns a hit's neighbourhood, `get` returns one full document,
//! and `put` writes one. Twelve read tools sit beside them (`recent`,
//! `list_by_tag`, `tags`, `links`, `hubs`, `orphans`, `broken`, `projects`,
//! `similar` and the three `prompt_*` tools) and return the same shapes the
//! CLI verbs do, as identifiers and titles. The tiering is the point rather than an organising
//! convenience — an agent's context budget is the binding constraint, and a
//! search that returns full documents spends it on the nine results that were
//! wrong. An agent that learns searching is expensive stops searching, and
//! then answers from memory.
//!
//! **Nothing here knows about MCP.** A tool is a name, a JSON schema and a
//! function from arguments to text; the protocol adapter lives in
//! [`crate::mcp_stdio`] and is replaceable without touching a handler
//! (`REPO_INVARIANTS.md` ENG-010). That is also what makes the surface
//! testable without speaking a wire protocol, which is why every semantic
//! test here calls [`ToolSurface::call`] directly and the protocol tests
//! assert only that the wire carries what the surface produced.
//!
//! Handlers call the same domain functions the CLI does. This module adds a
//! surface, not a second implementation of retrieval or of writing.

use serde_json::{Value, json};

use crate::storage;

/// One tool the surface offers.
pub struct ToolSpec {
    /// The name an agent calls it by.
    pub name: &'static str,
    /// What it does, as the agent reads it.
    pub description: &'static str,
    /// The JSON Schema its arguments must satisfy.
    pub schema: Value,
}

/// What a tool call produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolOutcome {
    /// The tool succeeded, with this as its output.
    Text(String),
    /// The tool failed, with this as the reason. Distinct from a transport
    /// error: the call was well-formed and reached the handler, which is a
    /// fact the agent needs in order to decide whether to retry.
    Failed(String),
}

/// A set of tools an agent can call.
///
/// The seam (`REPO_INVARIANTS.md` ENG-010). A protocol adapter enumerates
/// [`ToolSurface::specs`] and dispatches through [`ToolSurface::call`], and
/// knows nothing else about kb.
pub trait ToolSurface {
    /// Every tool, with its schema.
    fn specs(&self) -> Vec<ToolSpec>;
    /// Invoke `name` with `arguments`.
    ///
    /// Never fails as a `Result`: a tool that could not do its job returns
    /// [`ToolOutcome::Failed`], because an agent needs the reason as content
    /// it can read rather than as a transport-level error it cannot.
    fn call(&self, name: &str, arguments: &Value) -> ToolOutcome;
}

/// How many results `search` returns when the caller does not say.
///
/// Smaller than the CLI's twenty. A CLI result list is scanned by a human who
/// skips what is irrelevant at no cost; an agent pays for every line in the
/// budget it has left to reason with.
pub const DEFAULT_SEARCH_RESULTS: usize = 10;

/// How many rows a list-style read tool returns when the caller does not say.
///
/// Never larger than the CLI verb's own default (`recent` 50, `hubs` 20,
/// `similar` 10), and smaller where the CLI has none, for the reason
/// [`DEFAULT_SEARCH_RESULTS`] is smaller than the CLI's twenty. A list that
/// was cut reports its full size in `total`, so a short list is never read as
/// a complete one.
pub const DEFAULT_LIST_RESULTS: usize = 10;

/// The tool surface over a kb corpus.
///
/// Two paths since T029: the content-addressed store that is authoritative,
/// and the derived index that is searched and read. The superseded database
/// is not among them, which is the point — an agent that finds a record can
/// read it, and one it writes can be found.
pub struct KbTools {
    store: std::path::PathBuf,
    index: std::path::PathBuf,
}

impl KbTools {
    /// Tools over the store and index at their configured locations.
    #[must_use]
    pub fn new() -> Self {
        Self {
            store: crate::store::configured_store_path(),
            index: crate::index::configured_index_path(),
        }
    }

    /// Tools over an explicitly named store and index.
    #[must_use]
    pub fn with_paths(
        store: impl Into<std::path::PathBuf>,
        index: impl Into<std::path::PathBuf>,
    ) -> Self {
        Self {
            store: store.into(),
            index: index.into(),
        }
    }

    /// Open the store, or say why not.
    fn open_store(&self) -> Result<crate::store::GitBlobStore, String> {
        crate::store::GitBlobStore::open_or_init(&self.store).map_err(|e| e.to_string())
    }

    /// Open the derived index, or say why not.
    fn open_index(&self) -> Result<crate::index::Index, String> {
        crate::index::Index::open(&self.index).map_err(|e| e.to_string())
    }
}

impl Default for KbTools {
    fn default() -> Self {
        Self::new()
    }
}

/// Read a required string argument.
fn required<'a>(arguments: &'a Value, name: &str) -> Result<&'a str, String> {
    arguments
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("{name} is required and must be a non-empty string"))
}

/// Read an optional positive integer argument.
fn optional_count(arguments: &Value, name: &str) -> Option<usize> {
    arguments
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
}

/// Read an optional list-of-strings argument.
fn optional_strings(arguments: &Value, name: &str) -> Vec<String> {
    arguments
        .get(name)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

impl KbTools {
    /// Identifiers and titles for a query, and nothing else.
    ///
    /// The ranking is [`crate::search::resolve`] — the planner, the fusion
    /// and the cross-encoder window `kb search` runs (T030). It was FTS5
    /// keyword matching alone until then, which served agents 0.133 authored
    /// recall@5 while the operator at a terminal was getting 0.833 for the
    /// same question.
    fn search(&self, arguments: &Value) -> Result<String, String> {
        let query = required(arguments, "query")?;
        let mut request = crate::search::Request::new(query, DEFAULT_SEARCH_RESULTS);
        if let Some(limit) = optional_count(arguments, "limit") {
            request.limit = limit;
        }
        if let Some(corpus) = arguments.get("corpus").and_then(Value::as_str) {
            corpus.clone_into(&mut request.corpus);
        }
        if let Some(project) = arguments.get("project").and_then(Value::as_str) {
            request.project =
                Some(tftio_lib::project::Slug::new(project).map_err(|e| format!("project: {e}"))?);
        }
        if let Some(context) = arguments.get("context").and_then(Value::as_str) {
            request.context = Some(context.to_owned());
        }
        let index = self.open_index()?;
        let outcome = crate::search::resolve(&index, &request).map_err(|e| e.to_string())?;
        // No provenance here, deliberately: this tier is identifiers and
        // titles only (see the module doc's tiering), and a hit carries no
        // fields it would cost an agent's budget to have paid for without
        // asking. `get` (the next tier up) carries provenance.
        let hits: Vec<Value> = outcome
            .hits
            .iter()
            .map(|hit| {
                json!({
                    "id": hit.id,
                    "title": hit.title,
                    "similarity": hit.similarity,
                    "corpus": hit.corpus,
                })
            })
            .collect();
        // The count is stated rather than left to be inferred from the array's
        // length, because an agent that reads "3 results" knows the corpus was
        // searched and answered; an empty array alone reads equally as "no
        // matches" and as "something went wrong upstream".
        // Where the time went, per stage. A service threshold on the dense
        // scan (T025) is checkable only if the stage is reported separately
        // from the cross-encoder that dominates a reranked call, and the
        // caller's own clock cannot tell the two apart.
        let mut payload = json!({
            "count": hits.len(),
            "results": hits,
            "stages_ms": outcome.stages_ms(),
        });
        // A stage that was configured and failed is reported; one that was
        // never configured is not. The first is a degraded ranking the agent
        // must be able to see (ENG-004); the second is a deployment choice,
        // and a line about it on every call would spend context to say
        // nothing had gone wrong.
        let degraded: Vec<&str> = outcome
            .notes
            .iter()
            .filter(|note| note.asked_for)
            .map(|note| note.message.as_str())
            .collect();
        if !degraded.is_empty()
            && let Some(object) = payload.as_object_mut()
        {
            object.insert("degraded".to_owned(), json!(degraded));
        }
        render(&payload)
    }

    /// What a record links to and what links back.
    ///
    /// Read from the derived index (T029), so the neighbourhood an agent sees
    /// is the neighbourhood of the corpus it just searched.
    fn context(&self, arguments: &Value) -> Result<String, String> {
        let id = required(arguments, "id")?;
        let index = self.open_index()?;
        if index.record(id).is_err() {
            return Err(format!("no node with id: {id}"));
        }
        let neighborhood = index.links_of(id).map_err(|e| e.to_string())?;
        let describe = |rows: &[crate::index::IndexLinkRow], outgoing: bool| -> Vec<Value> {
            rows.iter()
                .filter_map(|row| {
                    let target = if outgoing {
                        row.target_id.clone()
                    } else {
                        Some(row.source_id.clone())
                    }?;
                    let title = index
                        .record(&target)
                        .map(|found| found.title)
                        .unwrap_or_default();
                    Some(json!({ "id": target, "title": title }))
                })
                .collect()
        };
        render(&json!({
            "id": id,
            "links_to": describe(&neighborhood.outgoing, true),
            "linked_from": describe(&neighborhood.incoming, false),
        }))
    }

    /// One record's full text.
    ///
    /// Read from the derived index, which is what `search` ranked and what
    /// `put` wrote through (T029). Reading the superseded database here would
    /// mean an agent could find a record and then be told it does not exist.
    fn get(&self, arguments: &Value) -> Result<String, String> {
        let id = required(arguments, "id")?;
        let store = self.open_store()?;
        let index = self.open_index()?;
        let row = index
            .record(id)
            .map_err(|_| format!("no node with id: {id}"))?;
        let text = crate::write::read_raw(&store, &index, id).map_err(|e| e.to_string())?;
        let domains = index.domains_of(id).unwrap_or_default();
        let mut rendered =
            crate::org_meta::render_metadata_around(id, &row.created, &row.updated, &text);
        rendered.push_str(&crate::org_meta::provenance_text(&row, &domains));
        Ok(rendered)
    }

    /// Write one record.
    fn put(&self, arguments: &Value) -> Result<String, String> {
        let body = required(arguments, "document")?;
        let mut document = crate::parser::parse_document(body).map_err(|e| e.to_string())?;
        // The stored body never carries a metadata drawer: the id and the
        // timestamps are the database's, and a drawer in the body would be a
        // second copy free to disagree with them.
        let _ = crate::org_meta::hydrate(&mut document);
        // Normalized and placed by the same functions the CLI uses, so a tag
        // written through this surface and one written through `kb create`
        // land under the same spelling. Two normalizations would split a tag
        // across two names that no single query returns.
        let tags: Vec<String> = optional_strings(arguments, "tags")
            .iter()
            .map(|tag| storage::normalize_tag(tag))
            .filter(|tag| !tag.is_empty())
            .collect();
        if !tags.is_empty() {
            storage::place_tags(&mut document, &tags);
        }
        let id = arguments
            .get("id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map_or_else(|| uuid::Uuid::new_v4().to_string(), str::to_owned);
        // Into the store, which is what the index is derived from and
        // therefore what `search` can find (T029). Writing to the superseded
        // database would return an identifier for a record no search reaches.
        let store = self.open_store()?;
        let index = self.open_index()?;
        let options = crate::write::WriteOptions::note(&id).map_err(|e| e.to_string())?;
        let written = crate::write::put_record(&store, &index, &id, &document, &options)
            .map_err(|e| e.to_string())?;
        render(&json!({ "id": written.id, "superseded": written.superseded }))
    }
}

/// A list of `(id, title)` pairs as the tiered summary every list tool shares.
///
/// `count` is what is returned and `total` what existed, so an agent that
/// reads a truncated list can tell it was truncated (`REPO_INVARIANTS.md`
/// ENG-004).
fn summaries(pairs: &[(String, String)], limit: usize) -> Value {
    let mut payload = bounded(pairs, limit);
    if let Some(object) = payload.as_object_mut() {
        object.insert("total".to_owned(), json!(pairs.len()));
    }
    payload
}

/// [`summaries`] for a list the index already cut to `limit`, which therefore
/// cannot say how many there were and does not pretend to.
fn bounded(pairs: &[(String, String)], limit: usize) -> Value {
    let rows: Vec<Value> = pairs
        .iter()
        .take(limit)
        .map(|(id, title)| json!({ "id": id, "title": title }))
        .collect();
    json!({ "count": rows.len(), "results": rows })
}

/// One row of the unified link graph, as `kb links` reports it.
fn link_row(row: &crate::index::IndexLinkRow) -> Value {
    json!({
        "source_id": row.source_id,
        "link_type": row.link_type,
        "target_id": row.target_id,
        "target_slug": row.target_slug,
    })
}

impl KbTools {
    /// The most recently updated records, newest first.
    fn recent(&self, arguments: &Value) -> Result<String, String> {
        let limit = optional_count(arguments, "limit").unwrap_or(DEFAULT_LIST_RESULTS);
        let index = self.open_index()?;
        let pairs = index
            .recent(crate::search::CORPUS_KB, limit)
            .map_err(|e| e.to_string())?;
        render(&bounded(&pairs, limit))
    }

    /// Records carrying a tag, under the CLI's tag normalization.
    fn list_by_tag(&self, arguments: &Value) -> Result<String, String> {
        let tag = required(arguments, "tag")?;
        let limit = optional_count(arguments, "limit").unwrap_or(DEFAULT_LIST_RESULTS);
        let index = self.open_index()?;
        let pairs = index
            .records_tagged(&storage::normalize_tag(tag))
            .map_err(|e| e.to_string())?;
        render(&summaries(&pairs, limit))
    }

    /// The tag vocabulary in use, with a count for each.
    fn tags(&self, arguments: &Value) -> Result<String, String> {
        let limit = optional_count(arguments, "limit").unwrap_or(DEFAULT_LIST_RESULTS);
        let index = self.open_index()?;
        let counts = index.tag_counts().map_err(|e| e.to_string())?;
        let rows: Vec<Value> = counts
            .iter()
            .take(limit)
            .map(|(tag, count)| json!({ "tag": tag, "count": count }))
            .collect();
        render(&json!({ "count": rows.len(), "total": counts.len(), "tags": rows }))
    }

    /// A record's outgoing and incoming links, with their link types.
    fn links(&self, arguments: &Value) -> Result<String, String> {
        let id = required(arguments, "id")?;
        let index = self.open_index()?;
        if index.record(id).is_err() {
            return Err(format!("no node with id: {id}"));
        }
        let neighborhood = index.links_of(id).map_err(|e| e.to_string())?;
        render(&json!({
            "id": id,
            "outgoing": neighborhood.outgoing.iter().map(link_row).collect::<Vec<_>>(),
            "incoming": neighborhood.incoming.iter().map(link_row).collect::<Vec<_>>(),
        }))
    }

    /// The most-linked records.
    fn hubs(&self, arguments: &Value) -> Result<String, String> {
        let limit = optional_count(arguments, "limit").unwrap_or(DEFAULT_LIST_RESULTS);
        let index = self.open_index()?;
        let hubs = index
            .hubs(crate::search::CORPUS_KB, limit)
            .map_err(|e| e.to_string())?;
        let rows: Vec<Value> = hubs
            .iter()
            .map(|hub| json!({ "id": hub.id, "title": hub.title, "in_degree": hub.in_degree }))
            .collect();
        render(&json!({ "count": rows.len(), "results": rows }))
    }

    /// Records nothing links to.
    fn orphans(&self, arguments: &Value) -> Result<String, String> {
        let limit = optional_count(arguments, "limit").unwrap_or(DEFAULT_LIST_RESULTS);
        let index = self.open_index()?;
        let pairs = index
            .orphans(crate::search::CORPUS_KB)
            .map_err(|e| e.to_string())?;
        render(&summaries(&pairs, limit))
    }

    /// Bracket references that resolve to no record.
    fn broken(&self, arguments: &Value) -> Result<String, String> {
        let limit = optional_count(arguments, "limit").unwrap_or(DEFAULT_LIST_RESULTS);
        let index = self.open_index()?;
        let rows = index.broken_links().map_err(|e| e.to_string())?;
        let shown: Vec<Value> = rows.iter().take(limit).map(link_row).collect();
        render(&json!({ "count": shown.len(), "total": rows.len(), "results": shown }))
    }

    /// Every project slug asserted by a record, with counts.
    fn projects(&self) -> Result<String, String> {
        let index = self.open_index()?;
        let projects = index.project_counts().map_err(|e| e.to_string())?;
        let no_project = index.records_without_project().map_err(|e| e.to_string())?;
        render(&json!({
            "projects": projects.iter().map(|p| json!({
                "project": p.project,
                "count": p.count,
                "newest": p.newest,
            })).collect::<Vec<_>>(),
            "noProject": no_project,
        }))
    }

    /// The records nearest a record by stored embedding.
    fn similar(&self, arguments: &Value) -> Result<String, String> {
        let id = required(arguments, "id")?;
        let limit = optional_count(arguments, "limit").unwrap_or(DEFAULT_LIST_RESULTS);
        let index = self.open_index()?;
        let pairs = crate::search::similar(&index, id, limit).map_err(|e| e.to_string())?;
        render(&bounded(&pairs, limit))
    }

    /// The prompt templates available, built in and overridden.
    fn prompt_list() -> Result<String, String> {
        let templates: Vec<Value> = crate::prompt::list_templates()
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "source": t.source,
                    "path": t.path.as_ref().map(|p| p.to_string_lossy().into_owned()),
                })
            })
            .collect();
        render(&json!({ "count": templates.len(), "templates": templates }))
    }

    /// One template's raw source.
    fn prompt_show(arguments: &Value) -> Result<String, String> {
        let name = required(arguments, "name")?;
        let resolved = crate::prompt::resolve_template_source(name).map_err(|e| e.to_string())?;
        render(&json!({
            "name": resolved.name,
            "source": resolved.source,
            "path": resolved.path.as_ref().map(|p| p.to_string_lossy().into_owned()),
            "body": resolved.body,
        }))
    }

    /// A template rendered against the live corpus, as text.
    fn prompt_render(&self, arguments: &Value) -> Result<String, String> {
        let name = required(arguments, "name")?;
        let index = self.open_index()?;
        crate::prompt::render_prompt(&index, name).map_err(|e| e.to_string())
    }
}

/// Render a value as the compact JSON a tool returns.
fn render(value: &Value) -> Result<String, String> {
    serde_json::to_string(value).map_err(|e| e.to_string())
}

/// The tools that list records and tags: `recent`, `list_by_tag` and `tags`.
fn read_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "recent",
            description: "The most recently updated records, newest first, as identifiers and titles. Use it to see what was added or changed lately; use `search` for a topic and `list_by_tag` for a tag.",
            schema: json!({
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "How many records to return. Defaults to 10.",
                    },
                },
                "required": [],
            }),
        },
        ToolSpec {
            name: "list_by_tag",
            description: "Records carrying a tag, as identifiers and titles. The tag is normalized as stored tags are, so casing and separators need not match (`Silent Critic` finds `silent-critic`), but a spelling with no word break ('CICD' against 'ci-cd') is a different tag. An empty result is not proof of absence: call `tags` to see the spellings that exist.",
            schema: json!({
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "The tag, without surrounding colons.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "How many records to return. Defaults to 10; `total` reports how many matched.",
                    },
                },
                "required": ["tag"],
            }),
        },
        ToolSpec {
            name: "tags",
            description: "The tag vocabulary in use, with a record count for each, most used first. Read it before tagging a record and reuse an existing spelling: a variant splits records across two tags that no single query returns. Also the first step when `list_by_tag` came back empty.",
            schema: json!({
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "How many tags to return. Defaults to 10; `total` reports how many exist.",
                    },
                },
                "required": [],
            }),
        },
    ]
}

/// The tools that read the link graph and the project index.
fn graph_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "links",
            description: "A record's outgoing and incoming links, each row as `{source_id, link_type, target_id, target_slug}`. `link_type` is `id` or `name`; a `name` row with a null `target_id` is a bracket reference no record answers. Use `context` for neighbour titles; use this when link types matter.",
            schema: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "The record's identifier." },
                },
                "required": ["id"],
            }),
        },
        ToolSpec {
            name: "hubs",
            description: "The most-linked records, ranked by in-degree across id-links and resolved name-links, as `{id, title, in_degree}`. Shows which records the corpus is organised around.",
            schema: json!({
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "How many hubs to return. Defaults to 10.",
                    },
                },
                "required": [],
            }),
        },
        ToolSpec {
            name: "orphans",
            description: "Records nothing links to under either link type, as identifiers and titles. Use it to find isolated records; use `list_by_tag`, `recent` or `search` to find records by tag, date or content.",
            schema: json!({
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "How many records to return. Defaults to 10; `total` reports how many exist.",
                    },
                },
                "required": [],
            }),
        },
        ToolSpec {
            name: "broken",
            description: "Bracket references that resolve to no record: `name` link rows whose `target_id` is null because no record carries that `#+name:` slug. Id-links cannot be broken. Use `links` to inspect one record.",
            schema: json!({
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "How many rows to return. Defaults to 10; `total` reports how many exist.",
                    },
                },
                "required": [],
            }),
        },
        ToolSpec {
            name: "projects",
            description: "Every project slug asserted by at least one record, with its record count and its newest record's date, plus `noProject`, the count of records with none. Call it before filtering `search` by `project`. For the records themselves use `search`.",
            schema: json!({
                "type": "object",
                "properties": {
                },
                "required": [],
            }),
        },
    ]
}

/// The tools that read neighbourhoods and templates: `similar` and the
/// `prompt_*` tools.
fn neighbour_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "similar",
            description: "Records semantically nearest a record you already hold, by stored embeddings rather than shared words; the record is never its own neighbour. Use it to follow a thread outward or to spot duplicates. Use `search` when you have a topic and no record. It reads stored vectors only, so it fails rather than degrading for a record that was never embedded, and the error names `kb embed` (the backfill) as the remedy.",
            schema: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "The record's identifier." },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "How many neighbours to return. Defaults to 10.",
                    },
                },
                "required": ["id"],
            }),
        },
        ToolSpec {
            name: "prompt_list",
            description: "The prompt templates `prompt_render` can render: built-ins plus user overrides, which win on a name collision. Each is `{name, source, path}`.",
            schema: json!({
                "type": "object",
                "properties": {
                },
                "required": [],
            }),
        },
        ToolSpec {
            name: "prompt_show",
            description: "One prompt template's raw source, whichever of the user override and the built-in wins. Read it to see what a template does before rendering it with `prompt_render`.",
            schema: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Template name, without extension.",
                    },
                },
                "required": ["name"],
            }),
        },
        ToolSpec {
            name: "prompt_render",
            description: "Render a named template against the live corpus and return the text. Templates can read `recent`, `orphans`, `hubs` and `tag_frequency`, and call `by_tag`, `search`, `all_nodes`, `get`, `links` and `link_distance`. kb assembles the text; the caller runs it, since kb sends it to no model.",
            schema: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Template name, without extension.",
                    },
                },
                "required": ["name"],
            }),
        },
    ]
}

impl ToolSurface for KbTools {
    fn specs(&self) -> Vec<ToolSpec> {
        vec![
            ToolSpec {
                name: "search",
                description: "Find records by content. Ranks keyword and semantic matches together, \
                     so a question in your own words works as well as the exact terms. Returns \
                     identifiers and titles only — call `get` for a record's text, and `context` \
                     for what it links to. Prefer several narrow searches over one broad query.",
                schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for, as keywords or as a question.",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "How many results to return. Defaults to 10.",
                        },
                        "corpus": {
                            "type": "string",
                            "enum": ["kb", "mail"],
                            "description": "Which corpus to search: `kb` for notes, the default, \
                                            or `mail` for the message archive.",
                        },
                        "project": {
                            "type": "string",
                            "description": "Restrict results to records asserting this project \
                                            slug. Omit for no project filter.",
                        },
                        "context": {
                            "type": "string",
                            "description": "Restrict results to records asserting this context \
                                            (`personal`, `work`, ...). Omit for no context filter.",
                        },
                    },
                    "required": ["query"],
                }),
            },
            ToolSpec {
                name: "context",
                description: "What a record links to and what links back to it, as identifiers and \
                     titles. The cheap way to decide which neighbouring record is worth \
                     fetching.",
                schema: json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "The record's identifier." },
                    },
                    "required": ["id"],
                }),
            },
            ToolSpec {
                name: "get",
                description: "One record's full text, in org-mode. The expensive tool: fetch a record \
                     only after `search` or `context` has given you a reason to.",
                schema: json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "The record's identifier." },
                    },
                    "required": ["id"],
                }),
            },
            ToolSpec {
                name: "put",
                description: "Write a new record. Deliberate: this stores something permanently, so \
                     use it for knowledge worth keeping rather than for working notes.",
                schema: json!({
                    "type": "object",
                    "properties": {
                        "document": {
                            "type": "string",
                            "description": "The record's body, as org-mode text.",
                        },
                        "id": {
                            "type": "string",
                            "description": "An identifier to store it under. One is minted if \
                                            omitted.",
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Tags to file it under.",
                        },
                    },
                    "required": ["document"],
                }),
            },
        ]
        .into_iter()
        .chain(read_specs())
        .chain(graph_specs())
        .chain(neighbour_specs())
        .collect()
    }

    fn call(&self, name: &str, arguments: &Value) -> ToolOutcome {
        let outcome = match name {
            "search" => self.search(arguments),
            "context" => self.context(arguments),
            "get" => self.get(arguments),
            "put" => self.put(arguments),
            "recent" => self.recent(arguments),
            "list_by_tag" => self.list_by_tag(arguments),
            "tags" => self.tags(arguments),
            "links" => self.links(arguments),
            "hubs" => self.hubs(arguments),
            "orphans" => self.orphans(arguments),
            "broken" => self.broken(arguments),
            "projects" => self.projects(),
            "similar" => self.similar(arguments),
            "prompt_list" => Self::prompt_list(),
            "prompt_show" => Self::prompt_show(arguments),
            "prompt_render" => self.prompt_render(arguments),
            other => Err(format!("no such tool: {other}")),
        };
        match outcome {
            Ok(text) => ToolOutcome::Text(text),
            Err(reason) => ToolOutcome::Failed(reason),
        }
    }
}
