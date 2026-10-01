//! The agent-facing tool surface, independent of the protocol carrying it.
//!
//! Four tools, tiered by token cost: `search` returns identifiers and titles,
//! `context` returns a hit's neighbourhood, `get` returns one full document,
//! and `put` writes one. The tiering is the point rather than an organising
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

/// Render a value as the compact JSON a tool returns.
fn render(value: &Value) -> Result<String, String> {
    serde_json::to_string(value).map_err(|e| e.to_string())
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
    }

    fn call(&self, name: &str, arguments: &Value) -> ToolOutcome {
        let outcome = match name {
            "search" => self.search(arguments),
            "context" => self.context(arguments),
            "get" => self.get(arguments),
            "put" => self.put(arguments),
            other => Err(format!("no such tool: {other}")),
        };
        match outcome {
            Ok(text) => ToolOutcome::Text(text),
            Err(reason) => ToolOutcome::Failed(reason),
        }
    }
}
