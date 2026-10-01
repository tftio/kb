//! The MCP tool surface over real stdio (T020).
//!
//! The tool schema is API that agents build prompts against, so it is covered
//! the way CLI-002 covers console output: by speaking the actual protocol to
//! the actual binary against a real database, rather than by calling the
//! handlers in process. A schema that is right in Rust and wrong on the wire
//! is wrong.
//!
//! Framing is newline-delimited JSON-RPC, which is what MCP's stdio transport
//! specifies; each request is one line and each response is one line.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// A running `kb-mcp` process, with the session already initialized.
struct Session {
    child: Child,
    /// `None` once the session has been shut down. Closing stdin is how the
    /// server is told to stop: killing it instead leaves it no chance to flush
    /// its coverage profile, which is why these tests would otherwise report
    /// the entire MCP surface as unexecuted.
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    next_id: i64,
}

impl Session {
    /// Start a server against `db` and complete the initialize handshake.
    fn start(db: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Self::start_with(db, &[])
    }

    /// [`Session::start`] with `env` layered on, for the corpora served by
    /// backends outside the database.
    fn start_with(db: &Path, env: &[(&str, &Path)]) -> Result<Self, Box<dyn std::error::Error>> {
        let home = db
            .parent()
            .ok_or("db path has no parent for HOME sandbox")?;
        let mut command = Command::new(env!("CARGO_BIN_EXE_kb-mcp"));
        for (key, value) in env {
            command.env(key, value);
        }
        let mut child = command
            .env("HOME", home)
            .env("XDG_CONFIG_HOME", home)
            .env("KB_DB_PATH", db)
            .env_remove("KB_EMBEDDING_BASE_URL")
            .env_remove("KB_EMBEDDING_MODEL")
            .env_remove("KB_RERANK_BASE_URL")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = child.stdin.take().ok_or("no stdin on the server")?;
        let stdout = BufReader::new(child.stdout.take().ok_or("no stdout on the server")?);
        let mut session = Self {
            child,
            stdin: Some(stdin),
            stdout,
            next_id: 1,
        };
        session.request(
            "initialize",
            &json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": { "name": "kb-tests", "version": "0" },
            }),
        )?;
        session.notify("notifications/initialized", &json!({}))?;
        Ok(session)
    }

    /// Send a request and read its response.
    fn request(
        &mut self,
        method: &str,
        params: &Value,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let id = self.next_id;
        self.next_id += 1;
        let message = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let stdin = self.stdin.as_mut().ok_or("the session is shut down")?;
        writeln!(stdin, "{message}")?;
        stdin.flush()?;
        loop {
            let mut line = String::new();
            if self.stdout.read_line(&mut line)? == 0 {
                return Err(format!("the server closed while waiting for {method}").into());
            }
            let parsed: Value = serde_json::from_str(line.trim())?;
            // Notifications carry no id and are not answers to anything.
            if parsed.get("id").and_then(Value::as_i64) == Some(id) {
                return Ok(parsed);
            }
        }
    }

    /// Send a notification, which has no response.
    fn notify(&mut self, method: &str, params: &Value) -> Result<(), Box<dyn std::error::Error>> {
        let message = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        let stdin = self.stdin.as_mut().ok_or("the session is shut down")?;
        writeln!(stdin, "{message}")?;
        stdin.flush()?;
        Ok(())
    }

    /// Call a tool and return the text of its first content block.
    fn call(
        &mut self,
        name: &str,
        arguments: &Value,
    ) -> Result<(String, bool), Box<dyn std::error::Error>> {
        let response = self.request(
            "tools/call",
            &json!({ "name": name, "arguments": arguments }),
        )?;
        let result = response
            .get("result")
            .ok_or_else(|| format!("no result in {response}"))?;
        let failed = result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let text = result
            .get("content")
            .and_then(Value::as_array)
            .and_then(|blocks| blocks.first())
            .and_then(|block| block.get("text"))
            .and_then(Value::as_str)
            .ok_or_else(|| format!("no text content in {result}"))?
            .to_owned();
        Ok((text, failed))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Closing stdin gives the server EOF, which is how a stdio transport
        // is told the session is over. It exits on its own, and only then does
        // it write out what it executed.
        drop(self.stdin.take());
        let _ = self.child.wait();
    }
}

/// A database of three notes, one of which links to another.
///
/// Bodies go through the parser rather than being assembled as an AST by
/// hand, so `[[id:alpha]]` becomes a link the graph can see. A hand-built
/// `Inline::Plain` containing the same characters is not a link, and a test
/// built that way would assert that the link graph is empty.
fn populated() -> Result<(tempfile::TempDir, std::path::PathBuf), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    // The store and index live where a sandboxed HOME puts them, because the
    // server and the CLI both resolve them from configuration rather than
    // being told (T029).
    let root = dir.path().join(".local/share/kb");
    std::fs::create_dir_all(&root)?;
    let store = kb::store::GitBlobStore::open_or_init(&root.join("store"))?;
    let index = kb::index::Index::open_for_rebuild(&root.join("index.db"))?;
    for (id, body) in [
        (
            "alpha",
            "* Ownership in Rust\n\nthe borrow checker and its rules\n",
        ),
        (
            "beta",
            "* Embedding models\n\nvectors for retrieval over notes\n",
        ),
        (
            "gamma",
            "* Retrieval notes\n\nsee [[id:alpha]] for the ownership piece\n",
        ),
    ] {
        let document = kb::parser::parse_document(body)?;
        let options = kb::write::WriteOptions::note(id)?;
        kb::write::put_record(&store, &index, id, &document, &options)?;
    }
    let db = dir.path().join("kb.db");
    Ok((dir, db))
}

#[test]
fn the_server_announces_four_tools_with_schemas() -> TestResult {
    let (_dir, db) = populated()?;
    let mut session = Session::start(&db)?;
    let response = session.request("tools/list", &json!({}))?;
    let tools = response
        .get("result")
        .and_then(|result| result.get("tools"))
        .and_then(Value::as_array)
        .ok_or_else(|| format!("no tools in {response}"))?;
    let names: Vec<&str> = tools
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .collect();
    assert_eq!(names, vec!["search", "context", "get", "put"]);
    for tool in tools {
        let schema = tool
            .get("inputSchema")
            .ok_or_else(|| format!("no schema on {tool}"))?;
        assert_eq!(
            schema.get("type").and_then(Value::as_str),
            Some("object"),
            "a tool schema is not an object schema: {tool}"
        );
        assert!(
            tool.get("description")
                .and_then(Value::as_str)
                .is_some_and(|text| !text.is_empty()),
            "a tool has no description: {tool}"
        );
    }
    Ok(())
}

/// The tiering is the surface's reason for existing, so it is asserted rather
/// than described: a search result must not carry the body an agent would
/// otherwise have paid for without asking.
#[test]
fn search_returns_identifiers_and_titles_and_no_bodies() -> TestResult {
    let (_dir, db) = populated()?;
    let mut session = Session::start(&db)?;
    // A word only the body of `alpha` contains, so one hit is expected and a
    // leaked body would be visible in the response.
    let (text, failed) = session.call("search", &json!({ "query": "checker" }))?;
    assert!(!failed, "search failed: {text}");
    let payload: Value = serde_json::from_str(&text)?;
    assert_eq!(payload.get("count").and_then(Value::as_u64), Some(1));
    let results = payload
        .get("results")
        .and_then(Value::as_array)
        .ok_or("no results array")?;
    let first = results.first().ok_or("no first result")?;
    assert_eq!(first.get("id").and_then(Value::as_str), Some("alpha"));
    assert_eq!(
        first.get("title").and_then(Value::as_str),
        Some("Ownership in Rust")
    );
    assert!(
        !text.contains("borrow checker"),
        "search returned the body: {text}"
    );
    Ok(())
}

/// A reranked call is dominated by the cross-encoder, so the caller's clock
/// cannot check the dense scan against a service threshold (T025). The
/// payload accounts for each stage that ran, and only those: with no
/// embedding endpoint configured there is no `embed` and no `kb-dense`, and
/// reporting them as zero would say they ran instantly.
#[test]
fn a_search_reports_where_its_time_went() -> TestResult {
    let (_dir, db) = populated()?;
    let mut session = Session::start(&db)?;
    let (text, failed) = session.call("search", &json!({ "query": "checker" }))?;
    assert!(!failed, "search failed: {text}");
    let payload: Value = serde_json::from_str(&text)?;
    let stages = payload
        .get("stages_ms")
        .and_then(Value::as_object)
        .ok_or("no stages_ms object")?;
    assert!(
        stages.get("kb-lexical").and_then(Value::as_f64).is_some(),
        "the lexical stage is not timed: {stages:?}"
    );
    assert!(
        !stages.contains_key("embed") && !stages.contains_key("kb-dense"),
        "a stage that never ran is reported as timed: {stages:?}"
    );
    assert!(
        !stages.contains_key("rerank"),
        "an unconfigured reranker is reported as timed: {stages:?}"
    );
    Ok(())
}

#[test]
fn search_honours_a_limit() -> TestResult {
    let (_dir, db) = populated()?;
    let mut session = Session::start(&db)?;
    let (text, failed) = session.call("search", &json!({ "query": "the", "limit": 1 }))?;
    assert!(!failed, "search failed: {text}");
    let payload: Value = serde_json::from_str(&text)?;
    assert_eq!(payload.get("count").and_then(Value::as_u64), Some(1));
    Ok(())
}

#[test]
fn context_reports_both_directions_of_the_link_graph() -> TestResult {
    let (_dir, db) = populated()?;
    let mut session = Session::start(&db)?;
    let (text, failed) = session.call("context", &json!({ "id": "alpha" }))?;
    assert!(!failed, "context failed: {text}");
    let payload: Value = serde_json::from_str(&text)?;
    let incoming = payload
        .get("linked_from")
        .and_then(Value::as_array)
        .ok_or("no linked_from")?;
    assert_eq!(
        incoming
            .first()
            .and_then(|entry| entry.get("id"))
            .and_then(Value::as_str),
        Some("gamma"),
        "the inbound link is missing: {text}"
    );
    assert!(
        !text.contains("borrow checker"),
        "context returned a body: {text}"
    );
    Ok(())
}

#[test]
fn get_returns_the_full_document() -> TestResult {
    let (_dir, db) = populated()?;
    let mut session = Session::start(&db)?;
    let (text, failed) = session.call("get", &json!({ "id": "alpha" }))?;
    assert!(!failed, "get failed: {text}");
    assert!(text.contains("Ownership in Rust"), "{text}");
    assert!(text.contains("borrow checker"), "{text}");
    assert!(text.contains(":ID: alpha"), "the drawer is missing: {text}");
    Ok(())
}

/// The acceptance round trip: a record written through the surface is found
/// by the surface and read back through it.
#[test]
fn a_record_written_by_put_is_found_by_search_and_read_by_get() -> TestResult {
    let (_dir, db) = populated()?;
    let mut session = Session::start(&db)?;
    let (written, failed) = session.call(
        "put",
        &json!({
            "document": "* Photosynthesis\n\nchloroplasts and the calvin cycle\n",
            "tags": ["Botany Notes"],
        }),
    )?;
    assert!(!failed, "put failed: {written}");
    let id = serde_json::from_str::<Value>(&written)?
        .get("id")
        .and_then(Value::as_str)
        .ok_or("put returned no id")?
        .to_owned();

    let (found, failed) = session.call("search", &json!({ "query": "chloroplasts" }))?;
    assert!(!failed, "search failed: {found}");
    assert!(
        found.contains(&id),
        "the written record was not found: {found}"
    );

    let (document, failed) = session.call("get", &json!({ "id": &id }))?;
    assert!(!failed, "get failed: {document}");
    assert!(document.contains("calvin cycle"), "{document}");
    // Tags normalize the way they do everywhere else, rather than a second
    // time in a second way.
    assert!(document.contains("botany-notes"), "{document}");
    Ok(())
}

/// A tool that could not do its job answers with the reason, as content the
/// agent can read. A protocol error would be rendered opaquely by the client
/// and the agent would learn only that something went wrong.
#[test]
fn a_missing_record_is_a_readable_failure_rather_than_a_protocol_error() -> TestResult {
    let (_dir, db) = populated()?;
    let mut session = Session::start(&db)?;
    let (text, failed) = session.call("get", &json!({ "id": "no-such-node" }))?;
    assert!(failed, "a missing record was reported as success: {text}");
    assert!(
        text.contains("no node with id"),
        "the reason was not given: {text}"
    );
    Ok(())
}

#[test]
fn a_missing_argument_is_refused_with_the_argument_named() -> TestResult {
    let (_dir, db) = populated()?;
    let mut session = Session::start(&db)?;
    let (text, failed) = session.call("search", &json!({}))?;
    assert!(failed, "a query-less search succeeded: {text}");
    assert!(text.contains("query"), "the argument was not named: {text}");
    Ok(())
}

#[test]
fn an_unknown_tool_is_refused_by_name() -> TestResult {
    let (_dir, db) = populated()?;
    let mut session = Session::start(&db)?;
    let (text, failed) = session.call("delete_everything", &json!({}))?;
    assert!(failed, "an unknown tool succeeded: {text}");
    assert!(text.contains("delete_everything"), "{text}");
    Ok(())
}

/// The measurement the acceptance check asks for, asserted as a bound so it
/// cannot drift upward unnoticed.
#[test]
fn a_search_response_stays_small() -> TestResult {
    let (_dir, db) = populated()?;
    let mut session = Session::start(&db)?;
    let (text, failed) = session.call("search", &json!({ "query": "the" }))?;
    assert!(!failed, "search failed: {text}");
    let payload: Value = serde_json::from_str(&text)?;
    let count = payload
        .get("count")
        .and_then(Value::as_u64)
        .ok_or("no count")?;
    assert!(count >= 2, "the fixture did not exercise several hits");
    // Roughly 25 bytes of envelope plus an id and a title per hit. A body
    // would be hundreds to hundreds of thousands.
    let per_hit = text.len() / usize::try_from(count)?;
    assert!(per_hit < 120, "a search hit costs {per_hit} bytes: {text}");
    Ok(())
}

/// Run `kb --db <db> <args...>` in the same sandbox the server gets, and
/// return its stdout.
///
/// The two processes must see the same world for a parity assertion to mean
/// anything: the same database, the same home, and no embedding or reranking
/// endpoint on either side, so the comparison is between two rankings and not
/// between two environments.
fn run_kb(db: &Path, args: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
    let home = db
        .parent()
        .ok_or("db path has no parent for HOME sandbox")?;
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_kb"));
    cmd.env("HOME", home)
        .env("XDG_CONFIG_HOME", home)
        .env_remove("KB_EMBEDDING_BASE_URL")
        .env_remove("KB_EMBEDDING_MODEL")
        .env_remove("KB_RERANK_BASE_URL")
        .arg("--db")
        .arg(db);
    for arg in args {
        cmd.arg(arg);
    }
    let out = cmd.stdin(Stdio::null()).output()?;
    if !out.status.success() {
        return Err(format!(
            "kb {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    Ok(String::from_utf8(out.stdout)?)
}

/// The tool and the CLI answer the same question the same way (T030).
///
/// Not "both return something plausible": the same hits, in the same order,
/// with the same similarity and the same corpus on each. The surface exists
/// to carry kb's retrieval to an agent, and an agent that gets a different
/// ranking from the one the operator sees cannot be debugged against what the
/// operator sees.
///
/// One field is deliberately not compared: `kb search --json` carries a
/// `provenance` block per hit (`PLAN-20260923-project-identity` T006), and
/// the MCP `search` tool does not — it is the cheap tier, identifiers and
/// titles only, and `provenance` is paid for by calling `get`. That
/// divergence is the point of the tiering this module's doc describes, not
/// a disagreement about ranking.
#[test]
fn the_tool_returns_what_kb_search_returns() -> TestResult {
    let (_dir, db) = populated()?;
    let cli: Value = serde_json::from_str(&run_kb(
        &db,
        &["search", "retrieval", "--limit", "3", "--json"],
    )?)?;
    let expected: Vec<Value> = cli
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("no data array in {cli}"))?
        .iter()
        .map(|hit| {
            json!({
                "id": hit.get("id"),
                "title": hit.get("title"),
                "similarity": hit.get("similarity"),
                "corpus": hit.get("corpus"),
            })
        })
        .collect();
    let mut session = Session::start(&db)?;
    let (text, failed) = session.call("search", &json!({ "query": "retrieval", "limit": 3 }))?;
    assert!(!failed, "search failed: {text}");
    let payload: Value = serde_json::from_str(&text)?;
    let results = payload
        .get("results")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("no results array in {payload}"))?;
    assert!(!expected.is_empty(), "the fixture matched nothing");
    assert_eq!(results, &expected, "the tool and the CLI disagree");
    Ok(())
}

/// Deliver one message into a Maildir under `home`, index it into the derived
/// index the mail corpus is served from, and return a mu index over it.
///
/// The mail corpus lives in two places outside the kb database — mu's Xapian
/// index for the lexical half and the derived index for records — so a test
/// that the tool can reach it has to build both. Neither is stubbed: what is
/// being asserted is that a corpus argument reaches real backends.
fn with_mail(home: &Path) -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let maildir = home.join("mail");
    let cur = maildir.join("Inbox").join("cur");
    std::fs::create_dir_all(&cur)?;
    std::fs::write(
        cur.join("one:2,S"),
        "From: Ada <ada@example.invalid>\n\
         To: reader@example.invalid\n\
         Subject: quarterly revenue figures\n\
         Date: Mon, 3 Feb 2025 09:00:00 +0000\n\
         Message-ID: <one@example.invalid>\n\
         \n\
         the revenue figures for the quarter are attached\n",
    )?;
    let corpus = kb::maildir::MaildirCorpus::scan(&maildir)?;
    let selection = kb::maildir::Selection::compute(
        &corpus,
        &kb::mail::BulkRule::default(),
        &std::collections::BTreeSet::new(),
    );
    let index_path = home.join(".local/share/kb/index.db");
    std::fs::create_dir_all(
        index_path
            .parent()
            .ok_or("the index path has no parent directory")?,
    )?;
    let index = kb::index::Index::open_for_rebuild(&index_path)?;
    kb::index::rebuild_mail(&corpus, &selection, &index)?;
    let muhome = tempfile::tempdir()?;
    for args in [
        vec![
            "init".to_owned(),
            "--muhome".to_owned(),
            muhome.path().display().to_string(),
            "--maildir".to_owned(),
            maildir.display().to_string(),
        ],
        vec![
            "index".to_owned(),
            "--muhome".to_owned(),
            muhome.path().display().to_string(),
        ],
    ] {
        let out = Command::new("mu").args(&args).output()?;
        if !out.status.success() {
            return Err(format!(
                "mu {:?} failed: {}",
                args,
                String::from_utf8_lossy(&out.stderr)
            )
            .into());
        }
    }
    Ok(muhome)
}

/// Whether `mu` is installed.
fn mu_available() -> bool {
    Command::new("mu")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// An agent can ask the mail archive, and gets the notes unless it does
/// (T030).
///
/// One query, two corpora, two disjoint answers. The default matters as much
/// as the argument: an agent that has been searching notes must keep getting
/// notes from the call it has always made.
#[test]
fn a_search_can_name_the_mail_corpus_and_defaults_to_kb() -> TestResult {
    if !mu_available() {
        eprintln!("skipping: mu not on PATH");
        return Ok(());
    }
    let (dir, db) = populated()?;
    {
        let root = dir.path().join(".local/share/kb");
        let store = kb::store::GitBlobStore::open_or_init(&root.join("store"))?;
        let index = kb::index::Index::open(&root.join("index.db"))?;
        let document = kb::parser::parse_document(
            "* Revenue notes\n\nthe quarterly revenue figures, as notes\n",
        )?;
        let options = kb::write::WriteOptions::note("delta")?;
        kb::write::put_record(&store, &index, "delta", &document, &options)?;
    }
    let muhome = with_mail(dir.path())?;
    let mut session = Session::start_with(&db, &[("MUHOME", muhome.path())])?;

    let (mail, failed) =
        session.call("search", &json!({ "query": "revenue", "corpus": "mail" }))?;
    assert!(!failed, "a mail search failed: {mail}");
    let payload: Value = serde_json::from_str(&mail)?;
    let hits = payload
        .get("results")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("no results array in {payload}"))?;
    assert!(
        hits.iter()
            .any(|hit| hit.get("id").and_then(Value::as_str) == Some("<one@example.invalid>")),
        "the mail corpus was not searched: {mail}"
    );
    assert!(
        hits.iter()
            .all(|hit| hit.get("corpus").and_then(Value::as_str) == Some("mail")),
        "a mail hit did not say so: {mail}"
    );

    let (notes, failed) = session.call("search", &json!({ "query": "revenue" }))?;
    assert!(!failed, "a default search failed: {notes}");
    let payload: Value = serde_json::from_str(&notes)?;
    let ids: Vec<&str> = payload
        .get("results")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("no results array in {payload}"))?
        .iter()
        .filter_map(|hit| hit.get("id").and_then(Value::as_str))
        .collect();
    assert_eq!(ids, vec!["delta"], "the default corpus is not kb: {notes}");
    Ok(())
}

/// A `project` argument narrows `search` to records asserting that slug
/// (`PLAN-20260923-project-identity` T006). `search` carries no provenance
/// field itself — that stays on `get`, the tier an agent pays for by
/// asking — but the filter still has to reach the SQL underneath it.
#[test]
fn search_with_a_project_argument_returns_only_that_projects_records() -> TestResult {
    let (dir, db) = populated()?;
    {
        let root = dir.path().join(".local/share/kb");
        let store = kb::store::GitBlobStore::open_or_init(&root.join("store"))?;
        let index = kb::index::Index::open(&root.join("index.db"))?;
        for (id, project, body) in [
            (
                "widget-alpha",
                "alpha-project",
                "* Widget\n\nnotes about the widget launch\n",
            ),
            (
                "widget-beta",
                "beta-project",
                "* Widget\n\nother notes about the widget rollout\n",
            ),
        ] {
            let document = kb::parser::parse_document(body)?;
            let options = kb::write::WriteOptions {
                kind: kb::record::ArtifactKind::Note,
                source: kb::record::SourceRef::new("node-id", id)?,
                provenance: kb::record::Provenance {
                    project: Some(tftio_lib::project::Slug::new(project)?),
                    ..kb::record::Provenance::default()
                },
            };
            kb::write::put_record(&store, &index, id, &document, &options)?;
        }
    }
    let mut session = Session::start(&db)?;

    let (text, failed) = session.call(
        "search",
        &json!({ "query": "widget", "project": "alpha-project" }),
    )?;
    assert!(!failed, "search failed: {text}");
    let payload: Value = serde_json::from_str(&text)?;
    let hits = payload
        .get("results")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("no results array in {payload}"))?;
    let ids: Vec<&str> = hits
        .iter()
        .filter_map(|hit| hit.get("id").and_then(Value::as_str))
        .collect();
    assert_eq!(ids, vec!["widget-alpha"], "{payload}");
    Ok(())
}

/// `get`'s response carries the same provenance block `search` and `kb get
/// --json` do.
#[test]
fn get_carries_the_records_provenance() -> TestResult {
    let (dir, db) = populated()?;
    {
        let root = dir.path().join(".local/share/kb");
        let store = kb::store::GitBlobStore::open_or_init(&root.join("store"))?;
        let index = kb::index::Index::open(&root.join("index.db"))?;
        let document = kb::parser::parse_document("* Widget\n\nnotes.\n")?;
        let options = kb::write::WriteOptions {
            kind: kb::record::ArtifactKind::Note,
            source: kb::record::SourceRef::new("node-id", "widget-alpha")?,
            provenance: kb::record::Provenance {
                project: Some(tftio_lib::project::Slug::new("alpha-project")?),
                ..kb::record::Provenance::default()
            },
        };
        kb::write::put_record(&store, &index, "widget-alpha", &document, &options)?;
    }
    let mut session = Session::start(&db)?;
    let (text, failed) = session.call("get", &json!({ "id": "widget-alpha" }))?;
    assert!(!failed, "get failed: {text}");
    assert!(
        text.contains("project: alpha-project"),
        "get's text output is missing provenance: {text}"
    );
    Ok(())
}
