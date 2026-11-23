//! The write tools of the agent surface, called through [`ToolSurface::call`].
//!
//! They assert what a write leaves in a real store and index, what it reports
//! about embedding, and that an embedding problem never costs the write.

use kb::cli_embed::CliEmbedder;
use kb::embedding::EmbeddingConfig;
use kb::mcp::{KbTools, ToolOutcome, ToolSurface};
use serde_json::{Value, json};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const MODEL: &str = "fixture-model";

/// Port 1 is reserved and never listening.
const DEAD_ENDPOINT: &str = "http://127.0.0.1:1/v1";

struct Fixture {
    _dir: tempfile::TempDir,
    store: std::path::PathBuf,
    index: std::path::PathBuf,
}

impl Fixture {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        Ok(Self {
            store: dir.path().join("store"),
            index: dir.path().join("index.db"),
            _dir: dir,
        })
    }

    fn tools(&self, embedder: Option<CliEmbedder>) -> KbTools {
        KbTools::with_paths(&self.store, &self.index).with_embedder(embedder)
    }

    fn index(&self) -> Result<kb::index::Index, Box<dyn std::error::Error>> {
        Ok(kb::index::Index::open(&self.index)?)
    }

    /// Store a record directly, as a session capture would, with a kind,
    /// source and provenance a tool must not disturb.
    fn seed_capture(&self, id: &str, body: &str) -> TestResult {
        let store = kb::store::GitBlobStore::open_or_init(&self.store)?;
        let index = kb::index::Index::open(&self.index)?;
        let options = kb::write::WriteOptions {
            kind: kb::record::ArtifactKind::SessionTranscript,
            source: kb::record::SourceRef::new("session", "abc-123")?,
            provenance: kb::record::Provenance {
                harness: Some("claude-code".to_owned()),
                ..kb::record::Provenance::default()
            },
        };
        let document = kb::parser::parse_document(body)?;
        kb::write::put_record(&store, &index, id, &document, &options)?;
        Ok(())
    }
}

fn embedder(url: &str) -> Result<CliEmbedder, Box<dyn std::error::Error>> {
    Ok(CliEmbedder::from_optional_config(Some(EmbeddingConfig {
        base_url: url.to_owned(),
        model: MODEL.to_owned(),
        api_key: None,
        document_prefix: String::new(),
        query_prefix: String::new(),
    }))?)
}

/// A mock embeddings endpoint answering every request with one vector.
fn endpoint() -> (mockito::ServerGuard, String) {
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"data":[{"embedding":[0.5,0.5]}]}"#)
        .create();
    let url = format!("{}/v1", server.url());
    (server, url)
}

fn text(outcome: ToolOutcome) -> Result<Value, Box<dyn std::error::Error>> {
    match outcome {
        ToolOutcome::Text(body) => Ok(serde_json::from_str(&body)?),
        ToolOutcome::Failed(reason) => Err(format!("tool failed: {reason}").into()),
    }
}

fn failure(outcome: ToolOutcome) -> Result<String, Box<dyn std::error::Error>> {
    match outcome {
        ToolOutcome::Failed(reason) => Ok(reason),
        ToolOutcome::Text(body) => Err(format!("expected a failure, got: {body}").into()),
    }
}

fn status<'a>(payload: &'a Value, path: &[&str]) -> Option<&'a str> {
    path.iter()
        .try_fold(payload, |value, key| value.get(*key))
        .and_then(Value::as_str)
}

fn spans(fixture: &Fixture, id: &str) -> Result<Vec<(i64, i64)>, Box<dyn std::error::Error>> {
    Ok(fixture.index()?.embedded_spans(id, MODEL)?)
}

fn tags_of(fixture: &Fixture, id: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let index = fixture.index()?;
    let mut carried = Vec::new();
    for (tag, _) in index.tag_counts()? {
        if index.records_tagged(&tag)?.iter().any(|(row, _)| row == id) {
            carried.push(tag);
        }
    }
    carried.sort();
    Ok(carried)
}

#[test]
fn a_tag_edit_gives_the_record_a_new_stream_hash() -> TestResult {
    // The reason the tag tools embed at all (plan D003): vectors are keyed by
    // stream hash, so if a tag edit left the hash alone the old vectors would
    // still serve and embedding again would be wasted work.
    let fixture = Fixture::new()?;
    let store = kb::store::GitBlobStore::open_or_init(&fixture.store)?;
    let index = kb::index::Index::open(&fixture.index)?;
    let document = kb::parser::parse_document("* Subject\n\nbody text\n")?;
    let options = kb::write::WriteOptions::note("rec")?;
    kb::write::put_record(&store, &index, "rec", &document, &options)?;
    let before = index.stream_hash_of("rec")?;
    kb::write::add_tags(&store, &index, "rec", &["fresh".to_owned()])?;
    let after = index.stream_hash_of("rec")?;
    assert!(before.is_some() && after.is_some());
    assert_ne!(before, after, "a tag edit left the stream hash unchanged");
    Ok(())
}

#[test]
fn put_without_an_embedder_succeeds_and_reports_nothing_embedded() -> TestResult {
    let fixture = Fixture::new()?;
    let tools = fixture.tools(None);
    let put = text(tools.call("put", &json!({ "id": "n1", "document": "* One\n\ntext\n" })))?;
    assert_eq!(status(&put, &["embedding", "status"]), Some("disabled"));
    assert!(fixture.index()?.record("n1").is_ok());
    assert!(spans(&fixture, "n1")?.is_empty());
    Ok(())
}

#[test]
fn put_embeds_the_record_it_wrote() -> TestResult {
    let fixture = Fixture::new()?;
    let (_server, url) = endpoint();
    let tools = fixture.tools(Some(embedder(&url)?));
    let put = text(tools.call("put", &json!({ "id": "n1", "document": "* One\n\ntext\n" })))?;
    assert_eq!(status(&put, &["embedding", "status"]), Some("enabled"));
    let first = put
        .pointer("/embedding/records/0")
        .ok_or("no per-record report")?;
    assert_eq!(
        first.get("status").and_then(Value::as_str),
        Some("embedded")
    );
    assert!(first.get("embedded").and_then(Value::as_u64) >= Some(1));
    assert!(!spans(&fixture, "n1")?.is_empty());
    Ok(())
}

#[test]
fn an_unreachable_endpoint_leaves_the_put_stored_and_says_why() -> TestResult {
    let fixture = Fixture::new()?;
    let tools = fixture.tools(Some(embedder(DEAD_ENDPOINT)?));
    let put = text(tools.call("put", &json!({ "id": "n1", "document": "* One\n\ntext\n" })))?;
    let reason = put
        .pointer("/embedding/records/0/reason")
        .and_then(Value::as_str)
        .ok_or("no failure reason")?;
    assert!(reason.contains("could not embed"), "reason was: {reason}");
    assert_eq!(
        put.pointer("/embedding/records/0/status")
            .and_then(Value::as_str),
        Some("failed")
    );
    assert!(
        fixture.index()?.record("n1").is_ok(),
        "the write was undone"
    );
    assert!(spans(&fixture, "n1")?.is_empty());
    Ok(())
}

#[test]
fn update_replaces_the_body_and_keeps_kind_source_and_provenance() -> TestResult {
    let fixture = Fixture::new()?;
    fixture.seed_capture("cap", "* Turn one\n\nfirst\n")?;
    let (_server, url) = endpoint();
    let tools = fixture.tools(Some(embedder(&url)?));
    let out = text(tools.call(
        "update",
        &json!({ "id": "cap", "document": "* Turn one\n\nrewritten\n", "tags": ["Fresh Tag"] }),
    ))?;
    assert_eq!(
        out.pointer("/embedding/records/0/status")
            .and_then(Value::as_str),
        Some("embedded")
    );
    let row = fixture.index()?.record("cap")?;
    assert_eq!(row.kind, "session-transcript");
    assert_eq!(row.source, "session:abc-123");
    assert_eq!(row.harness.as_deref(), Some("claude-code"));
    assert!(tags_of(&fixture, "cap")?.contains(&"fresh-tag".to_owned()));
    let shown = text_of(tools.call("get", &json!({ "id": "cap" })))?;
    assert!(shown.contains("rewritten"));
    assert!(!spans(&fixture, "cap")?.is_empty());
    Ok(())
}

fn text_of(outcome: ToolOutcome) -> Result<String, Box<dyn std::error::Error>> {
    match outcome {
        ToolOutcome::Text(body) => Ok(body),
        ToolOutcome::Failed(reason) => Err(format!("tool failed: {reason}").into()),
    }
}

#[test]
fn update_refuses_an_unknown_id_and_a_mismatched_drawer() -> TestResult {
    let fixture = Fixture::new()?;
    let tools = fixture.tools(None);
    let _ = text(tools.call("put", &json!({ "id": "real", "document": "* R\n\nx\n" })))?;
    let reason = failure(tools.call("update", &json!({ "id": "typo", "document": "* R\n\ny\n" })))?;
    assert!(reason.contains("no node with id: typo"), "{reason}");
    assert!(
        fixture.index()?.record("typo").is_err(),
        "update minted a record"
    );
    let drawer = ":PROPERTIES:\n:ID: other\n:END:\n* R\n\ny\n";
    let reason = failure(tools.call("update", &json!({ "id": "real", "document": drawer })))?;
    assert!(
        reason.contains("does not match target node id real"),
        "{reason}"
    );
    Ok(())
}

#[test]
fn update_still_succeeds_when_embedding_fails() -> TestResult {
    let fixture = Fixture::new()?;
    let tools = fixture.tools(Some(embedder(DEAD_ENDPOINT)?));
    let _ = text(tools.call("put", &json!({ "id": "n", "document": "* N\n\nold\n" })))?;
    let out = text(tools.call("update", &json!({ "id": "n", "document": "* N\n\nnew\n" })))?;
    assert_eq!(
        out.pointer("/embedding/records/0/status")
            .and_then(Value::as_str),
        Some("failed")
    );
    let shown = text_of(tools.call("get", &json!({ "id": "n" })))?;
    assert!(shown.contains("new"));
    Ok(())
}

#[test]
fn delete_removes_the_record_but_leaves_its_bytes_in_the_store() -> TestResult {
    let fixture = Fixture::new()?;
    let tools = fixture.tools(None);
    let _ = text(tools.call("put", &json!({ "id": "gone", "document": "* G\n\nbye\n" })))?;
    let hash = fixture
        .index()?
        .record("gone")?
        .record_hash
        .ok_or("no record hash")?;
    let out = text(tools.call("delete", &json!({ "id": "gone" })))?;
    assert_eq!(out.get("deleted").and_then(Value::as_bool), Some(true));
    assert!(fixture.index()?.record("gone").is_err());
    let store = kb::store::GitBlobStore::open_or_init(&fixture.store)?;
    let blob = kb::store::BlobStore::get(
        &store,
        &kb::store::BlobHash::from(&kb::record::ContentHash::new(&hash)?),
    )?;
    assert!(!blob.is_empty(), "the record blob was erased");
    let reason = failure(tools.call("delete", &json!({ "id": "gone" })))?;
    assert!(reason.contains("no node with id: gone"), "{reason}");
    Ok(())
}

#[test]
fn tags_add_keeps_provenance_and_leaves_vectors_under_the_new_stream_hash() -> TestResult {
    let fixture = Fixture::new()?;
    fixture.seed_capture("cap", "* Turn\n\nsome words\n")?;
    let (_server, url) = endpoint();
    let tools = fixture.tools(Some(embedder(&url)?));
    let index = fixture.index()?;
    let old_hash = index.stream_hash_of("cap")?.ok_or("no hash")?;
    drop(index);
    let out = text(tools.call(
        "tags_add",
        &json!({ "id": "cap", "tags": ["Silent Critic"] }),
    ))?;
    assert_eq!(out.get("added"), Some(&json!(["silent-critic"])));
    assert_eq!(
        out.pointer("/embedding/records/0/status")
            .and_then(Value::as_str),
        Some("embedded")
    );
    let index = fixture.index()?;
    let new_hash = index.stream_hash_of("cap")?.ok_or("no hash")?;
    assert_ne!(old_hash, new_hash);
    let row = index.record("cap")?;
    assert_eq!(row.kind, "session-transcript");
    assert_eq!(row.source, "session:abc-123");
    assert_eq!(row.harness.as_deref(), Some("claude-code"));
    let passages = index.passages("cap")?;
    assert!(!passages.is_empty());
    for passage in passages {
        assert_eq!(passage.stream_hash, new_hash);
        assert!(
            index
                .embedding(
                    &passage.stream_hash,
                    passage.span_start,
                    passage.span_len,
                    MODEL
                )?
                .is_some(),
            "no vector under the new stream hash"
        );
    }
    Ok(())
}

#[test]
fn a_tag_edit_that_changes_nothing_writes_and_embeds_nothing() -> TestResult {
    let fixture = Fixture::new()?;
    let (_server, url) = endpoint();
    let tools = fixture.tools(Some(embedder(&url)?));
    let _ = text(tools.call(
        "put",
        &json!({ "id": "n", "document": "* N\n\nx\n", "tags": ["kept"] }),
    ))?;
    let before = fixture.index()?.record("n")?.record_hash;
    let out = text(tools.call("tags_add", &json!({ "id": "n", "tags": ["kept"] })))?;
    assert_eq!(status(&out, &["embedding", "status"]), Some("unchanged"));
    assert_eq!(fixture.index()?.record("n")?.record_hash, before);
    let out = text(tools.call("tags_rm", &json!({ "id": "n", "tags": ["absent"] })))?;
    assert_eq!(status(&out, &["embedding", "status"]), Some("unchanged"));
    Ok(())
}

#[test]
fn tags_rm_removes_the_tag_and_embeds_again() -> TestResult {
    let fixture = Fixture::new()?;
    let (_server, url) = endpoint();
    let tools = fixture.tools(Some(embedder(&url)?));
    let _ = text(tools.call(
        "put",
        &json!({ "id": "n", "document": "* N\n\nx\n", "tags": ["a-tag", "b-tag"] }),
    ))?;
    let out = text(tools.call("tags_rm", &json!({ "id": "n", "tags": ["a-tag"] })))?;
    assert_eq!(out.get("removed"), Some(&json!(["a-tag"])));
    assert_eq!(tags_of(&fixture, "n")?, ["b-tag"]);
    assert_eq!(
        out.pointer("/embedding/records/0/status")
            .and_then(Value::as_str),
        Some("embedded")
    );
    let index = fixture.index()?;
    let hash = index.stream_hash_of("n")?.ok_or("no hash")?;
    assert!(index.passages("n")?.iter().all(|p| p.stream_hash == hash));
    assert!(!spans(&fixture, "n")?.is_empty());
    Ok(())
}

#[test]
fn tags_merge_rewrites_and_embeds_every_record_and_reports_each() -> TestResult {
    let fixture = Fixture::new()?;
    let (_server, url) = endpoint();
    let tools = fixture.tools(Some(embedder(&url)?));
    for id in ["m1", "m2"] {
        let _ = text(tools.call(
            "put",
            &json!({ "id": id, "document": "* M\n\nx\n", "tags": ["cicd"] }),
        ))?;
    }
    let _ = text(tools.call(
        "put",
        &json!({ "id": "other", "document": "* O\n\ny\n", "tags": ["unrelated"] }),
    ))?;
    let out = text(tools.call("tags_merge", &json!({ "from": "cicd", "to": "ci-cd" })))?;
    assert_eq!(out.get("count").and_then(Value::as_u64), Some(2));
    let records = out
        .pointer("/embedding/records")
        .and_then(Value::as_array)
        .ok_or("no per-record report")?;
    assert_eq!(records.len(), 2);
    for record in records {
        assert_eq!(
            record.get("status").and_then(Value::as_str),
            Some("embedded")
        );
    }
    for id in ["m1", "m2"] {
        assert_eq!(tags_of(&fixture, id)?, ["ci-cd"]);
        let index = fixture.index()?;
        let hash = index.stream_hash_of(id)?.ok_or("no hash")?;
        assert!(index.passages(id)?.iter().all(|p| {
            p.stream_hash == hash
                && index
                    .embedding(&p.stream_hash, p.span_start, p.span_len, MODEL)
                    .is_ok_and(|v| v.is_some())
        }));
    }
    let out = text(tools.call(
        "tags_merge",
        &json!({ "from": "nothing-has-this", "to": "ci-cd" }),
    ))?;
    assert_eq!(status(&out, &["embedding", "status"]), Some("unchanged"));
    let reason = failure(tools.call("tags_merge", &json!({ "from": "ci-cd", "to": "CI CD" })))?;
    assert!(reason.contains("same tag"), "{reason}");
    Ok(())
}

#[test]
fn a_failing_endpoint_never_undoes_a_tag_edit() -> TestResult {
    let fixture = Fixture::new()?;
    let tools = fixture.tools(Some(embedder(DEAD_ENDPOINT)?));
    let _ = text(tools.call(
        "put",
        &json!({ "id": "n", "document": "* N\n\nx\n", "tags": ["old"] }),
    ))?;
    let out = text(tools.call("tags_merge", &json!({ "from": "old", "to": "new" })))?;
    assert_eq!(
        out.pointer("/embedding/records/0/status")
            .and_then(Value::as_str),
        Some("failed")
    );
    assert_eq!(tags_of(&fixture, "n")?, ["new"]);
    let out = text(tools.call("tags_add", &json!({ "id": "n", "tags": ["more"] })))?;
    assert_eq!(
        out.pointer("/embedding/records/0/status")
            .and_then(Value::as_str),
        Some("failed")
    );
    assert_eq!(tags_of(&fixture, "n")?, ["more", "new"]);
    Ok(())
}

#[test]
fn tag_tools_reject_missing_arguments() -> TestResult {
    let fixture = Fixture::new()?;
    let tools = fixture.tools(None);
    let _ = text(tools.call("put", &json!({ "id": "n", "document": "* N\n\nx\n" })))?;
    assert!(failure(tools.call("tags_add", &json!({ "id": "n" })))?.contains("tags is required"));
    assert!(failure(tools.call("tags_rm", &json!({ "tags": ["a"] })))?.contains("id is required"));
    assert!(failure(tools.call("tags_merge", &json!({ "from": "a" })))?.contains("to is required"));
    assert!(
        failure(tools.call("tags_add", &json!({ "id": "ghost", "tags": ["a"] })))?
            .contains("ghost")
    );
    Ok(())
}

#[test]
fn a_surface_holding_an_embedder_can_be_dropped_inside_an_async_runtime() -> TestResult {
    // The server drops its tools when it stops, inside its own runtime; the
    // embedder owns a runtime too, and dropping one inside another panics
    // unless it is shut down in the background.
    let fixture = Fixture::new()?;
    let tools = fixture.tools(Some(embedder(DEAD_ENDPOINT)?));
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    runtime.block_on(async move { drop(tools) });
    Ok(())
}
