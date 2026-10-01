//! `kb embed` at the process boundary (CLI-002).
//!
//! Embedding was reachable only as `kb mail embed`, which hard-coded the one
//! corpus that had ever been served from the derived index. T029 moves the kb
//! corpus onto that index too, so the command that fills it has to name a
//! corpus rather than assume one.
//!
//! The endpoint is a mock rather than a live daemon: what is under test is
//! that the command selects the right passages, chunks them, stores what comes
//! back and resumes without repeating finished work — none of which is a
//! property of any particular model.

#![allow(
    clippy::significant_drop_tightening,
    reason = "the mock endpoint must outlive every subprocess that calls it"
)]

use kb::record::{ArtifactKind, ContentHash, CorpusId, RecordHeader, RecordId, SourceRef};
use kb::store::{BlobStore, GitBlobStore, RefName};
use std::path::Path;
use std::process::Command;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn run_kb(
    home: &Path,
    args: &[&str],
    env: &[(&str, &str)],
) -> Result<(bool, String, String), Box<dyn std::error::Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_kb"));
    command
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home)
        .env_remove("KB_EMBEDDING_BASE_URL")
        .env_remove("KB_EMBEDDING_MODEL")
        .env_remove("KB_MAIL_ROOT");
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command.args(args).output()?;
    Ok((
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

/// Put one note in the store under `corpus`.
fn store_note(
    store: &GitBlobStore,
    corpus: &str,
    id: &str,
    body: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let raw = format!("* Note {id}\n\n{body}\n");
    let stream = kb::record::normalize(ArtifactKind::Note, raw.as_bytes())?;
    let header = RecordHeader {
        id: RecordId::new(id)?,
        corpus: CorpusId::new(corpus)?,
        kind: ArtifactKind::Note,
        created: "2026-05-19T09:00:00Z".parse()?,
        updated: "2026-08-14T10:30:00Z".parse()?,
        source: SourceRef::new("node-id", id)?,
        tags: vec![],
        raw: ContentHash::new(store.put(raw.as_bytes())?.as_str())?,
        stream: ContentHash::new(store.put(stream.as_bytes())?.as_str())?,
        normalizer: kb::record::NORMALIZER_VERSION,
        provenance: kb::record::Provenance::default(),
    };
    let hash = store.put(&header.serialize())?;
    store.set_ref(&RefName::new(&format!("{corpus}/{id}"))?, &hash)?;
    Ok(())
}

/// A store of one kb note, indexed, with the paths to address it by.
fn indexed(dir: &Path) -> Result<(String, String), Box<dyn std::error::Error>> {
    let store_path = dir.join("store");
    let store = GitBlobStore::open_or_init(&store_path)?;
    store_note(&store, "kb", "n1", "the borrow checker and its rules")?;
    let index_path = dir.join("index.db");
    let store_arg = store_path.display().to_string();
    let index_arg = index_path.display().to_string();
    let (ok, _, stderr) = run_kb(
        dir,
        &["reindex", "--store", &store_arg, "--index", &index_arg],
        &[],
    )?;
    assert!(ok, "reindex failed: {stderr}");
    Ok((store_arg, index_arg))
}

/// The command embeds the corpus it is told to, and says what it did.
#[test]
fn embedding_a_corpus_stores_vectors_and_resumes_without_repeating_work() -> TestResult {
    let dir = tempfile::tempdir()?;
    let (_store, index) = indexed(dir.path())?;
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"data":[{"embedding":[0.5,0.5]}]}"#)
        .expect_at_least(1)
        .create();
    let url = format!("{}/v1", server.url());
    let live = [
        ("KB_EMBEDDING_BASE_URL", url.as_str()),
        ("KB_EMBEDDING_MODEL", "fixture-model"),
    ];

    let (ok, first, stderr) = run_kb(
        dir.path(),
        &["embed", "--corpus", "kb", "--index", &index],
        &live,
    )?;
    assert!(ok, "embed failed: {stderr}");
    assert!(first.contains("1 span embedded"), "stdout: {first}");

    let (ok, second, stderr) = run_kb(
        dir.path(),
        &["embed", "--corpus", "kb", "--index", &index],
        &live,
    )?;
    assert!(ok, "the second pass failed: {stderr}");
    assert!(
        second.contains("0 spans embedded, 1 already current"),
        "the second pass repeated finished work: {second}"
    );
    Ok(())
}

/// A corpus with nothing in it is not an error. `kb embed --corpus mail` on a
/// machine that has catalogued no mail has nothing to do, which is a report
/// rather than a failure.
#[test]
fn embedding_a_corpus_with_no_passages_reports_nothing_to_do() -> TestResult {
    let dir = tempfile::tempdir()?;
    let (_store, index) = indexed(dir.path())?;
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"data":[{"embedding":[0.5,0.5]}]}"#)
        .create();
    let url = format!("{}/v1", server.url());
    let live = [
        ("KB_EMBEDDING_BASE_URL", url.as_str()),
        ("KB_EMBEDDING_MODEL", "fixture-model"),
    ];

    let (ok, stdout, stderr) = run_kb(
        dir.path(),
        &["embed", "--corpus", "mail", "--index", &index],
        &live,
    )?;
    assert!(ok, "embed failed: {stderr}");
    assert!(stdout.contains("0 spans embedded"), "stdout: {stdout}");
    Ok(())
}

/// Without an endpoint the command says which variable is missing rather than
/// ranking nothing and exiting 0.
#[test]
fn embedding_without_configuration_says_what_is_missing() -> TestResult {
    let dir = tempfile::tempdir()?;
    let (_store, index) = indexed(dir.path())?;
    let (ok, _, stderr) = run_kb(
        dir.path(),
        &["embed", "--corpus", "kb", "--index", &index],
        &[],
    )?;
    assert!(!ok, "embedding without an endpoint reported success");
    assert!(
        stderr.contains("KB_EMBEDDING_BASE_URL"),
        "stderr does not name the variable: {stderr}"
    );
    Ok(())
}
