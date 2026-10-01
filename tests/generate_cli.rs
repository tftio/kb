//! `kb generate` at the process boundary (CLI-002).
#![allow(
    clippy::significant_drop_tightening,
    reason = "a mockito Server guard is held for the test's duration on purpose; dropping it early tears down the endpoint the child process is talking to"
)]

use std::path::Path;
use std::process::Command;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn store_note(store: &kb::store::GitBlobStore, id: &str) -> TestResult {
    use kb::record::{ArtifactKind, ContentHash, CorpusId, RecordHeader, RecordId, SourceRef};
    use kb::store::{BlobStore, RefName};
    let raw = format!("* Note {id}\n\nthe body of {id}.\n");
    let stream = kb::record::normalize(ArtifactKind::Note, raw.as_bytes())?;
    let header = RecordHeader {
        id: RecordId::new(id)?,
        corpus: CorpusId::new("kb")?,
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
    store.set_ref(&RefName::new(&format!("kb/{id}"))?, &hash)?;
    Ok(())
}

/// A home with a store, an index over it, and nothing generated yet.
fn prepared(home: &Path, ids: &[&str]) -> TestResult {
    let store_path = home.join(".local/share/kb/store");
    std::fs::create_dir_all(&store_path)?;
    let store = kb::store::GitBlobStore::open_or_init(&store_path)?;
    for id in ids {
        store_note(&store, id)?;
    }
    let index = kb::index::Index::open_for_rebuild(&home.join(".local/share/kb/index.db"))?;
    kb::index::rebuild(&store, &index, &kb::index::Scope::All)?;
    Ok(())
}

fn run(home: &Path, args: &[&str]) -> Result<(bool, String, String), Box<dyn std::error::Error>> {
    let out = Command::new(env!("CARGO_BIN_EXE_kb"))
        .args(args)
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home)
        .env("XDG_DATA_HOME", home.join(".local/share"))
        .env_remove("KB_GENERATION_BASE_URL")
        .env_remove("KB_GENERATION_MODEL")
        .output()?;
    Ok((
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    ))
}

fn endpoint(reply: &str) -> (mockito::ServerGuard, String) {
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(format!(
            r#"{{"choices":[{{"message":{{"content":"{reply}"}}}}]}}"#
        ))
        .expect_at_least(1)
        .create();
    let url = format!("{}/v1", server.url());
    (server, url)
}

/// The guarantee the task exists for, asserted where an operator would see
/// it: a second pass over the same scope contacts nothing.
#[test]
fn a_second_pass_over_the_same_scope_generates_nothing() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    prepared(home, &["n1", "n2"])?;
    let (_server, url) = endpoint("a generated paragraph");

    let (ok, first, stderr) = run(
        home,
        &["generate", "--model", "fixture", "--base-url", &url],
    )?;
    assert!(ok, "generate failed: {stderr}");
    assert!(first.contains("2 records generated"), "stdout: {first}");

    let (ok, second, stderr) = run(
        home,
        &["generate", "--model", "fixture", "--base-url", &url],
    )?;
    assert!(ok, "second pass failed: {stderr}");
    assert!(
        second.contains("0 records generated, 2 already current"),
        "the second pass paid for generation again: {second}"
    );
    Ok(())
}

/// The measurement loop: regenerate a named subset under a new prompt version
/// and leave everything else alone.
#[test]
fn a_named_subset_under_a_new_prompt_leaves_the_rest_alone() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    prepared(home, &["n1", "n2"])?;
    let (_server, url) = endpoint("first wording");
    run(
        home,
        &["generate", "--model", "fixture", "--base-url", &url],
    )?;

    let (ok, stdout, stderr) = run(
        home,
        &[
            "generate",
            "--model",
            "fixture",
            "--base-url",
            &url,
            "--record",
            "n1",
            "--prompt-version",
            "3",
        ],
    )?;

    assert!(ok, "subset generation failed: {stderr}");
    assert!(stdout.contains("1 record generated"), "stdout: {stdout}");
    let index = kb::index::Index::open(&home.join(".local/share/kb/index.db"))?;
    assert_eq!(
        index.generated_for("n1")?.len(),
        2,
        "both prompts must coexist"
    );
    assert_eq!(index.generated_for("n2")?.len(), 1, "n2 was regenerated");
    Ok(())
}

/// A named record the index does not hold is reported and made a nonzero
/// exit, because a subset run that silently generated nothing reads as
/// success.
#[test]
fn an_unknown_record_is_named_and_exits_nonzero() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    prepared(home, &["n1"])?;
    let (_server, url) = endpoint("a paragraph");

    let (ok, _, stderr) = run(
        home,
        &[
            "generate",
            "--model",
            "fixture",
            "--base-url",
            &url,
            "--record",
            "absent",
        ],
    )?;

    assert!(!ok);
    assert!(
        stderr.contains("no such record: absent"),
        "stderr: {stderr}"
    );
    Ok(())
}

/// An endpoint that is not there must say so rather than report a successful
/// run of nothing.
#[test]
fn an_unreachable_endpoint_is_reported() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    prepared(home, &["n1"])?;

    let (ok, _, stderr) = run(
        home,
        &[
            "generate",
            "--model",
            "fixture",
            "--base-url",
            "http://127.0.0.1:1/v1",
        ],
    )?;

    assert!(!ok);
    assert!(
        stderr.contains("n1"),
        "the failing record must be named: {stderr}"
    );
    Ok(())
}

/// A model that spends its whole budget thinking returns a successful
/// response with no content. Storing that would key a blank against the
/// source and never regenerate it.
#[test]
fn an_empty_completion_is_a_failure_not_an_artifact() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    prepared(home, &["n1"])?;
    let (_server, url) = endpoint("<think>thinking about it</think>");

    let (ok, _, stderr) = run(
        home,
        &["generate", "--model", "fixture", "--base-url", &url],
    )?;

    assert!(!ok, "an empty completion was accepted: {stderr}");
    let index = kb::index::Index::open(&home.join(".local/share/kb/index.db"))?;
    assert!(index.generated_for("n1")?.is_empty());
    Ok(())
}

/// A scope matching nothing is an error, not a successful run of zero, which
/// would read as "already generated".
#[test]
fn an_empty_scope_is_an_error() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    prepared(home, &["n1"])?;

    let (ok, _, stderr) = run(
        home,
        &["generate", "--model", "fixture", "--corpus", "nonexistent"],
    )?;

    assert!(!ok);
    assert!(stderr.contains("no records matched"), "stderr: {stderr}");
    Ok(())
}
