//! `kb mail scope`, `kb mail index` and `kb mail embed` at the process
//! boundary (CLI-002).
//!
//! The Maildir root comes from `--maildir` or `KB_MAIL_ROOT` and has no
//! default, so every test names a root inside its own temporary directory and
//! clears `KB_MAIL_ROOT` from the inherited environment: no test can reach a
//! real mailbox. `HOME` is redirected as in every other CLI test, so the
//! index path moves with it.
#![allow(
    clippy::significant_drop_tightening,
    reason = "a mockito Server guard is held for the test's duration on purpose; dropping it early tears down the endpoint the child process is talking to"
)]

use std::path::Path;
use std::process::Command;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const QUESTIONS: &str = r#"
[corpora]
known = ["kb", "mail"]
default = "kb"

[[question]]
id = "MQ01"
query = "what did Ada propose"
population = "mail"
[[question.expect]]
corpus = "mail"
node_id = "<needed@example.invalid>"
title = "the proposal"
folder = "Archive"
content_sha256 = "PLACEHOLDER"
authored_year = "2025"
"#;

/// The Maildir root each test configures, inside its temporary home.
fn maildir(home: &Path) -> std::path::PathBuf {
    home.join("Maildir")
}

fn deliver(
    root: &Path,
    folder: &str,
    uniq: &str,
    message_id: &str,
    extra: &str,
    body: &str,
) -> TestResult {
    let dir = root.join(folder).join("cur");
    std::fs::create_dir_all(&dir)?;
    std::fs::write(
        dir.join(format!("{uniq}:2,S")),
        format!(
            "From: Ada <ada@example.invalid>\n\
             To: reader@example.invalid\n\
             Subject: about {uniq}\n\
             Date: Mon, 3 Feb 2025 09:00:00 +0000\n\
             Message-ID: {message_id}\n\
             {extra}\
             \n\
             {body}\n"
        ),
    )?;
    Ok(())
}

/// Write a question file whose recorded hash matches what was delivered, so
/// drift reporting can be exercised in both directions.
fn questions(
    dir: &Path,
    home: &Path,
    matching: bool,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let delivered = std::fs::read(maildir(home).join("Archive/cur/u1:2,S"))?;
    let hash = if matching {
        kb::maildir::content_sha256(&delivered)
    } else {
        "0".repeat(64)
    };
    let path = dir.join("questions.toml");
    std::fs::write(&path, QUESTIONS.replace("PLACEHOLDER", &hash))?;
    Ok(path)
}

fn run(home: &Path, args: &[&str]) -> Result<(bool, String, String), Box<dyn std::error::Error>> {
    run_with(home, args, &[])
}

fn run_with(
    home: &Path,
    args: &[&str],
    env: &[(&str, &str)],
) -> Result<(bool, String, String), Box<dyn std::error::Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_kb"));
    command
        .args(args)
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home)
        .env("XDG_DATA_HOME", home.join(".local/share"))
        .env_remove("KB_EMBEDDING_BASE_URL")
        .env_remove("KB_EMBEDDING_MODEL")
        .env_remove("KB_MAIL_ROOT");
    for (name, value) in env {
        command.env(name, value);
    }
    let out = command.output()?;
    Ok((
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    ))
}

#[test]
fn scope_reports_what_the_increment_would_index() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    deliver(
        &maildir(home),
        "Archive",
        "u1",
        "<needed@example.invalid>",
        "",
        "the proposal",
    )?;
    deliver(
        &maildir(home),
        "Inbox",
        "u2",
        "<news@example.invalid>",
        "List-Id: <announce.example.invalid>\n",
        "a newsletter",
    )?;
    let question_file = questions(dir.path(), home, true)?;

    let (ok, stdout, stderr) = run(
        home,
        &[
            "mail",
            "scope",
            "--maildir",
            &maildir(home).display().to_string(),
            "--questions",
            &question_file.display().to_string(),
        ],
    )?;

    assert!(ok, "scope failed: {stderr}");
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    let field = |name: &str| report.get(name).cloned().unwrap_or_default();
    assert_eq!(
        field("maildir"),
        maildir(home).display().to_string().as_str()
    );
    assert_eq!(field("catalogued"), 2);
    assert_eq!(field("selected"), 1);
    assert_eq!(field("excluded_as_bulk"), 1);
    assert_eq!(field("drifted").as_array().map(Vec::len), Some(0));
    Ok(())
}

/// A figure measured against ground truth whose bytes have moved is not
/// comparable to the figure it is being compared with, so drift is a nonzero
/// exit rather than a line nobody reads.
#[test]
fn scope_reports_drifted_ground_truth_and_exits_nonzero() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    deliver(
        &maildir(home),
        "Archive",
        "u1",
        "<needed@example.invalid>",
        "",
        "the proposal",
    )?;
    let question_file = questions(dir.path(), home, false)?;

    let (ok, stdout, _) = run(
        home,
        &[
            "mail",
            "scope",
            "--maildir",
            &maildir(home).display().to_string(),
            "--questions",
            &question_file.display().to_string(),
        ],
    )?;

    assert!(!ok, "drift must not report success");
    let report: serde_json::Value = serde_json::from_str(&stdout)?;
    let drifted = report
        .get("drifted")
        .and_then(serde_json::Value::as_array)
        .ok_or("no drift array")?;
    assert_eq!(drifted.len(), 1);
    let moved = drifted.first().ok_or("no drift reported")?;
    assert_eq!(
        moved.get("message_id").and_then(serde_json::Value::as_str),
        Some("<needed@example.invalid>")
    );
    Ok(())
}

#[test]
fn index_writes_records_and_reports_the_selection() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    deliver(
        &maildir(home),
        "Archive",
        "u1",
        "<needed@example.invalid>",
        "",
        "the proposal",
    )?;
    deliver(
        &maildir(home),
        "Inbox",
        "u2",
        "<reply@example.invalid>",
        "In-Reply-To: <needed@example.invalid>\n",
        "agreed",
    )?;
    let question_file = questions(dir.path(), home, true)?;
    // The index must already exist: `mail index` opens rather than creates,
    // so a mistyped path is an error and not a silently empty corpus.
    let index = home.join("index.db");
    drop(kb::index::Index::open_for_rebuild(&index)?);

    let (ok, stdout, stderr) = run(
        home,
        &[
            "mail",
            "index",
            "--maildir",
            &maildir(home).display().to_string(),
            "--questions",
            &question_file.display().to_string(),
            "--index",
            &index.display().to_string(),
        ],
    )?;

    assert!(ok, "mail index failed: {stderr}");
    assert!(
        stdout.contains("2 selected of 2 catalogued"),
        "stdout: {stdout}"
    );
    let opened = kb::index::Index::open(&index)?;
    assert_eq!(
        opened.records_in("mail")?.len(),
        3,
        "two messages and a thread"
    );
    Ok(())
}

/// A misconfigured Maildir root is a misconfiguration worth naming, not an
/// empty mailbox worth reporting as success.
#[test]
fn a_missing_maildir_root_is_an_error_rather_than_an_empty_report() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    std::fs::create_dir_all(home.join(".local/share"))?;
    std::fs::write(
        home.join("questions.toml"),
        QUESTIONS.replace("PLACEHOLDER", &"0".repeat(64)),
    )?;

    let (ok, _, stderr) = run(
        home,
        &[
            "mail",
            "scope",
            "--maildir",
            &maildir(home).display().to_string(),
            "--questions",
            &home.join("questions.toml").display().to_string(),
        ],
    )?;

    assert!(!ok);
    assert!(stderr.contains("mail"), "stderr: {stderr}");
    Ok(())
}

#[test]
fn an_unreadable_question_file_is_reported() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    deliver(
        &maildir(home),
        "Archive",
        "u1",
        "<needed@example.invalid>",
        "",
        "x",
    )?;

    let (ok, _, stderr) = run(
        home,
        &[
            "mail",
            "scope",
            "--maildir",
            &maildir(home).display().to_string(),
            "--questions",
            "/nonexistent/q.toml",
        ],
    )?;

    assert!(!ok);
    assert!(stderr.contains("q.toml"), "stderr: {stderr}");
    Ok(())
}

/// Embedding without a configured endpoint must say which variables to set
/// rather than silently doing nothing.
#[test]
fn embedding_without_configuration_says_what_is_missing() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    deliver(
        &maildir(home),
        "Archive",
        "u1",
        "<needed@example.invalid>",
        "",
        "x",
    )?;

    let (ok, _, stderr) = run(home, &["mail", "embed"])?;

    assert!(!ok);
    assert!(
        stderr.contains("KB_EMBEDDING"),
        "the error must name the variables to set: {stderr}"
    );
    Ok(())
}

/// Embedding is resumable by construction: a vector is keyed by its span and
/// the model, so a run that stops partway leaves finished work behind and the
/// next run starts where it left off. That is the property a 34 MiB corpus
/// needs and the one a test can check cheaply.
#[test]
fn embedding_stores_vectors_and_skips_what_is_already_current() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    deliver(
        &maildir(home),
        "Archive",
        "u1",
        "<needed@example.invalid>",
        "",
        "the proposal",
    )?;
    let question_file = questions(dir.path(), home, true)?;
    let index = home.join("index.db");
    drop(kb::index::Index::open_for_rebuild(&index)?);
    let index_arg = index.display().to_string();
    let (built, _, stderr) = run(
        home,
        &[
            "mail",
            "index",
            "--maildir",
            &maildir(home).display().to_string(),
            "--questions",
            &question_file.display().to_string(),
            "--index",
            &index_arg,
        ],
    )?;
    assert!(built, "mail index failed: {stderr}");

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

    let (ok, first, stderr) = run_with(home, &["mail", "embed", "--index", &index_arg], &live)?;
    assert!(ok, "mail embed failed: {stderr}");
    assert!(first.contains("1 span embedded"), "stdout: {first}");

    let (ok, second, stderr) = run_with(home, &["mail", "embed", "--index", &index_arg], &live)?;
    assert!(ok, "second pass failed: {stderr}");
    assert!(
        second.contains("0 spans embedded, 1 already current"),
        "the second pass repeated finished work: {second}"
    );
    Ok(())
}

/// A long passage is embedded as several spans, because the embeddings table
/// is keyed by span and therefore already addresses any region of a stream.
#[test]
fn a_long_message_is_embedded_as_several_spans() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    let long = "a line of a very long message\n".repeat(400);
    deliver(
        &maildir(home),
        "Archive",
        "u1",
        "<needed@example.invalid>",
        "",
        &long,
    )?;
    let question_file = questions(dir.path(), home, true)?;
    let index = home.join("index.db");
    drop(kb::index::Index::open_for_rebuild(&index)?);
    let index_arg = index.display().to_string();
    run(
        home,
        &[
            "mail",
            "index",
            "--maildir",
            &maildir(home).display().to_string(),
            "--questions",
            &question_file.display().to_string(),
            "--index",
            &index_arg,
        ],
    )?;

    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"data":[{"embedding":[0.5,0.5]}]}"#)
        .expect_at_least(1)
        .create();
    let url = format!("{}/v1", server.url());

    let (ok, stdout, stderr) = run_with(
        home,
        &[
            "mail",
            "embed",
            "--index",
            &index_arg,
            "--chunk-bytes",
            "2000",
        ],
        &[
            ("KB_EMBEDDING_BASE_URL", url.as_str()),
            ("KB_EMBEDDING_MODEL", "fixture-model"),
        ],
    )?;

    assert!(ok, "mail embed failed: {stderr}");
    assert!(
        stdout.contains("7 spans embedded"),
        "a 12,000-byte message should chunk: {stdout}"
    );
    Ok(())
}

/// An endpoint that fails partway must say how far it got, because the work
/// is resumable and a run that reports nothing makes the operator redo it.
#[test]
fn a_failing_endpoint_reports_progress_before_stopping() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    deliver(
        &maildir(home),
        "Archive",
        "u1",
        "<needed@example.invalid>",
        "",
        "the proposal",
    )?;
    let question_file = questions(dir.path(), home, true)?;
    let index = home.join("index.db");
    drop(kb::index::Index::open_for_rebuild(&index)?);
    let index_arg = index.display().to_string();
    run(
        home,
        &[
            "mail",
            "index",
            "--maildir",
            &maildir(home).display().to_string(),
            "--questions",
            &question_file.display().to_string(),
            "--index",
            &index_arg,
        ],
    )?;

    let (ok, stdout, stderr) = run_with(
        home,
        &["mail", "embed", "--index", &index_arg],
        &[
            ("KB_EMBEDDING_BASE_URL", "http://127.0.0.1:1/v1"),
            ("KB_EMBEDDING_MODEL", "fixture-model"),
        ],
    )?;

    assert!(!ok);
    assert!(stderr.contains("stopped after 0 spans"), "stderr: {stderr}");
    assert!(stdout.contains("0 spans embedded"), "stdout: {stdout}");
    Ok(())
}

/// A full rebuild clears every corpus, but the store only repopulates its
/// own. Stopping there would make `kb reindex` silently delete the mail
/// corpus — and re-derivation cannot be the routine operation ST-004 calls
/// for if running it costs a corpus.
#[test]
fn reindex_rebuilds_every_corpus_not_only_the_store_backed_one() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    deliver(
        &maildir(home),
        "Archive",
        "u1",
        "<needed@example.invalid>",
        "",
        "the proposal",
    )?;
    let store = home.join(".local/share/kb/store");
    std::fs::create_dir_all(&store)?;
    drop(kb::store::GitBlobStore::open_or_init(&store)?);
    let index = home.join(".local/share/kb/index.db");
    let root = maildir(home).display().to_string();

    let (ok, stdout, stderr) = run(home, &["reindex", "--maildir", &root])?;

    assert!(ok, "reindex failed: {stderr}");
    assert!(stdout.contains("mail 1 record"), "stdout: {stdout}");
    let opened = kb::index::Index::open(&index)?;
    assert_eq!(opened.records_in("mail")?.len(), 1);
    Ok(())
}

/// A fresh store for the reindex tests, returning the index path it implies.
fn empty_store(home: &Path) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let store = home.join(".local/share/kb/store");
    std::fs::create_dir_all(&store)?;
    drop(kb::store::GitBlobStore::open_or_init(&store)?);
    Ok(home.join(".local/share/kb/index.db"))
}

/// Mail is indexed only when a Maildir root is configured. A mailbox sitting
/// at a conventional path is not configuration, so a rebuild with neither
/// `--maildir` nor `KB_MAIL_ROOT` must leave the mail corpus empty and say
/// why.
#[test]
fn reindex_skips_mail_when_no_maildir_is_configured() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    for conventional in [".local/share/mail", "Mail", "Maildir"] {
        deliver(
            &home.join(conventional),
            "Archive",
            "u1",
            "<needed@example.invalid>",
            "",
            "the proposal",
        )?;
    }
    let index = empty_store(home)?;

    let (ok, stdout, stderr) = run(home, &["reindex"])?;

    assert!(ok, "reindex failed: {stderr}");
    assert!(
        stdout.contains("mail skipped") && stdout.contains("KB_MAIL_ROOT"),
        "stdout: {stdout}"
    );
    let opened = kb::index::Index::open(&index)?;
    assert!(opened.records_in("mail")?.is_empty());
    Ok(())
}

/// `KB_MAIL_ROOT` alone is enough configuration for a rebuild to index mail.
#[test]
fn reindex_indexes_the_maildir_named_by_kb_mail_root() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    deliver(
        &maildir(home),
        "Archive",
        "u1",
        "<needed@example.invalid>",
        "",
        "the proposal",
    )?;
    let index = empty_store(home)?;
    let root = maildir(home).display().to_string();

    let (ok, stdout, stderr) = run_with(home, &["reindex"], &[("KB_MAIL_ROOT", root.as_str())])?;

    assert!(ok, "reindex failed: {stderr}");
    assert!(stdout.contains("mail 1 record"), "stdout: {stdout}");
    let opened = kb::index::Index::open(&index)?;
    assert_eq!(opened.records_in("mail")?.len(), 1);
    Ok(())
}

/// The flag is the more specific statement, so it wins over the variable.
#[test]
fn the_maildir_flag_overrides_kb_mail_root() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    let flagged = home.join("flagged");
    let variable = home.join("variable");
    deliver(
        &flagged,
        "Archive",
        "f1",
        "<flag@example.invalid>",
        "",
        "one",
    )?;
    deliver(
        &flagged,
        "Inbox",
        "f2",
        "<flag2@example.invalid>",
        "",
        "two",
    )?;
    deliver(
        &variable,
        "Archive",
        "v1",
        "<var@example.invalid>",
        "",
        "three",
    )?;
    let index = empty_store(home)?;
    let variable_arg = variable.display().to_string();
    let flagged_arg = flagged.display().to_string();

    let (ok, stdout, stderr) = run_with(
        home,
        &["reindex", "--maildir", &flagged_arg],
        &[("KB_MAIL_ROOT", variable_arg.as_str())],
    )?;

    assert!(ok, "reindex failed: {stderr}");
    assert!(stdout.contains("mail 2 records"), "stdout: {stdout}");
    let opened = kb::index::Index::open(&index)?;
    let records = opened.records_in("mail")?;
    assert_eq!(records.len(), 2, "{records:?}");
    Ok(())
}

/// `mail scope` has no mailbox to fall back on, so running it unconfigured
/// is an error naming what to set.
#[test]
fn scope_without_a_configured_maildir_is_an_error() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();

    let (ok, _, stderr) = run(home, &["mail", "scope"])?;

    assert!(!ok);
    assert!(stderr.contains("--maildir"), "stderr: {stderr}");
    Ok(())
}

/// An index rebuilt where the configured Maildir does not exist is a
/// correct index of one corpus, so absent mail is reported rather than
/// treated as a failure or passed over in silence.
#[test]
fn reindex_says_so_when_there_is_no_mail_on_this_machine() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    empty_store(home)?;
    let root = maildir(home).display().to_string();

    let (ok, stdout, stderr) = run(home, &["reindex", "--maildir", &root])?;

    assert!(ok, "reindex failed: {stderr}");
    assert!(
        stdout.contains("no mail at") && stdout.contains("empty on this machine"),
        "stdout: {stdout}"
    );
    Ok(())
}

/// A full rebuild must not cost the mail corpus its vectors.
///
/// The orphan sweep asks, per vector, whether any indexed passage still
/// covers the span it was computed from. A rebuild re-derives the
/// store-backed corpora first and mail afterwards, so sweeping in between
/// finds every mail vector describing text that is — for that moment —
/// not indexed, and deletes hours of embedding. Caught by rebuilding the
/// live index during T029: 11,573 vectors went in one command.
#[test]
fn a_full_rebuild_keeps_the_mail_corpus_vectors() -> TestResult {
    let dir = tempfile::tempdir()?;
    let home = dir.path();
    deliver(
        &maildir(home),
        "Archive",
        "u1",
        "<needed@example.invalid>",
        "",
        "the proposal",
    )?;
    let store = home.join(".local/share/kb/store");
    std::fs::create_dir_all(&store)?;
    drop(kb::store::GitBlobStore::open_or_init(&store)?);
    let index_path = home.join(".local/share/kb/index.db");
    let root = maildir(home).display().to_string();

    let (ok, _, stderr) = run(home, &["reindex", "--maildir", &root])?;
    assert!(ok, "the first reindex failed: {stderr}");

    // Give every mail passage a vector, the way an embedding pass would.
    {
        let index = kb::index::Index::open(&index_path)?;
        for record in index.records_in("mail")? {
            for passage in index.passages(&record)? {
                index.put_embedding(
                    &passage.stream_hash,
                    passage.span_start,
                    passage.span_len,
                    "a-model",
                    &kb::embedding::encode_embedding(&[0.5, 0.5]),
                )?;
            }
        }
    }
    let before = {
        let index = kb::index::Index::open(&index_path)?;
        index
            .records_in("mail")?
            .iter()
            .filter(|record| {
                index
                    .embedded_spans(record, "a-model")
                    .is_ok_and(|spans| !spans.is_empty())
            })
            .count()
    };
    assert!(before > 0, "the fixture embedded nothing");

    let (ok, stdout, stderr) = run(home, &["reindex", "--maildir", &root])?;
    assert!(ok, "the second reindex failed: {stderr}");
    assert!(
        !stdout.contains("dropped"),
        "a rebuild dropped vectors it should have kept: {stdout}"
    );

    let index = kb::index::Index::open(&index_path)?;
    let after = index
        .records_in("mail")?
        .iter()
        .filter(|record| {
            index
                .embedded_spans(record, "a-model")
                .is_ok_and(|spans| !spans.is_empty())
        })
        .count();
    assert_eq!(
        after, before,
        "the rebuild cost the mail corpus its vectors"
    );
    Ok(())
}
