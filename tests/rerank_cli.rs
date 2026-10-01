//! `kb search` under a configured cross-encoder (CLI-002).
//!
//! Reranking is the ordering stage T019 measured and T010 landed: it reorders
//! the top of the fused ranking and does nothing else. These tests hold it to
//! that. The reranker is a `mockito` server rather than a model, because what
//! needs testing at this boundary is the contract — which candidates are sent,
//! which order comes back, and what happens when the endpoint fails — not the
//! quality of a cross-encoder's judgement.
#![allow(
    clippy::significant_drop_tightening,
    reason = "a mockito Server guard is held for the test's duration on purpose; dropping it early tears down the endpoint the child process is talking to"
)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Variables that would otherwise leak a developer's live daemons into a test
/// process and make ranking machine-dependent.
const ISOLATED_ENV: [&str; 8] = [
    "KB_EMBEDDING_BASE_URL",
    "KB_EMBEDDING_MODEL",
    "KB_EMBEDDING_API_KEY",
    "KB_EMBEDDING_DOCUMENT_PREFIX",
    "KB_EMBEDDING_QUERY_PREFIX",
    "KB_EMBEDDING_MIN_SIMILARITY",
    "KB_RERANK_BASE_URL",
    "KB_RERANK_TOP_K",
];

struct Run {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

fn run_kb(
    db: &Path,
    stdin: Option<&str>,
    args: &[&str],
    env: &[(&str, &str)],
) -> Result<Run, Box<dyn std::error::Error>> {
    let home = db
        .parent()
        .ok_or("db path has no parent for HOME sandbox")?;
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_kb"));
    cmd.env("HOME", home)
        .env("XDG_CONFIG_HOME", home)
        .arg("--db")
        .arg(db);
    for var in ISOLATED_ENV {
        cmd.env_remove(var);
    }
    for (key, value) in env {
        cmd.env(key, value);
    }
    for arg in args {
        cmd.arg(arg);
    }
    cmd.stdin(if stdin.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    if let Some(payload) = stdin {
        child
            .stdin
            .as_mut()
            .ok_or("kb child stdin missing")?
            .write_all(payload.as_bytes())?;
    }
    let out = child.wait_with_output()?;
    Ok(Run {
        status: out.status,
        stdout: String::from_utf8(out.stdout)?,
        stderr: String::from_utf8(out.stderr)?,
    })
}

fn temp_db(name: &str) -> Result<(tempfile::TempDir, PathBuf), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join(format!("{name}.db"));
    Ok((dir, db))
}

/// Three nodes, all matching `rust`, seeded in an order FTS5 reproduces.
fn seed(db: &Path) -> TestResult {
    for (id, body) in [
        ("aaa", "* Alpha\n\nrust ownership notes\n"),
        ("bbb", "* Beta\n\nrust lifetime notes\n"),
        ("ccc", "* Gamma\n\nrust trait notes\n"),
    ] {
        let run = run_kb(db, Some(body), &["create", "--id", id], &[])?;
        assert!(run.status.success(), "seed create failed: {}", run.stderr);
    }
    Ok(())
}

/// Search and return the matched ids in ranked order.
fn ranked(
    db: &Path,
    args: &[&str],
    env: &[(&str, &str)],
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let run = run_kb(db, None, args, env)?;
    assert!(
        run.status.success(),
        "search {args:?} failed: {}{}",
        run.stdout,
        run.stderr
    );
    Ok(run
        .stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| line.split_whitespace().next().map(str::to_string))
        .collect())
}

/// A rerank response scoring documents in the order given.
fn scores(values: &[f32]) -> String {
    let results: Vec<String> = values
        .iter()
        .enumerate()
        .map(|(index, score)| format!(r#"{{"index":{index},"relevance_score":{score}}}"#))
        .collect();
    format!(r#"{{"results":[{}]}}"#, results.join(","))
}

#[test]
fn a_configured_reranker_reorders_the_ranking() -> TestResult {
    let (_dir, db) = temp_db("reorders")?;
    seed(&db)?;
    let fused = ranked(&db, &["search", "rust", "--no-vector"], &[])?;
    assert_eq!(fused.len(), 3, "expected three hits, got {fused:?}");

    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/v1/rerank")
        .with_header("content-type", "application/json")
        .with_body(scores(&[0.1, 0.2, 0.9]))
        .create();
    let url = server.url();
    let reranked = ranked(
        &db,
        &["search", "rust", "--no-vector"],
        &[("KB_RERANK_BASE_URL", &url)],
    )?;
    mock.assert();

    let mut expected: Vec<String> = fused;
    expected.reverse();
    assert_eq!(
        reranked, expected,
        "the reranker's order should be the returned order"
    );
    Ok(())
}

#[test]
fn reranking_returns_the_same_nodes_it_was_given() -> TestResult {
    let (_dir, db) = temp_db("same-nodes")?;
    seed(&db)?;
    let mut server = mockito::Server::new();
    let _mock = server
        .mock("POST", "/v1/rerank")
        .with_body(scores(&[0.9, 0.1, 0.5]))
        .create();
    let url = server.url();
    let mut reranked = ranked(
        &db,
        &["search", "rust", "--no-vector"],
        &[("KB_RERANK_BASE_URL", &url)],
    )?;
    reranked.sort();
    assert_eq!(reranked, vec!["aaa", "bbb", "ccc"]);
    Ok(())
}

#[test]
fn no_rerank_contacts_nothing_and_keeps_the_fused_order() -> TestResult {
    let (_dir, db) = temp_db("no-rerank")?;
    seed(&db)?;
    let fused = ranked(&db, &["search", "rust", "--no-vector"], &[])?;

    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/v1/rerank")
        .with_body(scores(&[0.1, 0.2, 0.9]))
        .expect(0)
        .create();
    let url = server.url();
    let unranked = ranked(
        &db,
        &["search", "rust", "--no-vector", "--no-rerank"],
        &[("KB_RERANK_BASE_URL", &url)],
    )?;
    mock.assert();
    assert_eq!(unranked, fused);
    Ok(())
}

#[test]
fn an_unreachable_reranker_keeps_the_fused_order_and_says_so() -> TestResult {
    let (_dir, db) = temp_db("unreachable")?;
    seed(&db)?;
    let fused = ranked(&db, &["search", "rust", "--no-vector"], &[])?;
    // Port 1 is reserved and unbound; nothing on this machine answers there.
    let run = run_kb(
        &db,
        None,
        &["search", "rust", "--no-vector"],
        &[("KB_RERANK_BASE_URL", "http://127.0.0.1:1")],
    )?;
    assert!(
        run.status.success(),
        "search should not fail: {}",
        run.stderr
    );
    let ids: Vec<String> = run
        .stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| line.split_whitespace().next().map(str::to_string))
        .collect();
    assert_eq!(ids, fused, "a dead reranker must not change the ranking");
    assert!(
        run.stderr.contains("fused ranking"),
        "a degraded ranking must be explained: {}",
        run.stderr
    );
    Ok(())
}

#[test]
fn a_reranker_answering_with_the_wrong_number_of_scores_is_refused() -> TestResult {
    let (_dir, db) = temp_db("mismatch")?;
    seed(&db)?;
    let fused = ranked(&db, &["search", "rust", "--no-vector"], &[])?;
    let mut server = mockito::Server::new();
    let _mock = server
        .mock("POST", "/v1/rerank")
        .with_body(scores(&[0.9]))
        .create();
    let url = server.url();
    let run = run_kb(
        &db,
        None,
        &["search", "rust", "--no-vector"],
        &[("KB_RERANK_BASE_URL", &url)],
    )?;
    assert!(run.status.success());
    let ids: Vec<String> = run
        .stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| line.split_whitespace().next().map(str::to_string))
        .collect();
    assert_eq!(ids, fused);
    assert!(
        run.stderr.contains("scores for"),
        "the mismatch must be named: {}",
        run.stderr
    );
    Ok(())
}

#[test]
fn the_window_bounds_what_reranking_may_touch() -> TestResult {
    let (_dir, db) = temp_db("window")?;
    seed(&db)?;
    let fused = ranked(&db, &["search", "rust", "--no-vector"], &[])?;
    let mut server = mockito::Server::new();
    let _mock = server
        .mock("POST", "/v1/rerank")
        .with_body(scores(&[0.1, 0.9]))
        .create();
    let url = server.url();
    let reranked = ranked(
        &db,
        &["search", "rust", "--no-vector"],
        &[("KB_RERANK_BASE_URL", &url), ("KB_RERANK_TOP_K", "2")],
    )?;
    let expected = vec![
        fused.get(1).ok_or("fused ranking too short")?.clone(),
        fused.first().ok_or("fused ranking too short")?.clone(),
        fused.get(2).ok_or("fused ranking too short")?.clone(),
    ];
    assert_eq!(
        reranked, expected,
        "the third candidate was outside the window and must not move"
    );
    Ok(())
}

#[test]
fn the_reranker_is_shown_the_text_the_vectors_were_computed_from() -> TestResult {
    let (_dir, db) = temp_db("shown")?;
    seed(&db)?;
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/v1/rerank")
        // The title is prepended and the property drawer is absent: this is
        // `embeddable_text`, not the rendered org `kb get` prints.
        .match_body(mockito::Matcher::AllOf(vec![
            mockito::Matcher::Regex("ownership notes".into()),
            mockito::Matcher::Regex(r#""query":"rust""#.into()),
        ]))
        .with_body(scores(&[0.3, 0.2, 0.1]))
        .create();
    let url = server.url();
    let run = run_kb(
        &db,
        None,
        &["search", "rust", "--no-vector"],
        &[("KB_RERANK_BASE_URL", &url)],
    )?;
    assert!(run.status.success(), "{}", run.stderr);
    mock.assert();
    assert!(
        !run.stdout.contains(":PROPERTIES:"),
        "search output should not carry a drawer"
    );
    Ok(())
}

#[test]
fn explain_reports_what_each_signal_returned() -> TestResult {
    let (_dir, db) = temp_db("explain")?;
    seed(&db)?;
    let run = run_kb(
        &db,
        None,
        &["search", "rust", "--no-vector", "--explain"],
        &[],
    )?;
    assert!(run.status.success(), "{}", run.stderr);
    assert!(
        run.stdout.contains("kb-lexical (lexical, kb)"),
        "the lexical signal is not named: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("fused:"),
        "fusion is not reported: {}",
        run.stdout
    );
    for id in ["aaa", "bbb", "ccc"] {
        assert!(
            run.stdout.contains(id),
            "{id} is missing from the trace: {}",
            run.stdout
        );
    }
    Ok(())
}

#[test]
fn explain_reports_what_reranking_changed() -> TestResult {
    let (_dir, db) = temp_db("explain-rerank")?;
    seed(&db)?;
    let mut server = mockito::Server::new();
    let _mock = server
        .mock("POST", "/v1/rerank")
        .with_body(scores(&[0.1, 0.2, 0.9]))
        .create();
    let url = server.url();
    let run = run_kb(
        &db,
        None,
        &["search", "rust", "--no-vector", "--explain", "--json"],
        &[("KB_RERANK_BASE_URL", &url)],
    )?;
    assert!(run.status.success(), "{}", run.stderr);
    let envelope: serde_json::Value = serde_json::from_str(&run.stdout)?;
    let trace = envelope
        .get("data")
        .ok_or("no data in the search envelope")?;
    let rerank = trace.get("rerank").ok_or("no rerank in the trace")?;
    assert_eq!(
        rerank.get("window").and_then(serde_json::Value::as_u64),
        Some(3)
    );
    let before = rerank.get("before").ok_or("no order before reranking")?;
    let after = rerank.get("after").ok_or("no order after reranking")?;
    assert_ne!(before, after, "the trace claims reranking changed nothing");
    let signals = trace
        .get("signals")
        .and_then(serde_json::Value::as_array)
        .ok_or("no signals in the trace")?;
    assert_eq!(
        signals.len(),
        1,
        "only the lexical signal runs under --no-vector"
    );
    // Each stage reports how long it took, because a dense-stage threshold
    // (T025) is unreadable from a total that a reranker dominates.
    assert!(
        signals
            .first()
            .and_then(|signal| signal.get("elapsed_ms"))
            .and_then(serde_json::Value::as_f64)
            .is_some_and(|ms| ms >= 0.0),
        "the signal's elapsed time is not reported: {signals:?}"
    );
    assert!(
        rerank
            .get("elapsed_ms")
            .and_then(serde_json::Value::as_f64)
            .is_some_and(|ms| ms >= 0.0),
        "the rerank stage's elapsed time is not reported: {rerank}"
    );
    Ok(())
}

#[test]
fn a_corpus_filter_reports_what_it_excluded() -> TestResult {
    let (_dir, db) = temp_db("corpus-filter")?;
    seed(&db)?;
    // The mail corpus is served from the derived index under XDG_DATA_HOME,
    // which is the per-test sandbox and holds no index — so the assertion is
    // about the filter's effect on the kb signals, not about mail results.
    let run = run_kb(
        &db,
        None,
        &[
            "search",
            "rust",
            "--no-vector",
            "--corpus",
            "mail",
            "--explain",
        ],
        &[],
    )?;
    if run.status.success() {
        assert!(
            !run.stdout.contains("kb-lexical"),
            "a mail query ran a kb signal: {}",
            run.stdout
        );
    } else {
        // Either backend may be the one missing from a sandbox — the derived
        // index or mu's — and both name themselves when they are.
        let reported = format!("{}{}", run.stdout, run.stderr);
        assert!(
            reported.contains("index") || reported.contains("mu"),
            "a missing backend must say which: {reported}"
        );
    }
    Ok(())
}

#[test]
fn explain_says_when_a_dead_reranker_left_the_fused_order() -> TestResult {
    let (_dir, db) = temp_db("explain-dead")?;
    seed(&db)?;
    let run = run_kb(
        &db,
        None,
        &["search", "rust", "--no-vector", "--explain"],
        &[("KB_RERANK_BASE_URL", "http://127.0.0.1:1")],
    )?;
    assert!(run.status.success(), "{}", run.stderr);
    assert!(
        run.stdout.contains("fused order stands"),
        "the trace does not say the reranking was abandoned: {}",
        run.stdout
    );
    assert!(
        run.stderr.contains("fused ranking"),
        "the degradation was not explained on stderr: {}",
        run.stderr
    );
    Ok(())
}

#[test]
fn explain_reports_a_reranking_that_changed_nothing() -> TestResult {
    let (_dir, db) = temp_db("explain-stable")?;
    seed(&db)?;
    let mut server = mockito::Server::new();
    let _mock = server
        .mock("POST", "/v1/rerank")
        // Descending scores agree with the fused order, so nothing moves.
        .with_body(scores(&[0.9, 0.5, 0.1]))
        .create();
    let url = server.url();
    let run = run_kb(
        &db,
        None,
        &["search", "rust", "--no-vector", "--explain"],
        &[("KB_RERANK_BASE_URL", &url)],
    )?;
    assert!(run.status.success(), "{}", run.stderr);
    assert!(
        run.stdout.contains("0 positions changed"),
        "a reranking that agreed with fusion is not reported as such: {}",
        run.stdout
    );
    Ok(())
}
