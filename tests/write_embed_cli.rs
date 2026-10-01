//! CLI tests for the embedding side of `kb create` and `kb update`.
//!
//! The failure contract is the point of this surface. The `SessionEnd` hook
//! captures conversations through `kb create`, so a write must never be lost
//! because a model is restarting: an unreachable endpoint leaves the node
//! stored, warns on stderr, and defers the vector to `kb embed`. A
//! configuration that names no endpoint at all is a deliberate choice and says
//! nothing, because a warning on every write is a warning nobody reads.
//!
//! Every spawned `kb` process is sandboxed: `HOME`, `XDG_CONFIG_HOME`, and
//! `--db` are pinned to the per-test temp directory and `KB_EMBEDDING_*` is
//! controlled explicitly.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const EMBEDDING_ENV: [&str; 6] = [
    "KB_EMBEDDING_BASE_URL",
    "KB_EMBEDDING_MODEL",
    "KB_EMBEDDING_API_KEY",
    "KB_EMBEDDING_DOCUMENT_PREFIX",
    "KB_EMBEDDING_QUERY_PREFIX",
    "KB_EMBEDDING_MIN_SIMILARITY",
];

const FIXTURE_MODEL: &str = "fixture-embedding-model";

/// Port 1 is reserved and never listening.
const DEAD_ENDPOINT: &str = "http://127.0.0.1:1/v1";

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
    let bin = env!("CARGO_BIN_EXE_kb");
    let home = db
        .parent()
        .ok_or("db path has no parent directory for HOME sandbox")?;
    let mut cmd = Command::new(bin);
    cmd.env("HOME", home)
        .env("XDG_CONFIG_HOME", home)
        .arg("--db")
        .arg(db);
    for var in EMBEDDING_ENV {
        cmd.env_remove(var);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    for a in args {
        cmd.arg(a);
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

/// A mock OpenAI-shaped embeddings endpoint answering every request with the
/// same vector. Held by the caller for the child's lifetime.
fn embedding_endpoint() -> (mockito::ServerGuard, String) {
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"data":[{"embedding":[0.5,0.5]}]}"#)
        .expect_at_least(1)
        .create();
    let url = format!("{}/v1", server.url());
    (server, url)
}

const fn live_env(url: &str) -> [(&str, &str); 2] {
    [
        ("KB_EMBEDDING_BASE_URL", url),
        ("KB_EMBEDDING_MODEL", FIXTURE_MODEL),
    ]
}

/// The index a sandboxed `kb` resolves under `HOME`.
fn index_of(db: &Path) -> Result<kb::index::Index, Box<dyn std::error::Error>> {
    let home = db
        .parent()
        .ok_or("db path has no parent directory for HOME sandbox")?;
    Ok(kb::index::Index::open(
        &home.join(".local/share/kb/index.db"),
    )?)
}

/// One row per stored vector of `id`, as (span start, span length).
///
/// Vectors are keyed by the span of the stream they were computed from since
/// T029, so what used to be a chunk index is now a byte range — the same fact
/// addressed more precisely.
fn embedding_rows(db: &Path, id: &str) -> Result<Vec<(i64, i64)>, Box<dyn std::error::Error>> {
    Ok(index_of(db)?.embedded_spans(id, FIXTURE_MODEL)?)
}

/// The L2 norm of each stored vector for `id`, by position.
fn embedding_norms(db: &Path, id: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let index = index_of(db)?;
    let stream = index.record(id)?.stream_hash;
    let mut norms = Vec::new();
    for (start, len) in embedding_rows(db, id)? {
        if let Some(blob) = index.embedding(&stream, start, len, FIXTURE_MODEL)? {
            let v = kb::embedding::decode_embedding(&blob)?;
            norms.push(v.iter().map(|x| x * x).sum::<f32>().sqrt());
        }
    }
    Ok(norms)
}

/// A body long enough to chunk. Distinct paragraphs rather than one giant
/// one, so the chunker packs at its own boundaries instead of hard-splitting.
fn long_body(paragraphs: usize) -> String {
    use std::fmt::Write as _;
    let mut s = String::from("* Long note\n\n");
    for i in 0..paragraphs {
        let _ = writeln!(
            s,
            "Paragraph {i} of a deliberately long document. {}\n",
            "filler sentence about ownership and lifetimes. ".repeat(20)
        );
    }
    s
}

/// The single row expected for `id`, failing the test with the actual row
/// list rather than panicking on an index.
fn only_row(rows: &[(i64, i64)]) -> Result<(i64, i64), Box<dyn std::error::Error>> {
    match rows {
        [row] => Ok(*row),
        other => Err(format!("expected exactly one embedding row, got {other:?}").into()),
    }
}

/// The stream a record's passages address.
fn stream_of(db: &Path, id: &str) -> Result<String, Box<dyn std::error::Error>> {
    Ok(index_of(db)?.record(id)?.stream_hash)
}

#[test]
fn a_created_node_is_embedded_and_its_vector_is_keyed_by_its_stream() -> TestResult {
    let (_dir, db) = temp_db("created")?;
    let (_server, url) = embedding_endpoint();

    let run = run_kb(
        &db,
        Some("* Alpha\n\nbody text\n"),
        &["create", "--id", "a"],
        &live_env(&url),
    )?;
    assert!(run.status.success(), "create failed: {}", run.stderr);
    assert!(
        run.stderr.is_empty(),
        "a successful embedding says nothing: {}",
        run.stderr
    );

    let row = only_row(&embedding_rows(&db, "a")?)?;
    let index = index_of(&db)?;
    let first = index
        .passages("a")?
        .first()
        .map(|passage| passage.span_start)
        .ok_or("the record has no passages")?;
    assert_eq!(row.0, first, "a short record is one span, its first");
    // The vector is keyed by the stream it was computed from, so it cannot
    // describe a revision the record does not have. That used to be a
    // `source_updated_at` column that had to agree with the node's timestamp;
    // since T029 it is a property of the address rather than a claim.
    let stream = stream_of(&db, "a")?;
    assert!(
        index
            .embedding(&stream, row.0, row.1, FIXTURE_MODEL)?
            .is_some(),
        "the vector is not keyed by the record's own stream"
    );
    Ok(())
}

#[test]
fn a_long_node_is_stored_as_tiling_spans() -> TestResult {
    let (_dir, db) = temp_db("chunked")?;
    let (_server, url) = embedding_endpoint();

    let body = long_body(20);
    let run = run_kb(
        &db,
        Some(&body),
        &["create", "--id", "long"],
        &live_env(&url),
    )?;
    assert!(run.status.success(), "create failed: {}", run.stderr);

    let rows = embedding_rows(&db, "long")?;
    assert!(
        rows.len() > 1,
        "a document past the chunk budget must produce several rows, got {}",
        rows.len()
    );
    // Spans tile the record: each starts where the last ended, and none
    // overlaps. A gap would be text no query can reach through this record;
    // an overlap would be the same sentence counted twice in a ranking.
    let mut expected_start = rows.first().map_or(0, |row| row.0);
    for (start, len) in &rows {
        assert_eq!(
            *start, expected_start,
            "spans do not tile the record: {rows:?}"
        );
        expected_start = start + len;
    }
    // There is no whole-record vector any more. A record is represented by
    // its best-matching span, which is what `rank_by_embedding` selects, so a
    // vector averaging the whole document would only dilute it.
    let index = index_of(&db)?;
    let covered: i64 = rows.iter().map(|(_, len)| len).sum();
    let held: i64 = index
        .passages("long")?
        .iter()
        .map(|passage| passage.span_len)
        .sum();
    assert_eq!(covered, held, "the spans do not cover every passage");

    // The stub answers every request with the same non-unit vector, and every
    // span carries it verbatim. Nothing is normalised on the way in: a
    // centroid used to be stored alongside the passages to stand for the
    // whole node, and dropping it is what lets a record be represented by its
    // best-matching span instead of by an average of everything it says.
    let norms = embedding_norms(&db, "long")?;
    assert_eq!(norms.len(), rows.len(), "a norm per stored span");
    for (ix, norm) in norms.iter().enumerate() {
        assert!(
            (norm - 1.0).abs() > 1e-3,
            "span {ix} should be the vector as returned, norm was {norm}"
        );
    }
    Ok(())
}

/// A node small enough to need no splitting stores exactly one row, because
/// the centroid of a single vector is that vector and adds nothing.
#[test]
fn a_short_node_stores_one_row_and_no_redundant_centroid() -> TestResult {
    let (_dir, db) = temp_db("single")?;
    let (_server, url) = embedding_endpoint();

    let run = run_kb(
        &db,
        Some("* Alpha\n\nshort enough to fit in one chunk\n"),
        &["create", "--id", "a"],
        &live_env(&url),
    )?;
    assert!(run.status.success(), "create failed: {}", run.stderr);
    assert_eq!(embedding_rows(&db, "a")?.len(), 1);
    Ok(())
}

/// A node that shrinks must not keep chunks describing text it no longer
/// contains; those orphans would go on matching queries forever.
#[test]
fn update_replaces_a_nodes_chunks_rather_than_accumulating_them() -> TestResult {
    let (_dir, db) = temp_db("replace")?;
    let (_server, url) = embedding_endpoint();

    let created = run_kb(
        &db,
        Some(&long_body(20)),
        &["create", "--id", "n"],
        &live_env(&url),
    )?;
    assert!(
        created.status.success(),
        "create failed: {}",
        created.stderr
    );
    let before = embedding_rows(&db, "n")?.len();
    assert!(before > 1, "fixture must start chunked, got {before}");
    let before_stream = stream_of(&db, "n")?;

    let updated = run_kb(
        &db,
        Some("* Long note\n\nnow it is short\n"),
        &["update", "n"],
        &live_env(&url),
    )?;
    assert!(
        updated.status.success(),
        "update failed: {}",
        updated.stderr
    );

    let after = only_row(&embedding_rows(&db, "n")?)?;
    let index = index_of(&db)?;
    let stream = stream_of(&db, "n")?;
    assert_ne!(
        stream, before_stream,
        "shortening left the stream unchanged"
    );
    assert!(
        index
            .embedding(&stream, after.0, after.1, FIXTURE_MODEL)?
            .is_some(),
        "the replacement is not keyed by the new stream"
    );
    assert!(
        index
            .embedding(&before_stream, 0, 2000, FIXTURE_MODEL)?
            .is_none(),
        "a vector for text the record no longer holds survived"
    );
    Ok(())
}

/// The operator's failure contract: the capture survives, the warning is
/// loud, and the work is deferred rather than dropped.
#[test]
fn an_unreachable_endpoint_leaves_the_node_stored_and_warns() -> TestResult {
    let (_dir, db) = temp_db("unreachable")?;

    let run = run_kb(
        &db,
        Some("* Alpha\n\nbody text\n"),
        &["create", "--id", "a"],
        &[
            ("KB_EMBEDDING_BASE_URL", DEAD_ENDPOINT),
            ("KB_EMBEDDING_MODEL", FIXTURE_MODEL),
        ],
    )?;
    assert!(
        run.status.success(),
        "a capture must not be lost to a restarting model: {}",
        run.stderr
    );
    assert!(
        run.stderr.contains("node a"),
        "the warning must name the node: {}",
        run.stderr
    );
    assert!(
        run.stderr.contains("127.0.0.1:1"),
        "the warning must name the endpoint tried: {}",
        run.stderr
    );
    assert!(
        run.stderr.contains("kb embed"),
        "the warning must name the remedy: {}",
        run.stderr
    );

    let got = run_kb(&db, None, &["get", "a"], &[])?;
    assert!(
        got.status.success(),
        "node must be retrievable: {}",
        got.stderr
    );
    assert!(got.stdout.contains("Alpha"), "{}", got.stdout);

    assert!(
        embedding_rows(&db, "a")?.is_empty(),
        "a failed embedding must write no rows"
    );
    Ok(())
}

/// Not configuring an endpoint is a choice, not a fault.
#[test]
fn embedding_disabled_writes_nothing_and_says_nothing() -> TestResult {
    let (_dir, db) = temp_db("disabled")?;

    for (args, body) in [
        (vec!["create", "--id", "a"], "* Alpha\n\nbody text\n"),
        (vec!["update", "a"], "* Alpha\n\nrevised body\n"),
    ] {
        let run = run_kb(&db, Some(body), &args, &[])?;
        assert!(run.status.success(), "{args:?} failed: {}", run.stderr);
        assert!(
            run.stderr.is_empty(),
            "{args:?} warned about a configuration choice: {}",
            run.stderr
        );
    }
    assert!(embedding_rows(&db, "a")?.is_empty());
    Ok(())
}

// ── Partial configuration on the write path ───────────────────────────

/// An endpoint with no model is a typo, not a choice. Left silent — as it was
/// once the model became required — every write accumulates without a vector
/// and nothing points at `backfill`.
#[test]
fn an_endpoint_without_a_model_warns_and_names_the_remedy() -> TestResult {
    let (_dir, db) = temp_db("partial_config")?;
    let (_server, url) = embedding_endpoint();

    let run = run_kb(
        &db,
        Some("* Partial\n\nstored without a vector\n"),
        &["create", "--id", "p1"],
        &[("KB_EMBEDDING_BASE_URL", url.as_str())],
    )?;
    assert!(
        run.status.success(),
        "the node must still be written: {}{}",
        run.stdout,
        run.stderr
    );
    assert!(
        run.stderr.contains("KB_EMBEDDING_MODEL"),
        "stderr must name the missing variable: {:?}",
        run.stderr
    );
    assert!(
        run.stderr.contains("kb embed"),
        "stderr must name the remedy: {:?}",
        run.stderr
    );
    assert!(
        embedding_rows(&db, "p1")?.is_empty(),
        "no vector should have been written"
    );
    Ok(())
}

/// The other half of the distinction: configuring nothing is a deliberate
/// choice, and a tool that nags about a choice is noise.
#[test]
fn a_wholly_unconfigured_environment_stays_silent() -> TestResult {
    let (_dir, db) = temp_db("no_config")?;

    let run = run_kb(
        &db,
        Some("* Silent\n\nno endpoint, no complaint\n"),
        &["create", "--id", "s1"],
        &[],
    )?;
    assert!(run.status.success(), "create failed: {}", run.stderr);
    assert!(
        !run.stderr.contains("embedding"),
        "an unconfigured environment must not mention embedding: {:?}",
        run.stderr
    );
    assert!(
        !run.stderr.contains("backfill"),
        "nor suggest a remedy for a choice: {:?}",
        run.stderr
    );
    assert!(embedding_rows(&db, "s1")?.is_empty());
    Ok(())
}
