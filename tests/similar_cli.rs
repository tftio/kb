//! CLI tests for `kb similar`, the semantic-neighbourhood verb.
//!
//! `kb similar` ranks stored vectors against the target node's own chunk-0
//! vector. It is the read verb that must keep working while the embedding
//! daemon is restarting, so these tests point it at a dead endpoint on purpose
//! and assert it neither contacts it nor cares.
//!
//! The vectors are hand-written rather than produced by a model, so what is
//! under test is the ranking and the exclusion of the target, not a model's
//! behaviour.
//!
//! Every spawned `kb` process is sandboxed: `HOME`, `XDG_CONFIG_HOME`, and
//! `--db` are pinned to the per-test temp directory, and `KB_EMBEDDING_*` is
//! controlled explicitly, so the result is identical on a clean CI runner and
//! on a developer machine with a live daemon.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::Value;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const EMBEDDING_ENV: [&str; 5] = [
    "KB_EMBEDDING_BASE_URL",
    "KB_EMBEDDING_MODEL",
    "KB_EMBEDDING_API_KEY",
    "KB_EMBEDDING_DOCUMENT_PREFIX",
    "KB_EMBEDDING_QUERY_PREFIX",
];

const FIXTURE_MODEL: &str = "fixture-embedding-model";

/// An endpoint that is guaranteed never to answer: port 1 is reserved. Any
/// verb that reaches the network with this configured will fail visibly.
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

/// The index a sandboxed `kb` resolves under `HOME`.
fn index_of(db: &Path) -> Result<kb::index::Index, Box<dyn std::error::Error>> {
    let home = db
        .parent()
        .ok_or("db path has no parent directory for HOME sandbox")?;
    Ok(kb::index::Index::open(
        &home.join(".local/share/kb/index.db"),
    )?)
}

/// Give every passage of `id` the vector `vector`, under `FIXTURE_MODEL`.
///
/// Vectors live in the derived index, keyed by the span they were computed
/// from (T029), so a fixture writes one per passage rather than one per
/// "chunk" of a node.
fn write_vector(db: &Path, id: &str, vector: &[f32]) -> TestResult {
    let index = index_of(db)?;
    for passage in index.passages(id)? {
        index.put_embedding(
            &passage.stream_hash,
            passage.span_start,
            passage.span_len,
            FIXTURE_MODEL,
            &kb::embedding::encode_embedding(vector),
        )?;
    }
    Ok(())
}

/// Five records around the unit circle. Ordered by closeness to `target`
/// ([1, 0]): `near` (cos 1.0), `mid` (~0.707), `far` (0.0), `opposite`
/// (-1.0). `unembedded` has no vector at all.
fn seed(db: &Path) -> TestResult {
    for id in ["target", "near", "mid", "far", "opposite", "unembedded"] {
        let body = format!("* {id}\n\nbody text for {id}\n");
        let run = run_kb(db, Some(&body), &["create", "--id", id], &[])?;
        assert!(run.status.success(), "seed create failed: {}", run.stderr);
    }
    write_vector(db, "target", &[1.0, 0.0])?;
    write_vector(db, "near", &[1.0, 0.0])?;
    write_vector(db, "mid", &[1.0, 1.0])?;
    write_vector(db, "far", &[0.0, 1.0])?;
    write_vector(db, "opposite", &[-1.0, 0.0])?;
    Ok(())
}

/// The identifiers in a `--json` envelope, in the order returned.
fn json_ids(stdout: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let v: Value = serde_json::from_str(stdout)?;
    let rows = v
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("no data array in envelope: {stdout}"))?;
    Ok(rows
        .iter()
        .filter_map(|r| r.get("id").and_then(Value::as_str).map(str::to_string))
        .collect())
}

#[test]
fn neighbours_are_ranked_by_similarity_and_never_include_the_target() -> TestResult {
    let (_dir, db) = temp_db("similar")?;
    seed(&db)?;

    let run = run_kb(
        &db,
        None,
        &["similar", "target", "--json"],
        &[("KB_EMBEDDING_MODEL", FIXTURE_MODEL)],
    )?;
    assert!(
        run.status.success(),
        "similar failed: {}{}",
        run.stdout,
        run.stderr
    );

    let ids = json_ids(&run.stdout)?;
    assert_eq!(ids, vec!["near", "mid", "far", "opposite"]);
    assert!(
        !ids.contains(&"target".to_string()),
        "a node is always its own nearest neighbour; it must not spend a result slot: {ids:?}"
    );
    Ok(())
}

#[test]
fn limit_truncates_from_the_near_end() -> TestResult {
    let (_dir, db) = temp_db("limit")?;
    seed(&db)?;

    let run = run_kb(
        &db,
        None,
        &["similar", "target", "--limit", "2", "--json"],
        &[("KB_EMBEDDING_MODEL", FIXTURE_MODEL)],
    )?;
    assert!(run.status.success(), "similar failed: {}", run.stderr);
    assert_eq!(json_ids(&run.stdout)?, vec!["near", "mid"]);
    Ok(())
}

/// The verb reads stored vectors and nothing else, so a configured endpoint
/// that can never answer must make no difference at all.
#[test]
fn a_dead_endpoint_changes_nothing_because_no_request_is_made() -> TestResult {
    let (_dir, db) = temp_db("offline")?;
    seed(&db)?;

    let offline = run_kb(
        &db,
        None,
        &["similar", "target", "--json"],
        &[
            ("KB_EMBEDDING_BASE_URL", DEAD_ENDPOINT),
            ("KB_EMBEDDING_MODEL", FIXTURE_MODEL),
        ],
    )?;
    assert!(
        offline.status.success(),
        "similar must not depend on the daemon: {}",
        offline.stderr
    );
    assert!(
        offline.stderr.is_empty(),
        "nothing was attempted, so nothing should be reported: {}",
        offline.stderr
    );
    assert_eq!(
        json_ids(&offline.stdout)?,
        vec!["near", "mid", "far", "opposite"]
    );
    Ok(())
}

/// An unembedded node is not a node without neighbours; saying so would be a
/// false claim rather than an incomplete one.
#[test]
fn a_node_with_no_stored_vector_fails_and_names_the_remedy() -> TestResult {
    let (_dir, db) = temp_db("noembed")?;
    seed(&db)?;

    let run = run_kb(
        &db,
        None,
        &["similar", "unembedded"],
        &[("KB_EMBEDDING_MODEL", FIXTURE_MODEL)],
    )?;
    assert!(
        !run.status.success(),
        "an empty success would read as 'nothing is similar', got: {}",
        run.stdout
    );
    let msg = format!("{}{}", run.stdout, run.stderr);
    assert!(
        msg.contains("kb embed"),
        "the message must name the remedy: {msg}"
    );
    assert!(
        msg.contains("unembedded"),
        "the message must name the node: {msg}"
    );
    Ok(())
}

/// A typo is a different problem from an unembedded node, and `kb backfill`
/// is no help with it.
#[test]
fn an_unknown_id_is_distinguished_from_an_unembedded_one() -> TestResult {
    let (_dir, db) = temp_db("unknown")?;
    seed(&db)?;

    let run = run_kb(&db, None, &["similar", "no-such-node"], &[])?;
    assert!(!run.status.success(), "stdout: {}", run.stdout);
    let msg = format!("{}{}", run.stdout, run.stderr);
    assert!(
        msg.contains("no node with id: no-such-node"),
        "the message must say the id is unknown: {msg}"
    );
    assert!(
        !msg.contains("kb backfill"),
        "backfill cannot fix a typo: {msg}"
    );
    Ok(())
}

/// With no model configured the verb still works, taking whichever model the
/// node was embedded under. A read verb should not require write-path
/// configuration to be present.
#[test]
fn an_unconfigured_model_falls_back_to_the_nodes_own() -> TestResult {
    let (_dir, db) = temp_db("nomodel")?;
    seed(&db)?;

    let run = run_kb(&db, None, &["similar", "target", "--json"], &[])?;
    assert!(run.status.success(), "similar failed: {}", run.stderr);
    assert_eq!(
        json_ids(&run.stdout)?,
        vec!["near", "mid", "far", "opposite"]
    );
    Ok(())
}
