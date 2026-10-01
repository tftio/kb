//! `kb queue` at the process boundary (CLI-002).
//!
//! The queue's three numbers exist to make one failure visible: the endpoint
//! acknowledging while nothing is ingested. That is only true if an operator
//! can actually read them, so what the command prints is asserted here rather
//! than assumed from the fact that the library computes it.

use std::path::Path;
use std::process::Command;

type TestResult = Result<(), Box<dyn std::error::Error>>;

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn run_kb(home: &Path, args: &[&str]) -> Result<Run, Box<dyn std::error::Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_kb"));
    command.env("HOME", home).env("XDG_CONFIG_HOME", home);
    for var in [
        "KB_EMBEDDING_BASE_URL",
        "KB_EMBEDDING_MODEL",
        "KB_EMBEDDING_API_KEY",
    ] {
        command.env_remove(var);
    }
    let output = command.args(args).output()?;
    Ok(Run {
        ok: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

/// A home with a queue holding `waiting` submissions.
fn seeded(waiting: &[&str]) -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let home = tempfile::tempdir()?;
    let queue = kb::ingest::Queue::open(&home.path().join(".local/share/kb/queue"))?;
    for id in waiting {
        queue.enqueue(&kb::ingest::Submission {
            id: (*id).to_owned(),
            corpus: "kb".to_owned(),
            document: format!("* Session {id}\n\nthe body of {id}.\n"),
            provenance: None,
        })?;
    }
    Ok(home)
}

#[test]
fn status_reports_the_three_numbers() -> TestResult {
    let home = seeded(&["cc-1", "cc-2"])?;
    let run = run_kb(home.path(), &["queue", "status"])?;
    assert!(run.ok, "queue status failed: {}", run.stderr);
    assert!(run.stdout.contains('2'), "no depth in: {}", run.stdout);
    assert!(
        run.stdout.contains("dead"),
        "the dead-letter count is not reported: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("oldest"),
        "the oldest wait is not reported: {}",
        run.stdout
    );
    Ok(())
}

/// The worker drains the queue the server accepted into, wherever that is.
/// `--queue` and `KB_QUEUE_PATH` are the same setting `kb-mcp` reads, so a
/// deployment that names one place names it for both.
#[test]
fn the_queue_can_be_named_by_flag_or_environment() -> TestResult {
    let home = tempfile::tempdir()?;
    let elsewhere = home.path().join("somewhere-else");
    let queue = kb::ingest::Queue::open(&elsewhere)?;
    queue.enqueue(&kb::ingest::Submission {
        id: "cc-7".to_owned(),
        corpus: "kb".to_owned(),
        document: "* Session cc-7\n\nthe body.\n".to_owned(),
        provenance: None,
    })?;
    let path = elsewhere.to_string_lossy().into_owned();

    let by_flag = run_kb(
        home.path(),
        &["queue", "--queue", &path, "status", "--json"],
    )?;
    assert!(by_flag.ok, "queue status failed: {}", by_flag.stderr);
    assert!(
        by_flag.stdout.contains("\"depth\":1") || by_flag.stdout.contains("\"depth\": 1"),
        "the named queue was not read: {}",
        by_flag.stdout
    );

    let mut command = Command::new(env!("CARGO_BIN_EXE_kb"));
    let output = command
        .env("HOME", home.path())
        .env("XDG_CONFIG_HOME", home.path())
        .env("KB_QUEUE_PATH", &elsewhere)
        .args(["queue", "status", "--json"])
        .output()?;
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("\"depth\":1") || stdout.contains("\"depth\": 1"),
        "KB_QUEUE_PATH was not honoured: {stdout}"
    );

    // The default location is untouched: nothing was ever enqueued there.
    let default = run_kb(home.path(), &["queue", "status", "--json"])?;
    assert!(default.ok, "{}", default.stderr);
    assert!(
        default.stdout.contains("\"depth\":0") || default.stdout.contains("\"depth\": 0"),
        "the default queue reports the named queue's contents: {}",
        default.stdout
    );
    Ok(())
}

/// An empty queue is a report, not a failure, and it says so rather than
/// printing nothing — silence reads as "the command did not run".
#[test]
fn status_of_an_empty_queue_says_so() -> TestResult {
    let home = seeded(&[])?;
    let run = run_kb(home.path(), &["queue", "status"])?;
    assert!(run.ok, "queue status failed: {}", run.stderr);
    assert!(
        !run.stdout.trim().is_empty(),
        "an empty queue printed nothing"
    );
    assert!(run.stdout.contains('0'), "stdout: {}", run.stdout);
    Ok(())
}

#[test]
fn status_emits_json_when_asked() -> TestResult {
    let home = seeded(&["cc-1"])?;
    let run = run_kb(home.path(), &["queue", "status", "--json"])?;
    assert!(run.ok, "queue status --json failed: {}", run.stderr);
    let value: serde_json::Value = serde_json::from_str(&run.stdout)?;
    let data = value.get("data").unwrap_or(&value);
    assert_eq!(
        data.get("depth").and_then(serde_json::Value::as_u64),
        Some(1)
    );
    assert_eq!(
        data.get("dead_lettered")
            .and_then(serde_json::Value::as_u64),
        Some(0)
    );
    Ok(())
}

/// Draining is the worker, run once. It reports what it did in both
/// directions, because "0 ingested" and "0 dead-lettered" are different
/// answers and only the pair distinguishes a working queue from a stuck one.
#[test]
fn draining_ingests_what_is_waiting_and_says_what_it_did() -> TestResult {
    let home = seeded(&["cc-3"])?;
    let run = run_kb(home.path(), &["queue", "drain"])?;
    assert!(run.ok, "queue drain failed: {}", run.stderr);
    assert!(
        run.stdout.contains('1') && run.stdout.contains("ingested"),
        "stdout: {}",
        run.stdout
    );

    let found = run_kb(home.path(), &["get", "cc-3"])?;
    assert!(
        found.ok,
        "the drained record is not readable: {}",
        found.stderr
    );
    assert!(found.stdout.contains("the body of cc-3"));

    let after = run_kb(home.path(), &["queue", "status"])?;
    assert!(after.stdout.contains('0'), "stdout: {}", after.stdout);
    Ok(())
}

/// Dead letters are listed with their reasons, because a count nobody can
/// expand into "which sessions, and why" is a number to ignore.
#[test]
fn dead_letters_are_listed_with_their_reasons() -> TestResult {
    let home = seeded(&[])?;
    let queue = kb::ingest::Queue::open(&home.path().join(".local/share/kb/queue"))?;
    queue.enqueue(&kb::ingest::Submission {
        id: "cc-bad\nsecond line".to_owned(),
        corpus: "kb".to_owned(),
        document: "* Fine\n\nbody.\n".to_owned(),
        provenance: None,
    })?;
    let drained = run_kb(home.path(), &["queue", "drain"])?;
    assert!(drained.ok, "queue drain failed: {}", drained.stderr);

    let run = run_kb(home.path(), &["queue", "dead"])?;
    assert!(run.ok, "queue dead failed: {}", run.stderr);
    assert!(
        run.stdout.contains("cc-bad"),
        "the dead letter is not listed: {}",
        run.stdout
    );
    assert!(
        run.stdout.to_lowercase().contains("record"),
        "the reason is not shown: {}",
        run.stdout
    );
    Ok(())
}
