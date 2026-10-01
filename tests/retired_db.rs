//! The retired database is not a retrieval path (T029).
//!
//! `kb.db` remains on disk as a frozen export source, which makes the claim
//! "nothing reads it any more" checkable rather than assumed: a database
//! holding an answer the store does not hold is a fixture that fails loudly if
//! any command still consults it. The commands exercised below are the whole
//! read surface — search, get, the graph and list verbs, and the template
//! context — because a single one still reading it would resurrect the split
//! that T029 exists to close.

mod common;

use std::path::Path;
use std::process::Command;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Run `kb` with `HOME` pinned to `home`, so the default store, index and
/// database paths all resolve inside the fixture.
fn run_kb(
    home: &Path,
    stdin: Option<&str>,
    args: &[&str],
) -> Result<(bool, String), Box<dyn std::error::Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_kb"));
    command.env("HOME", home).env("XDG_CONFIG_HOME", home);
    for var in [
        "KB_EMBEDDING_BASE_URL",
        "KB_EMBEDDING_MODEL",
        "KB_EMBEDDING_API_KEY",
        "KB_RERANK_BASE_URL",
        "KB_RERANK_TOP_K",
        "KB_DB_PATH",
        "KB_INDEX_PATH",
        "KB_STORE_PATH",
    ] {
        command.env_remove(var);
    }
    command.args(args);
    let output = if let Some(text) = stdin {
        use std::io::Write as _;
        command.stdin(std::process::Stdio::piped());
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());
        let mut child = command.spawn()?;
        child
            .stdin
            .as_mut()
            .ok_or("stdin was not piped")?
            .write_all(text.as_bytes())?;
        child.wait_with_output()?
    } else {
        command.output()?
    };
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok((output.status.success(), text))
}

/// A home holding one stored record and one legacy node, both answering to
/// the same distinctive word.
fn fixture() -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let home = tempfile::tempdir()?;
    let (ok, out) = run_kb(
        home.path(),
        Some("#+filetags: :shared:\n* Stored Record\n\nthe stored answer.\n"),
        &["create", "--id", "stored"],
    )?;
    assert!(ok, "create failed: {out}");

    let legacy = home.path().join(".local/share/kb/kb.db");
    let conn = common::legacy_db(&legacy)?;
    let ghost = kb::parser::parse_document(
        "#+name: phantom\n#+filetags: :shared:\n* Ghost Record\n\nthe monosodium answer.\n",
    )?;
    common::legacy_insert(&conn, "ghost", &ghost)?;
    Ok(home)
}

/// Every read verb, against a database holding an answer the store does not.
///
/// The assertion is on the ghost's title, name and body rather than on its
/// identifier, because `kb get ghost` names the identifier back in the error
/// that says it does not exist — which is the command passing, not failing.
#[test]
fn no_read_verb_consults_the_retired_database() -> TestResult {
    let home = fixture()?;
    let checks: [&[&str]; 8] = [
        &["search", "monosodium"],
        &["get", "ghost"],
        &["recent", "--limit", "10"],
        &["orphans"],
        &["hubs"],
        &["broken"],
        &["list-by-tag", "shared"],
        &["links", "ghost"],
    ];
    // The fixture is only worth anything if the ghost really is in the
    // database, so `kb export` — the one command that reads it on purpose —
    // is asked for it first.
    let exported = home.path().join("export.jsonl");
    let (ok, out) = run_kb(
        home.path(),
        None,
        &["export", "--out", &exported.display().to_string()],
    )?;
    assert!(ok, "export failed: {out}");
    let artifact = std::fs::read_to_string(&exported)?;
    assert!(
        artifact.contains("Ghost Record"),
        "the retired database does not hold the ghost, so the checks below prove nothing"
    );

    for args in checks {
        let (_, out) = run_kb(home.path(), None, args)?;
        for trace in ["Ghost Record", "monosodium", "phantom"] {
            assert!(
                !out.contains(trace),
                "`kb {}` surfaced {trace:?}, which exists only in the retired database:\n{out}",
                args.join(" ")
            );
        }
    }

    // And the record that does exist is still found, so the checks above are
    // not passing because every command returned nothing.
    let (ok, found) = run_kb(home.path(), None, &["search", "stored answer"])?;
    assert!(ok, "search failed: {found}");
    assert!(
        found.contains("stored"),
        "the stored record is absent: {found}"
    );
    Ok(())
}

/// Deleting the database changes no answer, which is the same claim stated
/// the other way round: what is not read cannot be missed.
#[test]
fn removing_the_retired_database_changes_no_answer() -> TestResult {
    let home = fixture()?;
    let (ok, before) = run_kb(home.path(), None, &["recent", "--limit", "10"])?;
    assert!(ok, "recent failed: {before}");
    assert!(
        before.contains("stored"),
        "the stored record is absent: {before}"
    );

    std::fs::remove_file(home.path().join(".local/share/kb/kb.db"))?;
    let (ok, after) = run_kb(home.path(), None, &["recent", "--limit", "10"])?;
    assert!(ok, "recent failed without the database: {after}");
    assert_eq!(
        before, after,
        "removing the retired database changed an answer"
    );
    Ok(())
}
