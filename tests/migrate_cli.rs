//! `kb export` and `kb import` at the process boundary (CLI-002).

use std::path::Path;
use std::process::Command;

mod common;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn run_kb(
    home: &Path,
    args: &[&str],
) -> Result<(bool, String, String), Box<dyn std::error::Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_kb"))
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home)
        .env_remove("KB_EMBEDDING_BASE_URL")
        .env_remove("KB_EMBEDDING_MODEL")
        .args(args)
        .output()?;
    Ok((
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

fn legacy_db(path: &Path) -> TestResult {
    let conn = common::legacy_db(path)?;
    let doc = kb::parser::parse_document("* A note\n\nwith a body.\n")?;
    common::legacy_insert(&conn, "n1", &doc)?;
    Ok(())
}

/// The migration is run once and checked as it goes, so both commands report
/// what they moved rather than succeeding silently.
#[test]
fn export_and_import_report_what_they_moved() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");
    legacy_db(&db)?;
    let artifact = dir.path().join("export.jsonl");

    let (ok, stdout, stderr) = run_kb(
        dir.path(),
        &[
            "--db",
            &db.display().to_string(),
            "export",
            "--out",
            &artifact.display().to_string(),
        ],
    )?;
    assert!(ok, "export failed: {stderr}");
    assert!(stdout.contains("1 node"), "stdout was: {stdout}");

    let (ok, stdout, stderr) = run_kb(
        dir.path(),
        &[
            "import",
            "--in",
            &artifact.display().to_string(),
            "--store",
            &dir.path().join("store").display().to_string(),
        ],
    )?;
    assert!(ok, "import failed: {stderr}");
    assert!(stdout.contains("1 record"), "stdout was: {stdout}");
    Ok(())
}

#[test]
fn export_from_a_missing_database_fails_and_names_it() -> TestResult {
    let dir = tempfile::tempdir()?;

    let (ok, _, stderr) = run_kb(
        dir.path(),
        &[
            "--db",
            &dir.path().join("absent.db").display().to_string(),
            "export",
            "--out",
            &dir.path().join("out.jsonl").display().to_string(),
        ],
    )?;

    assert!(!ok, "a missing database must not report success");
    assert!(stderr.contains("absent.db"), "stderr was: {stderr}");
    Ok(())
}

#[test]
fn import_of_a_missing_artifact_fails_and_names_it() -> TestResult {
    let dir = tempfile::tempdir()?;

    let (ok, _, stderr) = run_kb(
        dir.path(),
        &[
            "import",
            "--in",
            &dir.path().join("absent.jsonl").display().to_string(),
            "--store",
            &dir.path().join("store").display().to_string(),
        ],
    )?;

    assert!(!ok, "a missing artifact must not report success");
    assert!(stderr.contains("absent.jsonl"), "stderr was: {stderr}");
    Ok(())
}
