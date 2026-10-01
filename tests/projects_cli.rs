//! `kb projects` at the process boundary (CLI-002,
//! `PLAN-20260923-project-identity` T006).
//!
//! Lists every project slug asserted by at least one record's provenance,
//! with a count and the newest record's date under each, plus how many
//! records carry no project at all.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::Value;

type TestResult = Result<(), Box<dyn std::error::Error>>;

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
}

/// Run `kb` in a sandboxed home, with no embedding endpoint reachable.
fn run_kb(
    home: &Path,
    stdin: Option<&str>,
    args: &[&str],
) -> Result<Run, Box<dyn std::error::Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_kb"));
    command
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home)
        .env_remove("KB_EMBEDDING_BASE_URL")
        .env_remove("KB_EMBEDDING_MODEL")
        .env_remove("KB_RERANK_BASE_URL")
        .args(args)
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn()?;
    if let Some(payload) = stdin {
        child
            .stdin
            .as_mut()
            .ok_or("no stdin on the child")?
            .write_all(payload.as_bytes())?;
    }
    let out = child.wait_with_output()?;
    Ok(Run {
        ok: out.status.success(),
        stdout: String::from_utf8(out.stdout)?,
        stderr: String::from_utf8(out.stderr)?,
    })
}

/// Two records under `alpha-project`, one under `beta-project`, and one
/// carrying no provenance at all.
fn seeded() -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let home = tempfile::tempdir()?;
    for (id, project) in [("a1", "alpha-project"), ("a2", "alpha-project")] {
        let provenance = home.path().join(format!("{id}.json"));
        std::fs::write(&provenance, format!(r#"{{"project":"{project}"}}"#))?;
        let run = run_kb(
            home.path(),
            Some("* Note\n\nbody.\n"),
            &[
                "create",
                "--id",
                id,
                "--provenance-json",
                provenance.to_str().ok_or("non-utf8 path")?,
            ],
        )?;
        assert!(run.ok, "seeding {id} failed: {}", run.stderr);
    }
    let beta_provenance = home.path().join("b1.json");
    std::fs::write(&beta_provenance, r#"{"project":"beta-project"}"#)?;
    let run = run_kb(
        home.path(),
        Some("* Note\n\nbody.\n"),
        &[
            "create",
            "--id",
            "b1",
            "--provenance-json",
            beta_provenance.to_str().ok_or("non-utf8 path")?,
        ],
    )?;
    assert!(run.ok, "seeding b1 failed: {}", run.stderr);

    let run = run_kb(
        home.path(),
        Some("* Note\n\nbody.\n"),
        &["create", "--id", "no-project"],
    )?;
    assert!(run.ok, "seeding no-project failed: {}", run.stderr);
    Ok(home)
}

#[test]
fn projects_json_lists_slugs_with_counts_and_a_no_project_total() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["projects", "--json"])?;
    assert!(run.ok, "projects failed: {}", run.stderr);
    let payload: Value = serde_json::from_str(&run.stdout)?;
    let projects = payload
        .get("data")
        .and_then(|d| d.get("projects"))
        .and_then(Value::as_array)
        .ok_or_else(|| format!("no projects array: {}", run.stdout))?;

    let count_for = |slug: &str| -> Option<u64> {
        projects
            .iter()
            .find(|row| row.get("project").and_then(Value::as_str) == Some(slug))
            .and_then(|row| row.get("count"))
            .and_then(Value::as_u64)
    };
    assert_eq!(count_for("alpha-project"), Some(2), "{projects:?}");
    assert_eq!(count_for("beta-project"), Some(1), "{projects:?}");
    for row in projects {
        assert!(
            row.get("newest").and_then(Value::as_str).is_some(),
            "row missing newest date: {row}"
        );
    }

    let no_project = payload
        .get("data")
        .and_then(|d| d.get("noProject"))
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("no noProject count: {}", run.stdout))?;
    assert_eq!(no_project, 1, "expected exactly one record with no project");
    Ok(())
}

#[test]
fn projects_text_names_each_slug_and_the_no_project_line() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["projects"])?;
    assert!(run.ok, "projects failed: {}", run.stderr);
    assert!(
        run.stdout.contains("alpha-project"),
        "text output missing alpha-project: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("beta-project"),
        "text output missing beta-project: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("no project"),
        "text output missing the no-project line: {}",
        run.stdout
    );
    Ok(())
}

#[test]
fn projects_on_an_empty_index_reports_zero_of_everything() -> TestResult {
    let home = tempfile::tempdir()?;
    let run = run_kb(home.path(), None, &["projects", "--json"])?;
    assert!(run.ok, "projects failed: {}", run.stderr);
    let payload: Value = serde_json::from_str(&run.stdout)?;
    let projects = payload
        .get("data")
        .and_then(|d| d.get("projects"))
        .and_then(Value::as_array)
        .ok_or_else(|| format!("no projects array: {}", run.stdout))?;
    assert!(projects.is_empty(), "{projects:?}");
    let no_project = payload
        .get("data")
        .and_then(|d| d.get("noProject"))
        .and_then(Value::as_u64);
    assert_eq!(no_project, Some(0));
    Ok(())
}
