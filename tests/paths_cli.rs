//! The CLI verbs that take no `--store`/`--index` argument still follow
//! `KB_STORE_PATH` and `KB_INDEX_PATH`.
//!
//! The failure this guards against: `kb-mcp --http` honoured the variables
//! and served a record that `kb search` in the same environment could not
//! see, because `search`, `get`, `create` and their siblings opened the index under
//! `$HOME` regardless. A deployment points every process at one place with
//! one setting; this test is what makes that a property rather than a habit.
//!
//! Every spawned `kb` is sandboxed: `HOME` and `XDG_CONFIG_HOME` are pinned
//! to a per-test temp directory and the live embedding and reranking
//! endpoints are removed from its environment.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Run `kb <args...>` under `home`, with the store and index variables set
/// to `paths` when given, returning (status, stdout, stderr).
fn run_kb(
    home: &Path,
    paths: Option<(&Path, &Path)>,
    stdin: Option<&str>,
    args: &[&str],
) -> Result<(std::process::ExitStatus, String, String), Box<dyn std::error::Error>> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_kb"));
    cmd.env("HOME", home)
        .env("XDG_CONFIG_HOME", home)
        .arg("--db")
        .arg(home.join("kb.db"));
    for var in [
        "KB_EMBEDDING_BASE_URL",
        "KB_EMBEDDING_MODEL",
        "KB_EMBEDDING_API_KEY",
        "KB_RERANK_BASE_URL",
        "KB_RERANK_TOP_K",
        "KB_STORE_PATH",
        "KB_INDEX_PATH",
    ] {
        cmd.env_remove(var);
    }
    if let Some((store, index)) = paths {
        cmd.env("KB_STORE_PATH", store).env("KB_INDEX_PATH", index);
    }
    cmd.args(args);
    cmd.stdin(if stdin.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    })
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    if let Some(text) = stdin {
        child
            .stdin
            .take()
            .ok_or("stdin not piped")?
            .write_all(text.as_bytes())?;
    }
    let output = child.wait_with_output()?;
    Ok((
        output.status,
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

#[test]
fn the_verbs_without_path_arguments_follow_the_environment() -> TestResult {
    let home = tempfile::tempdir()?;
    let elsewhere = tempfile::tempdir()?;
    let store = elsewhere.path().join("store");
    let index = elsewhere.path().join("index.db");
    let paths = Some((store.as_path(), index.as_path()));

    let (status, _, stderr) = run_kb(
        home.path(),
        paths,
        Some("* Configured paths\n\nA record written where the environment says.\n"),
        &["create", "--id", "p1", "--json"],
    )?;
    assert!(status.success(), "create failed: {stderr}");

    // The record landed at the configured paths, not under HOME.
    assert!(store.join("objects").is_dir(), "no store at KB_STORE_PATH");
    assert!(index.is_file(), "no index at KB_INDEX_PATH");
    assert!(
        !home.path().join(".local/share/kb/index.db").exists(),
        "an index was created under HOME despite KB_INDEX_PATH"
    );

    // Reading follows the same variables...
    let (status, stdout, stderr) = run_kb(home.path(), paths, None, &["get", "p1"])?;
    assert!(status.success(), "get with the variables failed: {stderr}");
    assert!(
        stdout.contains("Configured paths"),
        "get returned: {stdout}"
    );

    let (status, stdout, _) = run_kb(
        home.path(),
        paths,
        None,
        &["search", "--json", "configured"],
    )?;
    assert!(status.success());
    assert!(
        stdout.contains("\"p1\""),
        "search did not find the record: {stdout}"
    );

    // ...and without them the same process sees an empty corpus under HOME.
    let (status, _, stderr) = run_kb(home.path(), None, None, &["get", "p1"])?;
    assert!(!status.success(), "get found a record under HOME: {stderr}");
    Ok(())
}

#[test]
fn an_empty_variable_means_the_default() -> TestResult {
    let home = tempfile::tempdir()?;
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_kb"));
    let output = cmd
        .env("HOME", home.path())
        .env("XDG_CONFIG_HOME", home.path())
        .env("KB_STORE_PATH", "")
        .env("KB_INDEX_PATH", "")
        .env_remove("KB_EMBEDDING_BASE_URL")
        .env_remove("KB_RERANK_BASE_URL")
        .arg("--db")
        .arg(home.path().join("kb.db"))
        .args(["recent", "--json"])
        .output()?;
    assert!(
        output.status.success(),
        "recent failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        home.path().join(".local/share/kb/index.db").is_file(),
        "an empty KB_INDEX_PATH did not fall back to the default under HOME"
    );
    Ok(())
}
