//! CLI integration tests for `kb prompt {render,list,show}`.
//!
//! `kb prompt` reads `$XDG_CONFIG_HOME`/`$HOME/.config/kb/prompts` for
//! user template overrides, so every spawned process pins `HOME`,
//! `XDG_CONFIG_HOME`, and `--db` to the per-test temp directory. Without
//! that sandbox the test would read the developer's real config and would
//! not reproduce on a clean CI runner with an empty `$HOME`.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::Value;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Run `kb --db <db> <args...>` with the given stdin payload, returning
/// (status, stdout, stderr). `HOME`/`XDG_CONFIG_HOME` are pinned to the
/// DB's parent temp dir so no real user prompt-template config is read.
fn run_kb(
    db: &Path,
    stdin: Option<&str>,
    args: &[&str],
) -> Result<(std::process::ExitStatus, String, String), Box<dyn std::error::Error>> {
    let bin = env!("CARGO_BIN_EXE_kb");
    let home = db
        .parent()
        .ok_or("db path has no parent directory for HOME sandbox")?;
    let mut cmd = Command::new(bin);
    cmd.env("HOME", home)
        .env("XDG_CONFIG_HOME", home)
        .arg("--db")
        .arg(db);
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
    Ok((
        out.status,
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    ))
}

fn extract_data(stdout: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let v: Value = serde_json::from_str(stdout.trim())?;
    v.get("data")
        .cloned()
        .ok_or_else(|| "envelope.data missing".into())
}

/// Anchors criterion 2d375ecc: `kb prompt list` enumerates built-in
/// templates and surfaces the cold-start-audit asset.
#[test]
fn prompt_list_verb() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");

    let (st, out, err) = run_kb(&db, None, &["prompt", "list", "--json"])?;
    assert!(st.success(), "prompt list failed: {err}\nstdout: {out}");
    let data = extract_data(&out)?;
    let arr = data.as_array().ok_or("prompt list: expected array")?;
    assert!(
        arr.iter().any(|t| t
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(|n| n == "cold-start-audit")
            && t.get("source").and_then(Value::as_str) == Some("builtin")),
        "expected cold-start-audit builtin in list: {arr:?}"
    );

    // Plain-text mode also works.
    let (st, out, err) = run_kb(&db, None, &["prompt", "list"])?;
    assert!(st.success(), "prompt list (text) failed: {err}");
    assert!(
        out.contains("cold-start-audit"),
        "text output should mention cold-start-audit: {out}"
    );
    Ok(())
}

/// Anchors criterion a84fcfab: `kb prompt show <name>` prints the
/// template body, both as JSON envelope and plain text.
#[test]
fn prompt_show_verb() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");

    let (st, out, err) = run_kb(&db, None, &["prompt", "show", "cold-start-audit", "--json"])?;
    assert!(st.success(), "prompt show failed: {err}\nstdout: {out}");
    let data = extract_data(&out)?;
    assert_eq!(
        data.get("name").and_then(Value::as_str),
        Some("cold-start-audit")
    );
    assert_eq!(data.get("source").and_then(Value::as_str), Some("builtin"));
    let body = data
        .get("body")
        .and_then(Value::as_str)
        .ok_or("prompt show: body string missing")?;
    assert!(body.contains("Cold-Start Audit"), "body shape: {body}");

    // Missing template emits a non-zero exit and a JSON error envelope.
    let (st, out, _err) = run_kb(&db, None, &["prompt", "show", "no-such-template", "--json"])?;
    assert!(!st.success(), "expected failure on missing template");
    let v: Value = serde_json::from_str(out.trim())?;
    assert_eq!(v.get("ok").and_then(Value::as_bool), Some(false));
    Ok(())
}
