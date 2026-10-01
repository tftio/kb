//! CLI tests for `kb meta agent emit-hooks`.
//!
//! kb declares its `SessionEnd` hook on `AGENT_SURFACE` by embedding
//! `scripts/session-end-kb.sh` with `include_str!`, so the file this
//! repository already lints and tests is the exact bytes the binary emits.
//! These tests pin that byte-identity end to end -- through the real
//! subprocess, not the library function directly -- and pin the target-aware
//! behavior documented on `tftio_lib::agent_hook`: Claude gets a `SessionEnd`
//! registration for the script it wrote, while Codex, which has no
//! `SessionEnd` equivalent, writes nothing and prints `{}`.
//!
//! `emit-skills` behavior is untouched by this task; its own tests are not
//! duplicated here.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

type TestResult = Result<(), Box<dyn std::error::Error>>;

struct Run {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

fn run_kb(
    home: &Path,
    stdin: Option<&str>,
    args: &[&str],
) -> Result<Run, Box<dyn std::error::Error>> {
    let bin = env!("CARGO_BIN_EXE_kb");
    let mut cmd = Command::new(bin);
    cmd.env("HOME", home).env("XDG_CONFIG_HOME", home);
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

fn session_end_script_source() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/session-end-kb.sh")
}

/// `emit-hooks --target claude` writes the embedded script byte-for-byte,
/// marks it executable, and prints a `SessionEnd` registration fragment
/// whose command points at that exact file.
#[test]
fn emit_hooks_claude_writes_byte_identical_executable_script_and_registration() -> TestResult {
    let home = tempfile::tempdir()?;
    let out = tempfile::tempdir()?;

    let run = run_kb(
        home.path(),
        None,
        &[
            "meta",
            "agent",
            "emit-hooks",
            "--target",
            "claude",
            "--out",
            out.path().to_str().ok_or("non-utf8 out path")?,
        ],
    )?;
    assert!(
        run.status.success(),
        "stdout={} stderr={}",
        run.stdout,
        run.stderr
    );

    let written = out.path().join("session-end-kb.sh");
    let written_bytes = fs::read(&written)?;
    let source_bytes = fs::read(session_end_script_source())?;
    assert_eq!(
        written_bytes, source_bytes,
        "emitted hook must be byte-identical to scripts/session-end-kb.sh"
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = fs::metadata(&written)?.permissions().mode();
        assert_ne!(mode & 0o111, 0, "emitted hook must be executable");
    }

    // stdout is the write log ("wrote <path>") followed by the JSON
    // registration fragment on its own trailing line; parse only that line.
    let json_line = run
        .stdout
        .lines()
        .next_back()
        .ok_or("emit-hooks printed no output")?;
    let fragment: serde_json::Value = serde_json::from_str(json_line)?;

    let command = fragment
        .pointer("/hooks/SessionEnd/0/hooks/0/command")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("missing SessionEnd command in fragment: {fragment}"))?;
    assert_eq!(command, written.display().to_string());

    let timeout = fragment
        .pointer("/hooks/SessionEnd/0/hooks/0/timeout")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("missing SessionEnd timeout in fragment: {fragment}"))?;
    assert_eq!(timeout, 30);

    let status_message = fragment
        .pointer("/hooks/SessionEnd/0/hooks/0/statusMessage")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("missing SessionEnd statusMessage in fragment: {fragment}"))?;
    assert_eq!(status_message, "Distilling session into kb");

    Ok(())
}

/// Codex has no `SessionEnd`-equivalent lifecycle event, so `emit-hooks
/// --target codex` must write nothing and print the empty fragment `{}`
/// rather than invent a key Codex never reads.
#[test]
fn emit_hooks_codex_writes_nothing_and_prints_empty_fragment() -> TestResult {
    let home = tempfile::tempdir()?;
    let out = tempfile::tempdir()?;

    let run = run_kb(
        home.path(),
        None,
        &[
            "meta",
            "agent",
            "emit-hooks",
            "--target",
            "codex",
            "--out",
            out.path().to_str().ok_or("non-utf8 out path")?,
        ],
    )?;
    assert!(
        run.status.success(),
        "stdout={} stderr={}",
        run.stdout,
        run.stderr
    );
    assert_eq!(run.stdout.trim(), "{}");
    assert!(
        fs::read_dir(out.path())?.next().is_none(),
        "codex has no SessionEnd equivalent; nothing should be written"
    );

    Ok(())
}
