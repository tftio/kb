//! `kb mail classify` at the process boundary (CLI-002).

use std::path::Path;
use std::process::Command;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn maildir(root: &Path) -> TestResult {
    for folder in ["Inbox", "Archive"] {
        std::fs::create_dir_all(root.join(folder).join("cur"))?;
    }
    std::fs::write(
        root.join("Archive/cur/1.,U=1:2,S"),
        "From: Ada <ada@example.com>\nTo: reader@example.invalid\n\
         Message-ID: <human@example.com>\nSubject: the schema\n\nShall we cut it?\n",
    )?;
    std::fs::write(
        root.join("Archive/cur/2.,U=2:2,S"),
        "From: news@example.com\nTo: reader@example.invalid\n\
         Message-ID: <bulk@example.com>\nList-Id: <announce.example.com>\n\nNewsletter.\n",
    )?;
    Ok(())
}

fn run(args: &[&str]) -> Result<(bool, String, String), Box<dyn std::error::Error>> {
    let out = Command::new(env!("CARGO_BIN_EXE_kb")).args(args).output()?;
    Ok((
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    ))
}

/// One line per message, carrying what a sampling pass needs to judge the
/// rule: the identifier, the verdict, and which signal fired.
#[test]
fn classify_emits_a_line_per_message() -> TestResult {
    let dir = tempfile::tempdir()?;
    maildir(dir.path())?;

    let (ok, stdout, stderr) = run(&[
        "mail",
        "classify",
        "--maildir",
        &dir.path().display().to_string(),
    ])?;

    assert!(ok, "classify failed: {stderr}");
    let lines = stdout.lines().filter(|l| !l.trim().is_empty()).count();
    assert_eq!(lines, 2, "stdout was: {stdout}");
    assert!(stdout.contains("<human@example.com>"), "{stdout}");
    assert!(
        stdout.contains("\"classification\":\"non-bulk\""),
        "{stdout}"
    );
    assert!(stdout.contains("\"classification\":\"bulk\""), "{stdout}");
    assert!(stdout.contains("\"signal\":\"list-id\""), "{stdout}");
    Ok(())
}

/// The identifier is the Message-ID, never the filename: Maildir flags
/// rewrite filenames on every flag change, so a filename cannot name a
/// message across two syncs.
#[test]
fn a_message_is_identified_by_its_message_id() -> TestResult {
    let dir = tempfile::tempdir()?;
    maildir(dir.path())?;

    let (_, stdout, _) = run(&[
        "mail",
        "classify",
        "--maildir",
        &dir.path().display().to_string(),
    ])?;

    assert!(
        !stdout.contains(",U=1:2,S"),
        "a filename leaked in: {stdout}"
    );
    Ok(())
}

/// A Maildir that is not there is an operator error, reported as one.
#[test]
fn a_missing_maildir_is_refused() -> TestResult {
    let dir = tempfile::tempdir()?;

    let (ok, _, stderr) = run(&[
        "mail",
        "classify",
        "--maildir",
        &dir.path().join("absent").display().to_string(),
    ])?;

    assert!(!ok, "a missing maildir must not report success");
    assert!(stderr.contains("absent"), "stderr was: {stderr}");
    Ok(())
}
