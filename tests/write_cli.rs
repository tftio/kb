//! CLI write-semantics tests for `kb create` and `kb update`.
//!
//! `update` replaces the whole document with whatever arrives on stdin, so a
//! caller that supplies nothing must be refused rather than silently emptying
//! the node. These tests drive the real `kb` binary with stdin closed, with an
//! explicit `--allow-empty`, and through the `kb get | kb update` round-trip
//! that is the supported way to retag a node without losing its body.
//!
//! The suite also pins where `--tag` values land in the stored document, which
//! is one ordered rule shared with `kb tags add`: an existing `#+filetags:`
//! line, else the first heading's tag list, else a new `#+filetags:` line. The
//! last branch is the repair for an empty level-1 heading being invented as a
//! tag carrier.
//!
//! The same harness pins the rest of the write path's refusal contracts: a
//! `create` whose `--id` already exists, an `update` whose document drawer
//! names a different node, and a `--markdown` import with `pandoc` off `PATH`.
//! Each of those already behaved correctly before these tests existed; the
//! tests convert accidental behavior into an asserted contract. The
//! `session-end-kb` hook depends directly on the first of them --
//! its `kb create --id <session-id> || kb update <session-id>` deduplication
//! is correct only because the conflicting create exits non-zero.
//!
//! Every spawned `kb` process is sandboxed: `HOME`, `XDG_CONFIG_HOME`, and
//! `--db` are pinned to the per-test temp directory so the result is identical
//! on a clean CI runner with an empty `$HOME`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Outcome of one `kb` invocation.
struct Run {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

/// Run `kb --db <db> <args...>`. `stdin` of `None` closes the child's stdin
/// (the `</dev/null` case); `Some(payload)` pipes the payload in.
fn run_kb(
    db: &Path,
    stdin: Option<&str>,
    args: &[&str],
) -> Result<Run, Box<dyn std::error::Error>> {
    run_kb_with_path(db, stdin, args, None)
}

/// As [`run_kb`], but with `PATH` overridden for this invocation when `path`
/// is `Some`. Used to manufacture a missing `pandoc` on a machine that has one.
/// The binary itself is reached through `CARGO_BIN_EXE_kb`, an absolute path,
/// so emptying `PATH` does not prevent the child from starting.
fn run_kb_with_path(
    db: &Path,
    stdin: Option<&str>,
    args: &[&str],
    path: Option<&Path>,
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
    if let Some(p) = path {
        cmd.env("PATH", p);
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

/// A fresh database path inside a per-test temp directory.
fn temp_db(name: &str) -> Result<(tempfile::TempDir, PathBuf), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join(format!("{name}.db"));
    Ok((dir, db))
}

const BODY: &str = "#+title: Original\n#+filetags: :alpha:\n\nreal body content\n";

/// Seed a node with a title, one filetag, and a body paragraph.
fn seed(db: &Path, id: &str) -> TestResult {
    let run = run_kb(db, Some(BODY), &["create", "--id", id])?;
    assert!(run.status.success(), "seed create failed: {}", run.stderr);
    Ok(())
}

#[test]
fn update_without_stdin_is_refused_and_leaves_the_node_intact() -> TestResult {
    let (_dir, db) = temp_db("refuse")?;
    seed(&db, "n1")?;

    let run = run_kb(&db, None, &["update", "n1", "--tag", "beta"])?;
    assert!(
        !run.status.success(),
        "update with no stdin should fail, got stdout: {}",
        run.stdout
    );
    let msg = format!("{}{}", run.stdout, run.stderr);
    assert!(
        msg.contains("empty document"),
        "error should name the empty body, got: {msg}"
    );
    assert!(
        msg.contains("kb get <id> | kb update <id>"),
        "error should point at the round-trip form, got: {msg}"
    );

    let after = run_kb(&db, Some(""), &["get", "n1", "--json"])?;
    assert!(after.status.success(), "get failed: {}", after.stderr);
    assert!(
        after.stdout.contains("Original"),
        "title should survive the refused update, got: {}",
        after.stdout
    );
    assert!(
        after.stdout.contains("real body content"),
        "body should survive the refused update, got: {}",
        after.stdout
    );
    assert!(
        !after.stdout.contains("beta"),
        "the refused tag must not be applied, got: {}",
        after.stdout
    );
    Ok(())
}

#[test]
fn update_with_whitespace_only_stdin_is_refused() -> TestResult {
    let (_dir, db) = temp_db("blank")?;
    seed(&db, "n1")?;

    let run = run_kb(&db, Some("   \n\t\n"), &["update", "n1", "--tag", "beta"])?;
    assert!(
        !run.status.success(),
        "whitespace-only stdin should fail, got stdout: {}",
        run.stdout
    );

    let after = run_kb(&db, Some(""), &["get", "n1", "--json"])?;
    assert!(
        after.stdout.contains("real body content"),
        "body should survive, got: {}",
        after.stdout
    );
    Ok(())
}

#[test]
fn update_with_allow_empty_clears_the_node() -> TestResult {
    let (_dir, db) = temp_db("allow")?;
    seed(&db, "n1")?;

    let run = run_kb(
        &db,
        None,
        &["update", "n1", "--allow-empty", "--tag", "beta"],
    )?;
    assert!(
        run.status.success(),
        "--allow-empty should succeed: {}{}",
        run.stdout,
        run.stderr
    );

    let after = run_kb(&db, Some(""), &["get", "n1", "--json"])?;
    assert!(
        !after.stdout.contains("real body content"),
        "body should be gone after an explicit --allow-empty, got: {}",
        after.stdout
    );
    assert!(
        after.stdout.contains("beta"),
        "the CLI tag should be applied, got: {}",
        after.stdout
    );
    Ok(())
}

#[test]
fn get_piped_into_update_preserves_the_body_and_merges_tags() -> TestResult {
    let (_dir, db) = temp_db("roundtrip")?;
    seed(&db, "n1")?;

    let got = run_kb(&db, Some(""), &["get", "n1"])?;
    assert!(got.status.success(), "get failed: {}", got.stderr);

    let run = run_kb(&db, Some(&got.stdout), &["update", "n1", "--tag", "gamma"])?;
    assert!(
        run.status.success(),
        "round-trip update failed: {}{}",
        run.stdout,
        run.stderr
    );

    let after = run_kb(&db, Some(""), &["get", "n1", "--json"])?;
    assert!(
        after.stdout.contains("real body content"),
        "round-trip must preserve the body, got: {}",
        after.stdout
    );
    assert!(
        after.stdout.contains("alpha") && after.stdout.contains("gamma"),
        "round-trip must merge the existing and new tags, got: {}",
        after.stdout
    );
    Ok(())
}

#[test]
fn create_without_stdin_is_refused() -> TestResult {
    let (_dir, db) = temp_db("create")?;

    let run = run_kb(&db, None, &["create", "--id", "n1", "--tag", "alpha"])?;
    assert!(
        !run.status.success(),
        "create with no stdin should fail, got stdout: {}",
        run.stdout
    );
    let msg = format!("{}{}", run.stdout, run.stderr);
    assert!(
        msg.contains("kb create"),
        "create error should carry the create hint, got: {msg}"
    );

    let after = run_kb(&db, Some(""), &["get", "n1", "--json"])?;
    assert!(
        !after.status.success(),
        "the refused create must not have stored a node, got: {}",
        after.stdout
    );
    Ok(())
}

#[test]
fn create_with_allow_empty_stores_an_empty_node() -> TestResult {
    let (_dir, db) = temp_db("create-allow")?;

    let run = run_kb(&db, None, &["create", "--id", "n1", "--allow-empty"])?;
    assert!(
        run.status.success(),
        "--allow-empty create should succeed: {}{}",
        run.stdout,
        run.stderr
    );

    let after = run_kb(&db, Some(""), &["get", "n1", "--json"])?;
    assert!(after.status.success(), "get failed: {}", after.stderr);
    Ok(())
}

/// A `create` whose `--id` is already taken must be refused outright, leaving
/// the existing node exactly as it was.
///
/// This is the contract `scripts/session-end-kb.sh` builds its
/// deduplication on: it runs `kb create --id <session-id>` and falls back to
/// `kb update <session-id>` on failure, so a resumed session refreshes its own
/// node instead of minting a second one. That fallback is correct only while
/// the conflicting create exits non-zero *and* writes nothing.
///
/// The message was originally the raw driver text `UNIQUE constraint failed:
/// nodes.id`, which named the column rather than the offending id and leaked
/// `SQLite` internals into an agent-facing surface. This test was written with a
/// deliberately loose `stderr` assertion so that repairing it would not
/// register as a test failure. The repair has landed, so the
/// assertion is now specific: the message names the conflicting id and the
/// command that would have worked.
///
/// What was pinned all along, and still is, is the behavior the hook depends on:
/// refusal, and an untouched node.
#[test]
fn create_with_a_conflicting_id_is_refused_and_leaves_the_existing_node_intact() -> TestResult {
    let (_dir, db) = temp_db("conflict")?;
    seed(&db, "n1")?;

    let run = run_kb(
        &db,
        Some("#+title: Replacement\n\ndifferent body\n"),
        &["create", "--id", "n1"],
    )?;
    assert!(
        !run.status.success(),
        "create onto an existing id should fail, got stdout: {}",
        run.stdout
    );
    assert!(
        run.stderr.contains("n1"),
        "the refusal must name the conflicting id, got: {}",
        run.stderr
    );
    assert!(
        run.stderr.contains("kb update n1"),
        "the refusal must point at the command that would have worked, got: {}",
        run.stderr
    );
    assert!(
        !run.stderr.contains("nodes.id"),
        "the physical schema must not reach an agent-facing surface, got: {}",
        run.stderr
    );

    let after = run_kb(&db, Some(""), &["get", "n1", "--json"])?;
    assert!(after.status.success(), "get failed: {}", after.stderr);
    assert!(
        after.stdout.contains("real body content"),
        "the original body must survive the refused create, got: {}",
        after.stdout
    );
    assert!(
        !after.stdout.contains("different body"),
        "the refused create must not have written its payload, got: {}",
        after.stdout
    );
    Ok(())
}

/// `update` accepts a document carrying a `:PROPERTIES:` drawer, but the
/// drawer's `:ID:` must name the node being updated. A mismatch is a caller
/// error -- most likely a document pasted from the wrong node -- and must be
/// refused rather than silently retargeted or silently stripped.
///
/// The HTTP half of this contract is already asserted in `src/api.rs` by
/// `api_orgtext_write_put_400_on_id_mismatch`; this closes the CLI half.
#[test]
fn update_with_a_mismatched_drawer_id_is_refused() -> TestResult {
    let (_dir, db) = temp_db("mismatch")?;
    seed(&db, "n1")?;

    let foreign = ":PROPERTIES:\n:ID: some-other-node\n:END:\n* Replacement\n\ndifferent body\n";
    let run = run_kb(&db, Some(foreign), &["update", "n1"])?;
    assert!(
        !run.status.success(),
        "a drawer :ID: naming another node should fail, got stdout: {}",
        run.stdout
    );
    let msg = format!("{}{}", run.stdout, run.stderr);
    assert!(
        msg.contains("some-other-node") && msg.contains("n1"),
        "the error should name both the drawer id and the target id, got: {msg}"
    );

    let after = run_kb(&db, Some(""), &["get", "n1", "--json"])?;
    assert!(
        after.stdout.contains("real body content"),
        "the original body must survive the refused update, got: {}",
        after.stdout
    );
    assert!(
        !after.stdout.contains("different body"),
        "the refused update must not have written its payload, got: {}",
        after.stdout
    );
    Ok(())
}

/// `--markdown` shells out to `pandoc`. When the binary is absent the failure
/// must be the documented `PandocNotFound` setup error -- a clean non-zero exit
/// naming pandoc -- and not a panic, and nothing may be stored.
///
/// pandoc is installed on the development machine, so its absence is
/// manufactured by giving this one invocation an empty `PATH`. `kb` itself is
/// reached by absolute path through `CARGO_BIN_EXE_kb`, so it still starts.
#[test]
fn create_markdown_without_pandoc_fails_cleanly() -> TestResult {
    let (dir, db) = temp_db("nopandoc")?;
    let empty_path = dir.path().join("empty-bin");
    std::fs::create_dir(&empty_path)?;

    let run = run_kb_with_path(
        &db,
        Some("# Heading\n\nsome markdown\n"),
        &["create", "--id", "n1"],
        Some(&empty_path),
    )?;
    assert!(
        run.status.success(),
        "the control create without --markdown should not need pandoc: {}",
        run.stderr
    );

    let run = run_kb_with_path(
        &db,
        Some("# Heading\n\nsome markdown\n"),
        &["create", "--id", "n2", "--markdown"],
        Some(&empty_path),
    )?;
    assert!(
        !run.status.success(),
        "--markdown without pandoc should fail, got stdout: {}",
        run.stdout
    );
    let msg = format!("{}{}", run.stdout, run.stderr);
    assert!(
        msg.contains("pandoc"),
        "the error should name pandoc, got: {msg}"
    );
    assert!(
        !msg.contains("panicked"),
        "a missing pandoc is a setup error, not a panic, got: {msg}"
    );

    let after = run_kb(&db, Some(""), &["get", "n2", "--json"])?;
    assert!(
        !after.status.success(),
        "the failed markdown import must not have stored a node, got: {}",
        after.stdout
    );
    Ok(())
}

/// A `--tag` supplied on the command line merges into the tag list of the
/// document's leading heading, leaving the heading's text alone.
#[test]
fn cli_tag_merges_into_an_existing_heading() -> TestResult {
    let (_dir, db) = temp_db("tag-heading")?;

    let run = run_kb(
        &db,
        Some("* Seed :alpha:\nbody text\n"),
        &["create", "--id", "n1", "--tag", "beta"],
    )?;
    assert!(run.status.success(), "create failed: {}", run.stderr);

    let after = run_kb(&db, Some(""), &["get", "n1"])?;
    assert!(
        after.stdout.contains("* Seed :alpha:beta:"),
        "the CLI tag should join the heading's tag list, got: {}",
        after.stdout
    );
    assert!(
        after.stdout.contains("body text"),
        "the body should be untouched, got: {}",
        after.stdout
    );
    Ok(())
}

/// A `--tag` supplied against a body with no heading is carried by a
/// `#+filetags:` line, org's document-level tag mechanism, which needs no
/// heading.
///
/// **This test previously asserted the opposite.** Until the defect was
/// fixed, `--tag` prepended an empty level-1 heading purely as a tag carrier,
/// storing `*  :beta:` ahead of the prose, and this test pinned that output
/// verbatim under a comment stating it documented a defect it did not endorse.
/// The pin existed so the repair could not silently change what the CLI
/// stores; it did its job, failing here and nowhere else when the placement
/// rule changed. It is rewritten rather than deleted so the defect's history
/// stays legible.
#[test]
fn cli_tag_on_a_headingless_body_is_carried_by_filetags_see_kb_5() -> TestResult {
    let (_dir, db) = temp_db("tag-headingless")?;

    let run = run_kb(
        &db,
        Some("just prose, no heading\n"),
        &["create", "--id", "n1", "--tag", "beta"],
    )?;
    assert!(run.status.success(), "create failed: {}", run.stderr);

    let after = run_kb(&db, Some(""), &["get", "n1"])?;
    assert!(
        after.stdout.contains("#+filetags: :beta:"),
        "expected a filetags line carrying the tag, got: {}",
        after.stdout
    );
    assert!(
        !after.stdout.contains('*'),
        "no heading may be invented to carry a tag (kb#5), got: {}",
        after.stdout
    );
    assert!(
        after.stdout.contains("just prose, no heading"),
        "the body should be untouched, got: {}",
        after.stdout
    );
    Ok(())
}

/// When the document already carries a `#+filetags:` line, that is where a
/// `--tag` value lands -- even though a heading is also present. The rule is
/// ordered, and a document-level flag belongs on the document-level carrier.
#[test]
fn cli_tag_prefers_an_existing_filetags_line_over_a_heading() -> TestResult {
    let (_dir, db) = temp_db("tag-filetags-first")?;

    let run = run_kb(
        &db,
        Some("#+filetags: :alpha:\n\n* Head :gamma:\nbody\n"),
        &["create", "--id", "n1", "--tag", "beta"],
    )?;
    assert!(run.status.success(), "create failed: {}", run.stderr);

    let after = run_kb(&db, Some(""), &["get", "n1"])?;
    assert!(
        after.stdout.contains("#+filetags: :alpha:beta:"),
        "expected the filetags line extended, got: {}",
        after.stdout
    );
    assert!(
        after.stdout.contains("* Head :gamma:"),
        "the heading's own tags should be untouched, got: {}",
        after.stdout
    );
    Ok(())
}

/// A `--tag` value is normalized before it enters the document, not only on
/// the way to the tag index. The destination can be a colon-delimited
/// `#+filetags:` value, where a raw spelling containing a colon would store as
/// two tokens and one containing a space would store a spelling `kb tags`
/// never shows.
#[test]
fn cli_tag_is_stored_normalized() -> TestResult {
    let (_dir, db) = temp_db("tag-normalized")?;

    let run = run_kb(
        &db,
        Some("just prose, no heading\n"),
        &["create", "--id", "n1", "--tag", "Silent Critic"],
    )?;
    assert!(run.status.success(), "create failed: {}", run.stderr);

    let after = run_kb(&db, Some(""), &["get", "n1"])?;
    assert!(
        after.stdout.contains("#+filetags: :silent-critic:"),
        "expected the normalized spelling in the document, got: {}",
        after.stdout
    );

    let listed = run_kb(&db, Some(""), &["list-by-tag", "silent-critic"])?;
    assert!(
        listed.stdout.contains("n1"),
        "the tag index should agree with the document, got: {}",
        listed.stdout
    );
    Ok(())
}

/// `kb create --provenance-json <file>` stores the file's provenance in the
/// record header, and `kb get` reports it back in both text and `--json`
/// output. This is the local (non-queue) write path's way to carry
/// provenance; T007's importer is expected to write a temp file and pass its
/// path here.
#[test]
fn create_with_provenance_json_is_readable_from_get() -> TestResult {
    let (dir, db) = temp_db("provenance")?;
    let provenance_path = dir.path().join("provenance.json");
    std::fs::write(
        &provenance_path,
        r#"{"project":"kb","project_source":"declared","remote":"github.com/tftio/kb",
            "context":"personal","domains":["clanker","silent-critic"],
            "harness":"claude-code","model":"claude-sonnet-5","session":"s-1",
            "cwd":"/Users/op/Projects/kb/main"}"#,
    )?;

    let run = run_kb(
        &db,
        Some("* A note\n\nbody.\n"),
        &[
            "create",
            "--id",
            "n1",
            "--provenance-json",
            provenance_path
                .to_str()
                .ok_or("provenance path is not UTF-8")?,
        ],
    )?;
    assert!(run.status.success(), "create failed: {}", run.stderr);

    let text = run_kb(&db, Some(""), &["get", "n1"])?;
    assert!(text.status.success(), "get failed: {}", text.stderr);
    assert!(
        text.stdout.contains("project: kb"),
        "text output missing provenance: {}",
        text.stdout
    );
    assert!(text.stdout.contains("harness: claude-code"));
    assert!(text.stdout.contains("domains: clanker, silent-critic"));

    let json = run_kb(&db, Some(""), &["get", "n1", "--json"])?;
    assert!(json.status.success(), "get --json failed: {}", json.stderr);
    let value: serde_json::Value = serde_json::from_str(&json.stdout)?;
    let provenance = value
        .get("data")
        .and_then(|d| d.get("provenance"))
        .ok_or("no provenance in the get response")?;
    let field = |key: &str| -> Result<&str, Box<dyn std::error::Error>> {
        provenance
            .get(key)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("provenance.{key} missing or not a string: {provenance}").into())
    };
    assert_eq!(field("project")?, "kb");
    assert_eq!(field("projectSource")?, "declared");
    assert_eq!(field("remote")?, "github.com/tftio/kb");
    assert_eq!(field("context")?, "personal");
    assert_eq!(
        provenance
            .get("domains")
            .and_then(serde_json::Value::as_array)
            .and_then(|a| a.first())
            .and_then(serde_json::Value::as_str),
        Some("clanker")
    );
    assert_eq!(field("harness")?, "claude-code");
    assert_eq!(field("model")?, "claude-sonnet-5");
    assert_eq!(field("session")?, "s-1");
    assert_eq!(field("cwd")?, "/Users/op/Projects/kb/main");
    Ok(())
}

/// A node created without `--provenance-json` reports `null` provenance
/// rather than an object with every field empty, so a caller can tell "no
/// provenance was ever asserted" from "provenance was asserted as absent".
#[test]
fn create_without_provenance_json_reports_null_provenance() -> TestResult {
    let (_dir, db) = temp_db("no-provenance")?;
    seed(&db, "n1")?;

    let json = run_kb(&db, Some(""), &["get", "n1", "--json"])?;
    assert!(json.status.success(), "get --json failed: {}", json.stderr);
    let value: serde_json::Value = serde_json::from_str(&json.stdout)?;
    let provenance = value.get("data").and_then(|d| d.get("provenance"));
    assert!(
        provenance.is_none_or(serde_json::Value::is_null),
        "expected null provenance, got: {provenance:?}"
    );
    Ok(())
}

/// Read `data.provenance.<key>` from a `kb get --json` response as a string.
fn provenance_field<'a>(
    value: &'a serde_json::Value,
    key: &str,
) -> Result<&'a str, Box<dyn std::error::Error>> {
    value
        .get("data")
        .and_then(|d| d.get("provenance"))
        .and_then(|p| p.get(key))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("provenance.{key} missing or not a string: {value}").into())
}

/// `kb update --provenance-json <file>` closes a gap `kb update` used to
/// leave: before it,
/// nothing could move an existing record's provenance without a
/// delete-and-recreate, which resets `created`, cascade-deletes tags and
/// links, and can lose the record outright if `create` fails after `delete`
/// succeeds. `update --provenance-json` replaces the provenance in place,
/// through the same `WriteOptions` path T005 gave `create`
/// (`read_provenance_options` in `src/cli_main.rs`), and `created` is
/// preserved exactly as every other `update` already preserves it.
#[test]
fn update_with_provenance_json_replaces_provenance_and_preserves_created() -> TestResult {
    let (dir, db) = temp_db("update-provenance")?;
    let provenance_a = dir.path().join("a.json");
    std::fs::write(&provenance_a, r#"{"project":"old-project"}"#)?;
    let provenance_b = dir.path().join("b.json");
    std::fs::write(
        &provenance_b,
        r#"{"project":"kb","project_source":"remote","remote":"github.com/tftio/kb"}"#,
    )?;

    let create = run_kb(
        &db,
        Some("* A note\n\noriginal body.\n"),
        &[
            "create",
            "--id",
            "n1",
            "--provenance-json",
            provenance_a
                .to_str()
                .ok_or("provenance path is not UTF-8")?,
            "--json",
        ],
    )?;
    assert!(create.status.success(), "create failed: {}", create.stderr);
    let created_before: serde_json::Value = serde_json::from_str(&create.stdout)?;
    let created_at = created_before
        .get("data")
        .and_then(|d| d.get("createdAt"))
        .and_then(serde_json::Value::as_str)
        .ok_or("create response missing createdAt")?
        .to_owned();

    let update = run_kb(
        &db,
        Some("* A note\n\nupdated body.\n"),
        &[
            "update",
            "n1",
            "--provenance-json",
            provenance_b
                .to_str()
                .ok_or("provenance path is not UTF-8")?,
            "--json",
        ],
    )?;
    assert!(update.status.success(), "update failed: {}", update.stderr);
    let updated: serde_json::Value = serde_json::from_str(&update.stdout)?;
    assert_eq!(
        updated.get("data").and_then(|d| d.get("createdAt")),
        Some(&serde_json::Value::String(created_at.clone())),
        "created must survive a provenance-changing update"
    );

    let json = run_kb(&db, Some(""), &["get", "n1", "--json"])?;
    assert!(json.status.success(), "get --json failed: {}", json.stderr);
    let value: serde_json::Value = serde_json::from_str(&json.stdout)?;
    assert_eq!(
        value.get("data").and_then(|d| d.get("createdAt")),
        Some(&serde_json::Value::String(created_at)),
        "created must still be the original after a later get"
    );
    assert_eq!(provenance_field(&value, "project")?, "kb");
    assert_eq!(provenance_field(&value, "projectSource")?, "remote");
    assert_eq!(provenance_field(&value, "remote")?, "github.com/tftio/kb");
    assert!(
        value
            .get("data")
            .and_then(|d| d.get("document"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|doc| doc.contains("updated body.")),
        "the document itself must still have moved to what update sent"
    );
    Ok(())
}

/// `kb update` without `--provenance-json` (T016) keeps the existing
/// record's provenance, kind and source exactly as `tags add`/`tags
/// rm`/`tags merge` already do (`existing_options`, `src/write.rs`): an
/// agent's ordinary `kb update` of a captured transcript must not silently
/// drop the project that capture asserted. This replaces the earlier
/// behaviour, where an update with no flag reset provenance to none.
#[test]
fn update_without_provenance_json_preserves_provenance_kind_and_source() -> TestResult {
    let (dir, db) = temp_db("update-no-provenance")?;
    let home = dir.path();
    let provenance_a = home.join("a.json");
    std::fs::write(
        &provenance_a,
        r#"{"project":"kb","project_source":"declared","remote":"github.com/tftio/kb"}"#,
    )?;

    let create = run_kb(
        &db,
        Some("* A note\n\noriginal body.\n"),
        &[
            "create",
            "--id",
            "n1",
            "--provenance-json",
            provenance_a
                .to_str()
                .ok_or("provenance path is not UTF-8")?,
        ],
    )?;
    assert!(create.status.success(), "create failed: {}", create.stderr);

    // `kb get --json` reports neither `kind` nor `source`, so those are
    // checked through the index directly (as in
    // `update_preserves_session_transcript_kind`), at the path the
    // subprocess wrote to.
    let index = kb::index::Index::open(&home.join(".local/share/kb/index.db"))?;
    let before_row = index.record("n1")?;
    assert_eq!(before_row.kind, "note");
    assert_eq!(before_row.source, "node-id:n1");

    let update = run_kb(&db, Some("* A note\n\nupdated body.\n"), &["update", "n1"])?;
    assert!(update.status.success(), "update failed: {}", update.stderr);

    let after_row = index.record("n1")?;
    assert_eq!(
        after_row.kind, before_row.kind,
        "an update with no --provenance-json must keep the existing kind"
    );
    assert_eq!(
        after_row.source, before_row.source,
        "an update with no --provenance-json must keep the existing source"
    );

    let json = run_kb(&db, Some(""), &["get", "n1", "--json"])?;
    assert!(json.status.success(), "get --json failed: {}", json.stderr);
    let value: serde_json::Value = serde_json::from_str(&json.stdout)?;
    assert_eq!(
        provenance_field(&value, "project")?,
        "kb",
        "an update with no --provenance-json must keep the existing provenance"
    );
    assert_eq!(provenance_field(&value, "projectSource")?, "declared");
    assert_eq!(provenance_field(&value, "remote")?, "github.com/tftio/kb");
    assert!(
        value
            .get("data")
            .and_then(|d| d.get("document"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|doc| doc.contains("updated body.")),
        "the document itself must still have moved to what update sent"
    );
    Ok(())
}

/// A `session-transcript` record's `kind` survives an ordinary `kb update`
/// (T016): nothing about replacing a transcript's body should be able to
/// turn it back into a `note`, the way `WriteOptions::note` used to on every
/// update before this task.
///
/// `kb create`/`kb update` never take a `--kind` flag; the only way an
/// ordinary write produces `session-transcript` is the capture queue
/// (`src/ingest.rs`), which assigns that kind to any submission tagged
/// `conversation` (`src/migrate.rs::TRANSCRIPT_TAG`). So this seeds the
/// record the way a real capture would: enqueue a tagged submission with
/// provenance, drain it, then run `kb update` on the result.
#[test]
fn update_preserves_session_transcript_kind() -> TestResult {
    let (dir, db) = temp_db("update-kind")?;
    let home = dir.path();
    let queue = kb::ingest::Queue::open(&home.join(".local/share/kb/queue"))?;
    queue.enqueue(&kb::ingest::Submission {
        id: "n1".to_owned(),
        corpus: "kb".to_owned(),
        document: "* A transcript  :conversation:\n\noriginal body.\n".to_owned(),
        provenance: Some(kb::record::RawProvenance {
            project: Some("kb".to_owned()),
            ..Default::default()
        }),
    })?;
    let drain = run_kb(&db, None, &["queue", "drain"])?;
    assert!(
        drain.status.success(),
        "queue drain failed: {}",
        drain.stderr
    );

    // `kb get --json` reports no `kind` field, so the record's kind is
    // checked through the index directly, at the same path the subprocess
    // wrote to (`kb::index::default_index_path`'s logic, `$HOME/.local/share/kb/index.db`).
    let index = kb::index::Index::open(&home.join(".local/share/kb/index.db"))?;
    let before_row = index.record("n1")?;
    assert_eq!(
        before_row.kind, "session-transcript",
        "the drained submission tagged :conversation: should have stored that kind"
    );

    let before = run_kb(&db, Some(""), &["get", "n1", "--json"])?;
    assert!(before.status.success(), "get failed: {}", before.stderr);
    let before_value: serde_json::Value = serde_json::from_str(&before.stdout)?;
    assert_eq!(
        provenance_field(&before_value, "project")?,
        "kb",
        "the drained submission should have stored the queued provenance: {before_value}"
    );

    let update = run_kb(
        &db,
        Some("* A transcript  :conversation:\n\nupdated body.\n"),
        &["update", "n1"],
    )?;
    assert!(update.status.success(), "update failed: {}", update.stderr);

    let after_row = index.record("n1")?;
    assert_eq!(
        after_row.kind, "session-transcript",
        "an update with no --provenance-json must keep the existing kind, got: {after_row:?}"
    );

    let json = run_kb(&db, Some(""), &["get", "n1", "--json"])?;
    assert!(json.status.success(), "get --json failed: {}", json.stderr);
    let value: serde_json::Value = serde_json::from_str(&json.stdout)?;
    assert_eq!(
        provenance_field(&value, "project")?,
        "kb",
        "an update with no --provenance-json must keep the existing provenance"
    );
    assert!(
        value
            .get("data")
            .and_then(|d| d.get("document"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|doc| doc.contains("updated body.")),
        "the document itself must still have moved to what update sent"
    );
    Ok(())
}

/// `kb create` infers `kind: session-transcript` from a document tagged
/// `conversation`, the same rule the capture queue worker applies
/// (`src/ingest.rs`, via `src/migrate.rs::kind_for_tags`). Plan
/// `PLAN-20260923-project-identity` T017: the T008 rehearsal found `kb
/// create` hardcoding every write to `kind: note`, which meant every new
/// transcript the importer wrote locally (rather than through the capture
/// queue) landed as a note. There is no `--kind` flag by design (invariant:
/// "the kind comes from the tag"); a document without the tag still creates
/// as an ordinary note.
#[test]
fn create_infers_session_transcript_kind_from_conversation_tag() -> TestResult {
    let (dir, db) = temp_db("create-kind-transcript")?;
    let home = dir.path();

    let transcript = run_kb(
        &db,
        Some("* A transcript  :conversation:\n\nsession body.\n"),
        &["create", "--id", "t1"],
    )?;
    assert!(
        transcript.status.success(),
        "create failed: {}",
        transcript.stderr
    );

    let note = run_kb(
        &db,
        Some("* An ordinary note\n\nauthored body.\n"),
        &["create", "--id", "n1"],
    )?;
    assert!(note.status.success(), "create failed: {}", note.stderr);

    // `kb get --json` reports no `kind` field, so it is checked through the
    // index directly, at the same path the subprocess wrote to.
    let index = kb::index::Index::open(&home.join(".local/share/kb/index.db"))?;
    assert_eq!(
        index.record("t1")?.kind,
        "session-transcript",
        "a document tagged :conversation: must create as a session transcript"
    );
    assert_eq!(
        index.record("n1")?.kind,
        "note",
        "a document without the conversation tag must still create as an ordinary note"
    );
    Ok(())
}
