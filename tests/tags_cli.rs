//! CLI tests for `kb tags`, `kb tags merge`, and the single-node tag edits
//! `kb tags add` and `kb tags rm`.
//!
//! Tag normalization silently fragments the namespace: two spellings a human
//! reads as one tag become two, and the resulting miss is indistinguishable
//! from an absent node. `kb tags` makes the vocabulary
//! inspectable so a caller can see what exists before inventing a variant, and
//! `kb tags merge` reconciles what is already fragmented.
//!
//! The merge assertions matter more than they look. `node_tags` is re-derived
//! from the stored document on every write, so a merge that touched only the
//! join table would be undone by the next `kb update` of an affected node --
//! and would appear to have worked until it silently had not. These tests pin
//! that the rewrite reaches the document.
//!
//! `tags add` and `tags rm` carry the same burden for a single node, plus one
//! more: they exist because the only prior way to retag was to pipe a node's
//! whole document back through `kb update --tag`, which replaces the body with
//! whatever stdin held. So the assertions read the stored document back rather
//! than trusting an exit status, and the headingless case is checked against
//! the trap `--tag` falls into -- injecting an empty level-1 heading to carry
//! the tag (pinned in `write_cli.rs`). `tags add` writes a
//! `#+filetags:` line there instead, org's document-level tag mechanism, which
//! needs no heading.
//!
//! Every spawned `kb` process is sandboxed: `HOME`, `XDG_CONFIG_HOME`, and
//! `--db` are pinned to the per-test temp directory so the result is identical
//! on a clean CI runner with an empty `$HOME`.

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
    db: &Path,
    stdin: Option<&str>,
    args: &[&str],
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

fn create(db: &Path, id: &str, body: &str) -> TestResult {
    let run = run_kb(db, Some(body), &["create", "--id", id])?;
    assert!(run.status.success(), "seed create failed: {}", run.stderr);
    Ok(())
}

/// Parse `kb tags` text output into `(tag, count)` pairs, in emitted order.
fn tag_lines(db: &Path, args: &[&str]) -> Result<Vec<(String, i64)>, Box<dyn std::error::Error>> {
    let run = run_kb(db, None, args)?;
    assert!(
        run.status.success(),
        "tags {args:?} failed: {}{}",
        run.stdout,
        run.stderr
    );
    let mut out = Vec::new();
    for line in run.stdout.lines().filter(|l| !l.trim().is_empty()) {
        let mut parts = line.split_whitespace();
        let count: i64 = parts.next().ok_or("missing count")?.parse()?;
        let tag = parts.next().ok_or("missing tag")?.to_string();
        out.push((tag, count));
    }
    Ok(out)
}

/// Three nodes: two under `rust`, one under `ci-cd`, one stranded under the
/// collapsed `cicd` spelling.
fn seed(db: &Path) -> TestResult {
    create(db, "a", "* Alpha :rust:ci-cd:\n\nfirst\n")?;
    create(db, "b", "* Beta :rust:\n\nsecond\n")?;
    create(db, "c", "* Gamma :cicd:\n\nthird\n")?;
    Ok(())
}

#[test]
fn tags_lists_the_vocabulary_by_count_then_name() -> TestResult {
    let (_dir, db) = temp_db("list")?;
    seed(&db)?;

    let tags = tag_lines(&db, &["tags"])?;

    assert_eq!(
        tags,
        vec![
            ("rust".to_string(), 2),
            ("ci-cd".to_string(), 1),
            ("cicd".to_string(), 1),
        ],
        "count descending, then tag ascending"
    );
    Ok(())
}

/// The fragmentation is visible precisely because the vocabulary is listable:
/// `ci-cd` and `cicd` appear as the two separate tags they are.
#[test]
fn fragmented_spellings_show_up_as_distinct_tags() -> TestResult {
    let (_dir, db) = temp_db("fragmented")?;
    seed(&db)?;

    let tags = tag_lines(&db, &["tags"])?;
    let names: Vec<&str> = tags.iter().map(|(t, _)| t.as_str()).collect();

    assert!(names.contains(&"ci-cd") && names.contains(&"cicd"));
    Ok(())
}

#[test]
fn tags_json_emits_an_envelope() -> TestResult {
    let (_dir, db) = temp_db("json")?;
    seed(&db)?;

    let run = run_kb(&db, None, &["tags", "--json"])?;

    assert!(run.status.success(), "{}", run.stderr);
    let parsed: serde_json::Value = serde_json::from_str(&run.stdout)?;
    assert!(
        parsed.to_string().contains("\"rust\""),
        "envelope should carry the tags: {}",
        run.stdout
    );
    Ok(())
}

#[test]
fn tags_on_an_empty_database_prints_nothing_and_succeeds() -> TestResult {
    let (_dir, db) = temp_db("empty")?;

    let run = run_kb(&db, None, &["tags"])?;

    assert!(run.status.success(), "{}", run.stderr);
    assert!(
        run.stdout.trim().is_empty(),
        "expected no output, got {:?}",
        run.stdout
    );
    Ok(())
}

#[test]
fn merge_moves_every_node_onto_the_target_tag() -> TestResult {
    let (_dir, db) = temp_db("merge")?;
    seed(&db)?;

    let run = run_kb(&db, None, &["tags", "merge", "cicd", "ci-cd"])?;
    assert!(run.status.success(), "{}", run.stderr);
    assert!(
        run.stdout.contains("merged cicd into ci-cd"),
        "unexpected report: {}",
        run.stdout
    );

    let tags = tag_lines(&db, &["tags"])?;
    assert_eq!(
        tags,
        vec![("ci-cd".to_string(), 2), ("rust".to_string(), 2)],
        "cicd is gone and ci-cd absorbed its node"
    );
    Ok(())
}

/// The durability property. A merge confined to `node_tags` would pass every
/// assertion above and fail this one.
#[test]
fn a_merged_tag_does_not_return_after_a_round_trip_update() -> TestResult {
    let (_dir, db) = temp_db("durable")?;
    seed(&db)?;
    let merge = run_kb(&db, None, &["tags", "merge", "cicd", "ci-cd"])?;
    assert!(merge.status.success(), "{}", merge.stderr);

    let got = run_kb(&db, None, &["get", "c"])?;
    assert!(got.status.success(), "{}", got.stderr);
    let updated = run_kb(&db, Some(&got.stdout), &["update", "c"])?;
    assert!(updated.status.success(), "{}", updated.stderr);

    let names: Vec<String> = tag_lines(&db, &["tags"])?
        .into_iter()
        .map(|(t, _)| t)
        .collect();
    assert!(
        !names.contains(&"cicd".to_string()),
        "the collapsed spelling came back: {names:?}"
    );
    Ok(())
}

#[test]
fn merging_a_tag_no_node_carries_reports_and_changes_nothing() -> TestResult {
    let (_dir, db) = temp_db("absent")?;
    seed(&db)?;
    let before = tag_lines(&db, &["tags"])?;

    let run = run_kb(&db, None, &["tags", "merge", "absent", "ci-cd"])?;

    assert!(run.status.success(), "{}", run.stderr);
    assert!(
        run.stdout.contains("nothing to merge"),
        "unexpected report: {}",
        run.stdout
    );
    assert_eq!(before, tag_lines(&db, &["tags"])?);
    Ok(())
}

#[test]
fn merging_a_tag_into_itself_is_refused() -> TestResult {
    let (_dir, db) = temp_db("selfmerge")?;
    seed(&db)?;

    let run = run_kb(&db, None, &["tags", "merge", "ci-cd", "CI/CD"])?;

    assert!(!run.status.success(), "expected a non-zero exit");
    let combined = format!("{}{}", run.stdout, run.stderr);
    assert!(
        combined.contains("nothing to merge"),
        "expected the same-tag refusal, got: {combined}"
    );
    Ok(())
}

/// Tags written in a `#+filetags:` line are rewritten too, not only heading
/// tags, and the neighbours in the line survive intact.
#[test]
fn merge_rewrites_filetags_lines() -> TestResult {
    let (_dir, db) = temp_db("filetags")?;
    create(
        &db,
        "f",
        "#+filetags: :alpha:bookclub:omega:\n\n* Node\n\nbody\n",
    )?;

    let run = run_kb(&db, None, &["tags", "merge", "bookclub", "book-club"])?;
    assert!(run.status.success(), "{}", run.stderr);

    let names: Vec<String> = tag_lines(&db, &["tags"])?
        .into_iter()
        .map(|(t, _)| t)
        .collect();
    assert!(names.contains(&"book-club".to_string()), "{names:?}");
    assert!(
        names.contains(&"alpha".to_string()),
        "neighbour lost: {names:?}"
    );
    assert!(
        names.contains(&"omega".to_string()),
        "neighbour lost: {names:?}"
    );
    assert!(!names.contains(&"bookclub".to_string()), "{names:?}");
    Ok(())
}

/// Both new verbs must reach the emitted skills, which is the only way the
/// vocabulary becomes discoverable to an agent rather than to a human reading
/// `--help`.
#[test]
fn the_new_verbs_are_declared_on_the_agent_surface() -> TestResult {
    let (_dir, db) = temp_db("surface")?;

    let run = run_kb(&db, None, &["meta", "agent", "list"])?;

    assert!(run.status.success(), "{}", run.stderr);
    assert!(run.stdout.contains("- tags:"), "{}", run.stdout);
    assert!(run.stdout.contains("- tags-merge:"), "{}", run.stdout);
    assert!(run.stdout.contains("- tags-add:"), "{}", run.stdout);
    assert!(run.stdout.contains("- tags-rm:"), "{}", run.stdout);
    Ok(())
}

// ── tags add / tags rm ─────────────────────────────────────────────────

/// The node's stored document as org text.
fn stored(db: &Path, id: &str) -> Result<String, Box<dyn std::error::Error>> {
    let run = run_kb(db, None, &["get", id])?;
    assert!(run.status.success(), "get failed: {}", run.stderr);
    Ok(run.stdout)
}

/// Whether `kb list-by-tag <tag>` returns the node - the check that the
/// projection followed the document rather than diverging from it.
fn listed_under(db: &Path, tag: &str, id: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let run = run_kb(db, None, &["list-by-tag", tag])?;
    assert!(run.status.success(), "list-by-tag failed: {}", run.stderr);
    Ok(run.stdout.contains(id))
}

const WITH_FILETAGS: &str = "#+filetags: :alpha:\n\n* Seeded\n\nreal body content\n";

/// The contrast with kb#5: a headingless node takes a tag without acquiring a
/// fabricated heading, and keeps its prose.
#[test]
fn tags_add_on_a_headingless_body_writes_filetags_and_invents_no_heading() -> TestResult {
    let (_dir, db) = temp_db("add-headingless")?;
    create(&db, "n1", "just prose, no heading\n")?;

    let run = run_kb(&db, None, &["tags", "add", "n1", "Beta"])?;

    assert!(run.status.success(), "tags add failed: {}", run.stderr);
    assert!(run.stdout.contains("beta"), "stdout was {:?}", run.stdout);
    let after = stored(&db, "n1")?;
    assert!(
        after.contains("#+filetags:") && after.contains("beta"),
        "expected a filetags line, got {after:?}"
    );
    assert!(
        !after.contains('*'),
        "tags add must not inject a heading; got {after:?}"
    );
    assert!(
        after.contains("just prose, no heading"),
        "body lost: {after:?}"
    );
    assert!(listed_under(&db, "beta", "n1")?);
    Ok(())
}

#[test]
fn tags_add_extends_an_existing_filetags_line_and_leaves_the_body_alone() -> TestResult {
    let (_dir, db) = temp_db("add-filetags")?;
    create(&db, "n1", WITH_FILETAGS)?;

    let run = run_kb(&db, None, &["tags", "add", "n1", "beta"])?;

    assert!(run.status.success(), "tags add failed: {}", run.stderr);
    let after = stored(&db, "n1")?;
    assert!(
        after.contains(":alpha:beta:"),
        "expected the existing filetags line extended, got {after:?}"
    );
    assert!(after.contains("real body content"), "body lost: {after:?}");
    assert!(listed_under(&db, "alpha", "n1")?);
    assert!(listed_under(&db, "beta", "n1")?);
    Ok(())
}

#[test]
fn tags_add_extends_the_first_heading_when_no_filetags_line_exists() -> TestResult {
    let (_dir, db) = temp_db("add-heading")?;
    create(&db, "n1", "* Seeded :alpha:\n\nreal body content\n")?;

    run_kb(&db, None, &["tags", "add", "n1", "beta"])?;

    let after = stored(&db, "n1")?;
    assert!(
        after.contains(":alpha:beta:"),
        "expected the heading's tag list extended, got {after:?}"
    );
    assert!(
        !after.contains("#+filetags:"),
        "tags belong where they already were: {after:?}"
    );
    Ok(())
}

#[test]
fn tags_add_normalizes_the_argument_before_storing_it() -> TestResult {
    let (_dir, db) = temp_db("add-normalize")?;
    create(&db, "n1", WITH_FILETAGS)?;

    run_kb(&db, None, &["tags", "add", "n1", "Silent Critic"])?;

    // Stored in the one form list-by-tag will match, not as supplied.
    assert!(listed_under(&db, "silent-critic", "n1")?);
    Ok(())
}

#[test]
fn tags_rm_removes_the_tag_and_leaves_the_body_intact() -> TestResult {
    let (_dir, db) = temp_db("rm")?;
    create(&db, "n1", WITH_FILETAGS)?;

    let run = run_kb(&db, None, &["tags", "rm", "n1", "Alpha"])?;

    assert!(run.status.success(), "tags rm failed: {}", run.stderr);
    let after = stored(&db, "n1")?;
    assert!(!after.contains("alpha"), "tag survived removal: {after:?}");
    assert!(after.contains("real body content"), "body lost: {after:?}");
    assert!(after.contains("Seeded"), "heading lost: {after:?}");
    assert!(!listed_under(&db, "alpha", "n1")?);
    Ok(())
}

/// `kb get` prefixes the body with a metadata drawer carrying `:UPDATED:`,
/// which a write legitimately moves. The document itself is what must come
/// back unchanged, so the drawer is dropped before comparing.
fn stored_body(db: &Path, id: &str) -> Result<String, Box<dyn std::error::Error>> {
    let text = stored(db, id)?;
    Ok(match text.split_once(":END:\n") {
        Some((_, body)) => body.to_string(),
        None => text,
    })
}

#[test]
fn tags_add_then_rm_restores_the_stored_document() -> TestResult {
    let (_dir, db) = temp_db("roundtrip")?;
    create(&db, "n1", WITH_FILETAGS)?;
    let before = stored_body(&db, "n1")?;

    run_kb(&db, None, &["tags", "add", "n1", "beta"])?;
    run_kb(&db, None, &["tags", "rm", "n1", "beta"])?;

    assert_eq!(stored_body(&db, "n1")?, before);
    Ok(())
}

#[test]
fn a_tag_edit_that_changes_nothing_says_so_and_still_succeeds() -> TestResult {
    let (_dir, db) = temp_db("noop")?;
    create(&db, "n1", WITH_FILETAGS)?;

    let already = run_kb(&db, None, &["tags", "add", "n1", "alpha"])?;
    assert!(already.status.success(), "{}", already.stderr);
    assert!(
        already.stdout.contains("no change"),
        "stdout was {:?}",
        already.stdout
    );

    let absent = run_kb(&db, None, &["tags", "rm", "n1", "never-applied"])?;
    assert!(absent.status.success(), "{}", absent.stderr);
    assert!(
        absent.stdout.contains("no change"),
        "stdout was {:?}",
        absent.stdout
    );
    Ok(())
}

#[test]
fn a_tag_edit_against_an_unknown_id_is_refused() -> TestResult {
    let (_dir, db) = temp_db("unknown-id")?;
    create(&db, "n1", WITH_FILETAGS)?;

    let run = run_kb(&db, None, &["tags", "add", "nosuch", "beta"])?;

    assert!(!run.status.success(), "expected a non-zero exit");
    assert!(
        run.stderr.contains("nosuch"),
        "the error should name the id; stderr was {:?}",
        run.stderr
    );
    Ok(())
}

#[test]
fn a_tag_argument_that_normalizes_away_is_refused_before_any_write() -> TestResult {
    let (_dir, db) = temp_db("empty-arg")?;
    create(&db, "n1", WITH_FILETAGS)?;
    let before = stored(&db, "n1")?;

    let run = run_kb(&db, None, &["tags", "add", "n1", "+++"])?;

    assert!(!run.status.success(), "expected a non-zero exit");
    assert!(!run.stderr.is_empty(), "expected a diagnostic on stderr");
    assert_eq!(stored(&db, "n1")?, before, "the node was written anyway");
    Ok(())
}

/// A tag added by an edit is in the document, so the next full update
/// re-derives it rather than dropping it. An edit that had written
/// `node_tags` directly would fail here and nowhere else.
#[test]
fn a_tag_added_by_an_edit_survives_a_round_trip_update() -> TestResult {
    let (_dir, db) = temp_db("durable-edit")?;
    create(&db, "n1", WITH_FILETAGS)?;
    run_kb(&db, None, &["tags", "add", "n1", "beta"])?;

    let got = run_kb(&db, None, &["get", "n1"])?;
    assert!(got.status.success(), "{}", got.stderr);
    let updated = run_kb(&db, Some(&got.stdout), &["update", "n1"])?;
    assert!(updated.status.success(), "{}", updated.stderr);

    assert!(listed_under(&db, "beta", "n1")?, "the tag did not survive");
    Ok(())
}

/// A case-boundary spelling now normalizes to the hyphenated form, so the two
/// converge without a merge.
#[test]
fn a_camelcase_tag_normalizes_to_the_hyphenated_spelling() -> TestResult {
    let (_dir, db) = temp_db("camel")?;
    create(&db, "x", "* X :silentCritic:\n\nbody\n")?;
    create(&db, "y", "* Y :silent-critic:\n\nbody\n")?;

    let tags = tag_lines(&db, &["tags"])?;

    assert_eq!(
        tags,
        vec![("silent-critic".to_string(), 2)],
        "both spellings should land on one tag"
    );
    Ok(())
}
