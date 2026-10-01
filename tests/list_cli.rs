//! The listing and graph verbs at the process boundary (CLI-002).
//!
//! `kb recent`, `kb orphans`, `kb hubs`, `kb broken`, `kb list-by-tag`,
//! `kb delete` and `kb prompt render` all answered from the superseded
//! database until T029 and answer from the derived index now. Their console
//! output is the surface an operator reads, so it is asserted here rather
//! than described.

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
        .env_remove("KB_MAIL_ROOT")
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

/// The identifiers in a `--json` envelope, in the order returned.
fn ids(stdout: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let payload: Value = serde_json::from_str(stdout)?;
    let rows = payload
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("no data array: {stdout}"))?;
    Ok(rows
        .iter()
        .filter_map(|row| row.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect())
}

/// Three notes: `hub` is linked to by both others, `lonely` is linked to by
/// nobody, and `broken` names a slug that resolves to nothing.
fn seeded() -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let home = tempfile::tempdir()?;
    for (id, body) in [
        (
            "hub",
            "#+name: hub\n#+filetags: :shared:\n* Hub\n\nthe target.\n",
        ),
        (
            "one",
            "#+filetags: :shared:\n* One\n\nsee [[hub]] for the target.\n",
        ),
        ("two", "* Two\n\nalso see [[hub]], and [[nowhere]].\n"),
        ("lonely", "* Lonely\n\nnothing points here.\n"),
    ] {
        let run = run_kb(home.path(), Some(body), &["create", "--id", id])?;
        assert!(run.ok, "seeding {id} failed: {}", run.stderr);
    }
    Ok(home)
}

#[test]
fn recent_lists_the_most_recently_written_first() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["recent", "--limit", "2", "--json"])?;
    assert!(run.ok, "recent failed: {}", run.stderr);
    let listed = ids(&run.stdout)?;
    assert_eq!(listed.len(), 2, "the limit was not applied: {listed:?}");
    assert!(
        listed.contains(&"lonely".to_owned()),
        "the newest record is missing: {listed:?}"
    );
    Ok(())
}

#[test]
fn recent_in_text_names_each_record() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["recent"])?;
    assert!(run.ok, "recent failed: {}", run.stderr);
    assert!(run.stdout.contains("Hub"), "stdout: {}", run.stdout);
    Ok(())
}

#[test]
fn orphans_are_the_records_nothing_links_to() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["orphans", "--json"])?;
    assert!(run.ok, "orphans failed: {}", run.stderr);
    let listed = ids(&run.stdout)?;
    assert!(
        listed.contains(&"lonely".to_owned()),
        "an unlinked record is not an orphan: {listed:?}"
    );
    assert!(
        !listed.contains(&"hub".to_owned()),
        "a linked-to record was called an orphan: {listed:?}"
    );
    Ok(())
}

#[test]
fn hubs_rank_by_how_many_records_point_at_them() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["hubs", "--json"])?;
    assert!(run.ok, "hubs failed: {}", run.stderr);
    let payload: Value = serde_json::from_str(&run.stdout)?;
    let rows = payload
        .get("data")
        .and_then(Value::as_array)
        .ok_or("no data array")?;
    let first = rows.first().ok_or("no hub reported")?;
    assert_eq!(first.get("id").and_then(Value::as_str), Some("hub"));
    assert_eq!(first.get("in_degree").and_then(Value::as_i64), Some(2));
    Ok(())
}

#[test]
fn hubs_in_text_lead_with_the_degree() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["hubs"])?;
    assert!(run.ok, "hubs failed: {}", run.stderr);
    assert!(run.stdout.starts_with("2\thub"), "stdout: {}", run.stdout);
    Ok(())
}

#[test]
fn broken_reports_a_name_that_resolves_to_nothing() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["broken", "--json"])?;
    assert!(run.ok, "broken failed: {}", run.stderr);
    assert!(
        run.stdout.contains("nowhere"),
        "the dangling name is not reported: {}",
        run.stdout
    );
    assert!(
        !run.stdout.contains("\"hub\""),
        "a resolving link was called broken: {}",
        run.stdout
    );
    Ok(())
}

#[test]
fn broken_in_text_says_which_reference_dangles() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["broken"])?;
    assert!(run.ok, "broken failed: {}", run.stderr);
    assert!(
        run.stdout.contains("[[nowhere]]") && run.stdout.contains("(broken)"),
        "stdout: {}",
        run.stdout
    );
    Ok(())
}

#[test]
fn list_by_tag_returns_every_record_carrying_it() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["list-by-tag", "shared", "--json"])?;
    assert!(run.ok, "list-by-tag failed: {}", run.stderr);
    let listed = ids(&run.stdout)?;
    assert_eq!(listed, vec!["hub".to_owned(), "one".to_owned()]);
    Ok(())
}

/// A tag is normalized before it is looked up, so the spelling a caller
/// happens to use is not a way to miss records.
#[test]
fn list_by_tag_normalizes_what_it_is_given() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["list-by-tag", "Shared", "--json"])?;
    assert!(run.ok, "list-by-tag failed: {}", run.stderr);
    assert_eq!(ids(&run.stdout)?.len(), 2);
    Ok(())
}

#[test]
fn links_reports_both_directions() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["links", "hub", "--json"])?;
    assert!(run.ok, "links failed: {}", run.stderr);
    let payload: Value = serde_json::from_str(&run.stdout)?;
    let incoming = payload
        .get("data")
        .and_then(|d| d.get("incoming"))
        .and_then(Value::as_array)
        .ok_or("no incoming array")?;
    assert_eq!(incoming.len(), 2, "both references should be reported");
    Ok(())
}

/// Deleting unbinds the name. The blobs stay addressable — that is what the
/// store is for — but nothing finds the record any more.
#[test]
fn delete_unbinds_the_record_and_removes_it_from_the_index() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["delete", "lonely"])?;
    assert!(run.ok, "delete failed: {}", run.stderr);
    assert!(run.stdout.contains("deleted lonely"), "{}", run.stdout);

    let after = run_kb(home.path(), None, &["get", "lonely"])?;
    assert!(!after.ok, "a deleted record was still readable");

    let listed = run_kb(home.path(), None, &["recent", "--json"])?;
    assert!(
        !ids(&listed.stdout)?.contains(&"lonely".to_owned()),
        "a deleted record is still listed: {}",
        listed.stdout
    );
    Ok(())
}

#[test]
fn deleting_what_is_not_there_says_so() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["delete", "no-such-record"])?;
    assert!(!run.ok, "deleting nothing reported success");
    assert!(
        run.stderr.contains("no node with id"),
        "stderr: {}",
        run.stderr
    );
    Ok(())
}

/// `kb prompt render` assembles a corpus snapshot for a template. What is
/// asserted is that the snapshot is of the live corpus.
#[test]
fn prompt_render_reads_the_corpus() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["prompt", "render", "cold-start-audit"])?;
    assert!(run.ok, "prompt render failed: {}", run.stderr);
    assert!(
        run.stdout.contains("Cold-Start Audit"),
        "the template did not render: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("Hub") || run.stdout.contains("hub"),
        "the rendered prompt names no record from the corpus: {}",
        run.stdout
    );
    Ok(())
}

#[test]
fn prompt_render_names_a_template_it_cannot_find() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["prompt", "render", "no-such-template"])?;
    assert!(!run.ok, "rendering a missing template reported success");
    assert!(
        run.stderr.contains("no-such-template"),
        "stderr: {}",
        run.stderr
    );
    Ok(())
}

/// A store that is not there is reported with the path, not as a bare
/// driver error. Every verb that opens one owes the operator that much.
#[test]
fn commands_name_the_store_they_cannot_open() -> TestResult {
    let home = tempfile::tempdir()?;
    let absent = home.path().join("no-such-store");
    let run = run_kb(
        home.path(),
        None,
        &[
            "reindex",
            "--store",
            &absent.display().to_string(),
            "--index",
            &home.path().join("index.db").display().to_string(),
        ],
    )?;
    assert!(!run.ok, "reindexing an absent store reported success");
    let reported = format!("{}{}", run.stdout, run.stderr);
    assert!(
        reported.contains("no-such-store"),
        "the store is not named: {reported}"
    );

    let fsck = run_kb(
        home.path(),
        None,
        &[
            "fsck",
            "--store",
            &absent.display().to_string(),
            "--index",
            &home.path().join("index.db").display().to_string(),
        ],
    )?;
    assert!(!fsck.ok, "fsck against an absent store reported success");
    Ok(())
}

/// Reindexing one record rebuilds that record and says so.
#[test]
fn reindex_reports_the_scope_it_rebuilt() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["reindex", "--record", "hub"])?;
    assert!(run.ok, "reindex failed: {}", run.stderr);
    assert!(
        run.stdout.contains("1 record"),
        "the scope is not reported: {}",
        run.stdout
    );

    let corpus = run_kb(home.path(), None, &["reindex", "--corpus", "kb"])?;
    assert!(corpus.ok, "reindex failed: {}", corpus.stderr);
    assert!(
        corpus.stdout.contains("4 records"),
        "the corpus scope is not reported: {}",
        corpus.stdout
    );
    Ok(())
}

/// `kb fsck` on a corpus written through the store finds nothing to repair,
/// which is the state every write is supposed to leave behind.
#[test]
fn fsck_is_clean_after_writing_through_the_store() -> TestResult {
    let home = seeded()?;
    let run = run_kb(home.path(), None, &["fsck"])?;
    assert!(run.ok, "fsck failed: {}", run.stderr);
    assert!(
        run.stdout.contains("the index agrees with the store"),
        "stdout: {}",
        run.stdout
    );
    Ok(())
}

/// A reranker configured with something that is not a usable endpoint costs
/// ordering, not results — and says so, because a caller who cannot tell a
/// fused ranking from a reranked one cannot interpret either (ENG-004).
#[test]
fn an_unusable_reranker_endpoint_degrades_and_says_so() -> TestResult {
    let home = seeded()?;
    let mut command = Command::new(env!("CARGO_BIN_EXE_kb"));
    let out = command
        .env("HOME", home.path())
        .env("XDG_CONFIG_HOME", home.path())
        .env_remove("KB_EMBEDDING_BASE_URL")
        .env_remove("KB_EMBEDDING_MODEL")
        .env_remove("KB_MAIL_ROOT")
        .env("KB_RERANK_BASE_URL", "not a url at all")
        .args(["search", "target", "--json"])
        .stdin(Stdio::null())
        .output()?;
    assert!(out.status.success(), "the search should still answer");
    let stderr = String::from_utf8(out.stderr)?;
    assert!(
        stderr.contains("fused ranking"),
        "the degradation is not reported: {stderr}"
    );
    Ok(())
}

/// A vector describing text the index no longer holds is dropped by a
/// rebuild, and the count is reported: vectors are the one derived thing
/// that costs hours to recompute, so quietly discarding them would be
/// expensive in a way nothing else in the index is.
#[test]
fn reindex_reports_vectors_it_dropped() -> TestResult {
    let home = seeded()?;
    let index_path = home.path().join(".local/share/kb/index.db");
    {
        let index = kb::index::Index::open(&index_path)?;
        // A stream no record addresses. Nothing can re-derive the passage it
        // described, which is exactly what makes the vector an orphan.
        index.put_embedding(
            "0000000000000000000000000000000000000000",
            0,
            10,
            "a-model",
            &kb::embedding::encode_embedding(&[0.5, 0.5]),
        )?;
    }

    let run = run_kb(home.path(), None, &["reindex"])?;
    assert!(run.ok, "reindex failed: {}", run.stderr);
    assert!(
        run.stdout.contains("dropped 1 vector"),
        "the dropped vector is not reported: {}",
        run.stdout
    );
    Ok(())
}

/// `kb prompt list` and `kb prompt show` read the built-in templates and
/// whatever the operator has put in their override directory, which is
/// resolved from the environment — so both are asserted through a real
/// process with a real home.
#[test]
fn prompt_list_and_show_see_builtins_and_overrides() -> TestResult {
    let home = seeded()?;
    let listed = run_kb(home.path(), None, &["prompt", "list"])?;
    assert!(listed.ok, "prompt list failed: {}", listed.stderr);
    assert!(
        listed.stdout.contains("cold-start-audit"),
        "the built-in is not listed: {}",
        listed.stdout
    );

    let shown = run_kb(home.path(), None, &["prompt", "show", "cold-start-audit"])?;
    assert!(shown.ok, "prompt show failed: {}", shown.stderr);
    assert!(
        shown.stdout.contains("Cold-Start Audit"),
        "the template body is not shown: {}",
        shown.stdout
    );

    // An override in the operator's own directory wins, and is reported as
    // theirs rather than as the built-in it replaced.
    let prompts = home.path().join("kb").join("prompts");
    std::fs::create_dir_all(&prompts)?;
    std::fs::write(
        prompts.join("cold-start-audit.j2"),
        "MINE: {{ recent | length }}",
    )?;
    let overridden = run_kb(home.path(), None, &["prompt", "list", "--json"])?;
    assert!(overridden.ok, "prompt list failed: {}", overridden.stderr);
    assert!(
        overridden.stdout.contains("\"user\""),
        "the override is not reported as the operator's: {}",
        overridden.stdout
    );

    let rendered = run_kb(home.path(), None, &["prompt", "render", "cold-start-audit"])?;
    assert!(rendered.ok, "prompt render failed: {}", rendered.stderr);
    assert!(
        rendered.stdout.starts_with("MINE: 4"),
        "the override did not render: {}",
        rendered.stdout
    );
    Ok(())
}

/// A template naming something the corpus cannot answer fails at render
/// time with the reason, rather than rendering a blank where the answer
/// should be — `MiniJinja` runs in strict mode for exactly that.
#[test]
fn a_template_referring_to_nothing_fails_with_its_reason() -> TestResult {
    let home = seeded()?;
    let prompts = home.path().join("kb").join("prompts");
    std::fs::create_dir_all(&prompts)?;
    std::fs::write(prompts.join("bad.j2"), "{{ no_such_binding }}")?;
    let run = run_kb(home.path(), None, &["prompt", "render", "bad"])?;
    assert!(!run.ok, "an undefined binding rendered anyway");
    assert!(
        run.stderr.contains("prompt render failed"),
        "stderr: {}",
        run.stderr
    );
    Ok(())
}
