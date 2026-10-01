//! CLI smoke tests for the unified link-graph verbs:
//! `kb links`, `kb orphans`, `kb hubs`, `kb broken`.
//!
//! Each test boots a fresh on-disk database via the `kb` binary,
//! populates a small graph that mixes id-links and name-links, and
//! asserts that the four verbs emit parseable JSON envelopes consistent
//! with the in-process tests in `src/storage.rs`. Every row carries the
//! `link_type` discriminator so id and name rows are distinguishable in
//! the agent-facing output.
//!
//! It also pins the write-time warning for dangling references: a
//! bracket reference that resolves to nothing is announced on stderr at
//! the moment it is written, while the node is still stored and the exit
//! status stays 0. Both halves are asserted, including the silence when
//! every reference resolves - a warning that fired on every write would
//! train its reader to ignore it.
//!
//! Every spawned `kb` process is sandboxed: `HOME`, `XDG_CONFIG_HOME`,
//! and `--db` are pinned to the per-test temp directory so the result is
//! identical on a clean CI runner with an empty `$HOME`.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::Value;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Run `kb --db <db> <args...>` with the given stdin payload, returning
/// (status, stdout, stderr). `HOME`/`XDG_CONFIG_HOME` are pinned to the
/// DB's parent temp dir so no real user config is consulted.
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
    // A developer's live endpoints must not reach the child. `create` warns on
    // stderr when it cannot embed a node it just stored, so a test asserting
    // silence otherwise passes or fails according to whether a daemon happens
    // to be running on the machine.
    for var in [
        "KB_EMBEDDING_BASE_URL",
        "KB_EMBEDDING_MODEL",
        "KB_EMBEDDING_API_KEY",
        "KB_RERANK_BASE_URL",
        "KB_RERANK_TOP_K",
    ] {
        cmd.env_remove(var);
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

/// Create a node from `org` under `id`, asserting the CLI succeeded.
fn create_node(db: &Path, org: &str, id: &str) -> TestResult {
    let (st, out, err) = run_kb(db, Some(org), &["create", "--id", id, "--json"])?;
    assert!(st.success(), "create {id} failed: {err}\nstdout: {out}");
    Ok(())
}

/// Run `kb links <node> --json` and return its single outgoing row,
/// asserting the verb succeeded and produced exactly one row.
fn sole_outgoing(db: &Path, node: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let (st, out, err) = run_kb(db, None, &["links", node, "--json"])?;
    assert!(st.success(), "links {node} failed: {err}");
    let data = extract_data(&out)?;
    let outgoing = data
        .get("outgoing")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("links {node}: outgoing array missing"))?;
    assert_eq!(outgoing.len(), 1, "links {node}: expected one outgoing row");
    let row = outgoing
        .first()
        .cloned()
        .ok_or_else(|| format!("links {node}: outgoing empty"))?;
    Ok(row)
}

/// Aggregated CLI verification for the consolidated `kb links / orphans
/// / hubs / broken` verbs against the unified link graph. The
/// criterion `kb.unified-cli-verbs` pins this test by name.
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "cohesive end-to-end CLI verification; splitting would fragment the fixture setup"
)]
fn unified_cli_verbs() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");

    // Named target ("alpha") referenced by `source` via [[alpha]].
    create_node(&db, "#+name: alpha\n* Target\n\nI am alpha.\n", "target")?;
    // `source` references [[alpha]] (name-link, resolves to target).
    create_node(&db, "* Source\n\nSee [[alpha]] for context.\n", "source")?;
    // `id-src` references target via [[id:target]] (id-link).
    create_node(
        &db,
        "* Id Source\n\nSee [[id:target]] for context.\n",
        "id-src",
    )?;
    // `broken-ref` mentions a slug nothing carries.
    create_node(
        &db,
        "* Broken\n\nSee [[ghost]] which does not exist.\n",
        "broken-ref",
    )?;
    // `lonely` carries no incoming refs at all.
    create_node(&db, "* Lonely\n\nNo one links to me.\n", "lonely")?;

    // ── kb links: shows both id-link and name-link incoming with link_type ──
    let (st, out, err) = run_kb(&db, None, &["links", "target", "--json"])?;
    assert!(st.success(), "links target failed: {err}");
    let data = extract_data(&out)?;
    let incoming = data
        .get("incoming")
        .and_then(Value::as_array)
        .ok_or("links target: incoming array missing")?;
    let by_src: std::collections::HashMap<&str, &str> = incoming
        .iter()
        .filter_map(|r| {
            let s = r.get("source_id").and_then(Value::as_str)?;
            let t = r.get("link_type").and_then(Value::as_str)?;
            Some((s, t))
        })
        .collect();
    assert_eq!(by_src.get("source"), Some(&"name"));
    assert_eq!(by_src.get("id-src"), Some(&"id"));

    // Outgoing for source: one name-link with link_type='name'.
    let row = sole_outgoing(&db, "source")?;
    assert_eq!(row.get("link_type").and_then(Value::as_str), Some("name"));
    assert_eq!(
        row.get("target_slug").and_then(Value::as_str),
        Some("alpha")
    );
    assert_eq!(row.get("target_id").and_then(Value::as_str), Some("target"));

    // Outgoing for id-src: one id-link with link_type='id'.
    let row = sole_outgoing(&db, "id-src")?;
    assert_eq!(row.get("link_type").and_then(Value::as_str), Some("id"));
    assert_eq!(row.get("target_id").and_then(Value::as_str), Some("target"));
    assert!(
        row.get("target_slug").is_some_and(Value::is_null),
        "id-link rows must carry target_slug=null"
    );

    // Outgoing for broken-ref: a broken name-link.
    let row = sole_outgoing(&db, "broken-ref")?;
    assert_eq!(row.get("link_type").and_then(Value::as_str), Some("name"));
    assert!(
        row.get("target_id").is_some_and(Value::is_null),
        "broken name-link must carry target_id=null"
    );
    assert_eq!(
        row.get("target_slug").and_then(Value::as_str),
        Some("ghost")
    );

    // ── kb links: human (non-JSON) output surfaces link_type ──
    let (st, out, err) = run_kb(&db, None, &["links", "target"])?;
    assert!(st.success(), "links target (text) failed: {err}");
    assert!(out.contains("[id]"), "human output must show [id]: {out}");
    assert!(
        out.contains("[name]"),
        "human output must show [name]: {out}"
    );

    // ── kb orphans: counts a node as orphan only when nothing of any
    //   link_type points at it ──
    let (st, out, err) = run_kb(&db, None, &["orphans", "--json"])?;
    assert!(st.success(), "orphans failed: {err}");
    let data = extract_data(&out)?;
    let ids: Vec<&str> = data
        .as_array()
        .ok_or("orphans: expected array")?
        .iter()
        .filter_map(|n| n.get("id").and_then(Value::as_str))
        .collect();
    // target has both a name-link AND an id-link pointing at it -> not orphan.
    assert!(!ids.contains(&"target"));
    assert!(ids.contains(&"source"));
    assert!(ids.contains(&"id-src"));
    assert!(ids.contains(&"broken-ref"));
    assert!(ids.contains(&"lonely"));

    // ── kb hubs: ranks by total in-degree across both link_types ──
    let (st, out, err) = run_kb(&db, None, &["hubs", "--json"])?;
    assert!(st.success(), "hubs failed: {err}");
    let data = extract_data(&out)?;
    let arr = data.as_array().ok_or("hubs: expected array")?;
    assert!(!arr.is_empty(), "expected at least one hub");
    let top = arr.first().ok_or("hubs: empty ranking")?;
    assert_eq!(top.get("id").and_then(Value::as_str), Some("target"));
    // target sums one id-link + one name-link incoming = 2.
    assert_eq!(top.get("in_degree").and_then(Value::as_i64), Some(2));

    // ── kb broken: only the unresolved name-link surfaces ──
    let (st, out, err) = run_kb(&db, None, &["broken", "--json"])?;
    assert!(st.success(), "broken failed: {err}");
    let data = extract_data(&out)?;
    let arr = data.as_array().ok_or("broken: expected array")?;
    assert_eq!(arr.len(), 1);
    let row = arr.first().ok_or("broken: empty list")?;
    assert_eq!(row.get("link_type").and_then(Value::as_str), Some("name"));
    assert_eq!(
        row.get("target_slug").and_then(Value::as_str),
        Some("ghost")
    );
    assert!(
        row.get("target_id").is_some_and(Value::is_null),
        "broken row must carry target_id=null"
    );
    Ok(())
}

// ── write-time warning on an unresolvable reference (kb#11) ────────────

const GHOST_BODY: &str = "* Source\n\nSee [[nosuch]] for context.\n";

#[test]
fn create_warns_on_an_unresolvable_reference_and_still_succeeds() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");

    let (st, out, err) = run_kb(&db, Some(GHOST_BODY), &["create", "--id", "n1"])?;

    // The exit status is load-bearing: the `session-end-kb` hook runs
    // `kb create --id <session-id>` and falls back to `kb update` when it
    // fails. A warning must not look like a failure.
    assert!(st.success(), "create should still succeed, stderr: {err}");
    assert!(err.contains("[[nosuch]]"), "stderr was: {err}");
    assert!(err.contains("#+name:"), "the rule should be stated: {err}");
    assert!(!out.trim().is_empty(), "the node view still goes to stdout");

    // And the node really is stored.
    let (get_st, get_out, _) = run_kb(&db, None, &["get", "n1"])?;
    assert!(get_st.success());
    assert!(get_out.contains("nosuch"), "body lost: {get_out}");
    Ok(())
}

/// A warning that fired on every write would train its reader to ignore it.
#[test]
fn create_says_nothing_when_every_reference_resolves() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");
    create_node(&db, "#+name: alpha\n* Target\n\nI am alpha.\n", "target")?;

    let (st, _, err) = run_kb(
        &db,
        Some("* Source\n\nSee [[alpha]] for context.\n"),
        &["create", "--id", "n1"],
    )?;

    assert!(st.success(), "create failed: {err}");
    assert!(err.trim().is_empty(), "expected silence, got: {err}");
    Ok(())
}

/// `update` replaces the whole document and can introduce a broken
/// reference exactly as `create` can.
#[test]
fn update_warns_on_an_unresolvable_reference() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");
    create_node(&db, "* Source\n\nnothing linked yet.\n", "n1")?;

    let (st, _, err) = run_kb(&db, Some(GHOST_BODY), &["update", "n1"])?;

    assert!(st.success(), "update should still succeed, stderr: {err}");
    assert!(err.contains("[[nosuch]]"), "stderr was: {err}");
    Ok(())
}

/// The warning goes to stderr precisely so that `--json` consumers are
/// unaffected. A caller parsing stdout must not have to strip it.
#[test]
fn the_warning_does_not_contaminate_json_output() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");

    let (st, out, err) = run_kb(&db, Some(GHOST_BODY), &["create", "--id", "n1", "--json"])?;

    assert!(st.success(), "create failed: {err}");
    assert!(err.contains("[[nosuch]]"), "the warning still fires: {err}");
    let parsed: Value = serde_json::from_str(out.trim())?;
    assert!(parsed.get("data").is_some(), "stdout was: {out}");
    Ok(())
}

/// A bare UUID is a *name*-link searching for a node named after that
/// UUID, not an id-link. It is one of the two mistakes that produced every
/// broken reference in the live corpus, and it now announces itself.
#[test]
fn a_uuid_without_the_id_prefix_warns_like_any_other_unresolved_slug() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");
    let uuid = "11111111-1111-4111-8111-111111111111";

    let (st, _, err) = run_kb(
        &db,
        Some(&format!("* Source\n\nSee [[{uuid}]].\n")),
        &["create", "--id", "n1"],
    )?;

    assert!(st.success(), "create failed: {err}");
    assert!(err.contains(uuid), "stderr was: {err}");
    assert!(
        err.contains("[[id:<uuid>]]"),
        "the remedy for this exact mistake should be named: {err}"
    );
    Ok(())
}
