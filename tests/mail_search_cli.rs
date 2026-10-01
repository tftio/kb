//! `kb search --corpus mail` at the process boundary (CLI-002).
//!
//! The mail corpus is served by two backends that live outside the kb
//! database — mu's Xapian index for lexical retrieval and the derived index
//! for vectors — so what is worth testing through the real binary is that the
//! CLI dispatches to them at all, reports what they returned, and names a
//! message by its subject rather than by the identifier it is addressed with.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

use kb::index::{Index, rebuild_mail};
use kb::maildir::{MaildirCorpus, Selection};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Variables that would otherwise leak a developer's live daemons into a test
/// process and make ranking machine-dependent.
const ISOLATED_ENV: [&str; 8] = [
    "KB_EMBEDDING_BASE_URL",
    "KB_EMBEDDING_MODEL",
    "KB_EMBEDDING_API_KEY",
    "KB_EMBEDDING_DOCUMENT_PREFIX",
    "KB_EMBEDDING_QUERY_PREFIX",
    "KB_EMBEDDING_MIN_SIMILARITY",
    "KB_RERANK_BASE_URL",
    "KB_RERANK_TOP_K",
];

fn deliver(root: &Path, uniq: &str, message_id: &str, subject: &str, body: &str) -> TestResult {
    let dir = root.join("Inbox").join("cur");
    fs::create_dir_all(&dir)?;
    fs::write(
        dir.join(format!("{uniq}:2,S")),
        format!(
            "From: Ada <ada@example.invalid>\n\
             To: reader@example.invalid\n\
             Subject: {subject}\n\
             Date: Mon, 3 Feb 2025 09:00:00 +0000\n\
             Message-ID: {message_id}\n\
             \n\
             {body}\n"
        ),
    )?;
    Ok(())
}

/// A sandboxed home holding a derived index over a two-message Maildir.
fn prepared() -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let home = tempfile::tempdir()?;
    let maildir = home.path().join("mail");
    deliver(
        &maildir,
        "one",
        "<one@example.invalid>",
        "quarterly revenue figures",
        "the revenue figures for the quarter are attached",
    )?;
    deliver(
        &maildir,
        "two",
        "<two@example.invalid>",
        "lunch on thursday",
        "shall we get lunch on thursday",
    )?;
    let corpus = MaildirCorpus::scan(&maildir)?;
    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &BTreeSet::new());
    let index_path = home.path().join(".local/share/kb/index.db");
    fs::create_dir_all(index_path.parent().ok_or("index path has no parent")?)?;
    let index = Index::open_for_rebuild(&index_path)?;
    rebuild_mail(&corpus, &selection, &index)?;
    Ok(home)
}

struct Run {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

/// Run `kb --db <home>/kb.db <args...>` in a sandboxed home, with `env`
/// layered on. `MUHOME` is how a caller points kb at a mu index other than
/// the default one, which is also how these tests avoid reading the
/// developer's own mail.
fn run_kb_env(
    home: &Path,
    args: &[&str],
    env: &[(&str, &Path)],
) -> Result<Run, Box<dyn std::error::Error>> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_kb"));
    cmd.env("HOME", home)
        .env("XDG_CONFIG_HOME", home)
        .arg("--db")
        .arg(home.join("kb.db"));
    for (key, value) in env {
        cmd.env(key, value);
    }
    for var in ISOLATED_ENV {
        cmd.env_remove(var);
    }
    for arg in args {
        cmd.arg(arg);
    }
    let out = cmd.stdin(Stdio::null()).output()?;
    Ok(Run {
        status: out.status,
        stdout: String::from_utf8(out.stdout)?,
        stderr: String::from_utf8(out.stderr)?,
    })
}

/// Whether `mu` is installed.
fn mu_available() -> bool {
    Command::new("mu")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Build a mu index over the Maildir under `home`, returning its muhome.
fn mu_index(home: &Path) -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let muhome = tempfile::tempdir()?;
    for args in [
        vec![
            "init".to_owned(),
            "--muhome".to_owned(),
            muhome.path().display().to_string(),
            "--maildir".to_owned(),
            home.join("mail").display().to_string(),
        ],
        vec![
            "index".to_owned(),
            "--muhome".to_owned(),
            muhome.path().display().to_string(),
        ],
    ] {
        let out = Command::new("mu").args(&args).output()?;
        if !out.status.success() {
            return Err(format!(
                "mu {:?} failed: {}",
                args,
                String::from_utf8_lossy(&out.stderr)
            )
            .into());
        }
    }
    Ok(muhome)
}

#[test]
fn a_mail_search_dispatches_to_the_mail_signals() -> TestResult {
    if !mu_available() {
        eprintln!("skipping: mu not on PATH");
        return Ok(());
    }
    let home = prepared()?;
    let muhome = mu_index(home.path())?;
    let run = run_kb_env(
        home.path(),
        &["search", "revenue", "--corpus", "mail", "--explain"],
        &[("MUHOME", muhome.path())],
    )?;
    assert!(run.status.success(), "{}{}", run.stdout, run.stderr);
    assert!(
        run.stdout.contains("mail-lexical (lexical, mail)"),
        "the mail lexical signal did not run: {}",
        run.stdout
    );
    assert!(
        !run.stdout.contains("kb-lexical"),
        "a mail query ran a kb signal: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("<one@example.invalid>"),
        "mu found nothing through the CLI: {}",
        run.stdout
    );
    Ok(())
}

/// With mu absent and no embedding endpoint, every signal a mail query has
/// failed — and an empty result would say the archive holds nothing on the
/// subject, which is the claim ENG-004 exists to prevent.
#[test]
fn a_mail_search_with_no_backend_at_all_fails_rather_than_answering_empty() -> TestResult {
    let home = prepared()?;
    let absent = home.path().join("no-such-muhome");
    let run = run_kb_env(
        home.path(),
        &["search", "revenue", "--corpus", "mail"],
        &[("MUHOME", absent.as_path())],
    )?;
    assert!(
        !run.status.success(),
        "a search with no working backend reported success: {}",
        run.stdout
    );
    Ok(())
}

/// Without a reachable embedding endpoint there is no query vector, so the
/// dense signal is not built at all. The lexical half still answers, which is
/// the degradation `kb search` has always made rather than failing.
#[test]
fn a_mail_search_without_an_endpoint_still_reaches_mu() -> TestResult {
    if !mu_available() {
        eprintln!("skipping: mu not on PATH");
        return Ok(());
    }
    let home = prepared()?;
    let muhome = mu_index(home.path())?;
    let run = run_kb_env(
        home.path(),
        &["search", "revenue", "--corpus", "mail"],
        &[("MUHOME", muhome.path())],
    )?;
    assert!(run.status.success(), "{}{}", run.stdout, run.stderr);
    assert!(
        run.stderr.contains("keyword-only ranking"),
        "the missing vector half was not explained: {}",
        run.stderr
    );
    Ok(())
}

/// A message is named by its subject. The identifier it is addressed by is
/// an angle-bracketed `Message-ID`, which tells a reader nothing.
#[test]
fn a_mail_hit_is_named_by_its_subject() -> TestResult {
    if !mu_available() {
        eprintln!("skipping: mu not on PATH");
        return Ok(());
    }
    let home = prepared()?;
    let index = Index::open(&home.path().join(".local/share/kb/index.db"))?;
    let ids = index.records_in("mail")?;
    assert!(
        ids.contains(&"<one@example.invalid>".to_owned()),
        "the fixture was not indexed: {ids:?}"
    );
    let muhome = mu_index(home.path())?;
    let run = run_kb_env(
        home.path(),
        &["search", "revenue", "--corpus", "mail", "--json"],
        &[("MUHOME", muhome.path())],
    )?;
    assert!(run.status.success(), "{}{}", run.stdout, run.stderr);
    if run.stdout.contains("one@example.invalid") {
        assert!(
            run.stdout.contains("quarterly revenue figures"),
            "the hit was not named by its subject: {}",
            run.stdout
        );
    }
    Ok(())
}
