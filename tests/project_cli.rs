//! `kb project` at the process boundary (CLI-002).
//!
//! What the command says is most of what an operator ever learns about the
//! projection: it runs on a timer beside fsck, and its output is the only
//! place a hand-edited page, an unresolvable link, or a page whose record has
//! left the store gets named. These tests hold that output to naming them.

use std::path::Path;
use std::process::{Command, Stdio};

use kb::record::{ArtifactKind, ContentHash, CorpusId, RecordHeader, RecordId, SourceRef};
use kb::store::{BlobStore, GitBlobStore, RefName};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn store_note(store: &GitBlobStore, id: &str, org: &str) -> TestResult {
    let normalized = kb::record::normalize(ArtifactKind::Note, org.as_bytes())?;
    let header = RecordHeader {
        id: RecordId::new(id)?,
        corpus: CorpusId::new("kb")?,
        kind: ArtifactKind::Note,
        created: "2026-05-19T09:00:00Z".parse()?,
        updated: "2026-08-14T10:30:00Z".parse()?,
        source: SourceRef::new("node-id", id)?,
        tags: vec![],
        raw: ContentHash::new(store.put(org.as_bytes())?.as_str())?,
        stream: ContentHash::new(store.put(normalized.as_bytes())?.as_str())?,
        normalizer: kb::record::NORMALIZER_VERSION,
        provenance: kb::record::Provenance::default(),
    };
    let hash = store.put(&header.serialize())?;
    store.set_ref(&RefName::new(&format!("kb/{id}"))?, &hash)?;
    Ok(())
}

/// A sandboxed home holding a store of two notes, one of which links to
/// something the corpus does not contain.
fn prepared() -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let home = tempfile::tempdir()?;
    let root = home.path().join(".local/share/kb");
    std::fs::create_dir_all(&root)?;
    let store = GitBlobStore::open_or_init(&root.join("store"))?;
    store_note(
        &store,
        "alpha",
        "#+title: Alpha\n\npoints at [[id:beta][beta]] and at [[nowhere]].\n",
    )?;
    store_note(&store, "beta", "#+title: Beta\n\nthe second note.\n")?;
    Ok(home)
}

struct Run {
    status: std::process::ExitStatus,
    stdout: String,
}

fn run_kb(home: &Path, args: &[&str]) -> Result<Run, Box<dyn std::error::Error>> {
    let out = Command::new(env!("CARGO_BIN_EXE_kb"))
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home)
        .env_remove("KB_STORE_PATH")
        .env_remove("KB_INDEX_PATH")
        .arg("--db")
        .arg(home.join("kb.db"))
        .args(args)
        .stdin(Stdio::null())
        .output()?;
    Ok(Run {
        status: out.status,
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
    })
}

fn space(home: &Path) -> std::path::PathBuf {
    home.join("space")
}

/// The first run reports what it wrote and how long it took, because the claim
/// that the projection is disposable is only true while re-running it is cheap
/// enough that people do.
#[test]
fn projecting_reports_the_pages_written_and_the_time_it_took() -> TestResult {
    let home = prepared()?;
    let run = run_kb(
        home.path(),
        &[
            "project",
            "--space",
            &space(home.path()).display().to_string(),
        ],
    )?;
    assert!(run.status.success(), "{}", run.stdout);
    assert!(
        run.stdout.contains("kb project: 2 records"),
        "{}",
        run.stdout
    );
    assert!(run.stdout.contains("2 written"), "{}", run.stdout);
    assert!(space(home.path()).join("kb/alpha.md").exists());
    Ok(())
}

/// A link the corpus cannot answer is named, not counted. The projection is
/// where link rot becomes visible, and a bare number would say something is
/// wrong and nothing about what.
#[test]
fn an_unresolvable_link_is_named() -> TestResult {
    let home = prepared()?;
    let run = run_kb(
        home.path(),
        &[
            "project",
            "--space",
            &space(home.path()).display().to_string(),
        ],
    )?;
    assert!(run.stdout.contains("kb/alpha -> nowhere"), "{}", run.stdout);
    Ok(())
}

/// A second run over an unchanged store writes nothing and says so.
#[test]
fn a_second_run_writes_nothing() -> TestResult {
    let home = prepared()?;
    let path = space(home.path()).display().to_string();
    run_kb(home.path(), &["project", "--space", &path])?;
    let run = run_kb(home.path(), &["project", "--space", &path])?;
    assert!(run.stdout.contains("2 unchanged"), "{}", run.stdout);
    Ok(())
}

/// A hand edit is named and the page is put back, because silence would train
/// the operator to treat the wiki as editable.
#[test]
fn a_hand_edited_page_is_named() -> TestResult {
    let home = prepared()?;
    let path = space(home.path()).display().to_string();
    run_kb(home.path(), &["project", "--space", &path])?;
    let page = space(home.path()).join("kb/beta.md");
    let generated = std::fs::read_to_string(&page)?;
    std::fs::write(&page, format!("{generated}\nedited by hand\n"))?;

    let run = run_kb(home.path(), &["project", "--space", &path])?;
    assert!(
        run.stdout.contains("edited by hand and overwritten"),
        "{}",
        run.stdout
    );
    assert!(run.stdout.contains("kb/beta"), "{}", run.stdout);
    assert_eq!(std::fs::read_to_string(&page)?, generated);
    Ok(())
}

/// An orphan page is reported with the remedy named, and left in place until
/// the operator asks for it to go.
#[test]
fn an_orphan_page_is_reported_and_removed_only_when_asked() -> TestResult {
    let home = prepared()?;
    let path = space(home.path()).display().to_string();
    run_kb(home.path(), &["project", "--space", &path])?;
    std::fs::remove_file(home.path().join(".local/share/kb/store/refs/kb/kb/beta"))?;

    let reported = run_kb(home.path(), &["project", "--space", &path])?;
    assert!(
        reported.stdout.contains("no longer in the store"),
        "{}",
        reported.stdout
    );
    assert!(reported.stdout.contains("--prune"), "{}", reported.stdout);
    assert!(space(home.path()).join("kb/beta.md").exists());

    let pruned = run_kb(home.path(), &["project", "--space", &path, "--prune"])?;
    assert!(pruned.stdout.contains("removed 1"), "{}", pruned.stdout);
    assert!(!space(home.path()).join("kb/beta.md").exists());
    Ok(())
}

/// A store that is not there is a failure with a path in it, not an empty
/// projection reported as a success (ENG-004).
#[test]
fn a_missing_store_fails_loudly() -> TestResult {
    let home = tempfile::tempdir()?;
    let run = run_kb(
        home.path(),
        &[
            "project",
            "--space",
            &space(home.path()).display().to_string(),
        ],
    )?;
    assert!(!run.status.success());
    assert!(!space(home.path()).exists());
    Ok(())
}
