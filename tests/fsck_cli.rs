//! `kb fsck` at the process boundary (CLI-002).
//!
//! What a reconciliation command says is most of what it is worth. A count of
//! drifted records tells an operator that something is wrong and nothing about
//! what, so these tests hold the output to naming the records, distinguishing
//! the kinds of drift, and saying plainly which kind it will not repair.

use std::path::Path;
use std::process::{Command, Stdio};

use kb::record::{ArtifactKind, ContentHash, CorpusId, RecordHeader, RecordId, SourceRef};
use kb::store::{BlobStore, GitBlobStore, RefName};

mod common;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn store_note(store: &GitBlobStore, id: &str, body: &str) -> TestResult {
    let raw = format!("* Note {id}\n\n{body}\n");
    let stream = kb::record::normalize(ArtifactKind::Note, raw.as_bytes())?;
    let header = RecordHeader {
        id: RecordId::new(id)?,
        corpus: CorpusId::new("kb")?,
        kind: ArtifactKind::Note,
        created: "2026-05-19T09:00:00Z".parse()?,
        updated: "2026-08-14T10:30:00Z".parse()?,
        source: SourceRef::new("node-id", id)?,
        tags: vec![],
        raw: ContentHash::new(store.put(raw.as_bytes())?.as_str())?,
        stream: ContentHash::new(store.put(stream.as_bytes())?.as_str())?,
        normalizer: kb::record::NORMALIZER_VERSION,
        provenance: kb::record::Provenance::default(),
    };
    let hash = store.put(&header.serialize())?;
    store.set_ref(&RefName::new(&format!("kb/{id}"))?, &hash)?;
    Ok(())
}

/// A sandboxed home with a store of two notes and an index built from it.
fn prepared() -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let home = tempfile::tempdir()?;
    let root = home.path().join(".local/share/kb");
    std::fs::create_dir_all(&root)?;
    let store = GitBlobStore::open_or_init(&root.join("store"))?;
    store_note(&store, "alpha", "the first body")?;
    store_note(&store, "beta", "the second body")?;
    let index = kb::index::Index::open_for_rebuild(&root.join("index.db"))?;
    kb::index::rebuild(&store, &index, &kb::index::Scope::All)?;
    Ok(home)
}

fn store_at(home: &Path) -> Result<GitBlobStore, Box<dyn std::error::Error>> {
    Ok(GitBlobStore::open(&home.join(".local/share/kb/store"))?)
}

struct Run {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
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
        stdout: String::from_utf8(out.stdout)?,
        stderr: String::from_utf8(out.stderr)?,
    })
}

#[test]
fn a_clean_store_reports_agreement_and_changes_nothing() -> TestResult {
    let home = prepared()?;
    let run = run_kb(home.path(), &["fsck", "--no-legacy"])?;
    assert!(run.status.success(), "{}{}", run.stdout, run.stderr);
    assert!(
        run.stdout.contains("checked 2 stored records"),
        "the survey did not say what it checked: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("the index agrees with the store"),
        "a clean store was not reported as clean: {}",
        run.stdout
    );
    assert!(
        !run.stdout.contains("--repair"),
        "a clean store should not suggest a repair: {}",
        run.stdout
    );
    Ok(())
}

#[test]
fn drift_is_named_by_record_and_repaired_only_when_asked() -> TestResult {
    let home = prepared()?;
    store_note(&store_at(home.path())?, "beta", "a body it did not have")?;

    let reported = run_kb(home.path(), &["fsck", "--no-legacy"])?;
    assert!(reported.status.success(), "{}", reported.stderr);
    assert!(
        reported.stdout.contains("indexed from different content"),
        "the drift was not classified: {}",
        reported.stdout
    );
    assert!(
        reported.stdout.contains("beta"),
        "the drifted record was not named: {}",
        reported.stdout
    );
    assert!(
        reported.stdout.contains("--repair"),
        "the remedy was not offered: {}",
        reported.stdout
    );

    // Reporting wrote nothing, so the same drift is still there.
    let again = run_kb(home.path(), &["fsck", "--no-legacy"])?;
    assert!(
        again.stdout.contains("indexed from different content"),
        "reporting repaired something: {}",
        again.stdout
    );

    let repaired = run_kb(home.path(), &["fsck", "--no-legacy", "--repair"])?;
    assert!(repaired.status.success(), "{}", repaired.stderr);
    assert!(
        repaired.stdout.contains("re-derived 1 record"),
        "the repair did not report its work: {}",
        repaired.stdout
    );

    let after = run_kb(home.path(), &["fsck", "--no-legacy"])?;
    assert!(
        after.stdout.contains("the index agrees with the store"),
        "the repair did not converge: {}",
        after.stdout
    );
    Ok(())
}

/// The acceptance property, through the binary: a second pass over an
/// unchanged store has nothing to do.
#[test]
fn repairing_a_clean_store_twice_does_nothing_both_times() -> TestResult {
    let home = prepared()?;
    for _ in 0..2 {
        let run = run_kb(home.path(), &["fsck", "--no-legacy", "--repair"])?;
        assert!(run.status.success(), "{}", run.stderr);
        assert!(
            run.stdout.contains("re-derived 0 records"),
            "a clean store was re-derived: {}",
            run.stdout
        );
        assert!(
            run.stdout.contains("dropped 0 records"),
            "a clean store lost rows: {}",
            run.stdout
        );
    }
    Ok(())
}

/// Nodes the superseded database holds and the store has never seen are
/// reported with the remedy named, and left exactly where they are: importing
/// them is an archival write, not a derivation.
#[test]
fn unarchived_nodes_are_reported_with_their_remedy() -> TestResult {
    let home = prepared()?;
    let legacy = common::legacy_db(&home.path().join("kb.db"))?;
    common::legacy_insert(
        &legacy,
        "cc-never-archived",
        &tftio_org::ast::Document {
            blocks: vec![tftio_org::ast::Block::Heading {
                level: 1,
                title: tftio_org::ast::Title("a session capture".into()),
                tags: vec![],
                children: vec![],
            }],
        },
    )?;
    drop(legacy);

    let run = run_kb(home.path(), &["fsck", "--repair"])?;
    assert!(run.status.success(), "{}{}", run.stdout, run.stderr);
    assert!(
        run.stdout.contains("never archived to the store"),
        "the unarchived node was not reported: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("cc-never-archived"),
        "the unarchived node was not named: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("kb export"),
        "the remedy was not named: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("re-derived 0 records"),
        "an unarchived node was treated as repairable: {}",
        run.stdout
    );
    Ok(())
}

/// A machine that has finished the migration has no legacy database, and
/// should not be told it has a problem for that.
#[test]
fn an_absent_legacy_database_is_not_a_finding() -> TestResult {
    let home = prepared()?;
    let run = run_kb(home.path(), &["fsck"])?;
    assert!(run.status.success(), "{}{}", run.stdout, run.stderr);
    assert!(
        !run.stdout.contains("never archived"),
        "an absent legacy database was reported as drift: {}",
        run.stdout
    );
    Ok(())
}

/// The deep survey re-normalizes rather than comparing addresses, and reports
/// the two kinds of difference apart: content the normalizer would no longer
/// produce, and a version banner that moved for some other artifact kind.
#[test]
fn a_deep_survey_reports_content_and_version_separately() -> TestResult {
    let home = prepared()?;
    let store = store_at(home.path())?;
    let raw = "* Note behind\n\na body\n";
    let current = kb::record::normalize(kb::record::ArtifactKind::Note, raw.as_bytes())?;
    let mut older = b"kb-stream/1\n".to_vec();
    older.extend_from_slice(current.payload());
    let header = RecordHeader {
        id: RecordId::new("behind")?,
        corpus: CorpusId::new("kb")?,
        kind: ArtifactKind::Note,
        created: "2026-05-19T09:00:00Z".parse()?,
        updated: "2026-08-14T10:30:00Z".parse()?,
        source: SourceRef::new("node-id", "behind")?,
        tags: vec![],
        raw: ContentHash::new(store.put(raw.as_bytes())?.as_str())?,
        stream: ContentHash::new(store.put(&older)?.as_str())?,
        normalizer: kb::record::NORMALIZER_VERSION,
        provenance: kb::record::Provenance::default(),
    };
    let hash = store.put(&header.serialize())?;
    store.set_ref(&RefName::new("kb/behind")?, &hash)?;

    // Index it first, so the only thing left to report about it is its
    // normalizer version rather than its absence.
    let indexed = run_kb(home.path(), &["fsck", "--no-legacy", "--repair"])?;
    assert!(
        indexed.status.success(),
        "{}{}",
        indexed.stdout,
        indexed.stderr
    );

    let run = run_kb(home.path(), &["fsck", "--no-legacy", "--deep"])?;
    assert!(run.status.success(), "{}{}", run.stdout, run.stderr);
    assert!(
        run.stdout.contains("the index agrees with the store"),
        "the fixture was not reconciled first: {}",
        run.stdout
    );
    assert!(
        run.stdout
            .contains("carries the content the current normalizer produces"),
        "a content-identical corpus was reported as stale: {}",
        run.stdout
    );
    assert!(
        run.stdout
            .contains("1 record normalized under an older version"),
        "the version lag was not counted: {}",
        run.stdout
    );
    assert!(
        !run.stdout.contains("    behind"),
        "a record behind only in version was named: {}",
        run.stdout
    );
    Ok(())
}

/// A first run after a schema change can list thousands. Naming every one
/// would produce a report nobody reads, so the list is bounded and the
/// remainder counted.
#[test]
fn a_long_list_of_drift_is_bounded_and_the_rest_counted() -> TestResult {
    let home = prepared()?;
    let store = store_at(home.path())?;
    for n in 0..25 {
        store_note(
            &store,
            &format!("added-{n:02}"),
            "a body added after indexing",
        )?;
    }
    let run = run_kb(home.path(), &["fsck", "--no-legacy"])?;
    assert!(run.status.success(), "{}{}", run.stdout, run.stderr);
    assert!(
        run.stdout.contains("25 records absent from the index"),
        "the count was not reported: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("... and 5 more"),
        "the list was not bounded: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("added-00") && !run.stdout.contains("added-24"),
        "the wrong end of the list was shown: {}",
        run.stdout
    );
    Ok(())
}

/// A stored stream whose content the normalizer would no longer produce is
/// real staleness, and is named rather than counted — unlike a version banner
/// that moved for some other artifact kind.
#[test]
fn a_deep_survey_names_a_record_whose_content_would_differ() -> TestResult {
    let home = prepared()?;
    let store = store_at(home.path())?;
    let raw = "* Note wrong\n\na body\n";
    let header = RecordHeader {
        id: RecordId::new("wrong")?,
        corpus: CorpusId::new("kb")?,
        kind: ArtifactKind::Note,
        created: "2026-05-19T09:00:00Z".parse()?,
        updated: "2026-08-14T10:30:00Z".parse()?,
        source: SourceRef::new("node-id", "wrong")?,
        tags: vec![],
        raw: ContentHash::new(store.put(raw.as_bytes())?.as_str())?,
        stream: ContentHash::new(
            store
                .put(b"kb-stream/3\ntext this record does not contain\n")?
                .as_str(),
        )?,
        normalizer: kb::record::NORMALIZER_VERSION,
        provenance: kb::record::Provenance::default(),
    };
    let hash = store.put(&header.serialize())?;
    store.set_ref(&RefName::new("kb/wrong")?, &hash)?;

    let run = run_kb(home.path(), &["fsck", "--no-legacy", "--deep"])?;
    assert!(run.status.success(), "{}{}", run.stdout, run.stderr);
    assert!(
        run.stdout
            .contains("1 record whose stored stream is not what the current normalizer produces"),
        "the stale record was not reported: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("    wrong"),
        "the stale record was not named: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("archival store"),
        "the reason it is not repaired was not given: {}",
        run.stdout
    );
    Ok(())
}
