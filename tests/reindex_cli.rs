//! `kb reindex` at the process boundary. Its console output is the surface an
//! operator reads to decide whether rebuilding is cheap enough to keep doing,
//! so it is covered here per CLI-002.

use kb::record::{ArtifactKind, ContentHash, CorpusId, RecordHeader, RecordId, SourceRef};
use kb::store::{BlobStore, GitBlobStore, RefName};
use std::path::Path;
use std::process::Command;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn run_kb(
    home: &Path,
    args: &[&str],
) -> Result<(bool, String, String), Box<dyn std::error::Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_kb"))
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home)
        .env_remove("KB_EMBEDDING_BASE_URL")
        .env_remove("KB_EMBEDDING_MODEL")
        .env_remove("KB_MAIL_ROOT")
        .args(args)
        .output()?;
    Ok((
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

fn store_note(store: &GitBlobStore, id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let raw = format!("* Note {id}\n\nCorpus content for {id}.\n");
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

/// A hand-built `kb-record/1` blob: no provenance lines at all, the exact
/// shape every record written before T005 has. `RecordHeader::serialize`
/// always writes the current format now, so a legacy blob has to be built by
/// hand to exist in a fixture store at all.
fn store_legacy_note(store: &GitBlobStore, id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let raw = format!("* Legacy {id}\n\nCorpus content for {id}.\n");
    let stream = kb::record::normalize(ArtifactKind::Note, raw.as_bytes())?;
    let raw_hash = store.put(raw.as_bytes())?;
    let stream_hash = store.put(stream.as_bytes())?;
    let blob = format!(
        "kb-record/1\n\
         id: {id}\n\
         corpus: kb\n\
         kind: note\n\
         created: 2026-05-19T09:00:00Z\n\
         updated: 2026-08-14T10:30:00Z\n\
         source: node-id:{id}\n\
         raw: {raw_hash}\n\
         stream: {stream_hash}\n\
         normalizer: {}\n",
        kb::record::NORMALIZER_VERSION,
    );
    let hash = store.put(blob.as_bytes())?;
    store.set_ref(&RefName::new(&format!("kb/{id}"))?, &hash)?;
    Ok(())
}

/// `kb reindex` must succeed over a store mixing `kb-record/1` blobs (no
/// provenance lines) and `kb-record/2` blobs (asserting provenance), and it
/// must fill the new provenance columns for the ones that carry it while
/// leaving them null for the ones that do not (ST-001).
#[test]
fn reindex_succeeds_over_a_store_mixing_format_one_and_format_two_blobs() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store_path = dir.path().join("store");
    let store = GitBlobStore::open_or_init(&store_path)?;
    store_legacy_note(&store, "legacy")?;
    store_note(&store, "current")?;
    let index_path = dir.path().join("index.db");

    let (ok, stdout, stderr) = run_kb(
        dir.path(),
        &[
            "reindex",
            "--store",
            &store_path.display().to_string(),
            "--index",
            &index_path.display().to_string(),
        ],
    )?;

    assert!(ok, "reindex over a mixed store failed: {stderr}");
    assert!(stdout.contains("2 records"), "stdout was: {stdout}");

    let index = kb::index::Index::open(&index_path)?;
    let legacy = index.record("legacy")?;
    assert_eq!(
        legacy.project, None,
        "a format-1 blob must index null provenance"
    );
    let current = index.record("current")?;
    assert_eq!(
        current.project, None,
        "store_note asserts no provenance either"
    );
    Ok(())
}

/// The command reports what it did and how long it took. The elapsed time is
/// the load-bearing part: an index that is expensive to rebuild becomes
/// authoritative in practice no matter what the invariants say.
#[test]
fn reindex_reports_counts_and_elapsed_time() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store_path = dir.path().join("store");
    let store = GitBlobStore::open_or_init(&store_path)?;
    store_note(&store, "n1")?;
    store_note(&store, "n2")?;

    let (ok, stdout, stderr) = run_kb(
        dir.path(),
        &[
            "reindex",
            "--store",
            &store_path.display().to_string(),
            "--index",
            &dir.path().join("index.db").display().to_string(),
        ],
    )?;

    assert!(ok, "reindex failed: {stderr}");
    assert!(stdout.contains("2 records"), "stdout was: {stdout}");
    assert!(stdout.contains("passages"), "stdout was: {stdout}");
    assert!(
        stdout.contains("ms") || stdout.contains('s'),
        "no elapsed time reported: {stdout}"
    );
    Ok(())
}

/// Scope reaches the command line, because the operator's common case is one
/// edited note rather than the whole corpus.
#[test]
fn reindex_accepts_a_record_scope() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store_path = dir.path().join("store");
    let store = GitBlobStore::open_or_init(&store_path)?;
    store_note(&store, "n1")?;
    store_note(&store, "n2")?;

    let (ok, stdout, stderr) = run_kb(
        dir.path(),
        &[
            "reindex",
            "--store",
            &store_path.display().to_string(),
            "--index",
            &dir.path().join("index.db").display().to_string(),
            "--record",
            "n1",
        ],
    )?;

    assert!(ok, "reindex failed: {stderr}");
    assert!(stdout.contains("1 record"), "stdout was: {stdout}");
    Ok(())
}

/// A store that is not there is an operator error with an obvious cause, so
/// it is reported as one rather than as a panic or an empty success.
#[test]
fn reindex_against_a_missing_store_fails_and_names_the_path() -> TestResult {
    let dir = tempfile::tempdir()?;
    let missing = dir.path().join("no-such-store");

    let (ok, _, stderr) = run_kb(
        dir.path(),
        &[
            "reindex",
            "--store",
            &missing.display().to_string(),
            "--index",
            &dir.path().join("index.db").display().to_string(),
        ],
    )?;

    assert!(!ok, "a missing store must not report success");
    assert!(
        stderr.contains("no-such-store"),
        "the error must name the path: {stderr}"
    );
    Ok(())
}

/// Rebuilding an empty store is a legitimate no-op, not a failure: it is what
/// the first run against a fresh store does.
#[test]
fn reindex_of_an_empty_store_succeeds() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store_path = dir.path().join("store");
    GitBlobStore::open_or_init(&store_path)?;

    let (ok, stdout, stderr) = run_kb(
        dir.path(),
        &[
            "reindex",
            "--store",
            &store_path.display().to_string(),
            "--index",
            &dir.path().join("index.db").display().to_string(),
        ],
    )?;

    assert!(ok, "reindex failed: {stderr}");
    assert!(stdout.contains("0 records"), "stdout was: {stdout}");
    Ok(())
}
