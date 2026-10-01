//! Behavior of the content-addressed blob store, exercised against real
//! temporary git repositories rather than a mock: the store's whole purpose is
//! to be the archival record, and a mock would assert only that this file
//! agrees with itself.

use kb::store::{BlobStore, GitBlobStore, RefName, StoreError};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// The archival guarantee, in its smallest form: what went in comes back.
#[test]
fn stored_bytes_come_back_unchanged() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(dir.path())?;
    let bytes = b"* Note\n\nthe archival bytes.\n";

    let hash = store.put(bytes)?;

    assert_eq!(store.get(&hash)?, bytes.to_vec());
    Ok(())
}

/// A record's public identity is its ref, never its hash: the ground-truth
/// question set and the `SessionEnd` hook both key on stable ids, and content
/// that changes must not change what a record is called.
#[test]
fn a_record_keeps_its_name_when_its_content_changes() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(dir.path())?;
    let name = RefName::new("nodes/b70049ea")?;

    let first = store.put(b"* Note\n\nas first written.\n")?;
    store.set_ref(&name, &first)?;
    let second = store.put(b"* Note\n\nas revised.\n")?;
    store.set_ref(&name, &second)?;

    assert_ne!(first, second, "the revision must be a different object");
    assert_eq!(
        store.read_ref(&name)?,
        second,
        "the ref follows the content"
    );
    assert_eq!(
        store.get(&first)?,
        b"* Note\n\nas first written.\n".to_vec(),
        "the superseded version is still addressable — nothing is overwritten"
    );
    Ok(())
}

/// Deduplication is the property that makes an appended transcript cheap. It
/// comes from content addressing rather than from kb, so this test guards a
/// backend property kb depends on rather than logic kb implements.
#[test]
fn storing_identical_bytes_twice_yields_one_object() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(dir.path())?;
    let bytes = b"* Transcript\n\nturn one.\n";

    let first = store.put(bytes)?;
    let second = store.put(bytes)?;

    assert_eq!(first, second);
    assert_eq!(count_loose_objects(dir.path())?, 1, "one object on disk");
    Ok(())
}

/// Reopening is how every later process reaches the store; if it silently
/// initialized a second empty repository instead, the corpus would appear to
/// vanish.
#[test]
fn reopening_the_store_finds_what_was_written() -> TestResult {
    let dir = tempfile::tempdir()?;
    let name = RefName::new("nodes/n1")?;
    let hash = {
        let store = GitBlobStore::open_or_init(dir.path())?;
        let hash = store.put(b"persisted.\n")?;
        store.set_ref(&name, &hash)?;
        hash
    };

    let reopened = GitBlobStore::open_or_init(dir.path())?;

    assert_eq!(reopened.read_ref(&name)?, hash);
    assert_eq!(reopened.get(&hash)?, b"persisted.\n".to_vec());
    Ok(())
}

/// An unbound name is a domain condition callers act on, not a crash.
#[test]
fn reading_an_unbound_ref_names_the_ref() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(dir.path())?;

    let outcome = store.read_ref(&RefName::new("nodes/never-written")?);

    assert!(
        matches!(&outcome, Err(StoreError::RefNotFound { name }) if name == "nodes/never-written"),
        "an unbound ref must not resolve; got: {outcome:?}"
    );
    Ok(())
}

/// A hash that addresses nothing is reported as such rather than as a bare
/// I/O failure the caller has to interpret.
#[test]
fn getting_an_absent_object_names_the_hash() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(dir.path())?;
    let absent = store.put(b"x")?;
    let other = GitBlobStore::open_or_init(tempfile::tempdir()?.path())?;

    let outcome = other.get(&absent);

    assert!(
        matches!(&outcome, Err(StoreError::ObjectNotFound { hash }) if hash == absent.as_str()),
        "an object stored elsewhere must not resolve here; got: {outcome:?}"
    );
    Ok(())
}

fn count_loose_objects(root: &std::path::Path) -> Result<usize, Box<dyn std::error::Error>> {
    let mut count = 0;
    for shard in std::fs::read_dir(root.join("objects"))? {
        let shard = shard?;
        let name = shard.file_name();
        let name = name.to_string_lossy();
        if name == "info" || name == "pack" {
            continue;
        }
        count += std::fs::read_dir(shard.path())?.count();
    }
    Ok(count)
}

/// Rebuilding the index means walking what the store holds, so the store has
/// to be able to say. Enumeration is by name prefix rather than wholesale,
/// because corpora are separately rebuildable and a mail corpus will dwarf
/// the rest.
#[test]
fn the_store_enumerates_the_names_it_holds() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(dir.path())?;
    let hash = store.put(b"content.\n")?;
    for name in ["nodes/n1", "nodes/n2", "mail/m1"] {
        store.set_ref(&RefName::new(name)?, &hash)?;
    }

    let nodes = store.list_refs("nodes/")?;
    let all = store.list_refs("")?;

    assert_eq!(
        nodes.iter().map(RefName::as_str).collect::<Vec<_>>(),
        vec!["nodes/n1", "nodes/n2"],
        "prefix filtering, in a stable order"
    );
    assert_eq!(all.len(), 3, "an empty prefix is everything: {all:?}");
    Ok(())
}

/// An empty store enumerates to nothing rather than failing: a first rebuild
/// runs against exactly that.
#[test]
fn an_empty_store_enumerates_to_nothing() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(dir.path())?;

    assert!(store.list_refs("")?.is_empty());
    Ok(())
}
