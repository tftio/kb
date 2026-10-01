//! Writing a record into the store, which is where writes go since T029.
//!
//! Retrieval reads the derived index and the index is derived from the store,
//! so a write that lands anywhere else is a write nobody can find. These tests
//! hold the whole loop to account: what is written is stored, indexed,
//! searchable and readable, and writing the same identifier twice supersedes
//! rather than duplicates.

use kb::index::Index;
use kb::store::GitBlobStore;
use kb::write::{Written, put_record};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// A store and an index over it, both empty.
fn workspace() -> Result<(tempfile::TempDir, GitBlobStore, Index), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    let index = Index::open_for_rebuild(&dir.path().join("index.db"))?;
    Ok((dir, store, index))
}

/// The loop a caller depends on: write, then find, then read.
#[test]
fn a_written_record_is_stored_indexed_and_findable() -> TestResult {
    let (_dir, store, index) = workspace()?;
    let document = kb::parser::parse_document("* Ownership in Rust\n\nthe borrow checker.\n")?;
    let options = kb::write::WriteOptions::note("alpha")?;
    let written: Written = put_record(&store, &index, "alpha", &document, &options)?;
    assert_eq!(written.id, "alpha");

    let expression = kb::storage::build_fts_query("checker", kb::storage::MatchMode::Keywords);
    assert_eq!(
        index.search_text_in("kb", &expression, None, None)?,
        vec!["alpha"]
    );
    assert!(index.record_text("alpha")?.contains("borrow checker"));
    assert_eq!(index.record("alpha")?.title, "Ownership in Rust");
    Ok(())
}

/// Writing the same identifier again supersedes it. The store keeps both
/// blobs — that is what content addressing is for — but the name resolves to
/// the new record and the index describes only that one.
#[test]
fn writing_an_identifier_twice_supersedes_rather_than_duplicates() -> TestResult {
    let (_dir, store, index) = workspace()?;
    let options = kb::write::WriteOptions::note("beta")?;
    let first = kb::parser::parse_document("* First\n\noriginal body.\n")?;
    let before = put_record(&store, &index, "beta", &first, &options)?;
    let second = kb::parser::parse_document("* Second\n\nrevised body.\n")?;
    let after = put_record(&store, &index, "beta", &second, &options)?;
    assert_ne!(
        before.record_hash, after.record_hash,
        "a changed body left the record address unchanged"
    );

    assert_eq!(index.record("beta")?.title, "Second");
    let stale = kb::storage::build_fts_query("original", kb::storage::MatchMode::Keywords);
    assert!(
        index.search_text_in("kb", &stale, None, None)?.is_empty(),
        "the superseded text is still indexed"
    );
    let current = kb::storage::build_fts_query("revised", kb::storage::MatchMode::Keywords);
    assert_eq!(
        index.search_text_in("kb", &current, None, None)?,
        vec!["beta"]
    );
    Ok(())
}

/// The creation time survives a rewrite; only the update time moves.
///
/// A record's age is a fact about when it was first written, and losing it on
/// every edit would make `created` mean `last edited` — which is what
/// `updated` already means.
#[test]
fn rewriting_a_record_keeps_the_time_it_was_created() -> TestResult {
    let (_dir, store, index) = workspace()?;
    let options = kb::write::WriteOptions::note("gamma")?;
    let document = kb::parser::parse_document("* Note\n\nbody.\n")?;
    put_record(&store, &index, "gamma", &document, &options)?;
    let created = index.record("gamma")?.created;

    let revised = kb::parser::parse_document("* Note\n\nrevised body.\n")?;
    put_record(&store, &index, "gamma", &revised, &options)?;
    let row = index.record("gamma")?;
    assert_eq!(row.created, created, "the creation time was overwritten");
    assert!(
        row.updated >= created,
        "the update time went backwards: {} < {created}",
        row.updated
    );
    Ok(())
}

/// Adding a tag rewrites the record: new blobs, a new address, the name
/// advanced to it. There is no cheaper path in a content-addressed store,
/// and editing an index row instead would make the index authoritative.
#[test]
fn adding_a_tag_rewrites_the_record() -> TestResult {
    let (_dir, store, index) = workspace()?;
    let document = kb::parser::parse_document("* Note\n\nbody.\n")?;
    let options = kb::write::WriteOptions::note("n1")?;
    let before = kb::write::put_record(&store, &index, "n1", &document, &options)?;

    let edit = kb::write::add_tags(&store, &index, "n1", &["Book Club".to_owned()])?;
    assert_eq!(edit.changed, vec!["book-club".to_owned()]);
    assert_eq!(edit.tags, vec!["book-club".to_owned()]);
    assert_ne!(
        index.record("n1")?.record_hash.as_deref(),
        Some(before.record_hash.as_str()),
        "the record's address did not change"
    );
    assert_eq!(index.tags_of("n1")?, vec!["book-club".to_owned()]);
    Ok(())
}

/// A tag the record already carries is not an edit, and not a write.
#[test]
fn adding_a_tag_twice_writes_nothing_the_second_time() -> TestResult {
    let (_dir, store, index) = workspace()?;
    let document = kb::parser::parse_document("* Note\n\nbody.\n")?;
    let options = kb::write::WriteOptions::note("n1")?;
    kb::write::put_record(&store, &index, "n1", &document, &options)?;
    kb::write::add_tags(&store, &index, "n1", &["rust".to_owned()])?;
    let settled = index.record("n1")?.record_hash;

    let edit = kb::write::add_tags(&store, &index, "n1", &["rust".to_owned()])?;
    assert!(edit.changed.is_empty(), "{:?}", edit.changed);
    assert_eq!(index.record("n1")?.record_hash, settled);
    Ok(())
}

#[test]
fn removing_a_tag_takes_it_off_the_record() -> TestResult {
    let (_dir, store, index) = workspace()?;
    let document = kb::parser::parse_document("#+filetags: :rust:ownership:\n* Note\n\nbody.\n")?;
    let options = kb::write::WriteOptions::note("n1")?;
    kb::write::put_record(&store, &index, "n1", &document, &options)?;

    let edit = kb::write::remove_tags(&store, &index, "n1", &["rust".to_owned()])?;
    assert_eq!(edit.changed, vec!["rust".to_owned()]);
    assert_eq!(index.tags_of("n1")?, vec!["ownership".to_owned()]);

    let again = kb::write::remove_tags(&store, &index, "n1", &["rust".to_owned()])?;
    assert!(again.changed.is_empty(), "{:?}", again.changed);
    Ok(())
}

/// A merge rewrites every record carrying the old tag and reports which.
#[test]
fn merging_a_tag_rewrites_every_record_that_carries_it() -> TestResult {
    let (_dir, store, index) = workspace()?;
    for id in ["a", "b"] {
        let document =
            kb::parser::parse_document(&format!("#+filetags: :bookclub:\n* {id}\n\nbody.\n"))?;
        let options = kb::write::WriteOptions::note(id)?;
        kb::write::put_record(&store, &index, id, &document, &options)?;
    }
    let untouched = kb::parser::parse_document("* c\n\nbody.\n")?;
    let options = kb::write::WriteOptions::note("c")?;
    kb::write::put_record(&store, &index, "c", &untouched, &options)?;
    let before = index.record("c")?.record_hash;

    let merge = kb::write::merge_tag(&store, &index, "bookclub", "Book Club")?;
    assert_eq!(merge.from, "bookclub");
    assert_eq!(merge.to, "book-club");
    assert_eq!(merge.rewritten, vec!["a".to_owned(), "b".to_owned()]);
    assert_eq!(index.tags_of("a")?, vec!["book-club".to_owned()]);
    assert_eq!(
        index.record("c")?.record_hash,
        before,
        "a record that did not carry the tag was rewritten anyway"
    );
    Ok(())
}

/// The refusals: a tag that normalizes to nothing, and a merge onto itself.
/// Both are caught before anything is written, because a merge that has
/// already rewritten half a corpus is not something to discover afterwards.
#[test]
fn a_merge_that_means_nothing_is_refused_before_it_writes() -> TestResult {
    let (_dir, store, index) = workspace()?;
    let document = kb::parser::parse_document("#+filetags: :rust:\n* n\n\nbody.\n")?;
    let options = kb::write::WriteOptions::note("n1")?;
    kb::write::put_record(&store, &index, "n1", &document, &options)?;
    let before = index.record("n1")?.record_hash;

    for (from, to) in [("  ", "rust"), ("rust", "  "), ("rust", "Rust")] {
        let refused = kb::write::merge_tag(&store, &index, from, to);
        assert!(
            refused.is_err(),
            "merging {from:?} onto {to:?} was not refused"
        );
    }
    assert_eq!(index.record("n1")?.record_hash, before);

    let refused = kb::write::add_tags(&store, &index, "n1", &["   ".to_owned()]);
    assert!(refused.is_err(), "an empty tag was accepted");
    Ok(())
}

/// `tags add` on a session-transcript record leaves its kind, source and
/// provenance unchanged. Before T005, every rewrite through `put_record` — a
/// tag edit included — hardcoded `kind: note`, so retagging a captured
/// transcript silently demoted it and its passages moved from turns to
/// sections.
#[test]
fn tags_add_preserves_kind_source_and_provenance() -> TestResult {
    let (_dir, store, index) = workspace()?;
    let document = kb::parser::parse_document(
        "#+filetags: :conversation:\n\
         * Human [2026-09-23 09:00]\n\nQuestion.\n\n\
         * Assistant [2026-09-23 09:01]\n\nAnswer.\n",
    )?;
    let options = kb::write::WriteOptions {
        kind: kb::record::ArtifactKind::SessionTranscript,
        source: kb::record::SourceRef::new("session-id", "cc-9")?,
        provenance: kb::record::RawProvenance {
            project: Some("kb".to_owned()),
            harness: Some("claude-code".to_owned()),
            ..kb::record::RawProvenance::default()
        }
        .validate()?,
    };
    put_record(&store, &index, "cc-9", &document, &options)?;

    let before = index.record("cc-9")?;
    assert_eq!(before.kind, "session-transcript");
    assert_eq!(before.source, "session-id:cc-9");
    assert_eq!(before.project.as_deref(), Some("kb"));
    assert_eq!(before.harness.as_deref(), Some("claude-code"));
    let passages_before = index.passages("cc-9")?.len();
    assert_eq!(passages_before, 2, "a transcript chunks on its turns");

    kb::write::add_tags(&store, &index, "cc-9", &["standing".to_owned()])?;

    let after = index.record("cc-9")?;
    assert_eq!(after.kind, "session-transcript", "kind was not preserved");
    assert_eq!(after.source, "session-id:cc-9", "source was not preserved");
    assert_eq!(
        after.project.as_deref(),
        Some("kb"),
        "provenance was not preserved"
    );
    assert_eq!(after.harness.as_deref(), Some("claude-code"));
    assert_eq!(
        index.passages("cc-9")?.len(),
        passages_before,
        "the tag edit changed how the record chunks"
    );
    Ok(())
}

/// Reading a record the index does not hold says so rather than returning
/// an empty document that would then be written back over it.
#[test]
fn reading_an_unknown_record_is_an_error() -> TestResult {
    let (_dir, store, index) = workspace()?;
    assert!(kb::write::read_raw(&store, &index, "absent").is_err());
    assert!(kb::write::read_document(&store, &index, "absent").is_err());
    Ok(())
}
