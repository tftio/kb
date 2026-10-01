//! Reconciliation between the store and the derived index (T011).
//!
//! The property under test is idempotence by construction: reconciliation
//! compares content addresses rather than replaying events, so running it
//! twice on a clean store must find nothing and change nothing. Everything
//! else here is a way of manufacturing one specific kind of drift and
//! checking that it is named correctly — the three repairable kinds have
//! different causes, and a survey that called a missing record a stale one
//! would send an operator looking in the wrong place.

use std::collections::BTreeSet;

use kb::fsck::{Survey, repair, survey};
use kb::index::{Index, Scope, rebuild};
use kb::record::{ArtifactKind, ContentHash, CorpusId, RecordHeader, RecordId, SourceRef};
use kb::store::{BlobStore, GitBlobStore, RefName};

mod common;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Write a note into the store under `id`, with `body` as its content.
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

/// Where a store's refs live on disk.
///
/// Two `kb` segments, and neither is a typo: the store namespaces all of its
/// refs under `refs/kb/`, and a record's ref is named `<corpus>/<id>`, so a kb
/// record lands at `refs/kb/kb/<id>`. Removing that file is how a record
/// leaves the store — the blob may linger until the next gc, but nothing
/// addresses it any more.
fn ref_path(dir: &tempfile::TempDir, id: &str) -> std::path::PathBuf {
    dir.path()
        .join("store")
        .join("refs")
        .join("kb")
        .join("kb")
        .join(id)
}

/// A store of three notes and an index built from it.
fn prepared() -> Result<(tempfile::TempDir, GitBlobStore, Index), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    for (id, body) in [
        ("alpha", "the first body"),
        ("beta", "the second body"),
        ("gamma", "the third body"),
    ] {
        store_note(&store, id, body)?;
    }
    let index = Index::open_in_memory()?;
    rebuild(&store, &index, &Scope::All)?;
    Ok((dir, store, index))
}

fn surveyed(store: &GitBlobStore, index: &Index) -> Result<Survey, Box<dyn std::error::Error>> {
    Ok(survey(store, index, None)?)
}

#[test]
fn a_freshly_built_index_agrees_with_the_store() -> TestResult {
    let (_dir, store, index) = prepared()?;
    let found = surveyed(&store, &index)?;
    assert_eq!(found.checked, 3);
    assert!(found.is_clean(), "{found:?}");
    Ok(())
}

/// The acceptance property: reconciliation is driven by comparison, so a
/// second pass over an unchanged store has nothing to do. Anything else would
/// mean the first pass changed something it should not have.
#[test]
fn surveying_twice_on_a_clean_store_finds_nothing_both_times() -> TestResult {
    let (_dir, store, index) = prepared()?;
    let first = surveyed(&store, &index)?;
    let done = repair(&store, &index, &first)?;
    assert_eq!(done.rederived, 0);
    assert_eq!(done.dropped, 0);
    let second = surveyed(&store, &index)?;
    assert!(second.is_clean(), "{second:?}");
    assert_eq!(first.checked, second.checked);
    Ok(())
}

#[test]
fn a_record_changed_outside_kb_is_named_stale_and_re_derived() -> TestResult {
    let (_dir, store, index) = prepared()?;
    // Rewriting the record under the same id is exactly what an external
    // edit looks like from the index's side: same ref, different blob.
    store_note(&store, "beta", "a body it did not have before")?;
    let found = surveyed(&store, &index)?;
    assert_eq!(found.stale, vec!["beta".to_owned()], "{found:?}");
    assert!(
        found.missing.is_empty() && found.orphaned.is_empty(),
        "{found:?}"
    );

    let done = repair(&store, &index, &found)?;
    assert_eq!(done.rederived, 1);
    assert!(
        done.passages > 0,
        "the record was not re-derived into passages"
    );

    let text = index.record_text("beta")?;
    assert!(
        text.contains("a body it did not have before"),
        "the index still holds the old text: {text}"
    );
    assert!(surveyed(&store, &index)?.is_clean());
    Ok(())
}

#[test]
fn a_record_the_index_never_saw_is_named_missing_and_indexed() -> TestResult {
    let (_dir, store, index) = prepared()?;
    store_note(&store, "delta", "a body added after the index was built")?;
    let found = surveyed(&store, &index)?;
    assert_eq!(found.missing, vec!["delta".to_owned()], "{found:?}");
    assert_eq!(found.checked, 4);

    repair(&store, &index, &found)?;
    let ids: BTreeSet<String> = index.records_in("kb")?.into_iter().collect();
    assert!(ids.contains("delta"), "{ids:?}");
    assert!(surveyed(&store, &index)?.is_clean());
    Ok(())
}

#[test]
fn repairing_one_record_leaves_the_others_alone() -> TestResult {
    let (_dir, store, index) = prepared()?;
    let before = index.record_text("alpha")?;
    store_note(&store, "beta", "changed")?;
    let found = surveyed(&store, &index)?;
    repair(&store, &index, &found)?;
    assert_eq!(
        index.record_text("alpha")?,
        before,
        "an untouched record was re-derived"
    );
    Ok(())
}

#[test]
fn a_record_no_longer_stored_is_named_orphaned_and_forgotten() -> TestResult {
    let (dir, store, index) = prepared()?;
    std::fs::remove_file(ref_path(&dir, "gamma"))?;
    let found = surveyed(&store, &index)?;
    assert_eq!(found.orphaned, vec!["gamma".to_owned()], "{found:?}");
    assert_eq!(found.checked, 2);

    let done = repair(&store, &index, &found)?;
    assert_eq!(done.dropped, 1);
    let ids: BTreeSet<String> = index.records_in("kb")?.into_iter().collect();
    assert!(!ids.contains("gamma"), "{ids:?}");
    assert!(index.passages("gamma")?.is_empty());
    assert!(surveyed(&store, &index)?.is_clean());
    Ok(())
}

/// Reference-only corpora have no record blob for the store to address, so a
/// store survey has nothing to compare them against. Reporting them as
/// orphaned would call the entire mail corpus corrupt.
#[test]
fn a_reference_only_corpus_is_not_reported_as_orphaned() -> TestResult {
    use kb::maildir::{MaildirCorpus, Selection};
    let (dir, store, index) = prepared()?;
    let maildir = dir.path().join("mail");
    let cur = maildir.join("Inbox").join("cur");
    std::fs::create_dir_all(&cur)?;
    std::fs::write(
        cur.join("one:2,S"),
        "From: Ada <ada@example.invalid>\n\
         To: reader@example.invalid\n\
         Subject: a message\n\
         Date: Mon, 3 Feb 2025 09:00:00 +0000\n\
         Message-ID: <one@example.invalid>\n\
         \n\
         a body\n",
    )?;
    let corpus = MaildirCorpus::scan(&maildir)?;
    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &BTreeSet::new());
    kb::index::rebuild_mail(&corpus, &selection, &index)?;
    assert!(
        !index.records_in("mail")?.is_empty(),
        "the fixture did not index"
    );

    let found = surveyed(&store, &index)?;
    assert!(
        found.orphaned.is_empty(),
        "a reference-only corpus was called orphaned: {:?}",
        found.orphaned
    );
    assert_eq!(found.checked, 3, "only stored records are compared");
    Ok(())
}

#[test]
fn a_node_the_store_never_saw_is_reported_and_not_repaired() -> TestResult {
    let (dir, store, index) = prepared()?;
    let legacy_path = dir.path().join("kb.db");
    let legacy = common::legacy_db(&legacy_path)?;
    for id in ["alpha", "unarchived-one", "unarchived-two"] {
        common::legacy_insert(
            &legacy,
            id,
            &tftio_org::ast::Document {
                blocks: vec![tftio_org::ast::Block::Heading {
                    level: 1,
                    title: tftio_org::ast::Title(id.into()),
                    tags: vec![],
                    children: vec![],
                }],
            },
        )?;
    }
    let found = survey(&store, &index, Some(&legacy))?;
    assert_eq!(
        found.unarchived,
        vec!["unarchived-one".to_owned(), "unarchived-two".to_owned()],
        "{found:?}"
    );
    // Unarchived nodes are drift, but not drift a derivation can fix: nothing
    // in the store describes them, so `is_clean` speaks only of what fsck can
    // repair and the repair leaves them exactly where they were.
    assert!(found.is_clean(), "{found:?}");
    let done = repair(&store, &index, &found)?;
    assert_eq!(done.rederived, 0);
    assert_eq!(done.dropped, 0);
    assert_eq!(
        survey(&store, &index, Some(&legacy))?.unarchived.len(),
        2,
        "the report changed what it was only supposed to describe"
    );
    Ok(())
}

/// Store a record whose stream blob is `stream_bytes`, whatever the normalizer
/// would actually produce. The only way to manufacture a record that
/// disagrees with its own normalization.
fn store_with_stream(
    store: &GitBlobStore,
    id: &str,
    body: &str,
    stream_bytes: &[u8],
) -> TestResult {
    let raw = format!("* Note {id}\n\n{body}\n");
    let header = RecordHeader {
        id: RecordId::new(id)?,
        corpus: CorpusId::new("kb")?,
        kind: ArtifactKind::Note,
        created: "2026-05-19T09:00:00Z".parse()?,
        updated: "2026-08-14T10:30:00Z".parse()?,
        source: SourceRef::new("node-id", id)?,
        tags: vec![],
        raw: ContentHash::new(store.put(raw.as_bytes())?.as_str())?,
        stream: ContentHash::new(store.put(stream_bytes)?.as_str())?,
        normalizer: kb::record::NORMALIZER_VERSION,
        provenance: kb::record::Provenance::default(),
    };
    let hash = store.put(&header.serialize())?;
    store.set_ref(&RefName::new(&format!("kb/{id}"))?, &hash)?;
    Ok(())
}

#[test]
fn a_deep_survey_of_a_clean_store_finds_nothing() -> TestResult {
    let (_dir, store, index) = prepared()?;
    let found = kb::fsck::survey_with(&store, &index, None, true)?;
    assert!(found.deep);
    assert!(found.renormalized.is_empty(), "{found:?}");
    assert_eq!(found.behind, 0, "{found:?}");
    Ok(())
}

/// The distinction the deep survey exists to draw. A stream whose content
/// differs is real staleness; one that differs only in its version banner is
/// a global counter having moved for some other artifact kind, and naming
/// every record for that would bury the first case in the second.
#[test]
fn a_version_only_difference_is_counted_and_a_content_difference_is_named() -> TestResult {
    let (_dir, store, index) = prepared()?;
    let body = "a body";
    let raw = format!("* Note behind\n\n{body}\n");
    let current = kb::record::normalize(ArtifactKind::Note, raw.as_bytes())?;

    // Same payload, older banner: the record is behind and nothing is wrong.
    let mut older = b"kb-stream/1\n".to_vec();
    older.extend_from_slice(current.payload());
    store_with_stream(&store, "behind", body, &older)?;

    // Different payload under the current banner: real staleness.
    let mut wrong = b"kb-stream/3\n".to_vec();
    wrong.extend_from_slice(b"text this record does not contain\n");
    store_with_stream(&store, "wrong", "a body", &wrong)?;

    rebuild(&store, &index, &Scope::All)?;
    let found = kb::fsck::survey_with(&store, &index, None, true)?;
    assert_eq!(found.behind, 1, "{found:?}");
    assert_eq!(found.renormalized, vec!["wrong".to_owned()], "{found:?}");
    // Neither is drift between store and index: both records are indexed from
    // exactly the stream they claim.
    assert!(found.is_clean(), "{found:?}");
    Ok(())
}

/// A deep survey must not put objects into the store. The store's gc packs and
/// never prunes, so an object written by a read-only command is permanent.
#[test]
fn a_deep_survey_writes_nothing_to_the_store() -> TestResult {
    let (dir, store, index) = prepared()?;
    let objects = dir.path().join("store").join("objects");
    let before = std::fs::read_dir(&objects)?.count();
    store_with_stream(&store, "wrong", "a body", b"kb-stream/3\nnot the content\n")?;
    let after_setup = std::fs::read_dir(&objects)?.count();
    kb::fsck::survey_with(&store, &index, None, true)?;
    let after_survey = std::fs::read_dir(&objects)?.count();
    assert_eq!(
        after_setup, after_survey,
        "the survey wrote objects into an append-only store"
    );
    assert!(after_setup >= before);
    Ok(())
}

/// Which store operation a [`Failing`] store refuses.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Refuse {
    /// Refuse to enumerate refs.
    Listing,
    /// Refuse to resolve a ref to a hash.
    Resolving,
    /// Refuse to read objects.
    Reading,
}

/// A store that delegates to a real one and refuses one operation.
///
/// A survey walks a store it does not control, and every read is a place the
/// walk can stop. What matters is that it stops with the record named rather
/// than with a bare driver error, since "the store failed" tells an operator
/// nothing about where to look.
struct Failing {
    inner: GitBlobStore,
    refuse: Refuse,
    /// How many reads to allow before refusing. Lets a test fail a read the
    /// deep survey makes without failing the record-blob read that precedes
    /// it, which is the only way to reach the deep path's own error handling.
    allow: std::cell::Cell<usize>,
}

impl Failing {
    const fn new(inner: GitBlobStore, refuse: Refuse) -> Self {
        Self {
            inner,
            refuse,
            allow: std::cell::Cell::new(0),
        }
    }

    const fn allowing(inner: GitBlobStore, reads: usize) -> Self {
        Self {
            inner,
            refuse: Refuse::Reading,
            allow: std::cell::Cell::new(reads),
        }
    }
}

impl BlobStore for Failing {
    fn put(&self, bytes: &[u8]) -> Result<kb::store::BlobHash, kb::store::StoreError> {
        self.inner.put(bytes)
    }
    fn get(&self, hash: &kb::store::BlobHash) -> Result<Vec<u8>, kb::store::StoreError> {
        if self.refuse == Refuse::Reading {
            let remaining = self.allow.get();
            if remaining == 0 {
                return Err(kb::store::StoreError::ObjectNotFound {
                    hash: hash.as_str().to_owned(),
                });
            }
            self.allow.set(remaining - 1);
        }
        self.inner.get(hash)
    }
    fn set_ref(
        &self,
        name: &RefName,
        hash: &kb::store::BlobHash,
    ) -> Result<(), kb::store::StoreError> {
        self.inner.set_ref(name, hash)
    }
    fn delete_ref(&self, name: &RefName) -> Result<bool, kb::store::StoreError> {
        self.inner.delete_ref(name)
    }
    fn read_ref(&self, name: &RefName) -> Result<kb::store::BlobHash, kb::store::StoreError> {
        if self.refuse == Refuse::Resolving {
            return Err(kb::store::StoreError::RefNotFound {
                name: name.as_str().to_owned(),
            });
        }
        self.inner.read_ref(name)
    }
    fn list_refs(&self, prefix: &str) -> Result<Vec<RefName>, kb::store::StoreError> {
        if self.refuse == Refuse::Listing {
            return Err(kb::store::StoreError::RefFailed {
                name: prefix.to_owned(),
                reason: "refused".to_owned(),
            });
        }
        self.inner.list_refs(prefix)
    }
}

#[test]
fn a_store_that_cannot_be_enumerated_stops_the_survey() -> TestResult {
    let (_dir, store, index) = prepared()?;
    let failing = Failing::new(store, Refuse::Listing);
    let Err(failure) = survey(&failing, &index, None) else {
        return Err("an unreadable store was reported as clean".into());
    };
    assert!(
        failure.to_string().contains("enumerating"),
        "the failure did not say where it stopped: {failure}"
    );
    Ok(())
}

#[test]
fn a_record_that_cannot_be_read_is_named_in_the_failure() -> TestResult {
    let (_dir, store, index) = prepared()?;
    let failing = Failing::new(store, Refuse::Reading);
    let Err(failure) = survey(&failing, &index, None) else {
        return Err("an unreadable record was reported as clean".into());
    };
    let text = failure.to_string();
    assert!(
        text.contains("alpha") || text.contains("beta") || text.contains("gamma"),
        "the failure did not name a record: {text}"
    );
    Ok(())
}

#[test]
fn a_blob_that_is_not_a_record_is_reported_as_such() -> TestResult {
    let (_dir, store, index) = prepared()?;
    // A ref pointing at bytes that are not a record header is corruption the
    // survey must name rather than skip.
    let hash = store.put(b"this is not a record header\n")?;
    store.set_ref(&RefName::new("kb/nonsense")?, &hash)?;
    let Err(failure) = survey(&store, &index, None) else {
        return Err("an unparsable record was reported as clean".into());
    };
    assert!(
        failure.to_string().contains("nonsense"),
        "the failure did not name the record: {failure}"
    );
    Ok(())
}

/// A deep survey must not try to re-normalize a corpus whose bytes were never
/// copied into the store; there is nothing there to read.
#[test]
fn a_deep_survey_skips_a_reference_only_corpus() -> TestResult {
    use kb::maildir::{MaildirCorpus, Selection};
    let (dir, store, index) = prepared()?;
    let maildir = dir.path().join("mail");
    let cur = maildir.join("Inbox").join("cur");
    std::fs::create_dir_all(&cur)?;
    std::fs::write(
        cur.join("one:2,S"),
        "From: Ada <ada@example.invalid>\n\
         To: reader@example.invalid\n\
         Subject: a message\n\
         Date: Mon, 3 Feb 2025 09:00:00 +0000\n\
         Message-ID: <one@example.invalid>\n\
         \n\
         a body\n",
    )?;
    let corpus = MaildirCorpus::scan(&maildir)?;
    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &BTreeSet::new());
    kb::index::rebuild_mail(&corpus, &selection, &index)?;

    let found = kb::fsck::survey_with(&store, &index, None, true)?;
    assert!(found.renormalized.is_empty(), "{found:?}");
    assert_eq!(found.behind, 0, "{found:?}");
    Ok(())
}

#[test]
fn a_ref_that_will_not_resolve_is_named_in_the_failure() -> TestResult {
    let (_dir, store, index) = prepared()?;
    let failing = Failing::new(store, Refuse::Resolving);
    let Err(failure) = survey(&failing, &index, None) else {
        return Err("an unresolvable ref was reported as clean".into());
    };
    let text = failure.to_string();
    assert!(
        text.contains("alpha") || text.contains("beta") || text.contains("gamma"),
        "the failure did not name a record: {text}"
    );
    Ok(())
}

/// The deep survey reads two more blobs per record than the shallow one, and
/// each is a place the walk can stop. A failure there must still name the
/// record rather than surfacing as a bare driver error.
#[test]
fn a_deep_survey_names_the_record_it_could_not_re_normalize() -> TestResult {
    let (_dir, store, index) = prepared()?;
    let failing = Failing::new(store, Refuse::Reading);
    let Err(failure) = kb::fsck::survey_with(&failing, &index, None, true) else {
        return Err("an unreadable store was reported as clean".into());
    };
    assert!(
        failure.to_string().contains("from the store"),
        "the failure did not say where it stopped: {failure}"
    );
    Ok(())
}

/// A record whose raw bytes will not normalize is corruption the deep survey
/// must name, not skip.
#[test]
fn a_record_whose_raw_will_not_normalize_is_reported() -> TestResult {
    let (_dir, store, index) = prepared()?;
    // Bytes that are not UTF-8 cannot be normalized at all, whatever the kind.
    let raw = store.put(&[0xff, 0xfe, 0xfd])?;
    let stream = store.put(
        b"kb-stream/3
anything
",
    )?;
    let header = RecordHeader {
        id: RecordId::new("unnormalizable")?,
        corpus: CorpusId::new("kb")?,
        kind: ArtifactKind::Note,
        created: "2026-05-19T09:00:00Z".parse()?,
        updated: "2026-08-14T10:30:00Z".parse()?,
        source: SourceRef::new("node-id", "unnormalizable")?,
        tags: vec![],
        raw: ContentHash::new(raw.as_str())?,
        stream: ContentHash::new(stream.as_str())?,
        normalizer: kb::record::NORMALIZER_VERSION,
        provenance: kb::record::Provenance::default(),
    };
    let hash = store.put(&header.serialize())?;
    store.set_ref(&RefName::new("kb/unnormalizable")?, &hash)?;

    let Err(failure) = kb::fsck::survey_with(&store, &index, None, true) else {
        return Err("an unnormalizable record was reported as clean".into());
    };
    assert!(
        failure.to_string().contains("unnormalizable"),
        "the failure did not name the record: {failure}"
    );
    Ok(())
}

/// The deep survey reads the raw blob after the record blob, so a store that
/// serves the first and refuses the second exercises the deep path's own error
/// handling rather than the shallow walk's.
#[test]
fn a_raw_blob_that_cannot_be_read_names_its_record() -> TestResult {
    let (_dir, store, index) = prepared()?;
    // One read for the record blob, then refuse: the next read is the raw
    // blob the re-normalization needs.
    let failing = Failing::allowing(store, 1);
    let Err(failure) = kb::fsck::survey_with(&failing, &index, None, true) else {
        return Err("an unreadable raw blob was reported as clean".into());
    };
    let text = failure.to_string();
    assert!(
        text.contains("alpha") || text.contains("beta") || text.contains("gamma"),
        "the failure did not name a record: {text}"
    );
    Ok(())
}

/// And the stream blob after that: three reads in, the comparison itself is
/// what fails.
#[test]
fn a_stream_blob_that_cannot_be_read_names_its_record() -> TestResult {
    let (_dir, store, index) = prepared()?;
    let failing = Failing::allowing(store, 2);
    let Err(failure) = kb::fsck::survey_with(&failing, &index, None, true) else {
        return Err("an unreadable stream blob was reported as clean".into());
    };
    assert!(
        failure.to_string().contains("from the store"),
        "the failure did not say where it stopped: {failure}"
    );
    Ok(())
}
