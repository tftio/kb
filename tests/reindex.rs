//! Rebuilding the index: the operation the whole architecture rests on.
//!
//! Everything else this plan buys is only worth what a rebuild costs, so
//! these tests are about the properties that keep rebuilding cheap and safe
//! to reach for — it is atomic, it is scopable, and it does not throw away
//! the derived data that was expensive to produce.

use kb::index::{Index, Scope, rebuild};
use kb::record::{ArtifactKind, ContentHash, CorpusId, RecordHeader, RecordId, SourceRef};
use kb::store::{BlobStore, GitBlobStore, RefName};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Put one record into the store under a corpus, and return its header.
fn store_record(
    store: &GitBlobStore,
    corpus: &str,
    id: &str,
    body: &str,
) -> Result<RecordHeader, Box<dyn std::error::Error>> {
    let raw = body.as_bytes();
    let stream = kb::record::normalize(ArtifactKind::Note, raw)?;
    let header = RecordHeader {
        id: RecordId::new(id)?,
        corpus: CorpusId::new(corpus)?,
        kind: ArtifactKind::Note,
        created: "2026-05-19T09:00:00Z".parse()?,
        updated: "2026-08-14T10:30:00Z".parse()?,
        source: SourceRef::new("node-id", id)?,
        tags: vec![],
        raw: ContentHash::new(store.put(raw)?.as_str())?,
        stream: ContentHash::new(store.put(stream.as_bytes())?.as_str())?,
        normalizer: kb::record::NORMALIZER_VERSION,
        provenance: kb::record::Provenance::default(),
    };
    let hash = store.put(&header.serialize())?;
    store.set_ref(&RefName::new(&format!("{corpus}/{id}"))?, &hash)?;
    Ok(header)
}

/// A rebuild that fails partway must leave the index as it was. The failure
/// mode this prevents is the worst one available: an index that looks
/// complete, is missing records nobody knows about, and answers queries
/// confidently anyway.
#[test]
fn a_failed_rebuild_leaves_the_index_untouched() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_record(&store, "kb", "n1", "* Good\n\nfine.\n")?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;
    let before = index.dump()?;

    // A record naming a stream the store does not hold, indexed after n1.
    let broken = RecordHeader {
        stream: ContentHash::new("0000000000000000000000000000000000000000")?,
        ..store_record(&store, "kb", "n2", "* Broken\n\nnope.\n")?
    };
    let hash = store.put(&broken.serialize())?;
    store.set_ref(&RefName::new("kb/n2")?, &hash)?;

    let outcome = rebuild(&store, &index, &Scope::All);

    assert!(outcome.is_err(), "the broken record must fail the rebuild");
    assert_eq!(
        index.dump()?,
        before,
        "a failed rebuild must roll back, not leave a partial index"
    );
    Ok(())
}

/// Rebuilding one corpus must not disturb another. Without this, adding a
/// mail message would mean rebuilding every note, and at mail scale nobody
/// would rebuild at all.
#[test]
fn a_corpus_can_be_rebuilt_without_touching_the_others() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_record(&store, "kb", "n1", "* Note\n\nauthored.\n")?;
    store_record(&store, "mail", "m1", "* Message\n\ncorrespondence.\n")?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;
    let whole = index.dump()?;

    let report = rebuild(&store, &index, &Scope::Corpus("mail".to_owned()))?;

    assert_eq!(report.records, 1, "only the mail record was rebuilt");
    assert_eq!(
        index.dump()?,
        whole,
        "a scoped rebuild changed rows outside its scope"
    );
    Ok(())
}

/// One record, for the case the operator actually hits: a note edited in
/// Emacs, needing to become searchable without touching anything else.
#[test]
fn a_single_record_can_be_rebuilt_on_its_own() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_record(&store, "kb", "n1", "* First\n\noriginal.\n")?;
    store_record(&store, "kb", "n2", "* Second\n\nuntouched.\n")?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;

    store_record(&store, "kb", "n1", "* First\n\nrevised entirely.\n")?;
    let report = rebuild(&store, &index, &Scope::Record("n1".to_owned()))?;

    assert_eq!(report.records, 1);
    assert!(
        index.search_text("revised")?.contains(&"n1".to_owned()),
        "the revision is not searchable"
    );
    assert!(
        index.search_text("original")?.is_empty(),
        "the superseded text is still indexed"
    );
    assert!(
        index.search_text("untouched")?.contains(&"n2".to_owned()),
        "an unrelated record lost its index rows"
    );
    Ok(())
}

/// Vectors are the expensive part. Re-chunking must not discard them, or
/// every chunking experiment costs a full re-embedding of the corpus and the
/// experiment stops being worth running — which is the exact failure this
/// architecture exists to prevent.
#[test]
fn a_rebuild_keeps_vectors_whose_passages_are_unchanged() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    let header = store_record(&store, "kb", "n1", "* Note\n\nembedded once.\n")?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;
    let passage = index.passages("n1")?;
    let first = passage.first().ok_or("no passage")?;
    index.put_embedding(
        header.stream.as_str(),
        first.span_start,
        first.span_len,
        "test-model",
        &[0u8, 1, 2, 3],
    )?;

    rebuild(&store, &index, &Scope::All)?;

    assert_eq!(
        index.embedding(
            header.stream.as_str(),
            first.span_start,
            first.span_len,
            "test-model"
        )?,
        Some(vec![0u8, 1, 2, 3]),
        "the vector was discarded by a rebuild that changed nothing about its passage"
    );
    Ok(())
}

/// A vector belongs to a span of a particular stream. When the stream
/// changes, the old vector describes text that is no longer there, and
/// keeping it would let a search return a passage that no longer says what
/// the vector says it says.
#[test]
fn a_vector_does_not_survive_the_text_it_described() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    let header = store_record(&store, "kb", "n1", "* Note\n\nthe original claim.\n")?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;
    let passage = index.passages("n1")?;
    let first = passage.first().ok_or("no passage")?;
    index.put_embedding(
        header.stream.as_str(),
        first.span_start,
        first.span_len,
        "test-model",
        &[9u8; 4],
    )?;

    store_record(&store, "kb", "n1", "* Note\n\na wholly different claim.\n")?;
    rebuild(&store, &index, &Scope::All)?;
    // The sweep is the caller's step after a full rebuild, because a full
    // rebuild re-derives the store-backed corpora and leaves the referenced
    // ones to be re-derived after it: sweeping inside it would call every
    // vector of a corpus that has not been rebuilt yet an orphan.
    index.drop_orphaned_derivations()?;

    let stale = index.embedding(
        header.stream.as_str(),
        first.span_start,
        first.span_len,
        "test-model",
    )?;
    assert!(
        stale.is_none()
            || index
                .passages("n1")?
                .iter()
                .all(|p| p.span_start != first.span_start),
        "a vector outlived the stream it was computed against"
    );
    Ok(())
}

/// The number that decides whether anyone rebuilds.
#[test]
fn a_rebuild_reports_how_long_it_took() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_record(&store, "kb", "n1", "* Note\n\ntimed.\n")?;
    let index = Index::open(&dir.path().join("index.db"))?;

    let report = rebuild(&store, &index, &Scope::All)?;

    assert!(
        report.elapsed.as_nanos() > 0,
        "a rebuild reported no elapsed time"
    );
    Ok(())
}
