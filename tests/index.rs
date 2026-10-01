//! Behavior of the derived index: it holds only what the store can produce
//! again, and throwing it away costs nothing but the time to rebuild.

use kb::index::{Index, Scope, rebuild};
use kb::record::{ArtifactKind, ContentHash, CorpusId, RecordHeader, RecordId, SourceRef, Tag};
use kb::store::{BlobStore, GitBlobStore, RefName};

mod common;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const NOTE: &[u8] = b"* Background\n\nSee [[storage-plan]] for context.\n\n\
                      * Decision\n\nWhat was chosen.\n";

/// Write a record into the store as the three blobs it is made of, and bind
/// its name. This is what T008's import will do at corpus scale.
fn store_a_note(
    store: &GitBlobStore,
    id: &str,
) -> Result<RecordHeader, Box<dyn std::error::Error>> {
    let stream = kb::record::normalize(ArtifactKind::Note, NOTE)?;
    let raw_hash = store.put(NOTE)?;
    let stream_hash = store.put(stream.as_bytes())?;
    let header = RecordHeader {
        id: RecordId::new(id)?,
        corpus: CorpusId::new("kb")?,
        kind: ArtifactKind::Note,
        created: "2026-05-19T09:00:00Z".parse()?,
        updated: "2026-08-14T10:30:00Z".parse()?,
        source: SourceRef::new("node-id", id)?,
        tags: vec![Tag::new("storage")?, Tag::new("retrieval")?],
        raw: ContentHash::new(raw_hash.as_str())?,
        stream: ContentHash::new(stream_hash.as_str())?,
        normalizer: kb::record::NORMALIZER_VERSION,
        provenance: kb::record::Provenance::default(),
    };
    let record_hash = store.put(&header.serialize())?;
    store.set_ref(&RefName::new(&format!("nodes/{id}"))?, &record_hash)?;
    Ok(header)
}

/// [`store_a_note`], but asserting `project` in its header, for the
/// project-filtered retrieval tests (`PLAN-20260923-project-identity` T006).
fn store_a_note_with_project(
    store: &GitBlobStore,
    id: &str,
    project: &str,
) -> Result<RecordHeader, Box<dyn std::error::Error>> {
    let stream = kb::record::normalize(ArtifactKind::Note, NOTE)?;
    let raw_hash = store.put(NOTE)?;
    let stream_hash = store.put(stream.as_bytes())?;
    let header = RecordHeader {
        id: RecordId::new(id)?,
        corpus: CorpusId::new("kb")?,
        kind: ArtifactKind::Note,
        created: "2026-05-19T09:00:00Z".parse()?,
        updated: "2026-08-14T10:30:00Z".parse()?,
        source: SourceRef::new("node-id", id)?,
        tags: vec![Tag::new("storage")?, Tag::new("retrieval")?],
        raw: ContentHash::new(raw_hash.as_str())?,
        stream: ContentHash::new(stream_hash.as_str())?,
        normalizer: kb::record::NORMALIZER_VERSION,
        provenance: kb::record::Provenance {
            project: Some(tftio_lib::project::Slug::new(project)?),
            ..kb::record::Provenance::default()
        },
    };
    let record_hash = store.put(&header.serialize())?;
    store.set_ref(&RefName::new(&format!("nodes/{id}"))?, &record_hash)?;
    Ok(header)
}

/// The index is a projection of the store and nothing more.
#[test]
fn a_rebuild_populates_the_index_from_the_store() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    let index = Index::open(&dir.path().join("index.db"))?;

    let report = rebuild(&store, &index, &Scope::All)?;

    assert_eq!(report.records, 1);
    assert_eq!(report.passages, 2, "one per section");
    assert_eq!(report.authored_links, 1, "the [[storage-plan]] reference");
    Ok(())
}

/// The property the whole architecture rests on: the index is disposable.
/// If truncating and rebuilding produced anything different, something in it
/// would be authoritative and ST-002 would be false.
#[test]
fn every_table_can_be_truncated_and_rebuilt_from_the_store() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    store_a_note(&store, "n2")?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;
    let before = index.dump()?;

    index.truncate()?;
    assert_eq!(
        index.dump()?,
        String::new(),
        "truncate must empty everything"
    );
    rebuild(&store, &index, &Scope::All)?;

    assert_eq!(index.dump()?, before);
    Ok(())
}

/// Rebuilding over a populated index replaces rather than accumulates. A
/// rebuild that doubled every row would still look plausible until a search
/// returned each hit twice.
#[test]
fn rebuilding_twice_leaves_the_index_unchanged() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    let index = Index::open(&dir.path().join("index.db"))?;

    rebuild(&store, &index, &Scope::All)?;
    let once = index.dump()?;
    rebuild(&store, &index, &Scope::All)?;

    assert_eq!(index.dump()?, once);
    Ok(())
}

/// Every row says which corpus it belongs to, what it is called in the system
/// it came from, and the hashes of both the bytes that arrived and the stream
/// its spans address. Without all four a row cannot be checked against the
/// store, which is what makes rebuild possible and provenance verifiable.
#[test]
fn every_record_row_names_its_corpus_source_and_both_hashes() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    let header = store_a_note(&store, "n1")?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;

    let row = index.record("n1")?;

    assert_eq!(row.corpus, "kb");
    assert_eq!(row.source, "node-id:n1");
    assert_eq!(row.raw_hash, header.raw.as_str());
    assert_eq!(row.stream_hash, header.stream.as_str());
    assert!(
        row.record_hash.is_some_and(|hash| !hash.is_empty()),
        "a copied record has a record blob, and its hash is what reconciliation compares"
    );
    Ok(())
}

/// The old database is an export source and a frozen archive. Nothing in the
/// new index may write to it, now or by later accident.
#[test]
fn the_old_database_is_never_written_to() -> TestResult {
    let dir = tempfile::tempdir()?;
    let legacy = dir.path().join("kb.db");
    drop(common::legacy_db(&legacy)?);
    let before = std::fs::read(&legacy)?;

    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;

    assert_eq!(
        std::fs::read(&legacy)?,
        before,
        "the legacy database changed during a rebuild"
    );
    Ok(())
}

/// A record naming a stream the store does not hold is a broken store, not a
/// record to index halfway. Saying so names the record.
#[test]
fn a_record_whose_stream_is_missing_fails_loudly() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    let header = RecordHeader {
        stream: ContentHash::new("0000000000000000000000000000000000000000")?,
        ..store_a_note(&store, "n1")?
    };
    let hash = store.put(&header.serialize())?;
    store.set_ref(&RefName::new("nodes/n1")?, &hash)?;
    let index = Index::open(&dir.path().join("index.db"))?;

    let outcome = rebuild(&store, &index, &Scope::All);

    assert!(
        format!("{outcome:?}").contains("n1"),
        "the failure must name the record; got: {outcome:?}"
    );
    assert!(outcome.is_err());
    Ok(())
}

/// The full-text index is part of the rebuild, not something bolted on
/// afterwards: it is external-content FTS5 over the cached passage text, so a
/// rebuild that populated `passages` but not the index would search empty.
#[test]
fn passages_are_searchable_after_a_rebuild() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;

    let hits = index.search_text("Background")?;
    let misses = index.search_text("unrelated")?;

    assert_eq!(
        hits,
        vec!["n1".to_owned()],
        "full-text search found nothing"
    );
    assert!(
        misses.is_empty(),
        "matched something it should not: {misses:?}"
    );
    Ok(())
}

/// Truncation has to empty the full-text index too. An external-content FTS5
/// table left populated would keep returning rowids whose passages are gone.
#[test]
fn truncation_empties_the_full_text_index_too() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;

    index.truncate()?;

    assert!(index.search_text("Background")?.is_empty());
    Ok(())
}

/// Passage levels reach the index as the source's own vocabulary, so a query
/// can ask for turns without knowing how transcripts happen to be stored.
#[test]
fn passage_levels_are_recorded_per_kind() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    let transcript = b"* Human [t]\n\nq\n\n* Assistant [t]\n\na\n";
    let thread = b"From a@x Thu Aug 14 09:00:00 2026\r\nFrom: a@x\r\n\r\nfirst\r\n\
                   From b@x Thu Aug 14 09:30:00 2026\r\nFrom: b@x\r\n\r\nsecond\r\n";
    store_a_record(&store, "s1", ArtifactKind::SessionTranscript, transcript)?;
    store_a_record(&store, "t1", ArtifactKind::MailThread, thread)?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;

    let dump = index.dump()?;

    assert!(dump.contains("\tturn\t"), "no turn-level passage: {dump}");
    assert!(
        dump.contains("\tthread\t"),
        "no thread-level passage: {dump}"
    );
    assert!(
        dump.contains("\tmessage\t"),
        "no message-level passage: {dump}"
    );
    Ok(())
}

/// Each class of link target is stored as its own kind, so a broken-reference
/// report can distinguish a record that has not been written yet from a URL
/// that was never meant to resolve inside the corpus.
#[test]
fn link_targets_are_stored_by_class() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    let note = b"* Links\n\n[[id:b70049ea]] and [[a-slug]] and [[https://example.com]].\n";
    store_a_record(&store, "n1", ArtifactKind::Note, note)?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;

    let dump = index.dump()?;

    assert!(
        dump.contains("\trecord\tb70049ea\t"),
        "no record link: {dump}"
    );
    assert!(dump.contains("\tname\ta-slug\t"), "no name link: {dump}");
    assert!(
        dump.contains("\turl\thttps://example.com\t"),
        "no url link: {dump}"
    );
    Ok(())
}

/// Write a record of any kind into the store.
fn store_a_record(
    store: &GitBlobStore,
    id: &str,
    kind: ArtifactKind,
    raw: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = kb::record::normalize(kind, raw)?;
    let raw_hash = store.put(raw)?;
    let stream_hash = store.put(stream.as_bytes())?;
    let header = RecordHeader {
        id: RecordId::new(id)?,
        corpus: CorpusId::new("kb")?,
        kind,
        created: "2026-05-19T09:00:00Z".parse()?,
        updated: "2026-08-14T10:30:00Z".parse()?,
        source: SourceRef::new("node-id", id)?,
        tags: vec![],
        raw: ContentHash::new(raw_hash.as_str())?,
        stream: ContentHash::new(stream_hash.as_str())?,
        normalizer: kb::record::NORMALIZER_VERSION,
        provenance: kb::record::Provenance::default(),
    };
    let record_hash = store.put(&header.serialize())?;
    store.set_ref(&RefName::new(&format!("nodes/{id}"))?, &record_hash)?;
    Ok(())
}

/// Whether a record's bytes live in the store or only outside it decides
/// whether a reference that stops resolving is corruption or an expected
/// consequence of not owning them. The index records it per row, so a reader
/// of the index can tell without consulting a registry that may since have
/// changed.
#[test]
fn each_record_row_records_how_its_corpus_stores_bytes() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;

    assert_eq!(index.record("n1")?.storage, "copy");
    Ok(())
}

/// An index built under an older schema is discarded rather than migrated.
/// That is what disposability means in practice, and the alternative is worse
/// than it sounds: `CREATE TABLE IF NOT EXISTS` leaves an old file short a
/// column, and the failure surfaces as "no such column" from whatever query
/// happens to need it first.
#[test]
fn an_index_from_an_older_schema_is_refused_with_its_remedy() -> TestResult {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("index.db");
    {
        let conn = rusqlite::Connection::open(&path)?;
        conn.execute_batch("CREATE TABLE records (record_id TEXT);")?;
        conn.pragma_update(None, "user_version", 0)?;
    }

    let outcome = Index::open(&path);

    // Display, not Debug: the remedy lives in the message an operator reads.
    let message = match &outcome {
        Err(e) => e.to_string(),
        Ok(_) => String::from("<opened>"),
    };
    assert!(
        outcome.is_err(),
        "an older index must not be opened as current"
    );
    assert!(
        message.contains("reindex"),
        "the error must name the remedy; got: {message}"
    );
    Ok(())
}

/// Rebuilding is that remedy, so it recreates rather than refusing: the index
/// holds nothing authoritative, and refusing would leave the operator to
/// delete a file by hand to run the command whose whole purpose is to rebuild
/// it.
#[test]
fn rebuilding_recreates_an_index_from_an_older_schema() -> TestResult {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("index.db");
    {
        let conn = rusqlite::Connection::open(&path)?;
        conn.execute_batch("CREATE TABLE records (record_id TEXT);")?;
        conn.pragma_update(None, "user_version", 0)?;
    }
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;

    let index = Index::open_for_rebuild(&path)?;
    let report = rebuild(&store, &index, &Scope::All)?;

    assert_eq!(report.records, 1);
    assert_eq!(index.record("n1")?.storage, "copy");
    Ok(())
}

/// A record row carries the record's name (T029).
///
/// The superseded database kept a `title` column, and retrieval read it to
/// name a hit. Retiring that database means the index has to recover the same
/// name from the store, which it can: an org artifact declares one with
/// `#+title:` or names itself with its first heading.
#[test]
fn a_record_row_carries_the_title_derived_from_its_raw_artifact() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    let raw = b"#+title: The storage plan\n\n* Background\n\nWhy it exists.\n";
    let stream = kb::record::normalize(ArtifactKind::Note, raw)?;
    let header = RecordHeader {
        id: RecordId::new("titled")?,
        corpus: CorpusId::new("kb")?,
        kind: ArtifactKind::Note,
        created: "2026-05-19T09:00:00Z".parse()?,
        updated: "2026-05-19T09:00:00Z".parse()?,
        source: SourceRef::new("node-id", "titled")?,
        tags: vec![],
        raw: ContentHash::new(store.put(raw)?.as_str())?,
        stream: ContentHash::new(store.put(stream.as_bytes())?.as_str())?,
        normalizer: kb::record::NORMALIZER_VERSION,
        provenance: kb::record::Provenance::default(),
    };
    let hash = store.put(&header.serialize())?;
    store.set_ref(&RefName::new("kb/titled")?, &hash)?;

    let index = Index::open_for_rebuild(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;
    assert_eq!(index.record("titled")?.title, "The storage plan");
    Ok(())
}

/// A record that declares no title is named by its first heading, and one
/// with neither is nameless rather than named after its body.
#[test]
fn an_untitled_record_falls_back_to_its_heading() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "plain")?;
    let index = Index::open_for_rebuild(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;
    assert_eq!(index.record("plain")?.title, "Background");
    Ok(())
}

/// A schema change does not cost the vectors (T029).
///
/// Everything in the index is derived, which is what makes it disposable —
/// but a vector is derived from an embedding endpoint over hours, not from
/// the store in seconds, and its key is the content it was computed from
/// rather than anything the schema decides. Recomputing 80,000 of them
/// because a column was added to another table is a cost the disposability
/// argument never claimed to justify.
#[test]
fn a_schema_change_keeps_the_vectors_it_cannot_recompute() -> TestResult {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("index.db");
    {
        let index = Index::open_for_rebuild(&path)?;
        index.put_embedding(
            "streamhash",
            0,
            12,
            "a-model",
            &kb::embedding::encode_embedding(&[0.25, 0.5]),
        )?;
        drop(index);
        // Stamp an older schema, which is what an index built before this
        // change looks like to the next rebuild.
        let conn = rusqlite::Connection::open(&path)?;
        conn.pragma_update(None, "user_version", 4)?;
    }
    let index = Index::open_for_rebuild(&path)?;
    assert!(
        index.embedding("streamhash", 0, 12, "a-model")?.is_some(),
        "the rebuild discarded a vector it cannot recompute"
    );
    Ok(())
}

/// An index stamped with a *later* schema is refused too, not only an
/// earlier one. A binary that opened a newer index would write the shape it
/// knows into a file something else is still reading.
#[test]
fn an_index_from_another_schema_is_refused_whichever_way_it_differs() -> TestResult {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("index.db");
    {
        let conn = rusqlite::Connection::open(&path)?;
        conn.execute_batch("CREATE TABLE records (record_id TEXT);")?;
        conn.pragma_update(None, "user_version", kb::index::INDEX_SCHEMA_VERSION + 7)?;
    }
    let outcome = Index::open(&path);
    let message = match &outcome {
        Err(e) => e.to_string(),
        Ok(_) => String::from("<opened>"),
    };
    assert!(outcome.is_err(), "a newer index must not be opened");
    assert!(
        message.contains("kb reindex"),
        "the remedy is not named: {message}"
    );
    Ok(())
}

/// A record with no vector under the model asked about is a different fact
/// from a record with no neighbours, and the two have different remedies.
#[test]
fn similarity_distinguishes_an_unembedded_record_from_a_lonely_one() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    store_a_note(&store, "n2")?;
    let index = Index::open_for_rebuild(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;

    assert!(
        index.similar_to("n1", "a-model", 5)?.is_none(),
        "a record with no vector must not report neighbours"
    );
    assert!(
        index.model_of("n1")?.is_none(),
        "a record with no vector names no model"
    );

    for passage in index.passages("n1")? {
        index.put_embedding(
            &passage.stream_hash,
            passage.span_start,
            passage.span_len,
            "a-model",
            &kb::embedding::encode_embedding(&[1.0, 0.0]),
        )?;
    }
    assert_eq!(index.model_of("n1")?.as_deref(), Some("a-model"));
    let neighbours = index
        .similar_to("n1", "a-model", 5)?
        .ok_or("an embedded record should report a neighbour list")?;
    // `n2` holds the same text as `n1`, so it addresses the same stream and
    // is described by the same vector. Two records that say the same thing
    // are as similar as it is possible to be, and content addressing gets
    // that right without embedding the second one.
    assert_eq!(neighbours.len(), 1, "{neighbours:?}");
    let (neighbour, cosine) = neighbours.first().ok_or("no neighbour")?;
    assert_eq!(neighbour, "n2");
    assert!((cosine - 1.0).abs() < 1e-6, "{cosine}");
    assert!(
        !neighbours.iter().any(|(id, _)| id == "n1"),
        "a record is not its own neighbour: {neighbours:?}"
    );
    Ok(())
}

/// The spans a record has vectors for, and the model they were computed
/// under, are readable back — which is what makes an embedding pass
/// resumable rather than repeated.
#[test]
fn embedded_spans_report_what_has_a_vector() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    let index = Index::open_for_rebuild(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;
    assert!(index.embedded_spans("n1", "a-model")?.is_empty());

    let first = index
        .passages("n1")?
        .first()
        .cloned()
        .ok_or("the record has no passages")?;
    index.put_embedding(
        &first.stream_hash,
        first.span_start,
        first.span_len,
        "a-model",
        &kb::embedding::encode_embedding(&[0.5, 0.5]),
    )?;
    assert_eq!(
        index.embedded_spans("n1", "a-model")?,
        vec![(first.span_start, first.span_len)]
    );
    assert!(
        index.embedded_spans("n1", "another-model")?.is_empty(),
        "a vector under one model must not answer for another"
    );
    Ok(())
}

/// Reindexing a named record leaves every other record alone. That is what
/// makes a write cheap: the index is re-derived for what changed, not for
/// the corpus.
#[test]
fn reindexing_one_record_leaves_the_others_untouched() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    store_a_note(&store, "n2")?;
    let index = Index::open_for_rebuild(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;
    let before = index.record("n2")?;

    let mut one = std::collections::BTreeSet::new();
    one.insert("n1".to_owned());
    let report = kb::index::reindex_records(&store, &index, &one)?;
    assert_eq!(report.records, 1, "only the named record is re-derived");
    assert_eq!(index.record("n2")?.record_hash, before.record_hash);
    Ok(())
}

/// Reindexing nothing is nothing, not a full rebuild.
#[test]
fn reindexing_an_empty_set_does_no_work() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    let index = Index::open_for_rebuild(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;
    let empty = std::collections::BTreeSet::new();
    let report = kb::index::reindex_records(&store, &index, &empty)?;
    assert_eq!(report.records, 0);
    assert_eq!(report.passages, 0);
    Ok(())
}

/// A store that refuses one operation, so the rebuild's error paths can be
/// walked without corrupting a real store.
///
/// A rebuild reads a store it does not control, and every read is a place
/// the walk can stop. What matters is that it stops naming the record rather
/// than with a bare driver error: "the store failed" tells an operator
/// nothing about where to look.
struct Failing {
    inner: GitBlobStore,
    /// Reads to allow before refusing. Failing the first read stops the walk
    /// at the record blob; allowing one and failing the next stops it at the
    /// stream, which is a different arm.
    allow: std::cell::Cell<usize>,
    list: bool,
}

impl kb::store::BlobStore for Failing {
    fn put(&self, bytes: &[u8]) -> Result<kb::store::BlobHash, kb::store::StoreError> {
        self.inner.put(bytes)
    }
    fn get(&self, hash: &kb::store::BlobHash) -> Result<Vec<u8>, kb::store::StoreError> {
        let remaining = self.allow.get();
        if remaining == 0 {
            return Err(kb::store::StoreError::ObjectNotFound {
                hash: hash.as_str().to_owned(),
            });
        }
        self.allow.set(remaining - 1);
        self.inner.get(hash)
    }
    fn set_ref(
        &self,
        name: &RefName,
        hash: &kb::store::BlobHash,
    ) -> Result<(), kb::store::StoreError> {
        self.inner.set_ref(name, hash)
    }
    fn read_ref(&self, name: &RefName) -> Result<kb::store::BlobHash, kb::store::StoreError> {
        self.inner.read_ref(name)
    }
    fn list_refs(&self, prefix: &str) -> Result<Vec<RefName>, kb::store::StoreError> {
        if self.list {
            return Err(kb::store::StoreError::RefFailed {
                name: prefix.to_owned(),
                reason: "refused".to_owned(),
            });
        }
        self.inner.list_refs(prefix)
    }
    fn delete_ref(&self, name: &RefName) -> Result<bool, kb::store::StoreError> {
        self.inner.delete_ref(name)
    }
}

/// A store that cannot be enumerated fails the rebuild rather than producing
/// an index of nothing, which would read as an empty corpus.
#[test]
fn a_store_that_cannot_be_listed_fails_the_rebuild() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    let refusing = Failing {
        inner: store,
        allow: std::cell::Cell::new(usize::MAX),
        list: true,
    };
    let index = Index::open_for_rebuild(&dir.path().join("index.db"))?;
    assert!(rebuild(&refusing, &index, &Scope::All).is_err());

    let mut one = std::collections::BTreeSet::new();
    one.insert("n1".to_owned());
    assert!(kb::index::reindex_records(&refusing, &index, &one).is_err());
    Ok(())
}

/// A record whose blob cannot be read stops the rebuild with the record
/// named, and the index is rolled back rather than left half-derived.
#[test]
fn an_unreadable_record_names_itself_and_rolls_the_rebuild_back() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    let index = Index::open_for_rebuild(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;
    let before = index.dump()?;

    let refusing = Failing {
        inner: store,
        allow: std::cell::Cell::new(0),
        list: false,
    };
    let failed = rebuild(&refusing, &index, &Scope::All);
    let message = match &failed {
        Err(e) => e.to_string(),
        Ok(_) => String::from("<succeeded>"),
    };
    assert!(failed.is_err(), "an unreadable store rebuilt cleanly");
    assert!(message.contains("n1"), "the record is not named: {message}");
    assert_eq!(
        index.dump()?,
        before,
        "a failed rebuild left the index changed"
    );

    let mut one = std::collections::BTreeSet::new();
    one.insert("n1".to_owned());
    refusing.allow.set(0);
    assert!(kb::index::reindex_records(&refusing, &index, &one).is_err());
    assert_eq!(index.dump()?, before);
    Ok(())
}

/// A record whose stream is unreadable stops at a different arm: the record
/// blob was read and parsed, and the bytes its passages address are gone.
#[test]
fn an_unreadable_stream_stops_the_rebuild_after_the_header() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    let refusing = Failing {
        inner: store,
        // The record blob and the raw artifact are read before the stream.
        allow: std::cell::Cell::new(2),
        list: false,
    };
    let index = Index::open_for_rebuild(&dir.path().join("index.db"))?;
    let failed = rebuild(&refusing, &index, &Scope::All);
    let message = match &failed {
        Err(e) => e.to_string(),
        Ok(_) => String::from("<succeeded>"),
    };
    assert!(failed.is_err(), "a missing stream rebuilt cleanly");
    assert!(message.contains("n1"), "the record is not named: {message}");
    Ok(())
}

/// The full-text match must be the query's outermost loop.
///
/// It once was not, by the planner's choice rather than by the query's
/// construction: given a corpus filter, `SQLite` drove the join from `records`
/// and evaluated the match once per passage, which turned a search of the
/// live corpus from ten milliseconds into two and a half minutes at 100% CPU.
/// Two builds of the same `SQLite` version chose differently, so the plan is
/// asserted here rather than trusted.
#[test]
fn the_text_match_is_never_the_innermost_loop() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;

    let path = index.path().ok_or("the index has no path")?;
    let conn = rusqlite::Connection::open(&path)?;
    let mut statement = conn.prepare(&format!(
        "EXPLAIN QUERY PLAN {}",
        kb::index::SEARCH_TEXT_IN_SQL
    ))?;
    let plan = statement
        .query_map(
            rusqlite::params![
                "\"background\"*",
                "kb",
                Option::<&str>::None,
                Option::<&str>::None
            ],
            |row| row.get::<_, String>(3),
        )?
        .collect::<Result<Vec<String>, _>>()?;

    let first = plan.first().ok_or("the planner reported no loops")?;
    assert!(
        first.contains("passages_fts") || first.contains("matched"),
        "the match is not the outermost loop, so it is re-run per row:\n  {}",
        plan.join("\n  ")
    );
    Ok(())
}

/// The project filter (`PLAN-20260923-project-identity` T006) is applied
/// alongside the corpus predicate, not as a join that could change the
/// plan's outer loop: with a real project value bound, the text match still
/// has to be the outermost loop, exactly as it is with no filter.
#[test]
fn the_text_match_is_never_the_innermost_loop_with_a_project_filter() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note_with_project(&store, "n1", "kb")?;
    let index = Index::open(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;

    let path = index.path().ok_or("the index has no path")?;
    let conn = rusqlite::Connection::open(&path)?;
    let mut statement = conn.prepare(&format!(
        "EXPLAIN QUERY PLAN {}",
        kb::index::SEARCH_TEXT_IN_SQL
    ))?;
    let plan = statement
        .query_map(
            rusqlite::params!["\"background\"*", "kb", Some("kb"), Option::<&str>::None],
            |row| row.get::<_, String>(3),
        )?
        .collect::<Result<Vec<String>, _>>()?;

    let first = plan.first().ok_or("the planner reported no loops")?;
    assert!(
        first.contains("passages_fts") || first.contains("matched"),
        "the project filter moved the match out of the outermost loop:\n  {}",
        plan.join("\n  ")
    );

    let matched: Vec<String> = index.search_text_in("kb", "\"background\"*", Some("kb"), None)?;
    assert_eq!(matched, vec!["n1"], "the project filter did not match");
    assert!(
        index
            .search_text_in("kb", "\"background\"*", Some("other-project"), None)?
            .is_empty(),
        "the project filter matched a record under a different project"
    );
    Ok(())
}

/// A vector belongs to a record through the stream the record names: every
/// record in the corpus, then every vector under its stream. That attribution
/// has to hold for a record whose passage was embedded as several chunks
/// (both chunks count, and the pooling sees both), for two records that say
/// the same thing (one stream, both records), for another corpus (never
/// ranked) and for a vector under a stream no record holds (never ranked).
/// The plan is asserted too, because the statement this replaced returned
/// the same rows through a `DISTINCT` over the vector blobs and that temp
/// b-tree cost more than the scan (T025).
#[test]
fn dense_ranking_attributes_vectors_by_stream_and_never_sorts_the_blobs() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    store_a_note(&store, "n2")?;
    let index = Index::open_for_rebuild(&dir.path().join("index.db"))?;
    rebuild(&store, &index, &Scope::All)?;

    let passages = index.passages("n1")?;
    let first = passages.first().ok_or("n1 has no passages")?;
    let strong = kb::embedding::encode_embedding(&[1.0, 0.0]);
    let weak = kb::embedding::encode_embedding(&[0.0, 1.0]);
    // Two chunks of one passage under the record's stream.
    index.put_embedding(&first.stream_hash, first.span_start, 4, "m", &strong)?;
    index.put_embedding(&first.stream_hash, first.span_start + 4, 4, "m", &weak)?;
    // A vector under a stream no record holds.
    index.put_embedding("no-such-stream", 0, 4, "m", &strong)?;

    let ranked = index.rank_by_embedding_with_pooling(
        &[1.0, 0.0],
        "m",
        "kb",
        kb::index::DensePooling::Maximum,
        None,
        None,
    )?;
    let ids: Vec<&str> = ranked.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, vec!["n1", "n2"], "{ranked:?}");
    for (_, score) in &ranked {
        assert!(
            (score - 1.0).abs() < 1e-6,
            "the best chunk did not score: {ranked:?}"
        );
    }
    let mean = index.rank_by_embedding_with_pooling(
        &[1.0, 0.0],
        "m",
        "kb",
        kb::index::DensePooling::MeanTopThree,
        None,
        None,
    )?;
    let (_, pooled) = mean.first().ok_or("nothing ranked under mean pooling")?;
    assert!(
        (pooled - 0.5).abs() < 1e-6,
        "both chunks must reach the pooling: {mean:?}"
    );
    assert!(
        index
            .rank_by_embedding(&[1.0, 0.0], "m", "mail")?
            .is_empty(),
        "another corpus's query ranked this corpus's records"
    );

    let path = index.path().ok_or("the index has no path")?;
    let conn = rusqlite::Connection::open(&path)?;
    let mut statement = conn.prepare(&format!(
        "EXPLAIN QUERY PLAN {}",
        kb::index::RANK_BY_EMBEDDING_SQL
    ))?;
    let plan = statement
        .query_map(
            rusqlite::params!["m", "kb", Option::<&str>::None, Option::<&str>::None],
            |row| row.get::<_, String>(3),
        )?
        .collect::<Result<Vec<String>, _>>()?;
    assert!(
        !plan.iter().any(|line| line.contains("TEMP B-TREE")),
        "the dense scan sorts or de-duplicates the vector blobs:\n  {}",
        plan.join("\n  ")
    );
    assert!(
        plan.iter().any(|line| line.contains("records_corpus")),
        "the dense scan does not walk the corpus's records:\n  {}",
        plan.join("\n  ")
    );
    Ok(())
}

/// Opening a current index is a read, and a read must not wait on a writer.
/// Every tool call opens the index, so an open that needed the write lock
/// turned a running rebuild into a five-second stall for every concurrent
/// search and, past the busy timeout, into "database is locked" (T025). The
/// writer here holds the lock for the whole test; the open and the read
/// under it have to finish in well under the timeout.
#[test]
fn opening_a_current_index_does_not_wait_for_a_writer() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    store_a_note(&store, "n1")?;
    let path = dir.path().join("index.db");
    {
        let index = Index::open_for_rebuild(&path)?;
        rebuild(&store, &index, &Scope::All)?;
    }
    let writer = rusqlite::Connection::open(&path)?;
    writer.execute_batch("BEGIN IMMEDIATE")?;

    let started = std::time::Instant::now();
    let index = Index::open(&path)?;
    let row = index.record("n1")?;
    let elapsed = started.elapsed();
    writer.execute_batch("ROLLBACK")?;
    assert_eq!(row.corpus, "kb");
    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "opening and reading a current index waited {elapsed:?} on a writer"
    );
    Ok(())
}

#[test]
fn the_configured_index_path_is_the_variable_or_the_default() {
    use std::ffi::OsString;
    assert_eq!(
        kb::index::index_path_from(Some(OsString::from("/srv/kb/index.db"))),
        std::path::PathBuf::from("/srv/kb/index.db")
    );
    assert_eq!(
        kb::index::index_path_from(Some(OsString::new())),
        kb::index::default_index_path()
    );
    assert_eq!(
        kb::index::index_path_from(None),
        kb::index::default_index_path()
    );
    assert_eq!(
        kb::store::store_path_from(Some(OsString::from("/srv/kb/store"))),
        std::path::PathBuf::from("/srv/kb/store")
    );
    assert_eq!(
        kb::store::store_path_from(None),
        kb::store::default_store_path()
    );
}
