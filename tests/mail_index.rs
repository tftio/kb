//! Projecting the mail corpus into the derived index.
//!
//! Mail is reference-only, so what lands here is a catalogue and derived
//! passages, never the bytes. Two consequences are asserted throughout: a
//! mail record has no record blob for the store to address, and mail text
//! stays out of the kb FTS because mu/Xapian already does that job better.

use kb::index::{Index, rebuild_mail};
use kb::maildir::{MaildirCorpus, Selection};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn deliver(
    root: &Path,
    folder: &str,
    uniq: &str,
    message_id: &str,
    extra: &str,
    body: &str,
) -> TestResult {
    let dir = root.join(folder).join("cur");
    fs::create_dir_all(&dir)?;
    fs::write(
        dir.join(format!("{uniq}:2,S")),
        format!(
            "From: Ada <ada@example.invalid>\n\
             To: reader@example.invalid\n\
             Subject: about {uniq}\n\
             Date: Mon, 3 Feb 2025 09:00:00 +0000\n\
             Message-ID: {message_id}\n\
             {extra}\
             \n\
             {body}\n"
        ),
    )?;
    Ok(())
}

/// A kb record in a store, so a mail rebuild can be shown not to disturb it.
fn store_note(
    dir: &tempfile::TempDir,
    id: &str,
) -> Result<kb::store::GitBlobStore, Box<dyn std::error::Error>> {
    use kb::record::{ArtifactKind, ContentHash, CorpusId, RecordHeader, RecordId, SourceRef};
    use kb::store::{BlobStore, RefName};
    let store = kb::store::GitBlobStore::open_or_init(dir.path())?;
    let raw = format!("* Note {id}\n\nbody.\n");
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
    Ok(store)
}

fn indexed(dir: &tempfile::TempDir) -> Result<Index, Box<dyn std::error::Error>> {
    let corpus = MaildirCorpus::scan(dir.path())?;
    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &BTreeSet::new());
    let index = Index::open_in_memory()?;
    rebuild_mail(&corpus, &selection, &index)?;
    Ok(index)
}

/// A reference-only record has no record blob, so there is no address to put
/// in `record_hash`. Fabricating one would make an absent thing look present.
#[test]
fn a_mail_record_declares_its_storage_and_has_no_record_blob() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(
        dir.path(),
        "Inbox",
        "u1",
        "<a@example.invalid>",
        "",
        "hello",
    )?;

    let index = indexed(&dir)?;

    let row = index.record("<a@example.invalid>")?;
    assert_eq!(row.corpus, "mail");
    assert_eq!(row.kind, "mail-message");
    assert_eq!(row.source, "message-id:<a@example.invalid>");
    assert_eq!(row.storage, "reference");
    assert_eq!(row.record_hash, None);
    Ok(())
}

/// The catalogue is what makes a dead reference legible: where the message
/// was, and what it hashed to when the index last saw it.
#[test]
fn the_catalogue_records_where_the_bytes_are_and_what_they_hashed_to() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(dir.path(), "Archive", "u1", "<a@example.invalid>", "", "hi")?;

    let index = indexed(&dir)?;

    let entry = index
        .mail_catalogue("<a@example.invalid>")?
        .ok_or("message was not catalogued")?;
    assert_eq!(entry.folder, "Archive");
    assert_eq!(entry.locator, "u1");
    assert_eq!(entry.content_sha256.len(), 64);
    assert!(!entry.ground_truth_override);
    Ok(())
}

/// mu/Xapian is mail's lexical engine and already indexes senders, dates,
/// identifiers and exact strings. A second implementation here would be one
/// more thing to keep honest for no gain.
#[test]
fn mail_text_is_cached_for_embedding_but_kept_out_of_the_kb_fts() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(
        dir.path(),
        "Inbox",
        "u1",
        "<a@example.invalid>",
        "",
        "a distinctive phrase about quarterly revenue",
    )?;

    let index = indexed(&dir)?;

    let passages = index.passages("<a@example.invalid>")?;
    assert_eq!(passages.len(), 1);
    assert!(
        passages
            .first()
            .ok_or("no passage")?
            .text
            .contains("distinctive phrase"),
        "the passage text is what the embedder and reranker read"
    );
    assert!(
        index.search_text("distinctive")?.is_empty(),
        "mail text reached the kb FTS"
    );
    Ok(())
}

/// A reply and its parent are one document as well as two, and a question
/// about the exchange may be answerable from neither message alone.
#[test]
fn a_reply_and_its_parent_become_a_thread_record() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(
        dir.path(),
        "Inbox",
        "u1",
        "<a@example.invalid>",
        "",
        "the ask",
    )?;
    deliver(
        dir.path(),
        "Inbox",
        "u2",
        "<b@example.invalid>",
        "In-Reply-To: <a@example.invalid>\n",
        "the answer",
    )?;

    let index = indexed(&dir)?;

    let row = index.record("thread:<a@example.invalid>")?;
    assert_eq!(row.kind, "mail-thread");
    let text = index
        .passages("thread:<a@example.invalid>")?
        .into_iter()
        .map(|p| p.text)
        .collect::<String>();
    assert!(text.contains("the ask"), "thread lost the parent: {text}");
    assert!(text.contains("the answer"), "thread lost the reply: {text}");
    Ok(())
}

/// A thread of one is the message. Indexing it twice would put two
/// near-identical vectors in front of every query that matches it.
#[test]
fn a_lone_message_produces_no_thread_record() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(
        dir.path(),
        "Inbox",
        "u1",
        "<a@example.invalid>",
        "",
        "alone",
    )?;

    let index = indexed(&dir)?;

    assert!(index.records_of_kind("mail-thread")?.is_empty());
    Ok(())
}

/// A reply to a message the discriminant excluded joins no thread. The scope
/// bound is not a reply's to widen.
#[test]
fn a_thread_does_not_reach_outside_the_increment() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(
        dir.path(),
        "Inbox",
        "u1",
        "<bulk@example.invalid>",
        "List-Id: <announce.example.invalid>\n",
        "a newsletter",
    )?;
    deliver(
        dir.path(),
        "Inbox",
        "u2",
        "<b@example.invalid>",
        "In-Reply-To: <bulk@example.invalid>\n",
        "a human reply",
    )?;

    let index = indexed(&dir)?;

    assert_eq!(
        index.records_in("mail")?,
        vec!["<b@example.invalid>".to_owned()],
        "the excluded parent was dragged in"
    );
    Ok(())
}

/// The index is disposable, so rebuilding must converge rather than
/// accumulate.
#[test]
fn rebuilding_twice_leaves_the_same_index() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(dir.path(), "Inbox", "u1", "<a@example.invalid>", "", "one")?;
    deliver(
        dir.path(),
        "Inbox",
        "u2",
        "<b@example.invalid>",
        "In-Reply-To: <a@example.invalid>\n",
        "two",
    )?;
    let corpus = MaildirCorpus::scan(dir.path())?;
    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &BTreeSet::new());
    let index = Index::open_in_memory()?;

    let first = rebuild_mail(&corpus, &selection, &index)?;
    let before = index.dump()?;
    let second = rebuild_mail(&corpus, &selection, &index)?;

    assert_eq!(first.records, second.records);
    assert_eq!(first.passages, second.passages);
    assert_eq!(before, index.dump()?);
    Ok(())
}

/// Rebuilding one corpus must not disturb another: the two read different
/// canonical sources and fail independently.
#[test]
fn rebuilding_mail_leaves_other_corpora_alone() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(dir.path(), "Inbox", "u1", "<a@example.invalid>", "", "mail")?;
    let index = Index::open_in_memory()?;
    let store_dir = tempfile::tempdir()?;
    let store = store_note(&store_dir, "n1")?;
    kb::index::rebuild(&store, &index, &kb::index::Scope::All)?;
    let corpus = MaildirCorpus::scan(dir.path())?;
    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &BTreeSet::new());

    rebuild_mail(&corpus, &selection, &index)?;

    assert!(index.record("n1").is_ok(), "the kb record was cleared");
    assert!(index.record("<a@example.invalid>").is_ok());
    Ok(())
}

/// The mirror image, and the one that bit: a full rebuild from the store is
/// the scheduled `kb reindex`, and until T025 it cleared the mail rows in
/// its own transaction and left them absent until the mail rebuild's commit
/// seconds later. A query in that window answered from an index presented as
/// complete. The store-backed rebuild now replaces only what the store
/// backs; mail survives it untouched, catalogue and passages included.
#[test]
fn a_full_rebuild_from_the_store_leaves_mail_in_place() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(dir.path(), "Inbox", "u1", "<a@example.invalid>", "", "mail")?;
    let index = Index::open_in_memory()?;
    let corpus = MaildirCorpus::scan(dir.path())?;
    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &BTreeSet::new());
    rebuild_mail(&corpus, &selection, &index)?;
    let mail_passages_before = index.passages("<a@example.invalid>")?.len();
    assert!(
        mail_passages_before > 0,
        "the fixture message produced no passages"
    );
    let store_dir = tempfile::tempdir()?;
    let store = store_note(&store_dir, "n1")?;

    let report = kb::index::rebuild(&store, &index, &kb::index::Scope::All)?;

    assert_eq!(report.records, 1, "the store's record was rebuilt");
    assert!(index.record("n1").is_ok(), "the kb record was not rebuilt");
    assert!(
        index.record("<a@example.invalid>").is_ok(),
        "the full rebuild cleared the mail record"
    );
    assert_eq!(
        index.passages("<a@example.invalid>")?.len(),
        mail_passages_before,
        "the mail passages did not survive the store rebuild"
    );
    assert!(
        index.mail_catalogue("<a@example.invalid>")?.is_some(),
        "the mail catalogue was cleared by the store rebuild"
    );
    // And a second full rebuild still replaces the store's rows rather than
    // accumulating them.
    kb::index::rebuild(&store, &index, &kb::index::Scope::All)?;
    assert_eq!(index.records_in("kb")?.len(), 1);
    Ok(())
}

/// Scoped clearing is what makes the two rebuild paths independent.
#[test]
fn clearing_the_mail_corpus_leaves_the_catalogue_consistent() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(dir.path(), "Inbox", "u1", "<a@example.invalid>", "", "one")?;
    let index = indexed(&dir)?;

    let empty = MaildirCorpus::scan(&{
        let other = dir.path().join("empty");
        fs::create_dir_all(other.join("Inbox").join("cur"))?;
        other
    })?;
    rebuild_mail(&empty, &Selection::default(), &index)?;

    assert!(
        index.records_in("mail")?.is_empty(),
        "a rebuild from an empty Maildir left records behind"
    );
    assert_eq!(
        index.mail_catalogue("<a@example.invalid>")?,
        None,
        "a rebuild from an empty Maildir left catalogue rows behind"
    );
    Ok(())
}

/// Embedding walks a corpus's passages, so the accessor has to return them
/// all in a stable order rather than a record at a time.
#[test]
fn a_corpus_yields_its_passages_in_record_and_stream_order() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(dir.path(), "Inbox", "u1", "<a@example.invalid>", "", "one")?;
    deliver(
        dir.path(),
        "Inbox",
        "u2",
        "<b@example.invalid>",
        "In-Reply-To: <a@example.invalid>\n",
        "two",
    )?;

    let index = indexed(&dir)?;

    let passages = index.passages_in("mail")?;
    assert_eq!(passages.len(), 5, "two messages plus a three-span thread");
    let ordering: Vec<_> = passages
        .iter()
        .map(|p| (p.record_id.clone(), p.span_start))
        .collect();
    let mut sorted = ordering.clone();
    sorted.sort();
    assert_eq!(ordering, sorted, "passages came back unordered");
    assert!(index.passages_in("kb")?.is_empty());
    Ok(())
}

/// The catalogue explains why a message is in the index despite the rule, so
/// a measurement can state what fraction of the corpus the rule did not
/// select.
#[test]
fn a_readmitted_message_is_recorded_as_an_override() -> TestResult {
    let dir = tempfile::tempdir()?;
    let folder = dir.path().join("Inbox").join("cur");
    fs::create_dir_all(&folder)?;
    fs::write(
        folder.join("u1:2,S"),
        "From: news@example.invalid\n\
         List-Id: <announce.example.invalid>\n\
         Message-ID: <needed@example.invalid>\n\
         \n\
         a newsletter that answers a question\n",
    )?;
    let corpus = MaildirCorpus::scan(dir.path())?;
    let ground_truth = BTreeSet::from(["<needed@example.invalid>".to_owned()]);
    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &ground_truth);
    let index = Index::open_in_memory()?;

    rebuild_mail(&corpus, &selection, &index)?;

    let entry = index
        .mail_catalogue("<needed@example.invalid>")?
        .ok_or("not catalogued")?;
    assert!(entry.ground_truth_override);
    Ok(())
}

/// A vector for part of a passage is not an orphan. Long messages embed as
/// several spans, so requiring an exact span match discards every chunk of
/// every long document on the next rebuild — 4,175 of 11,504 vectors, and
/// most of an hour's work, when this was first run against the real corpus.
#[test]
fn a_chunk_of_a_passage_survives_a_rebuild() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(
        dir.path(),
        "Inbox",
        "u1",
        "<a@example.invalid>",
        "",
        "a long message",
    )?;
    let corpus = MaildirCorpus::scan(dir.path())?;
    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &BTreeSet::new());
    let index = Index::open_in_memory()?;
    rebuild_mail(&corpus, &selection, &index)?;
    let passage = index
        .passages("<a@example.invalid>")?
        .into_iter()
        .next()
        .ok_or("no passage")?;
    // A sub-span, as chunking produces for a message past the model's window.
    let half = passage.span_len / 2;
    index.put_embedding(
        &passage.stream_hash,
        passage.span_start,
        half,
        "m",
        &[1, 2, 3, 4],
    )?;

    rebuild_mail(&corpus, &selection, &index)?;

    assert!(
        index
            .embedding(&passage.stream_hash, passage.span_start, half, "m")?
            .is_some(),
        "a chunk of a passage was mistaken for an orphan"
    );
    Ok(())
}

/// A vector for text that is genuinely gone must still be dropped, or the
/// index accumulates vectors nothing addresses.
#[test]
fn a_vector_for_vanished_text_is_still_dropped() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(dir.path(), "Inbox", "u1", "<a@example.invalid>", "", "one")?;
    let corpus = MaildirCorpus::scan(dir.path())?;
    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &BTreeSet::new());
    let index = Index::open_in_memory()?;
    index.put_embedding("a-stream-nothing-addresses", 0, 10, "m", &[1, 2, 3, 4])?;

    rebuild_mail(&corpus, &selection, &index)?;

    assert!(
        index
            .embedding("a-stream-nothing-addresses", 0, 10, "m")?
            .is_none()
    );
    Ok(())
}

/// A span reaching past the passage it claims to be part of addresses text
/// that is not there, whatever its stream hash says.
#[test]
fn a_span_overrunning_its_passage_is_an_orphan() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(dir.path(), "Inbox", "u1", "<a@example.invalid>", "", "one")?;
    let corpus = MaildirCorpus::scan(dir.path())?;
    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &BTreeSet::new());
    let index = Index::open_in_memory()?;
    rebuild_mail(&corpus, &selection, &index)?;
    let passage = index
        .passages("<a@example.invalid>")?
        .into_iter()
        .next()
        .ok_or("no passage")?;
    index.put_embedding(
        &passage.stream_hash,
        passage.span_start,
        passage.span_len + 1,
        "m",
        &[1, 2, 3, 4],
    )?;

    rebuild_mail(&corpus, &selection, &index)?;

    assert!(
        index
            .embedding(
                &passage.stream_hash,
                passage.span_start,
                passage.span_len + 1,
                "m"
            )?
            .is_none()
    );
    Ok(())
}

/// Rebuilding everything must survive the presence of passages the FTS never
/// indexed. `passages_fts` is an external-content table, so emptying it with
/// a plain DELETE reads each row's text back out of `passages` to know which
/// tokens to remove — and for a mail row there are none, which corrupts the
/// index. The failure surfaces later, as a rebuild that refuses, while
/// `PRAGMA integrity_check` still reports ok because the damage is
/// FTS5-internal rather than structural.
#[test]
fn a_full_rebuild_survives_passages_the_fts_never_indexed() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(
        dir.path(),
        "Inbox",
        "u1",
        "<a@example.invalid>",
        "",
        "mail text",
    )?;
    let store_dir = tempfile::tempdir()?;
    let store = store_note(&store_dir, "n1")?;
    let index = Index::open_in_memory()?;
    kb::index::rebuild(&store, &index, &kb::index::Scope::All)?;
    let corpus = MaildirCorpus::scan(dir.path())?;
    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &BTreeSet::new());
    rebuild_mail(&corpus, &selection, &index)?;

    kb::index::rebuild(&store, &index, &kb::index::Scope::All)?;

    assert!(index.record("n1").is_ok());
    assert!(
        !index.search_text("body")?.is_empty(),
        "the kb corpus stopped being searchable after a rebuild"
    );
    Ok(())
}
