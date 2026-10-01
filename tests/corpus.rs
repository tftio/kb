//! Behavior of the per-corpus resolver interface.
//!
//! The surface is deliberately two functions. Format knowledge is
//! irreducible — a Maildir is not a git store — but it belongs in resolution
//! rather than spread through chunking, embedding and indexing, which are
//! shared and corpus-agnostic.

use kb::corpus::{Corpus, CorpusError, KbCorpus, Registry, Storage};
use kb::record::{ArtifactKind, ContentHash, CorpusId, RecordHeader, RecordId, SourceRef};
use kb::store::{BlobStore, GitBlobStore, RefName};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn store_note(store: &GitBlobStore, id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let raw = format!("* Note {id}\n\nbody for {id}.\n");
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

/// Resolution returns the bytes that arrived, which is what a citation has to
/// be checkable against.
#[test]
fn the_kb_corpus_resolves_a_record_to_its_archival_bytes() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(dir.path())?;
    store_note(&store, "n1")?;
    let corpus = KbCorpus::new(store);

    let bytes = corpus.resolve(&SourceRef::new("node-id", "n1")?)?;

    assert_eq!(String::from_utf8(bytes)?, "* Note n1\n\nbody for n1.\n");
    Ok(())
}

/// An id the corpus does not hold is a domain answer naming the id, not a
/// bare I/O failure the caller has to interpret.
#[test]
fn resolving_an_unknown_id_names_it() -> TestResult {
    let dir = tempfile::tempdir()?;
    let corpus = KbCorpus::new(GitBlobStore::open_or_init(dir.path())?);

    let outcome = corpus.resolve(&SourceRef::new("node-id", "absent")?);

    assert!(
        matches!(&outcome, Err(CorpusError::NotFound { id, .. }) if id == "absent"),
        "outcome was: {outcome:?}"
    );
    Ok(())
}

/// Enumeration is paged and resumable from a cursor. kb could be walked whole
/// — the store is local — but Slack and Bluesky resolvers reach over
/// rate-limited APIs where `enumerate` cannot be called casually, and
/// retrofitting paging into an interface with three implementations is worse
/// than designing it with one.
#[test]
fn enumeration_is_paged_and_resumable() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(dir.path())?;
    for id in ["n1", "n2", "n3", "n4", "n5"] {
        store_note(&store, id)?;
    }
    let corpus = KbCorpus::new(store);

    let first = corpus.enumerate(None, 2)?;
    assert_eq!(first.ids.len(), 2);
    let second = corpus.enumerate(first.next.as_deref(), 2)?;
    let third = corpus.enumerate(second.next.as_deref(), 2)?;

    let mut seen: Vec<String> = first
        .ids
        .iter()
        .chain(&second.ids)
        .chain(&third.ids)
        .map(|s| s.value().to_owned())
        .collect();
    seen.sort();
    assert_eq!(seen, vec!["n1", "n2", "n3", "n4", "n5"]);
    assert!(third.next.is_none(), "the last page ends the walk");
    Ok(())
}

/// Resuming from a cursor never repeats or skips, which is what makes an
/// interrupted enumeration safe to restart rather than something to redo.
#[test]
fn a_resumed_enumeration_neither_repeats_nor_skips() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(dir.path())?;
    for id in ["a", "b", "c", "d"] {
        store_note(&store, id)?;
    }
    let corpus = KbCorpus::new(store);

    let page = corpus.enumerate(None, 3)?;
    let resumed = corpus.enumerate(page.next.as_deref(), 3)?;

    let first: Vec<&str> = page.ids.iter().map(kb::record::SourceRef::value).collect();
    let rest: Vec<&str> = resumed
        .ids
        .iter()
        .map(kb::record::SourceRef::value)
        .collect();
    assert_eq!(first, vec!["a", "b", "c"]);
    assert_eq!(rest, vec!["d"]);
    Ok(())
}

/// Whether a corpus copies into the store or only references it decides
/// whether a dead external reference is a bug or expected, so it is declared
/// rather than inferred.
#[test]
fn every_corpus_declares_how_it_stores() -> TestResult {
    let dir = tempfile::tempdir()?;
    let corpus = KbCorpus::new(GitBlobStore::open_or_init(dir.path())?);

    assert_eq!(corpus.storage(), Storage::CopyIntoStore);
    assert_eq!(
        Registry::default().storage_of("mail"),
        Some(Storage::ReferenceOnly)
    );
    Ok(())
}

/// The vocabulary is the ground-truth question set's, not the registry's.
/// A registry that invented an identifier would silently break the join
/// between an index row's corpus and an `expect` entry's, and the break would
/// read as a retrieval regression rather than a naming mismatch.
#[test]
fn the_registry_conforms_to_the_question_sets_vocabulary() -> TestResult {
    let toml = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/resources/eval/retrieval-questions.toml"
    ))?;
    let known = toml
        .lines()
        .find(|l| l.trim_start().starts_with("known ="))
        .ok_or("the question set records no corpus vocabulary")?;

    for id in Registry::default().ids() {
        assert!(
            known.contains(&format!("\"{id}\"")),
            "registry defines {id:?}, which the question set does not list: {known}"
        );
    }
    Ok(())
}

/// A second implementation, kept in tests, so the interface is exercised by
/// something that is not the store it was designed around — without touching
/// mail, which is T015's.
#[test]
fn a_second_corpus_exercises_the_interface() -> TestResult {
    struct Stub {
        docs: Vec<(String, Vec<u8>)>,
    }
    impl Corpus for Stub {
        fn id(&self) -> &'static str {
            "mail"
        }
        fn storage(&self) -> Storage {
            Storage::ReferenceOnly
        }
        fn resolve(&self, source: &SourceRef) -> Result<Vec<u8>, CorpusError> {
            self.docs
                .iter()
                .find(|(k, _)| k == source.value())
                .map(|(_, v)| v.clone())
                .ok_or_else(|| CorpusError::NotFound {
                    corpus: "mail".to_owned(),
                    id: source.value().to_owned(),
                })
        }
        fn enumerate(
            &self,
            cursor: Option<&str>,
            limit: usize,
        ) -> Result<kb::corpus::Page, CorpusError> {
            let start = cursor.and_then(|c| c.parse::<usize>().ok()).unwrap_or(0);
            let ids = self
                .docs
                .iter()
                .skip(start)
                .take(limit)
                .map(|(k, _)| SourceRef::new("message-id", k))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| CorpusError::Unusable {
                    corpus: "mail".to_owned(),
                    reason: e.to_string(),
                })?;
            let next = start + ids.len();
            Ok(kb::corpus::Page {
                next: (next < self.docs.len()).then(|| next.to_string()),
                ids,
            })
        }
    }

    let stub = Stub {
        docs: vec![
            ("<m1@x>".to_owned(), b"first message".to_vec()),
            ("<m2@x>".to_owned(), b"second message".to_vec()),
        ],
    };

    assert_eq!(stub.storage(), Storage::ReferenceOnly);
    assert_eq!(
        stub.resolve(&SourceRef::new("message-id", "<m2@x>")?)?,
        b"second message".to_vec()
    );
    let page = stub.enumerate(None, 1)?;
    assert_eq!(page.ids.len(), 1);
    assert!(page.next.is_some());
    Ok(())
}

/// A record whose bytes are not a record is a broken store, not a missing
/// id, and the two want different responses: one is re-derivable, the other
/// is corruption.
#[test]
fn a_record_that_is_not_a_record_is_reported_as_unusable() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(dir.path())?;
    let hash = store.put(b"this is not a kb record\n")?;
    store.set_ref(&RefName::new("kb/broken")?, &hash)?;
    let corpus = KbCorpus::new(store);

    let outcome = corpus.resolve(&SourceRef::new("node-id", "broken")?);

    assert!(
        matches!(&outcome, Err(CorpusError::Unusable { corpus, reason })
                 if corpus == "kb" && reason.contains("broken")),
        "outcome was: {outcome:?}"
    );
    Ok(())
}

/// An id that cannot even be formed into a record name is refused rather than
/// passed through to the store, where it would fail as something else.
#[test]
fn an_unusable_identifier_is_refused() -> TestResult {
    let dir = tempfile::tempdir()?;
    let corpus = KbCorpus::new(GitBlobStore::open_or_init(dir.path())?);

    let outcome = corpus.resolve(&SourceRef::new("node-id", "has spaces and ~carets")?);

    assert!(outcome.is_err(), "outcome was: {outcome:?}");
    Ok(())
}

/// Enumerating an empty corpus ends immediately rather than paging forever.
#[test]
fn an_empty_corpus_enumerates_to_one_empty_page() -> TestResult {
    let dir = tempfile::tempdir()?;
    let corpus = KbCorpus::new(GitBlobStore::open_or_init(dir.path())?);

    let page = corpus.enumerate(None, 10)?;

    assert!(page.ids.is_empty());
    assert!(
        page.next.is_none(),
        "an empty walk must not ask to be resumed"
    );
    Ok(())
}

/// A corpus nobody registered is neither copy nor reference, and saying so is
/// better than guessing: guessing "copy" makes a dead reference read as
/// corruption, and guessing "reference" hides real corruption.
#[test]
fn an_unregistered_corpus_has_no_storage_class() {
    assert_eq!(Registry::default().storage_of("slack"), None);
    assert_eq!(Storage::ReferenceOnly.as_str(), "reference");
}
