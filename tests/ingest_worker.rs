//! The worker that turns queued submissions into records (T022).
//!
//! The queue's promise is that the bytes survive; this is what happens next.
//! The tests below are the four failure shapes that matter: the worker dying
//! mid-flight, the same session arriving twice, a submission that cannot be
//! made into a record, and the embedding endpoint being down — which under
//! this design drains the queue into dead letters while the endpoint carries
//! on acknowledging, and is therefore the thing to be able to see.

use kb::index::Index;
use kb::ingest::{Queue, Submission};
use kb::store::GitBlobStore;

type TestResult = Result<(), Box<dyn std::error::Error>>;

struct Workspace {
    _dir: tempfile::TempDir,
    store: GitBlobStore,
    index: Index,
    queue: Queue,
}

fn workspace() -> Result<Workspace, Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let store = GitBlobStore::open_or_init(&dir.path().join("store"))?;
    let index = Index::open_for_rebuild(&dir.path().join("index.db"))?;
    let queue = Queue::open(&dir.path().join("queue"))?;
    Ok(Workspace {
        _dir: dir,
        store,
        index,
        queue,
    })
}

fn submission(id: &str, body: &str) -> Submission {
    Submission {
        id: id.to_owned(),
        corpus: "kb".to_owned(),
        document: format!("* Session {id}\n\n{body}\n"),
        provenance: None,
    }
}

/// A submission that was accepted before the worker existed is ingested when
/// one appears: this is the crash between acknowledgement and processing,
/// which is the whole reason the bytes are on disk rather than in memory.
#[test]
fn a_submission_accepted_before_the_worker_started_is_ingested() -> TestResult {
    let work = workspace()?;
    work.queue
        .enqueue(&submission("cc-1", "the borrow checker"))?;

    let drained = kb::ingest::drain(&work.store, &work.index, &work.queue, None)?;

    assert_eq!(drained.ingested, 1);
    assert_eq!(drained.dead_lettered, 0);
    assert!(
        work.queue.pending()?.is_empty(),
        "the queue was not drained"
    );
    let expression = kb::storage::build_fts_query("borrow", kb::storage::MatchMode::Keywords);
    assert_eq!(
        work.index.search_text_in("kb", &expression, None, None)?,
        vec!["cc-1"]
    );
    Ok(())
}

/// Re-posting a session changes nothing. Records carry stable ids and the
/// document is the same, so the record hashes to the same address.
#[test]
fn re_posting_an_ingested_session_leaves_the_corpus_unchanged() -> TestResult {
    let work = workspace()?;
    work.queue.enqueue(&submission("cc-2", "unchanged"))?;
    kb::ingest::drain(&work.store, &work.index, &work.queue, None)?;
    let before = work.index.record("cc-2")?.record_hash;

    work.queue.enqueue(&submission("cc-2", "unchanged"))?;
    let again = kb::ingest::drain(&work.store, &work.index, &work.queue, None)?;

    assert_eq!(again.ingested, 1, "the re-post was not processed");
    assert_eq!(
        work.index.record("cc-2")?.record_hash,
        before,
        "re-posting the same session produced a different record"
    );
    Ok(())
}

/// A submission that cannot be made into a record is kept and counted rather
/// than dropped, and the queue moves on to the next one.
#[test]
fn a_submission_that_cannot_be_stored_is_dead_lettered_and_the_rest_proceed() -> TestResult {
    let work = workspace()?;
    // An id spanning lines is refused by the record definition, which is the
    // shape a malformed post actually takes: the queue accepted the bytes and
    // only the record layer can judge them.
    work.queue.enqueue(&Submission {
        id: "cc-3\nsecond line".to_owned(),
        corpus: "kb".to_owned(),
        document: "* Fine\n\nbody.\n".to_owned(),
        provenance: None,
    })?;
    work.queue.enqueue(&submission("cc-4", "ordinary"))?;

    let drained = kb::ingest::drain(&work.store, &work.index, &work.queue, None)?;

    assert_eq!(drained.ingested, 1, "the good submission was not ingested");
    assert_eq!(
        drained.dead_lettered, 1,
        "the bad one was not dead-lettered"
    );
    assert!(work.index.record("cc-4").is_ok());
    let status = work.queue.status()?;
    assert_eq!(status.depth, 0);
    assert_eq!(status.dead_lettered, 1);
    let kept = work.queue.dead()?;
    let refused = kept.first().ok_or("nothing was dead-lettered")?;
    assert!(
        !refused.reason.is_empty(),
        "a dead letter must say why it was refused"
    );
    Ok(())
}

/// A submission carrying the `conversation` tag is stored as a session
/// transcript, chunked on its turns rather than its sections, and its
/// provenance is carried through to the record header. This is the T005 fix:
/// `put_record` used to hardcode `kind: note` for every write, including the
/// queue worker's, so a captured transcript was chunked as though it were an
/// authored note.
#[test]
fn a_conversation_submission_is_stored_as_a_session_transcript_with_its_provenance() -> TestResult {
    let work = workspace()?;
    work.queue.enqueue(&Submission {
        id: "cc-6".to_owned(),
        corpus: "kb".to_owned(),
        document: "#+filetags: :conversation:\n\
                   * Human [2026-09-23 09:00]\n\nWhat did we decide?\n\n\
                   * Assistant [2026-09-23 09:01]\n\nTo carry provenance in the header.\n"
            .to_owned(),
        provenance: Some(kb::record::RawProvenance {
            project: Some("kb".to_owned()),
            project_source: Some("declared".to_owned()),
            remote: Some("github.com/tftio/kb".to_owned()),
            context: Some("personal".to_owned()),
            domains: vec!["clanker".to_owned()],
            harness: Some("claude-code".to_owned()),
            model: Some("claude-sonnet-5".to_owned()),
            session: Some("cc-6".to_owned()),
            cwd: Some("/Users/op/Projects/kb/main".to_owned()),
        }),
    })?;

    let drained = kb::ingest::drain(&work.store, &work.index, &work.queue, None)?;
    assert_eq!(drained.ingested, 1);
    assert_eq!(drained.dead_lettered, 0);

    let row = work.index.record("cc-6")?;
    assert_eq!(row.kind, "session-transcript");
    assert_eq!(row.project.as_deref(), Some("kb"));
    assert_eq!(row.project_source.as_deref(), Some("declared"));
    assert_eq!(row.remote.as_deref(), Some("github.com/tftio/kb"));
    assert_eq!(row.context.as_deref(), Some("personal"));
    assert_eq!(row.harness.as_deref(), Some("claude-code"));
    assert_eq!(row.model.as_deref(), Some("claude-sonnet-5"));
    assert_eq!(row.session.as_deref(), Some("cc-6"));
    assert_eq!(row.cwd.as_deref(), Some("/Users/op/Projects/kb/main"));
    assert_eq!(work.index.domains_of("cc-6")?, vec!["clanker".to_owned()]);

    let passages = work.index.passages("cc-6")?;
    assert_eq!(
        passages.len(),
        2,
        "a session transcript chunks on turns, not sections: {passages:?}"
    );
    Ok(())
}

/// A submission with no `conversation` tag is stored as an ordinary note.
#[test]
fn a_submission_without_the_conversation_tag_is_stored_as_a_note() -> TestResult {
    let work = workspace()?;
    work.queue
        .enqueue(&submission("cc-7", "an authored note, not a transcript"))?;

    kb::ingest::drain(&work.store, &work.index, &work.queue, None)?;

    assert_eq!(work.index.record("cc-7")?.kind, "note");
    Ok(())
}

/// The failure mode this shape introduces, made visible. With the embedding
/// endpoint unreachable the record is still stored and still findable by text
/// — losing that would make capture depend on the endpoint's health, which is
/// what this design exists to avoid — but the submission is dead-lettered so
/// the gap is counted rather than assumed away.
#[test]
fn an_unreachable_embedding_endpoint_dead_letters_but_still_stores() -> TestResult {
    let work = workspace()?;
    work.queue
        .enqueue(&submission("cc-5", "distinctive text"))?;
    // Port zero is never listening, so this is a refusal rather than a wait.
    let embedder =
        kb::cli_embed::CliEmbedder::from_optional_config(Some(kb::embedding::EmbeddingConfig {
            base_url: "http://127.0.0.1:1/v1".to_owned(),
            model: "fixture-model".to_owned(),
            api_key: None,
            document_prefix: String::new(),
            query_prefix: String::new(),
        }))?;

    let drained = kb::ingest::drain(&work.store, &work.index, &work.queue, Some(&embedder))?;

    assert_eq!(drained.ingested, 0);
    assert_eq!(drained.dead_lettered, 1);
    assert!(
        work.index.record("cc-5").is_ok(),
        "the record was not stored, so capture depends on the embedding endpoint"
    );
    let expression = kb::storage::build_fts_query("distinctive", kb::storage::MatchMode::Keywords);
    assert_eq!(
        work.index.search_text_in("kb", &expression, None, None)?,
        vec!["cc-5"]
    );
    let kept = work.queue.dead()?;
    let refused = kept.first().ok_or("nothing was dead-lettered")?;
    assert!(
        refused.reason.contains("embed"),
        "the reason does not name the embedding failure: {}",
        refused.reason
    );
    Ok(())
}
