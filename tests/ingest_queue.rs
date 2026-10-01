//! The ingest queue's durability contract (T022).
//!
//! The endpoint acknowledges a submission only once its bytes are on disk, so
//! everything downstream — the worker, the dead-letter directory, the status
//! surface — rests on what this module promises about files. These tests hold
//! it to that: a submission survives the process that accepted it, a partial
//! write is never mistaken for a complete one, and nothing is discarded
//! silently.

use kb::ingest::{Queue, Submission};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn submission(id: &str, body: &str) -> Submission {
    Submission {
        id: id.to_owned(),
        corpus: "kb".to_owned(),
        document: format!("* {id}\n\n{body}\n"),
        provenance: None,
    }
}

/// The property acknowledgement claims: the bytes outlive the process.
#[test]
fn an_accepted_submission_survives_the_process_that_accepted_it() -> TestResult {
    let dir = tempfile::tempdir()?;
    {
        let queue = Queue::open(dir.path())?;
        queue.enqueue(&submission("cc-1", "the session"))?;
    }
    let queue = Queue::open(dir.path())?;
    let pending = queue.pending()?;
    assert_eq!(pending.len(), 1, "the submission did not survive");
    let first = pending.first().ok_or("nothing is pending")?;
    assert_eq!(first.submission.id, "cc-1");
    assert!(first.submission.document.contains("the session"));
    Ok(())
}

/// A file still being written is not a submission. The write goes to a
/// temporary name and is renamed into place, because a reader that saw the
/// partial file would parse half a document and dead-letter a session that
/// arrived intact.
#[test]
fn a_half_written_file_is_not_offered_to_the_worker() -> TestResult {
    let dir = tempfile::tempdir()?;
    let queue = Queue::open(dir.path())?;
    std::fs::write(dir.path().join("incoming/cc-2.json.partial"), b"{\"id\":")?;
    assert!(
        queue.pending()?.is_empty(),
        "a partially written file was offered as a submission"
    );
    Ok(())
}

/// Re-posting a session is a no-op rather than a second copy: records carry
/// stable ids, so the same session posted twice is the same submission.
#[test]
fn posting_the_same_session_twice_leaves_one_submission() -> TestResult {
    let dir = tempfile::tempdir()?;
    let queue = Queue::open(dir.path())?;
    queue.enqueue(&submission("cc-3", "first"))?;
    queue.enqueue(&submission("cc-3", "second"))?;
    let pending = queue.pending()?;
    assert_eq!(pending.len(), 1, "the same id enqueued twice");
    let only = pending.first().ok_or("nothing is pending")?;
    assert!(
        only.submission.document.contains("second"),
        "the later post did not supersede the earlier one"
    );
    Ok(())
}

/// What the worker cannot process is kept and counted, never dropped.
#[test]
fn a_submission_the_worker_refuses_is_kept_and_counted() -> TestResult {
    let dir = tempfile::tempdir()?;
    let queue = Queue::open(dir.path())?;
    queue.enqueue(&submission("cc-4", "unprocessable"))?;
    let pending = queue.pending()?;
    let waiting = pending.first().ok_or("nothing is pending")?;

    queue.dead_letter(waiting, "the embedding endpoint refused")?;

    assert!(queue.pending()?.is_empty(), "it is still pending");
    let status = queue.status()?;
    assert_eq!(status.depth, 0);
    assert_eq!(status.dead_lettered, 1);
    let kept = queue.dead()?;
    assert_eq!(kept.len(), 1);
    let refused = kept.first().ok_or("nothing was dead-lettered")?;
    assert!(
        refused.reason.contains("embedding endpoint"),
        "the reason was not kept: {:?}",
        refused.reason
    );
    assert!(refused.submission.document.contains("unprocessable"));
    Ok(())
}

/// Completing a submission removes it, and only it.
#[test]
fn completing_one_submission_leaves_the_rest() -> TestResult {
    let dir = tempfile::tempdir()?;
    let queue = Queue::open(dir.path())?;
    queue.enqueue(&submission("cc-5", "done"))?;
    queue.enqueue(&submission("cc-6", "waiting"))?;
    let pending = queue.pending()?;
    let first = pending
        .iter()
        .find(|p| p.submission.id == "cc-5")
        .ok_or("cc-5 is not pending")?;

    queue.complete(first)?;

    let left = queue.pending()?;
    assert_eq!(left.len(), 1);
    assert_eq!(
        left.first().ok_or("nothing is pending")?.submission.id,
        "cc-6"
    );
    Ok(())
}

/// The three observables the failure mode needs. Depth alone cannot
/// distinguish a queue that is empty because everything was ingested from one
/// that is empty because everything was dead-lettered.
#[test]
fn status_reports_depth_dead_letters_and_the_oldest_wait() -> TestResult {
    let dir = tempfile::tempdir()?;
    let queue = Queue::open(dir.path())?;
    let empty = queue.status()?;
    assert_eq!(empty.depth, 0);
    assert_eq!(empty.dead_lettered, 0);
    assert!(empty.oldest.is_none(), "an empty queue has no oldest entry");

    queue.enqueue(&submission("cc-7", "waiting"))?;
    let status = queue.status()?;
    assert_eq!(status.depth, 1);
    assert!(
        status.oldest.is_some(),
        "a queue with an entry reports how long it has waited"
    );
    Ok(())
}

// ── The refusals and the faults ─────────────────────────────────────────
//
// A queue that only works when the filesystem cooperates is a queue that
// loses submissions when it does not. These exercise the paths that report a
// fault rather than the ones that do the work.

/// A submission with no id has nowhere to be stored under and is refused
/// before anything is written.
#[test]
fn a_submission_with_no_id_is_refused() -> TestResult {
    let dir = tempfile::tempdir()?;
    let queue = Queue::open(dir.path())?;
    let refused = queue.enqueue(&Submission {
        id: "   ".to_owned(),
        corpus: "kb".to_owned(),
        document: "* Note\n\nbody.\n".to_owned(),
        provenance: None,
    });
    assert!(refused.is_err(), "an idless submission was accepted");
    assert!(queue.pending()?.is_empty());
    Ok(())
}

/// An id is a remote caller's string, and it becomes a file name. One that
/// traverses would write outside the queue directory entirely.
#[test]
fn an_id_that_traverses_cannot_escape_the_queue() -> TestResult {
    let dir = tempfile::tempdir()?;
    let queue = Queue::open(&dir.path().join("queue"))?;
    queue.enqueue(&Submission {
        id: "../../escaped".to_owned(),
        corpus: "kb".to_owned(),
        document: "* Note\n\nbody.\n".to_owned(),
        provenance: None,
    })?;
    assert_eq!(queue.pending()?.len(), 1);
    let written: Vec<_> = std::fs::read_dir(dir.path())?
        .filter_map(Result::ok)
        .map(|entry| entry.file_name())
        .collect();
    assert_eq!(
        written,
        vec![std::ffi::OsString::from("queue")],
        "{written:?}"
    );
    Ok(())
}

/// A queue directory that cannot be created is a fault the caller must see,
/// not something to carry on without.
#[test]
fn a_queue_that_cannot_be_created_reports_it() -> TestResult {
    let dir = tempfile::tempdir()?;
    let blocked = dir.path().join("queue");
    std::fs::write(&blocked, b"not a directory")?;
    assert!(
        Queue::open(&blocked).is_err(),
        "a file was opened as a queue"
    );
    Ok(())
}

/// A queue file that reached its final name is a promise that it parses.
/// Breaking that promise is a fault rather than something to skip, because
/// skipping it would silently drop a session that was acknowledged.
#[test]
fn a_complete_file_that_does_not_parse_is_a_fault() -> TestResult {
    let dir = tempfile::tempdir()?;
    let queue = Queue::open(dir.path())?;
    std::fs::write(dir.path().join("incoming/cc-9.json"), b"{not json")?;
    let fault = queue
        .pending()
        .err()
        .ok_or("a corrupt submission was read as empty")?;
    assert!(
        fault.to_string().contains("cc-9"),
        "the fault does not name the file: {fault}"
    );
    Ok(())
}

/// A queue whose directories have gone is a fault on every operation, rather
/// than an empty queue that quietly reports nothing to do.
#[test]
fn a_queue_directory_that_disappears_is_reported() -> TestResult {
    let dir = tempfile::tempdir()?;
    let queue = Queue::open(dir.path())?;
    std::fs::remove_dir_all(dir.path().join("incoming"))?;
    assert!(queue.pending().is_err(), "a missing queue read as empty");
    assert!(queue.status().is_err());
    assert!(
        queue
            .enqueue(&Submission {
                id: "cc-10".to_owned(),
                corpus: "kb".to_owned(),
                document: "* Note\n\nbody.\n".to_owned(),
                provenance: None,
            })
            .is_err(),
        "a submission was accepted into a queue that is not there"
    );
    Ok(())
}

/// Completing something already gone is success: two workers racing on one
/// submission is not a fault, and the second finding it gone is the outcome
/// either way.
#[test]
fn completing_a_submission_twice_is_not_a_fault() -> TestResult {
    let dir = tempfile::tempdir()?;
    let queue = Queue::open(dir.path())?;
    queue.enqueue(&submission("cc-11", "body"))?;
    let pending = queue.pending()?;
    let entry = pending.first().ok_or("nothing is pending")?;
    queue.complete(entry)?;
    queue.complete(entry)?;
    assert!(queue.pending()?.is_empty());
    Ok(())
}

/// Removing something that is not a file is a fault, not a silent success.
/// The only tolerated absence is the file already being gone.
#[test]
fn a_queue_entry_that_cannot_be_removed_is_reported() -> TestResult {
    let dir = tempfile::tempdir()?;
    let queue = Queue::open(dir.path())?;
    queue.enqueue(&submission("cc-12", "body"))?;
    let pending = queue.pending()?;
    let entry = pending.first().ok_or("nothing is pending")?.clone();
    // Replace the file with a directory: removing it now fails for a reason
    // that is neither success nor "already gone".
    std::fs::remove_file(&entry.path)?;
    std::fs::create_dir(&entry.path)?;
    assert!(
        queue.complete(&entry).is_err(),
        "a failed removal was reported as success"
    );
    Ok(())
}
