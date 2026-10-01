//! The ingest queue: what an acknowledgement means (T022).
//!
//! Capture used to write through kb directly, so a session was captured only
//! if the store, the index and the embedding endpoint were all reachable at
//! the moment a terminal closed. This module is the durable middle: the
//! endpoint puts a submission here and acknowledges, and a worker turns it
//! into a record whenever it can.
//!
//! **Acknowledgement means the bytes are on disk. It does not mean the record
//! is ingested.** Those are different claims and conflating them is what makes
//! the failure mode of this shape invisible: the accept layer keeps returning
//! success while the worker drains everything into the dead-letter directory.
//! [`Status`] therefore reports three numbers rather than one.
//!
//! Durability is fsync-and-rename. The submission is written to a temporary
//! name, flushed to the platform, renamed into place, and the directory itself
//! flushed — because a rename is metadata, and a directory entry that has not
//! reached the disk is a submission that vanishes in a power cut after it was
//! acknowledged.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// What a client posts: a rendered document and the identity to store it under.
///
/// The document is rendered on the client, because rendering is also the
/// redaction — see the Decision Log entry of 2026-08-21. What arrives here has
/// already had tool output dropped, so the queue and the dead-letter directory
/// hold no material the store would not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Submission {
    /// The record id to store under. Stable per session, which is what makes
    /// re-posting a no-op.
    pub id: String,
    /// The corpus the record belongs to.
    pub corpus: String,
    /// The rendered artifact.
    pub document: String,
    /// Where the work that produced this submission was happening, if the
    /// client knows. Raw, unvalidated JSON: the worker turns it into the
    /// record header's typed [`crate::record::Provenance`] at the point it
    /// writes the record, which is the boundary ENG-006 asks for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<crate::record::RawProvenance>,
}

/// A submission waiting to be processed, and where it is.
#[derive(Debug, Clone)]
pub struct Pending {
    /// What was posted.
    pub submission: Submission,
    /// The file holding it, which is what [`Queue::complete`] removes.
    pub path: PathBuf,
    /// When it was accepted.
    pub accepted: SystemTime,
}

/// A submission the worker refused, kept with the reason it was refused.
#[derive(Debug, Clone)]
pub struct DeadLettered {
    /// What was posted.
    pub submission: Submission,
    /// Why the worker could not process it.
    pub reason: String,
    /// The file holding it.
    pub path: PathBuf,
}

/// The three numbers that distinguish a working queue from a failing one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Status {
    /// Submissions waiting.
    pub depth: usize,
    /// Submissions the worker refused.
    pub dead_lettered: usize,
    /// How long the oldest waiting submission has waited, if any.
    pub oldest: Option<Duration>,
}

/// Why a queue operation failed.
#[derive(Debug, Error)]
pub enum IngestError {
    /// The queue directory could not be read or written.
    #[error("queue io at {path}: {source}")]
    Io {
        /// What was being read or written.
        path: PathBuf,
        /// The underlying failure.
        source: std::io::Error,
    },
    /// A queue file did not hold a submission.
    #[error("malformed submission at {path}: {reason}")]
    Malformed {
        /// The file.
        path: PathBuf,
        /// What was wrong with it.
        reason: String,
    },
    /// A submission was refused before it was written.
    #[error("{0}")]
    Refused(String),
}

/// The suffix a complete submission carries. A file being written carries
/// another, so a reader cannot mistake one for the other.
const SUBMISSION: &str = "json";

/// The on-disk shape of a queue file: the submission plus what the queue knows
/// about it. Separate from [`Submission`] because the reason a dead letter was
/// refused is the queue's knowledge, not the client's claim.
#[derive(Debug, Serialize, Deserialize)]
struct Entry {
    submission: Submission,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

/// Build the io fault for one path.
///
/// One construction site rather than eight. The repetition it replaces was
/// not only noise: each copy was a separate arm that no test could reach
/// without a filesystem that fails on demand, which made the module look far
/// less exercised than it is.
fn io(path: &Path) -> impl Fn(std::io::Error) -> IngestError + '_ {
    move |source| IngestError::Io {
        path: path.to_owned(),
        source,
    }
}

/// Where the capture queue lives when nothing says otherwise.
///
/// Beside the store it feeds, so a backup of one directory covers both what
/// is stored and what is waiting to be. `KB_QUEUE_PATH` overrides it at every
/// entry point that opens a queue.
#[must_use]
pub fn default_queue_path() -> PathBuf {
    dirs::home_dir().map_or_else(
        || PathBuf::from("queue"),
        |home| home.join(".local/share/kb/queue"),
    )
}

/// A durable queue rooted at a directory.
#[derive(Debug, Clone)]
pub struct Queue {
    incoming: PathBuf,
    dead: PathBuf,
}

impl Queue {
    /// Open the queue under `root`, creating its directories.
    ///
    /// # Errors
    ///
    /// [`IngestError::Io`] if the directories cannot be created.
    pub fn open(root: &Path) -> Result<Self, IngestError> {
        let incoming = root.join("incoming");
        let dead = root.join("dead");
        for directory in [&incoming, &dead] {
            std::fs::create_dir_all(directory).map_err(io(directory))?;
        }
        Ok(Self { incoming, dead })
    }

    /// Accept a submission, returning only once its bytes are durable.
    ///
    /// # Errors
    ///
    /// [`IngestError::Refused`] for a submission with no id, and
    /// [`IngestError::Io`] if it cannot be written.
    pub fn enqueue(&self, submission: &Submission) -> Result<PathBuf, IngestError> {
        if submission.id.trim().is_empty() {
            return Err(IngestError::Refused("a submission needs an id".to_owned()));
        }
        let entry = Entry {
            submission: submission.clone(),
            reason: None,
        };
        let destination = self
            .incoming
            .join(format!("{}.{SUBMISSION}", file_stem(&submission.id)));
        write_durably(&destination, &entry)?;
        Ok(destination)
    }

    /// Every submission waiting, oldest first.
    ///
    /// # Errors
    ///
    /// [`IngestError::Io`] if the directory cannot be read, or
    /// [`IngestError::Malformed`] if a complete submission does not parse — a
    /// file that reached its final name is a promise that it is readable, so
    /// failing to read one is a fault rather than something to skip.
    pub fn pending(&self) -> Result<Vec<Pending>, IngestError> {
        let mut waiting = Vec::new();
        for (path, entry, accepted) in Self::entries(&self.incoming)? {
            waiting.push(Pending {
                submission: entry.submission,
                path,
                accepted,
            });
        }
        waiting.sort_by_key(|pending| pending.accepted);
        Ok(waiting)
    }

    /// Every submission the worker refused.
    ///
    /// # Errors
    ///
    /// As [`Queue::pending`].
    pub fn dead(&self) -> Result<Vec<DeadLettered>, IngestError> {
        let mut refused = Vec::new();
        for (path, entry, _) in Self::entries(&self.dead)? {
            refused.push(DeadLettered {
                submission: entry.submission,
                reason: entry.reason.unwrap_or_default(),
                path,
            });
        }
        Ok(refused)
    }

    /// Move a submission to the dead-letter directory with the reason.
    ///
    /// Written durably before the original is removed: the reverse order can
    /// lose a submission entirely, which is the one thing this queue promises
    /// not to do.
    ///
    /// # Errors
    ///
    /// [`IngestError::Io`] if it cannot be written or removed.
    pub fn dead_letter(&self, pending: &Pending, reason: &str) -> Result<(), IngestError> {
        let entry = Entry {
            submission: pending.submission.clone(),
            reason: Some(reason.to_owned()),
        };
        let destination = self.dead.join(format!(
            "{}.{SUBMISSION}",
            file_stem(&pending.submission.id)
        ));
        write_durably(&destination, &entry)?;
        remove(&pending.path)
    }

    /// Remove a submission that has been ingested.
    ///
    /// # Errors
    ///
    /// [`IngestError::Io`] if it cannot be removed.
    pub fn complete(&self, pending: &Pending) -> Result<(), IngestError> {
        remove(&pending.path)
    }

    /// Depth, dead-letter count, and how long the oldest submission has waited.
    ///
    /// # Errors
    ///
    /// As [`Queue::pending`].
    pub fn status(&self) -> Result<Status, IngestError> {
        let waiting = self.pending()?;
        let now = SystemTime::now();
        let oldest = waiting
            .first()
            .and_then(|pending| now.duration_since(pending.accepted).ok());
        Ok(Status {
            depth: waiting.len(),
            dead_lettered: self.dead()?.len(),
            oldest,
        })
    }

    /// Read every complete entry in one directory.
    fn entries(directory: &Path) -> Result<Vec<(PathBuf, Entry, SystemTime)>, IngestError> {
        let listing = std::fs::read_dir(directory).map_err(io(directory))?;
        let mut found = Vec::new();
        for item in listing {
            let item = item.map_err(io(directory))?;
            let path = item.path();
            if path.extension().and_then(std::ffi::OsStr::to_str) != Some(SUBMISSION) {
                continue;
            }
            let bytes = std::fs::read(&path).map_err(io(&path))?;
            let entry: Entry =
                serde_json::from_slice(&bytes).map_err(|e| IngestError::Malformed {
                    path: path.clone(),
                    reason: e.to_string(),
                })?;
            let accepted = item
                .metadata()
                .and_then(|meta| meta.modified())
                .unwrap_or_else(|_| SystemTime::now());
            found.push((path, entry, accepted));
        }
        Ok(found)
    }
}

/// A file name for `id` that cannot escape the queue directory.
///
/// Record ids are constrained elsewhere, but this is the boundary a remote
/// caller reaches first, and a name that traverses would write outside the
/// queue.
fn file_stem(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Write `entry` to `destination` so that it is there after a power cut.
fn write_durably(destination: &Path, entry: &Entry) -> Result<(), IngestError> {
    use std::io::Write as _;

    let bytes = serde_json::to_vec_pretty(entry).map_err(|e| IngestError::Malformed {
        path: destination.to_owned(),
        reason: e.to_string(),
    })?;
    let staging = destination.with_extension("json.partial");
    {
        let mut file = std::fs::File::create(&staging).map_err(io(&staging))?;
        file.write_all(&bytes).map_err(io(&staging))?;
        file.sync_all().map_err(io(&staging))?;
    }
    std::fs::rename(&staging, destination).map_err(io(destination))?;
    // The rename is metadata: without this the directory entry can be lost
    // even though the file's contents reached the disk.
    if let Some(parent) = destination.parent() {
        std::fs::File::open(parent)
            .and_then(|dir| dir.sync_all())
            .map_err(io(parent))?;
    }
    Ok(())
}

/// Remove a queue file, treating an already-absent file as success: two
/// workers racing on one submission is not a fault, and the second one
/// finding it gone is the outcome either way.
fn remove(path: &Path) -> Result<(), IngestError> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(io(path)(source)),
    }
}

/// What one pass over the queue did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Drained {
    /// Submissions turned into records.
    pub ingested: usize,
    /// Submissions kept with a reason instead.
    pub dead_lettered: usize,
}

/// Turn every waiting submission into a record.
///
/// The order is store, index, then embed, and it is deliberate. Storing first
/// means a session survives an embedding endpoint that is down, which is the
/// coupling this whole design exists to break; a record with no vector is
/// still findable by text and `kb embed` fills it in later.
///
/// **An embedding failure still dead-letters the submission**, after the
/// record is stored. That is the failure mode this shape introduces and the
/// reason [`Status`] reports three numbers: the endpoint keeps acknowledging
/// while nothing is fully ingested, and a queue that is empty because
/// everything succeeded looks exactly like one that is empty because
/// everything failed. Re-queueing a dead letter is safe — ids are stable, so
/// the second pass is the same write.
///
/// A submission that fails is never left pending: retrying it forever would
/// stall every submission behind it, which is the silent-stall failure a
/// local spool has and this design is chosen to avoid.
///
/// # Errors
///
/// [`IngestError`] if the queue itself cannot be read or written. A submission
/// that cannot be *processed* is dead-lettered rather than returned as an
/// error: one bad session is not a reason to stop draining.
pub fn drain(
    store: &impl crate::store::BlobStore,
    index: &crate::index::Index,
    queue: &Queue,
    embedder: Option<&crate::cli_embed::CliEmbedder>,
) -> Result<Drained, IngestError> {
    let mut done = Drained::default();
    for pending in queue.pending()? {
        match process(store, index, &pending, embedder) {
            Ok(()) => {
                queue.complete(&pending)?;
                done.ingested = done.ingested.saturating_add(1);
            }
            Err(reason) => {
                queue.dead_letter(&pending, &reason)?;
                done.dead_lettered = done.dead_lettered.saturating_add(1);
            }
        }
    }
    Ok(done)
}

/// Produce one record, returning the reason it could not be produced.
fn process(
    store: &impl crate::store::BlobStore,
    index: &crate::index::Index,
    pending: &Pending,
    embedder: Option<&crate::cli_embed::CliEmbedder>,
) -> Result<(), String> {
    let document = crate::parser::parse_document(&pending.submission.document)
        .map_err(|e| format!("the posted document does not parse: {e}"))?;
    // Idempotence has to be checked rather than assumed. A record header
    // carries `updated`, so writing the same document twice produces two
    // different record addresses and a supersession that supersedes nothing.
    // Comparing what is stored against what would be stored is what makes a
    // re-post cost nothing, which is the property reconciliation relies on:
    // it re-posts whatever the server might be missing, and being wrong about
    // that must be free.
    let unchanged = crate::write::read_raw(store, index, &pending.submission.id)
        .is_ok_and(|stored| stored == crate::generator::generate(&document));
    if !unchanged {
        // A captured conversation chunks on its turns; an authored note on
        // its sections. This is the same rule `migrate.rs`'s one-way import
        // applies to the superseded database's export, kept in one place so
        // the two paths cannot drift onto different tags.
        let kind = crate::migrate::kind_for_tags(&crate::storage::tag_names(&document));
        let provenance = pending
            .submission
            .provenance
            .clone()
            .map(crate::record::RawProvenance::validate)
            .transpose()
            .map_err(|e| format!("the submission's provenance is not usable: {e}"))?
            .unwrap_or_default();
        let options = crate::write::WriteOptions {
            kind,
            source: crate::record::SourceRef::new("node-id", &pending.submission.id)
                .map_err(|e| format!("the submission's id is not usable: {e}"))?,
            provenance,
        };
        crate::write::put_record(store, index, &pending.submission.id, &document, &options)
            .map_err(|e| format!("the record could not be stored: {e}"))?;
    }
    if let Some(embedder) = embedder {
        crate::write::embed_record(
            index,
            embedder,
            &pending.submission.id,
            crate::cli_embed::DOCUMENT_CHUNK_CHARS,
        )
        .map_err(|e| format!("stored, but could not embed: {e}"))?;
    }
    Ok(())
}

/// The queue as an HTTP surface: what the endpoint needs and nothing more.
///
/// It holds the index *path* rather than an open index, for the reason the
/// tool surface does: the index is re-derived and replaced by a rebuild, and a
/// long-lived handle to the file that was there when the server started would
/// serve a corpus that no longer exists.
#[derive(Debug, Clone)]
pub struct Ingest {
    queue: Queue,
    index: PathBuf,
}

impl Ingest {
    /// An ingest surface over `queue`, reporting the ids held in `index`.
    #[must_use]
    pub const fn new(queue: Queue, index: PathBuf) -> Self {
        Self { queue, index }
    }

    /// Accept a posted submission, returning the id acknowledged.
    ///
    /// # Errors
    ///
    /// [`IngestError::Malformed`] if the body is not a submission, and
    /// [`IngestError::Refused`] or [`IngestError::Io`] as [`Queue::enqueue`].
    pub fn accept(&self, body: &[u8]) -> Result<String, IngestError> {
        let submission: Submission =
            serde_json::from_slice(body).map_err(|e| IngestError::Malformed {
                path: PathBuf::from("<posted body>"),
                reason: e.to_string(),
            })?;
        self.queue.enqueue(&submission)?;
        Ok(submission.id)
    }

    /// The record ids the server already holds in `corpus`.
    ///
    /// What reconciliation compares against. Ids only: a client deciding what
    /// to post needs to know what is there, not what it says.
    ///
    /// # Errors
    ///
    /// [`IngestError::Io`] if the index cannot be opened or read.
    pub fn ids(&self, corpus: &str) -> Result<Vec<String>, IngestError> {
        let open =
            |e: crate::index::IndexError| io(&self.index)(std::io::Error::other(e.to_string()));
        let index = crate::index::Index::open(&self.index).map_err(open)?;
        index.records_in(corpus).map_err(open)
    }
}
