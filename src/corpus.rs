//! What a corpus has to provide, and nothing more.
//!
//! The per-corpus surface is two functions: resolve an identifier to bytes,
//! and enumerate the identifiers there are. Chunking, embedding, generation
//! and indexing are shared and corpus-agnostic, because they operate on the
//! normalized stream rather than on whatever the source happened to be.
//!
//! That split is the thin answer to "do we need specialized ingesters".
//! Format knowledge is irreducible — a Maildir is not a git object database,
//! and a Slack export is neither — but it belongs in resolution rather than
//! spread through everything downstream.
//!
//! Enumeration is paged and resumable from an opaque cursor even though every
//! corpus this plan implements is local and could be walked whole. Slack and
//! Bluesky resolvers reach over rate-limited APIs where `enumerate` cannot be
//! called casually, and retrofitting paging into an interface with three
//! implementations is worse than designing it with one.

use crate::record::SourceRef;
use crate::store::{BlobHash, BlobStore, GitBlobStore, RefName, StoreError};
use thiserror::Error;

/// Whether a corpus keeps its bytes in the store or only points at them.
///
/// This decides whether a reference that no longer resolves is a bug or an
/// expected consequence of living outside the store, so it is declared per
/// corpus rather than inferred per failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    /// The bytes are copied into the content-addressed store, which then owns
    /// them. A dead reference is corruption.
    CopyIntoStore,
    /// The bytes stay where they are and the index holds a reference plus a
    /// content hash. A dead reference means the source moved or was deleted,
    /// which is expected and detectable rather than corrupt.
    ReferenceOnly,
}

impl Storage {
    /// The stored form, for the index column that records it.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CopyIntoStore => "copy",
            Self::ReferenceOnly => "reference",
        }
    }
}

/// Failures a corpus can report.
#[derive(Debug, Error)]
pub enum CorpusError {
    /// The corpus does not hold that identifier.
    #[error("{corpus}: no record {id}")]
    NotFound {
        /// Which corpus was asked.
        corpus: String,
        /// The identifier that did not resolve.
        id: String,
    },

    /// The corpus catalogued this identifier once and no longer holds the
    /// bytes.
    ///
    /// Distinct from [`CorpusError::NotFound`] on purpose. A reference-only
    /// corpus does not own what it points at, so a citation going dead is an
    /// expected outcome rather than corruption — but it is also not the same
    /// as never having held the record, and reporting it as such would either
    /// invent a deletion or hide one.
    #[error("{corpus}: {id} no longer resolves (last seen in {location})")]
    Vanished {
        /// Which corpus held it.
        corpus: String,
        /// The citation that has gone dead.
        id: String,
        /// Where it was when the catalogue last saw it.
        location: String,
    },

    /// The corpus could not be read at all.
    #[error("{corpus} is unusable: {reason}")]
    Unusable {
        /// Which corpus failed.
        corpus: String,
        /// What went wrong.
        reason: String,
    },
}

/// One page of an enumeration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page {
    /// The identifiers in this page.
    pub ids: Vec<SourceRef>,
    /// An opaque cursor resuming after this page, or `None` at the end.
    ///
    /// Opaque because what resumes a walk differs per corpus — an offset here,
    /// a continuation token from an API there — and a caller that inspected it
    /// would couple itself to one implementation.
    pub next: Option<String>,
}

/// A source of records.
pub trait Corpus {
    /// The corpus identifier, which must be one the ground-truth question set
    /// already names.
    fn id(&self) -> &'static str;

    /// Whether this corpus copies into the store or only references it.
    fn storage(&self) -> Storage;

    /// The bytes for one identifier, as they arrived.
    ///
    /// # Errors
    ///
    /// [`CorpusError::NotFound`] if the corpus does not hold it, or
    /// [`CorpusError::Unusable`] if the corpus cannot be read.
    fn resolve(&self, source: &SourceRef) -> Result<Vec<u8>, CorpusError>;

    /// Up to `limit` identifiers, resuming after `cursor`.
    ///
    /// # Errors
    ///
    /// [`CorpusError::Unusable`] if the corpus cannot be enumerated.
    fn enumerate(&self, cursor: Option<&str>, limit: usize) -> Result<Page, CorpusError>;
}

/// The kb corpus: records in the content-addressed store.
#[derive(Debug)]
pub struct KbCorpus {
    store: GitBlobStore,
}

impl KbCorpus {
    /// Wrap a store as the kb corpus.
    #[must_use]
    pub const fn new(store: GitBlobStore) -> Self {
        Self { store }
    }

    /// Map a store failure onto a corpus one, keeping the identifier that was
    /// asked for rather than reporting a bare hash.
    fn failure(&self, id: &str, error: &StoreError) -> CorpusError {
        match error {
            StoreError::RefNotFound { .. } | StoreError::ObjectNotFound { .. } => {
                CorpusError::NotFound {
                    corpus: self.id().to_owned(),
                    id: id.to_owned(),
                }
            }
            other => CorpusError::Unusable {
                corpus: self.id().to_owned(),
                reason: other.to_string(),
            },
        }
    }
}

impl Corpus for KbCorpus {
    fn id(&self) -> &'static str {
        "kb"
    }

    fn storage(&self) -> Storage {
        Storage::CopyIntoStore
    }

    fn resolve(&self, source: &SourceRef) -> Result<Vec<u8>, CorpusError> {
        let id = source.value();
        let name = RefName::new(&format!("kb/{id}")).map_err(|e| self.failure(id, &e))?;
        let record_hash = self
            .store
            .read_ref(&name)
            .map_err(|e| self.failure(id, &e))?;
        let blob = self
            .store
            .get(&record_hash)
            .map_err(|e| self.failure(id, &e))?;
        let header =
            crate::record::RecordHeader::parse(&blob).map_err(|e| CorpusError::Unusable {
                corpus: self.id().to_owned(),
                reason: format!("record {id}: {e}"),
            })?;
        // The archival bytes, not the record header: a citation is checkable
        // against what arrived, and the header is bookkeeping about it.
        let raw = BlobHash::new(header.raw.as_str()).map_err(|e| self.failure(id, &e))?;
        self.store.get(&raw).map_err(|e| self.failure(id, &e))
    }

    fn enumerate(&self, cursor: Option<&str>, limit: usize) -> Result<Page, CorpusError> {
        let names = self
            .store
            .list_refs("kb/")
            .map_err(|e| CorpusError::Unusable {
                corpus: self.id().to_owned(),
                reason: e.to_string(),
            })?;
        // The cursor is the last id returned rather than an offset: an offset
        // into a list that grew between calls would skip whatever was inserted
        // before it, and records are named, not positioned.
        let start = cursor.map_or(0, |after| {
            names
                .iter()
                .position(|n| n.as_str().strip_prefix("kb/") == Some(after))
                .map_or(0, |i| i.saturating_add(1))
        });
        let mut ids = Vec::new();
        for name in names.iter().skip(start).take(limit) {
            let bare = name.as_str().strip_prefix("kb/").unwrap_or(name.as_str());
            ids.push(
                SourceRef::new("node-id", bare).map_err(|e| CorpusError::Unusable {
                    corpus: self.id().to_owned(),
                    reason: e.to_string(),
                })?,
            );
        }
        let consumed = start.saturating_add(ids.len());
        let next = (consumed < names.len())
            .then(|| ids.last().map(|s| s.value().to_owned()))
            .flatten();
        Ok(Page { ids, next })
    }
}

/// Which corpora exist and how each stores its bytes.
///
/// The identifiers are the ground-truth question set's, not this registry's.
/// T013 fixed them before this task existed, deliberately: the question set
/// and its committed run records are the control for every retrieval claim in
/// the plan, and a registry that invented an identifier would break the join
/// between an index row's corpus and an `expect` entry's — a break that would
/// read as a retrieval regression rather than as a naming mismatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Registry {
    entries: Vec<(String, Storage)>,
}

impl Default for Registry {
    fn default() -> Self {
        Self {
            entries: vec![
                ("kb".to_owned(), Storage::CopyIntoStore),
                ("mail".to_owned(), Storage::ReferenceOnly),
            ],
        }
    }
}

impl Registry {
    /// Every corpus identifier the registry defines.
    #[must_use]
    pub fn ids(&self) -> Vec<&str> {
        self.entries.iter().map(|(id, _)| id.as_str()).collect()
    }

    /// How a named corpus stores its bytes, or `None` if it is not registered.
    #[must_use]
    pub fn storage_of(&self, id: &str) -> Option<Storage> {
        self.entries
            .iter()
            .find(|(name, _)| name == id)
            .map(|(_, storage)| *storage)
    }
}
