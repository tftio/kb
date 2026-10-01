//! Content-addressed blob store.
//!
//! The store is the archival record: bytes go in, a hash comes back, and the
//! bytes come out unchanged forever after. Everything else kb holds — the
//! `SQLite` index, its vectors, its generated text — is derived from what is
//! stored here and can be rebuilt from it (ST-002).
//!
//! git's object database is the implementation, chosen for packfile delta
//! compression, gc and integrity checking. That choice is confined to this
//! module: callers see [`BlobStore`], [`BlobHash`] and [`StoreError`], never a
//! `gix` type, so the backend could be replaced without touching them
//! (ENG-010).

use std::path::Path;
use thiserror::Error;

/// Failures the store can report. All are recoverable domain errors: the
/// caller is told which object or path was involved, never left to infer it
/// from a bare I/O message (ENG-004, RS-005).
#[derive(Debug, Error)]
pub enum StoreError {
    /// The store could not be opened, and could not be created either.
    #[error("could not open or create the blob store at {path}: {reason}")]
    Open {
        /// Filesystem path the store was expected at.
        path: String,
        /// Underlying failure, for display.
        reason: String,
    },

    /// Bytes could not be written to the object database.
    #[error("could not store {len} bytes: {reason}")]
    Write {
        /// Length of the rejected payload.
        len: usize,
        /// Underlying failure, for display.
        reason: String,
    },

    /// A hash was syntactically not an object id.
    #[error("not a valid object hash: {hash}")]
    MalformedHash {
        /// The text that failed to parse.
        hash: String,
    },

    /// No object with that hash is in the store.
    #[error("no object {hash} in the blob store")]
    ObjectNotFound {
        /// The hash that did not resolve.
        hash: String,
    },

    /// An object exists under that hash but is not stored content.
    #[error("object {hash} is not a blob")]
    NotABlob {
        /// The hash that resolved to the wrong object kind.
        hash: String,
    },

    /// A name was rejected before anything was written.
    #[error("not a usable record name: {name} ({reason})")]
    InvalidRefName {
        /// The rejected name.
        name: String,
        /// Which rule it broke.
        reason: String,
    },

    /// Nothing is bound to that name.
    #[error("no record named {name} in the blob store")]
    RefNotFound {
        /// The name that did not resolve.
        name: String,
    },

    /// A name exists but could not be read or written.
    #[error("could not resolve record name {name}: {reason}")]
    RefFailed {
        /// The name involved.
        name: String,
        /// Underlying failure, for display.
        reason: String,
    },
}

/// The stable, public name of a record.
///
/// Names are how records are addressed from outside the store — the
/// ground-truth question set, the `SessionEnd` hook and every index row key on
/// them — while hashes address content underneath. Binding a name to a new
/// hash is how a record changes without becoming a different record.
///
/// Accepted names are a deliberately conservative subset of what git permits:
/// slash-separated components of ASCII letters, digits, `-`, `_` and `.`, with
/// no empty component and no component that starts or ends with `.` or ends in
/// `.lock`. Being stricter than the backend is the point — a name valid here
/// stays valid if the backend is ever replaced.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RefName(String);

impl RefName {
    /// Validate `name` as a record name.
    ///
    /// # Errors
    ///
    /// [`StoreError::InvalidRefName`], naming the rule that was broken.
    pub fn new(name: &str) -> Result<Self, StoreError> {
        let refuse = |reason: &str| StoreError::InvalidRefName {
            name: name.to_owned(),
            reason: reason.to_owned(),
        };
        if name.is_empty() {
            return Err(refuse("empty"));
        }
        if name.len() > MAX_REF_NAME_LEN {
            return Err(refuse("longer than 255 bytes"));
        }
        for component in name.split('/') {
            if component.is_empty() {
                return Err(refuse("empty path component"));
            }
            if component.starts_with('.') || component.ends_with('.') {
                return Err(refuse("a component starts or ends with '.'"));
            }
            if component.to_ascii_lowercase().ends_with(".lock") {
                return Err(refuse("a component ends with '.lock'"));
            }
            if !component
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
            {
                return Err(refuse(
                    "only ASCII letters, digits, '-', '_' and '.' are allowed",
                ));
            }
        }
        Ok(Self(name.to_owned()))
    }

    /// The name as written.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RefName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Longest accepted record name, in bytes.
const MAX_REF_NAME_LEN: usize = 255;

/// Where record names live in the underlying repository. Private: the
/// namespace is an implementation detail of the git backend, and callers
/// address records by [`RefName`] alone.
const REF_NAMESPACE: &str = "refs/kb/";

/// The hash of stored bytes.
///
/// A hash addresses content and nothing else. It is deliberately not a
/// record's identity — see [`BlobStore`] — so this type carries no notion of
/// which record, corpus, or version the bytes belong to.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlobHash(String);

impl BlobHash {
    /// Take a hash back from stored text.
    ///
    /// The index stores hashes as text and hands them back when rebuilding,
    /// so they re-enter the store as strings and are narrowed here rather
    /// than trusted (ENG-006).
    ///
    /// # Errors
    ///
    /// [`StoreError::MalformedHash`] if `hex` is not hexadecimal.
    pub fn new(hex: &str) -> Result<Self, StoreError> {
        if hex.is_empty() || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(StoreError::MalformedHash {
                hash: hex.to_owned(),
            });
        }
        Ok(Self(hex.to_owned()))
    }

    /// The hash in hexadecimal.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A record's content hash is a store address, and the conversion cannot fail.
///
/// Both types accept exactly non-empty ASCII hexadecimal, so narrowing one to
/// the other through [`BlobHash::new`] produced an error branch nothing could
/// reach — and an unreachable error branch is worse than no branch: it asks
/// every caller to handle a case that cannot arise, and reads as though the
/// two types disagreed about what a hash is.
impl From<&crate::record::ContentHash> for BlobHash {
    fn from(hash: &crate::record::ContentHash) -> Self {
        Self(hash.as_str().to_owned())
    }
}

impl std::fmt::Display for BlobHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A content-addressed blob store.
///
/// `put` and `get` are the whole archival surface.
pub trait BlobStore {
    /// Store `bytes` and return the hash addressing them. Storing identical
    /// bytes twice yields the same hash and one stored object.
    ///
    /// # Errors
    ///
    /// [`StoreError::Write`] if the object database rejects the write.
    fn put(&self, bytes: &[u8]) -> Result<BlobHash, StoreError>;

    /// Retrieve the bytes `hash` addresses.
    ///
    /// # Errors
    ///
    /// [`StoreError::MalformedHash`] if `hash` is not an object id,
    /// [`StoreError::ObjectNotFound`] if nothing is stored under it,
    /// [`StoreError::NotABlob`] if it addresses something other than content.
    fn get(&self, hash: &BlobHash) -> Result<Vec<u8>, StoreError>;

    /// Bind `name` to `hash`, replacing any previous binding. The bytes the
    /// name pointed at before remain addressable by their own hash.
    ///
    /// # Errors
    ///
    /// [`StoreError::MalformedHash`] if `hash` is not an object id, or
    /// [`StoreError::RefFailed`] if the binding could not be written.
    fn set_ref(&self, name: &RefName, hash: &BlobHash) -> Result<(), StoreError>;

    /// Resolve `name` to the hash currently bound to it.
    ///
    /// # Errors
    ///
    /// [`StoreError::RefNotFound`] if nothing is bound to `name`, or
    /// [`StoreError::RefFailed`] if the binding could not be read.
    fn read_ref(&self, name: &RefName) -> Result<BlobHash, StoreError>;

    /// Every name beginning with `prefix`, in a stable order. An empty prefix
    /// is every name in the store.
    ///
    /// This is what makes the index rebuildable: rebuilding walks the store
    /// rather than trusting the index to know what it holds. By prefix rather
    /// than wholesale because corpora are separately rebuildable, and one
    /// corpus will eventually dwarf the others.
    ///
    /// # Errors
    ///
    /// [`StoreError::RefFailed`] if the names could not be enumerated.
    fn list_refs(&self, prefix: &str) -> Result<Vec<RefName>, StoreError>;

    /// Unbind `name`, returning whether anything was bound to it.
    ///
    /// **The bytes stay.** Deleting is unbinding a name, not erasing content:
    /// the blobs remain addressable by their own hashes, which is what makes a
    /// deletion recoverable by anyone who kept the address and what keeps an
    /// earlier record that referenced them intact. Reclaiming space is the
    /// separate, deliberate business of packing and garbage collection.
    ///
    /// # Errors
    ///
    /// [`StoreError::RefFailed`] if the binding could not be removed.
    fn delete_ref(&self, name: &RefName) -> Result<bool, StoreError>;
}

/// Where the store lives when nothing says otherwise.
///
/// `$HOME/.local/share/kb/store`, a bare repository beside the `SQLite` index
/// at `$HOME/.local/share/kb/kb.db` (see [`crate::storage::default_db_path`],
/// overridable by `KB_DB_PATH`).
///
/// The two are siblings but not peers. The store is authoritative and must be
/// backed up; the index beside it is derived and disposable, rebuildable from
/// the store whenever it is lost, stale, or deliberately thrown away (ST-002,
/// ST-004). Pointing `KB_DB_PATH` elsewhere moves only the index — it does not
/// move, and cannot corrupt, the record.
#[must_use]
pub fn default_store_path() -> std::path::PathBuf {
    dirs::home_dir().map_or_else(
        || std::path::PathBuf::from("store"),
        |home| home.join(".local/share/kb/store"),
    )
}

/// The store path the process is configured with: `KB_STORE_PATH` when set
/// and non-empty, else [`default_store_path`].
///
/// This is what every CLI verb that does not take a `--store` argument
/// opens. The verbs that do take one declare the same variable through clap,
/// so a deployment that points the server, the worker and the CLI at one
/// place with one setting is pointing all of them. Without this, `kb get`
/// would read under `$HOME` while `kb-mcp` read the variable, which is the
/// disagreement this exists to rule out.
#[must_use]
#[allow(
    clippy::disallowed_methods,
    reason = "the store location is deployment config read once at the binary edge, like the embedding endpoint (REPO_INVARIANTS.md ENG-013)"
)]
pub fn configured_store_path() -> std::path::PathBuf {
    store_path_from(std::env::var_os("KB_STORE_PATH"))
}

/// Pure variant of [`configured_store_path`]: the caller supplies what
/// `KB_STORE_PATH` held. Empty counts as unset.
#[must_use]
pub fn store_path_from(value: Option<std::ffi::OsString>) -> std::path::PathBuf {
    value
        .filter(|value| !value.is_empty())
        .map_or_else(default_store_path, std::path::PathBuf::from)
}

/// A [`BlobStore`] over a bare git repository.
#[derive(Debug)]
pub struct GitBlobStore {
    repo: gix::Repository,
}

impl GitBlobStore {
    /// Open an existing store, refusing to create one.
    ///
    /// Reading operations use this rather than [`Self::open_or_init`],
    /// because creating a store on demand turns a mistyped path into an
    /// empty corpus reported as a successful, empty result — the quiet wrong
    /// answer this codebase is meant not to give (ENG-004).
    ///
    /// # Errors
    ///
    /// [`StoreError::Open`] if there is no usable store at `path`.
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        gix::open(path)
            .map(|repo| Self { repo })
            .map_err(|e| StoreError::Open {
                path: path.display().to_string(),
                reason: e.to_string(),
            })
    }

    /// Open the store at `path`, creating a bare repository there if none
    /// exists. Idempotent: reopening finds everything previously stored.
    ///
    /// # Errors
    ///
    /// [`StoreError::Open`] if the path holds something that is not a usable
    /// repository, or if creating one fails.
    pub fn open_or_init(path: &Path) -> Result<Self, StoreError> {
        let opened = gix::open(path).or_else(|_| {
            // The store's parent is created on the way. Since T029 the first
            // `kb create` on a machine is what brings the store into
            // existence, and failing because `~/.local/share/kb` does not
            // exist yet would make a first write an installation step.
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| StoreError::Open {
                    path: parent.display().to_string(),
                    reason: e.to_string(),
                })?;
            }
            gix::init_bare(path).map_err(|e| StoreError::Open {
                path: path.display().to_string(),
                reason: e.to_string(),
            })
        })?;
        Ok(Self { repo: opened })
    }
}

impl BlobStore for GitBlobStore {
    fn put(&self, bytes: &[u8]) -> Result<BlobHash, StoreError> {
        let id = self.repo.write_blob(bytes).map_err(|e| StoreError::Write {
            len: bytes.len(),
            reason: e.to_string(),
        })?;
        Ok(BlobHash(id.detach().to_hex().to_string()))
    }

    fn get(&self, hash: &BlobHash) -> Result<Vec<u8>, StoreError> {
        let oid = object_id(hash)?;
        let object = self
            .repo
            .find_object(oid)
            .map_err(|_| StoreError::ObjectNotFound {
                hash: hash.0.clone(),
            })?;
        let blob = object.try_into_blob().map_err(|_| StoreError::NotABlob {
            hash: hash.0.clone(),
        })?;
        Ok(blob.data.clone())
    }

    fn set_ref(&self, name: &RefName, hash: &BlobHash) -> Result<(), StoreError> {
        let oid = object_id(hash)?;
        self.repo
            .reference(
                qualified(name),
                oid,
                gix::refs::transaction::PreviousValue::Any,
                format!("kb: bind {name}"),
            )
            .map(|_| ())
            .map_err(|e| StoreError::RefFailed {
                name: name.0.clone(),
                reason: e.to_string(),
            })
    }

    fn list_refs(&self, prefix: &str) -> Result<Vec<RefName>, StoreError> {
        self.names(prefix)
    }

    fn delete_ref(&self, name: &RefName) -> Result<bool, StoreError> {
        let Ok(reference) = self.repo.find_reference(qualified(name).as_str()) else {
            return Ok(false);
        };
        reference
            .delete()
            .map(|()| true)
            .map_err(|e| StoreError::RefFailed {
                name: name.0.clone(),
                reason: e.to_string(),
            })
    }

    fn read_ref(&self, name: &RefName) -> Result<BlobHash, StoreError> {
        let reference =
            self.repo
                .find_reference(qualified(name).as_str())
                .map_err(|e| match e {
                    gix::reference::find::existing::Error::NotFound { .. } => {
                        StoreError::RefNotFound {
                            name: name.0.clone(),
                        }
                    }
                    other @ gix::reference::find::existing::Error::Find(_) => {
                        StoreError::RefFailed {
                            name: name.0.clone(),
                            reason: other.to_string(),
                        }
                    }
                })?;
        reference
            .target()
            .try_id()
            .map(|id| BlobHash(id.to_hex().to_string()))
            .ok_or_else(|| StoreError::RefFailed {
                name: name.0.clone(),
                reason: "the name points at another name rather than at content".to_owned(),
            })
    }
}

impl GitBlobStore {
    /// Names under the store's namespace, sorted, with the namespace stripped.
    fn names(&self, prefix: &str) -> Result<Vec<RefName>, StoreError> {
        let platform = self.repo.references().map_err(|e| StoreError::RefFailed {
            name: format!("{prefix}*"),
            reason: e.to_string(),
        })?;
        let iter = platform
            .prefixed(format!("{REF_NAMESPACE}{prefix}").as_str())
            .map_err(|e| StoreError::RefFailed {
                name: format!("{prefix}*"),
                reason: e.to_string(),
            })?;
        let mut names = Vec::new();
        for reference in iter.flatten() {
            let full = reference.name().as_bstr().to_string();
            if let Some(bare) = full.strip_prefix(REF_NAMESPACE) {
                names.push(RefName::new(bare)?);
            }
        }
        names.sort();
        Ok(names)
    }
}

/// Place a record name in the store's namespace.
fn qualified(name: &RefName) -> String {
    format!("{REF_NAMESPACE}{name}")
}

/// Narrow a [`BlobHash`] to an object id, refusing anything that is not one.
fn object_id(hash: &BlobHash) -> Result<gix::ObjectId, StoreError> {
    gix::ObjectId::from_hex(hash.0.as_bytes()).map_err(|_| StoreError::MalformedHash {
        hash: hash.0.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        BlobHash, BlobStore, GitBlobStore, MAX_REF_NAME_LEN, RefName, StoreError,
        default_store_path,
    };

    /// The store and the index it derives are siblings under one data
    /// directory, so a single path is all an operator has to know, and the
    /// backup story can treat them differently without hunting for either.
    #[test]
    fn the_store_sits_beside_the_index_it_derives() {
        let store = default_store_path();
        let index = crate::storage::default_db_path();
        assert_eq!(
            store.parent(),
            index.parent(),
            "store {store:?} and index {index:?} should share a directory"
        );
        assert_eq!(
            store.file_name().and_then(std::ffi::OsStr::to_str),
            Some("store")
        );
    }

    fn rejection(name: &str) -> StoreError {
        RefName::new(name).expect_err("name should have been refused")
    }

    fn refused_because(name: &str) -> String {
        match rejection(name) {
            StoreError::InvalidRefName { reason, .. } => reason,
            other => panic!("expected InvalidRefName, got: {other}"),
        }
    }

    /// A hash that is not an object id is refused before the object database
    /// is consulted, so the caller is told the hash is wrong rather than that
    /// the object is missing.
    #[test]
    fn a_hash_that_is_not_an_object_id_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = GitBlobStore::open_or_init(dir.path()).expect("store");

        let err = store
            .get(&BlobHash("not-a-hash".to_owned()))
            .expect_err("a malformed hash must not resolve");

        assert!(
            matches!(&err, StoreError::MalformedHash { hash } if hash == "not-a-hash"),
            "error was: {err}"
        );
    }

    /// git's object database holds trees and commits too. Asking for content
    /// and getting structure is a distinguishable failure, not a decode error.
    #[test]
    fn an_object_that_is_not_content_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = GitBlobStore::open_or_init(dir.path()).expect("store");
        let tree = store
            .repo
            .write_object(gix::objs::Tree::empty())
            .expect("write an empty tree");
        let hash = BlobHash(tree.detach().to_hex().to_string());

        let err = store.get(&hash).expect_err("a tree is not stored content");

        assert!(
            matches!(&err, StoreError::NotABlob { hash: h } if *h == hash.as_str()),
            "error was: {err}"
        );
    }

    /// Failing to open the store names the path, because the usual cause is
    /// the path being wrong rather than the repository being broken.
    #[test]
    fn a_store_path_that_cannot_hold_a_repository_names_itself() {
        let dir = tempfile::tempdir().expect("tempdir");
        let occupied = dir.path().join("not-a-directory");
        std::fs::write(&occupied, b"a file sits here").expect("write");

        let err = GitBlobStore::open_or_init(&occupied).expect_err("a file cannot hold a store");

        assert!(
            matches!(&err, StoreError::Open { path, .. } if path == &occupied.display().to_string()),
            "error was: {err}"
        );
    }

    /// Hashes and names are quoted into error messages, logs and index rows,
    /// so their rendering is behavior rather than a formality.
    #[test]
    fn a_hash_and_a_name_render_as_themselves() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = GitBlobStore::open_or_init(dir.path()).expect("store");
        let hash = store.put(b"rendered.\n").expect("put");
        let name = RefName::new("nodes/n1").expect("name");

        assert_eq!(hash.to_string(), hash.as_str());
        assert_eq!(hash.to_string().len(), 40, "a full hex object id");
        assert_eq!(name.to_string(), "nodes/n1");
    }

    /// A name that points at another name is not a record. kb never writes
    /// one, so meeting one means the repository was edited by something else,
    /// and guessing which content was meant would be worse than refusing.
    #[test]
    fn a_name_pointing_at_another_name_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = GitBlobStore::open_or_init(dir.path()).expect("store");
        let hash = store.put(b"the real content.\n").expect("put");
        store
            .set_ref(&RefName::new("nodes/n1").expect("name"), &hash)
            .expect("bind");
        std::fs::write(
            dir.path().join("refs/kb/nodes/alias"),
            b"ref: refs/kb/nodes/n1\n",
        )
        .expect("write a symbolic ref");

        let err = store
            .read_ref(&RefName::new("nodes/alias").expect("name"))
            .expect_err("a symbolic name is not a record");

        assert!(
            matches!(&err, StoreError::RefFailed { name, reason }
                if name == "nodes/alias" && reason.contains("points at another name")),
            "error was: {err}"
        );
    }

    /// A ref file the backend cannot parse is distinguished from an absent
    /// one: absent is ordinary, unreadable means something corrupted the
    /// store, and conflating them would hide that.
    #[test]
    fn an_unreadable_name_is_not_reported_as_an_absent_one() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = GitBlobStore::open_or_init(dir.path()).expect("store");
        std::fs::create_dir_all(dir.path().join("refs/kb/nodes")).expect("mkdir");
        std::fs::write(
            dir.path().join("refs/kb/nodes/corrupt"),
            b"not an object id\n",
        )
        .expect("write a corrupt ref");

        let err = store
            .read_ref(&RefName::new("nodes/corrupt").expect("name"))
            .expect_err("a corrupt ref must not resolve");

        assert!(
            matches!(&err, StoreError::RefFailed { name, .. } if name == "nodes/corrupt"),
            "error was: {err}"
        );
    }

    #[test]
    fn an_ordinary_record_name_is_accepted() {
        let name = RefName::new("nodes/b70049ea-001f-49fd-ba5e-4344fbde9d92")
            .expect("a uuid-shaped node name is valid");
        assert_eq!(name.as_str(), "nodes/b70049ea-001f-49fd-ba5e-4344fbde9d92");
    }

    #[test]
    fn an_empty_name_is_refused() {
        assert_eq!(refused_because(""), "empty");
    }

    #[test]
    fn an_empty_path_component_is_refused() {
        assert_eq!(refused_because("nodes//n1"), "empty path component");
        assert_eq!(refused_because("/n1"), "empty path component");
        assert_eq!(refused_because("nodes/"), "empty path component");
    }

    /// `.` prefixes and suffixes are how git names its own bookkeeping and how
    /// a relative path sneaks in.
    #[test]
    fn a_component_bounded_by_a_dot_is_refused() {
        assert_eq!(
            refused_because("nodes/.hidden"),
            "a component starts or ends with '.'"
        );
        assert_eq!(refused_because(".."), "a component starts or ends with '.'");
    }

    /// git writes `<ref>.lock` while updating a reference; a record named that
    /// way would collide with the backend's own locking.
    #[test]
    fn a_lock_suffixed_component_is_refused() {
        assert_eq!(
            refused_because("nodes/n1.lock"),
            "a component ends with '.lock'"
        );
        // macOS, where this runs, has a case-insensitive filesystem by
        // default, so `n1.LOCK` would collide with git's lock file just as
        // `n1.lock` does.
        assert_eq!(
            refused_because("nodes/n1.LOCK"),
            "a component ends with '.lock'"
        );
    }

    #[test]
    fn punctuation_git_treats_as_syntax_is_refused() {
        for name in [
            "nodes/n 1",
            "nodes/n~1",
            "nodes/n^1",
            "nodes/n:1",
            "nodes/n?",
        ] {
            assert_eq!(
                refused_because(name),
                "only ASCII letters, digits, '-', '_' and '.' are allowed",
                "name was: {name}"
            );
        }
    }

    #[test]
    fn an_overlong_name_is_refused() {
        let long = "n".repeat(MAX_REF_NAME_LEN + 1);
        assert_eq!(refused_because(&long), "longer than 255 bytes");
        assert!(
            RefName::new(&"n".repeat(MAX_REF_NAME_LEN)).is_ok(),
            "the boundary itself is allowed"
        );
    }
}

#[cfg(test)]
mod hash_tests {
    use super::{BlobHash, StoreError};

    /// Hashes re-enter the store as text read back out of the index, so text
    /// that is not a hash is refused there rather than reaching the object
    /// database as a lookup that fails obscurely.
    #[test]
    fn text_that_is_not_a_hash_is_refused() {
        for candidate in ["", "not-hex", "beef!"] {
            let outcome = BlobHash::new(candidate);
            assert!(
                matches!(&outcome, Err(StoreError::MalformedHash { hash }) if hash == candidate),
                "{candidate:?} was accepted: {outcome:?}"
            );
        }
        assert!(BlobHash::new("2aae6c35c94fcfb415dbe95f408b9ce91ee846ed").is_ok());
    }
}

#[cfg(test)]
mod open_tests {
    use super::{BlobStore, GitBlobStore, StoreError};

    /// Reading from a store that is not there must fail rather than conjure
    /// an empty one: a mistyped path would otherwise read as a corpus that
    /// has lost everything.
    #[test]
    fn opening_a_store_that_does_not_exist_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let absent = dir.path().join("not-created");

        let outcome = GitBlobStore::open(&absent);

        assert!(
            matches!(&outcome, Err(StoreError::Open { path, .. }) if path.contains("not-created")),
            "outcome was: {outcome:?}"
        );
    }

    /// The creating constructor is still what ingest uses, and what it
    /// creates is openable afterwards.
    #[test]
    fn a_created_store_can_be_reopened_without_creating() {
        let dir = tempfile::tempdir().expect("tempdir");
        let hash = {
            let created = GitBlobStore::open_or_init(dir.path()).expect("init");
            created.put(b"content.\n").expect("put")
        };

        let reopened = GitBlobStore::open(dir.path()).expect("open");

        assert_eq!(reopened.get(&hash).expect("get"), b"content.\n".to_vec());
    }
}
