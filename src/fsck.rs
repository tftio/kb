//! Reconciliation between what the store holds and what the index says.
//!
//! The index is derived and disposable (ST-002), which makes it correct by
//! construction only for as long as something re-derives it. A record written
//! to the store while the index was closed, a rebuild interrupted before it
//! committed, a normalizer whose version moved — each leaves the index saying
//! something the store no longer supports, and none of them announces itself.
//!
//! **Driven by hashes, never by events.** A record's blob address is the whole
//! of its identity in content terms, so comparing the address the store holds
//! against the address the index recorded answers "is this row still true?"
//! without any history of what happened. That single property is what makes
//! reconciliation idempotent: a missed signal costs latency rather than
//! correctness, a duplicated signal costs a comparison, and the delivery
//! mechanism — scheduled task, file watcher, or somebody typing the command —
//! becomes a free choice rather than part of the correctness argument.
//!
//! **What it will not repair, it reports.** Records the index holds and the
//! store no longer does are removed, because a derived row whose source is
//! gone is by definition stale. Nodes the legacy database holds and the store
//! has never seen are only reported: importing them is an archival write, not
//! a derivation, and silently performing one under a command called `fsck`
//! would put content into the permanent store on a schedule nobody reviewed.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use thiserror::Error;

use crate::index::{Index, IndexError, RebuildReport, reindex_records, stored_record_hashes};
use crate::record::RecordHeader;
use crate::store::BlobStore;

/// Why a survey or a repair could not be completed.
#[derive(Debug, Error)]
pub enum FsckError {
    /// The index could not be read or written.
    #[error("{0}")]
    Index(#[from] IndexError),
    /// The store could not be enumerated or read.
    #[error("reading {record} from the store: {source}")]
    Store {
        /// Which ref was being read.
        record: String,
        /// What the store reported.
        source: crate::store::StoreError,
    },
    /// A record blob will not parse as a record header.
    #[error("parsing {record}: {source}")]
    Record {
        /// Which record could not be parsed.
        record: String,
        /// What the parse reported.
        source: crate::record::RecordError,
    },
    /// The legacy database could not be read.
    #[error("reading the legacy database: {0}")]
    Legacy(#[from] rusqlite::Error),
}

/// What a survey found.
///
/// Every field is a set of record ids rather than a count, because a count
/// tells an operator that something is wrong and nothing about what. The three
/// repairable kinds are kept apart rather than merged into "differs": they have
/// different causes, and a survey that reported forty stale records where the
/// truth is forty missing ones would send someone looking in the wrong place.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Survey {
    /// How many stored records were compared.
    pub checked: usize,
    /// In the store, absent from the index.
    pub missing: Vec<String>,
    /// In both, addressing different record blobs.
    pub stale: Vec<String>,
    /// In the index, no longer in the store.
    pub orphaned: Vec<String>,
    /// In the legacy database, never archived to the store. Reported, never
    /// repaired.
    pub unarchived: Vec<String>,
    /// Whose stored normalized stream carries different *content* from what
    /// the current normalizer produces. Only populated by a deep survey, and
    /// reported rather than repaired: correcting it means writing a new record
    /// blob into the archival store, which is not a derivation.
    pub renormalized: Vec<String>,
    /// How many records carry an older normalizer version whose payload is
    /// nonetheless identical. Counted rather than named: the version is a
    /// single global counter, so a bump made for one artifact kind marks every
    /// record of every kind behind, and naming thousands of records nothing is
    /// wrong with is how a report teaches its reader to skip it.
    pub behind: usize,
    /// Whether the survey re-normalized rather than only comparing addresses.
    pub deep: bool,
    /// How long the survey took.
    pub elapsed: Duration,
}

impl Survey {
    /// Whether the index agrees with the store in every respect this can
    /// check.
    #[must_use]
    pub const fn is_clean(&self) -> bool {
        self.missing.is_empty() && self.stale.is_empty() && self.orphaned.is_empty()
    }

    /// The records a repair would re-derive.
    #[must_use]
    pub fn repairable(&self) -> BTreeSet<String> {
        self.missing
            .iter()
            .chain(self.stale.iter())
            .cloned()
            .collect()
    }
}

/// What a repair did.
#[derive(Debug, Clone, Default)]
pub struct Repair {
    /// Records re-derived from the store.
    pub rederived: usize,
    /// Index rows dropped because the store no longer holds their record.
    pub dropped: usize,
    /// Passages written by the re-derivation.
    pub passages: usize,
    /// Vectors dropped because no passage contains their span any more.
    pub stale_vectors_dropped: usize,
    /// How long the repair took.
    pub elapsed: Duration,
}

/// Compare every stored record against what the index recorded.
///
/// Reads only. `legacy`, when given, is the superseded node database: nodes it
/// holds that the store has never seen are reported as `unarchived`. That is
/// not a hypothetical — the capture hook writes there and to nothing else, so
/// the set grows by roughly one node per session until something imports them.
///
/// # Errors
///
/// [`FsckError`] if the store cannot be enumerated or read, a record blob will
/// not parse, or either database fails.
pub fn survey(
    store: &impl BlobStore,
    index: &Index,
    legacy: Option<&rusqlite::Connection>,
) -> Result<Survey, FsckError> {
    survey_with(store, index, legacy, false)
}

/// [`survey`], optionally re-normalizing every stored record.
///
/// A shallow survey compares addresses, which answers "does the index still
/// describe what the store holds?" and nothing more. It cannot see a record
/// whose stored stream was produced by a normalizer the code has since
/// changed: the record blob is untouched, so every address still agrees, while
/// the passages address a normalization the parser would no longer produce.
///
/// A deep survey answers that by re-normalizing each record's stored raw bytes
/// and comparing the result to the stream the record claims. It is the honest
/// form of the question, and strictly better than comparing normalizer version
/// numbers: the version is a single global counter, so a bump made for one
/// artifact kind marks every other kind stale as well, and a check that
/// reports thousands of records nobody needs to act on is a check whose
/// reader learns to ignore it.
///
/// Reference-only corpora are skipped: their raw bytes were never copied into
/// the store, so there is nothing here to re-normalize.
///
/// # Errors
///
/// [`FsckError`] if the store cannot be enumerated or read, a record blob will
/// not parse, or either database fails.
pub fn survey_with(
    store: &impl BlobStore,
    index: &Index,
    legacy: Option<&rusqlite::Connection>,
    deep: bool,
) -> Result<Survey, FsckError> {
    let started = Instant::now();
    let recorded: BTreeMap<String, String> = stored_record_hashes(index)?;
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut found = Survey {
        deep,
        ..Survey::default()
    };

    let names = store.list_refs("").map_err(|e| FsckError::Store {
        record: "<enumerating>".to_owned(),
        source: e,
    })?;
    for name in &names {
        let label = name.as_str().to_owned();
        let hash = store.read_ref(name).map_err(|e| FsckError::Store {
            record: label.clone(),
            source: e,
        })?;
        let blob = store.get(&hash).map_err(|e| FsckError::Store {
            record: label.clone(),
            source: e,
        })?;
        let header = RecordHeader::parse(&blob).map_err(|e| FsckError::Record {
            record: label.clone(),
            source: e,
        })?;
        let id = header.id.as_str().to_owned();
        found.checked += 1;
        match recorded.get(&id) {
            None => found.missing.push(id.clone()),
            Some(recorded_hash) if recorded_hash != hash.as_str() => found.stale.push(id.clone()),
            Some(_) => {}
        }
        if deep {
            match renormalization(store, &header)? {
                Renormalization::Identical => {}
                Renormalization::VersionOnly => found.behind = found.behind.saturating_add(1),
                Renormalization::ContentDiffers => found.renormalized.push(id.clone()),
            }
        }
        seen.insert(id);
    }
    found.orphaned = recorded
        .keys()
        .filter(|id| !seen.contains(*id))
        .cloned()
        .collect();
    if let Some(conn) = legacy {
        found.unarchived = unarchived_nodes(conn, &seen)?;
    }
    found.elapsed = started.elapsed();
    Ok(found)
}

/// How a record's stored stream compares with what the current normalizer
/// produces from its stored raw bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Renormalization {
    /// Byte-for-byte the same, version header included.
    Identical,
    /// Same content under a different normalizer version. The record is
    /// behind; nothing derived from it is wrong.
    VersionOnly,
    /// Different content. The passages address text the normalizer would no
    /// longer produce.
    ContentDiffers,
}

/// Compare a record's stored stream with what the current normalizer produces.
///
/// Only ever called for a record the survey reached through a store ref, which
/// a reference-only corpus does not have: mail records are written into the
/// index directly and put nothing in the store, so there is nothing here to
/// re-normalize and nothing to guard against.
fn renormalization(
    store: &impl BlobStore,
    header: &RecordHeader,
) -> Result<Renormalization, FsckError> {
    let raw_hash = crate::store::BlobHash::from(&header.raw);
    let raw = store.get(&raw_hash).map_err(|e| FsckError::Store {
        record: header.id.as_str().to_owned(),
        source: e,
    })?;
    let produced = crate::record::normalize(header.kind, &raw).map_err(|e| FsckError::Record {
        record: header.id.as_str().to_owned(),
        source: e,
    })?;
    let stream_hash = crate::store::BlobHash::from(&header.stream);
    let stored = store.get(&stream_hash).map_err(|e| FsckError::Store {
        record: header.id.as_str().to_owned(),
        source: e,
    })?;
    // Compared as bytes rather than as addresses. Asking the store for the
    // address of the freshly normalized stream would mean **writing** it, and
    // a survey that puts objects into an append-only archive is not a survey —
    // the store's gc packs and never prunes, so every deep run would leave
    // permanent garbage behind. Two blobs with equal bytes have equal
    // addresses, so nothing is given up by comparing the bytes.
    if stored == produced.as_bytes() {
        return Ok(Renormalization::Identical);
    }
    let stored_stream = crate::record::NormalizedStream::from_bytes(stored);
    if stored_stream.payload() == produced.payload() {
        return Ok(Renormalization::VersionOnly);
    }
    Ok(Renormalization::ContentDiffers)
}

/// Nodes the legacy database holds that the store has never seen.
fn unarchived_nodes(
    conn: &rusqlite::Connection,
    archived: &BTreeSet<String>,
) -> Result<Vec<String>, FsckError> {
    let mut statement = conn.prepare("SELECT id FROM nodes")?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
    let mut absent = Vec::new();
    for row in rows {
        let id = row?;
        if !archived.contains(&id) {
            absent.push(id);
        }
    }
    absent.sort();
    Ok(absent)
}

/// Re-derive what a survey found repairable and drop what the store no longer
/// holds.
///
/// Takes the survey rather than re-taking it, so what is repaired is exactly
/// what was reported. A repair that surveyed again could act on drift the
/// operator never saw.
///
/// # Errors
///
/// [`FsckError`] if the store cannot be read or the index cannot be written.
pub fn repair(store: &impl BlobStore, index: &Index, found: &Survey) -> Result<Repair, FsckError> {
    let started = Instant::now();
    let mut done = Repair::default();
    let wanted = found.repairable();
    if !wanted.is_empty() {
        let report: RebuildReport = reindex_records(store, index, &wanted)?;
        done.rederived = report.records;
        done.passages = report.passages;
        done.stale_vectors_dropped = report.stale_vectors_dropped;
    }
    for id in &found.orphaned {
        index.forget(id)?;
        done.dropped += 1;
    }
    done.elapsed = started.elapsed();
    Ok(done)
}
