//! One-way migration out of the superseded database.
//!
//! The operator's directive of 2026-08-14 governs this module: the existing
//! kb is not a going concern. `kb.db` is read and never written, becomes a
//! frozen archive once its export is verified, and no path back to the old
//! schema is built or maintained.
//!
//! The transfer is the easy part; the verification is the task. The export
//! artifact holds exactly the text `kb get` renders, so a byte comparison
//! against a pre-export capture is a comparison against what the operator
//! would actually have read.

use crate::record::{
    ArtifactKind, ContentHash, CorpusId, RecordHeader, RecordId, SourceRef, Tag, normalize,
};
use crate::store::{BlobStore, StoreError};
use crate::{org_meta, record, storage};
use rusqlite::{Connection, OpenFlags};
use std::io::{BufRead, Write};
use std::path::Path;
use thiserror::Error;

/// Failures in exporting or importing.
#[derive(Debug, Error)]
pub enum MigrateError {
    /// The superseded database could not be read.
    #[error("reading {path}: {reason}")]
    Source {
        /// Which database.
        path: String,
        /// What went wrong.
        reason: String,
    },

    /// The export artifact could not be written or read.
    #[error("export artifact {path}: {reason}")]
    Artifact {
        /// Which file.
        path: String,
        /// What went wrong.
        reason: String,
    },

    /// A node could not be turned into a record.
    #[error("node {id}: {reason}")]
    Node {
        /// Which node.
        id: String,
        /// What went wrong.
        reason: String,
    },

    /// The store rejected a write.
    #[error("store: {0}")]
    Store(#[from] StoreError),
}

/// What an export produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExportReport {
    /// Nodes written to the artifact.
    pub nodes: usize,
    /// Bytes of node text written.
    pub bytes: usize,
}

/// What an import produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ImportReport {
    /// Records written to the store.
    pub records: usize,
    /// Records that were transcripts rather than authored notes.
    pub transcripts: usize,
}

/// The tag the transcript importer puts on every captured conversation. It is
/// what distinguishes a transcript from an authored note, and therefore
/// whether passages fall on turns or on sections.
///
/// `pub(crate)` because [`crate::ingest`]'s queue worker applies this exact
/// rule to a posted submission's tags, and must consume this constant rather
/// than duplicate the string literal.
pub(crate) const TRANSCRIPT_TAG: &str = "conversation";

/// The kind a write should assert for a document carrying `tags`, by the
/// same rule everywhere a document turns into a record: a `conversation` tag
/// means a captured transcript, chunked on its turns; anything else is an
/// authored note, chunked on its sections.
///
/// [`crate::ingest`]'s queue worker and `kb create` (T017) both call this
/// rather than testing [`TRANSCRIPT_TAG`] themselves, so the rule cannot
/// drift into two definitions of what a transcript is.
pub(crate) fn kind_for_tags<S: AsRef<str>>(tags: &[S]) -> ArtifactKind {
    if tags.iter().any(|t| t.as_ref() == TRANSCRIPT_TAG) {
        ArtifactKind::SessionTranscript
    } else {
        ArtifactKind::Note
    }
}

/// Export every node from the superseded database into a line-per-node
/// artifact.
///
/// The database is opened read-only, so the export cannot write to it even by
/// accident — it is 630MB and the only copy of the corpus until the store
/// holds it.
///
/// # Errors
///
/// [`MigrateError::Source`] if the database cannot be read,
/// [`MigrateError::Artifact`] if the artifact cannot be written, and
/// [`MigrateError::Node`] if a node cannot be rendered.
pub fn export_from_db(db: &Path, artifact: &Path) -> Result<ExportReport, MigrateError> {
    let conn = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(|e| {
        MigrateError::Source {
            path: db.display().to_string(),
            reason: e.to_string(),
        }
    })?;
    let file = std::fs::File::create(artifact).map_err(|e| MigrateError::Artifact {
        path: artifact.display().to_string(),
        reason: e.to_string(),
    })?;
    let mut out = std::io::BufWriter::new(file);
    let mut report = ExportReport::default();

    let rows = storage::list_all_nodes(&conn, usize::MAX, 0).map_err(|e| MigrateError::Source {
        path: db.display().to_string(),
        reason: e.to_string(),
    })?;
    for (node_id, _title) in rows {
        let id = node_id.0;
        let full = storage::get_node_full(&conn, &id)
            .map_err(|e| MigrateError::Node {
                id: id.clone(),
                reason: e.to_string(),
            })?
            .ok_or_else(|| MigrateError::Node {
                id: id.clone(),
                reason: "listed but not readable".to_owned(),
            })?;
        // Exactly what `kb get` prints, so verification compares against what
        // the operator would have read rather than against an encoding
        // invented here.
        let org =
            org_meta::render_with_metadata(&id, &full.created_at, &full.updated_at, &full.document);
        let line = serde_json::json!({
            "id": id,
            "created_at": full.created_at,
            "updated_at": full.updated_at,
            "tags": full.tags.iter().map(|t| t.0.clone()).collect::<Vec<_>>(),
            "org": org,
        });
        report.bytes = report.bytes.saturating_add(org.len());
        report.nodes = report.nodes.saturating_add(1);
        writeln!(out, "{line}").map_err(|e| MigrateError::Artifact {
            path: artifact.display().to_string(),
            reason: e.to_string(),
        })?;
    }
    out.flush().map_err(|e| MigrateError::Artifact {
        path: artifact.display().to_string(),
        reason: e.to_string(),
    })?;
    Ok(report)
}

/// Import an export artifact into the content-addressed store.
///
/// Each node becomes three blobs — the org text as it was rendered, the
/// normalized stream its passages will address, and the self-describing
/// record header — with the node id as the record's name. Idempotent: the
/// same artifact imported twice writes the same objects and rebinds the same
/// names.
///
/// # Errors
///
/// [`MigrateError::Artifact`] if the artifact cannot be read,
/// [`MigrateError::Node`] if a line is not a usable node, and
/// [`MigrateError::Store`] if a write fails.
pub fn import_into_store(
    artifact: &Path,
    store: &impl BlobStore,
) -> Result<ImportReport, MigrateError> {
    let file = std::fs::File::open(artifact).map_err(|e| MigrateError::Artifact {
        path: artifact.display().to_string(),
        reason: e.to_string(),
    })?;
    let mut report = ImportReport::default();
    for line in std::io::BufReader::new(file).lines() {
        let line = line.map_err(|e| MigrateError::Artifact {
            path: artifact.display().to_string(),
            reason: e.to_string(),
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let kind = import_one(&line, store, &mut report)?;
        if kind == ArtifactKind::SessionTranscript {
            report.transcripts = report.transcripts.saturating_add(1);
        }
    }
    Ok(report)
}

/// Import one exported node, returning what kind of record it became.
fn import_one(
    line: &str,
    store: &impl BlobStore,
    report: &mut ImportReport,
) -> Result<ArtifactKind, MigrateError> {
    let value: serde_json::Value =
        serde_json::from_str(line).map_err(|e| MigrateError::Artifact {
            path: "<line>".to_owned(),
            reason: e.to_string(),
        })?;
    let field = |key: &str| -> Result<String, MigrateError> {
        value
            .get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| MigrateError::Artifact {
                path: "<line>".to_owned(),
                reason: format!("missing {key}"),
            })
    };
    let id = field("id")?;
    let fail = |reason: String| MigrateError::Node {
        id: id.clone(),
        reason,
    };
    let org = field("org")?;
    let tags: Vec<String> = value
        .get("tags")
        .and_then(serde_json::Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();

    // A captured conversation chunks on its turns; an authored note on its
    // sections. Flattening both to one kind would degrade retrieval in a way
    // no count would show.
    let kind = kind_for_tags(&tags);

    let raw = org.as_bytes();
    let stream = normalize(kind, raw).map_err(|e| fail(e.to_string()))?;
    let raw_hash = store.put(raw)?;
    let stream_hash = store.put(stream.as_bytes())?;
    let header = RecordHeader {
        id: RecordId::new(&id).map_err(|e| fail(e.to_string()))?,
        corpus: CorpusId::new("kb").map_err(|e| fail(e.to_string()))?,
        kind,
        created: parse_time(&field("created_at")?).map_err(&fail)?,
        updated: parse_time(&field("updated_at")?).map_err(&fail)?,
        source: SourceRef::new("node-id", &id).map_err(|e| fail(e.to_string()))?,
        tags: tags
            .iter()
            .map(|t| Tag::new(t))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| fail(e.to_string()))?,
        raw: ContentHash::new(raw_hash.as_str()).map_err(|e| fail(e.to_string()))?,
        stream: ContentHash::new(stream_hash.as_str()).map_err(|e| fail(e.to_string()))?,
        normalizer: record::NORMALIZER_VERSION,
        provenance: record::Provenance::default(),
    };
    let record_hash = store.put(&header.serialize())?;
    store.set_ref(
        &crate::store::RefName::new(&format!("kb/{id}")).map_err(|e| fail(e.to_string()))?,
        &record_hash,
    )?;
    report.records = report.records.saturating_add(1);
    Ok(kind)
}

/// Read a stored timestamp. The superseded schema already holds RFC 3339.
fn parse_time(value: &str) -> Result<chrono::DateTime<chrono::Utc>, String> {
    chrono::DateTime::parse_from_rfc3339(value)
        .map(|t| t.with_timezone(&chrono::Utc))
        .map_err(|e| format!("timestamp {value:?}: {e}"))
}
