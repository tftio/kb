//! Writing a record into the store, which is where every write goes.
//!
//! Retrieval reads the derived index, and the index is derived from the store
//! (ST-001, ST-002). A write that lands anywhere else is a write nobody can
//! find — which is exactly what `kb create` and the agent surface's `put` did
//! until T029 moved retrieval off `kb.db`: they wrote a row into a database no
//! signal consults, returned an identifier, and left the caller to discover by
//! searching that what they had just written was not there.
//!
//! One function, so the CLI and the agent surface cannot drift into two
//! notions of what writing means, and so T022's ingest worker has a path to
//! call rather than a procedure to reimplement.
//!
//! **The index write is not the commit.** The store is written first and the
//! index is re-derived from it afterwards; if the process dies between the
//! two, the record is stored and the index is stale, which is the state
//! `kb fsck` is built to find and repair. The reverse order would leave the
//! index describing a record the store does not hold, which nothing can
//! repair because the bytes were never written.

use std::collections::BTreeSet;

use tftio_org::ast::Document;
use thiserror::Error;

use crate::index::Index;
use crate::record::{
    self, ArtifactKind, ContentHash, CorpusId, Provenance, RecordHeader, RecordId, SourceRef,
};
use crate::store::{BlobHash, BlobStore, RefName};

/// What a write asserts about the record it produces, beyond its content.
///
/// `put_record` used to hardcode `kind: note`, `corpus: kb` and
/// `source: node-id:<id>` for every write, which is the defect this type
/// fixes: the queue worker had no way to say a submission was a captured
/// transcript, so every record it wrote landed as a note and was chunked on
/// sections rather than turns. A caller that knows more than "an authored
/// note" — the queue worker importing a transcript, a future importer
/// carrying session provenance — builds one of these instead of accepting
/// the default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteOptions {
    /// What kind of artifact this write produces.
    pub kind: ArtifactKind,
    /// The record's identity in the system it came from.
    pub source: SourceRef,
    /// Where the work that produced it was happening, if known.
    pub provenance: Provenance,
}

impl WriteOptions {
    /// The ordinary case: an authored note, sourced by its own record id,
    /// asserting no provenance. What every write used before this type
    /// existed, and what `kb create` and `kb update` still pass unless a
    /// caller supplies more.
    ///
    /// # Errors
    ///
    /// [`record::RecordError`] if `id` cannot be used as a [`SourceRef`]
    /// value (it cannot: every character `RecordId` accepts, `SourceRef`
    /// accepts too, so this only reports a defect in the caller's id).
    pub fn note(id: &str) -> Result<Self, record::RecordError> {
        Ok(Self {
            kind: ArtifactKind::Note,
            source: SourceRef::new("node-id", id)?,
            provenance: Provenance::default(),
        })
    }
}

/// What a write produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Written {
    /// The identifier the record is stored under.
    pub id: String,
    /// The address of the record blob, which is what reconciliation compares.
    pub record_hash: String,
    /// Whether the record already existed under this identifier.
    pub superseded: bool,
}

/// Why a write did not happen.
#[derive(Debug, Error)]
pub enum WriteError {
    /// The store rejected a blob or a name.
    #[error(transparent)]
    Store(#[from] crate::store::StoreError),
    /// The document could not be made into a record.
    #[error(transparent)]
    Record(#[from] record::RecordError),
    /// The index could not be re-derived for the record just written.
    #[error(transparent)]
    Index(#[from] crate::index::IndexError),
    /// The write was refused before anything was written.
    #[error("{0}")]
    Refused(String),
}

/// Write `document` as the record `id`, and re-derive its index rows.
///
/// The record's tags are the document's own: a tag is part of what was
/// written rather than metadata attached alongside it, so placing them in the
/// document before calling this is the caller's job and reading them back out
/// is not.
///
/// # Errors
///
/// [`WriteError`] naming the layer that refused: the store, the record
/// definition, or the index.
pub fn put_record(
    store: &impl BlobStore,
    index: &Index,
    id: &str,
    document: &Document,
    options: &WriteOptions,
) -> Result<Written, WriteError> {
    let existing = index.record(id).ok();
    let now = chrono::Utc::now();
    // A record's age is a fact about when it was first written. Taking the
    // creation time from the row that is being replaced is what keeps
    // `created` from quietly coming to mean `last edited`.
    let created = existing
        .as_ref()
        .and_then(|row| chrono::DateTime::parse_from_rfc3339(&row.created).ok())
        .map_or(now, |parsed| parsed.with_timezone(&chrono::Utc));

    let raw = crate::generator::generate(document);
    let stream = record::normalize(options.kind, raw.as_bytes())?;
    let raw_hash = store.put(raw.as_bytes())?;
    let stream_hash = store.put(stream.as_bytes())?;
    let header = RecordHeader {
        id: RecordId::new(id)?,
        corpus: CorpusId::new(crate::search::CORPUS_KB)?,
        kind: options.kind,
        created,
        updated: now,
        source: options.source.clone(),
        tags: document_tags(document)?,
        raw: ContentHash::new(raw_hash.as_str())?,
        stream: ContentHash::new(stream_hash.as_str())?,
        normalizer: record::NORMALIZER_VERSION,
        provenance: options.provenance.clone(),
    };
    let record_hash = store.put(&header.serialize())?;
    store.set_ref(
        &RefName::new(&format!("{}/{id}", crate::search::CORPUS_KB))?,
        &record_hash,
    )?;

    let mut one = BTreeSet::new();
    one.insert(id.to_owned());
    crate::index::reindex_records(store, index, &one)?;

    Ok(Written {
        id: id.to_owned(),
        record_hash: record_hash.as_str().to_owned(),
        superseded: existing.is_some(),
    })
}

/// The write options a rewrite of an existing record should carry: its
/// current kind, source and provenance, unchanged by whatever prompted the
/// rewrite.
///
/// `tags add`, `tags rm` and `tags merge` each rewrite a record's blobs to
/// change its tags, and nothing about a tag edit should be able to turn a
/// `session-transcript` back into a `note` or drop the provenance a capture
/// asserted — that was the bug `put_record`'s old hardcoded defaults caused.
/// `kb update` (T016) reuses the same starting point for the same reason: an
/// ordinary update, with no `--provenance-json`, must not be able to do what
/// a tag edit already cannot. Read from the stored header rather than the
/// index row, because the row (`RecordRow`) does not carry provenance or a
/// typed kind.
///
/// # Errors
///
/// [`WriteError`] if the record is unknown or its header blob cannot be read
/// or parsed.
pub(crate) fn existing_options(
    store: &impl BlobStore,
    index: &Index,
    id: &str,
) -> Result<WriteOptions, WriteError> {
    let row = index.record(id)?;
    let hash = row.record_hash.ok_or_else(|| {
        WriteError::Refused(format!("record {id} has no record blob to preserve"))
    })?;
    let blob = store.get(&BlobHash::from(&ContentHash::new(&hash)?))?;
    let header = RecordHeader::parse(&blob)?;
    Ok(WriteOptions {
        kind: header.kind,
        source: header.source,
        provenance: header.provenance,
    })
}

/// Read a record's artifact as text.
///
/// The raw bytes for a corpus that copies them into the store, and the
/// normalized stream for one that only references them. A reader wants what
/// was written: normalizing is for spans and hashes, and showing its output
/// as the record would report `#+filetags:` and property drawers as having
/// been dropped when they are sitting in the store untouched.
///
/// # Errors
///
/// [`WriteError`] if the record is unknown or its bytes cannot be read.
pub fn read_raw(store: &impl BlobStore, index: &Index, id: &str) -> Result<String, WriteError> {
    let row = index.record(id)?;
    if row.record_hash.is_none() {
        return Ok(index.record_text(id)?);
    }
    let hash = crate::store::BlobHash::from(&ContentHash::new(&row.raw_hash)?);
    let raw = store.get(&hash)?;
    Ok(String::from_utf8_lossy(&raw).into_owned())
}

/// Read a record's raw artifact back as a document.
///
/// From the raw blob rather than the index's cached text, because the raw is
/// what was written and the cached text is a normalization of it. An edit
/// applied to the normalization and written back would rewrite the artifact
/// as a side effect of changing a tag.
///
/// # Errors
///
/// [`WriteError`] if the record is unknown, its bytes are unreadable, or they
/// do not parse.
pub fn read_document(
    store: &impl BlobStore,
    index: &Index,
    id: &str,
) -> Result<tftio_org::ast::Document, WriteError> {
    let text = read_raw(store, index, id)?;
    crate::parser::parse_document(&text).map_err(|e| {
        WriteError::Record(record::RecordError::Unparsable {
            kind: ArtifactKind::Note,
            reason: e.to_string(),
        })
    })
}

/// Add tags to one record, rewriting it.
///
/// A tag is part of the artifact, so changing one means writing a new record:
/// new blobs, a new address, and the name advanced to it. There is no cheaper
/// path in a content-addressed store, and pretending otherwise — editing an
/// index row and leaving the artifact behind — is how the index stops being
/// derived.
///
/// # Errors
///
/// [`WriteError`] if the record cannot be read or written.
pub fn add_tags(
    store: &impl BlobStore,
    index: &Index,
    id: &str,
    tags: &[String],
) -> Result<crate::storage::TagEdit, WriteError> {
    let requested = crate::storage::normalize_arguments(tags)
        .map_err(|e| WriteError::Refused(e.to_string()))?;
    let mut document = read_document(store, index, id)?;
    let present: std::collections::HashSet<String> =
        crate::storage::tag_names(&document).into_iter().collect();
    let additions: Vec<String> = requested
        .into_iter()
        .filter(|tag| !present.contains(tag))
        .collect();
    if !additions.is_empty() {
        crate::storage::place_tags(&mut document, &additions);
        let options = existing_options(store, index, id)?;
        put_record(store, index, id, &document, &options)?;
    }
    Ok(crate::storage::TagEdit {
        id: id.to_owned(),
        changed: additions,
        tags: crate::storage::tag_names(&document),
    })
}

/// Remove tags from one record, rewriting it.
///
/// # Errors
///
/// [`WriteError`] if the record cannot be read or written.
pub fn remove_tags(
    store: &impl BlobStore,
    index: &Index,
    id: &str,
    tags: &[String],
) -> Result<crate::storage::TagEdit, WriteError> {
    let requested = crate::storage::normalize_arguments(tags)
        .map_err(|e| WriteError::Refused(e.to_string()))?;
    let mut document = read_document(store, index, id)?;
    let present: std::collections::HashSet<String> =
        crate::storage::tag_names(&document).into_iter().collect();
    let removals: Vec<String> = requested
        .into_iter()
        .filter(|tag| present.contains(tag))
        .collect();
    if !removals.is_empty() {
        let doomed: std::collections::HashSet<String> = removals.iter().cloned().collect();
        crate::storage::remove_tags_in_blocks(&mut document.blocks, &doomed);
        let options = existing_options(store, index, id)?;
        put_record(store, index, id, &document, &options)?;
    }
    Ok(crate::storage::TagEdit {
        id: id.to_owned(),
        changed: removals,
        tags: crate::storage::tag_names(&document),
    })
}

/// Rewrite every record carrying `from` to carry `to` instead.
///
/// # Errors
///
/// [`WriteError`] if a record cannot be read or written, and
/// [`WriteError::Refused`] for a merge that names an empty tag or merges a
/// tag onto itself.
pub fn merge_tag(
    store: &impl BlobStore,
    index: &Index,
    from: &str,
    to: &str,
) -> Result<crate::storage::TagMerge, WriteError> {
    let from_norm = crate::storage::normalize_tag(from);
    let to_norm = crate::storage::normalize_tag(to);
    if from_norm.is_empty() {
        return Err(WriteError::Refused(format!(
            "source tag {from:?} normalizes to nothing"
        )));
    }
    if to_norm.is_empty() {
        return Err(WriteError::Refused(format!(
            "target tag {to:?} normalizes to nothing"
        )));
    }
    if from_norm == to_norm {
        return Err(WriteError::Refused(format!(
            "{from:?} and {to:?} are the same tag ({from_norm}); nothing to merge"
        )));
    }
    let mut rewritten = Vec::new();
    for (id, _) in index.records_tagged(&from_norm)? {
        let mut document = read_document(store, index, &id)?;
        if crate::storage::rewrite_tag_in_blocks(&mut document.blocks, &from_norm, &to_norm) {
            let options = existing_options(store, index, &id)?;
            put_record(store, index, &id, &document, &options)?;
            rewritten.push(id);
        }
    }
    Ok(crate::storage::TagMerge {
        from: from_norm,
        to: to_norm,
        rewritten,
    })
}

/// The document's tags, in the form a record header accepts.
///
/// Read from the document rather than taken as an argument, so that a tag
/// written into the body and a tag stored on the record cannot disagree.
fn document_tags(document: &Document) -> Result<Vec<record::Tag>, record::RecordError> {
    let mut tags: Vec<String> = crate::storage::extract_tags(document)
        .into_iter()
        .map(|tag| tag.0)
        .collect();
    tags.sort();
    tags.dedup();
    tags.iter().map(|tag| record::Tag::new(tag)).collect()
}

/// What embedding one record's passages did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Embedded {
    /// Spans sent to the model and stored.
    pub embedded: usize,
    /// Spans that already had a vector under this model.
    pub skipped: usize,
    /// Bytes of text embedded, for the cost record.
    pub bytes: usize,
}

impl Embedded {
    /// Fold another record's result into this one.
    pub const fn add(&mut self, other: Self) {
        self.embedded = self.embedded.saturating_add(other.embedded);
        self.skipped = self.skipped.saturating_add(other.skipped);
        self.bytes = self.bytes.saturating_add(other.bytes);
    }
}

/// Embed the spans of one record that do not yet have a vector.
///
/// Resumable by construction: a vector is keyed by the stream it was computed
/// from, the span within that stream, and the model, so a span that already
/// has one is skipped and a run that stops partway leaves its finished work
/// behind. That is what lets the same function serve a corpus-wide first pass
/// and the single record a write just produced.
///
/// # Errors
///
/// [`EmbedRecordError::Endpoint`] if the endpoint refuses or cannot be
/// reached, and [`EmbedRecordError::Index`] if the index cannot be read or
/// written.
pub fn embed_record(
    index: &Index,
    embedder: &crate::cli_embed::CliEmbedder,
    id: &str,
    chunk_bytes: usize,
) -> Result<Embedded, EmbedRecordError> {
    let mut done = Embedded::default();
    for passage in index.passages(id)? {
        let start = usize::try_from(passage.span_start).unwrap_or(0);
        for (span_start, span_len) in
            crate::embed_text::chunk_span(&passage.text, start, chunk_bytes)
        {
            let (Ok(sql_start), Ok(sql_len)) = (i64::try_from(span_start), i64::try_from(span_len))
            else {
                continue;
            };
            if index
                .embedding(&passage.stream_hash, sql_start, sql_len, embedder.model())?
                .is_some()
            {
                done.skipped = done.skipped.saturating_add(1);
                continue;
            }
            let offset = span_start.saturating_sub(start);
            let text = passage
                .text
                .get(offset..offset.saturating_add(span_len))
                .unwrap_or_default();
            let vector = embedder.embed_document(text)?;
            index.put_embedding(
                &passage.stream_hash,
                sql_start,
                sql_len,
                embedder.model(),
                &crate::embedding::encode_embedding(&vector),
            )?;
            done.embedded = done.embedded.saturating_add(1);
            done.bytes = done.bytes.saturating_add(text.len());
        }
    }
    Ok(done)
}

/// Why embedding one record stopped.
#[derive(Debug, Error)]
pub enum EmbedRecordError {
    /// The endpoint refused or could not be reached.
    #[error(transparent)]
    Endpoint(#[from] crate::cli_embed::EmbedError),
    /// The index could not be read or written.
    #[error(transparent)]
    Index(#[from] crate::index::IndexError),
}
