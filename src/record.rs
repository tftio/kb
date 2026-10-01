//! What a stored record is.
//!
//! A record is three things held together. The **raw artifact** is the bytes
//! as they arrived, kept because ingest is lossy and the loss is otherwise
//! irreversible. The **normalized stream** is a deterministic, versioned
//! rendering of those bytes that passages address by byte offset. The
//! **header** carries everything the index needs, so an index row can be
//! rebuilt from the record alone and nothing in the index is authoritative
//! (ST-001, ST-002).
//!
//! The stream is versioned rather than merely deterministic. A byte offset
//! into raw source stays a valid offset when the parser changes while quietly
//! coming to mean something else — this repository has shipped two parser
//! fixes that would have done exactly that. Putting the normalizer's version
//! in the stream makes the stream's hash change when the interpretation
//! changes, so spans computed under the old reading are invalidated rather
//! than silently repointed.

use crate::{canonical, generator, parser};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The normalizer's version. Bumping it changes the bytes of every stream and
/// therefore every stream hash, which is the point: spans are only meaningful
/// against the interpretation that produced them.
pub const NORMALIZER_VERSION: u32 = 3;

/// Failures in reading or normalizing a record.
#[derive(Debug, Error)]
pub enum RecordError {
    /// The raw bytes were not UTF-8 text.
    #[error("{kind} artifact is not valid UTF-8: {reason}")]
    NotText {
        /// Kind being normalized when the decode failed.
        kind: ArtifactKind,
        /// Decoder failure detail, for display.
        reason: String,
    },

    /// The text did not parse as the format its kind implies.
    #[error("{kind} artifact did not parse: {reason}")]
    Unparsable {
        /// Kind being normalized when the parse failed.
        kind: ArtifactKind,
        /// Parser failure detail, for display.
        reason: String,
    },

    /// A header field was rejected at the boundary.
    #[error("record field {field} is not usable: {value:?} ({reason})")]
    InvalidField {
        /// Which field.
        field: String,
        /// What was offered.
        value: String,
        /// Which rule it broke.
        reason: String,
    },

    /// The bytes were not a record header.
    #[error("not a kb record: {reason}")]
    NotARecord {
        /// What was wrong.
        reason: String,
    },

    /// A required header field was absent.
    #[error("record header is missing {field}")]
    MissingField {
        /// Which field.
        field: String,
    },
}

/// What a record is, which decides how it normalizes and where its passage
/// boundaries fall.
///
/// A closed set, dispatched over exhaustively: a new corpus adds a variant and
/// the compiler names every place that has to account for it (ENG-009).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ArtifactKind {
    /// An authored org-mode note.
    Note,
    /// A captured agent session, whose top-level headings are its turns.
    SessionTranscript,
    /// A single mail message.
    MailMessage,
    /// A mail thread, as a sequence of messages.
    MailThread,
}

impl ArtifactKind {
    /// The kind's stable wire name, used in the record header.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Note => "note",
            Self::SessionTranscript => "session-transcript",
            Self::MailMessage => "mail-message",
            Self::MailThread => "mail-thread",
        }
    }
}

impl std::fmt::Display for ArtifactKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A deterministic, versioned rendering of a raw artifact.
///
/// Passages address this by byte offset, so its bytes are the interpretation
/// spans were computed against. It opens with `kb-stream/<version>\n`, which
/// is what ties the two together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedStream {
    bytes: Vec<u8>,
}

impl NormalizedStream {
    /// The whole stream, version header included. Span offsets are relative
    /// to this.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The stream past its `kb-stream/<version>` header.
    ///
    /// What a reconciliation compares when asking whether the normalizer would
    /// produce different *content* for a record. The whole stream cannot
    /// answer that: the version is embedded in the first line, so a bump made
    /// for one artifact kind changes the bytes of every record of every kind,
    /// and a byte comparison would call the entire corpus stale for a change
    /// that touched none of it.
    #[must_use]
    pub fn payload(&self) -> &[u8] {
        self.bytes.get(payload_offset(&self.bytes)..).unwrap_or(&[])
    }

    /// Take stored bytes back as a stream.
    ///
    /// Rebuilding the index reads streams back out of the store rather than
    /// re-normalizing from raw, since the stored bytes are by definition the
    /// interpretation the existing spans were computed against.
    #[must_use]
    pub const fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }
}

/// A record's stable identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RecordId(String);

/// Which corpus a record belongs to.
///
/// A validated string rather than an enum, deliberately: the corpus
/// vocabulary is fixed by the ground-truth question set (T013), and a closed
/// enum here would let this module invent identifiers the question set does
/// not use, silently breaking the join between an index row and an `expect`
/// entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CorpusId(String);

/// A tag, in the lowercase-kebab form kb stores.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tag(String);

/// A record's identity in the system it came from — a `Message-ID`, a session
/// id, a node id — as `scheme` and `value`.
///
/// Kept distinct from [`RecordId`] because a reference-only corpus resolves
/// through it: for mail, this is what finds the message in the Maildir, and
/// the filename never is, since flags rewrite it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceRef {
    scheme: String,
    value: String,
}

/// The hash of stored content, as hexadecimal.
///
/// Deliberately a plain validated string rather than the store's own hash
/// type: this module describes records and must not depend on which blob
/// store holds them.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContentHash(String);

/// Reject `value` with the reason it broke.
fn refuse<T>(field: &str, value: &str, reason: &str) -> Result<T, RecordError> {
    Err(RecordError::InvalidField {
        field: field.to_owned(),
        value: value.to_owned(),
        reason: reason.to_owned(),
    })
}

/// Fields must survive a line-oriented header, so nothing may contain a line
/// break or the `key: value` separator's leading colon.
fn line_safe(field: &str, value: &str) -> Result<String, RecordError> {
    if value.is_empty() {
        return refuse(field, value, "empty");
    }
    if value.contains(['\n', '\r']) {
        return refuse(field, value, "contains a line break");
    }
    Ok(value.to_owned())
}

impl RecordId {
    /// Validate `id` as a record identity.
    ///
    /// # Errors
    ///
    /// [`RecordError::InvalidField`] if it is empty or spans lines.
    pub fn new(id: &str) -> Result<Self, RecordError> {
        line_safe("id", id).map(Self)
    }

    /// The id as written.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl CorpusId {
    /// Validate `corpus` as a corpus identifier: lowercase ASCII, digits and
    /// hyphens.
    ///
    /// # Errors
    ///
    /// [`RecordError::InvalidField`] if it is empty or carries anything else.
    pub fn new(corpus: &str) -> Result<Self, RecordError> {
        let corpus = line_safe("corpus", corpus)?;
        if !corpus
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        {
            return refuse(
                "corpus",
                &corpus,
                "only lowercase ASCII letters, digits and '-' are allowed",
            );
        }
        Ok(Self(corpus))
    }

    /// The corpus as written.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Tag {
    /// Validate `tag` in kb's stored lowercase-kebab form.
    ///
    /// # Errors
    ///
    /// [`RecordError::InvalidField`] if it is empty or carries anything else.
    pub fn new(tag: &str) -> Result<Self, RecordError> {
        let tag = line_safe("tag", tag)?;
        if !tag
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        {
            return refuse(
                "tag",
                &tag,
                "only lowercase ASCII letters, digits and '-' are allowed",
            );
        }
        Ok(Self(tag))
    }

    /// The tag as written.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl SourceRef {
    /// Validate a source identity.
    ///
    /// # Errors
    ///
    /// [`RecordError::InvalidField`] if either half is empty, spans lines, or
    /// the scheme carries a colon.
    pub fn new(scheme: &str, value: &str) -> Result<Self, RecordError> {
        let scheme = line_safe("source scheme", scheme)?;
        if scheme.contains(':') {
            return refuse("source scheme", &scheme, "contains a colon");
        }
        Ok(Self {
            scheme,
            value: line_safe("source value", value)?,
        })
    }

    /// Which naming system the value belongs to.
    #[must_use]
    pub fn scheme(&self) -> &str {
        &self.scheme
    }

    /// The identity within that system.
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }
}

/// Where the work that produced a record was happening, and how sure that is.
///
/// Every field is independently optional: an old record blob carries none of
/// this, and even a fresh capture may carry only some of it — a session
/// outside any known project has no slug, and a hand-authored note written
/// through `kb create` typically carries none of it at all. These are header
/// fields rather than tags, deliberately: a bare project-slug tag would
/// collide with the pre-existing `project` tag (a memory-type label carried
/// over from an old import — see the README) and with topical tags such as
/// `clanker` that already exist for other reasons, and provenance is the
/// writer's assertion about where the work happened rather than something
/// derived from the document's own text.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Provenance {
    /// The project slug the session or write resolved to.
    pub project: Option<tftio_lib::project::Slug>,
    /// How `project` was resolved.
    pub project_source: Option<tftio_lib::project::Source>,
    /// The working directory's normalized origin remote, independent of
    /// whether it is what resolved `project`.
    pub remote: Option<tftio_lib::project::NormalizedRemote>,
    /// The credential boundary the session ran under (`personal`, `work`,
    /// ...). Not a project — see the ADR's "Context is not a project"
    /// constraint (`PLAN-20260923-project-identity`).
    pub context: Option<String>,
    /// Topical domains the session or note touches, repeated.
    pub domains: Vec<String>,
    /// The harness that captured the session (`claude-code`, `codex`, ...).
    pub harness: Option<String>,
    /// The model in use when the record was produced.
    pub model: Option<String>,
    /// The harness session id, for correlating a record back to its
    /// transcript source.
    pub session: Option<String>,
    /// The working directory the session ran in.
    pub cwd: Option<String>,
}

impl Provenance {
    /// Whether every field is absent.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

/// The wire shape of a [`Provenance`] object in JSON.
///
/// The ingest queue's `Submission.provenance` and `kb create
/// --provenance-json` both deserialize into this before boundary validation
/// turns it into the typed [`Provenance`] (`REPO_INVARIANTS.md` ENG-006).
/// Every field is a plain string so a caller can build one without depending
/// on `tftio_lib::project` directly.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawProvenance {
    /// See [`Provenance::project`].
    #[serde(default)]
    pub project: Option<String>,
    /// See [`Provenance::project_source`]: one of `declared`, `path`,
    /// `remote`, `derived`.
    #[serde(default)]
    pub project_source: Option<String>,
    /// See [`Provenance::remote`]. Must already be normalized: the value a
    /// caller offers must equal what `tftio_lib::project::normalize_remote`
    /// produces from it.
    #[serde(default)]
    pub remote: Option<String>,
    /// See [`Provenance::context`].
    #[serde(default)]
    pub context: Option<String>,
    /// See [`Provenance::domains`].
    #[serde(default)]
    pub domains: Vec<String>,
    /// See [`Provenance::harness`].
    #[serde(default)]
    pub harness: Option<String>,
    /// See [`Provenance::model`].
    #[serde(default)]
    pub model: Option<String>,
    /// See [`Provenance::session`].
    #[serde(default)]
    pub session: Option<String>,
    /// See [`Provenance::cwd`].
    #[serde(default)]
    pub cwd: Option<String>,
}

impl RawProvenance {
    /// Validate every field at the boundary, producing the typed
    /// [`Provenance`] a record header carries (ENG-006).
    ///
    /// # Errors
    ///
    /// [`RecordError::InvalidField`] naming the first field that fails: an
    /// unrecognized `project_source` label, a `remote` that is not already
    /// normalized, or any free-form field that is empty or spans a line
    /// (`remote` and `cwd` are named explicitly by the plan; every other
    /// free-form field is held to the same rule, since each is one line of a
    /// line-oriented header).
    pub fn validate(self) -> Result<Provenance, RecordError> {
        Ok(Provenance {
            project: self
                .project
                .map(|v| validate_slug("project", v))
                .transpose()?,
            project_source: self
                .project_source
                .as_deref()
                .map(validate_project_source)
                .transpose()?,
            remote: self.remote.as_deref().map(validate_remote).transpose()?,
            context: self.context.map(|v| line_safe("context", &v)).transpose()?,
            domains: self
                .domains
                .iter()
                .map(|v| line_safe("domain", v))
                .collect::<Result<_, _>>()?,
            harness: self.harness.map(|v| line_safe("harness", &v)).transpose()?,
            model: self.model.map(|v| line_safe("model", &v)).transpose()?,
            session: self.session.map(|v| line_safe("session", &v)).transpose()?,
            cwd: self.cwd.map(|v| line_safe("cwd", &v)).transpose()?,
        })
    }
}

/// Validate `value` as a [`tftio_lib::project::Slug`].
fn validate_slug(field: &str, value: String) -> Result<tftio_lib::project::Slug, RecordError> {
    tftio_lib::project::Slug::new(value.clone()).map_err(|e| RecordError::InvalidField {
        field: field.to_owned(),
        value,
        reason: e.to_string(),
    })
}

/// Read one of `tftio_lib::project::Source`'s stable labels back.
fn validate_project_source(value: &str) -> Result<tftio_lib::project::Source, RecordError> {
    use tftio_lib::project::Source;
    match value {
        "declared" => Ok(Source::Declared),
        "path" => Ok(Source::Path),
        "remote" => Ok(Source::Remote),
        "derived" => Ok(Source::Derived),
        other => refuse(
            "project-source",
            other,
            "not one of declared, path, remote, derived",
        ),
    }
}

/// Validate `value` as an already-normalized remote: it must equal its own
/// normalization, so the header never carries a remote spelling that
/// `tftio_lib::project::normalize_remote` would reduce further.
fn validate_remote(value: &str) -> Result<tftio_lib::project::NormalizedRemote, RecordError> {
    // `normalize_remote` normalizes a URL's structure; it has no opinion on
    // an embedded line break, which would corrupt this line-oriented header
    // regardless of whether the remote is otherwise well-formed.
    if value.contains(['\n', '\r']) {
        return refuse("remote", value, "contains a line break");
    }
    let normalized =
        tftio_lib::project::normalize_remote(value).map_err(|e| RecordError::InvalidField {
            field: "remote".to_owned(),
            value: value.to_owned(),
            reason: e.to_string(),
        })?;
    if normalized.as_str() != value {
        return refuse("remote", value, "not already normalized");
    }
    Ok(normalized)
}

impl ContentHash {
    /// Validate `hash` as hexadecimal.
    ///
    /// # Errors
    ///
    /// [`RecordError::InvalidField`] if it is empty or not hex.
    pub fn new(hash: &str) -> Result<Self, RecordError> {
        let hash = line_safe("hash", hash)?;
        if !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return refuse("hash", &hash, "not hexadecimal");
        }
        Ok(Self(hash))
    }

    /// The hash as written.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// What a link points at.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LinkTarget {
    /// Another record, by id.
    Record(RecordId),
    /// A record by its `#+name:` slug, which may not exist yet.
    Name(String),
    /// Something outside the corpus.
    Url(String),
}

/// A link the source itself wrote.
///
/// These are deterministic facts about the corpus — an org link, an RFC
/// `References` edge, a literal URL — and are re-derived from the stream
/// rather than stored, so they cannot drift from the content that carries
/// them. Each cites the span it was found in, which is what makes the claim
/// checkable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthoredLink {
    /// What it points at.
    pub target: LinkTarget,
    /// Where in the stream it was written.
    pub span: Span,
}

/// Who produced an inferred link, and how sure they were.
///
/// Named `LinkProvenance` rather than `Provenance` so it cannot be confused
/// with [`Provenance`], which names where the *work that wrote a record*
/// happened; the two describe unrelated things that happen to share a word.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkProvenance {
    /// Which generator asserted the link.
    pub generator: String,
    /// The exact model or rule version behind it.
    pub model_version: String,
    /// How sure it was, in `0.0..=1.0`.
    pub confidence: f32,
}

/// A link the system inferred.
///
/// Never interchangeable with an [`AuthoredLink`]: an embedding coincidence
/// must not be able to become equivalent to a relationship the author wrote.
/// Provenance is a required field rather than an option, so an inferred link
/// that cannot say where it came from cannot be constructed (ENG-009).
#[derive(Debug, Clone, PartialEq)]
pub struct InferredLink {
    /// What it points at.
    pub target: LinkTarget,
    /// Which record it was inferred from.
    pub source: RecordId,
    /// Who inferred it, and how sure.
    pub provenance: LinkProvenance,
}

/// A byte range in a [`NormalizedStream`].
///
/// Spans are the unit of citation: an index entry says which bytes of which
/// stream it came from, and the claim can be checked by reading them. That is
/// only worth anything against a stream whose hash pins the interpretation,
/// which is why spans address the normalized stream rather than raw source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Span {
    start: usize,
    len: usize,
}

impl Span {
    /// A span of `len` bytes beginning at `start`.
    #[must_use]
    pub const fn new(start: usize, len: usize) -> Self {
        Self { start, len }
    }

    /// First byte addressed.
    #[must_use]
    pub const fn start(self) -> usize {
        self.start
    }

    /// Number of bytes addressed.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Whether the span addresses nothing.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// One past the last byte addressed.
    #[must_use]
    pub const fn end(self) -> usize {
        self.start.saturating_add(self.len)
    }
}

/// What a passage is a passage *of*.
///
/// The three-level model — artifact, document, passage — is about identity,
/// containment and retrieval respectively. This names the containment level a
/// given passage sits at, which differs by source: a conversation's unit is a
/// turn, a note's is a section, correspondence has messages inside threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PassageLevel {
    /// One turn of a conversation.
    Turn,
    /// One section of an authored document.
    Section,
    /// One mail message.
    Message,
    /// A whole mail thread, containing its messages.
    Thread,
}

/// A retrievable span of a record, at a named containment level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Passage {
    /// Which bytes of the stream this is.
    pub span: Span,
    /// What the span is a passage of.
    pub level: PassageLevel,
}

/// Divide `stream` into the passages its kind implies.
///
/// Boundaries come from the source's own structure rather than from a
/// character budget: a turn, a section, a message. A budget-driven split lands
/// mid-argument, and an embedding of half an argument retrieves for neither
/// half.
#[must_use]
pub fn passages(kind: ArtifactKind, stream: &NormalizedStream) -> Vec<Passage> {
    let bytes = stream.as_bytes();
    let payload = payload_offset(bytes);
    match kind {
        ArtifactKind::SessionTranscript => org_sections(bytes, payload, PassageLevel::Turn),
        ArtifactKind::Note => org_sections(bytes, payload, PassageLevel::Section),
        ArtifactKind::MailMessage => vec![Passage {
            span: Span::new(payload, bytes.len().saturating_sub(payload)),
            level: PassageLevel::Message,
        }],
        ArtifactKind::MailThread => {
            let mut found = vec![Passage {
                span: Span::new(payload, bytes.len().saturating_sub(payload)),
                level: PassageLevel::Thread,
            }];
            found.extend(mbox_messages(bytes, payload));
            found
        }
    }
}

/// Where the payload begins: just past the `kb-stream/<version>\n` header.
fn payload_offset(bytes: &[u8]) -> usize {
    bytes
        .iter()
        .position(|b| *b == b'\n')
        .map_or(bytes.len(), |i| i.saturating_add(1))
}

/// Split at column-zero `* ` headings, which is where org puts a top-level
/// section and where the transcript importer puts each turn.
///
/// A document with no heading is one passage covering the whole payload
/// rather than none. Some notes are keyword-only — a `#+title:`, a
/// `#+filetags:`, a body — and content that yields no passage is content that
/// cannot be retrieved, which is indistinguishable from not having stored it.
fn org_sections(bytes: &[u8], payload: usize, level: PassageLevel) -> Vec<Passage> {
    let starts = line_starts_matching(bytes, payload, b"* ");
    if starts.is_empty() {
        let len = bytes.len().saturating_sub(payload);
        return if len == 0 {
            Vec::new()
        } else {
            vec![Passage {
                span: Span::new(payload, len),
                level,
            }]
        };
    }
    spans_between(starts, bytes.len())
        .map(|span| Passage { span, level })
        .collect()
}

/// Split an mbox at its `From ` envelope lines. This is the standard way
/// concatenated messages are delimited, so a thread export is readable by
/// anything that reads mail, and no kb-specific separator has to be invented.
fn mbox_messages(bytes: &[u8], payload: usize) -> Vec<Passage> {
    let starts = line_starts_matching(bytes, payload, b"From ");
    spans_between(starts, bytes.len())
        .map(|span| Passage {
            span,
            level: PassageLevel::Message,
        })
        .collect()
}

/// Offsets of every line at or after `from` that begins with `marker`.
fn line_starts_matching(bytes: &[u8], from: usize, marker: &[u8]) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut line_start = from;
    while line_start < bytes.len() {
        if bytes
            .get(line_start..)
            .is_some_and(|rest| rest.starts_with(marker))
        {
            starts.push(line_start);
        }
        match bytes
            .get(line_start..)
            .and_then(|rest| rest.iter().position(|b| *b == b'\n'))
        {
            Some(offset) => line_start = line_start.saturating_add(offset).saturating_add(1),
            None => break,
        }
    }
    starts
}

/// Turn a list of boundary offsets into spans, each running to the next
/// boundary and the last running to `end`.
fn spans_between(starts: Vec<usize>, end: usize) -> impl Iterator<Item = Span> {
    let ends: Vec<usize> = starts
        .iter()
        .skip(1)
        .copied()
        .chain(std::iter::once(end))
        .collect();
    starts
        .into_iter()
        .zip(ends)
        .map(|(start, stop)| Span::new(start, stop.saturating_sub(start)))
}

/// The record format's own version, distinct from the normalizer's: this one
/// changes when the header's shape changes, that one when the reading of
/// content changes.
///
/// Version 2 adds the optional provenance lines between `normalizer:` and the
/// `tag:` lines (`project`, `project-source`, `remote`, `context`, repeated
/// `domain`, `harness`, `model`, `session`, `cwd`). `parse` accepts both 1 and
/// 2: every provenance field is optional, so a version-1 blob parses with
/// [`Provenance::default`] and `kb reindex` succeeds over a store mixing both
/// (ST-001).
pub const RECORD_FORMAT_VERSION: u32 = 2;

/// Every record format version this build can still read.
const SUPPORTED_RECORD_FORMAT_VERSIONS: [u32; 2] = [1, 2];

/// Everything the index needs, carried by the record itself.
///
/// This is what makes rebuild possible (ST-001): hand [`index_row_from_blob`]
/// the stored bytes and it reconstructs the row, so nothing in the index is
/// the only copy of anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordHeader {
    /// Stable identity; the name a ref binds.
    pub id: RecordId,
    /// Which corpus it belongs to.
    pub corpus: CorpusId,
    /// What it is, which decides normalization and passage boundaries.
    pub kind: ArtifactKind,
    /// When the source says it came into being.
    pub created: chrono::DateTime<chrono::Utc>,
    /// When it last changed.
    pub updated: chrono::DateTime<chrono::Utc>,
    /// Its identity in the system it came from.
    pub source: SourceRef,
    /// Tags, in stored form.
    pub tags: Vec<Tag>,
    /// Hash of the raw archival bytes.
    pub raw: ContentHash,
    /// Hash of the normalized stream its spans address.
    pub stream: ContentHash,
    /// Which normalizer produced that stream.
    pub normalizer: u32,
    /// Where the work that produced this record was happening. Absent on
    /// every record written before `kb-record/2`.
    pub provenance: Provenance,
}

/// The index's view of a record: exactly the header, because everything the
/// index holds beyond this is derived from the stream rather than asserted.
pub type IndexRow = RecordHeader;

/// First line of every record blob.
const RECORD_MAGIC: &str = "kb-record/";

/// Append one `key: value` line, or a bare value when `key` is empty, which
/// is how the opening `kb-record/<version>` line is written.
fn field(out: &mut String, key: &str, value: &str) {
    if !key.is_empty() {
        out.push_str(key);
        out.push_str(": ");
    }
    out.push_str(value);
    out.push('\n');
}

impl RecordHeader {
    /// Serialize to the bytes stored as the record.
    ///
    /// Line-oriented and deterministic: the same header always produces the
    /// same bytes, which is required of anything content-addressed. Text
    /// rather than a binary encoding because the store is a git repository,
    /// where text deltas well and can be read by anything.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = String::new();
        out.push_str(RECORD_MAGIC);
        field(&mut out, "", &RECORD_FORMAT_VERSION.to_string());
        field(&mut out, "id", self.id.as_str());
        field(&mut out, "corpus", self.corpus.as_str());
        field(&mut out, "kind", self.kind.as_str());
        field(&mut out, "created", &self.created.to_rfc3339());
        field(&mut out, "updated", &self.updated.to_rfc3339());
        let source = [self.source.scheme(), self.source.value()].join(":");
        field(&mut out, "source", &source);
        field(&mut out, "raw", self.raw.as_str());
        field(&mut out, "stream", self.stream.as_str());
        field(&mut out, "normalizer", &self.normalizer.to_string());
        if let Some(project) = &self.provenance.project {
            field(&mut out, "project", project.as_str());
        }
        if let Some(source) = self.provenance.project_source {
            field(&mut out, "project-source", source.label());
        }
        if let Some(remote) = &self.provenance.remote {
            field(&mut out, "remote", remote.as_str());
        }
        if let Some(context) = &self.provenance.context {
            field(&mut out, "context", context);
        }
        for domain in &self.provenance.domains {
            field(&mut out, "domain", domain);
        }
        if let Some(harness) = &self.provenance.harness {
            field(&mut out, "harness", harness);
        }
        if let Some(model) = &self.provenance.model {
            field(&mut out, "model", model);
        }
        if let Some(session) = &self.provenance.session {
            field(&mut out, "session", session);
        }
        if let Some(cwd) = &self.provenance.cwd {
            field(&mut out, "cwd", cwd);
        }
        for tag in &self.tags {
            field(&mut out, "tag", tag.as_str());
        }
        out.into_bytes()
    }

    /// Parse a record blob back into its header.
    ///
    /// # Errors
    ///
    /// [`RecordError::NotARecord`] if the bytes do not open as a record,
    /// [`RecordError::MissingField`] if a required field is absent, and
    /// [`RecordError::InvalidField`] if one is present but unusable.
    pub fn parse(blob: &[u8]) -> Result<Self, RecordError> {
        let text = std::str::from_utf8(blob).map_err(|e| RecordError::NotARecord {
            reason: e.to_string(),
        })?;
        let mut lines = text.lines();
        let opening = lines.next().unwrap_or_default();
        let Some(version) = opening
            .strip_prefix(RECORD_MAGIC)
            .and_then(|v| v.parse::<u32>().ok())
        else {
            return Err(RecordError::NotARecord {
                reason: format!("expected a leading {RECORD_MAGIC}<version> line"),
            });
        };
        if !SUPPORTED_RECORD_FORMAT_VERSIONS.contains(&version) {
            return Err(RecordError::NotARecord {
                reason: format!("unsupported record format version {version}"),
            });
        }
        let mut fields: Vec<(&str, &str)> = Vec::new();
        for line in lines {
            if let Some((key, value)) = line.split_once(':') {
                fields.push((key.trim(), value.trim()));
            }
        }
        let one = |key: &str| -> Result<String, RecordError> {
            fields
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| (*v).to_owned())
                .ok_or_else(|| RecordError::MissingField {
                    field: key.to_owned(),
                })
        };
        // Fetched in the order the header declares, so a record missing
        // several fields always reports the first one rather than whichever
        // the compiler happened to evaluate first.
        let id = one("id")?;
        let corpus = one("corpus")?;
        let kind = one("kind")?;
        let created = one("created")?;
        let updated = one("updated")?;
        let source = one("source")?;
        let raw = one("raw")?;
        let stream = one("stream")?;
        let normalizer = one("normalizer")?;
        let (scheme, value) = source
            .split_once(':')
            .ok_or_else(|| RecordError::InvalidField {
                field: "source".to_owned(),
                value: source.clone(),
                reason: "expected <scheme>:<value>".to_owned(),
            })?;
        let provenance = parse_provenance(&fields)?;
        Ok(Self {
            id: RecordId::new(&id)?,
            corpus: CorpusId::new(&corpus)?,
            kind: parse_kind(&kind)?,
            created: parse_time("created", &created)?,
            updated: parse_time("updated", &updated)?,
            source: SourceRef::new(scheme, value)?,
            tags: fields
                .iter()
                .filter(|(k, _)| *k == "tag")
                .map(|(_, v)| Tag::new(v))
                .collect::<Result<Vec<_>, _>>()?,
            raw: ContentHash::new(&raw)?,
            stream: ContentHash::new(&stream)?,
            normalizer: normalizer.parse().map_err(|_| RecordError::InvalidField {
                field: "normalizer".to_owned(),
                value: normalizer.clone(),
                reason: "not a version number".to_owned(),
            })?,
            provenance,
        })
    }

    /// The index row this record implies.
    ///
    /// The row *is* the header: everything else the index holds — vectors,
    /// cached span text, generated summaries — is derived from the stream
    /// rather than asserted by the record, so there is nothing else to carry.
    #[must_use]
    pub fn index_row(&self) -> Self {
        self.clone()
    }
}

/// Rebuild an index row from stored bytes and nothing else.
///
/// # Errors
///
/// Whatever [`RecordHeader::parse`] reports.
pub fn index_row_from_blob(blob: &[u8]) -> Result<IndexRow, RecordError> {
    RecordHeader::parse(blob).map(|header| header.index_row())
}

/// Read the optional provenance block back out of a parsed header's fields.
///
/// Split out of [`RecordHeader::parse`] to keep that function under the line
/// budget; the provenance block is the reason it grew past it.
fn parse_provenance(fields: &[(&str, &str)]) -> Result<Provenance, RecordError> {
    let opt = |key: &str| -> Option<String> {
        fields
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| (*v).to_owned())
    };
    Ok(Provenance {
        project: opt("project")
            .map(|v| validate_slug("project", v))
            .transpose()?,
        project_source: opt("project-source")
            .as_deref()
            .map(validate_project_source)
            .transpose()?,
        remote: opt("remote").as_deref().map(validate_remote).transpose()?,
        context: opt("context")
            .map(|v| line_safe("context", &v))
            .transpose()?,
        domains: fields
            .iter()
            .filter(|(k, _)| *k == "domain")
            .map(|(_, v)| line_safe("domain", v))
            .collect::<Result<Vec<_>, _>>()?,
        harness: opt("harness")
            .map(|v| line_safe("harness", &v))
            .transpose()?,
        model: opt("model").map(|v| line_safe("model", &v)).transpose()?,
        session: opt("session")
            .map(|v| line_safe("session", &v))
            .transpose()?,
        cwd: opt("cwd").map(|v| line_safe("cwd", &v)).transpose()?,
    })
}

/// Read a kind's wire name back.
fn parse_kind(value: &str) -> Result<ArtifactKind, RecordError> {
    match value {
        "note" => Ok(ArtifactKind::Note),
        "session-transcript" => Ok(ArtifactKind::SessionTranscript),
        "mail-message" => Ok(ArtifactKind::MailMessage),
        "mail-thread" => Ok(ArtifactKind::MailThread),
        other => Err(RecordError::InvalidField {
            field: "kind".to_owned(),
            value: other.to_owned(),
            reason: "not a known artifact kind".to_owned(),
        }),
    }
}

/// Read an RFC 3339 timestamp.
fn parse_time(field: &str, value: &str) -> Result<chrono::DateTime<chrono::Utc>, RecordError> {
    chrono::DateTime::parse_from_rfc3339(value)
        .map(|t| t.with_timezone(&chrono::Utc))
        .map_err(|e| RecordError::InvalidField {
            field: field.to_owned(),
            value: value.to_owned(),
            reason: e.to_string(),
        })
}

/// Extract the links the source itself wrote, with the span of each.
///
/// Derived from the stream rather than stored in the header, so they cannot
/// disagree with the content that carries them — and so a re-chunk or a
/// re-parse re-derives them for free (ST-002).
#[must_use]
pub fn authored_links(stream: &NormalizedStream) -> Vec<AuthoredLink> {
    let text = String::from_utf8_lossy(stream.as_bytes());
    let mut links = Vec::new();
    let mut rest = text.as_ref();
    let mut base = 0usize;
    while let Some(open) = rest.find("[[") {
        let after_open = open.saturating_add(2);
        let Some(tail) = rest.get(after_open..) else {
            break;
        };
        let Some(close) = tail.find("]]") else {
            break;
        };
        let inner = tail.get(..close).unwrap_or_default();
        let whole = close.saturating_add(4);
        if !inner.is_empty() && !inner.contains('\n') {
            links.push(AuthoredLink {
                target: link_target(inner),
                span: Span::new(base.saturating_add(open), whole),
            });
        }
        let consumed = after_open.saturating_add(close).saturating_add(2);
        base = base.saturating_add(consumed);
        rest = rest.get(consumed..).unwrap_or_default();
    }
    links
}

/// Classify what a bracket reference points at.
fn link_target(inner: &str) -> LinkTarget {
    if let Some(id) = inner.strip_prefix("id:") {
        return RecordId::new(id)
            .map_or_else(|_| LinkTarget::Name(inner.to_owned()), LinkTarget::Record);
    }
    if inner.contains("://") || inner.starts_with("mailto:") {
        return LinkTarget::Url(inner.to_owned());
    }
    LinkTarget::Name(inner.to_owned())
}

/// Render `raw` into its normalized stream.
///
/// # Errors
///
/// [`RecordError::NotText`] if the bytes are not UTF-8, or
/// [`RecordError::Unparsable`] if they do not parse as the kind implies.
pub fn normalize(kind: ArtifactKind, raw: &[u8]) -> Result<NormalizedStream, RecordError> {
    normalize_with(kind, raw, NORMALIZER_VERSION)
}

/// Normalize under a named version.
///
/// Exists so the versioning claim is testable rather than asserted: the whole
/// argument for versioning the stream is that a future normalizer produces
/// different bytes for unchanged input, and that cannot be demonstrated from
/// a single compiled-in constant.
fn normalize_with(
    kind: ArtifactKind,
    raw: &[u8],
    version: u32,
) -> Result<NormalizedStream, RecordError> {
    let body = match kind {
        // Org artifacts are the operator's own text and are UTF-8 by
        // construction; anything else is a defect worth reporting rather than
        // silently transcoding.
        ArtifactKind::Note | ArtifactKind::SessionTranscript => {
            let text = std::str::from_utf8(raw).map_err(|e| RecordError::NotText {
                kind,
                reason: e.to_string(),
            })?;
            normalize_org(kind, text)?
        }
        // Mail is not reliably UTF-8 and its body is not its bytes: 5,357 of
        // the first increment's 6,305 messages are multipart and 788 are
        // HTML, so canonicalizing means reading the MIME tree rather than
        // reading the file. The character set is the part's, declared in its
        // own headers, which is why this path takes bytes.
        ArtifactKind::MailMessage => crate::message::readable(raw),
        ArtifactKind::MailThread => normalize_mail_thread(raw),
    };
    let mut bytes = format!("kb-stream/{version}\n").into_bytes();
    bytes.extend_from_slice(body.as_bytes());
    Ok(NormalizedStream { bytes })
}

/// Canonical mail thread: each member message normalized in place, still
/// delimited by the `From ` envelope line it arrived with.
///
/// Normalizing a thread as though it were one message would keep only the
/// first message's headers and swallow the rest into a body, which is how the
/// thread's structure gets lost — and with it the ability to cite the message
/// that actually said the thing.
fn normalize_mail_thread(raw: &[u8]) -> String {
    let text = String::from_utf8_lossy(raw).replace("\r\n", "\n");
    let mut out = String::new();
    for (envelope, message) in split_mbox(&text) {
        out.push_str(envelope);
        out.push('\n');
        out.push_str(&crate::message::readable(message.as_bytes()));
    }
    if out.is_empty() {
        return crate::message::readable(text.as_bytes());
    }
    out
}

/// Split an mbox into `(envelope line, message)` pairs at column-zero
/// `From ` lines. Yields nothing when the text carries no envelope line at
/// all, which is the caller's signal that this is a bare message.
fn split_mbox(text: &str) -> Vec<(&str, &str)> {
    let mut messages = Vec::new();
    let mut current: Option<(&str, usize)> = None;
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        if line.starts_with("From ") {
            if let Some((envelope, start)) = current.take() {
                messages.push((envelope, text.get(start..offset).unwrap_or_default()));
            }
            current = Some((
                line.trim_end_matches('\n'),
                offset.saturating_add(line.len()),
            ));
        }
        offset = offset.saturating_add(line.len());
    }
    if let Some((envelope, start)) = current {
        messages.push((envelope, text.get(start..).unwrap_or_default()));
    }
    messages
}

/// Canonical org: parse, collapse to the canonical AST, render back. Two
/// documents that render identically normalize identically, which is what
/// makes the stream hash a statement about content rather than about
/// formatting.
fn normalize_org(kind: ArtifactKind, text: &str) -> Result<String, RecordError> {
    let parsed = parser::parse_document(text).map_err(|e| RecordError::Unparsable {
        kind,
        reason: e.to_string(),
    })?;
    Ok(generator::generate(&canonical::canonicalize(&parsed)))
}

/// The one-line name of a record, derived from the text it was made of.
///
/// Derived rather than stored, because a title is not a fact about a record
/// separate from its content: it is the content's own first claim about
/// itself. The superseded `kb.db` kept a `title` column, which is why the
/// derivation exists at all — retiring that column (T029) means the index has
/// to recover from the store what the column used to hold, and it can, for
/// every record but a handful whose title was only ever in the database.
///
/// Org artifacts name themselves with `#+title:` where they have one and with
/// their first top-level heading otherwise, minus the tag suffix, since
/// `:design:adr:` is filing rather than part of the name. A message is named by
/// its subject, which its normalized stream opens with and which is the only
/// line that describes it in one.
///
/// A record with none of those is nameless. Falling back to the first line of
/// the body would produce something that reads like a title and is not one,
/// which is worse than an empty string a caller can recognise.
#[must_use]
pub fn title_of(kind: ArtifactKind, text: &str) -> String {
    match kind {
        ArtifactKind::Note | ArtifactKind::SessionTranscript => org_title(text),
        ArtifactKind::MailMessage | ArtifactKind::MailThread => subject_title(text),
    }
}

/// The `#+name:` slug an org artifact declares, if any.
///
/// A `[[name]]` link resolves against this rather than against the title: a
/// slug is a stable handle the author chose, and a title is prose that
/// changes. Derived here for the same reason [`title_of`] is — the superseded
/// database materialized it in a column, and the index has to recover it from
/// the store.
#[must_use]
pub fn name_slug_of(text: &str) -> Option<String> {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("#+name:") {
            let trimmed = rest.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_owned());
            }
        }
    }
    None
}

/// `#+title:` if the text declares one, else its first top-level heading.
fn org_title(text: &str) -> String {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("#+title:") {
            return rest.trim().to_owned();
        }
    }
    text.lines()
        .find_map(|line| line.strip_prefix("* "))
        .map(strip_tag_suffix)
        .unwrap_or_default()
}

/// A heading without its trailing `:tag:tag:` block.
///
/// Matched structurally rather than by regular expression: a suffix qualifies
/// only if it opens and closes with a colon and everything between the colons
/// is tag-shaped, so a heading ending in a bare colon or in prose containing
/// one keeps every word.
fn strip_tag_suffix(heading: &str) -> String {
    let trimmed = heading.trim_end();
    let Some(open) = trimmed.rfind(|c: char| c.is_whitespace()) else {
        return trimmed.to_owned();
    };
    let Some(suffix) = trimmed.get(open.saturating_add(1)..) else {
        return trimmed.to_owned();
    };
    let tags_only = suffix.len() > 2
        && suffix.starts_with(':')
        && suffix.ends_with(':')
        && suffix.trim_matches(':').split(':').all(|tag| {
            !tag.is_empty()
                && tag
                    .chars()
                    .all(|c| c.is_alphanumeric() || "_@%-".contains(c))
        });
    if tags_only {
        trimmed.get(..open).unwrap_or(trimmed).trim_end().to_owned()
    } else {
        trimmed.to_owned()
    }
}

/// The `subject:` line a normalized message opens with.
fn subject_title(text: &str) -> String {
    for line in text.lines() {
        if let Some(subject) = line.strip_prefix("subject: ") {
            return subject.trim().to_owned();
        }
        if line.trim().is_empty() {
            break;
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::{ArtifactKind, NORMALIZER_VERSION, RecordHeader, normalize_with, title_of};

    /// A note names itself with `#+title:` when it has one.
    #[test]
    fn a_title_keyword_names_the_record() {
        let raw =
            "#+title: Talking Machines\n\n* 2015-09-24 [[https://example.invalid][permalink]]\n";
        assert_eq!(title_of(ArtifactKind::Note, raw), "Talking Machines");
    }

    /// Without one, the first heading names it — with its tag suffix removed,
    /// since `:design:adr:` is filing rather than part of the name.
    #[test]
    fn the_first_heading_names_a_record_that_has_no_title_keyword() {
        let raw = ":PROPERTIES:\n:ID: x\n:END:\n* Choosing a queue backend :design:adr:\n\nbody\n";
        assert_eq!(
            title_of(ArtifactKind::Note, raw),
            "Choosing a queue backend"
        );
    }

    /// A message is named by its subject, which its normalized stream opens
    /// with. Nothing else in a message describes it in one line.
    #[test]
    fn a_message_is_named_by_its_subject() {
        let stream = "from: ada@example.invalid\nsubject: quarterly revenue\n\nbody\n";
        assert_eq!(
            title_of(ArtifactKind::MailMessage, stream),
            "quarterly revenue"
        );
    }

    /// A record with nothing to name it is nameless rather than named after
    /// the first line of its body, which would read as a title and be wrong.
    #[test]
    fn a_record_with_no_heading_and_no_subject_is_nameless() {
        assert_eq!(title_of(ArtifactKind::Note, "just a paragraph\n"), "");
    }

    const NOTE: &[u8] = b"* A note\n\nunchanged input.\n";

    /// The reason the version is in the stream at all. Spans are byte offsets
    /// into an interpretation; when the interpretation changes the stream
    /// hash must change with it, so the spans derived from the old reading
    /// are invalidated rather than left pointing at text that now parses
    /// differently.
    #[test]
    fn bumping_the_normalizer_changes_the_stream_for_unchanged_input() {
        let current = normalize_with(ArtifactKind::Note, NOTE, NORMALIZER_VERSION)
            .expect("normalize under the current version");
        let next = normalize_with(ArtifactKind::Note, NOTE, NORMALIZER_VERSION + 1)
            .expect("normalize under the next version");

        assert_ne!(
            current.as_bytes(),
            next.as_bytes(),
            "the same input normalized identically under two normalizer versions"
        );

        // Stated in the currency the index actually uses: content addressing
        // means the hash follows the bytes, so a version bump invalidates the
        // stored stream and every span computed against it.
        let dir = tempfile::tempdir().expect("tempdir");
        let store = crate::store::GitBlobStore::open_or_init(dir.path()).expect("store");
        let hash_of = |stream: &super::NormalizedStream| {
            crate::store::BlobStore::put(&store, stream.as_bytes()).expect("put")
        };
        assert_ne!(hash_of(&current), hash_of(&next));
    }

    /// Content that is not text cannot be normalized, and saying so names the
    /// kind so the caller knows which ingest path produced it.
    #[test]
    fn bytes_that_are_not_text_are_refused_by_kind() {
        let outcome = normalize_with(ArtifactKind::Note, &[0xff, 0xfe], NORMALIZER_VERSION);

        assert!(
            matches!(
                outcome,
                Err(super::RecordError::NotText {
                    kind: ArtifactKind::Note,
                    ..
                })
            ),
            "outcome was: {outcome:?}"
        );
    }

    /// Header parsing rejects a record whose kind it does not know, rather
    /// than defaulting to one and normalizing it wrongly ever after.
    #[test]
    fn a_record_of_an_unknown_kind_is_refused() {
        let blob = b"kb-record/1\nid: n1\ncorpus: kb\nkind: hologram\n\
                     created: 2026-05-19T09:00:00Z\nupdated: 2026-05-19T09:00:00Z\n\
                     source: node-id:n1\nraw: ab\nstream: cd\nnormalizer: 1\n";

        let outcome = RecordHeader::parse(blob);

        assert!(
            matches!(
                outcome,
                Err(super::RecordError::InvalidField { ref field, .. }) if field == "kind"
            ),
            "outcome was: {outcome:?}"
        );
    }

    /// A record missing a field the index needs is refused by name, because
    /// the alternative is an index row with a silent hole in it.
    #[test]
    fn a_record_missing_a_required_field_names_it() {
        let blob = b"kb-record/1\nid: n1\ncorpus: kb\nkind: note\n";

        let outcome = RecordHeader::parse(blob);

        assert!(
            matches!(
                outcome,
                Err(super::RecordError::MissingField { ref field }) if field == "created"
            ),
            "outcome was: {outcome:?}"
        );
    }
}

#[cfg(test)]
mod field_tests {
    use super::{
        ArtifactKind, ContentHash, CorpusId, LinkTarget, NormalizedStream, RecordError,
        RecordHeader, RecordId, SourceRef, Span, Tag, authored_links, normalize,
    };

    fn reason(outcome: Result<impl std::fmt::Debug, RecordError>) -> String {
        match outcome {
            Err(RecordError::InvalidField { reason, .. }) => reason,
            other => panic!("expected InvalidField, got: {other:?}"),
        }
    }

    /// Every field is narrowed at the boundary (ENG-006), so a record that
    /// exists is a record that serializes, round-trips and indexes. A value
    /// admitted here would fail much later, in the index, as corruption.
    #[test]
    fn fields_that_would_not_survive_the_header_are_refused() {
        assert_eq!(reason(RecordId::new("")), "empty");
        assert_eq!(reason(RecordId::new("n1\nid: n2")), "contains a line break");
        assert_eq!(
            reason(CorpusId::new("KB")),
            "only lowercase ASCII letters, digits and '-' are allowed"
        );
        assert_eq!(
            reason(Tag::new("Silent Critic")),
            "only lowercase ASCII letters, digits and '-' are allowed"
        );
        assert_eq!(
            reason(SourceRef::new("mail:id", "m1")),
            "contains a colon",
            "a scheme carrying a colon would split ambiguously on parse"
        );
        assert_eq!(reason(ContentHash::new("zzzz")), "not hexadecimal");
    }

    /// The wire names are stored in every record blob, so they are format,
    /// not decoration: renaming one silently orphans every record that
    /// carries the old spelling.
    #[test]
    fn every_kind_round_trips_through_its_wire_name() {
        for kind in [
            ArtifactKind::Note,
            ArtifactKind::SessionTranscript,
            ArtifactKind::MailMessage,
            ArtifactKind::MailThread,
        ] {
            assert_eq!(super::parse_kind(kind.as_str()).ok(), Some(kind));
            assert_eq!(kind.to_string(), kind.as_str());
        }
    }

    #[test]
    fn a_span_reports_its_extent() {
        let span = Span::new(12, 30);
        assert_eq!(span.start(), 12);
        assert_eq!(span.len(), 30);
        assert_eq!(span.end(), 42);
        assert!(!span.is_empty());
        assert!(Span::new(12, 0).is_empty());
    }

    /// A literal URL is a link out of the corpus, not a reference to a record
    /// that has not been written yet, and conflating them would fill the
    /// broken-link report with things that were never meant to resolve.
    #[test]
    fn a_bracketed_url_is_a_link_out_of_the_corpus() {
        let stream = normalize(
            ArtifactKind::Note,
            b"* A\n\nSee [[https://example.com/spec]] and [[not-written-yet]].\n",
        )
        .expect("normalize");

        let links = authored_links(&stream);

        assert_eq!(
            links.iter().map(|l| l.target.clone()).collect::<Vec<_>>(),
            vec![
                LinkTarget::Url("https://example.com/spec".to_owned()),
                LinkTarget::Name("not-written-yet".to_owned()),
            ]
        );
    }

    /// An unterminated bracket is text, not a link: transcripts quote code
    /// and prose that opens brackets without closing them.
    #[test]
    fn an_unclosed_bracket_is_not_a_link() {
        let stream = NormalizedStream::from_bytes(b"kb-stream/1\nsee [[unterminated\n".to_vec());

        assert!(authored_links(&stream).is_empty());
    }

    #[test]
    fn a_record_whose_bytes_are_not_text_is_refused() {
        let outcome = RecordHeader::parse(&[b'k', 0xff, 0xfe]);

        assert!(
            matches!(outcome, Err(RecordError::NotARecord { .. })),
            "outcome was: {outcome:?}"
        );
    }

    #[test]
    fn a_source_without_a_scheme_is_refused() {
        let outcome = RecordHeader::parse(&blob_with("source", "bare-value"));

        assert_eq!(reason(outcome), "expected <scheme>:<value>");
    }

    #[test]
    fn a_normalizer_that_is_not_a_version_is_refused() {
        let outcome = RecordHeader::parse(&blob_with("normalizer", "one"));

        assert_eq!(reason(outcome), "not a version number");
    }

    #[test]
    fn a_timestamp_that_is_not_rfc3339_is_refused() {
        let outcome = RecordHeader::parse(&blob_with("created", "19th of May"));

        assert!(
            matches!(
                outcome,
                Err(RecordError::InvalidField { ref field, .. }) if field == "created"
            ),
            "outcome was: {outcome:?}"
        );
    }

    /// A well-formed record blob with one field replaced.
    fn blob_with(field: &str, value: &str) -> Vec<u8> {
        let mut out = String::from("kb-record/1\n");
        for (key, default) in [
            ("id", "n1"),
            ("corpus", "kb"),
            ("kind", "note"),
            ("created", "2026-05-19T09:00:00Z"),
            ("updated", "2026-05-19T09:00:00Z"),
            ("source", "node-id:n1"),
            ("raw", "ab"),
            ("stream", "cd"),
            ("normalizer", "1"),
        ] {
            let written = if key == field { value } else { default };
            out.push_str(key);
            out.push_str(": ");
            out.push_str(written);
            out.push('\n');
        }
        out.into_bytes()
    }

    /// A well-formed header with no provenance, for the provenance tests to
    /// build on.
    fn header() -> RecordHeader {
        RecordHeader {
            id: RecordId::new("n1").expect("id"),
            corpus: CorpusId::new("kb").expect("corpus"),
            kind: ArtifactKind::Note,
            created: "2026-05-19T09:00:00Z".parse().expect("created"),
            updated: "2026-05-19T09:00:00Z".parse().expect("updated"),
            source: SourceRef::new("node-id", "n1").expect("source"),
            tags: vec![Tag::new("rust").expect("tag")],
            raw: ContentHash::new("ab").expect("raw"),
            stream: ContentHash::new("cd").expect("stream"),
            normalizer: 1,
            provenance: super::Provenance::default(),
        }
    }

    /// A header carrying every provenance field, so the round-trip test
    /// exercises each one rather than only the empty case.
    fn header_with_full_provenance() -> RecordHeader {
        RecordHeader {
            provenance: super::Provenance {
                project: Some(tftio_lib::project::Slug::new("kb").expect("slug")),
                project_source: Some(tftio_lib::project::Source::Declared),
                remote: Some(
                    tftio_lib::project::normalize_remote("https://github.com/tftio/kb")
                        .expect("remote"),
                ),
                context: Some("personal".to_owned()),
                domains: vec!["clanker".to_owned(), "silent-critic".to_owned()],
                harness: Some("claude-code".to_owned()),
                model: Some("claude-sonnet-5".to_owned()),
                session: Some("s-123".to_owned()),
                cwd: Some("/Users/op/Projects/kb/main".to_owned()),
            },
            ..header()
        }
    }

    /// A record round-trips through serialize and parse byte-identically
    /// with every provenance field set.
    #[test]
    fn a_header_with_full_provenance_round_trips() {
        let head = header_with_full_provenance();
        let blob = head.serialize();

        let parsed = RecordHeader::parse(&blob).expect("parse");

        assert_eq!(parsed, head);
        assert_eq!(parsed.serialize(), blob, "serialize must be deterministic");
    }

    /// A record round-trips with no provenance field set: the format bump
    /// costs nothing for a record that asserts none of it.
    #[test]
    fn a_header_with_no_provenance_round_trips() {
        let head = header();
        let blob = head.serialize();

        assert!(
            !String::from_utf8_lossy(&blob).contains("project"),
            "no provenance line should be written when nothing is set"
        );
        let parsed = RecordHeader::parse(&blob).expect("parse");
        assert_eq!(parsed, head);
        assert!(parsed.provenance.is_empty());
    }

    /// A `kb-record/1` blob, carrying no provenance lines at all, still
    /// parses — the invariant `kb reindex` depends on to succeed over a
    /// mixed store.
    #[test]
    fn a_version_one_blob_parses_with_default_provenance() {
        let blob = blob_with("id", "n1");
        assert!(String::from_utf8_lossy(&blob).starts_with("kb-record/1\n"));

        let parsed = RecordHeader::parse(&blob).expect("parse");

        assert!(parsed.provenance.is_empty());
    }

    /// An unrecognized record format version is refused rather than
    /// silently misread.
    #[test]
    fn an_unknown_record_format_version_is_refused() {
        let text = String::from_utf8(blob_with("id", "n1")).expect("utf8");
        let blob = text.replace("kb-record/1", "kb-record/99").into_bytes();
        let outcome = RecordHeader::parse(&blob);
        assert!(
            matches!(outcome, Err(RecordError::NotARecord { .. })),
            "outcome was: {outcome:?}"
        );
    }

    /// A remote that is not already normalized is refused: the header must
    /// never carry a spelling `normalize_remote` would reduce further.
    #[test]
    fn a_remote_that_is_not_normalized_is_refused() {
        let outcome = super::RawProvenance {
            remote: Some("https://github.com/tftio/kb.git".to_owned()),
            ..Default::default()
        }
        .validate();

        assert_eq!(reason(outcome), "not already normalized");
    }

    /// An already-normalized remote is accepted.
    #[test]
    fn an_already_normalized_remote_is_accepted() {
        let provenance = super::RawProvenance {
            remote: Some("github.com/tftio/kb".to_owned()),
            ..Default::default()
        }
        .validate()
        .expect("normalized remote");

        assert_eq!(
            provenance
                .remote
                .map(tftio_lib::project::NormalizedRemote::into_string),
            Some("github.com/tftio/kb".to_owned())
        );
    }

    /// `project_source` only accepts the four stable labels
    /// `tftio_lib::project::Source` reports.
    #[test]
    fn an_unknown_project_source_label_is_refused() {
        let outcome = super::RawProvenance {
            project_source: Some("guessed".to_owned()),
            ..Default::default()
        }
        .validate();

        assert_eq!(
            reason(outcome),
            "not one of declared, path, remote, derived"
        );
    }

    /// `remote` and `cwd` are named explicitly by the plan as fields that
    /// must reject an embedded line break, since a value like that could
    /// otherwise inject a forged header line.
    #[test]
    fn remote_and_cwd_reject_an_embedded_newline() {
        assert_eq!(
            reason(
                super::RawProvenance {
                    remote: Some("github.com/tftio/kb\nproject: evil".to_owned()),
                    ..Default::default()
                }
                .validate()
            ),
            "contains a line break"
        );
        assert_eq!(
            reason(
                super::RawProvenance {
                    cwd: Some("/tmp\nproject: evil".to_owned()),
                    ..Default::default()
                }
                .validate()
            ),
            "contains a line break"
        );
    }
}
