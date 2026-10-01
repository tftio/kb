//! The derived index.
//!
//! Everything here is a projection of the store and can be thrown away
//! (ST-002). That is not a nicety: it is what turns a retrieval experiment
//! from a schema migration against the only copy of the corpus into a
//! rebuild, which is the change this whole architecture exists to make.
//!
//! Two consequences follow, and both are deliberate. The index denormalizes
//! freely — passage text is cached beside the row that addresses it — because
//! drift is not a risk against something rebuilt in minutes, and because git
//! serves whole blobs, so reading a 1KB span from a 260KB note would
//! otherwise cost the whole object. And no column may hold anything the store
//! cannot produce again: a value that exists only here would quietly make the
//! index authoritative for it.

use crate::record::{self, RecordError, RecordHeader};
use crate::store::{BlobHash, BlobStore, StoreError};
use rusqlite::Connection;
use std::path::Path;
use thiserror::Error;

/// Where the derived index lives when nothing says otherwise:
/// `$HOME/.local/share/kb/index.db`, beside the store it is derived from.
///
/// Deliberately not the same file as the superseded `kb.db`: that database is
/// a frozen export source, and pointing the index at it would put a
/// disposable artifact and an archive in one file.
#[must_use]
pub fn default_index_path() -> std::path::PathBuf {
    dirs::home_dir().map_or_else(
        || std::path::PathBuf::from("index.db"),
        |home| home.join(".local/share/kb/index.db"),
    )
}

/// The index path the process is configured with: `KB_INDEX_PATH` when set
/// and non-empty, else [`default_index_path`]. The counterpart of
/// [`crate::store::configured_store_path`], for the same reason.
#[must_use]
#[allow(
    clippy::disallowed_methods,
    reason = "the index location is deployment config read once at the binary edge, like the embedding endpoint (REPO_INVARIANTS.md ENG-013)"
)]
pub fn configured_index_path() -> std::path::PathBuf {
    index_path_from(std::env::var_os("KB_INDEX_PATH"))
}

/// Pure variant of [`configured_index_path`]: the caller supplies what
/// `KB_INDEX_PATH` held. Empty counts as unset.
#[must_use]
pub fn index_path_from(value: Option<std::ffi::OsString>) -> std::path::PathBuf {
    value
        .filter(|value| !value.is_empty())
        .map_or_else(default_index_path, std::path::PathBuf::from)
}

/// The statement behind [`Index::rank_by_embedding`]: every vector under the
/// stream of every record in the corpus, by stream identity.
///
/// Public so a test can ask `SQLite` for its plan, as with
/// [`SEARCH_TEXT_IN_SQL`]: the plan must walk the corpus's records and seek
/// their vectors, never sort the vector blobs — the `DISTINCT` the earlier
/// containment join needed cost more than the scan itself.
pub const RANK_BY_EMBEDDING_SQL: &str = "SELECT r.record_id, e.vector
     FROM embeddings e
     JOIN records r ON r.stream_hash = e.stream_hash
     WHERE e.model = ?1 AND r.corpus = ?2
       AND (?3 IS NULL OR r.project = ?3)
       AND (?4 IS NULL OR r.context = ?4)";

/// The statement behind [`Index::search_text_in`].
///
/// Public so a test can ask `SQLite` what plan it chooses for it: the shape of
/// that plan is a correctness property here, not an optimization. See
/// `tests/index.rs::the_text_match_is_never_the_innermost_loop`.
pub const SEARCH_TEXT_IN_SQL: &str = "WITH matched AS MATERIALIZED (
         SELECT rowid AS passage_id, rank AS score
         FROM passages_fts
         WHERE passages_fts MATCH ?1
     )
     SELECT p.record_id, MIN(m.score) AS best
     FROM matched m
     JOIN passages p ON p.passage_id = m.passage_id
     JOIN records r ON r.record_id = p.record_id
     WHERE r.corpus = ?2
       AND (?3 IS NULL OR r.project = ?3)
       AND (?4 IS NULL OR r.context = ?4)
     GROUP BY p.record_id
     ORDER BY best";

/// The index schema's version. It is not a migration counter: the index is
/// rebuilt rather than migrated, so a change here means the next rebuild
/// writes the new shape and the old file is discarded.
///
/// Version 6 adds the nullable provenance columns on `records` (`project`,
/// `project_source`, `remote`, `context`, `harness`, `model`, `session`,
/// `cwd`), the `record_domains` table for the repeated `domain` header line,
/// and indexes on `records.project` and `records.context`.
pub const INDEX_SCHEMA_VERSION: u32 = 6;

/// How passage similarities become one dense score for a record.
///
/// The choice is explicit because records contain radically different numbers
/// of passage vectors. A maximum gives every passage another chance to set the
/// record's score; the other modes are measured alternatives to that length
/// bias rather than hidden changes to the dense signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensePooling {
    /// The record scores as its single best passage.
    Maximum,
    /// The record scores as the mean of its best three passages, or all of
    /// them when it has fewer than three.
    MeanTopThree,
    /// The maximum less the empirical expectation of a maximum over the same
    /// number of draws from this query's corpus-wide passage scores.
    LengthNormalizedMaximum,
}

/// The pooling mode shipped when no experiment names another one.
pub const DEFAULT_DENSE_POOLING: DensePooling = DensePooling::MeanTopThree;

/// Fit the expected maximum for every record length the selected mode needs.
fn expected_maxima_for(
    records: &std::collections::BTreeMap<String, Vec<f32>>,
    population: &[f32],
    pooling: DensePooling,
) -> std::collections::BTreeMap<usize, f32> {
    if pooling != DensePooling::LengthNormalizedMaximum {
        return std::collections::BTreeMap::new();
    }
    records
        .values()
        .map(Vec::len)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .filter_map(|draws| {
            expected_empirical_maximum(population, draws).map(|expected| (draws, expected))
        })
        .collect()
}

/// The exact expected maximum of `draws` independent samples from an
/// empirical distribution whose observations are sorted ascending.
///
/// For observation `x_i`, `F(x_i)^n - F(x_{i-1})^n` is the probability that
/// the maximum of `n` draws lands on it. Repeated values need no special case:
/// their adjacent probability masses sum to the same value.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the empirical fit computes in f64 and returns a cosine-scale f32; the corpus counts are exactly representable at this scale"
)]
fn expected_empirical_maximum(sorted: &[f32], draws: usize) -> Option<f32> {
    if sorted.is_empty() || draws == 0 {
        return None;
    }
    let count = sorted.len() as f64;
    let exponent = draws as f64;
    let mut prior_cdf_power = 0.0_f64;
    let mut expected = 0.0_f64;
    for (offset, score) in sorted.iter().enumerate() {
        let cdf = (offset + 1) as f64 / count;
        let cdf_power = cdf.powf(exponent);
        expected = f64::from(*score).mul_add(cdf_power - prior_cdf_power, expected);
        prior_cdf_power = cdf_power;
    }
    Some(expected as f32)
}

/// Pool one record's passage scores under a fit computed for the whole query.
fn pool_record(
    scores: &[f32],
    pooling: DensePooling,
    expected_maxima: &std::collections::BTreeMap<usize, f32>,
) -> Option<f32> {
    let maximum = scores.iter().copied().max_by(f32::total_cmp)?;
    match pooling {
        DensePooling::Maximum => Some(maximum),
        DensePooling::MeanTopThree => {
            let mut ordered = scores.to_vec();
            ordered.sort_by(|left, right| right.total_cmp(left));
            let count = ordered.len().min(3);
            let divisor = match count {
                1 => 1.0,
                2 => 2.0,
                3 => 3.0,
                _ => return None,
            };
            Some(ordered.iter().take(count).sum::<f32>() / divisor)
        }
        DensePooling::LengthNormalizedMaximum => expected_maxima
            .get(&scores.len())
            .map(|expected| maximum - expected),
    }
}

/// Failures in building or reading the index.
#[derive(Debug, Error)]
pub enum IndexError {
    /// The database rejected an operation.
    #[error("index database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// The store could not supply something the index needs.
    #[error("record {record}: {source}")]
    Store {
        /// Which record was being indexed.
        record: String,
        /// What the store said.
        source: StoreError,
    },

    /// Stored bytes were not a usable record.
    #[error("record {record}: {source}")]
    Record {
        /// Which record was being indexed.
        record: String,
        /// What was wrong with it.
        source: RecordError,
    },

    /// A record was asked for that the index does not hold.
    #[error("no record {0} in the index")]
    NotFound(String),

    /// The index on disk was built under a different schema.
    #[error(
        "the index at {path} was built under schema version {found}; this build writes \
         version {expected}. The index is derived and disposable: run `kb reindex` to \
         rebuild it from the store."
    )]
    SchemaMismatch {
        /// Where the stale index is.
        path: String,
        /// The version stamped in the file.
        found: u32,
        /// The version this build writes.
        expected: u32,
    },
}

/// Every table, created together. There is no incremental migration path by
/// design — see [`INDEX_SCHEMA_VERSION`].
const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS records (
    record_id     TEXT PRIMARY KEY,
    corpus        TEXT NOT NULL,
    kind          TEXT NOT NULL,
    source        TEXT NOT NULL,
    created       TEXT NOT NULL,
    updated       TEXT NOT NULL,
    -- The address of the raw bytes when the corpus copies them into the
    -- store, and a digest of them when it only references them. Which one it
    -- is follows from `storage`, and no reader has to guess: a reference-only
    -- corpus has no store address to give.
    raw_hash      TEXT NOT NULL,
    stream_hash   TEXT NOT NULL,
    -- The address of the record header blob, and NULL exactly when `storage`
    -- is `reference`: a corpus that puts nothing in the store has no record
    -- blob for the store to address. Fabricating a hash here would make an
    -- absent thing look present.
    record_hash   TEXT,
    normalizer    INTEGER NOT NULL,
    storage       TEXT NOT NULL,
    -- The record's one-line name, derived from the artifact it was made of
    -- rather than stored alongside it. Cached here because naming a hit is
    -- part of every result list and re-reading the raw blob for each would
    -- make a search proportional to the corpus (T029).
    title         TEXT NOT NULL DEFAULT '',
    -- The `#+name:` slug a `[[name]]` link resolves against, or NULL where the
    -- artifact declares none. Separate from the title because a slug is a
    -- handle the author chose and a title is prose that changes.
    name_slug     TEXT,
    -- Provenance (`PLAN-20260923-project-identity` T005): where the work
    -- that produced this record was happening, re-derived from the header's
    -- optional `project`, `project-source`, `remote`, `context`, `harness`,
    -- `model`, `session` and `cwd` lines. NULL on every column exactly when
    -- the record carries none of it, which is every `kb-record/1` blob and
    -- any `kb-record/2` blob a writer chose to assert nothing in.
    project        TEXT,
    project_source TEXT,
    remote         TEXT,
    context        TEXT,
    harness        TEXT,
    model          TEXT,
    session        TEXT,
    cwd            TEXT
);
CREATE INDEX IF NOT EXISTS records_name_slug ON records(name_slug);
CREATE INDEX IF NOT EXISTS records_corpus ON records(corpus);
CREATE INDEX IF NOT EXISTS records_source ON records(source);
CREATE INDEX IF NOT EXISTS records_project ON records(project);
CREATE INDEX IF NOT EXISTS records_context ON records(context);

CREATE TABLE IF NOT EXISTS record_tags (
    record_id TEXT NOT NULL,
    tag       TEXT NOT NULL,
    PRIMARY KEY (record_id, tag)
);
CREATE INDEX IF NOT EXISTS record_tags_tag ON record_tags(tag);

-- The header's repeated `domain:` line. A separate table for the same reason
-- `record_tags` is: a record may carry any number of domains, and the value
-- is re-derivable from the blob (ST-002), never asserted only here.
CREATE TABLE IF NOT EXISTS record_domains (
    record_id TEXT NOT NULL,
    domain    TEXT NOT NULL,
    PRIMARY KEY (record_id, domain)
);
CREATE INDEX IF NOT EXISTS record_domains_domain ON record_domains(domain);

CREATE TABLE IF NOT EXISTS passages (
    passage_id  INTEGER PRIMARY KEY,
    record_id   TEXT NOT NULL,
    stream_hash TEXT NOT NULL,
    level       TEXT NOT NULL,
    span_start  INTEGER NOT NULL,
    span_len    INTEGER NOT NULL,
    -- Named `body` because `passages_fts` is an external-content table and
    -- takes its column names from this one, and `--match body:` is a filter
    -- callers already know.
    body        TEXT NOT NULL,
    -- The record's name, repeated here because `passages_fts` is an
    -- external-content table and reads every indexed column out of this one.
    title       TEXT NOT NULL DEFAULT '',
    -- Whether this passage was mirrored into `passages_fts`. Explicit because
    -- `passages_fts` is an external-content table: it reads text back out of
    -- this one to know which tokens a delete should remove, so deleting a row
    -- it never indexed removes tokens that were never added and corrupts it.
    -- Mail passages are cached here for embedding and reranking and are
    -- deliberately absent from the FTS, which makes partial membership a
    -- property this table has to record rather than infer.
    fts_indexed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS passages_record ON passages(record_id);
-- Dropping orphaned vectors asks, per vector, whether any passage on the same
-- stream contains its span. Without this the question is a full scan of the
-- passage table per vector, which took a rebuild from 1.4 seconds to over two
-- minutes once the corpus had vectors at all.
CREATE INDEX IF NOT EXISTS passages_stream ON passages(stream_hash, span_start, span_len);

CREATE TABLE IF NOT EXISTS authored_links (
    record_id   TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target      TEXT NOT NULL,
    span_start  INTEGER NOT NULL,
    span_len    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS authored_links_target ON authored_links(target);

CREATE TABLE IF NOT EXISTS inferred_links (
    source_record_id TEXT NOT NULL,
    target_kind      TEXT NOT NULL,
    target           TEXT NOT NULL,
    generator        TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    confidence       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    stream_hash TEXT NOT NULL,
    span_start  INTEGER NOT NULL,
    span_len    INTEGER NOT NULL,
    model       TEXT NOT NULL,
    vector      BLOB NOT NULL,
    PRIMARY KEY (stream_hash, span_start, span_len, model)
);

CREATE TABLE IF NOT EXISTS generated_artifacts (
    source_hash       TEXT NOT NULL,
    generator_version TEXT NOT NULL,
    prompt_version    TEXT NOT NULL,
    content           TEXT NOT NULL,
    PRIMARY KEY (source_hash, generator_version, prompt_version)
);

-- Where a referenced message actually is, and what it hashed to when the
-- catalogue last saw it. Separate from `records` because a path is a fact
-- about this machine at this moment, whereas a Message-ID, a folder and a
-- content hash are facts about the message. Holds exactly what the index
-- refers to: the Maildir remains canonical for everything else.
CREATE TABLE IF NOT EXISTS mail_catalogue (
    message_id            TEXT PRIMARY KEY,
    folder                TEXT NOT NULL,
    locator               TEXT NOT NULL,
    content_sha256        TEXT NOT NULL,
    -- Whether the discriminant classified this message as bulk and a
    -- ground-truth question readmitted it. Recorded so a measurement can
    -- state what fraction of the corpus the rule did not select.
    ground_truth_override INTEGER NOT NULL
);

-- Two columns because `--match` has always offered `title:` and `body:`
-- filters, and a reader who has learned them should not lose them when the
-- index behind the query changes (T029). `body` is the passage's own text;
-- `title` is the record's name, repeated on each of its passages so that a
-- filter can reach it from any of them.
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
    title,
    body,
    content='passages',
    content_rowid='passage_id'
);
";

/// Tables emptied by [`Index::truncate`], in an order that leaves no row
/// referring to a vanished one at any point.
const DERIVED_TABLES: [&str; 9] = [
    "passages_fts",
    "mail_catalogue",
    "embeddings",
    "authored_links",
    "inferred_links",
    "record_tags",
    "record_domains",
    "passages",
    "records",
];

/// A passage as the index holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassageRow {
    /// Which record it belongs to.
    pub record_id: String,
    /// The stream its span addresses.
    pub stream_hash: String,
    /// What the span is a passage of.
    pub level: String,
    /// First byte.
    pub span_start: i64,
    /// Length in bytes.
    pub span_len: i64,
    /// The span's text, cached beside the row that addresses it.
    pub text: String,
}

/// One generated artifact and what produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedArtifact {
    /// The canonical content it was generated from.
    pub source_hash: String,
    /// What generated it.
    pub generator_version: String,
    /// Which form and prompt version produced it.
    pub prompt_version: String,
    /// The generated text.
    pub content: String,
}

/// One catalogued message: where its bytes are and what they hashed to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MailCatalogueRow {
    /// The message's identity.
    pub message_id: String,
    /// Folder relative to the Maildir root.
    pub folder: String,
    /// Filename up to the flag separator: stable across flag changes.
    pub locator: String,
    /// SHA-256 over the delivered bytes when the catalogue last saw them.
    pub content_sha256: String,
    /// Whether the discriminant excluded this message and a ground-truth
    /// question readmitted it.
    pub ground_truth_override: bool,
}

/// One record, as the index holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordRow {
    /// Stable identity.
    pub record_id: String,
    /// Which corpus it belongs to.
    pub corpus: String,
    /// What it is.
    pub kind: String,
    /// Its identity in the system it came from, as `scheme:value`.
    pub source: String,
    /// Hash of the bytes as they arrived.
    pub raw_hash: String,
    /// Hash of the normalized stream its passages address.
    pub stream_hash: String,
    /// Hash of the record blob itself, which is what reconciliation compares.
    ///
    /// `None` exactly when [`Self::storage`] is `reference`: a corpus that
    /// puts nothing in the store has no record blob for the store to address,
    /// and the type says so rather than leaving callers to interpret an empty
    /// string.
    pub record_hash: Option<String>,
    /// When the record was first written, as RFC 3339.
    pub created: String,
    /// When it was last written.
    pub updated: String,
    /// The record's one-line name, derived from its artifact. Empty when the
    /// artifact offers nothing to name it with, which is a fact about the
    /// record rather than a lookup failure.
    pub title: String,
    /// How this record's corpus stores its bytes: `copy` or `reference`. A
    /// reference that stops resolving is corruption for the first and an
    /// expected consequence for the second, and the row says which without
    /// consulting a registry that may since have changed.
    pub storage: String,
    /// The project slug the session or write resolved to, if any.
    pub project: Option<String>,
    /// How `project` was resolved (`declared`, `path`, `remote`, `derived`).
    pub project_source: Option<String>,
    /// The working directory's normalized origin remote, if known.
    pub remote: Option<String>,
    /// The credential boundary the session ran under, if known.
    pub context: Option<String>,
    /// The harness that captured the session, if known.
    pub harness: Option<String>,
    /// The model in use when the record was produced, if known.
    pub model: Option<String>,
    /// The harness session id, if known.
    pub session: Option<String>,
    /// The working directory the session ran in, if known.
    pub cwd: Option<String>,
}

/// One project's row in `kb projects`: the slug, how many records assert it,
/// and the most recent one's `created` timestamp.
///
/// Derived entirely from `records.project` and `records.created`
/// (`PLAN-20260923-project-identity` T006) — nothing here is stored only by
/// this query (ST-002).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectCount {
    /// The project slug.
    pub project: String,
    /// How many records assert it.
    pub count: usize,
    /// The `created` timestamp of the most recently created record under
    /// it, as RFC 3339.
    pub newest: String,
}

/// What a rebuild produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RebuildReport {
    /// Records indexed.
    pub records: usize,
    /// Passages derived from them.
    pub passages: usize,
    /// Source-authored links found.
    pub authored_links: usize,
    /// Vectors dropped because the text they described is no longer indexed.
    pub stale_vectors_dropped: usize,
    /// How long it took. Reported because the number decides whether anyone
    /// rebuilds: an index nobody dares drop is authoritative in practice
    /// whatever the invariants say.
    pub elapsed: std::time::Duration,
}

/// How much of the index a rebuild covers.
///
/// Scoping exists so that the cheap derivations can be redone without the
/// expensive ones: re-chunking one corpus should not mean re-reading every
/// other, and editing one note should not mean rebuilding the corpus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    /// Everything the store holds.
    All,
    /// One corpus, by identifier.
    Corpus(String),
    /// One record, by id.
    Record(String),
}

impl Scope {
    /// The ref prefix this scope enumerates. Records are named
    /// `<corpus>/<id>`, so a corpus is a prefix and everything is the empty
    /// prefix; a single record still enumerates broadly and is filtered by
    /// id, because a record's corpus is not known before it is read.
    fn prefix(&self) -> String {
        match self {
            Self::All | Self::Record(_) => String::new(),
            Self::Corpus(corpus) => format!("{corpus}/"),
        }
    }

    /// Whether a record with this header belongs to the scope.
    fn covers(&self, header: &RecordHeader) -> bool {
        match self {
            Self::All => true,
            Self::Corpus(corpus) => header.corpus.as_str() == corpus,
            Self::Record(id) => header.id.as_str() == id,
        }
    }
}

/// The derived index.
#[derive(Debug)]
pub struct Index {
    conn: Connection,
}

impl Index {
    /// Open the index at `path`, creating the schema if it is not there.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the file cannot be opened or the schema
    /// cannot be created.
    pub fn open(path: &Path) -> Result<Self, IndexError> {
        let conn = open_at(path)?;
        let found: u32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        // An index from another schema is refused rather than opened. Opening
        // it would leave `CREATE TABLE IF NOT EXISTS` a no-op over the old
        // shape, and the mismatch would surface later as "no such column"
        // from whichever query needed the missing one first.
        if found != 0 && found != INDEX_SCHEMA_VERSION {
            return Err(IndexError::SchemaMismatch {
                path: path.display().to_string(),
                found,
                expected: INDEX_SCHEMA_VERSION,
            });
        }
        if found == 0 && table_exists(&conn, "records")? {
            return Err(IndexError::SchemaMismatch {
                path: path.display().to_string(),
                found,
                expected: INDEX_SCHEMA_VERSION,
            });
        }
        // An index already at this schema is opened and nothing more. Until
        // T025 every open re-ran the bootstrap, whose `PRAGMA user_version`
        // is a write, so every *read* — each tool call opens the index —
        // needed the write lock: behind a running `kb reindex` it waited out
        // rusqlite's five-second busy timeout and then failed with "database
        // is locked". The matrix measured both the stalls (p95 4.5–5.9s under
        // reindex against 0.4s idle) and the failures (13 of 3,240 queries)
        // before the cause was found here.
        if found == INDEX_SCHEMA_VERSION {
            return Ok(Self { conn });
        }
        Self::bootstrap(conn)
    }

    /// Open for rebuilding, discarding any index built under another schema.
    ///
    /// Rebuilding is the remedy the mismatch error names, so this recreates
    /// rather than refusing: the index holds nothing authoritative, and
    /// refusing here would leave the operator deleting a file by hand to run
    /// the one command whose purpose is to rebuild it.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the file cannot be opened or recreated.
    pub fn open_for_rebuild(path: &Path) -> Result<Self, IndexError> {
        let conn = open_at(path)?;
        let found: u32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if found != INDEX_SCHEMA_VERSION {
            let kept = carry_embeddings(&conn)?;
            let names: Vec<String> = {
                let mut stmt = conn.prepare(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view') \
                     AND name NOT LIKE 'sqlite_%'",
                )?;
                let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
                rows.collect::<Result<Vec<_>, _>>()?
            };
            for name in names {
                conn.execute_batch(&format!("DROP TABLE IF EXISTS \"{name}\""))?;
            }
            let index = Self::bootstrap(conn)?;
            index.restore_embeddings(&kept)?;
            return Ok(index);
        }
        Self::bootstrap(conn)
    }

    /// Put back the vectors carried across a schema change.
    fn restore_embeddings(&self, rows: &[EmbeddingRow]) -> Result<(), IndexError> {
        for row in rows {
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings
                 (stream_hash, span_start, span_len, model, vector)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    row.stream_hash,
                    row.span_start,
                    row.span_len,
                    row.model,
                    row.vector
                ],
            )?;
        }
        Ok(())
    }

    /// Open an index that exists only for the duration of the process.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the schema cannot be created.
    pub fn open_in_memory() -> Result<Self, IndexError> {
        Self::bootstrap(Connection::open_in_memory()?)
    }

    /// Create the schema and stamp its version.
    fn bootstrap(conn: Connection) -> Result<Self, IndexError> {
        conn.execute_batch("PRAGMA journal_mode = WAL;")?;
        conn.execute_batch(SCHEMA)?;
        conn.pragma_update(None, "user_version", INDEX_SCHEMA_VERSION)?;
        Ok(Self { conn })
    }

    /// Empty every derived table.
    ///
    /// The whole index is derived, so this empties all of it. That it is
    /// safe to call is the property [`rebuild`] depends on.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if a table cannot be emptied.
    pub fn truncate(&self) -> Result<(), IndexError> {
        for table in DERIVED_TABLES {
            self.conn.execute(&format!("DELETE FROM {table}"), [])?;
        }
        Ok(())
    }

    /// Remove the derived rows a rebuild of `scope` is about to replace.
    ///
    /// Vectors and generated artifacts are deliberately not removed: they are
    /// keyed by content rather than by row, cost orders of magnitude more to
    /// produce than everything else here, and a re-chunk that lands on the
    /// same spans should keep them. What no longer matches any passage is
    /// swept afterwards by [`Index::drop_orphaned_derivations`].
    fn clear(&self, scope: &Scope) -> Result<(), IndexError> {
        match scope {
            Scope::All => {
                // Everything the store backs, and nothing it does not. A
                // reference-only corpus (mail) is re-derived by its own
                // rebuild in its own transaction; clearing it here too left
                // the index without mail from this commit until that one —
                // several seconds per `kb reindex`, during which every mail
                // query answered from an index presented as complete (T025
                // measured it: mail-dense returned nothing and mail recall
                // fell to zero while a rebuild ran). Only rows that were
                // indexed into FTS are deleted from it: an external-content
                // table answers a row delete by reading that row's text back
                // out of `passages`, and removing tokens that were never
                // added corrupts it.
                self.conn.execute(
                    "DELETE FROM passages_fts WHERE rowid IN (
                         SELECT p.passage_id FROM passages p
                         JOIN records r ON r.record_id = p.record_id
                         WHERE r.storage != 'reference' AND p.fts_indexed = 1)",
                    [],
                )?;
                for table in [
                    "authored_links",
                    "record_tags",
                    "record_domains",
                    "passages",
                ] {
                    self.conn.execute(
                        &format!(
                            "DELETE FROM {table} WHERE record_id IN
                             (SELECT record_id FROM records WHERE storage != 'reference')"
                        ),
                        [],
                    )?;
                }
                self.conn
                    .execute("DELETE FROM records WHERE storage != 'reference'", [])?;
            }
            Scope::Corpus(corpus) => {
                self.conn.execute(
                    "DELETE FROM passages_fts WHERE rowid IN (
                         SELECT p.passage_id FROM passages p
                         JOIN records r ON r.record_id = p.record_id
                         WHERE r.corpus = ?1 AND p.fts_indexed = 1)",
                    [corpus],
                )?;
                for table in [
                    "authored_links",
                    "record_tags",
                    "record_domains",
                    "passages",
                ] {
                    self.conn.execute(
                        &format!(
                            "DELETE FROM {table} WHERE record_id IN
                             (SELECT record_id FROM records WHERE corpus = ?1)"
                        ),
                        [corpus],
                    )?;
                }
                self.conn
                    .execute("DELETE FROM records WHERE corpus = ?1", [corpus])?;
            }
            Scope::Record(id) => {
                self.conn.execute(
                    "DELETE FROM passages_fts WHERE rowid IN
                     (SELECT passage_id FROM passages
                      WHERE record_id = ?1 AND fts_indexed = 1)",
                    [id],
                )?;
                for table in [
                    "authored_links",
                    "record_tags",
                    "record_domains",
                    "passages",
                    "records",
                ] {
                    self.conn
                        .execute(&format!("DELETE FROM {table} WHERE record_id = ?1"), [id])?;
                }
            }
        }
        Ok(())
    }

    /// Forget everything the index derived from one record.
    ///
    /// What reconciliation does with a record the store no longer holds: the
    /// rows are derived from something that is gone, so they cannot be
    /// re-derived and cannot be trusted. Vectors are swept separately, by
    /// [`Index::drop_orphaned_derivations`], because a vector is addressed by
    /// content rather than by record and may still describe a passage some
    /// other record shares.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the deletion fails.
    pub fn forget(&self, record_id: &str) -> Result<(), IndexError> {
        self.clear(&Scope::Record(record_id.to_owned()))
    }

    /// Drop vectors that no longer describe any indexed passage.
    ///
    /// A vector addresses a span of a named stream. When a record's content
    /// changes its stream hash changes with it, so the old vector describes
    /// text that is no longer in the index; keeping it would let a search
    /// match a passage on what it used to say.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the sweep fails.
    pub fn drop_orphaned_derivations(&self) -> Result<usize, IndexError> {
        let removed = self.conn.execute(
            // Containment rather than equality. A passage longer than the
            // model's window is embedded as several spans, so an exact match
            // would call every chunk of every long document an orphan and
            // discard it on the next rebuild -- which it did, taking 4,175 of
            // 11,504 vectors and most of an hour's work the first time this
            // ran against the real corpus. A span reaching past the passage it
            // claims to be part of still addresses text that is not there, so
            // the upper bound stays checked.
            "DELETE FROM embeddings WHERE NOT EXISTS (
                 SELECT 1 FROM passages p
                 WHERE p.stream_hash = embeddings.stream_hash
                   AND embeddings.span_start >= p.span_start
                   AND embeddings.span_start + embeddings.span_len
                       <= p.span_start + p.span_len)",
            [],
        )?;
        Ok(removed)
    }

    /// Store a vector for one passage of one stream, under one model.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the write fails.
    pub fn put_embedding(
        &self,
        stream_hash: &str,
        span_start: i64,
        span_len: i64,
        model: &str,
        vector: &[u8],
    ) -> Result<(), IndexError> {
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings
             (stream_hash, span_start, span_len, model, vector)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![stream_hash, span_start, span_len, model, vector],
        )?;
        Ok(())
    }

    /// The vector for one passage under one model, if there is one.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the query fails.
    pub fn embedding(
        &self,
        stream_hash: &str,
        span_start: i64,
        span_len: i64,
        model: &str,
    ) -> Result<Option<Vec<u8>>, IndexError> {
        let mut statement = self.conn.prepare(
            "SELECT vector FROM embeddings
             WHERE stream_hash = ?1 AND span_start = ?2 AND span_len = ?3 AND model = ?4",
        )?;
        let mut rows =
            statement.query(rusqlite::params![stream_hash, span_start, span_len, model])?;
        match rows.next()? {
            Some(row) => Ok(Some(row.get(0)?)),
            None => Ok(None),
        }
    }

    /// Records ranked by the default pooling of their vectors' cosine scores
    /// against `query_vec`, under `model`, within `corpus`.
    ///
    /// The dense signal over the derived index, and the counterpart of
    /// [`Index::search_text`]. A record is represented by its best-matching
    /// span rather than by an average, because a long thread that answers a
    /// question in one message answers it, and averaging would dilute exactly
    /// the evidence being looked for.
    ///
    /// Vectors are keyed by the stream they were computed from, and a record
    /// names its stream, so a vector is attributed to a record by equality of
    /// `stream_hash` — every record in the corpus, then every vector under its
    /// stream. Until T025 the join went through `passages` by span
    /// containment and applied `DISTINCT` to rows carrying the 4KB vector
    /// blob; against the real kb corpus that cost 855ms a query, 460ms of it
    /// in a temp b-tree over the blobs and the rest in the fan-out, while the
    /// stream join returns the identical key set in about 90ms. Containment
    /// remains the relation [`Index::drop_orphaned_derivations`] uses to decide
    /// a vector still describes an indexed passage; ranking does not need to
    /// re-check it, because the sweep runs inside the same transaction as
    /// every rebuild that could orphan one.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn rank_by_embedding(
        &self,
        query_vec: &[f32],
        model: &str,
        corpus: &str,
    ) -> Result<Vec<(String, f32)>, IndexError> {
        self.rank_by_embedding_with_pooling(
            query_vec,
            model,
            corpus,
            DEFAULT_DENSE_POOLING,
            None,
            None,
        )
    }

    /// Records ranked after `pooling` combines each record's passage scores.
    ///
    /// Length normalization fits its null expectation to the passage-score
    /// distribution produced by this query over this corpus. For a record with
    /// `n` passage scores, the correction is the exact expected maximum of `n`
    /// independent draws from that empirical distribution. The fit therefore
    /// moves with the embedding model and query shape rather than assuming a
    /// universal penalty.
    ///
    /// `project` and `context` (`PLAN-20260923-project-identity` T006)
    /// narrow ranking to records asserting that provenance; `None` applies
    /// no filter.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn rank_by_embedding_with_pooling(
        &self,
        query_vec: &[f32],
        model: &str,
        corpus: &str,
        pooling: DensePooling,
        project: Option<&str>,
        context: Option<&str>,
    ) -> Result<Vec<(String, f32)>, IndexError> {
        let mut statement = self.conn.prepare(RANK_BY_EMBEDDING_SQL)?;
        let rows = statement
            .query_map(rusqlite::params![model, corpus, project, context], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
            })?;
        let mut by_record: std::collections::BTreeMap<String, Vec<f32>> =
            std::collections::BTreeMap::new();
        let mut population = Vec::new();
        for row in rows {
            let (record_id, blob) = row?;
            if let Ok(vector) = crate::embedding::decode_embedding(&blob) {
                let score = crate::storage::cosine_similarity(query_vec, &vector);
                population.push(score);
                by_record.entry(record_id).or_default().push(score);
            }
        }
        population.sort_by(f32::total_cmp);
        let expected_maxima = expected_maxima_for(&by_record, &population, pooling);
        let mut ranked: Vec<(String, f32)> = by_record
            .into_iter()
            .filter_map(|(record_id, scores)| {
                pool_record(&scores, pooling, &expected_maxima).map(|score| (record_id, score))
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(ranked)
    }

    /// What the catalogue holds for one referenced message, if anything.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn mail_catalogue(&self, message_id: &str) -> Result<Option<MailCatalogueRow>, IndexError> {
        let found = self
            .conn
            .query_row(
                "SELECT message_id, folder, locator, content_sha256, ground_truth_override
                 FROM mail_catalogue WHERE message_id = ?1",
                [message_id],
                |r| {
                    Ok(MailCatalogueRow {
                        message_id: r.get(0)?,
                        folder: r.get(1)?,
                        locator: r.get(2)?,
                        content_sha256: r.get(3)?,
                        ground_truth_override: r.get::<_, i64>(4)? != 0,
                    })
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => IndexError::NotFound(message_id.to_owned()),
                other => IndexError::Database(other),
            });
        match found {
            Ok(row) => Ok(Some(row)),
            Err(IndexError::NotFound(_)) => Ok(None),
            Err(other) => Err(other),
        }
    }

    /// Every record id in one corpus, in id order.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn records_in(&self, corpus: &str) -> Result<Vec<String>, IndexError> {
        let mut statement = self
            .conn
            .prepare("SELECT record_id FROM records WHERE corpus = ?1 ORDER BY record_id")?;
        let rows = statement.query_map([corpus], |r| r.get(0))?;
        Ok(rows.collect::<Result<Vec<String>, _>>()?)
    }

    /// Every record id of one kind, in id order.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn records_of_kind(&self, kind: &str) -> Result<Vec<String>, IndexError> {
        let mut statement = self
            .conn
            .prepare("SELECT record_id FROM records WHERE kind = ?1 ORDER BY record_id")?;
        let rows = statement.query_map([kind], |r| r.get(0))?;
        Ok(rows.collect::<Result<Vec<String>, _>>()?)
    }

    /// The canonical content hash of one record, or `None` if it is not held.
    ///
    /// This is what a generated artifact is keyed on: it changes when the
    /// record's text changes and not when the index is merely rebuilt.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the query fails.
    pub fn stream_hash_of(&self, record_id: &str) -> Result<Option<String>, IndexError> {
        let mut statement = self
            .conn
            .prepare("SELECT stream_hash FROM records WHERE record_id = ?1")?;
        let mut rows = statement.query([record_id])?;
        match rows.next()? {
            Some(row) => Ok(Some(row.get(0)?)),
            None => Ok(None),
        }
    }

    /// A record's text, as a generator should see it.
    ///
    /// Assembled from the cached passages in stream order, which for a record
    /// whose passages tile it reconstructs the normalized payload. The store
    /// is not consulted: generation runs against what the index holds, and a
    /// reference-only corpus has nothing in the store to consult.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the query fails.
    pub fn record_text(&self, record_id: &str) -> Result<String, IndexError> {
        let passages = self.passages(record_id)?;
        // Only the finest level, so a record whose coarser passages overlap
        // its finer ones does not present the same words to the model twice.
        let finest = passages
            .iter()
            .filter(|p| p.level != "thread")
            .map(|p| p.text.as_str())
            .collect::<Vec<_>>();
        Ok(finest.join("\n"))
    }

    /// A stored artifact for one key, if it exists.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the query fails.
    pub fn generated(
        &self,
        source_hash: &str,
        generator_version: &str,
        prompt_version: &str,
    ) -> Result<Option<String>, IndexError> {
        let mut statement = self.conn.prepare(
            "SELECT content FROM generated_artifacts
             WHERE source_hash = ?1 AND generator_version = ?2 AND prompt_version = ?3",
        )?;
        let mut rows = statement.query(rusqlite::params![
            source_hash,
            generator_version,
            prompt_version
        ])?;
        match rows.next()? {
            Some(row) => Ok(Some(row.get(0)?)),
            None => Ok(None),
        }
    }

    /// Store one generated artifact under its key.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the write fails.
    pub fn put_generated(
        &self,
        source_hash: &str,
        generator_version: &str,
        prompt_version: &str,
        content: &str,
    ) -> Result<(), IndexError> {
        self.conn.execute(
            "INSERT OR REPLACE INTO generated_artifacts
             (source_hash, generator_version, prompt_version, content)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![source_hash, generator_version, prompt_version, content],
        )?;
        Ok(())
    }

    /// Every artifact generated for one record, whatever produced it.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the query fails.
    pub fn generated_for(&self, record_id: &str) -> Result<Vec<GeneratedArtifact>, IndexError> {
        let mut statement = self.conn.prepare(
            "SELECT g.source_hash, g.generator_version, g.prompt_version, g.content
             FROM generated_artifacts g
             JOIN records r ON r.stream_hash = g.source_hash
             WHERE r.record_id = ?1
             ORDER BY g.generator_version, g.prompt_version",
        )?;
        let rows = statement.query_map([record_id], |r| {
            Ok(GeneratedArtifact {
                source_hash: r.get(0)?,
                generator_version: r.get(1)?,
                prompt_version: r.get(2)?,
                content: r.get(3)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<GeneratedArtifact>, _>>()?)
    }

    /// Every passage in one corpus, in record and stream order.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the query fails.
    pub fn passages_in(&self, corpus: &str) -> Result<Vec<PassageRow>, IndexError> {
        let mut statement = self.conn.prepare(
            "SELECT p.record_id, p.stream_hash, p.level, p.span_start, p.span_len, p.body
             FROM passages p JOIN records r ON r.record_id = p.record_id
             WHERE r.corpus = ?1
             ORDER BY p.record_id, p.span_start",
        )?;
        let rows = statement.query_map([corpus], |r| {
            Ok(PassageRow {
                record_id: r.get(0)?,
                stream_hash: r.get(1)?,
                level: r.get(2)?,
                span_start: r.get(3)?,
                span_len: r.get(4)?,
                text: r.get(5)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<PassageRow>, _>>()?)
    }

    /// The passages of a record, in stream order.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the query fails.
    pub fn passages(&self, record_id: &str) -> Result<Vec<PassageRow>, IndexError> {
        let mut statement = self.conn.prepare(
            "SELECT record_id, stream_hash, level, span_start, span_len, body
             FROM passages WHERE record_id = ?1 ORDER BY span_start",
        )?;
        let rows = statement
            .query_map([record_id], |r| {
                Ok(PassageRow {
                    record_id: r.get(0)?,
                    stream_hash: r.get(1)?,
                    level: r.get(2)?,
                    span_start: r.get(3)?,
                    span_len: r.get(4)?,
                    text: r.get(5)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// The row for `record_id`.
    ///
    /// # Errors
    ///
    /// [`IndexError::NotFound`] if no such record is indexed, or
    /// [`IndexError::Database`] if the query fails.
    pub fn record(&self, record_id: &str) -> Result<RecordRow, IndexError> {
        self.conn
            .query_row(
                "SELECT record_id, corpus, kind, source, raw_hash, stream_hash, record_hash,
                        storage, title, created, updated,
                        project, project_source, remote, context, harness, model, session, cwd
                 FROM records WHERE record_id = ?1",
                [record_id],
                |r| {
                    Ok(RecordRow {
                        record_id: r.get(0)?,
                        corpus: r.get(1)?,
                        kind: r.get(2)?,
                        source: r.get(3)?,
                        raw_hash: r.get(4)?,
                        stream_hash: r.get(5)?,
                        record_hash: r.get(6)?,
                        storage: r.get(7)?,
                        title: r.get(8)?,
                        created: r.get(9)?,
                        updated: r.get(10)?,
                        project: r.get(11)?,
                        project_source: r.get(12)?,
                        remote: r.get(13)?,
                        context: r.get(14)?,
                        harness: r.get(15)?,
                        model: r.get(16)?,
                        session: r.get(17)?,
                        cwd: r.get(18)?,
                    })
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => IndexError::NotFound(record_id.to_owned()),
                other => IndexError::Database(other),
            })
    }

    /// Record ids whose passages match `query`, best first, de-duplicated.
    ///
    /// Lexical retrieval over the cached passage text. One of the signals the
    /// T026 planner will fuse rather than the whole of retrieval.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the query is malformed or fails.
    pub fn search_text(&self, query: &str) -> Result<Vec<String>, IndexError> {
        let mut statement = self.conn.prepare(
            "SELECT p.record_id, MIN(rank) AS best
             FROM passages_fts f
             JOIN passages p ON p.passage_id = f.rowid
             WHERE passages_fts MATCH ?1
             GROUP BY p.record_id
             ORDER BY best",
        )?;
        let ids = statement
            .query_map([query], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ids)
    }

    /// Name-links from one record that resolve to no record in the index.
    ///
    /// A `[[name]]` reference is resolved against record names rather than
    /// identifiers, so writing one that matches nothing is the commonest way
    /// to author a dangling reference — and the only moment it is cheap to
    /// fix is the moment it is written.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn unresolved_names(&self, record_id: &str) -> Result<Vec<String>, IndexError> {
        let mut statement = self.conn.prepare(
            "SELECT DISTINCT l.target
             FROM authored_links l
             WHERE l.record_id = ?1
               AND l.target_kind = 'name'
               AND NOT EXISTS (SELECT 1 FROM records r
                                WHERE r.record_id = l.target OR r.name_slug = l.target)
             ORDER BY l.target",
        )?;
        let names = statement
            .query_map([record_id], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(names)
    }

    /// Where this index lives, when it is a file.
    ///
    /// Exists for callers that must hand a `'static` closure a way to reopen
    /// it — `kb prompt`'s template callables, which cannot borrow.
    #[must_use]
    pub fn path(&self) -> Option<String> {
        self.conn.path().map(ToOwned::to_owned)
    }

    /// The tags filed against one record, in stored order.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn tags_of(&self, record_id: &str) -> Result<Vec<String>, IndexError> {
        let mut statement = self
            .conn
            .prepare("SELECT tag FROM record_tags WHERE record_id = ?1 ORDER BY tag")?;
        let tags = statement
            .query_map([record_id], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(tags)
    }

    /// The domains filed against one record's provenance, in stored order.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn domains_of(&self, record_id: &str) -> Result<Vec<String>, IndexError> {
        let mut statement = self
            .conn
            .prepare("SELECT domain FROM record_domains WHERE record_id = ?1 ORDER BY domain")?;
        let domains = statement
            .query_map([record_id], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(domains)
    }

    /// Every project slug carried by at least one record, with its record
    /// count and its newest record's `created` timestamp, most recent
    /// project first (`PLAN-20260923-project-identity` T006, `kb projects`).
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn project_counts(&self) -> Result<Vec<ProjectCount>, IndexError> {
        let mut statement = self.conn.prepare(
            "SELECT project, COUNT(*), MAX(created)
             FROM records
             WHERE project IS NOT NULL
             GROUP BY project
             ORDER BY MAX(created) DESC, project",
        )?;
        let rows = statement
            .query_map([], |row| {
                Ok(ProjectCount {
                    project: row.get(0)?,
                    count: row.get::<_, i64>(1)?.try_into().unwrap_or(0),
                    newest: row.get(2)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// How many records carry no project.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn records_without_project(&self) -> Result<usize, IndexError> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM records WHERE project IS NULL",
            [],
            |row| row.get(0),
        )?;
        Ok(count.try_into().unwrap_or(0))
    }

    /// Record ids in `corpus` whose passages match `expression`, best first.
    ///
    /// The corpus filter is the difference between this and
    /// [`Index::search_text`], and it is not optional for a served query: one
    /// index holds every corpus since T029, so an unfiltered match answers a
    /// question about notes with a mail message.
    ///
    /// `expression` is an FTS5 expression, already built by
    /// [`crate::storage::build_fts_query`] under the caller's match mode.
    /// Taking the expression rather than the user's words keeps the rewriting
    /// — and therefore what a query means — in one place while the evaluation
    /// moves here.
    ///
    /// `project` and `context` (`PLAN-20260923-project-identity` T006)
    /// narrow the match to records asserting that provenance; `None` applies
    /// no filter, so a search with neither set returns exactly what it
    /// returned before those columns existed.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the expression is malformed or the query
    /// fails.
    pub fn search_text_in(
        &self,
        corpus: &str,
        expression: &str,
        project: Option<&str>,
        context: Option<&str>,
    ) -> Result<Vec<String>, IndexError> {
        let mut statement = self.conn.prepare(SEARCH_TEXT_IN_SQL)?;
        let ids = statement
            .query_map(
                rusqlite::params![expression, corpus, project, context],
                |row| row.get::<_, String>(0),
            )?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ids)
    }

    /// The spans of one record that have a vector under `model`.
    ///
    /// Ordered by position, so a caller sees the record's own layout rather
    /// than the table's.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn embedded_spans(
        &self,
        record_id: &str,
        model: &str,
    ) -> Result<Vec<(i64, i64)>, IndexError> {
        let mut statement = self.conn.prepare(
            "SELECT DISTINCT e.span_start, e.span_len
             FROM embeddings e
             JOIN passages p
               ON p.stream_hash = e.stream_hash
              AND e.span_start >= p.span_start
              AND e.span_start + e.span_len <= p.span_start + p.span_len
             WHERE p.record_id = ?1 AND e.model = ?2
             ORDER BY e.span_start",
        )?;
        let rows = statement
            .query_map(rusqlite::params![record_id, model], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// The model one record's vectors were computed under, if it has any.
    ///
    /// The fallback for a caller that has configured no model: a record
    /// embedded under some model is still comparable with everything else
    /// embedded under that model, and refusing to answer because the
    /// environment is quiet would make a read verb depend on a daemon it never
    /// contacts.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn model_of(&self, record_id: &str) -> Result<Option<String>, IndexError> {
        let mut statement = self.conn.prepare(
            "SELECT e.model
             FROM embeddings e
             JOIN passages p ON p.stream_hash = e.stream_hash
             WHERE p.record_id = ?1
             ORDER BY e.model
             LIMIT 1",
        )?;
        let found = statement
            .query_map([record_id], |row| row.get::<_, String>(0))?
            .next()
            .transpose()?;
        Ok(found)
    }

    /// The records most like `record_id`, by the vectors of its passages.
    ///
    /// A record is represented by its best-matching span in both directions.
    /// This neighbour operation deliberately retains pairwise maximum
    /// similarity: unlike query retrieval, it compares all passages of two
    /// known records, and T031 did not measure a replacement for that relation.
    ///
    /// Returns `Ok(None)` when the record has no vector under `model` — a
    /// different fact from having no neighbours, and one with a different
    /// remedy.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn similar_to(
        &self,
        record_id: &str,
        model: &str,
        limit: usize,
    ) -> Result<Option<Vec<(String, f32)>>, IndexError> {
        let mut vectors = self.conn.prepare(
            "SELECT e.vector
             FROM embeddings e
             JOIN passages p ON p.stream_hash = e.stream_hash
             WHERE p.record_id = ?1 AND e.model = ?2",
        )?;
        let mine: Vec<Vec<f32>> = vectors
            .query_map(rusqlite::params![record_id, model], |row| {
                row.get::<_, Vec<u8>>(0)
            })?
            .filter_map(std::result::Result::ok)
            .filter_map(|blob| crate::embedding::decode_embedding(&blob).ok())
            .collect();
        if mine.is_empty() {
            return Ok(None);
        }
        let mut best: std::collections::BTreeMap<String, f32> = std::collections::BTreeMap::new();
        let mut others = self.conn.prepare(
            "SELECT p.record_id, e.vector
             FROM embeddings e
             JOIN passages p ON p.stream_hash = e.stream_hash
             WHERE e.model = ?1 AND p.record_id != ?2",
        )?;
        let rows = others.query_map(rusqlite::params![model, record_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        for row in rows {
            let (id, blob) = row?;
            let Ok(theirs) = crate::embedding::decode_embedding(&blob) else {
                continue;
            };
            for ours in &mine {
                let score = crate::storage::cosine_similarity(ours, &theirs);
                best.entry(id.clone())
                    .and_modify(|current| *current = current.max(score))
                    .or_insert(score);
            }
        }
        let mut ranked: Vec<(String, f32)> = best.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(limit);
        Ok(Some(ranked))
    }

    /// One record's links, outgoing and incoming.
    ///
    /// A link resolves when its target names a record the index holds, by
    /// identifier or by title; an unresolved name-link is reported with the
    /// name it was written with, since that is what has to be corrected. URL
    /// targets are left out: they are references to the world rather than
    /// edges in the graph, and nothing this answers is a question about them.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn links_of(&self, record_id: &str) -> Result<Neighborhood, IndexError> {
        let mut outgoing = self.conn.prepare(
            "SELECT l.target_kind, l.target,
                    (SELECT r.record_id FROM records r
                      WHERE r.record_id = l.target OR r.name_slug = l.target LIMIT 1)
             FROM authored_links l
             WHERE l.record_id = ?1 AND l.target_kind IN ('record', 'name')
             ORDER BY l.span_start",
        )?;
        let out = outgoing
            .query_map([record_id], |row| {
                Ok(link_row(
                    record_id,
                    &row.get::<_, String>(0)?,
                    row.get(1)?,
                    row.get(2)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        let mut incoming = self.conn.prepare(
            "SELECT l.record_id, l.target_kind, l.target
             FROM authored_links l
             JOIN records r ON r.record_id = ?1
             WHERE l.target_kind IN ('record', 'name')
               AND (l.target = r.record_id OR l.target = r.name_slug)
             ORDER BY l.record_id",
        )?;
        let inc = incoming
            .query_map([record_id], |row| {
                let source: String = row.get(0)?;
                let kind: String = row.get(1)?;
                let target: String = row.get(2)?;
                Ok(link_row(&source, &kind, target, Some(record_id.to_owned())))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Neighborhood {
            outgoing: out,
            incoming: inc,
        })
    }

    /// Every name-link in the corpus that resolves to no record.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn broken_links(&self) -> Result<Vec<IndexLinkRow>, IndexError> {
        let mut statement = self.conn.prepare(
            "SELECT DISTINCT l.record_id, l.target
             FROM authored_links l
             WHERE l.target_kind = 'name'
               AND NOT EXISTS (SELECT 1 FROM records r
                                WHERE r.record_id = l.target OR r.name_slug = l.target)
             ORDER BY l.record_id, l.target",
        )?;
        let rows = statement
            .query_map([], |row| {
                Ok(link_row(
                    &row.get::<_, String>(0)?,
                    "name",
                    row.get(1)?,
                    None,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Records in `corpus` that nothing links to.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn orphans(&self, corpus: &str) -> Result<Vec<(String, String)>, IndexError> {
        let mut statement = self.conn.prepare(
            "SELECT r.record_id, r.title
             FROM records r
             WHERE r.corpus = ?1
               AND NOT EXISTS (SELECT 1 FROM authored_links l
                                WHERE l.target = r.record_id OR l.target = r.name_slug)
             ORDER BY r.record_id",
        )?;
        let rows = statement
            .query_map([corpus], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// The most linked-to records in `corpus`, with their in-degree.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn hubs(&self, corpus: &str, limit: usize) -> Result<Vec<Hub>, IndexError> {
        let mut statement = self.conn.prepare(
            "SELECT r.record_id, r.title, COUNT(*) AS degree
             FROM authored_links l
             JOIN records r ON r.record_id = l.target OR r.name_slug = l.target
             WHERE r.corpus = ?1
             GROUP BY r.record_id
             ORDER BY degree DESC, r.record_id
             LIMIT ?2",
        )?;
        let rows = statement
            .query_map(
                rusqlite::params![corpus, i64::try_from(limit).unwrap_or(i64::MAX)],
                |row| {
                    Ok(Hub {
                        id: row.get(0)?,
                        title: row.get(1)?,
                        in_degree: row.get::<_, i64>(2)?,
                    })
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// The most recently updated records in `corpus`, newest first.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn recent(&self, corpus: &str, limit: usize) -> Result<Vec<(String, String)>, IndexError> {
        let mut statement = self.conn.prepare(
            "SELECT record_id, title FROM records
             WHERE corpus = ?1 ORDER BY updated DESC, record_id LIMIT ?2",
        )?;
        let rows = statement
            .query_map(
                rusqlite::params![corpus, i64::try_from(limit).unwrap_or(i64::MAX)],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Records filed under `tag`, with their titles.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn records_tagged(&self, tag: &str) -> Result<Vec<(String, String)>, IndexError> {
        let mut statement = self.conn.prepare(
            "SELECT r.record_id, r.title
             FROM record_tags t JOIN records r ON r.record_id = t.record_id
             WHERE t.tag = ?1
             ORDER BY r.record_id",
        )?;
        let rows = statement
            .query_map([tag], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Every tag in use, with how many records carry it.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if the index cannot be read.
    pub fn tag_counts(&self) -> Result<Vec<(String, i64)>, IndexError> {
        let mut statement = self.conn.prepare(
            "SELECT tag, COUNT(*) AS uses FROM record_tags
             GROUP BY tag ORDER BY uses DESC, tag",
        )?;
        let rows = statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// A stable textual dump of every derived row.
    ///
    /// Exists so that "rebuilding produces the same index" is a claim a test
    /// can check in full, rather than a spot check of the columns whoever
    /// wrote the test happened to think of.
    ///
    /// # Errors
    ///
    /// [`IndexError::Database`] if a table cannot be read.
    pub fn dump(&self) -> Result<String, IndexError> {
        let mut out = String::new();
        for (table, columns) in [
            (
                "records",
                "record_id, corpus, kind, source, created, updated, raw_hash, stream_hash, \
                 record_hash, normalizer, storage, project, project_source, remote, context, \
                 harness, model, session, cwd",
            ),
            ("record_tags", "record_id, tag"),
            ("record_domains", "record_id, domain"),
            (
                "passages",
                "record_id, stream_hash, level, span_start, span_len, body",
            ),
            (
                "authored_links",
                "record_id, target_kind, target, span_start, span_len",
            ),
            (
                "inferred_links",
                "source_record_id, target_kind, target, generator, model_version, confidence",
            ),
            ("embeddings", "stream_hash, span_start, span_len, model"),
            (
                "generated_artifacts",
                "source_hash, generator_version, prompt_version, content",
            ),
        ] {
            let sql = format!("SELECT {columns} FROM {table} ORDER BY {columns}");
            let mut statement = self.conn.prepare(&sql)?;
            let column_count = statement.column_count();
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                let mut line = String::from(table);
                for i in 0..column_count {
                    line.push('\t');
                    line.push_str(
                        &row.get::<_, rusqlite::types::Value>(i)
                            .map_or_else(|_| "?".to_owned(), display_value),
                    );
                }
                out.push_str(&line);
                out.push('\n');
            }
        }
        Ok(out)
    }
}

/// Open the index file, creating the directory it lives in.
///
/// A read verb on a machine that has never built an index should report an
/// empty corpus, not fail because `~/.local/share/kb` does not exist yet: the
/// index is derived and disposable, and its absence is a state to be
/// described rather than an error to be raised.
fn open_at(path: &Path) -> Result<Connection, IndexError> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| {
            IndexError::Database(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CANTOPEN),
                Some(format!("{}: {e}", parent.display())),
            ))
        })?;
    }
    Ok(Connection::open(path)?)
}

/// A record's links in both directions.
#[derive(Debug, Clone, Default)]
pub struct Neighborhood {
    /// Links this record makes.
    pub outgoing: Vec<IndexLinkRow>,
    /// Links other records make to it.
    pub incoming: Vec<IndexLinkRow>,
}

/// One edge of the link graph, as the index holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexLinkRow {
    /// The record the link is written in.
    pub source_id: String,
    /// `id` when the link addresses a record directly, `name` when it names
    /// one. The distinction is what tells a reader whether an unresolved link
    /// is a typo in a name or a reference to something deleted.
    pub link_type: String,
    /// The record the link resolves to, if any.
    pub target_id: Option<String>,
    /// The name a name-link was written with.
    pub target_slug: Option<String>,
}

/// A record and how many links point at it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hub {
    /// The record's identifier.
    pub id: String,
    /// Its name.
    pub title: String,
    /// How many links point at it.
    pub in_degree: i64,
}

/// Assemble one link row from the columns the index stores.
fn link_row(source: &str, kind: &str, target: String, resolved: Option<String>) -> IndexLinkRow {
    if kind == "record" {
        IndexLinkRow {
            source_id: source.to_owned(),
            link_type: "id".to_owned(),
            target_id: resolved.or(Some(target)),
            target_slug: None,
        }
    } else {
        IndexLinkRow {
            source_id: source.to_owned(),
            link_type: "name".to_owned(),
            target_id: resolved,
            target_slug: Some(target),
        }
    }
}

/// One stored vector, as carried across a schema change.
struct EmbeddingRow {
    stream_hash: String,
    span_start: i64,
    span_len: i64,
    model: String,
    vector: Vec<u8>,
}

/// Read every vector out of an index about to be recreated under a new
/// schema.
///
/// **The one table a rebuild does not discard.** Everything else in the index
/// is re-derived from the store in seconds, which is what makes the index
/// disposable (ST-004). A vector is not: it is derived from an embedding
/// endpoint over hours, and its key — the stream it was computed from, the
/// span within that stream, and the model — is content-addressed, so no
/// change to any other table's columns can invalidate it. Recomputing tens of
/// thousands of them because a column was added elsewhere is a cost the
/// disposability argument never claimed to justify.
///
/// Carried only when the table's own shape is what this schema expects. A
/// change to the vector encoding or the key must not be papered over by
/// reloading rows that meant something else, so an unrecognised shape is
/// discarded and recomputed — the behaviour every table had before this.
fn carry_embeddings(conn: &Connection) -> Result<Vec<EmbeddingRow>, IndexError> {
    let columns: Vec<String> = {
        let Ok(mut stmt) = conn.prepare("SELECT name FROM pragma_table_info('embeddings')") else {
            return Ok(Vec::new());
        };
        let Ok(rows) = stmt.query_map([], |r| r.get::<_, String>(0)) else {
            return Ok(Vec::new());
        };
        rows.collect::<Result<Vec<_>, _>>()?
    };
    if columns != EMBEDDING_COLUMNS {
        return Ok(Vec::new());
    }
    let mut stmt =
        conn.prepare("SELECT stream_hash, span_start, span_len, model, vector FROM embeddings")?;
    let rows = stmt.query_map([], |r| {
        Ok(EmbeddingRow {
            stream_hash: r.get(0)?,
            span_start: r.get(1)?,
            span_len: r.get(2)?,
            model: r.get(3)?,
            vector: r.get(4)?,
        })
    })?;
    Ok(rows.collect::<Result<Vec<_>, _>>()?)
}

/// The `embeddings` columns this schema knows how to carry.
const EMBEDDING_COLUMNS: [&str; 5] = ["stream_hash", "span_start", "span_len", "model", "vector"];

/// Rebuild the index from the store, replacing whatever was there.
///
/// **A full rebuild does not sweep orphaned vectors; its caller must.** The
/// store holds the corpora it holds, and a corpus that only references its
/// bytes — mail — is re-derived by a separate pass afterwards. Sweeping here
/// would run while those rows are absent and delete every vector describing
/// them, which is hours of embedding for a command whose whole claim is that
/// re-derivation is routine (ST-004). A scoped rebuild sweeps, because
/// nothing else is mid-flight.
///
/// This walks the store rather than consulting the index about what it holds,
/// which is what makes the result a projection rather than an accumulation.
///
/// # Errors
///
/// [`IndexError::Store`] if a record or the stream it names cannot be read,
/// [`IndexError::Record`] if stored bytes are not a usable record, and
/// [`IndexError::Database`] if a write fails.
pub fn rebuild(
    store: &impl BlobStore,
    index: &Index,
    scope: &Scope,
) -> Result<RebuildReport, IndexError> {
    let started = std::time::Instant::now();
    let mut report = RebuildReport::default();
    let names = store
        .list_refs(&scope.prefix())
        .map_err(|e| IndexError::Store {
            record: "<enumerating>".to_owned(),
            source: e,
        })?;

    // One transaction for the whole rebuild. An interrupted rebuild rolls
    // back to the previous index rather than leaving a partial one that
    // looks complete -- which is the failure the invariant forbids, and the
    // one that would be hardest to notice.
    index.conn.execute_batch("BEGIN IMMEDIATE")?;
    let outcome = rebuild_within(store, index, scope, &names, &mut report);
    if outcome.is_err() {
        index.conn.execute_batch("ROLLBACK")?;
        outcome?;
    }
    index.conn.execute_batch("COMMIT")?;
    report.elapsed = started.elapsed();
    Ok(report)
}

/// The body of a rebuild, inside the caller's transaction.
fn rebuild_within(
    store: &impl BlobStore,
    index: &Index,
    scope: &Scope,
    names: &[crate::store::RefName],
    report: &mut RebuildReport,
) -> Result<(), IndexError> {
    index.clear(scope)?;
    for name in names {
        let label = name.as_str().to_owned();
        let hash = store.read_ref(name).map_err(|e| IndexError::Store {
            record: label.clone(),
            source: e,
        })?;
        let blob = store.get(&hash).map_err(|e| IndexError::Store {
            record: label.clone(),
            source: e,
        })?;
        let header = RecordHeader::parse(&blob).map_err(|e| IndexError::Record {
            record: label.clone(),
            source: e,
        })?;
        if !scope.covers(&header) {
            continue;
        }
        index_one(store, index, &header, &hash, report)?;
    }
    // The orphan sweep is the caller's, not this walk's. A full rebuild
    // re-derives the store-backed corpora here and the referenced ones
    // afterwards; sweeping now would call every vector of a corpus that has
    // not been rebuilt yet an orphan and delete it — which is how a routine
    // `kb reindex` silently cost the mail corpus its whole embedding run.
    if !matches!(scope, Scope::All) {
        report.stale_vectors_dropped = index.drop_orphaned_derivations()?;
    }
    Ok(())
}

/// Re-derive the index rows of named records, leaving every other record
/// alone.
///
/// The repair half of reconciliation. A full rebuild would do the same job and
/// is a single command (ST-004), but it re-derives 3,910 records to fix one —
/// and a reconciliation that costs a full rebuild is a reconciliation nobody
/// schedules. Records are cleared and re-derived individually, inside one
/// transaction, so an interruption leaves the index as it was rather than
/// half-repaired.
///
/// Ids naming no ref in the store are reported back rather than skipped
/// silently: they are the drift `fsck` cannot repair, and treating them as
/// nothing to do would make a missing record look reconciled.
///
/// # Errors
///
/// [`IndexError::Store`] if the store cannot be enumerated or read,
/// [`IndexError::Record`] if a record blob will not parse, and
/// [`IndexError::Database`] if the index cannot be written.
pub fn reindex_records(
    store: &impl BlobStore,
    index: &Index,
    ids: &std::collections::BTreeSet<String>,
) -> Result<RebuildReport, IndexError> {
    let started = std::time::Instant::now();
    let mut report = RebuildReport::default();
    if ids.is_empty() {
        report.elapsed = started.elapsed();
        return Ok(report);
    }
    let names = store.list_refs("").map_err(|e| IndexError::Store {
        record: "<enumerating>".to_owned(),
        source: e,
    })?;
    index.conn.execute_batch("BEGIN IMMEDIATE")?;
    let outcome = reindex_records_within(store, index, ids, &names, &mut report);
    if outcome.is_err() {
        index.conn.execute_batch("ROLLBACK")?;
        outcome?;
    }
    index.conn.execute_batch("COMMIT")?;
    report.elapsed = started.elapsed();
    Ok(report)
}

fn reindex_records_within(
    store: &impl BlobStore,
    index: &Index,
    ids: &std::collections::BTreeSet<String>,
    names: &[crate::store::RefName],
    report: &mut RebuildReport,
) -> Result<(), IndexError> {
    for name in names {
        let label = name.as_str().to_owned();
        let hash = store.read_ref(name).map_err(|e| IndexError::Store {
            record: label.clone(),
            source: e,
        })?;
        let blob = store.get(&hash).map_err(|e| IndexError::Store {
            record: label.clone(),
            source: e,
        })?;
        let header = RecordHeader::parse(&blob).map_err(|e| IndexError::Record {
            record: label.clone(),
            source: e,
        })?;
        if !ids.contains(header.id.as_str()) {
            continue;
        }
        index.clear(&Scope::Record(header.id.as_str().to_owned()))?;
        index_one(store, index, &header, &hash, report)?;
    }
    report.stale_vectors_dropped = index.drop_orphaned_derivations()?;
    Ok(())
}

/// Every record the index holds that came from the store, with the record
/// blob's address the index recorded for it.
///
/// Reference-only corpora are excluded by construction rather than by name: a
/// corpus that puts nothing in the store has no record blob to address, so its
/// `record_hash` is NULL and it has no ref for a store survey to compare
/// against. Including them would report the whole mail corpus as orphaned.
///
/// # Errors
///
/// [`IndexError::Database`] if the index cannot be read.
pub fn stored_record_hashes(
    index: &Index,
) -> Result<std::collections::BTreeMap<String, String>, IndexError> {
    let mut statement = index
        .conn
        .prepare("SELECT record_id, record_hash FROM records WHERE record_hash IS NOT NULL")?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut found = std::collections::BTreeMap::new();
    for row in rows {
        let (id, hash) = row?;
        found.insert(id, hash);
    }
    Ok(found)
}

/// Index one record and everything derived from it.
/// Write one record's `records` row, its `record_tags` rows and its
/// `record_domains` rows.
///
/// Split out of [`index_one`] to keep that function under the line budget;
/// the provenance columns are the reason it grew past it.
fn insert_record_row(
    index: &Index,
    header: &RecordHeader,
    record_hash: &BlobHash,
    title: &str,
    name_slug: Option<&str>,
) -> Result<(), IndexError> {
    let provenance = &header.provenance;
    index.conn.execute(
        "INSERT OR REPLACE INTO records
         (record_id, corpus, kind, source, created, updated, raw_hash, stream_hash,
          record_hash, normalizer, storage, title, name_slug,
          project, project_source, remote, context, harness, model, session, cwd)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13,
                 ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)",
        rusqlite::params![
            header.id.as_str(),
            header.corpus.as_str(),
            header.kind.as_str(),
            format!("{}:{}", header.source.scheme(), header.source.value()),
            header.created.to_rfc3339(),
            header.updated.to_rfc3339(),
            header.raw.as_str(),
            header.stream.as_str(),
            record_hash.as_str(),
            header.normalizer,
            // An unregistered corpus is recorded as such rather than guessed:
            // guessing "copy" would make a dead reference read as corruption,
            // and guessing "reference" would hide real corruption.
            crate::corpus::Registry::default()
                .storage_of(header.corpus.as_str())
                .map_or("unregistered", crate::corpus::Storage::as_str),
            title,
            name_slug,
            provenance
                .project
                .as_ref()
                .map(tftio_lib::project::Slug::as_str),
            provenance
                .project_source
                .map(tftio_lib::project::Source::label),
            provenance
                .remote
                .as_ref()
                .map(tftio_lib::project::NormalizedRemote::as_str),
            provenance.context.as_deref(),
            provenance.harness.as_deref(),
            provenance.model.as_deref(),
            provenance.session.as_deref(),
            provenance.cwd.as_deref(),
        ],
    )?;
    for tag in &header.tags {
        index.conn.execute(
            "INSERT OR REPLACE INTO record_tags (record_id, tag) VALUES (?1, ?2)",
            rusqlite::params![header.id.as_str(), tag.as_str()],
        )?;
    }
    for domain in &provenance.domains {
        index.conn.execute(
            "INSERT OR REPLACE INTO record_domains (record_id, domain) VALUES (?1, ?2)",
            rusqlite::params![header.id.as_str(), domain],
        )?;
    }
    Ok(())
}

fn index_one(
    store: &impl BlobStore,
    index: &Index,
    header: &RecordHeader,
    record_hash: &BlobHash,
    report: &mut RebuildReport,
) -> Result<(), IndexError> {
    let label = header.id.as_str().to_owned();
    // The name comes from the raw artifact rather than the stream: an org
    // note declares its title with `#+title:`, which normalizing drops. The
    // extra read is what retiring the superseded database's `title` column
    // costs a rebuild (T029).
    let raw = store
        .get(&BlobHash::from(&header.raw))
        .map_err(|e| IndexError::Store {
            record: label.clone(),
            source: e,
        })?;
    let text = String::from_utf8_lossy(&raw);
    let title = record::title_of(header.kind, &text);
    let name_slug = record::name_slug_of(&text);
    insert_record_row(index, header, record_hash, &title, name_slug.as_deref())?;
    report.records = report.records.saturating_add(1);

    let stream_hash = BlobHash::from(&header.stream);
    let stream_bytes = store.get(&stream_hash).map_err(|e| IndexError::Store {
        record: label.clone(),
        source: e,
    })?;
    let stream = record::NormalizedStream::from_bytes(stream_bytes);

    for passage in record::passages(header.kind, &stream) {
        let text = String::from_utf8_lossy(
            stream
                .as_bytes()
                .get(passage.span.start()..passage.span.end())
                .unwrap_or_default(),
        )
        .into_owned();
        index.conn.execute(
            "INSERT INTO passages
             (record_id, stream_hash, level, span_start, span_len, body, title, fts_indexed)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1)",
            rusqlite::params![
                header.id.as_str(),
                header.stream.as_str(),
                passage_level(passage.level),
                sql_offset(passage.span.start()),
                sql_offset(passage.span.len()),
                text,
                title,
            ],
        )?;
        let rowid = index.conn.last_insert_rowid();
        index.conn.execute(
            "INSERT INTO passages_fts (rowid, title, body) VALUES (?1, ?2, ?3)",
            rusqlite::params![rowid, title, text],
        )?;
        report.passages = report.passages.saturating_add(1);
    }

    for link in record::authored_links(&stream) {
        let (kind, target) = link_target(&link.target);
        index.conn.execute(
            "INSERT INTO authored_links
             (record_id, target_kind, target, span_start, span_len)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                header.id.as_str(),
                kind,
                target,
                sql_offset(link.span.start()),
                sql_offset(link.span.len()),
            ],
        )?;
        report.authored_links = report.authored_links.saturating_add(1);
    }
    Ok(())
}

/// Whether a table of that name exists.
fn table_exists(conn: &Connection, name: &str) -> Result<bool, IndexError> {
    let count: i64 = conn.query_row(
        "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
        [name],
        |r| r.get(0),
    )?;
    Ok(count > 0)
}

/// One column value as the dump renders it.
///
/// A vector is summarized by length rather than printed: the dump exists to
/// compare two rebuilds, and 1024 floats of noise would bury the columns that
/// carry meaning while proving nothing the length does not.
fn display_value(value: rusqlite::types::Value) -> String {
    match value {
        rusqlite::types::Value::Null => "NULL".to_owned(),
        rusqlite::types::Value::Integer(i) => i.to_string(),
        rusqlite::types::Value::Real(f) => f.to_string(),
        rusqlite::types::Value::Text(s) => s,
        rusqlite::types::Value::Blob(b) => format!("{} bytes", b.len()),
    }
}

/// Byte offsets are `usize` in the domain and signed in `SQLite`. Saturating
/// rather than failing: a span longer than `i64::MAX` bytes cannot exist, so
/// the branch is unreachable and an error path for it would be noise.
fn sql_offset(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// A passage level's stored name.
const fn passage_level(level: record::PassageLevel) -> &'static str {
    match level {
        record::PassageLevel::Turn => "turn",
        record::PassageLevel::Section => "section",
        record::PassageLevel::Message => "message",
        record::PassageLevel::Thread => "thread",
    }
}

/// A link target's stored kind and value.
fn link_target(target: &record::LinkTarget) -> (&'static str, String) {
    match target {
        record::LinkTarget::Record(id) => ("record", id.as_str().to_owned()),
        record::LinkTarget::Name(name) => ("name", name.clone()),
        record::LinkTarget::Url(url) => ("url", url.clone()),
    }
}

/// Rebuild the mail corpus's rows from the Maildir.
///
/// A second rebuild path rather than a branch inside [`rebuild`], because the
/// two read different canonical sources: kb's records come from the store,
/// and mail's come from a Maildir the PKB does not own. What they share is
/// the property that matters — both discard and re-derive rather than
/// migrate, so the index stays disposable (ST-002).
///
/// Mail passages are cached in `passages` and deliberately kept out of
/// `passages_fts`. Lexical retrieval for mail is mu/Xapian's, which already
/// indexes senders, dates, identifiers and exact strings; duplicating that
/// here would be a second implementation to keep honest for no gain.
///
/// # Errors
///
/// [`IndexError::Database`] if the index cannot be written, or
/// [`IndexError::Record`] if a message cannot be normalized.
pub fn rebuild_mail(
    corpus: &crate::maildir::MaildirCorpus,
    selection: &crate::maildir::Selection,
    index: &Index,
) -> Result<RebuildReport, IndexError> {
    let started = std::time::Instant::now();
    let mut report = RebuildReport::default();
    index.conn.execute_batch("BEGIN IMMEDIATE")?;
    let outcome = rebuild_mail_within(corpus, selection, index, &mut report);
    if outcome.is_err() {
        index.conn.execute_batch("ROLLBACK")?;
        outcome?;
    }
    index.conn.execute_batch("COMMIT")?;
    report.elapsed = started.elapsed();
    Ok(report)
}

fn rebuild_mail_within(
    corpus: &crate::maildir::MaildirCorpus,
    selection: &crate::maildir::Selection,
    index: &Index,
    report: &mut RebuildReport,
) -> Result<(), IndexError> {
    index.clear(&Scope::Corpus("mail".to_owned()))?;
    index
        .conn
        .execute("DELETE FROM mail_catalogue", rusqlite::params![])?;
    let overrides: std::collections::BTreeSet<&str> =
        selection.overrides.iter().map(String::as_str).collect();
    let mut read = Vec::new();
    for message_id in &selection.selected {
        let Some(entry) = corpus.catalogued(message_id) else {
            continue;
        };
        let source = record::SourceRef::new(crate::maildir::SCHEME, message_id).map_err(|e| {
            IndexError::Record {
                record: message_id.clone(),
                source: e,
            }
        })?;
        // A message catalogued a moment ago and gone now is the corpus
        // behaving as a reference-only corpus does. Abandoning the whole
        // rebuild over it would make the index hostage to somebody else's
        // deletions.
        let Ok(bytes) = crate::corpus::Corpus::resolve(corpus, &source) else {
            continue;
        };
        index.conn.execute(
            "INSERT OR REPLACE INTO mail_catalogue
             (message_id, folder, locator, content_sha256, ground_truth_override)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                message_id,
                entry.folder,
                entry.locator,
                entry.content_sha256,
                i64::from(overrides.contains(message_id.as_str())),
            ],
        )?;
        let readable = crate::message::parse(&bytes);
        index_mail_message(index, message_id, &entry.content_sha256, &readable, report)?;
        read.push(readable);
    }
    index_mail_threads(index, &read, report)?;
    report.stale_vectors_dropped = index.drop_orphaned_derivations()?;
    Ok(())
}

/// Write one message's record and its message-level passage.
fn index_mail_message(
    index: &Index,
    message_id: &str,
    content_sha256: &str,
    readable: &crate::message::Readable,
    report: &mut RebuildReport,
) -> Result<(), IndexError> {
    let stream = record::normalize(
        record::ArtifactKind::MailMessage,
        readable.render().as_bytes(),
    )
    .map_err(|e| IndexError::Record {
        record: message_id.to_owned(),
        source: e,
    })?;
    let stream_hash = crate::maildir::content_sha256(stream.as_bytes());
    let when = readable.date.trim();
    write_reference_record(
        index,
        &ReferenceRecord {
            record_id: message_id,
            kind: record::ArtifactKind::MailMessage,
            source: &format!("{}:{message_id}", crate::maildir::SCHEME),
            when,
            raw_hash: content_sha256,
            stream_hash: &stream_hash,
            title: record::title_of(
                record::ArtifactKind::MailMessage,
                &String::from_utf8_lossy(stream.as_bytes()),
            ),
        },
    )?;
    report.records = report.records.saturating_add(1);
    write_passages(
        index,
        message_id,
        &stream_hash,
        record::ArtifactKind::MailMessage,
        &stream,
        report,
    )
}

/// Assemble threads from the identifier headers and index one record each.
///
/// Only groups of two or more become thread records: a thread of one is the
/// message, and indexing it twice would put two near-identical vectors in
/// front of every query that matches it.
fn index_mail_threads(
    index: &Index,
    read: &[crate::message::Readable],
    report: &mut RebuildReport,
) -> Result<(), IndexError> {
    for (root, members) in crate::maildir::threads(read) {
        if members.len() < 2 {
            continue;
        }
        let mbox = crate::maildir::as_mbox(&members);
        let stream =
            record::normalize(record::ArtifactKind::MailThread, mbox.as_bytes()).map_err(|e| {
                IndexError::Record {
                    record: root.clone(),
                    source: e,
                }
            })?;
        let stream_hash = crate::maildir::content_sha256(stream.as_bytes());
        let record_id = format!("thread:{root}");
        let when = members.first().map(|m| m.date.trim()).unwrap_or_default();
        write_reference_record(
            index,
            &ReferenceRecord {
                record_id: &record_id,
                kind: record::ArtifactKind::MailThread,
                source: &format!("thread:{root}"),
                when,
                raw_hash: &crate::maildir::content_sha256(mbox.as_bytes()),
                stream_hash: &stream_hash,
                // A thread is named by its first message's subject, which is
                // what the mbox rendering opens with.
                title: record::title_of(
                    record::ArtifactKind::MailThread,
                    &String::from_utf8_lossy(stream.as_bytes()),
                ),
            },
        )?;
        report.records = report.records.saturating_add(1);
        write_passages(
            index,
            &record_id,
            &stream_hash,
            record::ArtifactKind::MailThread,
            &stream,
            report,
        )?;
    }
    Ok(())
}

/// The columns a reference-only record supplies.
struct ReferenceRecord<'a> {
    record_id: &'a str,
    kind: record::ArtifactKind,
    source: &'a str,
    when: &'a str,
    raw_hash: &'a str,
    stream_hash: &'a str,
    /// The record's one-line name. Derived from the normalized stream rather
    /// than from a raw blob, because a referenced corpus puts no raw bytes in
    /// the store to read one from.
    title: String,
}

fn write_reference_record(index: &Index, row: &ReferenceRecord<'_>) -> Result<(), IndexError> {
    index.conn.execute(
        "INSERT OR REPLACE INTO records
         (record_id, corpus, kind, source, created, updated, raw_hash, stream_hash,
          record_hash, normalizer, storage, title, name_slug)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL, ?9, ?10, ?11, ?12)",
        rusqlite::params![
            row.record_id,
            "mail",
            row.kind.as_str(),
            row.source,
            row.when,
            row.when,
            row.raw_hash,
            row.stream_hash,
            record::NORMALIZER_VERSION,
            crate::corpus::Storage::ReferenceOnly.as_str(),
            row.title,
            // A message declares no `#+name:` slug: nothing links to mail by
            // name, and inventing one would make a message answer a
            // `[[name]]` link written about a note.
            None::<String>,
        ],
    )?;
    Ok(())
}

/// Cache one record's passages, without touching the FTS.
fn write_passages(
    index: &Index,
    record_id: &str,
    stream_hash: &str,
    kind: record::ArtifactKind,
    stream: &record::NormalizedStream,
    report: &mut RebuildReport,
) -> Result<(), IndexError> {
    for passage in record::passages(kind, stream) {
        let text = String::from_utf8_lossy(
            stream
                .as_bytes()
                .get(passage.span.start()..passage.span.end())
                .unwrap_or_default(),
        )
        .into_owned();
        index.conn.execute(
            "INSERT INTO passages
             (record_id, stream_hash, level, span_start, span_len, body, fts_indexed)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0)",
            rusqlite::params![
                record_id,
                stream_hash,
                passage_level(passage.level),
                sql_offset(passage.span.start()),
                sql_offset(passage.span.len()),
                text,
            ],
        )?;
        report.passages = report.passages.saturating_add(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Index, IndexError, display_value, expected_empirical_maximum};
    use rusqlite::types::Value;

    /// Every column type the schema can hold has to render, or a dump would
    /// silently omit the difference between two rebuilds.
    #[test]
    fn every_column_type_renders_in_a_dump() {
        assert_eq!(display_value(Value::Null), "NULL");
        assert_eq!(display_value(Value::Integer(42)), "42");
        assert_eq!(display_value(Value::Real(0.5)), "0.5");
        assert_eq!(display_value(Value::Text("kb".to_owned())), "kb");
        assert_eq!(display_value(Value::Blob(vec![0; 12])), "12 bytes");
    }

    /// An index that never touches disk, for callers measuring or testing
    /// against a corpus they do not want to keep.
    #[test]
    fn an_in_memory_index_starts_empty() {
        let index = Index::open_in_memory().expect("open");

        assert_eq!(index.dump().expect("dump"), String::new());
    }

    /// The length correction uses the empirical order statistics described in
    /// T031 rather than a fitted constant. For a uniform two-point empirical
    /// distribution, one draw has expectation 1/2 and two have 3/4.
    #[test]
    fn expected_maximum_is_fitted_from_the_empirical_distribution() {
        let distribution = [0.0, 1.0];

        assert_eq!(expected_empirical_maximum(&distribution, 1), Some(0.5));
        assert_eq!(expected_empirical_maximum(&distribution, 2), Some(0.75));
        assert_eq!(expected_empirical_maximum(&distribution, 0), None);
        assert_eq!(expected_empirical_maximum(&[], 1), None);
    }

    /// Asking for a record the index does not hold is a domain answer, not a
    /// database error the caller has to interpret.
    #[test]
    fn asking_for_an_unindexed_record_names_it() {
        let index = Index::open_in_memory().expect("open");

        let outcome = index.record("n-missing");

        assert!(
            matches!(&outcome, Err(IndexError::NotFound(id)) if id == "n-missing"),
            "outcome was: {outcome:?}"
        );
    }
}
