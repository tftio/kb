//! Brute-force scan cost at and beyond the current corpus size (T016).
//!
//! `PLAN-20260813-retrieval-evaluation-harness` deferred approximate nearest
//! neighbour search because the exact scan cost about 60ms of a 1400ms search
//! at 14,055 vectors. That was a measurement, and it was correct then. This
//! re-takes it at the size the corpus actually reached, and at the size a full
//! archive ingestion would reach, so the deferral is re-decided on a number
//! rather than renewed by habit.
//!
//! The companion measurement of the superseded database's own scan went with
//! that database's retrieval path (T029). Every corpus is served from the
//! derived index now, so there is one scan to measure.
//!
//! Ignored by default: it is a measurement rather than an assertion, it takes
//! minutes at the larger sizes, and a wall-clock threshold in the test suite
//! would fail on a loaded machine and teach its reader to ignore it. Run it
//! deliberately:
//!
//! ```text
//! cargo test --release --test scan_bench -- --ignored --nocapture
//! ```

use std::time::Instant;

use kb::embedding::encode_embedding;
use kb::index::Index;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// The embedding model's dimension. 1024 f32s is 4,096 bytes a vector, which
/// is what makes the scan a question about bytes read rather than about
/// arithmetic.
const DIMENSION: usize = 1024;

/// A deterministic unit-ish vector that differs per index.
///
/// Values vary so the cosine is not degenerate, but nothing here depends on
/// the distribution: a brute-force scan reads and multiplies every stored
/// vector whatever they contain, which is the property being measured.
fn vector(seed: usize) -> Vec<f32> {
    (0..DIMENSION)
        .map(|component| {
            let n =
                u16::try_from((seed.wrapping_mul(31).wrapping_add(component)) % 1000).unwrap_or(0);
            f32::from(n) / 1000.0
        })
        .collect()
}

/// An index holding `count` mail vectors and the passages they describe.
///
/// The rows are written through a plain connection to the same file rather
/// than through [`Index`], because records normally arrive from the store or
/// from a Maildir and there is no public way to fabricate a quarter of a
/// million of them — nor should there be. A benchmark manufacturing its own
/// fixture is allowed to know the schema; production code is not.
fn synthetic(dir: &std::path::Path, count: usize) -> Result<Index, Box<dyn std::error::Error>> {
    let path = dir.join("index.db");
    // Opening for rebuild creates the schema; the writes then go through a
    // second connection, and the index is reopened to scan.
    drop(Index::open_for_rebuild(&path)?);
    let conn = rusqlite::Connection::open(&path)?;
    conn.execute_batch("PRAGMA journal_mode = WAL; BEGIN IMMEDIATE")?;
    for n in 0..count {
        let record = format!("<synthetic-{n}@example.invalid>");
        let stream = format!("{n:040x}");
        conn.execute(
            "INSERT INTO records (record_id, corpus, kind, source, created, updated,
                                  raw_hash, stream_hash, record_hash, normalizer, storage)
             VALUES (?1, 'mail', 'mail-message', ?2, '2026-01-01T00:00:00Z',
                     '2026-01-01T00:00:00Z', ?3, ?3, NULL, 3, 'reference')",
            rusqlite::params![record, format!("message-id:{record}"), stream],
        )?;
        conn.execute(
            "INSERT INTO passages (record_id, stream_hash, level, span_start, span_len,
                                   text, fts_indexed)
             VALUES (?1, ?2, 'message', 0, 100, '', 0)",
            rusqlite::params![record, stream],
        )?;
        conn.execute(
            "INSERT INTO embeddings (stream_hash, span_start, span_len, model, vector)
             VALUES (?1, 0, 100, 'm', ?2)",
            rusqlite::params![stream, encode_embedding(&vector(n))],
        )?;
    }
    conn.execute_batch("COMMIT")?;
    // Fold the write-ahead log back into the database before measuring.
    // Without this the vectors are read out of a freshly written WAL rather
    // than out of the main file, which measured four times slower than the
    // live index at the same vector count — an artefact of how the fixture was
    // built, not a property of the scan.
    conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
    drop(conn);
    Ok(Index::open(&path)?)
}

/// Time `passes` scans and report the mean.
///
/// The model name is a parameter rather than a constant because it is the
/// scan's `WHERE` clause: naming a model the corpus is not filed under matches
/// no rows and measures nothing, which is what the first version of this
/// benchmark did to the live index — reporting 10ms for a scan that never
/// happened.
fn timed(
    index: &Index,
    model: &str,
    corpus: &str,
    passes: u32,
) -> Result<(f64, usize), Box<dyn std::error::Error>> {
    let query = vector(7);
    // One untimed pass, so the figure is a scan rather than a page-cache miss.
    let ranked = index.rank_by_embedding(&query, model, corpus)?;
    let started = Instant::now();
    for _ in 0..passes {
        index.rank_by_embedding(&query, model, corpus)?;
    }
    Ok((
        started.elapsed().as_secs_f64() / f64::from(passes),
        ranked.len(),
    ))
}

#[test]
#[ignore = "a measurement, not an assertion; run with --ignored --nocapture"]
fn scan_cost_at_the_live_corpus_size() -> TestResult {
    let path = kb::index::default_index_path();
    if !path.exists() {
        println!("no index at {}; skipping", path.display());
        return Ok(());
    }
    let index = Index::open(&path)?;
    let counting =
        rusqlite::Connection::open_with_flags(&path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let vectors: i64 =
        counting.query_row("SELECT count(*) FROM embeddings", [], |row| row.get(0))?;
    let model: String =
        counting.query_row("SELECT model FROM embeddings LIMIT 1", [], |row| row.get(0))?;
    let (seconds, ranked) = timed(&index, &model, "mail", 5)?;
    println!(
        "live mail index: {vectors} vectors under {model}, {ranked} records ranked, \
         {:.1}ms per scan",
        seconds * 1000.0
    );
    Ok(())
}

#[test]
#[ignore = "a measurement, not an assertion; run with --ignored --nocapture"]
fn scan_cost_scales_with_the_vector_count() -> TestResult {
    println!(
        "{:>10}  {:>12}  {:>12}",
        "vectors", "scan (ms)", "MB scanned"
    );
    for count in [12_500_usize, 25_000, 50_000, 100_000, 200_000] {
        let dir = tempfile::tempdir()?;
        let index = synthetic(dir.path(), count)?;
        let passes = if count > 100_000 { 3 } else { 5 };
        let (seconds, ranked) = timed(&index, "m", "mail", passes)?;
        assert_eq!(ranked, count, "the fixture did not rank what it stored");
        #[allow(
            clippy::cast_precision_loss,
            reason = "a vector count and a byte count, reported to one decimal place"
        )]
        let megabytes = (count * DIMENSION * 4) as f64 / (1024.0 * 1024.0);
        println!("{count:>10}  {:>12.1}  {megabytes:>12.1}", seconds * 1000.0);
    }
    Ok(())
}
