//! Fixtures shared by the integration suites.
//!
//! Only what more than one suite needs, and only what has nowhere better to
//! live. The legacy-database builders below are here because the writer for
//! that schema was deleted with the rest of the retired backend (T029): the
//! superseded database is something this repository reads and never writes,
//! so a test that needs one states its shape itself rather than asking
//! production code to keep a writer alive for the tests' benefit.

#![allow(
    dead_code,
    reason = "each integration binary compiles this module and uses part of it"
)]

use std::path::Path;

use tftio_org::ast::Document;

/// Create the superseded database's schema and open it.
///
/// Just the two tables `kb export` and `kb fsck` read: nodes and their tags.
///
/// # Errors
///
/// Returns the driver's error if the file cannot be opened or created.
pub fn legacy_db(path: &Path) -> Result<rusqlite::Connection, Box<dyn std::error::Error>> {
    let conn = rusqlite::Connection::open(path)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS nodes (
             id         TEXT PRIMARY KEY,
             title      TEXT NOT NULL,
             ast_blob   TEXT NOT NULL,
             created_at TEXT NOT NULL,
             updated_at TEXT NOT NULL,
             name_slug  TEXT
         );
         CREATE TABLE IF NOT EXISTS node_tags (
             node_id TEXT NOT NULL,
             tag     TEXT NOT NULL,
             PRIMARY KEY (node_id, tag)
         );",
    )?;
    Ok(conn)
}

/// Insert one node into a legacy database, the way `kb create` once did:
/// the document as an s-expression blob, its title materialized, and its
/// tags filed alongside.
///
/// # Errors
///
/// Returns the driver's error if the insert fails.
pub fn legacy_insert(
    conn: &rusqlite::Connection,
    id: &str,
    document: &Document,
) -> Result<(), Box<dyn std::error::Error>> {
    let title = kb::storage::extract_title(document);
    let blob = kb::sexp::encode_document(document);
    conn.execute(
        "INSERT OR REPLACE INTO nodes (id, title, ast_blob, created_at, updated_at)
         VALUES (?1, ?2, ?3, '2026-08-01T00:00:00.000Z', '2026-08-01T00:00:00.000Z')",
        rusqlite::params![id, title, blob],
    )?;
    for tag in kb::storage::extract_tags(document) {
        conn.execute(
            "INSERT OR IGNORE INTO node_tags (node_id, tag) VALUES (?1, ?2)",
            rusqlite::params![id, tag.0],
        )?;
    }
    Ok(())
}
