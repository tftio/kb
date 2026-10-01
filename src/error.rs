//! Error types for the kb crate.

use thiserror::Error;

/// Errors that can occur in kb operations.
#[derive(Debug, Error)]
pub enum KbError {
    /// A database operation failed.
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// A node was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// An I/O operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A stored `ast_blob` failed to decode, attributed to the owning node.
    #[error("corrupt ast_blob for node {node_id}: {reason}")]
    CorruptAstBlob {
        /// Id of the node whose blob failed to decode.
        node_id: String,
        /// Decoder failure detail (display only).
        reason: String,
    },

    /// A tag merge was refused before any node was touched. This is a
    /// caller error, not a database fault.
    #[error("{0}")]
    TagMerge(#[from] crate::storage::TagMergeRefusal),

    /// A single-node tag edit was refused before the node was touched.
    /// This is a caller error, not a database fault.
    #[error("{0}")]
    TagEdit(#[from] crate::storage::TagEditRefusal),
}
