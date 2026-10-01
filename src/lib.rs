//! Personal knowledge base — typed AST with org-mode as projection, SQLite-backed.
//!
//! The crate provides:
//! - [`parser`] — org-mode text → AST
//! - [`generator`] — AST → org-mode text
//! - [`storage`] — `SQLite` persistence with FTS5 search
//! - [`embed_text`] — composition of the text an embedding model sees
//!
//! The org-mode AST types are re-used from the companion [`tftio_org`] crate;
//! the parser is kept in-crate (see the crate `README` for the rationale).
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::disallowed_methods,
        reason = "test code uses fail-fast assertions against temp fixtures, mock servers, and sandboxed HOME/XDG env"
    )
)]

pub mod canonical;
pub mod cli_embed;
pub mod cli_main;
pub mod corpus;
pub mod embed_text;
pub mod embedding;
pub mod error;
pub mod fsck;
pub mod generate;
pub mod generator;
pub mod index;
pub mod ingest;
pub mod mail;
pub mod maildir;
pub mod markdown;
pub mod mcp;
pub mod mcp_http;
pub mod mcp_stdio;
pub mod message;
pub mod migrate;
pub mod mu;
pub mod org_meta;
pub mod parser;
pub mod project;
pub mod prompt;
pub mod record;
pub mod rerank;
pub mod retrieval;
pub mod search;
pub mod sexp;
pub mod storage;
pub mod store;
pub mod write;
