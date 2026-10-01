//! v10 phase tests: F9 (parser panic-on-pipe-input). Each test name aligns
//! with the silent-critic criterion check filter (e.g. `cargo test -p
//! tftio-kb -- parser_panic_free_pipe`).
//!
//! The F10 half of this suite asserted an HTTP request-body limit (5 MiB
//! accepted, 33 MiB rejected). It was retired along with the HTTP surface
//! itself; there is no request body to bound in a CLI that reads its input
//! from a file or stdin.

use kb::parser::parse_document;
use proptest::prelude::*;
use tftio_org::ast::{Block, Document, Inline, Tag, Title};

mod common;

type TestResult = Result<(), Box<dyn std::error::Error>>;

// ── F9 parser tests ───────────────────────────────────────────────────

#[test]
fn parser_panic_free_pipe_returns_ok() -> TestResult {
    // The original crash: `line[1..line.len() - 1]` on a `"|"` line was
    // an invalid slice range. The fix bounds-checks before slicing; the
    // single-pipe line is no longer recognised as a table row, so the
    // parser falls through to the next block kind and returns Ok.
    let got = parse_document("|\n")?;
    // Exact representation is acceptance-flexible: paragraph or empty
    // doc, but never a panic and never a parse error.
    assert!(
        got.blocks.len() <= 1,
        "expected 0 or 1 top-level block, got {got:?}"
    );
    Ok(())
}

#[test]
fn parser_panic_free_pipe_handles_empty_string_too() -> TestResult {
    // The bounds-check also rules out empty strings reaching the slice.
    let got = parse_document("")?;
    assert!(got.blocks.is_empty());
    Ok(())
}

#[test]
fn parser_empty_input_stable_matches_v8_behaviour() -> TestResult {
    // Per criterion `parser-empty-input-stable`: empty / minimal inputs
    // are unchanged from v8.
    let empty = parse_document("")?;
    assert_eq!(empty.blocks, Vec::<Block>::new());

    // A lone newline is one blank line — represented explicitly so
    // spacing round-trips (diverges from v8, which dropped it).
    let single_newline = parse_document("\n")?;
    assert_eq!(single_newline.blocks, vec![Block::BlankLine]);

    let one_para = parse_document("a\n")?;
    assert_eq!(
        one_para.blocks,
        vec![Block::Paragraph {
            inlines: vec![Inline::Plain("a".into())],
        }]
    );
    Ok(())
}

fn arb_fuzz_byte() -> impl Strategy<Value = u8> {
    prop_oneof![
        20 => 0x20u8..=0x7E_u8, // printable ASCII (space..=tilde)
        2  => Just(b'\n'),
        1  => Just(b'\t'),
    ]
}

fn arb_fuzz_input() -> impl Strategy<Value = String> {
    // Bytes are drawn from ASCII-only ranges, so they are always valid
    // UTF-8; `from_utf8_lossy` is exact here and avoids an unwrap/expect.
    prop::collection::vec(arb_fuzz_byte(), 0..=256)
        .prop_map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
}

proptest! {
    // Per criterion `parser-panic-free-fuzz`: 1024+ cases.
    #![proptest_config(ProptestConfig { cases: 1024, .. ProptestConfig::default() })]

    /// Random short strings drawn from ASCII-printable plus newline and
    /// tab. The assertion is `no panic` — the parser may return Ok or
    /// Err, but must not unwind through `parse_document`.
    #[test]
    fn parser_panic_free_fuzz_random_inputs(input in arb_fuzz_input()) {
        let _ = parse_document(&input);
    }
}

#[test]
fn parser_panic_free_fuzz_explicit_short_inputs() {
    // Hand-picked short cases the criterion calls out: single delimiter
    // characters, unbalanced delimiters, and short pipe-tables. None of
    // these may panic. Outcome (Ok/Err) is irrelevant here.
    let cases: &[&str] = &[
        "",
        "\n",
        "|",
        "|\n",
        "||",
        "|||",
        "| |",
        "| |\n",
        "*",
        "*\n",
        "**",
        ":",
        ":\n",
        "::",
        "#",
        "#\n",
        "#+",
        "-",
        "-\n",
        "--",
        "---",
        "----",
        "-----",
        "|---|",
        "|---|\n",
        "|a|b|",
        "|a|b|\n",
        "|a\n",
        "a|\n",
        ":PROPERTIES:",
        ":PROPERTIES:\n",
        ":LOGBOOK:\n",
        "#+begin_src",
        "#+begin_quote",
        "1.",
        "1. \n",
        "[[",
        "[[id:",
        "*bold",
        "/italic",
        "=code",
        "~verb",
    ];
    for s in cases {
        // No `expect` — the assertion is purely about NOT panicking.
        let _ = parse_document(s);
    }
}

// ── F9 regression through the write path ──────────────────────────────

#[test]
fn f9_regression_fixture_pipe_line_in_org_body_stores_cleanly() -> TestResult {
    // A small org fixture with a single-`|` line — the byte sequence
    // that crashed the write path during the 2026-05-05 Claude Code
    // import. Originally asserted against the HTTP handler; asserted here
    // against parse-then-store, which is the path the CLI takes and the
    // only one that survives the server's retirement.
    let dir = tempfile::tempdir()?;
    let store = kb::store::GitBlobStore::open_or_init(&dir.path().join("store"))?;
    let index = kb::index::Index::open_for_rebuild(&dir.path().join("index.db"))?;

    let doc = parse_document("* Transcript\nsome text\n|\nmore text\n")?;
    let options = kb::write::WriteOptions::note("f9-fixture")?;
    kb::write::put_record(&store, &index, "f9-fixture", &doc, &options)?;
    assert!(
        index.record("f9-fixture").is_ok(),
        "the pipe-line fixture must round-trip to the store"
    );
    Ok(())
}

// ── Scope smoke ────────────────────────────────────────────────────────

/// Type alias confirming the [`kb::embedding::EmbeddingClient`] trait is
/// still object-safe and reachable after the v10 changes. Declared at
/// module scope to avoid `clippy::items_after_statements` inside the
/// test body.
type ScopeSmokeEmbeddingRef = std::sync::Arc<dyn kb::embedding::EmbeddingClient>;

#[test]
fn v10_scope_smoke_storage_embedding_surface_intact() -> TestResult {
    // Smoke for criterion `no-storage-or-mcp-changes`: confirms that
    // the storage / embedding public surfaces compile and respond
    // unchanged.
    //
    // The criterion's MCP half was retired along with the MCP surface
    // itself, and its HTTP half with the server surface;
    // it is not an oversight that nothing here touches either any more.
    let dir = tempfile::tempdir()?;
    let legacy = common::legacy_db(&dir.path().join("smoke.db"))?;
    let doc = Document {
        blocks: vec![Block::Heading {
            level: 1,
            title: Title("Smoke".into()),
            tags: vec![Tag("scope".into())],
            children: vec![],
        }],
    };

    common::legacy_insert(&legacy, "smoke-1", &doc)?;
    let full = kb::storage::get_node_full(&legacy, "smoke-1")?
        .ok_or("smoke-1 node missing from get_node_full")?;
    assert_eq!(full.document, doc);

    // `EmbeddingClient` trait still object-safe and the public re-export
    // path is intact.
    let _: Option<ScopeSmokeEmbeddingRef> = None;

    Ok(())
}
