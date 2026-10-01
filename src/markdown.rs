//! Markdown ingestion: convert GitHub-flavored markdown to the kb AST.
//!
//! kb's canonical document model is org-mode (see [`tftio_org::ast`]). Rather
//! than maintain a second parser, markdown is converted to org text by an
//! external `pandoc` process and then parsed by [`crate::parser`]. The
//! dependency on `pandoc` is intentional and documented: a missing binary
//! yields a clear error rather than a panic, and the conversion is
//! ingest-only (kb never renders markdown back out).

use std::io::Write;
use std::process::{Command, Stdio};

use thiserror::Error;

use crate::parser::{self, ParseError};
use tftio_org::ast::Document;

/// Failure modes of markdown ingestion via `pandoc`.
///
/// A caller can branch [`MarkdownError::PandocNotFound`] (a setup /
/// environment problem the operator must fix) against the input-shaped
/// failures ([`MarkdownError::PandocFailed`], [`MarkdownError::Parse`],
/// …) that are attributable to the supplied body.
#[derive(Debug, Error)]
pub enum MarkdownError {
    /// `pandoc` is not on `PATH`. An environment/setup error, not a
    /// problem with the input body.
    #[error("pandoc not found on PATH; markdown import requires pandoc")]
    PandocNotFound,

    /// Spawning the `pandoc` child process failed for a reason other
    /// than the binary being absent.
    #[error("failed to spawn pandoc: {0}")]
    Spawn(std::io::Error),

    /// Waiting on the `pandoc` child process failed.
    #[error("failed to wait for pandoc: {0}")]
    Wait(std::io::Error),

    /// `pandoc` exited non-zero. Carries the trimmed stderr for display.
    #[error("pandoc exited with failure: {stderr}")]
    PandocFailed {
        /// Trimmed stderr captured from the failed `pandoc` invocation.
        stderr: String,
    },

    /// `pandoc` produced output that was not valid UTF-8.
    #[error("pandoc produced non-UTF-8 output: {0}")]
    NonUtf8Output(#[from] std::string::FromUtf8Error),

    /// The org text `pandoc` produced did not parse as a kb document.
    #[error("org parse failure after pandoc conversion: {0}")]
    Parse(#[from] ParseError),
}

/// Convert a GitHub-flavored markdown string into a kb [`Document`].
///
/// Shells out to `pandoc -f gfm -t org --wrap=preserve`, then parses the
/// resulting org text with [`parser::parse_document`].
///
/// # Errors
///
/// Returns [`MarkdownError`] when `pandoc` is not on `PATH`, exits
/// non-zero, emits non-UTF-8 output, or produces org text the kb parser
/// rejects.
pub fn markdown_to_document(raw: &str) -> Result<Document, MarkdownError> {
    let org = markdown_to_org(raw)?;
    Ok(parser::parse_document(&org)?)
}

/// Convert GitHub-flavored markdown to org text via `pandoc`.
///
/// stdin is written from a separate thread so a large conversion cannot
/// deadlock against a full stdout pipe.
fn markdown_to_org(raw: &str) -> Result<String, MarkdownError> {
    let mut child = Command::new("pandoc")
        .args(["-f", "gfm", "-t", "org", "--wrap=preserve"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => MarkdownError::PandocNotFound,
            _ => MarkdownError::Spawn(e),
        })?;

    // `Stdio::piped()` above guarantees a stdin handle; take it without a
    // fallible branch. If it were somehow absent the writer thread simply
    // would not run and pandoc would see empty input — surfaced as a
    // PandocFailed/Parse error downstream rather than a panic.
    let writer = child.stdin.take().map(|mut stdin| {
        let raw_owned = raw.to_string();
        std::thread::spawn(move || {
            // Drop closes the pipe; a write error surfaces as pandoc failure.
            if let Err(e) = stdin.write_all(raw_owned.as_bytes()) {
                tracing::warn!("failed to write stdin to pandoc: {e}");
            }
        })
    });

    let output = child.wait_with_output().map_err(MarkdownError::Wait)?;
    if let Some(writer) = writer {
        let _ = writer.join();
    }

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MarkdownError::PandocFailed {
            stderr: stderr.trim().to_string(),
        });
    }

    Ok(String::from_utf8(output.stdout)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tftio_org::ast::Block;

    /// Skip the test body when `pandoc` is not installed.
    fn pandoc_available() -> bool {
        Command::new("pandoc")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }

    #[test]
    fn markdown_heading_and_paragraph_convert() {
        if !pandoc_available() {
            eprintln!("skipping: pandoc not on PATH");
            return;
        }
        let doc = markdown_to_document("# Title\n\nbody text\n").unwrap();
        assert!(
            doc.blocks
                .iter()
                .any(|b| matches!(b, Block::Heading { .. })),
            "expected a heading block, got {:?}",
            doc.blocks
        );
    }

    #[test]
    fn markdown_strikethrough_round_trips_to_inline() {
        if !pandoc_available() {
            eprintln!("skipping: pandoc not on PATH");
            return;
        }
        // gfm ~~x~~ -> org +x+ -> Inline::Strikethrough
        let doc = markdown_to_document("~~struck~~\n").unwrap();
        let rendered = crate::generator::generate(&doc);
        assert!(
            rendered.contains("+struck+"),
            "expected strikethrough markup, got: {rendered}"
        );
    }
}
