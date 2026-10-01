//! Lexical retrieval over the Maildir, delegated to mu.
//!
//! kb's own FTS5 index knows a mail message as the readable text
//! [`crate::message`] normalized it into. mu knows it as a message: it has
//! parsed the MIME tree, resolved the headers, and built a Xapian index with
//! fields for correspondent, folder, flag and date that FTS5 has no notion of.
//! That index can cover hundreds of thousands of messages and is maintained
//! by something other than kb, which is the argument for using it rather than
//! reproducing it — a second full-text index over the same Maildir would be a
//! second thing to keep current, and the one that drifts is always the one
//! nobody is watching.
//!
//! The join needs no new identity scheme. mu reports a message's `Message-ID`,
//! and a mail record in the index *is* its bracketed `Message-ID`
//! ([`crate::maildir`]), so a mu result names a kb record directly.
//!
//! **This degrades loudly** (`REPO_INVARIANTS.md` ENG-004). A missing binary,
//! a missing index and a failed query are three different errors, and none of
//! them is an empty result: a mail query that silently returned nothing
//! because mu was absent would read as "the archive holds nothing on this
//! subject", which is the one answer this module must never give by accident.

use std::process::Command;

use thiserror::Error;

/// Why a mu query could not be answered.
#[derive(Debug, Error)]
pub enum MuError {
    /// `mu` is not on `PATH`. An environment error rather than a user one.
    #[error("mu not found on PATH; mail lexical retrieval requires mu")]
    NotFound,
    /// Spawning `mu` failed for a reason other than absence.
    #[error("failed to spawn mu: {0}")]
    Spawn(String),
    /// `mu` exited non-zero for a reason other than an empty result.
    #[error("mu failed: {0}")]
    Failed(String),
    /// `mu` answered with bytes that are not UTF-8.
    #[error("mu returned output that is not UTF-8")]
    Encoding,
}

/// Lexical retrieval over mail, as the planner needs it.
///
/// The domain interface (`REPO_INVARIANTS.md` ENG-010): the planner asks for
/// message identifiers matching a query and knows nothing about Xapian,
/// subprocesses, or mu's query language.
pub trait MailSearch {
    /// Message identifiers matching `query`, most relevant first, at most
    /// `limit` of them.
    ///
    /// Identifiers are returned bracketed, in the form a mail record is
    /// addressed by.
    ///
    /// # Errors
    ///
    /// [`MuError`] if the backend is absent, unusable, or fails. An empty
    /// archive and an absent backend must not both be an empty list.
    fn message_ids(&self, query: &str, limit: usize) -> Result<Vec<String>, MuError>;
}

/// Wrap `raw` in angle brackets unless it already is.
///
/// mu reports identifiers bare; a mail record is addressed by the bracketed
/// form the message's own `Message-ID` header carries.
#[must_use]
pub fn bracketed(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.starts_with('<') && trimmed.ends_with('>') {
        trimmed.to_owned()
    } else {
        format!("<{trimmed}>")
    }
}

/// Read message identifiers from `mu find --fields i --format plain` output.
///
/// Blank lines are dropped and order is preserved, because mu's order is its
/// relevance ranking and re-sorting it would discard the signal.
#[must_use]
pub fn parse_ids(stdout: &str, limit: usize) -> Vec<String> {
    stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(limit)
        .map(bracketed)
        .collect()
}

/// mu's exit code for a query that matched nothing.
///
/// Distinguished from a failure because an archive with nothing on a subject
/// is an answer, and only a genuine fault is an error. Measured rather than
/// read from a manual: mu 1.14.3 exits 2 here and 1 when the index itself
/// cannot be opened.
const NO_MATCHES: i32 = 2;

/// What mu says on stderr when a query matched nothing.
///
/// Checked alongside the exit code so a different failure that happens to
/// share it is reported rather than silently read as an empty archive.
///
/// This is as far as the distinction can be taken: mu's query parser is
/// forgiving, and an expression kb might consider malformed comes back as
/// this same "no matches" rather than as a parse error. That is mu's
/// behaviour and not something a caller can recover — and it matches kb's own
/// keyword mode, where no input is a syntax error either.
const NO_MATCHES_MESSAGE: &str = "no matches";

/// [`MailSearch`] backed by the `mu` binary.
#[derive(Debug, Clone)]
pub struct MuIndex {
    /// `--muhome`, when the index is not at mu's default location. Set by
    /// tests against a fixture index; `None` in normal operation.
    home: Option<std::path::PathBuf>,
}

impl MuIndex {
    /// A client against mu's default index.
    #[must_use]
    pub const fn new() -> Self {
        Self { home: None }
    }

    /// A client against the index under `home`.
    #[must_use]
    pub fn at(home: impl Into<std::path::PathBuf>) -> Self {
        Self {
            home: Some(home.into()),
        }
    }
}

impl Default for MuIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl MailSearch for MuIndex {
    fn message_ids(&self, query: &str, limit: usize) -> Result<Vec<String>, MuError> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let mut command = Command::new("mu");
        // `--muhome` is a per-subcommand option rather than a global one, so
        // it goes after `find`; mu rejects it outright in front.
        command.arg("find");
        if let Some(home) = &self.home {
            command.arg("--muhome").arg(home);
        }
        command
            .arg("--fields")
            .arg("i")
            .arg("--format")
            .arg("plain")
            .arg(query);
        let output = command.output().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                MuError::NotFound
            } else {
                MuError::Spawn(e.to_string())
            }
        })?;
        if !output.status.success() {
            let detail = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            if output.status.code() == Some(NO_MATCHES) && detail.contains(NO_MATCHES_MESSAGE) {
                return Ok(Vec::new());
            }
            return Err(MuError::Failed(detail));
        }
        let stdout = String::from_utf8(output.stdout).map_err(|_| MuError::Encoding)?;
        Ok(parse_ids(&stdout, limit))
    }
}

#[cfg(test)]
mod tests {
    use super::{MailSearch, MuError, MuIndex, bracketed, parse_ids};

    #[test]
    fn a_bare_identifier_gains_the_brackets_a_record_is_addressed_by() {
        assert_eq!(bracketed("a@b.example"), "<a@b.example>");
    }

    #[test]
    fn an_already_bracketed_identifier_is_left_alone() {
        assert_eq!(bracketed("<a@b.example>"), "<a@b.example>");
        assert_eq!(bracketed("  <a@b.example>  "), "<a@b.example>");
    }

    #[test]
    fn output_parses_in_the_order_mu_returned_it() {
        let ids = parse_ids("one@x\n\ntwo@x\nthree@x\n", 10);
        assert_eq!(ids, vec!["<one@x>", "<two@x>", "<three@x>"]);
    }

    #[test]
    fn the_limit_bounds_what_is_taken_from_the_front() {
        let ids = parse_ids("one@x\ntwo@x\nthree@x\n", 2);
        assert_eq!(ids, vec!["<one@x>", "<two@x>"]);
    }

    #[test]
    fn an_empty_query_asks_mu_nothing() {
        // No index is configured here, so a query that reached mu would fail.
        let found = MuIndex::at("/nonexistent")
            .message_ids("   ", 10)
            .unwrap_or_else(|e| panic!("an empty query should not reach mu: {e}"));
        assert!(found.is_empty());
    }

    #[test]
    fn an_absent_index_is_an_error_rather_than_an_empty_archive() {
        match MuIndex::at("/nonexistent/muhome").message_ids("anything", 10) {
            Err(MuError::Failed(_) | MuError::NotFound) => {}
            Err(other) => panic!("unexpected failure mode: {other}"),
            Ok(found) => panic!("an absent index returned {} results", found.len()),
        }
    }
}
