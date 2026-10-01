//! Deciding which mail enters the corpus.
//!
//! A mailbox of a quarter of a million messages cannot give each one an
//! embedding and a generated summary at any tolerable cost, so the first
//! increment is bounded to human-authored mail. "Human-authored" is the
//! property wanted;
//! "carries no bulk header" is the computable approximation of it, and the
//! gap between them is what T018 measures rather than assumes.
//!
//! The two error directions are not symmetric, and the whole design follows
//! from that. Retaining a newsletter costs a little noise in the index.
//! Dropping real correspondence removes ground truth silently — a question
//! whose answering message was never indexed measures the discriminant rather
//! than retrieval — so the rule is biased toward inclusion: anything it cannot
//! read is kept.

use std::collections::BTreeSet;

/// Header names that mark automated mail.
///
/// On a typical business mailbox most messages carry at least one of these,
/// `List-Id` and `List-Unsubscribe` being the most common and
/// `Auto-Submitted` the rarest.
pub const DEFAULT_SIGNALS: [&str; 4] = [
    "list-unsubscribe",
    "list-id",
    "precedence",
    "auto-submitted",
];

/// What the discriminant decided about one message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Classification {
    /// Automated: the named header signal fired.
    Bulk {
        /// Which configured signal matched, so a false negative can be traced
        /// to the rule that caused it rather than guessed at.
        signal: String,
    },
    /// Kept. Either no signal fired, or the message could not be read — the
    /// bias toward inclusion makes those the same answer.
    NonBulk,
}

/// A configured bulk-mail discriminant.
///
/// The signal set is data rather than code because the rule's error rate is
/// the deliverable: tuning it and re-measuring must not require a rebuild.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BulkRule {
    signals: BTreeSet<String>,
}

impl Default for BulkRule {
    fn default() -> Self {
        Self::from_signals(DEFAULT_SIGNALS)
    }
}

impl BulkRule {
    /// Build a rule over the given header names, matched case-insensitively.
    pub fn from_signals<I, S>(signals: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self {
            signals: signals
                .into_iter()
                .map(|s| s.as_ref().to_ascii_lowercase())
                .collect(),
        }
    }

    /// The configured signal names, for reporting what a run was measured
    /// under.
    #[must_use]
    pub fn signals(&self) -> Vec<&str> {
        self.signals.iter().map(String::as_str).collect()
    }

    /// Classify one message from its raw bytes as text.
    ///
    /// Only the header block is consulted, and only lines that begin a
    /// header: a continuation line belongs to the value above it, and a bulk
    /// signal quoted in the body is somebody talking about a newsletter
    /// rather than a newsletter.
    #[must_use]
    pub fn classify(&self, message: &str) -> Classification {
        let head = message.split_once("\n\n").map_or(message, |(head, _)| head);
        for line in head.lines() {
            if line.starts_with([' ', '\t']) {
                continue;
            }
            let Some((name, _)) = line.split_once(':') else {
                continue;
            };
            let name = name.trim().to_ascii_lowercase();
            if self.signals.contains(&name) {
                return Classification::Bulk { signal: name };
            }
        }
        Classification::NonBulk
    }
}

/// One classified message, as a sampling pass needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifiedMessage {
    /// `Message-ID`, the only stable identifier: Maildir flags rewrite
    /// filenames, so a filename cannot name a message across two syncs.
    pub message_id: String,
    /// Folder the message was found in, relative to the Maildir root.
    pub folder: String,
    /// What the rule decided.
    pub classification: Classification,
    /// `From` header, for a human judging whether the rule erred.
    pub from: String,
    /// `Subject` header, likewise.
    pub subject: String,
    /// `Date` header, so a sample can be stratified across years.
    pub date: String,
}

/// Read one header's unfolded value from a header block.
#[must_use]
pub fn header_value(head: &str, name: &str) -> String {
    let wanted = name.to_ascii_lowercase();
    let mut lines = head.lines();
    let Some(mut value) = lines.find_map(|line| {
        let (n, rest) = line.split_once(':')?;
        n.trim()
            .eq_ignore_ascii_case(&wanted)
            .then(|| rest.trim().to_owned())
    }) else {
        return String::new();
    };
    for continuation in lines {
        if !continuation.starts_with([' ', '\t']) {
            break;
        }
        value.push(' ');
        value.push_str(continuation.trim());
    }
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl BulkRule {
    /// Classify one message and extract what a reviewer needs to judge it.
    #[must_use]
    pub fn describe(&self, message: &str, folder: &str) -> ClassifiedMessage {
        let head = message.split_once("\n\n").map_or(message, |(h, _)| h);
        ClassifiedMessage {
            message_id: header_value(head, "message-id"),
            folder: folder.to_owned(),
            classification: self.classify(message),
            from: header_value(head, "from"),
            subject: header_value(head, "subject"),
            date: header_value(head, "date"),
        }
    }
}
