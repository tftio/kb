//! Mail as a reference-only corpus.
//!
//! The Maildir stays canonical. What the PKB holds is a catalogue —
//! `Message-ID`, folder, content hash — and the index derived from it; no mail
//! bytes enter the content-addressed store. That is a deliberate refusal to
//! become the second owner of a corpus which already has a canonical store and
//! a good lexical index in mu/Xapian.
//!
//! Two properties of Maildir shape everything here.
//!
//! **Filenames are not identifiers.** A Maildir filename carries the message's
//! flags after a `:2,` suffix, so marking a message read renames its file. The
//! `Message-ID` is the identity; the portion of the filename before `:2,` is
//! merely a locator, stable across flag changes because only the suffix moves.
//!
//! **Resolution can legitimately fail.** Not owning the bytes means a citation
//! can go dead through no fault of the index — the message was deleted, or the
//! account expired. Failing honestly is therefore a behaviour of this module
//! rather than an error path through it: a catalogued message whose file is
//! gone reports [`CorpusError::Vanished`] naming the citation and where it was,
//! which is distinguishable both from an identifier that was never held and
//! from an empty result.

use crate::corpus::{Corpus, CorpusError, Page, Storage};
use crate::mail::header_value;
use crate::record::SourceRef;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// The identifier scheme this corpus answers to.
pub const SCHEME: &str = "message-id";

/// The corpus identifier, fixed by the ground-truth question set.
const CORPUS: &str = "mail";

/// What the catalogue records about one message.
///
/// Deliberately not the filename: flags rewrite it. `locator` is the stable
/// portion, and the hash is over delivered bytes so that drift means content
/// drift rather than somebody having read the message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogueEntry {
    /// Folder relative to the Maildir root, as the question set records it.
    pub folder: String,
    /// The filename up to the flag separator: stable across flag changes.
    pub locator: String,
    /// SHA-256 over the delivered message bytes, headers and body.
    pub content_sha256: String,
}

/// SHA-256 of delivered message bytes, hex-encoded.
///
/// The scope is the file's whole content and never its name, so the same
/// message synced to two machines hashes identically even when one has read it.
#[must_use]
pub fn content_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().fold(String::new(), |mut acc, byte| {
        use std::fmt::Write as _;
        let _ = write!(acc, "{byte:02x}");
        acc
    })
}

/// A Maildir rooted at one account, catalogued by `Message-ID`.
#[derive(Debug, Clone)]
pub struct MaildirCorpus {
    root: PathBuf,
    entries: BTreeMap<String, CatalogueEntry>,
    /// Where each message was when the scan walked it.
    ///
    /// Held separately from the catalogue because it is not catalogue data: a
    /// path is a fact about this machine at scan time, whereas `Message-ID`,
    /// folder and content hash are facts about the message. It exists so
    /// resolution does not have to list a 30,000-entry folder per lookup,
    /// which made a full pass quadratic and took six minutes.
    paths: BTreeMap<String, PathBuf>,
    unidentified: usize,
}

/// Split a Maildir filename into its stable locator and its flag suffix.
fn locator_of(filename: &str) -> &str {
    filename
        .split_once(":2,")
        .map_or(filename, |(base, _)| base)
}

fn unusable(reason: impl std::fmt::Display) -> CorpusError {
    CorpusError::Unusable {
        corpus: CORPUS.to_owned(),
        reason: reason.to_string(),
    }
}

impl MaildirCorpus {
    /// Catalogue every message under `root`.
    ///
    /// A folder is a Maildir folder when it holds `cur` or `new`; anything else
    /// under the root — mbsync state, `.uidvalidity` — is not mail. Messages
    /// carrying no `Message-ID` are counted rather than dropped silently, so
    /// the catalogue's size stays explainable against the folder's.
    ///
    /// # Errors
    ///
    /// [`CorpusError::Unusable`] if the root cannot be read, which is a
    /// misconfiguration and better said than reported as an empty mailbox.
    pub fn scan(root: &Path) -> Result<Self, CorpusError> {
        let mut entries = BTreeMap::new();
        let mut paths = BTreeMap::new();
        let mut unidentified = 0_usize;
        let folders =
            fs::read_dir(root).map_err(|e| unusable(format!("{}: {e}", root.display())))?;
        let mut names: Vec<String> = Vec::new();
        for folder in folders {
            let folder = folder.map_err(|e| unusable(format!("{}: {e}", root.display())))?;
            if folder.path().is_dir()
                && let Some(name) = folder.file_name().to_str()
            {
                names.push(name.to_owned());
            }
        }
        names.sort();
        for folder in names {
            for sub in ["cur", "new"] {
                let dir = root.join(&folder).join(sub);
                if !dir.is_dir() {
                    continue;
                }
                Self::catalogue_dir(&dir, &folder, &mut entries, &mut paths, &mut unidentified)?;
            }
        }
        Ok(Self {
            root: root.to_owned(),
            entries,
            paths,
            unidentified,
        })
    }

    fn catalogue_dir(
        dir: &Path,
        folder: &str,
        entries: &mut BTreeMap<String, CatalogueEntry>,
        paths: &mut BTreeMap<String, PathBuf>,
        unidentified: &mut usize,
    ) -> Result<(), CorpusError> {
        let listing = fs::read_dir(dir).map_err(|e| unusable(format!("{}: {e}", dir.display())))?;
        for message in listing {
            let message = message.map_err(|e| unusable(format!("{}: {e}", dir.display())))?;
            let path = message.path();
            if !path.is_file() {
                continue;
            }
            let Ok(bytes) = fs::read(&path) else {
                // A message that cannot be read is not a reason to abandon the
                // account; count it with the rest of what the scan could not
                // identify.
                *unidentified = unidentified.saturating_add(1);
                continue;
            };
            let text = String::from_utf8_lossy(&bytes);
            let head = text
                .split_once("\n\n")
                .map_or_else(|| text.as_ref(), |(h, _)| h);
            let message_id = header_value(head, "message-id");
            if message_id.is_empty() {
                *unidentified = unidentified.saturating_add(1);
                continue;
            }
            let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
                *unidentified = unidentified.saturating_add(1);
                continue;
            };
            paths.insert(message_id.clone(), path.clone());
            entries.insert(
                message_id,
                CatalogueEntry {
                    folder: folder.to_owned(),
                    locator: locator_of(filename).to_owned(),
                    content_sha256: content_sha256(&bytes),
                },
            );
        }
        Ok(())
    }

    /// What the catalogue holds for one `Message-ID`.
    #[must_use]
    pub fn catalogued(&self, message_id: &str) -> Option<&CatalogueEntry> {
        self.entries.get(message_id)
    }

    /// Every catalogued message, in `Message-ID` order.
    pub fn catalogue(&self) -> impl Iterator<Item = (&str, &CatalogueEntry)> {
        self.entries.iter().map(|(id, entry)| (id.as_str(), entry))
    }

    /// How many messages the catalogue holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the catalogue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many files the scan could not identify, so the catalogue's count is
    /// explainable against the folder's.
    #[must_use]
    pub const fn unidentified(&self) -> usize {
        self.unidentified
    }

    /// Find the file a catalogued message currently occupies.
    ///
    /// The path recorded at scan time is tried first and is right for every
    /// message nobody has touched since. When it is not — the flag suffix moves
    /// whenever a message is read, replied to or flagged, renaming the file —
    /// the folder is searched by locator, which is the portion of the filename
    /// flags do not touch. The fallback is the correctness argument and the
    /// recorded path is the speed one; both are needed.
    fn locate(&self, message_id: &str, entry: &CatalogueEntry) -> Option<PathBuf> {
        if let Some(path) = self.paths.get(message_id)
            && path.is_file()
        {
            return Some(path.clone());
        }
        for sub in ["cur", "new"] {
            let dir = self.root.join(&entry.folder).join(sub);
            let Ok(listing) = fs::read_dir(&dir) else {
                continue;
            };
            for candidate in listing.flatten() {
                let name = candidate.file_name();
                let Some(name) = name.to_str() else { continue };
                if locator_of(name) == entry.locator {
                    return Some(candidate.path());
                }
            }
        }
        None
    }
}

impl Corpus for MaildirCorpus {
    fn id(&self) -> &'static str {
        CORPUS
    }

    fn storage(&self) -> Storage {
        Storage::ReferenceOnly
    }

    fn resolve(&self, source: &SourceRef) -> Result<Vec<u8>, CorpusError> {
        if source.scheme() != SCHEME {
            return Err(unusable(format!(
                "mail resolves {SCHEME}, not {}",
                source.scheme()
            )));
        }
        let id = source.value();
        let Some(entry) = self.entries.get(id) else {
            return Err(CorpusError::NotFound {
                corpus: CORPUS.to_owned(),
                id: id.to_owned(),
            });
        };
        let Some(path) = self.locate(id, entry) else {
            return Err(CorpusError::Vanished {
                corpus: CORPUS.to_owned(),
                id: id.to_owned(),
                location: entry.folder.clone(),
            });
        };
        fs::read(&path).map_err(|e| CorpusError::Vanished {
            corpus: CORPUS.to_owned(),
            id: id.to_owned(),
            location: format!("{} ({e})", entry.folder),
        })
    }

    fn enumerate(&self, cursor: Option<&str>, limit: usize) -> Result<Page, CorpusError> {
        // The cursor is the last `Message-ID` returned rather than an offset:
        // a sync between calls inserts messages anywhere in the ordering, and
        // an offset would step over whatever landed before it.
        let start = cursor.map_or(0, |after| {
            self.entries
                .keys()
                .position(|id| id == after)
                .map_or(0, |i| i.saturating_add(1))
        });
        let mut ids = Vec::new();
        for message_id in self.entries.keys().skip(start).take(limit) {
            ids.push(SourceRef::new(SCHEME, message_id).map_err(unusable)?);
        }
        let consumed = start.saturating_add(ids.len());
        let next = (consumed < self.entries.len())
            .then(|| ids.last().map(|s| s.value().to_owned()))
            .flatten();
        Ok(Page { ids, next })
    }
}

/// The folders the first increment covers.
///
/// Drafts are unsent, Junk is mail a filter already rejected, and Deleted is
/// mail the operator has judged; none is correspondence the index should
/// answer from. The bound is a constant rather than a parameter because the
/// invariant is that widening the increment be a decision, not a drift.
pub const INCREMENT_FOLDERS: [&str; 2] = ["Inbox", "Archive"];

/// Which catalogued messages the increment indexes, and why the rest are out.
///
/// Every exclusion direction is counted separately because T017 has to be able
/// to state what the measured corpus actually was: the discriminant's
/// selection, plus the ground-truth overrides, minus what never resolved.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Selection {
    /// `Message-ID`s to index, in catalogue order.
    pub selected: Vec<String>,
    /// Ground-truth messages the discriminant classified as bulk and the
    /// override readmitted. Logged because they are the difference between the
    /// measured corpus and the rule's own selection.
    pub overrides: Vec<String>,
    /// How many messages the discriminant excluded.
    pub excluded: usize,
    /// Ground-truth messages sitting in a folder the increment does not cover.
    /// Their questions will fail, and the reason has to be legible as scope
    /// rather than read as a retrieval miss.
    pub out_of_scope: Vec<String>,
    /// Ground-truth messages the catalogue does not hold at all.
    pub missing_ground_truth: Vec<String>,
}

impl Selection {
    /// Decide the increment over a catalogued Maildir.
    ///
    /// The rule is biased toward inclusion — anything unreadable is kept — and
    /// the override is biased further: a message a ground-truth question names
    /// is indexed whatever the discriminant says, since a question whose
    /// answering message was cut measures the discriminant rather than
    /// retrieval.
    #[must_use]
    pub fn compute(
        corpus: &MaildirCorpus,
        rule: &crate::mail::BulkRule,
        ground_truth: &std::collections::BTreeSet<String>,
    ) -> Self {
        let mut selection = Self::default();
        for (message_id, entry) in corpus.catalogue() {
            let in_scope = INCREMENT_FOLDERS.contains(&entry.folder.as_str());
            let named = ground_truth.contains(message_id);
            if !in_scope {
                if named {
                    selection.out_of_scope.push(message_id.to_owned());
                }
                continue;
            }
            let bulk = matches!(
                corpus
                    .resolve(&match SourceRef::new(SCHEME, message_id) {
                        Ok(source) => source,
                        Err(_) => continue,
                    })
                    .map(|bytes| rule.classify(&String::from_utf8_lossy(&bytes))),
                Ok(crate::mail::Classification::Bulk { .. })
            );
            if bulk && !named {
                selection.excluded = selection.excluded.saturating_add(1);
                continue;
            }
            if bulk {
                selection.overrides.push(message_id.to_owned());
            }
            selection.selected.push(message_id.to_owned());
        }
        for named in ground_truth {
            if corpus.catalogued(named).is_none() {
                selection.missing_ground_truth.push(named.clone());
            }
        }
        selection
    }
}

/// What a ground-truth question expects of one message.
///
/// The hash is the reason this is recorded at all: mail is reference-only, so
/// the bytes a measurement ran against can change under it, and a figure
/// compared across two runs of drifted ground truth compares nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MailExpectation {
    /// Folder the question set found the message in.
    pub folder: String,
    /// SHA-256 the question set recorded for it.
    pub content_sha256: String,
}

/// One message whose bytes no longer hash to what the question set recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Drift {
    /// Which citation moved.
    pub message_id: String,
    /// What the question set recorded.
    pub expected: String,
    /// What the message hashes to now, or empty if it is no longer catalogued.
    pub found: String,
}

/// What a ground-truth question set expects of the mail corpus.
///
/// Read from the committed question file rather than passed in, so the
/// override set and the measurement's expectations cannot drift apart.
///
/// Parsed line-wise rather than through a TOML deserializer because the file
/// mixes two corpora under one `node_id` key and only the neighbouring
/// `corpus` line separates them; the ordering is what carries the meaning.
///
/// # Errors
///
/// [`CorpusError::Unusable`] if the file cannot be read.
pub fn ground_truth_expectations(
    questions: &Path,
) -> Result<BTreeMap<String, MailExpectation>, CorpusError> {
    let text = fs::read_to_string(questions)
        .map_err(|e| unusable(format!("{}: {e}", questions.display())))?;
    let mut found = BTreeMap::new();
    let mut in_mail = false;
    let mut id = String::new();
    let mut folder = String::new();
    let mut hash = String::new();
    let mut flush = |id: &mut String, folder: &mut String, hash: &mut String| {
        if !id.is_empty() {
            found.insert(
                std::mem::take(id),
                MailExpectation {
                    folder: std::mem::take(folder),
                    content_sha256: std::mem::take(hash),
                },
            );
        }
        folder.clear();
        hash.clear();
    };
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("[[") {
            flush(&mut id, &mut folder, &mut hash);
            in_mail = false;
        }
        if let Some(rest) = trimmed.strip_prefix("corpus = ") {
            in_mail = rest.trim_matches('"') == CORPUS;
        }
        if !in_mail {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("node_id = ") {
            rest.trim().trim_matches('"').clone_into(&mut id);
        } else if let Some(rest) = trimmed.strip_prefix("folder = ") {
            rest.trim().trim_matches('"').clone_into(&mut folder);
        } else if let Some(rest) = trimmed.strip_prefix("content_sha256 = ") {
            rest.trim().trim_matches('"').clone_into(&mut hash);
        }
    }
    flush(&mut id, &mut folder, &mut hash);
    Ok(found)
}

/// The `Message-ID`s a ground-truth question set names for the mail corpus.
///
/// # Errors
///
/// [`CorpusError::Unusable`] if the file cannot be read.
pub fn ground_truth_ids(
    questions: &Path,
) -> Result<std::collections::BTreeSet<String>, CorpusError> {
    Ok(ground_truth_expectations(questions)?.into_keys().collect())
}

/// Every ground-truth message whose bytes no longer match what was recorded.
///
/// Reported per message rather than as a count, because the useful question is
/// which ground truth moved — a drifted expectation invalidates one figure,
/// not the run.
#[must_use]
pub fn verify(
    corpus: &MaildirCorpus,
    expectations: &BTreeMap<String, MailExpectation>,
) -> Vec<Drift> {
    expectations
        .iter()
        .filter_map(|(message_id, expected)| {
            let found = corpus
                .catalogued(message_id)
                .map_or(String::new(), |e| e.content_sha256.clone());
            (found != expected.content_sha256).then(|| Drift {
                message_id: message_id.clone(),
                expected: expected.content_sha256.clone(),
                found,
            })
        })
        .collect()
}

/// Group messages into threads by their identifier headers.
///
/// A thread is the transitive closure of `In-Reply-To` and `References` over
/// the messages present. Only messages in the increment participate: a reply
/// to something the discriminant excluded joins no thread rather than
/// dragging the excluded message in, since the scope bound is not the
/// question set's to widen and not a reply's either.
///
/// Returned keyed by the root — the earliest member by delivery order — with
/// members in that order, because a thread read out of order is a different
/// document from the one a reader would recognize.
#[must_use]
pub fn threads(
    messages: &[crate::message::Readable],
) -> Vec<(String, Vec<crate::message::Readable>)> {
    let present: BTreeMap<&str, usize> = messages
        .iter()
        .enumerate()
        .map(|(at, message)| (message.message_id.as_str(), at))
        .collect();
    // Union-find over positions. Threads are small and the corpus is one
    // pass, so the naive path-halving version is ample.
    let mut parent: Vec<usize> = (0..messages.len()).collect();
    let find = |parent: &mut Vec<usize>, start: usize| -> usize {
        let mut node = start;
        while parent.get(node).copied().unwrap_or(node) != node {
            let grand = parent
                .get(parent.get(node).copied().unwrap_or(node))
                .copied()
                .unwrap_or(node);
            if let Some(slot) = parent.get_mut(node) {
                *slot = grand;
            }
            node = grand;
        }
        node
    };
    for (at, message) in messages.iter().enumerate() {
        let referenced = message
            .references
            .iter()
            .map(String::as_str)
            .chain(std::iter::once(message.in_reply_to.as_str()));
        for other in referenced {
            let Some(&position) = present.get(other) else {
                continue;
            };
            let (a, b) = (find(&mut parent, at), find(&mut parent, position));
            if a != b
                && let Some(slot) = parent.get_mut(a)
            {
                *slot = b;
            }
        }
    }
    let mut grouped: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for at in 0..messages.len() {
        grouped.entry(find(&mut parent, at)).or_default().push(at);
    }
    grouped
        .into_values()
        .filter_map(|members| {
            // Chronological, not catalogue order: members arrive sorted by
            // Message-ID, which is arbitrary with respect to time, and a
            // conversation read out of order is a different document from the
            // one a reader would recognize. The identifier breaks ties so the
            // assembly is deterministic when two messages share a second.
            let mut collected: Vec<crate::message::Readable> = members
                .iter()
                .filter_map(|&at| messages.get(at).cloned())
                .collect();
            collected.sort_by(|a, b| {
                a.epoch
                    .cmp(&b.epoch)
                    .then_with(|| a.message_id.cmp(&b.message_id))
            });
            let root = collected.first()?.message_id.clone();
            Some((root, collected))
        })
        .collect()
}

/// Render a thread as an mbox, which is the form the thread normalizer reads.
///
/// The `From ` envelope is what separates messages in an mbox and therefore
/// what the thread's per-message spans are computed from.
#[must_use]
pub fn as_mbox(messages: &[crate::message::Readable]) -> String {
    let mut out = String::new();
    for message in messages {
        out.push_str("From ");
        out.push_str(if message.from.is_empty() {
            "unknown"
        } else {
            message.from.as_str()
        });
        out.push(' ');
        out.push_str(&message.date);
        out.push('\n');
        out.push_str(&message.render());
    }
    out
}
