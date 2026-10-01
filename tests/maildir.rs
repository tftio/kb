//! Behaviour of the Maildir reference corpus.
//!
//! Mail is reference-only: the Maildir stays canonical, the PKB holds a
//! catalogue and a derived index, and no mail bytes enter the store. Two
//! consequences drive most of what is asserted here. Identity is the
//! `Message-ID`, because Maildir encodes mutable flags in the filename and a
//! read message is a renamed file. And resolution can legitimately fail, so
//! failing honestly — naming the citation that no longer resolves — is a
//! behaviour rather than an error path.

use kb::corpus::{Corpus, CorpusError, Storage};
use kb::maildir::MaildirCorpus;
use kb::record::SourceRef;
use std::fs;
use std::path::Path;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Write one message into `folder`, returning the file it landed in.
fn deliver(
    root: &Path,
    folder: &str,
    uniq: &str,
    flags: &str,
    message_id: &str,
    body: &str,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let dir = root.join(folder).join("cur");
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{uniq}:2,{flags}"));
    let message = format!(
        "From: someone@example.invalid\n\
         To: reader@example.invalid\n\
         Subject: about {uniq}\n\
         Date: Mon, 3 Feb 2025 09:00:00 +0000\n\
         Message-ID: {message_id}\n\
         \n\
         {body}\n"
    );
    fs::write(&path, message)?;
    Ok(path)
}

fn corpus_with_one(dir: &tempfile::TempDir) -> Result<MaildirCorpus, Box<dyn std::error::Error>> {
    deliver(
        dir.path(),
        "Archive",
        "1772827811.39501_18990.mailhost,U=709",
        "S",
        "<a@example.invalid>",
        "the body",
    )?;
    Ok(MaildirCorpus::scan(dir.path())?)
}

/// The declaration is the point: a reference that stops resolving is expected
/// here and would be corruption in the store.
#[test]
fn the_mail_corpus_declares_itself_reference_only() -> TestResult {
    let dir = tempfile::tempdir()?;
    let corpus = corpus_with_one(&dir)?;

    assert_eq!(corpus.id(), "mail");
    assert_eq!(corpus.storage(), Storage::ReferenceOnly);
    Ok(())
}

#[test]
fn it_resolves_a_message_by_message_id() -> TestResult {
    let dir = tempfile::tempdir()?;
    let corpus = corpus_with_one(&dir)?;

    let bytes = corpus.resolve(&SourceRef::new("message-id", "<a@example.invalid>")?)?;

    assert!(String::from_utf8(bytes)?.contains("the body"));
    Ok(())
}

/// Flags live in the filename, so marking a message read renames its file.
/// A catalogue that recorded the filename would lose the message; one that
/// records the `Message-ID` does not.
#[test]
fn a_reflagged_message_still_resolves_with_its_hash_unchanged() -> TestResult {
    let dir = tempfile::tempdir()?;
    let path = deliver(
        dir.path(),
        "Inbox",
        "1786752334.91077_12.mailhost,U=817",
        "",
        "<b@example.invalid>",
        "unread when catalogued",
    )?;
    let corpus = MaildirCorpus::scan(dir.path())?;
    let before = corpus
        .catalogued("<b@example.invalid>")
        .map(ToOwned::to_owned);

    fs::rename(
        &path,
        path.with_file_name("1786752334.91077_12.mailhost,U=817:2,S"),
    )?;

    let bytes = corpus.resolve(&SourceRef::new("message-id", "<b@example.invalid>")?)?;
    assert!(String::from_utf8(bytes)?.contains("unread when catalogued"));
    assert_eq!(
        before.as_ref().map(|e| e.content_sha256.as_str()),
        corpus
            .catalogued("<b@example.invalid>")
            .map(|e| e.content_sha256.as_str()),
        "the hash covers delivered bytes, so a flag change must not move it"
    );
    Ok(())
}

/// The invariant this corpus exists to honour. The PKB does not own these
/// bytes, so when they go it says which citation went — not an error that
/// aborts a query, and not an empty result indistinguishable from "nothing
/// was ever there".
#[test]
fn a_deleted_message_reports_an_unresolvable_citation_naming_it() -> TestResult {
    let dir = tempfile::tempdir()?;
    let path = deliver(
        dir.path(),
        "Archive",
        "1772827811.39501_18990.mailhost,U=709",
        "S",
        "<gone@example.invalid>",
        "here for now",
    )?;
    let corpus = MaildirCorpus::scan(dir.path())?;
    fs::remove_file(&path)?;

    let outcome = corpus.resolve(&SourceRef::new("message-id", "<gone@example.invalid>")?);

    assert!(
        matches!(
            &outcome,
            Err(CorpusError::Vanished { id, location, .. })
                if id == "<gone@example.invalid>" && location == "Archive"
        ),
        "outcome was: {outcome:?}"
    );
    Ok(())
}

/// Distinct from a vanished one: this citation was never valid, so reporting
/// it as a deletion would invent a history.
#[test]
fn an_uncataloged_message_id_is_not_found_rather_than_vanished() -> TestResult {
    let dir = tempfile::tempdir()?;
    let corpus = corpus_with_one(&dir)?;

    let outcome = corpus.resolve(&SourceRef::new("message-id", "<never@example.invalid>")?);

    assert!(
        matches!(&outcome, Err(CorpusError::NotFound { id, .. }) if id == "<never@example.invalid>"),
        "outcome was: {outcome:?}"
    );
    Ok(())
}

/// A `node-id` reaching the mail corpus is a caller mixing up corpora, which
/// is worth saying rather than silently missing.
#[test]
fn a_foreign_identifier_scheme_is_refused() -> TestResult {
    let dir = tempfile::tempdir()?;
    let corpus = corpus_with_one(&dir)?;

    let outcome = corpus.resolve(&SourceRef::new("node-id", "20260817T101500")?);

    assert!(
        matches!(&outcome, Err(CorpusError::Unusable { reason, .. }) if reason.contains("node-id")),
        "outcome was: {outcome:?}"
    );
    Ok(())
}

#[test]
fn enumeration_pages_without_repeating_or_skipping() -> TestResult {
    let dir = tempfile::tempdir()?;
    for n in 0..7_u32 {
        deliver(
            dir.path(),
            "Archive",
            &format!("1772827811.{n}.mailhost,U=70{n}"),
            "S",
            &format!("<m{n}@example.invalid>"),
            "body",
        )?;
    }
    let corpus = MaildirCorpus::scan(dir.path())?;

    let mut seen = Vec::new();
    let mut cursor = None;
    loop {
        let page = corpus.enumerate(cursor.as_deref(), 3)?;
        seen.extend(page.ids.iter().map(|s| s.value().to_owned()));
        match page.next {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }

    let mut unique = seen.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(seen.len(), 7, "walk repeated or skipped: {seen:?}");
    assert_eq!(unique.len(), 7);
    Ok(())
}

/// Both Maildir subdirectories hold delivered mail; `new` is merely mail no
/// client has filed yet.
#[test]
fn messages_awaiting_filing_in_new_are_catalogued() -> TestResult {
    let dir = tempfile::tempdir()?;
    let folder = dir.path().join("Inbox").join("new");
    fs::create_dir_all(&folder)?;
    fs::write(
        folder.join("1787067912.58046_25.mailhost,U=831"),
        "Message-ID: <fresh@example.invalid>\n\nnot yet filed\n",
    )?;
    let corpus = MaildirCorpus::scan(dir.path())?;

    let bytes = corpus.resolve(&SourceRef::new("message-id", "<fresh@example.invalid>")?)?;

    assert!(String::from_utf8(bytes)?.contains("not yet filed"));
    Ok(())
}

/// The catalogue is what the index stores, so folder and content hash have to
/// come out of the scan rather than be recomputed at query time.
#[test]
fn the_catalogue_records_folder_and_content_hash() -> TestResult {
    let dir = tempfile::tempdir()?;
    let path = deliver(
        dir.path(),
        "Archive",
        "1772827811.39501_18990.mailhost,U=709",
        "S",
        "<c@example.invalid>",
        "catalogued",
    )?;
    let corpus = MaildirCorpus::scan(dir.path())?;

    let entry = corpus
        .catalogued("<c@example.invalid>")
        .ok_or("message was not catalogued")?;

    assert_eq!(entry.folder, "Archive");
    let expected = kb::maildir::content_sha256(&fs::read(&path)?);
    assert_eq!(entry.content_sha256, expected);
    assert_eq!(entry.content_sha256.len(), 64);
    Ok(())
}

/// Real Maildirs contain files without a `Message-ID`. Dropping them silently
/// would make the catalogue's count unexplainable against the folder's.
#[test]
fn a_message_without_an_identifier_is_skipped_and_counted() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(
        dir.path(),
        "Archive",
        "1772827811.1.mailhost,U=1",
        "S",
        "<ok@example.invalid>",
        "fine",
    )?;
    let folder = dir.path().join("Archive").join("cur");
    fs::write(folder.join("1772827811.2.mailhost,U=2:2,S"), "no headers\n")?;
    let corpus = MaildirCorpus::scan(dir.path())?;

    assert_eq!(corpus.len(), 1);
    assert_eq!(corpus.unidentified(), 1);
    Ok(())
}

/// A folder that is not a Maildir — mbsync state files, `.uidvalidity` — is
/// not mail and must not become a catalogue entry.
#[test]
fn a_root_with_no_maildir_folders_catalogues_nothing() -> TestResult {
    let dir = tempfile::tempdir()?;
    fs::write(dir.path().join(".uidvalidity"), "1\n")?;

    let corpus = MaildirCorpus::scan(dir.path())?;

    assert_eq!(corpus.len(), 0);
    Ok(())
}

/// A root that does not exist is a misconfiguration, and saying so beats
/// reporting an empty mailbox.
#[test]
fn a_missing_root_is_unusable_rather_than_empty() -> TestResult {
    let dir = tempfile::tempdir()?;

    let outcome = MaildirCorpus::scan(&dir.path().join("absent"));

    assert!(
        matches!(&outcome, Err(CorpusError::Unusable { corpus, .. }) if corpus == "mail"),
        "outcome was: {outcome:?}"
    );
    Ok(())
}

// --- Selecting the increment -------------------------------------------------

use kb::maildir::{INCREMENT_FOLDERS, Selection};
use std::collections::BTreeSet;

fn bulk(
    root: &Path,
    folder: &str,
    uniq: &str,
    message_id: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = root.join(folder).join("cur");
    fs::create_dir_all(&dir)?;
    fs::write(
        dir.join(format!("{uniq}:2,S")),
        format!(
            "From: news@example.invalid\n\
             List-Id: <announce.example.invalid>\n\
             Message-ID: {message_id}\n\
             \n\
             a newsletter\n"
        ),
    )?;
    Ok(())
}

/// The increment is Inbox and Archive. Drafts are unsent, Junk is filtered
/// mail and Deleted is mail the operator has already judged — indexing any of
/// them would widen the increment by accident.
#[test]
fn the_increment_covers_only_the_two_named_folders() -> TestResult {
    assert_eq!(INCREMENT_FOLDERS, ["Inbox", "Archive"]);
    let dir = tempfile::tempdir()?;
    deliver(
        dir.path(),
        "Inbox",
        "u1",
        "S",
        "<in@example.invalid>",
        "kept",
    )?;
    deliver(
        dir.path(),
        "Archive",
        "u2",
        "S",
        "<ar@example.invalid>",
        "kept",
    )?;
    for folder in ["Drafts", "Junk", "Deleted", "Notes"] {
        deliver(
            dir.path(),
            folder,
            &format!("u-{folder}"),
            "S",
            &format!("<{folder}@example.invalid>"),
            "out of scope",
        )?;
    }
    let corpus = MaildirCorpus::scan(dir.path())?;

    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &BTreeSet::new());

    assert_eq!(
        selection.selected,
        vec![
            "<ar@example.invalid>".to_owned(),
            "<in@example.invalid>".to_owned()
        ]
    );
    Ok(())
}

#[test]
fn bulk_mail_is_excluded_and_the_count_reported() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(
        dir.path(),
        "Inbox",
        "u1",
        "S",
        "<human@example.invalid>",
        "hi",
    )?;
    bulk(dir.path(), "Inbox", "u2", "<news@example.invalid>")?;
    let corpus = MaildirCorpus::scan(dir.path())?;

    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &BTreeSet::new());

    assert_eq!(
        selection.selected,
        vec!["<human@example.invalid>".to_owned()]
    );
    assert_eq!(selection.excluded, 1);
    Ok(())
}

/// Without this, a ground-truth question naming a message the discriminant
/// cut is unanswerable by construction: it would measure the discriminant
/// rather than retrieval. T014 requires at least one such question by design,
/// to exercise the false-negative direction.
#[test]
fn a_ground_truth_message_survives_the_discriminant_and_is_logged() -> TestResult {
    let dir = tempfile::tempdir()?;
    bulk(dir.path(), "Archive", "u1", "<needed@example.invalid>")?;
    let corpus = MaildirCorpus::scan(dir.path())?;
    let ground_truth = BTreeSet::from(["<needed@example.invalid>".to_owned()]);

    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &ground_truth);

    assert_eq!(
        selection.selected,
        vec!["<needed@example.invalid>".to_owned()]
    );
    assert_eq!(
        selection.overrides,
        vec!["<needed@example.invalid>".to_owned()],
        "every override is logged, so T017 can state what fraction of the \
         measured corpus the discriminant did not select"
    );
    assert_eq!(selection.excluded, 0);
    Ok(())
}

/// The override readmits messages the rule cut; it must not reach into
/// folders the increment excludes, or the scope bound would be decided by the
/// question set rather than by the increment.
#[test]
fn the_override_does_not_widen_the_folder_scope() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(dir.path(), "Junk", "u1", "S", "<spam@example.invalid>", "x")?;
    let corpus = MaildirCorpus::scan(dir.path())?;
    let ground_truth = BTreeSet::from(["<spam@example.invalid>".to_owned()]);

    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &ground_truth);

    assert!(selection.selected.is_empty());
    assert_eq!(
        selection.out_of_scope,
        vec!["<spam@example.invalid>".to_owned()],
        "a ground-truth message outside the increment is reported, not \
         silently absent: its question will fail and the reason must be legible"
    );
    Ok(())
}

/// A question naming a message no longer in the Maildir has to be visible,
/// since the alternative is a retrieval figure quietly measuring absence.
#[test]
fn a_ground_truth_message_absent_from_the_catalogue_is_reported() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(
        dir.path(),
        "Inbox",
        "u1",
        "S",
        "<here@example.invalid>",
        "x",
    )?;
    let corpus = MaildirCorpus::scan(dir.path())?;
    let ground_truth = BTreeSet::from(["<vanished@example.invalid>".to_owned()]);

    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &ground_truth);

    assert_eq!(
        selection.missing_ground_truth,
        vec!["<vanished@example.invalid>".to_owned()]
    );
    Ok(())
}

/// The override set comes out of the committed question file, so it cannot
/// drift from what the measurement expects. Reading the real file rather than
/// a fixture is the point: a fixture would encode this test's assumptions
/// about the format instead of the format.
#[test]
fn ground_truth_ids_come_from_the_committed_question_set() -> TestResult {
    let ids = kb::maildir::ground_truth_ids(Path::new("resources/eval/retrieval-questions.toml"))?;

    assert_eq!(ids.len(), 12, "the committed set names 12 mail messages");
    assert!(
        ids.iter()
            .all(|id| id.starts_with('<') && id.ends_with('>'))
    );
    assert!(ids.contains("<synthetic-01.02cf89023e67@example.invalid>"));
    Ok(())
}

/// kb node ids share the `node_id` key with mail Message-IDs; only the
/// `corpus` line separates them, and picking up a kb id would silently widen
/// the override set.
#[test]
fn ground_truth_ids_exclude_the_kb_corpus() -> TestResult {
    let ids = kb::maildir::ground_truth_ids(Path::new("resources/eval/retrieval-questions.toml"))?;

    assert!(
        !ids.iter().any(|id| !id.contains('@')),
        "a kb node id leaked into the mail override set: {ids:?}"
    );
    Ok(())
}

/// The question set records a content hash per expected message. Reading it
/// back is what makes drift detectable rather than silent: if a message's
/// bytes change, the ground truth it anchors is no longer the ground truth
/// that was measured.
#[test]
fn expectations_carry_the_folder_and_hash_the_question_set_recorded() -> TestResult {
    let expectations = kb::maildir::ground_truth_expectations(Path::new(
        "resources/eval/retrieval-questions.toml",
    ))?;

    let entry = expectations
        .get("<synthetic-01.02cf89023e67@example.invalid>")
        .ok_or("expectation missing")?;
    assert_eq!(entry.folder, "Archive");
    assert_eq!(
        entry.content_sha256,
        "5165c4fd9b6989eddeec4718c95dbf6408f92901218d0473c8df1ebd77246d99"
    );
    assert_eq!(expectations.len(), 12);
    Ok(())
}

/// Drift is reported per message rather than as a count, because the useful
/// question is which ground truth moved.
#[test]
fn a_changed_message_is_reported_as_drift() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(
        dir.path(),
        "Inbox",
        "u1",
        "S",
        "<d@example.invalid>",
        "before",
    )?;
    let corpus = MaildirCorpus::scan(dir.path())?;
    let expectations = std::collections::BTreeMap::from([(
        "<d@example.invalid>".to_owned(),
        kb::maildir::MailExpectation {
            folder: "Inbox".to_owned(),
            content_sha256: "0".repeat(64),
        },
    )]);

    let drift = kb::maildir::verify(&corpus, &expectations);

    assert_eq!(drift.len(), 1);
    let reported = drift.first().ok_or("no drift reported")?;
    assert_eq!(reported.message_id, "<d@example.invalid>");
    assert_eq!(reported.expected, "0".repeat(64));
    assert_eq!(
        reported.found,
        corpus
            .catalogued("<d@example.invalid>")
            .ok_or("no entry")?
            .content_sha256
    );
    Ok(())
}

/// A message whose bytes still hash the same is not drift, whatever its flags
/// now say.
#[test]
fn an_unchanged_message_is_not_reported_as_drift() -> TestResult {
    let dir = tempfile::tempdir()?;
    let path = deliver(
        dir.path(),
        "Inbox",
        "u1",
        "",
        "<s@example.invalid>",
        "steady",
    )?;
    let corpus = MaildirCorpus::scan(dir.path())?;
    let expectations = std::collections::BTreeMap::from([(
        "<s@example.invalid>".to_owned(),
        kb::maildir::MailExpectation {
            folder: "Inbox".to_owned(),
            content_sha256: kb::maildir::content_sha256(&fs::read(&path)?),
        },
    )]);

    assert!(kb::maildir::verify(&corpus, &expectations).is_empty());
    Ok(())
}

/// mbsync and mu leave state directories inside a Maildir. A directory is not
/// a message, and treating one as an unreadable message would inflate the
/// count of what the scan could not identify.
#[test]
fn a_directory_inside_cur_is_not_a_message() -> TestResult {
    let dir = tempfile::tempdir()?;
    deliver(
        dir.path(),
        "Inbox",
        "u1",
        "S",
        "<m@example.invalid>",
        "body",
    )?;
    fs::create_dir_all(dir.path().join("Inbox").join("cur").join("tmp.state"))?;

    let corpus = MaildirCorpus::scan(dir.path())?;

    assert_eq!(corpus.len(), 1);
    assert_eq!(
        corpus.unidentified(),
        0,
        "a directory is not an unread message"
    );
    Ok(())
}

/// One unreadable file is not a reason to abandon an account, but it is a
/// reason to say the catalogue is smaller than the folder.
#[test]
fn an_unreadable_message_is_counted_rather_than_fatal() -> TestResult {
    use std::os::unix::fs::PermissionsExt as _;
    let dir = tempfile::tempdir()?;
    deliver(
        dir.path(),
        "Inbox",
        "u1",
        "S",
        "<m@example.invalid>",
        "body",
    )?;
    let blocked = deliver(dir.path(), "Inbox", "u2", "S", "<n@example.invalid>", "x")?;
    fs::set_permissions(&blocked, fs::Permissions::from_mode(0o000))?;
    if fs::File::open(&blocked).is_ok() {
        // Root, and some CI sandboxes, read a mode-000 file regardless. The
        // assertion below is about an unreadable file, so it is unreachable
        // here rather than wrong; restore the mode and stop.
        fs::set_permissions(&blocked, fs::Permissions::from_mode(0o600))?;
        return Ok(());
    }

    let corpus = MaildirCorpus::scan(dir.path())?;

    assert_eq!(corpus.len(), 1);
    assert_eq!(corpus.unidentified(), 1);
    fs::set_permissions(&blocked, fs::Permissions::from_mode(0o600))?;
    Ok(())
}

/// An empty account is a legitimate state — a new mailbox, or one whose
/// folders have not synced yet — and is not the same as a missing root.
#[test]
fn an_empty_account_is_empty_rather_than_unusable() -> TestResult {
    let dir = tempfile::tempdir()?;
    fs::create_dir_all(dir.path().join("Inbox").join("cur"))?;

    let corpus = MaildirCorpus::scan(dir.path())?;

    assert!(corpus.is_empty());
    assert_eq!(corpus.len(), 0);
    Ok(())
}

/// A thread is a conversation, and a conversation read out of order is a
/// different document from the one a reader would recognize. Members arrive
/// from the catalogue in `Message-ID` order, which is arbitrary with respect
/// to time, so ordering has to be imposed.
#[test]
fn a_thread_reads_in_the_order_it_happened() -> TestResult {
    let dir = tempfile::tempdir()?;
    // Identifiers deliberately sort against chronology: 'z' arrived first.
    let dir_path = dir.path();
    let messages = [
        (
            "z",
            "<zfirst@example.invalid>",
            "Mon, 3 Feb 2025 09:00:00 +0000",
            "",
            "the opening ask",
        ),
        (
            "a",
            "<asecond@example.invalid>",
            "Mon, 3 Feb 2025 11:00:00 +0000",
            "In-Reply-To: <zfirst@example.invalid>\n",
            "the later reply",
        ),
    ];
    for (uniq, id, date, extra, body) in messages {
        let folder = dir_path.join("Inbox").join("cur");
        fs::create_dir_all(&folder)?;
        fs::write(
            folder.join(format!("{uniq}:2,S")),
            format!(
                "From: Ada <ada@example.invalid>\nTo: reader@example.invalid\n\
                 Subject: thread\nDate: {date}\nMessage-ID: {id}\n{extra}\n{body}\n"
            ),
        )?;
    }
    let corpus = MaildirCorpus::scan(dir_path)?;
    let read: Vec<_> = corpus
        .catalogue()
        .filter_map(|(id, _)| {
            corpus
                .resolve(&SourceRef::new("message-id", id).ok()?)
                .ok()
                .map(|bytes| kb::message::parse(&bytes))
        })
        .collect();

    let threads = kb::maildir::threads(&read);

    let (root, members) = threads.first().ok_or("no thread")?;
    assert_eq!(
        root, "<zfirst@example.invalid>",
        "the root is not the earliest"
    );
    assert_eq!(
        members
            .iter()
            .map(|m| m.message_id.as_str())
            .collect::<Vec<_>>(),
        vec!["<zfirst@example.invalid>", "<asecond@example.invalid>"],
        "the thread was assembled in identifier order rather than in time order"
    );
    let mbox = kb::maildir::as_mbox(members);
    assert!(
        mbox.find("the opening ask") < mbox.find("the later reply"),
        "the rendered thread is out of order"
    );
    Ok(())
}
