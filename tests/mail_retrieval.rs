//! Mail retrieval through the planner, against a real mu index (T026).
//!
//! The mail corpus is where the planner earns its shape. Its lexical signal
//! is not kb's FTS5 — mail passages are deliberately kept out of it — but mu's
//! Xapian index, which already knows senders, folders, dates and exact
//! identifiers. Its dense signal is the derived index's vectors. Neither knows
//! about the other, and the planner is the only thing that does.
//!
//! These tests build a fixture Maildir, index it with the real `mu`, index the
//! same Maildir into a real derived index, and assert that a query reaches
//! both and comes back fused. Where `mu` is not installed the mu-dependent
//! tests report that and stop, following `markdown.rs`'s precedent for
//! `pandoc`: a machine without the binary should not fail a suite for a
//! dependency the code correctly reports as absent.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

use kb::embedding::encode_embedding;
use kb::index::{Index, rebuild_mail};
use kb::maildir::{MaildirCorpus, Selection};
use kb::mu::{MailSearch, MuIndex};
use kb::retrieval::{DocumentSource, IndexDense, MailLexical, Planner, Query, Signal, SignalKind};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Whether `mu` is installed.
fn mu_available() -> bool {
    Command::new("mu")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn deliver(root: &Path, uniq: &str, message_id: &str, subject: &str, body: &str) -> TestResult {
    let dir = root.join("Inbox").join("cur");
    fs::create_dir_all(&dir)?;
    fs::create_dir_all(root.join("Inbox").join("new"))?;
    fs::create_dir_all(root.join("Inbox").join("tmp"))?;
    fs::write(
        dir.join(format!("{uniq}:2,S")),
        format!(
            "From: Ada <ada@example.invalid>\n\
             To: reader@example.invalid\n\
             Subject: {subject}\n\
             Date: Mon, 3 Feb 2025 09:00:00 +0000\n\
             Message-ID: {message_id}\n\
             \n\
             {body}\n"
        ),
    )?;
    Ok(())
}

/// A fixture Maildir holding three messages on distinct subjects.
fn fixture_maildir() -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    deliver(
        dir.path(),
        "one",
        "<one@example.invalid>",
        "quarterly revenue figures",
        "the revenue figures for the quarter are attached",
    )?;
    deliver(
        dir.path(),
        "two",
        "<two@example.invalid>",
        "lunch on thursday",
        "shall we get lunch on thursday",
    )?;
    deliver(
        dir.path(),
        "three",
        "<three@example.invalid>",
        "revenue review meeting",
        "a meeting to review the revenue numbers",
    )?;
    Ok(dir)
}

/// Build a mu index over `maildir`, returning the muhome it lives in.
fn mu_index(maildir: &Path) -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let home = tempfile::tempdir()?;
    let init = Command::new("mu")
        .arg("init")
        .arg("--muhome")
        .arg(home.path())
        .arg("--maildir")
        .arg(maildir)
        .output()?;
    if !init.status.success() {
        return Err(format!("mu init failed: {}", String::from_utf8_lossy(&init.stderr)).into());
    }
    let index = Command::new("mu")
        .arg("index")
        .arg("--muhome")
        .arg(home.path())
        .output()?;
    if !index.status.success() {
        return Err(format!(
            "mu index failed: {}",
            String::from_utf8_lossy(&index.stderr)
        )
        .into());
    }
    Ok(home)
}

/// The derived index over `maildir`, with a vector for each message.
///
/// Vectors are two-dimensional and hand-assigned so the dense ranking is a
/// fact about the fixture rather than about an embedding model: `one` and
/// `three` point one way, `two` the other.
fn derived_index(maildir: &Path) -> Result<Index, Box<dyn std::error::Error>> {
    let corpus = MaildirCorpus::scan(maildir)?;
    let selection = Selection::compute(&corpus, &kb::mail::BulkRule::default(), &BTreeSet::new());
    let index = Index::open_in_memory()?;
    rebuild_mail(&corpus, &selection, &index)?;
    for record_id in index.records_in("mail")? {
        let vector: Vec<f32> = if record_id.contains("two@") {
            vec![0.0, 1.0]
        } else {
            vec![1.0, 0.0]
        };
        for passage in index.passages(&record_id)? {
            index.put_embedding(
                &passage.stream_hash,
                passage.span_start,
                passage.span_len,
                "m",
                &encode_embedding(&vector),
            )?;
        }
    }
    Ok(index)
}

#[test]
fn mu_finds_a_fixture_message_by_its_subject() -> TestResult {
    if !mu_available() {
        eprintln!("skipping: mu not on PATH");
        return Ok(());
    }
    let maildir = fixture_maildir()?;
    let home = mu_index(maildir.path())?;
    let found = MuIndex::at(home.path()).message_ids("revenue", 10)?;
    assert!(
        found.contains(&"<one@example.invalid>".to_owned()),
        "mu did not find the message about revenue: {found:?}"
    );
    assert!(
        !found.contains(&"<two@example.invalid>".to_owned()),
        "mu matched a message that says nothing about revenue: {found:?}"
    );
    Ok(())
}

#[test]
fn mu_reports_identifiers_a_mail_record_is_addressed_by() -> TestResult {
    if !mu_available() {
        eprintln!("skipping: mu not on PATH");
        return Ok(());
    }
    let maildir = fixture_maildir()?;
    let home = mu_index(maildir.path())?;
    let index = derived_index(maildir.path())?;
    let records: BTreeSet<String> = index.records_in("mail")?.into_iter().collect();
    let found = MuIndex::at(home.path()).message_ids("revenue", 10)?;
    assert!(!found.is_empty(), "mu found nothing to join on");
    for id in &found {
        assert!(
            records.contains(id),
            "mu reported {id}, which names no record: {records:?}"
        );
    }
    Ok(())
}

#[test]
fn a_mail_query_returns_fused_mu_and_dense_candidates() -> TestResult {
    if !mu_available() {
        eprintln!("skipping: mu not on PATH");
        return Ok(());
    }
    let maildir = fixture_maildir()?;
    let home = mu_index(maildir.path())?;
    let index = derived_index(maildir.path())?;
    let mu = MuIndex::at(home.path());
    let vector = vec![1.0_f32, 0.0];
    let planner = Planner::new(vec![
        Box::new(MailLexical::new(&mu)),
        Box::new(IndexDense::new(
            &index,
            "mail",
            &vector,
            "m",
            kb::embedding::NO_MIN_SIMILARITY,
        )),
    ]);
    let resolved = planner.resolve(
        &Query {
            text: "revenue".into(),
            corpus: Some("mail".into()),
            limit: 100,
            project: None,
            context: None,
        },
        None,
    );

    let kinds: Vec<SignalKind> = resolved
        .trace
        .signals
        .iter()
        .map(|signal| signal.kind)
        .collect();
    assert_eq!(
        kinds,
        vec![SignalKind::Lexical, SignalKind::Dense],
        "both signals must have run"
    );
    for signal in &resolved.trace.signals {
        assert!(
            signal.error.is_none(),
            "{} failed: {:?}",
            signal.name,
            signal.error
        );
        assert!(
            !signal.candidates.is_empty(),
            "{} contributed nothing",
            signal.name
        );
    }

    let ids: Vec<&str> = resolved.hits.iter().map(|(id, _)| id.as_str()).collect();
    assert!(
        ids.contains(&"<one@example.invalid>"),
        "the message both signals favour is absent: {ids:?}"
    );
    // The dense side scores every mail record, so the fused result is the
    // union; what fusion decides is the order, and the message mu also found
    // must outrank the one it did not.
    let one = ids.iter().position(|id| *id == "<one@example.invalid>");
    let two = ids.iter().position(|id| *id == "<two@example.invalid>");
    assert!(
        one < two,
        "lexical agreement did not lift the message: {ids:?}"
    );

    // A dense score reaches the caller; a lexical-only hit carries none,
    // because the two are not on a common scale.
    let scored = resolved
        .hits
        .iter()
        .find(|(id, _)| id == "<one@example.invalid>")
        .ok_or("the fused hit vanished")?;
    assert!(
        scored.1.is_some(),
        "the dense score was not carried through"
    );
    Ok(())
}

#[test]
fn the_dense_signal_ranks_only_the_corpus_it_serves() -> TestResult {
    let maildir = fixture_maildir()?;
    let index = derived_index(maildir.path())?;
    let vector = vec![1.0_f32, 0.0];
    let ranked = index.rank_by_embedding(&vector, "m", "kb")?;
    assert!(
        ranked.is_empty(),
        "mail vectors answered a query about the kb corpus: {ranked:?}"
    );
    let mail = index.rank_by_embedding(&vector, "m", "mail")?;
    assert!(!mail.is_empty(), "the mail corpus ranked nothing");
    Ok(())
}

#[test]
fn a_query_matching_nothing_is_an_empty_answer_rather_than_a_failure() -> TestResult {
    if !mu_available() {
        eprintln!("skipping: mu not on PATH");
        return Ok(());
    }
    let maildir = fixture_maildir()?;
    let home = mu_index(maildir.path())?;
    let found = MuIndex::at(home.path()).message_ids("zygomorphic", 10)?;
    assert!(found.is_empty(), "got {found:?}");
    Ok(())
}

#[test]
fn a_default_client_reads_mus_own_index() {
    // Constructing against the default location contacts nothing; what is
    // asserted is only that the default exists and is the no-muhome form.
    let _client = MuIndex::default();
}

#[test]
fn an_indexed_record_supplies_the_text_it_is_reranked_by() -> TestResult {
    let maildir = fixture_maildir()?;
    let index = derived_index(maildir.path())?;
    let documents = kb::retrieval::IndexDocuments::new(&index);
    let text = documents
        .text("<one@example.invalid>")
        .ok_or("the record supplied no text")?;
    assert!(
        text.contains("quarterly revenue figures"),
        "the subject is missing from the reranked text: {text}"
    );
    assert!(
        documents.text("<absent@example.invalid>").is_none(),
        "a record that is not there must not produce text"
    );
    Ok(())
}

#[test]
fn a_blank_query_ranks_no_mail() -> TestResult {
    let maildir = fixture_maildir()?;
    let index = derived_index(maildir.path())?;
    let vector = vec![1.0_f32, 0.0];
    let signal = IndexDense::new(
        &index,
        "mail",
        &vector,
        "m",
        kb::embedding::NO_MIN_SIMILARITY,
    );
    let found = signal.candidates(&Query {
        text: "  ".into(),
        corpus: Some("mail".into()),
        limit: 10,
        project: None,
        context: None,
    })?;
    assert!(found.is_empty(), "got {found:?}");
    Ok(())
}

#[test]
fn the_lexical_signal_returns_only_messages_the_corpus_holds() -> TestResult {
    if !mu_available() {
        eprintln!("skipping: mu not on PATH");
        return Ok(());
    }
    let maildir = fixture_maildir()?;
    // A message mu indexes and the derived index is never told about, which
    // is the shape of the whole excluded-as-bulk population on a real
    // machine: mu sees every message, the corpus holds a small fraction.
    deliver(
        maildir.path(),
        "four",
        "<four@example.invalid>",
        "revenue newsletter",
        "this week in revenue",
    )?;
    let home = mu_index(maildir.path())?;

    let corpus = MaildirCorpus::scan(maildir.path())?;
    let selected: Vec<String> = corpus
        .catalogue()
        .map(|(id, _)| id.to_owned())
        .filter(|id| id != "<four@example.invalid>")
        .collect();
    let index = Index::open_in_memory()?;
    let selection = Selection {
        selected,
        overrides: Vec::new(),
        excluded: 1,
        out_of_scope: Vec::new(),
        missing_ground_truth: Vec::new(),
    };
    rebuild_mail(&corpus, &selection, &index)?;

    let mu = MuIndex::at(home.path());
    let unconstrained = MailLexical::new(&mu);
    let query = Query {
        text: "revenue".into(),
        corpus: Some("mail".into()),
        limit: 100,
        project: None,
        context: None,
    };
    let all = unconstrained.candidates(&query)?;
    assert!(
        all.iter().any(|c| c.id == "<four@example.invalid>"),
        "mu did not find the excluded message at all: {all:?}"
    );

    let membership = kb::retrieval::CataloguedMail::new(&index);
    let constrained = MailLexical::within(&mu, &membership);
    let kept = constrained.candidates(&query)?;
    assert!(
        !kept.iter().any(|c| c.id == "<four@example.invalid>"),
        "a message outside the corpus was returned: {kept:?}"
    );
    assert!(
        kept.iter().any(|c| c.id == "<one@example.invalid>"),
        "the filter removed a message the corpus does hold: {kept:?}"
    );
    Ok(())
}
