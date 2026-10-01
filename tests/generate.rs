//! Generated artifacts, addressed by what produced them.
//!
//! Without this key, "rebuildable" is aspirational. Re-chunking and
//! re-embedding the kb corpus is minutes; regenerating its summaries on the
//! local model is an overnight job, and over the mail increment T015 measured
//! 59 to 175 hours depending on form. If a rebuild means paying that, rebuilds
//! stop happening and the derived index quietly becomes authoritative again —
//! which is the failure this whole plan exists to undo.

use kb::generate::{Form, Generator, GeneratorError, Plan, generate_missing};
use kb::index::{Index, Scope};
use kb::store::{BlobStore, GitBlobStore, RefName};
use std::cell::RefCell;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// A generator that counts its calls and never contacts anything.
#[derive(Debug, Default)]
struct Counting {
    calls: RefCell<Vec<String>>,
    reply: String,
}

impl Counting {
    fn new(reply: &str) -> Self {
        Self {
            calls: RefCell::new(Vec::new()),
            reply: reply.to_owned(),
        }
    }
    fn count(&self) -> usize {
        self.calls.borrow().len()
    }
}

impl Generator for Counting {
    fn version(&self) -> String {
        "counting-stub".to_owned()
    }
    fn generate(&self, _form: Form, document: &str) -> Result<String, GeneratorError> {
        self.calls.borrow_mut().push(document.to_owned());
        Ok(self.reply.clone())
    }
}

/// Put one note in the store, as the real ingest path does.
fn store_note(store: &GitBlobStore, id: &str, body: &str) -> TestResult {
    use kb::record::{ArtifactKind, ContentHash, CorpusId, RecordHeader, RecordId, SourceRef};
    let raw = format!("* Note {id}\n\n{body}\n");
    let stream = kb::record::normalize(ArtifactKind::Note, raw.as_bytes())?;
    let header = RecordHeader {
        id: RecordId::new(id)?,
        corpus: CorpusId::new("kb")?,
        kind: ArtifactKind::Note,
        created: "2026-05-19T09:00:00Z".parse()?,
        updated: "2026-08-14T10:30:00Z".parse()?,
        source: SourceRef::new("node-id", id)?,
        tags: vec![],
        raw: ContentHash::new(store.put(raw.as_bytes())?.as_str())?,
        stream: ContentHash::new(store.put(stream.as_bytes())?.as_str())?,
        normalizer: kb::record::NORMALIZER_VERSION,
        provenance: kb::record::Provenance::default(),
    };
    let hash = store.put(&header.serialize())?;
    store.set_ref(&RefName::new(&format!("kb/{id}"))?, &hash)?;
    Ok(())
}

/// A store and an index over it, built through the real rebuild path rather
/// than by writing rows a production caller could not write.
fn index_with(
    dir: &tempfile::TempDir,
    records: &[(&str, &str)],
) -> Result<(GitBlobStore, Index), Box<dyn std::error::Error>> {
    let store = GitBlobStore::open_or_init(dir.path())?;
    for (id, body) in records {
        store_note(&store, id, body)?;
    }
    let index = Index::open_in_memory()?;
    kb::index::rebuild(&store, &index, &Scope::All)?;
    Ok((store, index))
}

/// The invariant the task exists for.
#[test]
fn an_unchanged_key_issues_no_model_call() -> TestResult {
    let dir = tempfile::tempdir()?;
    let (_store, index) = index_with(&dir, &[("n1", "the body of note one")])?;
    let generator = Counting::new("a generated summary");
    let plan = Plan::new(Form::Summary, 2);

    let first = generate_missing(&index, &plan, &generator, &["n1".to_owned()])?;
    let second = generate_missing(&index, &plan, &generator, &["n1".to_owned()])?;

    assert_eq!(first.generated, 1);
    assert_eq!(second.generated, 0);
    assert_eq!(second.reused, 1);
    assert_eq!(
        generator.count(),
        1,
        "regenerating an unchanged key contacted the model"
    );
    Ok(())
}

/// A new prompt version must be measurable against the old one, which means
/// both have to exist at once rather than one overwriting the other.
#[test]
fn two_prompt_versions_coexist_for_one_source() -> TestResult {
    let dir = tempfile::tempdir()?;
    let (_store, index) = index_with(&dir, &[("n1", "the body of note one")])?;
    let generator = Counting::new("first wording");
    generate_missing(
        &index,
        &Plan::new(Form::Summary, 2),
        &generator,
        &["n1".to_owned()],
    )?;
    let revised = Counting::new("second wording");

    generate_missing(
        &index,
        &Plan::new(Form::Summary, 3),
        &revised,
        &["n1".to_owned()],
    )?;

    let stored = index.generated_for("n1")?;
    assert_eq!(stored.len(), 2, "a prompt change overwrote its predecessor");
    assert!(stored.iter().any(|a| a.content == "first wording"));
    assert!(stored.iter().any(|a| a.content == "second wording"));
    Ok(())
}

/// Two forms of the same document under the same prompt version are two
/// artifacts, not one: a different question asked of the same text is a
/// different prompt.
#[test]
fn the_two_forms_do_not_collide() -> TestResult {
    let dir = tempfile::tempdir()?;
    let (_store, index) = index_with(&dir, &[("n1", "the body of note one")])?;
    let summary = Counting::new("a paragraph");
    let questions = Counting::new("a question?");

    generate_missing(
        &index,
        &Plan::new(Form::Summary, 2),
        &summary,
        &["n1".to_owned()],
    )?;
    generate_missing(
        &index,
        &Plan::new(Form::Questions, 2),
        &questions,
        &["n1".to_owned()],
    )?;

    assert_eq!(index.generated_for("n1")?.len(), 2);
    Ok(())
}

/// The measurement loop this architecture exists for: regenerate for the
/// ground-truth subset under a new prompt, score it, keep or discard, without
/// touching the several thousand records that were not named.
#[test]
fn generating_for_a_subset_leaves_the_rest_untouched() -> TestResult {
    let dir = tempfile::tempdir()?;
    let (_store, index) = index_with(&dir, &[("n1", "one"), ("n2", "two"), ("n3", "three")])?;
    let generator = Counting::new("generated");
    let plan = Plan::new(Form::Summary, 2);

    let report = generate_missing(&index, &plan, &generator, &["n2".to_owned()])?;

    assert_eq!(report.generated, 1);
    assert_eq!(generator.count(), 1);
    assert!(index.generated_for("n1")?.is_empty());
    assert_eq!(index.generated_for("n2")?.len(), 1);
    assert!(index.generated_for("n3")?.is_empty());
    Ok(())
}

/// The key is the source hash, so a record whose text changed is a different
/// source and must be regenerated rather than answered from the old artifact.
#[test]
fn changed_source_text_is_regenerated() -> TestResult {
    let dir = tempfile::tempdir()?;
    let (store, index) = index_with(&dir, &[("n1", "the original body")])?;
    let generator = Counting::new("generated");
    let plan = Plan::new(Form::Summary, 2);
    generate_missing(&index, &plan, &generator, &["n1".to_owned()])?;

    store_note(&store, "n1", "an entirely rewritten body")?;
    kb::index::rebuild(&store, &index, &Scope::All)?;
    let report = generate_missing(&index, &plan, &generator, &["n1".to_owned()])?;

    assert_eq!(
        report.generated, 1,
        "a rewritten record reused a stale artifact"
    );
    assert_eq!(generator.count(), 2);
    Ok(())
}

/// A record the index does not hold is named in the report rather than
/// silently skipped: a subset run that quietly generated nothing would read
/// as success.
#[test]
fn an_unknown_record_is_reported_rather_than_skipped() -> TestResult {
    let dir = tempfile::tempdir()?;
    let (_store, index) = index_with(&dir, &[("n1", "one")])?;
    let generator = Counting::new("generated");

    let report = generate_missing(
        &index,
        &Plan::new(Form::Summary, 2),
        &generator,
        &["absent".to_owned()],
    )?;

    assert_eq!(report.generated, 0);
    assert_eq!(report.unknown, vec!["absent".to_owned()]);
    Ok(())
}

/// A generator that fails must not leave the run looking complete, and must
/// keep what it already finished, because the work is expensive and resumable.
#[test]
fn a_failing_generator_keeps_finished_work_and_names_the_failure() -> TestResult {
    struct Failing;
    impl Generator for Failing {
        fn version(&self) -> String {
            "failing".to_owned()
        }
        fn generate(&self, _form: Form, document: &str) -> Result<String, GeneratorError> {
            if document.contains("one") {
                return Ok("fine".to_owned());
            }
            Err(GeneratorError::Unreachable(
                "the endpoint refused".to_owned(),
            ))
        }
    }
    let dir = tempfile::tempdir()?;
    let (_store, index) = index_with(&dir, &[("n1", "one"), ("n2", "two")])?;

    let outcome = generate_missing(
        &index,
        &Plan::new(Form::Summary, 2),
        &Failing,
        &["n1".to_owned(), "n2".to_owned()],
    );

    assert!(outcome.is_err(), "a failed run reported success");
    assert_eq!(
        index.generated_for("n1")?.len(),
        1,
        "finished work was discarded by a later failure"
    );
    Ok(())
}

/// An empty completion is a failure, not an artifact: storing it would key a
/// blank against the source and never regenerate it.
#[test]
fn an_empty_completion_is_refused() -> TestResult {
    let dir = tempfile::tempdir()?;
    let (_store, index) = index_with(&dir, &[("n1", "one")])?;
    let empty = Counting::new("   \n  ");

    let outcome = generate_missing(
        &index,
        &Plan::new(Form::Summary, 2),
        &empty,
        &["n1".to_owned()],
    );

    assert!(outcome.is_err());
    assert!(index.generated_for("n1")?.is_empty());
    Ok(())
}

/// The point of keying on content: a rebuild re-derives everything cheap and
/// keeps everything expensive. If a rebuild cost the generation, rebuilds
/// would stop happening and the index would quietly become authoritative
/// again.
#[test]
fn a_rebuild_re_derives_passages_and_keeps_generated_artifacts() -> TestResult {
    let dir = tempfile::tempdir()?;
    let (store, index) = index_with(&dir, &[("n1", "the body of note one")])?;
    let generator = Counting::new("a generated summary");
    generate_missing(
        &index,
        &Plan::new(Form::Summary, 2),
        &generator,
        &["n1".to_owned()],
    )?;

    kb::index::rebuild(&store, &index, &Scope::All)?;

    assert_eq!(index.generated_for("n1")?.len(), 1);
    assert_eq!(
        generator.count(),
        1,
        "a rebuild would have to pay for generation again"
    );
    Ok(())
}

/// Re-embedding under a different model must not invalidate generation: the
/// artifact is keyed on the document, not on how it was vectorized.
#[test]
fn changing_the_embedding_model_regenerates_nothing() -> TestResult {
    let dir = tempfile::tempdir()?;
    let (_store, index) = index_with(&dir, &[("n1", "the body of note one")])?;
    let generator = Counting::new("a generated summary");
    let plan = Plan::new(Form::Summary, 2);
    generate_missing(&index, &plan, &generator, &["n1".to_owned()])?;
    let passage = index
        .passages("n1")?
        .into_iter()
        .next()
        .ok_or("no passage")?;
    index.put_embedding(
        &passage.stream_hash,
        passage.span_start,
        passage.span_len,
        "some-other-embedding-model",
        &[1, 2, 3, 4],
    )?;

    let report = generate_missing(&index, &plan, &generator, &["n1".to_owned()])?;

    assert_eq!(report.reused, 1);
    assert_eq!(generator.count(), 1);
    Ok(())
}

/// A different generator is a different artifact, not a replacement: two
/// models' output for one document has to be comparable.
#[test]
fn a_different_generator_produces_a_distinct_artifact() -> TestResult {
    struct Other;
    impl Generator for Other {
        fn version(&self) -> String {
            "another-model".to_owned()
        }
        fn generate(&self, _form: Form, _document: &str) -> Result<String, GeneratorError> {
            Ok("the other model's wording".to_owned())
        }
    }
    let dir = tempfile::tempdir()?;
    let (_store, index) = index_with(&dir, &[("n1", "the body of note one")])?;
    let plan = Plan::new(Form::Summary, 2);
    generate_missing(&index, &plan, &Counting::new("first"), &["n1".to_owned()])?;

    generate_missing(&index, &plan, &Other, &["n1".to_owned()])?;

    let stored = index.generated_for("n1")?;
    assert_eq!(stored.len(), 2);
    assert_eq!(
        stored
            .iter()
            .map(|a| a.generator_version.as_str())
            .collect::<Vec<_>>(),
        vec!["another-model", "counting-stub"]
    );
    Ok(())
}

/// A thinking model emits its reasoning before its answer, and the reasoning
/// is the model talking to itself rather than about the document.
#[test]
fn a_thinking_block_is_stripped_from_a_completion() {
    assert_eq!(
        kb::generate::strip_thinking("<think>weighing it up</think>the answer"),
        "the answer"
    );
    assert_eq!(
        kb::generate::strip_thinking("before <THINK>x</THINK> after"),
        "before  after"
    );
    // Unterminated: everything after the opener is reasoning, so keeping it
    // would store the model's deliberation as the artifact.
    assert_eq!(
        kb::generate::strip_thinking("the answer<think>still going"),
        "the answer"
    );
    assert_eq!(
        kb::generate::strip_thinking("no block here"),
        "no block here"
    );
}
