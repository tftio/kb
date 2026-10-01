//! The planner over independently queryable signals (T026).
//!
//! Two things are being held to account here. The first is that restructuring
//! retrieval into signals changed its structure and not its ranking: for the
//! kb corpus, whose signals are exactly the two the previous pipeline used,
//! the planner must agree with `search_hybrid_scored_with` candidate for
//! candidate. The second is that the trace is complete enough to attribute a
//! result — what each signal returned, what was skipped, what fusion produced,
//! and what reranking changed.

use kb::embedding::encode_embedding;
use kb::index::DensePooling;
use kb::mu::{MailSearch, MuError};
use kb::rerank::{RerankError, Reranker};
use kb::retrieval::{
    Candidate, DocumentSource, IndexDense, IndexLexical, MailLexical, Planner, Query, Rerank,
    Signal, SignalError, SignalKind,
};
use kb::storage::MatchMode;
use tftio_org::ast::{Block, Document, Inline, Title};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn note(title: &str, body: &str) -> Document {
    Document {
        blocks: vec![Block::Heading {
            level: 1,
            title: Title(title.into()),
            tags: vec![],
            children: vec![Block::Paragraph {
                inlines: vec![Inline::Plain(body.into())],
            }],
        }],
    }
}

/// A note with one independently embedded passage per section.
fn sectioned_note(title: &str, bodies: &[&str]) -> Document {
    Document {
        blocks: bodies
            .iter()
            .enumerate()
            .map(|(offset, body)| Block::Heading {
                level: 1,
                title: Title(format!("{title} {offset}")),
                tags: vec![],
                children: vec![Block::Paragraph {
                    inlines: vec![Inline::Plain((*body).into())],
                }],
            })
            .collect(),
    }
}

/// A kb corpus of three notes, each with a vector, in a fresh store and the
/// index derived from it.
///
/// Vectors are two-dimensional and hand-assigned so the dense ranking is a
/// fact about the fixture rather than about an embedding model.
fn corpus() -> Result<(tempfile::TempDir, kb::index::Index), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let store = kb::store::GitBlobStore::open_or_init(&dir.path().join("store"))?;
    let index = kb::index::Index::open_for_rebuild(&dir.path().join("index.db"))?;
    for (id, title, body, vector) in [
        ("alpha", "alpha note", "the quick brown fox", [1.0_f32, 0.0]),
        ("beta", "beta note", "a different idea entirely", [0.0, 1.0]),
        ("gamma", "gamma note", "the fox and the idea", [0.7, 0.7]),
    ] {
        let options = kb::write::WriteOptions::note(id)?;
        kb::write::put_record(&store, &index, id, &note(title, body), &options)?;
        for passage in index.passages(id)? {
            index.put_embedding(
                &passage.stream_hash,
                passage.span_start,
                passage.span_len,
                "m",
                &encode_embedding(&vector),
            )?;
        }
    }
    Ok((dir, index))
}

/// The planner the kb corpus is served by, matching the previous pipeline.
fn kb_planner<'a>(index: &'a kb::index::Index, vector: &'a [f32], floor: f32) -> Planner<'a> {
    Planner::new(vec![
        Box::new(IndexLexical::new(index, "kb", MatchMode::Keywords)),
        Box::new(IndexDense::new(index, "kb", vector, "m", floor)),
    ])
}

/// The planner is fusion over its signals and nothing else.
///
/// This replaces an equality against `search_hybrid_scored_with`, the `kb.db`
/// pipeline the planner was built to reproduce (T026). That pipeline no longer
/// serves retrieval — T029 moved the kb corpus onto the derived index — so
/// what is held to account here is the property the comparison was protecting:
/// the fused order is the reciprocal-rank fusion of what the signals returned,
/// with each hit carrying the dense score where the dense side had one.
#[test]
fn the_planner_fuses_exactly_what_its_signals_returned() -> TestResult {
    let (_dir, index) = corpus()?;
    let vector = vec![0.0_f32, 1.0];
    let floor = kb::embedding::NO_MIN_SIMILARITY;
    for text in ["fox", "idea", "the", "alpha note", "nothing matches this"] {
        let query = Query {
            text: text.to_owned(),
            corpus: None,
            limit: 100,
            project: None,
            context: None,
        };
        let lexical = IndexLexical::new(&index, "kb", MatchMode::Keywords).candidates(&query)?;
        let dense = IndexDense::new(&index, "kb", &vector, "m", floor).candidates(&query)?;
        let expected = kb::retrieval::fuse(&[
            lexical.iter().map(|c| c.id.clone()).collect(),
            dense.iter().map(|c| c.id.clone()).collect(),
        ]);
        let resolved = kb_planner(&index, &vector, floor).resolve(&query, None);
        let ids: Vec<String> = resolved.hits.iter().map(|(id, _)| id.clone()).collect();
        assert_eq!(ids, expected, "the fused order changed for {text:?}");
        for (id, score) in &resolved.hits {
            let from_dense = dense.iter().find(|c| &c.id == id).and_then(|c| c.score);
            assert_eq!(
                *score, from_dense,
                "{id} carries a score the dense signal did not give it"
            );
        }
    }
    Ok(())
}

#[test]
fn a_blank_query_ranks_nothing_through_either_path() -> TestResult {
    let (_dir, index) = corpus()?;
    let vector = vec![0.0_f32, 1.0];
    let planner = kb_planner(&index, &vector, kb::embedding::NO_MIN_SIMILARITY);
    let resolved = planner.resolve(
        &Query {
            text: "   ".into(),
            corpus: None,
            limit: 100,
            project: None,
            context: None,
        },
        None,
    );
    assert!(resolved.hits.is_empty(), "got {:?}", resolved.hits);
    Ok(())
}

#[test]
fn pooling_modes_rank_a_long_and_short_record_by_their_stated_rules() -> TestResult {
    let dir = tempfile::tempdir()?;
    let store = kb::store::GitBlobStore::open_or_init(&dir.path().join("store"))?;
    let index = kb::index::Index::open_for_rebuild(&dir.path().join("index.db"))?;
    let long_options = kb::write::WriteOptions::note("long")?;
    kb::write::put_record(
        &store,
        &index,
        "long",
        &sectioned_note("long", &["strong", "weak one", "weak two"]),
        &long_options,
    )?;
    let short_options = kb::write::WriteOptions::note("short")?;
    kb::write::put_record(
        &store,
        &index,
        "short",
        &sectioned_note("short", &["consistently relevant"]),
        &short_options,
    )?;
    for (id, vectors) in [
        ("long", vec![[0.9_f32, 0.1], [0.1, 0.9], [0.1, 0.9]]),
        ("short", vec![[0.8_f32, 0.2]]),
    ] {
        let passages = index.passages(id)?;
        assert_eq!(
            passages.len(),
            vectors.len(),
            "fixture passage count for {id}"
        );
        for (passage, vector) in passages.iter().zip(vectors) {
            index.put_embedding(
                &passage.stream_hash,
                passage.span_start,
                passage.span_len,
                "m",
                &encode_embedding(&vector),
            )?;
        }
    }

    let query = Query {
        text: "relevant".into(),
        corpus: Some("kb".into()),
        limit: 100,
        project: None,
        context: None,
    };
    let vector = [1.0_f32, 0.0];
    let ranked = |pooling| -> Result<Vec<String>, SignalError> {
        Ok(IndexDense::with_options(
            &index,
            "kb",
            &vector,
            "m",
            kb::embedding::NO_MIN_SIMILARITY,
            pooling,
            100,
        )
        .candidates(&query)?
        .into_iter()
        .map(|candidate| candidate.id)
        .collect())
    };

    assert_eq!(ranked(DensePooling::Maximum)?, vec!["long", "short"]);
    assert_eq!(ranked(DensePooling::MeanTopThree)?, vec!["short", "long"]);
    assert_eq!(
        ranked(DensePooling::LengthNormalizedMaximum)?,
        vec!["short", "long"]
    );
    Ok(())
}

#[test]
fn the_trace_reports_what_each_signal_returned() -> TestResult {
    let (_dir, index) = corpus()?;
    let vector = vec![0.0_f32, 1.0];
    let planner = kb_planner(&index, &vector, kb::embedding::NO_MIN_SIMILARITY);
    let resolved = planner.resolve(
        &Query {
            text: "fox".into(),
            corpus: None,
            limit: 100,
            project: None,
            context: None,
        },
        None,
    );
    let names: Vec<&str> = resolved
        .trace
        .signals
        .iter()
        .map(|signal| signal.name.as_str())
        .collect();
    assert_eq!(names, vec!["kb-lexical", "kb-dense"]);
    let lexical = resolved
        .trace
        .signals
        .first()
        .ok_or("no lexical signal in the trace")?;
    assert_eq!(lexical.kind, SignalKind::Lexical);
    assert!(
        lexical.candidates.contains(&"alpha".to_owned()),
        "lexical missed the node containing the word: {:?}",
        lexical.candidates
    );
    let dense = resolved
        .trace
        .signals
        .get(1)
        .ok_or("no dense signal in the trace")?;
    assert_eq!(dense.kind, SignalKind::Dense);
    assert_eq!(
        dense.candidates.len(),
        3,
        "dense ranks every embedded node: {:?}",
        dense.candidates
    );
    assert_eq!(resolved.trace.fused.len(), resolved.hits.len());
    Ok(())
}

/// A signal that always fails, standing for an unavailable backend.
struct Broken;

impl Signal for Broken {
    fn name(&self) -> &'static str {
        "broken"
    }
    fn kind(&self) -> SignalKind {
        SignalKind::Lexical
    }
    fn corpus(&self) -> &'static str {
        "kb"
    }
    fn candidates(&self, _query: &Query) -> Result<Vec<Candidate>, SignalError> {
        Err(SignalError::Mail(MuError::NotFound))
    }
}

/// A signal that returns a fixed list, standing for any working backend.
struct Fixed {
    name: &'static str,
    corpus: &'static str,
    kind: SignalKind,
    ids: Vec<&'static str>,
}

impl Signal for Fixed {
    fn name(&self) -> &'static str {
        self.name
    }
    fn kind(&self) -> SignalKind {
        self.kind
    }
    fn corpus(&self) -> &'static str {
        self.corpus
    }
    fn candidates(&self, _query: &Query) -> Result<Vec<Candidate>, SignalError> {
        Ok(self
            .ids
            .iter()
            .map(|id| Candidate {
                id: (*id).to_owned(),
                score: None,
            })
            .collect())
    }
}

fn anything() -> Query {
    Query {
        text: "anything".into(),
        corpus: None,
        limit: 10,
        project: None,
        context: None,
    }
}

#[test]
fn a_failing_signal_is_reported_and_the_others_still_answer() -> TestResult {
    let planner = Planner::new(vec![
        Box::new(Broken),
        Box::new(Fixed {
            name: "working",
            corpus: "kb",
            kind: SignalKind::Lexical,
            ids: vec!["a", "b"],
        }),
    ]);
    let resolved = planner.resolve(&anything(), None);
    assert_eq!(
        resolved.hits.len(),
        2,
        "the working signal must still count"
    );
    let broken = resolved
        .trace
        .signals
        .first()
        .ok_or("no trace for the broken signal")?;
    assert!(
        broken.error.is_some(),
        "a failure must be recorded, not swallowed"
    );
    assert!(broken.candidates.is_empty());
    Ok(())
}

#[test]
fn a_corpus_filter_excludes_signals_before_they_run() -> TestResult {
    let planner = Planner::new(vec![
        Box::new(Fixed {
            name: "kb-lexical",
            corpus: "kb",
            kind: SignalKind::Lexical,
            ids: vec!["note"],
        }),
        Box::new(Fixed {
            name: "mail-lexical",
            corpus: "mail",
            kind: SignalKind::Lexical,
            ids: vec!["<message>"],
        }),
    ]);
    let resolved = planner.resolve(
        &Query {
            text: "anything".into(),
            corpus: Some("mail".into()),
            limit: 10,
            project: None,
            context: None,
        },
        None,
    );
    assert_eq!(
        resolved
            .hits
            .iter()
            .map(|(id, _)| id.as_str())
            .collect::<Vec<_>>(),
        vec!["<message>"]
    );
    let skipped = resolved
        .trace
        .skipped
        .first()
        .ok_or("the excluded signal is not recorded")?;
    assert_eq!(skipped.name, "kb-lexical");
    assert!(
        skipped.reason.contains("mail"),
        "the reason must name the filter: {}",
        skipped.reason
    );
    Ok(())
}

/// A mu that answers from a fixed table rather than a Xapian index.
struct FakeMu(Vec<&'static str>);

impl MailSearch for FakeMu {
    fn message_ids(&self, _query: &str, limit: usize) -> Result<Vec<String>, MuError> {
        Ok(self
            .0
            .iter()
            .take(limit)
            .map(|id| kb::mu::bracketed(id))
            .collect())
    }
}

#[test]
fn a_mail_query_fuses_the_lexical_and_dense_signals() {
    let mu = FakeMu(vec!["one@x", "two@x"]);
    let planner = Planner::new(vec![
        Box::new(MailLexical::new(&mu)),
        Box::new(Fixed {
            name: "mail-dense",
            corpus: "mail",
            kind: SignalKind::Dense,
            ids: vec!["<two@x>", "<three@x>"],
        }),
    ]);
    let resolved = planner.resolve(
        &Query {
            text: "quarterly report".into(),
            corpus: Some("mail".into()),
            limit: 10,
            project: None,
            context: None,
        },
        None,
    );
    let ids: Vec<&str> = resolved.hits.iter().map(|(id, _)| id.as_str()).collect();
    // `two@x` is the only message both signals put forward, so fusion ranks
    // it first; the rest follow in their own signal's order.
    assert_eq!(ids.first(), Some(&"<two@x>"));
    assert_eq!(ids.len(), 3, "the union of both signals: {ids:?}");
    assert!(ids.contains(&"<one@x>") && ids.contains(&"<three@x>"));
}

/// A cross-encoder that scores by position in a fixed preference list.
///
/// Documents are compared by equality rather than by substring: the
/// identifiers here are single letters, and a substring test would match
/// them inside any surrounding prose the document source added.
struct FakeReranker(Vec<&'static str>);

impl Reranker for FakeReranker {
    fn score(&self, _query: &str, documents: &[String]) -> Result<Vec<f32>, RerankError> {
        Ok(documents
            .iter()
            .map(|document| {
                self.0
                    .iter()
                    .position(|wanted| *wanted == document.as_str())
                    .and_then(|position| u16::try_from(position).ok())
                    .map_or(0.0, |position| 1.0 - f32::from(position) / 100.0)
            })
            .collect())
    }
}

/// Documents that are simply their own identifiers.
struct SelfDocuments;

impl DocumentSource for SelfDocuments {
    fn text(&self, id: &str) -> Option<String> {
        Some(id.to_owned())
    }
}

#[test]
fn the_trace_records_what_reranking_changed() -> TestResult {
    let planner = Planner::new(vec![Box::new(Fixed {
        name: "lexical",
        corpus: "kb",
        kind: SignalKind::Lexical,
        ids: vec!["a", "b", "c"],
    })]);
    let client = FakeReranker(vec!["c", "b", "a"]);
    let documents = SelfDocuments;
    let resolved = planner.resolve(
        &anything(),
        Some(Rerank {
            client: &client,
            documents: &documents,
            top_k: 20,
        }),
    );
    let ids: Vec<&str> = resolved.hits.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, vec!["c", "b", "a"]);
    let record = resolved.trace.rerank.ok_or("reranking is not recorded")?;
    assert_eq!(record.before, vec!["a", "b", "c"]);
    assert_eq!(record.after, vec!["c", "b", "a"]);
    assert_eq!(record.window, 3);
    assert!(record.error.is_none());
    Ok(())
}

/// A cross-encoder that is never reachable.
struct DeadReranker;

impl Reranker for DeadReranker {
    fn score(&self, _query: &str, _documents: &[String]) -> Result<Vec<f32>, RerankError> {
        Err(RerankError::Unreachable {
            endpoint: "http://127.0.0.1:1".into(),
            detail: "connection refused".into(),
        })
    }
}

#[test]
fn a_dead_reranker_leaves_the_fused_order_and_says_so_in_the_trace() -> TestResult {
    let planner = Planner::new(vec![Box::new(Fixed {
        name: "lexical",
        corpus: "kb",
        kind: SignalKind::Lexical,
        ids: vec!["a", "b"],
    })]);
    let client = DeadReranker;
    let documents = SelfDocuments;
    let resolved = planner.resolve(
        &anything(),
        Some(Rerank {
            client: &client,
            documents: &documents,
            top_k: 20,
        }),
    );
    let ids: Vec<&str> = resolved.hits.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, vec!["a", "b"]);
    let record = resolved
        .trace
        .rerank
        .ok_or("a failed rerank is still a rerank stage")?;
    assert!(record.error.is_some());
    assert_eq!(record.before, record.after);
    Ok(())
}

/// A document source that can read nothing.
struct NoDocuments;

impl DocumentSource for NoDocuments {
    fn text(&self, _id: &str) -> Option<String> {
        None
    }
}

#[test]
fn an_unreadable_document_abandons_reranking_rather_than_dropping_a_hit() -> TestResult {
    let planner = Planner::new(vec![Box::new(Fixed {
        name: "lexical",
        corpus: "kb",
        kind: SignalKind::Lexical,
        ids: vec!["a", "b"],
    })]);
    let client = FakeReranker(vec!["b", "a"]);
    let documents = NoDocuments;
    let resolved = planner.resolve(
        &anything(),
        Some(Rerank {
            client: &client,
            documents: &documents,
            top_k: 20,
        }),
    );
    let ids: Vec<&str> = resolved.hits.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, vec!["a", "b"], "no hit may be lost to a read failure");
    let record = resolved
        .trace
        .rerank
        .ok_or("the abandonment must be recorded")?;
    assert!(record.error.is_some());
    assert_eq!(record.window, 0);
    Ok(())
}

#[test]
fn a_signal_kind_names_itself_for_a_trace() {
    assert_eq!(SignalKind::Lexical.as_str(), "lexical");
    assert_eq!(SignalKind::Dense.as_str(), "dense");
}

#[test]
fn a_node_is_reranked_by_the_text_its_vector_was_computed_from() -> TestResult {
    let (_dir, index) = corpus()?;
    let documents = kb::retrieval::IndexDocuments::new(&index);
    let text = documents.text("alpha").ok_or("alpha has no text")?;
    assert!(
        text.contains("alpha note") && text.contains("quick brown fox"),
        "the reranker is shown neither title nor body: {text}"
    );
    assert!(
        !text.contains(":PROPERTIES:"),
        "the drawer reached the reranker: {text}"
    );
    assert!(
        documents.text("no-such-node").is_none(),
        "a node that is not there must not produce text"
    );
    Ok(())
}
