//! Retrieval as a planner over independently queryable signals.
//!
//! Retrieval used to be one pipeline: FTS5 and cosine, fused, and whatever
//! came out was the answer. That works while there is one corpus and one way
//! to search it. It stops working the moment a second corpus arrives with a
//! better lexical engine of its own — mu's Xapian index over the Maildir
//! knows correspondents, folders and dates that FTS5 over normalized text has
//! no notion of — because a single pipeline can only be extended by making
//! every query pay for every backend.
//!
//! So the pipeline becomes a plan. A [`Signal`] is one independently
//! queryable source of candidates; the [`Planner`] selects the signals a
//! query needs, runs them, fuses their rankings, and optionally reranks the
//! result. Adding a backend for one corpus is adding a signal, and it changes
//! nothing about any other corpus's retrieval.
//!
//! **Observability is the point, not a convenience.** Every resolution
//! carries a [`Trace`] recording what each signal returned, what was skipped
//! and why, what fusion produced, and what reranking changed. Without it a
//! change to fusion is unattributable: the result moved, and nothing says
//! which signal moved it. That is the property the captured design
//! conversation identified as the reason to keep the lexical engine
//! independently queryable in the first place.
//!
//! **Structure, not ranking.** For a corpus whose signals are unchanged, this
//! module composes exactly the calls the previous pipeline made, in the same
//! order, with the same constants — [`fuse`] is
//! [`crate::storage::reciprocal_rank_fusion`] at the same k. A signal that
//! fails is reported and dropped rather than failing the query, because a
//! degraded answer from the signals that did answer is worth more than no
//! answer, and the trace is what keeps the degradation from being silent.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::rerank::{self, Reranker};
use tftio_org::ast::NodeId;

use crate::storage;

/// What kind of evidence a signal supplies.
///
/// Recorded per signal so a trace can be read without knowing which backends
/// were configured: "the dense signal returned nothing" is interpretable,
/// "`mail-dense` returned nothing" requires knowing what `mail-dense` is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalKind {
    /// Term matching over text.
    Lexical,
    /// Vector proximity.
    Dense,
}

impl SignalKind {
    /// The name this kind is reported under.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Lexical => "lexical",
            Self::Dense => "dense",
        }
    }
}

/// One record a signal put forward, with its score where the signal has one.
///
/// `score` is `None` for a lexical signal deliberately rather than `0.0`: the
/// two kinds are not on a common scale, so *no score was computed* has to stay
/// distinguishable from *the score was low*.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// The record's identifier.
    pub id: String,
    /// The signal's own score, where it has one.
    pub score: Option<f32>,
}

/// What is being asked for.
#[derive(Debug, Clone)]
pub struct Query {
    /// The question, as the user asked it.
    pub text: String,
    /// Restrict retrieval to one corpus. The structural filter applied
    /// *before* any textual retriever runs, since a corpus a query excludes
    /// is a corpus whose signals should not be paid for.
    pub corpus: Option<String>,
    /// How many candidates each signal is asked for.
    pub limit: usize,
    /// Restrict retrieval to records carrying this project slug
    /// (`PLAN-20260923-project-identity` T006). Applied inside each
    /// index-backed signal's own SQL, beside the corpus predicate, rather
    /// than as a post-filter: a record another project's session wrote is
    /// never candidate evidence for this query.
    pub project: Option<String>,
    /// Restrict retrieval to records carrying this context (`personal`,
    /// `work`, ...). Independent of `project` — see the ADR's "Context is
    /// not a project" constraint.
    pub context: Option<String>,
}

/// Why one signal could not answer.
#[derive(Debug, Error)]
pub enum SignalError {
    /// The backing store failed.
    #[error("{0}")]
    Database(#[from] rusqlite::Error),
    /// The derived index failed.
    #[error("{0}")]
    Index(#[from] crate::index::IndexError),
    /// An external lexical engine failed or is absent.
    #[error("{0}")]
    Mail(#[from] crate::mu::MuError),
}

/// An independently queryable source of candidates.
///
/// The seam. Fusion and reranking consume signal outputs through this trait
/// and nothing else, so a backend joins for one corpus without any other
/// corpus's retrieval changing (`REPO_INVARIANTS.md` ENG-010).
pub trait Signal {
    /// What this signal is called in a trace.
    fn name(&self) -> &str;
    /// What kind of evidence it supplies.
    fn kind(&self) -> SignalKind;
    /// Which corpus it serves.
    ///
    /// Borrowed rather than `'static` since T029: one implementation serves
    /// every corpus the derived index holds, so which corpus a signal is for
    /// is a value it carries rather than a fact about its type.
    fn corpus(&self) -> &str;
    /// The candidates it puts forward for `query`, best first.
    ///
    /// # Errors
    ///
    /// [`SignalError`] if the backend fails. A signal that fails is reported
    /// in the trace and dropped from the fusion; it does not fail the query.
    fn candidates(&self, query: &Query) -> Result<Vec<Candidate>, SignalError>;
}

/// What one signal did.
#[derive(Debug, Clone)]
pub struct SignalTrace {
    /// The signal's name.
    pub name: String,
    /// What kind of evidence it supplies.
    pub kind: SignalKind,
    /// Which corpus it serves.
    pub corpus: String,
    /// The candidates it returned, in its own order.
    pub candidates: Vec<String>,
    /// Why it contributed nothing, when it failed.
    pub error: Option<String>,
    /// How long the signal took to answer, whether or not it answered.
    ///
    /// Recorded per signal rather than for the resolution as a whole because
    /// the stages sit on different resources — the dense scan is bytes read
    /// from the index, the lexical match is FTS5, mail's lexical half is a
    /// subprocess — and a service threshold on one of them (T025's dense-stage
    /// p95) cannot be checked from a total.
    pub elapsed: std::time::Duration,
}

/// Why a signal was not run at all.
#[derive(Debug, Clone)]
pub struct Skipped {
    /// The signal's name.
    pub name: String,
    /// What excluded it.
    pub reason: String,
}

/// What reranking changed.
#[derive(Debug, Clone)]
pub struct RerankTrace {
    /// The order before reranking.
    pub before: Vec<String>,
    /// The order after.
    pub after: Vec<String>,
    /// How many candidates the cross-encoder saw.
    pub window: usize,
    /// Why the fused order stands, when reranking could not be applied.
    pub error: Option<String>,
    /// How long the stage took, including the document reads and the
    /// cross-encoder call.
    pub elapsed: std::time::Duration,
}

/// Everything a resolution did, in the order it did it.
///
/// The observability bar: for any query it must be possible to ask what each
/// signal returned, what was excluded before they ran, what fusion produced,
/// and what reranking changed.
#[derive(Debug, Clone, Default)]
pub struct Trace {
    /// What each signal that ran returned.
    pub signals: Vec<SignalTrace>,
    /// What was excluded before any signal ran.
    pub skipped: Vec<Skipped>,
    /// The fused order.
    pub fused: Vec<String>,
    /// What reranking changed, when it ran.
    pub rerank: Option<RerankTrace>,
}

/// A resolved query: the ranking, and the account of how it was reached.
#[derive(Debug, Clone)]
pub struct Resolution {
    /// The ranking, each hit carrying a dense score where one was computed.
    pub hits: Vec<(String, Option<f32>)>,
    /// How it was reached.
    pub trace: Trace,
}

/// The reciprocal-rank-fusion constant.
///
/// 60 because that is what the pipeline this replaces used, and the invariant
/// on this restructure is that ranking does not move for a corpus whose
/// signals are unchanged. A different constant would be a ranking change
/// wearing a refactor's clothes.
pub const FUSION_K: usize = 60;

/// Fuse ranked candidate lists by reciprocal rank.
///
/// Delegates to [`crate::storage::reciprocal_rank_fusion`] rather than
/// reimplementing it, so the planner and the pipeline it replaces cannot
/// drift apart.
#[must_use]
pub fn fuse(lists: &[Vec<String>]) -> Vec<String> {
    let as_nodes: Vec<Vec<NodeId>> = lists
        .iter()
        .map(|list| list.iter().map(|id| NodeId(id.clone())).collect())
        .collect();
    storage::reciprocal_rank_fusion(FUSION_K, &as_nodes)
        .into_iter()
        .map(|NodeId(id)| id)
        .collect()
}

/// Where a reranking stage gets the text it scores.
///
/// A document is corpus-specific — a note's text and a message's text come
/// from different places — so the planner asks rather than knowing.
pub trait DocumentSource {
    /// The text representing `id`, or `None` if it cannot be read.
    fn text(&self, id: &str) -> Option<String>;
}

/// A reranking stage to apply after fusion.
pub struct Rerank<'a> {
    /// The cross-encoder.
    pub client: &'a dyn Reranker,
    /// Where its documents come from.
    pub documents: &'a dyn DocumentSource,
    /// How many of the fused candidates it sees.
    pub top_k: usize,
}

/// A plan over a set of signals.
pub struct Planner<'a> {
    signals: Vec<Box<dyn Signal + 'a>>,
}

impl<'a> Planner<'a> {
    /// A planner over `signals`.
    ///
    /// Order matters: fusion is order-independent in score but the trace
    /// reports signals in this order, and the pipeline this replaces fused
    /// lexical before dense.
    #[must_use]
    pub fn new(signals: Vec<Box<dyn Signal + 'a>>) -> Self {
        Self { signals }
    }

    /// Resolve `query`, optionally reranking the result.
    ///
    /// Never fails. A signal that errors is recorded in the trace and
    /// contributes nothing; a reranker that cannot answer leaves the fused
    /// order standing and says so in the trace. The alternative — failing a
    /// query because one of several sources of evidence is unavailable —
    /// throws away the evidence that *is* available.
    #[must_use]
    pub fn resolve(&self, query: &Query, rerank: Option<Rerank<'_>>) -> Resolution {
        let mut trace = Trace::default();
        let mut lists: Vec<Vec<String>> = Vec::new();
        let mut scores: BTreeMap<String, f32> = BTreeMap::new();
        for signal in &self.signals {
            if let Some(wanted) = &query.corpus
                && signal.corpus() != wanted
            {
                trace.skipped.push(Skipped {
                    name: signal.name().to_owned(),
                    reason: format!("corpus filter: {wanted}"),
                });
                continue;
            }
            let started = std::time::Instant::now();
            let answered = signal.candidates(query);
            let elapsed = started.elapsed();
            match answered {
                Ok(candidates) => {
                    for candidate in &candidates {
                        if let Some(score) = candidate.score {
                            scores
                                .entry(candidate.id.clone())
                                .and_modify(|current| *current = current.max(score))
                                .or_insert(score);
                        }
                    }
                    let ids: Vec<String> = candidates
                        .into_iter()
                        .map(|candidate| candidate.id)
                        .collect();
                    trace.signals.push(SignalTrace {
                        name: signal.name().to_owned(),
                        kind: signal.kind(),
                        corpus: signal.corpus().to_owned(),
                        candidates: ids.clone(),
                        error: None,
                        elapsed,
                    });
                    lists.push(ids);
                }
                Err(failure) => trace.signals.push(SignalTrace {
                    name: signal.name().to_owned(),
                    kind: signal.kind(),
                    corpus: signal.corpus().to_owned(),
                    candidates: Vec::new(),
                    error: Some(failure.to_string()),
                    elapsed,
                }),
            }
        }
        trace.fused = fuse(&lists);
        let ordered = match rerank {
            None => trace.fused.clone(),
            Some(stage) => {
                let started = std::time::Instant::now();
                let (ordered, mut record) = apply_rerank(&trace.fused, &stage, &query.text);
                record.elapsed = started.elapsed();
                trace.rerank = Some(record);
                ordered
            }
        };
        let hits = ordered
            .into_iter()
            .map(|id| {
                let score = scores.get(&id).copied();
                (id, score)
            })
            .collect();
        Resolution { hits, trace }
    }
}

/// Rerank the top of `fused`, returning the new order and what changed.
fn apply_rerank(
    fused: &[String],
    stage: &Rerank<'_>,
    question: &str,
) -> (Vec<String>, RerankTrace) {
    let mut record = RerankTrace {
        before: fused.to_vec(),
        after: fused.to_vec(),
        window: 0,
        error: None,
        elapsed: std::time::Duration::ZERO,
    };
    if fused.is_empty() || question.trim().is_empty() {
        return (fused.to_vec(), record);
    }
    let window: Vec<&String> = fused.iter().take(stage.top_k).collect();
    let mut documents: Vec<String> = Vec::with_capacity(window.len());
    for id in &window {
        let Some(text) = stage.documents.text(id) else {
            record.error = Some(format!("{id} could not be read for reranking"));
            return (fused.to_vec(), record);
        };
        documents.push(rerank::shown(&text));
    }
    record.window = documents.len();
    match stage.client.score(question, &documents) {
        Ok(relevance) => {
            let ordered = rerank::reorder(fused.to_vec(), &relevance);
            record.after.clone_from(&ordered);
            (ordered, record)
        }
        Err(failure) => {
            record.error = Some(failure.to_string());
            record.window = 0;
            (fused.to_vec(), record)
        }
    }
}

// ── Signals over the derived index ─────────────────────────────────────

/// Lexical retrieval over one corpus's passages in the derived index.
///
/// Index-backed since T029. It was `kb.db`'s FTS5 table, which served the kb
/// corpus alone and made "which index answers this query" a question with two
/// answers. The rewriting of the query text is unchanged — the same
/// [`storage::build_fts_query`] under the same [`storage::MatchMode`] — so
/// what moved is where the expression is evaluated, not what it means.
pub struct IndexLexical<'a> {
    index: &'a crate::index::Index,
    corpus: &'a str,
    mode: storage::MatchMode,
    name: String,
}

impl<'a> IndexLexical<'a> {
    /// A lexical signal over `corpus` in `index`, under `mode`.
    ///
    /// The signal names itself after the corpus it serves, because a trace
    /// that says `kb-lexical` for a mail query attributes the result to the
    /// wrong backend.
    #[must_use]
    pub fn new(index: &'a crate::index::Index, corpus: &'a str, mode: storage::MatchMode) -> Self {
        Self {
            index,
            corpus,
            mode,
            name: format!("{corpus}-lexical"),
        }
    }
}

impl Signal for IndexLexical<'_> {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> SignalKind {
        SignalKind::Lexical
    }
    fn corpus(&self) -> &str {
        self.corpus
    }
    fn candidates(&self, query: &Query) -> Result<Vec<Candidate>, SignalError> {
        let text = query.text.trim();
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let expression = storage::build_fts_query(text, self.mode);
        // A query of nothing but control characters cleans to nothing.
        // Matching on an empty expression is an FTS5 syntax error, and the
        // honest answer to a query with no searchable content is no results.
        if expression.is_empty() {
            return Ok(Vec::new());
        }
        Ok(self
            .index
            .search_text_in(
                self.corpus,
                &expression,
                query.project.as_deref(),
                query.context.as_deref(),
            )?
            .into_iter()
            .map(|id| Candidate { id, score: None })
            .collect())
    }
}

/// Dense retrieval over one corpus's vectors in the derived index.
///
/// The similarity floor is applied to the candidate list **before** the rank
/// cut and before fusion, never to results after it, because a floor on a
/// fused result would let a cosine suppress a genuine lexical match — which
/// is not a cosine's business.
pub struct IndexDense<'a> {
    index: &'a crate::index::Index,
    corpus: &'a str,
    vector: &'a [f32],
    model: &'a str,
    floor: f32,
    pooling: crate::index::DensePooling,
    candidate_limit: usize,
    name: String,
}

impl<'a> IndexDense<'a> {
    /// A dense signal over `corpus` in `index` for a query already embedded
    /// as `vector`.
    #[must_use]
    pub fn new(
        index: &'a crate::index::Index,
        corpus: &'a str,
        vector: &'a [f32],
        model: &'a str,
        floor: f32,
    ) -> Self {
        Self {
            index,
            corpus,
            vector,
            model,
            floor,
            pooling: crate::index::DEFAULT_DENSE_POOLING,
            candidate_limit: storage::VECTOR_CANDIDATES,
            name: format!("{corpus}-dense"),
        }
    }

    /// A dense signal with an explicit pooling rule and rank cut.
    ///
    /// This is the experimental seam for T031. [`IndexDense::new`] remains
    /// the shipped default, so adding an arm cannot change callers that did
    /// not name one.
    #[must_use]
    pub fn with_options(
        index: &'a crate::index::Index,
        corpus: &'a str,
        vector: &'a [f32],
        model: &'a str,
        floor: f32,
        pooling: crate::index::DensePooling,
        candidate_limit: usize,
    ) -> Self {
        Self {
            index,
            corpus,
            vector,
            model,
            floor,
            pooling,
            candidate_limit,
            name: format!("{corpus}-dense"),
        }
    }
}

impl Signal for IndexDense<'_> {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> SignalKind {
        SignalKind::Dense
    }
    fn corpus(&self) -> &str {
        self.corpus
    }
    fn candidates(&self, query: &Query) -> Result<Vec<Candidate>, SignalError> {
        // A blank query embeds to nothing worth ranking, and the lexical side
        // returns nothing for it either. Ranking the whole corpus by cosine
        // against a vector for the empty string would make every query with
        // no words return the entire corpus in an arbitrary order.
        if query.text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let mut ranked = self.index.rank_by_embedding_with_pooling(
            self.vector,
            self.model,
            self.corpus,
            self.pooling,
            query.project.as_deref(),
            query.context.as_deref(),
        )?;
        ranked.retain(|(_, score)| *score >= self.floor);
        ranked.truncate(self.candidate_limit);
        Ok(ranked
            .into_iter()
            .map(|(id, score)| Candidate {
                id,
                score: Some(score),
            })
            .collect())
    }
}

// ── Signals over the mail corpus ───────────────────────────────────────

/// Whether the corpus holds a given identifier.
///
/// mu indexes the whole Maildir. kb's mail corpus is only the increment the
/// selection admitted, typically a small fraction of it, and the rest were
/// excluded deliberately as bulk. Without this test the lexical signal returns
/// messages the corpus does not contain: no passages, no vector, no subject
/// to name them by, and no possibility of the dense signal ever agreeing.
/// That is not a retrieval result, it is a category error, and it was found
/// by running a real query rather than by reasoning about one.
pub trait CorpusMembership {
    /// Whether `id` names a record this corpus holds.
    fn holds(&self, id: &str) -> bool;
}

/// [`CorpusMembership`] over the mail catalogue.
pub struct CataloguedMail<'a> {
    index: &'a crate::index::Index,
}

impl<'a> CataloguedMail<'a> {
    /// Membership as `index` records it.
    #[must_use]
    pub const fn new(index: &'a crate::index::Index) -> Self {
        Self { index }
    }
}

impl CorpusMembership for CataloguedMail<'_> {
    fn holds(&self, id: &str) -> bool {
        self.index
            .mail_catalogue(id)
            .is_ok_and(|found| found.is_some())
    }
}

/// How many more candidates mu is asked for than the query wants.
///
/// mu ranks over the whole archive, so an arbitrary number of its best hits
/// may fall outside the corpus. Asking for exactly `limit` and then filtering
/// would return fewer than asked for whenever the excluded population ranks
/// well — which for a query matching newsletters is most of the time.
const MEMBERSHIP_OVERFETCH: usize = 10;

/// Lexical retrieval over the Maildir, delegated to mu.
///
/// The first specialized backend, and the reason this module exists: mu's
/// index knows correspondents, folders and dates, and joining it needs no new
/// identity scheme because a mail record *is* its bracketed `Message-ID`.
pub struct MailLexical<'a> {
    mu: &'a dyn crate::mu::MailSearch,
    membership: Option<&'a dyn CorpusMembership>,
}

impl<'a> MailLexical<'a> {
    /// A lexical signal backed by `mu`, returning whatever mu returns.
    #[must_use]
    pub const fn new(mu: &'a dyn crate::mu::MailSearch) -> Self {
        Self {
            mu,
            membership: None,
        }
    }

    /// A lexical signal constrained to the records `membership` holds.
    #[must_use]
    pub const fn within(
        mu: &'a dyn crate::mu::MailSearch,
        membership: &'a dyn CorpusMembership,
    ) -> Self {
        Self {
            mu,
            membership: Some(membership),
        }
    }
}

impl Signal for MailLexical<'_> {
    fn name(&self) -> &'static str {
        "mail-lexical"
    }
    fn kind(&self) -> SignalKind {
        SignalKind::Lexical
    }
    fn corpus(&self) -> &'static str {
        "mail"
    }
    fn candidates(&self, query: &Query) -> Result<Vec<Candidate>, SignalError> {
        let Some(membership) = self.membership else {
            return Ok(self
                .mu
                .message_ids(&query.text, query.limit)?
                .into_iter()
                .map(|id| Candidate { id, score: None })
                .collect());
        };
        let asked = query.limit.saturating_mul(MEMBERSHIP_OVERFETCH);
        Ok(self
            .mu
            .message_ids(&query.text, asked)?
            .into_iter()
            .filter(|id| membership.holds(id))
            .take(query.limit)
            .map(|id| Candidate { id, score: None })
            .collect())
    }
}

/// The text an indexed record is reranked by.
pub struct IndexDocuments<'a> {
    index: &'a crate::index::Index,
}

impl<'a> IndexDocuments<'a> {
    /// Documents read from `index`.
    #[must_use]
    pub const fn new(index: &'a crate::index::Index) -> Self {
        Self { index }
    }
}

impl DocumentSource for IndexDocuments<'_> {
    /// Empty text is `None` rather than `Some("")`.
    ///
    /// A record's text is the concatenation of its passages, so a record the
    /// index does not hold produces an empty string rather than an error.
    /// Handing that to a cross-encoder would score a blank document against
    /// the query and rank the missing record on the result — worse than
    /// abandoning the reranking pass, which at least leaves the fused order
    /// standing and says why.
    fn text(&self, id: &str) -> Option<String> {
        self.index
            .record_text(id)
            .ok()
            .filter(|text| !text.trim().is_empty())
    }
}
