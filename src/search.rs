//! The one search kb resolves, for every caller that asks for one.
//!
//! `kb search` and the agent surface's `search` tool ask the same question of
//! the same corpora and must get the same answer. Before this module they did
//! not: the CLI ran T026's planner over lexical and dense signals and finished
//! with T010's cross-encoder, while the tool called `search_fts` and took the
//! first N identifiers — 0.133 authored recall@5 served to agents against the
//! 0.833 the CLI measures. Two implementations that agree today are a defect
//! waiting for the next change to one of them; this is the single one.
//!
//! **Nothing here prints.** Every degradation is returned as a [`Note`],
//! because the same fact goes to stderr for a person at a terminal and into
//! the response body for an agent that has no stderr to read. What is *not*
//! negotiable is that it goes somewhere: a caller who cannot tell a hybrid
//! ranking from a keyword-only one reads an empty result as "the knowledge
//! base is silent on this subject" (`REPO_INVARIANTS.md` ENG-004).
//!
//! The thin I/O shell around the pure planner (ENG-008): it reads the
//! environment, opens the derived index, embeds the query and fetches titles,
//! and leaves ranking entirely to [`crate::retrieval`].

use crate::cli_embed::CliEmbedder;
use crate::{index, mu, rerank, retrieval, storage};

/// The corpus searched when none is named.
///
/// One corpus rather than all of them, deliberately: what an unfiltered query
/// should do across corpora is a ranking question — the populations calibrate
/// differently — and answering it by default would silently change every
/// figure this plan measured.
pub const CORPUS_KB: &str = "kb";

/// The mail corpus, served by mu and the derived index.
pub const CORPUS_MAIL: &str = "mail";

/// One search, as asked for.
///
/// The index is the caller's: the CLI searches the configured one and a
/// served surface searches whichever it was pointed at, and neither should
/// have to agree with the other about where it lives.
#[derive(Debug, Clone)]
pub struct Request {
    /// What to search for.
    pub query: String,
    /// Which corpus to search: [`CORPUS_KB`] or [`CORPUS_MAIL`].
    pub corpus: String,
    /// How many hits to return, applied after ranking.
    pub limit: usize,
    /// How the query text reaches FTS5.
    pub mode: storage::MatchMode,
    /// Whether to contact the embedding endpoint for the dense signal.
    pub vector: bool,
    /// A cosine floor overriding the model's recorded calibration.
    pub min_similarity: Option<f32>,
    /// How passage similarities become one score per record.
    pub dense_pooling: index::DensePooling,
    /// How many records the dense signal contributes before fusion.
    pub vector_candidates: usize,
    /// Whether to apply the cross-encoder stage where one is configured.
    pub rerank: bool,
    /// Restrict results to records carrying this project slug
    /// (`PLAN-20260923-project-identity` T006). `None` applies no filter, and
    /// a search with neither this nor `context` set returns exactly what it
    /// returned before this field existed.
    pub project: Option<tftio_lib::project::Slug>,
    /// Restrict results to records carrying this context (`personal`,
    /// `work`, ...). Not a project — see the ADR's "Context is not a
    /// project" constraint.
    pub context: Option<String>,
}

impl Request {
    /// A default search for `query`: the kb corpus, keyword mode, every
    /// stage that is configured.
    #[must_use]
    pub fn new(query: impl Into<String>, limit: usize) -> Self {
        Self {
            query: query.into(),
            corpus: CORPUS_KB.to_owned(),
            limit,
            mode: storage::MatchMode::Keywords,
            vector: true,
            min_similarity: None,
            dense_pooling: index::DEFAULT_DENSE_POOLING,
            vector_candidates: storage::VECTOR_CANDIDATES,
            rerank: true,
            project: None,
            context: None,
        }
    }
}

/// One ranked record.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    /// The record's identifier.
    pub id: String,
    /// A one-line description of it.
    pub title: String,
    /// The cosine the vector side scored it at, or `None` where no cosine was
    /// computed. Never `0.0` as a stand-in: *not scored* and *scored low* are
    /// different facts and the keyword and vector halves are not on a common
    /// scale.
    pub similarity: Option<f32>,
    /// The corpus it came from.
    pub corpus: String,
}

/// A ranking stage that did not run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Note {
    /// What to tell a reader, unprefixed.
    pub message: String,
    /// True when something that was configured failed or was set wrongly; false
    /// when a stage was simply never configured. A person at a terminal wants
    /// both — being told the ranking is keyword-only explains a thin result —
    /// while an agent's context budget should not carry a line on every call
    /// reporting a deployment choice nobody made by accident.
    pub asked_for: bool,
}

/// What a search produced.
#[derive(Debug)]
pub struct Outcome {
    /// The ranking, truncated to the requested limit.
    pub hits: Vec<Hit>,
    /// What every signal returned and what reranking changed.
    pub trace: retrieval::Trace,
    /// Every stage that did not run.
    pub notes: Vec<Note>,
    /// How long embedding the query took, when an embedding was obtained.
    /// `None` when the dense signal was not asked for or could not run.
    pub embed_elapsed: Option<std::time::Duration>,
}

impl Outcome {
    /// Milliseconds spent in each stage, keyed by stage name.
    ///
    /// `embed` is the query embedding, each signal appears under its own name
    /// (`kb-dense`, `mail-lexical`, …) and `rerank` is the cross-encoder
    /// stage where one ran. A stage that did not run is absent rather than
    /// zero, for the reason [`Hit::similarity`] is `None` rather than `0.0`:
    /// *did not run* and *ran in no time* are different facts. The whole
    /// resolution is not summed here because what a caller waited for is the
    /// caller's clock; this is the account of where the time inside went.
    #[must_use]
    pub fn stages_ms(&self) -> std::collections::BTreeMap<String, f64> {
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        let mut stages = std::collections::BTreeMap::new();
        if let Some(embed) = self.embed_elapsed {
            stages.insert("embed".to_owned(), ms(embed));
        }
        for signal in &self.trace.signals {
            stages.insert(signal.name.clone(), ms(signal.elapsed));
        }
        if let Some(rerank) = &self.trace.rerank {
            stages.insert("rerank".to_owned(), ms(rerank.elapsed));
        }
        stages
    }
}

/// Why a search produced nothing at all.
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    /// A signal the corpus depends on failed, so an empty result would be a
    /// lie rather than an answer.
    #[error("{0}")]
    Signal(String),
    /// The derived index serving the named corpus could not be opened.
    #[error(transparent)]
    Index(#[from] index::IndexError),
    /// Titles could not be read for the ranked identifiers.
    #[error(transparent)]
    Titles(#[from] rusqlite::Error),
}

/// Embed `query` for the vector half of the ranking.
///
/// Returns the vector paired with the model it was produced under, since
/// storage filters stored rows by that same identifier and the two must
/// agree.
fn embed_query(query: &str) -> Result<(Vec<f32>, String), crate::cli_embed::EmbedError> {
    let embedder = CliEmbedder::from_env()?;
    let vector = embedder.embed_query(query)?;
    Ok((vector, embedder.model().to_string()))
}

/// Resolve the vector half of the ranking, noting why there is not one.
///
/// A blank query is skipped without a note: there is nothing to embed, the
/// keyword side returns nothing either, and no ranking was degraded.
fn resolve_vector(query: &str, notes: &mut Vec<Note>) -> Option<(Vec<f32>, String)> {
    if query.trim().is_empty() {
        return None;
    }
    match embed_query(query) {
        Ok(pair) => Some(pair),
        Err(e) => {
            let asked_for = !e.is_disabled();
            notes.push(Note {
                message: format!("keyword-only ranking; {e}"),
                asked_for,
            });
            None
        }
    }
}

/// The similarity floor for `model`: the request's value if given, else
/// `KB_EMBEDDING_MIN_SIMILARITY`, else the model's recorded calibration.
///
/// An unparsable environment value is ignored in favour of the calibration
/// rather than treated as zero, so a typo cannot silently switch the floor
/// off — and it is noted, because a floor that is not the one somebody wrote
/// down is exactly the thing that makes a result set inexplicable.
#[allow(
    clippy::disallowed_methods,
    reason = "the floor is deployment configuration alongside the endpoint and model, read at the search boundary (REPO_INVARIANTS.md ENG-013)"
)]
fn resolve_min_similarity(flag: Option<f32>, model: &str, notes: &mut Vec<Note>) -> f32 {
    if let Some(v) = flag {
        return v;
    }
    if let Ok(raw) = std::env::var("KB_EMBEDDING_MIN_SIMILARITY") {
        if let Ok(v) = raw.trim().parse::<f32>() {
            return v;
        }
        notes.push(Note {
            message: format!("ignoring unparsable KB_EMBEDDING_MIN_SIMILARITY={raw:?}"),
            asked_for: true,
        });
    }
    crate::embedding::default_min_similarity(model)
}

/// The reranking stage to apply, or `None` where none is configured or the
/// request asked for the fused order.
///
/// **Never fatal, and never silent.** A reranker that cannot answer costs
/// ordering, not results: the fused ranking it was given is a complete answer
/// and is what a search returns when no endpoint is configured at all. A
/// caller who cannot tell the two apart cannot interpret either.
fn rerank_client(want: bool, notes: &mut Vec<Note>) -> Option<(rerank::HttpReranker, usize)> {
    if !want {
        return None;
    }
    let config = rerank::read_config_from_env()?;
    match rerank::HttpReranker::new(&config.base_url) {
        Ok(client) => Some((client, config.top_k)),
        Err(e) => {
            notes.push(Note {
                message: format!("fused ranking; {e}"),
                asked_for: true,
            });
            None
        }
    }
}

/// The signal failure that is a failed query rather than a degradation.
///
/// Two cases, and only two.
///
/// A failure of the kb corpus's **own** lexical index is a failed query:
/// under `--match` the expression came from the caller, and reporting
/// dense-only results for a query `SQLite` could not parse would present a
/// malformed search as a successful one (T028).
///
/// Every signal failing is also a failed query. A partial answer is worth
/// more than none, which is why one backend going down degrades rather than
/// fails — but when nothing answered there is no partial answer, and
/// returning an empty result would say the corpus holds nothing on the
/// subject. That is the one claim this must never make by accident
/// (`REPO_INVARIANTS.md` ENG-004).
///
/// An external backend failing while another signal answers is neither. mu
/// being absent costs the lexical half of a mail query; the dense half still
/// answers, and the notes say what was lost.
fn fatal_signal_failure(trace: &retrieval::Trace) -> Option<&str> {
    let own_lexical = trace.signals.iter().find(|signal| {
        signal.kind == retrieval::SignalKind::Lexical
            && signal.corpus == CORPUS_KB
            && signal.error.is_some()
    });
    let all_failed =
        !trace.signals.is_empty() && trace.signals.iter().all(|signal| signal.error.is_some());
    own_lexical
        .or_else(|| {
            all_failed
                .then(|| trace.signals.iter().find(|signal| signal.error.is_some()))
                .flatten()
        })
        .and_then(|signal| signal.error.as_deref())
}

/// Every degradation the trace records, as notes alongside those gathered
/// while the search was being set up.
///
/// A signal that was asked for and failed is `asked_for`; there is no other
/// kind in a trace, since a signal that was never configured is never planned.
fn notes_from_trace(trace: &retrieval::Trace, notes: &mut Vec<Note>) {
    for signal in &trace.signals {
        if let Some(error) = &signal.error {
            notes.push(Note {
                message: format!("{} contributed nothing; {error}", signal.name),
                asked_for: true,
            });
        }
    }
    if let Some(record) = &trace.rerank
        && let Some(error) = &record.error
    {
        notes.push(Note {
            message: format!("fused ranking; {error}"),
            asked_for: true,
        });
    }
}

/// Resolve `request` against the kb database `conn` and whatever backends the
/// named corpus is served by.
///
/// The whole search happens inside this one call because the planner's
/// signals borrow the connection, the mu client and the derived index; there
/// is no useful place to cut it that does not hand a caller a set of borrows
/// to reassemble.
///
/// # Errors
///
/// Returns [`SearchError::Signal`] when a signal the corpus depends on failed,
/// which is a failed query rather than an empty result;
/// [`SearchError::Index`] when the derived index serving the named corpus
/// cannot be opened; and [`SearchError::Titles`] when the ranked identifiers
/// cannot be named. A stage that merely degraded is not an error: it is a
/// [`Note`] on a successful [`Outcome`].
pub fn resolve(index: &index::Index, request: &Request) -> Result<Outcome, SearchError> {
    let mut notes = Vec::new();
    let mut embed_elapsed = None;
    let vector = if request.vector {
        let started = std::time::Instant::now();
        let vector = resolve_vector(&request.query, &mut notes);
        // Timed only when an embedding was obtained: an endpoint that is not
        // configured, or refused, did not run the stage, and the note says so.
        if vector.is_some() {
            embed_elapsed = Some(started.elapsed());
        }
        vector
    } else {
        None
    };
    let floor = vector
        .as_ref()
        .map_or(crate::embedding::NO_MIN_SIMILARITY, |(_, model)| {
            resolve_min_similarity(request.min_similarity, model, &mut notes)
        });
    let mail = request.corpus == CORPUS_MAIL;
    let mu = mu::MuIndex::new();

    let catalogue = mail.then(|| retrieval::CataloguedMail::new(index));
    let mut signals: Vec<Box<dyn retrieval::Signal + '_>> = Vec::new();
    if mail {
        // Mail's lexical half is mu's Xapian index rather than the derived
        // one: mu already holds the archive and re-deriving a second full-text
        // index over it would be a copy that can disagree.
        signals.push(catalogue.as_ref().map_or_else(
            || Box::new(retrieval::MailLexical::new(&mu)) as Box<dyn retrieval::Signal + '_>,
            |catalogue| Box::new(retrieval::MailLexical::within(&mu, catalogue)),
        ));
    } else {
        signals.push(Box::new(retrieval::IndexLexical::new(
            index,
            &request.corpus,
            request.mode,
        )));
    }
    if let Some((v, model)) = vector.as_ref() {
        signals.push(Box::new(retrieval::IndexDense::with_options(
            index,
            &request.corpus,
            v,
            model,
            floor,
            request.dense_pooling,
            request.vector_candidates,
        )));
    }
    let planner = retrieval::Planner::new(signals);

    let documents = retrieval::IndexDocuments::new(index);
    let client = rerank_client(request.rerank, &mut notes);
    let stage = client.as_ref().map(|(client, top_k)| retrieval::Rerank {
        client,
        documents: &documents,
        top_k: *top_k,
    });

    let resolved = planner.resolve(
        &retrieval::Query {
            text: request.query.clone(),
            corpus: Some(request.corpus.clone()),
            limit: storage::VECTOR_CANDIDATES,
            project: request
                .project
                .as_ref()
                .map(|slug| slug.as_str().to_owned()),
            context: request.context.clone(),
        },
        stage,
    );

    if let Some(failure) = fatal_signal_failure(&resolved.trace) {
        return Err(SearchError::Signal(failure.to_owned()));
    }
    notes_from_trace(&resolved.trace, &mut notes);

    let mut scored = resolved.hits;
    // Truncating after ranking rather than in the query keeps the ranking
    // itself identical to what it was before a limit existed, for any query
    // returning at most `limit` hits.
    scored.truncate(request.limit);
    let hits = name(index, &scored, &request.corpus);
    Ok(Outcome {
        hits,
        trace: resolved.trace,
        notes,
        embed_elapsed,
    })
}

/// Attach each ranked identifier's title.
///
/// Read from the index, which caches the name derived from each record's
/// artifact (T029). It was read from `kb.db`'s `title` column for the kb
/// corpus and derived from the stream for mail, which is two answers to one
/// question and one of them in a database being retired.
///
/// A record the index cannot name is named with an empty string rather than
/// dropped: the ranking put it there, and removing it from the results
/// because its title is missing would answer a different question than the
/// one asked.
fn name(index: &index::Index, scored: &[(String, Option<f32>)], corpus: &str) -> Vec<Hit> {
    scored
        .iter()
        .map(|(id, similarity)| Hit {
            title: index.record(id).map(|row| row.title).unwrap_or_default(),
            id: id.clone(),
            similarity: *similarity,
            corpus: corpus.to_owned(),
        })
        .collect()
}
