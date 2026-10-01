//! Cross-encoder reranking of the candidates retrieval already surfaced.
//!
//! kb ranks by fusing two bi-encoder signals: FTS5 over the text and cosine
//! over passage vectors embedded once, ahead of time. Query and document
//! never meet. That is what makes the ranking cheap enough to run over the
//! whole corpus, and it is also what makes it blunt — a passage's vector has
//! to stand for everything the passage might ever answer, so a document that
//! is merely near the question's subject can outrank the one that answers it.
//!
//! A cross-encoder scores the query and one document *together*, so the
//! question's words attend to the document's. It cannot be precomputed, which
//! is why it runs over a shortlist rather than a corpus.
//!
//! **A reranker can only reorder what retrieval surfaced.** T019 measured
//! this rather than assuming it: of the seven ground-truth questions whose
//! answer sat outside the top-20 window, zero changed rank, and the reranked
//! run's misses were identical to the control's. Reranking is the ordering
//! axis and nothing else; recall is bought elsewhere, by the generated
//! document-side representations of [`crate::generate`].
//!
//! Everything here is pure except [`HttpReranker`], which is the module's
//! only edge onto the network and the only thing a caller has to fake to
//! test a reranked path (`REPO_INVARIANTS.md` ENG-010).

use std::cmp::Ordering;
use std::env;
use std::time::Duration;

use thiserror::Error;

/// Base URL of a local reranking server exposing `/v1/rerank`.
///
/// Absent means reranking is off and `kb search` returns the fused ranking
/// exactly as it did before this module existed. That is the revert path:
/// unset one variable, no rebuild, no reindex, nothing stored to undo.
pub const ENV_BASE_URL: &str = "KB_RERANK_BASE_URL";

/// How many fused candidates the cross-encoder rescores.
pub const ENV_TOP_K: &str = "KB_RERANK_TOP_K";

/// The window size reranking uses when [`ENV_TOP_K`] says nothing.
///
/// 20 is what T019 measured, and the measurement gives the number its
/// meaning: reranking the top 20 put every answer it could find into the top
/// five — recall@5 equalled recall@10 in both reranked runs — for 1.795s of
/// scoring. Cost scales linearly in the window, so this is the tuning knob if
/// the latency ever has to come down.
pub const DEFAULT_TOP_K: usize = 20;

/// How much of a document the cross-encoder is shown.
///
/// Bounded by the serving runtime rather than by the model: llama.cpp
/// requires a non-causal model's whole sequence to fit one physical batch, so
/// a document longer than `--ubatch-size` is refused outright. kb's org text
/// runs about three characters per token, so 2000 characters is roughly 660
/// tokens and needs `-ub 2048`; at the 512 default even 1000 characters
/// fails. Recorded because the number is a property of the server's flags,
/// not of the corpus.
pub const DOCUMENT_CHARS: usize = 2000;

/// Where the reranker is and how much of the ranking it sees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankConfig {
    /// Base URL of the reranking server.
    pub base_url: String,
    /// How many fused candidates are rescored.
    pub top_k: usize,
}

/// Why a reranking pass could not be completed.
///
/// Both variants are recoverable at the call site by keeping the fused
/// ranking: a reranker that cannot answer costs ordering, not results.
#[derive(Debug, Error)]
pub enum RerankError {
    /// The endpoint could not be reached, or answered with a failure status.
    #[error("reranker at {endpoint} did not answer: {detail}")]
    Unreachable {
        /// The endpoint that was asked.
        endpoint: String,
        /// What went wrong, as the transport reported it.
        detail: String,
    },
    /// The endpoint answered with a score list that does not match the
    /// documents sent, which would silently drop or misattribute candidates.
    #[error("reranker returned {returned} scores for {sent} documents")]
    Mismatch {
        /// How many scores came back.
        returned: usize,
        /// How many documents were sent.
        sent: usize,
    },
}

/// Scores query-document pairs with a model that sees both together.
///
/// The domain interface, so nothing outside this module knows the reranker
/// speaks HTTP (`REPO_INVARIANTS.md` ENG-010). Scores are returned in the
/// order the documents were given, whatever order the endpoint answers in.
pub trait Reranker {
    /// Score each of `documents` against `query`, in the order given.
    ///
    /// # Errors
    ///
    /// [`RerankError::Unreachable`] if the endpoint fails, and
    /// [`RerankError::Mismatch`] if it answers with a different number of
    /// scores than documents sent.
    fn score(&self, query: &str, documents: &[String]) -> Result<Vec<f32>, RerankError>;
}

/// How much of `text` the cross-encoder is shown.
///
/// Truncation is by character rather than by byte so a multi-byte character
/// is never split; the endpoint is handed JSON, and half a character is not
/// text.
#[must_use]
pub fn shown(text: &str) -> String {
    text.chars().take(DOCUMENT_CHARS).collect()
}

/// Reorder the first `scores.len()` candidates by score, leaving the rest
/// alone.
///
/// Candidates beyond the window keep both their order and their position
/// beneath it: the reranker never saw them, so promoting or demoting them
/// would be a claim nothing measured. Ties preserve the fused order, so a
/// cross-encoder that cannot separate two candidates leaves retrieval's
/// judgement standing rather than reversing it arbitrarily.
///
/// A score list longer than the candidate list is honoured only as far as
/// there are candidates, and a candidate inside the window with no score
/// sorts last within it rather than disappearing — losing a result is a worse
/// failure than misordering one.
#[must_use]
pub fn reorder<T>(candidates: Vec<T>, scores: &[f32]) -> Vec<T> {
    let window = scores.len().min(candidates.len());
    let mut head: Vec<(usize, T)> = Vec::with_capacity(window);
    let mut tail: Vec<T> = Vec::new();
    for (position, candidate) in candidates.into_iter().enumerate() {
        if position < window {
            head.push((position, candidate));
        } else {
            tail.push(candidate);
        }
    }
    head.sort_by(|(left, _), (right, _)| {
        let a = scores.get(*left).copied().unwrap_or(f32::NEG_INFINITY);
        let b = scores.get(*right).copied().unwrap_or(f32::NEG_INFINITY);
        b.partial_cmp(&a)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    head.into_iter()
        .map(|(_, candidate)| candidate)
        .chain(tail)
        .collect()
}

/// Build a [`RerankConfig`] from process environment variables.
///
/// Returns `None` — meaning reranking is off — unless [`ENV_BASE_URL`] is set
/// to a non-empty value. An unparsable or zero [`ENV_TOP_K`] falls back to
/// [`DEFAULT_TOP_K`] rather than switching reranking off: a typo in a tuning
/// knob should not silently change which strategy is running.
#[must_use]
#[allow(
    clippy::disallowed_methods,
    reason = "the reranking endpoint is deployment config alongside the embedding endpoint, read once at the binary edge (REPO_INVARIANTS.md ENG-013)"
)]
pub fn read_config_from_env() -> Option<RerankConfig> {
    config_from_env_inputs(env::var(ENV_BASE_URL).ok(), env::var(ENV_TOP_K).ok())
}

/// Pure variant of [`read_config_from_env`] — the caller supplies the values,
/// so tests need not mutate the process environment.
fn config_from_env_inputs(base_url: Option<String>, top_k: Option<String>) -> Option<RerankConfig> {
    let base_url = base_url.filter(|s| !s.trim().is_empty())?;
    let top_k = top_k
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|k| *k > 0)
        .unwrap_or(DEFAULT_TOP_K);
    Some(RerankConfig {
        base_url: base_url.trim_end_matches('/').to_owned(),
        top_k,
    })
}

/// HTTP-backed [`Reranker`] against a local `/v1/rerank`.
///
/// Local by construction, like every other model this repository talks to:
/// the corpus is the operator's own notes and correspondence.
pub struct HttpReranker {
    runtime: tokio::runtime::Runtime,
    client: reqwest::Client,
    base_url: String,
}

impl HttpReranker {
    /// Build a reranker against a local endpoint.
    ///
    /// # Errors
    ///
    /// [`RerankError::Unreachable`] if no async runtime can be started.
    pub fn new(base_url: &str) -> Result<Self, RerankError> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| RerankError::Unreachable {
                endpoint: base_url.to_owned(),
                detail: e.to_string(),
            })?;
        Ok(Self {
            runtime,
            client: reqwest::Client::builder()
                .timeout(Duration::from_mins(5))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            base_url: base_url.trim_end_matches('/').to_owned(),
        })
    }
}

/// The parts of a rerank response this module reads.
#[derive(serde::Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

#[derive(serde::Deserialize)]
struct RerankResult {
    index: usize,
    relevance_score: f32,
}

impl Reranker for HttpReranker {
    fn score(&self, query: &str, documents: &[String]) -> Result<Vec<f32>, RerankError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let endpoint = format!("{}/v1/rerank", self.base_url);
        let body = serde_json::json!({ "query": query, "documents": documents });
        let parsed: RerankResponse = self.runtime.block_on(async {
            let response = self
                .client
                .post(&endpoint)
                .json(&body)
                .send()
                .await
                .map_err(|e| RerankError::Unreachable {
                    endpoint: endpoint.clone(),
                    detail: e.to_string(),
                })?;
            let status = response.status();
            if !status.is_success() {
                return Err(RerankError::Unreachable {
                    endpoint: endpoint.clone(),
                    detail: format!("answered {status}"),
                });
            }
            response
                .json::<RerankResponse>()
                .await
                .map_err(|e| RerankError::Unreachable {
                    endpoint: endpoint.clone(),
                    detail: e.to_string(),
                })
        })?;
        place_scores(&parsed.results, documents.len())
    }
}

/// Put each returned score back at the position of the document it scored.
///
/// The endpoint is free to answer in relevance order, and reading its list
/// positionally would then attribute every score to the wrong document — a
/// failure that produces a plausible ranking rather than an error, which is
/// why the index is honoured rather than the order.
fn place_scores(results: &[RerankResult], sent: usize) -> Result<Vec<f32>, RerankError> {
    if results.len() != sent {
        return Err(RerankError::Mismatch {
            returned: results.len(),
            sent,
        });
    }
    let mut scores = vec![f32::NEG_INFINITY; sent];
    for result in results {
        let slot = scores.get_mut(result.index).ok_or(RerankError::Mismatch {
            returned: results.len(),
            sent,
        })?;
        *slot = result.relevance_score;
    }
    Ok(scores)
}

#[cfg(test)]
#[allow(
    clippy::significant_drop_tightening,
    reason = "a mockito Server guard is held for the test's duration on purpose; dropping it early tears down the endpoint under test"
)]
mod tests {
    use super::{
        DEFAULT_TOP_K, DOCUMENT_CHARS, HttpReranker, RerankResult, Reranker,
        config_from_env_inputs, place_scores, reorder, shown,
    };

    #[test]
    fn reordering_sorts_the_window_by_score_and_leaves_the_tail_alone() {
        let candidates = vec!["a", "b", "c", "d", "e"];
        let ordered = reorder(candidates, &[0.1, 0.9, 0.5]);
        assert_eq!(ordered, vec!["b", "c", "a", "d", "e"]);
    }

    #[test]
    fn a_tie_inside_the_window_keeps_the_fused_order() {
        let ordered = reorder(vec!["a", "b", "c"], &[0.5, 0.5, 0.5]);
        assert_eq!(ordered, vec!["a", "b", "c"]);
    }

    #[test]
    fn more_scores_than_candidates_reorders_what_there_is() {
        let ordered = reorder(vec!["a", "b"], &[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(ordered, vec!["b", "a"]);
    }

    #[test]
    fn an_empty_score_list_changes_nothing() {
        let ordered = reorder(vec!["a", "b", "c"], &[]);
        assert_eq!(ordered, vec!["a", "b", "c"]);
    }

    #[test]
    fn a_document_is_truncated_by_character_rather_than_by_byte() {
        let text = "é".repeat(DOCUMENT_CHARS + 10);
        let seen = shown(&text);
        assert_eq!(seen.chars().count(), DOCUMENT_CHARS);
        assert!(text.starts_with(&seen));
    }

    #[test]
    fn a_short_document_is_shown_whole() {
        assert_eq!(shown("brief"), "brief");
    }

    #[test]
    fn no_endpoint_means_no_reranking() {
        assert!(config_from_env_inputs(None, Some("20".into())).is_none());
        assert!(config_from_env_inputs(Some("   ".into()), None).is_none());
    }

    #[test]
    fn an_endpoint_alone_reranks_the_default_window() {
        let config = config_from_env_inputs(Some("http://127.0.0.1:8081/".into()), None)
            .unwrap_or_else(|| panic!("an endpoint should configure reranking"));
        assert_eq!(config.base_url, "http://127.0.0.1:8081");
        assert_eq!(config.top_k, DEFAULT_TOP_K);
    }

    #[test]
    fn an_unusable_window_falls_back_rather_than_switching_reranking_off() {
        for raw in ["nonsense", "0", "-4"] {
            let config = config_from_env_inputs(Some("http://x".into()), Some(raw.into()))
                .unwrap_or_else(|| panic!("{raw} should not disable reranking"));
            assert_eq!(config.top_k, DEFAULT_TOP_K, "for {raw}");
        }
    }

    #[test]
    fn an_explicit_window_is_honoured() {
        let config = config_from_env_inputs(Some("http://x".into()), Some(" 5 ".into()))
            .unwrap_or_else(|| panic!("5 should configure a window"));
        assert_eq!(config.top_k, 5);
    }

    #[test]
    fn scores_are_placed_by_the_index_the_endpoint_returned() {
        let results = vec![
            RerankResult {
                index: 2,
                relevance_score: 0.9,
            },
            RerankResult {
                index: 0,
                relevance_score: 0.1,
            },
            RerankResult {
                index: 1,
                relevance_score: 0.5,
            },
        ];
        let placed = place_scores(&results, 3).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(placed, vec![0.1, 0.5, 0.9]);
    }

    #[test]
    fn a_short_answer_is_a_mismatch_rather_than_a_silent_gap() {
        let results = vec![RerankResult {
            index: 0,
            relevance_score: 0.4,
        }];
        assert!(place_scores(&results, 3).is_err());
    }

    #[test]
    fn nothing_to_score_asks_the_endpoint_nothing() {
        // Port 1 is reserved and unbound, so anything sent would fail: an
        // empty answer proves no request left.
        let client = HttpReranker::new("http://127.0.0.1:1")
            .unwrap_or_else(|e| panic!("building a client should not need a server: {e}"));
        let scored = client
            .score("anything", &[])
            .unwrap_or_else(|e| panic!("an empty pass should succeed: {e}"));
        assert!(scored.is_empty());
    }

    #[test]
    fn a_failure_status_is_reported_as_unreachable() {
        let mut server = mockito::Server::new();
        let mock = server.mock("POST", "/v1/rerank").with_status(503).create();
        let client =
            HttpReranker::new(&server.url()).unwrap_or_else(|e| panic!("client should build: {e}"));
        let failure = client
            .score("q", &["a document".to_owned()])
            .expect_err("a 503 is not a ranking");
        mock.assert();
        assert!(
            failure.to_string().contains("503"),
            "the status should be named: {failure}"
        );
    }

    #[test]
    fn an_unreadable_answer_is_reported_rather_than_scored() {
        let mut server = mockito::Server::new();
        let _mock = server
            .mock("POST", "/v1/rerank")
            .with_body("not json at all")
            .create();
        let client =
            HttpReranker::new(&server.url()).unwrap_or_else(|e| panic!("client should build: {e}"));
        assert!(client.score("q", &["a document".to_owned()]).is_err());
    }

    #[test]
    fn an_out_of_range_index_is_a_mismatch() {
        let results = vec![
            RerankResult {
                index: 7,
                relevance_score: 0.4,
            },
            RerankResult {
                index: 0,
                relevance_score: 0.4,
            },
        ];
        assert!(place_scores(&results, 2).is_err());
    }
}
