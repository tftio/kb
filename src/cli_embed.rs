//! The CLI's single edge onto the embedding endpoint.
//!
//! `kb` is synchronous and [`crate::embedding::EmbeddingClient::embed`] is
//! `async`, so something has to own a runtime. Confining that to one place
//! keeps `CLI-001` and `ENG-008` satisfiable: exactly one location reads
//! `KB_EMBEDDING_*`, exactly one constructs a runtime, and the
//! `clippy::disallowed_methods` allowance for environment access appears
//! once rather than at every call site that wants a vector.
//!
//! The error type separates **disabled**, **incomplete** and **unreachable**
//! because callers must behave differently. Someone who has configured no
//! endpoint has chosen keyword-only operation and should not be warned on
//! every write. Someone whose endpoint is set but whose model is missing has
//! not chosen anything — that is a typo, and staying silent about it lets
//! every write accumulate without a vector. Someone whose daemon is down has
//! an operational problem worth reporting. A single error string would make
//! all three indistinguishable, which is how the middle case went unnoticed
//! when the model became required.

use crate::embedding::{
    EmbeddingConfig, EmbeddingError, HttpEmbeddingClient, PrefixedEmbedder, http_embedding_client,
    read_embedding_config_from_env,
};

/// A failure obtaining an embedding from the CLI.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// Embedding is not configured at all. A deliberate choice, not a fault.
    ///
    /// Both variables are named because both are required, and a message
    /// naming only the endpoint sent an operator whose model was missing
    /// looking in the wrong place.
    #[error(
        "embedding is disabled: {} and {} must both be set",
        crate::embedding::ENV_BASE_URL,
        crate::embedding::ENV_MODEL
    )]
    Disabled,
    /// Configured in part: an endpoint is set but a required variable is not.
    ///
    /// Distinct from [`Self::Disabled`] because the two mean opposite things
    /// about intent. Nobody sets an endpoint in order *not* to embed, so this
    /// is a typo or a variable that never got exported — and left silent it
    /// lets every subsequent write accumulate without a vector.
    #[error("embedding is configured in part: {missing} is not set")]
    Incomplete {
        /// The environment variable whose absence caused this, so the message
        /// names what to set rather than describing the shape of the problem.
        missing: &'static str,
    },
    /// An endpoint is configured but did not answer.
    #[error("embedding endpoint {endpoint} did not answer: {source}")]
    Unreachable {
        /// The endpoint that was tried, so the message names what to check.
        endpoint: String,
        /// The underlying transport or protocol failure.
        #[source]
        source: EmbeddingError,
    },
    /// The async runtime could not be constructed.
    #[error("could not start the embedding runtime: {0}")]
    Runtime(#[from] std::io::Error),
}

impl EmbedError {
    /// Whether embedding was deliberately left unconfigured.
    ///
    /// Callers that stay silent on a configuration choice and warn on
    /// everything else branch on this rather than on a message. It is
    /// **false** for [`Self::Incomplete`] on purpose: a half-configured
    /// environment is a mistake, not a choice, so it belongs in the warning
    /// branch alongside an unreachable endpoint. That is the whole mechanism
    /// — no call site needs to know the new variant exists.
    #[must_use]
    pub const fn is_disabled(&self) -> bool {
        matches!(self, Self::Disabled)
    }
}

/// Character budget for one embedded chunk.
///
/// Roughly two pages of prose. The size is chosen for *retrieval*, not for
/// the model's limit: an embedding averages everything in its input, so a
/// chunk covering many subjects yields a vector close to none of them. Two
/// pages is about the span over which a passage stays on one subject, which
/// is what lets the per-node maximum in `rank_by_embedding` surface a node
/// for one sharply relevant section.
///
/// A budget near the model's 32k-token window would sit far above almost
/// every node in the corpus, so nearly nothing would chunk and the vector
/// half of hybrid search would be one coarse average per node — the very
/// condition chunking exists to remove.
///
/// The cost is more rows and more requests, neither of which binds here: at
/// 1024 dimensions a vector is 4KB, so even several thousand chunks is tens
/// of megabytes against an already 290MB database, and the requests go to a
/// local daemon.
pub const DOCUMENT_CHUNK_CHARS: usize = 6_000;

/// Embed `doc` as the rows to store for it, nearest-first by chunk index.
///
/// Chunk 0 is the **whole-node** vector and chunks 1..n are the individual
/// passages. Both are needed and they answer different questions: a
/// thematic query ("that conversation about the auth redesign") matches the
/// node taken as a whole, while a pinpoint query matches one passage.
/// Storing only passages would lose the first; storing only the whole would
/// lose the second, which is the defect this work exists to fix. Since
/// `rank_by_embedding` scores a node by its best row, whichever of the two
/// fits the query wins, with no need to guess in advance which kind of
/// query is being asked.
///
/// A document that fits in one chunk yields exactly one row: the centroid
/// of a single vector is that vector, so there is nothing to add.
///
/// The whole-node vector is the centroid of the passage vectors rather than
/// a separate embedding of the whole text, which costs no extra request and
/// works for nodes far past the model's window.
///
/// # Errors
///
/// Propagates the first [`EmbedError`] from embedding any chunk. A partial
/// result is never returned: half a node's passages would rank the node on
/// an arbitrary subset of its content.
pub fn embed_document_rows(
    embedder: &CliEmbedder,
    doc: &tftio_org::ast::Document,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    let chunks = crate::embed_text::chunk_document(doc, DOCUMENT_CHUNK_CHARS);
    let mut vectors = Vec::with_capacity(chunks.len() + 1);
    for chunk in &chunks {
        vectors.push(embedder.embed_document(chunk)?);
    }
    if vectors.len() < 2 {
        return Ok(vectors);
    }
    match crate::embedding::centroid(&vectors) {
        Some(whole) => {
            vectors.insert(0, whole);
            Ok(vectors)
        }
        // Ragged or degenerate vectors cannot be averaged. The passages are
        // still worth storing, so the node keeps pinpoint recall and loses
        // only the thematic row.
        None => Ok(vectors),
    }
}

/// The model identifier configured for this process, if any.
///
/// Callers that only need to *read* stored vectors — `kb similar` — must
/// know which model's rows to filter by, but have no reason to construct a
/// runtime or contact anything. This keeps their environment access in the
/// same one place as everyone else's.
#[must_use]
pub fn configured_model() -> Option<String> {
    read_embedding_config_from_env().map(|c| c.model)
}

/// Synchronous access to the configured embedding endpoint.
///
/// Owns a current-thread `tokio` runtime for the process's lifetime.
/// Current-thread rather than multi-threaded because the CLI issues one
/// request at a time and blocks on it; a worker pool would be threads that
/// never run anything.
pub struct CliEmbedder {
    runtime: tokio::runtime::Runtime,
    embedder: PrefixedEmbedder<HttpEmbeddingClient>,
    model: String,
    endpoint: String,
}

impl CliEmbedder {
    /// Build from the process environment.
    ///
    /// # Errors
    ///
    /// [`EmbedError::Disabled`] when nothing is configured,
    /// [`EmbedError::Incomplete`] when an endpoint is configured but another
    /// required variable is not, or [`EmbedError::Runtime`] if the runtime
    /// cannot start.
    pub fn from_env() -> Result<Self, EmbedError> {
        match crate::embedding::missing_requirement_from_env() {
            None => Self::from_optional_config(read_embedding_config_from_env()),
            Some(var) if var == crate::embedding::ENV_BASE_URL => Err(EmbedError::Disabled),
            Some(missing) => Err(EmbedError::Incomplete { missing }),
        }
    }

    /// Build from an already-resolved configuration, treating `None` as
    /// disabled. The seam tests use to avoid mutating the process
    /// environment.
    ///
    /// # Errors
    ///
    /// [`EmbedError::Disabled`] when `config` is `None`, or
    /// [`EmbedError::Runtime`] if the runtime cannot start.
    pub fn from_optional_config(config: Option<EmbeddingConfig>) -> Result<Self, EmbedError> {
        let Some(config) = config else {
            return Err(EmbedError::Disabled);
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let model = config.model.clone();
        let endpoint = config.base_url.clone();
        let embedder = PrefixedEmbedder::new(http_embedding_client(config.clone()), &config);
        Ok(Self {
            runtime,
            embedder,
            model,
            endpoint,
        })
    }

    /// The model identifier vectors are stored under.
    ///
    /// Storage filters by this on read, so a caller writing a row and a
    /// caller ranking against it must agree; both take it from here.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The configured base URL, for error messages that must name what to
    /// check.
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Embed `text` as a stored document.
    ///
    /// # Errors
    ///
    /// [`EmbedError::Unreachable`] on any transport, status, or
    /// response-shape failure. Never panics.
    pub fn embed_document(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.block_on(self.embedder.embed_document(text))
    }

    /// Embed `text` as a search query.
    ///
    /// # Errors
    ///
    /// [`EmbedError::Unreachable`] on any transport, status, or
    /// response-shape failure. Never panics.
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.block_on(self.embedder.embed_query(text))
    }

    fn block_on(
        &self,
        future: impl Future<Output = Result<Vec<f32>, EmbeddingError>>,
    ) -> Result<Vec<f32>, EmbedError> {
        self.runtime
            .block_on(future)
            .map_err(|source| EmbedError::Unreachable {
                endpoint: self.endpoint.clone(),
                source,
            })
    }
}

#[cfg(test)]
#[allow(
    clippy::significant_drop_tightening,
    reason = "a mockito Server guard is held for the test's duration on purpose; dropping it early tears down the endpoint under test"
)]
mod tests {
    use super::{CliEmbedder, EmbedError};
    use crate::embedding::EmbeddingConfig;

    /// The distinction the write path branches on. `Disabled` must stay the
    /// only silent state: a half-configured environment is a mistake, and
    /// treating it as a choice let every write accumulate without a vector.
    #[test]
    fn only_the_unconfigured_state_counts_as_deliberately_disabled() {
        assert!(EmbedError::Disabled.is_disabled());
        assert!(
            !EmbedError::Incomplete {
                missing: crate::embedding::ENV_MODEL
            }
            .is_disabled(),
            "a half-configured environment must reach the warning branch"
        );
    }

    /// Each message must name the variable to set, since that is the only
    /// actionable thing either state can offer.
    #[test]
    fn both_configuration_errors_name_the_variables_to_set() {
        let disabled = EmbedError::Disabled.to_string();
        assert!(
            disabled.contains(crate::embedding::ENV_BASE_URL),
            "{disabled}"
        );
        assert!(disabled.contains(crate::embedding::ENV_MODEL), "{disabled}");

        let incomplete = EmbedError::Incomplete {
            missing: crate::embedding::ENV_MODEL,
        }
        .to_string();
        assert!(
            incomplete.contains(crate::embedding::ENV_MODEL),
            "{incomplete}"
        );
        assert!(
            !incomplete.contains("disabled"),
            "an incomplete configuration is not a disabled one: {incomplete}"
        );
    }

    fn config_for(base_url: &str) -> EmbeddingConfig {
        EmbeddingConfig {
            base_url: base_url.to_string(),
            model: "Qwen3-Embedding-0.6B".into(),
            api_key: None,
            document_prefix: String::new(),
            query_prefix: "Query:".into(),
        }
    }

    #[test]
    fn no_configuration_is_reported_as_disabled_not_as_a_failure() {
        let Err(err) = CliEmbedder::from_optional_config(None) else {
            panic!("no config must not yield an embedder");
        };
        assert!(err.is_disabled(), "got {err:?}");
        assert!(
            err.to_string().contains("KB_EMBEDDING_BASE_URL"),
            "the message must name the variable to set: {err}"
        );
    }

    #[test]
    fn an_unreachable_endpoint_is_a_distinct_variant_and_does_not_panic() {
        // Port 1 is reserved and never listening, so this exercises the
        // transport failure path without depending on a fixture server.
        let embedder = CliEmbedder::from_optional_config(Some(config_for("http://127.0.0.1:1/v1")))
            .expect("a runtime and client are constructible without contacting anything");
        let Err(err) = embedder.embed_query("anything") else {
            panic!("an unreachable endpoint cannot return a vector");
        };
        assert!(!err.is_disabled(), "got {err:?}");
        assert!(matches!(err, EmbedError::Unreachable { .. }), "got {err:?}");
        assert!(
            err.to_string().contains("127.0.0.1:1"),
            "the message must name the endpoint tried: {err}"
        );
    }

    #[test]
    fn constructing_an_embedder_contacts_nothing() {
        // Construction happens on every invocation of a command that might
        // embed, so it must not pay a round trip — nor fail — when the
        // daemon is down.
        assert!(
            CliEmbedder::from_optional_config(Some(config_for("http://127.0.0.1:1/v1"))).is_ok()
        );
    }

    #[test]
    fn the_model_and_endpoint_are_exposed_for_storage_and_error_messages() {
        let embedder = CliEmbedder::from_optional_config(Some(config_for("http://127.0.0.1:1/v1")))
            .expect("constructible");
        assert_eq!(embedder.model(), "Qwen3-Embedding-0.6B");
        assert_eq!(embedder.endpoint(), "http://127.0.0.1:1/v1");
    }

    #[test]
    fn a_synchronous_call_drives_the_async_client_to_completion() {
        // The point of the edge: a blocking caller gets a vector back from
        // an async transport, with no runtime of its own.
        let mut server = mockito::Server::new();
        let mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"data":[{"embedding":[0.25,0.5,0.75]}]}"#)
            .expect(2)
            .create();

        let base = format!("{}/v1", server.url());
        let embedder =
            CliEmbedder::from_optional_config(Some(config_for(&base))).expect("constructible");

        assert_eq!(
            embedder.embed_document("text").expect("document embeds"),
            vec![0.25, 0.5, 0.75]
        );
        assert_eq!(
            embedder.embed_query("text").expect("query embeds"),
            vec![0.25, 0.5, 0.75]
        );
        mock.assert();
    }

    #[test]
    fn an_error_status_is_reported_as_unreachable_rather_than_a_panic() {
        let mut server = mockito::Server::new();
        let _mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(500)
            .create();

        let base = format!("{}/v1", server.url());
        let embedder =
            CliEmbedder::from_optional_config(Some(config_for(&base))).expect("constructible");
        let Err(err) = embedder.embed_query("text") else {
            panic!("a 500 is not a vector");
        };
        assert!(matches!(err, EmbedError::Unreachable { .. }), "got {err:?}");
    }

    #[test]
    fn a_malformed_response_is_reported_rather_than_a_panic() {
        let mut server = mockito::Server::new();
        let _mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("not json at all")
            .create();

        let base = format!("{}/v1", server.url());
        let embedder =
            CliEmbedder::from_optional_config(Some(config_for(&base))).expect("constructible");
        assert!(embedder.embed_query("text").is_err());
    }
}
