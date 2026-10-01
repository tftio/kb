//! OpenAI-compatible embedding generation.
//!
//! Mirrors the Haskell `KB.Embedding` module: configuration via the
//! `KB_EMBEDDING_*` environment variables, an async client trait, and a
//! packed little-endian `float32` BLOB codec compatible with `sqlite-vec`.
//!
//! Stored documents and search queries are embedded through
//! [`PrefixedEmbedder`], which applies the model's role-specific
//! instruction. The distinction is enforced by having two methods rather
//! than one flag, because embedding a query as a document produces a
//! plausible vector and quietly worse rankings instead of an error.
//!
//! Unless both `KB_EMBEDDING_BASE_URL` and `KB_EMBEDDING_MODEL` are set,
//! [`read_embedding_config_from_env`] returns `None`, which signals
//! "embedding disabled" to the caller. Neither has a default — see
//! [`ENV_MODEL`] for why assuming a model identifier is worse than declining
//! to embed at all. Failures inside [`HttpEmbeddingClient::embed`] are
//! reported as [`EmbeddingError`]; they never panic.

#![allow(
    clippy::significant_drop_tightening,
    reason = "lock/connection/server guards are intentionally held for the operation's duration; early drop would break atomicity"
)]

use std::env;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

/// A failure while requesting an embedding from the endpoint.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// The HTTP request failed or the response body could not be parsed.
    #[error("embedding request failed: {0}")]
    Http(#[from] reqwest::Error),
    /// The endpoint returned a non-success status code.
    #[error("embedding endpoint returned status {0}")]
    Status(reqwest::StatusCode),
    /// The endpoint returned a success response carrying no embedding data.
    #[error("embeddings response contained no data")]
    EmptyResponse,
}

/// A packed embedding blob had a length that is not a multiple of 4.
#[derive(Debug, thiserror::Error)]
#[error("embedding blob length {0} is not a multiple of 4")]
pub struct DecodeEmbeddingError(pub usize);

/// Connection details for an OpenAI-compatible embedding endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingConfig {
    /// Base URL up to and including `/v1` (e.g. `http://localhost:1234/v1`).
    pub base_url: String,
    /// Model identifier passed in the `model` JSON field.
    pub model: String,
    /// Bearer token. LM Studio / Ollama ignore it; `OpenAI` requires it.
    pub api_key: Option<String>,
    /// Instruction prepended to text embedded as a stored document.
    pub document_prefix: String,
    /// Instruction prepended to text embedded as a search query.
    pub query_prefix: String,
}

/// The variables whose presence enables embedding at all.
///
/// Public because the "embedding is disabled" message has to name what to
/// set, and a message that hardcoded the names could drift from the
/// variables actually read.
pub const ENV_BASE_URL: &str = "KB_EMBEDDING_BASE_URL";

/// The model identifier. Required, with **no** default.
///
/// There was a default — `text-embedding-bge-large-en-v1.5` — and it became
/// a hazard the moment the corpus moved to another model. `embeddings.model`
/// is part of that table's primary key and the filter on every read path, so
/// a fallback identifier does not disable embedding; it selects whichever
/// corpus happens to carry that name. In practice that meant a few hundred
/// superseded rows sitting beside thousands of current ones, and the result
/// was cosines near zero with no error anywhere: a query embedded by a host that
/// no longer serves bge, scored against vectors from a model that no longer
/// exists.
///
/// Naming a different model here would move the same trap one migration
/// along. No identifier is safe to assume, so absence disables embedding
/// instead — a loud degradation rather than a quiet wrong answer.
pub const ENV_MODEL: &str = "KB_EMBEDDING_MODEL";
const ENV_API_KEY: &str = "KB_EMBEDDING_API_KEY";
const ENV_DOCUMENT_PREFIX: &str = "KB_EMBEDDING_DOCUMENT_PREFIX";
const ENV_QUERY_PREFIX: &str = "KB_EMBEDDING_QUERY_PREFIX";

/// Qwen3-Embedding takes a task instruction on the query side and embeds
/// documents bare. The wording and the exact `Instruct: …\nQuery:` framing
/// are the model card's, reproduced verbatim: the model was trained on this
/// template, and paraphrasing it degrades retrieval quietly rather than
/// visibly.
const QWEN3_QUERY_PREFIX: &str =
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:";

/// bge-family models document a query-side instruction and no document-side
/// one. The 502 vectors already stored were computed without it, which is a
/// small mismatch on the read path rather than a reason to leave the query
/// side unprefixed for every future model.
const BGE_QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

/// nomic-family models prefix both sides, and are the reason prefixes are
/// configuration rather than a query-side constant.
const NOMIC_DOCUMENT_PREFIX: &str = "search_document: ";
const NOMIC_QUERY_PREFIX: &str = "search_query: ";

/// Cosine below which a Qwen3-Embedding candidate is not worth returning.
///
/// Measured against the 1831-node corpus rather than chosen: off-topic
/// queries peak at 0.25–0.29 (`the price of tin on commodity markets`
/// → 0.2916) and topical ones at 0.52–0.81 (`git commit signing fails
/// because 1Password is locked` → 0.8068), with the band between empty.
/// 0.4 sits in that gap with margin either side.
///
/// **Re-derived 2026-08-13** against the 3886-node corpus by sweeping this
/// value over 42 ground-truth questions with `mise run eval:retrieval`
/// (endpoint LM Studio at
/// `http://127.0.0.1:1234/v1` serving `text-embedding-qwen3-embedding-0.6b`).
/// The sweep re-confirms 0.4 and the value is deliberately unchanged.
///
/// | floor | recall@5 | recall@10 | MRR | authored recall@10 |
/// |---|---|---|---|---|
/// | 0.34 | 0.524 | 0.690 | 0.414 | 0.400 |
/// | 0.36 | 0.524 | **0.714** | 0.415 | 0.467 |
/// | 0.38 | 0.524 | **0.714** | 0.414 | 0.467 |
/// | 0.40 | 0.524 | **0.714** | 0.415 | 0.467 |
/// | 0.42 | 0.571 | 0.690 | 0.415 | 0.400 |
/// | 0.46 | 0.619 | 0.667 | 0.411 | 0.333 |
/// | 0.50 | 0.619 | 0.690 | 0.435 | 0.400 |
/// | 0.70 | 0.476 | 0.500 | 0.407 | 0.200 |
///
/// Recall@10 has a genuine plateau at 0.36–0.40 and 0.4 sits inside it, so
/// the value is not an artifact of where the sweep happened to sample.
/// Raising it to 0.46–0.50 buys aggregate recall@5 (0.524 → 0.619) and is
/// rejected: it costs recall@10, cuts recall@10 over the hand-authored nodes
/// from 0.467 to 0.333, and doubles the questions retrieving nothing expected
/// at all, from 4 to 9. MRR is flat at 0.411–0.438 across the whole range and
/// is non-monotonic within it (0.411 at 0.46, 0.438 at 0.48), which is sample
/// noise at 42 questions rather than signal, and must not be optimised
/// against.
const QWEN3_MIN_SIMILARITY: f32 = 0.4;

/// The floor that excludes nothing, being the least a cosine can be.
///
/// Not `0.0`: a floor of zero silently discards every negatively-aligned
/// candidate, which is filtering, not the absence of it. Naming the true
/// lower bound keeps "no floor" expressible as an ordinary value, so the
/// comparison needs no special case for it.
pub const NO_MIN_SIMILARITY: f32 = -1.0;

/// The similarity floor a model's vectors are calibrated for, or
/// [`NO_MIN_SIMILARITY`] when none is recorded.
///
/// Recorded per model for the same reason the prefixes are: a cosine
/// threshold has no stable meaning across models. The value that separates
/// signal from noise under Qwen3 is not the value that does so under bge or
/// nomic, and nothing in the schema would catch a borrowed threshold
/// silently becoming wrong when the model changed. So an unrecognised model
/// gets **no** floor rather than another model's — an invented threshold is
/// worse than none, exactly as with [`default_prefixes`].
///
/// Callers may override this: `KB_EMBEDDING_MIN_SIMILARITY` for a
/// deployment, `kb search --min-similarity` for one query.
#[must_use]
pub fn default_min_similarity(model: &str) -> f32 {
    if model.to_ascii_lowercase().contains("qwen3-embedding") {
        QWEN3_MIN_SIMILARITY
    } else {
        NO_MIN_SIMILARITY
    }
}

/// The instruction prefixes a model expects, as `(document, query)`.
///
/// Modern embedding models treat stored documents and search queries
/// asymmetrically, and the asymmetry differs by family: Qwen3 instructs the
/// query only, nomic prefixes both, bge instructs the query with different
/// wording. Embedding both sides bare — what this code did before — is not
/// an error at any layer; it silently costs retrieval quality. An unknown
/// model gets no prefix, which is the only safe default: an invented
/// instruction is worse than none.
#[must_use]
pub fn default_prefixes(model: &str) -> (String, String) {
    let m = model.to_ascii_lowercase();
    if m.contains("qwen3-embedding") {
        (String::new(), QWEN3_QUERY_PREFIX.to_string())
    } else if m.contains("nomic") {
        (
            NOMIC_DOCUMENT_PREFIX.to_string(),
            NOMIC_QUERY_PREFIX.to_string(),
        )
    } else if m.contains("bge") {
        (String::new(), BGE_QUERY_PREFIX.to_string())
    } else {
        (String::new(), String::new())
    }
}

/// Which required variable is absent, or `None` when both are present.
///
/// [`read_embedding_config_from_env`] collapses every reason for absence into
/// a single `None`, which is all a *client* needs but not enough to explain
/// itself to an operator. Reporting the specific variable lets a caller
/// distinguish "embedding is not in use", which deserves silence, from "an
/// endpoint is configured but the model is missing", which is a typo or an
/// unexported variable and deserves saying so.
///
/// The base URL is reported in preference to the model when both are absent:
/// it is the variable that decides whether embedding is in use at all.
#[must_use]
pub fn missing_requirement(base_url: Option<&str>, model: Option<&str>) -> Option<&'static str> {
    let present = |v: Option<&str>| v.is_some_and(|s| !s.is_empty());
    if !present(base_url) {
        Some(ENV_BASE_URL)
    } else if !present(model) {
        Some(ENV_MODEL)
    } else {
        None
    }
}

/// [`missing_requirement`] over the process environment.
#[must_use]
#[allow(
    clippy::disallowed_methods,
    reason = "the embedding endpoint/model are deployment config read once at the binary edge (REPO_INVARIANTS.md #5)"
)]
pub fn missing_requirement_from_env() -> Option<&'static str> {
    let base_url = env::var(ENV_BASE_URL).ok();
    let model = env::var(ENV_MODEL).ok();
    missing_requirement(base_url.as_deref(), model.as_deref())
}

/// Build an [`EmbeddingConfig`] from process environment variables.
///
/// Returns `None` — meaning embedding is disabled — unless **both**
/// `KB_EMBEDDING_BASE_URL` and `KB_EMBEDDING_MODEL` are set to non-empty
/// values. The model has no default; see [`ENV_MODEL`] for why an assumed
/// identifier is worse than none. `KB_EMBEDDING_API_KEY` is optional, and
/// `KB_EMBEDDING_DOCUMENT_PREFIX` / `KB_EMBEDDING_QUERY_PREFIX` override
/// [`default_prefixes`].
#[must_use]
#[allow(
    clippy::disallowed_methods,
    reason = "the embedding endpoint/model are deployment config, not user-facing tunables, so they come from the environment (12-factor) and are read once at the binary edge (REPO_INVARIANTS.md #5)"
)]
pub fn read_embedding_config_from_env() -> Option<EmbeddingConfig> {
    config_from_env_inputs(
        env::var(ENV_BASE_URL).ok(),
        env::var(ENV_MODEL).ok(),
        env::var(ENV_API_KEY).ok(),
        env::var(ENV_DOCUMENT_PREFIX).ok(),
        env::var(ENV_QUERY_PREFIX).ok(),
    )
}

/// Pure variant of [`read_embedding_config_from_env`] — caller supplies
/// the env var values. Used by tests so they don't need to mutate the
/// process environment.
///
/// The prefixes treat an explicitly-empty value differently from the other
/// fields: `Some("")` means "this model takes no prefix" and is honoured,
/// where an empty `KB_EMBEDDING_MODEL` falls back to the default. An
/// operator who needs to switch a prefix off has no other way to say so,
/// and unsetting the variable would restore the default instead.
fn config_from_env_inputs(
    base_url: Option<String>,
    model: Option<String>,
    api_key: Option<String>,
    document_prefix: Option<String>,
    query_prefix: Option<String>,
) -> Option<EmbeddingConfig> {
    let base_url = base_url?;
    if base_url.is_empty() {
        return None;
    }
    // Required, and an empty value counts as absent: a model named the empty
    // string would match no stored row, so treating it as a model would turn
    // a misconfiguration into a silently empty vector ranking.
    let model = model.filter(|s| !s.is_empty())?;
    let (default_document, default_query) = default_prefixes(&model);
    Some(EmbeddingConfig {
        base_url,
        model,
        api_key: api_key.filter(|s| !s.is_empty()),
        document_prefix: document_prefix.unwrap_or(default_document),
        query_prefix: query_prefix.unwrap_or(default_query),
    })
}

/// Asynchronous embedding generator. Mirrors the Haskell
/// `EmbeddingClient = Text -> IO (Either Text (Vector Float))`.
#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    /// Compute an embedding vector for `input`.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError`] on network, status, or empty-response
    /// failure. Must never panic.
    async fn embed(&self, input: &str) -> Result<Vec<f32>, EmbeddingError>;
}

/// An [`EmbeddingClient`] that applies the model's role-specific
/// instruction prefix before embedding.
///
/// Embedding a query as though it were a document is the failure this type
/// exists to prevent. It raises no error at any layer — the endpoint
/// answers, a vector comes back, results are returned — and shows up only
/// as rankings that are quietly worse than they should be. Making the two
/// roles separate methods means the choice has to be made explicitly at
/// every call site rather than being an argument someone forgets to pass.
///
/// The wrapper is generic over the underlying client so the prefix logic is
/// testable against a stub, with no HTTP involved.
pub struct PrefixedEmbedder<C> {
    client: C,
    document_prefix: String,
    query_prefix: String,
}

impl<C: EmbeddingClient> PrefixedEmbedder<C> {
    /// Wrap `client`, taking both prefixes from `config`.
    pub fn new(client: C, config: &EmbeddingConfig) -> Self {
        Self {
            client,
            document_prefix: config.document_prefix.clone(),
            query_prefix: config.query_prefix.clone(),
        }
    }

    /// Embed `text` as a stored document.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError`] on network, status, or empty-response
    /// failure.
    pub async fn embed_document(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.client
            .embed(&apply_prefix(&self.document_prefix, text))
            .await
    }

    /// Embed `text` as a search query.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError`] on network, status, or empty-response
    /// failure.
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.client
            .embed(&apply_prefix(&self.query_prefix, text))
            .await
    }
}

/// Prepend `prefix` to `text`, or return `text` unchanged when the prefix is
/// empty.
///
/// Kept separate and allocation-free in the empty case because most models
/// prefix one side only, so half of all calls take this path.
#[must_use]
pub fn apply_prefix(prefix: &str, text: &str) -> String {
    if prefix.is_empty() {
        text.to_string()
    } else {
        format!("{prefix}{text}")
    }
}

/// HTTP-backed [`EmbeddingClient`] built on `reqwest::Client`.
pub struct HttpEmbeddingClient {
    base_url: String,
    model: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl HttpEmbeddingClient {
    fn endpoint(&self) -> String {
        format!("{}/embeddings", self.base_url)
    }
}

#[async_trait]
impl EmbeddingClient for HttpEmbeddingClient {
    async fn embed(&self, input: &str) -> Result<Vec<f32>, EmbeddingError> {
        let body = serde_json::json!({
            "model": self.model,
            "input": input,
        });
        let mut req = self.client.post(self.endpoint()).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(EmbeddingError::Status(status));
        }
        let parsed: EmbeddingResponse = resp.json().await?;
        parsed
            .data
            .into_iter()
            .next()
            .map(|item| item.embedding)
            .ok_or(EmbeddingError::EmptyResponse)
    }
}

/// Construct an HTTP-backed [`EmbeddingClient`].
///
/// Builds a `reqwest::Client` with a 30s timeout. The returned client is
/// `Send + Sync` and can be wrapped in `Arc<dyn EmbeddingClient>` and
/// shared across worker tasks.
///
/// # Panics
///
/// Panics only if the underlying `reqwest::Client::builder()` fails —
/// that path requires a TLS / system configuration error and is not
/// expected at runtime.
#[must_use]
pub fn http_embedding_client(config: EmbeddingConfig) -> HttpEmbeddingClient {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        // `build()` only fails if the system TLS backend cannot be
        // initialized; `Client::new()` is the documented default and
        // reproduces the same failure mode without an explicit `expect`.
        .unwrap_or_else(|_| reqwest::Client::new());
    HttpEmbeddingClient {
        base_url: config.base_url,
        model: config.model,
        api_key: config.api_key,
        client,
    }
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
}

/// The normalised mean of `vectors`, standing for the text they were
/// computed from taken as a whole.
///
/// Chunking gives every passage of a long node its own vector, which is
/// what makes a pinpoint query find the one relevant section. It also
/// leaves nothing representing the node *as a whole*, which is what a
/// thematic query — "that long conversation about the auth redesign" — is
/// actually asking for. The centroid restores that at the cost of one row
/// and no additional request: it approximates the embedding of the
/// concatenated text closely enough for ranking, and models the same
/// averaging a single-chunk embedding performs internally.
///
/// Normalised because cosine similarity is scale-invariant but the stored
/// blob is compared against vectors the model returns at unit length;
/// keeping the corpus uniform costs nothing and avoids surprises for any
/// future consumer that assumes it.
///
/// Returns `None` for an empty input, for vectors of differing lengths
/// (which cannot be averaged meaningfully and would indicate two models'
/// output mixed together), and for a mean of zero norm.
#[must_use]
pub fn centroid(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let first = vectors.first()?;
    let dims = first.len();
    if dims == 0 || vectors.iter().any(|v| v.len() != dims) {
        return None;
    }
    let mut sum = vec![0.0_f32; dims];
    for vector in vectors {
        for (acc, value) in sum.iter_mut().zip(vector.iter()) {
            *acc += *value;
        }
    }
    let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return None;
    }
    Some(sum.into_iter().map(|x| x / norm).collect())
}

/// Encode a slice of `f32` as packed little-endian bytes (sqlite-vec
/// compatible).
#[must_use]
pub fn encode_embedding(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Decode packed little-endian `f32` bytes back into a vector.
///
/// # Errors
///
/// Returns an error if `bytes.len()` is not a multiple of 4.
pub fn decode_embedding(bytes: &[u8]) -> Result<Vec<f32>, DecodeEmbeddingError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(DecodeEmbeddingError(bytes.len()));
    }
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        // `chunks_exact(4)` yields slices of exactly 4 bytes, so the
        // conversion always succeeds; the `Ok` guard avoids indexing or
        // an `expect` while preserving behavior.
        if let Ok(arr) = <[u8; 4]>::try_from(chunk) {
            out.push(f32::from_le_bytes(arr));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Matcher;
    use proptest::prelude::*;

    // ── embedding_config ─────────────────────────────────────────────

    /// `config_from_env_inputs` with both prefix variables unset, which is
    /// the case every pre-prefix test was written against.
    fn config_without_prefix_env(
        base_url: Option<String>,
        model: Option<String>,
        api_key: Option<String>,
    ) -> Option<EmbeddingConfig> {
        config_from_env_inputs(base_url, model, api_key, None, None)
    }

    #[test]
    fn embedding_config_returns_none_when_url_unset() {
        assert_eq!(config_without_prefix_env(None, None, None), None);
    }

    #[test]
    fn embedding_config_returns_none_when_url_empty() {
        assert_eq!(
            config_without_prefix_env(Some(String::new()), None, None),
            None
        );
    }

    /// The model has no default. It once defaulted to bge-large, and when
    /// the corpus moved to Qwen3 that fallback quietly pointed every read at
    /// 502 superseded rows beside 5,845 current ones — a wrong answer with no
    /// error anywhere. Absence must disable embedding, not choose a corpus.
    #[test]
    fn a_base_url_without_a_model_disables_embedding_rather_than_guessing() {
        assert_eq!(
            config_without_prefix_env(Some("http://x/v1".into()), None, None),
            None
        );
    }

    /// A configured endpoint and model is still the ordinary case.
    #[test]
    fn a_base_url_with_a_model_configures_embedding() {
        let cfg = config_without_prefix_env(
            Some("http://x/v1".into()),
            Some("text-embedding-qwen3-embedding-0.6b".into()),
            None,
        )
        .unwrap();
        assert_eq!(cfg.base_url, "http://x/v1");
        assert_eq!(cfg.model, "text-embedding-qwen3-embedding-0.6b");
        assert!(cfg.api_key.is_none());
    }

    #[test]
    fn embedding_config_passes_through_explicit_model() {
        let cfg = config_without_prefix_env(
            Some("http://x/v1".into()),
            Some("custom-model".into()),
            None,
        )
        .unwrap();
        assert_eq!(cfg.model, "custom-model");
    }

    /// A model is supplied because it is now required; these tests are about
    /// the api key, not about what happens without a model.
    #[test]
    fn embedding_config_carries_api_key_when_set() {
        let cfg = config_without_prefix_env(
            Some("http://x/v1".into()),
            Some("some-model".into()),
            Some("sk-abc".into()),
        )
        .unwrap();
        assert_eq!(cfg.api_key.as_deref(), Some("sk-abc"));
    }

    #[test]
    fn embedding_config_treats_empty_api_key_as_none() {
        let cfg = config_without_prefix_env(
            Some("http://x/v1".into()),
            Some("some-model".into()),
            Some(String::new()),
        )
        .unwrap();
        assert!(cfg.api_key.is_none());
    }

    // ── prefixes ─────────────────────────────────────────────────────

    #[test]
    fn qwen3_instructs_the_query_and_leaves_documents_bare() {
        let (document, query) = default_prefixes("Qwen3-Embedding-0.6B");
        assert_eq!(document, "");
        assert!(
            query.starts_with("Instruct: "),
            "Qwen3 takes the model card's instruction template, got {query:?}"
        );
        assert!(query.ends_with("\nQuery:"));
    }

    #[test]
    fn model_matching_is_case_insensitive() {
        assert_eq!(
            default_prefixes("qwen3-embedding-0.6b"),
            default_prefixes("Qwen3-Embedding-0.6B")
        );
    }

    #[test]
    fn bge_instructs_the_query_with_its_own_wording() {
        let (document, query) = default_prefixes("text-embedding-bge-large-en-v1.5");
        assert_eq!(document, "");
        assert_eq!(query, BGE_QUERY_PREFIX);
    }

    #[test]
    fn nomic_prefixes_both_sides() {
        let (document, query) = default_prefixes("nomic-embed-text-v1.5");
        assert_eq!(document, NOMIC_DOCUMENT_PREFIX);
        assert_eq!(query, NOMIC_QUERY_PREFIX);
    }

    #[test]
    fn an_unknown_model_gets_no_prefix() {
        // Inventing an instruction for a model that was not trained on one
        // is worse than sending none.
        assert_eq!(
            default_prefixes("some-future-model"),
            (String::new(), String::new())
        );
    }

    #[test]
    fn the_similarity_floor_is_recorded_for_the_model_it_was_calibrated_on() {
        assert!((default_min_similarity("text-embedding-qwen3-embedding-0.6b") - 0.4).abs() < 1e-6);
        // Case-insensitive on the same substring `default_prefixes` matches,
        // so the two cannot disagree about which model is which.
        assert!(
            (default_min_similarity("Qwen3-Embedding-0.6B")
                - default_min_similarity("qwen3-embedding-0.6b"))
            .abs()
                < 1e-6
        );
    }

    #[test]
    fn a_model_with_no_calibration_gets_no_floor_rather_than_a_borrowed_one() {
        // A cosine threshold means something different under every model, so
        // reusing Qwen3's would silently discard good candidates — or none.
        for model in [
            "some-future-model",
            "text-embedding-bge-large-en-v1.5",
            "nomic-embed-text-v1.5",
        ] {
            assert!(
                (default_min_similarity(model) - NO_MIN_SIMILARITY).abs() < f32::EPSILON,
                "{model} borrowed a floor"
            );
        }
    }

    /// The sentinel has to be the true lower bound of a cosine. Were it
    /// `0.0`, "no floor" would still discard every negatively-aligned
    /// candidate — filtering, dressed as its absence.
    #[test]
    fn the_absent_floor_cannot_exclude_any_possible_cosine() {
        const { assert!(NO_MIN_SIMILARITY <= -1.0) }
    }

    #[test]
    fn explicit_environment_values_override_the_model_defaults() {
        let cfg = config_from_env_inputs(
            Some("http://x/v1".into()),
            Some("Qwen3-Embedding-0.6B".into()),
            None,
            Some("doc: ".into()),
            Some("qry: ".into()),
        )
        .unwrap();
        assert_eq!(cfg.document_prefix, "doc: ");
        assert_eq!(cfg.query_prefix, "qry: ");
    }

    #[test]
    fn an_explicitly_empty_prefix_switches_the_default_off() {
        // Unlike the model field, an empty prefix is honoured rather than
        // replaced by the default: it is the only way an operator can say
        // "this model takes no instruction".
        let cfg = config_from_env_inputs(
            Some("http://x/v1".into()),
            Some("Qwen3-Embedding-0.6B".into()),
            None,
            None,
            Some(String::new()),
        )
        .unwrap();
        assert_eq!(cfg.query_prefix, "");
    }

    #[test]
    fn prefixes_default_from_the_model_when_unset() {
        let cfg = config_without_prefix_env(
            Some("http://x/v1".into()),
            Some("Qwen3-Embedding-0.6B".into()),
            None,
        )
        .unwrap();
        assert_eq!(cfg.query_prefix, QWEN3_QUERY_PREFIX);
        assert_eq!(cfg.document_prefix, "");
    }

    #[test]
    fn apply_prefix_leaves_text_untouched_when_the_prefix_is_empty() {
        assert_eq!(apply_prefix("", "hello"), "hello");
        assert_eq!(apply_prefix("p: ", "hello"), "p: hello");
    }

    /// Records the text handed to the transport, so a test can assert what
    /// [`PrefixedEmbedder`] actually sent without an HTTP endpoint.
    struct RecordingClient {
        seen: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl EmbeddingClient for RecordingClient {
        async fn embed(&self, input: &str) -> Result<Vec<f32>, EmbeddingError> {
            if let Ok(mut seen) = self.seen.lock() {
                seen.push(input.to_string());
            }
            Ok(vec![0.0])
        }
    }

    fn qwen3_config() -> EmbeddingConfig {
        config_without_prefix_env(
            Some("http://x/v1".into()),
            Some("Qwen3-Embedding-0.6B".into()),
            None,
        )
        .expect("base url is set")
    }

    #[tokio::test]
    async fn embed_query_and_embed_document_send_different_payloads() {
        // The whole point of the distinction. Under Qwen3 the same text
        // reaches the endpoint instructed as a query and bare as a
        // document; if these ever matched, the asymmetry would be lost
        // without any error to notice.
        let embedder = PrefixedEmbedder::new(
            RecordingClient {
                seen: std::sync::Mutex::new(Vec::new()),
            },
            &qwen3_config(),
        );
        embedder.embed_document("hybrid retrieval").await.unwrap();
        embedder.embed_query("hybrid retrieval").await.unwrap();

        let seen = embedder.client.seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 2);
        let document = seen.first().expect("document payload");
        let query = seen.get(1).expect("query payload");
        assert_ne!(document, query);
        assert_eq!(document, "hybrid retrieval");
        assert_eq!(query, &format!("{QWEN3_QUERY_PREFIX}hybrid retrieval"));
    }

    #[tokio::test]
    async fn a_model_with_no_prefixes_sends_the_two_roles_identically() {
        // Not every model is asymmetric, and for those the wrapper must be
        // transparent rather than inventing a difference.
        let config = config_without_prefix_env(
            Some("http://x/v1".into()),
            Some("some-future-model".into()),
            None,
        )
        .expect("base url is set");
        let embedder = PrefixedEmbedder::new(
            RecordingClient {
                seen: std::sync::Mutex::new(Vec::new()),
            },
            &config,
        );
        embedder.embed_document("text").await.unwrap();
        embedder.embed_query("text").await.unwrap();

        let seen = embedder.client.seen.lock().unwrap().clone();
        assert_eq!(seen, vec!["text".to_string(), "text".to_string()]);
    }

    /// An empty value is absence, not a model named the empty string —
    /// which would match no stored row and turn a misconfiguration into a
    /// silently empty vector ranking.
    #[test]
    fn an_empty_model_is_absence_rather_than_a_model() {
        assert_eq!(
            config_without_prefix_env(Some("http://x/v1".into()), Some(String::new()), None),
            None
        );
    }

    /// The classifier has to name the *specific* missing variable, because
    /// the whole point is telling "embedding is not in use" apart from "an
    /// endpoint is set and the model is missing" — states that were
    /// indistinguishable, and silent, once the model became required.
    #[test]
    fn the_missing_requirement_is_named_rather_than_merely_detected() {
        assert_eq!(missing_requirement(None, None), Some(ENV_BASE_URL));
        assert_eq!(
            missing_requirement(None, Some("some-model")),
            Some(ENV_BASE_URL),
            "the base URL decides whether embedding is in use at all"
        );
        assert_eq!(
            missing_requirement(Some("http://x/v1"), None),
            Some(ENV_MODEL)
        );
        assert_eq!(
            missing_requirement(Some("http://x/v1"), Some("some-model")),
            None
        );
    }

    /// An empty value is absence here too, matching
    /// [`config_from_env_inputs`]. Were the two to disagree, one would report
    /// a missing variable while the other built a client from it.
    #[test]
    fn an_empty_value_counts_as_missing_for_both_variables() {
        assert_eq!(missing_requirement(Some(""), Some("m")), Some(ENV_BASE_URL));
        assert_eq!(
            missing_requirement(Some("http://x/v1"), Some("")),
            Some(ENV_MODEL)
        );
        // The agreement itself: whatever the classifier calls missing, the
        // config builder must decline to build.
        for (url, model) in [
            (Some(""), Some("m")),
            (Some("http://x/v1"), Some("")),
            (None, None),
        ] {
            assert!(
                missing_requirement(url, model).is_some(),
                "classifier disagrees with the builder for {url:?}/{model:?}"
            );
            assert_eq!(
                config_from_env_inputs(
                    url.map(str::to_string),
                    model.map(str::to_string),
                    None,
                    None,
                    None
                ),
                None
            );
        }
    }

    /// No model identifier may be hardcoded as a fallback. A grep is the
    /// honest test here: the defect was a constant nobody looked at again
    /// after the corpus moved, and only absence of the string proves it gone.
    #[test]
    fn no_model_identifier_is_hardcoded_as_a_fallback() {
        // Only the code above `mod tests` counts: the test module names the
        // retired model deliberately, both to check bge's prefix and in this
        // very assertion.
        let source = include_str!("embedding.rs");
        let production = source.split("#[cfg(test)]").next().unwrap_or(source);
        let offenders: Vec<&str> = production
            .lines()
            .filter(|line| {
                let code = line.trim_start();
                !code.starts_with("//") && !code.starts_with("///")
            })
            .filter(|line| line.contains("bge-large-en-v1.5"))
            .collect();
        assert!(
            offenders.is_empty(),
            "a bge identifier survives outside a comment: {offenders:?}"
        );
    }

    // ── embedding_blob_codec ─────────────────────────────────────────

    #[test]
    fn embedding_blob_codec_empty_vec_roundtrips() {
        let v: Vec<f32> = vec![];
        assert_eq!(decode_embedding(&encode_embedding(&v)).unwrap(), v);
    }

    #[test]
    fn embedding_blob_codec_basic_roundtrip() {
        let v = vec![0.0, 1.0, -1.0, 0.5, f32::MIN, f32::MAX];
        let bytes = encode_embedding(&v);
        assert_eq!(bytes.len(), 4 * v.len());
        assert_eq!(decode_embedding(&bytes).unwrap(), v);
    }

    #[test]
    fn embedding_blob_codec_rejects_truncated_buffer() {
        let bad: Vec<u8> = vec![0; 7];
        let err = decode_embedding(&bad).unwrap_err();
        assert!(matches!(err, DecodeEmbeddingError(7)), "got: {err:?}");
    }

    #[test]
    fn embedding_blob_codec_is_little_endian() {
        // 1.0_f32 in IEEE-754 LE bytes is 00 00 80 3F.
        let bytes = encode_embedding(&[1.0_f32]);
        assert_eq!(bytes, vec![0x00, 0x00, 0x80, 0x3F]);
    }

    proptest! {
        #[test]
        fn embedding_blob_codec_property_roundtrip(
            v in proptest::collection::vec(any::<f32>(), 0..32)
        ) {
            let bytes = encode_embedding(&v);
            let back = decode_embedding(&bytes).unwrap();
            // f32 NaN is not == itself, so compare bit patterns.
            prop_assert_eq!(back.len(), v.len());
            for (a, b) in v.iter().zip(back.iter()) {
                prop_assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }

    // ── embedding_client ─────────────────────────────────────────────

    fn make_client(server_url: &str, api_key: Option<&str>) -> HttpEmbeddingClient {
        http_embedding_client(EmbeddingConfig {
            base_url: server_url.to_string(),
            model: "test-model".into(),
            api_key: api_key.map(str::to_string),
            document_prefix: String::new(),
            query_prefix: String::new(),
        })
    }

    #[tokio::test]
    async fn embedding_client_posts_model_and_input() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/embeddings")
            .match_header("content-type", "application/json")
            .match_body(Matcher::PartialJsonString(
                r#"{"model":"test-model","input":"hello"}"#.into(),
            ))
            .with_status(200)
            .with_body(r#"{"data":[{"embedding":[0.1,0.2,0.3]}]}"#)
            .create_async()
            .await;
        let client = make_client(&server.url(), None);
        let v = client.embed("hello").await.unwrap();
        assert_eq!(v, vec![0.1_f32, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn embedding_client_sets_bearer_auth_when_api_key_present() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/embeddings")
            .match_header("authorization", "Bearer sk-abc")
            .with_status(200)
            .with_body(r#"{"data":[{"embedding":[1.0]}]}"#)
            .create_async()
            .await;
        let client = make_client(&server.url(), Some("sk-abc"));
        let v = client.embed("hi").await.unwrap();
        assert_eq!(v, vec![1.0_f32]);
    }

    #[tokio::test]
    async fn embedding_client_omits_auth_when_api_key_absent() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/embeddings")
            .match_header("authorization", Matcher::Missing)
            .with_status(200)
            .with_body(r#"{"data":[{"embedding":[2.0]}]}"#)
            .create_async()
            .await;
        let client = make_client(&server.url(), None);
        let v = client.embed("hi").await.unwrap();
        assert_eq!(v, vec![2.0_f32]);
    }

    #[tokio::test]
    async fn embedding_client_returns_err_on_empty_data() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/embeddings")
            .with_status(200)
            .with_body(r#"{"data":[]}"#)
            .create_async()
            .await;
        let client = make_client(&server.url(), None);
        let err = client.embed("hi").await.unwrap_err();
        assert!(matches!(err, EmbeddingError::EmptyResponse), "got: {err:?}");
    }

    #[tokio::test]
    async fn embedding_client_returns_err_on_http_500() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/embeddings")
            .with_status(500)
            .with_body("boom")
            .create_async()
            .await;
        let client = make_client(&server.url(), None);
        let err = client.embed("hi").await.unwrap_err();
        assert!(matches!(err, EmbeddingError::Status(_)), "got: {err:?}");
    }

    #[tokio::test]
    async fn embedding_client_returns_err_on_malformed_json() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/embeddings")
            .with_status(200)
            .with_body("not json")
            .create_async()
            .await;
        let client = make_client(&server.url(), None);
        let err = client.embed("hi").await.unwrap_err();
        assert!(matches!(err, EmbeddingError::Http(_)), "got: {err:?}");
    }

    #[tokio::test]
    async fn embedding_client_returns_err_on_unreachable_endpoint() {
        // A random unused port; reqwest fails fast with a connection refused.
        let client = make_client("http://127.0.0.1:1", None);
        let err = client.embed("hi").await.unwrap_err();
        assert!(matches!(err, EmbeddingError::Http(_)), "got: {err:?}");
    }

    // ── http_embedding_client_async (criterion: http-embedding-client-uses-async-reqwest) ─

    /// Asserts the `HttpEmbeddingClient` is async-only — building inside an
    /// outer #[`tokio::main`] runtime never panics, and the `.embed()` call
    /// awaits async send/json APIs. Calling this test inside a tokio
    /// runtime (which v5's `reqwest::blocking` client could not) demonstrates
    /// the original bug is fixed at the unit level.
    #[tokio::test]
    async fn http_embedding_client_async_construction_does_not_panic_in_tokio() {
        let _client = http_embedding_client(EmbeddingConfig {
            base_url: "http://127.0.0.1:1/v1".into(),
            model: "test-model".into(),
            api_key: None,
            document_prefix: String::new(),
            query_prefix: String::new(),
        });
    }

    // ── embedding_client_async (criterion: embedding-trait-async) ─────

    /// Asserts the `EmbeddingClient` trait is async — a stub impl satisfying
    /// the trait must use an async fn (or equivalent), and the call site
    /// must await it. Compiles only against the new async signature.
    struct StubAsyncClient;

    #[async_trait]
    impl EmbeddingClient for StubAsyncClient {
        async fn embed(&self, _input: &str) -> Result<Vec<f32>, EmbeddingError> {
            Ok(vec![1.0, 2.0, 3.0])
        }
    }

    #[tokio::test]
    async fn embedding_client_async_trait_method_is_awaitable() {
        let client = StubAsyncClient;
        let v = client.embed("anything").await.unwrap();
        assert_eq!(v, vec![1.0_f32, 2.0, 3.0]);
    }

    // ── centroid ─────────────────────────────────────────────────────

    #[test]
    fn the_centroid_of_one_vector_is_that_vector_normalised() {
        let out = centroid(&[vec![3.0, 4.0]]).expect("a single vector has a centroid");
        assert!((out[0] - 0.6).abs() < 1e-6, "{out:?}");
        assert!((out[1] - 0.8).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn the_centroid_lies_between_the_vectors_it_averages() {
        // Two orthogonal unit vectors average to the diagonal between them,
        // equidistant from both — which is what "represents the whole"
        // means for a node covering two subjects.
        let out = centroid(&[vec![1.0, 0.0], vec![0.0, 1.0]]).expect("centroid exists");
        let expected = 1.0_f32 / 2.0_f32.sqrt();
        assert!((out[0] - expected).abs() < 1e-6, "{out:?}");
        assert!((out[1] - expected).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn the_centroid_is_unit_length() {
        let out = centroid(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![0.0, 1.0, 0.0],
        ])
        .expect("centroid exists");
        let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm was {norm}");
    }

    #[test]
    fn a_centroid_that_cannot_be_formed_is_none_rather_than_a_wrong_answer() {
        assert!(centroid(&[]).is_none(), "nothing to average");
        assert!(centroid(&[vec![]]).is_none(), "no dimensions to average");
        assert!(
            centroid(&[vec![1.0, 0.0], vec![1.0]]).is_none(),
            "ragged input means two models' output was mixed; averaging it would be meaningless"
        );
        assert!(
            centroid(&[vec![1.0, 0.0], vec![-1.0, 0.0]]).is_none(),
            "opposed vectors sum to zero, which has no direction to normalise"
        );
    }
}
