//! Verify, against a live endpoint, that the model answering `kb`'s
//! embedding requests is the model whose vectors are already in the
//! database.
//!
//! These tests are `#[ignore]`d: they require a running endpoint and are
//! not part of the CI gate. Run them with
//! `mise run verify:embedding-endpoint`.
//!
//! The response's `model` field is *not* evidence. The endpoint kb runs
//! against on this machine has been observed returning one model's
//! vectors under another model's name, and to answer a request naming a
//! model it does not serve with whatever model happens to be loaded. The
//! only reliable identification is the vector itself: a model is what it
//! computes.
//!
//! A fingerprint is the vector a model produces for [`PROBE`], stored as
//! packed little-endian f32 under `resources/fingerprints/<model>.f32`.
//! To record one for a new model, run the endpoint, then write the bytes
//! of `CliEmbedder::embed_document(PROBE)` to that path.

use std::path::PathBuf;

use kb::cli_embed::CliEmbedder;
use kb::embedding::decode_embedding;
use kb::storage::cosine_similarity;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// The text every fingerprint is taken over. Changing this invalidates
/// every recorded fingerprint.
const PROBE: &str = "the quick brown fox";

/// Two runs of one model over one input agree to well within this; two
/// different models do not come close to it. The observed cosine between
/// Qwen3-Embedding-0.6B and BGE-large over [`PROBE`] is -0.009.
const AGREEMENT: f32 = 0.999;

fn fingerprint_path(model: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resources/fingerprints")
        .join(format!("{model}.f32"))
}

#[test]
#[ignore = "requires a live embedding endpoint"]
fn the_endpoint_serves_the_model_whose_vectors_are_in_the_database() -> TestResult {
    let embedder = CliEmbedder::from_env()?;
    let model = embedder.model().to_owned();
    let path = fingerprint_path(&model);
    if !path.exists() {
        return Err(format!(
            "no fingerprint recorded for {model}; kb cannot tell which model is answering \
             at {}. Record one at {} before embedding anything under this name.",
            embedder.endpoint(),
            path.display()
        )
        .into());
    }

    let expected = decode_embedding(&std::fs::read(&path)?)?;
    let actual = embedder.embed_document(PROBE)?;

    if actual.len() != expected.len() {
        return Err(format!(
            "{model} at {} returned {} dimensions; the recorded fingerprint has {}. \
             A different model is answering.",
            embedder.endpoint(),
            actual.len(),
            expected.len()
        )
        .into());
    }

    let agreement = cosine_similarity(&actual, &expected);
    assert!(
        agreement >= AGREEMENT,
        "{model} at {} disagrees with its recorded fingerprint (cosine {agreement:.6}, \
         required {AGREEMENT}). A different model is answering; embedding under this \
         name would corrupt the corpus.",
        embedder.endpoint()
    );
    Ok(())
}

#[test]
#[ignore = "requires a live embedding endpoint"]
fn the_configured_model_gets_qwen3s_query_instruction() -> TestResult {
    let embedder = CliEmbedder::from_env()?;
    let model = embedder.model();
    assert!(
        model.contains("qwen3-embedding"),
        "the configured model is {model}, which default_prefixes does not recognise as \
         Qwen3; queries would be embedded without its instruction prefix and rank worse \
         than the corpus was built for"
    );
    Ok(())
}
