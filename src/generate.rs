//! Generated artifacts, addressed by what produced them.
//!
//! A generated summary or question set is derived data like a vector or a
//! passage, but it is derived data with a price. Re-chunking and re-embedding
//! the kb corpus takes minutes; regenerating its summaries on the local model
//! is an overnight job, and T015 measured 59 to 175 hours over the mail
//! increment depending on form. If a rebuild means paying that, rebuilds stop
//! happening and the derived index quietly becomes authoritative again — which
//! is the failure this plan exists to undo (ST-002, ST-004).
//!
//! The answer is to key each artifact on what actually determines it: the
//! canonical content, the generator, and the prompt. Then a re-chunk
//! regenerates nothing, an embedding-model change regenerates nothing, and
//! only a generator or prompt change costs a full pass — which is correct,
//! because those are the changes that make the old artifact wrong.
//!
//! It also unlocks the measurement loop the architecture exists for:
//! regenerate for a named subset under a new prompt version, score it against
//! the ground truth, and keep or discard without touching anything else.

use crate::index::{Index, IndexError};
use thiserror::Error;

/// What is being asked of the document.
///
/// A form is part of the prompt identity rather than a separate column: a
/// different question asked of the same text under the same prompt version is
/// a different prompt, and keying them together would let one overwrite the
/// other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    /// One paragraph describing what the document is about.
    Summary,
    /// Questions the document answers, one per line.
    Questions,
}

impl Form {
    /// The stored form name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Summary => "summary",
            Self::Questions => "questions",
        }
    }

    /// The instruction sent to the model.
    ///
    /// Held verbatim from the T002 measurement campaign
    /// (`scripts/generate_index.py`, prompt version 2). Wording is part of the
    /// key, so a figure measured under one wording cannot be compared with a
    /// figure measured under another — which is exactly why changing this text
    /// must also change [`Plan::prompt_version`].
    #[must_use]
    pub const fn instruction(self) -> &'static str {
        match self {
            Self::Summary => {
                "Write one paragraph describing what this document is about, in the third \
                 person, as a reference work would describe it. Name the specific subjects, \
                 people, systems and decisions it covers. Do not editorialise, do not \
                 preface, and reply with the paragraph alone."
            }
            Self::Questions => {
                "Write five questions this document answers, one per line, as a person would \
                 naturally ask them in the first person where that fits. Be specific to the \
                 document's actual content. No numbering, no preamble, questions alone."
            }
        }
    }
}

/// What to generate, and under which prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// Which form to ask for.
    pub form: Form,
    /// The prompt's version, which changes whenever its wording does.
    pub prompt_version: u32,
}

impl Plan {
    /// A plan for one form under one prompt version.
    #[must_use]
    pub const fn new(form: Form, prompt_version: u32) -> Self {
        Self {
            form,
            prompt_version,
        }
    }

    /// The stored prompt identity: the form and its version together.
    #[must_use]
    pub fn prompt_key(&self) -> String {
        format!("{}/{}", self.form.as_str(), self.prompt_version)
    }
}

/// Something that turns a document into generated text.
///
/// A trait so the expensive thing is injectable: the guarantee this module
/// exists to provide — that an unchanged key issues no model call — is only
/// checkable if the calls can be counted (ENG-010).
pub trait Generator {
    /// What produced the artifact, recorded so a model change is legible as a
    /// different artifact rather than an unexplained difference.
    fn version(&self) -> String;

    /// Generate for one document.
    ///
    /// # Errors
    ///
    /// [`GeneratorError`] if the generator cannot answer.
    fn generate(&self, form: Form, document: &str) -> Result<String, GeneratorError>;
}

/// Why a generation could not be produced.
#[derive(Debug, Error)]
pub enum GeneratorError {
    /// The generator could not be reached or refused.
    #[error("generator unreachable: {0}")]
    Unreachable(String),
    /// The generator answered with nothing usable.
    ///
    /// Distinct from a transport failure because it is the more dangerous
    /// case: an empty completion stored as an artifact keys a blank against
    /// the source and is never regenerated.
    #[error("generator returned no usable text for {0}")]
    Empty(String),
}

/// Why a generation pass could not complete.
#[derive(Debug, Error)]
pub enum GenerateError {
    /// The index could not be read or written.
    #[error(transparent)]
    Index(#[from] IndexError),
    /// The generator failed on a record, named so the pass can be resumed.
    #[error("{record}: {source}")]
    Generator {
        /// Which record was being generated.
        record: String,
        /// What went wrong.
        source: GeneratorError,
    },
}

/// What a generation pass did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GenerateReport {
    /// Artifacts produced by contacting the generator.
    pub generated: usize,
    /// Records already carrying an artifact under this key, left alone.
    pub reused: usize,
    /// Record ids the index does not hold.
    ///
    /// Named rather than counted, because a subset run that quietly generated
    /// nothing would otherwise read as success.
    pub unknown: Vec<String>,
}

/// Generate for the named records, skipping any whose key is already present.
///
/// The key is the record's stream hash — its canonical content — with the
/// generator and prompt identities. A record whose text changed has a
/// different stream and is regenerated; a record that was merely re-chunked or
/// re-embedded is not.
///
/// Artifacts are committed as they are produced rather than at the end.
/// Generation is expensive and resumable, and a pass that discards an hour of
/// finished work because the endpoint died on the last record is a pass nobody
/// runs twice.
///
/// # Errors
///
/// [`GenerateError::Index`] if the index cannot be read or written, or
/// [`GenerateError::Generator`] naming the record the generator failed on.
pub fn generate_missing(
    index: &Index,
    plan: &Plan,
    generator: &impl Generator,
    records: &[String],
) -> Result<GenerateReport, GenerateError> {
    let mut report = GenerateReport::default();
    let prompt_key = plan.prompt_key();
    let version = generator.version();
    for record_id in records {
        let Some(source_hash) = index.stream_hash_of(record_id)? else {
            report.unknown.push(record_id.clone());
            continue;
        };
        if index
            .generated(&source_hash, &version, &prompt_key)?
            .is_some()
        {
            report.reused = report.reused.saturating_add(1);
            continue;
        }
        let document = index.record_text(record_id)?;
        let content = generator.generate(plan.form, &document).map_err(|source| {
            GenerateError::Generator {
                record: record_id.clone(),
                source,
            }
        })?;
        if content.trim().is_empty() {
            return Err(GenerateError::Generator {
                record: record_id.clone(),
                source: GeneratorError::Empty(record_id.clone()),
            });
        }
        index.put_generated(&source_hash, &version, &prompt_key, content.trim())?;
        report.generated = report.generated.saturating_add(1);
    }
    Ok(report)
}

/// Qwen3's convention for suppressing the thinking block.
///
/// Prepended to every prompt because without it this model exhausts its token
/// budget on reasoning and returns empty content — measured during T002 as
/// 2,000 reasoning tokens and nothing else. Part of the prompt, and therefore
/// part of what a prompt version identifies.
const NO_THINK: &str = "/no_think\n";

/// How much of a document the generator is shown.
///
/// Held from the T002 campaign so figures remain comparable. A longer document
/// costs proportionally more time for a summary that a reader would not
/// distinguish, and the opening of a record is what it is usually about.
pub const DOCUMENT_CHARS: usize = 6_000;

/// A generator backed by a local chat-completions endpoint.
///
/// Local by construction, like every other model this repository talks to:
/// the corpus is the operator's own notes and other people's correspondence.
#[derive(Debug)]
pub struct HttpGenerator {
    runtime: tokio::runtime::Runtime,
    client: reqwest::Client,
    base_url: String,
    model: String,
    budget: u32,
}

impl HttpGenerator {
    /// Build a generator against a local endpoint.
    ///
    /// # Errors
    ///
    /// [`GeneratorError::Unreachable`] if no async runtime can be started.
    pub fn new(base_url: &str, model: &str, budget: u32) -> Result<Self, GeneratorError> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| GeneratorError::Unreachable(e.to_string()))?;
        Ok(Self {
            runtime,
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_owned(),
            model: model.to_owned(),
            budget,
        })
    }

    /// Which model this generator speaks to.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }
}

/// The parts of a chat completion this module reads.
#[derive(serde::Deserialize)]
struct Completion {
    choices: Vec<Choice>,
}

#[derive(serde::Deserialize)]
struct Choice {
    message: CompletionMessage,
}

#[derive(serde::Deserialize)]
struct CompletionMessage {
    #[serde(default)]
    content: String,
}

impl Generator for HttpGenerator {
    fn version(&self) -> String {
        self.model.clone()
    }

    fn generate(&self, form: Form, document: &str) -> Result<String, GeneratorError> {
        let truncated: String = document.chars().take(DOCUMENT_CHARS).collect();
        let prompt = format!("{NO_THINK}{}\n\n{truncated}", form.instruction());
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.budget,
            "temperature": 0.0,
        });
        let url = format!("{}/chat/completions", self.base_url);
        let text = self.runtime.block_on(async {
            let response = self
                .client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| GeneratorError::Unreachable(e.to_string()))?;
            let status = response.status();
            if !status.is_success() {
                return Err(GeneratorError::Unreachable(format!(
                    "{url} answered {status}"
                )));
            }
            let parsed: Completion = response
                .json()
                .await
                .map_err(|e| GeneratorError::Unreachable(e.to_string()))?;
            Ok(parsed
                .choices
                .into_iter()
                .next()
                .map(|c| c.message.content)
                .unwrap_or_default())
        })?;
        let cleaned = strip_thinking(&text);
        if cleaned.trim().is_empty() {
            // A reasoning model that spent its budget thinking returns a
            // successful response with no content. Reporting that as an
            // artifact would key a blank against the source forever.
            return Err(GeneratorError::Empty(self.model.clone()));
        }
        Ok(cleaned)
    }
}

/// Drop a reasoning model's thinking block.
///
/// Emitted even under `/no_think` by some builds, and it is the model talking
/// to itself rather than about the document.
#[must_use]
pub fn strip_thinking(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.to_ascii_lowercase().find("<think>") {
        out.push_str(rest.get(..open).unwrap_or_default());
        let after = rest.get(open..).unwrap_or_default();
        match after.to_ascii_lowercase().find("</think>") {
            Some(close) => rest = after.get(close.saturating_add(8)..).unwrap_or_default(),
            // An unterminated block means everything after it is thinking.
            None => return out.trim().to_owned(),
        }
    }
    out.push_str(rest);
    out.trim().to_owned()
}
