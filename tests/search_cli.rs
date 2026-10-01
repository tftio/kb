//! CLI search-semantics tests for `kb search` and its `--match` flag.
//!
//! `kb search` was documented as accepting FTS5 query syntax while in fact
//! rewriting every query into prefix-matched conjunctive keyword search, so
//! boolean queries returned nothing and exited 0. These tests
//! replay that reproduction against the real binary: the default mode keeps its
//! forgiving behavior, `--match` sends the expression to FTS5 verbatim, and a
//! malformed expression is a reported error rather than silence.
//!
//! The file also covers the hybrid ranking added on top: `kb search` fuses a
//! vector ranking into the result when an embedding endpoint answers, falls
//! back to keyword-only with a note on stderr when it does not, and is forced
//! keyword-only by `--no-vector`.
//!
//! Every spawned `kb` process is sandboxed: `HOME`, `XDG_CONFIG_HOME`, and
//! `--db` are pinned to the per-test temp directory, and every `KB_EMBEDDING_*`
//! variable is stripped unless a test sets one, so the result is identical on a
//! clean CI runner and on a developer machine with a daemon configured.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Variables that would otherwise leak a developer's live embedding daemon
/// into a test process and make ranking machine-dependent.
const EMBEDDING_ENV: [&str; 8] = [
    "KB_EMBEDDING_BASE_URL",
    "KB_EMBEDDING_MODEL",
    "KB_EMBEDDING_API_KEY",
    "KB_EMBEDDING_DOCUMENT_PREFIX",
    "KB_EMBEDDING_QUERY_PREFIX",
    "KB_EMBEDDING_MIN_SIMILARITY",
    // A configured cross-encoder reorders the top of every ranking, so
    // leaving it set would make these ordering assertions depend on a model
    // running on the developer's machine.
    "KB_RERANK_BASE_URL",
    "KB_RERANK_TOP_K",
];

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Outcome of one `kb` invocation.
struct Run {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

/// Run `kb --db <db> <args...>`, piping `stdin` in when supplied. Embedding
/// is disabled: no endpoint variable reaches the child.
fn run_kb(
    db: &Path,
    stdin: Option<&str>,
    args: &[&str],
) -> Result<Run, Box<dyn std::error::Error>> {
    run_kb_env(db, stdin, args, &[])
}

/// [`run_kb`] with `env` layered over the stripped embedding environment.
fn run_kb_env(
    db: &Path,
    stdin: Option<&str>,
    args: &[&str],
    env: &[(&str, &str)],
) -> Result<Run, Box<dyn std::error::Error>> {
    let bin = env!("CARGO_BIN_EXE_kb");
    let home = db
        .parent()
        .ok_or("db path has no parent directory for HOME sandbox")?;
    let mut cmd = Command::new(bin);
    cmd.env("HOME", home)
        .env("XDG_CONFIG_HOME", home)
        .arg("--db")
        .arg(db);
    for var in EMBEDDING_ENV {
        cmd.env_remove(var);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    for a in args {
        cmd.arg(a);
    }
    cmd.stdin(if stdin.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    if let Some(payload) = stdin {
        child
            .stdin
            .as_mut()
            .ok_or("kb child stdin missing")?
            .write_all(payload.as_bytes())?;
    }
    let out = child.wait_with_output()?;
    Ok(Run {
        status: out.status,
        stdout: String::from_utf8(out.stdout)?,
        stderr: String::from_utf8(out.stderr)?,
    })
}

fn temp_db(name: &str) -> Result<(tempfile::TempDir, PathBuf), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join(format!("{name}.db"));
    Ok((dir, db))
}

/// The two-node corpus from the kb#3 reproduction. Both nodes mention
/// `rust`; only Alpha says `ownership`.
fn seed(db: &Path) -> TestResult {
    for (id, body) in [
        ("a", "* Alpha\n\nrust ownership notes\n"),
        ("b", "* Beta\n\nembedding vectors for rust\n"),
    ] {
        let run = run_kb(db, Some(body), &["create", "--id", id])?;
        assert!(run.status.success(), "seed create failed: {}", run.stderr);
    }
    Ok(())
}

/// Search and return the matched ids, sorted so ranking order does not make
/// the assertions brittle.
fn ids(db: &Path, args: &[&str]) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let run = run_kb(db, None, args)?;
    assert!(
        run.status.success(),
        "search {args:?} failed: {}{}",
        run.stdout,
        run.stderr
    );
    let mut out: Vec<String> = run
        .stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| l.split_whitespace().next().map(str::to_string))
        .collect();
    out.sort();
    Ok(out)
}

/// Search and return the matched ids in ranked order.
fn ordered_ids(
    db: &Path,
    args: &[&str],
    env: &[(&str, &str)],
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let run = run_kb_env(db, None, args, env)?;
    assert!(
        run.status.success(),
        "search {args:?} failed: {}{}",
        run.stdout,
        run.stderr
    );
    Ok(run
        .stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| l.split_whitespace().next().map(str::to_string))
        .collect())
}

/// A mock OpenAI-shaped embeddings endpoint that answers every request with
/// `vector`. The returned `Server` must be held for the child's lifetime.
fn embedding_endpoint(vector: &[f32]) -> (mockito::ServerGuard, String) {
    let body = format!(
        r#"{{"data":[{{"embedding":[{}]}}]}}"#,
        vector
            .iter()
            .map(|f| format!("{f:?}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(body)
        .expect_at_least(1)
        .create();
    let url = format!("{}/v1", server.url());
    (server, url)
}

/// The index a sandboxed `kb` resolves under `HOME`.
fn index_of(db: &Path) -> Result<kb::index::Index, Box<dyn std::error::Error>> {
    let home = db
        .parent()
        .ok_or("db path has no parent directory for HOME sandbox")?;
    Ok(kb::index::Index::open(
        &home.join(".local/share/kb/index.db"),
    )?)
}

/// Give `id`'s passages the vectors in `chunks`, in order, under
/// `FIXTURE_MODEL`.
///
/// A vector is keyed by the span it was computed from since T029, so a
/// fixture assigns one per passage rather than one per "chunk index" of a
/// node. A record with fewer passages than vectors takes the ones it has
/// room for, which is what makes the multi-passage fixtures below explicit
/// about their headings.
fn write_chunks(db: &Path, id: &str, chunks: &[Vec<f32>]) -> TestResult {
    let index = index_of(db)?;
    let passages = index.passages(id)?;
    assert!(
        passages.len() >= chunks.len(),
        "{id} has {} passages for {} vectors",
        passages.len(),
        chunks.len()
    );
    for (passage, vector) in passages.iter().zip(chunks) {
        index.put_embedding(
            &passage.stream_hash,
            passage.span_start,
            passage.span_len,
            FIXTURE_MODEL,
            &kb::embedding::encode_embedding(vector),
        )?;
    }
    Ok(())
}

const FIXTURE_MODEL: &str = "fixture-embedding-model";

/// A three-node corpus whose vectors are hand-written rather than computed,
/// so the ranking under test is the fusion and not a model's behaviour.
///
/// The query vector is `[0, 1]`. Only `a` matches the keyword `alpha`. `c`
/// matches no keyword and its *first* passage is orthogonal to the query —
/// only its third aligns, which is why it is written with three headings.
/// `b` is negatively aligned throughout.
fn seed_vectors(db: &Path) -> TestResult {
    for (id, body) in [
        ("a", "* Alpha\n\nalpha ownership notes\n"),
        ("b", "* Beta\n\nunrelated material\n"),
        (
            "c",
            "* Gamma\n\nnothing lexical in common\n\n* Delta\n\nnor here\n\n\
             * Epsilon\n\nnor in this one\n",
        ),
    ] {
        let run = run_kb(db, Some(body), &["create", "--id", id])?;
        assert!(run.status.success(), "seed create failed: {}", run.stderr);
    }
    write_chunks(db, "a", &[vec![1.0, 0.0]])?;
    write_chunks(db, "b", &[vec![0.0, -1.0]])?;
    write_chunks(db, "c", &[vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]])?;
    Ok(())
}

/// The headline behaviour: a node reachable by no keyword, whose *later*
/// chunk aligns with the query, is surfaced by the hybrid ranking.
#[test]
fn a_later_chunk_carries_its_node_into_hybrid_results() -> TestResult {
    let (_dir, db) = temp_db("hybrid")?;
    seed_vectors(&db)?;
    let (_server, url) = embedding_endpoint(&[0.0, 1.0]);
    let env = [
        ("KB_EMBEDDING_BASE_URL", url.as_str()),
        ("KB_EMBEDDING_MODEL", FIXTURE_MODEL),
    ];

    let hybrid = ordered_ids(&db, &["search", "alpha"], &env)?;
    assert!(
        hybrid.contains(&"c".to_string()),
        "a node matched only by its third chunk must appear: {hybrid:?}"
    );
    // Keyword ranking is [a]; vector ranking is [c, a, b]. RRF (k=60):
    // a = 1/61 + 1/62, c = 1/61, b = 1/63.
    assert_eq!(hybrid, vec!["a", "c", "b"]);

    // Without the vector half none of that is reachable.
    let keyword_only = ordered_ids(&db, &["search", "alpha", "--no-vector"], &env)?;
    assert_eq!(keyword_only, vec!["a"]);
    Ok(())
}

/// `--no-vector` is the reproducible mode: identical bytes with and without
/// a reachable endpoint, and no note, because nothing was degraded.
#[test]
fn no_vector_output_is_byte_identical_to_keyword_only_search() -> TestResult {
    let (_dir, db) = temp_db("novector")?;
    seed_vectors(&db)?;
    let (_server, url) = embedding_endpoint(&[0.0, 1.0]);

    for args in [
        vec!["search", "alpha", "--no-vector"],
        vec!["search", "alpha", "--no-vector", "--json"],
        vec!["search", "--match", "alpha OR beta", "--no-vector"],
    ] {
        let configured = run_kb_env(
            &db,
            None,
            &args,
            &[
                ("KB_EMBEDDING_BASE_URL", url.as_str()),
                ("KB_EMBEDDING_MODEL", FIXTURE_MODEL),
            ],
        )?;
        let disabled = run_kb(&db, None, &args)?;
        assert_eq!(configured.stdout, disabled.stdout, "{args:?}");
        assert_eq!(configured.status.code(), disabled.status.code(), "{args:?}");
        assert!(
            configured.stderr.is_empty(),
            "--no-vector degrades nothing and must not warn, got: {}",
            configured.stderr
        );
    }
    Ok(())
}

/// With no daemon configured the search still answers, on a clean stdout,
/// and says on stderr that the ranking is keyword-only. An agent that cannot
/// tell the two rankings apart reads a thin result as an absent topic.
#[test]
fn an_unavailable_endpoint_degrades_loudly_on_stderr_and_never_on_stdout() -> TestResult {
    let (_dir, db) = temp_db("degraded")?;
    seed_vectors(&db)?;

    let unconfigured = run_kb(&db, None, &["search", "alpha"])?;
    assert!(
        unconfigured.status.success(),
        "a missing daemon is not a search failure: {}",
        unconfigured.stderr
    );
    assert_eq!(
        unconfigured
            .stdout
            .lines()
            .filter_map(|l| l.split_whitespace().next())
            .collect::<Vec<_>>(),
        vec!["a"],
        "keyword results must still be returned"
    );
    assert!(
        unconfigured.stderr.contains("keyword-only"),
        "the degradation must be stated: {}",
        unconfigured.stderr
    );
    assert!(
        unconfigured.stderr.contains("KB_EMBEDDING_BASE_URL"),
        "the note must name what is unset: {}",
        unconfigured.stderr
    );

    // A configured-but-dead endpoint is the other half: port 1 is reserved
    // and never listening.
    let dead = run_kb_env(
        &db,
        None,
        &["search", "alpha"],
        &[
            ("KB_EMBEDDING_BASE_URL", "http://127.0.0.1:1/v1"),
            ("KB_EMBEDDING_MODEL", FIXTURE_MODEL),
        ],
    )?;
    assert!(dead.status.success(), "stderr: {}", dead.stderr);
    assert!(
        dead.stderr.contains("keyword-only") && dead.stderr.contains("127.0.0.1:1"),
        "the note must name the endpoint tried: {}",
        dead.stderr
    );
    assert!(
        !dead.stdout.contains("keyword-only"),
        "stdout must stay a clean payload: {}",
        dead.stdout
    );
    Ok(())
}

/// The default mode is unchanged: conjunctive, prefix-matched, forgiving.
#[test]
fn bare_search_is_prefix_matched_and_conjunctive() -> TestResult {
    let (_dir, db) = temp_db("bare")?;
    seed(&db)?;

    assert_eq!(ids(&db, &["search", "rust"])?, vec!["a", "b"]);
    assert_eq!(ids(&db, &["search", "rust ownership"])?, vec!["a"]);
    // Prefix matching: a shorter term is more forgiving than a longer one.
    assert_eq!(ids(&db, &["search", "embed"])?, vec!["b"]);
    Ok(())
}

/// The kb#3 headline reproduction, both halves: silence without the flag,
/// the union with it.
#[test]
fn disjunction_returns_nothing_without_match_and_the_union_with_it() -> TestResult {
    let (_dir, db) = temp_db("disjunction")?;
    seed(&db)?;

    let bare = run_kb(&db, None, &["search", "rust OR embedding"])?;
    assert!(
        bare.status.success(),
        "the defect is silence, not failure: {}",
        bare.stderr
    );
    assert!(
        bare.stdout.trim().is_empty(),
        "keyword mode should match nothing here, got: {}",
        bare.stdout
    );

    assert_eq!(
        ids(&db, &["search", "--match", "rust OR embedding"])?,
        vec!["a", "b"]
    );
    Ok(())
}

/// The other reproduction from the issue: a column filter.
#[test]
fn column_filter_works_only_under_match() -> TestResult {
    let (_dir, db) = temp_db("column")?;
    seed(&db)?;

    assert!(ids(&db, &["search", "title:Alpha"])?.is_empty());
    assert_eq!(ids(&db, &["search", "--match", "title:Alpha"])?, vec!["a"]);
    assert_eq!(
        ids(&db, &["search", "--match", "body:ownership"])?,
        vec!["a"]
    );
    Ok(())
}

#[test]
fn negation_and_phrases_work_under_match() -> TestResult {
    let (_dir, db) = temp_db("ops")?;
    seed(&db)?;

    assert_eq!(
        ids(&db, &["search", "--match", "rust NOT ownership"])?,
        vec!["b"]
    );
    assert_eq!(
        ids(&db, &["search", "--match", "\"rust ownership\""])?,
        vec!["a"]
    );
    assert!(
        ids(&db, &["search", "--match", "\"ownership rust\""])?.is_empty(),
        "a phrase query must not match the reversed order"
    );
    Ok(())
}

/// The new failure mode the flag introduces. A malformed expression must
/// exit non-zero and attribute the failure to `--match`, not present an
/// opaque `SQLite` string as a database fault.
#[test]
fn a_malformed_match_expression_is_reported_and_names_the_flag() -> TestResult {
    let (_dir, db) = temp_db("malformed")?;
    seed(&db)?;

    let run = run_kb(&db, None, &["search", "--match", "rust AND"])?;
    assert!(
        !run.status.success(),
        "a dangling operator should fail, got stdout: {}",
        run.stdout
    );
    let msg = format!("{}{}", run.stdout, run.stderr);
    assert!(
        msg.contains("fts5"),
        "error should surface the FTS5 diagnosis, got: {msg}"
    );
    assert!(
        msg.contains("--match"),
        "error should attribute the failure to the flag, got: {msg}"
    );

    // The same input is harmless without the flag: it is just three
    // literal keyword terms, so it matches nothing and exits 0.
    let bare = run_kb(&db, None, &["search", "rust AND"])?;
    assert!(
        bare.status.success(),
        "no input can be a syntax error in keyword mode: {}",
        bare.stderr
    );
    Ok(())
}

/// The `--help` text is the surface an operator reads. It must describe the
/// flag and must not repeat the FTS5 claim for the default mode.
#[test]
fn search_help_documents_match_and_drops_the_blanket_fts5_claim() -> TestResult {
    let (_dir, db) = temp_db("help")?;
    let run = run_kb(&db, None, &["search", "--help"])?;
    assert!(run.status.success(), "help failed: {}", run.stderr);
    assert!(
        run.stdout.contains("--match"),
        "help should list --match, got: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("--no-vector"),
        "help should list --no-vector, got: {}",
        run.stdout
    );
    assert!(
        !run.stdout.contains("FTS5 query string"),
        "the positional argument must no longer claim FTS5 syntax, got: {}",
        run.stdout
    );
    Ok(())
}

/// What the similarity figure must still tell an agent, now that the measured
/// calibration bands no longer fit the description.
///
/// The bands — 0.70-0.85 for transcripts, 0.41-0.67 for authored notes,
/// 0.47-0.82 for mail, all under text-embedding-qwen3-embedding-0.6b — were
/// carried in this description until 2026-09-10, when the description was
/// trimmed to fit Claude Code's `skillListingMaxDescChars`. Past that cap a
/// listing entry is dropped rather than truncated, so the figures were reaching
/// no agent at all. What survives is the conclusion they were there to support:
/// that similarity calibrates per corpus, and that a page of mid-range results
/// is not evidence of absence. The wording that inverted it must stay gone.
///
/// Asserted through the real binary rather than against the constant, since the
/// constant reaching the agent is the property that matters (CLI-002).
#[test]
fn the_agent_surface_warns_against_reading_absence_from_similarity() -> TestResult {
    let (_dir, db) = temp_db("calibration")?;
    let run = run_kb(&db, None, &["meta", "agent", "describe", "search"])?;
    assert!(run.status.success(), "describe failed: {}", run.stderr);

    assert!(
        run.stdout.contains("calibrates per corpus"),
        "the description must say similarity is not comparable across corpora, got: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("not evidence of absence"),
        "the description must warn against reading absence from a low score, got: {}",
        run.stdout
    );
    assert!(
        run.stdout.contains("NOT that the node scored zero"),
        "the description must distinguish a null score from a zero one, got: {}",
        run.stdout
    );
    assert!(
        !run.stdout
            .contains("a page of results in the 0.4s means the knowledge base has"),
        "the inverted guidance must stay gone, got: {}",
        run.stdout
    );
    Ok(())
}

/// `--json` must work in both modes; the envelope shape is unchanged.
#[test]
fn json_output_is_available_in_both_modes() -> TestResult {
    let (_dir, db) = temp_db("json")?;
    seed(&db)?;

    for args in [
        vec!["search", "rust", "--json"],
        vec!["search", "--match", "rust OR embedding", "--json"],
    ] {
        let run = run_kb(&db, None, &args)?;
        assert!(run.status.success(), "{args:?} failed: {}", run.stderr);
        assert!(
            run.stdout.contains("\"Alpha\""),
            "{args:?} should return Alpha, got: {}",
            run.stdout
        );
    }
    Ok(())
}

// ── Result-set bound ──────────────────────────────────────────────────

/// A corpus larger than the default limit, every node sharing one keyword so
/// the keyword side alone exceeds the bound. Vectors are written for every
/// node so both halves of the fusion are oversized.
/// `index / span` as an exact `f32`. Goes through `u16` so the conversion is
/// lossless rather than a `usize as f32` cast.
fn descending_component(index: usize, span: usize) -> f32 {
    let i = f32::from(u16::try_from(index).unwrap_or(u16::MAX));
    let n = f32::from(u16::try_from(span).unwrap_or(u16::MAX));
    i / n
}

fn seed_many(db: &Path, count: usize) -> TestResult {
    for i in 0..count {
        let id = format!("n{i:04}");
        let body = format!("* Node {i}\n\ncommonword body {i}\n");
        let run = run_kb(db, Some(&body), &["create", "--id", &id])?;
        assert!(run.status.success(), "seed create failed: {}", run.stderr);
        let x = 1.0 - descending_component(i, count);
        write_chunks(db, &id, &[vec![x, 1.0 - x]])?;
    }
    Ok(())
}

/// Before the bound existed, `kb search` printed every node in the database
/// for any query at all — on the live corpus, 1831 of 1831 for a query that
/// matched nothing. The fixtures held three nodes, where "returns
/// everything" and "returns the right three in order" are the same
/// observation, which is why the existing tests passed.
#[test]
fn search_returns_at_most_the_default_limit() -> TestResult {
    let (_dir, db) = temp_db("limit_default")?;
    seed_many(&db, 45)?;
    let (_server, url) = embedding_endpoint(&[0.0, 1.0]);
    let env = [
        ("KB_EMBEDDING_BASE_URL", url.as_str()),
        ("KB_EMBEDDING_MODEL", FIXTURE_MODEL),
    ];

    let hits = ordered_ids(&db, &["search", "commonword"], &env)?;
    assert_eq!(hits.len(), 20, "got {} hits: {hits:?}", hits.len());
    Ok(())
}

/// The bound must hold for the case that exposed it: a query matching no
/// keyword at all, where the result is the vector ranking alone.
#[test]
fn a_query_matching_no_keyword_returns_at_most_the_limit_not_the_corpus() -> TestResult {
    let (_dir, db) = temp_db("limit_nomatch")?;
    seed_many(&db, 45)?;
    let (_server, url) = embedding_endpoint(&[0.0, 1.0]);
    let env = [
        ("KB_EMBEDDING_BASE_URL", url.as_str()),
        ("KB_EMBEDDING_MODEL", FIXTURE_MODEL),
    ];

    let hits = ordered_ids(&db, &["search", "zzzznothingmatchesthis"], &env)?;
    assert_eq!(hits.len(), 20, "got {} hits: {hits:?}", hits.len());
    Ok(())
}

#[test]
fn the_limit_flag_is_honoured_above_and_below_the_default() -> TestResult {
    let (_dir, db) = temp_db("limit_flag")?;
    seed_many(&db, 45)?;
    let (_server, url) = embedding_endpoint(&[0.0, 1.0]);
    let env = [
        ("KB_EMBEDDING_BASE_URL", url.as_str()),
        ("KB_EMBEDDING_MODEL", FIXTURE_MODEL),
    ];

    for want in ["3", "30"] {
        let hits = ordered_ids(&db, &["search", "commonword", "--limit", want], &env)?;
        assert_eq!(
            hits.len(),
            want.parse::<usize>()?,
            "--limit {want} gave {} hits",
            hits.len()
        );
    }
    Ok(())
}

/// The limit truncates a ranking rather than changing it, so the first N of a
/// larger request must equal the whole of a request for N.
#[test]
fn a_smaller_limit_is_a_prefix_of_a_larger_one() -> TestResult {
    let (_dir, db) = temp_db("limit_prefix")?;
    seed_many(&db, 45)?;
    let (_server, url) = embedding_endpoint(&[0.0, 1.0]);
    let env = [
        ("KB_EMBEDDING_BASE_URL", url.as_str()),
        ("KB_EMBEDDING_MODEL", FIXTURE_MODEL),
    ];

    let wide = ordered_ids(&db, &["search", "commonword", "--limit", "40"], &env)?;
    let narrow = ordered_ids(&db, &["search", "commonword", "--limit", "7"], &env)?;
    assert_eq!(wide.get(..7).unwrap_or_default(), narrow.as_slice());
    Ok(())
}

#[test]
fn dense_pooling_and_candidate_width_are_cli_measurement_controls() -> TestResult {
    let (_dir, db) = temp_db("dense_pooling")?;
    for (id, body) in [
        (
            "long",
            "* Strong\n\nrelevant once\n* Weak one\n\nother\n* Weak two\n\nother again\n",
        ),
        ("short", "* Consistent\n\nrelevant throughout\n"),
    ] {
        let run = run_kb(&db, Some(body), &["create", "--id", id])?;
        assert!(run.status.success(), "seed create failed: {}", run.stderr);
    }
    write_chunks(
        &db,
        "long",
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.1, 0.9]],
    )?;
    write_chunks(&db, "short", &[vec![0.8, 0.2]])?;
    let (_server, url) = embedding_endpoint(&[1.0, 0.0]);
    let env = [
        ("KB_EMBEDDING_BASE_URL", url.as_str()),
        ("KB_EMBEDDING_MODEL", FIXTURE_MODEL),
    ];
    let base = [
        "search",
        "nothinglexical",
        "--min-similarity",
        "-1",
        "--limit",
        "2",
        "--vector-candidates",
        "2",
    ];
    let mut maximum = base.to_vec();
    maximum.extend(["--dense-pooling", "max"]);
    let mut mean = base.to_vec();
    mean.extend(["--dense-pooling", "mean-top-three"]);
    let mut normalized = base.to_vec();
    normalized.extend(["--dense-pooling", "length-normalized-max"]);

    assert_eq!(ordered_ids(&db, &base, &env)?, vec!["short", "long"]);
    assert_eq!(ordered_ids(&db, &maximum, &env)?, vec!["long", "short"]);
    assert_eq!(ordered_ids(&db, &mean, &env)?, vec!["short", "long"]);
    assert_eq!(ordered_ids(&db, &normalized, &env)?, vec!["short", "long"]);

    let narrow = ordered_ids(
        &db,
        &[
            "search",
            "nothinglexical",
            "--min-similarity",
            "-1",
            "--limit",
            "2",
            "--vector-candidates",
            "1",
            "--dense-pooling",
            "max",
        ],
        &env,
    )?;
    assert_eq!(narrow, vec!["long"]);
    Ok(())
}

/// Keyword-only search is unaffected below the limit: the bound must not have
/// changed the pre-hybrid implementation for the queries it already handled.
#[test]
fn keyword_only_search_below_the_limit_is_unchanged() -> TestResult {
    let (_dir, db) = temp_db("limit_keyword")?;
    seed(&db)?;
    assert_eq!(ids(&db, &["search", "rust"])?, vec!["a", "b"]);
    assert_eq!(ids(&db, &["search", "ownership"])?, vec!["a"]);
    Ok(())
}

// ── Similarity reporting and the per-model floor ──────────────────────

/// One search hit as the JSON envelope reports it: id and, where a vector
/// candidate contributed, its cosine.
type ScoredHit = (String, Option<f64>);

/// Extract `(id, similarity)` pairs from a `--json` search envelope.
fn json_similarities(
    db: &Path,
    args: &[&str],
    env: &[(&str, &str)],
) -> Result<Vec<ScoredHit>, Box<dyn std::error::Error>> {
    let run = run_kb_env(db, None, args, env)?;
    assert!(
        run.status.success(),
        "search {args:?} failed: {}{}",
        run.stdout,
        run.stderr
    );
    let v: serde_json::Value = serde_json::from_str(&run.stdout)?;
    let data = v
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or("envelope has no data array")?;
    Ok(data
        .iter()
        .map(|h| {
            let id = h
                .get("id")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
            (id, h.get("similarity").and_then(serde_json::Value::as_f64))
        })
        .collect())
}

/// `a` matches the keyword and is aligned with the query vector; `b` matches
/// the keyword only and is orthogonal to it.
fn seed_scored(db: &Path) -> TestResult {
    for (id, body) in [
        ("a", "* Alpha\n\nshared keyword here\n"),
        ("b", "* Beta\n\nshared keyword too\n"),
    ] {
        let run = run_kb(db, Some(body), &["create", "--id", id])?;
        assert!(run.status.success(), "seed create failed: {}", run.stderr);
    }
    // `a` scores 0.8 against the query vector [0, 1] and `b` scores 0.0, so
    // a floor of 0.9 prunes both and 0.5 keeps only `a`. Deliberately not
    // 1.0: a candidate identical to the query clears every floor a cosine
    // can express, which would make the pruning tests vacuous.
    write_chunks(db, "a", &[vec![0.6, 0.8]])?;
    write_chunks(db, "b", &[vec![1.0, 0.0]])?;
    Ok(())
}

#[test]
fn json_reports_the_cosine_for_a_vector_ranked_hit() -> TestResult {
    let (_dir, db) = temp_db("sim_report")?;
    seed_scored(&db)?;
    let (_server, url) = embedding_endpoint(&[0.0, 1.0]);
    let env = [
        ("KB_EMBEDDING_BASE_URL", url.as_str()),
        ("KB_EMBEDDING_MODEL", FIXTURE_MODEL),
    ];

    let hits = json_similarities(&db, &["search", "shared", "--json"], &env)?;
    let a = hits
        .iter()
        .find(|(id, _)| id == "a")
        .ok_or("node a missing")?;
    assert!(
        a.1.is_some_and(|s| (s - 0.8).abs() < 1e-4),
        "node a should score ~0.8, got {:?}",
        a.1
    );
    Ok(())
}

/// `null` must mean "no cosine was computed", not "the cosine was zero" —
/// under `--no-vector` there is no vector ranking to have an opinion.
#[test]
fn similarity_is_null_when_no_vector_ranking_contributed() -> TestResult {
    let (_dir, db) = temp_db("sim_null")?;
    seed_scored(&db)?;

    let hits = json_similarities(&db, &["search", "shared", "--no-vector", "--json"], &[])?;
    assert!(!hits.is_empty(), "keyword search returned nothing");
    assert!(
        hits.iter().all(|(_, s)| s.is_none()),
        "expected all-null similarities, got {hits:?}"
    );
    Ok(())
}

/// The floor prunes vector candidates but must never suppress a genuine
/// lexical match — that is not a cosine's business.
#[test]
fn the_floor_never_suppresses_a_keyword_match() -> TestResult {
    let (_dir, db) = temp_db("sim_floor_keyword")?;
    seed_scored(&db)?;
    let (_server, url) = embedding_endpoint(&[0.0, 1.0]);
    let env = [
        ("KB_EMBEDDING_BASE_URL", url.as_str()),
        ("KB_EMBEDDING_MODEL", FIXTURE_MODEL),
    ];

    // A floor above every stored vector's score: both keyword hits survive.
    let hits = json_similarities(
        &db,
        &["search", "shared", "--min-similarity", "0.9", "--json"],
        &env,
    )?;
    let mut ids: Vec<&str> = hits.iter().map(|(id, _)| id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec!["a", "b"], "keyword hits were filtered by cosine");
    assert!(
        hits.iter().all(|(_, s)| s.is_none()),
        "a pruned candidate must report null, got {hits:?}"
    );
    Ok(())
}

/// The case the floor exists for: nothing similar and nothing lexical, so
/// the result is empty rather than a page of weak neighbours.
#[test]
fn a_query_below_the_floor_with_no_keyword_match_returns_nothing() -> TestResult {
    let (_dir, db) = temp_db("sim_floor_empty")?;
    seed_scored(&db)?;
    let (_server, url) = embedding_endpoint(&[0.0, 1.0]);
    let env = [
        ("KB_EMBEDDING_BASE_URL", url.as_str()),
        ("KB_EMBEDDING_MODEL", FIXTURE_MODEL),
    ];

    let hits = json_similarities(
        &db,
        &[
            "search",
            "zzzznothingmatchesthis",
            "--min-similarity",
            "0.9",
            "--json",
        ],
        &env,
    )?;
    assert!(hits.is_empty(), "expected no hits, got {hits:?}");

    // Without the floor the same query returns the vector neighbours.
    let unfiltered = json_similarities(&db, &["search", "zzzznothingmatchesthis", "--json"], &env)?;
    assert!(
        !unfiltered.is_empty(),
        "the fixture must have vector neighbours for the floor to remove"
    );
    Ok(())
}

#[test]
fn the_environment_sets_the_floor_and_the_flag_overrides_it() -> TestResult {
    let (_dir, db) = temp_db("sim_precedence")?;
    seed_scored(&db)?;
    let (_server, url) = embedding_endpoint(&[0.0, 1.0]);

    // Env floor above every score: node `b`, a keyword-only hit, reports null.
    let env_high = [
        ("KB_EMBEDDING_BASE_URL", url.as_str()),
        ("KB_EMBEDDING_MODEL", FIXTURE_MODEL),
        ("KB_EMBEDDING_MIN_SIMILARITY", "0.9"),
    ];
    let hits = json_similarities(&db, &["search", "shared", "--json"], &env_high)?;
    assert!(
        hits.iter().all(|(_, s)| s.is_none()),
        "env floor was not applied: {hits:?}"
    );

    // The flag overrides the environment, restoring the scores.
    let hits = json_similarities(
        &db,
        &["search", "shared", "--min-similarity", "-1.0", "--json"],
        &env_high,
    )?;
    assert!(
        hits.iter().any(|(_, s)| s.is_some()),
        "--min-similarity did not override the environment: {hits:?}"
    );
    Ok(())
}

/// Whatever the floor says, `--no-vector` is the pre-hybrid implementation.
#[test]
fn no_vector_text_output_is_unaffected_by_any_floor_setting() -> TestResult {
    let (_dir, db) = temp_db("sim_no_vector")?;
    seed_scored(&db)?;
    let baseline = run_kb(&db, None, &["search", "shared", "--no-vector"])?.stdout;
    for floor in ["-1.0", "0.5", "0.9"] {
        let run = run_kb_env(
            &db,
            None,
            &["search", "shared", "--no-vector", "--min-similarity", floor],
            &[],
        )?;
        assert_eq!(
            run.stdout, baseline,
            "floor {floor} changed --no-vector output"
        );
    }
    Ok(())
}

/// A negative floor must be accepted as a *value*. Without
/// `allow_negative_numbers`, clap reads `-1.0` as an unknown flag and the
/// only way to disable the floor is the `=` form — which nothing documents.
#[test]
fn a_negative_floor_is_a_value_not_a_flag() -> TestResult {
    let (_dir, db) = temp_db("sim_negative")?;
    seed_scored(&db)?;
    for form in [
        vec!["search", "shared", "--min-similarity", "-1.0"],
        vec!["search", "shared", "--min-similarity=-1.0"],
    ] {
        let run = run_kb(&db, None, &form)?;
        assert!(
            run.status.success(),
            "{form:?} was rejected: {}{}",
            run.stdout,
            run.stderr
        );
    }
    Ok(())
}

/// A base URL with no model must degrade loudly, not rank against whichever
/// corpus a fallback identifier happens to name. The retired default was
/// bge-large, which on the live database meant 502 superseded rows beside
/// 5,845 current ones and cosines near zero with no error anywhere.
#[test]
fn an_endpoint_without_a_model_degrades_to_keyword_only_and_says_so() -> TestResult {
    let (_dir, db) = temp_db("no_model")?;
    seed_scored(&db)?;
    let (_server, url) = embedding_endpoint(&[0.0, 1.0]);

    let run = run_kb_env(
        &db,
        None,
        &["search", "shared", "--json"],
        &[("KB_EMBEDDING_BASE_URL", url.as_str())],
    )?;
    assert!(
        run.status.success(),
        "search should still succeed: {}{}",
        run.stdout,
        run.stderr
    );
    assert!(
        run.stderr.contains("keyword-only"),
        "stderr must say ranking degraded: {:?}",
        run.stderr
    );
    assert!(
        run.stderr.contains("KB_EMBEDDING_MODEL"),
        "stderr must name the missing variable: {:?}",
        run.stderr
    );
    // The degradation is on stderr only; stdout stays machine-readable.
    let v: serde_json::Value = serde_json::from_str(&run.stdout)?;
    let data = v
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or("envelope has no data array")?;
    assert!(!data.is_empty(), "keyword ranking returned nothing");
    assert!(
        data.iter()
            .all(|h| h.get("similarity").is_some_and(serde_json::Value::is_null)),
        "no vector ranking ran, so every similarity must be null: {data:?}"
    );
    Ok(())
}

/// kb tells agents that without `--match` no input is a syntax error. An odd
/// quotation mark falsified that: it closed the phrase kb's escaping opens,
/// and the rest of the query reached FTS5 as an expression. The error named
/// an FTS5 column, which reads as a corpus fault rather than an escaping one
/// — sending whoever debugs it in the wrong direction entirely.
#[test]
fn ordinary_prose_with_a_stray_quote_searches_rather_than_erroring() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");
    run_kb(
        &db,
        Some("* Rust\n\nOwnership and borrowing.\n"),
        &["create", "--id", "n1"],
    )?;

    for query in [
        "quote \" then rust: language",
        "trailing quote \"",
        "he said \"hello",
        "\"",
    ] {
        let run = run_kb(&db, None, &["search", query, "--json"])?;
        assert!(
            run.status.success(),
            "search failed for {query:?}: {}",
            run.stderr
        );
        assert!(
            !run.stdout.contains("\"ok\":false"),
            "search reported an error for {query:?}: {}",
            run.stdout
        );
    }
    Ok(())
}

/// `--match` is where the grammar is available, so a malformed expression is
/// a real error there and must still be reported as one, naming the flag.
#[test]
fn match_mode_still_reports_a_malformed_expression() -> TestResult {
    let dir = tempfile::tempdir()?;
    let db = dir.path().join("kb.db");
    run_kb(
        &db,
        Some("* Rust\n\nOwnership.\n"),
        &["create", "--id", "n1"],
    )?;

    let run = run_kb(
        &db,
        None,
        &["search", "unterminated \"", "--match", "--json"],
    )?;

    assert!(
        !run.status.success() || run.stdout.contains("\"ok\":false"),
        "a malformed --match expression must be reported: {} {}",
        run.stdout,
        run.stderr
    );
    Ok(())
}

/// A hit says which corpus it came from.
///
/// Added when the evaluation harness scored all eighteen mail questions as
/// misses: it matched an expectation's corpus against the hit's, the payload
/// carried no corpus, and every hit therefore read as a kb one. Retrieval had
/// been correct all along and the measurement said otherwise, which is the
/// worst kind of wrong a harness can be.
#[test]
fn a_json_hit_names_the_corpus_it_came_from() -> TestResult {
    let (_dir, db) = temp_db("hit-corpus")?;
    seed(&db)?;
    let run = run_kb(&db, None, &["search", "rust", "--no-vector", "--json"])?;
    assert!(run.status.success(), "{}{}", run.stdout, run.stderr);
    let envelope: serde_json::Value = serde_json::from_str(&run.stdout)?;
    let hits = envelope
        .get("data")
        .and_then(serde_json::Value::as_array)
        .ok_or("no data array in the search envelope")?;
    assert!(!hits.is_empty(), "the fixture returned no hits");
    for hit in hits {
        assert_eq!(
            hit.get("corpus").and_then(serde_json::Value::as_str),
            Some("kb"),
            "a hit does not name its corpus: {hit}"
        );
    }
    Ok(())
}

// ── --project / --context filtering (PLAN-20260923-project-identity T006) ──

/// Two records under two different projects, both matching the same keyword,
/// so `--project` is the only thing that can separate them.
fn seed_two_projects(db: &Path) -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let provenance_dir = tempfile::tempdir()?;
    for (id, project, body) in [
        (
            "proj-a",
            "alpha-project",
            "* Widget\n\nnotes about the widget launch\n",
        ),
        (
            "proj-b",
            "beta-project",
            "* Widget\n\nother notes about the widget rollout\n",
        ),
    ] {
        let path = provenance_dir.path().join(format!("{id}.json"));
        std::fs::write(&path, format!(r#"{{"project":"{project}"}}"#))?;
        let run = run_kb(
            db,
            Some(body),
            &[
                "create",
                "--id",
                id,
                "--provenance-json",
                path.to_str().ok_or("provenance path is not UTF-8")?,
            ],
        )?;
        assert!(run.status.success(), "seed create failed: {}", run.stderr);
    }
    Ok(provenance_dir)
}

/// `--project` restricts results to records asserting that slug, and a
/// search with no `--project` still finds both — the filter narrows, it does
/// not replace, keyword matching.
#[test]
fn project_flag_narrows_results_to_one_project() -> TestResult {
    let (_dir, db) = temp_db("project-filter")?;
    let _provenance = seed_two_projects(&db)?;

    let both = ids(&db, &["search", "widget", "--no-vector"])?;
    assert_eq!(
        both,
        vec!["proj-a", "proj-b"],
        "unfiltered search: {both:?}"
    );

    let only_alpha = ids(
        &db,
        &[
            "search",
            "widget",
            "--no-vector",
            "--project",
            "alpha-project",
        ],
    )?;
    assert_eq!(
        only_alpha,
        vec!["proj-a"],
        "--project alpha-project: {only_alpha:?}"
    );

    let only_beta = ids(
        &db,
        &[
            "search",
            "widget",
            "--no-vector",
            "--project",
            "beta-project",
        ],
    )?;
    assert_eq!(
        only_beta,
        vec!["proj-b"],
        "--project beta-project: {only_beta:?}"
    );
    Ok(())
}

/// `--json` carries the same provenance block `kb get --json` does, per
/// record, so a caller does not have to fetch each hit to see why it
/// matched the filter.
#[test]
fn json_search_results_carry_the_provenance_block() -> TestResult {
    let (_dir, db) = temp_db("project-filter-json")?;
    let _provenance = seed_two_projects(&db)?;

    let run = run_kb(
        &db,
        None,
        &[
            "search",
            "widget",
            "--no-vector",
            "--project",
            "alpha-project",
            "--json",
        ],
    )?;
    assert!(run.status.success(), "{}{}", run.stdout, run.stderr);
    let envelope: serde_json::Value = serde_json::from_str(&run.stdout)?;
    let hits = envelope
        .get("data")
        .and_then(serde_json::Value::as_array)
        .ok_or("no data array in the search envelope")?;
    assert_eq!(hits.len(), 1, "expected exactly one hit: {hits:?}");
    let provenance = hits
        .first()
        .and_then(|hit| hit.get("provenance"))
        .ok_or("hit carries no provenance field")?;
    assert_eq!(
        provenance
            .get("project")
            .and_then(serde_json::Value::as_str),
        Some("alpha-project"),
        "provenance block: {provenance}"
    );
    Ok(())
}

/// A search naming no project or context returns exactly what it returned
/// before this task: the invariant that a filter narrows rather than
/// changes unfiltered behavior.
#[test]
fn no_project_or_context_filter_changes_nothing() -> TestResult {
    let (_dir, db) = temp_db("project-filter-absent")?;
    let _provenance = seed_two_projects(&db)?;

    let unfiltered = ids(&db, &["search", "widget", "--no-vector"])?;
    let explicit_none = ids(&db, &["search", "widget", "--no-vector"])?;
    assert_eq!(unfiltered, explicit_none);
    assert_eq!(unfiltered, vec!["proj-a", "proj-b"]);
    Ok(())
}

/// `--project` is parsed against the slug grammar at the CLI boundary: an
/// invalid slug is a clear error rather than a filter that silently matches
/// nothing.
#[test]
fn an_invalid_project_slug_is_a_clear_error() -> TestResult {
    let (_dir, db) = temp_db("project-filter-invalid")?;
    seed(&db)?;

    let run = run_kb(
        &db,
        None,
        &["search", "rust", "--no-vector", "--project", "Not A Slug"],
    )?;
    assert!(
        !run.status.success(),
        "an invalid --project value must be rejected: {}",
        run.stdout
    );
    assert!(
        run.stderr.contains("project"),
        "the error should name the offending flag: {}",
        run.stderr
    );
    Ok(())
}

/// `--context` filters the same way `--project` does, independently.
#[test]
fn context_flag_narrows_results_to_one_context() -> TestResult {
    let (_dir, db) = temp_db("context-filter")?;
    let provenance_dir = tempfile::tempdir()?;
    for (id, context, body) in [
        (
            "ctx-a",
            "personal",
            "* Gadget\n\nnotes about the gadget launch\n",
        ),
        (
            "ctx-b",
            "work",
            "* Gadget\n\nother notes about the gadget rollout\n",
        ),
    ] {
        let path = provenance_dir.path().join(format!("{id}.json"));
        std::fs::write(&path, format!(r#"{{"context":"{context}"}}"#))?;
        let run = run_kb(
            &db,
            Some(body),
            &[
                "create",
                "--id",
                id,
                "--provenance-json",
                path.to_str().ok_or("provenance path is not UTF-8")?,
            ],
        )?;
        assert!(run.status.success(), "seed create failed: {}", run.stderr);
    }

    let only_personal = ids(
        &db,
        &["search", "gadget", "--no-vector", "--context", "personal"],
    )?;
    assert_eq!(only_personal, vec!["ctx-a"], "{only_personal:?}");
    Ok(())
}
