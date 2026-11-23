//! The read tools of the agent surface, called through [`ToolSurface::call`].
//!
//! These assert what each tool says about a real store and index, with no
//! protocol in between: the wire tests elsewhere assert only that the wire
//! carries what the surface produced.

use kb::mcp::{KbTools, ToolOutcome, ToolSurface};
use serde_json::{Value, json};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const MODEL: &str = "fixture-model";

/// A corpus of five records.
///
/// `alpha` links to `beta` by id and to `no-such-slug` by name, so it has an
/// outgoing id-link and a broken name-link, `beta` has an incoming link, and
/// `gamma`, `delta` and `lonely` have none. `alpha` and `gamma` carry vectors;
/// `lonely` has none.
struct Corpus {
    _dir: tempfile::TempDir,
    tools: KbTools,
}

fn corpus() -> Result<Corpus, Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let store_path = dir.path().join("store");
    let index_path = dir.path().join("index.db");
    let store = kb::store::GitBlobStore::open_or_init(&store_path)?;
    let index = kb::index::Index::open_for_rebuild(&index_path)?;
    for (id, body) in [
        (
            "alpha",
            "#+filetags: :rust: :silent-critic:\n* Alpha\n\nsee [[id:beta]] and [[no-such-slug]]\n",
        ),
        ("beta", "#+filetags: :rust:\n* Beta\n\nplain text\n"),
        ("gamma", "* Gamma\n\nmore plain text\n"),
        ("delta", "* Delta\n\neven more\n"),
        ("lonely", "* Lonely\n\nnever embedded\n"),
    ] {
        let document = kb::parser::parse_document(body)?;
        let options = kb::write::WriteOptions::note(id)?;
        kb::write::put_record(&store, &index, id, &document, &options)?;
    }
    for (id, vector) in [
        ("alpha", [1.0_f32, 0.0]),
        ("beta", [1.0, 1.0]),
        ("gamma", [0.0, 1.0]),
    ] {
        for passage in index.passages(id)? {
            index.put_embedding(
                &passage.stream_hash,
                passage.span_start,
                passage.span_len,
                MODEL,
                &kb::embedding::encode_embedding(&vector),
            )?;
        }
    }
    drop(index);
    Ok(Corpus {
        _dir: dir,
        tools: KbTools::with_paths(store_path, index_path),
    })
}

fn text(outcome: ToolOutcome) -> Result<Value, Box<dyn std::error::Error>> {
    match outcome {
        ToolOutcome::Text(body) => Ok(serde_json::from_str(&body)?),
        ToolOutcome::Failed(reason) => Err(format!("tool failed: {reason}").into()),
    }
}

fn failure(outcome: ToolOutcome) -> Result<String, Box<dyn std::error::Error>> {
    match outcome {
        ToolOutcome::Failed(reason) => Ok(reason),
        ToolOutcome::Text(body) => Err(format!("expected a failure, got: {body}").into()),
    }
}

fn ids(payload: &Value) -> Vec<String> {
    payload
        .get("results")
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .filter_map(|row| row.get("id").and_then(Value::as_str).map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

fn sorted(mut values: Vec<String>) -> Vec<String> {
    values.sort();
    values
}

#[test]
fn the_surface_lists_twenty_one_tools_and_the_required_fields_match_the_cli() -> TestResult {
    let corpus = corpus()?;
    let specs = corpus.tools.specs();
    let names: Vec<&str> = specs.iter().map(|spec| spec.name).collect();
    assert_eq!(
        names,
        vec![
            "search",
            "context",
            "get",
            "put",
            "recent",
            "list_by_tag",
            "tags",
            "links",
            "hubs",
            "orphans",
            "broken",
            "projects",
            "similar",
            "prompt_list",
            "prompt_show",
            "prompt_render",
            "update",
            "delete",
            "tags_add",
            "tags_rm",
            "tags_merge",
        ]
    );
    let required = |name: &str| -> Vec<String> {
        specs
            .iter()
            .find(|spec| spec.name == name)
            .and_then(|spec| spec.schema.get("required"))
            .and_then(Value::as_array)
            .map(|fields| {
                fields
                    .iter()
                    .filter_map(|f| f.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    };
    assert_eq!(required("list_by_tag"), ["tag"]);
    assert_eq!(required("links"), ["id"]);
    assert_eq!(required("similar"), ["id"]);
    assert_eq!(required("prompt_show"), ["name"]);
    assert_eq!(required("prompt_render"), ["name"]);
    assert_eq!(required("update"), ["id", "document"]);
    assert_eq!(required("delete"), ["id"]);
    assert_eq!(required("tags_add"), ["id", "tags"]);
    assert_eq!(required("tags_rm"), ["id", "tags"]);
    assert_eq!(required("tags_merge"), ["from", "to"]);
    for name in [
        "recent",
        "tags",
        "hubs",
        "orphans",
        "broken",
        "projects",
        "prompt_list",
    ] {
        assert!(required(name).is_empty(), "{name} requires nothing");
    }
    Ok(())
}

#[test]
fn recent_returns_identifiers_and_titles_within_the_limit() -> TestResult {
    let corpus = corpus()?;
    let all = text(corpus.tools.call("recent", &json!({})))?;
    assert_eq!(all.get("count").and_then(Value::as_u64), Some(5));
    assert_eq!(
        sorted(ids(&all)),
        ["alpha", "beta", "delta", "gamma", "lonely"]
    );
    let first = all
        .get("results")
        .and_then(Value::as_array)
        .and_then(|rows| rows.first())
        .ok_or("no rows")?;
    assert!(first.get("title").and_then(Value::as_str).is_some());
    assert!(first.get("body").is_none(), "a body leaked: {first}");

    let two = text(corpus.tools.call("recent", &json!({ "limit": 2 })))?;
    assert_eq!(ids(&two).len(), 2);
    Ok(())
}

#[test]
fn list_by_tag_normalizes_the_query_as_the_cli_does() -> TestResult {
    let corpus = corpus()?;
    let found = text(
        corpus
            .tools
            .call("list_by_tag", &json!({ "tag": "Silent Critic" })),
    )?;
    assert_eq!(ids(&found), ["alpha"]);
    let rust = text(corpus.tools.call("list_by_tag", &json!({ "tag": "rust" })))?;
    assert_eq!(sorted(ids(&rust)), ["alpha", "beta"]);
    let capped = text(
        corpus
            .tools
            .call("list_by_tag", &json!({ "tag": "rust", "limit": 1 })),
    )?;
    assert_eq!(capped.get("count").and_then(Value::as_u64), Some(1));
    assert_eq!(
        capped.get("total").and_then(Value::as_u64),
        Some(2),
        "a cut list must say how long it was"
    );
    let none = text(
        corpus
            .tools
            .call("list_by_tag", &json!({ "tag": "absent" })),
    )?;
    assert_eq!(none.get("count").and_then(Value::as_u64), Some(0));
    Ok(())
}

#[test]
fn list_by_tag_requires_a_tag() -> TestResult {
    let corpus = corpus()?;
    let reason = failure(corpus.tools.call("list_by_tag", &json!({})))?;
    assert!(reason.contains("tag is required"), "{reason}");
    Ok(())
}

#[test]
fn tags_reports_the_stored_vocabulary_most_used_first() -> TestResult {
    let corpus = corpus()?;
    let payload = text(corpus.tools.call("tags", &json!({})))?;
    let rows = payload
        .get("tags")
        .and_then(Value::as_array)
        .ok_or("no tags")?;
    let pairs: Vec<(&str, u64)> = rows
        .iter()
        .filter_map(|row| {
            Some((
                row.get("tag")?.as_str()?,
                row.get("count").and_then(Value::as_u64)?,
            ))
        })
        .collect();
    assert_eq!(pairs, vec![("rust", 2), ("silent-critic", 1)]);
    let capped = text(corpus.tools.call("tags", &json!({ "limit": 1 })))?;
    assert_eq!(capped.get("count").and_then(Value::as_u64), Some(1));
    assert_eq!(capped.get("total").and_then(Value::as_u64), Some(2));
    Ok(())
}

#[test]
fn links_reports_both_directions_with_link_types() -> TestResult {
    let corpus = corpus()?;
    let alpha = text(corpus.tools.call("links", &json!({ "id": "alpha" })))?;
    let outgoing = alpha
        .get("outgoing")
        .and_then(Value::as_array)
        .ok_or("no outgoing")?;
    assert!(
        outgoing.iter().any(|row| {
            row.get("link_type").and_then(Value::as_str) == Some("id")
                && row.get("target_id").and_then(Value::as_str) == Some("beta")
        }),
        "no id-link to beta: {alpha}"
    );
    assert!(
        outgoing.iter().any(|row| {
            row.get("link_type").and_then(Value::as_str) == Some("name")
                && row.get("target_slug").and_then(Value::as_str) == Some("no-such-slug")
                && row.get("target_id").is_some_and(Value::is_null)
        }),
        "no broken name-link: {alpha}"
    );
    let beta = text(corpus.tools.call("links", &json!({ "id": "beta" })))?;
    let incoming = beta
        .get("incoming")
        .and_then(Value::as_array)
        .ok_or("no incoming")?;
    assert!(
        incoming
            .iter()
            .any(|row| row.get("source_id").and_then(Value::as_str) == Some("alpha")),
        "beta is not linked from alpha: {beta}"
    );
    Ok(())
}

#[test]
fn links_of_an_unknown_record_fails_by_name() -> TestResult {
    let corpus = corpus()?;
    let reason = failure(corpus.tools.call("links", &json!({ "id": "nope" })))?;
    assert!(reason.contains("no node with id: nope"), "{reason}");
    let missing = failure(corpus.tools.call("links", &json!({})))?;
    assert!(missing.contains("id is required"), "{missing}");
    Ok(())
}

#[test]
fn hubs_rank_by_in_degree() -> TestResult {
    let corpus = corpus()?;
    let payload = text(corpus.tools.call("hubs", &json!({})))?;
    let first = payload
        .get("results")
        .and_then(Value::as_array)
        .and_then(|rows| rows.first())
        .ok_or("no hubs")?;
    assert_eq!(first.get("id").and_then(Value::as_str), Some("beta"));
    assert_eq!(first.get("in_degree").and_then(Value::as_u64), Some(1));
    let none = text(corpus.tools.call("hubs", &json!({ "limit": 1 })))?;
    assert_eq!(none.get("count").and_then(Value::as_u64), Some(1));
    Ok(())
}

#[test]
fn orphans_are_the_records_nothing_links_to() -> TestResult {
    let corpus = corpus()?;
    let payload = text(corpus.tools.call("orphans", &json!({})))?;
    assert_eq!(
        sorted(ids(&payload)),
        ["alpha", "delta", "gamma", "lonely"],
        "beta is linked to and must not be listed"
    );
    let capped = text(corpus.tools.call("orphans", &json!({ "limit": 1 })))?;
    assert_eq!(capped.get("count").and_then(Value::as_u64), Some(1));
    assert_eq!(capped.get("total").and_then(Value::as_u64), Some(4));
    Ok(())
}

#[test]
fn broken_lists_unresolved_bracket_references() -> TestResult {
    let corpus = corpus()?;
    let payload = text(corpus.tools.call("broken", &json!({})))?;
    let rows = payload
        .get("results")
        .and_then(Value::as_array)
        .ok_or("no results")?;
    assert_eq!(rows.len(), 1, "{payload}");
    let row = rows.first().ok_or("no row")?;
    assert_eq!(row.get("source_id").and_then(Value::as_str), Some("alpha"));
    assert_eq!(
        row.get("target_slug").and_then(Value::as_str),
        Some("no-such-slug")
    );
    Ok(())
}

#[test]
fn projects_counts_records_with_and_without_a_project() -> TestResult {
    let corpus = corpus()?;
    let payload = text(corpus.tools.call("projects", &json!({})))?;
    assert_eq!(
        payload
            .get("projects")
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(0)
    );
    assert_eq!(payload.get("noProject").and_then(Value::as_u64), Some(5));
    Ok(())
}

#[test]
fn similar_ranks_neighbours_by_stored_vector_and_excludes_the_start() -> TestResult {
    let corpus = corpus()?;
    let payload = text(corpus.tools.call("similar", &json!({ "id": "alpha" })))?;
    assert_eq!(ids(&payload), ["beta", "gamma"]);
    let one = text(
        corpus
            .tools
            .call("similar", &json!({ "id": "alpha", "limit": 1 })),
    )?;
    assert_eq!(ids(&one), ["beta"]);
    Ok(())
}

#[test]
fn similar_of_an_unembedded_record_fails_and_names_the_remedy() -> TestResult {
    let corpus = corpus()?;
    let reason = failure(corpus.tools.call("similar", &json!({ "id": "lonely" })))?;
    assert!(reason.contains("lonely"), "{reason}");
    assert!(reason.contains("kb embed"), "{reason}");
    assert!(reason.contains("backfill"), "{reason}");
    Ok(())
}

#[test]
fn similar_of_an_unknown_record_is_a_different_failure() -> TestResult {
    let corpus = corpus()?;
    let reason = failure(corpus.tools.call("similar", &json!({ "id": "nope" })))?;
    assert!(reason.contains("no node with id: nope"), "{reason}");
    assert!(!reason.contains("backfill"), "{reason}");
    Ok(())
}

#[test]
fn prompt_tools_list_show_and_render_the_builtin_template() -> TestResult {
    let corpus = corpus()?;
    let listed = text(corpus.tools.call("prompt_list", &json!({})))?;
    let has_builtin = listed
        .get("templates")
        .and_then(Value::as_array)
        .ok_or("no templates")?
        .iter()
        .any(|t| t.get("name").and_then(Value::as_str) == Some("cold-start-audit"));
    assert!(has_builtin, "{listed}");

    let shown = text(
        corpus
            .tools
            .call("prompt_show", &json!({ "name": "cold-start-audit" })),
    )?;
    assert!(
        shown
            .get("body")
            .and_then(Value::as_str)
            .is_some_and(|body| !body.is_empty()),
        "{shown}"
    );

    match corpus
        .tools
        .call("prompt_render", &json!({ "name": "cold-start-audit" }))
    {
        ToolOutcome::Text(rendered) => assert!(!rendered.is_empty()),
        ToolOutcome::Failed(reason) => return Err(reason.into()),
    }
    Ok(())
}

#[test]
fn prompt_tools_fail_for_an_unknown_template_and_a_missing_name() -> TestResult {
    let corpus = corpus()?;
    for tool in ["prompt_show", "prompt_render"] {
        let unknown = failure(
            corpus
                .tools
                .call(tool, &json!({ "name": "no-such-template" })),
        )?;
        assert!(unknown.contains("no-such-template"), "{tool}: {unknown}");
        let missing = failure(corpus.tools.call(tool, &json!({})))?;
        assert!(missing.contains("name is required"), "{tool}: {missing}");
    }
    Ok(())
}

#[test]
fn the_four_existing_tools_keep_their_output_shapes() -> TestResult {
    let corpus = corpus()?;
    let found = text(corpus.tools.call("search", &json!({ "query": "plain" })))?;
    assert!(found.get("count").is_some() && found.get("results").is_some());
    let context = text(corpus.tools.call("context", &json!({ "id": "alpha" })))?;
    assert!(context.get("links_to").is_some() && context.get("linked_from").is_some());
    match corpus.tools.call("get", &json!({ "id": "alpha" })) {
        ToolOutcome::Text(body) => assert!(body.contains("Alpha")),
        ToolOutcome::Failed(reason) => return Err(reason.into()),
    }
    let put = text(
        corpus
            .tools
            .call("put", &json!({ "document": "* Fresh\n\nnew\n" })),
    )?;
    assert!(put.get("id").is_some() && put.get("superseded").is_some());
    Ok(())
}
