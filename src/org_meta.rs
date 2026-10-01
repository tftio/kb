//! Document-level kb metadata as an org property-drawer envelope.
//!
//! kb storage metadata — the node id and its created/updated timestamps —
//! is projected into org text as a leading `:PROPERTIES:` drawer on
//! export and stripped back out on import. The drawer is an envelope: it
//! is synthesized from storage facts at render time and never persisted
//! into the document AST. Both the `kb` CLI and the HTTP server route
//! their `text/org` output and input through these two functions.

use crate::generator;
use serde_json::json;
use tftio_org::ast::{Block, Document};

/// Property-drawer keys carrying kb storage metadata. These are
/// dehydrated into the org export and hydrated back out on import; they
/// are never stored in the document body.
pub const RESERVED_META_KEYS: [&str; 3] = ["ID", "CREATED", "UPDATED"];

/// Render `document` as org text prefixed with a document-level property
/// drawer carrying the node's kb metadata.
///
/// The drawer is built from the supplied storage facts, not from the
/// document AST, so the generator's `Document -> String` round-trip
/// invariant (`tests/roundtrip.rs`) is unaffected.
#[must_use]
pub fn render_with_metadata(
    id: &str,
    created_at: &str,
    updated_at: &str,
    document: &Document,
) -> String {
    // The generator emits property values verbatim (`:KEY:value`), so the
    // `:KEY: value` separator space must live in the value itself. `hydrate`
    // trims it back off on import.
    let drawer = Document {
        blocks: vec![Block::PropertyDrawer {
            entries: vec![
                ("ID".to_string(), format!(" {id}")),
                ("CREATED".to_string(), format!(" {created_at}")),
                ("UPDATED".to_string(), format!(" {updated_at}")),
            ],
        }],
    };
    format!(
        "{}{}",
        generator::generate(&drawer),
        generator::generate(document)
    )
}

/// [`render_with_metadata`] for text that is already rendered.
///
/// The index caches a record's text rather than its AST, so a reader that has
/// the text has nothing to generate from. Re-parsing it only to render it
/// again would put the parser between a record and its own bytes, which is
/// the round trip `tests/roundtrip.rs` covers and not something a read should
/// depend on.
#[must_use]
pub fn render_metadata_around(id: &str, created_at: &str, updated_at: &str, text: &str) -> String {
    let drawer = Document {
        blocks: vec![Block::PropertyDrawer {
            entries: vec![
                ("ID".to_string(), format!(" {id}")),
                ("CREATED".to_string(), format!(" {created_at}")),
                ("UPDATED".to_string(), format!(" {updated_at}")),
            ],
        }],
    };
    format!("{}{text}", generator::generate(&drawer))
}

/// The provenance object a caller reads a record's header through.
///
/// `null` when the record asserts none of it, an object naming only the
/// fields it set otherwise (`PLAN-20260923-project-identity` T005, T006).
/// Shared by `kb get --json`, `kb search --json` and the MCP `get` tool, so a
/// record's provenance reads the same whether it was found by id or by
/// search.
#[must_use]
pub fn provenance_json(row: &crate::index::RecordRow, domains: &[String]) -> serde_json::Value {
    if row.project.is_none()
        && row.project_source.is_none()
        && row.remote.is_none()
        && row.context.is_none()
        && domains.is_empty()
        && row.harness.is_none()
        && row.model.is_none()
        && row.session.is_none()
        && row.cwd.is_none()
    {
        return serde_json::Value::Null;
    }
    json!({
        "project": row.project,
        "projectSource": row.project_source,
        "remote": row.remote,
        "context": row.context,
        "domains": domains,
        "harness": row.harness,
        "model": row.model,
        "session": row.session,
        "cwd": row.cwd,
    })
}

/// The provenance block appended to a record's text output: empty when the
/// record asserts none of it.
#[must_use]
pub fn provenance_text(row: &crate::index::RecordRow, domains: &[String]) -> String {
    let mut lines = Vec::new();
    if let Some(v) = &row.project {
        lines.push(format!("project: {v}"));
    }
    if let Some(v) = &row.project_source {
        lines.push(format!("project-source: {v}"));
    }
    if let Some(v) = &row.remote {
        lines.push(format!("remote: {v}"));
    }
    if let Some(v) = &row.context {
        lines.push(format!("context: {v}"));
    }
    if !domains.is_empty() {
        lines.push(format!("domains: {}", domains.join(", ")));
    }
    if let Some(v) = &row.harness {
        lines.push(format!("harness: {v}"));
    }
    if let Some(v) = &row.model {
        lines.push(format!("model: {v}"));
    }
    if let Some(v) = &row.session {
        lines.push(format!("session: {v}"));
    }
    if let Some(v) = &row.cwd {
        lines.push(format!("cwd: {v}"));
    }
    if lines.is_empty() {
        return String::new();
    }
    format!("\nprovenance:\n  {}\n", lines.join("\n  "))
}

/// Hydrate kb metadata out of an imported document.
///
/// When `document` opens with a property drawer carrying an `ID` entry,
/// the kb-reserved entries ([`RESERVED_META_KEYS`]) are removed so they
/// never reach the stored body. Non-reserved entries in the same drawer
/// are preserved; the drawer block is dropped entirely when only
/// reserved entries remain. Returns the `ID` value when present.
pub fn hydrate(document: &mut Document) -> Option<String> {
    let Some(Block::PropertyDrawer { entries }) = document.blocks.first() else {
        return None;
    };
    if !entries.iter().any(|(k, _)| k == "ID") {
        return None;
    }
    // Property values are stored verbatim, including the `:KEY: value`
    // separator space; trim it so the returned id is the bare value.
    let id = entries
        .iter()
        .find(|(k, _)| k == "ID")
        .map(|(_, v)| v.trim().to_string());
    let Some(Block::PropertyDrawer { entries }) = document.blocks.first_mut() else {
        unreachable!("checked above")
    };
    entries.retain(|(k, _)| !RESERVED_META_KEYS.contains(&k.as_str()));
    if entries.is_empty() {
        document.blocks.remove(0);
    }
    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::canonicalize;
    use crate::parser;
    use tftio_org::ast::{Inline, Tag, Title};

    fn sample_doc() -> Document {
        Document {
            blocks: vec![Block::Heading {
                level: 1,
                title: Title("My Note".to_string()),
                tags: vec![Tag("rust".to_string())],
                children: vec![Block::Paragraph {
                    inlines: vec![Inline::Plain("body text".to_string())],
                }],
            }],
        }
    }

    #[test]
    fn render_prepends_metadata_drawer() {
        let org = render_with_metadata(
            "node-1",
            "2026-05-19 09:00:00",
            "2026-05-19 10:30:00",
            &sample_doc(),
        );
        let expected_prefix = ":PROPERTIES:\n:ID: node-1\n\
             :CREATED: 2026-05-19 09:00:00\n:UPDATED: 2026-05-19 10:30:00\n:END:\n";
        assert!(
            org.starts_with(expected_prefix),
            "missing metadata drawer, got: {org}"
        );
        assert!(org.contains("* My Note :rust:"));
    }

    #[test]
    fn export_round_trips_through_hydrate() {
        let org = render_with_metadata("node-1", "t0", "t1", &sample_doc());
        let mut doc = parser::parse_document(&org).expect("export must reparse");
        let id = hydrate(&mut doc);
        assert_eq!(id.as_deref(), Some("node-1"));
        // The reserved drawer is gone; the body matches the original doc
        // in canonical form (the parser/generator round-trip).
        assert_eq!(doc, canonicalize(&sample_doc()));
    }

    #[test]
    fn hydrate_noop_without_metadata_drawer() {
        let mut doc = sample_doc();
        let before = doc.clone();
        assert_eq!(hydrate(&mut doc), None);
        assert_eq!(doc, before);
    }

    #[test]
    fn hydrate_preserves_non_reserved_drawer_entries() {
        let mut doc = Document {
            blocks: vec![Block::PropertyDrawer {
                entries: vec![
                    ("ID".to_string(), "node-9".to_string()),
                    ("CUSTOM".to_string(), "keep-me".to_string()),
                ],
            }],
        };
        assert_eq!(hydrate(&mut doc).as_deref(), Some("node-9"));
        assert_eq!(
            doc.blocks,
            vec![Block::PropertyDrawer {
                entries: vec![("CUSTOM".to_string(), "keep-me".to_string())],
            }]
        );
    }
}
