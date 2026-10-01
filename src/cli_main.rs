//! CLI entrypoint for the kb knowledge base tool.
//!
//! Follows the `todoer` / `silent-critic` pattern: `ToolSpec`, agent
//! surface, metadata routing, and domain command dispatch all live on the
//! library side; `main.rs` is a thin binary stub.

use std::io::{IsTerminal, Read};

use serde_json::json;
use tftio_lib::{
    AgentCapability, AgentHook, AgentSurfaceSpec, CommandSelector, FlagSelector, HookEvent,
    JsonOutput, LicenseType, ToolSpec, error::print_error, map_standard_command, render_response,
    run_cli_no_doctor_from, workspace_tool,
};

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
pub use tftio_lib::MetaCommand;

use crate::markdown;
use crate::org_meta;
use crate::parser;
use crate::prompt;
use crate::retrieval;
use crate::storage;
use crate::store::BlobStore as _;
use tftio_org::ast::Document;

// ── Agent surface ──────────────────────────────────────────────────────

const SEARCH_COMMAND: CommandSelector = CommandSelector::new(&["search"]);
const GET_COMMAND: CommandSelector = CommandSelector::new(&["get"]);
const CREATE_COMMAND: CommandSelector = CommandSelector::new(&["create"]);
const UPDATE_COMMAND: CommandSelector = CommandSelector::new(&["update"]);
const DELETE_COMMAND: CommandSelector = CommandSelector::new(&["delete"]);
const RECENT_COMMAND: CommandSelector = CommandSelector::new(&["recent"]);
const LIST_BY_TAG_COMMAND: CommandSelector = CommandSelector::new(&["list-by-tag"]);
const LINKS_COMMAND: CommandSelector = CommandSelector::new(&["links"]);
const ORPHANS_COMMAND: CommandSelector = CommandSelector::new(&["orphans"]);
const HUBS_COMMAND: CommandSelector = CommandSelector::new(&["hubs"]);
const BROKEN_COMMAND: CommandSelector = CommandSelector::new(&["broken"]);
const PROJECTS_COMMAND: CommandSelector = CommandSelector::new(&["projects"]);
const PROMPT_RENDER_COMMAND: CommandSelector = CommandSelector::new(&["prompt", "render"]);
const PROMPT_LIST_COMMAND: CommandSelector = CommandSelector::new(&["prompt", "list"]);
const PROMPT_SHOW_COMMAND: CommandSelector = CommandSelector::new(&["prompt", "show"]);

const SEARCH_MATCH_FLAG: FlagSelector = FlagSelector::new(&["search"], "match");
const SEARCH_NO_VECTOR_FLAG: FlagSelector = FlagSelector::new(&["search"], "no-vector");
const SEARCH_LIMIT_FLAG: FlagSelector = FlagSelector::new(&["search"], "limit");
const SEARCH_MIN_SIMILARITY_FLAG: FlagSelector = FlagSelector::new(&["search"], "min-similarity");
const SEARCH_NO_RERANK_FLAG: FlagSelector = FlagSelector::new(&["search"], "no-rerank");
const SEARCH_CORPUS_FLAG: FlagSelector = FlagSelector::new(&["search"], "corpus");
const SEARCH_PROJECT_FLAG: FlagSelector = FlagSelector::new(&["search"], "project");
const SEARCH_CONTEXT_FLAG: FlagSelector = FlagSelector::new(&["search"], "context");
const SEARCH_EXPLAIN_FLAG: FlagSelector = FlagSelector::new(&["search"], "explain");
const SEARCH_JSON_FLAG: FlagSelector = FlagSelector::new(&["search"], "json");
const GET_JSON_FLAG: FlagSelector = FlagSelector::new(&["get"], "json");
const CREATE_ID_FLAG: FlagSelector = FlagSelector::new(&["create"], "id");
const CREATE_TAG_FLAG: FlagSelector = FlagSelector::new(&["create"], "tag");
const CREATE_MARKDOWN_FLAG: FlagSelector = FlagSelector::new(&["create"], "markdown");
const CREATE_ALLOW_EMPTY_FLAG: FlagSelector = FlagSelector::new(&["create"], "allow-empty");
const CREATE_PROVENANCE_JSON_FLAG: FlagSelector = FlagSelector::new(&["create"], "provenance-json");
const CREATE_JSON_FLAG: FlagSelector = FlagSelector::new(&["create"], "json");
const UPDATE_TAG_FLAG: FlagSelector = FlagSelector::new(&["update"], "tag");
const UPDATE_MARKDOWN_FLAG: FlagSelector = FlagSelector::new(&["update"], "markdown");
const UPDATE_ALLOW_EMPTY_FLAG: FlagSelector = FlagSelector::new(&["update"], "allow-empty");
const UPDATE_PROVENANCE_JSON_FLAG: FlagSelector = FlagSelector::new(&["update"], "provenance-json");
const UPDATE_JSON_FLAG: FlagSelector = FlagSelector::new(&["update"], "json");
const RECENT_LIMIT_FLAG: FlagSelector = FlagSelector::new(&["recent"], "limit");
const RECENT_JSON_FLAG: FlagSelector = FlagSelector::new(&["recent"], "json");
const LIST_BY_TAG_JSON_FLAG: FlagSelector = FlagSelector::new(&["list-by-tag"], "json");
const LINKS_JSON_FLAG: FlagSelector = FlagSelector::new(&["links"], "json");
const ORPHANS_JSON_FLAG: FlagSelector = FlagSelector::new(&["orphans"], "json");
const HUBS_LIMIT_FLAG: FlagSelector = FlagSelector::new(&["hubs"], "limit");
const HUBS_JSON_FLAG: FlagSelector = FlagSelector::new(&["hubs"], "json");
const BROKEN_JSON_FLAG: FlagSelector = FlagSelector::new(&["broken"], "json");
const PROJECTS_JSON_FLAG: FlagSelector = FlagSelector::new(&["projects"], "json");
const TAGS_COMMAND: CommandSelector = CommandSelector::new(&["tags"]);
const TAGS_MERGE_COMMAND: CommandSelector = CommandSelector::new(&["tags", "merge"]);
const TAGS_ADD_COMMAND: CommandSelector = CommandSelector::new(&["tags", "add"]);
const TAGS_RM_COMMAND: CommandSelector = CommandSelector::new(&["tags", "rm"]);
const TAGS_JSON_FLAG: FlagSelector = FlagSelector::new(&["tags"], "json");
const TAGS_MERGE_JSON_FLAG: FlagSelector = FlagSelector::new(&["tags", "merge"], "json");
const TAGS_ADD_JSON_FLAG: FlagSelector = FlagSelector::new(&["tags", "add"], "json");
const TAGS_RM_JSON_FLAG: FlagSelector = FlagSelector::new(&["tags", "rm"], "json");
const PROMPT_LIST_JSON_FLAG: FlagSelector = FlagSelector::new(&["prompt", "list"], "json");
const PROMPT_SHOW_JSON_FLAG: FlagSelector = FlagSelector::new(&["prompt", "show"], "json");
const SIMILAR_COMMAND: CommandSelector = CommandSelector::new(&["similar"]);
const SIMILAR_LIMIT_FLAG: FlagSelector = FlagSelector::new(&["similar"], "limit");
const SIMILAR_JSON_FLAG: FlagSelector = FlagSelector::new(&["similar"], "json");
const BACKFILL_COMMAND: CommandSelector = CommandSelector::new(&["backfill"]);
const BACKFILL_MODEL_FLAG: FlagSelector = FlagSelector::new(&["backfill"], "model");
const BACKFILL_LIMIT_FLAG: FlagSelector = FlagSelector::new(&["backfill"], "limit");
const BACKFILL_STALE_ONLY_FLAG: FlagSelector = FlagSelector::new(&["backfill"], "stale-only");
const BACKFILL_DRY_RUN_FLAG: FlagSelector = FlagSelector::new(&["backfill"], "dry-run");
const BACKFILL_JSON_FLAG: FlagSelector = FlagSelector::new(&["backfill"], "json");
const GLOBAL_DB_FLAG: FlagSelector = FlagSelector::new(&[], "db");

const SEARCH_CAPABILITY: AgentCapability = AgentCapability::new(
    "search",
    "Hybrid keyword and vector search across knowledge-base nodes. The query is \
     prefix-matched and conjunctive by default: every whitespace-separated token must \
     appear, each as a prefix, and FTS5 operators are matched literally, so `a OR b` \
     finds nothing. Pass --match to send the query to FTS5 verbatim. At most 20 results \
     unless --limit says otherwise.",
    &[SEARCH_COMMAND],
    &[
        SEARCH_MATCH_FLAG,
        SEARCH_NO_VECTOR_FLAG,
        SEARCH_LIMIT_FLAG,
        SEARCH_MIN_SIMILARITY_FLAG,
        SEARCH_NO_RERANK_FLAG,
        SEARCH_CORPUS_FLAG,
        SEARCH_PROJECT_FLAG,
        SEARCH_CONTEXT_FLAG,
        SEARCH_EXPLAIN_FLAG,
        SEARCH_JSON_FLAG,
        GLOBAL_DB_FLAG,
    ],
)
.with_when_to_use(
    "the user wants to find knowledge-base nodes by content. Ranking is hybrid, so a \
     natural-language question works as well as keywords and a paraphrase can find a node \
     sharing none of its words. Prefer several narrow searches over one boolean \
     expression; reach for --match when the query needs a disjunction, an exclusion, an \
     exact phrase, or a column filter. --corpus defaults to kb, so correspondence needs \
     --corpus mail explicitly. --project <slug> and --context <name> narrow results to \
     records asserting that provenance; absent, neither filters anything",
)
.with_when_not_to_use(
    "the user already knows the exact node id (use get instead), \
     or wants nodes by tag rather than content (use list-by-tag instead), \
     or wants nodes like a node they already have (use similar instead). \
     An empty result is not proof of absence - fall back to list-by-tag and recent",
)
.with_output(
    "a JSON array of {id, title, similarity} summaries. Ranking is environment-dependent: \
     order is by fused rank, not by similarity, so the scores are not monotonic down the \
     list. When no embedding endpoint answers, or under --no-vector, ranking is FTS5 \
     relevance alone, every similarity is null and recall is lower; a note on stderr says \
     which happened. Where \
     KB_RERANK_BASE_URL configures a cross-encoder, a third stage reorders the top 20 \
     candidates, which is strictly a reordering - a node retrieval never surfaced is not \
     reachable by reranking. --explain replaces the results with an account of how they \
     were reached, which distinguishes a node no signal surfaced from one ranked below \
     the cut",
)
.with_constraints(
    "--match enables the full FTS5 syntax: OR, NOT, NEAR, quoted phrases, term* \
     prefixes, and the title:/body: column filters. Under --json each hit carries a \
     similarity, null when the hit came from keyword matching alone - null means no \
     vector score was computed, NOT that the node scored zero, and similarity \
     calibrates per corpus, so results in the 0.4s are not evidence of absence.",
);

const GET_CAPABILITY: AgentCapability = AgentCapability::new(
    "get",
    "Retrieve a single knowledge-base node by id, returning the full document, \
     derived title, tag list, and storage timestamps",
    &[GET_COMMAND],
    &[GET_JSON_FLAG, GLOBAL_DB_FLAG],
)
.with_when_to_use("the user references a specific node id and wants to see its full contents")
.with_when_not_to_use(
    "the user wants to browse or discover nodes (use search, recent, or list-by-tag instead)",
)
.with_output("a JSON node view: id, title, tags, document AST, createdAt, updatedAt");

const CREATE_CAPABILITY: AgentCapability = AgentCapability::new(
    "create",
    "Create a new knowledge-base node from stdin: org-mode text, a JSON kb AST Document, \
     or GitHub-flavored markdown with --markdown, converted via pandoc. --id supplies an \
     id, otherwise a UUIDv4 is minted. Tags from --tag are merged with any parsed from \
     the body and normalized to lowercase kebab-case, so run tags first and reuse an \
     existing spelling rather than coining a variant that strands nodes under a second \
     name.",
    &[CREATE_COMMAND],
    &[
        CREATE_ID_FLAG,
        CREATE_TAG_FLAG,
        CREATE_MARKDOWN_FLAG,
        CREATE_ALLOW_EMPTY_FLAG,
        CREATE_PROVENANCE_JSON_FLAG,
        CREATE_JSON_FLAG,
        GLOBAL_DB_FLAG,
    ],
)
.with_when_to_use(
    "the user wants to persist a new piece of knowledge — a note, a transcript, a design doc, \
     or any org-mode text — into the knowledge base",
)
.with_when_not_to_use(
    "the node already exists and should be modified (use update instead), \
     or the id supplied via --id is already taken (the command will fail with a conflict error)",
)
.with_output(
    "a JSON node view of the created node, including the server-assigned or caller-supplied id",
)
.with_constraints(
    "Normalization cannot recover a word break nobody wrote: 'Silent Critic' and \
     'silentCritic' both store as 'silent-critic', but 'CICD' stores as 'cicd', a \
     different tag from the 'ci-cd' of 'CI/CD'. A [[slug]] matching no node's #+name: is \
     stored as a broken link and warned about on stderr; the node is still created and \
     the exit status is still 0. stdin must be a pipe or a redirected file, and the body \
     must be non-blank unless --allow-empty is passed. --provenance-json <file> reads a \
     JSON object naming where the work happened — project, project_source, remote, \
     context, domains, harness, model, session, cwd, every field optional and a string \
     (domains a string array) — and stores it in the record header rather than the body; \
     it must be a file path, never `-`, because stdin already carries the document body.",
);

const UPDATE_CAPABILITY: AgentCapability = AgentCapability::new(
    "update",
    "Replace an existing node's document in full, from stdin: org-mode text, a JSON kb \
     AST Document, or GitHub-flavored markdown with --markdown. Tags from --tag are \
     merged with any parsed from the new body and normalized exactly as create does. To \
     change only the tags, do not use this command: tags add and tags rm edit a node's \
     tags without touching its body and without reading stdin.",
    &[UPDATE_COMMAND],
    &[
        UPDATE_TAG_FLAG,
        UPDATE_MARKDOWN_FLAG,
        UPDATE_ALLOW_EMPTY_FLAG,
        UPDATE_PROVENANCE_JSON_FLAG,
        UPDATE_JSON_FLAG,
        GLOBAL_DB_FLAG,
    ],
)
.with_when_to_use("the user wants to replace the contents of an existing knowledge-base node")
.with_when_not_to_use(
    "the node does not yet exist (use create instead), \
     the id is unknown (the command will fail with 'no node with id'), \
     or only the tags need changing (use tags add or tags rm; this command replaces the \
     stored body with whatever stdin holds)",
)
.with_output("a JSON node view of the updated node with the refreshed updatedAt timestamp")
.with_constraints(
    "Tag normalization follows create's rule -- 'Silent Critic' and 'silentCritic' both \
     store as 'silent-critic'; 'CICD' stores as 'cicd', which is not the 'ci-cd' of \
     'CI/CD' -- and tags are written into the stored document by the same placement \
     rule. A [[slug]] in the new body that matches no node's #+name: is stored as a \
     broken link and reported as a warning on stderr, without changing the exit status. \
     stdin must be a pipe or a redirected file, and the body must be non-blank unless \
     --allow-empty is passed. created is always preserved. Without --provenance-json, \
     the node's existing kind, source and provenance are kept exactly as tags \
     add/rm/merge already keep them, so an ordinary update of a captured transcript \
     does not turn it back into a plain note or drop its project. With \
     --provenance-json <file>, the same JSON shape create's own flag takes, only the \
     provenance is replaced with exactly the file's contents; kind and source still \
     come from the existing node.",
);

const DELETE_CAPABILITY: AgentCapability = AgentCapability::new(
    "delete",
    "Permanently delete a knowledge-base node by id. This cascades to node tags and links. \
     The prior document is preserved in the audit log.",
    &[DELETE_COMMAND],
    &[GLOBAL_DB_FLAG],
)
.with_when_to_use("the user explicitly asks to delete a specific node from the knowledge base")
.with_when_not_to_use(
    "the user is unsure about deletion, or the node id is unknown \
     (the command will report 'no node with id' rather than silently succeeding)",
)
.with_output("plain-text confirmation 'deleted <id>' on success");

const RECENT_CAPABILITY: AgentCapability = AgentCapability::new(
    "recent",
    "List the most recently updated knowledge-base nodes, newest first. \
     Defaults to 50 nodes; use --limit to adjust.",
    &[RECENT_COMMAND],
    &[RECENT_LIMIT_FLAG, RECENT_JSON_FLAG, GLOBAL_DB_FLAG],
)
.with_when_to_use("the user wants to see what was recently added or changed in the knowledge base")
.with_when_not_to_use(
    "the user wants nodes matching a specific topic (use search) or tag (use list-by-tag)",
)
.with_output("a JSON array of {id, title} summaries ordered by updatedAt descending");

const LIST_BY_TAG_CAPABILITY: AgentCapability = AgentCapability::new(
    "list-by-tag",
    "List knowledge-base node summaries for all nodes carrying a given tag. \
     The tag is supplied without surrounding colons (e.g. 'rust' not ':rust:'). \
     The query tag is normalized the same way stored tags are, so casing and \
     separators do not have to match: 'Silent Critic', 'silentCritic', and \
     'silent-critic' all find the same nodes. Normalization cannot reconcile a \
     spelling that carries no word break at all - 'CICD' normalizes to 'cicd', \
     which is a different tag from the 'ci-cd' of 'CI/CD' - so an empty result \
     may mean the nodes are filed under a variant spelling rather than that no \
     such nodes exist.",
    &[LIST_BY_TAG_COMMAND],
    &[LIST_BY_TAG_JSON_FLAG, GLOBAL_DB_FLAG],
)
.with_when_to_use("the user wants to enumerate all nodes under a specific tag")
.with_when_not_to_use(
    "the user wants full-text search across node bodies (use search instead), \
     or wants to browse by recency (use recent instead). \
     An empty result is not proof of absence: run tags to see which spellings \
     actually exist before concluding the tag is unused",
)
.with_output("a JSON array of {id, title} summaries for nodes carrying the tag");

const LINKS_CAPABILITY: AgentCapability = AgentCapability::new(
    "links",
    "Show the forward (outgoing) and back (incoming) links for a node under the unified \
     link graph. Both `[[id:UUID]]` id-links and `[[name]]` bracket name-links coexist \
     in a single physical table discriminated by a `link_type` column ('id' | 'name'). \
     Each row reports `{source_id, link_type, target_id, target_slug}`. For id-link rows \
     `target_id` is always non-null and `target_slug` is null; for name-link rows \
     `target_slug` is always non-null and `target_id` is null when the bracket reference \
     is broken (no node carries that `#+name:` slug yet).",
    &[LINKS_COMMAND],
    &[LINKS_JSON_FLAG, GLOBAL_DB_FLAG],
)
.with_when_to_use(
    "the user wants to see what a node references (by either id or name) and what references it back",
)
.with_when_not_to_use(
    "the user wants global metrics like hubs / orphans / broken (use those verbs instead)",
)
.with_output(
    "a JSON object {outgoing: [...], incoming: [...]} of unified link rows including \
     link_type for the node",
);

const ORPHANS_CAPABILITY: AgentCapability = AgentCapability::new(
    "orphans",
    "List orphan nodes — nodes that nothing else points at under either link_type. \
     A node is an orphan when no `links` row of any link_type ('id' or 'name') has a \
     resolved `target_id` equal to its id.",
    &[ORPHANS_COMMAND],
    &[ORPHANS_JSON_FLAG, GLOBAL_DB_FLAG],
)
.with_when_to_use("the user wants to find isolated nodes in the knowledge base graph")
.with_when_not_to_use(
    "the user wants nodes by tag (use list-by-tag), by recency (use recent), \
     or by full-text content (use search)",
)
.with_output(
    "a JSON array of {id, title} summaries for nodes with zero incoming links of any link_type",
);

const HUBS_CAPABILITY: AgentCapability = AgentCapability::new(
    "hubs",
    "List the most-linked nodes ranked by total in-degree across both link_types \
     (id-links plus resolved name-links) in the unified graph. Defaults to top 20.",
    &[HUBS_COMMAND],
    &[HUBS_LIMIT_FLAG, HUBS_JSON_FLAG, GLOBAL_DB_FLAG],
)
.with_when_to_use("the user wants to see which nodes act as graph hubs in the knowledge base")
.with_when_not_to_use(
    "the user wants the full link table rather than ranked summaries (use links instead)",
)
.with_output("a JSON array of {id, title, in_degree} entries ordered by in-degree descending");

const BROKEN_CAPABILITY: AgentCapability = AgentCapability::new(
    "broken",
    "List broken bracket references in the unified link graph: rows with link_type='name' \
     whose `target_id` is NULL because no node carries the referenced `#+name:` slug. \
     Id-links cannot be broken by construction — the schema-level CHECK enforces a \
     non-null target_id for link_type='id'.",
    &[BROKEN_COMMAND],
    &[BROKEN_JSON_FLAG, GLOBAL_DB_FLAG],
)
.with_when_to_use("the user wants to find dangling bracket references in the knowledge base")
.with_when_not_to_use("the user wants to inspect a single node's links (use links instead)")
.with_output(
    "a JSON array of {source_id, link_type, target_id, target_slug} rows for unresolved \
     bracket references",
);

const PROJECTS_CAPABILITY: AgentCapability = AgentCapability::new(
    "projects",
    "List every project slug asserted by at least one record's provenance, with the \
     record count and the newest record's date under each, plus a count of records \
     carrying no project.",
    &[PROJECTS_COMMAND],
    &[PROJECTS_JSON_FLAG, GLOBAL_DB_FLAG],
)
.with_when_to_use(
    "the user wants to know what projects the corpus holds records for, or how many \
     records a given slug has, before filtering search by --project",
)
.with_when_not_to_use(
    "the user wants the records themselves rather than counts (use search --project \
     instead)",
)
.with_output(
    "a JSON object of {projects: [{project, count, newest}], noProject}, projects ordered \
     newest first",
);

const PROMPT_RENDER_CAPABILITY: AgentCapability = AgentCapability::new(
    "prompt-render",
    "Render a MiniJinja template against the local kb corpus and write the rendered text \
     to stdout. Templates have access to a documented query surface — `recent`, `orphans`, \
     `hubs`, `tag_frequency` as eagerly bound corpus snapshots, and `by_tag`, `search`, \
     `all_nodes`, `get`, `links`, `link_distance` as callables. Built-in templates ship \
     embedded with the crate and are overridable by files at \
     `$XDG_CONFIG_HOME/kb/prompts/<name>.j2`. kb assembles; the user executes — kb does not \
     pipe the rendered text into any LLM, that is the caller's job.",
    &[PROMPT_RENDER_COMMAND],
    &[GLOBAL_DB_FLAG],
)
.with_when_to_use(
    "the user wants to assemble an LLM prompt from kb contents using a named template",
)
.with_when_not_to_use(
    "the user wants to inspect available templates (use prompt list) or view a template's \
     source (use prompt show)",
)
.with_output("the rendered template text on stdout — no envelope; pipe it directly to an LLM");

const PROMPT_LIST_CAPABILITY: AgentCapability = AgentCapability::new(
    "prompt-list",
    "List every template available to `kb prompt`, merging built-in templates embedded with \
     the crate with user overrides found under `$XDG_CONFIG_HOME/kb/prompts/<name>.j2`. \
     User overrides win on name collision.",
    &[PROMPT_LIST_COMMAND],
    &[PROMPT_LIST_JSON_FLAG, GLOBAL_DB_FLAG],
)
.with_when_to_use("the user wants to enumerate the prompt templates kb can render")
.with_when_not_to_use("the user already knows the template name and wants to render or inspect it")
.with_output("a JSON array of {name, source, path} entries, sorted by name");

const PROMPT_SHOW_CAPABILITY: AgentCapability = AgentCapability::new(
    "prompt-show",
    "Print the raw source of the named template — whichever wins resolution between the \
     user override and the built-in. Useful for inspecting what a template will do before \
     rendering it.",
    &[PROMPT_SHOW_COMMAND],
    &[PROMPT_SHOW_JSON_FLAG, GLOBAL_DB_FLAG],
)
.with_when_to_use("the user wants to read a template's body without rendering it")
.with_when_not_to_use("the user wants the rendered output (use prompt render)")
.with_output("the template body as plain text; with --json, a {name, source, path, body} envelope");

const TAGS_CAPABILITY: AgentCapability = AgentCapability::new(
    "tags",
    "List the tag vocabulary actually in use, with the node count for each, ordered by \
     count descending then name ascending. Tags are stored normalized to lowercase \
     kebab-case, and this reports that stored form - the only form list-by-tag will \
     match. Its purpose is to be read before writing: tags are a primary retrieval axis, \
     and inventing a spelling variant of one that already exists splits the nodes across \
     two tags that no single query returns. Normalization does not prevent that, because \
     it cannot recover a word break nobody wrote ('CICD' and 'CI/CD' normalize to the \
     distinct tags 'cicd' and 'ci-cd').",
    &[TAGS_COMMAND],
    &[TAGS_JSON_FLAG, GLOBAL_DB_FLAG],
)
.with_when_to_use(
    "before tagging a new node, to reuse an existing spelling instead of inventing one; \
     when a list-by-tag query came back empty and the nodes may be filed under a variant; \
     or when the user asks what tags exist",
)
.with_when_not_to_use(
    "the user wants the nodes carrying a tag rather than the vocabulary itself \
     (use list-by-tag instead)",
)
.with_output("a JSON array of {tag, count} objects, most-used first");

const TAGS_MERGE_CAPABILITY: AgentCapability = AgentCapability::new(
    "tags-merge",
    "Move every node carrying one tag onto another, reconciling two spellings of what is \
     really one tag. This rewrites the stored document of each affected node, not just \
     the tag index, because the index is re-derived from the document on every write - a \
     merge that touched only the index would be undone by the next update. Each rewritten \
     node is audited and its updated_at is bumped, exactly as an update would be. Both \
     arguments are normalized before matching, so the source may be spelled any way that \
     normalizes to the stored tag. Nodes not carrying the source tag are untouched, and \
     merging a tag no node carries is reported rather than treated as an error.",
    &[TAGS_MERGE_COMMAND],
    &[TAGS_MERGE_JSON_FLAG, GLOBAL_DB_FLAG],
)
.with_when_to_use(
    "the user has identified two tags that are spellings of the same thing - typically \
     after reading tags - and wants the nodes consolidated under one of them",
)
.with_when_not_to_use(
    "the two tags are genuinely distinct, or the user has not confirmed which spelling \
     should win; this is a bulk mutation across every node carrying the source tag and \
     is reversible only through the audit log",
)
.with_output("a JSON object of {from, to, rewritten, count}, listing the ids rewritten");

const TAGS_ADD_CAPABILITY: AgentCapability = AgentCapability::new(
    "tags-add",
    "Add one or more tags to a single node without re-supplying its body. This is the \
     metadata-only mutation: unlike update, it does not read stdin and cannot replace the \
     stored document. Each tag is normalized to lowercase kebab-case before storage \
     ('Silent Critic' and 'silentCritic' both store as 'silent-critic'; 'CICD' stores as \
     'cicd', which is not the 'ci-cd' of 'CI/CD'), so run tags first and reuse an existing \
     spelling. The tags are written into the stored document - extending an existing \
     #+filetags: line, else the first heading's tags, else a new #+filetags: line - because \
     the tag index is re-derived from the document on every write. A tag the node already \
     carries is reported and written nowhere, leaving updated_at untouched.",
    &[TAGS_ADD_COMMAND],
    &[TAGS_ADD_JSON_FLAG, GLOBAL_DB_FLAG],
)
.with_when_to_use(
    "the user wants a node filed under an additional tag, or a capture was tagged \
     incompletely and the body should not change",
)
.with_when_not_to_use(
    "the body is changing too (use update, which takes --tag in the same pass), or two \
     existing spellings need reconciling across the corpus (use tags merge)",
)
.with_output(
    "a JSON object of {id, added, tags}, where added lists only the tags that were not \
     already present and tags is the node's resulting tag set",
);

const TAGS_RM_CAPABILITY: AgentCapability = AgentCapability::new(
    "tags-rm",
    "Remove one or more tags from a single node without re-supplying its body. The only \
     command that deletes a tag outright - tags merge moves a tag onto another but never \
     removes one. Arguments are normalized to lowercase kebab-case before matching, so a \
     tag may be spelled any way that normalizes to the stored form. Every occurrence the \
     tag extractor can see is removed from the stored document: heading tags at any depth \
     and tokens in #+filetags: values, with an emptied #+filetags: line dropped entirely. \
     The rest of the body is untouched, the prior document is preserved in the audit log, \
     and removing a tag the node does not carry is reported rather than treated as an error.",
    &[TAGS_RM_COMMAND],
    &[TAGS_RM_JSON_FLAG, GLOBAL_DB_FLAG],
)
.with_when_to_use("a node carries a tag that does not belong to it and the body should not change")
.with_when_not_to_use(
    "the tag is wrong everywhere rather than on this node (use tags merge to consolidate \
     spellings), or the node itself should go (use delete)",
)
.with_output(
    "a JSON object of {id, removed, tags}, where removed lists only the tags that were \
     actually present and tags is the node's resulting tag set",
);

const SIMILAR_CAPABILITY: AgentCapability = AgentCapability::new(
    "similar",
    "Nodes semantically nearest an existing node, by stored embeddings rather than by \
     shared words. The starting node is identified by id and is never returned as its own \
     neighbour. Neighbours are ranked by cosine similarity against the node's whole-node \
     vector, so this answers 'what else is about this' rather than 'what else says these \
     words'. Defaults to 10 neighbours; use --limit to adjust.",
    &[SIMILAR_COMMAND],
    &[SIMILAR_LIMIT_FLAG, SIMILAR_JSON_FLAG, GLOBAL_DB_FLAG],
)
.with_when_to_use(
    "the user has a node in hand and wants related reading - following a thread outward \
     from something already found, spotting duplicates, or gathering the neighbourhood of \
     a topic before writing about it",
)
.with_when_not_to_use(
    "the user has a topic or question rather than a node id (use search instead), or wants \
     an exact-word match (use search --match). This verb reads only stored vectors and \
     contacts no endpoint, so it fails rather than degrading when the starting node has \
     never been embedded - the error names backfill as the remedy",
)
.with_output(
    "a JSON array of {id, title} summaries in descending similarity. Exits nonzero when \
     the id is unknown or carries no embedding, which is a real failure rather than an \
     empty result",
);

const BACKFILL_CAPABILITY: AgentCapability = AgentCapability::new(
    "backfill",
    "Embed knowledge-base nodes that have no vector under the configured model, or whose \
     vector is older than the node it describes. Selection is a single query over that \
     condition rather than checkpoint state, so an interrupted run is resumed simply by \
     running it again - it converges on full coverage without duplicating or losing rows. \
     --stale-only narrows to refreshing out-of-date vectors and leaves never-embedded \
     nodes alone; --limit bounds one run; --dry-run reports the count while contacting no \
     endpoint and writing nothing.",
    &[BACKFILL_COMMAND],
    &[
        BACKFILL_MODEL_FLAG,
        BACKFILL_LIMIT_FLAG,
        BACKFILL_STALE_ONLY_FLAG,
        BACKFILL_DRY_RUN_FLAG,
        BACKFILL_JSON_FLAG,
        GLOBAL_DB_FLAG,
    ],
)
.with_when_to_use(
    "vector coverage has to be restored or extended: after nodes were written while the \
     endpoint was unreachable, after switching models, or to check with --dry-run how much \
     of the corpus is unembedded",
)
.with_when_not_to_use(
    "the aim is to search or to find neighbours (use search or similar) - this verb writes \
     rather than reads. It requires a reachable embedding endpoint except under --dry-run, \
     and --model must agree with the configured model rather than override it, since a \
     mismatch would file one model's vectors under another's name",
)
.with_output(
    "a report of {model, selected, embedded, failed, dryRun}, where failed lists node ids \
     by id rather than counting them. Exits nonzero if any selected node failed to embed, \
     so a partial run is distinguishable from a clean one",
);

// The installed registration this mirrors: `~/.config/claude/hooks/session-end-kb.sh`,
// timeout 30, statusMessage "Distilling session into kb"
// that an agent installer writes into Claude Code's settings. Embedding the script
// with `include_str!` means the file kb already lints and tests under
// `scripts/session-end-kb.sh` is the exact bytes `meta agent emit-hooks` writes out.
const SESSION_END_HOOK: AgentHook = AgentHook::new(
    "session-end-kb",
    HookEvent::SessionEnd,
    include_str!("../scripts/session-end-kb.sh"),
    30,
)
.with_status_message("Distilling session into kb");

const AGENT_SURFACE: AgentSurfaceSpec = AgentSurfaceSpec::new(&[
    SEARCH_CAPABILITY,
    SIMILAR_CAPABILITY,
    BACKFILL_CAPABILITY,
    GET_CAPABILITY,
    CREATE_CAPABILITY,
    UPDATE_CAPABILITY,
    DELETE_CAPABILITY,
    RECENT_CAPABILITY,
    LIST_BY_TAG_CAPABILITY,
    TAGS_CAPABILITY,
    TAGS_MERGE_CAPABILITY,
    TAGS_ADD_CAPABILITY,
    TAGS_RM_CAPABILITY,
    LINKS_CAPABILITY,
    ORPHANS_CAPABILITY,
    HUBS_CAPABILITY,
    BROKEN_CAPABILITY,
    PROJECTS_CAPABILITY,
    PROMPT_RENDER_CAPABILITY,
    PROMPT_LIST_CAPABILITY,
    PROMPT_SHOW_CAPABILITY,
])
.with_hooks(&[SESSION_END_HOOK]);

const TOOL_SPEC: ToolSpec = workspace_tool(
    "kb",
    "kb",
    env!("CARGO_PKG_VERSION"),
    LicenseType::MIT,
    true,
    false,
)
.with_agent_surface(&AGENT_SURFACE);

// ── Entrypoint ─────────────────────────────────────────────────────────

/// Return the process exit code for the kb CLI.
#[must_use]
pub fn main_exit_code() -> i32 {
    let env = process_env();
    run_cli_no_doctor_from::<Cli, _, _, _>(
        &TOOL_SPEC,
        &env,
        std::env::args_os(),
        metadata_command,
        |cli| Ok(run(cli)),
    )
}

/// Read process-edge environment values once at the binary edge.
#[allow(
    clippy::disallowed_methods,
    reason = "agent token / HOME read once at the process edge (REPO_INVARIANTS.md #5)"
)]
fn process_env() -> tftio_lib::ProcessEnv {
    tftio_lib::ProcessEnv {
        agent: tftio_lib::AgentModeContext::from_tokens(
            std::env::var(tftio_lib::AGENT_TOKEN_ENV).ok(),
            std::env::var(tftio_lib::AGENT_TOKEN_EXPECTED_ENV).ok(),
        ),
        home: std::env::var_os("HOME").map(std::path::PathBuf::from),
    }
}

/// Map a parsed CLI to a shared metadata command, if one was requested.
#[must_use]
pub fn metadata_command(cli: &Cli) -> Option<tftio_lib::StandardCommand> {
    match &cli.command {
        Command::Meta { command } => Some(map_standard_command(command, JsonOutput::Text)),
        _ => None,
    }
}

// ── Domain dispatch ────────────────────────────────────────────────────

/// Dispatch a `kb mail` subcommand.
///
/// Pulled out of [`run`] so that function stays under clippy's line count:
/// this arm is the one nested match among otherwise-flat dispatch, and it is
/// self-contained, so extracting it costs nothing but a call.
fn run_mail(command: MailCommand) -> i32 {
    match command {
        MailCommand::Classify { maildir, folder } => run_mail_classify(&maildir, &folder),
        MailCommand::Scope { maildir, questions } => run_mail_scope(&maildir, &questions),
        MailCommand::Index {
            maildir,
            questions,
            index,
        } => run_mail_index(&maildir, &questions, index.as_deref()),
        MailCommand::Embed {
            index,
            limit,
            chunk_bytes,
        } => run_embed(
            "mail embed",
            CORPUS_MAIL,
            index.as_deref(),
            limit,
            chunk_bytes,
        ),
    }
}

/// Dispatch a parsed non-metadata CLI to its handler and return an exit code.
#[must_use]
#[allow(
    clippy::too_many_lines,
    reason = "a flat dispatch table over every verb, one match arm per Command variant; \
              splitting it would trade one obvious list for several that have to be read \
              together"
)]
pub fn run(cli: Cli) -> i32 {
    match cli.command {
        Command::Meta { .. } => unreachable!("meta commands are routed before run"),
        Command::Search(args) => run_search(&args),
        Command::Generate(args) => run_generate(&args),
        Command::Mail { command } => run_mail(command),
        Command::Embed {
            corpus,
            index,
            limit,
            chunk_bytes,
        } => run_embed("embed", &corpus, index.as_deref(), limit, chunk_bytes),
        Command::Export { out } => run_export(&cli.db, &out),
        Command::Import { input, store } => run_import(&input, &store),
        Command::Reindex {
            store,
            index,
            corpus,
            record,
            maildir,
        } => run_reindex(
            &store,
            &index,
            corpus.as_deref(),
            record.as_deref(),
            maildir.as_deref(),
        ),
        Command::Fsck {
            store,
            index,
            repair,
            no_legacy,
            deep,
        } => run_fsck(
            &store,
            &index,
            repair,
            (!no_legacy).then_some(&cli.db),
            deep,
        ),
        Command::Project {
            store,
            space,
            prune,
        } => run_project(&store, &space, prune),
        Command::Get { id, json } => run_get(&id, JsonOutput::from_flag(json)),
        Command::Create {
            id,
            tags,
            markdown,
            allow_empty,
            provenance_json,
            json,
        } => run_create(
            id.as_deref(),
            &tags,
            markdown,
            allow_empty,
            provenance_json.as_deref(),
            JsonOutput::from_flag(json),
        ),
        Command::Update {
            id,
            tags,
            markdown,
            allow_empty,
            provenance_json,
            json,
        } => run_update(
            &id,
            &tags,
            markdown,
            allow_empty,
            provenance_json.as_deref(),
            JsonOutput::from_flag(json),
        ),
        Command::Delete { id } => run_delete(&id),
        Command::Similar { id, limit, json } => {
            run_similar(&id, limit, JsonOutput::from_flag(json))
        }
        Command::Recent { limit, json } => run_recent(limit, JsonOutput::from_flag(json)),
        Command::ListByTag { tag, json } => run_list_by_tag(&tag, JsonOutput::from_flag(json)),
        Command::Links { id, json } => run_links(&id, JsonOutput::from_flag(json)),
        Command::Orphans { json } => run_orphans(JsonOutput::from_flag(json)),
        Command::Hubs { limit, json } => run_hubs(limit, JsonOutput::from_flag(json)),
        Command::Broken { json } => run_broken(JsonOutput::from_flag(json)),
        Command::Projects { json } => run_projects(JsonOutput::from_flag(json)),
        Command::Queue {
            queue,
            store,
            index,
            command,
        } => run_queue(&queue, &store, &index, command),
        Command::Prompt { command } => run_prompt(command),
        Command::Tags { json, command } => run_tags(json, command),
    }
}

fn run_tags(json: bool, command: Option<TagsCommand>) -> i32 {
    match command {
        None => run_tags_list(JsonOutput::from_flag(json)),
        Some(TagsCommand::Merge {
            from,
            to,
            json: sub_json,
        }) => run_tags_merge(&from, &to, JsonOutput::from_flag(json || sub_json)),
        Some(TagsCommand::Add {
            id,
            tags,
            json: sub_json,
        }) => run_tags_edit(
            TagEditKind::Add,
            &id,
            &tags,
            JsonOutput::from_flag(json || sub_json),
        ),
        Some(TagsCommand::Rm {
            id,
            tags,
            json: sub_json,
        }) => run_tags_edit(
            TagEditKind::Remove,
            &id,
            &tags,
            JsonOutput::from_flag(json || sub_json),
        ),
    }
}

/// Which direction a single-node tag edit runs in.
///
/// The two commands differ only in the storage call and the word used for
/// the changed set, so they share one handler rather than duplicating the
/// open-report-render sequence.
#[derive(Debug, Clone, Copy)]
enum TagEditKind {
    Add,
    Remove,
}

impl TagEditKind {
    /// The command name used in output envelopes and error messages.
    const fn command(self) -> &'static str {
        match self {
            Self::Add => "tags-add",
            Self::Remove => "tags-rm",
        }
    }

    /// The JSON key naming what changed, and the verb used in text output.
    const fn changed_key(self) -> &'static str {
        match self {
            Self::Add => "added",
            Self::Remove => "removed",
        }
    }
}

fn run_tags_edit(kind: TagEditKind, id: &str, tags: &[String], json: JsonOutput) -> i32 {
    let command = kind.command();
    let (store, index) = match open_store_and_index() {
        Ok(pair) => pair,
        Err(e) => return print_error(command, json, &e),
    };
    let edit = match kind {
        TagEditKind::Add => crate::write::add_tags(&store, &index, id, tags),
        TagEditKind::Remove => crate::write::remove_tags(&store, &index, id, tags),
    };
    let edit = match edit {
        Ok(e) => e,
        Err(e) => return print_error(command, json, &e.to_string()),
    };
    let mut payload = serde_json::Map::new();
    payload.insert("id".into(), json!(edit.id));
    payload.insert(kind.changed_key().into(), json!(edit.changed));
    payload.insert("tags".into(), json!(edit.tags));
    println!(
        "{}",
        render_response(
            command,
            json,
            serde_json::Value::Object(payload),
            tag_edit_text(kind, &edit),
        )
    );
    0
}

/// Text rendering for a tag edit: the node's resulting tag set, plus what
/// changed. A call that changed nothing says so rather than printing an
/// empty list that reads like a successful write.
fn tag_edit_text(kind: TagEditKind, edit: &storage::TagEdit) -> String {
    let rendered: Vec<String> = edit.tags.iter().map(|t| format!(":{t}")).collect();
    let tags = rendered.join("");
    if edit.changed.is_empty() {
        return format!("{}  no change  {tags}\n", edit.id);
    }
    format!(
        "{}  {} {}  {tags}\n",
        edit.id,
        kind.changed_key(),
        edit.changed.join(",")
    )
}

fn run_tags_list(json: JsonOutput) -> i32 {
    let index = match crate::index::Index::open(&crate::index::configured_index_path()) {
        Ok(index) => index,
        Err(e) => return print_error("tags", json, &e.to_string()),
    };
    let counts = match index.tag_counts() {
        Ok(counts) => counts,
        Err(e) => return print_error("tags", json, &e.to_string()),
    };
    println!(
        "{}",
        render_response("tags", json, tags_json(&counts), tags_text(&counts))
    );
    0
}

fn run_tags_merge(from: &str, to: &str, json: JsonOutput) -> i32 {
    let (store, index) = match open_store_and_index() {
        Ok(pair) => pair,
        Err(e) => return print_error("tags-merge", json, &e),
    };
    let merge = match crate::write::merge_tag(&store, &index, from, to) {
        Ok(m) => m,
        Err(e) => return print_error("tags-merge", json, &e.to_string()),
    };
    println!(
        "{}",
        render_response(
            "tags-merge",
            json,
            json!({
                "from": merge.from,
                "to": merge.to,
                "rewritten": merge.rewritten,
                "count": merge.rewritten.len(),
            }),
            merge_text(&merge),
        )
    );
    0
}

fn merge_text(merge: &storage::TagMerge) -> String {
    let storage::TagMerge {
        from,
        to,
        rewritten,
    } = merge;
    if rewritten.is_empty() {
        return format!("no node carries {from}; nothing to merge");
    }
    let mut lines: Vec<String> = rewritten.iter().map(|id| format!("  {id}")).collect();
    lines.insert(
        0,
        format!(
            "merged {from} into {to} across {} node(s):",
            rewritten.len()
        ),
    );
    lines.join("\n")
}

fn tags_json(counts: &[(String, i64)]) -> serde_json::Value {
    let arr: Vec<serde_json::Value> = counts
        .iter()
        .map(|(tag, count)| json!({"tag": tag, "count": count}))
        .collect();
    serde_json::Value::Array(arr)
}

fn tags_text(counts: &[(String, i64)]) -> String {
    counts
        .iter()
        .map(|(tag, count)| format!("{count:>6}  {tag}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn run_queue(queue: &Path, store: &Path, index: &Path, command: QueueCommand) -> i32 {
    match command {
        QueueCommand::Status { json } => run_queue_status(queue, JsonOutput::from_flag(json)),
        QueueCommand::Drain { json } => {
            run_queue_drain(queue, store, index, JsonOutput::from_flag(json))
        }
        QueueCommand::Dead { json } => run_queue_dead(queue, JsonOutput::from_flag(json)),
    }
}

fn run_queue_status(path: &Path, json: JsonOutput) -> i32 {
    let queue = match crate::ingest::Queue::open(path) {
        Ok(queue) => queue,
        Err(e) => return print_error("queue-status", json, &e.to_string()),
    };
    let status = match queue.status() {
        Ok(status) => status,
        Err(e) => return print_error("queue-status", json, &e.to_string()),
    };
    let oldest = status.oldest.map(|waited| waited.as_secs());
    let text = format!(
        "kb queue: {} waiting, {} dead-lettered, oldest {}",
        status.depth,
        status.dead_lettered,
        oldest.map_or_else(|| "none".to_owned(), |seconds| format!("{seconds}s"))
    );
    println!(
        "{}",
        render_response(
            "queue-status",
            json,
            json!({
                "depth": status.depth,
                "dead_lettered": status.dead_lettered,
                "oldest_seconds": oldest,
            }),
            text,
        )
    );
    0
}

fn run_queue_drain(path: &Path, store_path: &Path, index_path: &Path, json: JsonOutput) -> i32 {
    let queue = match crate::ingest::Queue::open(path) {
        Ok(queue) => queue,
        Err(e) => return print_error("queue-drain", json, &e.to_string()),
    };
    // The worker writes where it is told, like `kb reindex`: a drain that
    // wrote to the default store while the server accepted into a named
    // queue would ingest a snapshot's submissions into the live corpus.
    let store = match crate::store::GitBlobStore::open_or_init(store_path) {
        Ok(store) => store,
        Err(e) => return print_error("queue-drain", json, &e.to_string()),
    };
    let index = match crate::index::Index::open(index_path) {
        Ok(index) => index,
        Err(e) => return print_error("queue-drain", json, &e.to_string()),
    };
    // Embedding is optional here for the same reason it is optional in the
    // worker: a configured endpoint that is down dead-letters, but an
    // unconfigured one is a machine that does not embed, and refusing to
    // drain on that basis would strand capture behind a setting.
    let embedder = crate::cli_embed::CliEmbedder::from_env().ok();
    let drained = match crate::ingest::drain(&store, &index, &queue, embedder.as_ref()) {
        Ok(drained) => drained,
        Err(e) => return print_error("queue-drain", json, &e.to_string()),
    };
    let text = format!(
        "kb queue: {} ingested, {} dead-lettered",
        drained.ingested, drained.dead_lettered
    );
    println!(
        "{}",
        render_response(
            "queue-drain",
            json,
            json!({
                "ingested": drained.ingested,
                "dead_lettered": drained.dead_lettered,
            }),
            text,
        )
    );
    0
}

fn run_queue_dead(path: &Path, json: JsonOutput) -> i32 {
    let queue = match crate::ingest::Queue::open(path) {
        Ok(queue) => queue,
        Err(e) => return print_error("queue-dead", json, &e.to_string()),
    };
    let refused = match queue.dead() {
        Ok(refused) => refused,
        Err(e) => return print_error("queue-dead", json, &e.to_string()),
    };
    let text = if refused.is_empty() {
        "kb queue: no dead letters".to_owned()
    } else {
        refused
            .iter()
            .map(|entry| format!("{}\t{}", entry.submission.id, entry.reason))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let rows: Vec<serde_json::Value> = refused
        .iter()
        .map(|entry| json!({ "id": entry.submission.id, "reason": entry.reason }))
        .collect();
    println!("{}", render_response("queue-dead", json, json!(rows), text));
    0
}

fn run_prompt(command: PromptCommand) -> i32 {
    match command {
        PromptCommand::Render { name } => run_prompt_render(&name),
        PromptCommand::List { json } => run_prompt_list(JsonOutput::from_flag(json)),
        PromptCommand::Show { name, json } => run_prompt_show(&name, JsonOutput::from_flag(json)),
    }
}

fn run_prompt_render(name: &str) -> i32 {
    let index = match crate::index::Index::open(&crate::index::configured_index_path()) {
        Ok(index) => index,
        Err(e) => {
            eprintln!("kb: {e:#}");
            return 1;
        }
    };
    match prompt::render_prompt(&index, name) {
        Ok(text) => {
            // Render is a write-through to stdout — no envelope. The
            // user pipes it to an LLM. Always include a trailing
            // newline for shell ergonomics.
            if text.ends_with('\n') {
                print!("{text}");
            } else {
                println!("{text}");
            }
            0
        }
        Err(e) => {
            eprintln!("kb: prompt render failed: {e}");
            1
        }
    }
}

fn run_prompt_list(json: JsonOutput) -> i32 {
    let summaries = prompt::list_templates();
    let payload = serde_json::Value::Array(
        summaries
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "source": t.source,
                    "path": t.path.as_ref().map(|p| p.to_string_lossy().into_owned()),
                })
            })
            .collect(),
    );
    let text = summaries
        .iter()
        .map(|t| {
            t.path.as_ref().map_or_else(
                || format!("{}\t{}", t.name, t.source),
                |p| format!("{}\t{}\t{}", t.name, t.source, p.display()),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    println!("{}", render_response("prompt-list", json, payload, text));
    0
}

fn run_prompt_show(name: &str, json: JsonOutput) -> i32 {
    let resolved = match prompt::resolve_template_source(name) {
        Ok(r) => r,
        Err(e) => return print_error("prompt-show", json, &e.to_string()),
    };
    let payload = json!({
        "name": resolved.name,
        "source": resolved.source,
        "path": resolved.path.as_ref().map(|p| p.to_string_lossy().into_owned()),
        "body": resolved.body,
    });
    println!(
        "{}",
        render_response("prompt-show", json, payload, resolved.body)
    );
    0
}

/// kb — personal knowledge base CLI.
#[derive(Debug, Parser)]
#[command(name = "kb")]
pub struct Cli {
    /// Path to the `SQLite` database file. Defaults to
    /// `$HOME/.local/share/kb/kb.db`; overridable via `KB_DB_PATH`.
    /// The superseded database. Read by `kb export` and by `kb fsck`'s
    /// check for nodes the store never archived, and by nothing else since
    /// T029: no retrieval path, no write path and no template binding
    /// consults it.
    #[arg(long, env = "KB_DB_PATH", default_value_os_t = crate::storage::default_db_path())]
    pub db: PathBuf,
    /// The subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// Subcommands for the kb knowledge base tool.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Shared metadata commands (version, license, completions).
    Meta {
        /// The shared metadata subcommand to run.
        #[command(subcommand)]
        command: MetaCommand,
    },
    /// Keyword search across nodes.
    Search(SearchArgs),
    /// Generate summaries or questions for records, skipping anything already
    /// generated under the same key.
    ///
    /// Addressed by content, generator and prompt, so a re-chunk or an
    /// embedding-model change costs nothing and only a generator or prompt
    /// change costs a pass.
    Generate(GenerateArgs),

    /// Mail-corpus tooling.
    Mail {
        /// The mail subcommand to run.
        #[command(subcommand)]
        command: MailCommand,
    },
    /// Export every node from the superseded database into a portable
    /// artifact.
    ///
    /// Reads the database read-only and never writes to it. One way: no path
    /// back into the old schema is provided.
    Export {
        /// Where to write the export artifact.
        #[arg(long)]
        out: std::path::PathBuf,
    },
    /// Import an export artifact into the content-addressed store.
    Import {
        /// The export artifact to read.
        #[arg(long = "in")]
        input: std::path::PathBuf,
        /// Path to the blob store, created if absent.
        #[arg(long, env = "KB_STORE_PATH", default_value_os_t = crate::store::default_store_path())]
        store: std::path::PathBuf,
    },
    /// Embed one corpus's passages into the derived index, skipping any
    /// already embedded.
    ///
    /// Resumable by construction: a vector is keyed by its span and the
    /// model, so a run that stops partway leaves the work it finished behind
    /// and the next run starts where it left off. That is what makes a
    /// multi-hour first pass over a corpus an ordinary operation rather than
    /// something to be attempted only when nothing can interrupt it.
    Embed {
        /// Which corpus to embed.
        #[arg(long, default_value = CORPUS_KB)]
        corpus: String,
        /// Index to write. Defaults to the configured index path.
        #[arg(long)]
        index: Option<std::path::PathBuf>,
        /// Stop after this many spans, for a bounded first pass.
        #[arg(long)]
        limit: Option<usize>,
        /// Largest span sent to the model in one request.
        #[arg(long, default_value_t = crate::cli_embed::DOCUMENT_CHUNK_CHARS)]
        chunk_bytes: usize,
    },

    /// Rebuild the derived index from the content-addressed store.
    ///
    /// The index holds nothing that cannot be re-derived, so this is a
    /// routine operation rather than a recovery procedure (ST-004).
    Reindex {
        /// Path to the blob store.
        #[arg(long, env = "KB_STORE_PATH", default_value_os_t = crate::store::default_store_path())]
        store: std::path::PathBuf,
        /// Path to the derived index. Separate from `--db`, which addresses
        /// the superseded schema.
        #[arg(long, env = "KB_INDEX_PATH", default_value_os_t = crate::index::default_index_path())]
        index: std::path::PathBuf,
        /// Rebuild only this corpus.
        #[arg(long)]
        corpus: Option<String>,
        /// Rebuild only this record.
        #[arg(long, conflicts_with = "corpus")]
        record: Option<String>,
        /// Maildir root whose mail a full rebuild re-indexes, containing
        /// folders such as `Inbox` and `Archive`. There is no default: with
        /// neither this flag nor `KB_MAIL_ROOT` set, the mail corpus is
        /// skipped.
        #[arg(long, env = "KB_MAIL_ROOT")]
        maildir: Option<std::path::PathBuf>,
    },
    /// Reconcile the derived index against the store, reporting drift and
    /// repairing what can be re-derived.
    ///
    /// Compares each stored record's blob address against the address the
    /// index recorded for it. Driven by hashes rather than by events, so it
    /// is idempotent: running it twice on a clean store changes nothing.
    Fsck {
        /// Path to the blob store.
        #[arg(long, env = "KB_STORE_PATH", default_value_os_t = crate::store::default_store_path())]
        store: std::path::PathBuf,
        /// Path to the derived index.
        #[arg(long, env = "KB_INDEX_PATH", default_value_os_t = crate::index::default_index_path())]
        index: std::path::PathBuf,
        /// Re-derive what differs and drop what the store no longer holds.
        /// Without it nothing is written and the drift is only reported.
        #[arg(long)]
        repair: bool,
        /// Skip the check for nodes the superseded database holds and the
        /// store has never archived.
        #[arg(long)]
        no_legacy: bool,
        /// Also re-normalize every stored record and report any whose stored
        /// stream is not what the current normalizer produces. Reads and
        /// re-parses every record, so it costs more than the address
        /// comparison; nothing is written either way.
        #[arg(long)]
        deep: bool,
    },
    /// Render the store into a `SilverBullet` space as markdown pages.
    ///
    /// One-directional and regenerable: every page names the record and
    /// content hash it came from, nothing is ever read back out of the space,
    /// and a full re-projection reproduces it byte for byte.
    Project {
        /// Path to the blob store.
        #[arg(long, env = "KB_STORE_PATH", default_value_os_t = crate::store::default_store_path())]
        store: std::path::PathBuf,
        /// Directory to write the space into, created if absent.
        #[arg(long, env = "KB_SPACE_PATH", default_value_os_t = crate::project::default_space_path())]
        space: std::path::PathBuf,
        /// Delete pages kb wrote whose record the store no longer holds.
        /// Without it they are only reported, because the space is the
        /// operator's own directory.
        #[arg(long)]
        prune: bool,
    },
    /// Retrieve a node by id.
    Get {
        /// Node id.
        id: String,
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
    /// Create a new node. Body is read from stdin as org text or JSON Document.
    Create {
        /// Optional caller-provided id. Defaults to a server-minted `UUIDv4`.
        #[arg(long)]
        id: Option<String>,
        /// Tags to attach to the node (merged with any tags in the body).
        /// Normalized to lowercase kebab-case: punctuation and whitespace
        /// collapse to single hyphens and a case boundary is a word break, so
        /// `Silent Critic` and `silentCritic` both become `silent-critic`. A
        /// spelling with no boundary at all is not split (`CICD` -> `cicd`,
        /// which differs from the `ci-cd` of `CI/CD`); run `kb tags` to see
        /// which spellings already exist.
        #[arg(long = "tag")]
        tags: Vec<String>,
        /// Treat stdin as GitHub-flavored markdown, converting it via pandoc.
        #[arg(long)]
        markdown: bool,
        /// Store the node even when stdin holds no document text.
        #[arg(long)]
        allow_empty: bool,
        /// Path to a JSON file naming where this node's work happened —
        /// project, `project_source`, remote, context, domains, harness,
        /// model, session, cwd, every key optional. Stored in the record
        /// header, not the body. Must be a real file path: `-` is not
        /// accepted, because stdin already carries the document body.
        #[arg(long = "provenance-json")]
        provenance_json: Option<PathBuf>,
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
    /// Replace an existing node's document. Body is read from stdin as org text or JSON Document.
    Update {
        /// Node id to update.
        id: String,
        /// Tags to attach to the node (merged with any tags in the body).
        /// Normalized to lowercase kebab-case: punctuation and whitespace
        /// collapse to single hyphens and a case boundary is a word break, so
        /// `Silent Critic` and `silentCritic` both become `silent-critic`. A
        /// spelling with no boundary at all is not split (`CICD` -> `cicd`,
        /// which differs from the `ci-cd` of `CI/CD`); run `kb tags` to see
        /// which spellings already exist.
        #[arg(long = "tag")]
        tags: Vec<String>,
        /// Treat stdin as GitHub-flavored markdown, converting it via pandoc.
        #[arg(long)]
        markdown: bool,
        /// Replace the node with an empty document when stdin holds no text.
        #[arg(long)]
        allow_empty: bool,
        /// Path to a JSON file naming where this node's work happened, the
        /// same shape `kb create --provenance-json` accepts. When given, the
        /// updated record's provenance is exactly the file's contents; its
        /// kind and source still come from the existing node. Omitted, the
        /// node's existing kind, source and provenance are all kept
        /// unchanged, exactly as `kb tags add`/`tags rm`/`tags merge` keep
        /// them. Must be a real file path: `-` is not accepted, because
        /// stdin already carries the document body.
        #[arg(long = "provenance-json")]
        provenance_json: Option<PathBuf>,
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
    /// Delete a node by id.
    Delete {
        /// Node id to delete.
        id: String,
    },
    /// Nodes semantically nearest a given node, by stored embeddings.
    Similar {
        /// Node id to find neighbours of.
        id: String,
        /// Maximum number of neighbours to return.
        #[arg(long, default_value = "10")]
        limit: usize,
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
    /// List most recently updated nodes.
    Recent {
        /// Maximum number of nodes to return.
        #[arg(long, default_value = "50")]
        limit: usize,
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
    /// List nodes carrying a given tag.
    ListByTag {
        /// Tag name (without surrounding colons).
        tag: String,
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
    /// Show forward and back `[[name]]` references for a node.
    Links {
        /// Node id.
        id: String,
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
    /// List orphan nodes — those nothing else `[[name]]`-references.
    Orphans {
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
    /// List the most-linked nodes ranked by `[[name]]` in-degree.
    Hubs {
        /// Maximum number of hubs to return.
        #[arg(long, default_value = "20")]
        limit: usize,
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
    /// List broken `[[name]]` references (no matching `#+name:` slug).
    Broken {
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
    /// List every project slug asserted by at least one record, with its
    /// record count and its newest record's date
    /// (`PLAN-20260923-project-identity` T006).
    Projects {
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
    /// Inspect and drain the capture queue.
    ///
    /// The queue is what a `SessionEnd` hook posts into; the worker turns its
    /// submissions into records. Acknowledgement means the bytes are durable,
    /// not that the record exists, so these verbs are how the difference is
    /// seen.
    Queue {
        /// Path to the queue directory. The same path `kb-mcp` accepts into,
        /// so the worker drains what the endpoint acknowledged.
        #[arg(long, env = "KB_QUEUE_PATH", default_value_os_t = crate::ingest::default_queue_path())]
        queue: PathBuf,
        /// Path to the blob store the worker writes records into.
        #[arg(long, env = "KB_STORE_PATH", default_value_os_t = crate::store::default_store_path())]
        store: PathBuf,
        /// Path to the derived index the worker indexes into.
        #[arg(long, env = "KB_INDEX_PATH", default_value_os_t = crate::index::default_index_path())]
        index: PathBuf,
        /// The `kb queue` sub-verb to run.
        #[command(subcommand)]
        command: QueueCommand,
    },
    /// Run a `MiniJinja` template against the local kb corpus.
    Prompt {
        /// The `kb prompt` sub-verb to run.
        #[command(subcommand)]
        command: PromptCommand,
    },
    /// List the normalized tag vocabulary with node counts.
    Tags {
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
        /// The `kb tags` sub-verb to run; omit to list the vocabulary.
        #[command(subcommand)]
        command: Option<TagsCommand>,
    },
}

/// Sub-verbs for `kb tags`.
#[derive(Debug, Subcommand)]
pub enum TagsCommand {
    /// Move every node carrying one tag onto another, rewriting the stored
    /// documents so the change survives a later update.
    Merge {
        /// Tag to merge away. Normalized before lookup.
        from: String,
        /// Tag to merge into.
        to: String,
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
    /// Add tags to one node's stored document, leaving its body alone.
    Add {
        /// Node id to tag.
        id: String,
        /// Tags to add. Normalized before storage.
        #[arg(required = true)]
        tags: Vec<String>,
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
    /// Remove tags from one node's stored document, leaving its body alone.
    Rm {
        /// Node id to untag.
        id: String,
        /// Tags to remove. Normalized before matching.
        #[arg(required = true)]
        tags: Vec<String>,
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
}

/// Sub-verbs for `kb queue`.
#[derive(Debug, Clone, Copy, Subcommand)]
pub enum QueueCommand {
    /// Report depth, dead-letter count and how long the oldest submission has
    /// waited.
    Status {
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
    /// Process every waiting submission once.
    Drain {
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
    /// List the submissions the worker refused, with the reasons.
    Dead {
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
}

/// Sub-verbs for `kb prompt`.
#[derive(Debug, Subcommand)]
pub enum PromptCommand {
    /// Render the named template against the live corpus and write the
    /// result to stdout.
    Render {
        /// Template name (no extension).
        name: String,
    },
    /// List available templates — built-ins plus user overrides at
    /// `$XDG_CONFIG_HOME/kb/prompts/<name>.j2`.
    List {
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
    /// Print the raw template source for the named template.
    Show {
        /// Template name (no extension).
        name: String,
        /// Emit JSON envelope.
        #[arg(long)]
        json: bool,
    },
}

// ── Helpers ────────────────────────────────────────────────────────────

fn read_stdin() -> Result<String, i32> {
    let mut buf = String::new();
    match std::io::stdin().read_to_string(&mut buf) {
        Ok(_) => Ok(buf),
        Err(e) => {
            eprintln!("kb: failed to read stdin: {e}");
            Err(1)
        }
    }
}

/// Map the `--match` flag onto the storage-level search mode.
const fn match_mode_from_flag(flag: bool) -> storage::MatchMode {
    if flag {
        storage::MatchMode::Fts5
    } else {
        storage::MatchMode::Keywords
    }
}

/// Remediation text appended to a rejected `--match` expression.
const MATCH_SYNTAX_HINT: &str = "--match sends the query to SQLite FTS5 verbatim, \
     so the expression must be valid FTS5 (`a OR b`, `a NOT b`, `\"exact phrase\"`, \
     `title:word`); drop --match to search for the text literally as keywords";

/// Render a search failure. Under [`storage::MatchMode::Fts5`] a malformed
/// expression reaches `SQLite` and comes back as an opaque `fts5: syntax error
/// near ...`; attribute it to the flag that caused it rather than presenting
/// a user error as a database fault. In keyword mode every token is quoted
/// before it reaches `SQLite`, so a syntax error is unreachable and any error
/// here is a genuine database failure.
///
/// Takes rendered text rather than a driver error because the planner fuses
/// backends whose error types have nothing in common, and a signal reports
/// its failure as a string. Nothing is lost: the FTS5 recognition was always
/// on the rendered message.
fn search_error_from(text: &str, mode: storage::MatchMode) -> String {
    if mode == storage::MatchMode::Fts5 && text.contains("fts5") {
        format!("{text}; {MATCH_SYNTAX_HINT}")
    } else {
        text.to_owned()
    }
}

/// Render a `kb create` insert failure caused by an `--id` that is already
/// taken, or `None` for any other error.
///
/// The driver reports this as `UNIQUE constraint failed: nodes.id`, which
/// names the column rather than the offending value and leaks `SQLite` and
/// the physical schema into an agent-facing surface. The
/// HTTP path has always worded it properly - `AppError::Conflict`, rendered
/// as `409 id already exists` - and this is the CLI's equivalent.
///
/// Recognition is on the driver's **structured** error rather than a
/// substring of its rendered text: an extended result code identifies the
/// kind of constraint, and the message identifies which column. A
/// substring match on `UNIQUE constraint failed` alone would misreport a
/// violation on some other table added later as an id conflict.
///
/// `nodes.id` is declared `TEXT PRIMARY KEY`, so the extended code
/// `SQLite` actually returns here is `SQLITE_CONSTRAINT_PRIMARYKEY`
/// (1555) even though the rendered message says "UNIQUE constraint
/// failed" - measured, not assumed. `SQLITE_CONSTRAINT_UNIQUE` is
/// accepted too, so the recognition survives a schema that declares the
/// column unique by some other route.
///
/// The insert is left to report the condition itself rather than
/// pre-checking with a lookup, which would add a query and a race window
/// the current code does not have.
fn id_conflict_text(id: &str) -> String {
    format!(
        "a node with id {id} already exists; \
         use `kb update {id}` to replace its document, \
         or omit --id to mint a new one"
    )
}

/// Remediation text appended to a rejected `kb create` body.
const CREATE_BODY_HINT: &str = "pipe an org document in \
     (e.g. `cat note.org | kb create --tag <tag>`), \
     or pass --allow-empty to store a node with no content";

/// Remediation text appended to a rejected `kb update` body.
const UPDATE_BODY_HINT: &str = "update replaces the whole document: pipe the new body in, \
     use `kb get <id> | kb update <id> --tag <tag>` to add tags while keeping the stored body, \
     or pass --allow-empty to deliberately empty the node";

/// Reasons a stdin document body is refused before it reaches storage.
///
/// `create` and `update` take the whole document on stdin, so a caller that
/// supplies nothing — an interactive shell, `</dev/null`, a hook or agent tool
/// call with no piped input — would otherwise store an empty document. For
/// `update` that is silent data loss, since the empty document replaces the
/// node's title and body.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
enum BodyInputError {
    /// stdin is an interactive terminal, so there is no document to read.
    #[error("no document on stdin (stdin is a terminal); {hint}")]
    Interactive {
        /// Verb-specific remediation text.
        hint: &'static str,
    },

    /// stdin was readable but held no document text.
    #[error("refusing to store an empty document; {hint}")]
    Empty {
        /// Verb-specific remediation text.
        hint: &'static str,
    },
}

/// Reject an interactive stdin before any read is attempted.
///
/// Reading a document from a terminal is not a supported workflow: the command
/// would block with no prompt. `--allow-empty` does not relax this — an
/// intentionally empty body is supplied as `</dev/null`, which is not a
/// terminal.
const fn check_stdin_source(is_terminal: bool, hint: &'static str) -> Result<(), BodyInputError> {
    if is_terminal {
        Err(BodyInputError::Interactive { hint })
    } else {
        Ok(())
    }
}

/// Reject a body that carries no document text unless the caller opted in.
fn check_body_present(
    raw: &str,
    allow_empty: bool,
    hint: &'static str,
) -> Result<(), BodyInputError> {
    if allow_empty || !raw.trim().is_empty() {
        Ok(())
    } else {
        Err(BodyInputError::Empty { hint })
    }
}

/// Failure modes of parsing a stdin body into a [`Document`].
#[derive(Debug, thiserror::Error)]
enum BodyParseError {
    /// The body looked like JSON but did not deserialize into a Document.
    #[error("invalid JSON Document: {0}")]
    Json(#[from] serde_json::Error),

    /// The body was treated as org text but the parser rejected it.
    #[error("{0}")]
    Org(#[from] parser::ParseError),

    /// `--markdown` was set and pandoc-backed conversion failed.
    #[error("{0}")]
    Markdown(#[from] markdown::MarkdownError),
}

/// Parse a stdin body into a [`Document`]. With `markdown`, the body is
/// converted from GitHub-flavored markdown via pandoc; otherwise it is
/// parsed as a JSON Document (when it starts with `{`) or as org text.
fn parse_input_body(raw: &str, markdown: bool) -> Result<Document, BodyParseError> {
    if markdown {
        Ok(markdown::markdown_to_document(raw)?)
    } else {
        parse_body(raw)
    }
}

/// Parse stdin into a [`Document`], trying JSON first (when it looks like
/// JSON) then org.
fn parse_body(raw: &str) -> Result<Document, BodyParseError> {
    if raw.trim().starts_with('{') {
        Ok(serde_json::from_str::<Document>(raw)?)
    } else {
        Ok(parser::parse_document(raw)?)
    }
}

/// Merge CLI-supplied `--tag` values into the document through the crate's
/// one placement rule, [`storage::place_tags`].
///
/// The tags are normalized on the way in rather than left as supplied. The
/// destination for a document with no heading is a `#+filetags:` value,
/// where the delimiter is a colon: a raw `--tag "a:b"` would land as two
/// tokens, and a raw `--tag "Silent Critic"` would store a spelling no
/// `kb tags` output ever shows. Normalizing here makes the document carry
/// what the tag index carries.
///
/// An argument that normalizes to nothing at all is dropped rather than
/// stored as an empty tag; `extract_tags` would discard it on the next
/// read regardless.
fn merge_cli_tags(doc: &mut Document, cli_tags: &[String]) {
    let normalized: Vec<String> = cli_tags
        .iter()
        .map(|t| storage::normalize_tag(t))
        .filter(|t| !t.is_empty())
        .collect();
    if normalized.is_empty() {
        return;
    }
    storage::place_tags(doc, &normalized);
}

/// Warn on stderr about bracket references in the just-written node that
/// resolve to nothing.
///
/// A `[[slug]]` that matches no node's `#+name:` is stored with a null
/// `target_id` and was, until this, reported nowhere: the write exited 0
/// with a silent stdout and stderr, and `kb broken` was the only feedback
/// anyone got, and only if they thought to run it.
///
/// This deliberately does **not** change the exit status. A reference to
/// a node not yet written is legitimate, and the `session-end-kb` hook
/// (`scripts/session-end-kb.sh`) runs `kb create --id <session-id>` and falls back to
/// `kb update` on failure - a warning that failed the write would break
/// session capture. The message goes to stderr so `--json` output on
/// stdout stays machine-readable.
///
/// A failure to run the query is swallowed: the node is already written,
/// and a diagnostic about a diagnostic helps nobody.
fn warn_unresolved_links(index: &crate::index::Index, record_id: &str) {
    if let Ok(names) = index.unresolved_names(record_id)
        && !names.is_empty()
    {
        eprintln!("{}", unresolved_links_warning(&names));
    }
}

/// The text of the unresolved-link warning, as a pure function of the
/// slugs so it can be asserted without a database.
///
/// It names the rule as well as the slugs, because the two mistakes that
/// produced every broken reference in the live corpus - a UUID written
/// without the `id:` prefix, and `[[name]]` used with a foreign
/// filename-slug convention - are only fixable by someone who knows what
/// a name-link resolves against.
fn unresolved_links_warning(slugs: &[String]) -> String {
    let quoted: Vec<String> = slugs.iter().map(|s| format!("[[{s}]]")).collect();
    let subject = if slugs.len() == 1 {
        "reference resolves"
    } else {
        "references resolve"
    };
    format!(
        "kb: warning: {} {subject} to nothing: {}\n\
         kb: a [[name]] link resolves against a node's `#+name:` slug; \
         to link by id write [[id:<uuid>]], not [[<uuid>]]. \
         The node was stored; run `kb broken` to review.",
        slugs.len(),
        quoted.join(", ")
    )
}

fn summaries_json(pairs: &[(String, String)]) -> serde_json::Value {
    let arr: Vec<serde_json::Value> = pairs
        .iter()
        .map(|(id, title)| json!({"id": id, "title": title}))
        .collect();
    serde_json::Value::Array(arr)
}

/// [`summaries_json`] with each hit's cosine attached as `similarity`.
///
/// A hit the vector side never scored carries `null`, not `0.0`: the keyword
/// and vector halves are not on a common scale, so *no cosine was computed*
/// must stay distinguishable from *the cosine was low*. Under `--no-vector`
/// every entry is `null`, which is the honest report — there was no vector
/// ranking to have an opinion.
fn scored_summaries_json(
    index: &crate::index::Index,
    pairs: &[(String, String)],
    scores: &std::collections::BTreeMap<&str, Option<f32>>,
    corpus: &str,
) -> serde_json::Value {
    let arr: Vec<serde_json::Value> = pairs
        .iter()
        .map(|(id, title)| {
            let similarity = scores.get(id.as_str()).copied().flatten();
            let provenance = index
                .record(id)
                .ok()
                .map_or(serde_json::Value::Null, |row| {
                    let domains = index.domains_of(id).unwrap_or_default();
                    org_meta::provenance_json(&row, &domains)
                });
            json!({
                "id": id,
                "title": title,
                "similarity": similarity,
                "corpus": corpus,
                "provenance": provenance,
            })
        })
        .collect();
    serde_json::Value::Array(arr)
}

fn summaries_text(pairs: &[(String, String)]) -> String {
    pairs
        .iter()
        .map(|(id, title)| format!("{id}  {title}"))
        .collect::<Vec<_>>()
        .join("\n")
}

// ── Command handlers ───────────────────────────────────────────────────

/// Results `kb search` returns when `--limit` is not given.
///
/// A bound is not cosmetic. The vector half of the fusion ranks every node
/// that has an embedding, so before this existed `kb search` printed the
/// whole corpus for every query — including one that matched nothing.
/// `kb similar` has always defaulted to 10; an unbounded `search` was also
/// an inconsistency between two commands answering the same kind of
/// question.
pub const DEFAULT_SEARCH_LIMIT: usize = 20;

/// The ground-truth question set, whose mail `node_id`s override the
/// discriminant.
const DEFAULT_QUESTIONS: &str = "resources/eval/retrieval-questions.toml";

/// Which form `kb generate` asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum GenerateForm {
    /// One paragraph describing the document.
    Summary,
    /// Questions the document answers.
    Questions,
}

impl GenerateForm {
    const fn to_form(self) -> crate::generate::Form {
        match self {
            Self::Summary => crate::generate::Form::Summary,
            Self::Questions => crate::generate::Form::Questions,
        }
    }
}

/// How `kb search` combines passage similarities into one dense record score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum DensePoolingArg {
    /// Use the record's single highest passage score.
    #[value(name = "max")]
    Maximum,
    /// Average the record's best three passage scores.
    MeanTopThree,
    /// Subtract the corpus-fitted expected maximum for the record's passage count.
    #[value(name = "length-normalized-max")]
    LengthNormalizedMaximum,
}

impl DensePoolingArg {
    const fn to_pooling(self) -> crate::index::DensePooling {
        match self {
            Self::Maximum => crate::index::DensePooling::Maximum,
            Self::MeanTopThree => crate::index::DensePooling::MeanTopThree,
            Self::LengthNormalizedMaximum => crate::index::DensePooling::LengthNormalizedMaximum,
        }
    }
}

/// Parse a count whose zero value would silently disable the stage it sizes.
fn positive_usize(raw: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|error| error.to_string())
        .and_then(|value| {
            (value > 0)
                .then_some(value)
                .ok_or_else(|| "value must be at least 1".to_owned())
        })
}

/// Validate a `--project` value against `tftio_lib::project::Slug`'s
/// grammar at the CLI boundary (`PLAN-20260923-project-identity` T006), so
/// clap reports a malformed slug as its own error rather than the filter
/// silently matching nothing.
fn parse_project_slug(raw: &str) -> Result<tftio_lib::project::Slug, String> {
    tftio_lib::project::Slug::new(raw).map_err(|e| e.to_string())
}

/// Everything `kb search` was asked to do.
#[derive(Debug, clap::Args)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each is an independent switch on the command line, not a state machine; collapsing them into enums would change a published CLI surface (REPO_INVARIANTS.md CLI-002)"
)]
pub struct SearchArgs {
    /// Keywords to search for. Every token must appear and each matches as
    /// a prefix; FTS5 operators are matched literally unless --match is set.
    pub query: String,
    /// Interpret the query as a verbatim `SQLite` FTS5 expression, enabling
    /// OR, NOT, NEAR, "phrases", `term*` and the `title:` / `body:` column
    /// filters.
    #[arg(long)]
    pub r#match: bool,
    /// Rank by keyword relevance alone, contacting no embedding endpoint.
    /// Makes results reproducible and independent of daemon availability.
    #[arg(long)]
    pub no_vector: bool,
    /// Maximum number of results to return.
    #[arg(long, default_value_t = DEFAULT_SEARCH_LIMIT)]
    pub limit: usize,
    /// Drop vector candidates scoring below this cosine. Overrides the
    /// model's recorded calibration and `KB_EMBEDDING_MIN_SIMILARITY`.
    /// Pass -1 to disable the floor entirely; negative values are
    /// accepted as values rather than parsed as flags.
    #[arg(long, allow_negative_numbers = true)]
    pub min_similarity: Option<f32>,
    /// How passage similarities become one dense score per record.
    #[arg(long, value_enum, default_value_t = DensePoolingArg::MeanTopThree)]
    pub dense_pooling: DensePoolingArg,
    /// How many dense records enter fusion. The default remains the measured
    /// production cut; larger values are retrieval experiments.
    #[arg(
        long,
        default_value_t = crate::storage::VECTOR_CANDIDATES,
        value_parser = positive_usize
    )]
    pub vector_candidates: usize,
    /// Return the fused ranking without the cross-encoder stage, even
    /// where `KB_RERANK_BASE_URL` configures one. The reordering is
    /// reproducible without it and costs no requests.
    #[arg(long)]
    pub no_rerank: bool,
    /// Which corpus to search. The structural filter, applied before any
    /// textual retriever runs. Defaults to the kb corpus; `mail` dispatches
    /// lexical retrieval to mu and dense retrieval to the derived index.
    #[arg(long, default_value = CORPUS_KB)]
    pub corpus: String,
    /// Restrict results to records asserting this project slug
    /// (`PLAN-20260923-project-identity` T006). Parsed against the slug
    /// grammar at the CLI boundary, so an invalid value is a clear error
    /// rather than a filter that silently matches nothing.
    #[arg(long, value_parser = parse_project_slug)]
    pub project: Option<tftio_lib::project::Slug>,
    /// Restrict results to records asserting this context (`personal`,
    /// `work`, ...). A plain string: context is a credential boundary, not
    /// a project, and carries no grammar of its own.
    #[arg(long)]
    pub context: Option<String>,
    /// Report what each signal returned, what was skipped, what fusion
    /// produced and what reranking changed, instead of the results.
    #[arg(long)]
    pub explain: bool,
    /// Emit JSON envelope.
    #[arg(long)]
    pub json: bool,
}

// The corpora a search names, defined where searching is implemented. Kept
// re-exported here because `--corpus`'s default is one of them and because
// they have been part of this module's published surface since T017.
pub use crate::search::{CORPUS_KB, CORPUS_MAIL};

/// Everything `kb generate` was asked to do.
#[derive(Debug, clap::Args)]
pub struct GenerateArgs {
    /// What to ask of each document.
    #[arg(long, value_enum, default_value_t = GenerateForm::Summary)]
    pub form: GenerateForm,
    /// The prompt's version. Changing the wording must change this, or a
    /// figure measured under one wording gets compared with one measured
    /// under another.
    #[arg(long, default_value_t = 2)]
    pub prompt_version: u32,
    /// Generate for one corpus.
    #[arg(long, conflicts_with = "record")]
    pub corpus: Option<String>,
    /// Generate for named records only, repeatable.
    #[arg(long)]
    pub record: Vec<String>,
    /// Stop after this many records.
    #[arg(long)]
    pub limit: Option<usize>,
    /// Index to read and write. Defaults to the configured index path.
    #[arg(long)]
    pub index: Option<std::path::PathBuf>,
    /// Chat-completions endpoint.
    #[arg(
        long,
        env = "KB_GENERATION_BASE_URL",
        default_value = "http://127.0.0.1:1234/v1"
    )]
    pub base_url: String,
    /// Generating model.
    #[arg(long, env = "KB_GENERATION_MODEL")]
    pub model: String,
    /// Token budget per completion.
    #[arg(long, default_value_t = 3000)]
    pub budget: u32,
}

/// Generate for a scope, reporting what was produced and what was reused.
fn run_generate(args: &GenerateArgs) -> i32 {
    let path = args
        .index
        .clone()
        .unwrap_or_else(crate::index::default_index_path);
    let index = match crate::index::Index::open(&path) {
        Ok(index) => index,
        Err(error) => return print_error("generate", JsonOutput::Text, &error.to_string()),
    };
    let mut records = if args.record.is_empty() {
        // No scope named is the whole index rather than nothing: an empty
        // selection reported as a successful run of zero would read as
        // "already generated".
        let corpus = args.corpus.as_deref().unwrap_or("kb");
        match index.records_in(corpus) {
            Ok(found) => found,
            Err(error) => return print_error("generate", JsonOutput::Text, &error.to_string()),
        }
    } else {
        args.record.clone()
    };
    if let Some(cap) = args.limit {
        records.truncate(cap);
    }
    if records.is_empty() {
        return print_error("generate", JsonOutput::Text, "no records matched the scope");
    }
    let generator =
        match crate::generate::HttpGenerator::new(&args.base_url, &args.model, args.budget) {
            Ok(generator) => generator,
            Err(error) => return print_error("generate", JsonOutput::Text, &error.to_string()),
        };
    let plan = crate::generate::Plan::new(args.form.to_form(), args.prompt_version);
    let started = std::time::Instant::now();
    match crate::generate::generate_missing(&index, &plan, &generator, &records) {
        Ok(report) => {
            println!(
                "kb generate: {} {} generated, {} already current, {} unknown, in {} ({} under {} by {})",
                report.generated,
                plural(report.generated, "record", "records"),
                report.reused,
                report.unknown.len(),
                format_elapsed(started.elapsed()),
                records.len(),
                plan.prompt_key(),
                args.model,
            );
            for unknown in &report.unknown {
                eprintln!("kb generate: no such record: {unknown}");
            }
            i32::from(!report.unknown.is_empty())
        }
        Err(error) => print_error("generate", JsonOutput::Text, &error.to_string()),
    }
}

/// Subcommands for the mail corpus.
#[derive(Debug, clap::Subcommand)]
pub enum MailCommand {
    /// Classify every message in a Maildir as bulk or non-bulk, one JSON
    /// object per line.
    ///
    /// Reads only; the Maildir is never modified. The rule's error rate is
    /// what T018 measures, so this exists to feed a sampling pass rather than
    /// to make a decision on its own.
    Classify {
        /// Maildir root, containing folders such as `Inbox` and `Archive`.
        #[arg(long, env = "KB_MAIL_ROOT")]
        maildir: std::path::PathBuf,
        /// Folders to walk, relative to the root.
        #[arg(long, default_values_t = [String::from("Inbox"), String::from("Archive")])]
        folder: Vec<String>,
    },

    /// Report which messages the first increment would index, and why the
    /// rest are out.
    ///
    /// The folders are fixed (`Inbox` and `Archive`); widening them is meant
    /// to be a code change rather than a flag.
    Scope {
        /// Maildir root, containing folders such as `Inbox` and `Archive`.
        /// Required: there is no default mailbox.
        #[arg(long, env = "KB_MAIL_ROOT")]
        maildir: std::path::PathBuf,
        /// Ground-truth question set, whose mail `node_id`s override the
        /// discriminant.
        #[arg(long, default_value = DEFAULT_QUESTIONS)]
        questions: std::path::PathBuf,
    },

    /// Rebuild the mail corpus's rows in the derived index.
    ///
    /// Discards and re-derives rather than migrating, like every other
    /// rebuild path: the index is disposable and the Maildir is canonical.
    /// Other corpora are left untouched.
    Index {
        /// Maildir root, containing folders such as `Inbox` and `Archive`.
        /// Required: there is no default mailbox.
        #[arg(long, env = "KB_MAIL_ROOT")]
        maildir: std::path::PathBuf,
        /// Ground-truth question set, whose mail `node_id`s override the
        /// discriminant.
        #[arg(long, default_value = DEFAULT_QUESTIONS)]
        questions: std::path::PathBuf,
        /// Index to write. Defaults to the configured index path.
        #[arg(long)]
        index: Option<std::path::PathBuf>,
    },

    /// Embed the mail corpus's passages, skipping any already embedded.
    ///
    /// Resumable by construction: a vector is keyed by its span and the
    /// model, so a run that stops partway leaves the work it finished behind
    /// and the next run starts where it left off.
    Embed {
        /// Index to write. Defaults to the configured index path.
        #[arg(long)]
        index: Option<std::path::PathBuf>,
        /// Stop after this many spans, for a bounded first pass.
        #[arg(long)]
        limit: Option<usize>,
        /// Largest span sent to the model in one request.
        #[arg(long, default_value_t = crate::cli_embed::DOCUMENT_CHUNK_CHARS)]
        chunk_bytes: usize,
    },
}

/// Embed one corpus's passages, reporting throughput and cost.
///
/// Corpus-general since T029: the derived index serves every corpus, so the
/// command that fills it names the one it means rather than assuming the only
/// one that had ever been served from it.
fn run_embed(
    command: &str,
    corpus: &str,
    index_path: Option<&std::path::Path>,
    limit: Option<usize>,
    chunk_bytes: usize,
) -> i32 {
    let embedder = match crate::cli_embed::CliEmbedder::from_env() {
        Ok(embedder) => embedder,
        Err(error) => return print_error(command, JsonOutput::Text, &error.to_string()),
    };
    let path = index_path.map_or_else(crate::index::default_index_path, ToOwned::to_owned);
    let index = match crate::index::Index::open(&path) {
        Ok(index) => index,
        Err(error) => return print_error(command, JsonOutput::Text, &error.to_string()),
    };
    let records = match index.records_in(corpus) {
        Ok(records) => records,
        Err(error) => return print_error(command, JsonOutput::Text, &error.to_string()),
    };
    let started = std::time::Instant::now();
    let mut total = crate::write::Embedded::default();
    for record in &records {
        // The bound is checked between records rather than between spans: a
        // record is the unit worth having embedded, and stopping inside one
        // leaves a document whose second half no query can reach.
        if limit.is_some_and(|cap| total.embedded >= cap) {
            break;
        }
        match crate::write::embed_record(&index, &embedder, record, chunk_bytes) {
            Ok(done) => total.add(done),
            Err(error) => {
                // Report what was finished before failing: the work is
                // resumable and a run that says nothing about its progress
                // makes the operator redo it.
                eprintln!(
                    "kb {command}: stopped after {} spans: {error}",
                    total.embedded
                );
                report_embed(command, total, started.elapsed());
                return 1;
            }
        }
    }
    report_embed(command, total, started.elapsed())
}

fn report_embed(command: &str, done: crate::write::Embedded, elapsed: std::time::Duration) -> i32 {
    let crate::write::Embedded {
        embedded,
        skipped,
        bytes,
    } = done;
    let each = if embedded == 0 {
        0.0
    } else {
        elapsed.as_secs_f64() / f64::from(u32::try_from(embedded).unwrap_or(u32::MAX))
    };
    println!(
        "kb {command}: {embedded} {} embedded, {skipped} already current, {:.1} MiB of text in {} ({:.3}s per span)",
        plural(embedded, "span", "spans"),
        bytes_as_mib(bytes),
        format_elapsed(elapsed),
        each,
    );
    0
}

/// Rebuild the mail corpus as part of a full re-derivation.
///
/// Absent mail is normal rather than an error: mail is indexed only when a
/// Maildir root is configured, and an index rebuilt without one is a correct
/// index of one corpus. Reported either way, so "no mail" is a statement
/// rather than a silence.
fn reindex_mail(index: &crate::index::Index, root: Option<&std::path::Path>) {
    let Some(root) = root else {
        println!(
            "kb reindex: mail skipped — no Maildir configured (pass --maildir or set KB_MAIL_ROOT)"
        );
        return;
    };
    if !root.is_dir() {
        println!(
            "kb reindex: no mail at {} — the mail corpus is empty on this machine",
            root.display()
        );
        return;
    }
    let questions = std::path::Path::new(DEFAULT_QUESTIONS);
    let ground_truth = crate::maildir::ground_truth_ids(questions).unwrap_or_default();
    let corpus = match crate::maildir::MaildirCorpus::scan(root) {
        Ok(corpus) => corpus,
        Err(error) => {
            eprintln!("kb reindex: mail could not be read: {error}");
            return;
        }
    };
    let selection = crate::maildir::Selection::compute(
        &corpus,
        &crate::mail::BulkRule::default(),
        &ground_truth,
    );
    match crate::index::rebuild_mail(&corpus, &selection, index) {
        Ok(report) => println!(
            "kb reindex: mail {} {}, {} {} in {}",
            report.records,
            plural(report.records, "record", "records"),
            report.passages,
            plural(report.passages, "passage", "passages"),
            format_elapsed(report.elapsed),
        ),
        Err(error) => eprintln!("kb reindex: mail rebuild failed: {error}"),
    }
}

/// Rebuild the mail corpus's rows, reporting what was written.
fn run_mail_index(
    root: &std::path::Path,
    questions: &std::path::Path,
    index_path: Option<&std::path::Path>,
) -> i32 {
    let expectations = match crate::maildir::ground_truth_expectations(questions) {
        Ok(found) => found,
        Err(error) => return print_error("mail index", JsonOutput::Text, &error.to_string()),
    };
    let ground_truth: std::collections::BTreeSet<String> = expectations.keys().cloned().collect();
    let corpus = match crate::maildir::MaildirCorpus::scan(root) {
        Ok(corpus) => corpus,
        Err(error) => return print_error("mail index", JsonOutput::Text, &error.to_string()),
    };
    let selection = crate::maildir::Selection::compute(
        &corpus,
        &crate::mail::BulkRule::default(),
        &ground_truth,
    );
    let drift = crate::maildir::verify(&corpus, &expectations);
    for moved in &drift {
        eprintln!(
            "kb mail index: ground truth drifted: {} expected {} found {}",
            moved.message_id,
            moved.expected,
            if moved.found.is_empty() {
                "nothing"
            } else {
                moved.found.as_str()
            }
        );
    }
    let path = index_path.map_or_else(crate::index::default_index_path, ToOwned::to_owned);
    let index = match crate::index::Index::open(&path) {
        Ok(index) => index,
        Err(error) => return print_error("mail index", JsonOutput::Text, &error.to_string()),
    };
    match crate::index::rebuild_mail(&corpus, &selection, &index) {
        Ok(report) => {
            println!(
                "kb mail index: {} {}, {} {} in {} ({} selected of {} catalogued, {} excluded as bulk, {} {} readmitted by ground truth)",
                report.records,
                plural(report.records, "record", "records"),
                report.passages,
                plural(report.passages, "passage", "passages"),
                format_elapsed(report.elapsed),
                selection.selected.len(),
                corpus.len(),
                selection.excluded,
                selection.overrides.len(),
                plural(selection.overrides.len(), "message", "messages"),
            );
            i32::from(!drift.is_empty())
        }
        Err(error) => print_error("mail index", JsonOutput::Text, &error.to_string()),
    }
}

/// Report the first increment's scope over the configured Maildir.
fn run_mail_scope(root: &std::path::Path, questions: &std::path::Path) -> i32 {
    let expectations = match crate::maildir::ground_truth_expectations(questions) {
        Ok(found) => found,
        Err(error) => return print_error("mail scope", JsonOutput::Text, &error.to_string()),
    };
    let ground_truth: std::collections::BTreeSet<String> = expectations.keys().cloned().collect();
    let corpus = match crate::maildir::MaildirCorpus::scan(root) {
        Ok(corpus) => corpus,
        Err(error) => return print_error("mail scope", JsonOutput::Text, &error.to_string()),
    };
    let rule = crate::mail::BulkRule::default();
    let selection = crate::maildir::Selection::compute(&corpus, &rule, &ground_truth);
    let drift = crate::maildir::verify(&corpus, &expectations);
    println!(
        "{}",
        serde_json::json!({
            "maildir": root.display().to_string(),
            "folders": crate::maildir::INCREMENT_FOLDERS,
            "catalogued": corpus.len(),
            "unidentified": corpus.unidentified(),
            "selected": selection.selected.len(),
            "excluded_as_bulk": selection.excluded,
            "ground_truth": ground_truth.len(),
            "overrides": selection.overrides,
            "out_of_scope": selection.out_of_scope,
            "missing_ground_truth": selection.missing_ground_truth,
            "drifted": drift.iter().map(|d| serde_json::json!({
                "message_id": d.message_id,
                "expected": d.expected,
                "found": d.found,
            })).collect::<Vec<_>>(),
        })
    );
    // Drift is not a crash: the bytes are somebody else's and may legitimately
    // move. It is a nonzero status because a figure measured against drifted
    // ground truth is not comparable to the one it is being compared with.
    i32::from(!drift.is_empty())
}

/// Walk a Maildir and print one classification per message.
fn run_mail_classify(maildir: &std::path::Path, folders: &[String]) -> i32 {
    if !maildir.is_dir() {
        return print_error(
            "mail classify",
            JsonOutput::Text,
            &format!("no Maildir at {}", maildir.display()),
        );
    }
    let rule = crate::mail::BulkRule::default();
    let mut seen = 0usize;
    for folder in folders {
        for sub in ["cur", "new"] {
            let dir = maildir.join(folder).join(sub);
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let Ok(bytes) = std::fs::read(entry.path()) else {
                    continue;
                };
                // Mail is not reliably UTF-8; a lossy read is right here
                // because only header names and a few values are consulted,
                // and refusing a message would drop it from the measurement.
                let text = String::from_utf8_lossy(&bytes);
                let described = rule.describe(&text, folder);
                let (classification, signal) = match &described.classification {
                    crate::mail::Classification::Bulk { signal } => ("bulk", signal.as_str()),
                    crate::mail::Classification::NonBulk => ("non-bulk", ""),
                };
                println!(
                    "{}",
                    serde_json::json!({
                        "message_id": described.message_id,
                        "folder": described.folder,
                        "classification": classification,
                        "signal": signal,
                        "from": described.from,
                        "subject": described.subject,
                        "date": described.date,
                    })
                );
                seen = seen.saturating_add(1);
            }
        }
    }
    eprintln!("kb mail classify: {seen} messages");
    0
}

/// Export the superseded corpus, reporting what was written.
fn run_export(db_path: &std::path::Path, out: &std::path::Path) -> i32 {
    match crate::migrate::export_from_db(db_path, out) {
        Ok(report) => {
            println!(
                "kb export: {} {} ({:.1} MiB of node text) to {}",
                report.nodes,
                plural(report.nodes, "node", "nodes"),
                bytes_as_mib(report.bytes),
                out.display(),
            );
            0
        }
        Err(e) => print_error("export", JsonOutput::Text, &e.to_string()),
    }
}

/// Import an export artifact into the store, reporting what was written.
fn run_import(input: &std::path::Path, store_path: &std::path::Path) -> i32 {
    let store = match crate::store::GitBlobStore::open_or_init(store_path) {
        Ok(s) => s,
        Err(e) => return print_error("import", JsonOutput::Text, &e.to_string()),
    };
    match crate::migrate::import_into_store(input, &store) {
        Ok(report) => {
            println!(
                "kb import: {} {} ({} {}) into {}",
                report.records,
                plural(report.records, "record", "records"),
                report.transcripts,
                plural(report.transcripts, "transcript", "transcripts"),
                store_path.display(),
            );
            0
        }
        Err(e) => print_error("import", JsonOutput::Text, &e.to_string()),
    }
}

/// Render a byte count in mebibytes, for figures an operator compares against
/// what `du` tells them.
fn bytes_as_mib(bytes: usize) -> f64 {
    #[allow(
        clippy::cast_precision_loss,
        reason = "a corpus size in bytes is far below f64's exact-integer range"
    )]
    let value = bytes as f64;
    value / 1_048_576.0
}

/// Rebuild the derived index from the store, and report what it cost.
///
/// The elapsed time is not decoration. The architecture's claim is that a
/// retrieval experiment is a rebuild rather than a migration, and that claim
/// is only true while rebuilding is cheap enough that people actually do it —
/// so the number is put in front of the operator every time.
/// Report drift between store and index, and repair it when asked.
///
/// Reporting and repairing are one command with a flag rather than two
/// commands, because the repair acts on the survey it just printed: a
/// `kb fsck` followed by a separate `kb repair` would act on drift that may
/// have changed in between, and an operator who read the first would be
/// approving something else.
fn run_fsck(
    store_path: &std::path::Path,
    index_path: &std::path::Path,
    repair: bool,
    legacy_db: Option<&std::path::Path>,
    deep: bool,
) -> i32 {
    let store = match crate::store::GitBlobStore::open(store_path) {
        Ok(s) => s,
        Err(e) => return print_error("fsck", JsonOutput::Text, &e.to_string()),
    };
    let index = match crate::index::Index::open_for_rebuild(index_path) {
        Ok(i) => i,
        Err(e) => return print_error("fsck", JsonOutput::Text, &e.to_string()),
    };
    // A legacy database that is not there is not a failure: the check exists
    // for a machine mid-migration, and every other machine should not be told
    // it has a problem for having finished.
    let legacy = legacy_db
        .filter(|path| path.exists())
        .and_then(|path| path.to_str())
        .and_then(|path| storage::open_db(path).ok());
    let found = match crate::fsck::survey_with(&store, &index, legacy.as_ref(), deep) {
        Ok(found) => found,
        Err(e) => return print_error("fsck", JsonOutput::Text, &e.to_string()),
    };
    report_survey(&found);
    if !repair {
        if !found.is_clean() {
            println!("kb fsck: run with --repair to re-derive what differs");
        }
        return 0;
    }
    match crate::fsck::repair(&store, &index, &found) {
        Ok(done) => {
            println!(
                "kb fsck: re-derived {} {} into {} {}, dropped {} {} in {}",
                done.rederived,
                plural(done.rederived, "record", "records"),
                done.passages,
                plural(done.passages, "passage", "passages"),
                done.dropped,
                plural(done.dropped, "record", "records"),
                format_elapsed(done.elapsed),
            );
            if done.stale_vectors_dropped > 0 {
                println!(
                    "kb fsck: dropped {} {} whose text is no longer indexed",
                    done.stale_vectors_dropped,
                    plural(done.stale_vectors_dropped, "vector", "vectors"),
                );
            }
            0
        }
        Err(e) => print_error("fsck", JsonOutput::Text, &e.to_string()),
    }
}

/// Print what a survey found, naming the records rather than counting them.
fn report_survey(found: &crate::fsck::Survey) {
    println!(
        "kb fsck: checked {} stored {} in {}",
        found.checked,
        plural(found.checked, "record", "records"),
        format_elapsed(found.elapsed),
    );
    for (label, ids) in [
        ("absent from the index", &found.missing),
        ("indexed from different content", &found.stale),
        ("indexed but no longer stored", &found.orphaned),
    ] {
        if ids.is_empty() {
            continue;
        }
        println!(
            "kb fsck: {} {} {label}",
            ids.len(),
            plural(ids.len(), "record", "records")
        );
        for id in ids.iter().take(FSCK_NAMED) {
            println!("    {id}");
        }
        if ids.len() > FSCK_NAMED {
            println!("    ... and {} more", ids.len() - FSCK_NAMED);
        }
    }
    if found.is_clean() {
        println!("kb fsck: the index agrees with the store");
    }
    if !found.renormalized.is_empty() {
        println!(
            "kb fsck: {} {} whose stored stream is not what the current normalizer \
             produces; this is not repaired here — correcting it rewrites the record in \
             the archival store",
            found.renormalized.len(),
            plural(found.renormalized.len(), "record", "records"),
        );
        for id in found.renormalized.iter().take(FSCK_NAMED) {
            println!("    {id}");
        }
        if found.renormalized.len() > FSCK_NAMED {
            println!("    ... and {} more", found.renormalized.len() - FSCK_NAMED);
        }
    } else if found.deep {
        println!(
            "kb fsck: every stored stream carries the content the current normalizer produces"
        );
    }
    if found.behind > 0 {
        println!(
            "kb fsck: {} {} normalized under an older version whose content is unchanged; \
             the version is one global counter, so a bump for one artifact kind marks every \
             kind behind",
            found.behind,
            plural(found.behind, "record", "records"),
        );
    }
    if !found.unarchived.is_empty() {
        println!(
            "kb fsck: {} {} in the superseded database and never archived to the store; \
             this is not repaired here — `kb export` and `kb import` do it",
            found.unarchived.len(),
            plural(found.unarchived.len(), "node", "nodes"),
        );
        for id in found.unarchived.iter().take(FSCK_NAMED) {
            println!("    {id}");
        }
        if found.unarchived.len() > FSCK_NAMED {
            println!("    ... and {} more", found.unarchived.len() - FSCK_NAMED);
        }
    }
}

/// Render the store into a `SilverBullet` space, and report what changed.
///
/// The store is opened rather than created: a mistyped path must not produce
/// an empty space reported as a successful projection (ENG-004).
fn run_project(store_path: &std::path::Path, space: &std::path::Path, prune: bool) -> i32 {
    let store = match crate::store::GitBlobStore::open(store_path) {
        Ok(s) => s,
        Err(e) => return print_error("project", JsonOutput::Text, &e.to_string()),
    };
    match crate::project::project(&store, space, prune) {
        Ok(done) => {
            report_projection(&done, space);
            0
        }
        Err(e) => print_error("project", JsonOutput::Text, &e.to_string()),
    }
}

/// Print what a projection did, naming what an operator would otherwise have
/// to go looking for.
fn report_projection(done: &crate::project::Projection, space: &std::path::Path) {
    println!(
        "kb project: {} {} into {} in {} — {} written, {} refreshed, {} unchanged",
        done.records,
        plural(done.records, "record", "records"),
        space.display(),
        format_elapsed(done.elapsed),
        done.written,
        done.refreshed,
        done.unchanged,
    );
    if !done.hand_edited.is_empty() {
        println!(
            "kb project: {} {} edited by hand and overwritten; the space is a view, \
             and edits belong in the record",
            done.hand_edited.len(),
            plural(done.hand_edited.len(), "page", "pages"),
        );
        name_some(&done.hand_edited);
    }
    if !done.orphaned.is_empty() {
        println!(
            "kb project: {} {} whose record is no longer in the store{}",
            done.orphaned.len(),
            plural(done.orphaned.len(), "page", "pages"),
            if done.pruned > 0 {
                String::new()
            } else {
                "; run with --prune to remove them".to_owned()
            },
        );
        name_some(&done.orphaned);
    }
    if done.pruned > 0 {
        println!(
            "kb project: removed {} orphan {}",
            done.pruned,
            plural(done.pruned, "page", "pages"),
        );
    }
    if !done.unresolved.is_empty() {
        println!(
            "kb project: {} {} naming nothing the projection contains, rendered as text",
            done.unresolved.len(),
            plural(done.unresolved.len(), "link", "links"),
        );
        name_some(&done.unresolved);
    }
}

/// Name the first few of a list and count the rest, the way `kb fsck` does.
fn name_some(items: &[String]) {
    for item in items.iter().take(FSCK_NAMED) {
        println!("    {item}");
    }
    if items.len() > FSCK_NAMED {
        println!("    ... and {} more", items.len() - FSCK_NAMED);
    }
}

/// How many drifted records `kb fsck` names before summarising the rest.
///
/// Naming them is the point — a count says something is wrong and nothing
/// about what — but a first run after a schema change can list thousands, and
/// a report nobody can read is a report nobody reads.
const FSCK_NAMED: usize = 20;

fn run_reindex(
    store_path: &std::path::Path,
    index_path: &std::path::Path,
    corpus: Option<&str>,
    record: Option<&str>,
    maildir: Option<&std::path::Path>,
) -> i32 {
    let store = match crate::store::GitBlobStore::open(store_path) {
        Ok(s) => s,
        Err(e) => return print_error("reindex", JsonOutput::Text, &e.to_string()),
    };
    let index = match crate::index::Index::open_for_rebuild(index_path) {
        Ok(i) => i,
        Err(e) => return print_error("reindex", JsonOutput::Text, &e.to_string()),
    };
    let scope = match (corpus, record) {
        (Some(c), _) => crate::index::Scope::Corpus(c.to_owned()),
        (None, Some(r)) => crate::index::Scope::Record(r.to_owned()),
        (None, None) => crate::index::Scope::All,
    };
    match crate::index::rebuild(&store, &index, &scope) {
        Ok(mut report) => {
            println!(
                "kb reindex: {} {}, {} {}, {} {} in {}",
                report.records,
                plural(report.records, "record", "records"),
                report.passages,
                plural(report.passages, "passage", "passages"),
                report.authored_links,
                plural(report.authored_links, "link", "links"),
                format_elapsed(report.elapsed),
            );
            // A full rebuild clears every corpus but the store only
            // repopulates its own, so stopping here would leave `kb reindex`
            // silently deleting the mail corpus and requiring a second
            // command to notice. Re-derivation is meant to be the routine
            // operation (ST-004), which it cannot be if running it costs a
            // corpus.
            if matches!(scope, crate::index::Scope::All) {
                reindex_mail(&index, maildir);
                // Only now is every corpus back, so only now can a vector be
                // called an orphan. Sweeping inside the store-backed rebuild
                // would delete every mail vector on the way past.
                match index.drop_orphaned_derivations() {
                    Ok(dropped) => report.stale_vectors_dropped = dropped,
                    Err(e) => return print_error("reindex", JsonOutput::Text, &e.to_string()),
                }
            }
            if report.stale_vectors_dropped > 0 {
                println!(
                    "kb reindex: dropped {} {} whose text is no longer indexed",
                    report.stale_vectors_dropped,
                    plural(report.stale_vectors_dropped, "vector", "vectors"),
                );
            }
            0
        }
        Err(e) => print_error("reindex", JsonOutput::Text, &e.to_string()),
    }
}

/// Pick the singular or plural form for a count.
const fn plural<'a>(count: usize, one: &'a str, many: &'a str) -> &'a str {
    if count == 1 { one } else { many }
}

/// Render a duration the way an operator reads it, not the way `Debug` does.
fn format_elapsed(elapsed: std::time::Duration) -> String {
    let seconds = elapsed.as_secs_f64();
    if seconds >= 1.0 {
        format!("{seconds:.1}s")
    } else {
        format!("{}ms", elapsed.as_millis())
    }
}

/// The trace, rendered for `--explain`.
fn trace_json(trace: &retrieval::Trace) -> serde_json::Value {
    json!({
        "signals": trace.signals.iter().map(|signal| json!({
            "name": signal.name,
            "kind": signal.kind.as_str(),
            "corpus": signal.corpus,
            "count": signal.candidates.len(),
            "candidates": signal.candidates,
            "error": signal.error,
            "elapsed_ms": signal.elapsed.as_secs_f64() * 1000.0,
        })).collect::<Vec<_>>(),
        "skipped": trace.skipped.iter().map(|skipped| json!({
            "name": skipped.name,
            "reason": skipped.reason,
        })).collect::<Vec<_>>(),
        "fused": trace.fused,
        "rerank": trace.rerank.as_ref().map(|record| json!({
            "window": record.window,
            "before": record.before,
            "after": record.after,
            "error": record.error,
            "elapsed_ms": record.elapsed.as_secs_f64() * 1000.0,
        })),
    })
}

/// The trace, rendered for a human.
fn trace_text(trace: &retrieval::Trace) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for signal in &trace.signals {
        let _ = write!(
            out,
            "{} ({}, {}): {} candidates in {:.1}ms",
            signal.name,
            signal.kind.as_str(),
            signal.corpus,
            signal.candidates.len(),
            signal.elapsed.as_secs_f64() * 1000.0
        );
        if let Some(error) = &signal.error {
            let _ = write!(out, " — failed: {error}");
        }
        out.push('\n');
        for id in signal.candidates.iter().take(TRACE_CANDIDATES) {
            let _ = writeln!(out, "    {id}");
        }
    }
    for skipped in &trace.skipped {
        let _ = writeln!(out, "{} skipped: {}", skipped.name, skipped.reason);
    }
    let _ = writeln!(out, "fused: {} candidates", trace.fused.len());
    if let Some(record) = &trace.rerank {
        let _ = write!(out, "reranked: window {}", record.window);
        if let Some(error) = &record.error {
            let _ = write!(out, " — fused order stands: {error}");
        } else {
            let moved = record
                .before
                .iter()
                .zip(record.after.iter())
                .filter(|(before, after)| before != after)
                .count();
            let _ = write!(out, ", {moved} positions changed");
        }
        out.push('\n');
    }
    out
}

/// How many of a signal's candidates the human-readable trace lists.
///
/// The JSON form carries all of them; the text form is for reading, and a
/// dense signal contributes a hundred identifiers that nobody reads past the
/// top of.
const TRACE_CANDIDATES: usize = 10;

/// Explain every degradation on stderr, so a caller can tell a full answer
/// from a partial one. stdout stays a clean payload in both output modes.
///
/// Everything the search reported is printed, including a stage that was
/// never configured: a person who has set no endpoint still benefits from
/// being told that the thin result they are looking at was ranked by
/// keywords alone.
fn report_degradations(notes: &[crate::search::Note]) {
    for note in notes {
        eprintln!("kb search: {}", note.message);
    }
}

/// Search, over the derived index.
///
/// Takes no database path since T029: retrieval reads the index for every
/// corpus, and `--db` addresses the superseded schema that no signal consults.
fn run_search(args: &SearchArgs) -> i32 {
    let mode = match_mode_from_flag(args.r#match);
    let json = JsonOutput::from_flag(args.json);
    let request = crate::search::Request {
        query: args.query.clone(),
        corpus: args.corpus.clone(),
        limit: args.limit,
        mode,
        vector: !args.no_vector,
        min_similarity: args.min_similarity,
        dense_pooling: args.dense_pooling.to_pooling(),
        vector_candidates: args.vector_candidates,
        rerank: !args.no_rerank,
        project: args.project.clone(),
        context: args.context.clone(),
    };
    let index = match crate::index::Index::open(&crate::index::configured_index_path()) {
        Ok(index) => index,
        Err(e) => return print_error("search", json, &e.to_string()),
    };
    let outcome = match crate::search::resolve(&index, &request) {
        Ok(outcome) => outcome,
        // Only a signal failure gets the `--match` syntax hint: it is the one
        // error whose cause may be the expression the caller typed.
        Err(crate::search::SearchError::Signal(failure)) => {
            return print_error("search", json, &search_error_from(&failure, mode));
        }
        Err(e) => return print_error("search", json, &e.to_string()),
    };
    report_degradations(&outcome.notes);

    if args.explain {
        println!(
            "{}",
            render_response(
                "search",
                json,
                trace_json(&outcome.trace),
                trace_text(&outcome.trace)
            )
        );
        return 0;
    }

    let pairs: Vec<(String, String)> = outcome
        .hits
        .iter()
        .map(|hit| (hit.id.clone(), hit.title.clone()))
        .collect();
    let by_id: std::collections::BTreeMap<&str, Option<f32>> = outcome
        .hits
        .iter()
        .map(|hit| (hit.id.as_str(), hit.similarity))
        .collect();
    println!(
        "{}",
        render_response(
            "search",
            json,
            scored_summaries_json(&index, &pairs, &by_id, &args.corpus),
            // Text output carries no score: the number is for programmatic
            // consumers, and adding a column would change every existing
            // caller's parse.
            summaries_text(&pairs)
        )
    );
    0
}

/// Read one record, from the index that `kb search` ranked.
///
/// Reading the superseded database here would mean a record could be found
/// and then reported missing, which is the incoherence T029 exists to end.
fn run_get(id: &str, json: JsonOutput) -> i32 {
    let (store, index) = match open_store_and_index() {
        Ok(pair) => pair,
        Err(e) => return print_error("get", json, &e),
    };
    let Ok(row) = index.record(id) else {
        return print_error("get", json, &format!("no node with id: {id}"));
    };
    let body = match crate::write::read_raw(&store, &index, id) {
        Ok(text) => text,
        Err(e) => return print_error("get", json, &e.to_string()),
    };
    let tags = index.tags_of(id).unwrap_or_default();
    let domains = index.domains_of(id).unwrap_or_default();
    let payload = json!({
        "id": id,
        "title": row.title,
        "tags": tags,
        "document": body,
        "createdAt": row.created,
        "updatedAt": row.updated,
        "provenance": org_meta::provenance_json(&row, &domains),
    });
    let mut text = org_meta::render_metadata_around(id, &row.created, &row.updated, &body);
    text.push_str(&org_meta::provenance_text(&row, &domains));
    println!("{}", render_response("get", json, payload, text));
    0
}

/// Embed a node that has just been stored, replacing its prior vectors.
///
/// Never fatal, by operator decision: the `SessionEnd` hook writes through
/// this path, and a capture must not be lost because a model happens to be
/// restarting. A failure leaves the node stored and unembedded, which is
/// precisely the state `kb backfill` selects for, so the work is deferred
/// rather than dropped.
///
/// Silent when embedding is disabled. A user who has configured no endpoint
/// has chosen keyword-only operation, and a warning on every write would
/// train its reader to ignore warnings.
fn run_create(
    id: Option<&str>,
    tags: &[String],
    markdown: bool,
    allow_empty: bool,
    provenance_json: Option<&Path>,
    json: JsonOutput,
) -> i32 {
    if let Err(e) = check_stdin_source(std::io::stdin().is_terminal(), CREATE_BODY_HINT) {
        return print_error("create", json, &e.to_string());
    }
    let raw = match read_stdin() {
        Ok(r) => r,
        Err(code) => return code,
    };
    if let Err(e) = check_body_present(&raw, allow_empty, CREATE_BODY_HINT) {
        return print_error("create", json, &e.to_string());
    }
    let parsed = parse_input_body(&raw, markdown);
    let mut doc = match parsed {
        Ok(d) => d,
        Err(e) => return print_error("create", json, &e.to_string()),
    };
    // Strip any kb metadata drawer so it never enters the stored body.
    // create ignores the drawer's :ID:; the node id comes from --id or
    // a freshly minted UUID.
    let _ = org_meta::hydrate(&mut doc);
    merge_cli_tags(&mut doc, tags);

    let nid = id.map_or_else(|| uuid::Uuid::new_v4().to_string(), String::from);
    let (store, index) = match open_store_and_index() {
        Ok(pair) => pair,
        Err(e) => return print_error("create", json, &e),
    };
    // An identifier that already names a record is a conflict rather than a
    // rewrite. `kb update` is how a record is replaced, and creating over one
    // by accident is how a note is lost.
    if index.record(&nid).is_ok() {
        return print_error("create", json, &id_conflict_text(&nid));
    }
    let options = match read_provenance_options(&nid, &doc, provenance_json) {
        Ok(options) => options,
        Err(e) => return print_error("create", json, &e),
    };
    match crate::write::put_record(&store, &index, &nid, &doc, &options) {
        Ok(_) => {}
        Err(e) => return print_error("create", json, &e.to_string()),
    }
    warn_unresolved_links(&index, &nid);
    embed_written_record(&index, &nid);
    render_written("create", &index, &nid, json)
}

/// The [`crate::write::WriteOptions`] `kb create` writes with: kind inferred
/// from `document`'s tags by the same `conversation` → `session-transcript`
/// rule the queue worker applies (T017; see
/// [`crate::migrate::kind_for_tags`]), and provenance either none or
/// whatever `--provenance-json <file>` names. `create` has no existing
/// record to preserve anything from, so kind is always derived fresh rather
/// than read back from a prior write.
fn read_provenance_options(
    id: &str,
    document: &Document,
    provenance_json: Option<&Path>,
) -> Result<crate::write::WriteOptions, String> {
    let mut base = crate::write::WriteOptions::note(id).map_err(|e| e.to_string())?;
    base.kind = crate::migrate::kind_for_tags(&crate::storage::tag_names(document));
    apply_provenance_json(base, provenance_json)
}

/// The [`crate::write::WriteOptions`] `kb update` writes with (T016): the
/// existing record's kind, source and provenance, unchanged unless
/// `--provenance-json <file>` is given, in which case only the provenance is
/// replaced by the file's contents. This is what makes an ordinary
/// `kb update` of a captured transcript behave like `tags add`/`tags
/// rm`/`tags merge` rather than like `kb create`: it cannot turn a
/// `session-transcript` back into a `note` or silently drop the project an
/// earlier capture asserted.
///
/// # Errors
///
/// A message naming the record if it is unknown or its header cannot be
/// read, or naming the file if `--provenance-json` was given and the file
/// could not be read, parsed, or validated.
fn update_write_options(
    store: &impl crate::store::BlobStore,
    index: &crate::index::Index,
    id: &str,
    provenance_json: Option<&Path>,
) -> Result<crate::write::WriteOptions, String> {
    let base = crate::write::existing_options(store, index, id).map_err(|e| e.to_string())?;
    apply_provenance_json(base, provenance_json)
}

/// `base`, or `base` with its provenance replaced by `--provenance-json
/// <file>`'s contents when one is given. Shared by `kb create` and `kb
/// update` so the two commands' provenance files are parsed and validated by
/// exactly one code path.
///
/// The file, not stdin, is the source: the command's document body already
/// owns stdin, so a provenance value read from the same stream would have to
/// interleave with or follow the body in some format neither command
/// defines. A file path has no such conflict and is what T007's importer
/// passes.
fn apply_provenance_json(
    base: crate::write::WriteOptions,
    provenance_json: Option<&Path>,
) -> Result<crate::write::WriteOptions, String> {
    let Some(path) = provenance_json else {
        return Ok(base);
    };
    let bytes = std::fs::read(path)
        .map_err(|e| format!("reading provenance file {}: {e}", path.display()))?;
    let raw: crate::record::RawProvenance = serde_json::from_slice(&bytes)
        .map_err(|e| format!("provenance file {} is not usable: {e}", path.display()))?;
    let provenance = raw
        .validate()
        .map_err(|e| format!("provenance file {} is not usable: {e}", path.display()))?;
    Ok(crate::write::WriteOptions { provenance, ..base })
}

/// Open the store and the derived index a write goes through.
fn open_store_and_index() -> Result<(crate::store::GitBlobStore, crate::index::Index), String> {
    let store = crate::store::GitBlobStore::open_or_init(&crate::store::configured_store_path())
        .map_err(|e| e.to_string())?;
    let index = crate::index::Index::open(&crate::index::configured_index_path())
        .map_err(|e| e.to_string())?;
    Ok((store, index))
}

/// Print the record a write produced, read back from the index.
fn render_written(command: &str, index: &crate::index::Index, id: &str, json: JsonOutput) -> i32 {
    let row = match index.record(id) {
        Ok(row) => row,
        Err(e) => return print_error(command, json, &e.to_string()),
    };
    let text = index.record_text(id).unwrap_or_default();
    let tags = index.tags_of(id).unwrap_or_default();
    let payload = json!({
        "id": id,
        "title": row.title,
        "tags": tags,
        "document": text,
        "createdAt": row.created,
        "updatedAt": row.updated,
    });
    let listed: Vec<String> = tags.iter().map(|tag| format!(":{tag}")).collect();
    let line = format!("{id}  {}  {}\n", row.title, listed.join(""));
    println!("{}", render_response(command, json, payload, line));
    0
}

/// Embed what was just written, or say why not.
///
/// Never fatal, by operator decision: the capture path writes through here
/// and a capture must not be lost because a model happens to be restarting. A
/// failure leaves the record stored and unembedded, which is precisely the
/// state `kb embed` resumes from, so the work is deferred rather than
/// dropped.
///
/// Silent when embedding is disabled. Somebody who has configured no endpoint
/// has chosen keyword-only operation, and a warning on every write would
/// train its reader to ignore warnings.
fn embed_written_record(index: &crate::index::Index, id: &str) {
    let embedder = match crate::cli_embed::CliEmbedder::from_env() {
        Ok(embedder) => embedder,
        Err(e) if e.is_disabled() => return,
        Err(e) => {
            eprintln!("kb: node {id} was stored without an embedding: {e}");
            eprintln!("kb: run `kb embed` once that is corrected");
            return;
        }
    };
    if let Err(e) =
        crate::write::embed_record(index, &embedder, id, crate::cli_embed::DOCUMENT_CHUNK_CHARS)
    {
        eprintln!("kb: node {id} was stored without an embedding: {e}");
        eprintln!("kb: run `kb embed` once the endpoint is available");
    }
}

fn run_update(
    id: &str,
    tags: &[String],
    markdown: bool,
    allow_empty: bool,
    provenance_json: Option<&Path>,
    json: JsonOutput,
) -> i32 {
    if let Err(e) = check_stdin_source(std::io::stdin().is_terminal(), UPDATE_BODY_HINT) {
        return print_error("update", json, &e.to_string());
    }
    let raw = match read_stdin() {
        Ok(r) => r,
        Err(code) => return code,
    };
    if let Err(e) = check_body_present(&raw, allow_empty, UPDATE_BODY_HINT) {
        return print_error("update", json, &e.to_string());
    }
    let parsed = parse_input_body(&raw, markdown);
    let mut doc = match parsed {
        Ok(d) => d,
        Err(e) => return print_error("update", json, &e.to_string()),
    };
    // Hydrate the kb metadata drawer out of the body. On the round-trip
    // path the drawer's :ID: must name the node being updated; a mismatch
    // signals an imported file applied to the wrong node.
    if let Some(drawer_id) = org_meta::hydrate(&mut doc)
        && drawer_id != id
    {
        return print_error(
            "update",
            json,
            &format!("document :ID: {drawer_id} does not match target node id {id}"),
        );
    }
    merge_cli_tags(&mut doc, tags);

    let (store, index) = match open_store_and_index() {
        Ok(pair) => pair,
        Err(e) => return print_error("update", json, &e),
    };
    // Updating something that does not exist is a mistake worth naming:
    // `put_record` would happily create it, and a typo in an identifier would
    // silently mint a record instead of correcting one.
    if index.record(id).is_err() {
        return print_error("update", json, &format!("no node with id: {id}"));
    }
    // T016: absent `--provenance-json`, this keeps the existing record's
    // kind, source and provenance exactly as `tags add`/`tags rm`/`tags
    // merge` already do; with the flag, only the provenance is replaced by
    // the file's contents (`update_write_options`, shares
    // `apply_provenance_json` with `create`'s provenance parsing).
    let options = match update_write_options(&store, &index, id, provenance_json) {
        Ok(options) => options,
        Err(e) => return print_error("update", json, &e),
    };
    if let Err(e) = crate::write::put_record(&store, &index, id, &doc, &options) {
        return print_error("update", json, &e.to_string());
    }
    warn_unresolved_links(&index, id);
    embed_written_record(&index, id);
    render_written("update", &index, id, json)
}

/// Delete a record: unbind its name and forget its index rows.
///
/// The blobs stay addressable — deleting is unbinding rather than erasing —
/// so a deletion is recoverable by anyone who kept the address, and packing
/// is what eventually reclaims the space.
fn run_delete(id: &str) -> i32 {
    let (store, index) = match open_store_and_index() {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("kb: {e}");
            return 1;
        }
    };
    let name = match crate::store::RefName::new(&format!("{CORPUS_KB}/{id}")) {
        Ok(name) => name,
        Err(e) => {
            eprintln!("kb: {e}");
            return 1;
        }
    };
    match store.delete_ref(&name) {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("kb: no node with id: {id}");
            return 1;
        }
        Err(e) => {
            eprintln!("kb: delete failed: {e}");
            return 1;
        }
    }
    if let Err(e) = index.forget(id) {
        eprintln!("kb: {id} was unbound but its index rows remain: {e}");
        eprintln!("kb: run `kb fsck --repair` to reconcile");
        return 1;
    }
    println!("deleted {id}");
    0
}

/// What to tell a user whose record has no vector to search from.
///
/// An empty success would read as "nothing in the knowledge base resembles
/// this node", which is a different claim and a false one.
fn no_embedding_text(id: &str) -> String {
    format!("no stored embedding for node {id}; run `kb embed` to embed records that have none")
}

fn run_similar(id: &str, limit: usize, json: JsonOutput) -> i32 {
    let index = match crate::index::Index::open(&crate::index::configured_index_path()) {
        Ok(index) => index,
        Err(e) => return print_error("similar", json, &e.to_string()),
    };
    // Distinguish an unknown id from a known one that was never embedded:
    // the remedies differ, and `kb embed` cannot help with a typo.
    if index.record(id).is_err() {
        return print_error("similar", json, &format!("no node with id: {id}"));
    }
    // The configured model when there is one, else the model this record's
    // own vectors were computed under: `kb similar` is a read verb and must
    // keep working while the embedding daemon is restarting.
    let model = match crate::cli_embed::configured_model() {
        Some(model) => model,
        None => match index.model_of(id) {
            Ok(Some(model)) => model,
            Ok(None) => return print_error("similar", json, &no_embedding_text(id)),
            Err(e) => return print_error("similar", json, &e.to_string()),
        },
    };
    let neighbours = match index.similar_to(id, &model, limit) {
        Ok(Some(neighbours)) => neighbours,
        Ok(None) => return print_error("similar", json, &no_embedding_text(id)),
        Err(e) => return print_error("similar", json, &e.to_string()),
    };
    let pairs = named(&index, neighbours.into_iter().map(|(id, _)| id));
    println!(
        "{}",
        render_response(
            "similar",
            json,
            summaries_json(&pairs),
            summaries_text(&pairs)
        )
    );
    0
}

/// Pair each identifier with the name the index holds for it.
fn named(
    index: &crate::index::Index,
    ids: impl IntoIterator<Item = String>,
) -> Vec<(String, String)> {
    ids.into_iter()
        .map(|id| {
            let title = index.record(&id).map(|row| row.title).unwrap_or_default();
            (id, title)
        })
        .collect()
}

fn run_recent(limit: usize, json: JsonOutput) -> i32 {
    let index = match crate::index::Index::open(&crate::index::configured_index_path()) {
        Ok(index) => index,
        Err(e) => return print_error("recent", json, &e.to_string()),
    };
    let pairs = match index.recent(CORPUS_KB, limit) {
        Ok(rows) => rows,
        Err(e) => return print_error("recent", json, &e.to_string()),
    };
    println!(
        "{}",
        render_response(
            "recent",
            json,
            summaries_json(&pairs),
            summaries_text(&pairs)
        )
    );
    0
}

/// Serialize one row of the unified link graph. `link_type` is always
/// present; `target_id` and `target_slug` are NULL where they don't
/// apply or where a name-link is broken.
fn link_row_json(row: &crate::index::IndexLinkRow) -> serde_json::Value {
    json!({
        "source_id": row.source_id,
        "link_type": row.link_type,
        "target_id": row.target_id,
        "target_slug": row.target_slug,
    })
}

/// Render one row of the unified link graph for human display, surfacing
/// the `link_type` discriminator so id-links and name-links are
/// distinguishable in plain text output.
fn link_row_text(row: &crate::index::IndexLinkRow) -> String {
    match (row.link_type.as_str(), &row.target_id, &row.target_slug) {
        ("id", Some(tgt), _) => format!("{}  [id]  -> {}", row.source_id, tgt),
        ("name", Some(tgt), Some(slug)) => {
            format!("{}  [name]  [[{slug}]]  -> {tgt}", row.source_id)
        }
        ("name", None, Some(slug)) => {
            format!("{}  [name]  [[{slug}]]  -> (broken)", row.source_id)
        }
        _ => format!(
            "{}  [{}]  target_id={:?} target_slug={:?}",
            row.source_id, row.link_type, row.target_id, row.target_slug
        ),
    }
}

fn run_links(id: &str, json: JsonOutput) -> i32 {
    let index = match crate::index::Index::open(&crate::index::configured_index_path()) {
        Ok(index) => index,
        Err(e) => return print_error("links", json, &e.to_string()),
    };
    let nb = match index.links_of(id) {
        Ok(n) => n,
        Err(e) => return print_error("links", json, &e.to_string()),
    };
    let payload = json!({
        "outgoing": nb.outgoing.iter().map(link_row_json).collect::<Vec<_>>(),
        "incoming": nb.incoming.iter().map(link_row_json).collect::<Vec<_>>(),
    });
    let mut text_lines = vec!["outgoing:".to_string()];
    for row in &nb.outgoing {
        text_lines.push(format!("  {}", link_row_text(row)));
    }
    text_lines.push("incoming:".to_string());
    for row in &nb.incoming {
        text_lines.push(format!("  {}", link_row_text(row)));
    }
    println!(
        "{}",
        render_response("links", json, payload, text_lines.join("\n"))
    );
    0
}

fn run_orphans(json: JsonOutput) -> i32 {
    let index = match crate::index::Index::open(&crate::index::configured_index_path()) {
        Ok(index) => index,
        Err(e) => return print_error("orphans", json, &e.to_string()),
    };
    let pairs = match index.orphans(CORPUS_KB) {
        Ok(rows) => rows,
        Err(e) => return print_error("orphans", json, &e.to_string()),
    };
    println!(
        "{}",
        render_response(
            "orphans",
            json,
            summaries_json(&pairs),
            summaries_text(&pairs)
        )
    );
    0
}

fn run_hubs(limit: usize, json: JsonOutput) -> i32 {
    let index = match crate::index::Index::open(&crate::index::configured_index_path()) {
        Ok(index) => index,
        Err(e) => return print_error("hubs", json, &e.to_string()),
    };
    let hubs = match index.hubs(CORPUS_KB, limit) {
        Ok(h) => h,
        Err(e) => return print_error("hubs", json, &e.to_string()),
    };
    let payload = serde_json::Value::Array(
        hubs.iter()
            .map(|h| json!({"id": h.id, "title": h.title, "in_degree": h.in_degree}))
            .collect(),
    );
    let text = hubs
        .iter()
        .map(|h| format!("{}\t{}\t{}", h.in_degree, h.id, h.title))
        .collect::<Vec<_>>()
        .join("\n");
    println!("{}", render_response("hubs", json, payload, text));
    0
}

fn run_broken(json: JsonOutput) -> i32 {
    let index = match crate::index::Index::open(&crate::index::configured_index_path()) {
        Ok(index) => index,
        Err(e) => return print_error("broken", json, &e.to_string()),
    };
    let rows = match index.broken_links() {
        Ok(r) => r,
        Err(e) => return print_error("broken", json, &e.to_string()),
    };
    let payload = serde_json::Value::Array(rows.iter().map(link_row_json).collect::<Vec<_>>());
    let text = rows
        .iter()
        .map(link_row_text)
        .collect::<Vec<_>>()
        .join("\n");
    println!("{}", render_response("broken", json, payload, text));
    0
}

/// `kb projects`: every slug asserted by at least one record, its record
/// count and its newest record's date, plus how many records carry none
/// (`PLAN-20260923-project-identity` T006).
fn run_projects(json: JsonOutput) -> i32 {
    let index = match crate::index::Index::open(&crate::index::configured_index_path()) {
        Ok(index) => index,
        Err(e) => return print_error("projects", json, &e.to_string()),
    };
    let projects = match index.project_counts() {
        Ok(rows) => rows,
        Err(e) => return print_error("projects", json, &e.to_string()),
    };
    let no_project = match index.records_without_project() {
        Ok(count) => count,
        Err(e) => return print_error("projects", json, &e.to_string()),
    };
    let payload = json!({
        "projects": projects.iter().map(|p| json!({
            "project": p.project,
            "count": p.count,
            "newest": p.newest,
        })).collect::<Vec<_>>(),
        "noProject": no_project,
    });
    let mut lines: Vec<String> = projects
        .iter()
        .map(|p| {
            format!(
                "{}  {} {}  newest {}",
                p.project,
                p.count,
                plural(p.count, "record", "records"),
                p.newest
            )
        })
        .collect();
    lines.push(format!(
        "(no project)  {} {}",
        no_project,
        plural(no_project, "record", "records")
    ));
    println!(
        "{}",
        render_response("projects", json, payload, lines.join("\n"))
    );
    0
}

fn run_list_by_tag(tag: &str, json: JsonOutput) -> i32 {
    let index = match crate::index::Index::open(&crate::index::configured_index_path()) {
        Ok(index) => index,
        Err(e) => return print_error("list-by-tag", json, &e.to_string()),
    };
    let pairs = match index.records_tagged(&storage::normalize_tag(tag)) {
        Ok(rows) => rows,
        Err(e) => return print_error("list-by-tag", json, &e.to_string()),
    };
    println!(
        "{}",
        render_response(
            "list-by-tag",
            json,
            summaries_json(&pairs),
            summaries_text(&pairs)
        )
    );
    0
}

#[cfg(test)]
mod tests {
    use super::{format_elapsed, plural};
    use std::time::Duration;

    /// Rebuild time is the figure that decides whether the index stays
    /// disposable in practice, so it is rendered for a person rather than
    /// printed as a Debug struct.
    #[test]
    fn elapsed_time_reads_as_an_operator_expects() {
        assert_eq!(format_elapsed(Duration::from_millis(420)), "420ms");
        assert_eq!(format_elapsed(Duration::from_millis(2200)), "2.2s");
        assert_eq!(format_elapsed(Duration::from_secs(90)), "90.0s");
    }

    #[test]
    fn counts_of_one_read_singular() {
        assert_eq!(plural(1, "record", "records"), "record");
        assert_eq!(plural(0, "record", "records"), "records");
        assert_eq!(plural(2, "record", "records"), "records");
    }

    use super::{
        BACKFILL_CAPABILITY, BodyInputError, CREATE_BODY_HINT, CREATE_CAPABILITY,
        LIST_BY_TAG_CAPABILITY, SEARCH_CAPABILITY, SIMILAR_CAPABILITY, TAGS_CAPABILITY,
        UPDATE_BODY_HINT, UPDATE_CAPABILITY, check_body_present, check_stdin_source,
        id_conflict_text, match_mode_from_flag, merge_text, retrieval, search_error_from, storage,
        tags_text, trace_json, trace_text, unresolved_links_warning,
    };
    use tftio_lib::AgentCapability;

    /// The capability description as declared. `AgentCapability` exposes it
    /// through `summary()`; the struct fields are crate-private to `tftio_lib`.
    fn description(capability: &AgentCapability) -> &'static str {
        capability
            .summary()
            .unwrap_or_else(|| panic!("{} declares no description", capability.name()))
    }

    /// Every field `render_skill_md` puts in front of an agent, joined.
    ///
    /// A guard on the summary alone conflates two different claims: that a fact
    /// reaches the agent, and that it reaches the agent in one particular field.
    /// Only the first is the invariant. The emitted artifact renders the summary,
    /// both triggers, the output shape and the constraints, and an agent reads the
    /// whole body before invoking the verb -- so a fact stated under `## Constraints`
    /// is as reachable as one in the opening paragraph. The frontmatter description
    /// is length-bounded (Pi rejects a skill over 1024 characters outright), and
    /// that bound is what forces long operational detail out of the summary; a guard
    /// that insisted on the summary would make the bound unsatisfiable.
    fn agent_facing_text(capability: &AgentCapability) -> String {
        [
            capability.summary(),
            capability.when_to_use(),
            capability.when_not_to_use(),
            capability.output(),
            capability.constraints(),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ")
    }

    /// Every capability that writes or queries a tag must say how tags are
    /// normalized. Silence here is what let the namespace fragment
    /// unnoticed.
    #[test]
    fn the_tag_capabilities_document_normalization() {
        for capability in [
            &CREATE_CAPABILITY,
            &UPDATE_CAPABILITY,
            &LIST_BY_TAG_CAPABILITY,
            &TAGS_CAPABILITY,
        ] {
            let text = agent_facing_text(capability);
            assert!(
                text.contains("normaliz"),
                "{} does not mention normalization",
                capability.name()
            );
            assert!(
                text.contains("silent-critic") || text.contains("ci-cd"),
                "{} states the rule without a worked example",
                capability.name()
            );
        }
    }

    /// The rule's limit has to be stated wherever the rule is, or the
    /// description implies spelling does not matter - which is how the
    /// stranded `cicd` nodes came about.
    #[test]
    fn the_tag_capabilities_state_the_limit_of_normalization() {
        for capability in [
            &CREATE_CAPABILITY,
            &UPDATE_CAPABILITY,
            &LIST_BY_TAG_CAPABILITY,
            &TAGS_CAPABILITY,
        ] {
            assert!(
                agent_facing_text(capability).contains("CICD"),
                "{} does not say that an unbroken spelling is not split",
                capability.name()
            );
        }
    }

    /// An agent that believes ranking is deterministic will read an empty or
    /// thin result as evidence of absence. It is not: recall depends on
    /// whether an embedding endpoint answered, which the agent cannot see.
    /// A trace is the only account of why a result looks the way it does, so
    /// both renderings have to carry every part of it — including the signals
    /// that never ran, which are invisible in the results by definition.
    #[test]
    fn a_trace_renders_every_part_of_what_happened() {
        let trace = retrieval::Trace {
            signals: vec![retrieval::SignalTrace {
                name: "mail-lexical".into(),
                kind: retrieval::SignalKind::Lexical,
                corpus: "mail".into(),
                candidates: vec!["<one@x>".into()],
                error: None,
                elapsed: std::time::Duration::from_millis(3),
            }],
            skipped: vec![retrieval::Skipped {
                name: "kb-lexical".into(),
                reason: "corpus filter: mail".into(),
            }],
            fused: vec!["<one@x>".into()],
            rerank: Some(retrieval::RerankTrace {
                before: vec!["<one@x>".into(), "<two@x>".into()],
                after: vec!["<two@x>".into(), "<one@x>".into()],
                window: 2,
                error: None,
                elapsed: std::time::Duration::from_millis(1500),
            }),
        };

        let text = trace_text(&trace);
        assert!(
            text.contains("mail-lexical (lexical, mail): 1 candidates in 3.0ms"),
            "{text}"
        );
        assert!(
            text.contains("kb-lexical skipped: corpus filter: mail"),
            "{text}"
        );
        assert!(text.contains("fused: 1 candidates"), "{text}");
        assert!(text.contains("2 positions changed"), "{text}");

        let payload = trace_json(&trace);
        assert_eq!(payload["skipped"][0]["name"], "kb-lexical");
        assert_eq!(payload["signals"][0]["kind"], "lexical");
        assert_eq!(payload["signals"][0]["count"], 1);
        assert_eq!(payload["signals"][0]["elapsed_ms"], 3.0);
        assert_eq!(payload["rerank"]["window"], 2);
        assert_eq!(payload["rerank"]["elapsed_ms"], 1500.0);
    }

    /// A signal that failed contributed nothing, and both renderings have to
    /// say so — a zero-candidate signal that failed and one that simply found
    /// nothing are different facts.
    #[test]
    fn a_failed_signal_is_visible_in_both_renderings() {
        let trace = retrieval::Trace {
            signals: vec![retrieval::SignalTrace {
                name: "mail-lexical".into(),
                kind: retrieval::SignalKind::Lexical,
                corpus: "mail".into(),
                candidates: vec![],
                error: Some("mu not found on PATH".into()),
                elapsed: std::time::Duration::ZERO,
            }],
            skipped: vec![],
            fused: vec![],
            rerank: Some(retrieval::RerankTrace {
                before: vec![],
                after: vec![],
                window: 0,
                error: Some("connection refused".into()),
                elapsed: std::time::Duration::ZERO,
            }),
        };
        let text = trace_text(&trace);
        assert!(text.contains("failed: mu not found on PATH"), "{text}");
        assert!(
            text.contains("fused order stands: connection refused"),
            "{text}"
        );
        let payload = trace_json(&trace);
        assert_eq!(payload["signals"][0]["error"], "mu not found on PATH");
        assert_eq!(payload["rerank"]["error"], "connection refused");
    }

    #[test]
    fn the_search_capability_says_ranking_depends_on_the_endpoint() {
        let text = SEARCH_CAPABILITY
            .output()
            .unwrap_or_else(|| panic!("search declares no output"));
        assert!(
            text.contains("environment-dependent"),
            "search does not say ranking varies: {text}"
        );
        assert!(
            text.contains("--no-vector") && text.contains("null"),
            "search does not say what happens without an endpoint: {text}"
        );
    }

    /// The score is only useful to an agent that knows it is not a rank.
    #[test]
    fn the_search_capability_explains_the_similarity_it_reports() {
        let described = agent_facing_text(&SEARCH_CAPABILITY);
        assert!(
            described.contains("similarity"),
            "search does not mention the reported score"
        );
        assert!(
            described.contains("null"),
            "search does not distinguish an absent score from a zero one"
        );
        assert!(
            SEARCH_CAPABILITY
                .output()
                .is_some_and(|t| t.contains("not by similarity")),
            "search does not warn that results are ordered by fused rank"
        );
    }

    /// Every declared capability needs the three fields an agent reads to
    /// choose between verbs. A capability that says only what it does, and
    /// not when to reach for it or what comes back, is the drift this task
    /// exists to close.
    #[test]
    fn the_embedding_capabilities_are_fully_described() {
        for capability in [&SIMILAR_CAPABILITY, &BACKFILL_CAPABILITY] {
            let name = capability.name();
            assert!(
                !description(capability).is_empty(),
                "{name} declares no description"
            );
            assert!(
                capability.when_to_use().is_some(),
                "{name} does not say when to use it"
            );
            assert!(
                capability.when_not_to_use().is_some(),
                "{name} does not say when not to"
            );
            assert!(
                capability.output().is_some(),
                "{name} does not say what it returns"
            );
        }
    }

    /// `similar` reads stored vectors only. An agent told it might contact an
    /// endpoint would treat its failure as transient and retry, when the real
    /// remedy is a backfill.
    #[test]
    fn the_similar_capability_names_backfill_as_the_remedy_for_a_missing_vector() {
        let text = format!(
            "{} {}",
            description(&SIMILAR_CAPABILITY),
            SIMILAR_CAPABILITY.when_not_to_use().unwrap_or_default()
        );
        assert!(
            text.contains("backfill"),
            "similar does not name the remedy for an unembedded node: {text}"
        );
    }

    /// A verb that writes has to say so, and has to say that its exit status
    /// distinguishes a partial run from a clean one.
    #[test]
    fn the_backfill_capability_states_that_it_writes_and_how_it_reports_failure() {
        let text = format!(
            "{} {} {}",
            description(&BACKFILL_CAPABILITY),
            BACKFILL_CAPABILITY.when_not_to_use().unwrap_or_default(),
            BACKFILL_CAPABILITY.output().unwrap_or_default()
        );
        assert!(text.contains("writes"), "backfill does not say it writes");
        assert!(
            text.contains("nonzero"),
            "backfill does not say how a partial run is detected: {text}"
        );
        assert!(
            text.contains("--dry-run"),
            "backfill does not offer the read-only form"
        );
    }

    #[test]
    fn the_unresolved_link_warning_names_every_slug_and_the_rule() {
        let one = unresolved_links_warning(&["nosuch".to_string()]);
        assert!(one.contains("1 reference resolves to nothing"), "{one}");
        assert!(one.contains("[[nosuch]]"), "{one}");

        let many = unresolved_links_warning(&["alpha".to_string(), "beta".to_string()]);
        assert!(many.contains("2 references resolve to nothing"), "{many}");
        assert!(many.contains("[[alpha]], [[beta]]"), "{many}");

        // The rule matters as much as the slugs: both mistakes in the live
        // corpus are only fixable by someone who knows it.
        for text in [&one, &many] {
            assert!(text.contains("#+name:"), "{text}");
            assert!(text.contains("[[id:<uuid>]]"), "{text}");
            assert!(text.contains("The node was stored"), "{text}");
        }
    }

    /// A taken identifier is refused in the caller's terms.
    ///
    /// This used to be recognition of a `SQLite` extended error code, because
    /// the conflict surfaced as a primary-key violation on `nodes.id`. Writes
    /// go to the store now (T029) and a taken name is found by looking, so
    /// what survives is the property that mattered: the message names the id
    /// and the remedy, and neither the engine nor the physical schema reaches
    /// an agent-facing surface.
    #[test]
    fn a_taken_id_is_reported_as_a_conflict_naming_the_id() {
        let text = id_conflict_text("n1");
        assert!(text.contains("n1"), "{text}");
        assert!(text.contains("kb update n1"), "{text}");
        assert!(text.contains("omit --id"), "{text}");
        assert!(!text.contains("nodes.id"), "{text}");
        assert!(!text.to_lowercase().contains("sqlite"), "{text}");
    }

    #[test]
    fn tags_text_renders_count_then_tag() {
        let rendered = tags_text(&[("rust".into(), 12), ("ci-cd".into(), 1)]);
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].trim_start().starts_with("12  rust"), "{rendered}");
        assert!(lines[1].trim_start().starts_with("1  ci-cd"), "{rendered}");
    }

    #[test]
    fn merge_text_reports_the_no_op_case_distinctly() {
        let empty = storage::TagMerge {
            from: "absent".into(),
            to: "target".into(),
            rewritten: vec![],
        };
        assert!(merge_text(&empty).contains("nothing to merge"));

        let one = storage::TagMerge {
            from: "cicd".into(),
            to: "ci-cd".into(),
            rewritten: vec!["node-1".into()],
        };
        let text = merge_text(&one);
        assert!(
            text.contains("merged cicd into ci-cd across 1 node(s)"),
            "{text}"
        );
        assert!(text.contains("node-1"), "{text}");
    }

    #[test]
    fn the_match_flag_selects_fts5_and_its_absence_selects_keywords() {
        assert_eq!(match_mode_from_flag(true), storage::MatchMode::Fts5);
        assert_eq!(match_mode_from_flag(false), storage::MatchMode::Keywords);
    }

    #[test]
    fn an_fts5_syntax_error_under_match_is_attributed_to_the_flag() {
        let err = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(1),
            Some("fts5: syntax error near \"\"".into()),
        );
        let text = search_error_from(&err.to_string(), storage::MatchMode::Fts5);
        assert!(text.contains("fts5"), "diagnosis should survive: {text}");
        assert!(text.contains("--match"), "flag should be named: {text}");
    }

    /// A genuine database failure must not be dressed up as a query-syntax
    /// problem just because the caller passed `--match`.
    #[test]
    fn a_non_syntax_error_is_reported_verbatim_in_either_mode() {
        let err = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(1),
            Some("disk I/O error".into()),
        );
        for mode in [storage::MatchMode::Fts5, storage::MatchMode::Keywords] {
            let text = search_error_from(&err.to_string(), mode);
            assert!(!text.contains("--match"), "{mode:?}: {text}");
        }
    }

    /// In keyword mode every token is quoted before it reaches `SQLite`, so a
    /// syntax error is unreachable there and the hint would be misleading.
    #[test]
    fn the_match_hint_is_not_attached_in_keyword_mode() {
        let err = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(1),
            Some("fts5: syntax error near \"\"".into()),
        );
        let text = search_error_from(&err.to_string(), storage::MatchMode::Keywords);
        assert!(!text.contains("--match"), "{text}");
    }

    #[test]
    fn interactive_stdin_is_rejected() {
        let err = check_stdin_source(true, UPDATE_BODY_HINT).unwrap_err();
        assert_eq!(
            err,
            BodyInputError::Interactive {
                hint: UPDATE_BODY_HINT
            }
        );
        assert!(err.to_string().contains("stdin is a terminal"));
        assert!(err.to_string().contains("kb get <id> | kb update <id>"));
    }

    #[test]
    fn piped_stdin_is_accepted() {
        assert!(check_stdin_source(false, UPDATE_BODY_HINT).is_ok());
    }

    #[test]
    fn empty_body_is_rejected_by_default() {
        for raw in ["", "   ", "\n\n", " \t\r\n "] {
            let err = check_body_present(raw, false, UPDATE_BODY_HINT).unwrap_err();
            assert_eq!(
                err,
                BodyInputError::Empty {
                    hint: UPDATE_BODY_HINT
                },
                "expected {raw:?} to be refused as empty"
            );
            assert!(
                err.to_string()
                    .contains("refusing to store an empty document")
            );
        }
    }

    #[test]
    fn empty_body_is_accepted_with_allow_empty() {
        assert!(check_body_present("", true, UPDATE_BODY_HINT).is_ok());
    }

    #[test]
    fn non_blank_body_is_accepted() {
        assert!(check_body_present("#+title: x\n", false, CREATE_BODY_HINT).is_ok());
    }

    #[test]
    fn create_hint_does_not_advertise_the_update_round_trip() {
        let err = check_body_present("", false, CREATE_BODY_HINT).unwrap_err();
        assert!(err.to_string().contains("kb create"));
        assert!(!err.to_string().contains("kb update"));
    }
}
