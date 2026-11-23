---
plan_format_version: 2
plan_id: PLAN-20261123-kb-server-agent-surface
title: Make the kb server the one interface agents use
status: approved
created_at: 2026-11-23
updated_at: 2026-11-23
owner: jfb
source:
  type: manual
  url: null
  external_id: null
  imported_at: null
bug:
  summary: >
    Each machine holds its own kb store, the vector half of search has been off
    since 2026-09-16, and agents reach kb only through the local `kb` CLI, so no
    agent session can use one shared, fully searchable corpus.
  severity: medium
  affected_area: src/mcp.rs, src/mcp_http.rs, src/cli_main.rs (agent capabilities), agent-content mise-tasks/lib/build.bash, clanker personal domain configuration, the yadm home directory's LaunchAgents
  user_impact: >
    A session on one machine cannot find what a session on another captured; new
    records carry no vectors, so paraphrase search and `similar` fail for them; and
    the corpus on this laptop holds 24 records.
execution:
  requires_operator_approval_before_implementation: true
  requires_plan_updates_during_execution: true
  task_graph_status: approved
project: null
---

<!-- AGENT_GENERATED: t; AGENT_MODEL: claude-opus-5-5; AGENT_DATE: 2026-11-23 -->

# ADR: Make the kb server the one interface agents use

## Problem Statement

kb is a personal knowledge base: a content-addressed store (a bare git
repository of record blobs) that is authoritative, and a SQLite index derived
from it that is searched. Today every machine that runs `kb` has its own store
and index under `~/.local/share/kb`. Agents reach kb through the `kb-*` skills,
which tell the agent to run the local `kb` command, so an agent can only see
the corpus on the machine it runs on. On this laptop that corpus holds 24
records.

The vector half of `kb search` and the `similar` verb have not worked for new
records since 2026-09-16, when `KB_EMBEDDING_BASE_URL` and `KB_EMBEDDING_MODEL`
were removed from the shell environment (README, "Retrieval is keyword-only for
recent nodes").

kb already contains most of a server. `kb-mcp --http ADDR` serves MCP over
streamable HTTP behind a bearer token, and serves an ingest endpoint
(`POST /ingest`, `GET /ingest/ids`) into a capture queue on the same listener.
`kb queue drain` turns queued submissions into records. The `SessionEnd` hook
posts to the ingest endpoint when `KB_INGEST_URL` is set, and
`kb-import --post URL --reconcile` re-posts whatever the server lacks. What is
missing is an agent surface on the server that covers what agents do with kb,
the client wiring that points every harness at the server, and a host to run it
on. The host is the subject of a companion plan in the systems repository,
`PLAN-20261123-hemulen-kb-container`.

## Source Material

### Ticket

No ticket was provided. The operator directed the work in an agent session on
2026-11-23.

### Discussion Summary

The session began by comparing kb and mnene with Diderot (diderot.dev), a
self-hosted Go server from Bluesky that indexes chat, issues, docs and agent
transcripts into Postgres and Qdrant. The operator was interested in the server
shape: one kb server on hemulen that every agent uses.

Facts established in the session, from the repositories and the live network:

- `PLAN-20260822-hemulen-kb-serving` (systems repository) deployed `kb-mcp` on
  hemulen as a jfb user service on 2026-08-22. Its population task, T004, never
  ran. On 2026-09-24 the deployment was removed (systems commit `ea9ae36`, T010
  of the hemulen cargo-tools plan) because "the operator wants kb stood up
  later as a self-hosted server". The token `kb/mcp_token` was kept in
  hemulen's sops document for the successor.
- hemulen still serves `text-embedding-qwen3-embedding-0.6b` (Q8_0, 1024
  dimensions) through llama.cpp on Vulkan at `10.0.1.2:8080`, and a reranker at
  `:8081` (`nix/hosts/hemulen/services/local-inference.nix`). Measured from this
  laptop on 2026-11-23: one query embeds in 0.21s; ~1000-character passages
  embed at about 20 per second.
- The MCP surface has four tools: `search`, `context`, `get`, `put`
  (`src/mcp.rs`). The agent capabilities declared in `src/cli_main.rs` number
  twenty-two and include `recent`, `list-by-tag`, `tags` and tag edits,
  `update`, `delete`, `links`, `hubs`, `orphans`, `broken`, `projects`,
  `similar`, `backfill` and the `prompt` verbs. After this plan the MCP surface
  has twenty-one tools: the four existing ones, twelve read tools and five
  write tools (Decision 1). The difference from twenty-two is that `create` is
  served by the existing `put` tool and `backfill` stays a CLI-only operator
  verb.
- The MCP `put` tool writes a record but does not embed it; `kb queue drain`
  embeds what it writes when an embedder is configured (`src/ingest.rs`
  `process`).

Operator decisions taken in the session, each on 2026-11-23:

1. Agents reach kb through the server's MCP tools ("let's fix the MCP"), not
   through a remote mode of the CLI or through ssh.
2. One shared bearer token for every client is acceptable.
3. Work content is not a concern: no work context is in use.
4. Embeddings are computed on the server. Client-side precomputed vectors were
   discussed and are deferred (see Alternatives Considered).
5. The server needs scheduled reconciliation for captures lost while it was
   unreachable.
6. Clients reach the server over the tailnet.
7. The corpus starts fresh. Nothing is migrated from any existing store.
8. The host is an nspawn container on hemulen, not a native user service and
   not an LXC guest on snufkin. Mail is out of scope.

## Context

The server today. `src/bin/kb-mcp.rs` serves stdio by default and HTTP only
when `--http ADDR` names an address. `src/mcp_http.rs` refuses to start without
`KB_MCP_TOKEN`, checks the token before routing, and rejects a `Host` header
not in `KB_MCP_ALLOWED_HOSTS` or the bound address. Ingest routes are
`/ingest` (POST, accept a submission) and `/ingest/ids` (GET, list served ids).
`--store`, `--index` and `--queue` take the same environment variables the CLI
reads (`KB_STORE_PATH`, `KB_INDEX_PATH`, `KB_QUEUE_PATH`).

The tool surface. `src/mcp.rs` defines a protocol-independent
`ToolSurface` trait (`REPO_INVARIANTS.md` ENG-010). Handlers call the same
domain functions the CLI does. The surface is tiered by token cost: `search`
returns identifiers and titles, `context` a hit's neighbourhood, `get` one full
document, `put` one write. `DEFAULT_SEARCH_RESULTS` is 10.

The ingest submission. `Submission { id, corpus, document, provenance }`
(`src/ingest.rs`). The document is rendered on the client, because rendering
drops tool calls and tool output, which is the redaction (Decision Log entry of
2026-08-21 in the content-addressed-store plan). The worker parses the
document, skips an unchanged re-post, writes the record, and embeds it if an
embedder is configured.

How vectors are keyed. The index's `embeddings` table has primary key
`(stream_hash, span_start, span_len, model)` (`src/index.rs`). A vector is
filed under the model name string, and `embed_record` (`src/write.rs`) chunks
each passage with `embed_text::chunk_span` before embedding.

Tag edits change the stream hash. `add_tags`, `remove_tags` and `merge_tag`
(`src/write.rs`) write the tag into the document and store it again through
`put_record`. `put_record` derives the stream from the regenerated document,
which still contains the `#+filetags:` line or heading tags, so a tag edit
produces a new `stream_hash`. The record's existing vectors are filed under the
old hash, so after a tag edit the record has no vectors until it is embedded
again. This was read from the code on 2026-11-23 and not yet confirmed by a
test.

Concurrency. The index opens with `PRAGMA journal_mode = WAL`
(`src/index.rs`). No busy timeout and no store-level write lock were found in
`src/index.rs` or `src/store.rs` on 2026-11-23. On the server, `kb-mcp` write
tools and the `kb queue drain` timer will write to the same store and index.

The clients.

- agent-content builds a `tftio` plugin per harness and registers mnene's stdio
  MCP server in each (`mise-tasks/lib/build.bash`, `mnene_mcp_server_json`, and
  `reconcile_opencode_mcp` in `mise-tasks/lib/package_install.bash` for
  OpenCode). It also installs the `kb-*` skills emitted by
  `kb meta agent emit-skills` into every Claude and Codex root
  (`mise run install-kb-skills`), and `mise run doctor` diffs them.
- clanker removes every inherited environment variable not listed in
  `defaults.passthrough_environment` or the domain's `credentials`, so a token
  reaches a session only through that configuration
  (`~/.config/clanker/config.toml`, `~/.config/clanker/local.toml`).
- The `SessionEnd` hook (`scripts/session-end-kb.sh`) posts with
  `--post "$KB_INGEST_URL"` when that variable is set; `kb-import` reads the
  token from `KB_INGEST_TOKEN`. A failed post is not retried or saved for later.
- The laptop's launchd agents are managed in the yadm home directory
  (`~/Library/LaunchAgents`, `io.tft.*`), not in the systems repository
  (`systems/docs/macos.md`).

The network. The tailnet's subnet router `snufkin-subnet` advertises
`10.0.0.0/16` and is primary for it. This laptop accepts subnet routes
(`RouteAll: true`). The tailnet's only split-DNS route is
`h.tft.io -> 10.0.1.53`, so `*.home.arpa` names do not resolve off the LAN and
clients address the server as `https://kb.h.tft.io`, through hemulen's Caddy.

## Constraints

- `REPO_INVARIANTS.md` binds: the store is authoritative and the index is
  derived (ST-001, ST-002); tool handlers call the CLI's domain functions and
  nothing in `src/mcp.rs` knows about the protocol (ENG-010); the coverage floor
  holds (RS-007).
- No plaintext secret lands in any repository. The token reaches clients
  through clanker's secret-backed environment and reaches the server through
  sops (companion plan).
- Redaction stays on the client: the server accepts rendered documents, not raw
  transcripts.
- Adding a variable to clanker's environment filter changes the operator's
  security boundary; the operator approved adding the kb token on 2026-11-23.
- A release, a push, a commit, and any change to installed harness
  configuration need operator approval at the time.
- agent-content documents per-harness invocation separately and never assumes
  one shared syntax. Each harness's remote-MCP header and environment syntax is
  confirmed against that harness's current documentation when the task runs,
  not written from memory.

## Non-Goals

- The host. Container, address, DNS, Caddy route, sops wiring, timers, and
  backups are `PLAN-20261123-hemulen-kb-container` in the systems repository.
- The mail corpus.
- Migrating any existing store. The operator accepted losing the one under
  hemulen's `/home/jfb/.local/share/kb` on 2026-11-23, and the companion plan
  removes it. Past transcripts on clients are not back-filled either; see T007.
- A remote mode for the `kb` CLI. Operator maintenance verbs (`reindex`,
  `fsck`, `embed`, `backfill`, `queue`) run on the server.
- Precomputed client-side vectors. Deferred; see Alternatives Considered.
- Per-client tokens, read-only tokens, or access control finer than the one
  shared bearer token.
- Codex session capture. Codex has no `SessionEnd` event, which is unchanged.

## Decision

1. The MCP surface covers what agents do with kb. Add read tools `recent`,
   `list_by_tag`, `tags`, `links`, `hubs`, `orphans`, `broken`, `projects`,
   `similar`, `prompt_list`, `prompt_show`, `prompt_render`, and write tools
   `update`, `delete`, `tags_add`, `tags_rm`, `tags_merge`. Each calls the
   domain function its CLI verb calls. Tool descriptions carry the guidance the
   capability text carries today (for example, read `tags` before tagging).
   `backfill`, `reindex`, `fsck`, `embed` and `queue` stay CLI-only operator
   verbs.
2. Every server write embeds. `put`, `update`, `tags_add`, `tags_rm` and
   `tags_merge` embed each record they write when an embedder is configured,
   as `kb queue drain` already does. The tag tools are included because a tag
   edit gives the record a new stream hash and so leaves it without vectors
   (see Context). `delete` writes no record and embeds nothing. A failure to
   embed is reported in the tool result and does not undo the write. A
   scheduled `kb embed` on the host (companion plan) catches up anything
   missed.
3. Concurrent writers are proven safe before deployment, by a test that
   runs MCP writes and `kb queue drain` against one store and index at once.
   Whatever guard the test shows is needed (a SQLite busy timeout, a store
   write lock) is added.
4. Agents use MCP, not the `kb-*` skills. agent-content registers kb's HTTP
   MCP server in every generated harness package, reading the token from
   `KB_MCP_TOKEN`, and stops installing the CLI-based `kb-*` skills. kb keeps
   its `AgentSurfaceSpec` for the CLI.
5. clanker passes the token and the ingest URL into personal-context
   sessions: `KB_MCP_TOKEN` and `KB_INGEST_TOKEN` (one secret under two names)
   and `KB_INGEST_URL=https://kb.h.tft.io`. The value is the server's base
   URL: `IngestClient` (`python/kb_import/ingest.py`) strips a trailing slash
   and appends `/ingest` and `/ingest/ids` itself, and the `SessionEnd` hook
   passes `KB_INGEST_URL` to `--post` unchanged. A value ending in `/ingest`
   would post to `/ingest/ingest`, which the server routes to MCP, not to
   capture.
6. The laptop reconciles on a schedule. A launchd agent in the yadm home
   directory runs `kb-import --post <url> --reconcile` periodically.
7. The client URL is `https://kb.h.tft.io`. MCP clients are configured
   with `https://kb.h.tft.io/mcp`; the server routes every authenticated path
   other than `/ingest` and `/ingest/ids` to MCP (`src/mcp_http.rs`
   `ingest_route`), so `/mcp` is a convention, not a route. Capture clients
   are configured with the base URL only, and `kb-import` adds the `/ingest`
   paths (Decision 5).

## Alternatives Considered

- A remote mode for the `kb` CLI, so the existing `kb-*` skills keep
  working unchanged. Rejected by the operator in favour of MCP: it would make
  the CLI a second network client and keep two agent interfaces in sync.
- `ssh hemulen kb ...` from agent sessions. Rejected: clanker's
  environment filter removes `SSH_AUTH_SOCK` from sessions, and letting it
  through widens the security boundary far more than one bearer token does.
- Precomputed vectors from clients. A submission would carry vectors
  computed on the client, so a bulk load could use a faster machine's GPU.
  Deferred, not rejected. The server must embed queries with the same model in
  any case, because an agent's `search` sends text. Vectors are keyed by
  `(stream_hash, span_start, span_len, model)`, so the client would have to
  reproduce the server's passage extraction and chunking exactly, which today
  lives only in Rust (`embed_text::chunk_span`, `index.passages`) while the
  posting client is Python. Vectors from a different runtime or quantization
  filed under the same model string would mix silently. If it is ever built,
  each submitted vector set must name the model file, the quantization and the
  chunker version, and the server must reject any set whose names differ from
  its own. Measured server throughput (about 20 passages a second) does not
  call for it now.
- One token per client, or a read-only token. Rejected by the operator for
  now; one shared token.

## Consequences

- Agents on any machine on the tailnet see one corpus with working vector
  search.
- The `kb-*` skills leave agent sessions. `mise run install-kb-skills` and the
  kb section of `mise run doctor` in agent-content change or go away, and the
  agent-content `CLAUDE.md` sections describing them must be rewritten.
- Search and capture depend on hemulen and on the tailnet. While the server is
  unreachable, MCP calls fail and captures wait for the next reconcile.
- The shared token can both read and write. Anyone holding it can change the
  corpus.
- `kb-import` on each client must stay version-matched to `kb` on the server.

## Operator Guidance Log

Append-only. Add a dated entry whenever an agent needs operator guidance or operator
guidance changes the plan. A new plan legitimately has no entries yet; this note is
what satisfies the requirement until the first one is written.

### 2026-11-23 — jfb (operator)

Question or decision needed: Whether to serve kb from one server, and the shape
of the server and its clients.

Decision or guidance received: Fix the MCP surface and use it as the agent
interface; one shared token; no work contexts in use; embed on the server, with
precomputed vectors explained and not yet chosen; scheduled reconciliation is
needed; clients reach the server over the tailnet; start the corpus fresh; run
the server in an nspawn container on hemulen; leave mail out.

Impact on plan: These are the decisions recorded in the Discussion Summary and
the Decision section.

### 2026-11-23 — jfb (operator), later the same session

Question or decision needed: Whether the previous corpus on hemulen must be
preserved.

Decision or guidance received: "I am comfortable losing the existing repo, to
be honest; it doesn't have enough content to be useful."

Impact on plan: The companion plan removes hemulen's old corpus (its T008).
Checking what "start fresh" means for clients showed that
`kb-import --reconcile` posts every transcript under the harness roots that the
server lacks, so T007 now limits the scheduled reconcile with `--since` set to
the go-live date.

### 2026-11-23 — jfb (operator), plan review

Question or decision needed: Seven review findings on this plan.

Decision or guidance received: Embed after tag edits; add `kb embed` to the
concurrency test; leave the cross-plan dependencies unrecorded, since the two
plans run in parallel and are reconciled when both finish; drop the
dependencies that are not needed; fix a typo in T007; state how the
twenty-one MCP tools relate to the twenty-two CLI capabilities.

Impact on plan: Decision 2, T002 and T003 changed; T006 now has no
dependencies, T005 depends on T006, T007 on T004 and T006, and T008 on T005
and T007. T002 still depends on T001 because both add tools to `KbTools` in
`src/mcp.rs`, and doing them in parallel would cause conflicting edits.

### 2026-11-23 — jfb (operator), ingest URL

Question or decision needed: Decision 5 set `KB_INGEST_URL` to the `/ingest`
path and left the correct form to T006.

Decision or guidance received: Settle it in the plan from the code.

Impact on plan: `python/kb_import/ingest.py` `IngestClient` appends `/ingest`
and `/ingest/ids` to a base URL, so Decision 5, T006 and T007 now use
`https://kb.h.tft.io`, and T006 no longer has to determine the form. Decision
7 now notes that the server routes every non-ingest path to MCP, so `/mcp` is
a convention.

### 2026-11-23 — jfb (operator), execution

Question or decision needed: Approval to implement, and how commits, branches
and out-of-repository work are handled.

Decision or guidance received: Execute the plan with one Sonnet-class subagent
per task. Subagents may make one signed commit per task on
`feat/mcp-agent-surface`, branched from `docs/plan-kb-server-agent-surface`;
nothing is pushed. T006 starts in parallel with T001, but its agent prepares
the clanker and secret changes and applies nothing until the operator approves
them.

Impact on plan: Plan and task graph status set to approved. T004's merge and
release, and every change to clanker, agent-content and the home directory,
still need approval at the time.

## Decision Log

Append-only. Decisions affecting architecture, behavior, sequencing, or tradeoffs, as
distinct from operational chatter. A new plan legitimately has no entries yet; this
note is what satisfies the requirement until the first one is written.

### 2026-11-23 — D001: agents use MCP; the `kb-*` CLI skills are not installed for agents

The server's MCP tools become the only agent interface. This keeps one
interface for every harness and avoids putting SSH or a second network client
in agent sessions. Reversible: the skills can be emitted and installed again.

### 2026-11-23 — D002: embeddings are computed on the server

Queries must be embedded on the server whatever happens to documents, and the
measured throughput of hemulen's embedding server is enough for this corpus.
Client-side precomputed vectors are deferred with the guard conditions recorded
in Alternatives Considered.

### 2026-11-23 — D003: tag edits embed the record they rewrite

A tag edit stores the document again with the tag written into it, which
produces a new `stream_hash`; the record's vectors stay filed under the old
hash, so without re-embedding, a tag edit removes the record from vector search
until the scheduled `kb embed` runs. The MCP tag tools therefore embed after
writing, as `put` and `update` do. T002 confirms the hash change by test before
relying on it.

### 2026-11-23 — D004: `similar` resolves through `search::similar`

The CLI kept the choice of embedding model and the three failure cases (unknown
id, no stored vector, unreadable index) inside `run_similar`, so the MCP tool
could not call it. T001 moved that logic into `search::similar` with a typed
`SimilarError`, and `kb similar` now calls it, so the two surfaces cannot
disagree about when `similar` fails. The no-vector message names `kb embed`
and the word backfill together, because the CLI text and the capability text
had named different remedies. Reversible: the function is a straight move.

### 2026-11-23 — D005: the tag-edit hash change is confirmed; the embedder is injected

T002 confirmed by test that add_tags changes the record's stream_hash, so D003
stands unchanged and the tag tools embed. The embedder reaches the tool surface
as a value (`KbTools::with_embedder`), built once by `kb-mcp` from the
environment, rather than being read inside `src/mcp.rs`. That keeps the surface
independent of the environment (ENG-010) and lets tests hand it a mock
endpoint. An unset endpoint is silent, a half-configured one is named on stderr,
and the server starts in both cases. `CliEmbedder` also changed to shut its
runtime down in the background, because dropping it inside the server's runtime
panicked at shutdown.

# Task Graph

<!-- TASK_GRAPH:BEGIN -->
```yaml
tasks:
  - id: T001
    title: Add the read tools to the MCP surface
    status: done
    depends_on: []
    owner: null
    description: >
      Add MCP tools recent, list_by_tag, tags, links, hubs, orphans, broken,
      projects, similar, prompt_list, prompt_show and prompt_render to KbTools
      in src/mcp.rs. Each handler calls the domain function its CLI verb calls
      and returns compact JSON in the existing tiered style (identifiers and
      titles unless the tool's purpose is to return a body). Move each
      capability's agent guidance from src/cli_main.rs into the tool
      description, shortened for an agent's context budget.
    work_items:
      - Add one ToolSpec and one handler per tool, with a JSON schema whose required fields match the CLI verb's required arguments.
      - Give list-style tools a limit argument with a default no larger than the CLI's.
      - Report the vector-less case for similar the way the CLI does, naming backfill as the remedy.
      - Add ToolSurface::call tests per tool, and one protocol test confirming tools/list over stdio and HTTP returns every tool.
    invariants:
      - No handler reimplements retrieval, tag normalization, or link resolution; each calls the CLI's domain function (ENG-010).
      - src/mcp.rs does not import anything protocol-specific.
      - The four existing tools keep their names, schemas and output shapes.
      - The coverage floor in REPO_INVARIANTS.md RS-007 holds.
    acceptance_checks:
      - "`mise run test` passes."
      - "`mise run ci` passes."
      - An MCP tools/list against `kb-mcp` (stdio) lists the four existing tools and the twelve new ones.
    completion_evidence: |
      Files touched: src/mcp.rs (twelve handlers at lines 366-505, specs in
      read_specs, graph_specs and neighbour_specs at 512-700, DEFAULT_LIST_RESULTS
      at line 80, dispatch in call); src/search.rs (new SimilarError and
      similar, lines 455-518); src/cli_main.rs (run_similar at line 3364 now
      calls search::similar; no_embedding_text and named removed);
      tests/mcp_read_tools.rs (new, 17 ToolSurface::call tests); tests/mcp_stdio.rs
      and tests/mcp_http.rs (the tool-list tests now assert all sixteen names in
      order).
      Commands: `mise run test` passed, 882 run, 882 passed, 9 skipped. `mise run
      ci`: fmt, taplo, shellcheck, typos, clippy, nextest, doc, machete, package,
      lockfiles, ruff, pyrefly, pytest and kb-import package passed; check:coverage
      ran and reported 94.25% line coverage against the 93% floor, without error;
      `cargo audit` and `cargo deny` failed on RUSTSEC-2026-0592 (url 2.5.8,
      stack overflow in Url::origin on nested blob: URLs). Cargo.lock is unchanged
      by this task, and `cargo update -p url` locked 0 packages, so no fixed url
      was available to the resolver on 2026-11-23. The failure is independent of
      this change and needs a dependency bump once url 2.5.9 resolves.
      tools/list over stdio: with `kb-mcp` run under a scratch HOME and
      KB_STORE_PATH and KB_INDEX_PATH set, the initialize, notifications/initialized
      and tools/list lines piped to the binary returned 16 tools: search, context,
      get, put, recent, list_by_tag, tags, links, hubs, orphans, broken, projects,
      similar, prompt_list, prompt_show, prompt_render. The same list over HTTP is
      asserted by tests/mcp_http.rs the_same_sixteen_tools_are_offered_over_http.
      Deviations: (1) similar needed a domain function, because the CLI kept the
      model choice and the no-vector error inside run_similar; it moved to
      search::similar and both callers use it (Decision Log D004). (2) The CLI's
      no-vector message named `kb embed`, while the capability text said backfill;
      the shared message now names both. (3) links fails for an unknown id, as
      context does, where the CLI returns empty lists. (4) List tools without a
      CLI limit (list_by_tag, tags, orphans, broken) take a limit defaulting to
      10 and report total, so a cut list says it was cut; recent defaults to 10
      against the CLI's 50, hubs to 10 against 20.

  - id: T002
    title: Add the write tools and embed every server write
    status: done
    depends_on: [T001]
    owner: null
    description: >
      Add MCP tools update, delete, tags_add, tags_rm and tags_merge, each
      calling the domain function its CLI verb calls, with the same
      preservation rules (update and tag edits keep kind, source and
      provenance). Make put, update, tags_add, tags_rm and tags_merge embed
      each record they wrote when an embedder is configured from the
      environment, using the same embed_record path and chunk size kb queue
      drain uses. Tag edits are included because they give the record a new
      stream hash, which leaves it without vectors. tags_merge embeds every
      record it rewrote. An embedding failure is reported in the tool result
      and leaves the write in place.
    work_items:
      - Add the five write tools with schemas and handlers.
      - Wire an optional embedder into KbTools, built from the environment once at server start.
      - Embed after put, update, tags_add, tags_rm and tags_merge; report embedded, skipped, or the failure reason in the result, per record for tags_merge.
      - Add a test confirming that a tag edit changes the record's stream_hash, so the reason for embedding after tag edits is checked rather than assumed. If the test shows the hash does not change, drop embedding from the tag tools and record that in the Decision Log.
      - Tests for each write tool, for embed-on-write with a test embedder, and for an embedding failure that leaves the record stored.
    invariants:
      - update and tag edits preserve the record's kind, source and provenance exactly as the CLI does (src/write.rs existing_options).
      - delete keeps its audit record.
      - A write never fails because embedding failed.
      - No server process requires an embedder to start.
    acceptance_checks:
      - "`mise run test` passes."
      - With KB_EMBEDDING_BASE_URL and KB_EMBEDDING_MODEL pointing at a test endpoint, a put over MCP produces embeddings rows for the record's passages.
      - With no embedder configured, put over MCP succeeds and reports that nothing was embedded.
      - With a test embedder configured, a tags_add over MCP leaves the record with embeddings rows under its new stream_hash.
    completion_evidence: |
      Files touched: src/mcp.rs (with_embedder at line 130, embedding_report at 141,
      handlers update, delete, edit_tags and tags_merge at 379-475, write_specs at
      857, dispatch at 1041); src/write.rs (embed_written at 462, DeleteError at 472,
      delete_record at 503); src/cli_embed.rs (CliEmbedder drops its runtime in the
      background, Drop at 270); src/bin/kb-mcp.rs (embedder_from_env at 85, built
      once at start); src/cli_main.rs (run_delete at 3326 and embed_written_record at
      3241 now call the shared functions); src/ingest.rs (process at 453 calls
      embed_written); tests/mcp_write_tools.rs (new, 14 tests); tests/mcp_stdio.rs,
      tests/mcp_http.rs and tests/mcp_read_tools.rs (tool lists now assert all
      twenty-one names in order, and the new required fields).
      Commands: the stream-hash test (a_tag_edit_gives_the_record_a_new_stream_hash)
      was written first and passed: the hash before and after add_tags differ, so D003
      stands. `mise run test` passed, 897 run, 897 passed, 9 skipped. `mise run ci`:
      formatting, taplo, shellcheck, typos, clippy, nextest, doc, machete, package,
      lockfiles and the Python checks passed; check:coverage reported 94.28% line
      coverage against the 93% floor; `cargo audit` and `cargo deny` failed on
      RUSTSEC-2026-0592 (url 2.5.8), the known failure that is independent of this
      change. Cargo.toml and Cargo.lock are unchanged.
      Live check: kb-mcp over stdio under a scratch HOME with KB_STORE_PATH and
      KB_INDEX_PATH set. tools/list returned 21 tools. With no embedder, put returned
      embedding.status "disabled" and the record was stored; tags_add did the same.
      With a local fake endpoint (a Python server answering /v1/embeddings) and
      KB_EMBEDDING_BASE_URL and KB_EMBEDDING_MODEL set, put returned one record with
      status "embedded", embedded 1, and so did tags_add; the server exited 0 when
      stdin closed. With KB_EMBEDDING_BASE_URL set and no model, the server started,
      named the problem on stderr and stored writes unembedded.
      Result shape: every writing tool returns embedding as {status, records}, where
      status is "disabled" (no endpoint), "enabled" (records lists one entry per
      record with status "embedded" and embedded and skipped counts, or "failed" with
      the reason), or "unchanged" (a tag edit that rewrote nothing, so nothing was
      embedded). tags_merge lists every rewritten record.
      Deviations: (1) The chunk size for written records was a constant named at each
      call site, not a selection; embed_written now names it once and the CLI, the
      queue worker and the MCP tools all call it. (2) delete had no domain function,
      only CLI code; it moved to write::delete_record with a typed DeleteError, and
      `kb delete` calls it with unchanged messages. The store keeps no audit log:
      "keeps its audit record" holds as the store behaves today, where delete unbinds
      the name and the blobs stay addressable by hash. (3) A real defect found by the
      live check: dropping a CliEmbedder inside the server's async runtime panicked at
      shutdown. CliEmbedder now shuts its runtime down in the background, and a test
      drops a surface holding an embedder inside a runtime. (4) A tag edit that
      changed nothing reports "unchanged" and does not embed. (5) update does not take
      provenance_json; the CLI's flag is not in this task. (6) tags_add, tags_rm and
      update take a tags argument as arrays of strings; tags_add and tags_rm require
      at least one.
      Open concern for T003: in a first pipelined session, two tool calls sent back to
      back against an empty index raced on creating it (the second saw schema version
      0). Calls are dispatched concurrently, so this is the kind of fault T003 should
      cover.

  - id: T003
    title: Prove concurrent server writers are safe
    status: not_started
    depends_on: [T002]
    owner: null
    description: >
      On the server, kb-mcp write tools, the kb queue drain timer, and a
      scheduled kb embed write to one store and one index at the same time.
      Write a test that runs MCP writes, queue drains and kb embed
      concurrently against one store and index, then checks that every write
      is present, the index matches the store (kb fsck reports no drift), no
      embedding row is lost or duplicated, and no call failed with a lock
      error. Add whatever guard the test shows is missing, such as a SQLite
      busy timeout on every index connection or a store write lock.
    work_items:
      - Write the concurrency test, running the three writers (MCP write tools, kb queue drain, kb embed) as separate processes or threads against one store and index.
      - Record which failures appear without a guard, if any.
      - Add the minimal guard and record it in REPO_INVARIANTS.md if it is a new rule.
    invariants:
      - The index stays derivable from the store (ST-002).
      - No write is silently dropped; a write that cannot proceed fails with a reason.
    acceptance_checks:
      - The concurrency test passes repeatedly (at least 20 consecutive runs).
      - "`kb fsck` against the test's store and index reports no drift."
    completion_evidence: null

  - id: T004
    title: Release kb with the new surface
    status: not_started
    depends_on: [T003]
    owner: null
    description: >
      Update README.md's agent-surface and retrieval sections and CHANGELOG
      material for the new tools and embed-on-write, then release through the
      repository's release-plz flow, so the systems flake input and kb-import
      on clients can pin the released version. Merging and releasing need
      operator approval at the time.
    work_items:
      - Update README.md, including the keyword-only section, which will no longer describe the server.
      - Open the pull request and, on approval, merge it and let release-plz cut the release.
    invariants:
      - kb and kb-import carry the same version (README, "Installing kb-import").
    acceptance_checks:
      - A tagged release exists and `kb meta version` and `kb-import --version` built from it report the same version.
    completion_evidence: null

  - id: T005
    title: Register the kb server in every agent-content harness package
    status: not_started
    depends_on: [T004, T006]
    owner: null
    description: >
      In agent-content, register kb's HTTP MCP server at
      https://kb.h.tft.io/mcp in the generated Claude, Codex, OpenCode and Pi
      packages beside mnene, with the bearer token read from KB_MCP_TOKEN in
      each harness's own syntax. Stop installing the kb-* skills for agents,
      and change mise run doctor's kb section accordingly. Update
      agent-content's CLAUDE.md and AGENTS.md sections that describe kb skill
      artifacts. Confirm each harness's remote-MCP and header syntax against
      its current documentation, and record the URL and fetch date beside the
      definition, as build.bash already does for OpenCode.
    work_items:
      - Add the kb server definition to mise-tasks/lib/build.bash for Claude and Codex manifests, and to reconcile_opencode_mcp for OpenCode.
      - Add the Pi registration through Pi's package mechanism, or record that Pi cannot register it and why.
      - Retire or narrow install-kb-skills and the doctor kb-skill diff.
      - Update the tests that cover the build output and doctor.
    invariants:
      - The token value never appears in any generated file; only the variable name does.
      - mnene's registration is unchanged.
      - "`mise run validate` passes."
    acceptance_checks:
      - "`mise run build` output contains the kb server definition in each harness package and no token value."
      - "`mise run validate` passes."
      - After an approved reinstall, `claude mcp list` in a personal clanker session lists kb as connected once the server is deployed.
    completion_evidence: null

  - id: T006
    title: Pass the kb token and ingest URL into personal sessions
    status: not_started
    depends_on: []
    owner: null
    description: >
      Map the kb token into personal-context sessions as KB_MCP_TOKEN and
      KB_INGEST_TOKEN, and set KB_INGEST_URL=https://kb.h.tft.io (the base
      URL; see Decision 5), through clanker's secret-backed environment and
      the personal domain's credentials list. Decide whether the change
      belongs in tftio/clanker's config/config.toml or in this machine's
      ~/.config/clanker/local.toml. The operator approved widening the filter
      for the kb token on 2026-11-23; the concrete change is shown before it
      is applied.
    work_items:
      - Read the current credentials and env_from_secrets entries and choose the file.
      - Add the secret source for the token to the SOPS-backed shell environment, without writing its value into any repository.
      - Apply the clanker change on approval.
    invariants:
      - No other variable is added to the filter.
      - "`clanker doctor` passes."
    acceptance_checks:
      - In a new personal clanker session, KB_MCP_TOKEN and KB_INGEST_TOKEN are set, KB_INGEST_URL is https://kb.h.tft.io with no path, and the token's value is never printed during the check.
    completion_evidence: null

  - id: T007
    title: Schedule reconciliation on the laptop
    status: not_started
    depends_on: [T004, T006]
    owner: null
    description: >
      Add a LaunchAgent to the yadm home directory (an io.tft.kb-reconcile
      plist beside the existing io.tft agents) that runs
      kb-import --post <url> --reconcile --since <go-live date> on a schedule,
      where <url> is the base URL https://kb.h.tft.io (Decision 5).
      Reconcile discovers every session file under the harness configuration
      roots, so without --since its first run would post every transcript
      still on disk (30 under ~/.config/claude on 2026-11-23), which
      contradicts starting the corpus fresh. The go-live date is the date
      the companion plan's T006 completes. Supply the token and URL the way
      the home directory supplies secrets to its other agents. Discover that mechanism before writing; if the home directory
      has none, stop and ask the operator. systems/docs/macos.md mentions an
      io.tft.kb-server orphan in its history; read that history and remove or
      reuse the orphan if it still exists.
    work_items:
      - Read how existing io.tft LaunchAgents receive configuration and secrets.
      - Write the plist and any wrapper script, with logs under ~/.local/state.
      - Load it on approval.
    invariants:
      - The token is not written into the plist in plaintext.
      - A reconcile run against an unreachable server exits without modifying local state.
    acceptance_checks:
      - "`launchctl print gui/$(id -u)/io.tft.kb-reconcile` shows the agent loaded with its schedule."
      - "`kb-import --post <url> --reconcile --since <go-live date> --dry-run` run by hand lists only sessions from the go-live date onward that the server lacks."
    completion_evidence: null

  - id: T008
    title: Verify the whole path end to end
    status: not_started
    depends_on: [T005, T007]
    owner: null
    description: >
      After the companion plan's server is deployed and verified, verify the
      client side: MCP search, get, put, update and tag edits from Claude Code,
      Codex and OpenCode personal sessions, both on the LAN and off it over the
      tailnet; a Claude Code session captured by the SessionEnd hook appearing
      in the served corpus with vectors; a session captured while the server
      was stopped appearing after the next scheduled reconcile; and a search
      result for a new record carrying a non-null similarity.
    work_items:
      - Run each check and record the commands, outputs and timestamps.
      - Mark this plan complete and record the outcome in the companion plan.
    invariants:
      - Stopping the server for the downtime check is done with operator approval and restarted the same session.
    acceptance_checks:
      - Each listed check has recorded evidence in this task's completion_evidence.
      - "`kb queue status` on the server reports 0 dead-lettered after the checks."
    completion_evidence: null
```
<!-- TASK_GRAPH:END -->

# Task Details

## T001 — Add the read tools to the MCP surface

Status: `done`

Depends on: none

### Description

Add MCP tools `recent`, `list_by_tag`, `tags`, `links`, `hubs`, `orphans`,
`broken`, `projects`, `similar`, `prompt_list`, `prompt_show` and
`prompt_render` to `KbTools` in `src/mcp.rs`. Each handler calls the domain
function its CLI verb calls and returns compact JSON in the existing tiered
style: identifiers and titles, unless the tool exists to return a body. Move
each capability's agent guidance from `src/cli_main.rs` into the tool
description, shortened for an agent's context budget.

### Invariants

- No handler reimplements retrieval, tag normalization, or link resolution;
  each calls the CLI's domain function (ENG-010).
- `src/mcp.rs` does not import anything protocol-specific.
- The four existing tools keep their names, schemas and output shapes.
- The coverage floor in `REPO_INVARIANTS.md` RS-007 holds.

### Acceptance Checks

- `mise run test` passes.
- `mise run ci` passes.
- An MCP `tools/list` against `kb-mcp` (stdio) lists the four existing tools
  and the twelve new ones.

### Completion Evidence

Files touched: `src/mcp.rs` (handlers at lines 366-505, tool specs at 512-700,
`DEFAULT_LIST_RESULTS` at line 80); `src/search.rs` (`SimilarError` and
`similar`, lines 455-518); `src/cli_main.rs` (`run_similar` at line 3364 calls
`search::similar`; `no_embedding_text` and `named` removed);
`tests/mcp_read_tools.rs` (new, 17 `ToolSurface::call` tests);
`tests/mcp_stdio.rs` and `tests/mcp_http.rs` (tool-list tests assert all sixteen
names).

Commands:

- `mise run test`: passed, 882 run, 882 passed, 9 skipped.
- `mise run ci`: formatting, clippy, nextest, doc, machete, package, lockfile
  and Python checks passed. `check:coverage` reported 94.25% line coverage
  against the 93% floor. `cargo audit` and `cargo deny` failed on
  RUSTSEC-2026-0592 (`url` 2.5.8). `Cargo.lock` is unchanged by this task and
  `cargo update -p url` locked 0 packages, so no fixed `url` resolved on
  2026-11-23. The failure does not depend on this change.
- stdio `tools/list`: `kb-mcp` run under a scratch `HOME` with
  `KB_STORE_PATH` and `KB_INDEX_PATH` set, given initialize,
  `notifications/initialized` and `tools/list` on stdin, returned 16 tools:
  search, context, get, put, recent, list_by_tag, tags, links, hubs, orphans,
  broken, projects, similar, prompt_list, prompt_show, prompt_render. The HTTP
  list is asserted by `the_same_sixteen_tools_are_offered_over_http`.

Deviations from the plan:

- `similar` needed a domain function: the CLI held the model choice and the
  no-vector error inside `run_similar`. It moved to `search::similar` and both
  callers use it (D004).
- The CLI's no-vector message named `kb embed` while the capability text said
  backfill. The shared message now names both.
- `links` fails for an unknown id, as `context` does; the CLI returns empty
  lists.
- List tools with no CLI limit (`list_by_tag`, `tags`, `orphans`, `broken`) take
  a `limit` defaulting to 10 and report `total`. `recent` defaults to 10
  (CLI 50) and `hubs` to 10 (CLI 20).


## T002 — Add the write tools and embed every server write

Status: `done`

Depends on: T001

### Description

Add MCP tools `update`, `delete`, `tags_add`, `tags_rm` and `tags_merge`, each
calling the domain function its CLI verb calls, with the same preservation
rules: `update` and tag edits keep kind, source and provenance. Make `put`,
`update`, `tags_add`, `tags_rm` and `tags_merge` embed each record they wrote
when an embedder is configured from the environment, through the same
`embed_record` path and chunk size `kb queue drain` uses. Tag edits are
included because they give the record a new stream hash, which leaves it
without vectors (see Context); a test confirms this first, and if the hash
does not change, the tag tools do not embed and the Decision Log records why.
`tags_merge` embeds every record it rewrote. An embedding failure is reported
in the tool result and leaves the write in place.

### Invariants

- `update` and tag edits preserve the record's kind, source and provenance
  exactly as the CLI does (`src/write.rs` `existing_options`).
- `delete` keeps its audit record.
- A write never fails because embedding failed.
- No server process requires an embedder to start.

### Acceptance Checks

- `mise run test` passes.
- With `KB_EMBEDDING_BASE_URL` and `KB_EMBEDDING_MODEL` pointing at a test
  endpoint, a `put` over MCP produces `embeddings` rows for the record's
  passages.
- With no embedder configured, `put` over MCP succeeds and reports that nothing
  was embedded.
- With a test embedder configured, a `tags_add` over MCP leaves the record with
  `embeddings` rows under its new `stream_hash`.

### Completion Evidence

Files touched: src/mcp.rs (with_embedder at line 130, embedding_report at 141,
handlers update, delete, edit_tags and tags_merge at 379-475, write_specs at
857, dispatch at 1041); src/write.rs (embed_written at 462, DeleteError at 472,
delete_record at 503); src/cli_embed.rs (CliEmbedder drops its runtime in the
background, Drop at 270); src/bin/kb-mcp.rs (embedder_from_env at 85, built
once at start); src/cli_main.rs (run_delete at 3326 and embed_written_record at
3241 now call the shared functions); src/ingest.rs (process at 453 calls
embed_written); tests/mcp_write_tools.rs (new, 14 tests); tests/mcp_stdio.rs,
tests/mcp_http.rs and tests/mcp_read_tools.rs (tool lists now assert all
twenty-one names in order, and the new required fields).

Commands: the stream-hash test (a_tag_edit_gives_the_record_a_new_stream_hash)
was written first and passed: the hash before and after add_tags differ, so D003
stands. `mise run test` passed, 897 run, 897 passed, 9 skipped. `mise run ci`:
formatting, taplo, shellcheck, typos, clippy, nextest, doc, machete, package,
lockfiles and the Python checks passed; check:coverage reported 94.28% line
coverage against the 93% floor; `cargo audit` and `cargo deny` failed on
RUSTSEC-2026-0592 (url 2.5.8), the known failure that is independent of this
change. Cargo.toml and Cargo.lock are unchanged.
Live check: kb-mcp over stdio under a scratch HOME with KB_STORE_PATH and
KB_INDEX_PATH set. tools/list returned 21 tools. With no embedder, put returned
embedding.status "disabled" and the record was stored; tags_add did the same.
With a local fake endpoint (a Python server answering /v1/embeddings) and
KB_EMBEDDING_BASE_URL and KB_EMBEDDING_MODEL set, put returned one record with
status "embedded", embedded 1, and so did tags_add; the server exited 0 when
stdin closed. With KB_EMBEDDING_BASE_URL set and no model, the server started,
named the problem on stderr and stored writes unembedded.
Result shape: every writing tool returns embedding as {status, records}, where
status is "disabled" (no endpoint), "enabled" (records lists one entry per
record with status "embedded" and embedded and skipped counts, or "failed" with
the reason), or "unchanged" (a tag edit that rewrote nothing, so nothing was
embedded). tags_merge lists every rewritten record.
Deviations: (1) The chunk size for written records was a constant named at each
call site, not a selection; embed_written now names it once and the CLI, the
queue worker and the MCP tools all call it. (2) delete had no domain function,
only CLI code; it moved to write::delete_record with a typed DeleteError, and
`kb delete` calls it with unchanged messages. The store keeps no audit log:
"keeps its audit record" holds as the store behaves today, where delete unbinds
the name and the blobs stay addressable by hash. (3) A real defect found by the
live check: dropping a CliEmbedder inside the server's async runtime panicked at
shutdown. CliEmbedder now shuts its runtime down in the background, and a test
drops a surface holding an embedder inside a runtime. (4) A tag edit that
changed nothing reports "unchanged" and does not embed. (5) update does not take
provenance_json; the CLI's flag is not in this task. (6) tags_add, tags_rm and
update take a tags argument as arrays of strings; tags_add and tags_rm require
at least one.
Open concern for T003: in a first pipelined session, two tool calls sent back to
back against an empty index raced on creating it (the second saw schema version
0). Calls are dispatched concurrently, so this is the kind of fault T003 should
cover.

## T003 — Prove concurrent server writers are safe

Status: `not_started`

Depends on: T002

### Description

On the server, `kb-mcp` write tools, the `kb queue drain` timer and a scheduled
`kb embed` write to one store and one index at the same time. The index uses
WAL; no busy timeout or store write lock was found on 2026-11-23. Write a test
that runs all three writers (MCP write tools, `kb queue drain` and `kb embed`)
concurrently against one store and index, then checks that every write is
present, `kb fsck` reports no drift, no embedding row is lost or duplicated,
and no call failed with a lock error. Add whatever guard the test shows is
missing.

### Invariants

- The index stays derivable from the store (ST-002).
- No write is silently dropped; a write that cannot proceed fails with a reason.

### Acceptance Checks

- The concurrency test passes repeatedly (at least 20 consecutive runs).
- `kb fsck` against the test's store and index reports no drift.

### Completion Evidence

<Agents update this when the task is complete.>

## T004 — Release kb with the new surface

Status: `not_started`

Depends on: T003

### Description

Update `README.md`'s agent-surface and retrieval sections for the new tools and
embed-on-write, then release through the repository's release-plz flow, so the
systems flake input and `kb-import` on clients can pin the released version.
Merging and releasing need operator approval at the time.

### Invariants

- `kb` and `kb-import` carry the same version (README, "Installing kb-import").

### Acceptance Checks

- A tagged release exists, and `kb meta version` and `kb-import --version`
  built from it report the same version.

### Completion Evidence

<Agents update this when the task is complete.>

## T005 — Register the kb server in every agent-content harness package

Status: `not_started`

Depends on: T004, T006. T006 is needed because this task's last acceptance
check requires `KB_MCP_TOKEN` in a personal session.

### Description

In agent-content, register kb's HTTP MCP server at `https://kb.h.tft.io/mcp` in
the generated Claude, Codex, OpenCode and Pi packages beside mnene, with the
bearer token read from `KB_MCP_TOKEN` in each harness's own syntax. Stop
installing the `kb-*` skills for agents, and change `mise run doctor`'s kb
section to match. Update agent-content's `CLAUDE.md` and `AGENTS.md` sections
on kb skill artifacts. Confirm each harness's remote-MCP and header syntax
against its current documentation, and record the URL and fetch date beside the
definition, as `build.bash` already does for OpenCode.

### Invariants

- The token value never appears in any generated file; only the variable name
  does.
- mnene's registration is unchanged.
- `mise run validate` passes.

### Acceptance Checks

- `mise run build` output contains the kb server definition in each harness
  package and no token value.
- `mise run validate` passes.
- After an approved reinstall, `claude mcp list` in a personal clanker session
  lists kb as connected once the server is deployed.

### Completion Evidence

<Agents update this when the task is complete.>

## T006 — Pass the kb token and ingest URL into personal sessions

Status: `not_started`

Depends on: none. It changes clanker and the shell environment only, and can
run in parallel with T001 to T004.

### Description

Map the kb token into personal-context sessions as `KB_MCP_TOKEN` and
`KB_INGEST_TOKEN`, and set `KB_INGEST_URL=https://kb.h.tft.io` (the base URL;
see Decision 5), through clanker's secret-backed environment and the personal
domain's `credentials` list. Decide whether the change belongs in
tftio/clanker's `config/config.toml` or in this machine's
`~/.config/clanker/local.toml`. The operator approved widening the filter for
the kb token on 2026-11-23; the concrete change is shown before it is applied.

### Invariants

- No other variable is added to the filter.
- `clanker doctor` passes.

### Acceptance Checks

- In a new personal clanker session, `KB_MCP_TOKEN` and `KB_INGEST_TOKEN` are
  set, `KB_INGEST_URL` is `https://kb.h.tft.io` with no path, and the token's
  value is never printed during the check.

### Completion Evidence

<Agents update this when the task is complete.>

## T007 — Schedule reconciliation on the laptop

Status: `not_started`

Depends on: T004 (a released `kb-import` matching the server) and T006 (the
token's secret source).

### Description

Add a LaunchAgent to the yadm home directory, an `io.tft.kb-reconcile` plist
beside the existing `io.tft` agents, that runs
`kb-import --post <url> --reconcile --since <go-live date>` on a schedule,
where `<url>` is the base URL `https://kb.h.tft.io` (Decision 5).
Reconcile discovers every session file under the harness configuration roots,
so without `--since` its first run would post every transcript still on disk
(30 under `~/.config/claude` on 2026-11-23), which contradicts starting the
corpus fresh. The go-live date is the date the companion plan's T006
completes. Supply the token and URL the way the home directory supplies
secrets to its other agents.
Discover that mechanism before writing; if the home directory has none, stop
and ask the operator. `systems/docs/macos.md` mentions an `io.tft.kb-server`
orphan in its history; read that history and remove or reuse the orphan if it
still exists.

### Invariants

- The token is not written into the plist in plaintext.
- A reconcile run against an unreachable server exits without modifying local
  state.

### Acceptance Checks

- `launchctl print gui/$(id -u)/io.tft.kb-reconcile` shows the agent loaded
  with its schedule.
- `kb-import --post <url> --reconcile --since <go-live date> --dry-run` run by
  hand lists only sessions from the go-live date onward that the server lacks.

### Completion Evidence

<Agents update this when the task is complete.>

## T008 — Verify the whole path end to end

Status: `not_started`

Depends on: T005, T007

### Description

After the companion plan's server is deployed and verified, verify the client
side: MCP `search`, `get`, `put`, `update` and tag edits from Claude Code,
Codex and OpenCode personal sessions, both on the LAN and off it over the
tailnet; a Claude Code session captured by the `SessionEnd` hook appearing in
the served corpus with vectors; a session captured while the server was stopped
appearing after the next scheduled reconcile; and a search result for a new
record carrying a non-null similarity.

### Invariants

- Stopping the server for the downtime check is done with operator approval and
  restarted in the same session.

### Acceptance Checks

- Each listed check has recorded evidence in this task's completion evidence.
- `kb queue status` on the server reports 0 dead-lettered after the checks.

### Completion Evidence

<Agents update this when the task is complete.>

# Execution Protocol

Agents executing this plan must:

- Read the full plan and verify a task's dependencies are `done` before starting it.
- Set a task to `in_progress` and update `updated_at` when starting.
- Update the plan when they find it wrong, discover missing context, change the
  approach, need operator guidance, or alter the task graph.
- When operator guidance is needed: add an Operator Guidance Log entry, set affected
  tasks to `blocked`, and not guess when guessing risks rework, data loss, security,
  or product-behavior changes.
- When completing a task: verify invariants, run or describe acceptance checks, record
  `completion_evidence`, set status `done`, update `updated_at`, and update ADR
  sections only when the task changed context, decision, alternatives, or consequences.
- Name, in the completion evidence, the files the task actually touched. A path and a
  line number beats a filename.
- Treat the Operator Guidance Log, Decision Log, and completion evidence as
  append-only; supersede prior entries with a new dated entry rather than deleting them.
- Tasks T005 to T008 change agent-content, clanker, and the yadm home directory.
  Work there follows each repository's own conventions and approval gates, and
  records its evidence here.
