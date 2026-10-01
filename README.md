# kb

Personal knowledge base — typed AST with org-mode as projection, SQLite-backed

## Getting started

Toolchain, task execution, and hook tools are managed by mise.
The Rust toolchain is declared in `mise.toml`; rustup is an implementation
detail and `rust-toolchain.toml` is intentionally absent.
Entering the directory does not prepare, install, or regenerate anything: setup
is explicit, so no lockfile ever changes because someone walked into the
repository. From a fresh worktree:

```sh
mise trust --quiet
mise install
mise run setup:idea   # optional; regenerates the gitignored .idea/
```

Dependencies move on one deliberate command, and never on their own:

```sh
mise run update       # mise tools, cargo crates, Python packages, prek hooks
mise run check:locks  # read-only; fails if a lockfile is stale
```

## Installing kb-import

`kb-import` is the transcript importer, packaged separately from the `kb`
binary as the `kb-import` Python distribution under `python/kb_import`. Its
version is not carried twice — it is read out of `Cargo.toml`'s `[package]`
table at build time, so it always matches the crate version at the commit or
tag it was built from.

```sh
uv tool install git+https://github.com/tftio/kb@vX.Y.Z
```

or, from a local checkout:

```sh
uv tool install /path/to/kb
```

An install from a tag stays on that tag: `uv tool upgrade kb-import` does not
move it. To move to a newer release, reinstall from the new tag with
`uv tool install --force git+https://github.com/tftio/kb@vX.Y.Z`.

Because the two binaries are versioned from the same `Cargo.toml` and can be
installed independently, they can drift: `kb-import --version` should equal
`kb --version`'s version. If they differ, reinstall `kb-import` against the
tag or checkout that matches the installed `kb`.

## Building with Nix

`flake.nix` packages both binaries for a NixOS host that serves the corpus.
It builds the committed tree and nothing more — the test suite is the gate
above, not the package — and exports no NixOS module, since a service belongs
in the host's own configuration beside its other services.

```sh
nix build .#kb            # ./result/bin/{kb,kb-mcp}
nix run .#kb-mcp -- --help
nix flake check           # builds, then asks both binaries for --help
```

## Tasks

```sh
mise run check  # check-only hooks, as CI runs them
mise run lint   # manual autofix hooks
mise run test   # test suite
mise run ci     # full CI gate
```

The generated Rust gate includes formatting, TOML formatting, shell linting,
spelling, clippy, nextest, docs, unused-dependency detection, advisory audit,
license/source policy, packaging, and line coverage. Coverage is held to a
repository-local floor of 94% (the generated default is 100%), and no part of
the crate is exempt from it: kb is a single binary whose entry point
`src/bin/kb.rs` is itself fully covered, and interactive CLI dispatch is
covered partly in process and partly by the `tests/*_cli.rs` suites, which
spawn the real binary — that subprocess coverage is attributed and counts
toward the figure. The shortfall to 100% is ordinary uncovered error paths
and format variants, not unreachable code. See RS-007 in
`REPO_INVARIANTS.md`.

## Retrieval is keyword-only for recent nodes

**The embedding path is deliberately unwired as of 2026-09-16.** `KB_EMBEDDING_BASE_URL`
and `KB_EMBEDDING_MODEL` were removed from the shell environment, along with
`KB_INGEST_URL` and `KB_IMPORT_SCRIPT`. Capture was not: the `SessionEnd` hook still distils every
session in, and the store has kept growing since.

The consequence is a corpus split at that date:

| | up to 2026-09-16 | after |
|---|---|---|
| keyword search (FTS5) | yes | yes |
| vector half of `search` | yes | no vector stored |
| `similar` | yes | fails; the error names `backfill` |
| `backfill` | n/a | cannot run without an endpoint |

Nothing is lost and nothing is wrong — the vectors written before that date are
still in the derived index, and every node is still retrievable by keyword, tag,
and link. But **an empty or thin result from `search` over recent material is not
evidence of absence.** It may mean only that the node carries no vector, which is
exactly the failure `search`'s own documentation warns about: a `null` similarity
means no vector score was computed, not that the node scored zero. Fall back to
`list-by-tag`, `recent`, and `--match` with an explicit phrase before concluding
a topic is unrecorded.

To undo the split, restore the two `KB_EMBEDDING_*` variables and run `backfill`,
which selects on the absence of a current vector rather than on checkpoint state
and therefore converges on full coverage however many times it is interrupted.

## Mail corpus

kb can index a Maildir as a second, reference-only corpus: the messages stay
in the Maildir, and kb keeps a catalogue and derived search rows. Mail is off
unless a Maildir root is configured. There is no default mailbox path; set it
with `--maildir <path>` or the `KB_MAIL_ROOT` environment variable (the flag
wins when both are set). The root holds the folders to index, `Inbox` and
`Archive`.

```sh
KB_MAIL_ROOT=~/Mail kb mail scope    # what would be indexed, and why the rest is out
KB_MAIL_ROOT=~/Mail kb mail index    # rebuild the mail corpus's rows
kb reindex --maildir ~/Mail          # full rebuild, mail included
```

`kb reindex` with neither the flag nor the variable set rebuilds the store's
corpora and skips mail, saying so on stdout.

## Agent surface

kb's CLI carries a declarative agent surface (`tftio_lib::AgentSurfaceSpec`)
introspectable through `kb meta agent`:

```sh
kb meta agent list                       # capability names visible to an agent
kb meta agent describe <name>            # one capability's full contract
kb meta agent emit-skills --target claude --out <dir>   # render capabilities as skills
kb meta agent emit-hooks --target claude --out <dir>    # write declared lifecycle hooks
```

`emit-hooks` writes each hook kb declares whose lifecycle event the target
harness supports, marks the script executable, and prints the harness's own
registration-fragment shape on stdout so an installer can merge it straight
into the harness's settings file. kb declares one hook today: `session-end-kb`,
registered against `HookEvent::SessionEnd` with a 30-second timeout and the
status message "Distilling session into kb" — the same registration an
installer would otherwise write into Claude Code's `settings.json` by hand. Its script is `include_str!`'d from
`scripts/session-end-kb.sh`, so the file this repository lints and tests
under that path is the exact bytes `emit-hooks` writes out; nothing here
restates or diverges from it.

Codex has no `SessionEnd`-equivalent lifecycle event, so
`emit-hooks --target codex` writes nothing for this hook and prints the empty
fragment `{}`, rather than registering a key Codex never reads.

## Design notes

### Org-mode AST and parser

The typed org-mode AST (`Document`, `Block`, `Inline`, and friends) is re-used
from the companion [`tftio-org`](https://crates.io/crates/tftio-org) crate, which
was extracted from this crate. The AST types are byte-for-byte behaviour-identical
across the two crates, so deduplicating them costs nothing.

The **parser is deliberately kept in-crate** rather than taken from `tftio-org`.
`tftio-org` 0.1.0's `parser::parse_document` enters an infinite loop on pipe /
table input such as `"|\n"` — a defect this crate's own parser explicitly fixed
and guards with regression tests (`tests/v10_phase.rs`,
`parser_panic_free_pipe_returns_ok`). Because that defect ships in a published
dependency version that cannot be patched from here, adopting `tftio-org`'s parser
would reintroduce the hang. Keeping the proven in-crate parser preserves behaviour;
the AST dependency is still shared. Revisit this once a fixed `tftio-org` is
released.

### Record header and provenance (`kb-record/2`)

Every stored record opens with a line-oriented header
(`src/record.rs::RecordHeader`), and every header field is what the derived
index rebuilds from — nothing the index holds is authoritative
(`REPO_INVARIANTS.md` ST-001, ST-002). `PLAN-20260923-project-identity` T005
added an optional provenance block: `project`, `project-source`, `remote`,
`context`, a repeated `domain` line, `harness`, `model`, `session`, and `cwd`,
written between `normalizer:` and the `tag:` lines. Every one of these lines
is optional, which is what lets `kb-record/1` blobs (written before this
change) keep parsing and what lets `kb reindex` succeed over a store mixing
both formats: a record that carries none of the block simply indexes with
every provenance column `NULL`.

| header key | meaning |
|---|---|
| `project` | The project slug the session or write resolved to (`tftio_lib::project::Slug`). |
| `project-source` | How `project` was resolved: `declared`, `path`, `remote`, or `derived`. |
| `remote` | The working directory's normalized origin remote (`tftio_lib::project::normalize_remote`); the value must already equal its own normalization. |
| `context` | The credential boundary the session ran under (`personal`, `work`, ...). Not a project — a session can carry a context and no project. |
| `domain` | A topical domain the session or note touches. Repeated, like `tag:`. |
| `harness` | The harness that captured the session (`claude-code`, `codex`, ...). |
| `model` | The model in use when the record was produced. |
| `session` | The harness session id, for correlating a record back to its transcript source. |
| `cwd` | The working directory the session ran in. |

`context`, `harness`, `model`, and `session` are free-form, single-line
strings; `domain` is repeated. `remote` and `cwd` are rejected at
construction if they contain a line break, since either would otherwise be
able to forge a second header line; every field is validated once at the
boundary (a `RawProvenance` from JSON, or the header's own parser) and
carried as a typed `Provenance` from there on (`REPO_INVARIANTS.md` ENG-006).

Provenance is a header field rather than a tag, deliberately: a bare
project-slug tag would collide with topical tags that already share a
repository's name (`clanker`, `silent-critic`) and, worse, with the
**pre-existing `project` tag already present on about 420 records**. That tag
is a leftover of an old harness memory import (`type: project` carried over as
a filetag) and is a *memory-type label*, not this project axis — it happens to
co-occur almost entirely with `claude-memory` and `conversation` and says
nothing about which repository or body of work a record belongs to. Do not
read it as, or merge it with, the `project` provenance field.

`put_record` takes a `crate::write::WriteOptions { kind, source, provenance }`
rather than hardcoding `kind: note` for every write. `kb tags add`, `tags rm`,
and `tags merge` read a record's existing header first and preserve its kind,
source, and provenance across the rewrite a tag edit causes. The queue worker
does the same for a captured transcript: a submission carrying the
`conversation` tag is written as `kind: session-transcript` (the same rule
`migrate.rs`'s one-way import applies), and the queue's `Submission` carries
an optional `provenance` object that is validated and passed straight
through.

`kb create --provenance-json <file>` and `kb update --provenance-json <file>`
are the local (non-queue) write path's way to assert provenance: `<file>`
must be a real file path holding a JSON object with any of the keys above
(`domains` as a JSON array, `project_source` in place of the header's
hyphenated `project-source`), every key optional. `-` (stdin) is not
accepted, because both commands' document body already owns stdin. Both
flags share one parsing and validation path (`apply_provenance_json` in
`src/cli_main.rs`), so a provenance file that `create` accepts is accepted
by `update` too, and the reverse. `create` has no existing record, so it
always starts from `WriteOptions::note` and layers the file's provenance
(if any) on top. `update` (`PLAN-20260923-project-identity` T016) starts
from the *existing* record's kind, source and provenance
(`existing_options` in `src/write.rs`, the same helper `tags add`/`tags
rm`/`tags merge` use) and changes only what `--provenance-json` names:
given, the provenance is replaced outright with the file's contents while
kind and source still come from the existing record; omitted, kind, source
and provenance are all left exactly as they were. This makes an ordinary
`kb update` — the path an agent's `kb create --id <id> || kb update <id>`
capture dedup takes on a second write — behave like `tags add`/`tags
rm`/`tags merge` rather than like `kb create`: it cannot turn a
`session-transcript` back into a `note` or silently drop the project an
earlier capture asserted. `created` is unaffected either way: `update` has
always preserved it, and a `--provenance-json` update is not an exception.
Before `--provenance-json` existed, the only way to move an *existing*
record's provenance was `kb delete` followed by `kb create`, which resets
`created`, cascade-deletes the record's tags and links, and can lose the
record outright if `create` fails after `delete` succeeds — `update
--provenance-json` replaces that with a single in-place write.

`kb get` shows a node's provenance in both its text and `--json`
output, and the SilverBullet projection (`src/project.rs`) writes the same
fields into each page's frontmatter as `kb_project`, `kb_project_source`,
`kb_remote`, `kb_context`, `kb_domains`, `kb_harness`, `kb_model`,
`kb_session`, and `kb_cwd`.

### Filtering by project and context (`PLAN-20260923-project-identity` T006)

`kb search` and the MCP `search` tool both take `--project <slug>` /
`project` and `--context <name>` / `context`, applied beside the corpus
predicate inside `SEARCH_TEXT_IN_SQL` and `RANK_BY_EMBEDDING_SQL`
(`src/index.rs`) rather than as a post-filter over already-ranked hits. A
`--project` value is parsed against `tftio_lib::project::Slug`'s grammar at
the CLI and MCP boundaries, so an unusable slug is a clear error rather than
a filter that silently matches nothing; `--context` carries no grammar of
its own. Neither argument changes a search that does not pass it: the
predicates are `(? IS NULL OR column = ?)`, so an absent filter is a no-op,
not a different query plan. `kb search --json` carries the same
`provenance` block per hit that `kb get --json` does; the MCP `search` tool
deliberately does not, since it is the cheap identifiers-and-titles tier and
`provenance` is one of the fields an agent pays for by calling `get`
instead.

`kb projects` (text and `--json`) lists every project slug asserted by at
least one record's header, with its record count and its newest record's
`created` date, plus a count of records carrying no project at all — the
read side of the axis T005 wrote and T006 makes searchable.
