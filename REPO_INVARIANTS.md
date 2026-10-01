# Repository invariants

These invariants are binding for humans and coding agents working in this repository.
Bypasses must be explicit: use `INVARIANT-BYPASS(<ID>): <reason>` in the smallest
possible change and explain why the invariant cannot hold.

## Engineering invariants

| ID | Invariant | Enforcement |
|---|---|---|
| ENG-001 | Keep changes single-purpose and use Conventional Commits for committed work. | Review + `cog verify` |
| ENG-002 | Never commit secrets, credentials, or tokens; inject configuration from the environment rather than hardcoding it. | Review + secret scanning |
| ENG-003 | Every behavior change ships with tests that exercise real systems, assert outcomes rather than implementation, and cover error paths. Do not skip, xfail, or delete tests to make checks pass. | Tests + review |
| ENG-004 | Fail loudly with actionable context at the source; do not add silent `except` blocks or swallowed `Result` values. | Review |
| ENG-005 | Distinguish recoverable domain errors (return typed error values) from invariant violations (assert and crash with diagnostics). | Review |
| ENG-006 | Parse external data into typed containers at the system boundary; validate once on the way in and trust the types inside. | Review |
| ENG-007 | Keep retained state immutable; store snapshots in fields, module- and class-level bindings, closures, and caches, never aliased mutable objects. | Review |
| ENG-008 | Keep business logic in pure, deterministic functions; confine I/O, logging, and state mutation to a thin imperative shell. | Review |
| ENG-009 | Model data so illegal states are unrepresentable; dispatch over closed sum types exhaustively with no silent catch-all. | Review |
| ENG-010 | Wrap third-party dependencies behind domain-specific interfaces rather than threading their APIs through the codebase. | Review |
| ENG-011 | Fit the repository's established conventions; do not rewrite working code solely to change its library, tooling, or style. | Review |
| ENG-012 | Keep formatting, linting, typing, and tests clean before handing off work. | `mise run check` |
| ENG-013 | `mise.toml` plus `mise.lock` are the repository-owned tool declarations; hooks and CI must invoke `mise run` tasks rather than reimplementing checks. | Review + CI |

## Rust invariants

| ID | Invariant | Enforcement |
|---|---|---|
| RS-001 | Use Rust 2024 with the exact Rust toolchain declared in `mise.toml`; do not add `rust-toolchain.toml` as a second source of truth. | Review + `mise run check` |
| RS-002 | Commit `Cargo.lock` for every Rust repository, including libraries, so local and CI dependency resolution use the same artifact. | Review + CI |
| RS-003 | Deny unsafe code, missing docs, clippy warnings, unwrap/expect/panic/todo/unimplemented/dbg, wildcard imports, enum glob imports, and unchecked indexing. | `mise run check:clippy` |
| RS-004 | Keep command entry points thin; put behavior in the library crate and cover it with tests. | Tests + review |
| RS-005 | Model recoverable failures as typed error enums with `thiserror`; use `anyhow` only at binary or integration boundaries. | Review |
| RS-006 | Validate dependency advisories, licenses, duplicate dependency shape, unused dependencies, documentation, packaging, and spelling before handoff. | `mise run check` |
| RS-007 | Maintain at least 94% line coverage (repository-local floor; the generated default is 100%). **No entry point is a residual.** `src/main.rs` no longer exists, kb being a single binary, and the surviving entry point `src/bin/kb.rs` measures 100%. Interactive CLI dispatch (`src/cli_main.rs`) is likewise not a residual: it is covered in process by unit tests and out of process by the `tests/*_cli.rs` suites, which spawn the real binary via `CARGO_BIN_EXE_kb`, and that subprocess execution is attributed to the crate because `cargo llvm-cov` exports `LLVM_PROFILE_FILE` and the child inherits it. What remains below the whole-crate figure is ordinary uncovered branches rather than unreachable code, concentrated in `markdown.rs` (84%), `prompt.rs` (86%), `sexp.rs` (87%), `cli_main.rs` and `generator.rs` (89% each) — error paths and format variants, each reachable and each therefore a coverage debt rather than an exemption. Note that the first `Cover` column of the `llvm-cov` table is *region* coverage, several points below the line figure this invariant states; the gate is `--fail-under-lines`. Lowering the floor requires an explicit invariant bypass. | `mise run check:coverage` |

## Storage invariants

These govern the content-addressed store and the index derived from it. The store is the
single source of truth; the index is a disposable projection of it, kept only because
querying blobs directly would be slow. Each invariant below protects that split: if the
index could hold anything the store does not, or if code could read around the index, a
rebuild would stop being safe and the two would drift without anyone noticing.

| ID | Invariant | Enforcement |
|---|---|---|
| ST-001 | Stored content is self-describing: every blob carries a header from which its whole index row — id, corpus, timestamps, tags, source identity — is reconstructible from the blob alone. A field the index needs that no blob carries is a defect in the record definition, never a reason to let the index hold it authoritatively. | Tests + review |
| ST-002 | The index holds only what can be re-derived from the store. Every index table must be droppable and rebuildable from stored bytes alone, and no column may be the only copy of anything. | Tests + review |
| ST-003 | Nothing reads the store by path. Reads address content by hash through the index; a worktree is a convenience view for humans, not an interface for code, because a path-addressed read is a second source of truth that drifts silently. | Review |
| ST-004 | Re-derivation is a routine command rather than a recovery procedure: a full rebuild and a rebuild scoped to one corpus, record, or derivation stage are each a single invocation, idempotent, resumable after interruption, and each reports its elapsed time. This is what makes ST-003 hold in practice, since reaching around the index is only attractive while rebuilding is expensive. | Tests + review |

## CLI invariants

| ID | Invariant | Enforcement |
|---|---|---|
| CLI-001 | Keep command parsing and process I/O in `src/bin/kb.rs`; delegate domain behavior into `src/lib.rs`. | Tests + review |
| CLI-002 | Console output is user-facing API; cover non-trivial output behavior with integration tests. | Tests |
