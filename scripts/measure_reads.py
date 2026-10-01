#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Measure how often agent sessions read and write kb, from session logs on disk.

This is the scratch measurement from the operator-directed conversation that
produced `PLAN-20260923-project-identity` (kb node
`cc-0b05c389-fb58-439d-890b-2bcb1838dfc5`), ported to a committed, tested
script (that plan's T014). It answers one question: of the agent sessions
recorded on this machine, how many ever called into kb, and how many of those
calls were reads versus writes.

**2026-09-23 baseline**, recorded verbatim from the plan's Problem Statement
(measured over the Claude Code and Codex session logs on this machine; Claude
Code logs on disk reach back to August 2026, Codex logs to November 2025):

| source | window | sessions | with a kb read | with a kb write |
|---|---|---|---|---|
| Claude Code, personal | Aug to Sep 2026 | 192 | 33 | 7 |
| Claude Code, work | Aug to Sep 2026 | 166 | 12 | 0 |
| Codex | Nov 2025 to Sep 2026 | 945 | 56 | 145 |

Of the 45 Claude Code sessions with a read, the operator had mentioned kb in
22 of them. The `project` tag that exists on 420 kb records is not a project:
it co-occurs almost entirely with `claude-memory` and `conversation`, which
matches the harness memory format's `type: project` field carried over as a
tag during an import.

Design notes:

* **Recovered versus inferred.** The scratch script that produced the table
  above was run inside the conversation captured as kb node
  `cc-0b05c389-fb58-439d-890b-2bcb1838dfc5`, but kb's transcript importer
  drops tool calls by contract (`kb_import.sources`'s "Tool calls
  are dropped" design note), so the captured record holds only the
  conversation's prose, not the script's source or its exact pattern list.
  What is recovered from that record and the plan's Problem Statement,
  word for word, is the methodology: "a session as a read if an assistant
  turn called the kb MCP tools, a kb skill, or a kb query command in the
  shell", the two log roots and their glob shapes, and the operator-mention
  criterion ("22 ... had you mention kb or the knowledge base in a user
  message"). Every concrete pattern below -- the MCP tool names, the
  `kb-`-prefixed skill names, the CLI verb-to-read/write mapping, and the
  Codex tool-call record shapes -- is *inferred*, checked against kb's own
  `src/mcp.rs` (which exposes exactly four MCP tools: `search`, `context`,
  `get`, `put`) and against live session logs on this machine, not recovered
  from the original script.
* **What counts as a kb read or write.** Three surfaces, matched
  independently per session and OR'd together: an MCP tool call named
  `mcp__kb__<verb>`; a `kb-`-prefixed skill invocation; and a `kb <verb>`
  shell invocation. `KB_MCP_READ_TOOLS`, `KB_MCP_WRITE_TOOLS`,
  `KB_WRITE_SKILLS` and `KB_CLI_WRITE_VERBS`/`KB_CLI_READ_VERBS` are the
  patterns; anything not named there is not counted, which is a deliberate
  choice to avoid false positives from an agent merely *mentioning* a kb
  command in prose (a real hazard: the Skill tool's own listing enumerates
  every `kb-*` skill by name and description in a system reminder each
  session carries, so scanning free text for those names would count every
  session as a kb read).
* **Operator-mention detection is scoped to genuine human turns.** A Claude
  Code "user" record is sometimes a tool result rendered back to the model,
  not something the operator typed, and every user turn carries the harness's
  `<system-reminder>` blocks appended to it -- which, again, list every
  `kb-*` skill by name. Both are excluded before the mention pattern is
  applied (`_is_tool_result_message`, `SYSTEM_REMINDER_RE`), or essentially
  every session would appear operator-prompted.
* **Purity and I/O.** Log discovery, file reads and the optional
  `clanker project resolve` subprocess are confined to the driver at the
  bottom of this file (REPO_INVARIANTS.md ENG-008); every classification
  function above it takes already-decoded records and returns a value,
  with no I/O of its own, which is what makes them testable against a
  synthetic fixture with no real session logs on disk.
* **Read-only, per the plan's invariant for this task.** This script never
  opens a store, an index, or a log file for writing, and never invokes a kb
  or clanker subcommand that mutates anything -- `clanker project resolve`
  is documented as side-effect-free (`clanker project resolve --help`).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

Source = Literal["claude-code", "codex"]

# ── Untrusted-input accessors ──────────────────────────────────────────────
# Session logs are external data written by another program; these narrow
# `object` to a usable type at the boundary and return an empty value rather
# than raising, so one malformed line cannot abort a run over thousands of
# files (REPO_INVARIANTS.md ENG-006).


def _as_dict(value: object) -> dict[str, object]:
    """Narrow a JSON value to a mapping.

    Args:
        value: A value decoded from a session log line.

    Returns:
        The mapping, or an empty one when `value` is not a mapping.
    """
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    """Narrow a JSON value to a list.

    Args:
        value: A value decoded from a session log line.

    Returns:
        The list, or an empty one when `value` is not a list.
    """
    return value if isinstance(value, list) else []


def _as_str(value: object) -> str:
    """Narrow a JSON value to a string.

    Args:
        value: A value decoded from a session log line.

    Returns:
        The string, or the empty string when `value` is not a string.
    """
    return value if isinstance(value, str) else ""


# ── Read and write patterns (module-level constants) ───────────────────────
# See the module docstring's "What counts as a kb read or write" note. These
# are the only vocabulary this script recognizes; kb's own surfaces are the
# source of truth (`kb --help`, the emitted skills, `src/mcp.rs`), never
# restated logic from kb's Rust source.

#: The kb MCP server's read tools, named `mcp__<server>__<verb>` the way
#: Claude Code and Codex both expose an MCP tool to the model. `context` and
#: `get` are read verbs alongside `search`; `src/mcp.rs` exposes exactly
#: these four tool names, the fourth being `put`.
KB_MCP_READ_TOOLS: frozenset[str] = frozenset(
    {"mcp__kb__search", "mcp__kb__context", "mcp__kb__get"}
)

#: The kb MCP server's one write tool.
KB_MCP_WRITE_TOOLS: frozenset[str] = frozenset({"mcp__kb__put"})

#: Every kb agent-surface skill is named with this prefix (`kb-search`,
#: `kb-create`, ...). A skill invocation not named here is not a kb skill.
KB_SKILL_PREFIX = "kb-"

#: The subset of `kb-`-prefixed skills that mutate the store. Every other
#: `kb-`-prefixed skill name is a read (`kb-search`, `kb-get`, `kb-recent`,
#: `kb-list-by-tag`, `kb-links`, `kb-hubs`, `kb-orphans`, `kb-broken`,
#: `kb-projects`, `kb-prompt-list`, `kb-prompt-render`, `kb-prompt-show`,
#: `kb-similar`, `kb-tags`).
KB_WRITE_SKILLS: frozenset[str] = frozenset(
    {
        "kb-create",
        "kb-update",
        "kb-delete",
        "kb-tags-add",
        "kb-tags-rm",
        "kb-tags-merge",
        "kb-backfill",
    }
)

#: Matches a `kb <verb>` (and optional second word, for `tags add`-shaped
#: subcommands) shell invocation inside a larger command string. Requires a
#: non-identifier character or start-of-string before `kb`, so `kb` is never
#: matched as a substring of `kbps` or a path segment such as `/opt/kb/bin`.
KB_CLI_INVOCATION_RE = re.compile(r"(?<![\w./-])kb\s+([a-z][\w-]*)(?:\s+([a-z][\w-]*))?")

#: `kb <verb>` invocations that only read the store or index.
KB_CLI_READ_VERBS: frozenset[str] = frozenset(
    {
        "search",
        "get",
        "recent",
        "list-by-tag",
        "links",
        "hubs",
        "orphans",
        "broken",
        "projects",
        "prompt",
        "similar",
        "tags",
    }
)

#: `kb <verb>` invocations that always mutate the store, regardless of a
#: second word.
KB_CLI_WRITE_VERBS: frozenset[str] = frozenset({"create", "update", "delete", "backfill"})

#: Second words after `kb tags` that make the invocation a write; a bare
#: `kb tags` (no second word, or one outside this set, such as the tag name
#: in `kb tags list-by-tag`) lists the tag vocabulary and is a read.
KB_CLI_TAGS_WRITE_SUBVERBS: frozenset[str] = frozenset({"add", "rm", "merge"})

#: An operator message mentioning kb, either the bare word or "knowledge
#: base" (with or without a hyphen). Case-insensitive; `\bkb\b` will not
#: match "kbps" or "skb" and requires kb to be its own word.
KB_MENTION_RE = re.compile(r"\bkb\b|\bknowledge[\s-]?base\b", re.IGNORECASE)

#: Strips the harness's injected `<system-reminder>` blocks out of a user
#: turn before mention detection, since those blocks routinely name every
#: `kb-*` skill (see the module docstring).
SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)

#: Relative to a per-account directory, e.g. `~/.config/claude/personal`.
CLAUDE_CODE_SESSION_GLOB = "projects/*/*.jsonl"

#: Relative to a per-account directory, e.g. `~/.config/codex/personal`.
#: `archived_sessions` is deliberately not matched; those are a distinct,
#: much smaller directory this script does not scan.
CODEX_ROLLOUT_GLOB = "sessions/**/rollout-*.jsonl"

DEFAULT_CLAUDE_CODE_ROOT = Path("~/.config/claude").expanduser()
DEFAULT_CODEX_ROOT = Path("~/.config/codex").expanduser()
DEFAULT_CLANKER_BIN = "clanker"

#: Codex `response_item` payload types that represent a tool call rather than
#: conversational text. Only these are scanned for a kb invocation; assistant
#: and user prose is never scanned, for the same false-positive reason
#: mention detection excludes skill listings (see the module docstring).
CODEX_TOOL_CALL_PAYLOAD_TYPES: frozenset[str] = frozenset(
    {"function_call", "custom_tool_call", "local_shell_call"}
)


def classify_cli_text(text: str) -> tuple[bool, bool]:
    """Find every `kb <verb>` invocation in `text` and classify each.

    Args:
        text: Shell command text, or any string that may embed one (a Codex
            tool call's JSON-encoded arguments, for instance).

    Returns:
        `(saw_read, saw_write)`: whether any matched invocation was a read,
        and whether any was a write. An invocation whose verb is recognized
        as neither (`kb --help`, `kb reindex`, ...) contributes to neither.
    """
    saw_read = False
    saw_write = False
    for match in KB_CLI_INVOCATION_RE.finditer(text):
        verb, subverb = match.group(1), match.group(2)
        if (verb == "tags" and subverb in KB_CLI_TAGS_WRITE_SUBVERBS) or verb in KB_CLI_WRITE_VERBS:
            saw_write = True
        elif verb in KB_CLI_READ_VERBS:
            saw_read = True
    return saw_read, saw_write


def _classify_skill(skill: str) -> tuple[bool, bool]:
    """Classify a Skill-tool invocation by its `skill` argument.

    Args:
        skill: The `skill` field of a `Skill` tool call, e.g. `"kb-search"`.

    Returns:
        `(saw_read, saw_write)`, both `False` when `skill` is not a kb skill.
    """
    if not skill.startswith(KB_SKILL_PREFIX):
        return False, False
    return (False, True) if skill in KB_WRITE_SKILLS else (True, False)


# ── Session outcome ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SessionOutcome:
    """What one session's logged records show about its kb use.

    Attributes:
        when: The date used for `--since` filtering: the session's own
            earliest timestamp, or its file's modification time when no
            record carries one.
        cwd: The session's working directory, when the source records one.
        has_read: Whether any turn called a kb read surface.
        has_write: Whether any turn called a kb write surface.
        operator_mentioned_kb: Whether a genuine (non-tool-result) human
            message mentioned kb, per `KB_MENTION_RE`. `None` for a source
            that does not carry distinguishable human turns to check
            (Codex, in this script; see the module docstring).
    """

    when: date
    cwd: str | None
    has_read: bool
    has_write: bool
    operator_mentioned_kb: bool | None


def classify_claude_code_records(records: Sequence[object]) -> tuple[bool, bool, bool, str | None]:
    """Classify one Claude Code session's decoded JSONL records.

    Args:
        records: The session's records, in file order.

    Returns:
        `(has_read, has_write, operator_mentioned_kb, cwd)`.
    """
    has_read = False
    has_write = False
    mentioned = False
    cwd: str | None = None

    for record_raw in records:
        record = _as_dict(record_raw)
        record_type = _as_str(record.get("type"))
        if record_type not in ("user", "assistant"):
            continue
        if record.get("isMeta") or record.get("isSidechain"):
            # Meta records are harness-injected context; sidechain records
            # belong to a subagent transcript, not this session's own turns.
            continue
        if cwd is None:
            found_cwd = _as_str(record.get("cwd"))
            if found_cwd:
                cwd = found_cwd

        message = _as_dict(record.get("message"))
        content = message.get("content")

        if record_type == "assistant":
            read, write = _classify_claude_code_assistant_content(content)
            has_read = has_read or read
            has_write = has_write or write
        elif not mentioned and not _is_tool_result_message(content):
            text = SYSTEM_REMINDER_RE.sub("", _extract_user_text(content))
            if KB_MENTION_RE.search(text):
                mentioned = True

    return has_read, has_write, mentioned, cwd


def _classify_claude_code_assistant_content(content: object) -> tuple[bool, bool]:
    """Classify the tool calls in one assistant record's content blocks.

    Args:
        content: The record's `message.content`, expected to be a list of
            content blocks.

    Returns:
        `(saw_read, saw_write)` across every `tool_use` block present.
    """
    saw_read = False
    saw_write = False
    for block_raw in _as_list(content):
        block = _as_dict(block_raw)
        if block.get("type") != "tool_use":
            continue
        name = _as_str(block.get("name"))
        tool_input = _as_dict(block.get("input"))
        if name in KB_MCP_READ_TOOLS:
            saw_read = True
        elif name in KB_MCP_WRITE_TOOLS:
            saw_write = True
        elif name == "Skill":
            read, write = _classify_skill(_as_str(tool_input.get("skill")))
            saw_read = saw_read or read
            saw_write = saw_write or write
        elif name == "Bash":
            read, write = classify_cli_text(_as_str(tool_input.get("command")))
            saw_read = saw_read or read
            saw_write = saw_write or write
    return saw_read, saw_write


def _is_tool_result_message(content: object) -> bool:
    """Report whether a `user`-typed record is a tool result, not a human turn.

    Args:
        content: The record's `message.content`.

    Returns:
        `True` when any content block is a `tool_result`.
    """
    return any(_as_dict(block).get("type") == "tool_result" for block in _as_list(content))


def _extract_user_text(content: object) -> str:
    """Extract the plain text of a user turn, whatever shape it was sent in.

    Args:
        content: The record's `message.content`: a bare string, or a list of
            content blocks.

    Returns:
        The turn's text blocks joined by newlines; the empty string when
        `content` is neither shape or carries no text block.
    """
    if isinstance(content, str):
        return content
    parts = [
        _as_str(block.get("text"))
        for block in (_as_dict(b) for b in _as_list(content))
        if block.get("type") == "text"
    ]
    return "\n".join(parts)


def classify_codex_records(records: Sequence[object]) -> tuple[bool, bool, str | None]:
    """Classify one Codex rollout session's decoded JSONL records.

    Args:
        records: The session's records, in file order.

    Returns:
        `(has_read, has_write, cwd)`.
    """
    has_read = False
    has_write = False
    cwd: str | None = None

    for record_raw in records:
        record = _as_dict(record_raw)
        record_type = _as_str(record.get("type"))
        payload = _as_dict(record.get("payload"))

        if record_type == "session_meta":
            if cwd is None:
                found_cwd = _as_str(payload.get("cwd"))
                if found_cwd:
                    cwd = found_cwd
            continue

        if record_type != "response_item":
            continue
        if payload.get("type") not in CODEX_TOOL_CALL_PAYLOAD_TYPES:
            continue

        name = _as_str(payload.get("name"))
        if name in KB_MCP_READ_TOOLS:
            has_read = True
        elif name in KB_MCP_WRITE_TOOLS:
            has_write = True
        read, write = _classify_skill(name)
        has_read = has_read or read
        has_write = has_write or write

        read, write = classify_cli_text(_codex_tool_call_text(payload))
        has_read = has_read or read
        has_write = has_write or write

    return has_read, has_write, cwd


def _codex_tool_call_text(payload: dict[str, object]) -> str:
    """Collect the text worth scanning for a `kb <verb>` invocation.

    Codex has shipped at least three shapes for a tool call across the CLI
    versions on this machine: an older `function_call` whose `arguments` is a
    JSON-encoded string (a shell tool's `{"command": [...]}` among them), a
    newer `custom_tool_call` whose `input` is a JavaScript source string
    embedding a `cmd: "..."` shell command, and a `local_shell_call` whose
    `action.command` is an argv list. Rather than parse each shape exactly --
    the JavaScript wrapper in particular is not valid JSON -- every field that
    could plausibly hold command text is concatenated and handed to
    `classify_cli_text`, which matches the literal substring `kb <verb>`
    wherever it appears. This is deliberately permissive; see the module
    docstring's note on recovered versus inferred detail.

    Args:
        payload: A `response_item` record's `payload`, already known to be
            one of `CODEX_TOOL_CALL_PAYLOAD_TYPES`.

    Returns:
        The concatenated candidate text, possibly empty.
    """
    parts: list[str] = []
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        parts.append(arguments)
    tool_input = payload.get("input")
    if isinstance(tool_input, str):
        parts.append(tool_input)
    action = _as_dict(payload.get("action"))
    command = action.get("command")
    if isinstance(command, str):
        parts.append(command)
    elif isinstance(command, list):
        parts.append(" ".join(str(part) for part in command if isinstance(part, str)))
    return "\n".join(parts)


# ── Timestamps ──────────────────────────────────────────────────────────────


def _parse_timestamp(raw: str) -> datetime | None:
    """Parse an ISO-8601 timestamp as logged by either source.

    Args:
        raw: A timestamp string, typically ending in `Z`.

    Returns:
        The parsed, timezone-aware datetime, or `None` when `raw` does not
        parse.
    """
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _earliest_timestamp(records: Sequence[object]) -> datetime | None:
    """Find the earliest parseable timestamp among a session's records.

    Args:
        records: The session's records, in file order.

    Returns:
        The earliest timestamp found, from either a record's own
        `timestamp` field or, for Codex `session_meta` records, its
        `payload.timestamp`; `None` when no record carries one.
    """
    for record_raw in records:
        record = _as_dict(record_raw)
        raw = _as_str(record.get("timestamp"))
        if not raw:
            raw = _as_str(_as_dict(record.get("payload")).get("timestamp"))
        if not raw:
            continue
        parsed = _parse_timestamp(raw)
        if parsed is not None:
            return parsed
    return None


def session_date(records: Sequence[object], path: Path) -> date:
    """The date a session counts under for `--since` filtering.

    Args:
        records: The session's decoded records.
        path: The session's file, consulted for its modification time when
            no record carries a usable timestamp.

    Returns:
        The session's earliest timestamp's date, or the file's modification
        date as a fallback.
    """
    earliest = _earliest_timestamp(records)
    if earliest is not None:
        return earliest.astimezone(UTC).date()
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).date()


# ── Discovery and loading ───────────────────────────────────────────────────


def discover_sessions(root: Path, pattern: str) -> list[tuple[Path, str]]:
    """Find session files and the account each belongs to.

    Both Codex and Claude Code store sessions under a per-account directory
    directly below `root` (`personal`, `work`), mirroring
    `kb_import.cli`'s `discover_sessions`.

    Args:
        root: The configuration root holding one directory per account.
        pattern: A glob, relative to the account directory.

    Returns:
        Pairs of file path and account name, sorted by path.
    """
    found: list[tuple[Path, str]] = []
    if not root.is_dir():
        return found
    for account_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        found.extend((path, account_dir.name) for path in sorted(account_dir.glob(pattern)))
    return found


def load_records(path: Path) -> list[object]:
    """Decode one session's JSONL file, skipping unparsable lines.

    Args:
        path: The session file to read.

    Returns:
        The decoded records, in file order; empty when the file cannot be
        opened at all.
    """
    records: list[object] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records


def classify_session(
    source: Source, records: Sequence[object]
) -> tuple[bool, bool, bool | None, str | None]:
    """Classify a session's records, dispatching on its source.

    Args:
        source: Which log format `records` was decoded from.
        records: The session's decoded records.

    Returns:
        `(has_read, has_write, operator_mentioned_kb, cwd)`.
    """
    if source == "claude-code":
        return classify_claude_code_records(records)
    has_read, has_write, cwd = classify_codex_records(records)
    return has_read, has_write, None, cwd


# ── Project resolution ───────────────────────────────────────────────────────


class ProjectResolver:
    """Resolves a working directory's project slug through `clanker`.

    Never reimplements the slug grammar or remote normalization
    (`PLAN-20260923-project-identity`'s constraint that every consumer calls
    the one shared implementation): the only thing this class does is shell
    out to `clanker project resolve --dir <cwd>` and cache the answer, the
    same contract `kb_import.provenance`'s `resolve_project` relies
    on. A missing binary, a non-zero exit, or unexpected output all degrade
    to `None` rather than raising, since a resolution failure must not abort
    a read-only measurement run.
    """

    def __init__(self, clanker_bin: str = DEFAULT_CLANKER_BIN) -> None:
        """Create a resolver backed by `clanker_bin`.

        Args:
            clanker_bin: The `clanker` executable to invoke.
        """
        self._clanker_bin = clanker_bin
        self._cache: dict[str, str | None] = {}

    def resolve(self, cwd: str) -> str | None:
        """Resolve `cwd` to a project slug, memoized per working directory.

        Args:
            cwd: The working directory to resolve.

        Returns:
            The resolved slug, or `None` when nothing resolves or resolution
            failed.
        """
        if cwd not in self._cache:
            self._cache[cwd] = self._resolve_uncached(cwd)
        return self._cache[cwd]

    def _resolve_uncached(self, cwd: str) -> str | None:
        """Run `clanker project resolve --dir cwd` and take its slug field.

        Args:
            cwd: The working directory to resolve.

        Returns:
            The resolved slug, or `None`.
        """
        try:
            result = subprocess.run(
                [self._clanker_bin, "project", "resolve", "--dir", cwd],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        lines = result.stdout.splitlines()
        if not lines:
            return None
        parts = lines[0].split("\t")
        if len(parts) != 3 or not parts[0]:
            return None
        return parts[0]


# ── Aggregation ───────────────────────────────────────────────────────────


@dataclass
class Counts:
    """Session counts for one source/account, or one source/account/project.

    Attributes:
        sessions: Sessions seen.
        with_read: Sessions with at least one kb read.
        with_write: Sessions with at least one kb write.
        read_sessions_with_operator_mention: Of `with_read`, how many also had
            a human turn mentioning kb. Meaningful only where
            `operator_mentioned_kb` is computed (Claude Code); stays `0` for
            Codex rows, which is why it is reported alongside `with_read`
            rather than as a percentage.
    """

    sessions: int = 0
    with_read: int = 0
    with_write: int = 0
    read_sessions_with_operator_mention: int = 0

    def record(self, outcome: SessionOutcome) -> None:
        """Fold one session's outcome into these counts.

        Args:
            outcome: The session's classification.
        """
        self.sessions += 1
        if outcome.has_read:
            self.with_read += 1
            if outcome.operator_mentioned_kb:
                self.read_sessions_with_operator_mention += 1
        if outcome.has_write:
            self.with_write += 1

    def as_dict(self) -> dict[str, int]:
        """Render as a plain mapping for `--json` output.

        Returns:
            A dict with the same field names as the dataclass.
        """
        return {
            "sessions": self.sessions,
            "with_read": self.with_read,
            "with_write": self.with_write,
            "read_sessions_with_operator_mention": self.read_sessions_with_operator_mention,
        }


@dataclass
class Report:
    """The full result of one measurement run.

    Attributes:
        by_source_account: Counts keyed by `"<source>/<account>"`, e.g.
            `"claude-code/personal"`, `"codex/work"`.
        by_project: Present only with `--by-project`. Counts keyed first by
            `"<source>/<account>"` then by resolved project slug, with `None`
            standing for a session whose working directory resolved to no
            project (including one with no recorded working directory at
            all).
    """

    by_source_account: dict[str, Counts] = field(default_factory=dict)
    by_project: dict[str, dict[str | None, Counts]] | None = None


def measure(
    claude_code_root: Path,
    codex_root: Path,
    since: date | None,
    by_project: bool,
    resolver: ProjectResolver | None,
) -> Report:
    """Scan both log roots and build the full report.

    Args:
        claude_code_root: The Claude Code configuration root, holding one
            per-account directory each with `CLAUDE_CODE_SESSION_GLOB`.
        codex_root: The Codex configuration root, holding one per-account
            directory each with `CODEX_ROLLOUT_GLOB`.
        since: Only sessions dated on or after this date are counted; `None`
            counts everything found.
        by_project: Whether to also compute `Report.by_project`. Requires
            `resolver`.
        resolver: The project resolver to use when `by_project` is set;
            unused (and may be `None`) otherwise.

    Returns:
        The aggregated report.
    """
    report = Report(by_project={} if by_project else None)
    sources: tuple[tuple[Source, Path, str], ...] = (
        ("claude-code", claude_code_root, CLAUDE_CODE_SESSION_GLOB),
        ("codex", codex_root, CODEX_ROLLOUT_GLOB),
    )
    for source, root, pattern in sources:
        for path, account in discover_sessions(root, pattern):
            records = load_records(path)
            if not records:
                continue
            if since is not None and session_date(records, path) < since:
                continue
            has_read, has_write, mentioned, cwd = classify_session(source, records)
            outcome = SessionOutcome(
                when=session_date(records, path),
                cwd=cwd,
                has_read=has_read,
                has_write=has_write,
                operator_mentioned_kb=mentioned,
            )
            key = f"{source}/{account}"
            report.by_source_account.setdefault(key, Counts()).record(outcome)
            if report.by_project is not None:
                slug = resolver.resolve(cwd) if resolver is not None and cwd else None
                per_project = report.by_project.setdefault(key, {})
                per_project.setdefault(slug, Counts()).record(outcome)
    return report


# ── Output ──────────────────────────────────────────────────────────────────


def render_text(report: Report) -> str:
    """Render a report as aligned text tables.

    Args:
        report: The report to render.

    Returns:
        The rendered text, without a trailing newline.
    """
    lines: list[str] = ["source/account          sessions  with_read  with_write  read+mentioned"]
    for key in sorted(report.by_source_account):
        counts = report.by_source_account[key]
        lines.append(
            f"{key:<24}{counts.sessions:>9}{counts.with_read:>11}"
            f"{counts.with_write:>12}{counts.read_sessions_with_operator_mention:>16}"
        )
    if report.by_project is not None:
        lines.append("")
        lines.append("by project:")
        for key in sorted(report.by_project):
            lines.append(f"  {key}:")
            for slug in sorted(report.by_project[key], key=lambda s: s or ""):
                counts = report.by_project[key][slug]
                label = slug or "(no project)"
                lines.append(
                    f"    {label:<20}{counts.sessions:>9}{counts.with_read:>11}"
                    f"{counts.with_write:>12}"
                )
    return "\n".join(lines)


def render_json(report: Report) -> str:
    """Render a report as JSON.

    Args:
        report: The report to render.

    Returns:
        The rendered JSON text.
    """
    payload: dict[str, object] = {
        "by_source_account": {k: v.as_dict() for k, v in report.by_source_account.items()}
    }
    if report.by_project is not None:
        payload["by_project"] = {
            key: {(slug or ""): counts.as_dict() for slug, counts in projects.items()}
            for key, projects in report.by_project.items()
        }
    return json.dumps(payload, indent=2, sort_keys=True)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_since(raw: str) -> date:
    """Parse the `--since` flag.

    Args:
        raw: The flag's text, expected as `YYYY-MM-DD`.

    Returns:
        The parsed date.

    Raises:
        argparse.ArgumentTypeError: If `raw` does not parse.
    """
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--since must be YYYY-MM-DD, got {raw!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        The parser for this script's flags.
    """
    parser = argparse.ArgumentParser(
        description="Measure how often agent sessions read and write kb, from session logs."
    )
    parser.add_argument(
        "--since",
        type=_parse_since,
        default=None,
        help="Only count sessions dated on or after this date.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table.")
    parser.add_argument(
        "--by-project",
        action="store_true",
        help="Also break counts down by project, resolved through `clanker project resolve --dir`.",
    )
    parser.add_argument(
        "--claude-code-root",
        type=Path,
        default=DEFAULT_CLAUDE_CODE_ROOT,
        help="Claude Code configuration root (default: %(default)s).",
    )
    parser.add_argument(
        "--codex-root",
        type=Path,
        default=DEFAULT_CODEX_ROOT,
        help="Codex configuration root (default: %(default)s).",
    )
    parser.add_argument(
        "--clanker-bin",
        default=DEFAULT_CLANKER_BIN,
        help="The clanker executable to use for --by-project (default: %(default)s).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the measurement and print the report.

    Args:
        argv: Command-line arguments, excluding the program name; `None`
            uses `sys.argv[1:]`.

    Returns:
        The process exit code, always `0`: this script reads session logs
        only and has no failure mode that should abort a caller's pipeline.
    """
    args = build_parser().parse_args(argv)
    resolver = ProjectResolver(args.clanker_bin) if args.by_project else None
    report = measure(
        claude_code_root=args.claude_code_root,
        codex_root=args.codex_root,
        since=args.since,
        by_project=args.by_project,
        resolver=resolver,
    )
    print(render_json(report) if args.json else render_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
