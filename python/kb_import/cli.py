"""Import conversation transcripts from four sources into the kb knowledge base.

The four sources are the ChatGPT web export, the Codex CLI's `rollout-*.jsonl`
session files, the Claude.ai web export, and the Claude Code CLI's per-session
JSONL transcripts. Each has its own on-disk shape; all four are mapped onto the
common [`Conversation`][kb_import.model.Conversation] model
(`kb_import.model`) by the parsers in `kb_import.sources` and rendered by one
function, [`render`][kb_import.model.render], so a node imported from any
source is indistinguishable in form from any other.

This module is the CLI driver: source discovery, argument parsing, and the
two write paths (`import_conversations`, writing through the `kb` binary via
`kb_import.store`; `post_conversations`, posting to a capture server via
`kb_import.ingest`). It is the only module that performs process-level I/O
and argument parsing; `kb_import.model` and `kb_import.sources` are pure
(`REPO_INVARIANTS.md` ENG-008), and `kb_import.titles`, `kb_import.provenance`,
`kb_import.store`, and `kb_import.ingest` each own one slice of the I/O
boundary below the parsers. `kb_import.cli:main` is the console entry point;
no other module imports from this one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kb_import import __version__
from kb_import.ingest import IngestClient, IngestError, IngestTarget, submission_for
from kb_import.model import Conversation, TranscriptError, render
from kb_import.provenance import provenance_for
from kb_import.sources import (
    parse_chatgpt_export,
    parse_claude_code_transcript,
    parse_claude_export,
    parse_codex_rollout,
)
from kb_import.store import fetch_node, is_unchanged, write_node
from kb_import.titles import (
    DEFAULT_TITLE_MODEL,
    CommandError,
    CommandRunner,
    TitleCache,
    derive_title,
    run_command,
)

#: Where derived titles are remembered between runs.
DEFAULT_CACHE_PATH = Path.home() / ".local" / "state" / "kb-import" / "titles.json"


def title_cache_path() -> Path:
    """The title cache location: `$KB_IMPORT_TITLE_CACHE`, else `DEFAULT_CACHE_PATH`.

    A rehearsal against copies of the live store must not also read or write
    the live title cache -- doing so would let a scratch run mutate the one
    cache the live importer relies on for idempotence, and would let the
    live cache's cached titles mask a rehearsal from ever exercising
    derivation. The override makes "which cache" an explicit choice rather
    than a hard-coded path, the same way `KB_STORE_PATH`/`KB_INDEX_PATH`
    already let kb itself point at scratch copies.

    Returns:
        The path to use for this run's `TitleCache`.
    """
    override = os.environ.get("KB_IMPORT_TITLE_CACHE")
    return Path(override) if override else DEFAULT_CACHE_PATH


# ── Source discovery ───────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[object]:
    """Decode a JSONL file, skipping records that do not parse.

    A single corrupt line in one of 774 rollout files must not cost the other
    773, so a malformed record is skipped rather than raised.

    Args:
        path: The file to read.

    Returns:
        The decoded records, in file order.
    """
    records: list[object] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def discover_sessions(root: Path, pattern: str) -> list[tuple[Path, str]]:
    """Find CLI session files and the account each belongs to.

    Both Codex and Claude Code store sessions under a per-account directory, so
    the account is the path segment directly below `root`.

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


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        The parser for the importer's flags. Behavior behind these flags lands
        with the driver; the flags are defined here so the renderer and the
        parsers can be exercised against a stable interface.
    """
    parser = argparse.ArgumentParser(
        prog="kb-import",
        description="Import conversation transcripts into the kb knowledge base.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"kb-import {__version__}",
    )
    parser.add_argument(
        "--source",
        choices=["claude-export", "chatgpt-export", "codex", "claude-code", "all"],
        default="all",
        help="Which source to import from.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Import at most this many.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be imported without writing or contacting any endpoint.",
    )
    parser.add_argument("--since", default=None, help="Skip conversations older than YYYY-MM-DD.")
    parser.add_argument("--session", default=None, help="Import exactly one session by id.")
    parser.add_argument("--db", default=None, help="Path to the kb database.")
    parser.add_argument(
        "--post",
        default=os.environ.get("KB_INGEST_URL"),
        help=(
            "Post to a capture server's ingest endpoint instead of writing through kb. "
            "Defaults to $KB_INGEST_URL. The bearer token comes from $KB_INGEST_TOKEN."
        ),
    )
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help=(
            "With --post, ask the server which sessions it already holds and send only "
            "the rest. Heals gaps left by posts that never arrived."
        ),
    )
    parser.add_argument(
        "--claude-export", default=None, help="Directory holding a Claude.ai data export."
    )
    parser.add_argument(
        "--chatgpt-export", default=None, help="Directory holding a ChatGPT data export."
    )
    parser.add_argument(
        "--title-model", default=DEFAULT_TITLE_MODEL, help="Model alias used to derive titles."
    )
    parser.add_argument(
        "--transcript",
        default=None,
        help=(
            "Import exactly this CLI transcript file, skipping discovery. The SessionEnd "
            "hook uses this: it already knows the path, and scanning every account's "
            "sessions to find one file would grow slower with every session ever recorded."
        ),
    )
    parser.add_argument(
        "--account", default=None, help="Account tag; inferred from the path when omitted."
    )
    parser.add_argument(
        "--include-programmatic",
        action="store_true",
        help=(
            "Import Claude Code sessions started by `claude -p` or the SDK. Refused by "
            "default: the importer's own title derivation is one such session, so "
            "importing them feeds the corpus its own machinery."
        ),
    )

    provenance = parser.add_argument_group(
        "provenance",
        "Where the captured work happened. Every value is passed straight through to "
        "kb; no slug grammar or remote normalization lives in this script, and kb's own "
        "boundary validates what it is given on write.",
    )
    provenance.add_argument("--project", default=None, help="Project slug.")
    provenance.add_argument(
        "--project-source",
        dest="project_source",
        default=None,
        help="How --project was resolved: declared, path, remote, or derived.",
    )
    provenance.add_argument(
        "--remote", default=None, help="The working directory's origin remote, normalized."
    )
    provenance.add_argument(
        "--context", default=None, help="The credential boundary the session ran under."
    )
    provenance.add_argument(
        "--domain",
        dest="domains",
        action="append",
        default=None,
        help="A topical domain the session touched. Repeatable.",
    )
    provenance.add_argument(
        "--harness", default=None, help="The harness that captured the session."
    )
    provenance.add_argument("--model", default=None, help="The model in use when captured.")
    provenance.add_argument(
        "--clanker-session",
        dest="clanker_session",
        default=None,
        help="clanker's own launch session id, for correlating a record back to it.",
    )
    return parser


def _account_for(path: Path, root: Path, override: str | None) -> str:
    """Determine which account a session file belongs to.

    Args:
        path: The session file.
        root: The configuration root holding one directory per account.
        override: An explicit account name, or `None` to infer.

    Returns:
        The account name, or an empty string when it cannot be inferred.
    """
    if override:
        return override
    try:
        return path.resolve().relative_to(root.resolve()).parts[0]
    except (ValueError, IndexError):
        return ""


def _cli_sessions(args: argparse.Namespace, root: Path, pattern: str) -> list[tuple[Path, str]]:
    """List the CLI session files to import, honouring the narrowing flags.

    Args:
        args: Parsed command-line arguments.
        root: The configuration root holding one directory per account.
        pattern: A glob, relative to the account directory.

    Returns:
        Pairs of file path and account name.

    Raises:
        TranscriptError: If `--transcript` names a file that does not exist.
            Silently importing nothing would look identical to success.
    """
    if args.transcript:
        path = Path(args.transcript)
        if not path.is_file():
            raise TranscriptError(f"no transcript at {path}")
        return [(path, _account_for(path, root, args.account))]
    found = discover_sessions(root, pattern)
    if args.session:
        # Filter before parsing. Parsing every file to discard all but one made
        # the hook's cost grow with the size of the whole session archive.
        found = [(p, a) for p, a in found if args.session in p.stem]
    return [(p, args.account or a) for p, a in found]


@dataclass
class Report:
    """Per-source outcome counts.

    Attributes:
        created: Nodes newly written.
        updated: Existing nodes replaced.
        skipped: Conversations filtered out before any write.
        empty: Session files that parsed to no renderable turn at all. Counted
            rather than silently skipped, so a format the parser cannot yet
            read is visible instead of disappearing from the corpus.
        failed: Conversations whose write or parse raised, with the reason.
    """

    created: int = 0
    updated: int = 0
    skipped: int = 0
    empty: int = 0
    failed: list[str] = field(default_factory=list)

    def line(self, source: str, *, dry_run: bool) -> str:
        """Render a one-line summary.

        Args:
            source: The source these counts belong to.
            dry_run: Whether this was a dry run.

        Returns:
            A human-readable summary line.
        """
        verb = "would create" if dry_run else "created"
        return (
            f"{source}: {verb} {self.created}, updated {self.updated}, "
            f"skipped {self.skipped}, empty {self.empty}, failed {len(self.failed)}"
        )


def _load_export(path: Path) -> object:
    """Read a web export's `conversations.json`.

    Args:
        path: Either the export directory or the JSON file itself.

    Returns:
        The decoded contents.
    """
    target = path / "conversations.json" if path.is_dir() else path
    return json.loads(target.read_text(encoding="utf-8"))


def collect(args: argparse.Namespace) -> tuple[list[Conversation], dict[str, list[str]]]:
    """Gather conversations from every selected source.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A pair of `(conversations, empty)`: the conversations from the
        selected sources in source order, and a per-source mapping of the
        paths of session files that parsed to no renderable turn at all --
        `parse_codex_rollout` returning `None` -- so the driver can report
        them instead of letting them vanish. Filtered-out Claude Code
        transcripts (`is_programmatic`) are a deliberate skip, not this
        defect, and are not included.
    """
    wanted = args.source
    conversations: list[Conversation] = []
    empty: dict[str, list[str]] = {}

    if wanted in ("all", "claude-export") and args.claude_export:
        conversations.extend(parse_claude_export(_load_export(Path(args.claude_export))))

    if wanted in ("all", "chatgpt-export") and args.chatgpt_export:
        conversations.extend(parse_chatgpt_export(_load_export(Path(args.chatgpt_export))))

    if wanted in ("all", "codex"):
        root = Path.home() / ".config" / "codex"
        for path, account in _cli_sessions(args, root, "sessions/**/rollout-*.jsonl"):
            conversation = parse_codex_rollout(load_jsonl(path), account)
            if conversation is not None:
                conversations.append(conversation)
            else:
                empty.setdefault("codex", []).append(str(path))

    if wanted in ("all", "claude-code"):
        root = Path.home() / ".config" / "claude"
        for path, account in _cli_sessions(args, root, "projects/*/*.jsonl"):
            conversation = parse_claude_code_transcript(
                load_jsonl(path), path.stem, account, args.include_programmatic
            )
            if conversation is not None:
                conversations.append(conversation)

    return conversations, empty


def select(conversations: list[Conversation], args: argparse.Namespace) -> list[Conversation]:
    """Apply the `--session`, `--since`, and `--limit` filters.

    Args:
        conversations: Everything the sources produced.
        args: Parsed command-line arguments.

    Returns:
        The conversations to import.
    """
    chosen = conversations
    if args.session:
        chosen = [c for c in chosen if args.session in (c.source_id, c.node_id)]
    if args.since:
        chosen = [c for c in chosen if (c.started_at or "") >= args.since]
    if args.limit is not None:
        chosen = chosen[: args.limit]
    return chosen


def import_conversations(
    conversations: list[Conversation],
    args: argparse.Namespace,
    cache: TitleCache,
    run: CommandRunner,
) -> dict[str, Report]:
    """Write each conversation to kb, reporting per source.

    A failure on one conversation is reported and the rest continue: a single
    malformed record in one of hundreds of files must not cost the others.

    Args:
        conversations: The conversations to import.
        args: Parsed command-line arguments.
        cache: The title cache.
        run: The subprocess runner.

    Returns:
        One `Report` per source encountered.
    """
    reports: dict[str, Report] = {}
    for conversation in conversations:
        report = reports.setdefault(conversation.source, Report())
        if args.dry_run:
            report.created += 1
            continue
        try:
            fetched = fetch_node(conversation.node_id, args.db, run)
            provenance = provenance_for(conversation, args, run)
            title = derive_title(conversation, cache, run, args.title_model)
            rendered = render(conversation, title)
            if fetched is None:
                outcome = write_node(
                    conversation, title, args.db, run, exists=False, provenance=provenance
                )
            else:
                stored_document, stored_provenance = fetched
                # Idempotence (T008's backfill leans on this): an unchanged
                # document with unchanged provenance is a no-op; a provenance
                # change alone -- the common backfill case -- still counts as
                # a change even when the document text does not. Provenance
                # is compared only when this run actually computed some: this
                # importer asserts only provenance it knows (PLAN-20260923-
                # project-identity T016 follow-up), so a run that resolved
                # none -- no `--project`/flags and no captured `cwd` -- must
                # neither treat a stored record's provenance as "changed" nor
                # clear it. `stored_provenance == provenance` would compare
                # a possibly non-empty stored mapping against `{}` and
                # wrongly count every such record as needing a rewrite on
                # every run.
                provenance_changed = bool(provenance) and stored_provenance != provenance
                if is_unchanged(stored_document, rendered) and not provenance_changed:
                    report.skipped += 1
                    continue
                outcome = write_node(
                    conversation, title, args.db, run, exists=True, provenance=provenance
                )
        except (CommandError, OSError, ValueError) as error:
            report.failed.append(f"{conversation.node_id}: {error}")
            continue
        if outcome == "created":
            report.created += 1
        else:
            report.updated += 1
    return reports


def post_conversations(
    conversations: list[Conversation],
    args: argparse.Namespace,
    cache: TitleCache,
    run: CommandRunner,
    client: IngestTarget,
) -> dict[str, Report]:
    """Post each conversation to the capture server, reporting per source.

    The counterpart to `import_conversations`, and deliberately not a mode
    inside it: writing through `kb` reads the stored document back to decide
    whether anything changed, and a client posting to a server cannot — it can
    ask which ids are held, never what they say. The server settles that
    instead, which is why a re-post costs nothing there.

    A failure on one conversation is reported and the rest continue. A capture
    run that stops at the first unreachable moment would turn one bad session
    into a lost batch.

    Args:
        conversations: The conversations to post.
        args: Parsed command-line arguments.
        cache: The title cache.
        run: The subprocess runner, for title derivation.
        client: The server to post to.

    Returns:
        One `Report` per source encountered.
    """
    served: list[str] = []
    if args.reconcile and not args.dry_run:
        served = client.served_ids()

    reports: dict[str, Report] = {}
    for conversation in conversations:
        report = reports.setdefault(conversation.source, Report())
        if args.reconcile and conversation.node_id in served:
            report.skipped += 1
            continue
        if args.dry_run:
            report.created += 1
            continue
        try:
            title = derive_title(conversation, cache, run, args.title_model)
            provenance = provenance_for(conversation, args, run)
            client.post(
                submission_for(conversation.node_id, render(conversation, title), provenance)
            )
        except (IngestError, CommandError, OSError, ValueError) as error:
            report.failed.append(f"{conversation.node_id}: {error}")
            continue
        report.created += 1
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments, or `None` to read `sys.argv`.

    Returns:
        The process exit status: non-zero if any conversation failed.
    """
    args = build_parser().parse_args(argv)
    try:
        collected, empty = collect(args)
        conversations = select(collected, args)
    except (OSError, json.JSONDecodeError, TranscriptError) as error:
        print(f"kb-import: {error}", file=sys.stderr)
        return 2

    cache = TitleCache(title_cache_path())
    if args.post:
        try:
            client = IngestClient(args.post, os.environ.get("KB_INGEST_TOKEN", ""))
            reports = post_conversations(conversations, args, cache, run_command, client)
        except IngestError as error:
            # Reaching the server at all is a precondition, not a per-session
            # failure: reporting it once names the cause, where a hundred
            # identical per-session failures would bury it.
            print(f"kb-import: {error}", file=sys.stderr)
            return 2
    else:
        reports = import_conversations(conversations, args, cache, run_command)
    if not args.dry_run:
        cache.save()

    # Session files that parsed to no renderable turn at all are reported
    # here regardless of source, even a source `select` filtered out of
    # `conversations` entirely -- they never became a Conversation, so
    # `--session`/`--since`/`--limit` cannot have meant to hide them.
    for source, paths in empty.items():
        reports.setdefault(source, Report()).empty += len(paths)

    failures = 0
    for source in sorted(reports):
        report = reports[source]
        print(report.line(source, dry_run=args.dry_run))
        for path in empty.get(source, []):
            print(f"  empty {path}", file=sys.stderr)
        for failure in report.failed:
            print(f"  failed {failure}", file=sys.stderr)
        failures += len(report.failed)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
