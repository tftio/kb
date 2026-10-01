"""Provenance resolution.

No slug grammar or remote normalization lives here (T007's constraint):
`clanker project resolve` is the only thing this module ever asks to
interpret a working directory, and kb's `RawProvenance::validate` is what
ultimately accepts or refuses the result.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
from collections.abc import Iterator
from pathlib import Path

from kb_import.model import Conversation
from kb_import.titles import CommandError, CommandRunner

#: `RawProvenance`'s wire field names (`src/record.rs`), in the order its
#: header lines are written. Kept as a tuple so every place that builds or
#: reads a provenance mapping agrees on the vocabulary.
PROVENANCE_FIELDS = (
    "project",
    "project_source",
    "remote",
    "context",
    "domains",
    "harness",
    "model",
    "session",
    "cwd",
)


def resolve_project(cwd: str, run: CommandRunner) -> dict[str, str]:
    r"""Resolve a bare working directory's project through `clanker`.

    `clanker project resolve --dir <cwd>` prints one tab-separated line,
    `<slug>\\t<source>\\t<remote>`, with empty fields when nothing resolves,
    and exits `0` either way (`clanker project resolve --help`). Absence of
    `clanker` from `PATH`, a non-zero exit, or output that does not match that
    shape are all the same "nothing known" case to a caller -- this function
    degrades to an empty mapping rather than raising, because a resolution
    failure must not abort an import the way a malformed transcript would.

    Args:
        cwd: The working directory to resolve.
        run: The subprocess runner.

    Returns:
        A mapping holding whichever of `project`, `project_source`, `remote`
        the directory resolved to; empty when nothing did.
    """
    try:
        output = run(["clanker", "project", "resolve", "--dir", cwd], "")
    except (CommandError, OSError):
        return {}
    lines = output.splitlines()
    parts = lines[0].split("\t") if lines else []
    if len(parts) != 3:
        return {}
    slug, source, remote = parts
    resolved: dict[str, str] = {}
    if slug:
        resolved["project"] = slug
    if source:
        resolved["project_source"] = source
    if remote:
        resolved["remote"] = remote
    return resolved


def provenance_for(
    conversation: Conversation, args: argparse.Namespace, run: CommandRunner
) -> dict[str, object]:
    """Build the provenance object one conversation's write will carry.

    Explicit flags always win, which is how the hook passes what it already
    resolved from the `CLANKER_SESSION_*` markers it inherited. When
    `--project` was not given but the conversation itself names a working
    directory, that directory is resolved once through `clanker` -- the path
    T008's backfill relies on to recover project provenance for transcripts
    the hook never annotated.

    Args:
        conversation: The conversation being written; only `.cwd` is read.
        args: Parsed command-line arguments, carrying the provenance flags.
        run: The subprocess runner, for `clanker project resolve`.

    Returns:
        A `RawProvenance`-shaped mapping (`PROVENANCE_FIELDS`), holding only
        the fields that are actually known; empty when none are.
    """
    resolved: dict[str, str] = {}
    if args.project is None and conversation.cwd:
        resolved = resolve_project(conversation.cwd, run)

    provenance: dict[str, object] = {}
    project = args.project or resolved.get("project")
    if project:
        provenance["project"] = project
    project_source = args.project_source or resolved.get("project_source")
    if project_source:
        provenance["project_source"] = project_source
    remote = args.remote or resolved.get("remote")
    if remote:
        provenance["remote"] = remote
    if args.context:
        provenance["context"] = args.context
    if args.domains:
        provenance["domains"] = list(args.domains)
    if args.harness:
        provenance["harness"] = args.harness
    if args.model:
        provenance["model"] = args.model
    if args.clanker_session:
        provenance["session"] = args.clanker_session
    if conversation.cwd:
        provenance["cwd"] = conversation.cwd
    return provenance


@contextlib.contextmanager
def _provenance_file(provenance: dict[str, object]) -> Iterator[Path]:
    """Write `provenance` to a temporary JSON file for `--provenance-json`.

    The file, not stdin, is `kb create`'s provenance channel: stdin already
    carries the document body. Written and removed around a single `kb
    create` call, so nothing depends on this process's working directory and
    nothing is left behind on either success or failure.

    Args:
        provenance: The `RawProvenance`-shaped mapping to write.

    Yields:
        The temporary file's path.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="kb-provenance-", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(provenance, handle)
        path = Path(handle.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)
