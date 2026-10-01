"""The subprocess boundary and the title-derivation layer.

Everything here performs I/O -- this is what sits below the parsers and
`render` in `kb_import.model` and `kb_import.sources`, which perform none and
are what lets the whole parsing surface be tested without a database, an
endpoint, or a network (`REPO_INVARIANTS.md` ENG-008).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from kb_import.model import ROLE_LABELS, Conversation, _as_dict, _truncate


class CommandError(Exception):
    """A subprocess exited non-zero."""


#: A subprocess runner: argv and stdin in, stdout out, `CommandError` on
#: failure. Injected rather than called directly so the driver and the title
#: deriver can be tested without spawning anything.
type CommandRunner = Callable[[Sequence[str], str], str]

#: Model used to derive titles. An alias rather than a pinned name, so the
#: derivation follows the current small model instead of pinning to one that
#: will be retired.
DEFAULT_TITLE_MODEL = "haiku"

#: How much of a conversation the title deriver sees. A title is determined by
#: the opening exchange; feeding whole transcripts would cost tokens without
#: improving the result.
TITLE_PROMPT_CHARS = 2000


TITLE_INSTRUCTION = (
    "Below is the opening of a saved conversation transcript. Reply with a single "
    "specific title for it, under 80 characters, naming the actual subject rather "
    "than describing it as a conversation. No quotes, no preamble, no trailing "
    "punctuation. Reply with the title alone.\n\n"
)


def run_command(argv: Sequence[str], stdin: str = "") -> str:
    """Run a subprocess and return its stdout.

    Args:
        argv: The command and its arguments.
        stdin: Text to write to the process's standard input.

    Returns:
        The process's standard output.

    Raises:
        CommandError: If the process exits non-zero, carrying its stderr.
    """
    result = subprocess.run(list(argv), input=stdin, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CommandError(f"{argv[0]} exited {result.returncode}: {detail}")
    return result.stdout


class TitleCache:
    """Derived titles, remembered across runs.

    The cache is what makes LLM-derived titles affordable and reproducible.
    Without it every re-run pays a call per conversation and produces different
    titles, so an idempotent re-import would still rewrite every node.

    Writes are atomic — a temporary file in the same directory, then a rename —
    because the import is explicitly interruptible and a half-written JSON cache
    would lose every title derived so far.
    """

    def __init__(self, path: Path) -> None:
        """Load the cache at `path`, treating an unreadable one as empty.

        Args:
            path: Where the cache lives.
        """
        self.path = path
        self._titles: dict[str, str] = {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self._titles = {k: v for k, v in _as_dict(raw).items() if isinstance(v, str)}

    def get(self, node_id: str) -> str | None:
        """Look up a remembered title.

        Args:
            node_id: The node the title belongs to.

        Returns:
            The cached title, or `None`.
        """
        return self._titles.get(node_id)

    def set(self, node_id: str, title: str) -> None:
        """Remember a title.

        Args:
            node_id: The node the title belongs to.
            title: The derived title.
        """
        self._titles[node_id] = title

    def save(self) -> None:
        """Write the cache to disk atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._titles, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)


def derive_title(
    conversation: Conversation,
    cache: TitleCache,
    run: CommandRunner,
    model: str = DEFAULT_TITLE_MODEL,
) -> str:
    """Produce a title for `conversation`, deriving one if it is not cached.

    A derivation failure is not fatal. `claude` being absent, rate-limited, or
    slow degrades to the source-supplied title and then to the opening line of
    the first human turn, so a title outage costs quality rather than the
    import.

    Args:
        conversation: The conversation to title.
        cache: The title cache, consulted first and updated on success.
        run: The subprocess runner.
        model: Model alias passed to `claude -p`.

    Returns:
        A non-empty title.
    """
    cached = cache.get(conversation.node_id)
    if cached:
        return cached
    try:
        # `--no-session-persistence` is not optional. Without it every title
        # derivation leaves a transcript in the project directory, which the
        # next import then ingests as a conversation -- a feedback loop that
        # put ~1166 machine-generated nodes into the corpus before it was
        # caught. Titles are one-shot and never resumed, so nothing is lost.
        raw = run(
            ["claude", "-p", "--no-session-persistence", "--model", model],
            _title_prompt(conversation),
        )
    except (CommandError, OSError):
        return conversation.fallback_title
    title = _truncate(raw.strip().strip("\"'").splitlines()[0]) if raw.strip() else ""
    if not title:
        return conversation.fallback_title
    cache.set(conversation.node_id, title)
    return title


def _title_prompt(conversation: Conversation) -> str:
    """Build the title-derivation prompt.

    Args:
        conversation: The conversation to describe.

    Returns:
        The instruction followed by the opening of the transcript.
    """
    body: list[str] = []
    for message in conversation.renderable_messages:
        body.append(f"{ROLE_LABELS[message.role]}: {message.text.strip()}")
        if sum(len(part) for part in body) > TITLE_PROMPT_CHARS:
            break
    return TITLE_INSTRUCTION + "\n\n".join(body)[:TITLE_PROMPT_CHARS]
