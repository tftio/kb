"""The conversation model and its renderer, common to all four sources.

Design notes:

* **Purity.** These parsers and [`render`][] perform no I/O and hold no state.
  Subprocess calls, the filesystem, and the database appear only in
  `kb_import.store`, `kb_import.titles`, `kb_import.provenance`, and
  `kb_import.ingest`. This is `REPO_INVARIANTS.md` ENG-008, and it is what
  makes the parsers testable without a database or an endpoint.
* **`#+title:` is mandatory.** kb's `extract_title` prefers that keyword and
  otherwise falls back to the first heading, which here would title every node
  `Human [<timestamp>]`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

Role = Literal["human", "assistant"]


class TranscriptError(Exception):
    """A transcript could not be parsed.

    Raised only for input that is structurally broken in a way that would
    otherwise yield a silently partial conversation. Ordinary absences — a
    missing timestamp, an empty turn, a conversation of pure tool traffic — are
    represented in the model rather than raised.
    """


# ── Untrusted-input accessors ──────────────────────────────────────────────
# Export files are external data. These narrow `object` to a usable type at the
# boundary and return an empty value rather than raising, so one malformed
# record cannot abort a 774-file import (REPO_INVARIANTS.md ENG-006).


def _as_dict(value: object) -> dict[str, object]:
    """Narrow a JSON value to a mapping.

    Args:
        value: A value decoded from untrusted JSON.

    Returns:
        The mapping, or an empty one when `value` is not a mapping.
    """
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    """Narrow a JSON value to a list.

    Args:
        value: A value decoded from untrusted JSON.

    Returns:
        The list, or an empty one when `value` is not a list.
    """
    return value if isinstance(value, list) else []


def _as_str(value: object) -> str:
    """Narrow a JSON value to a string.

    Args:
        value: A value decoded from untrusted JSON.

    Returns:
        The string, or an empty one when `value` is not a string.
    """
    return value if isinstance(value, str) else ""


def _epoch_to_iso(value: object) -> str | None:
    """Convert an epoch-seconds timestamp to an ISO-8601 instant.

    ChatGPT records `create_time` as a float; every other source records an ISO
    string already.

    Args:
        value: The raw timestamp value.

    Returns:
        An ISO-8601 string in UTC, or `None` when `value` is not a usable
        number.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return datetime.fromtimestamp(float(value), tz=UTC).isoformat().replace("+00:00", "Z")


#: Tag carried by every imported node regardless of source.
COMMON_TAG = "conversation"

#: Heading label per role. These are the labels the 2026-05 Claude.ai import
#: used, and 65 existing nodes already follow them; changing them would split
#: the corpus into two rendering conventions.
ROLE_LABELS: dict[Role, str] = {"human": "Human", "assistant": "Assistant"}


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of a conversation, stripped of tool traffic.

    Attributes:
        role: Who produced the turn.
        timestamp: ISO-8601 instant the turn was produced, or `None` when the
            source records none. Codex rollouts and some Claude Code records
            omit per-message timestamps.
        text: The prose of the turn. May be empty when a turn carried only
            reasoning.
        reasoning: Model reasoning associated with the turn, empty when absent.
    """

    role: Role
    timestamp: str | None
    text: str
    reasoning: str = ""

    @property
    def is_renderable(self) -> bool:
        """Whether this message contributes anything to a rendered document.

        Returns:
            True when the message carries prose or reasoning. A turn that held
            only tool traffic arrives here empty and renders to nothing.
        """
        return bool(self.text.strip() or self.reasoning.strip())


@dataclass(frozen=True, slots=True)
class Conversation:
    """One conversation from any source, normalized.

    Attributes:
        source: Short source identifier, used for reporting and tagging.
        source_id: The identifier the source itself uses.
        node_id: The kb node id this conversation is written to. Derived from
            `source_id` by a per-source rule; see the module docstring of each
            parser for why a given source is or is not prefixed.
        title_hint: A title supplied by the source, empty when it supplies none.
            Used as the first fallback when title derivation fails.
        account: The account the conversation belongs to, empty for web exports
            which are inherently single-account.
        started_at: ISO-8601 instant the conversation began, or `None`.
        messages: The turns, in chronological order.
        tags: Tags to attach, excluding the account, which is added by the
            driver.
        cwd: The working directory the session ran in, when the source
            records one. `None` for the Claude.ai and ChatGPT web exports,
            which carry no notion of a directory at all.
        related_id: The node id of a related thread -- a Codex subagent's
            parent, or a resumed thread's origin -- or `None` when this
            conversation has none. Rendered as an `[[id:...]]` link so `kb
            links` surfaces the relationship.
        related_label: The label to render before `related_id`'s link.
            Meaningless when `related_id` is `None`.
    """

    source: str
    source_id: str
    node_id: str
    title_hint: str
    account: str
    started_at: str | None
    messages: tuple[Message, ...]
    tags: tuple[str, ...] = field(default=())
    cwd: str | None = None
    related_id: str | None = None
    related_label: str = ""

    @property
    def renderable_messages(self) -> tuple[Message, ...]:
        """The messages that will appear in the rendered document.

        Returns:
            Every message carrying prose or reasoning, in order.
        """
        return tuple(m for m in self.messages if m.is_renderable)

    @property
    def is_empty(self) -> bool:
        """Whether this conversation would render to no content at all.

        Returns:
            True when nothing survives tool-call removal. Such a conversation
            yields no node rather than an empty one.
        """
        return not self.renderable_messages

    @property
    def fallback_title(self) -> str:
        """The title to use when derivation is unavailable or fails.

        Prefers the source's own title; failing that, the opening line of the
        first human turn. Truncated to kb's 80-character title limit.

        Returns:
            A non-empty title string.
        """
        if self.title_hint.strip():
            return _truncate(self.title_hint.strip())
        for message in self.renderable_messages:
            if message.role == "human" and message.text.strip():
                return _truncate(message.text.strip().splitlines()[0])
        return _truncate(f"{self.source} conversation {self.source_id}")


def _truncate(text: str, limit: int = 80) -> str:
    """Truncate to kb's title limit.

    Args:
        text: The text to truncate.
        limit: Maximum length; kb's `extract_title` truncates at 80.

    Returns:
        `text` unchanged when short enough, else its first `limit` characters.
    """
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit]


def escape_org(text: str) -> str:
    """Neutralize org structure that transcript prose would otherwise become.

    A conversation about org-mode, about kb, or about this importer contains
    lines that *look* like org markup, and a verbatim renderer puts them at
    column zero where org markup lives. kb then parses them as structure rather
    than as text.

    That is not hypothetical. The 2026-08-06 import wrote 221 nodes carrying
    tags they never should have had, 220 of them tagged `topic-tags-lowercase`
    from the old SessionEnd prompt's own template line
    (`* <concise specific title> :<topic-tags-lowercase>:`), and one transcript
    quoting a generated node re-tagged itself `agent-summary` — the very tag the
    id scheme was designed to protect. One node failed to store outright, kb
    rejecting a malformed tag its content had produced.

    Only column zero is structural, so a single leading space is enough, and it
    is invisible in every org renderer. Two prefixes matter:

    * `*` opens a heading, and a heading's trailing `:a:b:` becomes real tags.
    * `#+` opens a keyword, and `#+filetags:` injects tags directly.

    List bullets and table pipes are left alone: they parse as prose-bearing
    blocks, carry no tags, and escaping them would corrupt legitimate formatting
    in the archived text for no gain.

    Args:
        text: One message's prose or reasoning, verbatim.

    Returns:
        The same text with structural lines shifted one column right.
    """
    return "\n".join(
        f" {line}" if line.startswith(("*", "#+")) else line for line in text.splitlines()
    )


def _heading(message: Message) -> str:
    """Render one message's org heading line.

    Args:
        message: The message to label.

    Returns:
        A level-one heading naming the role, with the timestamp when the source
        supplied one.
    """
    label = ROLE_LABELS[message.role]
    return f"* {label} [{message.timestamp}]" if message.timestamp else f"* {label}"


def render(conversation: Conversation, title: str) -> str:
    """Render a conversation as kb-ready org text.

    Each message becomes a level-one heading, which gives kb's chunker a break
    point at every turn: `embed_text::collect_units` pushes a heading's title as
    its own unit before recursing into its children, so a section boundary is
    always an available seam for the packer.

    Args:
        conversation: The conversation to render. Must not be empty; callers
            check `is_empty` first.
        title: The node title, written to the mandatory `#+title:` line.

    Returns:
        Org text ready to pipe to `kb create` or `kb update`.

    Raises:
        ValueError: If `conversation` has no renderable message, or `title` is
            blank. Both would produce a node kb cannot title correctly.
    """
    if conversation.is_empty:
        raise ValueError(f"conversation {conversation.source_id} has no renderable message")
    if not title.strip():
        raise ValueError(f"conversation {conversation.source_id} was given a blank title")

    tags = [COMMON_TAG, *conversation.tags]
    if conversation.account:
        tags.append(conversation.account)

    lines = [f"#+title: {_truncate(title)}", f"#+filetags: :{':'.join(tags)}:", ""]
    if conversation.related_id:
        lines.extend([f"{conversation.related_label}: [[id:{conversation.related_id}]]", ""])
    for message in conversation.renderable_messages:
        lines.append(_heading(message))
        if message.reasoning.strip():
            lines.extend([":THINKING:", escape_org(message.reasoning.strip()), ":END:", ""])
        if message.text.strip():
            lines.extend([escape_org(message.text.strip()), ""])
    return "\n".join(lines).rstrip() + "\n"
