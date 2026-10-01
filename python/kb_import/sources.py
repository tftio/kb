"""The four transcript parsers: Claude.ai, ChatGPT, Codex, and Claude Code.

Each source has its own on-disk shape; all four are mapped onto the common
[`Conversation`][kb_import.model.Conversation] model and rendered by one
function ([`render`][kb_import.model.render]), so a node imported from any
source is indistinguishable in form from any other.

Design notes:

* **Tool calls are dropped** from every source. That is the operator's stated
  contract for "verbatim", and it doubles as the main secret-hygiene measure:
  tool output is where environment dumps, file contents, and command output
  live.
* **Reasoning is kept** in a `:THINKING:` block. kb's AST has no generic drawer
  constructor, so this round-trips as a paragraph rather than as structure — the
  text and its marker lines both enter the embeddings. This was verified against
  a live kb before the format was chosen.
* **Provenance carries no grammar of its own (T007,
  `PLAN-20260923-project-identity`).** The importer never validates a project
  slug or normalizes a remote; a bare working directory is resolved by
  shelling out to `clanker project resolve --dir <cwd>` (see
  `kb_import.provenance`), which is the only thing here that links against
  `tftio_lib::project`, and kb's own `RawProvenance::validate` is what
  ultimately accepts or refuses whatever comes back. Only the Claude Code and
  Codex sources carry a working directory at all (`cwd` on a Claude Code JSONL
  record, `payload.cwd` on a Codex `session_meta` record); the Claude.ai and
  ChatGPT web exports carry none, because a browser conversation was never
  running "in" a directory. Claude Code records also carry `gitBranch`, which
  is read but not stored: kb's `Provenance` (`src/record.rs`) has no branch
  field, adding one is a Rust change outside this task's scope, and the plan
  left the decision to a later task rather than block capture on it -- see
  this task's completion evidence.

These parsers and their helpers perform no I/O and hold no state
(`REPO_INVARIANTS.md` ENG-008); the filesystem appears only in
`kb_import.cli`'s source discovery.
"""

from __future__ import annotations

from kb_import.model import (
    Conversation,
    Message,
    Role,
    TranscriptError,
    _as_dict,
    _as_list,
    _as_str,
    _epoch_to_iso,
)

# ── Claude.ai web export ───────────────────────────────────────────────────


def parse_claude_export(raw: object) -> list[Conversation]:
    """Parse a Claude.ai `conversations.json` export.

    Each entry carries `uuid`, `name`, `created_at`, and a flat `chat_messages`
    array whose `sender` is already `human` or `assistant`. Message content is a
    list of blocks typed `text`, `thinking`, `tool_use`, `tool_result`, or
    `flag`; only the first two survive. Older entries carry the prose in a
    top-level `text` field instead, which is used when the block list yields
    nothing.

    Node ids are the **bare** source uuid, with no prefix. 65 nodes from the
    2026-05 imports are already keyed that way, and prefixing would duplicate
    every one of them instead of updating it. This is the only source that is
    not prefixed, and the asymmetry is deliberate.

    Args:
        raw: The decoded contents of `conversations.json`.

    Returns:
        One `Conversation` per entry that has anything to render, in file order.
    """
    conversations: list[Conversation] = []
    for entry_raw in _as_list(raw):
        entry = _as_dict(entry_raw)
        uuid = _as_str(entry.get("uuid"))
        if not uuid:
            continue
        messages = [
            m
            for m in (_claude_message(_as_dict(r)) for r in _as_list(entry.get("chat_messages")))
            if m is not None
        ]
        conversation = Conversation(
            source="claude-export",
            source_id=uuid,
            node_id=uuid,
            title_hint=_as_str(entry.get("name")),
            account="",
            started_at=_as_str(entry.get("created_at")) or None,
            messages=tuple(messages),
            tags=("claude-export",),
        )
        if not conversation.is_empty:
            conversations.append(conversation)
    return conversations


def _claude_message(record: dict[str, object]) -> Message | None:
    """Map one Claude.ai `chat_messages` entry onto a `Message`.

    Args:
        record: The raw message object.

    Returns:
        The message, or `None` when its sender is unrecognized.
    """
    sender = _as_str(record.get("sender"))
    if sender not in ("human", "assistant"):
        return None
    role: Role = "human" if sender == "human" else "assistant"

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    for block_raw in _as_list(record.get("content")):
        block = _as_dict(block_raw)
        match _as_str(block.get("type")):
            case "text":
                text_parts.append(_as_str(block.get("text")))
            case "thinking":
                thinking_parts.append(_as_str(block.get("thinking")))
            case _:
                # tool_use, tool_result, and flag are dropped by contract.
                pass

    text = "\n\n".join(p for p in text_parts if p.strip()) or _as_str(record.get("text"))
    return Message(
        role=role,
        timestamp=_as_str(record.get("created_at")) or None,
        text=text,
        reasoning="\n\n".join(p for p in thinking_parts if p.strip()),
    )


# ── ChatGPT web export ─────────────────────────────────────────────────────


def parse_chatgpt_export(raw: object) -> list[Conversation]:
    """Parse a ChatGPT `conversations.json` export.

    `mapping` is a **tree** of message nodes, not a list: an edited prompt or a
    regenerated reply creates a sibling branch, and `current_node` names the
    leaf of the branch the user actually ended on. The conversation is recovered
    by walking `parent` links from `current_node` to the root and reversing.

    **Everything off that chain is discarded**, which is correct — abandoned
    branches are drafts the user replaced — but it is invisible in the output.
    A reader who does not know the format will otherwise take this parser for
    accidentally lossy and try to "repair" it. Note also that nodes in this
    export carry no `children` key at all, so walking downward is not merely
    worse, it is impossible.

    Args:
        raw: The decoded contents of `conversations.json`.

    Returns:
        One `Conversation` per entry that has anything to render, in file order.

    Raises:
        TranscriptError: If a conversation's `current_node` is absent from its
            own mapping. A silently truncated conversation is indistinguishable
            from a short one, so this is raised rather than absorbed.
    """
    conversations: list[Conversation] = []
    for entry_raw in _as_list(raw):
        entry = _as_dict(entry_raw)
        conversation_id = _as_str(entry.get("conversation_id")) or _as_str(entry.get("id"))
        if not conversation_id:
            continue
        mapping = _as_dict(entry.get("mapping"))
        chain = _chatgpt_chain(mapping, _as_str(entry.get("current_node")), conversation_id)
        messages = [m for m in (_chatgpt_message(node) for node in chain) if m is not None]
        conversation = Conversation(
            source="chatgpt-export",
            source_id=conversation_id,
            node_id=f"chatgpt-{conversation_id}",
            title_hint=_as_str(entry.get("title")),
            account="",
            started_at=_epoch_to_iso(entry.get("create_time")),
            messages=tuple(messages),
            tags=("chatgpt-export",),
        )
        if not conversation.is_empty:
            conversations.append(conversation)
    return conversations


def _chatgpt_chain(
    mapping: dict[str, object], current_node: str, conversation_id: str
) -> list[dict[str, object]]:
    """Recover the surviving branch of a ChatGPT conversation tree.

    Args:
        mapping: The conversation's node table.
        current_node: The id of the leaf the conversation ended on.
        conversation_id: Used only to attribute an error.

    Returns:
        The nodes from root to leaf, in chronological order.

    Raises:
        TranscriptError: If `current_node` is empty or names a node the mapping
            does not contain.
    """
    if not current_node or current_node not in mapping:
        raise TranscriptError(
            f"conversation {conversation_id}: current_node {current_node!r} is not in its mapping"
        )
    chain: list[dict[str, object]] = []
    seen: set[str] = set()
    cursor = current_node
    while cursor and cursor in mapping and cursor not in seen:
        seen.add(cursor)
        node = _as_dict(mapping[cursor])
        chain.append(node)
        cursor = _as_str(node.get("parent"))
    chain.reverse()
    return chain


def _chatgpt_message(node: dict[str, object]) -> Message | None:
    """Map one ChatGPT mapping node onto a `Message`.

    Args:
        node: A node from the conversation mapping.

    Returns:
        The message, or `None` for the root node, a system or tool turn, or a
        turn whose content type carries no readable text.
    """
    message = _as_dict(node.get("message"))
    if not message:
        return None
    author = _as_str(_as_dict(message.get("author")).get("role"))
    if author not in ("user", "assistant"):
        return None
    role: Role = "human" if author == "user" else "assistant"

    content = _as_dict(message.get("content"))
    content_type = _as_str(content.get("content_type"))
    text = ""
    reasoning = ""
    if content_type == "text":
        text = "\n\n".join(_as_str(p) for p in _as_list(content.get("parts")) if _as_str(p).strip())
    elif content_type == "reasoning_recap":
        reasoning = _as_str(content.get("content"))
    else:
        # Multimodal and tool content types carry no prose worth importing.
        return None

    return Message(
        role=role,
        timestamp=_epoch_to_iso(message.get("create_time")),
        text=text,
        reasoning=reasoning,
    )


# ── Codex CLI rollouts ─────────────────────────────────────────────────────


#: Prefixes that mark a `response_item`/`message` turn as injected
#: environment context (AGENTS.md contents, `<environment_context>`) rather
#: than something a human typed. These arrive with role `user`, so role alone
#: cannot exclude them; the `event_msg`/`user_message` path never saw them
#: because Codex does not surface them there.
_CODEX_INJECTED_PREFIXES = ("<environment_context>", "# AGENTS.md instructions for")


def _is_injected_codex_context(text: str) -> bool:
    """Report whether a `response_item` message turn is injected context.

    Args:
        text: The turn's extracted text.

    Returns:
        True when `text` opens with one of `_CODEX_INJECTED_PREFIXES`.
    """
    return text.strip().startswith(_CODEX_INJECTED_PREFIXES)


def _codex_response_item_text(payload: dict[str, object]) -> str:
    """Extract the prose of a `response_item`/`message` payload.

    Content parts are typed `input_text` (user/developer) or `output_text`
    (assistant); both carry their prose in a `text` field, which is all this
    reads. `input_image` parts carry no text and contribute nothing.

    Args:
        payload: The `message` payload.

    Returns:
        The joined text of every content part that has any, empty when none do.
    """
    return "\n\n".join(
        part
        for part in (
            _as_str(_as_dict(c).get("text")).strip() for c in _as_list(payload.get("content"))
        )
        if part
    )


def parse_codex_rollout(records: list[object], account: str) -> Conversation | None:
    """Parse a Codex `rollout-*.jsonl` session.

    Conversation text is preferentially taken from the `event_msg` records
    typed `user_message` and `agent_message`, **not** from
    `response_item`/`message`. Both carry assistant text, and using both
    duplicates every reply. The `event_msg` side was chosen on measurement
    over 61 sessions: it holds 408 user messages against `response_item`'s
    490, and the 82-record excess is synthetic feedback — tool output and
    environment context fed back to the model — rather than anything a human
    typed.

    Codex CLI releases at and after about 0.144 stop emitting
    `event_msg`/`user_message`/`agent_message` altogether; a session from one
    of those releases carries its turns only as `response_item`/`message`
    records (role `user` or `assistant`; role `developer` is a system-prompt
    injection and is dropped, mirroring what the `event_msg` path always
    excluded by construction). When a file has neither, this parser falls
    back to that path, extracting text from each content part's
    `input_text`/`output_text` field and dropping any user turn that opens
    with `<environment_context>` or `# AGENTS.md instructions for` --
    injected context that Codex renders as role `user` but that no human
    typed (see `_is_injected_codex_context`). When a file carries both
    formats, the `event_msg` turns are used and the `response_item` turns are
    ignored, so nothing is double-counted.

    Reasoning is largely unavailable here. A `reasoning` record's text lives in
    `encrypted_content`, which is opaque; only its `summary` is readable, and
    over those same 61 sessions just 84 of 3792 reasoning records had a
    non-empty one. Those that do are short headlines rather than prose. The
    parser takes what it can and does not pretend to more, and does not
    attempt to recover reasoning for the `response_item`-only fallback.

    Every rollout file is one Codex *thread*. A resumed session_meta record
    carries `session_id` (the resumed thread's own id, unchanged) plus
    `forked_from_id` naming the thread it was resumed from -- a distinct kb
    node linked back to its origin. A subagent fan-out is different: the
    subagent's `session_meta` carries the **parent's** `session_id` but its
    own `payload.id` (`thread_source: "subagent"`, or a `source.subagent`
    mapping on Codex CLI releases before that field existed); every subagent
    of one parent shares that `session_id`, so keying its node on
    `session_id` collides every sibling onto one record and the last file
    written wins. A subagent instead gets its own node,
    `codex-<own payload.id>`, tagged `codex-subagent` and linked to its
    parent thread.

    Args:
        records: The decoded JSONL records, in file order.
        account: The account segment of the session's path, used as a tag.

    Returns:
        The conversation, or `None` when the session holds no human or
        assistant turn.
    """
    session_id = ""
    own_id = ""
    is_subagent = False
    parent_thread_id: str | None = None
    forked_from_id: str | None = None
    started_at: str | None = None
    cwd: str | None = None
    event_messages: list[Message] = []
    response_messages: list[Message] = []
    pending_reasoning: list[str] = []

    for record_raw in records:
        record = _as_dict(record_raw)
        payload = _as_dict(record.get("payload"))
        timestamp = _as_str(record.get("timestamp")) or None
        record_type = record.get("type")

        if record_type == "session_meta":
            if session_id:
                # A subagent's rollout can replay its parent thread's own
                # session_meta later in the same file (history replay), with
                # no thread_source/source.subagent of its own -- identity
                # comes from the FIRST session_meta only, or that replay
                # would silently overwrite own_id/is_subagent and make the
                # file look like the parent's own thread again.
                continue
            session_id = _as_str(payload.get("session_id")) or _as_str(payload.get("id"))
            own_id = _as_str(payload.get("id")) or session_id
            started_at = _as_str(payload.get("timestamp")) or None
            cwd = _as_str(payload.get("cwd")) or None
            source = _as_dict(payload.get("source"))
            spawn = _as_dict(_as_dict(source.get("subagent")).get("thread_spawn"))
            is_subagent = _as_str(payload.get("thread_source")) == "subagent" or bool(spawn)
            if is_subagent:
                parent_thread_id = (
                    _as_str(payload.get("parent_thread_id"))
                    or _as_str(spawn.get("parent_thread_id"))
                    or None
                )
            else:
                forked_from_id = _as_str(payload.get("forked_from_id")) or None
            continue

        if record_type == "event_msg":
            match _as_str(payload.get("type")):
                case "user_message":
                    event_messages.append(
                        Message(
                            role="human", timestamp=timestamp, text=_as_str(payload.get("message"))
                        )
                    )
                    pending_reasoning.clear()
                case "agent_message":
                    event_messages.append(
                        Message(
                            role="assistant",
                            timestamp=timestamp,
                            text=_as_str(payload.get("message")),
                            reasoning="\n\n".join(pending_reasoning),
                        )
                    )
                    pending_reasoning.clear()
                case _:
                    pass
            continue

        if payload.get("type") == "reasoning":
            for part_raw in _as_list(payload.get("summary")):
                part = _as_str(_as_dict(part_raw).get("text")).strip()
                if part:
                    pending_reasoning.append(part)
            continue

        if record_type == "response_item" and payload.get("type") == "message":
            role = _as_str(payload.get("role"))
            if role not in ("user", "assistant"):
                continue
            text = _codex_response_item_text(payload)
            if not text or (role == "user" and _is_injected_codex_context(text)):
                continue
            response_messages.append(
                Message(
                    role="human" if role == "user" else "assistant", timestamp=timestamp, text=text
                )
            )

    if not session_id:
        return None

    messages = event_messages if event_messages else response_messages

    if is_subagent:
        node_id = f"codex-{own_id}"
        source_id = own_id
        tags: tuple[str, ...] = ("codex-session", "codex-subagent")
        related_id = f"codex-{parent_thread_id}" if parent_thread_id else None
        related_label = "Parent thread"
    else:
        node_id = f"codex-{session_id}"
        source_id = session_id
        tags = ("codex-session",)
        related_id = f"codex-{forked_from_id}" if forked_from_id else None
        related_label = "Resumed from"

    conversation = Conversation(
        source="codex",
        source_id=source_id,
        node_id=node_id,
        title_hint="",
        account=account,
        started_at=started_at,
        messages=tuple(messages),
        tags=tags,
        cwd=cwd,
        related_id=related_id,
        related_label=related_label,
    )
    return None if conversation.is_empty else conversation


# ── Claude Code CLI transcripts ────────────────────────────────────────────


def is_programmatic(records: list[object]) -> bool:
    """Report whether a Claude Code transcript came from a non-interactive run.

    Every record carries an `entrypoint`: `cli` for a session a human typed
    into, `sdk-cli` for one started by `claude -p` or the SDK. The distinction
    matters because the importer derives titles by shelling out to `claude -p`,
    and each of those calls used to leave a transcript that the next run then
    imported as if it were a conversation. Of 2725 transcripts on this machine
    when the defect was found, 2495 were `sdk-cli` and only 230 were real
    sessions.

    The test fails open. A transcript carrying no `entrypoint` at all is
    reported as interactive, so a future Claude Code release that renames the
    field degrades to over-collection rather than to silent data loss.

    Args:
        records: The decoded JSONL records, in file order.

    Returns:
        True when at least one entrypoint was seen and none of them was `cli`.
    """
    seen = {
        entrypoint
        for entrypoint in (_as_str(_as_dict(record).get("entrypoint")) for record in records)
        if entrypoint
    }
    return bool(seen) and "cli" not in seen


def parse_claude_code_transcript(
    records: list[object], session_id: str, account: str, include_programmatic: bool = False
) -> Conversation | None:
    """Parse a Claude Code per-session JSONL transcript.

    **One line is not one message.** A streamed assistant reply is split across
    several records sharing a `message.id`; in one sampled transcript 521 of
    2309 message groups spanned more than one record, up to six. Grouping by
    message id before rendering is what keeps a reply whole instead of
    shattering it into fragments. The existing `SessionEnd` hook's own distiller
    prompt warns about exactly this.

    Node ids are always prefixed `cc-`. The bare session id is occupied by the
    1148 summary nodes the hook wrote under its previous contract, and writing
    there would overwrite them.

    Args:
        records: The decoded JSONL records, in file order.
        session_id: The session id, taken from the transcript's filename.
        account: The account the session belongs to, used as a tag.
        include_programmatic: Import `sdk-cli` sessions too. Off by default;
            see `is_programmatic` for why they are refused.

    Returns:
        The conversation, or `None` when nothing survives tool-call removal or
        when the transcript is programmatic and not explicitly included.
    """
    if not include_programmatic and is_programmatic(records):
        return None

    title_hint = ""
    groups: dict[str, list[dict[str, object]]] = {}
    order: list[str] = []

    for record_raw in records:
        record = _as_dict(record_raw)
        record_type = _as_str(record.get("type"))
        if record_type == "ai-title":
            # Claude Code generates its own session title. Preferring it saves
            # a title-derivation call per session.
            title_hint = _as_str(record.get("aiTitle")) or title_hint
            continue
        if record_type not in ("user", "assistant"):
            continue
        if record.get("isMeta") or record.get("isSidechain"):
            # Meta records are system-injected context; sidechain records
            # belong to a subagent, not to this conversation.
            continue
        message = _as_dict(record.get("message"))
        key = _as_str(message.get("id")) or _as_str(record.get("uuid"))
        if not key:
            continue
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)

    messages = [m for m in (_claude_code_message(groups[k]) for k in order) if m is not None]
    conversation = Conversation(
        source="claude-code",
        source_id=session_id,
        node_id=f"cc-{session_id}",
        title_hint=title_hint,
        account=account,
        started_at=_first_timestamp(groups, order),
        messages=tuple(messages),
        tags=("claude-code-session",),
        cwd=_first_cwd(records),
    )
    return None if conversation.is_empty else conversation


def _first_cwd(records: list[object]) -> str | None:
    """Find the first working directory recorded in a Claude Code transcript.

    Every `user`/`assistant` record carries a top-level `cwd`, set once for
    the whole session rather than per turn, so the first one seen is the
    session's. `gitBranch` sits beside it on the same records; it is read
    nowhere in this module -- see the module docstring's judgment-call note.

    Args:
        records: The decoded JSONL records, in file order.

    Returns:
        The first non-empty `cwd` seen, or `None`.
    """
    for record_raw in records:
        cwd = _as_str(_as_dict(record_raw).get("cwd"))
        if cwd:
            return cwd
    return None


def _first_timestamp(groups: dict[str, list[dict[str, object]]], order: list[str]) -> str | None:
    """Find the earliest record timestamp in a grouped transcript.

    Args:
        groups: Records grouped by message id.
        order: Group keys in first-seen order.

    Returns:
        The first timestamp present, or `None`.
    """
    for key in order:
        for record in groups[key]:
            stamp = _as_str(record.get("timestamp"))
            if stamp:
                return stamp
    return None


def _claude_code_message(group: list[dict[str, object]]) -> Message | None:
    """Merge one message id's records into a single `Message`.

    Args:
        group: Every record sharing one `message.id`, in file order.

    Returns:
        The merged message, or `None` when the group is unusable or held only
        tool traffic.
    """
    first = group[0]
    record_type = _as_str(first.get("type"))
    if record_type not in ("user", "assistant"):
        return None
    role: Role = "human" if record_type == "user" else "assistant"

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    for record in group:
        content = _as_dict(record.get("message")).get("content")
        if isinstance(content, str):
            # User turns are sometimes a bare string rather than a block list.
            text_parts.append(content)
            continue
        for block_raw in _as_list(content):
            block = _as_dict(block_raw)
            match _as_str(block.get("type")):
                case "text":
                    text_parts.append(_as_str(block.get("text")))
                case "thinking":
                    thinking_parts.append(_as_str(block.get("thinking")))
                case _:
                    # tool_use and tool_result are dropped by contract.
                    pass

    return Message(
        role=role,
        timestamp=_as_str(first.get("timestamp")) or None,
        text="\n\n".join(p for p in text_parts if p.strip()),
        reasoning="\n\n".join(p for p in thinking_parts if p.strip()),
    )
