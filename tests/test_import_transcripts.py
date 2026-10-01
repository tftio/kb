"""Tests for the transcript importer's model and renderer.

The importer is the installed `kb_import` package. Every test here runs
without a database, an embedding endpoint, or a network, which is what keeps
the parsers cheap to exercise.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from kb_import import cli as kbi
from kb_import import ingest, model, provenance, sources, store, titles


def _conversation(
    *messages: model.Message,
    title_hint: str = "",
    account: str = "personal",
    tags: tuple[str, ...] = ("codex-session",),
) -> model.Conversation:
    """Build a Conversation with sensible defaults for the fields under test."""
    return model.Conversation(
        source="codex",
        source_id="abc123",
        node_id="codex-abc123",
        title_hint=title_hint,
        account=account,
        started_at="2026-01-01T00:00:00Z",
        messages=messages,
        tags=tags,
    )


def test_a_message_holding_only_tool_traffic_is_not_renderable() -> None:
    assert not model.Message(
        role="assistant", timestamp=None, text="   ", reasoning=""
    ).is_renderable


def test_reasoning_alone_makes_a_message_renderable() -> None:
    message = model.Message(role="assistant", timestamp=None, text="", reasoning="thought")
    assert message.is_renderable


def test_a_conversation_of_only_tool_traffic_is_empty() -> None:
    empty = model.Message(role="assistant", timestamp=None, text="", reasoning="")
    assert _conversation(empty).is_empty


def test_render_refuses_an_empty_conversation() -> None:
    with pytest.raises(ValueError, match="no renderable message"):
        model.render(_conversation(), "a title")


def test_render_refuses_a_blank_title() -> None:
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hello"))
    with pytest.raises(ValueError, match="blank title"):
        model.render(conversation, "   ")


def test_every_document_opens_with_a_title_keyword() -> None:
    """Guard kb's title precedence.

    Without the keyword, kb's `extract_title` falls back to the first heading
    and titles every imported node `Human [<timestamp>]`.
    """
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hello"))
    assert model.render(conversation, "A specific title").startswith("#+title: A specific title\n")


def test_the_title_is_truncated_to_kbs_eighty_character_limit() -> None:
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hello"))
    rendered = model.render(conversation, "x" * 200)
    assert rendered.splitlines()[0] == "#+title: " + "x" * 80


def test_filetags_carry_the_common_tag_the_source_tag_and_the_account() -> None:
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hello"))
    assert ":conversation:codex-session:personal:" in model.render(conversation, "t")


def test_a_web_export_without_an_account_emits_no_account_tag() -> None:
    conversation = _conversation(
        model.Message(role="human", timestamp=None, text="hello"),
        account="",
        tags=("claude-export",),
    )
    assert "#+filetags: :conversation:claude-export:\n" in model.render(conversation, "t")


def test_each_message_becomes_its_own_heading() -> None:
    """Headings are what give kb's chunker a seam at every turn."""
    conversation = _conversation(
        model.Message(role="human", timestamp="2026-01-01T00:00:00Z", text="q"),
        model.Message(role="assistant", timestamp="2026-01-01T00:00:05Z", text="a"),
    )
    rendered = model.render(conversation, "t")
    assert "* Human [2026-01-01T00:00:00Z]" in rendered
    assert "* Assistant [2026-01-01T00:00:05Z]" in rendered


def test_a_message_without_a_timestamp_still_gets_a_heading() -> None:
    conversation = _conversation(model.Message(role="human", timestamp=None, text="q"))
    assert "* Human\n" in model.render(conversation, "t")


def test_reasoning_renders_in_a_thinking_block_before_the_prose() -> None:
    conversation = _conversation(
        model.Message(role="assistant", timestamp=None, text="the reply", reasoning="the thought")
    )
    rendered = model.render(conversation, "t")
    assert ":THINKING:\nthe thought\n:END:" in rendered
    assert rendered.index(":END:") < rendered.index("the reply")


def test_unrenderable_messages_are_omitted_from_the_document() -> None:
    conversation = _conversation(
        model.Message(role="human", timestamp=None, text="kept"),
        model.Message(role="assistant", timestamp=None, text="", reasoning=""),
    )
    assert model.render(conversation, "t").count("* ") == 1


def test_fallback_title_prefers_the_source_title() -> None:
    conversation = _conversation(
        model.Message(role="human", timestamp=None, text="the first thing said"),
        title_hint="A human-written title",
    )
    assert conversation.fallback_title == "A human-written title"


def test_fallback_title_uses_the_first_human_line_when_the_source_has_none() -> None:
    conversation = _conversation(
        model.Message(role="assistant", timestamp=None, text="assistant speaks first"),
        model.Message(role="human", timestamp=None, text="the question\nand more"),
    )
    assert conversation.fallback_title == "the question"


def test_fallback_title_is_never_empty() -> None:
    conversation = _conversation(model.Message(role="assistant", timestamp=None, text="only me"))
    assert conversation.fallback_title.strip()


def test_the_cli_parses_its_flags() -> None:
    args = kbi.build_parser().parse_args(["--source", "codex", "--dry-run", "--limit", "5"])
    assert args.source == "codex"
    assert args.dry_run
    assert args.limit == 5


def test_the_cli_parses_its_provenance_flags() -> None:
    args = kbi.build_parser().parse_args(
        [
            "--project",
            "kb",
            "--project-source",
            "remote",
            "--remote",
            "github.com/tftio/kb",
            "--context",
            "personal",
            "--domain",
            "eng",
            "--domain",
            "infra",
            "--harness",
            "claude",
            "--model",
            "sonnet",
            "--clanker-session",
            "01a0cef5-8afd-75b0-bccd-9eb65e5e10dc",
        ]
    )
    assert args.project == "kb"
    assert args.project_source == "remote"
    assert args.remote == "github.com/tftio/kb"
    assert args.context == "personal"
    assert args.domains == ["eng", "infra"]
    assert args.harness == "claude"
    assert args.model == "sonnet"
    assert args.clanker_session == "01a0cef5-8afd-75b0-bccd-9eb65e5e10dc"


def test_the_provenance_flags_default_to_none() -> None:
    args = kbi.build_parser().parse_args([])
    assert args.project is None
    assert args.project_source is None
    assert args.remote is None
    assert args.context is None
    assert args.domains is None
    assert args.harness is None
    assert args.model is None
    assert args.clanker_session is None


# ── Claude.ai web export ───────────────────────────────────────────────────

CLAUDE_EXPORT: list[object] = [
    {
        "uuid": "11111111-2222-3333-4444-555555555555",
        "name": "Choosing a kettle",
        "created_at": "2026-03-01T09:00:00Z",
        "chat_messages": [
            {
                "sender": "human",
                "created_at": "2026-03-01T09:00:00Z",
                "text": "",
                "content": [{"type": "text", "text": "Which kettle should I buy?"}],
            },
            {
                "sender": "assistant",
                "created_at": "2026-03-01T09:00:30Z",
                "text": "",
                "content": [
                    {"type": "thinking", "thinking": "weighing capacity against spout control"},
                    {"type": "tool_use", "name": "search", "input": {"q": "kettles"}},
                    {"type": "tool_result", "content": "a large amount of tool output"},
                    {"type": "text", "text": "A gooseneck, if you care about pour rate."},
                ],
            },
        ],
    }
]


def test_claude_export_parses_field_by_field() -> None:
    (conversation,) = sources.parse_claude_export(CLAUDE_EXPORT)
    assert conversation.source == "claude-export"
    assert conversation.source_id == "11111111-2222-3333-4444-555555555555"
    assert conversation.title_hint == "Choosing a kettle"
    assert conversation.account == ""
    assert conversation.started_at == "2026-03-01T09:00:00Z"
    assert conversation.tags == ("claude-export",)
    first, second = conversation.renderable_messages
    assert (first.role, first.text) == ("human", "Which kettle should I buy?")
    assert second.role == "assistant"
    assert second.text == "A gooseneck, if you care about pour rate."
    assert second.reasoning == "weighing capacity against spout control"


def test_claude_export_node_id_is_the_bare_uuid() -> None:
    """65 nodes are already keyed at the bare uuid; a prefix would duplicate them."""
    (conversation,) = sources.parse_claude_export(CLAUDE_EXPORT)
    assert conversation.node_id == "11111111-2222-3333-4444-555555555555"


def test_claude_export_drops_tool_traffic_from_the_rendered_document() -> None:
    (conversation,) = sources.parse_claude_export(CLAUDE_EXPORT)
    rendered = model.render(conversation, "t")
    assert "a large amount of tool output" not in rendered
    assert "search" not in rendered


def test_claude_export_falls_back_to_the_top_level_text_field() -> None:
    """Older entries carry prose in `text` rather than in a block list."""
    raw: list[object] = [
        {
            "uuid": "u1",
            "name": "",
            "chat_messages": [{"sender": "human", "text": "legacy prose", "content": []}],
        }
    ]
    (conversation,) = sources.parse_claude_export(raw)
    assert conversation.renderable_messages[0].text == "legacy prose"


def test_a_claude_conversation_of_only_tool_traffic_yields_nothing() -> None:
    raw: list[object] = [
        {
            "uuid": "u1",
            "name": "",
            "chat_messages": [
                {"sender": "assistant", "text": "", "content": [{"type": "tool_use", "name": "x"}]}
            ],
        }
    ]
    assert sources.parse_claude_export(raw) == []


# ── ChatGPT web export ─────────────────────────────────────────────────────


def _chatgpt_node(node_id: str, parent: str | None, role: str, text: str) -> dict[str, object]:
    """Build one ChatGPT mapping node."""
    return {
        "id": node_id,
        "parent": parent,
        "message": {
            "id": node_id,
            "author": {"role": role},
            "create_time": 1772000000.0,
            "content": {"content_type": "text", "parts": [text]},
        },
    }


CHATGPT_EXPORT: list[object] = [
    {
        "conversation_id": "conv-1",
        "title": "Kettles again",
        "create_time": 1772000000.0,
        "current_node": "n3",
        "mapping": {
            "root": {"id": "root", "parent": None, "message": None},
            "n1": _chatgpt_node("n1", "root", "user", "which kettle?"),
            "n2-abandoned": _chatgpt_node("n2-abandoned", "n1", "assistant", "A DISCARDED DRAFT"),
            "n3": _chatgpt_node("n3", "n1", "assistant", "the surviving reply"),
        },
    }
]


def test_chatgpt_export_keeps_only_the_surviving_branch() -> None:
    """Keep only the branch the user ended on.

    Edits and regenerations create sibling branches; only the `current_node`
    ancestor chain is the conversation the user actually ended on.
    """
    (conversation,) = sources.parse_chatgpt_export(CHATGPT_EXPORT)
    texts = [m.text for m in conversation.renderable_messages]
    assert texts == ["which kettle?", "the surviving reply"]
    assert "A DISCARDED DRAFT" not in model.render(conversation, "t")


def test_chatgpt_export_parses_field_by_field() -> None:
    (conversation,) = sources.parse_chatgpt_export(CHATGPT_EXPORT)
    assert conversation.source == "chatgpt-export"
    assert conversation.source_id == "conv-1"
    assert conversation.node_id == "chatgpt-conv-1"
    assert conversation.title_hint == "Kettles again"
    assert conversation.tags == ("chatgpt-export",)
    assert conversation.renderable_messages[0].role == "human"


def test_chatgpt_messages_are_chronological_root_first() -> None:
    (conversation,) = sources.parse_chatgpt_export(CHATGPT_EXPORT)
    assert conversation.renderable_messages[0].text == "which kettle?"


def test_a_dangling_current_node_is_attributed_to_its_conversation() -> None:
    """A silently truncated conversation is indistinguishable from a short one."""
    raw: list[object] = [
        {"conversation_id": "conv-broken", "current_node": "missing", "mapping": {"root": {}}}
    ]
    with pytest.raises(model.TranscriptError, match="conv-broken"):
        sources.parse_chatgpt_export(raw)


def test_chatgpt_reasoning_recap_becomes_reasoning() -> None:
    raw: list[object] = [
        {
            "conversation_id": "c",
            "current_node": "n1",
            "mapping": {
                "n1": {
                    "id": "n1",
                    "parent": None,
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"content_type": "reasoning_recap", "content": "a recap"},
                    },
                }
            },
        }
    ]
    (conversation,) = sources.parse_chatgpt_export(raw)
    assert conversation.renderable_messages[0].reasoning == "a recap"


# ── Codex CLI rollouts ─────────────────────────────────────────────────────

CODEX_ROLLOUT: list[object] = [
    {
        "timestamp": "2026-07-20T11:47:09Z",
        "type": "session_meta",
        "payload": {"session_id": "sess-1", "timestamp": "2026-07-20T11:47:02Z", "cwd": "/tmp"},
    },
    {
        "timestamp": "2026-07-20T11:47:10Z",
        "type": "event_msg",
        "payload": {"type": "user_message", "message": "check the runner setup"},
    },
    {
        "timestamp": "2026-07-20T11:47:12Z",
        "type": "response_item",
        "payload": {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "a thought"}],
        },
    },
    {
        "timestamp": "2026-07-20T11:47:20Z",
        "type": "response_item",
        "payload": {"type": "custom_tool_call", "name": "shell", "input": "SECRET TOOL INPUT"},
    },
    {
        "timestamp": "2026-07-20T11:47:21Z",
        "type": "response_item",
        "payload": {"type": "custom_tool_call_output", "output": "SECRET TOOL OUTPUT"},
    },
    {
        "timestamp": "2026-07-20T11:47:30Z",
        "type": "response_item",
        "payload": {"type": "message", "role": "assistant", "content": [{"text": "the reply"}]},
    },
    {
        "timestamp": "2026-07-20T11:47:30Z",
        "type": "event_msg",
        "payload": {"type": "agent_message", "message": "the reply"},
    },
    {
        "timestamp": "2026-07-20T11:47:31Z",
        "type": "response_item",
        "payload": {"type": "message", "role": "developer", "content": [{"text": "SYSTEM PROMPT"}]},
    },
]


def test_codex_parses_field_by_field() -> None:
    conversation = sources.parse_codex_rollout(CODEX_ROLLOUT, "personal")
    assert conversation is not None
    assert conversation.source == "codex"
    assert conversation.source_id == "sess-1"
    assert conversation.node_id == "codex-sess-1"
    assert conversation.account == "personal"
    assert conversation.started_at == "2026-07-20T11:47:02Z"
    assert conversation.cwd == "/tmp"
    assert conversation.tags == ("codex-session",)
    human, assistant = conversation.renderable_messages
    assert (human.role, human.text) == ("human", "check the runner setup")
    assert (assistant.role, assistant.text) == ("assistant", "the reply")
    assert assistant.reasoning == "a thought"


def test_codex_assistant_text_is_not_duplicated() -> None:
    """Guard against double-counting assistant turns.

    `response_item/message` and `event_msg/agent_message` both carry assistant
    text; emitting both would double every reply.
    """
    conversation = sources.parse_codex_rollout(CODEX_ROLLOUT, "personal")
    assert conversation is not None
    assert [m.text for m in conversation.renderable_messages].count("the reply") == 1


def test_codex_drops_developer_prompts_and_tool_traffic() -> None:
    conversation = sources.parse_codex_rollout(CODEX_ROLLOUT, "personal")
    assert conversation is not None
    rendered = model.render(conversation, "t")
    assert "SYSTEM PROMPT" not in rendered
    assert "SECRET TOOL INPUT" not in rendered
    assert "SECRET TOOL OUTPUT" not in rendered


def test_codex_reasoning_without_a_readable_summary_is_simply_absent() -> None:
    """Treat unreadable reasoning as absent.

    Reasoning text lives in `encrypted_content`; only 84 of 3792 sampled
    records had a readable `summary`, so absence is the normal case.
    """
    records: list[object] = [
        {"type": "session_meta", "payload": {"session_id": "s"}},
        {
            "type": "response_item",
            "payload": {"type": "reasoning", "summary": [], "encrypted_content": "opaque"},
        },
        {"type": "event_msg", "payload": {"type": "agent_message", "message": "hello"}},
    ]
    conversation = sources.parse_codex_rollout(records, "personal")
    assert conversation is not None
    assert conversation.renderable_messages[0].reasoning == ""


def test_a_codex_session_with_no_conversation_yields_nothing() -> None:
    records: list[object] = [
        {"type": "session_meta", "payload": {"session_id": "s"}},
        {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "shell"}},
    ]
    assert sources.parse_codex_rollout(records, "personal") is None


def _codex_turn(role: str, text: str) -> dict[str, object]:
    return {
        "timestamp": "2026-07-20T11:47:10Z",
        "type": "event_msg",
        "payload": {"type": f"{role}_message", "message": text},
    }


def test_codex_subagent_threads_get_their_own_node_linked_to_the_parent() -> None:
    """A subagent fan-out must never collide onto the parent's record.

    Codex writes one rollout file per subagent, all carrying the *parent's*
    `session_id` but each its own `payload.id`. Keying the kb node on
    `session_id` collided every sibling onto one record, and the last file
    processed won -- the defect PLAN-20260923-project-identity's Operator
    Guidance Log recorded against `codex-019f76b7...` and `codex-019f8bba...`.
    """
    parent_records: list[object] = [
        {
            "type": "session_meta",
            "payload": {"session_id": "parent-1", "id": "parent-1", "cwd": "/tmp"},
        },
        _codex_turn("user", "parent turn"),
        _codex_turn("agent", "parent reply"),
    ]
    sub1_records: list[object] = [
        {
            "type": "session_meta",
            "payload": {
                "session_id": "parent-1",
                "id": "sub-1",
                "parent_thread_id": "parent-1",
                "thread_source": "subagent",
                "source": {"subagent": {"thread_spawn": {"parent_thread_id": "parent-1"}}},
            },
        },
        _codex_turn("user", "sub1 turn"),
        _codex_turn("agent", "sub1 reply"),
    ]
    sub2_records: list[object] = [
        {
            "type": "session_meta",
            "payload": {
                "session_id": "parent-1",
                "id": "sub-2",
                "parent_thread_id": "parent-1",
                "thread_source": "subagent",
                "source": {"subagent": {"thread_spawn": {"parent_thread_id": "parent-1"}}},
            },
        },
        _codex_turn("user", "sub2 turn"),
        _codex_turn("agent", "sub2 reply"),
    ]

    parent = sources.parse_codex_rollout(parent_records, "personal")
    sub1 = sources.parse_codex_rollout(sub1_records, "personal")
    sub2 = sources.parse_codex_rollout(sub2_records, "personal")
    assert parent is not None
    assert sub1 is not None
    assert sub2 is not None

    node_ids = {parent.node_id, sub1.node_id, sub2.node_id}
    assert node_ids == {"codex-parent-1", "codex-sub-1", "codex-sub-2"}

    assert parent.tags == ("codex-session",)
    assert parent.related_id is None

    for sub in (sub1, sub2):
        assert "codex-subagent" in sub.tags
        assert sub.related_id == "codex-parent-1"
        rendered = model.render(sub, "t")
        assert "[[id:codex-parent-1]]" in rendered

    # Importing all three writes three distinct records; none overwrites
    # another's stored content.
    runner = _KbRunner()
    cache = titles.TitleCache(Path("/tmp/does-not-exist-t018.json"))
    reports = kbi.import_conversations([parent, sub1, sub2], _args(), cache, runner)
    assert reports["codex"].created == 3
    assert set(runner.stored) == {"codex-parent-1", "codex-sub-1", "codex-sub-2"}
    assert "parent turn" in runner.stored["codex-parent-1"]
    assert "sub1 turn" in runner.stored["codex-sub-1"]
    assert "sub2 turn" in runner.stored["codex-sub-2"]
    assert "sub1 turn" not in runner.stored["codex-parent-1"]
    assert "sub2 turn" not in runner.stored["codex-parent-1"]


def test_codex_subagent_identity_survives_a_replayed_parent_session_meta() -> None:
    """A second, later session_meta must never override the file's own identity.

    62 real rollout files carry two `session_meta` records: a subagent's own,
    followed by its parent thread's `session_meta` replayed later in the same
    file for history context -- observed on live files under
    `~/.config/codex/*/sessions`. That replayed record has no
    `thread_source`/`source.subagent` of its own; honoring it would silently
    turn the subagent's node back into the parent's, recreating the exact
    collision this parser exists to prevent.
    """
    records: list[object] = [
        {
            "type": "session_meta",
            "payload": {
                "session_id": "parent-1",
                "id": "sub-1",
                "parent_thread_id": "parent-1",
                "thread_source": "subagent",
                "source": {"subagent": {"thread_spawn": {"parent_thread_id": "parent-1"}}},
            },
        },
        _codex_turn("user", "sub1 turn"),
        # The parent's own session_meta, replayed later in the same file.
        {
            "type": "session_meta",
            "payload": {"session_id": "parent-1", "id": "parent-1", "thread_source": "user"},
        },
        _codex_turn("agent", "sub1 reply"),
    ]
    conversation = sources.parse_codex_rollout(records, "personal")
    assert conversation is not None
    assert conversation.node_id == "codex-sub-1"
    assert "codex-subagent" in conversation.tags
    assert conversation.related_id == "codex-parent-1"


def test_codex_forked_resume_links_back_to_its_origin_thread() -> None:
    """A genuine user resume (not a subagent) is its own node, linked to origin."""
    records: list[object] = [
        {
            "type": "session_meta",
            "payload": {
                "session_id": "resumed-1",
                "id": "resumed-1",
                "forked_from_id": "origin-1",
                "thread_source": "user",
                "source": "cli",
            },
        },
        _codex_turn("user", "continuing"),
        _codex_turn("agent", "sure"),
    ]
    conversation = sources.parse_codex_rollout(records, "personal")
    assert conversation is not None
    assert conversation.node_id == "codex-resumed-1"
    assert "codex-subagent" not in conversation.tags
    assert conversation.related_id == "codex-origin-1"
    rendered = model.render(conversation, "t")
    assert "[[id:codex-origin-1]]" in rendered


def test_codex_reads_response_item_messages_when_event_msg_is_absent() -> None:
    """Codex CLI >= ~0.144 stops writing event_msg user_message/agent_message.

    Those sessions carry their turns only as `response_item`/`message`
    records. `developer` role and injected environment context arriving as
    role `user` must both be excluded.
    """
    records: list[object] = [
        {"type": "session_meta", "payload": {"session_id": "s1", "id": "s1"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "SYSTEM PROMPT"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "# AGENTS.md instructions for /repo\n<INSTRUCTIONS>...",
                    }
                ],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "please read the plan"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "reading it now"}],
            },
        },
    ]
    conversation = sources.parse_codex_rollout(records, "personal")
    assert conversation is not None
    human, assistant = conversation.renderable_messages
    assert (human.role, human.text) == ("human", "please read the plan")
    assert (assistant.role, assistant.text) == ("assistant", "reading it now")
    rendered = model.render(conversation, "t")
    assert "SYSTEM PROMPT" not in rendered
    assert "AGENTS.md instructions" not in rendered


def test_codex_event_msg_wins_over_response_item_when_both_are_present() -> None:
    """When a file carries both formats, only the `event_msg` turns are used."""
    records: list[object] = [
        {"type": "session_meta", "payload": {"session_id": "s2", "id": "s2"}},
        _codex_turn("user", "event msg turn"),
        _codex_turn("agent", "event msg reply"),
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "response item turn"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "response item reply"}],
            },
        },
    ]
    conversation = sources.parse_codex_rollout(records, "personal")
    assert conversation is not None
    texts = [m.text for m in conversation.renderable_messages]
    assert texts == ["event msg turn", "event msg reply"]
    assert "response item turn" not in texts
    assert "response item reply" not in texts


def test_an_empty_codex_rollout_is_reported_not_silently_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`collect` must surface files that parse to no renderable turn.

    Before this fix, `parse_codex_rollout` returning `None` for an empty
    session and for a rollout carrying only tool traffic looked identical to
    the caller: both vanished with no record anywhere.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    sessions = tmp_path / ".config" / "codex" / "personal" / "sessions"
    sessions.mkdir(parents=True)
    empty_path = sessions / "rollout-2026-07-20T00-00-00-empty.jsonl"
    empty_path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"type": "session_meta", "payload": {"session_id": "empty-1"}},
                {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "shell"}},
            ]
        )
    )
    nonempty_path = sessions / "rollout-2026-07-20T00-01-00-real.jsonl"
    nonempty_path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"type": "session_meta", "payload": {"session_id": "real-1"}},
                _codex_turn("user", "hi"),
                _codex_turn("agent", "hello"),
            ]
        )
    )

    conversations, empty = kbi.collect(_args("--source", "codex"))
    assert [c.node_id for c in conversations] == ["codex-real-1"]
    assert empty["codex"] == [str(empty_path)]


# ── Claude Code CLI transcripts ────────────────────────────────────────────

CLAUDE_CODE: list[object] = [
    {"type": "ai-title", "aiTitle": "Investigate the runner setup", "sessionId": "s1"},
    {
        "type": "user",
        "uuid": "u1",
        "timestamp": "2026-08-01T10:00:00Z",
        "cwd": "/Users/op/Projects/kb/main",
        "gitBranch": "main",
        "message": {"id": "m1", "content": "what is failing?"},
    },
    {
        "type": "user",
        "uuid": "u2",
        "isMeta": True,
        "timestamp": "2026-08-01T10:00:01Z",
        "message": {"id": "m-meta", "content": "INJECTED META CONTEXT"},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "timestamp": "2026-08-01T10:00:05Z",
        "message": {"id": "m2", "content": [{"type": "thinking", "thinking": "considering"}]},
    },
    {
        "type": "assistant",
        "uuid": "a2",
        "timestamp": "2026-08-01T10:00:06Z",
        "message": {"id": "m2", "content": [{"type": "text", "text": "The runner "}]},
    },
    {
        "type": "assistant",
        "uuid": "a3",
        "timestamp": "2026-08-01T10:00:07Z",
        "message": {
            "id": "m2",
            "content": [
                {"type": "text", "text": "is not registered."},
                {"type": "tool_use", "name": "bash", "input": "SECRET"},
                {"type": "tool_result", "content": "SECRET RESULT"},
            ],
        },
    },
]


def test_claude_code_streaming_chunks_become_one_message() -> None:
    """Group streamed records by message id.

    One line is not one message: 521 of 2309 sampled groups spanned several
    records, up to six.
    """
    conversation = sources.parse_claude_code_transcript(CLAUDE_CODE, "s1", "personal")
    assert conversation is not None
    assert model.render(conversation, "t").count("* Assistant") == 1
    assistant = conversation.renderable_messages[1]
    assert assistant.text == "The runner \n\nis not registered."
    assert assistant.reasoning == "considering"


def test_claude_code_node_ids_are_always_cc_prefixed() -> None:
    """The bare session id holds the hook's pre-existing agent-summary node."""
    conversation = sources.parse_claude_code_transcript(CLAUDE_CODE, "s1", "personal")
    assert conversation is not None
    assert conversation.node_id == "cc-s1"


def test_claude_code_carries_the_records_cwd() -> None:
    """`cwd` is what lets a bare-transcript re-import resolve a project."""
    conversation = sources.parse_claude_code_transcript(CLAUDE_CODE, "s1", "personal")
    assert conversation is not None
    assert conversation.cwd == "/Users/op/Projects/kb/main"


def test_a_claude_code_transcript_without_a_cwd_leaves_it_none() -> None:
    records: list[object] = [
        {
            "type": "user",
            "uuid": "u1",
            "timestamp": "2026-08-01T10:00:00Z",
            "message": {"id": "m1", "content": "hi"},
        },
    ]
    conversation = sources.parse_claude_code_transcript(records, "s2", "personal")
    assert conversation is not None
    assert conversation.cwd is None


def test_claude_code_takes_its_title_from_the_ai_title_record() -> None:
    conversation = sources.parse_claude_code_transcript(CLAUDE_CODE, "s1", "personal")
    assert conversation is not None
    assert conversation.title_hint == "Investigate the runner setup"


def test_claude_code_drops_meta_records_and_tool_traffic() -> None:
    conversation = sources.parse_claude_code_transcript(CLAUDE_CODE, "s1", "personal")
    assert conversation is not None
    rendered = model.render(conversation, "t")
    assert "INJECTED META CONTEXT" not in rendered
    assert "SECRET" not in rendered


def test_claude_code_accepts_a_bare_string_message_content() -> None:
    conversation = sources.parse_claude_code_transcript(CLAUDE_CODE, "s1", "personal")
    assert conversation is not None
    assert conversation.renderable_messages[0].text == "what is failing?"


def test_a_claude_code_session_of_only_tool_traffic_yields_nothing() -> None:
    records: list[object] = [
        {
            "type": "assistant",
            "uuid": "a1",
            "message": {"id": "m1", "content": [{"type": "tool_use", "name": "bash"}]},
        }
    ]
    assert sources.parse_claude_code_transcript(records, "s1", "personal") is None


# ── Programmatic-session exclusion ─────────────────────────────────────────
#
# Title derivation shells out to `claude -p`, and each such call used to leave
# a transcript that the next import ingested as a conversation. On the machine
# where this was found, 2495 of 2725 transcripts were `sdk-cli` and roughly
# 1166 machine-generated nodes had already reached the corpus.


def _entrypoint_records(entrypoint: str | None) -> list[object]:
    record: dict[str, object] = {
        "type": "user",
        "uuid": "u1",
        "message": {"id": "m1", "content": "derive a title for this"},
    }
    if entrypoint is not None:
        record["entrypoint"] = entrypoint
    return [record]


def test_a_programmatic_session_is_refused() -> None:
    records = _entrypoint_records("sdk-cli")
    assert sources.is_programmatic(records)
    assert sources.parse_claude_code_transcript(records, "s1", "personal") is None


def test_include_programmatic_imports_it_anyway() -> None:
    conversation = sources.parse_claude_code_transcript(
        _entrypoint_records("sdk-cli"), "s1", "personal", True
    )
    assert conversation is not None
    assert conversation.node_id == "cc-s1"


def test_one_interactive_record_makes_the_whole_transcript_interactive() -> None:
    """A resumed session mixes entrypoints; any `cli` record settles it."""
    records = [*_entrypoint_records("sdk-cli"), *_entrypoint_records("cli")]
    assert not sources.is_programmatic(records)


def test_a_transcript_without_entrypoints_fails_open() -> None:
    """A renamed field must cost over-collection, never silent data loss."""
    records = _entrypoint_records(None)
    assert not sources.is_programmatic(records)
    assert sources.parse_claude_code_transcript(records, "s1", "personal") is not None


def test_include_programmatic_is_off_by_default() -> None:
    assert not _args().include_programmatic
    assert _args("--include-programmatic").include_programmatic


# ── Title derivation and cache ─────────────────────────────────────────────


class _Runner:
    """A stub subprocess runner recording its calls."""

    def __init__(self, *, output: str = "A Derived Title", fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.stdins: list[str] = []
        self.output = output
        self.fail = fail

    def __call__(self, argv: Sequence[str], stdin: str = "") -> str:
        self.calls.append(list(argv))
        self.stdins.append(stdin)
        if self.fail:
            raise titles.CommandError("claude exited 1: rate limited")
        return self.output


def _titled(title_hint: str = "") -> model.Conversation:
    return _conversation(
        model.Message(role="human", timestamp=None, text="what is a gooseneck kettle"),
        title_hint=title_hint,
    )


def test_a_derived_title_is_cached_and_not_derived_twice(tmp_path: Path) -> None:
    cache = titles.TitleCache(tmp_path / "titles.json")
    runner = _Runner()
    conversation = _titled()
    first = kbi.derive_title(conversation, cache, runner)
    second = kbi.derive_title(conversation, cache, runner)
    assert first == second == "A Derived Title"
    assert len(runner.calls) == 1


def test_title_derivation_leaves_no_transcript_behind(tmp_path: Path) -> None:
    """Close the feedback loop at its source.

    Without `--no-session-persistence` every derivation writes a transcript
    into the project directory, which the next run then imports as if it were
    a conversation.
    """
    runner = _Runner()
    kbi.derive_title(_titled(), titles.TitleCache(tmp_path / "titles.json"), runner)
    assert "--no-session-persistence" in runner.calls[0]


def test_a_failed_derivation_falls_back_to_the_source_title(tmp_path: Path) -> None:
    cache = titles.TitleCache(tmp_path / "titles.json")
    title = kbi.derive_title(_titled("The source title"), cache, _Runner(fail=True))
    assert title == "The source title"


def test_a_failed_derivation_is_not_cached(tmp_path: Path) -> None:
    """Caching a fallback would freeze it in place once the endpoint recovers."""
    cache = titles.TitleCache(tmp_path / "titles.json")
    kbi.derive_title(_titled("The source title"), cache, _Runner(fail=True))
    assert cache.get("codex-abc123") is None


def test_an_empty_model_reply_falls_back_rather_than_titling_a_node_blank(tmp_path: Path) -> None:
    cache = titles.TitleCache(tmp_path / "titles.json")
    title = kbi.derive_title(_titled("The source title"), cache, _Runner(output="   \n"))
    assert title == "The source title"


def test_a_derived_title_is_stripped_of_quotes_and_truncated(tmp_path: Path) -> None:
    cache = titles.TitleCache(tmp_path / "titles.json")
    title = kbi.derive_title(_titled(), cache, _Runner(output='"' + "y" * 200 + '"'))
    assert title == "y" * 80


def test_the_cache_survives_a_round_trip_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "titles.json"
    cache = titles.TitleCache(path)
    cache.set("n1", "remembered")
    cache.save()
    assert titles.TitleCache(path).get("n1") == "remembered"


def test_an_unreadable_cache_is_treated_as_empty(tmp_path: Path) -> None:
    """An interrupted or corrupt cache must not abort an import."""
    path = tmp_path / "titles.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert titles.TitleCache(path).get("anything") is None


def test_saving_the_cache_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    path = tmp_path / "titles.json"
    cache = titles.TitleCache(path)
    cache.set("n1", "t")
    cache.save()
    assert not list(path.parent.glob("*.tmp"))


def test_title_cache_path_honors_the_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rehearsal must not touch the live title cache.

    Without an override, every run shares one cache path
    (`~/.local/state/kb-import/titles.json`), which is exactly what a
    rehearsal against scratch copies of the store must not also read or
    write: doing so would mutate the live importer's own idempotence cache,
    and its already-cached titles would mask the rehearsal from ever
    exercising derivation.
    """
    override = tmp_path / "scratch-titles.json"
    monkeypatch.setenv("KB_IMPORT_TITLE_CACHE", str(override))
    assert kbi.title_cache_path() == override


def test_title_cache_path_defaults_without_the_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KB_IMPORT_TITLE_CACHE", raising=False)
    assert kbi.title_cache_path() == kbi.DEFAULT_CACHE_PATH


# ── Stored-document comparison ─────────────────────────────────────────────


def test_the_property_drawer_is_stripped_before_comparison() -> None:
    stored = ":PROPERTIES:\n:ID: x\n:CREATED: y\n:UPDATED: z\n:END:\n#+title: t\n\n* Human\nhi"
    assert store._strip_property_drawer(stored).startswith("#+title: t")


def test_a_document_without_a_drawer_is_returned_unchanged() -> None:
    assert store._strip_property_drawer("#+title: t\n") == "#+title: t\n"


def test_a_trailing_newline_does_not_count_as_a_change() -> None:
    """Ignore the trailing newline kb adds.

    kb's generator appends one and the renderer does not. Comparing naively
    would rewrite and re-embed the entire corpus on every run.
    """
    assert store.is_unchanged("#+title: t\n\n* Human\nhi\n\n", "#+title: t\n\n* Human\nhi\n")


def test_a_real_change_is_still_detected() -> None:
    assert not store.is_unchanged("#+title: t\n\n* Human\nhi\n", "#+title: t\n\n* Human\nbye\n")


# ── Provenance ────────────────────────────────────────────────────────────


class _ClankerRunner:
    """A stub `clanker`, recording every invocation."""

    def __init__(self, line: str = "", *, present: bool = True) -> None:
        self.line = line
        self.present = present
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str], stdin: str = "") -> str:  # noqa: ARG002
        self.calls.append(list(argv))
        if not self.present:
            raise FileNotFoundError("clanker: command not found")
        return self.line


def test_resolve_project_parses_the_tab_separated_line() -> None:
    runner = _ClankerRunner("kb\tremote\tgithub.com/tftio/kb\n")
    resolved = provenance.resolve_project("/Users/op/Projects/kb/main", runner)
    assert resolved == {
        "project": "kb",
        "project_source": "remote",
        "remote": "github.com/tftio/kb",
    }
    assert runner.calls == [
        ["clanker", "project", "resolve", "--dir", "/Users/op/Projects/kb/main"]
    ]


def test_resolve_project_an_unresolved_directory_is_empty_fields() -> None:
    runner = _ClankerRunner("\t\t\n")
    assert provenance.resolve_project("/tmp", runner) == {}


def test_resolve_project_a_missing_clanker_binary_resolves_to_nothing() -> None:
    runner = _ClankerRunner(present=False)
    assert provenance.resolve_project("/tmp", runner) == {}


def test_resolve_project_a_refused_call_resolves_to_nothing() -> None:
    def run(_argv: Sequence[str], _stdin: str = "") -> str:
        raise titles.CommandError("clanker exited 2")

    assert provenance.resolve_project("/tmp", run) == {}


def test_provenance_for_prefers_explicit_flags_over_resolution() -> None:
    runner = _ClankerRunner("other\tpath\tgithub.com/x/y\n")
    conversation = _conversation(
        model.Message(role="human", timestamp=None, text="hi"),
        tags=("claude-code-session",),
    )
    conversation = dataclasses.replace(conversation, cwd="/some/dir")
    args = kbi.build_parser().parse_args(["--project", "kb", "--context", "personal"])
    result = provenance.provenance_for(conversation, args, runner)
    assert result["project"] == "kb"
    assert result["context"] == "personal"
    assert result["cwd"] == "/some/dir"
    # An explicit --project means the working directory is never resolved.
    assert runner.calls == []


def test_provenance_for_resolves_a_bare_cwd() -> None:
    """The acceptance case: a transcript with only `cwd` resolves through clanker."""
    runner = _ClankerRunner("kb\tremote\tgithub.com/tftio/kb\n")
    conversation = _conversation(
        model.Message(role="human", timestamp=None, text="hi"),
        tags=("claude-code-session",),
    )
    conversation = dataclasses.replace(conversation, cwd="/Users/op/Projects/kb")
    args = kbi.build_parser().parse_args([])
    result = provenance.provenance_for(conversation, args, runner)
    assert result == {
        "project": "kb",
        "project_source": "remote",
        "remote": "github.com/tftio/kb",
        "cwd": "/Users/op/Projects/kb",
    }


def test_provenance_for_carries_every_flag() -> None:
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hi"))
    args = kbi.build_parser().parse_args(
        [
            "--project",
            "kb",
            "--project-source",
            "declared",
            "--remote",
            "github.com/tftio/kb",
            "--context",
            "personal",
            "--domain",
            "eng",
            "--harness",
            "claude",
            "--model",
            "sonnet",
            "--clanker-session",
            "sess-id",
        ]
    )
    result = provenance.provenance_for(conversation, args, lambda _a, _s="": "")
    assert result == {
        "project": "kb",
        "project_source": "declared",
        "remote": "github.com/tftio/kb",
        "context": "personal",
        "domains": ["eng"],
        "harness": "claude",
        "model": "sonnet",
        "session": "sess-id",
    }


def test_provenance_for_nothing_known_is_an_empty_mapping() -> None:
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hi"))
    args = kbi.build_parser().parse_args([])
    assert provenance.provenance_for(conversation, args, lambda _a, _s="": "") == {}


def test_a_provenance_file_round_trips_as_json() -> None:
    payload: dict[str, object] = {"project": "kb", "domains": ["eng", "infra"]}
    with provenance._provenance_file(payload) as path:
        assert json.loads(path.read_text()) == payload
        written = path
    assert not written.exists()


# ── kb driver ──────────────────────────────────────────────────────────────


class _KbRunner:
    """A stub kb, holding stored documents and provenance in memory.

    `get` answers `--json` the way the real CLI does: `data.document` is the
    stored text with no drawer or provenance block attached, and
    `data.provenance` renames `project_source` to `projectSource`, matching
    `src/cli_main.rs`'s `provenance_json` -- the asymmetry `fetch_node`'s
    `_normalize_stored_provenance` exists to undo.
    """

    def __init__(
        self,
        *,
        stored: dict[str, str] | None = None,
        provenance: dict[str, dict[str, object]] | None = None,
        fail_on: str = "",
    ) -> None:
        self.stored: dict[str, str] = dict(stored or {})
        self.provenance: dict[str, dict[str, object]] = dict(provenance or {})
        self.calls: list[list[str]] = []
        self.fail_on = fail_on

    def __call__(self, argv: Sequence[str], stdin: str = "") -> str:
        args = list(argv)
        self.calls.append(args)
        if args[0] == "claude":
            return "A Derived Title"
        if args[0] == "clanker":
            return ""
        # `kb` may be followed by `--db <path>` before the verb.
        rest = args[3:] if args[1:2] == ["--db"] else args[1:]
        verb = rest[0]
        if verb == "get":
            node_id = rest[1]
            if node_id not in self.stored:
                raise titles.CommandError("kb exited 1: no node with id")
            stored_provenance = self.provenance.get(node_id)
            payload: dict[str, object] | None = None
            if stored_provenance:
                payload = dict(stored_provenance)
                if "project_source" in payload:
                    payload["projectSource"] = payload.pop("project_source")
            data = {"document": self.stored[node_id], "provenance": payload}
            return json.dumps({"command": "get", "ok": True, "data": data})
        if verb == "delete":
            node_id = rest[1]
            self.stored.pop(node_id, None)
            self.provenance.pop(node_id, None)
            return "{}"
        # `create` and `update` (the only remaining verbs) both accept
        # `--provenance-json`, and given it both store exactly the file's
        # contents. Absent it, the real `kb update` (PLAN-20260923-project-
        # identity T016) preserves whatever provenance the node already had
        # -- there being none to preserve on a fresh `create` makes that verb
        # look the same as the old "reset to none" behaviour without this
        # stub having to special-case it.
        node_id = rest[rest.index("--id") + 1] if verb == "create" else rest[1]
        if self.fail_on and self.fail_on in node_id:
            raise titles.CommandError("kb exited 1: refused")
        self.stored[node_id] = stdin
        if "--provenance-json" in rest:
            path = Path(rest[rest.index("--provenance-json") + 1])
            self.provenance[node_id] = json.loads(path.read_text())
        return "{}"


def _args(*extra: str) -> argparse.Namespace:
    return kbi.build_parser().parse_args(["--db", "/tmp/nonexistent.db", *extra])


def test_write_node_passes_no_tag_flags() -> None:
    """Keep tag authority in the document.

    `render` already writes the tags into `#+filetags:`, and kb appends `--tag`
    values to that line rather than merging. Passing both produced a doubled tag
    list that grew on every re-import.
    """
    runner = _KbRunner()
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hi"))
    store.write_node(conversation, "A title", None, runner, exists=False)
    assert "--tag" not in runner.calls[0]


def test_write_node_attaches_provenance_on_create() -> None:
    runner = _KbRunner()
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hi"))
    store.write_node(
        conversation, "A title", None, runner, exists=False, provenance={"project": "kb"}
    )
    assert runner.provenance["codex-abc123"] == {"project": "kb"}
    assert "--provenance-json" in runner.calls[0]


def test_write_node_update_without_provenance_does_not_attach_the_flag() -> None:
    runner = _KbRunner(stored={"codex-abc123": "old body"})
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hi"))
    outcome = store.write_node(conversation, "A title", None, runner, exists=True)
    assert outcome == "updated"
    assert [call[1] for call in runner.calls] == ["update"]
    assert "--provenance-json" not in runner.calls[0]


def test_write_node_update_with_provenance_attaches_the_flag_in_place() -> None:
    """A provenance change on an existing record is one in-place `kb update`.

    `kb update --provenance-json` replaces delete-then-create (coordinator
    review of 998bf6e): a provenance change on an existing record is a single
    in-place `kb update` call, never a `kb delete`.
    """
    runner = _KbRunner(
        stored={"codex-abc123": "old body"}, provenance={"codex-abc123": {"project": "old"}}
    )
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hi"))
    outcome = store.write_node(
        conversation, "A title", None, runner, exists=True, provenance={"project": "kb"}
    )
    assert outcome == "updated"
    assert [call[1] for call in runner.calls] == ["update"]
    assert "delete" not in [call[1] for call in runner.calls]
    assert "--provenance-json" in runner.calls[0]
    assert runner.provenance["codex-abc123"] == {"project": "kb"}


def test_an_unchanged_conversation_is_skipped_rather_than_rewritten(tmp_path: Path) -> None:
    runner = _KbRunner()
    cache = titles.TitleCache(tmp_path / "t.json")
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hi"))
    args = _args()
    first = kbi.import_conversations([conversation], args, cache, runner)
    second = kbi.import_conversations([conversation], args, cache, runner)
    assert (first["codex"].created, first["codex"].skipped) == (1, 0)
    assert (second["codex"].created, second["codex"].skipped) == (0, 1)


def test_an_unchanged_transcript_with_unchanged_provenance_reports_no_change(
    tmp_path: Path,
) -> None:
    """Idempotence for T008's backfill: re-running with the same provenance is a no-op."""
    runner = _KbRunner()
    cache = titles.TitleCache(tmp_path / "t.json")
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hi"))
    args = _args("--project", "kb", "--context", "personal")
    first = kbi.import_conversations([conversation], args, cache, runner)
    second = kbi.import_conversations([conversation], args, cache, runner)
    assert (first["codex"].created, first["codex"].updated) == (1, 0)
    assert (second["codex"].created, second["codex"].updated, second["codex"].skipped) == (
        0,
        0,
        1,
    )
    assert runner.provenance["codex-abc123"] == {"project": "kb", "context": "personal"}
    assert "delete" not in [call[1] for call in runner.calls]


def test_a_rerun_that_adds_provenance_to_an_existing_record_updates_it(tmp_path: Path) -> None:
    """The other half of T008's idempotence requirement.

    Newly resolved provenance on an otherwise-unchanged transcript still
    counts as a change.
    """
    runner = _KbRunner()
    cache = titles.TitleCache(tmp_path / "t.json")
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hi"))

    # The pre-T007 state: written with no provenance at all.
    first = kbi.import_conversations([conversation], _args(), cache, runner)
    assert (first["codex"].created, first["codex"].updated) == (1, 0)
    assert "codex-abc123" not in runner.provenance

    # A later run resolves a project the first run did not have.
    second = kbi.import_conversations([conversation], _args("--project", "kb"), cache, runner)
    assert (second["codex"].created, second["codex"].updated, second["codex"].skipped) == (
        0,
        1,
        0,
    )
    assert runner.provenance["codex-abc123"] == {"project": "kb"}

    # A third run with the same provenance is a true no-op again.
    third = kbi.import_conversations([conversation], _args("--project", "kb"), cache, runner)
    assert (third["codex"].created, third["codex"].updated, third["codex"].skipped) == (0, 0, 1)

    # Landing provenance on an existing record never goes through `kb
    # delete` (coordinator review of 998bf6e): every write above the first
    # is an in-place `kb update --provenance-json`.
    assert "delete" not in [call[1] for call in runner.calls]


def test_a_rerun_with_no_provenance_leaves_stored_provenance_alone(tmp_path: Path) -> None:
    """A run that resolves no provenance must not clear what is stored.

    Before this fix, the skip check compared a freshly computed empty
    provenance mapping against whatever was stored and treated any mismatch
    as a change, rewriting a stored-nonempty/computed-empty record on every
    run -- breaking T008's "second run reports zero changed" idempotence for
    any record a later run cannot re-resolve provenance for (no
    `--project`/flags and no captured `cwd`). This importer asserts only
    provenance it knows, so an unchanged document with no newly resolved
    provenance must be a true no-op: no `create`/`update` call at all, and
    the record's existing provenance must survive untouched.
    """
    runner = _KbRunner()
    cache = titles.TitleCache(tmp_path / "t.json")
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hi"))

    # Seed a record carrying provenance from an earlier, better-informed run.
    first = kbi.import_conversations([conversation], _args("--project", "kb"), cache, runner)
    assert (first["codex"].created, first["codex"].updated) == (1, 0)
    assert runner.provenance["codex-abc123"] == {"project": "kb"}
    calls_before = list(runner.calls)

    # A later run resolves no provenance at all, and the document is unchanged.
    second = kbi.import_conversations([conversation], _args(), cache, runner)
    assert (second["codex"].created, second["codex"].updated, second["codex"].skipped) == (
        0,
        0,
        1,
    )
    new_calls = runner.calls[len(calls_before) :]
    verbs = [call[3] for call in new_calls]
    assert "create" not in verbs
    assert "update" not in verbs
    assert runner.provenance["codex-abc123"] == {"project": "kb"}


def test_a_rerun_with_no_provenance_and_a_changed_document_updates_without_the_flag(
    tmp_path: Path,
) -> None:
    """The document-changed half of the same rule.

    A changed document with no newly resolved provenance still must be
    written -- the document itself is the thing that changed -- but as a
    plain `kb update` with no `--provenance-json`, which (T016) preserves
    the record's existing provenance rather than clearing it.
    """
    runner = _KbRunner()
    cache = titles.TitleCache(tmp_path / "t.json")
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hi"))
    first = kbi.import_conversations([conversation], _args("--project", "kb"), cache, runner)
    assert (first["codex"].created, first["codex"].updated) == (1, 0)
    assert runner.provenance["codex-abc123"] == {"project": "kb"}

    changed = dataclasses.replace(
        conversation,
        messages=(model.Message(role="human", timestamp=None, text="hi again"),),
    )
    second = kbi.import_conversations([changed], _args(), cache, runner)
    assert (second["codex"].created, second["codex"].updated, second["codex"].skipped) == (
        0,
        1,
        0,
    )
    update_calls = [call for call in runner.calls if call[3] == "update"]
    assert update_calls, "expected the changed document to trigger an update"
    assert "--provenance-json" not in update_calls[-1]
    assert "hi again" in runner.stored["codex-abc123"]
    assert runner.provenance["codex-abc123"] == {"project": "kb"}


def test_a_dry_run_makes_no_subprocess_call_at_all(tmp_path: Path) -> None:
    runner = _KbRunner()
    cache = titles.TitleCache(tmp_path / "t.json")
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hi"))
    reports = kbi.import_conversations([conversation], _args("--dry-run"), cache, runner)
    assert runner.calls == []
    assert reports["codex"].created == 1


def test_one_failure_does_not_abort_the_rest(tmp_path: Path) -> None:
    runner = _KbRunner(fail_on="doomed")
    cache = titles.TitleCache(tmp_path / "t.json")
    message = model.Message(role="human", timestamp=None, text="hi")
    doomed = model.Conversation(
        source="codex",
        source_id="doomed",
        node_id="codex-doomed",
        title_hint="t",
        account="",
        started_at=None,
        messages=(message,),
        tags=("codex-session",),
    )
    survivor = _conversation(message)
    reports = kbi.import_conversations([doomed, survivor], _args(), cache, runner)
    assert reports["codex"].created == 1
    assert len(reports["codex"].failed) == 1
    assert "codex-doomed" in reports["codex"].failed[0]


def test_select_applies_the_session_filter() -> None:
    message = model.Message(role="human", timestamp=None, text="hi")
    a = _conversation(message)
    assert kbi.select([a], _args("--session", "abc123")) == [a]
    assert kbi.select([a], _args("--session", "other")) == []


def test_select_applies_the_since_filter() -> None:
    message = model.Message(role="human", timestamp=None, text="hi")
    a = _conversation(message)
    assert kbi.select([a], _args("--since", "2025-01-01")) == [a]
    assert kbi.select([a], _args("--since", "2027-01-01")) == []


def test_select_applies_the_limit() -> None:
    message = model.Message(role="human", timestamp=None, text="hi")
    conversations = [_conversation(message), _conversation(message)]
    assert len(kbi.select(conversations, _args("--limit", "1"))) == 1


def test_load_jsonl_skips_a_corrupt_line(tmp_path: Path) -> None:
    """One bad record in one of 774 files must not cost the other 773."""
    path = tmp_path / "s.jsonl"
    path.write_text('{"a": 1}\nnot json at all\n{"b": 2}\n', encoding="utf-8")
    assert kbi.load_jsonl(path) == [{"a": 1}, {"b": 2}]


# ── Org-injection escaping ─────────────────────────────────────────────────


def test_a_heading_line_in_prose_is_escaped() -> None:
    """Guard the 2026-08-06 regression.

    A transcript discussing org-mode contains lines that look like org markup.
    Rendered at column zero they became real headings, and their trailing
    `:a:b:` became real tags on the node.
    """
    injected = "* Summary :agent-summary:injected:"
    conversation = _conversation(model.Message(role="human", timestamp=None, text=injected))
    rendered = model.render(conversation, "t")
    assert "\n * Summary :agent-summary:injected:" in rendered
    assert "\n* Summary" not in rendered


def test_a_keyword_line_in_prose_is_escaped() -> None:
    """`#+filetags:` at column zero injects tags directly."""
    conversation = _conversation(
        model.Message(role="human", timestamp=None, text="#+filetags: :hijacked:")
    )
    assert "\n #+filetags: :hijacked:" in model.render(conversation, "t")


def test_reasoning_is_escaped_too() -> None:
    conversation = _conversation(
        model.Message(role="assistant", timestamp=None, text="reply", reasoning="* not a heading")
    )
    assert "\n * not a heading" in model.render(conversation, "t")


def test_the_real_message_headings_are_still_headings() -> None:
    """Escaping must not disarm the structure the renderer itself emits."""
    conversation = _conversation(model.Message(role="human", timestamp=None, text="hi"))
    assert "\n* Human\n" in "\n" + model.render(conversation, "t")


def test_list_bullets_and_tables_are_left_alone() -> None:
    """They carry no tags, and escaping them would corrupt archived formatting."""
    body = "- a bullet\n| a | table |\n1. numbered"
    conversation = _conversation(model.Message(role="human", timestamp=None, text=body))
    assert "- a bullet" in model.render(conversation, "t")
    assert "| a | table |" in model.render(conversation, "t")


def test_escape_org_preserves_ordinary_prose_exactly() -> None:
    prose = "a normal line\nanother one\n  indented already"
    assert model.escape_org(prose) == prose


# ── Change-detection false positive (PLAN-20260923-project-identity T017) ──
#
# The T008 rehearsal found the importer reporting an already-imported
# transcript as changed on every re-run, rewriting it, bumping `updated_at`
# and re-embedding it for nothing. `is_unchanged` itself
# (`stored.rstrip() == rendered.rstrip()`) is correct: it was comparing
# `rendered` (this run's `render()` output) against `stored` (what
# `fetch_node` reads back through `kb get --json`) faithfully. The mismatch
# was upstream of both: kb's own parser round-trips a document through
# `parse_document` and `generator::generate` on every write
# (`src/write.rs::put_record`), and `is_paragraph_line` (`src/parser.rs`)
# trimmed a line before testing it for a heading star or a `#+` keyword
# prefix, while the parsers that actually claim those block kinds
# (`try_parse_heading`, `try_parse_keyword`) require them at column 0,
# untrimmed. `escape_org` (`kb_import.model`) defuses a
# literal `*`/`#+` at the start of transcript prose by indenting it one
# column exactly so it is *not* column 0 -- but the trimmed check still
# saw a heading/keyword shape, excluded the line from the paragraph it
# belongs to, and no other block parser claimed an indented `*`/`#+` line
# either, so it was silently dropped (`parse_blocks`'s residue). The stored
# document then permanently disagreed with every future `render()` of the
# same conversation, and `is_unchanged` correctly, repeatedly, reported
# "changed" for content that had not changed at all.
#
# This is a full round trip through the real `kb` binary this repository
# builds, not a mock of it: the bug lived in kb's own parser, so a stub
# `CommandRunner` returning a hand-written "stored" string could not have
# caught it, and would not catch a regression either.


def _kb_binary() -> Path:
    """The `kb` binary this worktree builds, building it if necessary.

    Args:
        None.

    Returns:
        Path to `target/debug/kb`. `cargo nextest run` (this repository's
        `check:test` task) already builds it as a side effect of building
        `tests/write_cli.rs`'s `CARGO_BIN_EXE_kb`, and `prek.toml` runs that
        hook before the `pytest` hook -- but a fallback build keeps this test
        honest when it is run on its own.
    """
    repo_root = Path(__file__).resolve().parent.parent
    binary = repo_root / "target" / "debug" / "kb"
    if not binary.exists():
        subprocess.run(["cargo", "build", "--quiet", "--bin", "kb"], cwd=repo_root, check=True)
    return binary


def _kb_runner(kb_bin: Path, store: Path, index: Path) -> titles.CommandRunner:
    """A `CommandRunner` that runs the real `kb` binary against a scratch store.

    Args:
        kb_bin: Path to the `kb` binary.
        store: A scratch `KB_STORE_PATH`, unique to this test.
        index: A scratch `KB_INDEX_PATH`, unique to this test.

    Returns:
        A runner substitutable for `run_command`, isolated from any real
        store or index.
    """
    env = dict(os.environ)
    env["KB_STORE_PATH"] = str(store)
    env["KB_INDEX_PATH"] = str(index)

    def run(argv: Sequence[str], stdin: str = "") -> str:
        real_argv = [str(kb_bin), *argv[1:]] if argv and argv[0] == "kb" else list(argv)
        result = subprocess.run(
            real_argv, input=stdin, capture_output=True, text=True, env=env, check=False
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise titles.CommandError(f"{real_argv[0]} exited {result.returncode}: {detail}")
        return result.stdout

    return run


#: A minimized, synthetic Claude Code transcript reproducing the structural
#: trigger of a real one (T017's rehearsal sample, sanitized rather than
#: quoted): a human turn and an assistant turn each containing a line that
#: `escape_org` indents because it starts with `*` or `#+`.
_ESCAPED_SHAPE_RECORDS: list[object] = [
    {
        "type": "user",
        "uuid": "u1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:00Z",
        "cwd": "/Users/op/Projects/kb/main",
        "message": {
            "id": "m1",
            "content": "Here is a checklist:\n* not a real heading\nplease look at it.",
        },
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:05Z",
        "message": {
            "id": "m2",
            "content": [
                {
                    "type": "text",
                    "text": "Noted. One config line looked like:\n#+title: not real\ndone.",
                }
            ],
        },
    },
]


def test_a_transcript_containing_the_escaped_shape_round_trips_unchanged(tmp_path: Path) -> None:
    """Regression for the T008 false positive: create, fetch, compare.

    Before the `src/parser.rs::is_paragraph_line` fix, kb's own parse/
    generate round trip silently dropped the escaped lines on `kb create`,
    so the very first `kb get --json` already disagreed with `render()` and
    `is_unchanged` reported "changed" forever. This asserts the stored
    document, fetched exactly as the importer fetches it, compares equal to
    a fresh render on the very next comparison -- which `import_conversations`
    relies on to report 0 updated on an unchanged second run.
    """
    conversation = sources.parse_claude_code_transcript(
        _ESCAPED_SHAPE_RECORDS, "deadbeef", "personal"
    )
    assert conversation is not None
    rendered = model.render(conversation, "Escaped structural lines regression")
    # The fixture is pointless if escape_org did not actually trigger.
    assert "\n * not a real heading" in rendered
    assert "\n #+title: not real" in rendered

    run = _kb_runner(_kb_binary(), tmp_path / "store", tmp_path / "index.db")
    outcome = store.write_node(
        conversation, "Escaped structural lines regression", None, run, exists=False
    )
    assert outcome == "created"

    fetched = store.fetch_node(conversation.node_id, None, run)
    assert fetched is not None
    stored_document, _stored_provenance = fetched
    assert store.is_unchanged(stored_document, rendered), (
        "the stored document must already match the very next render; a "
        f"mismatch means kb's parse/generate round trip lost content.\n"
        f"stored:   {stored_document!r}\nrendered: {rendered!r}"
    )

    # A second write must be a true no-op, matching the rehearsal's
    # acceptance check: a re-run reports 0 updated.
    second_fetch = store.fetch_node(conversation.node_id, None, run)
    assert second_fetch is not None
    second_stored, _ = second_fetch
    assert store.is_unchanged(second_stored, rendered)


#: A second, independent trigger shape from the same rehearsal: a shell
#: comment (`# ...`) quoted inside a fenced code block, landing as a
#: heading's direct child. `escape_org` does not touch it -- a bare `#` is
#: not org-structural at column 0 the way `*`/`#+` are -- so kb's own parser
#: is what has to round-trip it, and `try_parse_heading`'s child-collection
#: loop used to drop it (see `src/parser.rs::try_parse_non_heading_block`).
_COMMENT_UNDER_HEADING_RECORDS: list[object] = [
    {
        "type": "user",
        "uuid": "u1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:00Z",
        "cwd": "/Users/op/Projects/kb/main",
        "message": {"id": "m1", "content": "What is the exact push command?"},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:05Z",
        "message": {
            "id": "m2",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "The exact commands:\n\n```bash\n"
                        "# here, on the laptop\nyadm push origin main\n```"
                    ),
                }
            ],
        },
    },
]


def test_a_transcript_with_a_fenced_comment_round_trips_unchanged(tmp_path: Path) -> None:
    """Regression for the second T017 false-positive cause.

    Before the `src/parser.rs::try_parse_non_heading_block` fix, a `#
    comment`-shaped line inside a fenced code block, directly under a
    heading, was silently dropped on `kb create` -- the child-collection
    loop tried only seven block kinds and then a bare paragraph fallback,
    and `is_paragraph_line` correctly refuses a `# ` line (it belongs to
    `try_parse_comment`, which the loop never tried). This is exactly the
    shape sampled from a real rehearsal transcript (`cc-9dc96a20...`,
    `PLAN-20260923-project-identity` T017's Operator Guidance Log entry).
    """
    conversation = sources.parse_claude_code_transcript(
        _COMMENT_UNDER_HEADING_RECORDS, "cafef00d", "personal"
    )
    assert conversation is not None
    rendered = model.render(conversation, "Fenced comment regression")
    # The fixture is pointless if the comment line is not actually present.
    assert "\n# here, on the laptop\n" in rendered

    run = _kb_runner(_kb_binary(), tmp_path / "store", tmp_path / "index.db")
    outcome = store.write_node(conversation, "Fenced comment regression", None, run, exists=False)
    assert outcome == "created"

    fetched = store.fetch_node(conversation.node_id, None, run)
    assert fetched is not None
    stored_document, _stored_provenance = fetched
    assert store.is_unchanged(stored_document, rendered), (
        "the stored document must already match the very next render; a "
        f"mismatch means kb's parse/generate round trip lost content.\n"
        f"stored:   {stored_document!r}\nrendered: {rendered!r}"
    )


#: A third, independent trigger shape from the same rehearsal: an assistant
#: reply that opens a numbered item (`6. ...`) and continues with plain `-`
#: bullets, which real model replies do. `src/parser.rs::try_parse_list`
#: used to swallow every subsequent bullet into the first item's
#: `ListType`, so every `-` item was silently renumbered `7.`, `8.`, … on
#: `kb create` -- content preserved, marker kind wrong, permanently
#: disagreeing with every later render of the same reply.
_MIXED_LIST_MARKER_RECORDS: list[object] = [
    {
        "type": "user",
        "uuid": "u1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:00Z",
        "cwd": "/Users/op/Projects/kb/main",
        "message": {"id": "m1", "content": "Summarize the requests."},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:05Z",
        "message": {
            "id": "m2",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "6. **All user messages:**\n"
                        '- "/pr-auditor 1543"\n'
                        '- "proceed"\n'
                        '- "can we fix these issues?"'
                    ),
                }
            ],
        },
    },
]


def test_a_transcript_with_a_mixed_marker_list_round_trips_unchanged(tmp_path: Path) -> None:
    """Regression for the third T017 false-positive cause.

    Sampled and minimized from a real rehearsal transcript
    (`cc-0d9071e4...`, `PLAN-20260923-project-identity` T017's Operator
    Guidance Log entry): a reply's `-` items were coming back from kb
    renumbered as `7.`, `8.`, … after the very first `kb create`.
    """
    conversation = sources.parse_claude_code_transcript(
        _MIXED_LIST_MARKER_RECORDS, "0ddba11c", "personal"
    )
    assert conversation is not None
    rendered = model.render(conversation, "Mixed list marker regression")
    # The fixture is pointless if the dash items are not actually present.
    assert '\n- "/pr-auditor 1543"\n' in rendered

    run = _kb_runner(_kb_binary(), tmp_path / "store", tmp_path / "index.db")
    outcome = store.write_node(
        conversation, "Mixed list marker regression", None, run, exists=False
    )
    assert outcome == "created"

    fetched = store.fetch_node(conversation.node_id, None, run)
    assert fetched is not None
    stored_document, _stored_provenance = fetched
    assert store.is_unchanged(stored_document, rendered), (
        "the stored document must already match the very next render; a "
        f"mismatch means kb renumbered a `-` item as `N.`.\n"
        f"stored:   {stored_document!r}\nrendered: {rendered!r}"
    )


#: A fourth, independent trigger shape from the same rehearsal: prose
#: quoting an org block keyword this parser does not implement
#: (`#+begin_export`, `#+begin_signature` -- real org keywords, just not
#: ones `src/parser.rs` builds an AST node for). `escape_org` indents them
#: because they start with `#+`; `src/parser.rs::is_paragraph_line` used to
#: exclude *any* `#+begin_...` line from paragraph content as if it opened
#: one of the three block kinds this parser does implement, so an
#: unimplemented one matched no block parser either and was dropped.
_UNIMPLEMENTED_BLOCK_KEYWORD_RECORDS: list[object] = [
    {
        "type": "user",
        "uuid": "u1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:00Z",
        "cwd": "/Users/op/Projects/kb/main",
        "message": {"id": "m1", "content": "Show the org-msg signature snippet."},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:05Z",
        "message": {
            "id": "m2",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "```elisp\n"
                        '(defvar my/org-msg-signature "\n\n'
                        "#+begin_signature\n#+begin_export html\n"
                        "<table><tr><td>me@example.com</td></tr></table>\n"
                        '#+end_export\n#+end_signature\n")\n```'
                    ),
                }
            ],
        },
    },
]


def test_a_transcript_with_an_unimplemented_block_keyword_round_trips_unchanged(
    tmp_path: Path,
) -> None:
    """Regression for the fourth T017 false-positive cause.

    Sampled and minimized from a real rehearsal transcript
    (`cc-b0f241f4...`, `PLAN-20260923-project-identity` T017's Operator
    Guidance Log entry): an Emacs `org-msg` signature variable, quoted in a
    fenced code block, whose nested `#+begin_export`/`#+begin_signature`
    lines vanished on the very first `kb create`.
    """
    conversation = sources.parse_claude_code_transcript(
        _UNIMPLEMENTED_BLOCK_KEYWORD_RECORDS, "decafbad", "personal"
    )
    assert conversation is not None
    rendered = model.render(conversation, "Unimplemented block keyword regression")
    # The fixture is pointless if escape_org did not actually trigger.
    assert "\n #+begin_signature\n" in rendered

    run = _kb_runner(_kb_binary(), tmp_path / "store", tmp_path / "index.db")
    outcome = store.write_node(
        conversation, "Unimplemented block keyword regression", None, run, exists=False
    )
    assert outcome == "created"

    fetched = store.fetch_node(conversation.node_id, None, run)
    assert fetched is not None
    stored_document, _stored_provenance = fetched
    assert store.is_unchanged(stored_document, rendered), (
        "the stored document must already match the very next render; a "
        f"mismatch means kb dropped an unimplemented block keyword.\n"
        f"stored:   {stored_document!r}\nrendered: {rendered!r}"
    )


#: A fifth, independent trigger shape from the same rehearsal (redesigned
#: after a first attempt regressed -- see
#: `src/parser.rs::is_list_continuation_line`'s doc comment for the full
#: account): a bullet's prose, a blank line, then an indented markdown
#: table -- a "loose list" shape real model replies produce routinely.
_LOOSE_LIST_TABLE_RECORDS: list[object] = [
    {
        "type": "user",
        "uuid": "u1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:00Z",
        "cwd": "/Users/op/Projects/kb/main",
        "message": {"id": "m1", "content": "What does the pad send today?"},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:05Z",
        "message": {
            "id": "m2",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "- **What the pad sends today**: only keys 1 and 3 need rules.\n\n"
                        "  | Control | Sends now | Rule |\n"
                        "  |---|---|---|\n"
                        "  | key 1 | `al_local_machine_browser` | previous track |"
                    ),
                }
            ],
        },
    },
]


def test_a_transcript_with_a_loose_list_table_round_trips_unchanged(tmp_path: Path) -> None:
    """Regression for the fifth T017 false-positive cause.

    Sampled and minimized from a real rehearsal transcript
    (`cc-ec002615...`, `PLAN-20260923-project-identity` T017's Operator
    Guidance Log entry): an indented table continuing a bullet across a
    blank line came back from kb with its indentation stripped.
    """
    conversation = sources.parse_claude_code_transcript(
        _LOOSE_LIST_TABLE_RECORDS, "fadedbee", "personal"
    )
    assert conversation is not None
    rendered = model.render(conversation, "Loose list table regression")
    # The fixture is pointless if the indented table is not actually present.
    assert "\n\n  | Control | Sends now | Rule |\n" in rendered

    run = _kb_runner(_kb_binary(), tmp_path / "store", tmp_path / "index.db")
    outcome = store.write_node(conversation, "Loose list table regression", None, run, exists=False)
    assert outcome == "created"

    fetched = store.fetch_node(conversation.node_id, None, run)
    assert fetched is not None
    stored_document, _stored_provenance = fetched
    assert store.is_unchanged(stored_document, rendered), (
        "the stored document must already match the very next render; a "
        f"mismatch means kb dropped the table's indentation.\n"
        f"stored:   {stored_document!r}\nrendered: {rendered!r}"
    )


#: The shape that regressed a first attempt at the fix above: a fresh
#: numbered list opening right after an escaped bold line
#: (`escape_org` indents `**Suggested order:**` because it starts with
#: `*`), itself right after a blank line that ends a *prior* numbered list.
_ESCAPED_BOLD_THEN_FRESH_LIST_RECORDS: list[object] = [
    {
        "type": "user",
        "uuid": "u1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:00Z",
        "cwd": "/Users/op/Projects/kb/main",
        "message": {"id": "m1", "content": "What's the suggested order?"},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:05Z",
        "message": {
            "id": "m2",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "16. item sixteen\n"
                        "17. item seventeen, blocked by items 6 and 7.\n\n"
                        "**Suggested order:**\n"
                        "1. Clear items 1-5\n"
                        "2. Decide item 6"
                    ),
                }
            ],
        },
    },
]


def test_a_fresh_list_after_an_escaped_bold_line_keeps_its_own_numbering(
    tmp_path: Path,
) -> None:
    """Regression for the bug the first loose-list fix caused.

    Sampled and minimized from a real rehearsal transcript
    (`cc-7b992f21...`): a fresh `1. Clear items...` list, opening right
    after an escaped `**Suggested order:**` line, must not be folded into
    the prior `16./17.` list and renumbered from `18.`.
    """
    conversation = sources.parse_claude_code_transcript(
        _ESCAPED_BOLD_THEN_FRESH_LIST_RECORDS, "deadfa11", "personal"
    )
    assert conversation is not None
    rendered = model.render(conversation, "Escaped bold then fresh list regression")
    # The fixture is pointless if escape_org did not actually trigger, or if
    # the fresh list does not actually start at 1.
    assert "\n **Suggested order:**\n" in rendered
    assert "\n1. Clear items 1-5\n" in rendered

    run = _kb_runner(_kb_binary(), tmp_path / "store", tmp_path / "index.db")
    outcome = store.write_node(
        conversation, "Escaped bold then fresh list regression", None, run, exists=False
    )
    assert outcome == "created"

    fetched = store.fetch_node(conversation.node_id, None, run)
    assert fetched is not None
    stored_document, _stored_provenance = fetched
    assert store.is_unchanged(stored_document, rendered), (
        "the stored document must already match the very next render; a "
        f"mismatch means the fresh list was folded into the prior one.\n"
        f"stored:   {stored_document!r}\nrendered: {rendered!r}"
    )


#: A sixth, independent trigger shape, found by a final rehearsal
#: verification after the first five were fixed: an indented `# comment`
#: line, the shape a real Python/shell comment quoted inside a fenced code
#: block takes. Real org syntax permits an indented comment (Worg's Org
#: Syntax §2.3 excludes only headings, inlinetasks, footnote definitions,
#: and diary sexps from being indented), but `src/parser.rs`'s
#: `Block::Comment` carries only its text, no position, so the generator
#: always re-emitted it at column 0 -- losing the indentation on every
#: write.
_INDENTED_COMMENT_RECORDS: list[object] = [
    {
        "type": "user",
        "uuid": "u1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:00Z",
        "cwd": "/Users/op/Projects/kb/main",
        "message": {"id": "m1", "content": "What does this test do?"},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:05Z",
        "message": {
            "id": "m2",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "```python\n"
                        "def test_mirror_count_sql_excludes_temp_namespaces():\n"
                        "    # A concurrent session's temp tables appear as pg_temp_N.\n"
                        "    assert True\n"
                        "```"
                    ),
                }
            ],
        },
    },
]


def test_a_transcript_with_an_indented_comment_round_trips_unchanged(tmp_path: Path) -> None:
    """Regression for the sixth T017 false-positive cause (comment half).

    Sampled and minimized from real rehearsal transcripts
    (`cc-b90c37fc...`, `cc-c58df4cd...`): an indented Python comment quoted
    inside a fenced code block came back from kb with its indentation
    stripped.
    """
    conversation = sources.parse_claude_code_transcript(
        _INDENTED_COMMENT_RECORDS, "c0ffee11", "personal"
    )
    assert conversation is not None
    rendered = model.render(conversation, "Indented comment regression")
    # The fixture is pointless if the indented comment is not actually present.
    assert "\n    # A concurrent session's temp tables" in rendered

    run = _kb_runner(_kb_binary(), tmp_path / "store", tmp_path / "index.db")
    outcome = store.write_node(conversation, "Indented comment regression", None, run, exists=False)
    assert outcome == "created"

    fetched = store.fetch_node(conversation.node_id, None, run)
    assert fetched is not None
    stored_document, _stored_provenance = fetched
    assert store.is_unchanged(stored_document, rendered), (
        "the stored document must already match the very next render; a "
        f"mismatch means kb stripped the comment's indentation.\n"
        f"stored:   {stored_document!r}\nrendered: {rendered!r}"
    )


#: The horizontal-rule half of the same sixth cause: `Block::HorizontalRule`
#: is a bare unit variant carrying no data at all, so an indented `-----`
#: (an ASCII-table rule quoted inside a transcript) lost its indentation
#: the same way.
_INDENTED_HORIZONTAL_RULE_RECORDS: list[object] = [
    {
        "type": "user",
        "uuid": "u1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:00Z",
        "cwd": "/Users/op/Projects/kb/main",
        "message": {"id": "m1", "content": "What's the savings breakdown?"},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:05Z",
        "message": {
            "id": "m2",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Elastic IPs released       16   x $3.65     58\n"
                        "                                         -----\n"
                        "                                           116"
                    ),
                }
            ],
        },
    },
]


def test_a_transcript_with_an_indented_horizontal_rule_round_trips_unchanged(
    tmp_path: Path,
) -> None:
    """Regression for the sixth T017 false-positive cause (rule half).

    Sampled and minimized from a real rehearsal transcript
    (`cc-89c9fe92...`): an indented ASCII-table rule came back from kb with
    its indentation stripped.
    """
    conversation = sources.parse_claude_code_transcript(
        _INDENTED_HORIZONTAL_RULE_RECORDS, "decade11", "personal"
    )
    assert conversation is not None
    rendered = model.render(conversation, "Indented horizontal rule regression")
    # The fixture is pointless if the indented rule is not actually present.
    assert "\n                                         -----\n" in rendered

    run = _kb_runner(_kb_binary(), tmp_path / "store", tmp_path / "index.db")
    outcome = store.write_node(
        conversation, "Indented horizontal rule regression", None, run, exists=False
    )
    assert outcome == "created"

    fetched = store.fetch_node(conversation.node_id, None, run)
    assert fetched is not None
    stored_document, _stored_provenance = fetched
    assert store.is_unchanged(stored_document, rendered), (
        "the stored document must already match the very next render; a "
        f"mismatch means kb stripped the rule's indentation.\n"
        f"stored:   {stored_document!r}\nrendered: {rendered!r}"
    )


#: T019 pattern A (`PLAN-20260923-project-identity`): `#+name:`/`#+begin_src`/
#: `#+end_src` lines quoted in an assistant reply about a noweb src block all
#: start with `#+`, so `escape_org` indents every one of them by one column —
#: the same mechanism `_ESCAPED_SHAPE_RECORDS` above exercises for a bare
#: `#+title:`-shaped line. Before `try_parse_src_block` required column 0
#: (T019), the indented `#+begin_src`/`#+end_src` pair was still recognized
#: as a real `Block::SrcBlock`, which carries no indentation field, so the
#: leading space was stripped on every render even though the sibling
#: `#+name:` keyword line survived (T017 already required column 0 there).
#: Minimized from the T018 rehearsal's 17 remaining Codex differences.
_INDENTED_SRC_BLOCK_RECORDS: list[object] = [
    {
        "type": "user",
        "uuid": "u1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:00Z",
        "cwd": "/Users/op/Projects/kb/main",
        "message": {"id": "m1", "content": "Show me a noweb src block example."},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:05Z",
        "message": {
            "id": "m2",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Here is one:\n\n#+name: mysrc\n#+begin_src org :noweb yes\n"
                        "example content\n#+end_src"
                    ),
                }
            ],
        },
    },
]


def test_a_transcript_with_an_indented_src_block_round_trips_unchanged(tmp_path: Path) -> None:
    """Regression for T019 pattern A.

    Before the `src/parser.rs::try_parse_src_block` column-0 fix, the
    `#+begin_src`/`#+end_src` pair -- escaped by `escape_org` because each
    line starts with `#+` -- came back from kb with its indentation
    stripped.
    """
    conversation = sources.parse_claude_code_transcript(
        _INDENTED_SRC_BLOCK_RECORDS, "5010c0de", "personal"
    )
    assert conversation is not None
    rendered = model.render(conversation, "Indented src block regression")
    # The fixture is pointless if escape_org did not actually trigger.
    assert "\n #+name: mysrc\n" in rendered
    assert "\n #+begin_src org :noweb yes\n" in rendered
    assert "\n #+end_src\n" in rendered

    run = _kb_runner(_kb_binary(), tmp_path / "store", tmp_path / "index.db")
    outcome = store.write_node(
        conversation, "Indented src block regression", None, run, exists=False
    )
    assert outcome == "created"

    fetched = store.fetch_node(conversation.node_id, None, run)
    assert fetched is not None
    stored_document, _stored_provenance = fetched
    assert store.is_unchanged(stored_document, rendered), (
        "the stored document must already match the very next render; a "
        f"mismatch means kb stripped the src block's indentation.\n"
        f"stored:   {stored_document!r}\nrendered: {rendered!r}"
    )


#: T019 pattern B (`PLAN-20260923-project-identity`), table half: a markdown
#: table quoted with its own indentation (not touched by `escape_org`, which
#: only escapes lines starting with `*`/`#+`) came back from kb with its
#: indentation stripped, because `try_parse_table` used to recognize a
#: table's opening `|` after trimming.
_INDENTED_TABLE_RECORDS: list[object] = [
    {
        "type": "user",
        "uuid": "u1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:00Z",
        "cwd": "/Users/op/Projects/kb/main",
        "message": {"id": "m1", "content": "What's the target schema?"},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:05Z",
        "message": {
            "id": "m2",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Nested under the note:\n\n | Field | Type |\n |---|---|\n | id | UUID |"
                    ),
                }
            ],
        },
    },
]


def test_a_transcript_with_an_indented_table_round_trips_unchanged(tmp_path: Path) -> None:
    """Regression for T019 pattern B (table half).

    Before the `src/parser.rs::try_parse_table` column-0 fix, an indented
    table quoted in a reply came back from kb with its indentation stripped.
    """
    conversation = sources.parse_claude_code_transcript(
        _INDENTED_TABLE_RECORDS, "5ca1ab1e", "personal"
    )
    assert conversation is not None
    rendered = model.render(conversation, "Indented table regression")
    # The fixture is pointless if the indented table is not actually present.
    assert "\n | Field | Type |\n" in rendered

    run = _kb_runner(_kb_binary(), tmp_path / "store", tmp_path / "index.db")
    outcome = store.write_node(conversation, "Indented table regression", None, run, exists=False)
    assert outcome == "created"

    fetched = store.fetch_node(conversation.node_id, None, run)
    assert fetched is not None
    stored_document, _stored_provenance = fetched
    assert store.is_unchanged(stored_document, rendered), (
        "the stored document must already match the very next render; a "
        f"mismatch means kb stripped the table's indentation.\n"
        f"stored:   {stored_document!r}\nrendered: {rendered!r}"
    )


#: T019 pattern B (`PLAN-20260923-project-identity`), drawer half: a
#: `:PROPERTIES:` drawer quoted with its own indentation (also untouched by
#: `escape_org`, which does not recognize `:` as structural) came back from
#: kb with its indentation stripped.
_INDENTED_DRAWER_RECORDS: list[object] = [
    {
        "type": "user",
        "uuid": "u1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:00Z",
        "cwd": "/Users/op/Projects/kb/main",
        "message": {"id": "m1", "content": "Show me a CUSTOM_ID example."},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:05Z",
        "message": {
            "id": "m2",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Here's a minimal one:\n\n :PROPERTIES:\n :CUSTOM_ID: n-example\n :END:"
                    ),
                }
            ],
        },
    },
]


def test_a_transcript_with_an_indented_property_drawer_round_trips_unchanged(
    tmp_path: Path,
) -> None:
    """Regression for T019 pattern B (drawer half).

    Before the `src/parser.rs::try_parse_property_drawer` column-0 fix, an
    indented `:PROPERTIES:` drawer quoted in a reply came back from kb with
    its indentation stripped.
    """
    conversation = sources.parse_claude_code_transcript(
        _INDENTED_DRAWER_RECORDS, "d2a41e2d", "personal"
    )
    assert conversation is not None
    rendered = model.render(conversation, "Indented property drawer regression")
    # The fixture is pointless if the indented drawer is not actually present.
    assert "\n :PROPERTIES:\n" in rendered

    run = _kb_runner(_kb_binary(), tmp_path / "store", tmp_path / "index.db")
    outcome = store.write_node(
        conversation, "Indented property drawer regression", None, run, exists=False
    )
    assert outcome == "created"

    fetched = store.fetch_node(conversation.node_id, None, run)
    assert fetched is not None
    stored_document, _stored_provenance = fetched
    assert store.is_unchanged(stored_document, rendered), (
        "the stored document must already match the very next render; a "
        f"mismatch means kb stripped the drawer's indentation.\n"
        f"stored:   {stored_document!r}\nrendered: {rendered!r}"
    )


#: T019 pattern C (`PLAN-20260923-project-identity`): two adjacent numbered
#: lists with no blank-line separator, the second restarting at `1.`. Before
#: `try_parse_list`'s restart check, the second list was folded into the
#: first and renumbered `6.`, `7.`, continuing the first list's count.
_ADJACENT_RESTARTING_LISTS_RECORDS: list[object] = [
    {
        "type": "user",
        "uuid": "u1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:00Z",
        "cwd": "/Users/op/Projects/kb/main",
        "message": {"id": "m1", "content": "List the rules, then restate two of them."},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:05Z",
        "message": {
            "id": "m2",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "1. first rule\n2. second rule\n3. third rule\n4. fourth rule\n"
                        "5. fifth rule\n1. sixth rule restated as one\n"
                        "2. seventh rule restated as two"
                    ),
                }
            ],
        },
    },
]


def test_a_transcript_with_adjacent_restarting_lists_round_trips_unchanged(
    tmp_path: Path,
) -> None:
    """Regression for T019 pattern C.

    Before the `src/parser.rs::try_parse_list` restart check, a fresh
    numbered list starting right after another (no blank line, number
    strictly lower) came back from kb folded into the first list and
    renumbered continuing its count.
    """
    conversation = sources.parse_claude_code_transcript(
        _ADJACENT_RESTARTING_LISTS_RECORDS, "1157ed1e", "personal"
    )
    assert conversation is not None
    rendered = model.render(conversation, "Adjacent restarting lists regression")
    # The fixture is pointless if both lists are not actually present.
    assert "\n5. fifth rule\n1. sixth rule restated as one\n" in rendered

    run = _kb_runner(_kb_binary(), tmp_path / "store", tmp_path / "index.db")
    outcome = store.write_node(
        conversation, "Adjacent restarting lists regression", None, run, exists=False
    )
    assert outcome == "created"

    fetched = store.fetch_node(conversation.node_id, None, run)
    assert fetched is not None
    stored_document, _stored_provenance = fetched
    assert store.is_unchanged(stored_document, rendered), (
        "the stored document must already match the very next render; a "
        f"mismatch means kb merged and renumbered the second list.\n"
        f"stored:   {stored_document!r}\nrendered: {rendered!r}"
    )


#: T019 pattern D (`PLAN-20260923-project-identity`): a `rustc`-style
#: diagnostic quoted verbatim, whose gutter column is sometimes just
#: whitespace and a lone `|`. Before the `is_paragraph_line` fix, that line
#: matched neither `try_parse_table` (which already declines a
#: one-character row) nor the paragraph fallback (which excluded any
#: trimmed `|...|` shape, including a single `|`), so it fell through to
#: residue and was silently dropped.
_GUTTER_ONLY_PIPE_RECORDS: list[object] = [
    {
        "type": "user",
        "uuid": "u1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:00Z",
        "cwd": "/Users/op/Projects/kb/main",
        "message": {"id": "m1", "content": "Why does this fail to build?"},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "entrypoint": "cli",
        "timestamp": "2026-09-01T10:00:05Z",
        "message": {
            "id": "m2",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Build failed:\n\n   --> src/lib.rs:10:4\n    |\n"
                        "10  | fn example() -> ()\n    |    ^^^^^^^\n    |\n"
                        "    = note: unused"
                    ),
                }
            ],
        },
    },
]


def test_a_transcript_with_a_gutter_only_pipe_line_round_trips_unchanged(
    tmp_path: Path,
) -> None:
    """Regression for T019 pattern D.

    Before the `src/parser.rs::is_paragraph_line` fix, a gutter-only `|`
    line quoted from a compiler diagnostic was silently dropped on
    `kb create`.
    """
    conversation = sources.parse_claude_code_transcript(
        _GUTTER_ONLY_PIPE_RECORDS, "9011e777", "personal"
    )
    assert conversation is not None
    rendered = model.render(conversation, "Gutter-only pipe line regression")
    # The fixture is pointless if the gutter-only line is not actually present.
    assert "\n    |\n" in rendered

    run = _kb_runner(_kb_binary(), tmp_path / "store", tmp_path / "index.db")
    outcome = store.write_node(
        conversation, "Gutter-only pipe line regression", None, run, exists=False
    )
    assert outcome == "created"

    fetched = store.fetch_node(conversation.node_id, None, run)
    assert fetched is not None
    stored_document, _stored_provenance = fetched
    assert store.is_unchanged(stored_document, rendered), (
        "the stored document must already match the very next render; a "
        f"mismatch means kb dropped the gutter-only pipe line.\n"
        f"stored:   {stored_document!r}\nrendered: {rendered!r}"
    )


# ── Posting to the ingest endpoint (T022) ──────────────────────────────────


def test_only_the_sessions_the_server_lacks_are_posted():
    # The comparison is the whole of reconciliation, so it is a pure function
    # over two id sets rather than something the posting loop decides as it
    # goes: what is skipped has to be inspectable without a server.
    served = ["cc-a", "cc-c"]
    local = ["cc-c", "cc-b", "cc-a", "cc-b"]
    assert ingest.missing_ids(local, served) == ["cc-b"]


def test_a_server_holding_nothing_means_everything_is_missing():
    assert ingest.missing_ids(["cc-b", "cc-a"], []) == ["cc-a", "cc-b"]


def test_a_submission_names_the_record_and_carries_the_rendered_document():
    submission = ingest.submission_for("cc-1", "* Session\n\nbody.\n")
    assert submission["id"] == "cc-1"
    assert submission["corpus"] == "kb"
    assert str(submission["document"]).startswith("* Session")


def test_posting_sends_the_submission_to_the_ingest_path():
    sent: list[tuple[str, dict[str, str], bytes]] = []

    def transport(url: str, headers: dict[str, str], body: bytes) -> bytes:
        sent.append((url, headers, body))
        return b'{"accepted":"cc-1"}'

    client = ingest.IngestClient("http://kb.example:8080", "a-token", transport)
    client.post(ingest.submission_for("cc-1", "* Session\n\nbody.\n"))

    url, headers, body = sent[0]
    assert url == "http://kb.example:8080/ingest"
    assert headers["Authorization"] == "Bearer a-token"
    assert json.loads(body)["id"] == "cc-1"


def test_the_served_ids_are_read_from_the_listing_endpoint():
    def transport(url: str, headers: dict[str, str], body: bytes) -> bytes:
        assert url == "http://kb.example:8080/ingest/ids?corpus=kb"
        assert headers["Authorization"] == "Bearer a-token"
        assert body == b""
        return b'["cc-1","cc-2"]'

    client = ingest.IngestClient("http://kb.example:8080", "a-token", transport)
    assert client.served_ids() == ["cc-1", "cc-2"]


def test_a_trailing_slash_on_the_server_url_does_not_double_up():
    seen: list[str] = []

    def transport(url: str, _headers: dict[str, str], _body: bytes) -> bytes:
        seen.append(url)
        return b"[]"

    ingest.IngestClient("http://kb.example:8080/", "t", transport).served_ids()
    assert seen == ["http://kb.example:8080/ingest/ids?corpus=kb"]


def test_a_refused_post_is_reported_rather_than_swallowed():
    def transport(_url: str, _headers: dict[str, str], _body: bytes) -> bytes:
        raise OSError("connection refused")

    client = ingest.IngestClient("http://kb.example:8080", "a-token", transport)
    with pytest.raises(ingest.IngestError, match="connection refused"):
        client.post(ingest.submission_for("cc-1", "* S\n\nb.\n"))


def test_an_ingest_client_needs_a_token():
    # An unauthenticated post is refused by the server, so building a client
    # without a token produces a run whose every post fails at the far end
    # rather than one that fails here where the cause is legible.
    with pytest.raises(ingest.IngestError, match="token"):
        ingest.IngestClient("http://kb.example:8080", "", lambda _u, _h, _b: b"")


def _posting_args(**overrides) -> argparse.Namespace:
    """Namespace with the fields the posting loop reads."""
    defaults = {
        "dry_run": False,
        "title_model": "haiku",
        "db": None,
        "reconcile": False,
        "project": None,
        "project_source": None,
        "remote": None,
        "context": None,
        "domains": None,
        "harness": None,
        "model": None,
        "clanker_session": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class _RecordingClient:
    """An IngestClient stand-in that records what it was asked to send."""

    def __init__(self, served: list[str] | None = None, fail: set[str] | None = None) -> None:
        self.posted: list[dict[str, object]] = []
        self._served = served or []
        self._fail = fail or set()

    def served_ids(self, corpus: str = "kb") -> list[str]:  # noqa: ARG002
        return self._served

    def post(self, submission: dict[str, object]) -> None:
        if submission["id"] in self._fail:
            raise ingest.IngestError("the server refused it")
        self.posted.append(submission)


def _stub_title(monkeypatch) -> None:
    """Stop title derivation from shelling out to claude."""
    monkeypatch.setattr(kbi, "derive_title", lambda _conversation, _cache, _run, _model: "A Title")


def test_posting_a_conversation_with_a_cwd_resolves_and_carries_provenance(monkeypatch, tmp_path):
    """Acceptance check for a submission's resolved project.

    A Claude Code transcript with `cwd`, and a stub `clanker`, posts a
    submission carrying the resolved project.
    """
    _stub_title(monkeypatch)
    conversation = dataclasses.replace(
        _conversation(
            model.Message(role="human", text="one", timestamp=""), tags=("claude-code-session",)
        ),
        cwd="/Users/op/Projects/kb/main",
    )
    client = _RecordingClient()
    run = _ClankerRunner("kb\tremote\tgithub.com/tftio/kb\n")

    reports = kbi.post_conversations(
        [conversation], _posting_args(), titles.TitleCache(tmp_path / "titles.json"), run, client
    )

    assert reports["codex"].created == 1
    assert client.posted[0]["provenance"] == {
        "project": "kb",
        "project_source": "remote",
        "remote": "github.com/tftio/kb",
        "cwd": "/Users/op/Projects/kb/main",
    }


def test_posting_sends_every_selected_conversation(monkeypatch, tmp_path):
    _stub_title(monkeypatch)
    conversations = [
        _conversation(model.Message(role="human", text="one", timestamp=""), title_hint="first"),
        _conversation(model.Message(role="human", text="two", timestamp=""), title_hint="second"),
    ]
    conversations[1] = dataclasses.replace(conversations[1], node_id="codex-second")
    client = _RecordingClient()

    reports = kbi.post_conversations(
        conversations,
        _posting_args(),
        titles.TitleCache(tmp_path / "titles.json"),
        lambda _argv, _stdin: "",
        client,
    )

    assert [submission["id"] for submission in client.posted] == [
        "codex-abc123",
        "codex-second",
    ]
    assert reports["codex"].created == 2
    assert reports["codex"].failed == []


def test_reconciling_skips_what_the_server_already_holds(monkeypatch, tmp_path):
    _stub_title(monkeypatch)
    held = _conversation(model.Message(role="human", text="one", timestamp=""))
    missing = dataclasses.replace(
        _conversation(model.Message(role="human", text="two", timestamp="")),
        node_id="codex-missing",
    )
    client = _RecordingClient(served=["codex-abc123"])

    reports = kbi.post_conversations(
        [held, missing],
        _posting_args(reconcile=True),
        titles.TitleCache(tmp_path / "titles.json"),
        lambda _argv, _stdin: "",
        client,
    )

    assert [submission["id"] for submission in client.posted] == ["codex-missing"]
    assert reports["codex"].skipped == 1
    assert reports["codex"].created == 1


def test_a_refused_post_is_recorded_and_the_rest_continue(monkeypatch, tmp_path):
    _stub_title(monkeypatch)
    bad = _conversation(model.Message(role="human", text="one", timestamp=""))
    good = dataclasses.replace(
        _conversation(model.Message(role="human", text="two", timestamp="")), node_id="codex-good"
    )
    client = _RecordingClient(fail={"codex-abc123"})

    reports = kbi.post_conversations(
        [bad, good],
        _posting_args(),
        titles.TitleCache(tmp_path / "titles.json"),
        lambda _argv, _stdin: "",
        client,
    )

    assert [submission["id"] for submission in client.posted] == ["codex-good"]
    assert reports["codex"].created == 1
    assert len(reports["codex"].failed) == 1
    assert "codex-abc123" in reports["codex"].failed[0]


def test_a_dry_run_posts_nothing(monkeypatch, tmp_path):
    _stub_title(monkeypatch)
    client = _RecordingClient()
    reports = kbi.post_conversations(
        [_conversation(model.Message(role="human", text="one", timestamp=""))],
        _posting_args(dry_run=True),
        titles.TitleCache(tmp_path / "titles.json"),
        lambda _argv, _stdin: "",
        client,
    )
    assert client.posted == []
    assert reports["codex"].created == 1
