"""Tests for `scripts/measure_reads.py`, over synthetic fixture JSONL.

Every fixture here is fabricated for this test file; none of it is drawn from
a real session log. `conftest.py` puts `scripts/` on the import path so this
module, a path-inserted script, can be imported by name; the transcript
importer is the installed `kb_import` package and needs no such path
insertion.
"""

from __future__ import annotations

import json
import stat
from datetime import date
from pathlib import Path

import measure_reads as mr
import pytest

# ── classify_cli_text ────────────────────────────────────────────────────────


def test_a_bare_kb_search_is_a_read() -> None:
    assert mr.classify_cli_text("kb search foo") == (True, False)


def test_kb_create_from_stdin_is_a_write() -> None:
    assert mr.classify_cli_text("cat notes.org | kb create --tag x") == (False, True)


def test_tags_add_is_a_write_but_bare_tags_is_a_read() -> None:
    assert mr.classify_cli_text("kb tags add clanker cc-123") == (False, True)
    assert mr.classify_cli_text("kb tags") == (True, False)


def test_tags_rm_and_merge_are_writes() -> None:
    assert mr.classify_cli_text("kb tags rm stale cc-1") == (False, True)
    assert mr.classify_cli_text("kb tags merge old new") == (False, True)


def test_kb_as_a_path_segment_or_word_fragment_does_not_match() -> None:
    assert mr.classify_cli_text("cd /opt/kb/bin && ls") == (False, False)
    assert mr.classify_cli_text("echo kbps") == (False, False)


def test_an_unrecognized_verb_matches_neither() -> None:
    assert mr.classify_cli_text("kb --help") == (False, False)
    assert mr.classify_cli_text("kb reindex") == (False, False)


def test_a_command_with_both_a_read_and_a_write_reports_both() -> None:
    assert mr.classify_cli_text("kb search x; kb create --tag y") == (True, True)


# ── _classify_skill ──────────────────────────────────────────────────────────


def test_kb_search_skill_is_a_read() -> None:
    assert mr._classify_skill("kb-search") == (True, False)


def test_kb_create_skill_is_a_write() -> None:
    assert mr._classify_skill("kb-create") == (False, True)


def test_a_non_kb_skill_is_neither() -> None:
    assert mr._classify_skill("mnene-search") == (False, False)


# ── Claude Code fixtures ─────────────────────────────────────────────────────


def _cc_user(text: str, *, cwd: str = "/Users/op/repo", ts: str = "2026-09-10T00:00:00Z") -> dict:
    return {
        "type": "user",
        "timestamp": ts,
        "cwd": cwd,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _cc_tool_result(*, cwd: str = "/Users/op/repo", ts: str = "2026-09-10T00:00:01Z") -> dict:
    return {
        "type": "user",
        "timestamp": ts,
        "cwd": cwd,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "kb-search"}],
        },
    }


def _cc_assistant_tool(
    name: str,
    tool_input: dict[str, object],
    *,
    cwd: str = "/Users/op/repo",
    ts: str = "2026-09-10T00:00:02Z",
) -> dict:
    return {
        "type": "assistant",
        "timestamp": ts,
        "cwd": cwd,
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tu1", "name": name, "input": tool_input}],
        },
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_claude_code_mcp_search_is_a_read_with_no_write() -> None:
    records = [
        _cc_user("please look this up"),
        _cc_assistant_tool("mcp__kb__search", {"query": "x"}),
    ]
    has_read, has_write, _mentioned, cwd = mr.classify_claude_code_records(records)
    assert (has_read, has_write) == (True, False)
    assert cwd == "/Users/op/repo"


def test_claude_code_mcp_put_is_a_write() -> None:
    records = [_cc_assistant_tool("mcp__kb__put", {"document": "..."})]
    has_read, has_write, _mentioned, _cwd = mr.classify_claude_code_records(records)
    assert (has_read, has_write) == (False, True)


def test_claude_code_kb_skill_invocation_counts_as_a_read() -> None:
    records = [_cc_assistant_tool("Skill", {"skill": "kb-search", "args": "foo"})]
    has_read, has_write, _mentioned, _cwd = mr.classify_claude_code_records(records)
    assert (has_read, has_write) == (True, False)


def test_claude_code_kb_write_skill_invocation_counts_as_a_write() -> None:
    records = [_cc_assistant_tool("Skill", {"skill": "kb-tags-add"})]
    has_read, has_write, _mentioned, _cwd = mr.classify_claude_code_records(records)
    assert (has_read, has_write) == (False, True)


def test_claude_code_bash_kb_invocation_is_detected() -> None:
    records = [_cc_assistant_tool("Bash", {"command": "kb search bookclub"})]
    has_read, has_write, _mentioned, _cwd = mr.classify_claude_code_records(records)
    assert (has_read, has_write) == (True, False)


def test_claude_code_session_with_no_kb_activity_reads_and_writes_nothing() -> None:
    records = [
        _cc_user("what's the weather"),
        _cc_assistant_tool("Bash", {"command": "ls -la"}),
    ]
    has_read, has_write, mentioned, _cwd = mr.classify_claude_code_records(records)
    assert (has_read, has_write, mentioned) == (False, False, False)


def test_operator_mention_of_kb_is_detected() -> None:
    records = [
        _cc_user("can you check kb for prior art on this"),
        _cc_assistant_tool("mcp__kb__search", {"query": "prior art"}),
    ]
    _has_read, _has_write, mentioned, _cwd = mr.classify_claude_code_records(records)
    assert mentioned is True


def test_operator_mention_requires_the_word_boundary() -> None:
    records = [_cc_user("what's a good kbps for this stream")]
    _has_read, _has_write, mentioned, _cwd = mr.classify_claude_code_records(records)
    assert mentioned is False


def test_operator_mention_matches_knowledge_base_phrase() -> None:
    records = [_cc_user("please search the knowledge base for this")]
    _has_read, _has_write, mentioned, _cwd = mr.classify_claude_code_records(records)
    assert mentioned is True


def test_a_tool_result_message_is_not_an_operator_mention() -> None:
    # The tool result's own content literally contains "kb-search", but it was
    # never typed by a human, so it must not count.
    records = [_cc_tool_result()]
    _has_read, _has_write, mentioned, _cwd = mr.classify_claude_code_records(records)
    assert mentioned is False


def test_a_system_reminder_listing_kb_skills_is_not_an_operator_mention() -> None:
    text = (
        "unrelated question<system-reminder>Available skills:\n"
        "- kb-search: search the knowledge base\n</system-reminder>"
    )
    records = [_cc_user(text)]
    _has_read, _has_write, mentioned, _cwd = mr.classify_claude_code_records(records)
    assert mentioned is False


def test_meta_and_sidechain_records_are_ignored() -> None:
    meta = {**_cc_user("mentions kb"), "isMeta": True}
    sidechain = {**_cc_assistant_tool("mcp__kb__search", {}), "isSidechain": True}
    has_read, _has_write, mentioned, _cwd = mr.classify_claude_code_records([meta, sidechain])
    assert (has_read, mentioned) == (False, False)


# ── Codex fixtures ───────────────────────────────────────────────────────────


def _codex_session_meta(cwd: str, ts: str = "2026-07-01T00:00:00Z") -> dict:
    return {
        "type": "session_meta",
        "timestamp": ts,
        "payload": {"session_id": "019f0000", "id": "019f0000", "timestamp": ts, "cwd": cwd},
    }


def _codex_function_call(name: str, arguments: str) -> dict:
    return {
        "type": "response_item",
        "payload": {"type": "function_call", "name": name, "arguments": arguments},
    }


def _codex_custom_tool_call(input_js: str) -> dict:
    return {
        "type": "response_item",
        "payload": {"type": "custom_tool_call", "name": "exec", "input": input_js},
    }


def test_codex_shell_function_call_with_kb_search_is_a_read() -> None:
    records = [
        _codex_session_meta("/Users/op/bookclub"),
        _codex_function_call(
            "shell", json.dumps({"command": ["bash", "-lc", "kb search bookclub"]})
        ),
    ]
    has_read, has_write, cwd = mr.classify_codex_records(records)
    assert (has_read, has_write) == (True, False)
    assert cwd == "/Users/op/bookclub"


def test_codex_custom_tool_call_exec_with_kb_create_is_a_write() -> None:
    records = [
        _codex_custom_tool_call(
            "const r = await tools.exec_command({\n"
            '  cmd: "kb create --tag x",\n'
            '  workdir: "/x"\n'
            "});"
        )
    ]
    has_read, has_write, _cwd = mr.classify_codex_records(records)
    assert (has_read, has_write) == (False, True)


def test_codex_session_with_no_kb_tool_call_is_neither() -> None:
    records = [
        _codex_session_meta("/Users/op/other"),
        _codex_function_call("shell", json.dumps({"command": ["bash", "-lc", "ls -la"]})),
    ]
    has_read, has_write, _cwd = mr.classify_codex_records(records)
    assert (has_read, has_write) == (False, False)


def test_codex_agent_prose_mentioning_a_kb_skill_by_name_is_not_a_call() -> None:
    # Plain assistant/user message text is never scanned, only tool-call
    # fields -- otherwise an agent merely discussing "kb-search" would count.
    records = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I will use kb-search next."}],
            },
        }
    ]
    has_read, has_write, _cwd = mr.classify_codex_records(records)
    assert (has_read, has_write) == (False, False)


# ── session_date / discovery / load_records ─────────────────────────────────


def test_session_date_prefers_the_earliest_record_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [_cc_user("hi", ts="2026-08-15T00:00:00Z")])
    assert mr.session_date(mr.load_records(path), path) == date(2026, 8, 15)


def test_session_date_falls_back_to_file_mtime_when_no_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text('{"type":"user"}\n')
    result = mr.session_date(mr.load_records(path), path)
    assert isinstance(result, date)


def test_load_records_skips_unparsable_lines(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text('{"type":"user"}\nnot json\n{"type":"assistant"}\n')
    assert len(mr.load_records(path)) == 2


def test_load_records_on_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert mr.load_records(tmp_path / "missing.jsonl") == []


def test_discover_sessions_finds_files_under_each_account_directory(tmp_path: Path) -> None:
    root = tmp_path / "claude"
    for account in ("personal", "work"):
        project_dir = root / account / "projects" / "-a-b"
        project_dir.mkdir(parents=True)
        (project_dir / "session.jsonl").write_text("{}\n")
    found = mr.discover_sessions(root, mr.CLAUDE_CODE_SESSION_GLOB)
    assert {account for _path, account in found} == {"personal", "work"}
    assert len(found) == 2


def test_discover_sessions_on_a_missing_root_is_empty(tmp_path: Path) -> None:
    assert mr.discover_sessions(tmp_path / "does-not-exist", mr.CLAUDE_CODE_SESSION_GLOB) == []


# ── end-to-end measure() over a fixture tree ────────────────────────────────


def _build_fixture_tree(root: Path) -> None:
    """Build a small tree combining both sources and both accounts.

    Layout:
        claude/personal/projects/p1/read.jsonl   -- MCP read, operator-prompted
        claude/personal/projects/p1/write.jsonl  -- CLI write, unprompted
        claude/work/projects/p2/none.jsonl       -- no kb activity
        codex/personal/sessions/2026/07/01/rollout-x.jsonl -- CLI read
    """
    cc_personal = root / "claude" / "personal" / "projects" / "p1"
    cc_personal.mkdir(parents=True)
    _write_jsonl(
        cc_personal / "read.jsonl",
        [
            _cc_user("check kb for this please", ts="2026-09-10T00:00:00Z"),
            _cc_assistant_tool("mcp__kb__search", {"query": "x"}, ts="2026-09-10T00:00:01Z"),
        ],
    )
    _write_jsonl(
        cc_personal / "write.jsonl",
        [_cc_assistant_tool("Bash", {"command": "kb create --tag x"}, ts="2026-09-11T00:00:00Z")],
    )

    cc_work = root / "claude" / "work" / "projects" / "p2"
    cc_work.mkdir(parents=True)
    _write_jsonl(cc_work / "none.jsonl", [_cc_user("what time is it", ts="2026-09-12T00:00:00Z")])

    codex_personal = root / "codex" / "personal" / "sessions" / "2026" / "07" / "01"
    codex_personal.mkdir(parents=True)
    _write_jsonl(
        codex_personal / "rollout-x.jsonl",
        [
            _codex_session_meta("/Users/op/bookclub", ts="2026-07-01T00:00:00Z"),
            _codex_function_call("shell", json.dumps({"command": ["bash", "-lc", "kb get abc"]})),
        ],
    )


def test_measure_counts_each_source_and_account_separately(tmp_path: Path) -> None:
    _build_fixture_tree(tmp_path)
    report = mr.measure(
        claude_code_root=tmp_path / "claude",
        codex_root=tmp_path / "codex",
        since=None,
        by_project=False,
        resolver=None,
    )
    personal = report.by_source_account["claude-code/personal"]
    assert (personal.sessions, personal.with_read, personal.with_write) == (2, 1, 1)
    assert personal.read_sessions_with_operator_mention == 1

    work = report.by_source_account["claude-code/work"]
    assert (work.sessions, work.with_read, work.with_write) == (1, 0, 0)

    codex = report.by_source_account["codex/personal"]
    assert (codex.sessions, codex.with_read, codex.with_write) == (1, 1, 0)
    # Codex rows never compute an operator mention.
    assert codex.read_sessions_with_operator_mention == 0


def test_measure_respects_since(tmp_path: Path) -> None:
    _build_fixture_tree(tmp_path)
    report = mr.measure(
        claude_code_root=tmp_path / "claude",
        codex_root=tmp_path / "codex",
        since=date(2026, 9, 11),
        by_project=False,
        resolver=None,
    )
    personal = report.by_source_account["claude-code/personal"]
    # Only write.jsonl (2026-09-11) survives the filter; read.jsonl is 09-10.
    assert (personal.sessions, personal.with_read, personal.with_write) == (1, 0, 1)


def _stub_clanker(tmp_path: Path, output: str) -> Path:
    """A `clanker` stand-in for `project resolve --dir <path>`.

    Args:
        tmp_path: Test scratch directory the stub lives under.
        output: What `clanker project resolve` should print, tab-separated.

    Returns:
        The directory holding the stub, to pass as `--clanker-bin`.
    """
    bin_dir = tmp_path / "clanker-bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / "clanker"
    script.write_text(f"#!/bin/sh\nprintf '%s' '{output}'\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


def test_project_resolver_parses_the_tab_separated_line(tmp_path: Path) -> None:
    bin_dir = _stub_clanker(tmp_path, "kb\tremote\tgithub.com/tftio/kb")
    resolver = mr.ProjectResolver(str(bin_dir / "clanker"))
    assert resolver.resolve("/Users/op/kb") == "kb"


def test_project_resolver_returns_none_on_an_empty_slug(tmp_path: Path) -> None:
    bin_dir = _stub_clanker(tmp_path, "\t\t")
    resolver = mr.ProjectResolver(str(bin_dir / "clanker"))
    assert resolver.resolve("/Users/op/elsewhere") is None


def test_project_resolver_returns_none_when_the_binary_is_missing() -> None:
    resolver = mr.ProjectResolver("/no/such/clanker-binary")
    assert resolver.resolve("/Users/op/kb") is None


def test_project_resolver_caches_by_working_directory(tmp_path: Path) -> None:
    bin_dir = _stub_clanker(tmp_path, "kb\tremote\tgithub.com/tftio/kb")
    resolver = mr.ProjectResolver(str(bin_dir / "clanker"))
    assert resolver.resolve("/Users/op/kb") == "kb"
    # A second call for the same directory must not shell out again; prove it
    # by removing the binary and asserting the cached answer still comes back.
    (bin_dir / "clanker").unlink()
    assert resolver.resolve("/Users/op/kb") == "kb"


def test_measure_by_project_groups_sessions_under_their_resolved_slug(tmp_path: Path) -> None:
    _build_fixture_tree(tmp_path)
    bin_dir = _stub_clanker(tmp_path, "bookclub\tremote\tgithub.com/op/bookclub")
    resolver = mr.ProjectResolver(str(bin_dir / "clanker"))
    report = mr.measure(
        claude_code_root=tmp_path / "claude",
        codex_root=tmp_path / "codex",
        since=None,
        by_project=True,
        resolver=resolver,
    )
    assert report.by_project is not None
    codex_projects = report.by_project["codex/personal"]
    assert "bookclub" in codex_projects
    assert codex_projects["bookclub"].sessions == 1
    # Claude Code fixture sessions carry no origin remote the stub recognizes
    # by design of this test (the stub always answers "bookclub"), so this
    # only exercises grouping, not clanker's real resolution logic.
    personal_projects = report.by_project["claude-code/personal"]
    assert personal_projects["bookclub"].sessions == 2


# ── rendering ────────────────────────────────────────────────────────────────


def test_render_text_includes_every_source_account_key() -> None:
    counts = mr.Counts(sessions=3, with_read=1)
    report = mr.Report(by_source_account={"claude-code/personal": counts})
    text = mr.render_text(report)
    assert "claude-code/personal" in text
    assert "3" in text


def test_render_json_round_trips_counts() -> None:
    report = mr.Report(by_source_account={"codex/work": mr.Counts(sessions=5, with_write=2)})
    payload = json.loads(mr.render_json(report))
    assert payload["by_source_account"]["codex/work"]["sessions"] == 5
    assert payload["by_source_account"]["codex/work"]["with_write"] == 2


def test_render_json_includes_by_project_when_present() -> None:
    report = mr.Report(
        by_source_account={"claude-code/personal": mr.Counts(sessions=1)},
        by_project={"claude-code/personal": {"kb": mr.Counts(sessions=1), None: mr.Counts()}},
    )
    payload = json.loads(mr.render_json(report))
    assert "kb" in payload["by_project"]["claude-code/personal"]


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_since_flag_parses_iso_dates() -> None:
    parser = mr.build_parser()
    args = parser.parse_args(["--since", "2026-08-01"])
    assert args.since == date(2026, 8, 1)


def test_since_flag_rejects_a_malformed_date() -> None:
    parser = mr.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--since", "not-a-date"])


def test_main_over_an_empty_tree_exits_zero_and_prints_json(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    empty = tmp_path / "empty"
    code = mr.main(
        [
            "--json",
            "--claude-code-root",
            str(empty / "claude"),
            "--codex-root",
            str(empty / "codex"),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["by_source_account"] == {}


def test_main_over_the_fixture_tree_prints_a_text_table(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    _build_fixture_tree(tmp_path)
    code = mr.main(
        [
            "--claude-code-root",
            str(tmp_path / "claude"),
            "--codex-root",
            str(tmp_path / "codex"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "claude-code/personal" in out
    assert "codex/personal" in out
