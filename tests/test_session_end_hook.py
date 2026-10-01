"""The SessionEnd hook's one hard obligation (T022).

A hook runs as a terminal is closing, with a thirty-second budget. Whatever
happens to capture, the hook must not fail the session: a lost session is a gap
reconciliation can fill from the transcript still on disk, while a hook that
exits non-zero or hangs is a hook that interferes with the operator's work.

These tests run the real script with a stub importer, because the property
belongs to the shell rather than to anything the importer does.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent.parent / "scripts" / "session-end-kb.sh"

#: Every `CLANKER_SESSION*` marker T003 (`PLAN-20260923-project-identity`)
#: exports at launch. Cleared by default in `_run_hook` so a test's outcome
#: does not depend on whatever session happens to be running these tests --
#: this machine's own shell already carries some of these.
_CLANKER_SESSION_MARKERS = (
    "CLANKER_SESSION",
    "CLANKER_SESSION_VERSION",
    "CLANKER_SESSION_ID",
    "CLANKER_SESSION_INVOCATION",
    "CLANKER_SESSION_HARNESS",
    "CLANKER_SESSION_CONTEXT",
    "CLANKER_SESSION_CONTEXT_SOURCE",
    "CLANKER_SESSION_DOMAINS",
    "CLANKER_SESSION_DOMAIN_SOURCE",
    "CLANKER_SESSION_MODEL",
    "CLANKER_SESSION_MODEL_SOURCE",
    "CLANKER_SESSION_FAMILY",
    "CLANKER_SESSION_PROJECT",
    "CLANKER_SESSION_PROJECT_SOURCE",
    "CLANKER_SESSION_REMOTE",
)


def _path_without_clanker() -> str:
    """This process's `PATH`, minus every directory that holds a `clanker` executable.

    Built by filtering the real `PATH` rather than hardcoding a curated list:
    `clanker`'s install location is machine-specific (this development
    machine keeps it under both `~/.cargo/bin` and a mise shims directory), so
    a literal list would be portable to neither a different developer machine
    nor a CI runner. `bash`, `jq`, `uv`, `nohup` and `cut` all live in system
    or Homebrew directories `clanker` never does, so removing only the
    directories that hold `clanker` leaves the hook everything else it needs.

    Returns:
        A `PATH`-shaped string with no `clanker` reachable on it.
    """
    kept = [
        directory
        for directory in os.environ.get("PATH", "").split(os.pathsep)
        if directory and not os.access(Path(directory) / "clanker", os.X_OK)
    ]
    return os.pathsep.join(kept)


def _payload(transcript: Path, session: str = "abc123") -> str:
    """The hook payload Claude Code writes to the hook's stdin."""
    return json.dumps(
        {
            "session_id": session,
            "transcript_path": str(transcript),
            "cwd": "/tmp",
            "hook_event_name": "SessionEnd",
            "reason": "clear",
        }
    )


def _transcript(tmp_path: Path, lines: int = 40) -> Path:
    """A transcript long enough to clear the hook's floor."""
    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join('{"type":"user"}' for _ in range(lines)) + "\n")
    return path


def _stub_importer(tmp_path: Path, *, exit_code: int) -> Path:
    """An executable named `kb-import` in its own directory.

    It records how it was called and exits `exit_code`. Returns the directory
    holding the stub (to prepend to `PATH`), not the stub itself: the hook
    resolves `kb-import` with `command -v`, so what matters to a caller is
    where to put it on `PATH`.
    """
    bin_dir = tmp_path / "kb-import-bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / "kb-import"
    script.write_text(f'#!/bin/sh\necho "$@" > {str(tmp_path / "argv.txt")!r}\nexit {exit_code}\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


def _run_hook(
    tmp_path: Path, importer_bin_dir: Path, env: dict[str, str]
) -> subprocess.CompletedProcess:
    """Run the hook with a sandboxed HOME and the given environment.

    `importer_bin_dir` (holding the `kb-import` stub) is always prepended to
    `PATH`, including when `env` overrides `PATH` itself, so a test that
    scrubs or replaces `PATH` for another reason (hiding `clanker`, adding a
    fake one) does not accidentally hide the stub importer too.
    """
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    environment = {
        **os.environ,
        "HOME": str(home),
        # HOME is sandboxed so the hook writes its logs into the fixture, but
        # uv keeps its cache under HOME too, and rebuilding that per test costs
        # more than the wait below allows. The cache is read-only here.
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv")),
    }
    environment.pop("KB_SESSION_DISTILL", None)
    environment.pop("KB_INGEST_URL", None)
    # This machine's own shell may itself be a clanker session; without this,
    # a test's outcome would depend on whatever launched pytest rather than
    # on what it deliberately set up.
    for marker in _CLANKER_SESSION_MARKERS:
        environment.pop(marker, None)
    environment.update(env)
    base_path = environment.get("PATH", os.environ.get("PATH", ""))
    environment["PATH"] = f"{importer_bin_dir}{os.pathsep}{base_path}"
    return subprocess.run(
        ["bash", str(HOOK)],
        input=_payload(_transcript(tmp_path)),
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
        check=False,
    )


@pytest.mark.skipif(not HOOK.exists(), reason="the hook script is not in this checkout")
def test_a_failing_capture_does_not_fail_the_session(tmp_path):
    importer_bin = _stub_importer(tmp_path, exit_code=1)
    result = _run_hook(tmp_path, importer_bin, {"KB_INGEST_URL": "http://127.0.0.1:1"})
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not HOOK.exists(), reason="the hook script is not in this checkout")
def test_a_configured_server_makes_the_importer_post(tmp_path):
    importer_bin = _stub_importer(tmp_path, exit_code=0)
    result = _run_hook(tmp_path, importer_bin, {"KB_INGEST_URL": "http://kb.example:8080"})
    assert result.returncode == 0, result.stderr

    argv = _await_argv(tmp_path)
    assert "--post http://kb.example:8080" in argv, argv


@pytest.mark.skipif(not HOOK.exists(), reason="the hook script is not in this checkout")
def test_without_a_server_the_importer_writes_through_kb(tmp_path):
    # Capture must not stop working because a deployment has not happened, so
    # an unset KB_INGEST_URL is the pre-T022 path rather than an error.
    importer_bin = _stub_importer(tmp_path, exit_code=0)
    result = _run_hook(tmp_path, importer_bin, {})
    assert result.returncode == 0, result.stderr

    argv = _await_argv(tmp_path)
    assert "--post" not in argv, argv
    assert "--source claude-code" in argv, argv


def _stub_clanker(tmp_path: Path, output: str) -> tuple[Path, Path]:
    """A `clanker` stand-in for `project resolve --dir <path>`.

    Args:
        tmp_path: Test scratch directory; the stub and its record file both
            live under it.
        output: What `clanker project resolve` should print, tab-separated.

    Returns:
        `(bin_dir, record)`: the directory to prepend to `PATH`, and the file
        the stub writes its own argv to (space-joined, one line).
    """
    bin_dir = tmp_path / "clanker-bin"
    bin_dir.mkdir(exist_ok=True)
    record = tmp_path / "clanker_argv.txt"
    script = bin_dir / "clanker"
    script.write_text(
        f'#!/bin/sh\necho "$@" > "{record}"\ncat <<\'CLANKER_STUB_EOF\'\n{output}CLANKER_STUB_EOF\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, record


@pytest.mark.skipif(not HOOK.exists(), reason="the hook script is not in this checkout")
def test_inherited_clanker_session_markers_reach_the_importer_argv(tmp_path):
    """T003's `CLANKER_SESSION_*` markers are passed straight through."""
    importer_bin = _stub_importer(tmp_path, exit_code=0)
    result = _run_hook(
        tmp_path,
        importer_bin,
        {
            "CLANKER_SESSION": "1",
            "CLANKER_SESSION_PROJECT": "kb",
            "CLANKER_SESSION_PROJECT_SOURCE": "remote",
            "CLANKER_SESSION_REMOTE": "github.com/tftio/kb",
            "CLANKER_SESSION_CONTEXT": "personal",
            "CLANKER_SESSION_HARNESS": "claude",
            "CLANKER_SESSION_MODEL": "sonnet",
            "CLANKER_SESSION_ID": "sess-xyz",
            "CLANKER_SESSION_DOMAINS": "eng,infra",
        },
    )
    assert result.returncode == 0, result.stderr

    argv = _await_argv(tmp_path)
    assert "--project kb" in argv, argv
    assert "--project-source remote" in argv, argv
    assert "--remote github.com/tftio/kb" in argv, argv
    assert "--context personal" in argv, argv
    assert "--harness claude" in argv, argv
    assert "--model sonnet" in argv, argv
    assert "--clanker-session sess-xyz" in argv, argv
    assert "--domain eng" in argv, argv
    assert "--domain infra" in argv, argv


@pytest.mark.skipif(not HOOK.exists(), reason="the hook script is not in this checkout")
def test_without_clanker_session_the_hook_asks_clanker_to_resolve(tmp_path):
    """`CLANKER_SESSION` unset: the hook calls `clanker project resolve --dir`."""
    importer_bin = _stub_importer(tmp_path, exit_code=0)
    clanker_bin, record = _stub_clanker(tmp_path, "myproj\tpath\tgithub.com/x/y\n")
    result = _run_hook(
        tmp_path,
        importer_bin,
        {"PATH": f"{clanker_bin}{os.pathsep}{os.environ.get('PATH', '')}"},
    )
    assert result.returncode == 0, result.stderr

    argv = _await_argv(tmp_path)
    assert record.exists(), "the stub clanker was never invoked"
    assert record.read_text().split() == ["project", "resolve", "--dir", "/tmp"]
    assert "--project myproj" in argv, argv
    assert "--project-source path" in argv, argv
    assert "--remote github.com/x/y" in argv, argv


@pytest.mark.skipif(not HOOK.exists(), reason="the hook script is not in this checkout")
def test_with_no_clanker_binary_capture_proceeds_with_null_provenance(tmp_path):
    importer_bin = _stub_importer(tmp_path, exit_code=0)
    result = _run_hook(tmp_path, importer_bin, {"PATH": _path_without_clanker()})
    assert result.returncode == 0, result.stderr

    argv = _await_argv(tmp_path)
    assert "--source claude-code" in argv, argv
    assert "--project" not in argv, argv
    assert "--remote" not in argv, argv


@pytest.mark.skipif(not HOOK.exists(), reason="the hook script is not in this checkout")
def test_a_tiny_transcript_is_not_captured_at_all(tmp_path):
    importer_bin = _stub_importer(tmp_path, exit_code=0)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    short = tmp_path / "short.jsonl"
    short.write_text('{"type":"user"}\n')
    result = subprocess.run(
        ["bash", str(HOOK)],
        input=_payload(short),
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HOME": str(home),
            "PATH": f"{importer_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        },
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    time.sleep(0.5)
    assert not (tmp_path / "argv.txt").exists(), "a three-line session was captured"


def _run_hook_without_kb_import(tmp_path: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run the hook with no `kb-import` reachable on `PATH`.

    Unlike `_run_hook`, this never prepends the stub importer's directory.
    `HOME` is sandboxed to a tmp dir, so the hook's own
    `$HOME/.cargo/bin:$HOME/.local/bin` prefix resolves to empty directories;
    the real `PATH` is filtered down to entries that plausibly hold `bash`,
    `jq`, `nohup` and `cut` while excluding `/opt/homebrew/bin` and any
    directory that actually contains a `kb-import` executable, so a real
    installation on this development machine cannot leak into the test.
    """
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    kept = [
        directory
        for directory in os.environ.get("PATH", "").split(os.pathsep)
        if directory
        and directory != "/opt/homebrew/bin"
        and not os.access(Path(directory) / "kb-import", os.X_OK)
    ]
    environment = {
        **os.environ,
        "HOME": str(home),
        "PATH": os.pathsep.join(kept),
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv")),
    }
    environment.pop("KB_SESSION_DISTILL", None)
    environment.pop("KB_INGEST_URL", None)
    for marker in _CLANKER_SESSION_MARKERS:
        environment.pop(marker, None)
    environment.update(env)
    return subprocess.run(
        ["bash", str(HOOK)],
        input=_payload(_transcript(tmp_path)),
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
        check=False,
    )


@pytest.mark.skipif(not HOOK.exists(), reason="the hook script is not in this checkout")
def test_no_kb_import_on_path_exits_zero_and_logs_to_stderr_and_log(tmp_path):
    """No `kb-import` reachable on `PATH`: exit 0, message on stderr and in the log."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)

    result = _run_hook_without_kb_import(tmp_path, {"HOME": str(home)})
    assert result.returncode == 0, result.stderr
    assert "kb-import" in result.stderr, result.stderr
    assert "abc123" in result.stderr, result.stderr
    assert "not captured" in result.stderr, result.stderr

    log = home / ".local" / "state" / "kb-distill" / "abc123.log"
    assert log.exists(), "the hook did not log the missing importer"
    log_text = log.read_text()
    assert "kb-import" in log_text
    assert "abc123" in log_text
    assert "not captured" in log_text
    assert "uv tool install" in log_text

    time.sleep(0.5)
    assert not (tmp_path / "argv.txt").exists(), "no importer should have been invoked"


@pytest.mark.skipif(not HOOK.exists(), reason="the hook script is not in this checkout")
def test_the_hook_source_names_no_hard_coded_importer_path():
    """The stale `~/Projects/Repositories/kb/main/...` fallback must not reappear."""
    source = HOOK.read_text()
    assert "Projects/Repositories/kb" not in source, source


def _await_argv(tmp_path: Path, seconds: float = 10.0) -> str:
    """Wait for the detached child to record its arguments.

    The hook detaches deliberately — it must not block a closing terminal — so
    the assertion has to wait for the background process rather than assume it
    has already run.
    """
    argv = tmp_path / "argv.txt"
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if argv.exists():
            return argv.read_text()
        time.sleep(0.05)
    raise AssertionError("the detached importer never ran")
