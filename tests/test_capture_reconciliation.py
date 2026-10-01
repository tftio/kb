"""Capture's recovery path end to end (T022).

The other capture tests hold one piece still: the hook tests stub the importer,
and the importer tests stub the server. What neither shows is the claim the
design actually rests on — that a session lost to an unreachable server is not
lost at all, because the transcript is still on the client's disk and
reconciliation posts what the server does not hold.

These tests therefore run the real hook, the real importer, and a server on a
real socket, which is why they live apart from `test_import_transcripts.py`:
that module's tests are deliberately free of a network, and these are not.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import StubIngestServer
from kb_import import cli as kbi

REPOSITORY = Path(__file__).resolve().parent.parent
HOOK = REPOSITORY / "scripts" / "session-end-kb.sh"
SESSION = "aaaa1111-2222-3333-4444-555566667777"


def _kb_import_bin_dir(tmp_path: Path) -> Path:
    """A directory holding a `kb-import` that runs this venv's console script.

    The hook resolves `kb-import` with `command -v`, so these tests put a
    real, runnable `kb-import` on `PATH` rather than pointing the hook at a
    script path directly. `sys.executable`'s directory is the venv's `bin/`,
    which already holds the `kb-import` entry point `uv run` installs from
    `[project.scripts]`; symlinking it under a directory prepended to `PATH`
    gives the hook the same resolution a real installation would.

    Returns:
        The directory to prepend to `PATH`.
    """
    venv_kb_import = Path(sys.executable).parent / "kb-import"
    assert venv_kb_import.is_file(), (
        f"expected a kb-import console script at {venv_kb_import}; run `uv sync --group dev` first"
    )
    bin_dir = tmp_path / "kb-import-bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "kb-import").symlink_to(venv_kb_import)
    return bin_dir


def _transcript(directory: Path, session: str = SESSION, turns: int = 16) -> Path:
    """Write a Claude Code transcript long enough to clear the hook's floor.

    Args:
        directory: Where the file is written; created if it does not exist.
        session: The session id, which the filename carries and the node id
            derives from.
        turns: How many human/assistant pairs to write.

    Returns:
        The transcript's path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for turn in range(turns):
        records.append(
            {
                "type": "user",
                "uuid": f"u{turn}",
                "timestamp": "2026-08-21T09:00:00Z",
                "message": {"id": f"mu{turn}", "role": "user", "content": f"question {turn}"},
            }
        )
        records.append(
            {
                "type": "assistant",
                "uuid": f"a{turn}",
                "timestamp": "2026-08-21T09:00:01Z",
                "message": {
                    "id": f"ma{turn}",
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"answer {turn}"}],
                },
            }
        )
    path = directory / f"{session}.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _titles(home: Path, *node_ids: str) -> Path:
    """Pre-seed the title cache so no test derives a title with `claude -p`.

    Args:
        home: The sandboxed home directory the cache lives under.
        node_ids: The nodes to give a title.

    Returns:
        The cache file's path, which is where the importer looks under `home`.
    """
    path = home / ".local" / "state" / "kb-import" / "titles.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict.fromkeys(node_ids, "A Captured Session")))
    return path


def _await(condition, seconds: float = 20.0) -> None:
    """Wait for a detached process to reach an observable state.

    The hook detaches deliberately — it must not block a closing terminal — so
    an assertion about what it did has to wait rather than assume.

    Args:
        condition: Called until it returns true.
        seconds: How long to keep trying.

    Raises:
        AssertionError: If the condition never holds.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError("the detached capture never reached the expected state")


@pytest.mark.skipif(not HOOK.exists(), reason="the hook script is not in this checkout")
def test_a_captured_sessions_provenance_reaches_the_posted_submission(
    tmp_path, stub_ingest_server: StubIngestServer
):
    """T007's acceptance check.

    Posts a submission carrying provenance and confirms it is what the
    server -- and, through the queue worker's `put_record`
    (`src/ingest.rs`, `src/write.rs`), the stored record's header -- actually
    receives. The real hook and the real importer both run; only storage
    itself is stubbed, exactly as the rest of this module does.
    """
    home = tmp_path / "home"
    session = "bbbb2222-3333-4444-5555-666677778888"
    transcript = _transcript(
        home / ".config" / "claude" / "personal" / "projects" / "kb", session=session
    )
    _titles(home, f"cc-{session}")
    kb_import_bin = _kb_import_bin_dir(tmp_path)

    hook = subprocess.run(
        ["bash", str(HOOK)],
        input=json.dumps(
            {
                "session_id": session,
                "transcript_path": str(transcript),
                "cwd": str(tmp_path),
                "hook_event_name": "SessionEnd",
                "reason": "clear",
            }
        ),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={
            **os.environ,
            "HOME": str(home),
            "PATH": f"{kb_import_bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "KB_INGEST_URL": stub_ingest_server.url,
            "KB_INGEST_TOKEN": "a-token",
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv")),
            # Explicit for every marker the hook reads, so this test's outcome
            # does not depend on whatever launched pytest itself.
            "CLANKER_SESSION": "1",
            "CLANKER_SESSION_PROJECT": "kb",
            "CLANKER_SESSION_PROJECT_SOURCE": "declared",
            "CLANKER_SESSION_CONTEXT": "personal",
            "CLANKER_SESSION_REMOTE": "",
            "CLANKER_SESSION_HARNESS": "",
            "CLANKER_SESSION_MODEL": "",
            "CLANKER_SESSION_ID": "",
            "CLANKER_SESSION_DOMAINS": "",
        },
    )
    assert hook.returncode == 0, hook.stderr

    log = home / ".local" / "state" / "kb-distill" / f"{session}.log"
    _await(lambda: log.exists() and "captured kb node" in log.read_text())

    assert [submission["id"] for submission in stub_ingest_server.received] == [f"cc-{session}"]
    assert stub_ingest_server.received[0]["provenance"] == {
        "project": "kb",
        "project_source": "declared",
        "context": "personal",
    }


@pytest.mark.skipif(not HOOK.exists(), reason="the hook script is not in this checkout")
def test_a_session_the_server_never_received_is_delivered_by_reconciliation(
    monkeypatch, tmp_path, stub_ingest_server: StubIngestServer
):
    home = tmp_path / "home"
    transcript = _transcript(home / ".config" / "claude" / "personal" / "projects" / "kb")
    _titles(home, f"cc-{SESSION}")
    kb_import_bin = _kb_import_bin_dir(tmp_path)

    # Port 1 is not listening, so the post fails the way a server that is down
    # fails: refused at connect, inside the hook's own budget.
    hook = subprocess.run(
        ["bash", str(HOOK)],
        input=json.dumps(
            {
                "session_id": SESSION,
                "transcript_path": str(transcript),
                "cwd": str(tmp_path),
                "hook_event_name": "SessionEnd",
                "reason": "clear",
            }
        ),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={
            **os.environ,
            "HOME": str(home),
            "PATH": f"{kb_import_bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "KB_INGEST_URL": "http://127.0.0.1:1",
            "KB_INGEST_TOKEN": "a-token",
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv")),
        },
    )
    assert hook.returncode == 0, hook.stderr

    log = home / ".local" / "state" / "kb-distill" / f"{SESSION}.log"
    _await(lambda: log.exists() and "capture failed" in log.read_text())
    # Not just any failure: the post has to have been refused at the socket,
    # or this test would pass on a broken importer that never posted at all.
    assert "Connection refused" in log.read_text(), log.read_text()
    assert stub_ingest_server.received == [], "an unreachable server received something"

    # The gap is now real and the transcript is still on disk, which is the
    # whole of what reconciliation needs.
    monkeypatch.setattr(kbi, "DEFAULT_CACHE_PATH", _titles(home, f"cc-{SESSION}"))
    monkeypatch.setenv("KB_INGEST_TOKEN", "a-token")
    status = kbi.main(
        [
            "--source",
            "claude-code",
            "--transcript",
            str(transcript),
            "--account",
            "personal",
            "--post",
            stub_ingest_server.url,
            "--reconcile",
        ]
    )

    assert status == 0
    assert [submission["id"] for submission in stub_ingest_server.received] == [f"cc-{SESSION}"]
    assert "answer 0" in str(stub_ingest_server.received[0]["document"])


def test_a_reconciliation_run_reports_what_it_posted_and_what_was_already_held(
    monkeypatch, tmp_path, capsys, stub_ingest_server: StubIngestServer
):
    # CLI-002: the operator reads the size of the gap off this line, so the
    # counts are part of the surface rather than a by-product of it.
    home = tmp_path / "home"
    sessions = home / ".config" / "claude" / "personal" / "projects" / "kb"
    _transcript(sessions, session="held-session")
    _transcript(sessions, session="missing-session")
    monkeypatch.setattr(
        kbi, "DEFAULT_CACHE_PATH", _titles(home, "cc-held-session", "cc-missing-session")
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("KB_INGEST_TOKEN", "a-token")
    stub_ingest_server.held.append("cc-held-session")

    status = kbi.main(["--source", "claude-code", "--post", stub_ingest_server.url, "--reconcile"])

    assert status == 0
    assert capsys.readouterr().out.strip() == (
        "claude-code: created 1, updated 0, skipped 1, empty 0, failed 0"
    )
    assert [submission["id"] for submission in stub_ingest_server.received] == [
        "cc-missing-session"
    ]
