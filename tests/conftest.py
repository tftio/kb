"""Put the dev scripts' directory on the import path for the test suite.

The transcript importer is now the installed `kb_import` package and needs no
path insertion; tests import it directly. `scripts/` still holds the other dev
scripts (`eval_retrieval.py`, `bench_serving.py`, `measure_reads.py`, and the
rest), none of which is an installed package, so their directory is still put
on the path here. This lets the tests import them by name rather than through
`importlib` gymnastics, which matters for more than tidiness: a module loaded
by path is opaque to the type checker, so every value drawn from it degrades to
`object` and pyrefly can no longer verify the tests against the code they test.

`pyrefly.toml` mirrors this with a `search-path` entry, for the same reason.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class StubIngestServer:
    """A stand-in for T022's ingest endpoint, over a real socket.

    It serves the two routes the capture client uses -- the id listing that
    reconciliation compares against, and the accept that takes a submission --
    so a test drives the importer's own HTTP transport rather than a
    substituted one. What it does with a submission is deliberately less than
    the real endpoint does: it holds the id, which is all reconciliation can
    observe about a served corpus anyway.

    Attributes:
        received: Every submission posted, in order.
        held: The ids the listing reports, seeded at construction and extended
            by each accepted post.
    """

    def __init__(self, held: list[str] | None = None) -> None:
        """Bind a server to an ephemeral loopback port without starting it.

        Args:
            held: The ids the listing reports before anything is posted.
        """
        self.received: list[dict[str, object]] = []
        self.held: list[str] = list(held or [])
        server = self

        class Handler(BaseHTTPRequestHandler):
            """Route the two ingest paths; refuse everything else."""

            def do_GET(self) -> None:
                """Answer the id listing."""
                if not self.path.startswith("/ingest/ids"):
                    self.send_error(404)
                    return
                self._respond(200, json.dumps(server.held).encode("utf-8"))

            def do_POST(self) -> None:
                """Accept a submission and record it."""
                if self.path.rstrip("/") != "/ingest":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length") or 0)
                submission = json.loads(self.rfile.read(length) or b"{}")
                server.received.append(submission)
                server.held.append(submission["id"])
                self._respond(202, json.dumps({"accepted": submission["id"]}).encode("utf-8"))

            def _respond(self, status: int, body: bytes) -> None:
                """Write one JSON response."""
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                """Keep the request log out of the test output."""

        self._http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._http.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        """The base URL to hand the client."""
        host, port = self._http.server_address[:2]
        return f"http://{host!s}:{port}"

    def start(self) -> None:
        """Begin serving."""
        self._thread.start()

    def stop(self) -> None:
        """Stop serving and release the socket."""
        self._http.shutdown()
        self._http.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def stub_ingest_server() -> Iterator[StubIngestServer]:
    """A running `StubIngestServer`, torn down after the test."""
    server = StubIngestServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()
