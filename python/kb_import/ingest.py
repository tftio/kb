"""The ingest endpoint client (T022).

Posts to a capture server's ingest endpoint as an alternative to writing
through the `kb` binary directly.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from typing import Protocol

#: A transport: url, headers and body in, response body out. Injected so the
#: client's shape can be exercised without a server, and so the one place that
#: touches the network is one line.
type Transport = Callable[[str, dict[str, str], bytes], bytes]


class IngestTarget(Protocol):
    """What the posting loop needs of a capture server.

    A protocol rather than the concrete client because the loop uses exactly
    two operations, and saying so is what lets it be exercised against a
    recording double instead of a socket.
    """

    def served_ids(self, corpus: str = "kb") -> list[str]:
        """Every record id the server already holds in `corpus`."""
        ...

    def post(self, submission: dict[str, object]) -> None:
        """Send one submission, returning once it is acknowledged."""
        ...


class IngestError(Exception):
    """A post or a listing did not reach the server, or it was refused."""


def missing_ids(local: Iterable[str], served: Iterable[str]) -> list[str]:
    """The ids present locally and absent on the server, sorted and deduplicated.

    The whole of reconciliation, as a function of two id sets. Kept pure so
    that what a run would post can be inspected without a server, which is
    also what makes `--reconcile --dry-run` honest rather than a second
    implementation of the same comparison.

    Args:
        local: Every record id the local archive can produce.
        served: Every record id the server already holds.

    Returns:
        The difference, sorted.
    """
    return sorted(set(local) - set(served))


def submission_for(
    node_id: str, document: str, provenance: dict[str, object] | None = None
) -> dict[str, object]:
    """The body the ingest endpoint accepts.

    Args:
        node_id: The record id to store under.
        document: The rendered org text.
        provenance: The `RawProvenance`-shaped mapping to attach, if any. The
            worker applies it on every write regardless of whether the id is
            new (`crate::ingest::worker` calls `put_record` unconditionally),
            so a provenance change on an existing record lands here exactly
            as it does through `write_node`'s `kb update --provenance-json`.

    Returns:
        The submission, ready to serialize.
    """
    submission: dict[str, object] = {"id": node_id, "corpus": "kb", "document": document}
    if provenance:
        submission["provenance"] = provenance
    return submission


def http_transport(url: str, headers: dict[str, str], body: bytes) -> bytes:
    """The real transport: one bounded request.

    A timeout is not optional here. This runs from a `SessionEnd` hook with a
    thirty-second budget, and a server that accepts the connection and then
    stalls would hold the session open to the end of it.

    Args:
        url: The absolute URL.
        headers: Headers to send.
        body: The request body, empty for a GET.

    Returns:
        The response body.

    Raises:
        IngestError: If the request fails or the server refuses it.
    """
    request = urllib.request.Request(
        url, data=body or None, headers=headers, method="POST" if body else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise IngestError(f"{url} refused the request: {exc.code} {exc.reason}") from exc
    except OSError as exc:
        raise IngestError(f"{url} could not be reached: {exc}") from exc


class IngestClient:
    """The capture server, as this script uses it."""

    def __init__(self, base_url: str, token: str, transport: Transport = http_transport) -> None:
        """Bind a server and the credential for it.

        Args:
            base_url: The server's root, with or without a trailing slash.
            token: The bearer token, which the server requires.
            transport: How to make a request; injected for testing.

        Raises:
            IngestError: If no token was supplied. An unauthenticated post is
                refused at the far end, and failing here names the cause where
                it can still be acted on.
        """
        if not token:
            raise IngestError("no ingest token: set KB_INGEST_TOKEN")
        self._base = base_url.rstrip("/")
        self._token = token
        self._transport = transport

    def _headers(self, *, json_body: bool) -> dict[str, str]:
        """Headers for a request, with the credential."""
        headers = {"Authorization": f"Bearer {self._token}"}
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def served_ids(self, corpus: str = "kb") -> list[str]:
        """Every record id the server already holds in `corpus`.

        Args:
            corpus: The corpus to ask about.

        Returns:
            The ids, as the server reported them.

        Raises:
            IngestError: If the request fails or the answer does not parse.
        """
        url = f"{self._base}/ingest/ids?corpus={corpus}"
        try:
            body = self._transport(url, self._headers(json_body=False), b"")
        except OSError as exc:
            raise IngestError(f"{url} could not be reached: {exc}") from exc
        try:
            listed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise IngestError(f"{url} did not answer with JSON: {exc}") from exc
        if not isinstance(listed, list):
            raise IngestError(f"{url} did not answer with a list of ids")
        return [str(item) for item in listed]

    def post(self, submission: dict[str, object]) -> None:
        """Post one submission, returning once the server has acknowledged it.

        Args:
            submission: The body from `submission_for`.

        Raises:
            IngestError: If the request fails or the server refuses it.
        """
        url = f"{self._base}/ingest"
        payload = json.dumps(submission).encode()
        try:
            self._transport(url, self._headers(json_body=True), payload)
        except OSError as exc:
            raise IngestError(f"{url} could not be reached: {exc}") from exc
