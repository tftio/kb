"""Cross-encoder reranking over the candidates retrieval already surfaced.

kb's retrieval is a bi-encoder: passages are embedded once, ahead of time, and
compared to the query's vector by cosine. Query and document never meet, which
is what makes it cheap enough to run over the whole corpus and also what makes
it blunt — a passage's vector has to stand for everything it might ever answer.

A cross-encoder takes the query and one document *together* and scores the
pair, so the question's words can attend to the passage's. That is far more
accurate and cannot be precomputed, which is why it runs over a shortlist
rather than a corpus.

The consequence worth stating plainly, because it bounds what this can
measure: **a reranker can only reorder what retrieval surfaced.** If the
answer is not in the window, no score changes that, and a loss in such a case
is not the reranker's failure. [`window_contains_answer`] is what lets the two
be told apart in the run record.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

#: How much of a document the reranker is shown. Qwen3-Reranker accepts far
#: more, but a whole 260KB transcript would dominate the latency this task
#: exists to measure, and the opening of a passage is what a retrieval
#: decision usually turns on.
#: How much of a document the reranker is shown. Bounded by the serving
#: runtime rather than by the model: llama.cpp requires a non-causal model's
#: whole sequence to fit one physical batch, so a document longer than
#: `--ubatch-size` is refused outright. kb's org text runs about three
#: characters per token, so 2000 characters is roughly 660 tokens and needs
#: `-ub 2048`; at the 512 default even 1000 characters fails. Recorded because
#: the number is a property of the server's flags, not of the corpus.
DOCUMENT_CHARS = 2000


def readable_text(document: str) -> str:
    """Drop kb's leading property drawer from a document.

    The drawer is an id and two timestamps, identical in shape across every
    record, so it distinguishes no candidate from another — while consuming
    roughly a fifth of the characters the reranker is allowed to see. An
    unterminated drawer is left in place: blanking a document because its
    opener is malformed would score it as irrelevant.
    """
    if not document.startswith(":PROPERTIES:"):
        return document
    end = document.find(":END:\n")
    if end == -1:
        return document
    return document[end + len(":END:\n") :]


class RerankError(RuntimeError):
    """A reranking pass could not be completed or was inconsistent."""


def reorder(candidate_ids: list[str], scores: dict[str, float], top_k: int) -> list[str]:
    """Reorder the first `top_k` candidates by score, leaving the rest alone.

    Candidates below the window keep both their order and their position
    beneath the window: the reranker never saw them, so promoting or demoting
    them would be a claim the measurement cannot support. Ties preserve
    retrieval order, so a cross-encoder that cannot separate two candidates
    leaves the fused ranking's judgement standing.

    Raises:
        RerankError: If a candidate inside the window has no score, which
            would otherwise drop it silently.
    """
    window = candidate_ids[:top_k]
    tail = candidate_ids[top_k:]
    missing = [c for c in window if c not in scores]
    if missing:
        raise RerankError(f"no score returned for {len(missing)} candidate(s): {missing[:3]}")
    ranked = sorted(enumerate(window), key=lambda pair: (-scores[pair[1]], pair[0]))
    return [candidate for _, candidate in ranked] + tail


def window_contains_answer(rank: int | None, top_k: int) -> bool:
    """Whether the answer was inside the window the reranker could see.

    A question whose answer sits below the window, or is absent entirely, is
    one reranking could not have helped. Counting those as failures would
    blame the reranker for the retriever's miss.
    """
    return rank is not None and rank <= top_k


@dataclass(frozen=True, slots=True)
class RerankClient:
    """A client for a local reranking server's `/v1/rerank`.

    Local by construction, like every other model this repository talks to:
    the corpus is the operator's own notes and, later, other people's
    correspondence.
    """

    base_url: str
    timeout: float = 300.0

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Score each document against `query`, in the order given.

        Raises:
            RerankError: If the endpoint fails, or answers with a different
                number of scores than documents sent.
        """
        if not documents:
            return []
        payload = json.dumps({"query": query, "documents": documents}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/v1/rerank",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RerankError(f"reranker at {self.base_url} did not answer: {exc}") from exc
        results = body.get("results")
        if not isinstance(results, list) or len(results) != len(documents):
            raise RerankError(
                f"reranker returned {len(results) if isinstance(results, list) else 'no'} "
                f"scores for {len(documents)} documents"
            )
        scores = [0.0] * len(documents)
        for entry in results:
            index = entry.get("index")
            value = entry.get("relevance_score")
            if not isinstance(index, int) or not isinstance(value, (int, float)):
                raise RerankError(f"reranker returned an unusable result: {entry!r}")
            if not 0 <= index < len(documents):
                raise RerankError(f"reranker returned an out-of-range index: {index}")
            scores[index] = float(value)
        return scores
