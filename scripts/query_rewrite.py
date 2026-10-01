"""Rewrite a question into the register the documents are written in.

The measured defect this attacks is a register mismatch rather than a ranking
one. An authored biography node does not appear in the top 100 nodes by
cosine for "what is my background and what do I work on", where the whole
corpus tops out at 0.419; the same node ranks **first at 0.563** for
"<author> biography core facts". The document is not hard to retrieve. Natural
first-person questions simply do not embed near third-person descriptive
prose under this model.

Three strategies are offered, cheapest first:

* `template` strips the interrogative frame and moves first person onto the
  operator's name, deterministically and without a model.
* `paraphrase` asks a local model for the same thing, which can reach
  rewrites a fixed rule cannot.
* `hyde` asks a local model for a hypothetical *answer* and embeds that
  instead of the question, which is the strongest form of the idea: it makes
  the query document-shaped rather than merely less interrogative.

Nothing here edits any kb default. Rewriting happens in the harness, on the
read path, so a strategy is reverted by dropping a flag (T001's invariant).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

#: Bumped when a prompt changes, so a run record names the prompt that
#: produced it and two runs are only comparable when it matches.
PROMPT_VERSION = 1

#: How the corpus's author is named in its third-person prose. The authored
#: notes refer to the author this way, which is precisely why a first-person
#: question fails to reach them. Set it to the name your notes use.
OPERATOR = "op"

#: Interrogative openings the template removes. Ordered longest-first so that
#: "what did" is consumed before "what".
_OPENINGS = (
    r"what (?:did|do|does|is|are|was|were|has|have|had)",
    r"how (?:did|do|does|is|are|can|could|should|would)",
    r"why (?:did|do|does|is|are|was|were)",
    r"when (?:did|do|does|is|are|was|were)",
    r"where (?:did|do|does|is|are|was|were)",
    r"which (?:of|is|are|was|were)?",
    r"who (?:is|are|was|were)?",
    r"what|how|why|when|where|which|who",
)

_OPENING_RE = re.compile(rf"^\s*(?:{'|'.join(_OPENINGS)})\b\s*", re.IGNORECASE)

#: Conjunctions that join clauses of a compound question, kept in the split so
#: the rewritten phrase still reads as one thing.
_CONJUNCTION_RE = re.compile(r"(\s+(?:and|or)\s+)", re.IGNORECASE)

#: First-person forms and what they become. Possessives first, so "my" is not
#: consumed by the bare-pronoun rule.
_PERSON = (
    (re.compile(r"\bmy\b", re.IGNORECASE), f"{OPERATOR}'s"),
    (re.compile(r"\bour\b", re.IGNORECASE), f"{OPERATOR}'s"),
    (re.compile(r"\bmine\b", re.IGNORECASE), f"{OPERATOR}'s"),
    (re.compile(r"\bI\b"), OPERATOR),
    (re.compile(r"\bwe\b", re.IGNORECASE), OPERATOR),
    (re.compile(r"\b(?:me|us)\b", re.IGNORECASE), OPERATOR),
)

#: Labels instruction-tuned models prepend to their answers.
_LABEL_RE = re.compile(
    r"^\s*(?:rewritten(?: query)?|query|answer|result|output)\s*[:\-]\s*", re.IGNORECASE
)

#: Reasoning models emit a thinking block before their answer.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class RewriteError(RuntimeError):
    """A rewrite could not be produced.

    Raised rather than falling back to the original question: a silent
    fallback would record the control's numbers under a strategy's name, which
    is worse than no measurement.
    """


class Client(Protocol):
    """The one thing a rewriting model has to do."""

    def complete(self, prompt: str) -> str:
        """Return the model's completion for `prompt`."""
        ...


@dataclass(frozen=True, slots=True)
class Strategy:
    """One way of rewriting a question."""

    name: str
    needs_model: bool
    prompt: str


STRATEGIES: dict[str, Strategy] = {
    "none": Strategy(name="none", needs_model=False, prompt=""),
    "template": Strategy(name="template", needs_model=False, prompt=""),
    "paraphrase": Strategy(
        name="paraphrase",
        needs_model=True,
        prompt=(
            "Rewrite this question as a short descriptive noun phrase in the third "
            "person, as it would appear in a reference document's own prose. Refer to "
            f"the asker as {OPERATOR}. Do not answer the question. Do not explain. "
            "Reply with the phrase alone.\n\nQuestion: {question}"
        ),
    ),
    "hyde": Strategy(
        name="hyde",
        needs_model=True,
        prompt=(
            "Write two or three sentences of the passage that would answer this "
            "question, as it would appear in a personal knowledge base written in the "
            f"third person about {OPERATOR}. Invent plausible specifics. Do not "
            "hedge, do not say you lack information, and do not explain. Reply with "
            "the passage alone.\n\nQuestion: {question}"
        ),
    ),
}


def rewrite_template(question: str) -> str:
    """Rewrite `question` into descriptive register by fixed rule.

    Deterministic and model-free, so it costs nothing and cannot drift between
    runs. A question already in descriptive register is returned unchanged,
    since rewriting a statement would be damage rather than improvement.
    """
    # Collapse whitespace first: the opening patterns match single spaces, and
    # a question typed with a double space would otherwise keep its frame.
    text = " ".join(question.strip().rstrip("?").split())
    # A compound question carries an interrogative per clause -- "what is my
    # background and what do I work on" -- so each clause is stripped, not just
    # the first. Splitting on a conjunction that joins ordinary nouns is
    # harmless, since stripping an opening that is not there does nothing.
    clauses = _CONJUNCTION_RE.split(text)
    stripped = [
        part if _CONJUNCTION_RE.fullmatch(part) else _OPENING_RE.sub("", part) for part in clauses
    ]
    text = "".join(stripped)
    for pattern, replacement in _PERSON:
        text = pattern.sub(replacement, text)
    return " ".join(text.split())


def clean_completion(reply: str) -> str:
    """Narrow a model's reply to the text that should be embedded.

    Reasoning blocks, answer labels and surrounding quotation marks are all
    things a local instruction-tuned model emits routinely, and every one of
    them would otherwise be embedded as though it were part of the query.
    """
    text = _THINK_RE.sub("", reply).strip()
    text = _LABEL_RE.sub("", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    # Double quotes are model formatting rather than content, and an odd
    # number of them breaks `kb search`'s escaping in its default keyword
    # mode: the remainder is parsed as an FTS5 expression, so a stray quote
    # turns an ordinary query into "no such column". Recorded as a kb defect
    # in the plan; stripped here because a generated passage's quotation marks
    # are not part of what is being measured either way.
    text = text.replace('"', " ")
    return " ".join(text.split())


def is_degenerate(text: str, *, window: int = 12, threshold: float = 0.25) -> bool:
    """Whether `text` is a model stuck in a repetition loop.

    A local model asked to invent specifics sometimes emits its own
    deliberation instead, looping a phrase dozens of times. Searching for that
    would record the model's failure as the strategy's result, so it is
    detected rather than embedded: a passage in which one short phrase
    accounts for more than `threshold` of all phrases of that length is not
    prose.
    """
    words = text.split()
    if len(words) < window * 3:
        return False
    phrases = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    if not phrases:
        return False
    most_common = max(phrases.count(p) for p in set(phrases))
    return most_common / len(phrases) > threshold


def rewrite(question: str, strategy: str, client: Client | None) -> str:
    """Rewrite `question` under the named strategy.

    Raises:
        RewriteError: If the strategy is unknown, if it needs a model and none
            was supplied, or if the model produced nothing usable.
    """
    chosen = STRATEGIES.get(strategy)
    if chosen is None:
        raise RewriteError(
            f"unknown rewrite strategy {strategy!r}; expected one of {sorted(STRATEGIES)}"
        )
    if chosen.name == "none":
        return question
    if chosen.name == "template":
        return rewrite_template(question)
    if client is None:
        raise RewriteError(
            f"the {chosen.name!r} strategy needs a local model and none was configured"
        )
    rewritten = clean_completion(client.complete(chosen.prompt.format(question=question)))
    if is_degenerate(rewritten):
        raise RewriteError(
            f"the {chosen.name!r} rewrite of {question!r} degenerated into a repetition "
            "loop; refusing to record a model failure as a strategy's result"
        )
    if not rewritten:
        raise RewriteError(
            f"the {chosen.name!r} rewrite of {question!r} came back empty; "
            "refusing to search for nothing and record it as a measurement"
        )
    return rewritten


@dataclass(frozen=True, slots=True)
class LocalChatClient:
    """A chat-completions client against a local endpoint.

    Local by construction: the corpus includes the operator's own notes and,
    later, other people's correspondence, and none of it is sent to a hosted
    API.
    """

    base_url: str
    model: str
    timeout: float = 300.0
    #: Generous by default because the locally available models are reasoning
    #: models: a budget that a shorter answer would never approach is spent on
    #: a thinking block, and a run that stops at the cap returns empty content
    #: rather than a short answer.
    max_tokens: int = 2000

    def complete(self, prompt: str) -> str:
        """Ask the model for one completion.

        Raises:
            RewriteError: If the endpoint fails or answers in a shape that is
                not a completion.
        """
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RewriteError(f"rewriting model {self.model!r} did not answer: {exc}") from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RewriteError(
                f"rewriting model {self.model!r} answered in an unexpected shape: {body!r}"
            ) from exc
        if not isinstance(content, str):
            raise RewriteError(f"rewriting model {self.model!r} returned non-text content")
        return content
