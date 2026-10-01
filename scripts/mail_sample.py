"""Measure the bulk discriminant's false-negative rate.

The rule's deliverable is not the rule; it is the rule *plus a known error
rate*, because everything downstream depends on knowing how much real
correspondence a large cut removes. A message wrongly dropped is ground truth
that silently never existed.

Two comparisons live here, both of which need the operator's own Maildir and
so belong in `scripts/` rather than in the crate, per the precedent
`PLAN-20260813-retrieval-evaluation-harness` set.

* The **behavioural rule**: did the operator ever reply in this thread?
  Resolvable from `In-Reply-To` and `References` on the messages he sent.
  Replying is a far stronger signal that a message mattered than any header,
  so a bulk-classified message the operator replied to is a false negative
  with evidence rather than an opinion.
* A **stratified sample** of bulk-classified messages for hand-checking,
  drawn deterministically so a later run can reproduce exactly the set that
  was judged.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

#: Angle-bracketed message identifiers, as they appear in In-Reply-To and
#: References. Mail is inconsistent about spacing and folding, so they are
#: extracted by pattern rather than by splitting.
MESSAGE_ID_RE = re.compile(r"<[^<>@\s]+@[^<>\s]+>")


def header_block(raw: bytes) -> str:
    """The header block of a message, decoded leniently.

    Mail is not reliably UTF-8 and refusing a message would drop it from the
    measurement, which is the one outcome this whole task exists to avoid.
    """
    head = raw.split(b"\n\n", 1)[0]
    return head.decode("utf-8", errors="replace")


def unfold(head: str, name: str) -> str:
    """The unfolded value of one header, or the empty string."""
    wanted = name.lower()
    lines = head.splitlines()
    for index, line in enumerate(lines):
        if ":" not in line or line.startswith((" ", "\t")):
            continue
        if line.split(":", 1)[0].strip().lower() != wanted:
            continue
        value = [line.split(":", 1)[1].strip()]
        for continuation in lines[index + 1 :]:
            if not continuation.startswith((" ", "\t")):
                break
            value.append(continuation.strip())
        return " ".join(" ".join(value).split())
    return ""


def referenced_ids(head: str) -> set[str]:
    """Every message id this message's thread headers point at."""
    joined = f"{unfold(head, 'in-reply-to')} {unfold(head, 'references')}"
    return set(MESSAGE_ID_RE.findall(joined))


def messages(maildir: Path, folders: Sequence[str]) -> Iterable[tuple[Path, str]]:
    """Yield `(path, header block)` for every message in `folders`."""
    for folder in folders:
        for sub in ("cur", "new"):
            directory = maildir / folder / sub
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                try:
                    yield path, header_block(path.read_bytes())
                except OSError:
                    continue


def threads_the_operator_joined(maildir: Path, folders: Sequence[str], address: str) -> set[str]:
    """Message ids appearing in threads the operator himself wrote into.

    Includes the ids his own messages carry, so a reply to his reply is
    recognised as the same conversation.
    """
    joined: set[str] = set()
    for _path, head in messages(maildir, folders):
        if address.lower() not in unfold(head, "from").lower():
            continue
        joined |= referenced_ids(head)
        own = unfold(head, "message-id")
        joined |= set(MESSAGE_ID_RE.findall(own))
    return joined


def stratified_sample(rows: Sequence[dict[str, str]], size: int, seed: int) -> list[dict[str, str]]:
    """Draw `size` rows spread across years, deterministically.

    Stratified because the corpus spans 2022-2026 and mail conventions change:
    a sample drawn only from recent mail would measure a rule against the
    habits of one year.
    """
    by_year: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        parts = row.get("date", "").split()
        year = next((p for p in parts if p.isdigit() and len(p) == 4), "unknown")
        by_year.setdefault(year, []).append(row)
    rng = random.Random(seed)
    per_year = max(1, size // max(1, len(by_year)))
    drawn: list[dict[str, str]] = []
    for year in sorted(by_year):
        bucket = by_year[year]
        drawn.extend(rng.sample(bucket, min(per_year, len(bucket))))
    remaining = [r for r in rows if r not in drawn]
    if len(drawn) < size and remaining:
        drawn.extend(rng.sample(remaining, min(size - len(drawn), len(remaining))))
    return drawn[:size]


def main(argv: Sequence[str] | None = None) -> int:
    """Compare the header rule against the behavioural one and draw a sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classified", type=Path, required=True, help="JSONL from `kb mail classify`."
    )
    parser.add_argument("--maildir", type=Path, required=True)
    parser.add_argument(
        "--address", required=True, help="The mailbox owner's address, whose replies mark threads."
    )
    parser.add_argument("--folder", action="append", default=None)
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--out", type=Path, required=True, help="Where to write the review sample.")
    args = parser.parse_args(argv)
    folders = args.folder or ["Inbox", "Archive"]

    rows = [json.loads(line) for line in args.classified.read_text().splitlines() if line.strip()]
    bulk = [r for r in rows if r["classification"] == "bulk"]
    non_bulk = [r for r in rows if r["classification"] != "bulk"]
    print(f"classified: {len(bulk)} bulk, {len(non_bulk)} non-bulk")

    joined = threads_the_operator_joined(args.maildir, folders, args.address)
    print(f"threads the operator wrote into: {len(joined)} message ids")

    # A message the operator sent himself is not incoming correspondence being
    # dropped, and his out-of-office auto-replies carry `Precedence:` and sit
    # in every thread he touched -- so counting them would inflate the rule's
    # apparent error with his own automation.
    incoming = [r for r in bulk if args.address.lower() not in r.get("from", "").lower()]
    self_sent = len(bulk) - len(incoming)
    disputed = [r for r in incoming if r["message_id"] and r["message_id"] in joined]
    print(f"bulk-classified messages the operator sent himself: {self_sent}")
    print(
        f"incoming, bulk-classified, in a thread the operator joined: {len(disputed)} "
        f"({len(disputed) / max(1, len(incoming)):.2%} of incoming bulk)"
    )

    sample = stratified_sample(bulk, args.sample_size, args.seed)
    args.out.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "sample_size": len(sample),
                "bulk_total": len(bulk),
                "non_bulk_total": len(non_bulk),
                "behavioural_disputed": [r["message_id"] for r in disputed],
                "sample": sample,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(sample)} sampled messages for hand-checking to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
