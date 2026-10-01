"""The kb driver: reading and writing nodes through the `kb` binary.

* **`kb update --provenance-json <file>` moves an existing record's
  provenance in place.** T005 gave `kb create` this channel but not `kb
  update`; a first version of this task worked around that by having
  `write_node` run `kb delete` immediately followed by `kb create
  --provenance-json` against the same id. The coordinator's review of that
  commit (998bf6e) rejected it: it resets `created`, cascade-deletes the
  node's tags and links on the delete, and loses the record outright if
  `create` fails after `delete` succeeds -- unacceptable for what T008's
  backfill does to the 424 existing Claude Code transcripts. `kb update`
  gained its own `--provenance-json` instead, sharing `kb create`'s parsing
  and validation code path, and `write_node` passes it through on `update`
  whenever this run actually computed a non-empty provenance mapping,
  changed or not. **`kb update` with no `--provenance-json` now preserves a
  record's existing kind, source and provenance rather than resetting them**
  (`PLAN-20260923-project-identity` T016, after this task originally shipped
  against the opposite default); `import_conversations` relies on exactly
  that: a run that resolves no provenance at all for a conversation -- no
  `--project`/flags and no captured `cwd` -- passes no `--provenance-json`
  and never compares its empty mapping against what is stored, so it leaves
  a record's existing provenance untouched instead of treating "resolved
  nothing this time" as "clear what was there."
"""

from __future__ import annotations

import json

from kb_import.model import Conversation, _as_dict, _as_str, render
from kb_import.provenance import PROVENANCE_FIELDS, _provenance_file
from kb_import.titles import CommandError, CommandRunner


def fetch_node(
    node_id: str, db: str | None, run: CommandRunner
) -> tuple[str, dict[str, object]] | None:
    """Read a node's stored document and provenance, if it exists.

    Reads through `kb get --json` rather than the text form. The text form
    appends a human-readable `provenance:` block after the document (`kb
    get`'s own doc in `src/cli_main.rs`), which would have to be parsed back
    off before the document could be compared against a freshly rendered one;
    the JSON form's `data.document` never carries that block, or kb's
    synthesized `:PROPERTIES:` envelope, at all.

    Args:
        node_id: The id to fetch.
        db: Database path, or `None` for kb's default.
        run: The subprocess runner.

    Returns:
        `(document, provenance)` when the node exists, provenance normalized
        to `RawProvenance`'s wire field names and holding only the fields
        that are actually set; `None` when no such node exists. `kb get`
        exits non-zero for a missing node, which is the existence probe.
    """
    try:
        raw = run([*_kb(db), "get", node_id, "--json"], "")
    except CommandError:
        return None
    data = _as_dict(_as_dict(json.loads(raw)).get("data"))
    document = _strip_property_drawer(_as_str(data.get("document")))
    provenance = _normalize_stored_provenance(_as_dict(data.get("provenance")))
    return document, provenance


#: `kb get --json`'s provenance object renames one field for its camelCase
#: convention (`src/cli_main.rs`'s `provenance_json`); every other key
#: already matches `RawProvenance`'s wire names.
_GET_PROVENANCE_RENAME = {"projectSource": "project_source"}


def _normalize_stored_provenance(raw: dict[str, object]) -> dict[str, object]:
    """Normalize `kb get --json`'s provenance object for comparison.

    Args:
        raw: The `provenance` object from `kb get --json`; `{}` when the
            record carries none (kb reports that case as JSON `null`, which
            `_as_dict` narrows to `{}`).

    Returns:
        A mapping using `RawProvenance`'s wire field names
        (`PROVENANCE_FIELDS`), holding only the fields that are actually set.
        `kb get` includes every key with a `null` or empty value, which must
        not register as "set" here or a provenance-free record would never
        compare equal to itself.
    """
    normalized: dict[str, object] = {}
    for key, value in raw.items():
        if value in (None, "", []):
            continue
        normalized[_GET_PROVENANCE_RENAME.get(key, key)] = value
    return {field: normalized[field] for field in PROVENANCE_FIELDS if field in normalized}


def _strip_property_drawer(document: str) -> str:
    """Remove the metadata drawer kb adds on export.

    kb synthesizes a leading `:PROPERTIES:` drawer carrying the node id and
    timestamps. It is an envelope, never part of the stored body, so it must
    come off before comparing stored text against freshly rendered text.

    Args:
        document: The output of `kb get`.

    Returns:
        The document body.
    """
    lines = document.splitlines()
    if not lines or lines[0].strip() != ":PROPERTIES:":
        return document
    for index, line in enumerate(lines):
        if index and line.strip() == ":END:":
            return "\n".join(lines[index + 1 :])
    return document


def is_unchanged(stored: str, rendered: str) -> bool:
    """Whether a stored document already matches what would be written.

    Trailing whitespace is normalized before comparing. kb's generator appends
    a trailing newline that the renderer does not — measured at 284 bytes in,
    285 out — so a naive comparison reports every unchanged conversation as
    changed and rewrites, re-embeds, and bumps `updated_at` across the whole
    corpus on every run.

    Args:
        stored: The document currently in kb, envelope already removed.
        rendered: The document that would be written.

    Returns:
        True when the two are equal up to trailing whitespace.
    """
    return stored.rstrip() == rendered.rstrip()


def write_node(
    conversation: Conversation,
    title: str,
    db: str | None,
    run: CommandRunner,
    *,
    exists: bool,
    provenance: dict[str, object] | None = None,
) -> str:
    """Create or update the node for `conversation`.

    Args:
        conversation: The conversation to store.
        title: Its title.
        db: Database path, or `None` for kb's default.
        run: The subprocess runner.
        exists: Whether the node is already present.
        provenance: The `RawProvenance`-shaped mapping to attach, if any.
            Passed via `--provenance-json` on whichever of `kb create` or `kb
            update` this call makes, whenever there is anything to assert --
            this importer asserts only provenance it actually knows, and
            never passes an empty mapping just to make a call self-
            consistent. Since `kb update` (PLAN-20260923-project-identity
            T016) preserves a record's existing kind, source and provenance
            when `--provenance-json` is omitted, an empty `provenance` here
            correctly leaves whatever is already stored alone rather than
            clearing it.

    Returns:
        Either `"created"` or `"updated"`.

    Raises:
        CommandError: If kb rejects the write.
    """
    # No `--tag` flags. `render` already writes the tags into the document's
    # `#+filetags:` line, and kb derives a node's tags from its stored document.
    # Passing them again appends them to that line rather than merging: the
    # first re-import produced
    # `:conversation:codex-session:personal:conversation:codex-session:personal:`,
    # and each further run would have grown it again. The document is the single
    # source of tag truth, which also keeps the render/store round-trip stable
    # enough for `is_unchanged` to mean something.
    document = render(conversation, title)
    verb = "update" if exists else "create"
    id_args = [conversation.node_id] if exists else ["--id", conversation.node_id]
    if provenance:
        with _provenance_file(provenance) as path:
            run(
                [*_kb(db), verb, *id_args, "--provenance-json", str(path), "--json"],
                document,
            )
    else:
        run([*_kb(db), verb, *id_args, "--json"], document)
    return "updated" if exists else "created"


def _kb(db: str | None) -> list[str]:
    """Build the kb command prefix.

    Args:
        db: Database path, or `None` for kb's default.

    Returns:
        The argv prefix.
    """
    return ["kb", "--db", db] if db else ["kb"]
