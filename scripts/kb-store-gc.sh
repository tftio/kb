#!/usr/bin/env bash
# Pack the kb blob store.
#
# Packing is what makes the store cheap. A resumed session appends to its
# transcript and is stored again in full; loose, that is one copy per resume,
# and packed it is a delta against the previous one. Nothing does this on its
# own, so it is scheduled rather than hoped for.
#
# Deliberately `--no-prune`: this never deletes. Records are blobs bound to
# refs, so a blob whose ref has moved on -- the previous version of an edited
# note -- is unreachable, and an ordinary `git gc` would eventually delete it.
# Unreachable is not the same as unwanted here: superseded content is the
# archival history the store exists to hold, and the index cites blobs by hash
# without regard to what any ref currently points at. Pruning becomes safe only
# once records carry a commit chain that makes their history reachable, which
# is not built yet -- see the plan's Decision Log, 2026-08-14.
set -euo pipefail

STORE="${KB_STORE_PATH:-$HOME/.local/share/kb/store}"

if [ ! -d "$STORE/objects" ]; then
  echo "kb-store-gc: no blob store at $STORE" >&2
  exit 1
fi

before="$(git -C "$STORE" count-objects -v | awk '/^count:/ {print $2}')"
git -C "$STORE" gc --no-prune --quiet
after="$(git -C "$STORE" count-objects -v | awk '/^count:/ {print $2}')"
packed="$(git -C "$STORE" count-objects -v | awk '/^in-pack:/ {print $2}')"
size="$(git -C "$STORE" count-objects -v | awk '/^size-pack:/ {print $2}')"

echo "kb-store-gc: packed $STORE -- loose $before -> $after, in-pack $packed, pack size ${size}KiB"
