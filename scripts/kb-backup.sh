#!/usr/bin/env bash
# Snapshot the kb database into a directory that is actually backed up.
#
# `~/.local/share/kb/` is excluded from Time Machine and Backblaze, because a
# continuously-mutating multi-hundred-megabyte SQLite file is a poor incremental
# backup target. This writes one consistent copy per day into `~/Documents`,
# which is included, so the corpus survives a disk loss.
#
# VACUUM INTO is used rather than `cp`: it is consistent against a live database
# with writers attached, and it compacts as it copies. Measured at 2.9s for a
# 634 MB corpus, so daily is cheap.
#
# The copy is written under a temporary name and moved into place only after its
# integrity is verified, so an interrupted or corrupt run never leaves a file
# that looks like a good backup.
set -euo pipefail

DB="${KB_DB_PATH:-$HOME/.local/share/kb/kb.db}"
DEST="${KB_BACKUP_DIR:-$HOME/Documents/kb-backups}"
KEEP_DAYS="${KB_BACKUP_KEEP_DAYS:-7}"

if [ ! -f "$DB" ]; then
  echo "kb-backup: no database at $DB" >&2
  exit 1
fi

mkdir -p "$DEST"
final="$DEST/kb-$(date +%F).db"
partial="$DEST/.kb-$(date +%F).partial"
rm -f "$partial"

sqlite3 "$DB" "VACUUM INTO '$partial';"

# `integrity_check` prints `ok` and nothing else when the file is sound. Trust
# the content rather than sqlite3's exit status, which is 0 for a report of
# corruption too.
if [ "$(sqlite3 "$partial" 'PRAGMA integrity_check;')" != "ok" ]; then
  echo "kb-backup: integrity check failed; leaving $partial for inspection" >&2
  exit 1
fi

mv -f "$partial" "$final"
echo "kb-backup: wrote $final ($(du -h "$final" | cut -f1))"

# Prune by age, but never remove the copy just written. A lapse in the schedule
# must not leave the directory empty just because everything in it aged out --
# and on a same-day re-run `$final` is itself older than the cutoff only if the
# clock is wrong, so excluding it explicitly is the cheap guard.
pruned=0
while IFS= read -r old; do
  [ "$old" = "$final" ] && continue
  rm -f "$old"
  pruned=$((pruned + 1))
done < <(find "$DEST" -maxdepth 1 -name 'kb-*.db' -mtime +"$KEEP_DAYS" 2>/dev/null)

echo "kb-backup: pruned $pruned backup(s) older than $KEEP_DAYS days"
