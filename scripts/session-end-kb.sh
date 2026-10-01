#!/usr/bin/env bash
# SessionEnd hook: capture the just-closed Claude Code transcript in kb.
#
# Receives the hook payload on stdin:
#   { session_id, transcript_path, cwd, hook_event_name, reason }
#
# Two-tier model: the raw JSONL transcript stays where it is (grep-able audit
# log); this hook captures the readable projection of it - every human and
# assistant turn, model reasoning kept, tool calls and tool results dropped.
#
# Rendering is not done here. It is done by kb's transcript importer, so this
# hook and a bulk historical import cannot drift into two formats. Dropping
# tool calls is also the secret-hygiene measure: tool output is where
# environment dumps, file contents and command output live. That is why the
# rendering happens on this machine even when the record is destined for a
# server - see the plan's Decision Log, "Redaction must precede durability".
#
# Two destinations, chosen by whether a capture server is configured:
#
#   KB_INGEST_URL set - the importer posts the rendered document to that
#     server's ingest endpoint, which acknowledges once the bytes are durable
#     and turns them into a record asynchronously. A failed post is not
#     retried and nothing is spooled: the JSONL transcript is still on this
#     disk, so `kb-import --post ... --reconcile` fills the gap.
#
#   unset - the importer writes through kb locally, which is what happened
#     before T022 and what still happens on a machine with no server to talk
#     to. Capture must not stop working because a deployment has not happened.
set -uo pipefail

# Recursion guard: the importer derives a node title with `claude -p`, which is
# itself a Claude session whose SessionEnd would re-fire this hook. The guard
# variable is inherited by that child process.
if [ -n "${KB_SESSION_DISTILL:-}" ]; then exit 0; fi
export KB_SESSION_DISTILL=1

# kb lives in ~/.cargo/bin; `uv tool install` places kb-import in
# ~/.local/bin. Hooks do not always get a login-shell PATH.
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:$PATH"

payload=$(cat)
transcript=$(jq -r '.transcript_path // empty' <<<"$payload")
session_id=$(jq -r '.session_id // empty' <<<"$payload")
session_cwd=$(jq -r '.cwd // empty' <<<"$payload")

[ -n "$transcript" ] && [ -f "$transcript" ] && [ -n "$session_id" ] || exit 0

# Cheap floor: a three-line session is noise in a corpus searched by
# similarity, and posting it would cost a round trip to say so.
lines=$(wc -l <"$transcript")
[ "$lines" -ge 25 ] || exit 0

# Resolve the account config dir, mirroring clanker's convention:
# CLAUDE_CONFIG_DIR (already resolved by the wrapper for the session that just
# ended) wins; otherwise derive it from CLAUDE_ENV. The account becomes a kb
# tag, so a work session and a personal one stay distinguishable in the corpus.
if [ -z "${CLAUDE_CONFIG_DIR:-}" ] && [ -n "${CLAUDE_ENV:-}" ] &&
  [ -d "$HOME/.config/claude/$CLAUDE_ENV" ]; then
  CLAUDE_CONFIG_DIR="$HOME/.config/claude/$CLAUDE_ENV"
fi
if [ -n "${CLAUDE_CONFIG_DIR:-}" ]; then
  export CLAUDE_CONFIG_DIR
  account=$(basename "$CLAUDE_CONFIG_DIR")
else
  account="default"
fi

log_dir="$HOME/.local/state/kb-distill"
mkdir -p "$log_dir"
log="$log_dir/${session_id}.log"

IMPORTER=$(command -v kb-import) || IMPORTER=""
if [ -z "$IMPORTER" ]; then
  message="kb-import not found on PATH; session $session_id not captured. Install it with: uv tool install git+https://github.com/tftio/kb"
  echo "$message" >>"$log"
  echo "$message" >&2
  exit 0
fi
echo "Using kb importer: $IMPORTER" >>"$log"

# Detach: SessionEnd hooks have a short timeout and must not block exit. The
# post itself is bounded and quick, but title derivation is a `claude -p` call
# and rendering a long session is not free, so neither belongs inline.
#
# --transcript passes the path the payload already gave us, so the importer
# renders one file instead of scanning every account's session archive.
# Errors land in the per-session log, and this script exits 0 regardless: a
# capture that failed is a gap reconciliation can fill, while a hook that
# fails is a hook that interferes with closing a terminal.
#
# Provenance flags are computed inside the detached subshell too, alongside
# everything else that must not block the hook's own exit: this process
# already inherited whatever CLANKER_SESSION_* markers T003 exported at
# launch (PLAN-20260923-project-identity), and `bash -c` passes them through
# to the child unless something clears them. No slug grammar or remote
# normalization is repeated here -- `clanker project resolve` is asked
# directly, and kb's own boundary is what ultimately validates the result. A
# session not launched through clanker carries none of the markers; when
# `clanker` is not on PATH at all, provenance is simply omitted and capture
# proceeds with none -- this hook must never fail a capture over it.
# shellcheck disable=SC2016 # single quotes are the point: $1..$5 expand in the child shell
nohup bash -c '
  set -uo pipefail
  importer=$1; transcript=$2; session_id=$3; account=$4; session_cwd=$5

  destination=()
  if [ -n "${KB_INGEST_URL:-}" ]; then
    destination=(--post "$KB_INGEST_URL")
  fi

  provenance=()
  if [ -n "${CLANKER_SESSION:-}" ]; then
    [ -n "${CLANKER_SESSION_PROJECT:-}" ] && provenance+=(--project "$CLANKER_SESSION_PROJECT")
    [ -n "${CLANKER_SESSION_PROJECT_SOURCE:-}" ] &&
      provenance+=(--project-source "$CLANKER_SESSION_PROJECT_SOURCE")
    [ -n "${CLANKER_SESSION_REMOTE:-}" ] && provenance+=(--remote "$CLANKER_SESSION_REMOTE")
    [ -n "${CLANKER_SESSION_CONTEXT:-}" ] && provenance+=(--context "$CLANKER_SESSION_CONTEXT")
    [ -n "${CLANKER_SESSION_HARNESS:-}" ] && provenance+=(--harness "$CLANKER_SESSION_HARNESS")
    [ -n "${CLANKER_SESSION_MODEL:-}" ] && provenance+=(--model "$CLANKER_SESSION_MODEL")
    [ -n "${CLANKER_SESSION_ID:-}" ] && provenance+=(--clanker-session "$CLANKER_SESSION_ID")
    if [ -n "${CLANKER_SESSION_DOMAINS:-}" ]; then
      IFS="," read -r -a domains <<<"$CLANKER_SESSION_DOMAINS"
      for domain in "${domains[@]}"; do
        [ -n "$domain" ] && provenance+=(--domain "$domain")
      done
    fi
  elif command -v clanker >/dev/null 2>&1; then
    resolved=$(clanker project resolve --dir "$session_cwd" 2>/dev/null)
    resolved_project=$(cut -f1 <<<"$resolved")
    resolved_source=$(cut -f2 <<<"$resolved")
    resolved_remote=$(cut -f3 <<<"$resolved")
    [ -n "$resolved_project" ] && provenance+=(--project "$resolved_project")
    [ -n "$resolved_source" ] && provenance+=(--project-source "$resolved_source")
    [ -n "$resolved_remote" ] && provenance+=(--remote "$resolved_remote")
  fi

  "$importer" \
    --source claude-code \
    --transcript "$transcript" \
    --session "$session_id" \
    --account "$account" \
    "${destination[@]+"${destination[@]}"}" \
    "${provenance[@]+"${provenance[@]}"}" || {
    echo "capture failed with $?" >&2
    exit 1
  }

  echo "captured kb node cc-$session_id (account=$account cwd=$session_cwd${KB_INGEST_URL:+ via $KB_INGEST_URL})"
' _ "$IMPORTER" "$transcript" "$session_id" "$account" "$session_cwd" >>"$log" 2>&1 &
disown

exit 0
