#!/usr/bin/env bash
# Verify that the kb-import distribution packages and installs correctly:
# the built wheel and sdist carry the right version and contents, and a tool
# install of the wheel produces a working `kb-import` entry point. Everything
# happens under one temporary directory (build output, uv's tool dir, and its
# bin dir), so the operator's real `uv tool` state is never touched and no
# lockfile is read for write.
#
# Kept as its own script rather than an inline mise task body because the
# assertions are too long to read as one shell one-liner; it is exactly what
# `mise run check:python-package` runs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fail() {
  echo "check-python-package: $*" >&2
  exit 1
}

# --- expected version, from Cargo.toml's [package] table only -------------
# Restrict the search to lines between the [package] header and the next
# table header, so a dependency's own `version = "..."` later in the file is
# never a candidate.
expected_version="$(
  sed -n '/^\[package\]/,/^\[/{/^version[[:space:]]*=/p;}' Cargo.toml | head -n1 |
    sed -E 's/^version[[:space:]]*=[[:space:]]*"([^"]*)".*/\1/'
)"
[ -n "$expected_version" ] || fail "could not read [package].version from Cargo.toml"

# --- build wheel and sdist, writing only into $tmp -------------------------
uv build --out-dir "$tmp/dist" || fail "uv build failed"

wheel="$(find "$tmp/dist" -maxdepth 1 -name '*.whl' | head -n1)"
sdist="$(find "$tmp/dist" -maxdepth 1 -name '*.tar.gz' | head -n1)"
[ -n "$wheel" ] || fail "no wheel produced under $tmp/dist"
[ -n "$sdist" ] || fail "no sdist produced under $tmp/dist"

# --- wheel filename carries the expected version ---------------------------
case "$(basename "$wheel")" in
  kb_import-"$expected_version"-*.whl) ;;
  *) fail "wheel filename $(basename "$wheel") does not carry version $expected_version" ;;
esac

# --- wheel contents: only kb_import/ and its dist-info -------------------
bad_wheel_entries="$(
  unzip -Z1 "$wheel" |
    grep -v -E "^kb_import/" |
    grep -v -E "^kb_import-${expected_version}\.dist-info/" || true
)"
[ -z "$bad_wheel_entries" ] || fail "wheel contains entries outside kb_import/ and its dist-info: $bad_wheel_entries"

# --- sdist contents: no Rust tree ------------------------------------------
bad_sdist_entries="$(
  tar -tzf "$sdist" | grep -E '(^|/)(src|target)/|(^|/)Cargo\.lock$' || true
)"
[ -z "$bad_sdist_entries" ] || fail "sdist contains Rust-tree entries: $bad_sdist_entries"

# --- install the wheel into a throwaway tool dir, never the real one ------
UV_TOOL_DIR="$tmp/tools" UV_TOOL_BIN_DIR="$tmp/bin" \
  uv tool install "$wheel" || fail "uv tool install failed"

installed_bin="$tmp/bin/kb-import"
[ -x "$installed_bin" ] || fail "kb-import binary not found at $installed_bin after install"

version_output="$("$installed_bin" --version)"
expected_version_line="kb-import $expected_version"
[ "$version_output" = "$expected_version_line" ] ||
  fail "kb-import --version printed '$version_output', expected '$expected_version_line'"

"$installed_bin" --help >/dev/null || fail "kb-import --help failed to run"

echo "check-python-package: OK (version $expected_version)"
