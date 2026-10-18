#!/usr/bin/env bash
# Vendors pacod-id's Python package (pacod_id: token checks, the authorization feed, the rules
# engine, the internal-API client) into this repository, from a pacod-id checkout:
#
#   vendor/wheels/pacod_id-<version>-py3-none-any.whl
#   vendor/PACOD_ID_VERSION          the commit it was built from, and that commit's date
#
#   scripts/vendor-pacod-id.sh [<pacod-id checkout>]     (default: $PACOD_ID_SRC or ../pacod-id)
#
# The wheel is built from the checkout's committed tree at HEAD (git archive), never from its
# working copy, so the recorded commit is exactly what was packaged. The pacod-id image kb runs
# (`make images`) is built from the same checkout. pyproject.toml points uv at the wheel file
# ([tool.uv.sources]), so it is never resolved through an index: a package of the same name on
# PyPI cannot replace it. After vendoring, uv.lock is updated for the new file's hash.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-${PACOD_ID_SRC:-$(dirname "$ROOT")/pacod-id}}"

fail() { echo "vendor-pacod-id: $*" >&2; exit 1; }
git -C "$SRC" rev-parse --verify --quiet HEAD >/dev/null || fail "$SRC is not a git checkout of pacod-id"
git -C "$SRC" cat-file -e HEAD:python/pyproject.toml 2>/dev/null || fail "$SRC has no python/ package at HEAD"
command -v uv >/dev/null || fail "uv is needed"

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
git -C "$SRC" archive HEAD python | tar -x -C "$BUILD"
uv build --quiet --wheel --out-dir "$BUILD/dist" "$BUILD/python"

WHEELS="$ROOT/vendor/wheels"
mkdir -p "$WHEELS"
rm -f "$WHEELS"/pacod_id-*.whl
cp "$BUILD"/dist/pacod_id-*.whl "$WHEELS/"
{
  echo "commit $(git -C "$SRC" rev-parse HEAD)"
  echo "date $(git -C "$SRC" log -1 --format=%cI)"
} > "$ROOT/vendor/PACOD_ID_VERSION"

(cd "$ROOT" && uv lock --quiet --upgrade-package pacod-id)
echo "vendored pacod-id $(git -C "$SRC" rev-parse --short HEAD): $(cd "$WHEELS" && ls pacod_id-*.whl)"
