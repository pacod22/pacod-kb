#!/usr/bin/env bash
# Vendors two things from a pacod-id checkout into this repository:
#
#   vendor/wheels/pacod_id-<version>-py3-none-any.whl
#                                    pacod-id's Python package (pacod_id: token checks, the
#                                    authorization feed, the rules engine, the internal-API client)
#   web/static/pacod-ui/{tokens,console}.css
#                                    the design tokens of @pacod/ui, pacod-id's design package
#                                    (the look every pacod product's UI shares)
#   vendor/PACOD_ID_VERSION          the commit both were built from, and that commit's date
#
#   scripts/vendor-pacod-id.sh [<pacod-id checkout>]        (default: $PACOD_ID_SRC or ../pacod-id)
#   scripts/vendor-pacod-id.sh --css [<pacod-id checkout>]  only the CSS, from the recorded commit
#
# Both are built from the checkout's committed tree (git archive), never from its working copy,
# so the recorded commit is exactly what was packaged. The pacod-id image kb runs (`make
# images`) is built from the same checkout. pyproject.toml points uv at the wheel file
# ([tool.uv.sources]), so it is never resolved through an index: a package of the same name on
# PyPI cannot replace it. After vendoring, uv.lock is updated for the new file's hash. The CSS
# needs only node (packages/ui/scripts/build-tokens.js has no dependencies); the files are
# committed, so neither the UI image nor a checkout of kb needs node.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CSS_ONLY=false
if [ "${1:-}" = "--css" ]; then
  CSS_ONLY=true
  shift
fi
SRC="${1:-${PACOD_ID_SRC:-$(dirname "$ROOT")/pacod-id}}"

fail() { echo "vendor-pacod-id: $*" >&2; exit 1; }
git -C "$SRC" rev-parse --verify --quiet HEAD >/dev/null || fail "$SRC is not a git checkout of pacod-id"
command -v node >/dev/null || fail "node is needed (for the design tokens)"

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

# Builds @pacod/ui's tokens.css and console.css from commit $1 into web/static/pacod-ui/.
vendor_css() {
  local commit="$1" out="$ROOT/web/static/pacod-ui" f
  git -C "$SRC" cat-file -e "$commit:packages/ui/scripts/build-tokens.js" 2>/dev/null \
    || fail "$commit has no packages/ui/scripts/build-tokens.js"
  mkdir -p "$BUILD/css"
  git -C "$SRC" archive "$commit" packages/ui | tar -x -C "$BUILD/css"
  node "$BUILD/css/packages/ui/scripts/build-tokens.js"
  mkdir -p "$out"
  for f in tokens.css console.css; do
    {
      echo "/* @pacod/ui $f, vendored from pacod-id $commit by scripts/vendor-pacod-id.sh."
      echo "   Generated: do not edit; change the tokens in pacod-id and vendor again. */"
      cat "$BUILD/css/packages/ui/dist/$f"
    } > "$out/$f"
  done
}

if $CSS_ONLY; then
  COMMIT="$(awk '$1 == "commit" { print $2 }' "$ROOT/vendor/PACOD_ID_VERSION")"
  [ -n "$COMMIT" ] || fail "vendor/PACOD_ID_VERSION records no commit"
  vendor_css "$COMMIT"
  echo "vendored @pacod/ui tokens from pacod-id $(git -C "$SRC" rev-parse --short "$COMMIT")"
  exit 0
fi

git -C "$SRC" cat-file -e HEAD:python/pyproject.toml 2>/dev/null || fail "$SRC has no python/ package at HEAD"
command -v uv >/dev/null || fail "uv is needed"

COMMIT="$(git -C "$SRC" rev-parse HEAD)"
git -C "$SRC" archive "$COMMIT" python | tar -x -C "$BUILD"
uv build --quiet --wheel --out-dir "$BUILD/dist" "$BUILD/python"

WHEELS="$ROOT/vendor/wheels"
mkdir -p "$WHEELS"
rm -f "$WHEELS"/pacod_id-*.whl
cp "$BUILD"/dist/pacod_id-*.whl "$WHEELS/"
vendor_css "$COMMIT"
{
  echo "commit $COMMIT"
  echo "date $(git -C "$SRC" log -1 --format=%cI "$COMMIT")"
} > "$ROOT/vendor/PACOD_ID_VERSION"

(cd "$ROOT" && uv lock --quiet --upgrade-package pacod-id)
echo "vendored pacod-id $(git -C "$SRC" rev-parse --short "$COMMIT"): $(cd "$WHEELS" && ls pacod_id-*.whl), @pacod/ui tokens"
