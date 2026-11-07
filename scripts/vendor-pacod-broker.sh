#!/usr/bin/env bash
# Vendors pacod-broker's Python client from a pacod-broker checkout into this repository:
#
#   vendor/wheels/pacod_broker_client-<version>-py3-none-any.whl
#                                    the client (pacod_broker: discovery, resources, reputation);
#                                    the pipeline reads the broker with it (pipeline/broker.py)
#   vendor/PACOD_BROKER_VERSION      the commit it was built from, and that commit's date
#
#   scripts/vendor-pacod-broker.sh [<pacod-broker checkout>]  (default: $PACOD_BROKER_SRC or ../pacod-broker)
#
# Built from the checkout's committed tree (git archive), never from its working copy, so the
# recorded commit is exactly what was packaged. pyproject.toml points uv at the wheel file
# ([tool.uv.sources]), so it is never resolved through an index. After vendoring, uv.lock is
# updated for the new file's hash.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-${PACOD_BROKER_SRC:-$(dirname "$ROOT")/pacod-broker}}"

fail() { echo "vendor-pacod-broker: $*" >&2; exit 1; }
git -C "$SRC" rev-parse --verify --quiet HEAD >/dev/null || fail "$SRC is not a git checkout of pacod-broker"
git -C "$SRC" cat-file -e HEAD:clients/python/pyproject.toml 2>/dev/null \
  || fail "$SRC has no clients/python package at HEAD"
command -v uv >/dev/null || fail "uv is needed"

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

COMMIT="$(git -C "$SRC" rev-parse HEAD)"
git -C "$SRC" archive "$COMMIT" clients/python | tar -x -C "$BUILD"
uv build --quiet --wheel --out-dir "$BUILD/dist" "$BUILD/clients/python"

WHEELS="$ROOT/vendor/wheels"
mkdir -p "$WHEELS"
rm -f "$WHEELS"/pacod_broker_client-*.whl
cp "$BUILD"/dist/pacod_broker_client-*.whl "$WHEELS/"
{
  echo "commit $COMMIT"
  echo "date $(git -C "$SRC" log -1 --format=%cI "$COMMIT")"
} > "$ROOT/vendor/PACOD_BROKER_VERSION"

(cd "$ROOT" && uv lock --quiet --upgrade-package pacod-broker-client)
echo "vendored pacod-broker $(git -C "$SRC" rev-parse --short "$COMMIT"): $(cd "$WHEELS" && ls pacod_broker_client-*.whl)"
