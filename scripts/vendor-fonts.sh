#!/usr/bin/env bash
# Vendors the three typefaces of the pacod design (pacod-id's @pacod/ui tokens name them) into
# web/static/fonts/, so the UI loads no font from a third party:
#
#   Inter (text), Manrope (headings, buttons), JetBrains Mono (captions, code)
#
# Taken from fontsource's variable packages at the versions pacod-gw's console uses, latin,
# latin-ext, cyrillic and cyrillic-ext subsets only (kb shows bg/ru/sr/mk text). fonts.css is
# generated from each package's wght.css with only those subsets; each family's OFL licence is
# copied next to its files.
#
#   scripts/vendor-fonts.sh        (needs npm and the npm registry)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/web/static/fonts"
PACKAGES=(inter@5.3.0 manrope@5.3.0 jetbrains-mono@5.3.0)
SUBSETS='(latin|latin-ext|cyrillic|cyrillic-ext)'

command -v npm >/dev/null || { echo "vendor-fonts: npm is needed" >&2; exit 1; }
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

rm -rf "$OUT"
mkdir -p "$OUT"
{
  echo "/* Inter, Manrope and JetBrains Mono (variable, OFL-1.1), vendored by scripts/vendor-fonts.sh"
  echo "   from fontsource: ${PACKAGES[*]}. Generated: do not edit. */"
} > "$OUT/fonts.css"

for spec in "${PACKAGES[@]}"; do
  family="${spec%@*}"
  mkdir -p "$BUILD/$family" "$OUT/$family"
  (cd "$BUILD/$family" && npm pack --silent "@fontsource-variable/$spec" >/dev/null && tar -xzf ./*.tgz)
  pkg="$BUILD/$family/package"
  cp "$pkg/LICENSE" "$OUT/$family/LICENSE"
  # wght.css is a list of blocks, each "/* <family>-<subset>-wght-normal */" + one @font-face.
  awk -v family="$family" -v keep="^/[*] $family-$SUBSETS-wght-normal [*]/\$" '
    /^\/\*/ { on = ($0 ~ keep) }
    on { gsub(/url\(\.\/files\//, "url(" family "/"); print }
  ' "$pkg/wght.css" >> "$OUT/fonts.css"
  for f in $(grep -oE "$family/[a-z0-9-]+\\.woff2" "$OUT/fonts.css" | sort -u); do
    cp "$pkg/files/${f#"$family"/}" "$OUT/$f"
  done
done

echo "vendored fonts into web/static/fonts:"
(cd "$OUT" && find . -name '*.woff2' | sort && du -sh .)
