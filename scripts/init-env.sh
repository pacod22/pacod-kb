#!/usr/bin/env bash
# make setup: writes this machine's .env from .env.example, or completes an existing one.
#
# A new .env is a copy of .env.example. Then every secret the configured services need and the
# file leaves blank (or does not name) is generated in place; a value already written stays. No
# value is ever printed.
#
# It refuses to invent a secret that data already depends on, and writes nothing then: the kb
# database password once /data/postgres holds data (PostgreSQL took it on its first start), and
# pacod-id's while its database volume exists (its role was created with them, and what it
# stored is sealed under PID_KEK).
#
# Which secrets: always the pipeline's and the UI's; pacod-id's when the id profile runs; the
# development gateway's key when dev-gateway runs (COMPOSE_PROFILES, from the environment or .env).
set -euo pipefail
cd "$(dirname "$0")/.."

fail() { echo "setup: $*" >&2; exit 1; }

[ -f .env.example ] || fail ".env.example is missing"
new_file=
if [ ! -f .env ]; then
  umask 077
  cp .env.example .env
  new_file=1
fi

# The last value KEY is given in .env, as written ("" when blank or absent).
value_of() {
  local line
  line=$(grep -E "^$1=" .env | tail -n 1 || true)
  line=${line#*=}
  line=${line%%#*}
  printf '%s' "$(printf '%s' "$line" | tr -d '[:space:]')"
}
hex() { openssl rand -hex "$1"; }
generate() {
  case "$1" in
    hex*) hex "${1#hex}" ;;
    b64) openssl rand -base64 32 | tr -d '\n' ;;
    # pacod-id requires role holders' passwords to be 12+ characters and not common.
    password) openssl rand -base64 24 | tr -d '/+=\n' | cut -c1-24 ;;
    gw) printf 'pgw_kb_%s' "$(hex 24)" ;;
  esac
}

profiles=${COMPOSE_PROFILES:-$(value_of COMPOSE_PROFILES)}
has_profile() { case ",$profiles," in *,"$1",*) return 0 ;; *) return 1 ;; esac; }

# ---- what data exists ----------------------------------------------------------------------------
project=${COMPOSE_PROJECT_NAME:-$(basename "$PWD" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')}
id_data=
if docker info >/dev/null 2>&1; then
  if docker volume inspect "${project}_kb-id-pg-data" >/dev/null 2>&1; then
    id_data="volume ${project}_kb-id-pg-data"
  fi
fi
pg_data=
if [ -d /data/postgres ] && [ -n "$(ls -A /data/postgres 2>/dev/null)" ]; then
  pg_data="the folder /data/postgres"
fi

# ---- the secrets this machine lacks ----------------------------------------------------------------
# KEY:kind:guard -- guard names the data the value is bound to once it exists.
SECRETS="PIPELINE_API_SECRET:hex24: UI_SESSION_SECRET:hex32: KB_DATABASE_PASSWORD:hex16:pg"
if has_profile id; then
  SECRETS="$SECRETS PID_PG_SUPERUSER_PASSWORD:hex16:id PID_DB_PASSWORD:hex16:id
  PID_KEK:b64:id PID_TOKEN_PEPPER:b64:id KB_ID_UI_SECRET:hex32:id KB_ID_SERVICE_SECRET:hex32:id
  ID_ADMIN_PASSWORD:password:id"
fi
if has_profile dev-gateway; then
  SECRETS="$SECRETS PACOD_GW_API_KEY:gw:"
fi

# bash 3.2 (macOS) calls an empty array unbound under set -u: expand as ${arr[@]+"${arr[@]}"}.
keys=() values=() blocked=()
for spec in $SECRETS; do
  key=${spec%%:*} rest=${spec#*:}
  kind=${rest%%:*} guard=${rest#*:}
  [ -n "$(value_of "$key")" ] && continue
  case "$guard" in
    id) bound=$id_data ;;
    pg) bound=$pg_data ;;
    *) bound= ;;
  esac
  if [ -n "$bound" ]; then blocked+=("$key ($bound)"); continue; fi
  keys+=("$key")
  values+=("$(generate "$kind")")
done

if [ ${#blocked[@]} -gt 0 ]; then
  echo "setup: these secrets are blank, and data that depends on them already exists:" >&2
  printf '  %s\n' "${blocked[@]}" >&2
  cat >&2 <<END
A new value would not match it: PostgreSQL keeps the password of its first start, and pacod-id's
database and sealed data the values they were created with. Put the old values back into .env
(from a backup of it). Nothing was written.
END
  [ -z "$new_file" ] || rm -f .env
  exit 1
fi

if [ ${#keys[@]} -eq 0 ]; then
  echo "setup: .env already has every secret the configured services need (profiles: $profiles)."
  exit 0
fi

tmp=$(mktemp .env.XXXXXX)
trap 'rm -f "$tmp"' EXIT
cp .env "$tmp"
appended=()
for i in "${!keys[@]}"; do
  key=${keys[$i]} value=${values[$i]}
  if grep -qE "^$key=" "$tmp"; then
    # Replace the (blank) assignment, keeping a trailing comment.
    awk -v k="$key" -v v="$value" '
      BEGIN { done = 0 }
      !done && index($0, k "=") == 1 {
        comment = ""
        if (match($0, /[[:space:]]#.*$/)) comment = substr($0, RSTART)
        print k "=" v comment; done = 1; next
      }
      { print }' "$tmp" > "$tmp.new" && mv "$tmp.new" "$tmp"
  else
    appended+=("$key=$value")
  fi
done
if [ ${#appended[@]} -gt 0 ]; then
  { echo; echo "# ---- secrets added by make setup on $(date +%Y-%m-%d) ----"; printf '%s\n' "${appended[@]}"; } >> "$tmp"
fi
chmod 600 "$tmp"
mv "$tmp" .env
trap - EXIT

echo "setup: $( [ -n "$new_file" ] && echo "wrote .env from .env.example" || echo "completed .env" ); generated (not printed): ${keys[*]}"
if printf '%s\n' "${keys[@]}" | grep -qx ID_ADMIN_PASSWORD; then
  echo "setup: the first kb admin signs in as ID_ADMIN_EMAIL with ID_ADMIN_PASSWORD (grep ID_ADMIN_ .env)"
fi
