#!/bin/sh
# kb's role and database on the shared PostgreSQL (pacod-postgres): a role "kb" that is not a
# superuser and owns the database "kb", and kb's schema there (kb/*.sql, as the role kb so it owns
# what they create; the pipeline adds its other tables itself at start).
#
# pacod-db-init runs this at every start, before kb-pipeline, as a superuser through libpq's PG*
# variables (PGHOST, PGUSER, PGPASSWORD). Safe to run again: it creates what is missing, sets the
# role's password to KB_DATABASE_PASSWORD and applies the SQL files, which are idempotent. The
# password is read with psql's \getenv, so it is on no command line, and nothing secret is printed.
#
# kb's own stack (docker-compose.standalone.yml) connects as the role kb itself, the superuser of
# its PostgreSQL since before the shared network: then the role is left as it is. The full stack
# (pacod-demo) connects as postgres. Mount this directory whole: the SQL is read from kb/ next to
# this file.

set -eu

: "${PGHOST:?the PostgreSQL host, with PGUSER and PGPASSWORD of a superuser}"
: "${KB_DATABASE_PASSWORD:?the password for the role kb}"

sql_dir="$(dirname "$0")/kb"
set -- "$sql_dir"/*.sql
[ -f "$1" ] || { echo "kb-db: no SQL in $sql_dir (mount platform/postgres whole)" >&2; exit 1; }

psql -v ON_ERROR_STOP=1 -q -d postgres <<'SQL'
\getenv pw KB_DATABASE_PASSWORD
SELECT format('CREATE ROLE kb LOGIN PASSWORD %L', :'pw')
 WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'kb') \gexec
SELECT format('ALTER ROLE kb LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD %L', :'pw')
 WHERE current_user <> 'kb' \gexec
SELECT 'CREATE DATABASE kb OWNER kb'
 WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'kb') \gexec
REVOKE ALL ON DATABASE kb FROM PUBLIC;
GRANT CONNECT, CREATE, TEMPORARY ON DATABASE kb TO kb;
SQL

# One session: the tables and indexes belong to kb, whoever connected.
{
  echo 'SET ROLE kb; SET client_min_messages = warning;'
  for f in "$@"; do
    cat "$f"
    echo
  done
} | psql -v ON_ERROR_STOP=1 -q -d kb

echo "kb-db: the role and the database kb are in place, schema applied"
