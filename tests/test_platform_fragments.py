"""platform/: kb's share of the shared configuration (pacod-demo/docs/NETWORK.md, section 4)."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).parent.parent
_PG = _ROOT / "platform" / "postgres"


def test_kb_db_script_never_demotes_the_connecting_role() -> None:
    script = (_PG / "kb-db.sh").read_text(encoding="utf-8")
    # kb's own stack connects as kb, its PostgreSQL's superuser: ALTER ROLE would lock it out.
    assert "WHERE current_user <> 'kb' \\gexec" in script
    assert "LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE" in script
    # Secrets via \getenv, never on a command line; the schema is applied as kb.
    assert "\\getenv pw KB_DATABASE_PASSWORD" in script
    assert "SET ROLE kb;" in script


def test_the_jobs_index_waits_for_its_table() -> None:
    sql = (_PG / "kb" / "02-migrate-kb-jobs-index.sql").read_text(encoding="utf-8")
    assert "to_regclass('kb_jobs') IS NOT NULL" in sql


def test_the_orm_creates_the_jobs_index_with_its_table() -> None:
    from pipeline.db_models import KbJob

    (index,) = [i for i in KbJob.__table__.indexes if i.name == "ix_kb_jobs_created_at_status"]
    assert [str(e) for e in index.expressions] == ["kb_jobs.created_at DESC", "kb_jobs.status"]


def test_the_caddy_site_is_kb_ui() -> None:
    site = (_ROOT / "platform" / "caddy" / "kb.caddy").read_text(encoding="utf-8")
    assert "{$KB_SITE:http://kb.localhost} {" in site
    assert "reverse_proxy kb-ui:8080" in site
    assert "delete code" in site and "delete state" in site
    # Site blocks only: an imported file cannot carry global options.
    assert not site.lstrip().startswith("{\n")
