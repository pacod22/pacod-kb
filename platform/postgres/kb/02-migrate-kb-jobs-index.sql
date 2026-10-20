-- Knowledge Base PostgreSQL migration: kb_jobs index optimization
--
-- Migration: Add composite index on kb_jobs(created_at, status) to optimize
-- the Jobs page query as job history grows.
--
-- The Jobs page polls GET /jobs which queries the kb_jobs table ordered by
-- created_at DESC with a limit of 50. As the job history accumulates over
-- time (especially with auto-ingest and watch tasks), individual indexes on
-- created_at alone become less efficient for this high-frequency query.
--
-- This composite index allows the database optimizer to satisfy both the
-- ORDER BY and any future filtering on status in a single index scan.
--
-- kb_jobs is created by the pipeline (SQLAlchemy create_all, with this index:
-- pipeline/db_models.py), so on a new database the table does not exist yet
-- when platform/postgres/kb-db.sh first runs: then this does nothing, and the
-- next run adds the index to a database whose table predates it.
--
-- Applied: June 2026
-- Status: Safe to apply to existing databases (idempotent)

DO $$
BEGIN
    IF to_regclass('kb_jobs') IS NOT NULL THEN
        CREATE INDEX IF NOT EXISTS ix_kb_jobs_created_at_status
            ON kb_jobs (created_at DESC, status);
    END IF;
END
$$;
