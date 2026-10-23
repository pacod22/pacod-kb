"""kb CLI — entry point for all pipeline commands."""

from __future__ import annotations

from pathlib import Path

import typer

import kblog

app = typer.Typer(help="AI-ready knowledge base pipeline.")


@app.callback()
def _log_setup() -> None:
    # The log (kblog: LOG_LEVEL, LOG_FORMAT, SERVICE_NAME) on stderr: stdout is the commands'
    # output (kb export writes its JSON there). pipeline.config first: it reads .env.
    import pipeline.config  # noqa: F401

    kblog.setup("kb-pipeline", stream="stderr")


_DOC_TYPE_HELP = (
    "generic | web_page | legal_code | regulation | directive"
    " | decision | circular | article | paper | report | document | guide"
)


@app.command()
def fetch(
    urls: list[str] = typer.Argument(..., help="One or more URLs to fetch and convert to Markdown"),
    output_dir: str = typer.Option(".", "--output-dir", "-o", help="Directory to write .md files"),
    doc_type: str = typer.Option("web_page", "--doc-type", help=_DOC_TYPE_HELP),
) -> None:
    """Fetch one or more URLs and save each as a Markdown file (no KB ingestion)."""
    from pathlib import Path

    from pipeline.acquisition import slugify
    from pipeline.parsers.web import WebParser

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    parser = WebParser()
    used_names: set[str] = set()
    errors = 0

    for url in urls:
        try:
            result = parser.parse(url)
        except Exception as exc:
            typer.echo(f"ERROR  {url}  {exc}", err=True)
            errors += 1
            continue

        body = result.body
        if doc_type == "legal_code":
            from pipeline.promote_legal import promote_legal

            body = promote_legal(body)

        base = slugify(result.title or url.split("/")[-1] or "page")
        name = base
        counter = 2
        while name in used_names or (out / f"{name}.md").exists():
            name = f"{base}_{counter}"
            counter += 1
        used_names.add(name)

        out_path = out / f"{name}.md"
        out_path.write_text(body, encoding="utf-8")
        typer.echo(f"OK  {out_path}  ({result.title})")

    if errors:
        raise typer.Exit(code=1)


@app.command()
def ingest(
    source: str = typer.Argument(..., help="File path or URL to ingest"),
    doc_type: str = typer.Option("legal_code", "--doc-type", help=_DOC_TYPE_HELP),
    source_version: str = typer.Option("", "--source-version", help="Free-text version label"),
    supersedes: str | None = typer.Option(
        None, "--supersedes", help="doc_id of the prior version this document supersedes"
    ),
    skip_index: bool = typer.Option(False, "--skip-index", help="Skip embedding and indexing"),
    trusted: bool = typer.Option(
        False, "--trusted", help="Bypass the content-security scan (use only for trusted sources)"
    ),
    json_raw: bool = typer.Option(
        False, "--json-raw", help="Store .json files verbatim instead of converting to Markdown"
    ),
) -> None:
    """Ingest a document (file or URL) into the knowledge base."""
    from pipeline.acquisition import ingest_document
    from pipeline.config import load_config

    config = load_config()

    from pipeline.document_store import make_document_store
    from pipeline.indexer import index_document, make_embedder, make_fts_db
    from pipeline.vectorstore import make_vector_store

    store = make_document_store(config)
    _store = store if config.storage_backend == "database" else None

    try:
        doc_id, folder = ingest_document(
            source,
            config,
            doc_type=doc_type,
            source_version=source_version,
            supersedes=supersedes,
            trusted=trusted,
            store=_store,
            json_convert=not json_raw,
        )
    except Exception as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    folder_label = folder.name if folder is not None else "(database)"
    typer.echo(f"OK  {doc_id[:8]}  {folder_label}")

    if skip_index:
        return

    embedder = make_embedder(config)
    vector_store = make_vector_store(config)
    fts_db = make_fts_db(config)

    try:
        if folder is None and _store is not None:
            manifest = _store.load_manifest(doc_id)
            body = _store.load_body(doc_id)
            chunks = index_document(
                None,
                config,
                embedder,
                vector_store,
                fts_db,
                doc_data=(manifest, body),
            )
            _store.update_manifest(doc_id, chunk_count=len(chunks))
        else:
            chunks = index_document(folder, config, embedder, vector_store, fts_db)
        typer.echo(f"     {len(chunks)} chunks indexed")

        if supersedes:
            from datetime import UTC, datetime

            from pipeline.versioning import supersede_document

            today = datetime.now(UTC).date().isoformat()
            supersede_document(supersedes, today, config, vector_store, fts_db, store=_store)
            typer.echo(f"     superseded {supersedes[:8]}  valid_until={today}")
    except Exception as exc:
        typer.echo(f"WARN: indexing failed (non-fatal): {exc}", err=True)
    finally:
        fts_db.close()


@app.command()
def query(
    text: str = typer.Argument(..., help="Query text"),
    top_k: int = typer.Option(5, "--top-k", help="Number of results"),
    historical: bool = typer.Option(False, "--historical", help="Include superseded documents"),
    project: str = typer.Option("", help="Project slug to search (default: the default project)"),
) -> None:
    """Search the knowledge base with hybrid semantic + keyword retrieval."""
    from pipeline.config import load_config
    from pipeline.document_store import make_document_store
    from pipeline.indexer import make_embedder, make_fts_db
    from pipeline.projects import make_project_registry
    from pipeline.retrieval import SearchOptions, search
    from pipeline.vectorstore import QdrantVectorStore, make_vector_store

    config = load_config()
    store = make_document_store(config)
    registry = make_project_registry(config, getattr(store, "_factory", None))
    default = registry.ensure_default(config.qdrant_collection)
    target = registry.get(project) if project else default
    if target is None:
        typer.echo(f"ERROR: project '{project}' not found", err=True)
        raise typer.Exit(code=1)

    try:
        embedder = make_embedder(config)
        # Vectors and keyword hits from the same project (the keyword index spans all).
        vector_store = QdrantVectorStore(make_vector_store(config).client, target.collection_name)
        fts_db = make_fts_db(config)

        results = search(
            text,
            embedder,
            vector_store,
            fts_db,
            top_k=top_k,
            include_historical=historical,
            project=target.slug,
            options=SearchOptions(
                keyword_mode=config.keyword_query_mode,
                keyword_weight=config.rrf_keyword_weight,
            ),
        )
        fts_db.close()
    except Exception as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not results:
        typer.echo("No results found.")
        return

    for i, r in enumerate(results, start=1):
        path = " > ".join(r.heading_path) if r.heading_path else "(preamble)"
        typer.echo(f"\n[{i}] score={r.score:.4f}  doc={r.source_id[:8]}  path={path}")
        preview = r.text[:200].replace("\n", " ")
        typer.echo(f"    {preview}…")


@app.command()
def rebuild(
    project: str = typer.Option("", help="Project slug to rebuild (default: the default project)"),
    all_projects: bool = typer.Option(False, "--all-projects", help="Rebuild every project"),
    reuse: bool = typer.Option(
        True, "--reuse/--no-reuse", help="Keep vectors of chunks whose text is unchanged"
    ),
) -> None:
    """Drop and regenerate a project's index (or all projects') from the archive."""
    from pipeline.config import load_config
    from pipeline.document_store import make_document_store
    from pipeline.indexer import make_embedder, make_fts_db
    from pipeline.projects import make_project_registry
    from pipeline.vectorstore import QdrantVectorStore, make_vector_store
    from pipeline.versioning import rebuild_all_projects, rebuild_index

    config = load_config()
    store = make_document_store(config)
    _store = store if config.storage_backend == "database" else None
    registry = make_project_registry(config, getattr(store, "_factory", None))
    default = registry.ensure_default(config.qdrant_collection)
    target = registry.get(project) if project else default
    if target is None:
        typer.echo(f"ERROR: project '{project}' not found", err=True)
        raise typer.Exit(code=1)
    embedder = make_embedder(config)
    client = make_vector_store(config).client
    fts_db = make_fts_db(config)

    try:
        if all_projects:
            ok, err = rebuild_all_projects(
                config, embedder, client, fts_db, registry, store=_store, reuse_vectors=reuse
            )
        else:
            ok, err = rebuild_index(
                config,
                embedder,
                QdrantVectorStore(client, target.collection_name),
                fts_db,
                store=_store,
                project=target.slug,
                default_project=default.slug,
                reuse_vectors=reuse,
            )
    finally:
        fts_db.close()

    typer.echo(f"Rebuilt: {ok} documents, {err} errors")
    if err:
        raise typer.Exit(code=1)


@app.command()
def reindex(
    doc_id: str = typer.Argument(..., help="doc_id or 8-char prefix to re-index"),
) -> None:
    """Re-index a specific document (replace its chunks, keep source unchanged)."""
    import json

    from pipeline.config import load_config
    from pipeline.document_store import make_document_store
    from pipeline.indexer import index_document, make_embedder, make_fts_db
    from pipeline.vectorstore import make_vector_store

    config = load_config()
    store = make_document_store(config)
    _store = store if config.storage_backend == "database" else None

    embedder = make_embedder(config)
    vector_store = make_vector_store(config)
    fts_db = make_fts_db(config)

    try:
        if _store is not None:
            full_id = _store.find_by_prefix(doc_id) or doc_id
            manifest = _store.load_manifest(full_id)
            body = _store.load_body(full_id)
            chunks = index_document(
                None,
                config,
                embedder,
                vector_store,
                fts_db,
                doc_data=(manifest, body),
            )
            _store.update_manifest(full_id, chunk_count=len(chunks))
        else:
            from pipeline.versioning import find_doc_folder

            folder = find_doc_folder(doc_id, config, prefix=True)
            if folder is None:
                typer.echo(f"ERROR: document '{doc_id}' not found", err=True)
                raise typer.Exit(code=1)
            manifest_data = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            full_id = str(manifest_data["doc_id"])
            chunks = index_document(folder, config, embedder, vector_store, fts_db)
    finally:
        fts_db.close()

    typer.echo(f"Re-indexed {full_id[:8]}: {len(chunks)} chunks")


@app.command()
def status() -> None:
    """Show knowledge base status (documents, versions, chunks, failures, review queue)."""
    import json

    from pipeline.config import load_config

    config = load_config()
    docs_dir = config.documents_dir

    if not docs_dir.exists():
        typer.echo("No documents directory found. Run 'kb ingest' first.")
        return

    total = active = superseded = errors = chunk_total = 0
    for folder in sorted(docs_dir.iterdir()):
        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            continue
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        total += 1
        chunk_total += int(data.get("chunk_count", 0))
        if data.get("valid_until") is not None:
            superseded += 1
        elif data.get("status") == "ok":
            active += 1
        elif data.get("status") == "error":
            errors += 1

    review_count = 0
    if config.review_dir.exists():
        review_count = sum(1 for _ in config.review_dir.iterdir())

    typer.echo(f"Documents  : {total}")
    typer.echo(f"  Active   : {active}")
    typer.echo(f"  Superseded: {superseded}")
    typer.echo(f"  Errors   : {errors}")
    typer.echo(f"Chunks     : {chunk_total}")
    typer.echo(f"Review Q   : {review_count}")


@app.command()
def watch(
    once: bool = typer.Option(False, "--once", help="Run a single pass and exit"),
    interval: int = typer.Option(3600, "--interval", help="Polling interval in seconds"),
    watch_file: str = typer.Option(
        "", "--watch-file", help="Path to watch.json (default: KB_ROOT/watch.json)"
    ),
) -> None:
    """Poll configured web sources for changes; re-ingest on sha256 change."""
    import time
    from pathlib import Path

    from pipeline.config import load_config
    from pipeline.indexer import make_embedder, make_fts_db
    from pipeline.vectorstore import make_vector_store
    from pipeline.watch import watch_sources

    config = load_config()
    wf = Path(watch_file) if watch_file else config.kb_root / "watch.json"
    embedder = make_embedder(config)
    vector_store = make_vector_store(config)
    fts_db = make_fts_db(config)

    from pipeline.document_store import make_document_store

    store = make_document_store(config)
    _store = store if config.storage_backend == "database" else None

    try:
        while True:
            results = watch_sources(config, wf, embedder, vector_store, fts_db, store=_store)
            for r in results:
                s = r["status"]
                lbl = r.get("label", r["url"])
                if s == "unchanged":
                    typer.echo(f"  UNCHANGED  {lbl}")
                elif s == "new":
                    typer.echo(f"  NEW        {lbl}  ({r['doc_id']})")
                elif s == "updated":
                    typer.echo(f"  UPDATED    {lbl}  {r['prior_doc_id']} → {r['new_doc_id']}")
                elif s == "error":
                    typer.echo(f"  ERROR      {lbl}  {r['error']}", err=True)
            if once:
                break
            typer.echo(f"Sleeping {interval}s…")
            time.sleep(interval)
    finally:
        fts_db.close()


@app.command()
def export(
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file (default: stdout)"
    ),
) -> None:
    """Export all ingestion metadata and document bodies to a JSON file."""
    import json as _json
    from datetime import UTC, datetime

    from pipeline.config import load_config
    from pipeline.document_store import make_document_store

    config = load_config()
    store = make_document_store(config)
    docs = store.export_all()
    payload = {
        "export_version": "1",
        "exported_at": datetime.now(UTC).isoformat(),
        "storage_backend": config.storage_backend,
        "document_count": len(docs),
        "documents": docs,
    }
    text = _json.dumps(payload, indent=2, ensure_ascii=False)
    if output is not None:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Exported {len(docs)} documents → {output}")
    else:
        typer.echo(text)


@app.command()
def export_sources(
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file (default: stdout)"
    ),
    fmt: str = typer.Option("json", "--format", "-f", help="Output format: json or csv"),
) -> None:
    """Export ingested sources in accounting-agent ingested_sources table format."""
    import json as _json
    from datetime import UTC, datetime

    from pipeline.config import load_config
    from pipeline.document_store import _sources_to_csv, make_document_store

    if fmt not in ("json", "csv"):
        typer.echo(f"Unknown format {fmt!r}. Use 'json' or 'csv'.", err=True)
        raise typer.Exit(1)

    config = load_config()
    store = make_document_store(config)
    rows = store.export_as_ingested_sources()

    if fmt == "csv":
        text = "﻿" + _sources_to_csv(rows)  # BOM for Excel UTF-8
    else:
        payload = {
            "export_version": "1",
            "exported_at": datetime.now(UTC).isoformat(),
            "source_count": len(rows),
            "sources": rows,
        }
        text = _json.dumps(payload, indent=2, ensure_ascii=False)

    if output is not None:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Exported {len(rows)} sources → {output}")
    else:
        typer.echo(text)
