"""pacod-broker, read side: what the platform's catalogue says about the models kb can use.

pacod-broker lists models, embeddings, MCP servers, agents, tools and datasets that
providers announce, with prices, live status and reputation. kb only reads it, for the
admin's Broker page; it announces nothing.

* :func:`overview` — the gateway's models that name a broker resource
  (``broker_resource_id``) with that resource's reputation, price and status, and the models
  kb uses (a template followed to its model);
* :func:`catalogue` — every model and embedding the broker lists, each marked by how the
  gateway offers it: usable now under the name kb would call (a linked model, or a gateway
  template ``@name``), or not in the gateway;
* :func:`search` — a discovery query;
* :func:`probe` — whether the broker answers, is ready, and takes kb's credentials.

The client is pacod-broker's own (``pacod_broker``, vendored). It is always given the URL,
a token string and kb's name: given None it would read ``BROKER_*`` from the environment
and send the deployment's token to an address an admin set. Every failure becomes
:class:`BrokerUnavailableError`; the routes report it and never fail. Listings are written by
other providers: they are shown as text, never followed.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Any

import httpx
from pacod_broker import BrokerClient, BrokerError

if TYPE_CHECKING:
    from pipeline.broker_config import BrokerConnection
    from pipeline.gateway import GatewayModel

logger = logging.getLogger(__name__)

CLIENT_NAME = "pacod-kb"
KINDS = ("model", "embedding", "mcp_server", "agent", "tool", "compute", "dataset")
SEARCH_LIMIT = 20
# The broker caps a page at 200 and names the next one by a cursor; the catalogue reads at
# most MAX_CATALOGUE_PAGES per kind and says when it stopped short.
CATALOGUE_PAGE = 200
MAX_CATALOGUE_PAGES = 5
CATALOGUE_KINDS = ("model", "embedding")
MAX_FACT_IDS = 50
CACHE_SECS = 60.0
_FACT_WORKERS = 8
_PROBLEM_TYPE = re.compile(r"[a-z0-9_:/.-]{1,80}")


class BrokerUnavailableError(Exception):
    """No broker, or none that answers usefully; the message is safe for logs and the UI."""


def _client(conn: BrokerConnection, transport: httpx.BaseTransport | None) -> BrokerClient:
    if not conn.configured:
        raise BrokerUnavailableError("no broker is configured")
    return BrokerClient(
        conn.url,
        # Always a string (see the module docstring); blank = open mode, kb names itself.
        token=conn.token or "",
        client_name=CLIENT_NAME,
        timeout=conn.timeout,
        transport=transport,
    )


def _unavailable(exc: Exception, conn: BrokerConnection) -> BrokerUnavailableError:
    if isinstance(exc, BrokerError):
        if exc.status in (401, 403):
            return BrokerUnavailableError(f"the broker refused kb's credentials ({exc.status})")
        kind = f", {exc.kind}" if _PROBLEM_TYPE.fullmatch(exc.kind or "") else ""
        return BrokerUnavailableError(f"the broker answered {exc.status}{kind}")
    if isinstance(exc, httpx.TimeoutException):
        return BrokerUnavailableError(f"the broker did not answer within {conn.timeout:g} s")
    if isinstance(exc, ValueError):
        # A 200 that is not JSON: a web page (the broker's console, a proxy's login) answers
        # at this address, not pacod-broker's API.
        return BrokerUnavailableError(
            f"the answer from {conn.url} is not pacod-broker's API (its console, or a proxy?)"
        )
    return BrokerUnavailableError(f"the broker cannot be reached ({type(exc).__name__})")


# ---- a hit as kb shows it ---------------------------------------------------------------


def _price_text(price: Any) -> str:
    """``EUR input 0.25, output 0.5`` — the price in force now; ``free``; or blank."""
    if not isinstance(price, dict):
        return ""
    if price.get("free"):
        return "free"
    rates = price.get("rates")
    if not isinstance(rates, dict) or not rates:
        return ""
    parts = [
        f"{k} {v:g}"
        for k, v in list(rates.items())[:4]
        if isinstance(v, int | float) and not isinstance(v, bool)
    ]
    return f"{price.get('currency') or ''} {', '.join(parts)}".strip()


def _reputation(hit: dict[str, Any]) -> dict[str, Any] | None:
    """The resource's own score, else its provider's, with its label (``new`` while unproven)."""
    for key in ("reputation", "provider_reputation"):
        r = hit.get(key)
        if isinstance(r, dict) and isinstance(r.get("score"), int | float):
            return {
                "score": round(float(r["score"]), 3),
                "label": str(r.get("label") or ""),
                "of": "resource" if key == "reputation" else "provider",
            }
    return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def compact_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """One discovery hit, or one resource read, cut to what the Broker page shows."""
    res = _dict(hit.get("resource")) or hit
    provider = _dict(hit.get("provider"))
    live = _dict(hit.get("live"))
    model = _dict(res.get("model"))
    tags = [str(t) for t in (res.get("tags") or []) if isinstance(t, str)][:16]
    interfaces = [
        {"protocol": str(i.get("protocol") or ""), "url": str(i.get("url") or "")}
        for i in (res.get("interfaces") or [])[:4]
        if isinstance(i, dict)
    ]
    return {
        "id": str(hit.get("id") or res.get("id") or ""),
        "kind": str(res.get("kind") or ""),
        "name": str(res.get("title") or res.get("name") or ""),
        "summary": str(res.get("summary") or "")[:300],
        "provider": str(provider.get("title") or provider.get("name") or ""),
        "price": _price_text(hit.get("price") or hit.get("price_now")),
        "reputation": _reputation(hit),
        "status": str(live.get("status") or ""),
        "interfaces": interfaces,
        "tags": tags,
        # The name it is called by: a gateway offering announces a template as "@name".
        "upstream_id": str(model.get("upstream_id") or ""),
        "gateway": "gateway" in tags,
    }


# ---- search ----------------------------------------------------------------------------


def search(
    conn: BrokerConnection,
    query: str,
    kind: str | None = None,
    *,
    limit: int = SEARCH_LIMIT,
    transport: httpx.BaseTransport | None = None,
) -> list[dict[str, Any]]:
    """``POST /v1/discover`` with the words of *query*, and *kind* when the broker knows it."""
    body: dict[str, Any] = {"q": (query or "").strip()[:200], "limit": max(1, min(limit, 20))}
    if kind in KINDS:
        body["kind"] = kind
    client = _client(conn, transport)
    try:
        answer = client.discover(body)
    except (BrokerError, httpx.HTTPError, ValueError) as exc:
        raise _unavailable(exc, conn) from None
    finally:
        client.close()
    return [compact_hit(h) for h in _dict(answer).get("hits") or [] if isinstance(h, dict)]


# ---- the catalogue ---------------------------------------------------------------------

_lock = threading.Lock()
_catalogue: dict[tuple[str, str], tuple[float, list[dict[str, Any]], bool]] = {}
_facts: dict[tuple[str, str], tuple[float, dict[str, Any] | None]] = {}


def reset_cache() -> None:
    with _lock:
        _catalogue.clear()
        _facts.clear()


def _list_models(
    conn: BrokerConnection, transport: httpx.BaseTransport | None
) -> tuple[list[dict[str, Any]], bool]:
    client = _client(conn, transport)
    seen: dict[str, dict[str, Any]] = {}
    truncated = False
    try:
        for kind in CATALOGUE_KINDS:
            cursor: str | None = None
            for page in range(MAX_CATALOGUE_PAGES):
                body: dict[str, Any] = {"kind": kind, "limit": CATALOGUE_PAGE}
                if cursor:
                    body["cursor"] = cursor
                try:
                    answer = _dict(client.discover(body))
                except (BrokerError, httpx.HTTPError, ValueError) as exc:
                    # The first failure ends the listing: a dead broker costs one call.
                    raise _unavailable(exc, conn) from None
                for hit in answer.get("hits") or []:
                    if isinstance(hit, dict):
                        item = compact_hit(hit)
                        if item["id"]:
                            seen.setdefault(item["id"], item)
                cursor = str(answer.get("next_cursor") or "") or None
                if not cursor:
                    break
                if page == MAX_CATALOGUE_PAGES - 1:
                    truncated = True
    finally:
        client.close()
    items = sorted(seen.values(), key=lambda m: (m["kind"], m["name"].lower(), m["id"]))
    return items, truncated


def mark_usage(items: list[dict[str, Any]], rows: list[GatewayModel]) -> list[dict[str, Any]]:
    """Each item plus ``usage``: ``usable`` with the name kb would call, or ``not_in_gateway``."""
    linked = {r.broker_resource_id: r.id for r in rows if r.broker_resource_id}
    templates = {r.id for r in rows if r.kind == "template"}
    out = []
    for item in items:
        upstream = item.get("upstream_id") or ""
        call = linked.get(item["id"]) or (
            upstream if item.get("gateway") and upstream in templates else None
        )
        out.append(
            {**item, "usage": {"state": "usable" if call else "not_in_gateway", "call": call}}
        )
    return out


def catalogue(
    conn: BrokerConnection,
    rows: list[GatewayModel],
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Every model and embedding the broker lists (cached a minute), marked by the gateway."""
    key = (conn.url, conn.token)
    now = time.monotonic()
    with _lock:
        cached = _catalogue.get(key)
    if cached is not None and now - cached[0] < CACHE_SECS:
        items, truncated = cached[1], cached[2]
    else:
        items, truncated = _list_models(conn, transport)
        with _lock:
            _catalogue[key] = (now, items, truncated)
    return {"items": mark_usage(items, rows), "truncated": truncated}


# ---- facts of linked resources ----------------------------------------------------------


def _read_resource(client: BrokerClient, rid: str, conn: BrokerConnection) -> dict[str, Any] | None:
    """One resource as compact facts; None when the broker does not know it."""
    try:
        return compact_hit(_dict(client.resource(rid)))
    except BrokerError as exc:
        if exc.status in (401, 403):
            raise _unavailable(exc, conn) from None
        logger.info("Broker resource unreadable", extra={"resource": rid, "status": exc.status})
        return None
    except (httpx.HTTPError, ValueError) as exc:
        # Unreachable, too slow, or not the broker's API: the other ids would fare the same.
        raise _unavailable(exc, conn) from None


def resource_facts(
    conn: BrokerConnection,
    ids: list[str],
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, dict[str, Any] | None]:
    """``{id: compact resource | None}`` for broker resource ids, each read once a minute.

    An id the broker does not know is None. A broker that cannot be reached, is too slow, or
    refuses kb raises :class:`BrokerUnavailableError` after one call: the first missing id is read
    alone, the rest (at most MAX_FACT_IDS in all) concurrently under an overall deadline.
    """
    wanted = list(dict.fromkeys(i for i in ids if isinstance(i, str) and i.startswith("res_")))
    wanted = wanted[:MAX_FACT_IDS]
    now = time.monotonic()
    out: dict[str, dict[str, Any] | None] = {}
    missing: list[str] = []
    with _lock:
        for rid in wanted:
            cached = _facts.get((conn.url, rid))
            if cached is not None and now - cached[0] < CACHE_SECS:
                out[rid] = cached[1]
            else:
                missing.append(rid)
    if not missing:
        return out

    client = _client(conn, transport)
    read: dict[str, dict[str, Any] | None] = {}
    pool: concurrent.futures.ThreadPoolExecutor | None = None
    try:
        read[missing[0]] = _read_resource(client, missing[0], conn)
        rest = missing[1:]
        if rest:
            pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(_FACT_WORKERS, len(rest)), thread_name_prefix="broker-facts"
            )
            futures = {pool.submit(_read_resource, client, rid, conn): rid for rid in rest}
            try:
                for future in concurrent.futures.as_completed(futures, timeout=2 * conn.timeout):
                    read[futures[future]] = future.result()
            except concurrent.futures.TimeoutError:
                raise BrokerUnavailableError(
                    f"the broker did not answer within {2 * conn.timeout:g} s"
                ) from None
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        client.close()
    with _lock:
        for rid, facts in read.items():
            _facts[(conn.url, rid)] = (now, facts)
    out.update(read)
    return out


def overview(
    conn: BrokerConnection,
    rows: list[GatewayModel],
    in_use: dict[str, str],
    *,
    enabled: dict[str, bool] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """The Broker page's head: the models kb uses and the gateway's linked models, with facts.

    *in_use* maps a role (``converter``, ``security``) to the name kb calls; a template is
    followed to its model, whose ``broker_resource_id`` gives the reputation. *enabled* says
    which roles are switched on (all, when not given).
    """
    by_id = {r.id: r for r in rows}
    used: list[dict[str, Any]] = []
    for role, name in in_use.items():
        row = by_id.get(name)
        uses = row.model if row is not None and row.kind == "template" else None
        target = by_id.get(uses or name)
        used.append(
            {
                "role": role,
                "name": name,
                "enabled": True if enabled is None else enabled.get(role, True),
                "uses": uses,
                "listed": row is not None,
                "broker_resource_id": target.broker_resource_id if target is not None else None,
                "facts": None,
            }
        )
    linked: list[dict[str, Any]] = [
        {"id": r.id, "broker_resource_id": r.broker_resource_id, "facts": None, "used_as": []}
        for r in rows
        if r.kind == "model" and r.broker_resource_id
    ]
    for entry in linked:
        entry["used_as"] = [
            u["role"] for u in used if (u["uses"] or u["name"]) == entry["id"]
        ]

    broker_error: str | None = None
    ids = [str(e["broker_resource_id"]) for e in linked] + [
        str(u["broker_resource_id"]) for u in used if u["broker_resource_id"]
    ]
    if ids and conn.configured:
        try:
            facts = resource_facts(conn, ids, transport=transport)
        except BrokerUnavailableError as exc:
            broker_error = str(exc)
        else:
            for entry in [*linked, *used]:
                entry["facts"] = facts.get(str(entry["broker_resource_id"] or ""))
    elif not conn.configured:
        broker_error = "no broker is configured"
    return {"in_use": used, "linked": linked, "broker_error": broker_error}


# ---- the connection test ----------------------------------------------------------------


def probe(
    conn: BrokerConnection, *, transport: httpx.BaseTransport | None = None
) -> dict[str, Any]:
    """Whether the broker answers (``/healthz``), is ready (``/readyz``) and takes kb's
    credentials (``GET /v1/reputation/me``: the client and provider id it sees)."""
    out: dict[str, Any] = {
        "configured": conn.configured,
        "url": conn.url or None,
        "token": conn.sources.get("token", "none"),
        "reachable": False,
        "ready": False,
        "accepted": None,
        "client": None,
        "provider_id": None,
        "error": None,
        "hint": None,
    }
    if not conn.configured:
        out["error"] = "no broker is configured"
        return out
    try:
        with httpx.Client(transport=transport, timeout=conn.timeout) as http:
            health = http.get(f"{conn.url}/healthz")
    except httpx.HTTPError as exc:
        out["error"] = str(_unavailable(exc, conn))
        return out
    out["reachable"] = True
    if health.status_code != 200:
        out["error"] = f"{conn.url}/healthz answered {health.status_code}: is this pacod-broker?"
        return out

    client = _client(conn, transport)
    try:
        if not client.ready():
            out["error"] = "the broker is not ready (/readyz)"
            out["hint"] = "It may still be waiting for pacod-id; try again in a minute."
            return out
        out["ready"] = True
        me = _dict(client.me())
        out["accepted"] = True
        out["client"] = str(me.get("client") or "") or None
        out["provider_id"] = str(me.get("provider_id") or "") or None
    except BrokerError as exc:
        out["accepted"] = False if exc.status in (401, 403) else None
        out["error"] = str(_unavailable(exc, conn))
        if exc.status in (401, 403):
            out["hint"] = (
                "Check the token (a pacod-id service account's pidt_… with broker.consumer), "
                "or choose no token if this broker runs in open mode."
                if conn.token
                else "This broker wants a token: a pacod-id service account's pidt_… with "
                "broker.consumer."
            )
    except (httpx.HTTPError, ValueError) as exc:
        out["error"] = str(_unavailable(exc, conn))
    finally:
        client.close()
    return out
