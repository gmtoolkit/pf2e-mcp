#!/usr/bin/env python3
"""
TTRPG MCP server — semantic search over PF2e, SF2e, and Draw Steel content stored in Qdrant.

Env vars:
  VOYAGE_API_KEY       - Voyage AI key
  QDRANT_URL           - Qdrant Cloud URL
  QDRANT_API_KEY       - Qdrant Cloud API key
  QDRANT_COLLECTION    - PF2e collection (default: pf2e)
  SF2E_COLLECTION      - SF2e collection (default: sf2e)
  DS_COLLECTION        - Draw Steel collection (default: draw-steel)
  VOYAGE_MODEL         - (default: voyage-3)
  PORT                 - HTTP port (default: 8000)
  DATABASE_URL         - Postgres URL for rate limiting + call logging
  AUTH_ISSUER          - JWT issuer (default: https://gmkit.io)
  RATE_LIMIT_PER_DAY   - Max calls per user per day (default: 100)
"""
import asyncio
import os
import time
from contextvars import ContextVar
from functools import wraps
from typing import Optional

import asyncpg
import httpx
import jwt
import voyageai
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

VOYAGE_API_KEY    = os.environ["VOYAGE_API_KEY"]
VOYAGE_MODEL      = os.environ.get("VOYAGE_MODEL", "voyage-3")
QDRANT_URL        = os.environ.get("QDRANT_URL", "http://qdrant.infra.svc.cluster.local:6333")
QDRANT_API_KEY    = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "pf2e")
SF2E_COLLECTION   = os.environ.get("SF2E_COLLECTION", "sf2e")
DS_COLLECTION     = os.environ.get("DS_COLLECTION", "draw-steel")
DND5E_COLLECTION  = os.environ.get("DND5E_COLLECTION", "dnd5e")
DND2024_COLLECTION = os.environ.get("DND2024_COLLECTION", "dnd2024")
DH_COLLECTION     = os.environ.get("DH_COLLECTION", "daggerheart")
DATABASE_URL      = os.environ.get("DATABASE_URL", "")
AUTH_ISSUER       = os.environ.get("AUTH_ISSUER", "https://gmkit.io")
RATE_LIMIT        = int(os.environ.get("RATE_LIMIT_PER_DAY", "100"))

voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

PORT = int(os.environ.get("PORT", "8000"))

# --- JWKS cache ---
_jwks_cache: dict = {}
_jwks_fetched_at: float = 0.0
_JWKS_TTL = 3600

async def _get_public_keys() -> dict:
    global _jwks_cache, _jwks_fetched_at
    if time.time() - _jwks_fetched_at < _JWKS_TTL and _jwks_cache:
        return _jwks_cache
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{AUTH_ISSUER}/.well-known/jwks.json", timeout=5)
        r.raise_for_status()
        jwks = r.json()
    keys = {}
    for k in jwks.get("keys", []):
        keys[k["kid"]] = jwt.algorithms.RSAAlgorithm.from_jwk(k)
    _jwks_cache = keys
    _jwks_fetched_at = time.time()
    return keys

async def _validate_token(token: str) -> Optional[dict]:
    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        keys = await _get_public_keys()
        if kid not in keys:
            _jwks_fetched_at = 0
            keys = await _get_public_keys()
        public_key = keys.get(kid)
        if not public_key:
            return None
        payload = jwt.decode(
            token, public_key,
            algorithms=["RS256"],
            issuer=AUTH_ISSUER,
            options={"verify_aud": False},
        )
        return payload
    except Exception:
        return None

# --- DB pool ---
_db_pool: Optional[asyncpg.Pool] = None

async def _get_db() -> Optional[asyncpg.Pool]:
    global _db_pool
    if _db_pool is None and DATABASE_URL:
        _db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5, ssl=False)
    return _db_pool

# --- Rate limiting ---
async def _check_rate_limit(user_id: str) -> bool:
    pool = await _get_db()
    if pool is None:
        return True
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO rate_limits(user_id, date, call_count)
               VALUES($1, CURRENT_DATE, 1)
               ON CONFLICT(user_id, date) DO UPDATE
                 SET call_count = rate_limits.call_count + 1
               RETURNING call_count""",
            user_id,
        )
        return row["call_count"] <= RATE_LIMIT

# --- Tool call logging ---
_current_user: ContextVar[Optional[dict]] = ContextVar("current_user", default=None)

def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)

async def _log_tool_call(user_id: str, tool_name: str, tokens_in: int, tokens_out: int, latency_ms: int) -> None:
    if not user_id:
        return
    pool = await _get_db()
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO tool_calls(user_id, tool_name, tokens_in, tokens_out, latency_ms)
                   VALUES($1, $2, $3, $4, $5)""",
                user_id, tool_name, tokens_in, tokens_out, latency_ms,
            )
    except Exception:
        pass

def tracked_tool(fn):
    """Wraps a sync MCP tool: runs in thread, logs call + estimated token counts."""
    @wraps(fn)
    async def _wrapper(*args, **kwargs):
        start = time.perf_counter()
        user = _current_user.get()
        user_id = user.get("sub", "") if user else ""
        tokens_in = _est_tokens(" ".join(str(v) for v in list(args) + list(kwargs.values())))
        result = await asyncio.to_thread(fn, *args, **kwargs)
        latency = int((time.perf_counter() - start) * 1000)
        tokens_out = _est_tokens(result if isinstance(result, str) else str(result))
        asyncio.create_task(_log_tool_call(user_id, fn.__name__, tokens_in, tokens_out, latency))
        return result
    return mcp.tool()(_wrapper)

# --- Auth middleware ---
SKIP_PATHS = {"/health", "/.well-known/oauth-authorization-server"}

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in SKIP_PATHS:
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse({"error": "unauthorized"}, status_code=401,
                                headers={"WWW-Authenticate": f'Bearer realm="{AUTH_ISSUER}"'})
        payload = await _validate_token(auth[7:])
        if payload is None:
            return JSONResponse({"error": "invalid_token"}, status_code=401)
        user_id = payload.get("sub", "")
        if not await _check_rate_limit(user_id):
            return JSONResponse({"error": "rate_limit_exceeded", "detail": f"limit is {RATE_LIMIT}/day"}, status_code=429)
        request.state.user = payload
        token = _current_user.set(payload)
        try:
            response = await call_next(request)
        finally:
            _current_user.reset(token)
        return response

mcp = FastMCP(
    "ttrpg",
    instructions=(
        "Search TTRPG rules and content. "
        "For Pathfinder 2e: use search_pf2e or get_pf2e_entry. "
        "For Starfinder 2e: use search_sf2e or get_sf2e_entry. "
        "For Draw Steel (MCDM): use search_draw_steel or get_draw_steel_entry. "
        "For D&D 5e (2014 SRD): use search_dnd5e or get_dnd5e_entry. "
        "For D&D 2024 (5.5e SRD): use search_dnd2024 or get_dnd2024_entry. "
        "For Daggerheart (Darrington Press): use search_daggerheart or get_daggerheart_entry. "
        "Use the game-specific tool matching the user's question."
    ),
    host="0.0.0.0",
    port=PORT,
)


@tracked_tool
def search_pf2e(query: str, limit: int = 5, category: str = "") -> str:
    """
    Semantic search over all PF2e content from Archives of Nethys.

    Args:
        query:    Natural language query, e.g. "fire damage area spells level 3"
        limit:    Number of results to return (1-20, default 5)
        category: Optional filter — one of: action, ancestry, archetype, background,
                  class, condition, creature, equipment, feat, hazard, rules,
                  setting_article, skill, spell, trait, weapon (leave empty for all)
    """
    limit = max(1, min(20, limit))

    result = voyage.embed([query], model=VOYAGE_MODEL, input_type="query")
    vector = result.embeddings[0]

    search_filter = None
    if category:
        search_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))]
        )

    hits = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vector,
        query_filter=search_filter,
        limit=limit,
        with_payload=True,
    )

    if not hits:
        return "No results found."

    parts = []
    for i, hit in enumerate(hits, 1):
        p     = hit.payload
        name  = p.get("name", "Unknown").replace("_", " ")
        cat   = p.get("category", "")
        score = f"{hit.score:.3f}"
        chunk = p.get("chunk_index", 0)
        text  = p.get("text", "")

        header = (
            f"### {i}. {name} ({cat}) — score {score}"
            + (f" [chunk {chunk}]" if chunk > 0 else "")
        )
        parts.append(f"{header}\n\n{text}" if text else header)

    return "\n\n".join(parts)


@tracked_tool
def get_pf2e_entry(name: str, category: str = "") -> str:
    """
    Retrieve all chunks for a specific PF2e entry by name.

    Args:
        name:     Exact or partial name, e.g. "Fireball" or "Power Attack"
        category: Optional category filter to narrow results
    """
    query = f"{category} {name}".strip()
    result = voyage.embed([query], model=VOYAGE_MODEL, input_type="query")
    vector = result.embeddings[0]

    search_filter = None
    if category:
        search_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))]
        )

    hits = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vector,
        query_filter=search_filter,
        limit=20,
        with_payload=True,
    )

    name_lower = name.lower().replace(" ", "_")
    matched = [
        h for h in hits
        if name_lower in h.payload.get("name", "").lower()
        or name_lower in h.payload.get("s3_key", "").lower()
    ]

    if not matched:
        return f"No entry found for '{name}'."

    matched.sort(key=lambda h: h.payload.get("chunk_index", 0))

    entry_name = matched[0].payload.get("name", name).replace("_", " ")
    cat        = matched[0].payload.get("category", "")
    s3_key     = matched[0].payload.get("s3_key", "")

    lines = [f"# {entry_name}", f"**Category:** {cat}", f"**Source:** {s3_key}", ""]

    for h in matched:
        chunk_text = h.payload.get("text", "")
        if chunk_text:
            lines.append(chunk_text)
            lines.append("")

    if not any(h.payload.get("text") for h in matched):
        lines.append(f"_{len(matched)} chunk(s) found — text not in index. Re-run embedder to populate._")

    return "\n".join(lines)


@tracked_tool
def list_pf2e_categories() -> str:
    """List all available PF2e content categories in the database."""
    cats: dict[str, int] = {}
    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=["category"],
            with_vectors=False,
        )
        for point in results:
            c = point.payload.get("category", "unknown")
            cats[c] = cats.get(c, 0) + 1
        if offset is None:
            break

    lines = ["**PF2e Categories in database:**", ""]
    for cat, count in sorted(cats.items()):
        lines.append(f"- `{cat}` ({count} chunks)")
    return "\n".join(lines)


@tracked_tool
def search_sf2e(query: str, limit: int = 5, category: str = "") -> str:
    """
    Semantic search over all Starfinder 2e content from Archives of Nethys (2e.aonsrd.com).

    Args:
        query:    Natural language query, e.g. "plasma weapons" or "envoy class features"
        limit:    Number of results to return (1-20, default 5)
        category: Optional filter — one of: action, ancestry, archetype, background,
                  class, condition, creature, equipment, feat, hazard, rules,
                  skill, spell, trait, weapon (leave empty for all)
    """
    limit = max(1, min(20, limit))

    result = voyage.embed([query], model=VOYAGE_MODEL, input_type="query")
    vector = result.embeddings[0]

    search_filter = None
    if category:
        search_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))]
        )

    hits = qdrant.search(
        collection_name=SF2E_COLLECTION,
        query_vector=vector,
        query_filter=search_filter,
        limit=limit,
        with_payload=True,
    )

    if not hits:
        return "No results found."

    parts = []
    for i, hit in enumerate(hits, 1):
        p     = hit.payload
        name  = p.get("name", "Unknown").replace("_", " ")
        cat   = p.get("category", "")
        score = f"{hit.score:.3f}"
        chunk = p.get("chunk_index", 0)
        text  = p.get("text", "")

        header = (
            f"### {i}. {name} ({cat}) — score {score}"
            + (f" [chunk {chunk}]" if chunk > 0 else "")
        )
        parts.append(f"{header}\n\n{text}" if text else header)

    return "\n\n".join(parts)


@tracked_tool
def get_sf2e_entry(name: str, category: str = "") -> str:
    """
    Retrieve all chunks for a specific Starfinder 2e entry by name.

    Args:
        name:     Exact or partial name, e.g. "Envoy" or "Shock Grenade"
        category: Optional category filter to narrow results
    """
    query = f"{category} {name}".strip()
    result = voyage.embed([query], model=VOYAGE_MODEL, input_type="query")
    vector = result.embeddings[0]

    search_filter = None
    if category:
        search_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))]
        )

    hits = qdrant.search(
        collection_name=SF2E_COLLECTION,
        query_vector=vector,
        query_filter=search_filter,
        limit=20,
        with_payload=True,
    )

    name_lower = name.lower().replace(" ", "_")
    matched = [
        h for h in hits
        if name_lower in h.payload.get("name", "").lower()
        or name_lower in h.payload.get("s3_key", "").lower()
    ]

    if not matched:
        return f"No entry found for '{name}'."

    matched.sort(key=lambda h: h.payload.get("chunk_index", 0))

    entry_name = matched[0].payload.get("name", name).replace("_", " ")
    cat        = matched[0].payload.get("category", "")
    s3_key     = matched[0].payload.get("s3_key", "")

    lines = [f"# {entry_name}", f"**Category:** {cat}", f"**Source:** {s3_key}", ""]

    for h in matched:
        chunk_text = h.payload.get("text", "")
        if chunk_text:
            lines.append(chunk_text)
            lines.append("")

    if not any(h.payload.get("text") for h in matched):
        lines.append(f"_{len(matched)} chunk(s) found — text not in index. Re-run embedder to populate._")

    return "\n".join(lines)


@tracked_tool
def list_sf2e_categories() -> str:
    """List all available Starfinder 2e content categories in the database."""
    cats: dict[str, int] = {}
    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=SF2E_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=["category"],
            with_vectors=False,
        )
        for point in results:
            c = point.payload.get("category", "unknown")
            cats[c] = cats.get(c, 0) + 1
        if offset is None:
            break

    lines = ["**SF2e Categories in database:**", ""]
    for cat, count in sorted(cats.items()):
        lines.append(f"- `{cat}` ({count} chunks)")
    return "\n".join(lines)


@tracked_tool
def search_draw_steel(query: str, limit: int = 5, category: str = "") -> str:
    """
    Semantic search over Draw Steel (MCDM) rules, abilities, monsters, and more.

    Args:
        query:    Natural language query, e.g. "fury rage abilities" or "demon statblock"
        limit:    Number of results to return (1-20, default 5)
        category: Optional filter — one of: rules, bestiary, adventures (leave empty for all)
    """
    limit = max(1, min(20, limit))

    result = voyage.embed([query], model=VOYAGE_MODEL, input_type="query")
    vector = result.embeddings[0]

    search_filter = None
    if category:
        search_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))]
        )

    hits = qdrant.search(
        collection_name=DS_COLLECTION,
        query_vector=vector,
        query_filter=search_filter,
        limit=limit,
        with_payload=True,
    )

    if not hits:
        return "No results found."

    parts = []
    for i, hit in enumerate(hits, 1):
        p     = hit.payload
        name  = p.get("name", "Unknown").replace("_", " ")
        cat   = p.get("category", "")
        score = f"{hit.score:.3f}"
        chunk = p.get("chunk_index", 0)
        text  = p.get("text", "")

        header = (
            f"### {i}. {name} ({cat}) — score {score}"
            + (f" [chunk {chunk}]" if chunk > 0 else "")
        )
        parts.append(f"{header}\n\n{text}" if text else header)

    return "\n\n".join(parts)


@tracked_tool
def get_draw_steel_entry(name: str, category: str = "") -> str:
    """
    Retrieve all chunks for a specific Draw Steel entry by name.

    Args:
        name:     Exact or partial name, e.g. "Fury" or "Free Strike"
        category: Optional category filter — rules, bestiary, or adventures
    """
    query = f"{category} {name}".strip()
    result = voyage.embed([query], model=VOYAGE_MODEL, input_type="query")
    vector = result.embeddings[0]

    search_filter = None
    if category:
        search_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))]
        )

    hits = qdrant.search(
        collection_name=DS_COLLECTION,
        query_vector=vector,
        query_filter=search_filter,
        limit=20,
        with_payload=True,
    )

    name_lower = name.lower().replace(" ", "_")
    matched = [
        h for h in hits
        if name_lower in h.payload.get("name", "").lower()
        or name_lower in h.payload.get("s3_key", "").lower()
    ]

    if not matched:
        return f"No entry found for '{name}'."

    matched.sort(key=lambda h: h.payload.get("chunk_index", 0))

    entry_name = matched[0].payload.get("name", name).replace("_", " ")
    cat        = matched[0].payload.get("category", "")
    s3_key     = matched[0].payload.get("s3_key", "")

    lines = [f"# {entry_name}", f"**Category:** {cat}", f"**Source:** {s3_key}", ""]

    for h in matched:
        chunk_text = h.payload.get("text", "")
        if chunk_text:
            lines.append(chunk_text)
            lines.append("")

    if not any(h.payload.get("text") for h in matched):
        lines.append(f"_{len(matched)} chunk(s) found — text not in index._")

    return "\n".join(lines)


def _search_collection(collection: str, query: str, limit: int, category: str) -> str:
    limit = max(1, min(20, limit))
    result = voyage.embed([query], model=VOYAGE_MODEL, input_type="query")
    vector = result.embeddings[0]
    search_filter = None
    if category:
        search_filter = Filter(must=[FieldCondition(key="category", match=MatchValue(value=category))])
    hits = qdrant.search(collection_name=collection, query_vector=vector,
                         query_filter=search_filter, limit=limit, with_payload=True)
    if not hits:
        return "No results found."
    parts = []
    for i, hit in enumerate(hits, 1):
        p = hit.payload
        name  = p.get("name", "Unknown").replace("_", " ")
        cat   = p.get("category", "")
        score = f"{hit.score:.3f}"
        chunk = p.get("chunk_index", 0)
        text  = p.get("text", "")
        header = f"### {i}. {name} ({cat}) — score {score}" + (f" [chunk {chunk}]" if chunk > 0 else "")
        parts.append(f"{header}\n\n{text}" if text else header)
    return "\n\n".join(parts)


def _get_entry_collection(collection: str, name: str, category: str) -> str:
    query = f"{category} {name}".strip()
    result = voyage.embed([query], model=VOYAGE_MODEL, input_type="query")
    vector = result.embeddings[0]
    search_filter = None
    if category:
        search_filter = Filter(must=[FieldCondition(key="category", match=MatchValue(value=category))])
    hits = qdrant.search(collection_name=collection, query_vector=vector,
                         query_filter=search_filter, limit=20, with_payload=True)
    name_lower = name.lower().replace(" ", "_")
    matched = [h for h in hits if name_lower in h.payload.get("name", "").lower()
               or name_lower in h.payload.get("s3_key", "").lower()]
    if not matched:
        return f"No entry found for '{name}'."
    matched.sort(key=lambda h: h.payload.get("chunk_index", 0))
    entry_name = matched[0].payload.get("name", name).replace("_", " ")
    cat    = matched[0].payload.get("category", "")
    s3_key = matched[0].payload.get("s3_key", "")
    lines = [f"# {entry_name}", f"**Category:** {cat}", f"**Source:** {s3_key}", ""]
    for h in matched:
        chunk_text = h.payload.get("text", "")
        if chunk_text:
            lines.append(chunk_text)
            lines.append("")
    return "\n".join(lines)


@tracked_tool
def search_dnd5e(query: str, limit: int = 5, category: str = "") -> str:
    """
    Semantic search over D&D 5e SRD (2014) content.

    Args:
        query:    Natural language query, e.g. "fire damage spells level 3" or "fighter class features"
        limit:    Number of results to return (1-20, default 5)
        category: Optional filter — one of: spell, monster, class, subclass, equipment,
                  magic-item, feat, race, subrace, background, condition, rule (leave empty for all)
    """
    return _search_collection(DND5E_COLLECTION, query, limit, category)


@tracked_tool
def get_dnd5e_entry(name: str, category: str = "") -> str:
    """
    Retrieve a specific D&D 5e SRD entry by name.

    Args:
        name:     Exact or partial name, e.g. "Fireball" or "Fighter"
        category: Optional category filter
    """
    return _get_entry_collection(DND5E_COLLECTION, name, category)


@tracked_tool
def search_dnd2024(query: str, limit: int = 5, category: str = "") -> str:
    """
    Semantic search over D&D 2024 SRD (5.5e) content.

    Args:
        query:    Natural language query, e.g. "bardic inspiration mechanics" or "weapon mastery"
        limit:    Number of results to return (1-20, default 5)
        category: Optional filter — one of: spell, monster, class, equipment, magic-item,
                  feat, race, background, condition, rule (leave empty for all)
    """
    return _search_collection(DND2024_COLLECTION, query, limit, category)


@tracked_tool
def get_dnd2024_entry(name: str, category: str = "") -> str:
    """
    Retrieve a specific D&D 2024 SRD entry by name.

    Args:
        name:     Exact or partial name, e.g. "Fireball" or "Paladin"
        category: Optional category filter
    """
    return _get_entry_collection(DND2024_COLLECTION, name, category)


@tracked_tool
def search_daggerheart(query: str, limit: int = 5, category: str = "") -> str:
    """
    Semantic search over Daggerheart (Darrington Press) rules and content.

    Args:
        query:    Natural language query, e.g. "bard class abilities" or "adversary types"
        limit:    Number of results to return (1-20, default 5)
        category: Optional filter — one of: class, ancestry, community, domain, ability,
                  environment, adversary, item, rule (leave empty for all)
    """
    return _search_collection(DH_COLLECTION, query, limit, category)


@tracked_tool
def get_daggerheart_entry(name: str, category: str = "") -> str:
    """
    Retrieve a specific Daggerheart entry by name.

    Args:
        name:     Exact or partial name, e.g. "Bard" or "Seraph"
        category: Optional category filter
    """
    return _get_entry_collection(DH_COLLECTION, name, category)


@mcp.custom_route("/health", methods=["GET"])
async def health(request):
    return JSONResponse({"status": "ok"})


@mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
async def oauth_discovery(request):
    return JSONResponse({
        "issuer": AUTH_ISSUER,
        "authorization_endpoint": f"{AUTH_ISSUER}/oauth/authorize",
        "token_endpoint": f"{AUTH_ISSUER}/oauth/token",
        "registration_endpoint": f"{AUTH_ISSUER}/oauth/register",
        "jwks_uri": f"{AUTH_ISSUER}/.well-known/jwks.json",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
    })


if __name__ == "__main__":
    import uvicorn
    app = mcp.streamable_http_app()
    app.add_middleware(JWTAuthMiddleware)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
