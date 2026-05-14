#!/usr/bin/env python3
"""
PF2e MCP server — semantic search over Archives of Nethys data stored in Qdrant.

Env vars:
  VOYAGE_API_KEY    - Voyage AI key
  QDRANT_URL        - (default: http://qdrant.infra.svc.cluster.local:6333)
  QDRANT_COLLECTION - (default: pf2e)
  VOYAGE_MODEL      - (default: voyage-3)
  PORT              - HTTP port (default: 8000)
"""
import os

import voyageai
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

VOYAGE_API_KEY    = os.environ["VOYAGE_API_KEY"]
VOYAGE_MODEL      = os.environ.get("VOYAGE_MODEL", "voyage-3")
QDRANT_URL        = os.environ.get("QDRANT_URL", "http://qdrant.infra.svc.cluster.local:6333")
QDRANT_API_KEY    = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "pf2e")

voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

PORT = int(os.environ.get("PORT", "8000"))

mcp = FastMCP(
    "pf2e",
    instructions=(
        "Search Pathfinder 2e rules, spells, feats, creatures, items, and more "
        "from the Archives of Nethys. Use search_pf2e for semantic queries. "
        "Optionally filter by category."
    ),
    host="0.0.0.0",
    port=PORT,
)


@mcp.tool()
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


@mcp.tool()
def get_pf2e_entry(name: str, category: str = "") -> str:
    """
    Retrieve all chunks for a specific PF2e entry by name.

    Args:
        name:     Exact or partial name, e.g. "Fireball" or "Power Attack"
        category: Optional category filter to narrow results
    """
    # Search by name using semantic query — name match tends to score very high
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

    # Filter to hits where name roughly matches
    name_lower = name.lower().replace(" ", "_")
    matched = [
        h for h in hits
        if name_lower in h.payload.get("name", "").lower()
        or name_lower in h.payload.get("s3_key", "").lower()
    ]

    if not matched:
        return f"No entry found for '{name}'."

    # Sort by chunk_index to reassemble order
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


@mcp.tool()
def list_pf2e_categories() -> str:
    """List all available PF2e content categories in the database."""
    # Scroll a sample and collect unique categories
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


@mcp.custom_route("/health", methods=["GET"])
async def health(request):
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
