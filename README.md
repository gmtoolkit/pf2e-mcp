# TTRPG MCP

Semantic search over **Pathfinder 2e**, **Starfinder 2e**, and **Draw Steel** rules — served as an [MCP](https://modelcontextprotocol.io) server for Claude.

## What it does

Ask rules questions in plain English inside Claude and get accurate, cited answers sourced from Archives of Nethys (PF2e, SF2e) and the Draw Steel rules.

**Tools provided:**
- `search_pf2e` / `get_pf2e_entry` — Pathfinder 2e content
- `search_sf2e` / `get_sf2e_entry` — Starfinder 2e content
- `search_draw_steel` / `get_draw_steel_entry` — Draw Steel (MCDM) content

## Usage

### 1. Sign in at [gmkit.io](https://gmkit.io)

Authenticate with GitHub or Google. Free, 100 searches/day.

### 2. Add to Claude Code

```json
{
  "mcpServers": {
    "ttrpg": {
      "type": "http",
      "url": "https://mcp.gmkit.io/mcp/"
    }
  }
}
```

Add this to `~/.claude/mcp.json`. Claude Code will open a browser to authenticate on first use.

### 3. Ask away

```
"What does the Paladin's Retributive Strike do in PF2e?"
"Compare champion and censor across PF2e, SF2e, Draw Steel"
"What armor can a Kineticist wear?"
```

## Stack

- Python + [FastMCP](https://github.com/jlowin/fastmcp) — MCP server
- [Voyage AI](https://voyageai.com) — embeddings
- [Qdrant](https://qdrant.tech) — vector search
- OAuth 2.1 + JWT auth via [gmkit.io](https://gmkit.io)
- Deployed on [fly.io](https://fly.io)

## Support

If this saves you time at the table, consider buying me a coffee:

[![Buy me a coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-☕-FFDD00?style=for-the-badge&labelColor=000000)](https://coff.ee/gmkit)

**[coff.ee/gmkit](https://coff.ee/gmkit)**

---

## Legal

TTRPG MCP is an independent product published under the DRAW STEEL Creator License and is not affiliated with MCDM Productions, LLC. DRAW STEEL © 2024 MCDM Productions, LLC.

Pathfinder and Starfinder are trademarks of Paizo Inc. Content used under the [ORC License](https://paizo.com/licenses/orc).
