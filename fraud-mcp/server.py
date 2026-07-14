import asyncio
# pyrefly: ignore [missing-import]
from mcp.server.fastapi import FastServer
# pyrefly: ignore [missing-import]
from mcp.shared.exceptions import McpError
# pyrefly: ignore [missing-import]
from mcp.types import Tool, Resource, TextContent, INVALID_PARAMS

# Initialize FastServer (or standard Server depending on your transport layer)
# For simplicity with Claude Desktop locally, we can use the StdioServerTransport,
# but since we are tunneling via ngrok for a remote client, we will use SSE/HTTP.
# pyrefly: ignore [missing-import]
from mcp.server import Server
# pyrefly: ignore [missing-import]
from mcp.server.transport.types import ServerTransport
# pyrefly: ignore [missing-import]
import mcp.types as types

app = Server("fraud-mcp-server")

# --- STRETCH GOAL: Resource (Model Card) ---
@app.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="model://fraud-detection/card",
            name="Fraud Detection Model Card",
            description="Metadata, accuracy metrics, and trust scores for the active fraud models.",
            mimeType="text/markdown"
        )
    ]

@app.read_resource()
async def handle_read_resource(uri: str) -> str:
    if uri == "model://fraud-detection/card":
        return """# Fraud Detection Model Card
## Active Setups
1. **Ensemble-XG**: Trust Score: 0.94 (Best for high-volume card transactions)
2. **Neural-Shield v2**: Trust Score: 0.89 (Best for cross-border wire transfers)

*Recommendation: Trust Ensemble-XG most for standard retail transactions.*"""
    raise ValueError("Resource not found")

# --- MISSION: Tools ---
@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="score_transaction",
            description="Calculate the fraud risk score for a specific transaction.",
            inputSchema={
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "currency": {"type": "string"},
                    "location": {"type": "string"}
                },
                "required": ["amount", "currency"]
            }
        ),
        types.Tool(
            name="get_leaderboard",
            description="Get the performance leaderboard of active model setups.",
            inputSchema={"type": "object"}
        ),
        types.Tool(
            name="get_recent_stats",
            description="Get real-time statistics of processed transactions.",
            inputSchema={"type": "object"}
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "score_transaction":
        amount = arguments.get("amount", 0)
        # Mock logic
        risk_score = 0.85 if amount > 5000 else 0.12
        return [types.TextContent(type="text", text=f"Transaction Risk Score: {risk_score} (Flagged: {risk_score > 0.5})")]
        
    elif name == "get_leaderboard":
        return [types.TextContent(type="text", text="1. Ensemble-XG (94% Acc)\n2. Neural-Shield v2 (89% Acc)")]
        
    elif name == "get_recent_stats":
        return [types.TextContent(type="text", text="Total Processed (Last 1hr): 1,420 | Blocked: 14 | False Positives: 1")]
        
    else:
        raise McpError(INVALID_PARAMS, f"Unknown tool: {name}")

if __name__ == "__main__":
    # Run using stdio transport for local desktop connection
    # pyrefly: ignore [missing-import]
    from mcp.server.stdio import start_stdio_server
    asyncio.run(start_stdio_server(app))