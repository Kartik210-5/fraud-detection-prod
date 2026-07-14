from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fraud-mcp-server")

@mcp.resource("model://fraud-detection/card")
def model_card() -> str:
    return """
# Fraud Detection Model Card

## Active Setups

1. **Ensemble-XG**
   - Trust Score: 0.94
   - Best for high-volume card transactions

2. **Neural-Shield v2**
   - Trust Score: 0.89
   - Best for cross-border wire transfers

Recommendation:
Trust Ensemble-XG most for standard retail transactions.
"""

@mcp.tool()
def score_transaction(
    amount: float,
    currency: str,
    location: str = "Unknown",
) -> str:
    """
    Calculate the fraud risk score for a transaction.
    """

    risk_score = 0.85 if amount > 5000 else 0.12

    return (
        f"Transaction Risk Score: {risk_score:.2f}\n"
        f"Flagged: {risk_score > 0.5}\n"
        f"Currency: {currency}\n"
        f"Location: {location}"
    )


@mcp.tool()
def get_leaderboard() -> str:
    """
    Get active model leaderboard.
    """

    return """
1. Ensemble-XG (94% Accuracy)
2. Neural-Shield v2 (89% Accuracy)
"""


@mcp.tool()
def get_recent_stats() -> str:
    """
    Return recent transaction statistics.
    """

    return (
        "Total Processed (Last Hour): 1,420\n"
        "Blocked: 14\n"
        "False Positives: 1"
    )


if __name__ == "__main__":
    mcp.run()