"""Mock backend + tool registry for the toy support/commerce agent.

Three tools over a fake in-memory store. Each tool takes keyword args and
returns a string observation. The TOOLS registry also records each tool's
expected args, so on Day 2 your deterministic evaluator can check that the
agent called a real tool with sane arguments.

Support/commerce was chosen deliberately: it maps directly onto the agents
Razorpay and Swiggy run, which makes the cold-email framing land.
"""
from __future__ import annotations

from typing import Any, Callable

# --- fake in-memory data ---------------------------------------------------
_ORDERS = {
    "ORD1001": {"item": "Wireless Earbuds", "status": "Delivered", "amount": 2499, "date": "2026-05-30"},
    "ORD1002": {"item": "Yoga Mat", "status": "In Transit", "amount": 899, "date": "2026-06-08"},
    "ORD1003": {"item": "Coffee Grinder", "status": "Cancelled", "amount": 3199, "date": "2026-06-01"},
}

_REFUNDS = {
    "ORD1003": {"state": "Refund Initiated", "expected_days": 5, "amount": 3199},
}

_FAQS = {
    "return policy": "Items can be returned within 7 days of delivery if unused and in original packaging.",
    "refund time": "Refunds are credited to the original payment method within 5-7 business days.",
    "cancel order": "Orders can be cancelled for free any time before they are shipped.",
    "track order": "Use the order ID on the Orders page to see live tracking status.",
}


# --- tool implementations --------------------------------------------------
def order_lookup(order_id: str = "") -> str:
    o = _ORDERS.get(order_id.strip().upper())
    if not o:
        return f"No order found with id '{order_id}'."
    return (
        f"Order {order_id.upper()}: {o['item']}, status={o['status']}, "
        f"amount=Rs{o['amount']}, ordered on {o['date']}."
    )


def refund_status(order_id: str = "") -> str:
    r = _REFUNDS.get(order_id.strip().upper())
    if not r:
        return f"No active refund found for order '{order_id}'."
    return (
        f"Refund for {order_id.upper()}: {r['state']}, amount=Rs{r['amount']}, "
        f"expected in {r['expected_days']} business days."
    )


def faq_search(query: str = "") -> str:
    q = query.lower()
    for key, ans in _FAQS.items():
        if any(word in q for word in key.split()):
            return ans
    return "No FAQ matched that query."


# --- registry --------------------------------------------------------------
TOOLS: dict[str, dict[str, Any]] = {
    "order_lookup": {
        "func": order_lookup,
        "description": "Look up an order's status and details by its order_id.",
        "args": {"order_id": "string, e.g. ORD1001"},
    },
    "refund_status": {
        "func": refund_status,
        "description": "Check the refund state for a given order_id.",
        "args": {"order_id": "string, e.g. ORD1003"},
    },
    "faq_search": {
        "func": faq_search,
        "description": "Search the help-center FAQ with a natural-language query.",
        "args": {"query": "string, the user's question in plain words"},
    },
}


def run_tool(tool_name: str, args: dict[str, Any]) -> str:
    """Execute a tool by name. Returns an observation string (never raises)."""
    if tool_name not in TOOLS:
        return f"ERROR: unknown tool '{tool_name}'."
    func: Callable = TOOLS[tool_name]["func"]
    try:
        return func(**args)
    except TypeError as e:
        return f"ERROR: bad arguments for {tool_name}: {e}"


def tools_description() -> str:
    """Human-readable tool list injected into the agent's system prompt."""
    lines = []
    for name, spec in TOOLS.items():
        args = ", ".join(f"{k} ({v})" for k, v in spec["args"].items())
        lines.append(f"- {name}: {spec['description']} Args: {args}")
    return "\n".join(lines)
