"""Sample tasks for the toy agent.

`expected` holds gold info you'll use on Day 4 to measure whether the agent
(and then your evaluator) got it right:
  - tools:           which tool(s) a correct run should call
  - answer_contains: a substring a correct final answer should contain

The set is deliberately varied: single-tool, multi-tool, FAQ, and an
unknown-order case. Variety means your traces will contain a range of
behaviours for the evaluator to grade.
"""

TASKS = [
    {
        "id": "t01_order_status",
        "goal": "What's the status of my order ORD1002?",
        "expected": {"tools": ["order_lookup"], "answer_contains": "In Transit"},
    },
    {
        "id": "t02_refund",
        "goal": "I cancelled order ORD1003. When will I get my refund?",
        "expected": {"tools": ["refund_status"], "answer_contains": "5"},
    },
    {
        "id": "t03_policy",
        "goal": "What is your return policy?",
        "expected": {"tools": ["faq_search"], "answer_contains": "7 days"},
    },
    {
        "id": "t04_multi",
        "goal": "Did order ORD1003 ship, and if it was cancelled what's the refund status?",
        "expected": {"tools": ["order_lookup", "refund_status"], "answer_contains": "Refund"},
    },
    {
        "id": "t05_unknown",
        "goal": "What's the status of order ORD9999?",
        "expected": {"tools": ["order_lookup"], "answer_contains": "No order"},
    },
]
