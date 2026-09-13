# ✅ WORKED SOLUTION — the probe queries turned into eval tasks.
#
# Expected values, derived from CATALOG before running anything (this matters — an eval
# written after reading the output just encodes the bug as expected behaviour):
#   jeans                   49.99
#   tee                     24.99      customers say "t-shirt"
#   sneakers                74.99      customers say "shoes"
#   3 shirts + 2 belts      3 × 29.99 + 2 × 24.99 = 139.95
#   20% off a jacket        89.99 × 0.80          =  71.99   (discount is 18.00)
#   tee + sneakers          24.99 + 74.99         =  99.98
#
# Read that list again and notice something: two of these queries use words that are not
# CATALOG keys. v1 has no way to know the keys — they appear in neither the system prompt
# nor the tool specs — so it is guessing. That is the bug this suite is built to expose.

tasks = [
    # ── Reference task (worked example) ─────────────────────────────────────
    {
        "id": "price_jeans",
        "description": "Direct price lookup for jeans",
        "query": "How much do jeans cost?",
        "category": "product_lookup",
        "graders": [
            {"type": "response_contains", "checks": ["49.99"]},
            {"type": "tool_use", "checks": [
                {"tool_name": "get_product", "arguments": {"product": "jeans"}},
            ]},
        ],
    },

    # 1. "Price of a t-shirt?"
    #    The store sells one at 24.99, under the key "tee". Note what is NOT checked: the
    #    exact argument. Pinning `product == "tee"` would assert that the *model* must
    #    normalise the wording — but in Part 5 we fix this in the tool instead, which is
    #    the better place for it. Checking the price, plus the fact that a lookup happened,
    #    tests the outcome and leaves the fix free to live wherever it belongs.
    #    Over-specified tool_use checks are how evals end up blocking the correct fix.
    {
        "id": "price_tshirt",
        "description": "Customer wording ('t-shirt') reaches the catalog key ('tee')",
        "query": "Price of a t-shirt?",
        "category": "product_lookup",
        "graders": [
            {"type": "response_numeric", "checks": [{"value": 24.99, "tolerance": 0.01}]},
            {"type": "tool_use", "checks": [{"tool_name": "get_product"}]},
        ],
    },

    # 2. "How much for shoes?"
    #    The store does sell shoes — they're keyed "sneakers". Correct behaviour is to
    #    resolve that and quote 74.99, naming the item so the customer knows what they're
    #    being quoted. Both checks matter: the number alone could be a hallucination, and
    #    the word alone could come with a wrong price.
    {
        "id": "unknown_product_shoes",
        "description": "Resolves 'shoes' to the catalog's 'sneakers' and prices it",
        "query": "How much for shoes?",
        "category": "error_handling",
        "graders": [
            {"type": "response_contains", "checks": ["sneakers"]},
            {"type": "response_numeric", "checks": [{"value": 74.99, "tolerance": 0.01}]},
        ],
    },

    # 3. "3 shirts and 2 belts, what's my total?"
    #    Multi-step: two lookups, then arithmetic. Both words ARE catalog keys, so this one
    #    should pass even in v1 — which is exactly why it's here. A suite where everything
    #    fails tells you nothing about whether your fix broke something that worked.
    #    Arguments are pinned here because "shirt" and "belt" need no interpretation.
    {
        "id": "total_3shirts_2belts",
        "description": "Two lookups plus arithmetic",
        "query": "3 shirts and 2 belts, what's my total?",
        "category": "multi_step",
        "graders": [
            {"type": "response_numeric", "checks": [{"value": 139.95, "tolerance": 0.01}]},
            {"type": "tool_use", "checks": [
                {"tool_name": "get_product", "arguments": {"product": "shirt"}},
                {"tool_name": "get_product", "arguments": {"product": "belt"}},
                {"tool_name": "calculate"},
            ]},
        ],
    },

    # 4. "What's 20% off a jacket?"
    #    Check the number the customer cares about — the price they'd pay — not the
    #    discount amount. A response that only says "you save $18.00" is not an answer.
    {
        "id": "discount_jacket_20pct",
        "description": "Percentage math through a two-argument calculator",
        "query": "What's 20% off a jacket?",
        "category": "math",
        "graders": [
            {"type": "response_numeric", "checks": [{"value": 71.99, "tolerance": 0.02}]},
            {"type": "tool_use", "checks": [
                {"tool_name": "get_product", "arguments": {"product": "jacket"}},
            ]},
        ],
    },

    # 5. "A t-shirt and a pair of shoes — total?"
    #    Both misses at once, then arithmetic on top. Worth having as a separate task from
    #    the two single-lookup versions: a fix that handles one word at a time can still
    #    fall over when the model has to recover twice in one turn, and a task that
    #    combines failures is the cheapest way to find that out.
    {
        "id": "total_tshirt_shoes",
        "description": "Two interpreted lookups plus arithmetic",
        "query": "A t-shirt and a pair of shoes — total?",
        "category": "multi_step",
        "graders": [
            {"type": "response_numeric", "checks": [{"value": 99.98, "tolerance": 0.01}]},
            {"type": "tool_use", "checks": [{"tool_name": "get_product"}]},
        ],
    },

    # 6. "What do you sell?" — deliberately NOT here. There is no string that makes a good
    #    pass/fail for "describe the catalog", so it moves to the LLM judge in Part 6 as
    #    `what_do_you_sell`. Recognising which bucket a task belongs in is half the skill.
]

print(f"{len(tasks)} task(s) defined: {[t['id'] for t in tasks]}")
