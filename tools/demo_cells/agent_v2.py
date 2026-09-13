# ✅ WORKED SOLUTION — boutique v2. v1 is left untouched so we can compare.
#
# The single most important change is the least glamorous one: **the catalog is now
# visible to the model**, in the tool description, built from CATALOG so it can't go
# stale. v1 was guessing product keys. No amount of prompt polish fixes that, because it
# isn't a reasoning failure — the information simply wasn't there.
#
# Note that the *tool surface is unchanged* — same two tools, same arguments. Nothing here
# adds a capability; it only makes the existing ones legible. That keeps the before/after
# comparison honest.

# 1. A system prompt that actually says what the job is, and what not to do.
#    "Never state a price from memory" is the line that stops hallucinated prices.
SYSTEM_PROMPT_V2 = """You are a shopping assistant for a boutique clothing store.

- Always call get_product to look up a price. Never state a price from memory.
- Use calculate for every arithmetic step. Chain calls for multi-step math.
- If get_product tells you it interpreted the customer's wording as a different catalog
  item, say which item you priced so the customer isn't surprised.
- If something isn't in the catalog, say so plainly and suggest the closest item we do
  sell. Never invent a price for a product that doesn't exist.
- Give the final amount in dollars, and show the arithmetic briefly."""


# 2. Tools that fail *usefully*. A tool's error string is a prompt — the model reads it
#    and decides what to do next. `KeyError: 'shoes'` tells it nothing; a message that
#    names the alternatives tells it exactly how to recover.
#
#    Note the synonym table rather than fuzzy string matching. difflib would score
#    "shoes" closer to "socks" than to "sneakers" — an intentional mapping beats a
#    plausible-looking guess every time.
SYNONYMS = {
    "shoes": "sneakers", "sneaker": "sneakers", "trainers": "sneakers",
    "t-shirt": "tee", "tshirt": "tee", "t shirt": "tee", "tee shirt": "tee",
    "pants": "jeans", "trousers": "jeans", "denim": "jeans",
    "jumper": "sweater", "pullover": "sweater", "sweatshirt": "hoodie",
    "cap": "hat", "beanie": "hat", "frock": "dress",
}


def _resolve_product(raw: str):
    """Map customer wording onto a catalog key. Returns (key, matched_exactly)."""
    name = (raw or "").strip().lower()
    if name in CATALOG:
        return name, True
    if name in SYNONYMS:
        return SYNONYMS[name], False
    if name.endswith("s"):                            # "shirts" -> "shirt"
        singular = name[:-1]
        if singular in CATALOG:
            return singular, False
        if singular in SYNONYMS:                      # "t-shirts" -> "t-shirt" -> "tee"
            return SYNONYMS[singular], False
    return None, False


def get_product_v2(product: str):
    key, exact = _resolve_product(product)
    if key is None:
        raise ValueError(
            f"No catalog item matches '{product}'. "
            f"The catalog is: {', '.join(sorted(CATALOG))}. "
            f"Tell the customer we don't carry it and suggest the closest item we do."
        )
    if exact:
        return CATALOG[key]
    # Surface the resolution so the assistant can be transparent about what it priced.
    return f"{CATALOG[key]} (interpreted '{product}' as catalog item '{key}')"


LEGAL_OPS = ["+", "-", "*", "/", "**"]


def calculate_v2(op: str, input1: float, input2: float):
    if op not in LEGAL_OPS:
        # v1 fell through every branch and returned None here — a wrong answer that
        # looks like a working tool call. Failing loudly is strictly better.
        raise ValueError(f"Unsupported op '{op}'. Supported: {' '.join(LEGAL_OPS)}. "
                         f"For a percentage, multiply by the decimal (20% -> * 0.2).")
    if op == "/" and input2 == 0:
        raise ValueError("Division by zero.")
    return {"+": input1 + input2, "-": input1 - input2, "*": input1 * input2,
            "/": input1 / input2 if input2 else None,
            "**": input1 ** input2}[op]


# 3. Specs that describe reality. The `enum` on `op` is the single highest-value line
#    in this cell: it makes an illegal operator *unrepresentable* rather than something
#    you hope the model avoids. Constrain at the schema level before you prompt for it.
GET_PRODUCT_SPEC_V2 = {
    "name": "get_product",
    "description": (
        "Look up the current price of one item from the store catalog. Returns the price "
        "in USD. The full catalog is: " + ", ".join(sorted(CATALOG)) + ". "
        "Common synonyms are accepted (e.g. 'shoes' resolves to 'sneakers'); when that "
        "happens the result names the catalog item that was actually priced. Raises an "
        "error listing the catalog if nothing matches — never guess a price instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "product": {
                "type": "string",
                "description": "Singular product name as the customer said it, e.g. "
                               "'jeans', 'shirt', 't-shirt', 'shoes'.",
            },
        },
        "required": ["product"],
    },
}

CALCULATE_SPEC_V2 = {
    "name": "calculate",
    "description": (
        "Apply one arithmetic operation to two numbers and return the result. Handles a "
        "single step only — chain several calls for multi-step math (e.g. 20% off a "
        "$50 item: multiply 50 by 0.2, then subtract that from 50). There is no percent "
        "operator: convert percentages to decimals yourself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": LEGAL_OPS,
                "description": "The operation: + - * / or ** (exponent).",
            },
            "input1": {"type": "number", "description": "Left-hand operand."},
            "input2": {"type": "number", "description": "Right-hand operand."},
        },
        "required": ["op", "input1", "input2"],
    },
}


# ── Wiring (unchanged from the workbook) ───────────────────────────────────
TOOL_REGISTRY_V2 = {"get_product": get_product_v2, "calculate": calculate_v2}
ALL_TOOL_SPECS_V2 = [GET_PRODUCT_SPEC_V2, CALCULATE_SPEC_V2]


def execute_tool_v2(name, inputs):
    try:
        return str(TOOL_REGISTRY_V2[name](**inputs))
    except Exception as e:
        return f"Error: {e}"


def run_agent_v2(prompt, eval_mode=False, model=None):
    """Same loop as run_agent, pointed at the v2 prompt, specs and tools."""
    messages = [{"role": "user", "content": prompt}]
    total_input_tokens = total_output_tokens = 0

    while True:
        response = client.messages.create(
            model=model or AGENT_MODEL,
            system=SYSTEM_PROMPT_V2,
            max_tokens=1024,
            tools=ALL_TOOL_SPECS_V2,
            messages=messages,
        )
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        tool_results = [
            {"type": "tool_result", "tool_use_id": block.id,
             "content": execute_tool_v2(block.name, block.input)}
            for block in response.content if isinstance(block, ToolUseBlock)
        ]
        messages.append({"role": "user", "content": tool_results})

    if eval_mode:
        return {"messages": messages,
                "usage": {"input_tokens": total_input_tokens,
                          "output_tokens": total_output_tokens}}
    return "\n".join(b.text for b in response.content if isinstance(b, TextBlock))


print("boutique v2 ready.")
