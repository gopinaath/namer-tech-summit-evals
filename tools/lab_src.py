# Percent-format source for Bigger_Model_or_Better_Agent.ipynb.
# Rebuild with:  python tools/nbbuild.py

# %% [markdown]
# # Bigger Model, or Better Agent?
#
# ### NAMER Tech Summit · companion lab · Amazon Bedrock
#
# Every model-selection argument comes down to one question:
#
# > Our agent gets things wrong. Do we move it to a stronger model, or do we fix it?
#
# Both work, sometimes. They cost wildly different amounts. This notebook settles it by
# measurement rather than argument, on an agent built to fail in three realistic ways.
#
# You will measure two ladders on the same eval suite:
#
# | Ladder | What varies | The question it answers |
# |---|---|---|
# | **Capability** | Haiku 4.5 → Sonnet 5 → Fable 5.1 → Opus 5, agent untouched | What does buying a bigger model actually get me? |
# | **Engineering** | Haiku 4.5 throughout, agent tuned in four steps | What does fixing the agent get me, on the cheapest model? |
#
# Then it puts the two answers side by side — the tuned small model against the untuned
# large one — on accuracy *and* tokens *and* latency.
#
# > **This is a companion to `Building_an_Eval.ipynb`.** That workshop teaches you to build
# > an eval. This one is a worked experiment you *run*, not fill in — every cell is
# > complete. It reuses the same setup cell, so if the workshop ran, this will.
#
# **Budget:** ~190 agent runs, and all but 21 of them are on Haiku. A few dollars. The
# `RUNS`, `LADDER_MODELS` and `CONFIG_LADDER` knobs further down cut it right back.

# %% include=setup
# Kernel/install guard, Bedrock credentials, model IDs and `client`.
# Shared verbatim with Building_an_Eval.ipynb via tools/shared/setup.py.

# %% [markdown]
# ---
#
# ## The agent, and the three defects
#
# Same shop as the main workshop, rebuilt around the failure modes that actually show up
# in production integrations. The store's data is fine. Every defect is in the **seam
# between the model and the tools**:
#
# | # | Defect | Why it's realistic |
# |---|---|---|
# | 1 | **Prices are integers in minor units.** `4999` means `$49.99`. | Every payments API on earth does this. Stripe, Adyen, your billing service. |
# | 2 | **`search_catalog` is silently paginated** at 4 results, and its price cap is *also* in minor units. | Paginated list endpoints are the default. `next_offset` is easy to ignore. |
# | 3 | **`apply_discount` takes a whole percent** (`20`, not `0.2`) and silently accepts either. | Passing `0.2` returns a *plausible* wrong number. No error, no clue. |
#
# Notice what these have in common: **nothing is missing from the environment.** Every
# fact the model needs is reachable — `minor_unit_scale` is right there in the response,
# `next_offset` is right there in the response. The information is present and
# *undocumented*. That's the setup where capability can substitute for engineering, so
# it's the setup where the comparison is interesting.
#
# (Contrast the main workshop's `boutique`, whose catalog keys are genuinely unobtainable.
# No model can fix that one, and Part 7 shows exactly that.)

# %%
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── The store. Prices in minor units, like every real payments API. ───────────
CATALOG = {
    "jeans": 4999, "shirt": 2999, "dress": 5999, "jacket": 8999,
    "sneakers": 7499, "hat": 1999, "socks": 999, "hoodie": 4499,
    "shorts": 3499, "tee": 2499, "sweater": 5499, "belt": 2499,
}
# What customers say -> what the catalog calls it. The lookup tool knows these; the
# *model* is never told they exist unless a tool spec says so.
ALIASES = {"t-shirt": "tee", "tshirt": "tee", "t shirt": "tee", "shoes": "sneakers",
           "coat": "jacket", "trainers": "sneakers", "cap": "hat"}
PAGE_SIZE = 4

CATALOG_NAMES = ", ".join(sorted(CATALOG))


def get_product(name):
    """Defect 1: `price` is an integer in minor units, and says so only via a field name."""
    key = str(name).strip().lower()
    if key in CATALOG:
        return json.dumps({"sku": f"{key[:3].upper()}-001", "name": key,
                           "price": CATALOG[key], "currency": "USD",
                           "minor_unit_scale": 100, "in_stock": True})
    suggestion = ALIASES.get(key)
    return json.dumps({"error": "not_found", "query": key,
                       "did_you_mean": [suggestion] if suggestion else []})


def search_catalog(max_price=None, offset=0):
    """Defect 2: paginated at PAGE_SIZE, and `max_price` is in minor units too — so the
    obvious call, max_price=30 for "under $30", matches nothing and returns an empty list
    rather than an error."""
    items = sorted(CATALOG.items(), key=lambda kv: kv[1])
    if max_price is not None:
        items = [kv for kv in items if kv[1] <= float(max_price)]
    offset = int(offset or 0)
    page = items[offset:offset + PAGE_SIZE]
    following = offset + PAGE_SIZE if offset + PAGE_SIZE < len(items) else None
    return json.dumps({"items": [{"name": n, "price": p} for n, p in page],
                       "returned": len(page), "next_offset": following})


def apply_discount(price, discount):
    """Defect 3: `discount` is a whole percent. Pass 0.2 for "20% off" and you get back a
    number that is wrong by a plausible-looking margin, with no error at all."""
    return json.dumps({"price": int(round(float(price) * (1 - float(discount) / 100)))})


RAW_TOOLS = {"get_product": get_product, "search_catalog": search_catalog,
             "apply_discount": apply_discount}

print(f"{len(CATALOG)} products, {len(RAW_TOOLS)} tools. "
      f"Cheapest: socks at {CATALOG['socks']} (which is $9.99, not $999).")

# %% [markdown]
# ### Level 0 — the agent as most integrations are first written
#
# A one-line system prompt and tool specs that describe the *arguments* but say nothing
# about the *contract*: no units, no pagination, no return shape, no examples. Nothing
# here is unusual. This is what you get when the tool specs are generated from a function
# signature, or written by whoever wrote the function and already knew the answers.

# %%
PROMPT_L0 = "You are a helpful shopping assistant."

SPECS_L0 = [
    {"name": "get_product", "description": "Look up a product.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string", "description": "product"}},
                      "required": ["name"]}},
    {"name": "search_catalog", "description": "Search the catalog.",
     "input_schema": {"type": "object",
                      "properties": {"max_price": {"type": "number", "description": "max price"},
                                     "offset": {"type": "integer", "description": "offset"}},
                      "required": []}},
    {"name": "apply_discount", "description": "Apply a discount to a price.",
     "input_schema": {"type": "object",
                      "properties": {"price": {"type": "number", "description": "price"},
                                     "discount": {"type": "number", "description": "discount"}},
                      "required": ["price", "discount"]}},
]

# %% [markdown]
# ### The four tuning levers
#
# Each is a different place you can spend engineering effort, and they are **not**
# interchangeable — that turns out to be the most useful thing this notebook measures.
#
# | Config | Prompt | Tool specs | Tool code | The lever |
# |---|---|---|---|---|
# | `L0` | bare | bare | raw | nothing — the baseline |
# | `L1` | **tuned** | bare | raw | tell the *model* the conventions |
# | `L2` | **tuned** | **tuned** | raw | document the conventions **where the model looks** |
# | `L3` | bare | bare | **hardened** | leave the docs alone, fix the **interface** |
# | `L4` | **tuned** | **tuned** | **hardened** | every lever, plus a tool shaped like the task |
#
# `L3` is the interesting one. It changes no documentation at all — it renames the tools
# and their arguments so the names carry the contract (`price_usd`, not `price`), returns
# dollars instead of cents, drops the hidden pagination, and makes misuse return a loud
# error instead of a plausible wrong number. Whether that is enough on its own is exactly
# the sort of thing people assert confidently in design reviews and rarely measure.

# %%
# ── L1/L2: the tuned system prompt. Same words for both. ─────────────────────
PROMPT_TUNED = (
    "You are the shopping assistant for a clothing store.\n\n"
    "Rules you must follow:\n"
    "1. Never state a price you did not get from a tool. If a lookup fails, say the item "
    "is unavailable — do not estimate.\n"
    "2. Tool prices are integers in MINOR UNITS (cents). Divide by 100 before quoting a "
    "dollar figure: 4999 is $49.99, not $4,999. Price *arguments* you pass to tools are "
    "in minor units too.\n"
    "3. If a search result has a non-null `next_offset`, you have only seen part of the "
    "answer. Call again with that offset and keep going until it is null.\n"
    "4. If a lookup returns `did_you_mean`, retry with that value before concluding the "
    "item is not stocked.\n"
    "5. Percentage discounts are whole percents: 20 means 20% off, not 0.2.\n\n"
    "Answer in one short paragraph, quoting dollars to two decimal places."
)

# ── L2: the same three tools, fully documented. No code changes. ─────────────
SPECS_TUNED = [
    {"name": "get_product",
     "description": (
         "Look up one catalog product by name.\n"
         "Returns JSON {sku, name, price, currency, minor_unit_scale, in_stock}. `price` is "
         "an INTEGER IN MINOR UNITS — divide by `minor_unit_scale` (100) for dollars. A "
         "price of 4999 means $49.99, never $4,999.\n"
         "If the name is not a catalog key, returns {error: 'not_found', query, "
         "did_you_mean: [...]}. When `did_you_mean` is non-empty, call this tool again with "
         "that value before telling the customer anything is unavailable."),
     "input_schema": {"type": "object", "properties": {"name": {
         "type": "string",
         "description": f"The catalog key, lowercase. One of: {CATALOG_NAMES}. Map the "
                        f"customer's wording onto one of these yourself."}},
         "required": ["name"]}},
    {"name": "search_catalog",
     "description": (
         "List catalog products cheapest first, optionally under a price cap.\n"
         "Returns JSON {items: [{name, price}], returned, next_offset}. `price` is in minor "
         "units (divide by 100).\n"
         f"THIS RESULT IS PAGINATED at {PAGE_SIZE} items per call. If `next_offset` is not "
         "null there are MORE MATCHES you have not seen — call again with "
         "offset=next_offset and keep going until it is null. Never present one page as the "
         "complete list."),
     "input_schema": {"type": "object", "properties": {
         "max_price": {"type": "number",
                       "description": "Inclusive price cap IN MINOR UNITS: pass 3000 for "
                                      "$30.00. Passing 30 means 30 cents and will match "
                                      "nothing."},
         "offset": {"type": "integer",
                    "description": "Index to start from. Pass the previous next_offset."}},
         "required": []}},
    {"name": "apply_discount",
     "description": (
         "Apply a percentage discount to a price. Returns JSON {price}, in minor units.\n"
         "`discount` is a WHOLE PERCENT: pass 20 for 20% off, not 0.2. Passing 0.2 is "
         "accepted and returns a wrong answer that looks plausible."),
     "input_schema": {"type": "object", "properties": {
         "price": {"type": "number", "description": "Price in minor units (integer cents)."},
         "discount": {"type": "number", "minimum": 1, "maximum": 100,
                      "description": "Whole percent off, 1-100. 20 means 20%."}},
         "required": ["price", "discount"]}},
]

# %%
# ── L3: hardened tools. Same capability, an interface that resists misuse. ────
# The rule being applied: if a caller can get it wrong, that's the interface's fault.
# Dollars in and out, no hidden pagination, and misuse returns a loud error carrying the
# correction rather than a number that merely looks wrong.

def get_product_price(product_name):
    key = str(product_name).strip().lower()
    if key in CATALOG:
        return json.dumps({"product_name": key, "price_usd": CATALOG[key] / 100,
                           "in_stock": True})
    suggestion = ALIASES.get(key)
    return json.dumps({"error": "not_found", "message": (
        f"No product named '{key}'." + (
            f" Did you mean '{suggestion}'? Call this tool again with "
            f"product_name='{suggestion}'." if suggestion else
            " Call list_products to see everything that is stocked."))})


def list_products(max_price_usd=None):
    items = sorted(CATALOG.items(), key=lambda kv: kv[1])
    if max_price_usd is not None:
        items = [kv for kv in items if kv[1] <= float(max_price_usd) * 100]
    return json.dumps({"products": [{"product_name": n, "price_usd": p / 100}
                                    for n, p in items],
                       "count": len(items), "is_complete_list": True})


def apply_percent_discount(price_usd, percent_off):
    pct = float(percent_off)
    if not 1 <= pct <= 100:
        return json.dumps({"error": "invalid_percent_off", "message": (
            f"percent_off must be a whole percent from 1 to 100. You passed {pct}. "
            f"For 15% off, pass 15, not 0.15.")})
    return json.dumps({"price_usd": round(float(price_usd) * (1 - pct / 100), 2)})


HARDENED_TOOLS = {"get_product_price": get_product_price, "list_products": list_products,
                  "apply_percent_discount": apply_percent_discount}

# Descriptions stay as bare as L0's, deliberately. This config tests whether names and
# return shapes alone carry the contract — it is the "we'll fix the API, not write docs"
# position, measured.
SPECS_HARDENED = [
    {"name": "get_product_price", "description": "Look up a product.",
     "input_schema": {"type": "object", "properties": {
         "product_name": {"type": "string", "description": "product"}},
         "required": ["product_name"]}},
    {"name": "list_products", "description": "Search the catalog.",
     "input_schema": {"type": "object", "properties": {
         "max_price_usd": {"type": "number", "description": "max price"}},
         "required": []}},
    {"name": "apply_percent_discount", "description": "Apply a discount to a price.",
     "input_schema": {"type": "object", "properties": {
         "price_usd": {"type": "number", "description": "price"},
         "percent_off": {"type": "number", "description": "discount"}},
         "required": ["price_usd", "percent_off"]}},
]

# %%
# ── L4: every lever, plus one more — a tool shaped like the actual question. ──
# The tasks that survive every other fix are the ones needing several numbers combined.
# So stop asking the model to do arithmetic and give it an endpoint that prices a basket.
# This is the fix that has nothing to do with prompting at all.

def quote_order(items, percent_off=0):
    lines, unknown = [], []
    for item in items or []:
        name = str(item.get("product_name", "")).strip().lower()
        name = name if name in CATALOG else ALIASES.get(name, name)
        quantity = int(item.get("quantity", 1) or 1)
        if name not in CATALOG:
            unknown.append(name)
            continue
        lines.append({"product_name": name, "quantity": quantity,
                      "unit_price_usd": CATALOG[name] / 100,
                      "line_total_usd": round(CATALOG[name] * quantity / 100, 2)})
    if unknown:
        return json.dumps({"error": "unknown_products", "unknown": unknown,
                           "message": f"Not stocked: {', '.join(unknown)}. Valid "
                                      f"product_name values: {CATALOG_NAMES}."})
    pct = float(percent_off or 0)
    if pct and not 1 <= pct <= 100:
        return json.dumps({"error": "invalid_percent_off", "message": (
            f"percent_off is a whole percent from 1 to 100; you passed {pct}. "
            f"For 15% off pass 15, not 0.15.")})
    subtotal = round(sum(line["line_total_usd"] for line in lines), 2)
    discount = round(subtotal * pct / 100, 2)
    return json.dumps({"line_items": lines, "subtotal_usd": subtotal,
                       "discount_percent": pct, "discount_usd": discount,
                       "total_usd": round(subtotal - discount, 2)})


L4_TOOLS = {"get_product_price": get_product_price, "list_products": list_products,
            "quote_order": quote_order}

SPECS_L4 = [
    {"name": "get_product_price",
     "description": ("Look up the price of ONE product the store stocks. Returns "
                     "{product_name, price_usd, in_stock}; price_usd is already in dollars.\n"
                     "If the name is not stocked, returns {error, message}, and the message "
                     "names a likely alternative — call this tool again with that value "
                     "before telling the customer anything is unavailable."),
     "input_schema": {"type": "object", "properties": {"product_name": {
         "type": "string", "enum": sorted(CATALOG),
         "description": f"The catalog key. One of: {CATALOG_NAMES}. Map the customer's "
                        f"wording onto one of these — a 't-shirt' is a 'tee', 'shoes' are "
                        f"'sneakers', 'a coat' is a 'jacket'."}},
         "required": ["product_name"]}},
    {"name": "list_products",
     "description": ("Everything the store stocks, cheapest first, optionally under a price "
                     "cap. Returns {products: [{product_name, price_usd}], count, "
                     "is_complete_list}. Never paginated: one call returns every match, so "
                     "'what do you sell' is answerable from a single call."),
     "input_schema": {"type": "object", "properties": {"max_price_usd": {
         "type": "number", "description": "Inclusive cap in DOLLARS, e.g. 30 for $30.00."}},
         "required": []}},
    {"name": "quote_order",
     "description": ("Price a whole basket in one call, with an optional order-level "
                     "discount. Use this for ANY question involving quantities, more than "
                     "one product, or a discount — it does the arithmetic for you, so never "
                     "add or multiply prices yourself.\n"
                     "Returns {line_items, subtotal_usd, discount_percent, discount_usd, "
                     "total_usd}, all in dollars. Quote total_usd as the answer."),
     "input_schema": {"type": "object", "properties": {
         "items": {"type": "array", "description": "One entry per distinct product.",
                   "items": {"type": "object", "properties": {
                       "product_name": {"type": "string", "enum": sorted(CATALOG),
                                        "description": f"One of: {CATALOG_NAMES}."},
                       "quantity": {"type": "integer", "minimum": 1,
                                    "description": "How many of this product."}},
                       "required": ["product_name", "quantity"]}},
         "percent_off": {"type": "number", "minimum": 1, "maximum": 100,
                         "description": "Whole percent off the whole order, 1-100. 20 means "
                                        "20% off. Omit when there is no discount."}},
         "required": ["items"]}},
]

PROMPT_L4 = (
    "You are the shopping assistant for a clothing store that stocks exactly these "
    f"products: {CATALOG_NAMES}.\n\n"
    "How to answer:\n"
    "1. Map the customer's wording onto a stocked product yourself — a 't-shirt' is a "
    "'tee', 'shoes' are 'sneakers', 'a coat' is a 'jacket'. Never ask the customer to "
    "restate a product you can already identify.\n"
    "2. One product, no quantities, no discount -> `get_product_price`.\n"
    "3. Anything with quantities, several products, or a discount -> `quote_order`, once. "
    "Never do money arithmetic yourself; the tool is exact and you are not.\n"
    "4. 'What do you sell' / 'everything under $X' -> `list_products`, which returns the "
    "complete list in one call.\n"
    "5. If a product genuinely is not stocked, say so plainly and never invent a price.\n\n"
    "All tool prices are already in dollars. Answer in one short paragraph, quoting "
    "dollars to two decimal places."
)

# %%
# ── The five configurations, as data. ────────────────────────────────────────
CONFIGS = {
    "L0": {"prompt": PROMPT_L0,     "specs": SPECS_L0,       "tools": RAW_TOOLS,
           "label": "untouched"},
    "L1": {"prompt": PROMPT_TUNED,  "specs": SPECS_L0,       "tools": RAW_TOOLS,
           "label": "prompt only"},
    "L2": {"prompt": PROMPT_TUNED,  "specs": SPECS_TUNED,    "tools": RAW_TOOLS,
           "label": "prompt + tool docs"},
    "L3": {"prompt": PROMPT_L0,     "specs": SPECS_HARDENED, "tools": HARDENED_TOOLS,
           "label": "interface only"},
    "L4": {"prompt": PROMPT_L4,     "specs": SPECS_L4,       "tools": L4_TOOLS,
           "label": "everything"},
}

for _name, _cfg in CONFIGS.items():
    _chars = len(_cfg["prompt"]) + len(json.dumps(_cfg["specs"]))
    print(f"{_name}  {_cfg['label']:<20} {len(_cfg['specs'])} tools, "
          f"{_chars:>5} chars of prompt + specs")

# %% [markdown]
# ---
#
# ## The eval suite
#
# Seven tasks. Deliberately including two that the untuned agent usually passes even on the
# cheapest model, because a suite where everything fails tells you nothing about *which*
# change helped — and a suite rigged to make your point is not an eval, it's a slide.
#
# | Task | Query | What it probes |
# |---|---|---|
# | `units_simple` | How much do jeans cost? | Control. One lookup, one conversion. |
# | `alias_recovery` | How much is a t-shirt? | Control. Not a catalog key; `did_you_mean` says `tee`. |
# | `affordance_price` | What's 20% off a jacket? | Does the model even *recognise* `get_product` applies to "a jacket"? |
# | `affordance_slang` | What's the damage for a hoodie and a pair of sneakers? | Same, through colloquial phrasing. |
# | `empty_result` | Do you have anything under $15? | The obvious call returns `[]` rather than an error. Believe it, or recover? |
# | `paged_list` | List every item under $30, with prices. | Five matches, four per page. |
# | `compound` | 2 tees and a hat, 15% off the whole order. Total? | Units + quantities + percent convention, compounded. |
#
# The checks are all deterministic — no judge, so nothing here depends on a grader's own
# reliability. `not_numeric` matters as much as `numeric`: for `units_simple`, "$4,999" and
# "$49.99" are both confident answers, and only one is right.

# %%
def _numbers(text):
    return [float(n.replace(",", ""))
            for n in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)]


def check_numeric(text, spec):
    """Some number in the answer is `value`, within tolerance."""
    return any(abs(n - spec["value"]) <= spec.get("tol", 0.02) for n in _numbers(text))


def check_not_numeric(text, spec):
    """No number in the answer is near `value` — catches the confidently-wrong variant."""
    return not any(abs(n - spec["value"]) <= spec.get("tol", 0.5) for n in _numbers(text))


def check_contains_all(text, spec):
    low = text.lower()
    return all(word in low for word in spec["words"])


CHECKS = {"numeric": check_numeric, "not_numeric": check_not_numeric,
          "contains_all": check_contains_all}

TASKS = [
    {"id": "units_simple", "query": "How much do jeans cost?",
     "probes": "control: one lookup, one minor-unit conversion",
     "checks": [{"type": "numeric", "value": 49.99},
                {"type": "not_numeric", "value": 4999.0}]},
    {"id": "alias_recovery", "query": "How much is a t-shirt?",
     "probes": "control: not a catalog key, did_you_mean carries the answer",
     "checks": [{"type": "numeric", "value": 24.99}]},
    {"id": "affordance_price", "query": "What's 20% off a jacket?",
     "probes": "is the lookup tool recognised as applicable at all?",
     "checks": [{"type": "numeric", "value": 71.99}]},
    {"id": "affordance_slang",
     "query": "What's the damage for a hoodie and a pair of sneakers?",
     "probes": "same, through colloquial phrasing; two lookups",
     "checks": [{"type": "numeric", "value": 119.98, "tol": 0.03}]},
    {"id": "empty_result", "query": "Do you have anything under $15?",
     "probes": "cap is in minor units, so the obvious call returns [] not an error",
     "checks": [{"type": "numeric", "value": 9.99},
                {"type": "contains_all", "words": ["socks"]}]},
    {"id": "paged_list", "query": "List every item you sell for under $30, with prices.",
     "probes": "five matches, silently paginated at four",
     "checks": [{"type": "contains_all",
                 "words": ["socks", "hat", "belt", "tee", "shirt"]}]},
    {"id": "compound",
     "query": "I want 2 tees and a hat, with 15% off the whole order. Total?",
     "probes": "units + quantities + percent convention, compounded",
     "checks": [{"type": "numeric", "value": 59.47, "tol": 0.03}]},
]

print(f"{len(TASKS)} tasks, {sum(len(t['checks']) for t in TASKS)} checks, 0 judges.")

# %%
# ── Runner ───────────────────────────────────────────────────────────────────
# Deliberately small: one agent loop parameterised by config, one grader pass, one
# results dict keyed by (config, model). The main workshop's runner is the one to copy
# for real work — this one only has to support a grid.

MAX_TURNS = 12          # the raw agent has no guard of its own; don't fund an infinite loop


def run_agent(query, config_name, model, capture=None):
    """Run one query under one config. Returns (final_text, tool_calls, metrics)."""
    config = CONFIGS[config_name]
    messages = [{"role": "user", "content": query}]
    tokens_in = tokens_out = turns = 0
    started = time.time()

    for _ in range(MAX_TURNS):
        response = client.messages.create(model=model, system=config["prompt"],
                                          max_tokens=2048, tools=config["specs"],
                                          messages=messages)
        tokens_in += response.usage.input_tokens
        tokens_out += response.usage.output_tokens
        turns += 1
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                output = str(config["tools"][block.name](**block.input))
            except Exception as exc:                 # a tool bug is the agent's problem to
                output = f"Error: {exc}"             # handle, not the harness's to hide
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": output})
        messages.append({"role": "user", "content": results})

    final_text, calls = "", []
    for message in messages:
        if message["role"] != "assistant":
            continue
        for block in message["content"]:
            if block.type == "text":
                final_text = block.text
            elif block.type == "tool_use":
                calls.append({"name": block.name, "input": block.input})

    metrics = {"tokens_in": tokens_in, "tokens_out": tokens_out, "turns": turns,
               "tool_calls": len(calls), "seconds": round(time.time() - started, 2)}
    if capture is not None:
        capture.append({"query": query, "text": final_text, "calls": calls,
                        "metrics": metrics})
    return final_text, calls, metrics


def run_task(task, config_name, model):
    try:
        text, calls, metrics = run_agent(task["query"], config_name, model)
    except Exception as exc:
        # An infrastructure failure is not an agent failure. Keep them distinguishable or
        # you will spend the afternoon fixing a prompt to cure a throttle.
        return {"id": task["id"], "passed": False, "text": "",
                "error": f"{type(exc).__name__}: {exc}"[:160], "calls": [], "metrics": {}}
    failed = [c for c in task["checks"] if not CHECKS[c["type"]](text, c)]
    return {"id": task["id"], "passed": not failed, "text": text, "error": None,
            "calls": calls, "metrics": metrics, "failed_checks": failed}


def run_grid(cells, tasks, runs, max_workers=6):
    """cells: [(config_name, model)]. Returns {(config, model): [result, ...]}."""
    jobs = [(t, c, m) for c, m in cells for t in tasks for _ in range(runs)]
    out = {}
    print(f"{len(jobs)} agent runs across {len(cells)} cell(s)…", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_task, t, c, m): (c, m) for t, c, m in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            out.setdefault(futures[future], []).append(future.result())
            if done % 25 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}", flush=True)
    errors = [r for rs in out.values() for r in rs if r["error"]]
    if errors:
        print(f"\n⚠️  {len(errors)} run(s) never completed — infrastructure, not the agent. "
              f"First: {errors[0]['error']}")
        if len(errors) > len(jobs) / 4:
            print("    That volume is almost always Bedrock throttling. Re-run with "
                  "max_workers=1.")
    return out


def rates(results, tasks):
    """{task_id: fraction of runs passed}, plus the overall fraction."""
    per_task = {}
    for task in tasks:
        runs = [r for r in results if r["id"] == task["id"]]
        per_task[task["id"]] = sum(r["passed"] for r in runs) / len(runs) if runs else 0.0
    overall = sum(r["passed"] for r in results) / len(results) if results else 0.0
    return per_task, overall


def show_grid(grid, cells, tasks, title, column_label, legend=None):
    """One table: tasks down the side, grid cells across the top."""
    width = max(len(t["id"]) for t in tasks) + 2
    heads = [column_label(c) for c in cells]
    rule = "─" * (width + 11 * len(heads))
    print(f"\n{title}")
    if legend:
        print(legend)
    print(rule)
    print(" " * width + "".join(f"{h[:10]:>11}" for h in heads))
    for task in tasks:
        row = f"{task['id']:<{width}}"
        for cell in cells:
            per_task, _ = rates(grid[cell], tasks)
            row += f"{per_task[task['id']] * 100:>10.0f}%"
        print(row)
    print(rule)
    row = f"{'OVERALL':<{width}}"
    for cell in cells:
        _, overall = rates(grid[cell], tasks)
        row += f"{overall * 100:>10.0f}%"
    print(row)


def n_runs(count):
    return f"{count} run" if count == 1 else f"{count} runs"


print("Runner ready.")

# %% [markdown]
# ---
#
# ## Experiment 1 — the capability ladder
#
# The untouched `L0` agent, unchanged, on every model this account can reach. Nothing but
# the model varies.
#
# **Predict before you run it.** Write down a number for each model. The interesting part
# of this cell is not the table, it's the gap between the table and your prediction.
#
# > `RUNS = 3` is the floor for reading anything into a per-task cell, and even three runs
# > only distinguishes "always", "never" and "sometimes". Treat every intermediate
# > percentage as *unstable*, not as a measurement.

# %%
RUNS = 3

# Cheapest first, so the table reads left-to-right up the capability curve. Anything this
# account can't reach was already dropped by the setup cell.
LADDER_MODELS = [m for m in [FAST_MODEL, MODEL, FABLE_5_1_MODEL, BIG_MODEL]
                 if m in AVAILABLE_MODELS]

ladder_cells = [("L0", m) for m in LADDER_MODELS]
ladder = run_grid(ladder_cells, TASKS, RUNS)

show_grid(ladder, ladder_cells, TASKS,
          f"Untuned agent (L0), {n_runs(RUNS)} — only the model varies",
          lambda cell: short_model_name(cell[1]))

_scores = {m: rates(ladder[("L0", m)], TASKS)[1] * 100 for m in LADDER_MODELS}
_cheapest, _best = LADDER_MODELS[0], max(_scores, key=_scores.get)
_floor = [t["id"] for t in TASKS
          if all(rates(ladder[("L0", m)], TASKS)[0][t["id"]] == 0 for m in LADDER_MODELS)]
_ceiling = [t["id"] for t in TASKS
            if all(rates(ladder[("L0", m)], TASKS)[0][t["id"]] == 1 for m in LADDER_MODELS)]
_split = [t["id"] for t in TASKS if t["id"] not in _floor and t["id"] not in _ceiling]

if len(LADDER_MODELS) < 2:
    # Restricted account: one column is not a ladder. Say so rather than printing a
    # comparison of a model against itself.
    print(f"\nOnly {short_model_name(_cheapest)} is reachable on this account, so there is "
          f"no capability ladder to read — {_scores[_cheapest]:.0f}% is a baseline, not a "
          f"comparison. Experiment 2 works regardless; it only needs one model.")
else:
    print(f"\nCapability alone moved this agent from {_scores[_cheapest]:.0f}% "
          f"({short_model_name(_cheapest)}) to {_scores[_best]:.0f}% "
          f"({short_model_name(_best)}) — {_scores[_best] - _scores[_cheapest]:+.0f} "
          f"points, for a per-token price difference you can look up.")
    if _floor:
        print(f"Not passed by any of the {len(LADDER_MODELS)} models measured: "
              f"{', '.join(_floor)} — capability did not reach these.")
    if _ceiling:
        print(f"Passed by all {len(LADDER_MODELS)}: {', '.join(_ceiling)} — you were never "
              f"paying for these.")
    if _split:
        print(f"Only these responded to capability: {', '.join(_split)}. That set — not "
              f"the overall score — is what a model upgrade actually buys you.")
    _wrong_way = [(a, b) for a, b in zip(LADDER_MODELS, LADDER_MODELS[1:])
                  if _scores[b] < _scores[a]]
    for _a, _b in _wrong_way:
        print(f"And note {short_model_name(_b)} scored BELOW {short_model_name(_a)} "
              f"({_scores[_b]:.0f}% vs {_scores[_a]:.0f}%). Capability is not a scalar — "
              f"models don't fall on one line, so 'bigger' is not a plan.")

# %% [markdown]
# ### Why — read the transcripts, not the percentages
#
# A score tells you *that* it failed. Only the transcript tells you *what to fix*, and the
# fix is usually not what the score suggests. The next cell runs the split tasks once on
# the cheapest and the strongest model and prints the tool calls side by side.
#
# Watch for the three failure shapes worth knowing by name:
#
# - **Not recognising the affordance.** The model asks *you* for a price it could have
#   looked up. The tool was right there. `"Look up a product."` just wasn't enough for it
#   to believe `"a jacket"` was a valid `name`.
# - **Believing an empty result.** The tool returned `[]`, so the model reports that
#   nothing matches — fluently, and wrongly. No error was raised, so nothing prompted a
#   second thought.
# - **Silent arithmetic.** No tool for adding up a basket, so the model does it in its
#   head, and a single transposed digit becomes a total that is wrong by pennies and looks
#   entirely plausible.

# %%
DIAGNOSE = [t for t in TASKS if t["id"] in set(_split) | set(_floor)] or TASKS[-2:]
_pair = [LADDER_MODELS[0]] + ([_best] if _best != LADDER_MODELS[0] else [])

for _task in DIAGNOSE:
    print(f"\n{'═' * 78}\n{_task['id']}: {_task['query']}\n  probes: {_task['probes']}")
    for _model in _pair:
        _text, _calls, _m = run_agent(_task["query"], "L0", _model)
        _ok = not [c for c in _task["checks"] if not CHECKS[c["type"]](_text, c)]
        print(f"\n  {'─' * 74}\n  {short_model_name(_model):<12} "
              f"{'PASS' if _ok else 'FAIL'}   {_m['turns']} turns, "
              f"{_m['tool_calls']} tool calls, {_m['seconds']}s")
        for _c in _calls:
            print(f"     → {_c['name']}({_c['input']})")
        if not _calls:
            print("     → (no tool calls at all)")
        print(f"     A: {' '.join(_text.split())[:220]}")

# %% [markdown]
# ---
#
# ## Experiment 2 — the engineering ladder
#
# Now hold the model at the **cheapest** one and walk the four tuning levers. Same tasks,
# same checks, same runs, so the numbers are directly comparable to Experiment 1.
#
# The question is not "does tuning help" — of course it does. It's **which lever fixes
# which bug**, and whether the levers are additive. They are not.

# %%
CONFIG_LADDER = ["L0", "L1", "L2", "L3", "L4"]
TUNE_MODEL = LADDER_MODELS[0]

tune_cells = [(c, TUNE_MODEL) for c in CONFIG_LADDER]
tuned = run_grid(tune_cells, TASKS, RUNS)

show_grid(tuned, tune_cells, TASKS,
          f"{short_model_name(TUNE_MODEL)} only, {n_runs(RUNS)} — the agent varies, "
          f"the model does not",
          lambda cell: cell[0],
          legend="  ".join(f"{c}={CONFIGS[c]['label']}" for c in CONFIG_LADDER))

_tune_scores = {c: rates(tuned[(c, TUNE_MODEL)], TASKS)[1] * 100 for c in CONFIG_LADDER}
# max() over a dict returns the FIRST maximum, so ties resolve to the earliest config —
# the least invasive change that reached the ceiling. That's the answer you want anyway:
# if documentation alone got there, the extra tooling was work you didn't need to do.
_best_config = max(_tune_scores, key=_tune_scores.get)
print(f"\nOn {short_model_name(TUNE_MODEL)} alone, engineering moved this agent from "
      f"{_tune_scores['L0']:.0f}% to {_tune_scores[_best_config]:.0f}% "
      f"({_best_config}, {CONFIGS[_best_config]['label']}) — "
      f"{_tune_scores[_best_config] - _tune_scores['L0']:+.0f} points, at no extra cost "
      f"per request.")
_ties = [c for c in CONFIG_LADDER
         if c != _best_config and _tune_scores[c] == _tune_scores[_best_config]]
if _ties:
    print(f"{', '.join(_ties)} tied with it. {_best_config} is the cheapest way to get "
          f"there, so the extra levers in {_ties[-1]} bought nothing on THIS suite — which "
          f"is the only kind of evidence that should stop you building them.")

# The configs are NOT ordered by amount of effort — L3 is a different lever, not a bigger
# one — so a dip between adjacent columns is the finding, not a mistake. Name it.
_dips = [(a, b) for a, b in zip(CONFIG_LADDER, CONFIG_LADDER[1:])
         if _tune_scores[b] < _tune_scores[a]]
for _a, _b in _dips:
    print(f"Note {_b} ({CONFIGS[_b]['label']}) scores BELOW {_a} "
          f"({_tune_scores[_b]:.0f}% vs {_tune_scores[_a]:.0f}%) — and {_b} is not less "
          f"work than {_a}, just a different lever. This column order is not a ladder you "
          f"climb; it's a set of tools that fix different bugs. Compare the per-task rows "
          f"to see which.")
for _t in TASKS:
    _by = {c: rates(tuned[(c, TUNE_MODEL)], TASKS)[0][_t["id"]] for c in CONFIG_LADDER}
    _fixers = [c for c in CONFIG_LADDER if _by[c] == 1 and _by["L0"] < 1]
    if _by["L0"] < 1 and _fixers:
        print(f"  {_t['id']:<18} first fixed by {_fixers[0]} ({CONFIGS[_fixers[0]]['label']})")

# A reproducibility check that costs nothing: ("L0", TUNE_MODEL) was already measured in
# Experiment 1, so these two tables contain two independent samples of the identical
# config. Whatever they disagree by is pure run-to-run noise — and that number is the only
# honest yardstick you have for how much of a gap elsewhere in these tables is real.
_first = rates(ladder[("L0", TUNE_MODEL)], TASKS)
_again = rates(tuned[("L0", TUNE_MODEL)], TASKS)
print(f"\nNoise check — L0 on {short_model_name(TUNE_MODEL)} was measured twice, "
      f"independently, {n_runs(RUNS)} each: {_first[1] * 100:.0f}% in Experiment 1 vs "
      f"{_again[1] * 100:.0f}% here.")
_wobble = [t["id"] for t in TASKS if _first[0][t["id"]] != _again[0][t["id"]]]
if _wobble:
    print(f"Identical prompt, specs, tools and model — different numbers on: "
          f"{', '.join(_wobble)}. Nothing changed but the sample. Treat any gap of that "
          f"size elsewhere in these tables as noise until more runs say otherwise.")
else:
    print(f"Every task agreed. At {n_runs(RUNS)} that is reassuring rather than proof: it "
          f"means no task happened to land mid-range twice, not that none can.")

# %% [markdown]
# ---
#
# ## Experiment 3 — the decision
#
# Two options, priced. The **tuned cheap model** against the **untuned expensive one** —
# the actual choice in front of you, with no new API calls: both numbers were already
# measured above.
#
# Accuracy is the first column and the least interesting one. Tokens and latency are what
# you pay on every request for the rest of the agent's life.

# %%
def summarise(config_name, model):
    """One row of the decision table, from runs already in `ladder` or `tuned`."""
    cell = (config_name, model)
    source = ladder if cell in ladder else tuned
    runs = source[cell]
    ok = [r for r in runs if not r["error"]] or runs
    _, accuracy = rates(runs, TASKS)
    mean = lambda key: sum(r["metrics"].get(key, 0) for r in ok) / len(ok)
    return {"what": f"{short_model_name(model)} · {config_name} "
                    f"({CONFIGS[config_name]['label']})",
            "acc": accuracy * 100, "tin": mean("tokens_in"), "tout": mean("tokens_out"),
            "turns": mean("turns"), "sec": mean("seconds")}


baseline = summarise("L0", TUNE_MODEL)                 # where we started
tuned_small = summarise(_best_config, TUNE_MODEL)       # option A: fix the agent
untuned_big = summarise("L0", _best)                    # option B: buy capability

_rows = [baseline] + [r for r in (tuned_small, untuned_big) if r["what"] != baseline["what"]]
print(f"{'option':<44}{'pass':>7}{'tok in':>9}{'tok out':>9}{'turns':>7}{'sec':>7}")
print("─" * 83)
for _r in _rows:
    print(f"{_r['what']:<44}{_r['acc']:>6.0f}%{_r['tin']:>9.0f}{_r['tout']:>9.0f}"
          f"{_r['turns']:>7.1f}{_r['sec']:>7.1f}")
print()

if _best == TUNE_MODEL:
    # Either one model was reachable, or the cheapest model already scored highest. Either
    # way there is no "buy capability" column here, and claiming one would be a lie.
    print(f"{short_model_name(TUNE_MODEL)} is both the cheapest model available and the "
          f"highest-scoring untuned one, so this run has no capability option to price "
          f"against — the engineering ladder above is the whole result. Re-run with a "
          f"second model reachable to get the comparison this section is for.")
elif tuned_small["acc"] >= untuned_big["acc"]:
    print(f"The tuned cheap agent matches or beats the untuned expensive one: "
          f"{tuned_small['acc']:.0f}% on {short_model_name(TUNE_MODEL)} vs "
          f"{untuned_big['acc']:.0f}% on {short_model_name(_best)}, at "
          f"{tuned_small['sec'] / max(untuned_big['sec'], 0.01):.1f}× the latency and a "
          f"per-token price you can look up.\n"
          f"The engineering was paid once. The token bill recurs forever.")
else:
    print(f"On this suite the untuned {short_model_name(_best)} still leads: "
          f"{untuned_big['acc']:.0f}% vs {tuned_small['acc']:.0f}%. Worth stating plainly "
          f"rather than talking around — the gap is the tasks tuning didn't reach. Find "
          f"them in the Experiment 2 rows and decide whether they're worth another lever "
          f"or the bigger model.")

print(f"\nNote the token columns as well as the pass column. Tuning bought accuracy AND "
      f"changed the input-token count, because documentation is tokens too — a fair "
      f"comparison prices both. Per-token rates move, so look them up rather than trusting "
      f"a number embedded in a notebook; what's durable is that one of these columns is a "
      f"one-off engineering cost and the rest recur on every request for the life of the "
      f"agent.")

# %% [markdown]
# ---
#
# ## What this measured
#
# **Capability and engineering are not substitutes.** They fix disjoint sets of bugs, and
# the overall percentage hides which. Read your own per-task tables above, not this list —
# but the shape usually comes out like this:
#
# | Failure | Fixed by a bigger model? | Fixed by engineering? |
# |---|---|---|
# | Doesn't realise a tool applies | Often — stronger models guess the affordance | Yes, and cheaply: one example value in the argument description |
# | Believes an empty tool result | Sometimes, by retrying with different arguments | Yes — say the units in the spec, or take dollars in the first place |
# | Ignores `next_offset` | Unreliably, and it's *not monotonic in model size* | Yes — one sentence in the spec, or drop the pagination |
# | Silent mental arithmetic | Partly: stronger models slip less, but they still slip | Yes, completely — give it a tool that returns the total |
# | Information genuinely absent | **No.** Ever. | Only by putting the information somewhere reachable |
#
# Five things worth taking to a design review:
#
# 1. **"Use a bigger model" is a hypothesis, not a plan.** It is testable in one cell. The
#    number is often much smaller than the room assumes, and it is *never* uniform across
#    tasks — a small overall gain can be one task flipping.
# 2. **Capability is not a scalar.** Models don't fall on one line. Expect a cell where a
#    cheaper model beats a dearer one; it isn't a bug in your eval.
# 3. **The cheapest fix is usually an example, not an explanation.** A single valid value
#    in an argument description outperforms a paragraph of prose in the system prompt,
#    because it lands where the model is actually deciding what to pass.
# 4. **The interface is part of the prompt.** Argument names, return-field names and error
#    strings are read by the model on every call, whether or not you thought of them as
#    documentation. `price_usd` is a better spec than a sentence saying the price is in
#    dollars — but note from Experiment 2 that renaming alone did *not* rescue every task,
#    so it is a complement to documentation, not a replacement.
# 5. **The best fix often isn't in the prompt at all.** When a model keeps getting
#    arithmetic wrong, the durable fix is to stop asking it to do arithmetic — `L4`'s
#    `quote_order` returns the total, so there is nothing left to get wrong. But check your
#    own Experiment 2 row before reaching for it: if `L2` already reached the ceiling,
#    building that endpoint was work the eval would have told you to skip.
#
# ### And what it did not measure
#
# Be as clear about this as about the results, because it's where an eval gets oversold:
#
# - **Seven tasks and three runs is a sketch.** It's enough to separate "always" from
#   "never" and to catch a lever that does nothing. It is not enough to defend a 10-point
#   difference between two adjacent cells — the noise check at the end of Experiment 2
#   measures exactly how little that difference is worth here. Raise `RUNS` before you
#   quote a number at anyone.
# - **The tuning was written knowing these tasks.** That's the eval-overfitting trap the
#   main workshop warns about. Real confidence needs a held-out suite the prompt author
#   never saw — and `L4`'s `enum` of catalog names is the clearest example of a fix that
#   would not survive a catalog of fifty thousand products.
# - **Tuning has its own regressions.** If a cell in Experiment 2 went *down*, that's
#   real: a prompt rule that fixes one task can break another. This is the entire argument
#   for keeping the eval and re-running it, rather than reading a table once and moving on.
#
# ### Next
#
# - `Building_an_Eval.ipynb` — build the eval suite and graders this notebook takes for
#   granted, including the LLM judge for the behaviours no string match reaches.
# - Swap `CATALOG` and `TASKS` for your own agent's tools and failures. The two ladders and
#   the runner transfer unchanged; that's the reusable part.
