# The seven scenarios, in plain words

A talking guide to the eval suite in
[`Bigger_Model_or_Better_Agent.ipynb`](Bigger_Model_or_Better_Agent.ipynb) — what each task
asks, what the agent gets wrong, and the right and wrong answer side by side.

Written to be explained out loud. Every number here is the notebook's own: catalog values
from the `CATALOG` dict, expected answers from the `checks` in `TASKS`.

---

## Set the stage first

Say this once and the seven scenarios all follow from it.

The shop sells 12 items. **Every price is stored in cents, as a whole number** — jeans are
`4999`, meaning $49.99. That isn't a contrivance to make the demo fail; it's how Stripe,
Adyen and most billing systems represent money.

There are three tools, and the flaw in each is in its **documentation**, not its logic:

| Tool | What it does | The flaw |
|---|---|---|
| `get_product(name)` | Price for one item | Returns `4999`, and hints at cents only through a `minor_unit_scale: 100` field nobody reads |
| `search_catalog(max_price, offset)` | Items under a price | `max_price` is **also in cents** — and it silently returns only 4 results per call |
| `apply_discount(price, discount)` | Applies a discount | Wants `20` for 20%, not `0.2` — and accepts `0.2` without complaint |

The important property: **nothing is missing.** `minor_unit_scale` is right there in the
response. `next_offset` is right there in the response. Every fact the model needs is
reachable and undocumented. That's what makes the bigger-model-vs-better-agent comparison
fair — a stronger model genuinely *could* figure these out.

---

## 1. `units_simple` — the control

> **"How much do jeans cost?"**

The simplest possible request. One lookup, one division.

| | |
|---|---|
| Right | **$49.99** |
| Wrong | **"$4,999"** — a hundred times too expensive, stated with complete confidence |

This task carries **two** checks: the answer must contain `49.99`, **and must not contain
`4999`**. That second check is the lesson. "$4,999" isn't a fuzzy answer deserving partial
credit — it's a fluent, well-formatted lie. An eval that only looks for the right number
would score a confidently wrong answer the same as "I'm not sure".

---

## 2. `alias_recovery` — the other control

> **"How much is a t-shirt?"**

There is no "t-shirt" in the catalog. The item is called `tee`. So the lookup fails — but
the failure carries a breadcrumb:

```json
{"error": "not_found", "query": "t-shirt", "did_you_mean": ["tee"]}
```

| | |
|---|---|
| Right | Read the hint, call the tool **again** with `tee`, answer **$24.99** |
| Wrong | "Sorry, we don't sell those" — or worse, invent a plausible price |

**Can the agent recover from a failure that told it exactly how to recover?**

Both controls exist because a suite where everything fails teaches you nothing about
*which* change helped. And a suite rigged to prove your point isn't an eval, it's a slide.

---

## 3. `affordance_price` — does it know the tool applies?

> **"What's 20% off a jacket?"**

Jacket is $89.99, so 20% off is **$71.99**.

Two separate traps, which is why this is the task that stays broken longest:

**Trap one — the model doesn't realise it can look this up.** The tool description says
only `"Look up a product."`, and the argument is described as `"product"`. So instead of
calling it, the model turns around and asks *you*: "Happy to help — what's the jacket's
price?" The tool was sitting right there the whole time.

**Trap two — the percent convention.** If it does get the price, it may pass `0.2` to
`apply_discount`:

| | |
|---|---|
| Right | **$71.99** (passing `20`) |
| Wrong | **$89.81** (passing `0.2`) |

Look at $89.81. It's *plausible*. It looks like a small discount was applied. Nothing
errored, nothing logged, and no reviewer skimming the output would blink.

---

## 4. `affordance_slang` — same question, human phrasing

> **"What's the damage for a hoodie and a pair of sneakers?"**

"What's the damage" means "what's the total". The agent has to recognise a colloquial
phrase as a price request, do **two** lookups, and add the results itself — there is no
tool for addition.

| | |
|---|---|
| Right | $44.99 + $74.99 = **$119.98** |
| Wrong | Asking what a "hoodie" is, or adding the cents wrong and being off by a few pennies |

---

## 5. `empty_result` — the tool told the truth and misled it

*If you only have time to explain one scenario, use this one.*

> **"Do you have anything under $15?"**

The obvious call is `search_catalog(max_price=15)`. But that field is in **cents** — so the
agent has just asked for *anything under fifteen cents*.

Nothing matches. The tool returns an empty list. **No error. No warning. No exception.**

So the model reports, politely and fluently: *"I'm sorry, we don't have anything under
$15."*

Socks are **$9.99**. They were always there.

| | |
|---|---|
| Right | `max_price=1500` → **socks, $9.99** |
| Wrong | "Nothing under $15" |

The checks require both the number `9.99` and the word `socks`.

**The tool did not break.** It answered precisely the question it was asked, and the
question was wrong. This is the class of failure that monitoring and alerting never
catch — because from the system's point of view, nothing failed.

---

## 6. `paged_list` — the item that quietly disappears

> **"List every item you sell for under $30, with prices."**

Five items qualify: socks $9.99, hat $19.99, tee $24.99, belt $24.99, shirt $29.99.

The tool returns **four** of them, plus one quiet field: `next_offset: 4`. The agent has to
notice that value isn't null and ask for the second page.

When it doesn't, **the shirt silently vanishes.**

| | |
|---|---|
| Right | All five items listed |
| Wrong | Four items, presented as a complete list |

And the customer cannot tell. There's no truncation notice, no "…and more" — the reply
reads exactly like a finished answer.

Worth pointing out: **this bug hides behind scenario 5.** You have to get `max_price=3000`
right before you can even discover that pagination exists. Bugs stack, and the eval finds
them in the order they surface.

---

## 7. `compound` — everything at once

> **"I want 2 tees and a hat, with 15% off the whole order. Total?"**

| | |
|---|---|
| Right | (2 × $24.99) + $19.99 = $69.97, less 15% = **$59.47** |
| Wrong | **$69.87** — what you get passing `0.15` instead of `15` |

Every problem in one request: cents, multiplying by quantity, and the percent convention.
And because there's no basket-pricing tool, the model does the arithmetic **in its head** —
so a single transposed digit becomes a total that's wrong by pennies and looks entirely
reasonable.

This is the task that nothing fixes until the tools are properly documented. It's also the
reason the notebook's `L4` config adds a `quote_order` tool: the real fix isn't better
prompting, it's **not asking the model to do arithmetic at all.**

---

## The shape of the suite

| Group | Tasks | What it tests |
|---|---|---|
| **Controls** | `units_simple`, `alias_recovery` | Already passing — proves the suite isn't rigged, and that no fix broke what worked |
| **"Didn't know it could"** | `affordance_price`, `affordance_slang` | The tool exists and the model doesn't reach for it |
| **"Truthfully misled"** | `empty_result`, `paged_list` | The tool answered correctly and the answer was wrong |
| **All of it** | `compound` | Units + quantity + percent, compounded |

The thread running through all seven:

> **Not one failure is the model being stupid, and not one raises an error.** Every wrong
> answer is fluent, confident and numerically plausible. That is exactly why you need an
> eval to find them.

---

## Reading the results table

Scores come from **3 runs** per cell, so the only possible values are 0%, 33%, 67% and
100%. **33% and 67% mean one-of-three and two-of-three — that's noise, not a
measurement.** Three runs distinguishes "always", "never" and "sometimes", and nothing
finer.

In Experiment 2 the model is held constant (the cheapest one) and the **agent** varies
across five configurations:

| Config | Prompt | Tool specs | Tool code | The lever |
|---|---|---|---|---|
| `L0` | bare | bare | raw | nothing — the baseline |
| `L1` | tuned | bare | raw | tell the *model* the conventions |
| `L2` | tuned | tuned | raw | document them **where the model looks** |
| `L3` | bare | bare | hardened | leave the docs alone, fix the **interface** |
| `L4` | tuned | tuned | hardened | every lever, plus a tool shaped like the task |

**These columns are not ordered by effort.** `L3` is a *different* lever, not a bigger one,
so a score dropping as you read left-to-right is the finding rather than a mistake — see
the `affordance_price` row, which `L3` fails completely because rewriting a tool cannot fix
a model that never knew the tool applied.
