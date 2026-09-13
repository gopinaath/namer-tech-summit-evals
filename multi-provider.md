# Running this workshop on non-Anthropic Bedrock models

Investigation notes. **Partly shipped, partly not**, and the split matters:

- **Shipped.** [`Multi_Provider_on_Bedrock.ipynb`](Multi_Provider_on_Bedrock.ipynb) runs the
  workshop's agent, graders, runner and judge on Claude Haiku 4.5 **and** OpenAI's GPT 5.6 /
  GPT 6 reasoning models, over Bedrock's Converse API. Everything below about the Converse
  path, the silent failure, the `maxTokens` floor and the OpenAI reasoning profiles is the
  investigation that produced it. If you want the *lesson*, read the notebook; this file is the
  working record behind it.
- **Not shipped.** xAI / Grok and the four open-weight `gpt-oss` models. They were probed, they
  mostly work, and they are deliberately absent from the notebook — see
  [What stayed out, and why](#what-stayed-out-and-why). For those two families this file is
  still the only record.
- **Unchanged.** `Building_an_Eval.ipynb` and `Bigger_Model_or_Better_Agent.ipynb` remain
  Anthropic-only, on the Anthropic Messages wire format, on purpose. Nothing here proposes
  changing them.

All findings come from real API calls against **one AWS account in `us-west-2`**, first probed
**2026-09-12** and re-measured **2026-09-13**. Where the two dates disagree, the newer number
wins and the disagreement is written down rather than smoothed over. Model availability is
granted per account and provider behaviour drifts, so re-check on your own account before
quoting any of this.

---

## Verdict

**Other providers work on Bedrock, and they do tool use. They do not work with the main
workshop's client.** That harness is built on `anthropic.AnthropicBedrock`, which speaks the
Anthropic Messages wire format. Going multi-provider means moving to Bedrock's **Converse**
API — a real change to the agent loop, the transcript parser and the judge, not a change to
the model constants.

Adding a GPT or Grok ID to `MODEL` / `SWEEP_EXTRA` as things stand does **not** reliably raise
a clean error. See [The silent failure](#the-silent-failure) — that is the reason this was
never a five-minute change, and the reason it became its own notebook instead of a row in
Part 7.

What the port actually cost, once attempted: **the graders needed zero changes** and the
tool-use protocol ported whole. Six small things broke. That is the notebook's content.

---

## What's available

18 providers list models in this account: Amazon, Anthropic, Cohere, DeepSeek, Google,
Luma AI, Meta, MiniMax, Mistral AI, Moonshot AI, NVIDIA, OpenAI, Qwen, Stability AI,
TwelveLabs, Writer, Z.AI, xAI.

### OpenAI — 8 models

| modelId | Invocation | Modalities | In the notebook? |
|---|---|---|---|
| `openai.gpt-6-astra` | inference profile | text, image | **yes — the default** |
| `openai.gpt-5.6-sol` | inference profile | text, image | available, commented out |
| `openai.gpt-5.6-luna` | inference profile | text, image | available, commented out |
| `openai.gpt-5.6-terra` | inference profile | text, image | available, commented out |
| `openai.gpt-oss-120b-1:0` | **on-demand** | text | no — out of scope |
| `openai.gpt-oss-20b-1:0` | **on-demand** | text | no — out of scope |
| `openai.gpt-oss-safeguard-120b` | **on-demand** | text | no — out of scope |
| `openai.gpt-oss-safeguard-20b` | **on-demand** | text | no — out of scope |

### xAI — 1 model

| modelId | Invocation | Modalities | In the notebook? |
|---|---|---|---|
| `xai.grok-4.6` | inference profile | text, image | no — out of scope |

Profile-only models need a geography prefix — `global.openai.gpt-6-astra`,
`us.xai.grok-4.6`. The bare ID returns a 400 asking for provisioned throughput, exactly as
it does for Claude. The four `gpt-oss` models are the exception: they're genuinely
on-demand, so the bare ID is correct for them.

> Worth noting for the Part 7 talk track: **every Anthropic model in this account is
> `INFERENCE_PROFILE` only — none supports `ON_DEMAND`.** Some OpenAI models do. So "bare
> IDs never work on Bedrock" is the wrong generalisation; "bare IDs never work *for Claude*"
> is the right one.

---

## The three access paths

| Path | Wire format | Works for non-Anthropic? |
|---|---|---|
| `AnthropicBedrock` → `/model/<id>/invoke` | Anthropic Messages | **No** — see below |
| `bedrock-runtime` **Converse** | provider-agnostic | **Yes**, all tested models |
| `POST /openai/v1/chat/completions` | OpenAI chat completions | Yes, incl. Claude IDs |

Converse was the right target and is what the notebook uses: one request shape, one response
shape, one `toolConfig`, and it covers Claude too, so the harness needs *one* code path rather
than one per provider.

The OpenAI-compatible endpoint also works and is handy if you're porting existing OpenAI
code, but it isn't a good fit here — it would mean writing the harness *away* from the
idiomatic Bedrock API to gain nothing the Converse path doesn't already give us.

### The silent failure

This is the finding that shaped the notebook, and it is worse than "it doesn't work". Pointing
`AnthropicBedrock` at a non-Anthropic model fails in **three different ways**, and only one of
them is any use to you:

| Model | Result via `AnthropicBedrock` |
|---|---|
| `global.openai.gpt-6-astra` | `BadRequestError: 400 unknown_parameter: 'anthropic_version'` |
| `openai.gpt-oss-120b-1:0` | HTTP **200**, `response.content is None` — no exception |
| `global.xai.grok-4.6` | **hangs**, then `APITimeoutError` after 120s — no exception until then |

The 400 is fine: loud, obvious, fixed in a minute. It is also the one the notebook's in-scope
model produces, which is why the notebook can demonstrate the lesson live without spending two
minutes hanging.

The 200 is the hazard. Bedrock accepts the Anthropic body, the model answers in its own
response shape, and the Anthropic SDK finds no `content` field it recognises — so it hands
back a `Message` with `content = None`. Note `None`, not `[]`. Downstream, `parse_transcript`
filters `response.content` for `TextBlock` instances, so an empty result isn't a crash, it's an
**empty transcript**. Every grader fails it. A sweep row reads `0/9` and looks like a broken
agent or a missing model grant, when the cause is a wire-format mismatch two layers up. Nobody
diagnoses that from a score.

> **Re-measured 2026-09-13, and this replaces what this file said before.** The earlier record
> here put `global.xai.grok-4.6` in the silent-200 row alongside `gpt-oss-120b`. On re-test it
> does not return 200 at all — it hangs until the client's own timeout fires. Same behaviour
> via Converse without `reasoning_config` (below), so it is a model-level property rather than
> an SDK one. The old row was either wrong or describes behaviour that has since changed; see
> [Open questions](#open-questions). The generalisable point survives either way, and is
> arguably stronger for it: **the number of ways this fails is not two, and you should not
> assume you have seen the last one.**

---

## Tool use does work — over Converse

The eval is worthless without tool calling, so this was the decisive test: one `get_product`
tool, `toolConfig`, prompt `"How much do jeans cost?"`. Re-run and extended on 2026-09-13 to
cover all three `toolChoice` modes and a full tool-result round trip.

| Model | Basic tool call | `toolChoice` auto / any / tool | Result round trip |
|---|---|---|---|
| `global.anthropic.claude-haiku-4-5-…` *(control)* | ✓ | ✓ ✓ ✓ | ✓ `{json}` |
| `global.openai.gpt-6-astra` | ✓ | ✓ ✓ ✓ | ✓ `{json}` |
| `global.openai.gpt-5.6-sol` | ✓ | ✓ ✓ ✓ | ✓ `{json}` |
| `global.openai.gpt-5.6-luna` | ✓ | ✓ ✓ ✓ | ✓ `{json}` |
| `global.openai.gpt-5.6-terra` | ✓ | ✓ ✓ ✓ | ✓ `{json}` |
| `openai.gpt-oss-120b-1:0` *(out of scope)* | ✓ | ✓ ✓ ✓ | ✓ |
| `openai.gpt-oss-20b-1:0` *(out of scope)* | ✓ | ✓ ✓ ✓ | ✓ |
| `openai.gpt-oss-safeguard-120b` *(out of scope)* | ✓ | ✓ ✓ ✓ | ✓ |
| `openai.gpt-oss-safeguard-20b` *(out of scope)* | ✓ | ✓ ✓ ✓ | ✓ |
| `global.xai.grok-4.6` *(out of scope)* | ✓ **with `reasoning_config`** | ✓ ✓ ✓ | ✓ `{json}` |
| `us.xai.grok-4.6` *(out of scope)* | ✓ **with `reasoning_config`** | ✓ ✓ ✓ | ✓ `{json}` |

Identical tool name and arguments everywhere. **A Converse harness works across all of them**,
which is what made the notebook worth building — and the surprise is how complete the parity
is. All three `toolChoice` modes and the `{json}` result round trip worked first try on every
model tested, including the open-weight ones we expected to lack tool use entirely.

**Two things that are worth more than the ticks:**

- The `gpt-oss` **"safeguard"** variants are not restricted moderation endpoints. They answer
  arbitrary questions, call tools and accept forced tool calls exactly like the non-safeguard
  models of the same size. The name does not describe a capability limit on Bedrock Converse.
  Their model IDs also have **no `:0` suffix** (`openai.gpt-oss-safeguard-120b`) where the
  non-safeguard ones do (`openai.gpt-oss-120b-1:0`) — the convention is inconsistent, so don't
  generate these IDs.
- **Grok 4.6 hangs forever without `additionalModelRequestFields={"reasoning_config": "low"}`**
  (or `medium` / `high` / `xhigh`). Not a 400, not a 500, no response headers — the TCP socket
  blocks indefinitely, confirmed by raw socket test and by timeouts up to 240s.
  `get_foundation_model` documents the field in `converse.additionalRequestFieldsSchema` and
  nothing in the error surface tells you it is mandatory. **This contradicts what this file
  recorded on 2026-09-12**, when Grok tool use over Converse was logged as working with no such
  note; see [Open questions](#open-questions).

### Gotcha: `maxTokens` has a floor on the newer models

The four `gpt-5.6`/`gpt-6` profiles reject small token budgets:

```
maxTokens=8  -> ValidationException: integer_below_min_value  ("Expected a value >= 16, but got 8")
maxTokens=12 -> ValidationException: integer_below_min_value  ("Expected a value >= 16, but got 12")
maxTokens=16 -> accepted
```

Grok 4.6 has the same floor of 16. The four `gpt-oss` models accept 8. Haiku 4.5 accepts 8.

This matters because a **cheap liveness ping is the standard trick for checking model access**
— the setup's one-token probe (in `workshop_setup.py`, and Anthropic-only for exactly this
reason) would report these models as broken when they are fully functional. The notebook's
Converse ping therefore uses `PING_MAX_TOKENS = 16`: a multi-model probe has to be sized for
the strictest model in the pool. These are reasoning models, so the floor is presumably
reserved thinking budget.

Note also that the error message says `max_output_tokens` — the provider's *native* parameter
name, not Converse's `maxTokens`. The provider's error schema leaks through Bedrock's
abstraction, which is worth knowing when you go searching for the string.

### The other parameter that doesn't survive: `temperature`

- **GPT 5.6 / GPT 6:** `This model doesn't support the temperature field. Remove temperature
  and try again.` Rejected at **every** value tested (0.0, 0.5, 1.0) — a blanket ban, not a
  range restriction.
- **Grok 4.6:** same message, same blanket ban.
- **`claude-sonnet-5`:** rejected too, with a different message —
  `` `temperature` is deprecated for this model``.
- **Haiku 4.5, and the `gpt-oss` four:** accept it fine.

So "omit `temperature`" is not an OpenAI workaround, it is where the field is heading
generally. The casualty is any harness that pins `temperature=0` for deterministic LLM-judge
verdicts — which is most of them. The notebook sets it nowhere and says out loud that it has
therefore given up the illusion of a determinism knob.

---

## What the port actually took

The six steps this file used to list as future work were all done. Here is what each one cost,
because the estimate was wrong in an interesting direction:

| Step | Estimated | Actual |
|---|---|---|
| 1. Swap the client to `bedrock-runtime` / `converse` | real work | ~30 lines, both auth paths reused from the existing setup (`workshop_setup.py`) |
| 2. Rewrite the tool schema | real work | **4 lines** (`to_converse_tool`), and the JSON Schema passes through untouched |
| 3. Rewrite `parse_transcript` | real work | one function; dict lookups replace `isinstance` checks |
| 4. Rewrite the judge | real work | two lines changed — the forced tool call is the same idea in different spelling |
| 5. Raise the probe budget to `maxTokens >= 16` | trivial | trivial, and load-bearing |
| 6. Re-baseline everything | **the expensive one** | **avoided entirely** |

**Step 6 is why this is a separate notebook.** The estimate was right that re-baselining
`FACILITATOR.md`'s expected failures against a Converse harness would be expensive, and wrong
that it had to happen: building a *standalone* notebook with its own six-task suite leaves
Parts 1-7 and every number in the facilitator guide untouched. The middle path this file used
to propose — a Converse adapter used only by Part 7 — would have been worse, because it puts a
second wire format inside a notebook whose Part 2 asks attendees to read the harness.

**What was actually hard was none of the code.** The notebook ran clean end to end on the first
execution attempt; zero code fixes were required. Every correction was to the *prose*, because
several claims that seemed safe turned out to be measurably wrong:

- "All four GPT models are identical" — false as stated. Identical *capability surface*, yes.
  But `gpt-5.6-terra` returned **no `reasoningContent` block** on 4 of 4 plain calls where
  `astra`, `sol` and `luna` each returned one on 4 of 4. Same API, different response content.
  This is a better finding than the claim it replaced.
- `reasoningContent` presence is not simply "always" or "never" — it varies turn by turn within
  one conversation, and `reasoning.effort` moves some turns but not the `toolUse` turn.
- Token counts are not comparable across providers, and the gap is large: identical turn and
  tool-call counts on all six tasks, and Haiku reported **~20,500 input tokens against GPT's
  8,198**. Two tokenizers counting two serialisations of the same conversation.
- The `reasoning` column in the results table reads `0` for GPT, which invites exactly the
  wrong conclusion. It is a property of the workload's turn shapes, not of the model.

The lesson for anyone repeating this: **budget for re-measuring your own claims, not for
writing the code.** The Converse port is a day. Being sure of what you then say about it is
longer.

### The "is it worth it" question, resolved

The argument for was always good: "does a *different vendor's* model earn its cost on my
workload?" is a more interesting question than comparing three Claude models. The argument
against was that Converse's dict-based transcripts are less readable than typed blocks in a
teaching context, and Part 2 already asks attendees to read the harness.

Both were right, which is why the answer was a **separate notebook** rather than a change to
the main one. The main workshop keeps the readable typed-block harness; the companion notebook
pays the readability cost on purpose, because in *that* notebook the wire format is the
subject rather than an obstacle.

What the notebook teaches that this investigation could not: the investigation established
that Converse *works*. The notebook establishes what it costs — that an eval suite's graders
are wire-format agnostic if you parse to an intermediate representation, and that the residual
incompatibilities are all small, all findable only by calling the model, and all in places
where the failure is quiet rather than loud.

---

## What stayed out, and why

The notebook's scope is **Claude versus OpenAI's GPT 5.6+ reasoning models, through one wire
format.** It is not a provider survey, and the honest framing of a two-provider comparison is
narrower than "multi-provider on Bedrock" suggests. So:

**xAI / Grok is excluded**, despite working. Adding it would make the notebook a provider
survey, which changes what it teaches: with two providers every difference is attributable, and
with three the reader is reading a table instead of learning a portability layer. Grok would
also drag in a genuinely interesting but off-topic hazard — the `reasoning_config` hang — that
deserves more than a footnote and would dominate the section it landed in. The data above is
the record; it is not a feature request.

**The four `gpt-oss` open-weight models are excluded**, despite passing every probe. Two
reasons. They are a different *kind* of model — open-weight, on-demand, text-only — so putting
them next to frontier reasoning profiles invites comparisons the suite cannot support. And one
of them is the source of the silent-200 failure, which is far more useful to the notebook as a
described hazard than as a scored column.

**Sonnet 5 is excluded** from the notebook's model list for the same reason: it is Claude, and
a second Claude model adds a row without adding a provider. Its `temperature` deprecation is
mentioned in the notebook precisely because it shows the field is going away everywhere rather
than being an OpenAI quirk.

None of these exclusions is about capability. All three families work.

---

## Open questions

Written down rather than resolved, because guessing here would be worse than admitting it.

1. **Did Grok's Converse behaviour change, or was the 2026-09-12 record wrong?** On
   2026-09-12 this file logged Grok tool use over Converse as working, with no mention of
   `reasoning_config`. On 2026-09-13 a Converse call without
   `additionalModelRequestFields={"reasoning_config": "low"}` hangs indefinitely. Both
   observations are in the record; they cannot both describe the same request. The likeliest
   explanations are that the original probe passed the field and the note omitted it, or that
   Bedrock's routing for that profile changed between the two dates. Nobody has re-run the
   original script to find out. **Do not quote either date's Grok behaviour as settled.**
2. **Does Sonnet 5 emit `reasoningContent` on tool-use turns?** Recorded on 2026-09-12 as
   doing so unrequested, with a `reasoningText.signature`, and with *no* text block —
   `[reasoningContent, toolUse]` where Haiku returns `[text, toolUse]`. On 2026-09-13 the same
   call returned `['toolUse']` only, with no reasoning block. **Not reproduced.** The notebook
   therefore does not make the claim. If it is real it is intermittent, which would make it
   nastier than if it were consistent.
3. **What is the earliest botocore that has `converse`?** `requirements.txt` floors boto3 at a
   comfortably recent version rather than a measured minimum. The relevant fact — that
   `anthropic[bedrock]`'s own floor predates the operation — is established; the exact boundary
   is not.

---

## Reproducing this

Needs a Bedrock API key or IAM credentials in `.env`, and `boto3`
(`pip install -r requirements.txt` covers it).

```python
import os, re, json, botocore.session
env = dict(re.findall(r"^([A-Za-z_]+)=(.*)$", open(".env").read(), re.M))
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = env["AWS_BEARER_TOKEN_BEDROCK"]
region = env["AWS_REGION"]

bd = botocore.session.get_session().create_client("bedrock", region_name=region)
for m in bd.list_foundation_models(byProvider="OpenAI")["modelSummaries"]:
    print(m["modelId"], m.get("inferenceTypesSupported"))

# tool use over Converse
rt = botocore.session.get_session().create_client("bedrock-runtime", region_name=region)
r = rt.converse(
    modelId="global.openai.gpt-6-astra",
    messages=[{"role": "user", "content": [{"text": "How much do jeans cost?"}]}],
    inferenceConfig={"maxTokens": 1024},
    # For global.xai.grok-4.6 add:  additionalModelRequestFields={"reasoning_config": "low"}
    # Without it the call hangs indefinitely rather than returning an error.
    toolConfig={"tools": [{"toolSpec": {
        "name": "get_product",
        "description": "Look up the price of a product in the store catalog.",
        "inputSchema": {"json": {"type": "object",
                                 "properties": {"product": {"type": "string"}},
                                 "required": ["product"]}}}}]})
print(r["stopReason"], json.dumps(r["output"]["message"]["content"], indent=2, default=str))
```

`default=str` in that `json.dumps` is not decoration: reasoning models return
`reasoningContent.redactedContent` as **`bytes`**, and plain `json.dumps` raises
`TypeError: Object of type bytes is not JSON serializable`. That failure lands at save time,
after you have paid for the run.

Swap `byProvider` for `"xAI"`, `"Meta"`, `"Mistral AI"` and so on to widen the survey. For a
much more thorough version of the above, with a live capability matrix and graceful degradation
when a model isn't granted, run `Multi_Provider_on_Bedrock.ipynb`.

---

## Odds and ends worth keeping

Measured along the way, useful, and not big enough for a section of their own.

- **`toolUseId` prefixes depend on the endpoint, not just the provider.** The same Haiku 4.5
  model returns `toolu_…` over the Anthropic Messages API and `tooluse_…` over Converse
  (measured 2026-09-13: `tooluse_1jZoNl7QgRcpqTP6wfP3LT`). OpenAI models return `call_` + 32
  hex characters. Converse treats the ID as opaque and round-trips it fine — the point is that
  any code which regexes or length-checks one of these breaks on the other.
- **Reasoning effort has exactly one accepted spelling.**
  `additionalModelRequestFields={"reasoning": {"effort": "high"}}` is accepted by the GPT
  profiles; `{"reasoning_effort": "high"}` returns `unknown_parameter`. Haiku 4.5 rejects
  *both*, naming whichever key you sent (`reasoning: Extra inputs are not permitted`,
  `reasoning_effort: Extra inputs are not permitted`) — so this is a field that cannot be
  shared across the two providers even in principle. Grok uses a third spelling again
  (`reasoning_config`, a bare string rather than a nested object).
- **`usage` field sets differ.** Haiku 4.5 and the GPT profiles both return `inputTokens`,
  `outputTokens`, `totalTokens`, `cacheReadInputTokens`. The `gpt-oss` four return only the
  first three. Grok returns six, including a duplicate (`cacheReadInputTokenCount` *and*
  `cacheReadInputTokens`) and an always-empty `serverToolUsage`. A strict schema validator over
  `usage` will fail on at least one provider.
- **No provider reports reasoning tokens.** Every model that demonstrably generated reasoning
  content reported no count of it in `usage`. Token-based cost estimates for reasoning models
  undercount by an amount the API will not tell you.
- **Reasoning tokens consume the `maxTokens` budget.** On Grok, `maxTokens=64` on "Say OK" spent
  all 64 on reasoning and produced **zero text**, `stopReason=max_tokens`. At 256 it used ~111
  and answered. A caller with a small budget gets an empty answer and no error.

---

## Anthropic model availability, same account

Recorded here because it's the other half of the same survey. 13 Anthropic foundation
models list; 11 are usable, each under both a `global.` and a `us.` profile (22 working
profile IDs). Verified by a real invoke, not by reading grants.

**The ten the workshop references, all usable:** Haiku 4.5 · Sonnet 4.5, 4.6, 5 · Opus 4.5,
4.6, 4.7, 4.8, 5 · Fable 5.1. (Not an exhaustive list of the usable eleven — it is the set the
workshop's own tables cover.)

**Not usable — lifecycle, not permissions.** Two are among the 13 that list:
`claude-sonnet-4-20250514` and `claude-opus-4-1-20250805`, both marked `LEGACY` (*"you have
not been actively using it"*). That is what reconciles the count — **13 listed − 2 legacy =
11 usable.**

Two more, `claude-3-haiku` and `claude-3-sonnet`, are end of life and **no longer appear in
`list_foundation_models` at all**, so they are not part of the 13. Worth knowing separately,
because an old runbook that names them looks like a permissions problem and isn't.
(Re-listed 2026-09-13: still 13 entries, 13 distinct model IDs.)

All four return **404**, not 403. No grant will fix them — don't send anyone to the Model
access page for these.
