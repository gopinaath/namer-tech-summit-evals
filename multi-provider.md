# Running this workshop on non-Anthropic Bedrock models

Investigation notes, not a shipped feature. **The notebook is deliberately Anthropic-only.**
This file records what a multi-provider version would take, and the Bedrock facts we
established along the way — several of which are useful in their own right.

All findings verified by real API calls from **one AWS account in `us-west-2`, 2026-09-12**,
using a Bedrock API key. Model availability is granted per account, so re-check on your own
account before quoting any of these numbers.

---

## Verdict

**Other providers work on Bedrock, and they do tool use. They do not work with this
notebook's client.** The harness is built on `anthropic.AnthropicBedrock`, which speaks the
Anthropic Messages wire format. Porting to non-Anthropic models means moving the harness to
Bedrock's **Converse** API — a real change to the agent loop, the transcript parser and the
judge, not a change to the model constants.

Adding a GPT or Grok ID to `MODEL`/`SWEEP_EXTRA` as things stand does **not** raise a clean
error. See [The silent failure](#the-silent-failure) — it's the reason this isn't a
five-minute change.

---

## What's available

18 providers list models in this account: Amazon, Anthropic, Cohere, DeepSeek, Google,
Luma AI, Meta, MiniMax, Mistral AI, Moonshot AI, NVIDIA, OpenAI, Qwen, Stability AI,
TwelveLabs, Writer, Z.AI, xAI.

### OpenAI — 8 models

| modelId | Invocation | Modalities |
|---|---|---|
| `openai.gpt-6-astra` | inference profile | text, image |
| `openai.gpt-5.6-sol` | inference profile | text, image |
| `openai.gpt-5.6-luna` | inference profile | text, image |
| `openai.gpt-5.6-terra` | inference profile | text, image |
| `openai.gpt-oss-120b-1:0` | **on-demand** | text |
| `openai.gpt-oss-20b-1:0` | **on-demand** | text |
| `openai.gpt-oss-safeguard-120b` | **on-demand** | text |
| `openai.gpt-oss-safeguard-20b` | **on-demand** | text |

### xAI — 1 model

| modelId | Invocation | Modalities |
|---|---|---|
| `xai.grok-4.6` | inference profile | text, image |

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

Converse is the right target for a multi-provider harness: one request shape, one response
shape, one `toolConfig`, and it covers Claude too, so the workshop wouldn't need two code
paths.

The OpenAI-compatible endpoint also works and is handy if you're porting existing OpenAI
code, but it isn't a good fit here — it would mean rewriting the harness *away* from the
idiomatic Bedrock API to gain nothing the Converse path doesn't already give us.

### The silent failure

This is the finding that matters. Pointing `AnthropicBedrock` at a non-Anthropic model does
**not** fail uniformly:

| Model | Result |
|---|---|
| `openai.gpt-oss-120b-1:0` | HTTP **200**, `response.content is None` — no exception |
| `global.xai.grok-4.6` | HTTP **200**, `response.content is None` — no exception |
| `global.openai.gpt-6-astra` | `BadRequestError: 400 unknown_parameter: 'anthropic_version'` |

The 400 is fine — loud and obvious. The 200s are the hazard: Bedrock accepts the Anthropic
body, the model answers, and the reply comes back in OpenAI's
`choices[0].message.content` shape. The Anthropic SDK finds no `content` field it
recognises and hands back a `Message` with `content = None`.

Downstream, `parse_transcript` filters on `isinstance(block, TextBlock)` over
`response.content` — so an empty result isn't a crash, it's an **empty transcript**. Every
grader fails it. A sweep row would read `0/9` and look like a broken agent or a missing
model grant, when the real cause is a wire-format mismatch two layers up. Nobody would
diagnose that in a workshop.

---

## Tool use does work — over Converse

The eval is worthless without tool calling, so this was the decisive test: one
`get_product` tool, `toolConfig`, prompt `"How much do jeans cost?"`.

| Model | Result |
|---|---|
| `global.anthropic.claude-haiku-4-5-20251001-v1:0` *(control)* | ✓ `get_product({"product":"jeans"})`, `stopReason=tool_use` |
| `global.xai.grok-4.6` | ✓ same |
| `us.xai.grok-4.6` | ✓ same |
| `global.openai.gpt-5.6-sol` | ✓ same |
| `global.openai.gpt-5.6-luna` | ✓ same |
| `global.openai.gpt-6-astra` | ✓ same |

Six for six, identical tool name and arguments. A Converse-based harness would work across
all of them.

### Gotcha: `maxTokens` has a floor on the newer models

The four `gpt-5.6`/`gpt-6` profiles reject small token budgets:

```
maxTokens=12 -> ValidationException: integer_below_min_value  ("Invalid 'max_output...")
maxTokens=16 -> accepted
```

The `gpt-oss` models accept 12 happily. This matters because a **cheap liveness ping is the
standard trick for checking model access** — the setup cell's one-token probe would report
these models as broken when they're fine. Any multi-provider version needs `maxTokens >= 16`
on its probe. These are reasoning models; the floor is presumably reserved thinking budget.

---

## What a multi-provider version would actually take

Roughly, in dependency order:

1. **Swap the client.** `anthropic.AnthropicBedrock` → `boto3.client("bedrock-runtime")`,
   `messages.create(...)` → `converse(...)`. Drops the `anthropic` dependency but makes
   boto3 mandatory, so `requirements.txt` and both auth paths in `SETUP.md` change.
2. **Rewrite the tool schema.** Anthropic's `{"name", "description", "input_schema"}` becomes
   Converse's `{"toolSpec": {"name", "description", "inputSchema": {"json": ...}}}`.
3. **Rewrite `parse_transcript`.** Converse returns dicts (`content[i]["toolUse"]`,
   `["text"]`), not typed `TextBlock`/`ToolUseBlock` objects, so every `isinstance` check goes.
4. **Rewrite the judge.** Part 6's forced tool call is
   `tool_choice={"type":"tool","name":"submit_verdict"}`; Converse spells it
   `toolConfig={"toolChoice": {"tool": {"name": "submit_verdict"}}}`.
5. **Raise the probe budget** to `maxTokens >= 16` (above).
6. **Re-baseline everything.** `FACILITATOR.md` documents exact expected failures — which
   probes fail, `3 of 6` passing, `50% → 100%`. Those are measured against Haiku on the
   Anthropic path and would all need re-measuring.

Step 6 is the expensive one, and it's why this is a deliberate project rather than an
afternoon. The workshop's credibility rests on the facilitator guide predicting what
attendees will actually see.

### Is it worth it?

There's a genuine argument for it: "does a *different vendor's* model earn its cost on my
workload?" is a more interesting question than comparing four Claude models, and the answer
would be evidence rather than opinion — which is the whole thesis of the workshop.

The argument against, and the reason we didn't: this is a **Claude on Amazon Bedrock**
workshop with a 90-minute budget, and Converse's dict-based transcripts are noticeably less
readable than typed blocks in a teaching context. Part 2 already asks attendees to read the
harness. Making that code worse to gain a sweep row is a bad trade for *this* session.

A middle path, if the sweep is the only reason you want it: keep the Anthropic SDK for the
agent, and add a Converse adapter used **only** by Part 7. Contained, no re-baselining of
Parts 1-6, one extra function. That's the version to build first if someone asks.

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
    toolConfig={"tools": [{"toolSpec": {
        "name": "get_product",
        "description": "Look up the price of a product in the store catalog.",
        "inputSchema": {"json": {"type": "object",
                                 "properties": {"product": {"type": "string"}},
                                 "required": ["product"]}}}}]})
print(r["stopReason"], json.dumps(r["output"]["message"]["content"], indent=2))
```

Swap `byProvider` for `"xAI"`, `"Meta"`, `"Mistral AI"` and so on to widen the survey.

---

## Anthropic model availability, same account

Recorded here because it's the other half of the same survey. 13 Anthropic foundation
models list; 11 are usable, each under both a `global.` and a `us.` profile (22 working
profile IDs). Verified by a real invoke, not by reading grants.

**Usable:** Haiku 4.5 · Sonnet 4.5, 4.6, 5 · Opus 4.5, 4.6, 4.7, 4.8, 5 · Fable 5, 5.1

**Not usable — lifecycle, not permissions:** `claude-3-haiku`, `claude-3-sonnet` (end of
life), `claude-sonnet-4-20250514`, `claude-opus-4-1-20250805` (marked `LEGACY`, *"you have
not been actively using it"*).

Those return **404**, not 403. No grant will fix them — don't send anyone to the Model
access page for these.
