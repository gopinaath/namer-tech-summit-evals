# Percent-format source for Multi_Provider_on_Bedrock.ipynb.
# Rebuild with:  python tools/nbbuild.py

# %% [markdown]
# # Multi-Provider Evals on Bedrock: Claude and OpenAI GPT
#
# ### NAMER Tech Summit · companion notebook · Amazon Bedrock
#
# One eval suite, one agent, two providers, one wire format. Bedrock's **Converse** API
# gives you a single request and response shape for every model it hosts — so in principle
# the eval you built in `Building_an_Eval.ipynb` should run against a non-Anthropic model
# with no changes at all.
#
# In practice it takes six specific fixes, and this notebook is about those six.
#
# | | |
# |---|---|
# | **What this is** | Claude Haiku 4.5 vs OpenAI's GPT 5.6 / GPT 6 reasoning models, through `bedrock-runtime`'s Converse API |
# | **What this is not** | A survey of Bedrock's providers, a model leaderboard, or a re-teaching of the eval framework |
# | **What it teaches** | The portability layer: what breaks when you point one harness at two providers, and how each break presents |
# | **Prerequisite** | `Building_an_Eval.ipynb` — tasks, graders, runner, judge. This notebook assumes them and ports them. |
#
# **Scope, stated narrowly on purpose.** Two providers. One Anthropic control
# (`global.anthropic.claude-haiku-4-5-20251001-v1:0`) and one OpenAI reasoning model
# (`global.openai.gpt-6-astra`). The control earns its place twice: it is the baseline the
# GPT column is read against, and it is what makes the silent-failure demonstration in the
# next section legible at all.
#
# **Why one GPT model and not four.** This account can reach four: `gpt-6-astra`,
# `gpt-5.6-sol`, `gpt-5.6-luna`, `gpt-5.6-terra`. On every *capability* probe in this
# notebook they answer identically — same `temperature` rejection word for word, same
# `maxTokens` floor of 16, same three `toolChoice` modes, same `call_` + 32 hex tool-call
# IDs (re-checked on all four, 2026-09-13). Four identical table rows is not a finding,
# it's a bill. So one runs by default and the other three are one uncommented line away.
#
# They are not, however, interchangeable. `gpt-5.6-terra` returned **no `reasoningContent`
# block at all** on four plain calls, where `astra`, `sol` and `luna` each returned one on
# four out of four. Same API surface, different response content — which is the whole
# argument for probing the model you are going to call rather than the family it belongs to.
#
# **Cost.** About 70 model calls on the default two-model configuration, most of them tiny:
# four Anthropic setup pings, two Converse liveness pings, one deliberate failure, two
# side-by-side calls, sixteen capability probes, four `additionalModelRequestFields` probes, a
# two-turn smoke test, **34 agent turns** for six tasks × two models, and six judge fixtures. No
# dollar figure appears anywhere in this notebook — per-token rates move, and a hardcoded
# price goes stale silently. Token counts are printed; multiply them yourself against the
# current Bedrock pricing page.
#
# > Everything measured here comes from **one AWS account in `us-west-2`**, probed
# > 2026-09-12 and re-measured 2026-09-13. Model access is granted per account and per
# > model, and provider feature support drifts. Every table in the prose is dated for that
# > reason, and the notebook rebuilds the important one live rather than trusting it.

# %% include=setup
# Kernel/install guard, Bedrock credentials, model IDs and the anthropic `client`.
# Shared verbatim with Building_an_Eval.ipynb via tools/shared/setup.py.
# The anthropic client is used in exactly one cell here — the silent-failure demo.
# Everything else goes through the Converse client built in the next cell.

# %% [markdown]
# ---
#
# ## The Converse client
#
# The main workshop talks to Bedrock through `anthropic.AnthropicBedrock`, which speaks the
# Anthropic Messages wire format. That is the right client for a Claude-only workshop and
# the wrong one the moment a second provider appears.
#
# `bedrock-runtime`'s **Converse** API is the provider-agnostic path: one request shape, one
# response shape, one `toolConfig`, and it covers Claude too — so a Converse harness needs
# *one* code path, not one per provider. That last point is why we don't reach for
# Bedrock's OpenAI-compatible `/openai/v1/chat/completions` endpoint instead: it also works,
# but it would mean writing the harness away from the idiomatic Bedrock API to gain nothing.
#
# The credentials are already resolved — the setup cell above worked out the region, the
# auth mode and the profile. We reuse all three rather than re-deriving them, and both auth
# paths are supported: a Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK`, which botocore picks
# up from the environment on its own) or IAM (SSO, `aws configure`, an assumed role, with
# `AWS_PROFILE` honoured if set).
#
# > **The liveness ping is `maxTokens=16`, not `maxTokens=1`.** The one-token ping in the
# > setup cell is the standard cheap trick for "can this account reach this model", and it
# > reports the GPT reasoning models as **unreachable when they are fully functional** —
# > they reject any budget below 16. First portability bug, and it lands before you have
# > even run a task. A multi-model probe has to be sized for the strictest model in the
# > pool.

# %%
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# The kernel-safety guard in the setup cell above already ran and passed, so installing
# into this interpreter is exactly as safe here as it was there.
if importlib.util.find_spec("botocore") is None:
    print("Installing boto3 (Converse needs it) — first run only…", flush=True)
    _pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "boto3"],
                          capture_output=True, text=True)
    if _pip.returncode:
        raise SystemExit("pip install boto3 failed:\n"
                         + (_pip.stderr or _pip.stdout)[-500:])

import botocore.session
from botocore.config import Config
from botocore.exceptions import ClientError

# ── Client ────────────────────────────────────────────────────────────────────

def make_converse_client(read_timeout=900, max_attempts=5):
    """A `bedrock-runtime` client for whichever credential type the setup cell found.

    Two auth paths, same as `make_client()` above:

    * **api-key** — botocore reads `AWS_BEARER_TOKEN_BEDROCK` out of the environment and
      signs with it. Nothing is passed in explicitly, which also means the token never
      appears in this notebook's variables or output.
    * **iam** — a normal botocore session, honouring `AWS_PROFILE` when one is named.

    Retries are generous for the same reason the Anthropic client's are: Bedrock throttles
    per account and per model, and a throttled request that exhausts its retries shows up
    as a task ERROR that looks like an agent bug."""
    if AUTH_MODE == "api-key":
        os.environ.setdefault("AWS_BEARER_TOKEN_BEDROCK", _token)
        session = botocore.session.get_session()
    else:
        session = botocore.session.Session(profile=_profile or None)
    return session.create_client(
        "bedrock-runtime", region_name=_region,
        config=Config(read_timeout=read_timeout, connect_timeout=20,
                      retries={"max_attempts": max_attempts, "mode": "standard"}))


brt = make_converse_client()

# ── The models ────────────────────────────────────────────────────────────────
# The Anthropic control is the same ID the setup cell already verified, so this notebook
# can't drift from the main workshop's baseline model.
HAIKU = FAST_MODEL

# One OpenAI reasoning model by default. The other three answer every capability probe in
# this notebook identically — same errors word for word, same maxTokens floor, same toolChoice
# support (re-checked 2026-09-13) — so running them quadruples the GPT half of the bill to
# print four identical rows. Uncomment to widen the sweep; they are not interchangeable in
# every respect, and the meta-finding at the end of the notebook says where they diverge.
GPT_MODELS = {"gpt-6-astra": "global.openai.gpt-6-astra"}
# GPT_MODELS.update({"gpt-5.6-sol":   "global.openai.gpt-5.6-sol",
#                    "gpt-5.6-luna":  "global.openai.gpt-5.6-luna",
#                    "gpt-5.6-terra": "global.openai.gpt-5.6-terra"})

# These are inference profiles, so the geography prefix is part of the ID. A bare
# `openai.gpt-6-astra` returns a 400 asking for provisioned throughput, exactly as a bare
# `anthropic.claude-sonnet-5` does.
CANDIDATES = {"haiku-4-5": HAIKU, **GPT_MODELS}

PING_MAX_TOKENS = 16      # 8 and 12 are rejected by the GPT models; see the note above


def short_name(model_id):
    """Display name. Extends the setup cell's short_model_name() to OpenAI IDs."""
    return short_model_name(model_id).replace("openai.", "")


def ping_converse(model_id):
    """(reachable, reason). A model this account can't use is a 400, not a 404, so an
    exception here means 'not usable today' rather than 'something is broken'."""
    try:
        brt.converse(modelId=model_id,
                     messages=[{"role": "user", "content": [{"text": "ping"}]}],
                     inferenceConfig={"maxTokens": PING_MAX_TOKENS})
        return True, "ok"
    except ClientError as exc:
        return False, exc.response.get("Error", {}).get("Code", "ClientError")
    except Exception as exc:
        return False, type(exc).__name__


CONVERSE_MODELS, _unreachable = {}, {}
for _name, _id in CANDIDATES.items():
    _ok, _why = ping_converse(_id)
    (CONVERSE_MODELS if _ok else _unreachable)[_name] = _id if _ok else _why
    print(f"  {'✓' if _ok else '✗'} {_name:<14} {_id}{'' if _ok else f'  ({_why})'}")

print(f"\nConverse ready — {_region}, auth: {AUTH_MODE}. "
      f"{len(CONVERSE_MODELS)} of {len(CANDIDATES)} candidate models reachable.")
if not CONVERSE_MODELS:
    raise SystemExit("No models reachable over Converse — check the pings above.")
if len(CONVERSE_MODELS) == 1:
    print(f"Only {list(CONVERSE_MODELS)[0]} is reachable, so everything below runs as the "
          f"DEGENERATE SINGLE-PROVIDER CASE: a working eval, no cross-provider comparison. "
          f"Model access is granted per account and per model in the Bedrock console; these "
          f"are global inference profiles, so changing AWS_REGION will not help.")

DEFAULT_MODEL = CONVERSE_MODELS[list(CONVERSE_MODELS)[0]]

# %% [markdown]
# ---
#
# ## The silent failure — why the client has to change first
#
# Before porting anything, it's worth seeing what happens if you *don't*. The obvious
# five-minute version of this notebook is to add a GPT model ID to the main workshop's
# `SWEEP_EXTRA` and re-run Part 7. That doesn't fail cleanly, and the way it fails is the
# reason this notebook exists.
#
# `AnthropicBedrock` sends an Anthropic Messages body. Bedrock sometimes translates it for a
# non-Anthropic model and sometimes doesn't, and there are three measured outcomes:
#
# | What you get | Why it's bad |
# |---|---|
# | **400 with a clear message** | Fine. Loud, obvious, fixed in a minute. |
# | **HTTP 200 with `response.content = None`** | The model answered in its own response shape, the SDK found no `content` it recognised, and returned a `Message` with `None`. No exception. |
# | **A hang, then a timeout** | Also no exception until the timeout fires. |
#
# The middle one is the hazard. `parse_transcript()` filters `response.content` for
# `TextBlock` instances, and filtering `None` gives you an empty transcript, not a crash —
# so every grader fails, the sweep row reads `0/N`, and it looks exactly like a broken agent
# or a missing model grant. Nobody diagnoses a wire-format mismatch from that.
#
# The cell below runs the loud version, because it's the one this notebook's in-scope model
# produces. The silent version was measured on an out-of-scope on-demand OpenAI model on the
# same account and is not run here.

# %%
_gpt = next((i for n, i in CONVERSE_MODELS.items() if n != "haiku-4-5"), None)

if _gpt is None:
    print("No GPT model reachable on this account, so there is nothing to mis-address. "
          "The lesson still holds: the client you use has to speak the wire format the "
          "model speaks, and Bedrock does not guarantee it will tell you when it doesn't.")
else:
    print(f"Calling {short_name(_gpt)} through anthropic.AnthropicBedrock…\n")
    try:
        _bad = client.messages.create(
            model=_gpt, max_tokens=PING_MAX_TOKENS,
            messages=[{"role": "user", "content": "How much do jeans cost?"}])
        # Reached only if Bedrock accepted the Anthropic body. Then the question is
        # whether anything usable came back — and `content is None` is the trap.
        print(f"  HTTP 200. stop_reason={_bad.stop_reason!r}  content={_bad.content!r}")
        if not _bad.content:
            print("\n  ↑ THIS IS THE SILENT FAILURE. No exception, no error field, and an "
                  "empty\n    content list. Every grader downstream fails and none of them "
                  "can say why.")
    except Exception as _exc:
        print(f"  {type(_exc).__name__}: {str(_exc)[:400]}")
        print("\n  ↑ The loud version. `anthropic_version` is an Anthropic-Messages field; "
              "this\n    model has never heard of it. Note what is NOT wrong here: the "
              "model. It answers\n    this exact question correctly over Converse a few "
              "cells from now. The SDK is the\n    problem, not the model.")

# %% [markdown]
# > **The transferable habit:** the absence of an exception is not evidence that a call
# > succeeded. When you point an existing harness at a new model, assert on the *content* of
# > the first response, not just on the status. An eval harness whose failure mode is
# > "scores zero, silently" is worse than one that crashes.
#
# ---
#
# ## Translating what you already have
#
# If you have Anthropic-style tool specs — and if you did the main workshop, you do — this
# is the only piece of translation you have to write. It is mechanical, it is four lines of
# code, and it is the single most reusable thing in this notebook.
#
# | Anthropic Messages | Converse |
# |---|---|
# | `messages=[{"role": "user", "content": "text"}]` | `messages=[{"role": "user", "content": [{"text": "text"}]}]` |
# | `system="..."` | `system=[{"text": "..."}]` |
# | `max_tokens=1024` | `inferenceConfig={"maxTokens": 1024}` |
# | `tools=[{"name", "description", "input_schema"}]` | `toolConfig={"tools": [{"toolSpec": {"name", "description", "inputSchema": {"json": ...}}}]}` |
# | `tool_choice={"type": "tool", "name": "x"}` | `toolConfig={"toolChoice": {"tool": {"name": "x"}}}` |
# | `response.content` → typed blocks | `response["output"]["message"]["content"]` → dicts |
# | `block.type == "tool_use"` | `"toolUse" in block` |
# | `{"type": "tool_result", "tool_use_id":…, "content": str}` | `{"toolResult": {"toolUseId":…, "content": [{"json": {...}}]}}` |
# | `response.usage.input_tokens` | `response["usage"]["inputTokens"]` |
#
# Nothing in that table is conceptually new. It is renaming, one level of extra nesting, and
# dicts where the SDK gave you typed objects.

# %%
def to_converse_tool(spec):
    """Anthropic-style tool spec -> Converse toolSpec. THE takeaway of this notebook.

    Takes the `{"name", "description", "input_schema"}` shape used by
    `anthropic.messages.create(tools=...)` — the shape the main workshop's
    `ALL_TOOL_SPECS_V2` already uses — and returns Converse's
    `{"toolSpec": {..., "inputSchema": {"json": ...}}}`.

    The JSON Schema itself is untouched: `enum`, `required`, nested objects, descriptions
    all carry over verbatim. So if you already have Anthropic tool specs, you do not
    rewrite them to go multi-provider — you wrap them, once, here."""
    return {"toolSpec": {"name": spec["name"],
                         "description": spec["description"],
                         "inputSchema": {"json": spec["input_schema"]}}}


def converse_text_blocks(message):
    """The text of one Converse message, joined. Filters by block TYPE, never by index —
    the next section shows why that distinction has teeth."""
    return "\n".join(b["text"] for b in message["content"] if "text" in b)


_demo_spec = {"name": "get_product",
              "description": "Look up the price of a product in the store catalog.",
              "input_schema": {"type": "object",
                               "properties": {"product": {"type": "string"}},
                               "required": ["product"]}}
print("Anthropic spec ->  " + json.dumps(_demo_spec))
print("Converse spec  ->  " + json.dumps(to_converse_tool(_demo_spec)))

# %% [markdown]
# ---
#
# ## One call, two providers
#
# The same question, the same request, both models. Watch the **content blocks**, not the
# prose: that is where the providers diverge, and it is where a harness breaks.

# %%
for _name, _id in CONVERSE_MODELS.items():
    _r = brt.converse(modelId=_id,
                      messages=[{"role": "user",
                                 "content": [{"text": "How much do jeans cost?"}]}],
                      system=[{"text": "Answer in one short sentence."}],
                      inferenceConfig={"maxTokens": 300})
    _blocks = _r["output"]["message"]["content"]
    print(f"\n{'─' * 78}\n{_name:<14} stopReason={_r['stopReason']}  "
          f"usage={dict(sorted(_r['usage'].items()))}")
    print(f"  content block types: {[list(b)[0] for b in _blocks]}")
    for _b in _blocks:
        _kind = list(_b)[0]
        if _kind == "text":
            print(f"    [text] {' '.join(_b['text'].split())[:150]}")
        elif _kind == "reasoningContent":
            # `.get`, not `[...]`: `reasoningContent` also has a `reasoningText` form (that
            # is what Claude's extended thinking returns), and indexing straight into
            # `redactedContent` would crash the cell on a model that used it.
            _red = _b["reasoningContent"].get("redactedContent")
            print(f"    [reasoningContent] keys={sorted(_b['reasoningContent'])} "
                  + (f"redactedContent is {type(_red).__name__}, {len(_red)} long — "
                     "encrypted, not readable" if _red is not None
                     else "no redactedContent — this model returned reasoning some other way"))
        else:
            print(f"    [{_kind}] {json.dumps(_b[_kind], default=str)[:150]}")

    # The concrete bug this shape causes, demonstrated rather than asserted.
    try:
        print(f"  content[0]['text'] -> {_blocks[0]['text'][:40]!r}")
    except KeyError as _exc:
        print(f"  content[0]['text'] -> KeyError({_exc}) — the first block is not text")
    # And the second one, which bites at save time rather than at parse time.
    try:
        json.dumps(_blocks)
        print("  json.dumps(content) -> ok")
    except TypeError as _exc:
        print(f"  json.dumps(content) -> TypeError: {_exc}")

# %% [markdown]
# ### Two things that just happened
#
# **The envelope is identical; the contents are not.** Same `stopReason`, same
# `output.message.content` list, same four `usage` fields (`inputTokens`, `outputTokens`,
# `totalTokens`, `cacheReadInputTokens`). What differs is that the GPT reasoning models can
# emit a `reasoningContent` block, and when they do it arrives **before** the text. So
# `content[0]["text"]` works on Claude and raises `KeyError` on GPT.
#
# Worse than the exception: **presence varies turn by turn inside a single conversation**, and
# you cannot work out which turns from the request alone. Measured on `gpt-6-astra`,
# 2026-09-13:
#
# | Turn | `reasoningContent`? |
# |---|---|
# | plain question, no `toolConfig` | yes (4/4 samples) |
# | plain question, `reasoning.effort=high` | yes |
# | `toolConfig` sent, model answered without calling a tool | yes |
# | the turn that emitted `toolUse` | **no** — at default effort *and* at `high` |
# | the final answer after a `toolResult`, default effort | **no** |
# | the final answer after a `toolResult`, `reasoning.effort=high` | yes |
#
# So the same model, the same conversation, gives you a leading `reasoningContent` block on
# some turns and not others — which rules out both "index 0 is reasoning" and "index 0 is
# text". **Filter by block type. Never by position.** A parser that filters by type is
# correct on both providers and needs no branch on the model ID or the turn number.
#
# Note that the request is not *irrelevant* — `reasoning.effort` is one lever that moves the
# answer, and the last two rows are the same turn with and without it. But it only moves some
# of the turns: at `high` the `toolUse` turn still has no reasoning block. So the request
# narrows the possibilities and does not determine them, which is the worst case for a parser.
# You cannot compute the block layout ahead of the call from either side.
#
# **`redactedContent` is `bytes`, not a base64 string.** botocore decodes the Bedrock blob
# for you, which means `json.dumps(transcript)` raises
# `TypeError: Object of type bytes is not JSON serializable`. This one does not fail at
# parse time, or at run time, or on the first eight tasks — it fails when you go to *save*
# your results, after you have paid for the whole run. The fix is one line in the transcript
# sanitiser below, but you have to know it's coming.
#
# Note also what is *not* in `usage`: any count of reasoning tokens. The model generated
# reasoning content and the usage dict reports `inputTokens` / `outputTokens` /
# `totalTokens` / `cacheReadInputTokens` and nothing else. Token-based cost estimates for
# reasoning models undercount, and the API gives you no way to tell by how much.

# %% [markdown]
# ---
#
# ## The capability matrix, built live
#
# A shared wire format is necessary and not sufficient. Converse guarantees the shape of the
# request; it guarantees nothing about which *parameters* a given model honours. So probe
# the model you are actually going to call, with the parameters you are actually going to
# pass — the same discipline as the setup cell pinging each model instead of trusting that a
# valid credential implies access.
#
# Here is the matrix as measured on **one account in `us-west-2`**: all four GPT profiles on
# 2026-09-12, re-measured on 2026-09-13 (every row on `gpt-6-astra`; the `plain`,
# `toolChoice`, `temperature` and `maxTokens` rows on all four). Treat it as an example of
# the *shape* of the answer, not as the answer:
#
# | Probe | Haiku 4.5 | GPT 5.6 / GPT 6 (all four) |
# |---|---|---|
# | plain `converse` | ok | ok |
# | `system` prompt | ok | ok |
# | tool use | ok | ok |
# | `toolChoice` auto / any / tool | ok | ok |
# | tool-result round trip (`json` content) | ok | ok |
# | `temperature=0.0` | ok | **rejected — not a range limit, a blanket ban** |
# | `maxTokens=8` | ok | **rejected — floor is 16** |
#
# The next cell rebuilds that table against your own account and region, because per-account
# grants drift and per-model support drifts faster. Any model that fails a probe the harness
# genuinely needs is **dropped with a printed reason** rather than being left in to fill the
# results table with errors.

# %%
PROBE_TOOL = to_converse_tool(_demo_spec)
_probe_msg = [{"role": "user", "content": [{"text": "How much do jeans cost?"}]}]
_tools = {"tools": [PROBE_TOOL]}

# Each probe is (label, converse kwargs). Nothing in them depends on the model — that is
# the point: one request, every provider, and the differences fall out as failures.
PROBES = [
    ("plain",    {"messages": _probe_msg, "inferenceConfig": {"maxTokens": 64}}),
    ("system",   {"messages": _probe_msg, "system": [{"text": "Be brief."}],
                  "inferenceConfig": {"maxTokens": 64}}),
    ("tool_use", {"messages": _probe_msg, "inferenceConfig": {"maxTokens": 512},
                  "toolConfig": _tools}),
    ("tc:auto",  {"messages": _probe_msg, "inferenceConfig": {"maxTokens": 512},
                  "toolConfig": {**_tools, "toolChoice": {"auto": {}}}}),
    ("tc:any",   {"messages": _probe_msg, "inferenceConfig": {"maxTokens": 512},
                  "toolConfig": {**_tools, "toolChoice": {"any": {}}}}),
    ("tc:tool",  {"messages": _probe_msg, "inferenceConfig": {"maxTokens": 512},
                  "toolConfig": {**_tools,
                                 "toolChoice": {"tool": {"name": "get_product"}}}}),
    ("temp=0",   {"messages": _probe_msg,
                  "inferenceConfig": {"maxTokens": 64, "temperature": 0.0}}),
    ("maxTok=8", {"messages": _probe_msg, "inferenceConfig": {"maxTokens": 8}}),
]

PROBE_ERRORS = []


def probe(model_id, **kwargs):
    """Run one Converse call. Returns (status, response), status "ok" or a short reason.

    Short matters: the status goes in a table cell. The full message is kept in
    PROBE_ERRORS so a surprising result can be read verbatim without re-running."""
    try:
        return "ok", brt.converse(modelId=model_id, **kwargs)
    except ClientError as exc:
        error = exc.response.get("Error", {})
        message = error.get("Message", str(exc))
        PROBE_ERRORS.append((model_id, kwargs, message))
        if "temperature" in message:
            return "rejected: temperature", None
        if "integer_below_min_value" in message or "below minimum" in message:
            return "rejected: below min", None
        return f"rejected: {error.get('Code', 'error')}", None
    except Exception as exc:
        PROBE_ERRORS.append((model_id, kwargs, str(exc)))
        return f"failed: {type(exc).__name__}", None


CAPS, TOOL_IDS = {}, {}
for _name, _id in CONVERSE_MODELS.items():
    CAPS[_name] = {}
    for _probe_name, _kwargs in PROBES:
        # `_probe_status`, not `_status`: this is one probe's outcome string, not the status of
        # anything global, and the longer name says so at every use below.
        _probe_status, _resp = probe(_id, **_kwargs)
        CAPS[_name][_probe_name] = _probe_status
        # Record the tool-call ID format while we have a response that contains one:
        # it differs between providers and is worth seeing rather than being told.
        if _resp is not None and _probe_name.startswith("tc:"):
            for _b in _resp["output"]["message"]["content"]:
                if "toolUse" in _b:
                    TOOL_IDS.setdefault(_name, _b["toolUse"]["toolUseId"])

_cols = [p[0] for p in PROBES]
print(f"{'model':<14}" + "".join(f"{c:>22}" for c in _cols))
print("─" * (14 + 22 * len(_cols)))
for _name in CONVERSE_MODELS:
    print(f"{_name:<14}" + "".join(f"{CAPS[_name][c]:>22}" for c in _cols))

print("\ntoolUseId formats observed (Converse passes these back opaquely):")
for _name, _tid in TOOL_IDS.items():
    print(f"  {_name:<14} {_tid}")

# Graceful degradation. These three probes are what the harness below cannot work without;
# everything else it can route around. A model that fails one drops out with a reason
# rather than contributing a column of ERRORs to the results table.
REQUIRED = ["plain", "tool_use", "tc:auto"]
EVAL_MODELS = {}
for _name, _id in CONVERSE_MODELS.items():
    _missing = [p for p in REQUIRED if CAPS[_name][p] != "ok"]
    if _missing:
        print(f"\nDROPPED {_name}: failed required probe(s) {', '.join(_missing)}. The eval "
              f"below needs tool use, so this model cannot be scored by this harness.")
    else:
        EVAL_MODELS[_name] = _id

if EVAL_MODELS:
    # Point the agent's default at a model that passed the probes, not merely one that
    # answered a ping.
    DEFAULT_MODEL = list(EVAL_MODELS.values())[0]

print(f"\nEvaluating: {', '.join(EVAL_MODELS) or '(none)'}")
if PROBE_ERRORS:
    print(f"\n{len(PROBE_ERRORS)} probe(s) were rejected. Verbatim, because the exact "
          f"wording is what you will search for later:")
    for _m, _kw, _msg in PROBE_ERRORS:
        print(f"\n  {short_name(_m)}  {json.dumps(_kw.get('inferenceConfig', {}))}\n"
              f"    {' '.join(_msg.split())[:300]}")

# %% [markdown]
# ### Read your own table, then read this
#
# Three things in that matrix contradict a reasonable assumption, which is what makes them
# worth the API calls:
#
# **1. `temperature` is not range-limited on the GPT reasoning models, it is banned.**
# Not "0.0 is invalid, use 0.7" — the field itself is refused, at every value. The main
# workshop's judge sets `temperature` nowhere, but plenty of harnesses pin `temperature=0`
# for determinism, and every one of those raises `ValidationException` here. Note the
# direction of the surprise: the *newer* model is the one that supports *less*. Feature
# availability tracks recency and provider policy, not capability.
#
# Claude is not immune either. On this account `claude-sonnet-5` rejects `temperature` too,
# with a different message — `` `temperature` is deprecated for this model`` — while
# Haiku 4.5 still accepts it. So "omit temperature" is not an OpenAI workaround, it is
# where the whole field is heading. Sonnet 5 is deliberately out of scope here (this
# notebook is Claude vs GPT, not a survey of reasoning models), but if you add it to
# `CANDIDATES` that is the first thing you will hit.
#
# **2. The `maxTokens` floor is 16.** Both 8 and 12 are rejected with
# `integer_below_min_value`. The damage is not to your agent — 1024 is fine — it is to your
# *probes*. Every cheap-ping pattern in every harness assumes it can ask for one token.
#
# **3. Tool use ports completely.** All three `toolChoice` modes, the tool-result round trip
# with `json` content, parallel tool calls in one turn. This is the finding that makes the
# notebook practical rather than a list of complaints: the hard part of an agent harness —
# the tool-use protocol — is genuinely portable. What isn't portable is the small stuff
# around it.

# %% [markdown]
# ### The labelled exit from portability
#
# Converse has one field for parameters it does not model: `additionalModelRequestFields`,
# passed through to the provider verbatim. It is the correct place to put anything
# provider-specific — and by construction it is the one part of a Converse request that is
# guaranteed *not* to port.
#
# This notebook never sets it. The cell below sends it twice per model anyway, because
# "reasoning effort is controllable" is a claim worth measuring rather than repeating, and
# because both failures are more interesting than the success.

# %%
_EFFORT_FORMS = [("nested  {'reasoning': {'effort': 'high'}}", {"reasoning": {"effort": "high"}}),
                 ("flat    {'reasoning_effort': 'high'}", {"reasoning_effort": "high"})]

for _name, _id in CONVERSE_MODELS.items():
    print(f"\n{_name}")
    for _label, _extra in _EFFORT_FORMS:
        try:
            _r = brt.converse(modelId=_id, messages=_probe_msg,
                              inferenceConfig={"maxTokens": 512},
                              additionalModelRequestFields=_extra)
            _kinds = [list(b)[0] for b in _r["output"]["message"]["content"]]
            print(f"  {_label}   accepted — content blocks {_kinds}")
        except ClientError as _exc:
            _msg = _exc.response.get("Error", {}).get("Message", str(_exc))
            print(f"  {_label}   rejected — {' '.join(_msg.split())[:190]}")

# %% [markdown]
# Three things in that output, and only the first is the one you went looking for:
#
# 1. **There is exactly one accepted spelling.** `{"reasoning": {"effort": "high"}}` works on
#    the GPT models; `{"reasoning_effort": "high"}` returns `unknown_parameter`. The flat form
#    is the one an OpenAI-API habit reaches for first, so this is a likely first attempt and it
#    fails.
# 2. **Haiku 4.5 rejects both**, naming back whichever key you sent — `… Extra inputs are not
#    permitted`. Not a bad value: the field is unknown to it. So the setting cannot be shared
#    between the two providers even in principle, and a harness that wants it needs a branch on
#    the model.
# 3. **That branch is fine, as long as it is one branch.** Everything above this cell is the
#    portable layer; this field is where Converse admits providers differ. Keep provider-specific
#    settings here, in one place, and the rest of the harness stays model-agnostic. What you must
#    not do is let a provider-specific parameter leak into the agent loop — then every model you
#    add is a new conditional in the hot path.
#
# Note the second bullet is a real constraint on the notebook's own comparison: the GPT models
# will reason at whatever their default effort is, and there is no equivalent knob to give the
# control model. The two columns are not tuned to parity and cannot be.

# %% [markdown]
# ---
#
# ## The agent, on Converse
#
# The same `boutique` agent as the main workshop, with the same v2 fixes: a system prompt
# that forbids prices from memory, a synonym table so "shoes" reaches `sneakers`, error
# strings written as prompts, and tool specs that name the catalog. **The tool
# implementations are unchanged** — they are pure Python and have no idea which provider is
# calling them.
#
# What changes is the loop, and only in mechanical ways:
#
# - Content blocks are dicts, so `isinstance(block, ToolUseBlock)` becomes
#   `"toolUse" in block`.
# - Tool specs go through `to_converse_tool()`.
# - Tool results are `{"toolResult": {"toolUseId":…, "content": [{"json": …}]}}`.
# - **No `temperature` anywhere.** Rejected by GPT, deprecated on Sonnet 5.
# - A turn cap, because a loop with no cap is a budget with no cap.

# %%
# ── The store. Unchanged from the main workshop — pure Python, provider-agnostic. ──
CATALOG = {"jeans": 49.99, "shirt": 29.99, "dress": 59.99, "jacket": 89.99,
           "sneakers": 74.99, "hat": 19.99, "socks": 9.99, "hoodie": 44.99,
           "shorts": 34.99, "tee": 24.99, "sweater": 54.99, "belt": 24.99}

SYNONYMS = {"shoes": "sneakers", "sneaker": "sneakers", "trainers": "sneakers",
            "t-shirt": "tee", "tshirt": "tee", "t shirt": "tee", "tee shirt": "tee",
            "pants": "jeans", "trousers": "jeans", "denim": "jeans",
            "jumper": "sweater", "pullover": "sweater", "sweatshirt": "hoodie",
            "cap": "hat", "beanie": "hat", "frock": "dress"}

LEGAL_OPS = ["+", "-", "*", "/", "**"]


def _resolve_product(raw):
    name = (raw or "").strip().lower()
    if name in CATALOG:
        return name, True
    if name in SYNONYMS:
        return SYNONYMS[name], False
    if name.endswith("s"):
        singular = name[:-1]
        if singular in CATALOG:
            return singular, False
        if singular in SYNONYMS:
            return SYNONYMS[singular], False
    return None, False


def get_product(product):
    key, exact = _resolve_product(product)
    if key is None:
        raise ValueError(f"No catalog item matches '{product}'. The catalog is: "
                         f"{', '.join(sorted(CATALOG))}. Tell the customer we don't carry "
                         f"it and suggest the closest item we do.")
    if exact:
        return CATALOG[key]
    return f"{CATALOG[key]} (interpreted '{product}' as catalog item '{key}')"


def calculate(op, input1, input2):
    if op not in LEGAL_OPS:
        raise ValueError(f"Unsupported op '{op}'. Supported: {' '.join(LEGAL_OPS)}. "
                         f"For a percentage, multiply by the decimal (20% -> * 0.2).")
    if op == "/" and input2 == 0:
        raise ValueError("Division by zero.")
    return {"+": input1 + input2, "-": input1 - input2, "*": input1 * input2,
            "/": input1 / input2 if input2 else None, "**": input1 ** input2}[op]


TOOL_REGISTRY = {"get_product": get_product, "calculate": calculate}

SYSTEM_PROMPT = """You are a shopping assistant for a boutique clothing store.

- Always call get_product to look up a price. Never state a price from memory.
- Use calculate for every arithmetic step. Chain calls for multi-step math.
- If get_product tells you it interpreted the customer's wording as a different catalog
  item, say which item you priced so the customer isn't surprised.
- If something isn't in the catalog, say so plainly and suggest the closest item we do
  sell. Never invent a price for a product that doesn't exist.
- Give the final amount in dollars, and show the arithmetic briefly."""

# ── Tool specs: written in the Anthropic shape, translated once. ──────────────
# Deliberately authored in the format the main workshop uses, then passed through
# to_converse_tool(). That is the migration path a reader with existing specs takes, so the
# notebook should walk it rather than hand-writing Converse specs it doesn't have to.
ANTHROPIC_TOOL_SPECS = [
    {"name": "get_product",
     "description": ("Look up the current price of one item from the store catalog. Returns "
                     "the price in USD. The full catalog is: "
                     + ", ".join(sorted(CATALOG)) + ". Common synonyms are accepted (e.g. "
                     "'shoes' resolves to 'sneakers'); when that happens the result names "
                     "the catalog item that was actually priced. Raises an error listing "
                     "the catalog if nothing matches — never guess a price instead."),
     "input_schema": {"type": "object", "properties": {"product": {
         "type": "string",
         "description": "Singular product name as the customer said it, e.g. 'jeans', "
                        "'shirt', 't-shirt', 'shoes'."}},
         "required": ["product"]}},
    {"name": "calculate",
     "description": ("Apply one arithmetic operation to two numbers and return the result. "
                     "Handles a single step only — chain several calls for multi-step math "
                     "(e.g. 20% off a $50 item: multiply 50 by 0.2, then subtract that from "
                     "50). There is no percent operator: convert percentages to decimals "
                     "yourself."),
     "input_schema": {"type": "object", "properties": {
         "op": {"type": "string", "enum": LEGAL_OPS,
                "description": "The operation: + - * / or ** (exponent)."},
         "input1": {"type": "number", "description": "Left-hand operand."},
         "input2": {"type": "number", "description": "Right-hand operand."}},
         "required": ["op", "input1", "input2"]}},
]

CONVERSE_TOOL_CONFIG = {"tools": [to_converse_tool(s) for s in ANTHROPIC_TOOL_SPECS]}

MAX_TURNS = 8          # a loop with no cap is a budget with no cap
AGENT_MAX_TOKENS = 1024


def execute_tool(name, inputs):
    """A tool error is information for the model, not an exception for the harness."""
    try:
        return str(TOOL_REGISTRY[name](**inputs))
    except Exception as exc:
        return f"Error: {exc}"


def run_agent_converse(prompt, eval_mode=False, model=None):
    """The boutique agent over Converse. Works unchanged on Claude and GPT."""
    model = model or DEFAULT_MODEL
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    tokens_in = tokens_out = turns = 0
    stop_reason = "unknown"
    started = time.time()

    for _ in range(MAX_TURNS):
        response = brt.converse(
            modelId=model, messages=messages,
            system=[{"text": SYSTEM_PROMPT}],
            # No temperature. Rejected outright by the GPT reasoning models, deprecated on
            # Sonnet 5. Omitting it is the only setting that works everywhere.
            inferenceConfig={"maxTokens": AGENT_MAX_TOKENS},
            toolConfig=CONVERSE_TOOL_CONFIG)
        usage = response["usage"]
        tokens_in += usage["inputTokens"]
        tokens_out += usage["outputTokens"]
        turns += 1
        stop_reason = response["stopReason"]
        # Echo the assistant message back verbatim, reasoningContent bytes included. The
        # provider may require its own reasoning blocks on the next turn, and passing them
        # back untouched is both correct and less code than filtering them.
        messages.append(response["output"]["message"])

        if stop_reason != "tool_use":
            break

        # Filter by block type, NEVER by index: a reasoningContent block can precede the
        # toolUse blocks, and a turn can carry several toolUse blocks at once.
        results = []
        for block in response["output"]["message"]["content"]:
            if "toolUse" not in block:
                continue
            call = block["toolUse"]
            results.append({"toolResult": {
                # toolUseId is passed straight back. The format differs by provider
                # (`tooluse_…` from Haiku over Converse, `call_…` from GPT on this account)
                # and Converse treats it as opaque — so never validate, parse or regex it.
                # Note `tooluse_`, not the `toolu_` the Anthropic Messages API returns for the
                # same model: the format is a property of the endpoint as well as the provider.
                "toolUseId": call["toolUseId"],
                # `json` content verified on both providers. A [{"text": ...}] form also
                # exists in the API; this notebook did not need it and did not test it.
                "content": [{"json": {"result": execute_tool(call["name"],
                                                             call["input"])}}]}})
        messages.append({"role": "user", "content": results})

    metrics = {"input_tokens": tokens_in, "output_tokens": tokens_out, "turns": turns,
               "stop_reason": stop_reason, "seconds": round(time.time() - started, 2)}
    if eval_mode:
        return {"messages": messages, "usage": {"input_tokens": tokens_in,
                                                "output_tokens": tokens_out},
                "metrics": metrics}
    return "\n".join(converse_text_blocks(m) for m in messages
                     if m["role"] == "assistant").strip()


print(f"boutique on Converse ready — {len(ANTHROPIC_TOOL_SPECS)} tools, "
      f"turn cap {MAX_TURNS}, no temperature.")
print(f"Smoke test on {short_name(DEFAULT_MODEL)}: "
      f"{' '.join(run_agent_converse('How much do jeans cost?').split())[:120]}")

# %% [markdown]
# ---
#
# ## The eval framework, ported
#
# Here is the part that is genuinely encouraging. The graders from the main workshop operate
# on a **parsed shape** — `{"final_text": str, "tool_calls": [{"name", "arguments"}]}` — not
# on API types. So they port with **zero changes**. `grade_response_contains`,
# `grade_response_numeric` and `grade_tool_use` below are the workshop's functions,
# copied verbatim.
#
# The adapter is one function: `parse_converse_transcript()`. That is the whole cost of
# making an eval suite wire-format agnostic, and it is a good argument for parsing to an
# intermediate representation even when you only have one provider.
#
# Two details in the parser earn their comments: `reasoningContent` blocks are skipped
# explicitly and counted rather than ignored silently (a block you drop without noticing is
# how a transcript ends up mysteriously empty), and the sanitiser replaces the
# `redactedContent` **bytes** with a placeholder so results can be serialised at all.

# %%
def sanitise_block(block):
    """Make a Converse content block JSON-serialisable and readable.

    `reasoningContent.redactedContent` comes back from botocore as `bytes`, which
    json.dumps refuses. Replacing it with a note preserves the fact that reasoning
    happened — and how much of it — without carrying an unprintable blob into your
    saved results."""
    if isinstance(block, dict) and "reasoningContent" in block:
        red = block["reasoningContent"].get("redactedContent")
        return {"reasoningContent": f"<redacted, {len(red) if red else 0} bytes>"}
    return block


def parse_converse_transcript(messages):
    """Converse messages -> the same {final_text, tool_calls} shape the graders expect."""
    final_text, tool_calls, reasoning_blocks = "", [], 0
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        for block in msg["content"]:
            if "text" in block:
                final_text = block["text"]           # last text block wins
            elif "toolUse" in block:
                call = block["toolUse"]
                tool_calls.append({"name": call["name"], "arguments": call["input"],
                                   "id": call["toolUseId"]})
            elif "reasoningContent" in block:
                # Skipped on purpose, counted on purpose. Silently dropping an unknown
                # block type is how you end up with an empty transcript and no idea why.
                reasoning_blocks += 1
    for msg in messages:                             # match results back to their calls
        if msg["role"] != "user":
            continue
        for block in msg["content"]:
            if "toolResult" not in block:
                continue
            result = block["toolResult"]
            for call in tool_calls:
                if call["id"] == result["toolUseId"]:
                    call["result"] = json.dumps(result.get("content", []), default=str)
                    break
    return {"final_text": final_text, "tool_calls": tool_calls,
            "reasoning_blocks": reasoning_blocks, "messages": messages}


# ── Graders: copied verbatim from the main workshop. They parse a shape, not a type. ──

def grade_response_contains(result, check, context=None):
    text = result["final_text"].lower()
    if check.lower() in text:
        return {"score": 1.0, "reason": f"Found '{check}' in response"}
    return {"score": 0.0,
            "reason": f"'{check}' not found in: {result['final_text'][:200]}"}


def grade_response_numeric(result, check, context=None):
    if isinstance(check, (int, float)):
        value, tolerance = float(check), 0.01
    else:
        value, tolerance = float(check["value"]), float(check.get("tolerance", 0.01))
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", result["final_text"])
    for num_str in numbers:
        num = float(num_str.replace(",", ""))
        if abs(num - value) <= tolerance:
            return {"score": 1.0, "reason": f"Found {num} (expected {value} +/- {tolerance})"}
    return {"score": 0.0,
            "reason": f"Expected {value} (+/- {tolerance}), found: {numbers[:10]}"}


def grade_tool_use(result, check, context=None):
    tool_name, expected_args = check["tool_name"], check.get("arguments")
    for call in result["tool_calls"]:
        if call["name"] != tool_name:
            continue
        if expected_args is None:
            return {"score": 1.0, "reason": f"Tool '{tool_name}' was called"}
        actual = call.get("arguments", {})
        if all((isinstance(v, str) and isinstance(actual.get(k), str)
                and v.lower() == actual[k].lower()) or actual.get(k) == v
               for k, v in expected_args.items()):
            return {"score": 1.0,
                    "reason": f"Tool '{tool_name}' called with {expected_args}"}
    seen = [{"name": c["name"], "args": c.get("arguments", {})}
            for c in result["tool_calls"]]
    return {"score": 0.0, "reason": f"'{tool_name}' not called with {expected_args}. "
                                    f"Actual: {seen}"}


GRADER_REGISTRY = {"response_contains": grade_response_contains,
                   "response_numeric": grade_response_numeric,
                   "tool_use": grade_tool_use}

print(f"Parser + graders ready. Graders: {list(GRADER_REGISTRY)} — all deterministic, all "
      f"unchanged\nfrom Building_an_Eval.ipynb. The adapter above is the entire cost of "
      f"going multi-provider.")

# %% [markdown]
# ### The runner and the reporting
#
# Also ported from the main workshop, with three changes worth knowing about:
#
# - `run_eval()` defaults to `max_workers=1` rather than 3. These GPT inference profiles are
#   new on Bedrock and we do not know their throttle limits; a throttled request that
#   exhausts its retries becomes a task ERROR that reads like an agent bug.
# - The ERROR / FAIL distinction is kept, because on a cross-provider run it does most of
#   the diagnostic work. FAIL is the agent. ERROR is infrastructure, and on Bedrock a
#   column of ERRORs is nearly always throttling.
# - `save_results()` keeps `default=str` even though the transcript is sanitised first —
#   belt and braces against a block shape we have not seen.

# %%
def run_single_task(agent_fn, task, model=None):
    start = time.time()
    try:
        raw = agent_fn(task["query"], eval_mode=True, model=model)
    except Exception as exc:
        # Infrastructure failures look nothing like agent bugs. Keep them separable or you
        # will spend the afternoon fixing a prompt to cure a throttle.
        return {"task_id": task["id"], "task_description": task.get("description", ""),
                "query": task["query"], "passed": False, "grades": [],
                "error": traceback.format_exc(),
                "error_short": f"{type(exc).__name__}: {exc}".replace("\n", " ")[:200],
                "metrics": {"seconds": round(time.time() - start, 2)}}

    result = parse_converse_transcript(raw["messages"])
    metrics = dict(raw.get("metrics", {}))
    metrics.update({"tool_calls": len(result["tool_calls"]),
                    "reasoning_blocks": result["reasoning_blocks"]})

    grades = []
    context = {"query": task["query"], "task_id": task["id"], "model": model}
    for grader in task.get("graders", []):
        grader_fn = GRADER_REGISTRY.get(grader["type"])
        if grader_fn is None:
            grades.append({"type": grader["type"], "check": None, "score": 0.0,
                           "reason": f"Unknown grader: {grader['type']}"})
            continue
        for check in grader.get("checks", []):
            try:
                grade = grader_fn(result, check, context)
                grades.append({"type": grader["type"], "check": check,
                               "score": grade["score"], "reason": grade["reason"]})
            except Exception as exc:
                grades.append({"type": grader["type"], "check": check, "score": 0.0,
                               "reason": f"grader error: {type(exc).__name__}: {exc}"})

    return {"task_id": task["id"], "task_description": task.get("description", ""),
            "query": task["query"], "error": None,
            "passed": all(g["score"] == 1.0 for g in grades) if grades else False,
            "grades": grades, "metrics": metrics,
            "final_text": result["final_text"],
            "transcript": [sanitise_block(b) for m in raw["messages"]
                           for b in m["content"]]}


def run_eval(agent_fn, tasks, model=None, num_runs=1, max_workers=1, label=None):
    """Same contract as the main workshop's runner.

    max_workers defaults to 1 here, not 3: we did not measure the throttle limits on these
    GPT inference profiles, and a slow run that reports honest numbers beats a fast one
    that reports errors and sends you hunting a bug that isn't there."""
    resolved = model or DEFAULT_MODEL
    all_runs = []
    for _ in range(num_runs):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run_single_task, agent_fn, t, resolved): t for t in tasks}
            run_results = []
            for future in as_completed(futures):
                r = future.result()
                run_results.append(r)
                mark = "PASS" if r["passed"] else ("ERROR" if r.get("error") else "FAIL")
                print(f"  [{len(run_results)}/{len(tasks)}] {r['task_id']}: {mark}",
                      flush=True)
        order = {t["id"]: i for i, t in enumerate(tasks)}
        run_results.sort(key=lambda r: order.get(r["task_id"], 999))
        all_runs.append(run_results)
    return {"runs": all_runs,
            "config": {"model": resolved, "model_name": short_name(resolved),
                       "agent": getattr(agent_fn, "__name__", str(agent_fn)),
                       "num_runs": num_runs, "num_tasks": len(tasks), "label": label,
                       "region": _region, "api": "bedrock-runtime converse"}}


def pass_rates(results):
    """{task_id: fraction of runs that passed}, in task order."""
    rates = {}
    for run in results["runs"]:
        for r in run:
            rates.setdefault(r["task_id"], []).append(1.0 if r["passed"] else 0.0)
    return {tid: sum(scores) / len(scores) for tid, scores in rates.items()}


def print_summary(results):
    config = results["config"]
    print("=" * 64)
    print(f"{config['label'] or config['model_name']} — {config['num_tasks']} tasks, "
          f"{config['num_runs']} run(s)")
    print(f"Model: {config['model']}   via {config['api']} in {config['region']}")
    print("=" * 64)
    for run in results["runs"]:
        passed, total = sum(1 for r in run if r["passed"]), len(run)
        print(f"Overall: {passed}/{total} passed ({passed / total * 100:.0f}%)\n")
        for r in run:
            mark = "ERROR" if r.get("error") else ("PASS" if r["passed"] else "FAIL")
            print(f"  [{mark}] {r['task_id']}: {r['task_description']}")
            for g in r.get("grades", []):
                print(f"    {'+' if g['score'] == 1.0 else '-'} {g['type']}: "
                      f"{g['reason'][:130]}")
            if r.get("error"):
                print(f"    {r.get('error_short')}")
        ok = [r for r in run if not r.get("error")]
        if ok:
            print(f"\n  avg {sum(r['metrics']['seconds'] for r in ok) / len(ok):.2f}s, "
                  f"{sum(r['metrics']['tool_calls'] for r in ok) / len(ok):.1f} tool calls, "
                  f"{sum(r['metrics']['turns'] for r in ok) / len(ok):.1f} turns · "
                  f"tokens {sum(r['metrics']['input_tokens'] for r in ok):,} in / "
                  f"{sum(r['metrics']['output_tokens'] for r in ok):,} out · "
                  f"{sum(r['metrics']['reasoning_blocks'] for r in ok)} reasoning blocks")
        errored = [r for r in run if r.get("error")]
        if errored:
            print(f"\n  ⚠️  {len(errored)}/{total} task(s) never completed — that is "
                  f"infrastructure, not the agent. Wholesale errors on Bedrock are almost "
                  f"always throttling.")
        print()


def inspect_task(results, task_id, run_index=0):
    run = results["runs"][run_index]
    r = next((x for x in run if x["task_id"] == task_id), None)
    if r is None:
        print(f"'{task_id}' not found. Available: {[x['task_id'] for x in run]}")
        return
    print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['task_id']} — {r['query']}")
    print(f"Response: {r.get('final_text', '')}\n")
    if r.get("error"):
        print(r["error"])
        return
    for g in r["grades"]:
        print(f"  {'+' if g['score'] == 1.0 else '-'} {g['type']}: {g['reason']}")
    print(f"\nMetrics: {r['metrics']}\nTranscript:")
    for block in r.get("transcript", []):
        kind = next(iter(block), "?") if isinstance(block, dict) else "?"
        print(f"  [{kind}] {json.dumps(block.get(kind), default=str)[:220]}")


def save_results(results, directory="eval_results"):
    os.makedirs(directory, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9._-]", "-",
                 str(results["config"].get("label") or results["config"]["model_name"]))
    path = f"{directory}/converse_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as fh:
        # default=str is the belt to the sanitiser's braces: any bytes that reach here
        # from a block shape we have not seen become a string instead of a crash.
        json.dump(results, fh, indent=2, default=str)
    print(f"Results saved to {path}")
    return path


def compare_models(results_by_name, tasks):
    """Per-task pass/fail for every model, side by side."""
    names = list(results_by_name)
    rates = {n: pass_rates(results_by_name[n]) for n in names}
    width = max(len(t["id"]) for t in tasks) + 2
    print(f"{'task':<{width}}" + "".join(f"{n[:13]:>15}" for n in names))
    print("─" * (width + 15 * len(names)))
    for task in tasks:
        print(f"{task['id']:<{width}}"
              + "".join(f"{rates[n].get(task['id'], 0) * 100:>14.0f}%" for n in names))
    print("─" * (width + 15 * len(names)))
    print(f"{'OVERALL':<{width}}"
          + "".join(f"{sum(rates[n].values()) / len(rates[n]) * 100:>14.0f}%"
                    for n in names))
    return rates


print(f"Runner ready — will evaluate {len(EVAL_MODELS)} model(s): "
      f"{', '.join(EVAL_MODELS) or '(none)'}.")

# %% [markdown]
# ---
#
# ## The suite
#
# Six deterministic tasks, taken from the worked solutions of the main workshop. This
# notebook is not a fill-in exercise: it runs a known suite so that the only thing varying
# between columns is the provider.
#
# Expected values, derived from `CATALOG` rather than from output:
#
# | Task | Query | Expected |
# |---|---|---|
# | `price_jeans` | How much do jeans cost? | 49.99, via `get_product("jeans")` |
# | `price_tshirt` | Price of a t-shirt? | 24.99 — customer wording, catalog key is `tee` |
# | `unknown_product_shoes` | How much for shoes? | 74.99 and the word "sneakers" |
# | `total_3shirts_2belts` | 3 shirts and 2 belts? | 3 × 29.99 + 2 × 24.99 = 139.95 |
# | `discount_jacket_20pct` | What's 20% off a jacket? | 89.99 × 0.8 = 71.992, graded to ±0.02 |
# | `total_tshirt_shoes` | A t-shirt and a pair of shoes? | 24.99 + 74.99 = 99.98 |
#
# **Every grader here is deterministic, and that is a deliberate choice rather than a
# limitation.** The matrix above showed that `toolChoice` — and therefore the main
# workshop's forced-tool-call judge — ports to both providers, so an LLM judge *would* work.
# Two reasons it stays out of the scored suite anyway:
#
# 1. **A judge is a model.** Grading a cross-provider comparison with a model adds a second
#    variable to a measurement whose entire purpose is to isolate the first one. When a cell
#    disagrees between providers you want to be certain the disagreement is in the agent.
# 2. **You cannot pin it.** The determinism you are used to from `temperature=0` does not
#    exist on these models — the field is refused. A forced tool call constrains the
#    verdict's *shape*; nothing constrains the reasoning behind it.
#
# The judge is ported and demonstrated further down, on fixtures, so you can see that it
# works and decide for yourself.

# %%
TASKS = [
    {"id": "price_jeans", "description": "Direct price lookup",
     "query": "How much do jeans cost?",
     "graders": [{"type": "response_contains", "checks": ["49.99"]},
                 {"type": "tool_use", "checks": [
                     {"tool_name": "get_product", "arguments": {"product": "jeans"}}]}]},
    # Note what is NOT pinned: the argument. The fix for customer wording lives in the
    # tool, not in the model, so an eval that pinned product == "tee" would fail the
    # correct behaviour. Grade the outcome, not the implementation.
    {"id": "price_tshirt", "description": "Customer wording reaches the catalog key",
     "query": "Price of a t-shirt?",
     "graders": [{"type": "response_numeric",
                  "checks": [{"value": 24.99, "tolerance": 0.01}]},
                 {"type": "tool_use", "checks": [{"tool_name": "get_product"}]}]},
    {"id": "unknown_product_shoes", "description": "Resolves 'shoes' to 'sneakers'",
     "query": "How much for shoes?",
     "graders": [{"type": "response_contains", "checks": ["sneakers"]},
                 {"type": "response_numeric",
                  "checks": [{"value": 74.99, "tolerance": 0.01}]}]},
    {"id": "total_3shirts_2belts", "description": "Two lookups plus arithmetic",
     "query": "3 shirts and 2 belts, what's my total?",
     "graders": [{"type": "response_numeric",
                  "checks": [{"value": 139.95, "tolerance": 0.01}]},
                 {"type": "tool_use", "checks": [
                     {"tool_name": "get_product", "arguments": {"product": "shirt"}},
                     {"tool_name": "get_product", "arguments": {"product": "belt"}}]}]},
    {"id": "discount_jacket_20pct", "description": "Percentage math",
     "query": "What's 20% off a jacket?",
     "graders": [{"type": "response_numeric",
                  "checks": [{"value": 71.99, "tolerance": 0.02}]},
                 {"type": "tool_use", "checks": [
                     {"tool_name": "get_product", "arguments": {"product": "jacket"}}]}]},
    {"id": "total_tshirt_shoes", "description": "Two interpreted lookups plus arithmetic",
     "query": "A t-shirt and a pair of shoes — total?",
     "graders": [{"type": "response_numeric",
                  "checks": [{"value": 99.98, "tolerance": 0.01}]},
                 {"type": "tool_use", "checks": [{"tool_name": "get_product"}]}]},
]

print(f"{len(TASKS)} tasks, "
      f"{sum(len(c['checks']) for t in TASKS for c in t['graders'])} checks, 0 judges.")

# %% [markdown]
# ---
#
# ## Run it
#
# One run per model per task. That is an anecdote, not a baseline — the main workshop
# teaches multi-run evaluation and this notebook deliberately trades statistical depth for
# provider breadth. `max_workers=1`, for the throttle reasons in `run_eval`'s docstring.
#
# Raise `NUM_RUNS` if you want confidence in a per-task cell rather than a sketch of one.

# %%
NUM_RUNS = 1

model_results = {}
for _name, _id in EVAL_MODELS.items():
    print(f"\n=== {_name} ({_id}) ===")
    model_results[_name] = run_eval(run_agent_converse, TASKS, model=_id,
                                    num_runs=NUM_RUNS, max_workers=1, label=_name)

print(f"\n{'model':<14}{'pass':>8}{'err':>6}{'avg s':>8}{'tok in':>10}{'tok out':>10}"
      f"{'reasoning':>11}")
print("─" * 67)
for _name, _res in model_results.items():
    _run = _res["runs"][0]
    _ok = [r for r in _run if not r.get("error")]
    _passed = sum(1 for r in _run if r["passed"])
    _avg = sum(r["metrics"]["seconds"] for r in _ok) / max(len(_ok), 1)
    print(f"{_name:<14}"
          f"{f'{_passed}/{len(_run)}':>8}"
          f"{len(_run) - len(_ok):>6}"
          f"{_avg:>8.2f}"
          f"{sum(r['metrics']['input_tokens'] for r in _ok):>10,}"
          f"{sum(r['metrics']['output_tokens'] for r in _ok):>10,}"
          f"{sum(r['metrics']['reasoning_blocks'] for r in _ok):>11}")

for _name, _res in model_results.items():
    save_results(_res)

# %% [markdown]
# ### Two columns of that table are traps
#
# **The `reasoning` column will probably read `0` for the GPT model, and that is not a bug.**
# Every GPT turn in this suite either emitted `toolUse` or answered after a `toolResult` at
# default effort — and per the table further up, neither of those carries a
# `reasoningContent` block. The model demonstrably produced one a few cells ago on a plain
# question. So a per-run reasoning count of zero tells you about the *shape of your
# workload*, not about the model. If you drew a "does this model reason?" conclusion from an
# agent trace, you would be wrong.
#
# **The token columns are not comparable across providers.** On this account across three
# full passes on 2026-09-13, the two models took the *same* turn counts and the *same* tool
# calls on all six tasks (`[2, 2, 2, 4, 4, 3]` turns, `[1, 1, 1, 5, 3, 3]` calls, both
# models, every pass) — and Haiku reported **~20,500 input tokens** against GPT's
# **8,198** for that identical work (~1,250 vs ~580 out). That is not GPT being two and a half
# times cheaper; it is two providers counting two differently-serialised versions of the same
# conversation with two different tokenizers. Comparing providers on token counts, or on
# token counts multiplied by per-token rates, compares your arithmetic to their accounting.
# If cost is the question, price the whole run.
#
# And recall that neither column includes reasoning tokens, because `usage` does not report
# them. The GPT number quoted above is an undercount by an amount the API will not tell you.
#
# (Only reached Haiku? Then your table has one row and no GPT column. The numbers in this
# section are ours, not yours — but the caution generalises: token counts are a per-provider
# accounting artefact, so they are the wrong axis for any cross-provider comparison.)
#
# %% [markdown]
# ### Side by side
#
# The most valuable row in the table below is any task that passed on one provider and
# failed on the other. A latency difference is a price list; a behavioural difference is an
# engineering problem you now know about before your customers do.

# %%
if not model_results:
    print("No model passed the required capability probes, so nothing was scored. The "
          "probe table above says which check failed and why — that is the finding for "
          "this account, not a broken notebook.")
elif len(model_results) < 2:
    _only = list(model_results)[0]
    print(f"Only {_only} ran, so there is no comparison to draw — this is the degenerate "
          f"single-provider case. Everything above still worked: the Converse client, the "
          f"translator, the parser, the graders. What is missing is the second column, and "
          f"that needs model access granted in the Bedrock console for one of the GPT "
          f"profiles in GPT_MODELS.")
    print_summary(model_results[_only])
else:
    _rates = compare_models(model_results, TASKS)
    _overall = {n: sum(r.values()) / len(r) for n, r in _rates.items()}
    _split = [t["id"] for t in TASKS
              if len({_rates[n].get(t["id"], 0) for n in _rates}) > 1]
    _errors = {n: sum(1 for run in res["runs"] for r in run if r.get("error"))
               for n, res in model_results.items()}

    print()
    if any(_errors.values()):
        print(f"⚠️  Errored tasks: { {n: e for n, e in _errors.items() if e} }. Those rows "
              f"are not comparable — re-run before concluding anything from them.")
    if not _split:
        print(f"Every task landed the same way on all {len(_rates)} models "
              f"({', '.join(f'{n} {v * 100:.0f}%' for n, v in _overall.items())}). The "
              f"harness ports cleanly and the providers agree on this suite. The remaining "
              f"differences are token counts, latency, and the portability findings below — "
              f"which is where the actual work was.")
    else:
        print(f"These tasks did NOT agree across providers: {', '.join(_split)}.")
        print(f"That is the most interesting result in the notebook. Read the transcripts "
              f"before theorising:")
        for _tid in _split:
            for _n in _rates:
                print(f"\n{'═' * 78}\n{_n} · {_tid}")
                inspect_task(model_results[_n], _tid)
    print(f"\nOne run per task, so treat any single differing cell as a lead to "
          f"investigate rather than a measurement. Raise NUM_RUNS to tell a real "
          f"difference from a flaky one.")

# %% [markdown]
# ---
#
# ## The judge, ported (demonstration)
#
# The forced-tool-call judge from Part 6 of the main workshop, translated to Converse. It is
# not used to score the suite above — see the reasons in the suite section — but it is worth
# porting, because a lot of real suites cannot avoid a judge, and this is the pattern that
# survives the crossing:
#
# - The verdict schema is an ordinary Anthropic-style tool spec, put through
#   `to_converse_tool()`. No second copy of the schema.
# - `tool_choice={"type": "tool", "name": ...}` becomes
#   `toolConfig={"toolChoice": {"tool": {"name": ...}}}`.
# - **No `temperature`.** Which is the honest caveat: the determinism you get from
#   `temperature=0` is not available here. The forced tool call guarantees the verdict's
#   *shape*, not its stability.
#
# The cell runs three fixtures — one that must pass, two that must fail — on every reachable
# model. An unvalidated judge is a source of false confidence rather than a measurement, and
# validating it across providers is the only way to know the verdict means the same thing in
# both columns.

# %%
VERDICT_SPEC = {
    "name": "submit_verdict",
    "description": "Record your pass/fail judgement of the assistant's response.",
    "input_schema": {"type": "object", "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
        "reason": {"type": "string",
                   "description": "One sentence justifying the verdict."}},
        "required": ["verdict", "reason"]}}

VERDICT_TOOL_CONFIG = {"tools": [to_converse_tool(VERDICT_SPEC)],
                       "toolChoice": {"tool": {"name": "submit_verdict"}}}

JUDGE_PROMPT = """You are grading one response from an AI shopping assistant against one criterion.

<customer_query>
{query}
</customer_query>

<assistant_response>
{response}
</assistant_response>

<criterion>
{criterion}
</criterion>

Judge ONLY whether the response satisfies this criterion. Specifically:

- Ignore tone, length, formatting and friendliness. They are not the criterion.
- FAIL if the criterion is only partially met, or if it is met by vague implication
  rather than by something actually stated.
- FAIL if the response contains a factual error relevant to the criterion — a
  confident, well-written answer with a wrong number is a failure, not a pass.
- Do not reward the response for doing something impressive that the criterion
  did not ask for.

Submit your verdict with the submit_verdict tool."""

# Haiku by default: judging is a small, well-specified task and this is the cheapest model
# in the pool. Nothing in the function is Anthropic-specific, which the fixtures prove.
JUDGE_MODEL = CONVERSE_MODELS.get("haiku-4-5", DEFAULT_MODEL)


def grade_llm_judge(result, check, context=None, model=None):
    response = brt.converse(
        modelId=model or JUDGE_MODEL,
        messages=[{"role": "user", "content": [{"text": JUDGE_PROMPT.format(
            query=(context or {}).get("query", "unknown"),
            response=result["final_text"], criterion=check)}]}],
        inferenceConfig={"maxTokens": 300},          # no temperature; see above
        toolConfig=VERDICT_TOOL_CONFIG)
    # Filter by type, not by index — on a reasoning model the verdict is not block 0.
    verdict = next((b["toolUse"]["input"]
                    for b in response["output"]["message"]["content"] if "toolUse" in b),
                   None)
    if verdict is None:
        return {"score": 0.0, "reason": "judge returned no verdict"}
    return {"score": 1.0 if verdict.get("verdict") == "PASS" else 0.0,
            "reason": verdict.get("reason", "(no reason given)")}


# Registered so a reader who adds a judged task gets a working grader. The scored suite
# above deliberately uses none.
GRADER_REGISTRY["llm_judge"] = grade_llm_judge

_criterion = "Response states the price of jeans as $49.99"
_fixtures = [("should PASS", "Jeans are $49.99."),
             ("should FAIL", "Jeans are $39.99."),
             ("should FAIL", "We have a great selection of denim!")]

for _name, _id in EVAL_MODELS.items():
    print(f"\n{_name} as judge:")
    for _expected, _text in _fixtures:
        _g = grade_llm_judge({"final_text": _text}, _criterion,
                             {"query": "How much do jeans cost?"}, model=_id)
        _verdict = "PASS" if _g["score"] else "FAIL"
        _agrees = _verdict == _expected.split()[-1]
        print(f"  [{_verdict}] ({_expected}) {'✓' if _agrees else '← DISAGREES'} "
              f"{_text!r} — {_g['reason'][:90]}")

print("\nA judge that cannot separate the good fixture from the bad ones is not a grader, "
      "it is\nnoise with a schema. Check this before you trust any judged score — on every "
      "model you\nintend to judge with.")

# %% [markdown]
# ---
#
# ## What ports, and what does not
#
# ### Ports cleanly
#
# 1. **The tool-use protocol, entirely.** All three `toolChoice` modes, tool-result round
#    trips with `json` content, several `toolUse` blocks in one turn, and the forced-tool-call
#    judge. This is the hard part of an agent harness and it is genuinely provider-agnostic.
# 2. **System prompts.** `system=[{"text": ...}]`, no differences observed.
# 3. **The graders.** Zero changes. They consume `{final_text, tool_calls}`, and a parsed
#    intermediate representation is wire-format agnostic by construction. If you take one
#    design lesson from this notebook, take that one.
# 4. **The request/response envelope.** `stopReason`, `output.message.content`, and the same
#    four `usage` fields on both providers.
# 5. **Tool schemas.** JSON Schema passes through `to_converse_tool()` untouched — `enum`,
#    `required`, nested objects, descriptions.
#
# ### Does not port
#
# 1. **`temperature`.** Refused outright by the GPT reasoning models at every value —
#    `This model doesn't support the temperature field. Remove temperature and try again.`
#    Also deprecated on `claude-sonnet-5` (`` `temperature` is deprecated for this model``),
#    while Haiku 4.5 still accepts it. Any harness that pins `temperature=0` for determinism
#    raises `ValidationException`. Omit the field; then say out loud that you no longer have
#    even the illusion of a determinism knob.
# 2. **The `maxTokens` floor.** 16 on the GPT reasoning models; 8 and 12 are rejected with
#    `integer_below_min_value`. Claude accepts 8. The casualty is not your agent, it is your
#    liveness probe — a one-token ping reports a perfectly functional model as unreachable.
# 3. **Content-block position.** GPT reasoning models emit a `reasoningContent` block first
#    when they emit one at all, and presence swings turn by turn within a single
#    conversation — see the table earlier: yes on a plain answer, no on the turn that emits
#    `toolUse` even at `reasoning.effort=high`, no on the post-`toolResult` answer at default
#    effort, yes on that same answer at `high`. `content[0]["text"]` raises `KeyError`. The
#    providers also disagree on ordinary blocks: on this suite Haiku put a `text` block
#    *before* its `toolUse` blocks in the same message and GPT sent `toolUse` alone, so even
#    "the assistant says something before it calls a tool" is not portable. Filter by type.
#    This is the most dangerous of the six, because a parser that filters by index doesn't
#    crash — it returns an empty transcript, and an empty transcript scores 0 and blames your
#    agent.
# 4. **`redactedContent` is `bytes`.** Not a base64 string. `json.dumps(transcript)` raises
#    `TypeError: Object of type bytes is not JSON serializable`, at save time, after you have
#    paid for the run. It is also encrypted, so there is no readable reasoning to inspect —
#    structurally unlike Anthropic's extended thinking, which returns text.
# 5. **`toolUseId` format.** `call_` + 32 hex characters from OpenAI; `tooluse_` + a short
#    opaque string from Haiku over Converse. The round trip works because Converse treats the
#    ID as opaque — but any code that regexes, prefix-checks or length-asserts an ID breaks on
#    the other provider. The cell above printed the IDs your own account returned. Worth one
#    extra look if you are porting *from* the Anthropic SDK: the same Haiku model returns
#    `toolu_` on the Messages API and `tooluse_` here, so the prefix tracks the endpoint too,
#    not just the provider. Which is the argument for not depending on it at all.
# 6. **Reasoning tokens are invisible.** `usage` reports `inputTokens`, `outputTokens`,
#    `totalTokens`, `cacheReadInputTokens` and no reasoning count, on models that demonstrably
#    generate reasoning content. Token-based cost estimates undercount reasoning models by an
#    amount the API will not tell you.
#
# One more, adjacent, and measured in the `additionalModelRequestFields` cell above rather
# than asserted here: **reasoning effort is controllable, with exactly one spelling.**
# `additionalModelRequestFields={"reasoning": {"effort": "high"}}` is accepted;
# `{"reasoning_effort": "high"}` returns `unknown_parameter`; Haiku 4.5 rejects both, because
# the field does not exist for it. It is not one of the six because the notebook never sets it
# — but it is the one request field that cannot port even in principle, so if you do want it,
# put the branch in one place and keep it out of the agent loop.
#
# ### The meta-finding
#
# All four GPT models in this account — `gpt-6-astra`, `gpt-5.6-sol`, `gpt-5.6-luna`,
# `gpt-5.6-terra` — have **one API surface between them**. Re-checked 2026-09-13: the same
# `temperature` rejection to the word, the same `maxTokens` floor of 16, the same three
# `toolChoice` modes working, the same `call_` + 32 hex `toolUseId`. Four model names, one set
# of portability fixes. That is why the notebook runs one by default: four rows of the same
# answer costs four times as much and teaches nothing new.
#
# **The exception is the interesting part.** `gpt-5.6-terra` returned **no
# `reasoningContent` block** on four out of four plain calls, where `astra`, `sol` and `luna`
# returned one on four out of four. Identical request, identical error surface, different
# response content. So "same provider, same generation, same wire format" does not get you to
# "same thing comes back" — and the difference is in exactly the place that broke a parser two
# sections ago. Uncomment `GPT_MODELS.update(...)` and see what your account does; this claim
# has a shelf life measured in weeks.
#
# ### The habit underneath all six
#
# **A shared wire format is a floor, not a ceiling.** Converse solved the expensive problem —
# the tool-use protocol — and left six cheap ones that you can only find by calling the model
# you actually intend to use, with the parameters you actually intend to pass. That is the
# same lesson as the setup cell pinging each model instead of trusting a valid credential, and
# the same lesson as the main workshop's structured-outputs table: **capabilities are per
# model, not per endpoint, and not per provider.**

# %% [markdown]
# ---
#
# ## What this notebook did not measure
#
# Be at least as clear about this as about the results, because this is where a notebook gets
# oversold.
#
# - **One run per model per task.** An anecdote. The main workshop teaches multi-run
#   evaluation for good reason; this notebook spent its budget on a second provider instead.
#   Any single differing cell in that comparison is a lead, not a finding. `NUM_RUNS` is
#   right there.
# - **One Anthropic model and one OpenAI model, by default.** Not a leaderboard, not a
#   provider survey, and not a statement about GPT versus Claude quality. Six tasks cannot
#   characterise a model; they can only exercise a harness.
# - **Sonnet 5 is out of scope, and it has its own edge.** It rejects `temperature` with a
#   different message — `` `temperature` is deprecated for this model`` — where Haiku 4.5
#   still accepts the field. Two Anthropic models, two answers. Adding it to `CANDIDATES` is
#   one line and the probe matrix will tell you the rest; that is all we checked on it.
# - **No cost figure.** Token counts are printed and per-token rates are not, on purpose: a
#   hardcoded price goes stale silently, which is the one failure mode this repo cannot
#   afford. Multiply the printed tokens by the current Bedrock pricing page.
# - **Streaming, images, caching, guardrails, `ConverseStream`.** All untested here.
#   `list_foundation_models` reports `TEXT, IMAGE` input and `responseStreamingSupported:
#   true` for all four GPT profiles on this account (checked 2026-09-13); we sent text and
#   did not stream. An advertised modality is a claim by the catalog, not a measurement.
# - **Failure modes we saw but did not run.** Two other out-of-scope models on this account
#   produced a silent HTTP 200 with `content = None` and an indefinite hang respectively, via
#   the Anthropic SDK. Both are worse than the 400 demonstrated above, and neither is in this
#   notebook's scope. Assume a third failure mode exists that nobody has hit yet.
#
# ### Next
#
# - `Building_an_Eval.ipynb` — the eval framework this notebook ports, taught properly:
#   tasks, graders, a multi-run baseline, the LLM judge.
# - `Bigger_Model_or_Better_Agent.ipynb` — capability versus engineering, measured, on Claude.
# - Swap `CATALOG`, `ANTHROPIC_TOOL_SPECS` and `TASKS` for your own. `to_converse_tool()`,
#   `parse_converse_transcript()` and `run_agent_converse()` are the reusable parts, and the
#   graders you already have come across unchanged.
