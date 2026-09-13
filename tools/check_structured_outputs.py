#!/usr/bin/env python3
"""Check which Claude models accept `output_config` on Bedrock, for your account.

Part 6 tells attendees that structured outputs work on Bedrock for some Claude models and
return a 400 on others, and that a forced tool call is the pattern that works everywhere.
That is a claim about a moving target: Bedrock parity gaps close over time. Run this before
a session so the talk track matches what the room will see.

    python tools/check_structured_outputs.py                 # models the workshop uses
    python tools/check_structured_outputs.py --all           # every reachable Claude profile
    python tools/check_structured_outputs.py --compare-direct # also test api.anthropic.com
    python tools/check_structured_outputs.py --region us-east-1

Reads credentials from `.env` the same way the notebook does (nearest one, walking up), so
it needs no setup beyond what an attendee already has. Each probe is one tiny request.

Exit status is 0 if the run completed, 1 if it couldn't run at all — not a pass/fail on the
models, since "this model rejects output_config" is a finding, not an error.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor

# Models the notebook actually references, so the default run mirrors the workshop.
WORKSHOP_MODELS = [
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",   # JUDGE_MODEL / FAST_MODEL
    "global.anthropic.claude-sonnet-5",                  # MODEL
    "global.anthropic.claude-opus-5",                    # BIG_MODEL
    "global.anthropic.claude-fable-5-1",           # FABLE_5_1_MODEL
]

# A verdict schema shaped like Part 6's, so this tests the thing the judge would actually do.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"passed": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["passed", "reason"],
    "additionalProperties": False,
}
OUTPUT_CONFIG = {"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}}

# Asks for prose, so a model that merely *tends* to emit JSON fails: we're testing an
# enforced schema, not a cooperative model. Kept short, and MAX_TOKENS generous, because a
# truncated reply is unparseable JSON and would otherwise read as a failure to enforce.
PROMPT = [{
    "role": "user",
    "content": "Briefly: does the response 'Jeans are $49.99.' state a price? Give a verdict.",
}]
MAX_TOKENS = 2048


def load_env():
    """Nearest .env walking up from this file's parent, matching the notebook's loader."""
    here = pathlib.Path(__file__).resolve().parent.parent
    for d in [here, *here.parents]:
        if (d / ".env").is_file():
            text = (d / ".env").read_text()
            # `#?` so a commented-out ANTHROPIC_API_KEY is still found for --compare-direct;
            # people comment it out to force the Bedrock path, not to revoke it.
            return d / ".env", dict(re.findall(r"^\s*#?\s*([A-Z_][A-Z0-9_]*)=(.*)$", text, re.M))
    return None, {}


# Verdicts that are a statement about output_config. Anything else means we never got to
# ask the question, so it shouldn't count in the summary either way.
CONCLUSIVE = ("WORKS", "REJECTED", "NOT ENFORCED")


def classify(exc):
    """Turn an SDK exception into a short verdict plus the message worth showing."""
    msg = str(exc)
    if "output_config" in msg and "not permitted" in msg:
        return "REJECTED", "400 output_config.format: Extra inputs are not permitted"
    # 404s here are lifecycle, not permissions: the profile is listed but retired or
    # not-yet-granted. No grant or region change fixes it, so don't report it as a finding.
    if "not_found" in msg or "404" in msg:
        reason = "retired / not granted" if "Legacy" in msg or "end of its life" in msg \
            else "no such model on this endpoint"
        return "UNAVAILABLE", reason
    return type(exc).__name__.upper(), msg[:110]


def probe(client, model):
    """One request. Returns (model, verdict, detail)."""
    try:
        r = client.messages.create(model=model, max_tokens=MAX_TOKENS,
                                   messages=PROMPT, output_config=OUTPUT_CONFIG)
    except Exception as exc:                      # noqa: BLE001 — any failure is a datapoint
        return model, *classify(exc)

    text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Truncated JSON is unparseable but says nothing about enforcement — don't let a
        # too-small max_tokens masquerade as a model that ignores the schema.
        if r.stop_reason == "max_tokens":
            return model, "TRUNCATED", f"hit max_tokens={MAX_TOKENS}; raise it and re-run"
        if not text.strip():
            # Extended thinking can consume the whole budget before any text block.
            return model, "TRUNCATED", f"no text block (stop_reason={r.stop_reason})"
        # Accepted the parameter and ignored it — worse than a clean rejection, because it
        # would pass a smoke test and then fail on some later response.
        return model, "NOT ENFORCED", f"accepted, non-JSON reply: {text[:50]!r}"
    if set(obj) != set(VERDICT_SCHEMA["required"]):
        return model, "NOT ENFORCED", f"JSON but wrong keys: {sorted(obj)}"
    return model, "WORKS", "schema-valid JSON"


def bedrock_models(region, env, want_all):
    if not want_all:
        return WORKSHOP_MODELS
    import botocore.session
    if env.get("AWS_BEARER_TOKEN_BEDROCK"):
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = env["AWS_BEARER_TOKEN_BEDROCK"]
    bd = botocore.session.get_session().create_client("bedrock", region_name=region)
    return sorted(
        p["inferenceProfileId"]
        for page in bd.get_paginator("list_inference_profiles").paginate()
        for p in page["inferenceProfileSummaries"]
        if p["inferenceProfileId"].startswith("global.anthropic.")
    )


def direct_name(bedrock_id):
    """`global.anthropic.claude-haiku-4-5-20251001-v1:0` -> `claude-haiku-4-5`.

    The direct API takes an unprefixed name and doesn't accept Bedrock's date/version
    suffix, so strip both rather than 404 on every dated profile.
    """
    name = bedrock_id.split("anthropic.", 1)[-1]
    return re.sub(r"-(\d{8}-)?v\d+(:\d+)?$", "", name)


def report(title, rows):
    print(f"\n=== {title}")
    width = max((len(m) for m, _, _ in rows), default=10)
    for model, verdict, detail in rows:
        mark = {"WORKS": "✓", "REJECTED": "✗", "NOT ENFORCED": "!",
                "TRUNCATED": "~", "UNAVAILABLE": "-"}.get(verdict, "?")
        print(f"  {mark} {model:<{width}}  {verdict:<12} {detail}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="test every reachable global.anthropic.* profile, not just the five "
                         "the workshop uses")
    ap.add_argument("--compare-direct", action="store_true",
                    help="also test api.anthropic.com, to distinguish a Bedrock parity gap "
                         "from a model that lacks the feature entirely "
                         "(needs ANTHROPIC_API_KEY in .env)")
    ap.add_argument("--region", help="Bedrock region (default: AWS_REGION from .env)")
    args = ap.parse_args()

    try:
        import anthropic
    except ImportError:
        sys.exit("anthropic is not installed — pip install -r requirements.txt")

    env_path, env = load_env()
    region = args.region or env.get("AWS_REGION") or os.environ.get("AWS_REGION")
    if not region:
        sys.exit("No region. Set AWS_REGION in .env or pass --region.")
    print(f"env    : {env_path or '(none found)'}")
    print(f"region : {region}")

    token = env.get("AWS_BEARER_TOKEN_BEDROCK", "").strip()
    profile = env.get("AWS_PROFILE", "").strip() or os.environ.get("AWS_PROFILE", "")
    if token and not token.startswith("paste-your"):
        # The SDK refuses a bearer token and AWS credentials at the same time.
        os.environ.pop("AWS_PROFILE", None)
        client = anthropic.AnthropicBedrock(api_key=token, aws_region=region,
                                           timeout=120.0, max_retries=2)
        print("auth   : api-key")
    else:
        kw = {"aws_profile": profile} if profile else {}
        client = anthropic.AnthropicBedrock(aws_region=region, timeout=120.0,
                                           max_retries=2, **kw)
        print(f"auth   : IAM{' (profile ' + profile + ')' if profile else ''}")

    models = bedrock_models(region, env, args.all)
    with ThreadPoolExecutor(max_workers=3) as ex:
        rows = report(f"bedrock-runtime · {region}",
                      list(ex.map(lambda m: probe(client, m), models)))

    if args.compare_direct:
        key = env.get("ANTHROPIC_API_KEY", "").strip()
        if not key.startswith("sk-ant-"):
            print("\n=== api.anthropic.com\n  skipped — no ANTHROPIC_API_KEY in .env")
        else:
            direct = anthropic.Anthropic(api_key=key, timeout=120.0, max_retries=2)
            with ThreadPoolExecutor(max_workers=3) as ex:
                report("api.anthropic.com",
                       list(ex.map(lambda m: probe(direct, direct_name(m)), models)))

    asked = [(m, v) for m, v, _ in rows if v in CONCLUSIVE]
    works = [m for m, v in asked if v == "WORKS"]
    if not asked:
        sys.exit("\nNo model answered the question — nothing was reachable. Run the notebook's\n"
                 "setup cell first; it prints which models your account can actually call.")
    print(f"\n{len(works)}/{len(asked)} reachable models accept and enforce output_config "
          f"on Bedrock.")
    if not works:
        print("None of them. Part 6's forced-tool-call judge is the only option here — which\n"
              "is what the notebook does, so nothing to change.")
    elif len(works) < len(asked):
        print("Mixed support — the state Part 6 describes. Support is per model, not per\n"
              "endpoint, so keep the forced-tool-call judge: it is the only pattern that\n"
              "survives the Part 7 model sweep. Update the model names in Part 6's table if\n"
              "the split above differs from the one it lists.")
    else:
        print("Every reachable model now accepts it, so Part 6's table is stale — update the\n"
              "note in tools/workbook_src.py (and FACILITATOR.md / SETUP.md). The forced tool\n"
              "call is still worth teaching for cross-provider portability, but say so honestly.")


if __name__ == "__main__":
    main()
