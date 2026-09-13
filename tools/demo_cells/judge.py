# ✅ WORKED SOLUTION — LLM-as-judge with a schema-validated structured verdict.
#
# The verdict schema is expressed as a tool. Forcing that tool call is how we get
# schema-validated JSON on Bedrock, where `output_config={"format": ...}` is rejected
# with `400 output_config.format: Extra inputs are not permitted`.

VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Record your pass/fail judgement of the assistant's response.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
            "reason": {"type": "string", "description": "One sentence justifying the verdict."},
        },
        "required": ["verdict", "reason"],
    },
}

JUDGE_MODEL = FAST_MODEL  # judging is a small, well-specified task — Haiku is plenty

# The judge prompt does four things a naive version doesn't: it scopes the judgement to
# the single criterion, it names what to ignore, it spells out what a FAIL looks like,
# and it forbids rewarding confidence. Every one of those exists because a looser judge
# passed something it shouldn't have.
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


def grade_llm_judge(result, check, context=None):
    query = context["query"] if context else "unknown"

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        tools=[VERDICT_TOOL],
        tool_choice={"type": "tool", "name": "submit_verdict"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            query=query, response=result["final_text"], criterion=check)}],
    )

    verdict_block = next((b for b in response.content if isinstance(b, ToolUseBlock)), None)
    if verdict_block is None:  # forced tool_choice makes this very unlikely
        return {"score": 0.0, "reason": "judge did not return a verdict"}

    data = verdict_block.input
    return {"score": 1.0 if data.get("verdict") == "PASS" else 0.0,
            "reason": data.get("reason", "(no reason given)")}


# Same contract as the deterministic graders, so registration is all it takes.
GRADER_REGISTRY["llm_judge"] = grade_llm_judge
print(f"Graders loaded: {list(GRADER_REGISTRY.keys())}")

# ── Sanity-check the judge itself ─────────────────────────────────────────────
# An unvalidated judge is a source of false confidence, not a measurement. Feed it one
# response that should pass and one that should fail, and confirm it can tell them apart.
# If the bad case passes, the criterion is too loose — fix it before you trust any score.
_criterion = "Response states the price of jeans as $49.99"
for _label, _text in [("should PASS", "Jeans are $49.99."),
                      ("should FAIL", "Jeans are $39.99."),
                      ("should FAIL", "We have a great selection of denim!")]:
    _g = grade_llm_judge({"final_text": _text}, _criterion,
                         {"query": "How much do jeans cost?"})
    print(f"  [{'PASS' if _g['score'] else 'FAIL'}] ({_label}) {_text!r} — {_g['reason']}")
