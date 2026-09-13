# Building an Eval for an AI Agent

**NAMER Tech Summit · hands-on workshop · Claude on Amazon Bedrock**

A self-contained, ~90 minute workshop (plus 10 for setup). You build an eval suite for a
small AI shopping assistant, use it to find out exactly where the agent breaks, fix it, and
then prove the fix worked — with numbers rather than vibes.

Everything runs against **Claude on Amazon Bedrock**. Nothing in this folder depends on
the rest of the repository, so it can be zipped and handed out on its own.

---

## What's here

| Path | What it is |
|---|---|
| [`Building_an_Eval.ipynb`](Building_an_Eval.ipynb) | **The workbook.** What attendees work in. Cells marked ✏️ YOUR TURN are theirs to fill in. |
| [`demo/Building_an_Eval_DEMO.ipynb`](demo/Building_an_Eval_DEMO.ipynb) | **The worked demo.** Every blank filled. Run it live, or hand it out afterwards. |
| [`Bigger_Model_or_Better_Agent.ipynb`](Bigger_Model_or_Better_Agent.ipynb) | **The companion lab.** A worked experiment, not a fill-in: measures a bigger model against a better-engineered agent on the same suite. Optional, standalone, ~20 min. |
| [`FACILITATOR.md`](FACILITATOR.md) | Timings, talk track, the failures to expect, and what to do when the room gets stuck. |
| [`SETUP.md`](SETUP.md) | Bedrock credentials, Python environment, and the errors people actually hit. |
| `requirements.txt` | Two packages. That's the whole dependency list. |
| [`multi-provider.md`](multi-provider.md) | Why the notebook is Anthropic-only, and what running it on OpenAI/xAI models via Bedrock's Converse API would take. Investigation notes. |
| `tools/` | Notebook build scripts, plus `check_structured_outputs.py` — a pre-session check that Part 6's `output_config` table still matches your account. Only needed if you're *running or maintaining* the workshop. |

---

## What you'll build

`boutique` is a single-turn agent with two tools: look up a product price, and do
arithmetic. It answers fluently and it is wrong more often than it looks. Over seven
parts you build the machinery to prove that:

1. **Meet the agent** — probe it, notice that confident prose hides wrong numbers
2. **The eval framework** — tasks, a concurrent runner, three deterministic graders
3. **Write eval tasks** — turn observations into checks that can fail
4. **Run it** — get a baseline, then a *real* baseline across multiple runs
5. **Fix the agent** — and use `compare_results` to show the gain and catch regressions
6. **LLM-as-judge** — grade the responses no string match can handle
7. **Compare models** — Haiku, Sonnet, Fable 5 and Fable 5.1 on your own suite, scored
   against tokens and latency; then v1 *and* v2 across Sonnet 5, Opus 5 and Fable 5.1, to
   settle whether a higher-tier model can substitute for actually fixing the agent

The habit being taught, more than any single technique: **measure before you change,
compare after, and never trust a single run.**

### What attendees leave with

- A task schema, three deterministic graders, and an LLM judge that returns a
  schema-validated verdict on Bedrock via a forced tool call — the pattern that works on
  every model, including the newer ones where Bedrock still rejects `output_config`
- A concurrent runner with per-task error isolation and Bedrock-appropriate throttling behaviour
- A before/after comparison that flags regressions — the seed of a CI regression gate
- Every result set self-describing: `config` records the resolved model ID and the agent that
  produced it, so runs stay comparable weeks later
- An answer to the question every model-selection argument turns on — whether a higher-tier
  model fixes a broken agent or merely hides the bug — measured on their own suite

---

## Quick start

Full detail, including every error message you might hit, is in [`SETUP.md`](SETUP.md).

```bash
# 1. From this folder
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Credentials
cp .env.example .env                 # then edit .env
```

Put your Bedrock credentials in `.env` (gitignored — never committed):

```ini
AWS_BEARER_TOKEN_BEDROCK=<your Bedrock API key>
AWS_REGION=us-east-1
```

Already have working AWS credentials (SSO, `aws configure`, an assumed role)? Skip the
token, keep `AWS_REGION`, and set `AWS_PROFILE` if you use a named profile.

Then open `Building_an_Eval.ipynb` in VS Code, pick the `.venv` kernel (kernel picker,
top right), and run the first two cells. A green **"✓ Connected to Claude on Bedrock"**
banner means you're ready.

> The setup cell verifies your credentials with a real API call to each model, because a
> valid credential without model access looks identical until you use it — the single most
> common workshop failure. Never paste a key into a notebook cell — it belongs in `.env`.

### Prerequisites

- Python 3.9+
- Bedrock **model access** granted for Claude (Bedrock console → Model access). Granted per
  account, one model at a time. Any region with a Bedrock endpoint works — the notebook uses
  global inference profiles, so you don't need to hunt for a region that has every model.
- A few dollars of Bedrock usage per attendee for a full pass. Everything up to Part 7 runs
  on Haiku; Part 7 is the bulk of the cost — it repeats the suite on Sonnet, Fable 5 and
  Fable 5.1, then runs both agent versions across the tier ladder. Trim `SWEEP_EXTRA` and
  `TIER_MODELS` in those cells to cut it back. The companion lab is a few dollars more, most
  of it on Haiku; trim `RUNS`, `LADDER_MODELS` and `CONFIG_LADDER`.

---

## Running it as a workshop

Read [`FACILITATOR.md`](FACILITATOR.md) first. The short version:

| Part | Minutes | Mode |
|---|---|---|
| Setup | 10 | Together |
| Part 1 (meet the agent) | 10 | Together |
| Part 2 (framework walkthrough) | 10 | You talk, they read |
| Part 3 (write tasks) | 20 | **Solo / pairs** |
| Part 4 (run + read results) | 10 | Solo, then discuss |
| Part 5 (fix + compare) | 20 | **Solo / pairs** |
| Part 6 (LLM judge) | 10 | Solo |
| Part 7 + wrap-up | 10 | Together |

Parts 3 and 5 are the workshop. Everything else is scaffolding around them — if you're
running short, protect those two and talk through the rest from the demo notebook.

---

## The companion lab

[`Bigger_Model_or_Better_Agent.ipynb`](Bigger_Model_or_Better_Agent.ipynb) is separate and
optional. It answers the question Part 7 raises but can't fully settle, because the main
workshop's agent is missing information no model can recover:

> Our agent gets things wrong. Do we move it to a stronger model, or do we fix it?

It runs a second agent whose three defects are all *documentation* problems — prices in
minor units, a silently paginated search, a discount argument in whole percents — so both
answers are genuinely available. Then it measures two ladders on one suite of seven tasks:
capability (Haiku → Sonnet → Fable 5.1 → Opus, agent untouched) and engineering (Haiku
throughout, agent tuned four ways), and prices the winner of each against the other on
accuracy, tokens and latency.

Every cell is complete — attendees run it and read the tables rather than filling blanks.
Useful as a ~20 minute follow-on, a pre-read for a model-selection debate, or something to
hand to whoever asks "why not just use Opus?". It needs nothing from the workbook except
the same `.env`.

---

## Maintaining the workshop

Three notebooks are generated from two percent-format sources, so the setup cell, the
harness and the prose can't drift between variants:

```
tools/workbook_src.py      source for the workbook and the demo (plain Python, percent format)
tools/lab_src.py           source for the companion lab
tools/demo_cells/<id>.py   worked solutions that replace cells tagged `# %% id=<id>`
tools/shared/setup.py      cells spliced into both sources by `# %% include=setup`
tools/nbbuild.py           builds all three .ipynb files
```

`include=` exists so the credential and install handling has exactly one copy: the lab
needs the same three setup cells as the workbook, and two copies would drift. The lab
deliberately has its *own* runner rather than reusing the workbook's `run_eval` — it
measures a grid of (config × model) rather than one suite, which is a different shape.

```bash
python tools/nbbuild.py            # rebuild all three notebooks
python tools/nbbuild.py --check    # exit 1 if a notebook is stale (use in CI)
```

Edit the `.py` sources, not the `.ipynb` files — a direct notebook edit is overwritten on
the next build. The shipped notebooks are committed without outputs so diffs stay
readable.

---

## Credits

Adapted from the Anthropic Partner Basecamp *Building an Eval* build-along
(`day2/01_evals/`), restructured as a standalone Bedrock-only workshop with a worked
demo, a before/after comparison, and a judge that works on the Bedrock endpoint.
