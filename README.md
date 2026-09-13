# Building an Eval for an AI Agent

**NAMER Tech Summit · hands-on workshop · Claude on Amazon Bedrock**

A self-contained, ~90 minute workshop (plus 10 for setup). You build an eval suite for a
small AI shopping assistant, use it to find out exactly where the agent breaks, fix it, and
then prove the fix worked — with numbers rather than vibes.

The main workshop and the companion lab run against **Claude on Amazon Bedrock**. A separate
companion notebook, [`Multi_Provider_on_Bedrock.ipynb`](Multi_Provider_on_Bedrock.ipynb),
extends the same eval to **OpenAI's GPT 5.6 / GPT 6 reasoning models** through Bedrock's
Converse API. Nothing in this folder depends on the rest of the repository, so it can be
zipped and handed out on its own.

---

## What's here

| Path | What it is |
|---|---|
| [`Building_an_Eval.ipynb`](Building_an_Eval.ipynb) | **The workbook.** What attendees work in. Cells marked ✏️ YOUR TURN are theirs to fill in. |
| [`demo/Building_an_Eval_DEMO.ipynb`](demo/Building_an_Eval_DEMO.ipynb) | **The worked demo.** Every blank filled. Run it live, or hand it out afterwards. |
| [`Bigger_Model_or_Better_Agent.ipynb`](Bigger_Model_or_Better_Agent.ipynb) | **The companion lab.** A worked experiment, not a fill-in: measures a bigger model against a better-engineered agent on the same suite. Optional, standalone, ~20 min. |
| [`Multi_Provider_on_Bedrock.ipynb`](Multi_Provider_on_Bedrock.ipynb) | **The multi-provider companion.** Claude Haiku 4.5 against OpenAI's GPT 5.6 / GPT 6 reasoning models through Bedrock's Converse API — the same agent, the same graders, ported. Optional, standalone, ~20 min. |
| [`SETUP.md`](SETUP.md) | Bedrock credentials, Python environment, and the errors people actually hit. |
| `requirements.txt` | Three packages. That's the whole dependency list. |
| `workshop_setup.py` | The setup every notebook runs in its first cell: kernel safety check, Bedrock credentials from `.env`, and a real ping of each model. Imported, not edited — but every failure message it prints names the fix, so it's worth a look if setup misbehaves. |
| [`multi-provider.md`](multi-provider.md) | Investigation notes behind the multi-provider notebook: what was probed, what broke, and why the Converse path was chosen. Also the xAI and open-weight findings that stayed *out* of the notebook. |
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
7. **Compare models** — Haiku, Sonnet and Fable 5.1 on your own suite, scored
   against tokens and latency; then v1 *and* v2 across Sonnet 5, Opus 5 and Fable 5.1, to
   settle whether a higher-tier model can substitute for actually fixing the agent

The habit being taught, more than any single technique: **measure before you change,
compare after, and never trust a single run.**

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
top right), and run the first cell. A green **"✓ Connected to Claude on Bedrock"**
banner means you're ready.

### Prerequisites

- Python 3.9+
- Bedrock **model access** granted for Claude (Bedrock console → Model access). Granted per
  account, one model at a time. Any region with a Bedrock endpoint works — the notebook uses
  global inference profiles, so you don't need to hunt for a region that has every model.

---

## Running it as a workshop

---

## Optional lab # 1

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

## Optional lab # 2 :  multi-provider 

[`Multi_Provider_on_Bedrock.ipynb`](Multi_Provider_on_Bedrock.ipynb) answers the other
question the workshop raises and doesn't settle: **does this eval still work if the model
isn't Claude?**

---

## Maintaining the workshop

Four notebooks are generated from three percent-format sources, so the setup cell, the
harness and the prose can't drift between variants:

```
tools/workbook_src.py       source for the workbook and the demo (plain Python, percent format)
tools/lab_src.py            source for the companion lab
tools/multiprovider_src.py  source for the multi-provider Converse notebook
tools/demo_cells/<id>.py    worked solutions that replace cells tagged `# %% id=<id>`
tools/shared/setup.py       the setup cell, spliced into every source by `# %% include=setup`
tools/nbbuild.py            builds all four .ipynb files
workshop_setup.py           the setup logic itself — shipped to attendees, not built
```

Note the split: `tools/shared/setup.py` is the two *cells* an attendee sees (a markdown
intro and a short bootstrap), while `workshop_setup.py` at the repo root is the code those
cells import. `include=` gives the cells exactly one copy across the three sources; the
import gives the logic exactly one copy too, and keeps it out of the notebook where it was
the first thing anyone read. `workshop_setup.py` lives at the root rather than in `tools/`
because attendees need it — `tools/` is maintainers-only.

Each notebook deliberately keeps its *own* runner rather than reusing the workbook's
`run_eval` — the lab measures a grid of (config × model) and the multi-provider notebook runs
on a Converse client, which are different shapes.

```bash
python tools/nbbuild.py            # rebuild all four notebooks
python tools/nbbuild.py --check    # exit 1 if a notebook is stale (use in CI)
```

Edit the `.py` sources, not the `.ipynb` files — a direct notebook edit is overwritten on
the next build. The shipped notebooks are committed without outputs so diffs stay
readable.
