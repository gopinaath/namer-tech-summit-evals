# Setup

A Python environment and Bedrock credentials. Ten minutes. Every error the notebook can
print is listed at the bottom with the fix.

---

## 1. Python environment

**Python 3.9+**, in a venv — the notebook refuses to install into a bare system Python.

```bash
cd NAMER-Tech-Summit

python3 -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt      # anthropic[bedrock] + ipykernel
```

Then open the folder in VS Code (Python + Jupyter extensions), open
`Building_an_Eval.ipynb`, and pick the `.venv` interpreter in the kernel picker, top right.

conda works too — `conda create -n evals python=3.12`, then install and select that kernel.
Shared `base` doesn't count as isolated; make a named env.

**Hosted notebooks** (SageMaker Studio, notebook instances, any managed JupyterLab): skip
all of the above. The image's own Python is disposable, so the first cell detects the
container and installs straight into it.

---

## 2. Bedrock credentials

Request **Model access** for the Anthropic models in the Bedrock console first — granted per
account, one model at a time, and the usual reason a valid credential still fails. Any region
with a Bedrock endpoint works.

`cp .env.example .env`, then use whichever credential you have.

**Option A — Bedrock API key** (what most attendees get):

```ini
AWS_BEARER_TOKEN_BEDROCK=<paste the whole key>
AWS_REGION=us-east-1
```

**Option B — IAM** (SSO, `aws configure`, assumed role). Leave the bearer token empty:

```ini
AWS_REGION=us-east-1
# AWS_PROFILE=my-profile
```

Either way the credential needs `bedrock:InvokeModel`. Check IAM creds haven't expired with
`aws sts get-caller-identity`.

> `.env` is gitignored. **Never paste a credential into a notebook cell** — cells get
> saved, screenshotted and shared.

---

## 3. Verify

Run the first two cells of `Building_an_Eval.ipynb`. You're looking for:

```
✓ Ready — anthropic installed for /…/.venv/bin/python
✓ Connected to Claude on Bedrock — us-east-1, auth: api-key. Models available:
  haiku-4-5, sonnet-5, opus-5, fable-5, fable-5-1
```

The cell pings each model rather than inspecting the credential — an ungranted model looks
identical to a working one until you call it. Fewer models listed is fine; only Part 7 is
affected.

> `AWS_REGION` already set in your shell overrides `.env`.

---

## Models

Global inference profiles, so any Bedrock region works:

```python
FAST_MODEL      = "global.anthropic.claude-haiku-4-5-20251001-v1:0"  # agent + judge
MODEL           = "global.anthropic.claude-sonnet-5"
BIG_MODEL       = "global.anthropic.claude-opus-5"
FABLE_MODEL     = "global.anthropic.claude-fable-5"
FABLE_5_1_MODEL = "global.anthropic.claude-fable-5-1"
```

Only `FAST_MODEL` is required. The rest are compared in Part 7 and in the companion lab, and
the setup cell drops any your account can't reach, so a missing grant costs a table row
rather than a broken notebook. (The lab's two comparison sections say so explicitly when
they've been left with only one model to work with.)

Swap `global.` for your geography (`us.`, `eu.`, `apac.`) if data residency rules out global
routing, and stay in a matching region.

---

## Cost

**A few dollars per full pass.** Parts 1-6 are all Haiku. Part 7 is most of the spend, in two
cells: the sweep repeats the 9-task suite on Sonnet 5, Fable 5 and Fable 5.1, then the tier
comparison runs *both* agent versions on up to four models. Set `SWEEP_EXTRA = []` and trim
`TIER_MODELS` to cut it right back. `print_summary()` prints token counts as you go.

**The companion lab** (`Bigger_Model_or_Better_Agent.ipynb`) is a few dollars again: ~190
agent runs, of which only 21 are on Opus and the rest on Haiku. Three knobs cut it, in
descending order of saving:

| Knob | Default | Cheapest useful setting |
|---|---|---|
| `LADDER_MODELS` | every model your account reaches | two — the cheapest and one other |
| `RUNS` | `3` | `1`, but every number then becomes anecdote (the lab's own noise check will show you why) |
| `CONFIG_LADDER` | `["L0","L1","L2","L3","L4"]` | `["L0","L2"]` — keeps the before/after, drops the lever comparison |

It prints running token and latency totals in its final table rather than a dollar figure,
because per-token prices move and a number baked into a notebook goes stale silently.

---

## Troubleshooting

### "This kernel is a shared Python"

`pip install` in the notebook goes wherever the kernel lives, so it stops rather than mutate
a Python you share. The check is on the interpreter, not `VIRTUAL_ENV` — select the `.venv`
interpreter *in VS Code*, not just activate it in a terminal.

Hosted notebooks are detected (`/.dockerenv`, `/opt/ml`, `SAGEMAKER*`/`SM_*`, container
cgroups) and allowed through. If yours isn't, run this above the cell and re-run it:

```python
import os; os.environ["WORKSHOP_ALLOW_SYSTEM_PYTHON"] = "1"
```

### "📋 Add your Bedrock credentials to continue"

No usable credential found. Check `.env` sits next to the notebook, no quotes or spaces around
the value, and that you replaced the `paste-your-...` placeholder (ignored deliberately). The
cell re-reads `.env` on every run.

### "model '…' isn't usable from \<region\>"

Credential is fine, model isn't granted. Bedrock console → **Model access**. Changing
`AWS_REGION` won't help — these are global profiles, so the grant is account-level.

### "Bedrock denied the request for '…'" (403)

`AccessDenied` is ambiguous: either the credential lacks `bedrock:InvokeModel` or that model
isn't granted. Rule out the IAM policy first.

### "Bedrock rejected those credentials" (401 / 403)

API key: paste the whole value, confirm it isn't revoked. IAM: probably expired — `aws sso
login`, then re-run.

### "The IAM-credentials path needs boto3"

`pip install 'anthropic[bedrock]'`, or use a Bedrock API key, which needs no boto3.

### Tasks come back as `ERROR` rather than `FAIL`

`ERROR` means the request never completed — infrastructure, not your agent. **Every task
failing for one model** is a missing grant. **Scattered failures** are throttling: pass
`max_workers=1` to `run_eval`. The SDK already retries 429s (`max_retries=4`); if it's
chronic, request a quota increase.

### `400 output_config.format: Extra inputs are not permitted`

Structured outputs are accepted on Bedrock through the 4.6 families (Haiku 4.5, Sonnet 4.5,
Sonnet 4.6, Opus 4.5, Opus 4.6) and rejected from 4.7 onwards (Opus 4.7, Opus 4.8, Sonnet 5,
Opus 5, Fable 5, Fable 5.1) — even though all of them accept it on the Anthropic API directly.
A beta header doesn't unlock it. The newer model is the one missing the feature, so don't
assume support carries upward.

Part 6 therefore uses the alternative that works everywhere: a tool whose `input_schema` is
the verdict schema, forced with `tool_choice`. Re-check the current state for your account
with `python tools/check_structured_outputs.py`.

### `pip install` fails with "externally-managed-environment"

PEP 668. Use the venv in §1. Behind a proxy, set `HTTPS_PROXY` or use
`pip install -i <mirror-url> -r requirements.txt`.

### Run All hangs partway down

`INTERACTIVE_CHAT = True` and a cell is waiting on `input()`. Type `quit`, or set it back to
`False`.
