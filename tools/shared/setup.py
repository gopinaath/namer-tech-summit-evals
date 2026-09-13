# Shared setup cell: imports workshop_setup.connect() and binds what it returns.
# Spliced into a notebook source with:  # %%%% include=setup
# Rebuild with:  python tools/nbbuild.py
#
# The work itself lives in workshop_setup.py at the repo root: the kernel guard, credential
# resolution and model pinging whose only value is the message it prints when something is
# wrong. That was 330 lines across two cells; this is 29 in one, so the first thing an
# attendee reads is "Why evals?" rather than the plumbing. It ships in the same folder as the
# notebooks, so the handout is still self-contained.

# %% [markdown]
# ## Setup — connect to Claude on Amazon Bedrock
#
# Run the next cell. The first time, it writes a gitignored **`.env`** next to this
# notebook — fill it in, save, re-run. Either credential type works, and the cell detects
# which you have:
#
# | | Set this |
# |---|---|
# | **Bedrock API key** | `AWS_BEARER_TOKEN_BEDROCK` + `AWS_REGION` |
# | **IAM** (SSO, `aws configure`, assumed role) | `AWS_REGION`, plus `AWS_PROFILE` if named |
#
# Any region with a Bedrock endpoint will do — the notebook calls Claude through **global
# inference profiles**, so there's no region to hunt for. The cell pings each model and
# reports which ones your account can reach.
#
# Green **"✓ Connected to Claude on Bedrock"** means you're done. Yellow or red tells you
# what to fix. The code doing all of this is in **`workshop_setup.py`** beside this
# notebook — worth a read once setup is working, but nothing below depends on you opening it.

# %% id=setup
# Connect to Bedrock. Everything this cell does lives in workshop_setup.py — installing the
# SDK if needed, reading .env, and pinging each model. Safe to re-run at any point.
import importlib
import pathlib
import sys

# Find the workshop folder. Usually that's the working directory; the demo copy of this
# notebook sits one level down, hence the walk upwards.
_root = next((d for d in (pathlib.Path.cwd(), *pathlib.Path.cwd().parents)
              if (d / "workshop_setup.py").is_file()), None)
if _root is None:
    raise SystemExit("\n  ✗ Can't find workshop_setup.py. Open this notebook from the "
                     "workshop folder\n    (the one containing requirements.txt), then run "
                     "this cell again.\n")
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import workshop_setup

# Re-read the file rather than trusting Python's module cache, so editing workshop_setup.py
# takes effect when you re-run this cell — the same as when this code was pasted inline.
_setup = importlib.reload(workshop_setup).connect()

client, make_client, short_model_name = _setup.client, _setup.make_client, _setup.short_model_name
MODEL, FAST_MODEL, BIG_MODEL, FABLE_5_1_MODEL = (_setup.MODEL, _setup.FAST_MODEL,
                                                 _setup.BIG_MODEL, _setup.FABLE_5_1_MODEL)
OPTIONAL_MODELS, AVAILABLE_MODELS = _setup.OPTIONAL_MODELS, _setup.AVAILABLE_MODELS
AUTH_MODE, _region, _token, _profile = (_setup.auth_mode, _setup.region,
                                        _setup.token, _setup.profile)
