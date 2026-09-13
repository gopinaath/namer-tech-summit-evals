# Shared setup cells: kernel/install guard, then Bedrock credentials + model list.
# Spliced into a notebook source with:  # %%%% include=setup
# Rebuild with:  python tools/nbbuild.py

# %%
# Check the kernel, then install what's missing into it. Safe to re-run.
import importlib.util, os, subprocess, sys


def _disposable_runtime():
    """True in a hosted notebook — SageMaker Studio, a notebook instance, any managed
    JupyterLab image. These run as root in a container that is rebuilt on restart, so
    `pip install` into the image's own Python (usually /opt/conda) is the normal thing
    to do: there is no shared machine to damage."""
    if os.path.exists("/.dockerenv") or os.path.isdir("/opt/ml"):
        return True
    if any(k.startswith(("SAGEMAKER", "SM_", "STUDIO_LAB", "AWS_JUPYTER")) for k in os.environ):
        return True
    try:                                               # containerd / EKS / ECS / Fargate
        with open("/proc/1/cgroup") as fh:
            cgroup = fh.read()                         # read once: the handle is consumed
    except OSError:
        return False                                   # no /proc: macOS, Windows
    return any(m in cgroup for m in ("docker", "containerd", "kubepods", "ecs"))


def _safe_to_install():
    """True when `pip install` here can't hurt anything the attendee cares about.

    A pip install goes wherever this kernel lives, so on a laptop a system-Python kernel
    would mutate Python machine-wide. Judged from the interpreter, not from VIRTUAL_ENV: a
    system-Python kernel launched from an activated terminal inherits that variable and
    must not pass."""
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return True                                    # python -m venv, virtualenv
    if _disposable_runtime():
        return True                                    # hosted image — mutating it is fine
    conda = os.environ.get("CONDA_PREFIX")             # conda envs aren't PEP 405 venvs,
    return bool(                                       # so the check above misses them
        conda
        and os.path.realpath(sys.executable).startswith(os.path.realpath(conda) + os.sep)
        and os.environ.get("CONDA_DEFAULT_ENV", "base") != "base"   # shared base doesn't count
    )


# Same cell as the install below, deliberately: a separate guard cell can be skipped.
if not _safe_to_install() and os.environ.get("WORKSHOP_ALLOW_SYSTEM_PYTHON") != "1":
    raise SystemExit(
        f"\n  ✗ This kernel is a shared Python ({sys.executable}).\n"
        "    Installing here would change Python for your whole machine.\n\n"
        "    On a laptop, make an environment and select it as the kernel:\n"
        "        python3 -m venv .venv && source .venv/bin/activate\n"
        "        pip install -r requirements.txt\n"
        "        # then: VS Code kernel picker (top right) -> the .venv interpreter\n\n"
        "    In a hosted notebook where this Python IS disposable, run this above, then\n"
        "    re-run the cell:\n"
        '        import os; os.environ["WORKSHOP_ALLOW_SYSTEM_PYTHON"] = "1"\n\n'
        "    Details in SETUP.md.\n"
    )

# boto3 is only needed for the IAM-credentials path; a Bedrock API key works without it.
if importlib.util.find_spec("anthropic") is None:
    print("Installing anthropic — first run only, please wait…", flush=True)
    _pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anthropic"],
                          capture_output=True, text=True)
    if _pip.returncode:
        raise SystemExit("\n  ✗ pip install failed:\n      "
                         + "\n      ".join((_pip.stderr or _pip.stdout).strip().splitlines()[-3:])
                         + "\n  Behind a proxy or a private index? See SETUP.md.\n")

print(f"✓ Ready — anthropic installed for {sys.executable}")

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
# what to fix.

# %%
import os
import pathlib
import re

import anthropic


def _status(ok, msg):
    """Green/red banner in notebooks; plain text when run as a script."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None or shell.__class__.__name__ != "ZMQInteractiveShell":
            raise RuntimeError("not in a notebook kernel - use the plain-text banner")
        from IPython.display import display, HTML
        color = "#1a7f37" if ok else "#b42318"
        bg = "#e6f4ea" if ok else "#fdecea"
        icon = "✓" if ok else "✗"
        display(HTML(
            f'<div style="padding:12px 16px;border-radius:8px;background:{bg};'
            f'border:1.5px solid {color};color:{color};font-weight:600;'
            f'font-size:15px;font-family:sans-serif;">{icon} {msg}</div>'
        ))
    except Exception:
        print(("[OK] " if ok else "[!!] ") + msg)


def _needs_credentials(head, body):
    """Warning-yellow banner + stop, so setup fails here rather than several cells later."""
    _shown = False
    try:
        from IPython import get_ipython
        if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
            import html as _html
            from IPython.display import HTML, display
            display(HTML(
                '<div style="padding:12px 16px;border-radius:8px;background:#fff8c5;'
                'border:1.5px solid #9a6700;font-size:15px;font-family:sans-serif;">'
                '<div style="color:#9a6700;font-weight:600;">' + _html.escape(head) + '</div>'
                '<pre style="margin:10px 0 0;font-family:inherit;font-size:14px;font-weight:400;'
                'color:#141413;white-space:pre-wrap;">' + _html.escape(body) + '</pre></div>'
            ))
            _shown = True
    except Exception:
        pass
    if not _shown:
        print("\n" + head + ":\n   " + body.replace("\n", "\n   ") + "\n")
    raise SystemExit("Credentials missing — see the message above.")


_ENV_TEMPLATE = (
    "# NAMER Tech Summit — Building an Eval. Credentials for Claude on Amazon Bedrock.\n"
    "# This file is gitignored. Paste values after the = (no quotes, no spaces),\n"
    "# save, then re-run the setup cell.\n"
    "\n"
    "# --- Option A: Bedrock API key (what most attendees get) ---\n"
    "AWS_BEARER_TOKEN_BEDROCK=paste-your-bedrock-api-key-here\n"
    "AWS_REGION=us-east-1\n"
    "\n"
    "# --- Option B: IAM credentials (SSO, `aws configure`, assumed role) ---\n"
    "# Delete the AWS_BEARER_TOKEN_BEDROCK line above, keep AWS_REGION, and\n"
    "# optionally name a profile:\n"
    "# AWS_PROFILE=my-profile\n"
)


def _resolve_env_file():
    """Prefer a .env in this workshop folder, then the nearest one walking up (so a shared
    .env one level up still works). If none exists, create one right here."""
    here = pathlib.Path.cwd().resolve()
    for d in [here, *here.parents]:
        if (d / ".env").is_file():
            return d / ".env"
    return here / ".env"


_env_file = _resolve_env_file()
if not _env_file.exists():
    _env_file.write_text(_ENV_TEMPLATE)
    print(f"Created {_env_file.name} in {_env_file.parent} — open it, add your Bedrock "
          "credentials, save, then re-run this cell.")

# Tiny .env parser (no python-dotenv dependency). Re-read on every run, so pasting a value
# and re-running picks it up. A real value already in the environment (shell / CI) wins.
_file = {}
for _line in (_env_file.read_text().splitlines() if _env_file.exists() else []):
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        _file[_k.strip()] = _v.strip().strip('"').strip("'")
for _k, _v in _file.items():
    os.environ.setdefault(_k, _v)

_token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip()
if _token.startswith("paste-"):
    # The placeholder from the template never counts as a credential. Remove it from the
    # environment too: the SDK reads AWS_BEARER_TOKEN_BEDROCK itself, and refuses to accept
    # a bearer token and IAM credentials at the same time — so leaving the placeholder set
    # would break the IAM path with a confusing "cannot specify both" error.
    _token = ""
    os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
_region = os.environ.get("AWS_REGION", "").strip() or os.environ.get("AWS_DEFAULT_REGION", "").strip()
_profile = os.environ.get("AWS_PROFILE", "").strip()

if not _region:
    _needs_credentials(
        "📋 Bedrock needs a region",
        f"Open this file:  {_env_file}\n"
        "and set the region your Claude models are enabled in, e.g.:\n"
        "    AWS_REGION=us-east-1\n"
        "\n"
        "Save the file, then click ▶ on this cell again."
    )

if _token:
    AUTH_MODE = "api-key"
elif _profile or os.environ.get("AWS_ACCESS_KEY_ID") or (pathlib.Path.home() / ".aws" / "credentials").exists():
    AUTH_MODE = "iam"
else:
    _needs_credentials(
        "📋 Add your Bedrock credentials to continue",
        f"Open this file:  {_env_file}\n"
        "\n"
        "Option A — Bedrock API key (simplest):\n"
        "    AWS_BEARER_TOKEN_BEDROCK=<your Bedrock API key>\n"
        f"    AWS_REGION={_region or 'us-east-1'}\n"
        "\n"
        "Option B — IAM credentials: sign in with the AWS CLI (`aws sso login`, or\n"
        "`aws configure`), then set AWS_PROFILE here if you use a named profile.\n"
        "\n"
        "Save the file, then click ▶ on this cell again."
    )

# ── Models ────────────────────────────────────────────────────────────────
# Global cross-region inference profiles, so these work on-demand from any Bedrock region
# and nobody has to hunt for a region. If data residency rules out global routing, switch
# to your geography's prefix (`us.`, `eu.`, `apac.`) and stay in a matching region.
#
# Undated IDs are aliases that track the current snapshot; Haiku 4.5 only publishes a dated
# profile, hence the version suffix on that one.
MODEL = "global.anthropic.claude-sonnet-5"                       # the workhorse
FAST_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"   # cheap + quick (agent, judge)
BIG_MODEL = "global.anthropic.claude-opus-5"                     # the largest, for comparison
FABLE_MODEL = "global.anthropic.claude-fable-5"                  # \ two more points on the
FABLE_5_1_MODEL = "global.anthropic.claude-fable-5-1"            # / capability/cost curve

# Everything except FAST_MODEL is optional: the notebook runs start to finish on Haiku alone.
# The setup cell pings each of these and keeps the ones your account can actually reach, so a
# missing grant costs you a table row rather than an hour of debugging.
OPTIONAL_MODELS = [MODEL, BIG_MODEL, FABLE_MODEL, FABLE_5_1_MODEL]


def short_model_name(model_id):
    """'global.anthropic.claude-haiku-4-5-20251001-v1:0' -> 'haiku-4-5'.

    Profile IDs are long and contain a ':', which is fine in an API call but breaks table
    alignment and is an illegal filename character on Windows. Used for display and for
    naming saved result files."""
    name = str(model_id or "default")
    for prefix in ("global.", "us.", "eu.", "apac.", "anthropic.", "claude-"):
        name = name.replace(prefix, "")
    return re.sub(r"-\d{8}-v\d+:\d+$", "", name) or "default"


def make_client(timeout=900.0, max_retries=4):
    """A Bedrock client for whichever credential type you have.

    Talks to the `bedrock-runtime` endpoint for your region — the same API the AWS SDKs
    use, so anything you learn here transfers straight to boto3.

    max_retries is deliberately generous: Bedrock throttles per-account, and the eval
    runner fires several requests at once. The SDK retries 429s with backoff for us."""
    kwargs = {"aws_region": _region, "timeout": timeout, "max_retries": max_retries}
    if AUTH_MODE == "api-key":
        return anthropic.AnthropicBedrock(api_key=_token, **kwargs)
    if _profile:
        kwargs["aws_profile"] = _profile
    return anthropic.AnthropicBedrock(**kwargs)


# Connection check — verifies the credential AND that the model is reachable for you.
# A perfectly valid Bedrock credential still fails on a model your account hasn't been
# granted, so we ping the real model IDs rather than just checking the key's shape.
_probe = make_client(timeout=30.0, max_retries=1)


def _ping(model_id):
    """True if this exact model ID answers for these credentials in this region.

    `bedrock-runtime` reports an unusable model as a 400, not a 404 — "The provided model
    identifier is invalid" for an ID that doesn't exist in this region, and "Invocation ...
    with on-demand throughput isn't supported" for a plain model ID that needs an inference
    profile. Both mean "not usable here", so both count as False rather than an error."""
    try:
        _probe.messages.create(model=model_id, max_tokens=1,
                               messages=[{"role": "user", "content": "ping"}])
        return True
    except (anthropic.NotFoundError, anthropic.BadRequestError):
        return False


try:
    _agent_model_ok = _ping(FAST_MODEL)
except anthropic.PermissionDeniedError:
    # AccessDenied is ambiguous on Bedrock: the credential may lack bedrock:InvokeModel, or
    # it may be fine and this particular model just isn't granted. Name both — guessing one
    # sends people to the wrong console page.
    _status(False, f"Bedrock denied the request for '{FAST_MODEL}'. Either these credentials "
                   "lack bedrock:InvokeModel, or model access for Claude Haiku hasn't been "
                   "granted in this account (Bedrock console → Model access).")
    raise SystemExit("Access denied — see the message above.")
except anthropic.AuthenticationError:
    hint = ("Check AWS_BEARER_TOKEN_BEDROCK — paste the whole key — and that it has "
            "bedrock:InvokeModel permission."
            if AUTH_MODE == "api-key" else
            f"Your IAM credentials{' (profile ' + _profile + ')' if _profile else ''} were "
            "rejected. Refresh them (`aws sso login`) and confirm bedrock:InvokeModel.")
    _status(False, "Bedrock rejected those credentials. " + hint)
    raise SystemExit("Credentials not accepted — re-run this cell after fixing them.")
except ModuleNotFoundError:
    _status(False, "The IAM-credentials path needs boto3. Run: pip install 'anthropic[bedrock]' "
                   "— or switch to a Bedrock API key (AWS_BEARER_TOKEN_BEDROCK).")
    raise SystemExit("boto3 missing — see the message above.")
except Exception as exc:
    _status(False, "Could not reach Bedrock (" + type(exc).__name__ + "). Check your "
                   "connection / VPN, then run this cell again.")
    raise

if not _agent_model_ok:
    _status(False, f"Bedrock reached from {_region}, but this account can't use "
                   f"'{FAST_MODEL}'. Request model access in the Bedrock console (Model "
                   f"access). This is a global inference profile, so switching AWS_REGION "
                   f"won't help — the grant is account-level.")
    raise SystemExit("Model not available — see the message above.")

# Model access is granted per account, one model at a time — so the model powering the agent
# working tells you nothing about the others. Check them all now, for three tokens, rather
# than discovering it in Part 7 an hour from now. AVAILABLE_MODELS is what the sweep uses.
def _reachable(model_id):
    try:
        return _ping(model_id)
    except Exception:
        # The credential is already proven by this point, so anything else — an AccessDenied
        # for this specific model, a throttle, a dropped connection — means "can't use it
        # today". Treat it as unavailable rather than failing setup over an optional model.
        return False


AVAILABLE_MODELS = [FAST_MODEL] + [m for m in OPTIONAL_MODELS if _reachable(m)]
_missing = [m for m in OPTIONAL_MODELS if m not in AVAILABLE_MODELS]

_status(True, f"Connected to Claude on Bedrock — {_region}, auth: {AUTH_MODE}. "
              f"Models available: "
              f"{', '.join(short_model_name(m) for m in AVAILABLE_MODELS)}.")
if _missing:
    print(f"Note: {', '.join(short_model_name(m) for m in _missing)} not available to this "
          f"account — the Part 7 model comparison will skip them. Everything else works. "
          f"These are global inference profiles, so changing AWS_REGION won't help: request "
          f"access in the Bedrock console under Model access.")

# The working client used by everything below.
client = make_client()
