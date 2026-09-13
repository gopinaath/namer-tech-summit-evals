# Building an Eval for an AI Shopping Assistant — WORKED DEMO

### NAMER Tech Summit · facilitator / reference copy · Amazon Bedrock

> **This is the completed version.** Every ✏️ YOUR TURN cell is filled in. Use it to
> run the whole story end to end in front of a room, or hand it out afterwards as the
> reference. Participants work in `../Building_an_Eval.ipynb`.

`boutique` is a small shopping assistant that looks up product prices and does math.
It works... mostly. The job is to find out **exactly** where it breaks, **why**, and
whether a fix actually helped — with numbers, not vibes.

Everything runs against **Claude on Amazon Bedrock**.

**The story this notebook tells, in five beats:**

| Beat | Cell | What the room should see |
|---|---|---|
| 1. It looks fine | Part 1 probes | Fluent answers. Some are wrong. |
| 2. Measure it | Part 4 baseline | ~40-60% pass. Now the damage is visible. |
| 3. Diagnose | `inspect_task` | Every failure has a *different* root cause. |
| 4. Fix and re-measure | Part 5 `compare_results` | Pass rate jumps; no regressions. **This is the punchline.** |
| 5. Grade the unstringmatchable | Part 6 judge | Judge on reasoning, deterministic checks on facts. |

**Run All takes roughly 8-12 minutes** (the multi-run baselines and the model sweep
dominate). If you're tight on time, run Parts 1-5 live and talk through 6-7.
