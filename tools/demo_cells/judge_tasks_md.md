### Tasks that need a judge

Three tasks, and the thing to point out is how `better_deal` **mixes graders**:
`tool_use` pins the facts deterministically, and the judge assesses only the reasoning
built on top of them.

That split is the whole pattern: **deterministic where you can, judge where you must.**
Every check you can make deterministic is one you don't pay for, don't wait for, and
don't have to trust.

Also note the second criterion on `what_do_you_sell` — it checks for the *absence* of
invented inventory. On open-ended questions the interesting failure isn't an incomplete
answer, it's a confident one about products you don't stock, and a criterion phrased
only in the positive direction will never catch it.
