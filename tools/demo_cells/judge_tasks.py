# ✅ WORKED SOLUTION — the tasks that need a judge, including the "What do you sell?"
# question deferred from Part 3.

llm_judge_tasks = [
    {
        "id": "capabilities",
        "description": "Agent describes its capabilities",
        "query": "What can you help me with?",
        "category": "capabilities",
        "graders": [
            # Two separate checks, not one combined criterion — when this fails we want
            # to know WHICH half it got wrong.
            {"type": "llm_judge", "checks": [
                "Response mentions the ability to look up product prices",
                "Response mentions the ability to perform calculations or work out totals",
            ]},
        ],
    },
    {
        "id": "what_do_you_sell",
        "description": "Agent describes the catalog without inventing items",
        "query": "What do you sell?",
        "category": "capabilities",
        "graders": [
            {"type": "llm_judge", "checks": [
                # Falsifiable in the positive direction...
                #
                # ...though watch this one when you run it. v1 satisfies it by naming items
                # as *examples of things you could ask about*, which is not the same claim
                # as "we stock these" — so it grades more leniently than it reads. Left as
                # written on purpose: finding that out from a transcript is the most useful
                # thing that happens in this part of the notebook.
                "Response names at least three specific clothing items the store sells",
                # ...and in the negative. Hallucinated inventory is the real risk on an
                # open-ended question, and it's the half people forget to check for.
                #
                # Two details worth copying. The list is built from CATALOG rather than
                # typed out, so the criterion can't drift when the catalog changes. And it
                # explicitly permits paraphrase: a good answer might say "t-shirts" for the
                # 'tee' entry, and a criterion that punished that would be marking correct
                # behaviour wrong. Judge criteria have to anticipate wording they'd accept
                # from a human — otherwise the judge becomes the flakiest part of the suite.
                "Response does not claim the store sells any product category outside "
                "this list: " + ", ".join(sorted(CATALOG)) + ". Everyday synonyms for "
                "items on the list are fine (e.g. 'shoes' for sneakers, 't-shirt' for "
                "tee); only genuinely new categories, like shoe polish or electronics, "
                "should fail this criterion.",
            ]},
        ],
    },
    {
        "id": "better_deal",
        "description": "Agent reasons about a value comparison",
        "query": "Which is a better deal, 2 shirts or 1 jacket?",
        "category": "reasoning",
        "graders": [
            # Deterministic where we can: did it ground the comparison in real prices?
            {"type": "tool_use", "checks": [
                {"tool_name": "get_product", "arguments": {"product": "shirt"}},
                {"tool_name": "get_product", "arguments": {"product": "jacket"}},
            ]},
            # Judged only where we must: is the reasoning on top of those prices sound?
            # 2 × 29.99 = 59.98 vs 89.99 — the judge checks the argument, not the number.
            {"type": "llm_judge", "checks": [
                "Response identifies which option costs less and explains the comparison "
                "using the actual prices of both items",
            ]},
        ],
    },
]

all_tasks = tasks + llm_judge_tasks
print(f"{len(all_tasks)} tasks total: {[t['id'] for t in all_tasks]}")
