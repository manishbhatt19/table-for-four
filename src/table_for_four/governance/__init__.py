"""M4 — governance: the checks that sit between the model and the guest.

The product's guardrails have always been refusals at the tool boundary: a
booking may only name a restaurant Scout surfaced, at a time the backend
offered, for a party size the guest gave. Eleven of them stand in front of one
`create_booking` call.

What none of them cover is the **reply**. A handler can refuse to book a table
at a time nobody offered; nothing stops the sentence "how does 7pm sound?" from
reaching the guest. `grounding` is that check, and it is deliberately code
rather than a second model: a claim with an exact answer should be checked, not
estimated — the same argument `docs/week4_tree_of_thought.md` §3.4 makes about
constraints, applied one layer out.
"""
