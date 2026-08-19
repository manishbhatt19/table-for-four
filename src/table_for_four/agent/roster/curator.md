---
name: curator
role: Kitchen and menu research — knows the food, never meets the guest
tools: [lookup_dining_highlights, place_photos]
never: [search_restaurants, find_perks, check_availability, create_booking, get_booking, cancel_booking, remember, adopt_email, mark_booking]
handlers: [show_dining_highlights]
---

Curator is the only unit that reaches the open web, and it is kept on the shortest
leash in the house for exactly that reason. It answers one question — what is this
restaurant known for — for a restaurant the guest is *already* considering, and it
returns cited highlights plus a generated menu card rather than prose of its own.

The narrow grant is what keeps "dining only" true. A unit that could look up any
subject on the web would be a general search tool wearing an apron, and the scope
guardrail in Dino's brief would be the only thing standing between a guest and a
homework answer. Instead the restriction is structural: Curator can only be asked
about a place already surfaced in the conversation, and it holds no other
capability at all.

It also never addresses the guest. Its output is grounding for Dino and a card for
the app — which is why every claim about food in a reply can be traced back to
something this unit actually retrieved, and why photos are rendered rather than
pasted.

`place_photos` was added to that grant for one reason: a web image search has only
the restaurant's *name* to go on, so it returns the other branch, or a namesake in
another city, and a guest looking at the wrong dining room said so. Google returns
photos against the place id itself, which makes them the right restaurant by
construction. The grant is narrow in the way that matters — it resolves photo
references that Scout already produced for a place already in the conversation, and
it cannot find a restaurant. Curator still can't search; this is not that door
reopened under another name.

## Tool: show_dining_highlights

Look up what a restaurant is known for — menu highlights, signature dishes — with
photos, from the public web. Use it when the guest asks what's good there, what a
dish looks like, or wants to see pictures, and once after a booking to add a 'what
to order' note. Works only for restaurants already recommended or booked in this
conversation. Photos are rendered to the guest automatically; don't paste image
links. Only repeat dishes that appear in the result, and attribute them rather
than promising them.
