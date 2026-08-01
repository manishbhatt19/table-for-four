# Table for Four — Retrieval-Augmented Generation (RAG)

**CMU AI Agent Certification — Capstone · Week 3 Submission**
**Author:** Manish Bhatt

> Week 3 focuses on **RAG**: what it is, why this project needs it, and exactly how
> it is implemented here. The primary RAG surface is the **perks** layer — a hybrid
> semantic + metadata retrieval over an unstructured offers store — which is already
> built and wired into the concierge journey. A second, lighter retrieval surface
> (long-term member memory) is described at the end. As with the earlier
> submissions, this is design-and-rationale writing grounded in the working code.

---

## 1. What RAG is

A language model only knows what was in its training data, frozen at training time.
**Retrieval-Augmented Generation** fixes that by putting a *retrieval* step in front
of *generation*: instead of asking the model to answer from memory, you first fetch
the relevant facts from an external knowledge store and hand them to the model as
grounded context. Three parts:

1. **A knowledge store** — documents indexed for retrieval, usually as **embeddings**
   (vectors that place semantically similar text near each other).
2. **A retriever** — given a query, it embeds the query and returns the closest
   documents (plus any structured filtering).
3. **Generation** — the model reasons over the *retrieved* material, so its answer
   is grounded in real, current, inspectable data rather than invented.

RAG is the right tool when the knowledge is **large, changing, private, or
unstructured** — exactly where fine-tuning or prompt-stuffing falls down.

---

## 2. Why this project needs RAG

The concierge has to surface **perks** (offers/coupons) that fit a specific dining
intent — *"a birthday dinner, one guest is gluten-free, party of four, this
Friday."* Perks are written as free-text marketing blurbs (cuisine, vibe, dietary
fit, occasion). Two things make this a retrieval problem, not a prompting one:

- **The model was never trained on our offers.** They are our own (synthetic)
  data; the model cannot know them, and asking it to guess would fabricate offers.
- **The match is *semantic*, not keyword.** *"celiac-safe birthday"* should surface
  a *"gluten-free tasting flight for celebrations"* even with no shared keywords.
  Exact-match filtering can't do this; embedding similarity can.

So perks are a natural RAG fit: an unstructured, private knowledge store that must
be searched by meaning and then handed to the agent as grounded options.

> **Note — not everything retrieval-heavy is RAG.** Restaurant *search* (Google
> Places) is structured API retrieval by keyword/location, not RAG. The RAG layer
> is specifically the **perks** store, where semantic similarity over unstructured
> text is what adds value.

---

## 3. How we do it — the perks RAG pipeline

Implemented in [`mcp_servers/perks_server.py`](../mcp_servers/perks_server.py),
exposed as the MCP tool **`find_perks`**.

**a) Knowledge store — Chroma vector DB.**
Each perk is stored once in a local **Chroma** collection (`restaurant_perks`):
- the **document** is the perk's unstructured `blurb` — this is what gets embedded;
- the **metadata** is the structured fields (`place_id`, `min_party_size`,
  `valid_days`, `expiry`, `active`, `perk_type`, …) — kept alongside for filtering.

**b) Embeddings — local, offline, no API key.**
Documents are embedded with **`all-MiniLM-L6-v2`** (384-dim sentence embeddings) via
Chroma's default embedding function. It runs locally (~80 MB, downloaded once), so
the whole RAG path works with **no API key and no per-query cost** — consistent with
the project's offline-first principle. The collection uses **cosine** space, so a
distance converts cleanly to a `0–1` **similarity** score we can report and rank on.

**c) Retriever — *hybrid* semantic + structured.**
`find_perks` does three things in sequence (`query_perks`):
1. **Metadata pre-filter** (`where`): only `active` perks, optionally restricted to a
   candidate `place_id` set and to `min_party_size ≤ party_size`.
2. **Vector search**: embed the query text and fetch the nearest blurbs *within* that
   filtered set. We deliberately **over-fetch** (`n × 3`) so the next step still has
   enough candidates.
3. **Python post-filters** that Chroma's `where` can't express cleanly: drop
   **expired** perks (`expiry ≥ today`) and perks not valid on the requested **day**.

The result is a ranked list of perks, each carrying its `similarity` score, its
structured fields, and `source: "synthetic"`.

**d) Generation — grounding the concierge.**
The retrieved perks are handed to the orchestrator, which attaches them to the
restaurant shortlist. The LLM then *presents* real, retrieved offers ("2 of these 3
carry a perk that fits a gluten-free group of four") rather than inventing any —
retrieval grounds generation.

---

## 4. Why *hybrid* retrieval (and not pure vectors)

Pure vector search would happily return a semantically similar perk that is
**expired**, **too small a party**, or **for the wrong restaurant**. Pure metadata
filtering can't understand *"celiac-safe birthday."* The hybrid split assigns each
half the job it's good at:

| Concern | Handled by | Example |
|---|---|---|
| *Meaning* / intent | **Vector search** on the blurb | "gluten-free celebration" → celiac-safe tasting flight |
| *Hard constraints* | **Metadata `where`** | active only, this restaurant, party ≥ minimum |
| *Time validity* | **Post-filter** | not expired; valid on Friday |

The vector DB carries weight **only** where semantics genuinely add value; hard
facts stay as exact filters. This is the core design decision of the layer.

---

## 5. How RAG fits the agent journey

In the concierge journey ([`agent/concierge_chat.py`](../agent/concierge_chat.py),
`recommend_restaurants`), RAG runs at the recommendation step:

1. Search returns candidate restaurants (Google Places or fixture).
2. `find_perks` is called **scoped to those candidates' `place_id`s** — so perks
   line up with the exact shortlist, and the best perk per restaurant is attached.
3. Ava presents the shortlist and names which one or two carry an offer.

**A RAG nuance — semantic-only fallback on live data.** Our perks are synthetic and
keyed to the *fixture* restaurants. With **live** Google results (real, different
IDs), the `place_id`-scoped retrieval matches nothing. The system then re-queries
the perks store **without the `place_id` filter** — pure semantic retrieval by
cuisine/intent — and attaches the top one or two matches to the recommendations as
clearly-labeled **sample partner offers**. This is the same RAG engine used two
ways: *filtered* retrieval when we own the catalog, *open* semantic retrieval when
we only want the most relevant offer regardless of source.

---

## 6. A second retrieval surface — long-term member memory

[`agent/profile_memory.py`](../agent/profile_memory.py) also uses Chroma. Each
returning member has one profile document (keyed by email); we store the full
profile as JSON metadata and embed a **human-readable summary** of it. Today it is
retrieved primarily by key (recognize a returning guest and reuse their
preferences), but embedding the summary means the store is **retrieval-ready**: it
can later support semantic recall ("the guest who loves Sicilian wine") on the same
infrastructure. It is retrieval-augmented *personalization* built on the same
Chroma + local-embeddings stack as the perks RAG.

---

## 7. Design choices & trade-offs

- **Local embeddings over an embedding API** — zero key, zero cost, fully offline
  and reproducible for grading; the trade-off is a smaller model than a hosted one,
  which is more than adequate for short offer blurbs.
- **Synthetic, labeled knowledge** — there is no cleanly-licensable coupon dataset,
  so perks are synthetic and every result carries `source: "synthetic"`; sample
  offers on live restaurants are additionally flagged as illustrative. Provenance is
  part of the design, not an afterthought.
- **Over-fetch then post-filter** — cosine top-k *then* expiry/day filtering avoids
  an empty result set when the nearest matches happen to be time-invalid.
- **Cosine similarity surfaced as a score** — retrieval is inspectable: every perk
  reports *why* it ranked, supporting the governance/audit goal.

---

## 8. Responsible-AI angle

RAG is also a **safety** mechanism here. By forcing offers to come from a retrieved,
labeled store rather than the model's imagination, we make the "a perk is available"
claim **verifiable** and prevent hallucinated discounts. Every retrieved item is
tagged with its source; illustrative sample offers are marked as such; and nothing
retrieval-grounded is passed off as more real than it is — the same honesty stance
the booking gate and audit trail take elsewhere in the system.

---

## 9. Summary

The perks layer is a textbook **hybrid RAG**: an unstructured, private knowledge
store (perk blurbs) indexed as **local embeddings in Chroma**, retrieved by
**semantic similarity + structured metadata + time filters**, and handed to the
agent so its recommendations are **grounded, current, and inspectable** rather than
invented. The same engine powers both catalog-scoped retrieval (offline/fixture) and
open semantic retrieval (live data), and the same stack underpins long-term member
memory — making RAG a load-bearing, reusable part of the architecture, not a bolt-on.

---

*Week 3 submission · v1.0.*
