# Testing guide — Table for Four

Quick ways to exercise the system: the automated suite first, then hands-on checks
for the RAG layer, the bookings ledger, and the cancellation policy. Everything
runs offline unless noted.

---

## 1. Automated tests (fastest confidence)

```bash
uv run pytest -q                       # whole suite (should be all green)
uv run pytest tests/test_booking.py -q          # ledger + 24h cancellation
uv run pytest tests/test_perks_server.py -q     # perks hybrid retrieval + weighting
uv run pytest tests/test_perks_eval.py -q       # retrieval quality (hit@k / MRR)
uv run pytest tests/test_profile_memory.py -q   # memory, semantic recall, cancel flow
uv run pytest -q -k cancel                       # just the cancellation tests
```

Tests use an in-memory SQLite DB and offline fixtures, so they never touch your
real `bookings.db` or hit any paid API.

---

## 2. Perks RAG — evaluate & inspect

```bash
# Retrieval quality on 10 labelled intents (expect hit@3 = 1.0, MRR = 1.0)
uv run python -m table_for_four.mcp_servers.perks.eval

# How the semantic-vs-metadata weight trades off precision
uv run python -m table_for_four.mcp_servers.perks.eval --sweep

# Inspect ONE query: see the ranked perks + why (sim / fit / score) + filters
uv run python -m table_for_four.mcp_servers.perks.inspect "romantic dinner with wine" --party 2
uv run python -m table_for_four.mcp_servers.perks.inspect "tacos with friends" --day Tue --weight 0.3
uv run python -m table_for_four.mcp_servers.perks.inspect "gluten-free birthday" --blurb
```

**What to look for:** the right restaurant tops the list for each intent; lowering
`--weight` toward 0 lets metadata fit (party/day) reorder results.

Semantic **member** recall (the "guest who loves X" search):

```bash
uv run python -c "from table_for_four.agent.profile_memory import remember, find_members; remember('wine@x.com', {'name':'Giulia','cuisines':['Italian'],'interests':['Sicilian wine']}); remember('tacos@x.com', {'name':'Diego','interests':['street tacos']}); print([(m['name'], m['similarity']) for m in find_members('the guest who loves Sicilian wine')])"
```

Expect **Giulia** ranked first. (This writes to the real profile store; harmless demo data.)

---

## 3. Bookings ledger & cancellation — via the live API

Start the backend (serves interactive Swagger docs — the easiest way to poke it):

```bash
uv run uvicorn table_for_four.mcp_servers.booking.backend.app:app --port 8000
```

Open **http://localhost:8000/docs** and use "Try it out", or use PowerShell below.

### a) See open slots, then book

```powershell
# What times are open? (availability is deterministic per place+date)
Invoke-RestMethod "http://localhost:8000/availability?place_id=fixture-osteria-1&date=2026-09-04&party_size=2"

# Book one of the returned times (swap 19:00 for a slot that came back)
$body = @{ place_id='fixture-osteria-1'; restaurant_name='Osteria Midtown'
           address='127 W 44th St'; restaurant_phone='(212) 555-0142'
           website='https://example.com/osteria-midtown'
           date='2026-09-04'; time='19:00'; party_size=2
           guest_name='Manish'; guest_email='you@example.com' } | ConvertTo-Json
$booking = Invoke-RestMethod -Method Post "http://localhost:8000/bookings" -ContentType 'application/json' -Body $body
$booking      # note the confirmation_id, e.g. TF4-0001, status = confirmed
```

### b) Cancel — the 24-hour policy

The `now` field lets you simulate the clock so you can see both outcomes.

```powershell
$id = $booking.confirmation_id

# MORE than 24h before the booking -> cancels successfully
$ok = @{ now='2026-09-01T19:00:00'; reason='plans changed' } | ConvertTo-Json
Invoke-RestMethod -Method Post "http://localhost:8000/bookings/$id/cancel" -ContentType 'application/json' -Body $ok
# -> status = cancelled, cancelled_at + reason recorded
```

Book again, then try inside the window:

```powershell
# WITHIN 24h -> refused (409), returns the restaurant's phone + website to call
$late = @{ now='2026-09-04T10:00:00' } | ConvertTo-Json
Invoke-RestMethod -Method Post "http://localhost:8000/bookings/$id/cancel" -ContentType 'application/json' -Body $late
# (409) detail.error = within_24h, detail.restaurant_phone / detail.website
```

Real cancellations omit `now` and use the server clock.

### c) The ledger persists

```powershell
Invoke-RestMethod "http://localhost:8000/bookings"                              # all
Invoke-RestMethod "http://localhost:8000/bookings?email=you@example.com"        # by guest
Invoke-RestMethod "http://localhost:8000/bookings?status=cancelled"             # by status
```

Stop the server (Ctrl+C) and restart it — the bookings are still there (they live
in `src/table_for_four/mcp_servers/booking/backend/bookings.db`).

---

## 4. Talk to Dino (end-to-end, needs an OpenAI/OpenRouter key)

Requires `OPENAI_API_KEY` (or OpenRouter) in `.env`.

```bash
uv run python -m table_for_four chat            # terminal
uv run streamlit run src/table_for_four/ui/chat_app.py # web UI with live memory panel
```

**A booking + cancellation script to try:**

1. "Hi, I'd like an Italian table for four this Friday at 7." → Dino gathers details.
2. Give an email when asked (Dino won't book without one, or without a party size).
3. Pick a recommended restaurant, then an offered time → Dino books and gives a
   confirmation id.
4. "Actually, can you cancel that?" → if the date is >24h away Dino cancels and
   notes it; if it's <24h away Dino gives you the restaurant's phone to call.
5. Come back later as the same email → Dino remembers you and your past booking.

**Guardrails to verify:** Dino never invents a restaurant/time/email, never books
without email + party size, and never claims a cancellation the backend refused.
