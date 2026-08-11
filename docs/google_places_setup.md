# Enabling live Google Places data

The search server runs offline against a fixture until you provide an API key.
To use real restaurant data, enable **Places API (New)** and drop a key into
`.env`.

## 1. Create / pick a Google Cloud project

1. Go to <https://console.cloud.google.com/>.
2. Top bar → project selector → **New Project** (or reuse an existing one).

## 2. Enable billing

Places API requires a billing account, even though Google grants a recurring
free usage allotment. You will not be charged within the free tier, but the
account must exist.

- Left menu → **Billing** → link a billing account to the project.

## 3. Enable the API

- Left menu → **APIs & Services → Library**.
- Search for **Places API (New)** (the one labeled "New" — not the legacy
  "Places API"). Click **Enable**.

## 4. Create an API key

- **APIs & Services → Credentials → Create credentials → API key**.
- Copy the key.
- **Restrict it** (recommended): edit the key → *API restrictions* → limit to
  **Places API (New)**. This caps blast radius if the key leaks — a small but
  real governance/hygiene point worth noting in the writeup.

## 5. Wire it into the project

```bash
cp .env.example .env
```

Then set the key in `.env`:

```
GOOGLE_PLACES_API_KEY=AIza...your-key...
```

`.env` is gitignored — the key never enters version control.

## 6. Verify

```bash
uv run python -c "from table_for_four.mcp_servers.search.server import search_restaurants; import json; print(json.dumps(search_restaurants('Italian near Midtown Manhattan', open_now=True), indent=2))"
```

`"source": "live"` in the output confirms real API calls (offline mode reports
`"source": "fixture"`).

## Cost control note

The server sends an explicit **field mask** (`X-Goog-FieldMask`) requesting only
the fields the concierge reasons over. On Places API (New) the field mask
determines the billing SKU, so requesting fewer fields = lower cost per call.
This is deliberate and worth a line in the responsible-AI section of the
writeup.
