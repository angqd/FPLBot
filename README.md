# fplbot — automated Fantasy Premier League manager

```
 ingest ──► project (xPts) ──► optimise (MILP) ──► [LLM judge] ──► execute
 FPL API     per player/GW      squad, XI, C,       picks 1 of K     POST transfers
 snapshots   deterministic      roll-vs-transfer    solver plans     + lineup
```

## Why this shape

**Data: the official FPL API is enough. No scraping needed.** `bootstrap-static/` already
carries prices, ownership, status/news, `ep_next`, xG/90, xA/90, defensive contributions/90,
saves/90, bonus, and team strength ratings. `fixtures/` has every fixture by GW. FotMob or
Firecrawl would only add noise and maintenance. Add Understat/FBref later only if you want
richer shot data (`soccerdata` package), behind the same `project()` interface.

**Decision: deterministic core, LLM on top.** The MILP is the right tool for the hard part:
budget, 2/5/5/3, max 3 per club, XI formation, and free-transfer banking (up to 5,
2026/27 rules). An LLM can't reliably respect those constraints. What an LLM *is* good at
is soft context like press-conference news. So the solver emits K distinct legal plans and
the LLM can only choose an index. If the LLM fails, the bot falls back to the solver's
optimum.

**Roll vs. transfer is solved, not hand-coded.** Free transfers are a state variable in the
model, so "roll this week, make two next week" happens exactly when it scores higher over
the horizon. `max_transfers_per_gw: 2` and `allow_hits: false` encode the policy.

## Setup

```bash
pip install -e ".[judge,dev]"
cp .env.example .env        # fill FPL_ENTRY_ID + OIDC values (see comments in the file)
fplbot ingest               # first snapshot
fplbot draft                # best £100m squad for the next GW (enter it in the FPL UI once)
fplbot plan                 # transfers/roll plan for your live team
fplbot run --force          # the full weekly job (dry-run until you flip execute.dry_run)
```

Offline / CI: `python tests/make_fixture.py && fplbot draft --offline tests/fixtures`, `pytest`.
`--offline` also takes an ingest snapshot (`data/raw/<ts>/`, gzipped) or `latest` (newest snapshot).

Second account (profiles): `cp .env.example .env.bot`, fill it for the other account, then
`fplbot --profile bot plan` (or `FPLBOT_PROFILE=bot`). The profile loads only `.env.bot`, writes
its rotated refresh token back there, and keeps its decisions DB in `data/bot/` so idempotency
records don't collide with the main account; public-data snapshots in `data/raw/` stay shared.
`FPLBOT_ENV_FILE=/path/to/envfile` selects an env file explicitly. `.env.*` files are gitignored.

## Scheduling
`fplbot run` is idempotent. Every invocation snapshots data (this builds price and form
history). It only decides inside `hours_before_deadline` of the next deadline, and at most
once per GW. Run it a few times a day from the included GitHub Actions workflow, or from cron
on any box.

## Known gaps / next steps
1. **Auth bootstrap is manual once.** FPL moved to PingOne OIDC. Copy the refresh token,
   token URL and client_id from DevTools. Verify the token-request fields and the
   `/api/transfers/` payload against one manual transfer before setting `dry_run: false`.
2. **Initial squad creation** isn't exposed in the API. Enter the drafted squad in the UI once.
3. **Chips** (wildcard, free hit, bench boost, triple captain) aren't modelled yet. A wildcard
   is just `solve()` with unlimited free moves in GW1, which is how draft mode works.
4. **Price changes** aren't predicted. Snapshots accumulate `transfers_in_event` /
   `transfers_out_event`, which is what you'd need for a price-change model.
5. **Backtesting:** raw snapshots are kept gzipped in `data/raw/`. Replay them with
   `--offline`, or train a learned xPts model on vaastav/Fantasy-Premier-League history.
