# fplbot

An automated Fantasy Premier League manager. Each gameweek it pulls FPL data, projects
points for every player, works out the best transfers, picks the XI and captain, and
submits them. It runs in dry-run mode until you turn submitting on.

## How it works

```
 ingest  ──►  project  ──►  optimise  ──►  judge (optional)  ──►  execute
 FPL API      points per    squad, XI,      LLM picks one of      transfers
 snapshots    player/GW     captain, roll   the solver's plans    + lineup
```

1. **Ingest.** Reads the official FPL API (prices, fixtures, injuries, xG, xA, DefCon) and
   saves a snapshot each run.
2. **Project.** Estimates expected points per player per gameweek. Team attack and defence
   ratings come from a Poisson model fitted to this season's results and xG.
3. **Optimise.** An integer program picks transfers over the next few gameweeks within the
   budget, squad shape and 3-per-club rules. It decides between rolling a free transfer and
   using it because banked transfers are part of the model.
4. **Judge.** Optional. An LLM can choose between the solver's top plans using team news.
   It can't invent moves. If it fails, the bot takes the solver's best plan.
5. **Execute.** Sends the transfers and lineup to FPL. Dry run by default.

## Quick start

```bash
pip install -e ".[judge,dev]"
cp .env.example .env     # add your FPL team id and refresh token
fplbot ingest            # take a data snapshot
fplbot draft             # best £100m squad for the next gameweek
fplbot plan              # transfer plan for your current team
fplbot run --force       # the full weekly job (dry run)
```

To get your refresh token, log in at https://fantasy.premierleague.com/ and open DevTools.
Copy `refresh_token` from the localStorage entry that starts with `oidc.user:`. The bot
rotates the token and writes the new one back to `.env` on each use.

## Useful options

| Option | What it does |
|---|---|
| `--offline latest` | Run on the newest saved snapshot, with no network calls |
| `--offline tests/fixtures` | Run on synthetic data (`python tests/make_fixture.py` first) |
| `fplbot backtest` | Score past gameweeks: our projections vs FPL's expected points |
| `--profile bot` | Use `.env.bot` and a separate decision history, for a second team |

Settings live in `config.yaml`: planning horizon, free-transfer value, hits allowed, and
`execute.dry_run`.

## Scheduling

`fplbot run` is safe to call often. It always takes a snapshot, but it only makes a decision
in the hours before a deadline, and at most once per gameweek. Run it from cron or the
included GitHub Actions workflow.

## Status

- Working against the live API: login, selling prices, payloads, draft and plan (dry run).
- Projections (minutes model, shrinkage, DefCon, bonus, betting odds) beat FPL's own
  expected points in a backtest of past gameweeks.
- Not yet: chips, price-change timing, live submission.

See `HANDOFF.md` for the project's current state, and
`docs/research/report.md` for the research and roadmap.

## Tests

```bash
python tests/make_fixture.py && pytest -q
```
