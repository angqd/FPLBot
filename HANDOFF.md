# Handoff — fplbot

For the next agent (or human) picking this up. Read this before touching code. README.md is
the user-facing overview; this file is the state of play.

## Goal
Fully automated FPL manager: pull data weekly, project points, draft a squad, then every GW
either roll the free transfer or make up to two (no -4 hits by default), set XI + captain,
and submit — with minimal manual work.

## Status (as of 2026-11-26)
Phases 0-2 of the roadmap in `docs/research/report.md` are done. Everything runs in dry run.

| Area | State |
|---|---|
| Data ingest (`data/fpl_client.py`, `data/store.py`) | Working against the live API. gz snapshots in `data/raw/` (gitignored). |
| Team model (`model/team_strength.py`, `model/odds.py`) | Time-decayed Poisson fit on football-data.co.uk (goals 0.30, xG 0.35, odds-implied 0.35). Promoted-club prior (att -0.28, def -0.21). Optional Odds API blend, off without `ODDS_API_KEY`. |
| Player model (`model/xpts.py`) | Minutes model, shrinkage to position x price priors, DefCon Poisson tail, bonus blend. `blend_fpl_ep: 0` (backtest showed blending hurts). |
| Backtest (`eval/`, `fplbot backtest`) | Point-in-time replay of past GWs. 2026/27 GW2-11 next-GW MAE 1.94 vs FPL ep 2.33. `evaluate_configs()` for tuning. |
| MILP optimiser (`optimize/solver.py`) | Working. Draft + multi-GW planning, FT banking (cap 5), K candidate plans. Wildcard/Free Hit: unlimited first GW (`SquadState.unlimited_first_gw`); Free Hit revert not modelled. |
| Auth (`data/auth.py`) | Verified live: PingOne refresh + rotation. `--profile NAME` uses `.env.NAME`. |
| Executor (`execute/executor.py`) | Payload shapes match the web app. `dry_run: true`. Never submitted. |
| LLM judge (`judge/llm_judge.py`) | Written, untested. Off by default. Phase 4. |
| Scheduling | **Open question.** `.github/workflows/weekly.yml` is kept local only (gitignored, schedule commented out). Choose cron, GitHub Actions or manual before shadow mode. Only one place may hold the refresh token, because it rotates. |
| Tests | 61 pass: `python tests/make_fixture.py && pytest -q` |

`VIABILITY_LOG.md` (local only, gitignored) holds the dated evidence for every check.

## STEP 1 — Viability gate (do this before ANY other work)
The project isn't confirmed viable yet. Before improving the model, adding features or
refactoring, prove the APIs work and the payloads are right. **Getting transfer prices is
essential:** the optimiser's budget maths depends on them, and the transfer POST needs them.
If any check below fails and can't be fixed, stop and report back to the user. Don't build
around it.

Write each result into the "Viability results" table below as you go.

**1a. Public data.** `pip install -e ".[judge,dev]"` then `fplbot ingest` on a machine with internet.
   - Confirm `bootstrap-static/` and `fixtures/` return 200.
   - Confirm the fields the model uses exist this season: `now_cost`, `status`,
     `chance_of_playing_next_round`, `ep_next`, `expected_goals_per_90`,
     `expected_assists_per_90`, `defensive_contribution_per_90`, `saves_per_90`, `starts`,
     `bonus`, team `strength_*`, event `deadline_time` / `is_next`. Note anything missing.

**1b. Auth.** Log in at fantasy.premierleague.com. In DevTools → Network, find the POST to `.../as/token`.
   - Record the URL, `client_id`, and exact form fields. Fix `OIDCAuth.access_token()` if they differ.
   - Copy `refresh_token` from the `oidc.user:*` storage entry into `.env`.
   - Confirm the refresh works and that the rotated token is written back to `.env`.

**1c. Transfer prices (essential).** Call `my-team/{entry}/` with auth and confirm it returns:
   - `picks[].selling_price` and `picks[].purchase_price` for every owned player. Selling
     price ≠ `now_cost` once a player has risen; FPL keeps only half the profit, rounded down.
   - `transfers.bank`, `transfers.limit`, `transfers.made` (or the 2026/27 equivalents for
     free transfers).
   - Spot-check one risen player against the FPL UI's "selling price".
   - If selling prices aren't available from the API, **this is a blocker**. Report it.
     Don't fall back to `now_cost` silently: that breaks the budget constraint and can
     produce illegal transfers.

**1d. Payloads.** Make ONE transfer manually in the FPL UI. Capture the request bodies of the
   `/api/transfers/` POST and the lineup (`my-team`) POST.
   - Diff them against `build_transfer_payload` / `build_lineup_payload`: field names,
     price units (tenths), the `chip` field, and any new required fields.
   - Fix the builders to match. Then run `fplbot run --force` in dry-run and check that the
     payload it would send is identical in shape to the captured one.

**1e. Verdict.** Tell the user: viable / viable with fixes / blocked, with the evidence.
   Only after they confirm should you continue below or ever set `execute.dry_run: false`.

### Viability results
| Check | Result | Notes |
|---|---|---|
| 1a public endpoints + fields | **PARTIAL** (2026-11-25) | bootstrap + fixtures 200; 674 players, next GW12 (deadline 2026-11-27T18:30Z). All player fields present. **Team `strength_attack_*`/`strength_defence_*` are 0 for all 20 teams** (only `strength_overall_*` 2–5 populated) → `project()` raises ZeroDivisionError on live data. Needs a new team-strength source (e.g. from 110 finished fixture scores, or summed player xG / xGC). `game_config.scoring` matches DEFAULT_SCORING (GK goal 10, DefCon 2 for DEF/MID/FWD, 0 GK); DefCon thresholds not in API. `rules.max_extra_free_transfers=4` (cap 5 ✓), `transfers_sell_on_fee=0.5` (half-profit ✓). `--offline` expects uncompressed json, not data/raw snapshots. |
| 1b OIDC refresh + rotation | **PASS** (2026-11-25) | Live refresh via `scripts/viability_check.py`: access token issued, rotated refresh token written back to .env. Config from FPL's JS bundle: token URL `https://account.premierleague.com/as/token`, client_id `bfcbaf69-aade-4c1b-8f00-c1cb8a193030`, header `X-API-Authorization: Bearer`. Refresh token lives in localStorage `oidc.user:https://account.premierleague.com/as:<client_id>` (note: `fantasy.premierleague.com.` with a trailing dot is a different origin with no session). Team id comes from `me/` → `player.entry`. |
| 1c selling/purchase price, bank, FTs | **PASS** (2026-11-25) | `my-team/` returns `picks[].selling_price` + `purchase_price` for all 15; half-profit rounded-down rule holds for all 10 risen/fallen players (e.g. Groß 5.5→6.3 sells 5.9). `transfers` = `{cost, status, limit, made, bank, value}` (tenths); FTs = `limit - made` (`limit` null under WC/FH). Also returns `chips[].status_for_entry`. |
| 1d transfer + lineup payload match | PASS (static, 2026-11-25) | Web app sends `transfers/` `{chip, entry, event, transfers:[{element_in, element_out, purchase_price: now_cost, selling_price: pick.selling_price}]}` and `my-team/{entry}/` `{chip, picks:[{element, position, is_captain, is_vice_captain}]}`, identical to the builders. Web app FT formula is `limit===null ? 0 : max(limit-made, 0)`, so `cli._state_from_api`'s `max(1, ...)` floor is a bug. `scripts/viability_check.py` re-checks shapes against the live squad. |
| 1e verdict to user | **Viable with fixes** (2026-11-25) | APIs, auth, prices and payloads all work. Fixes before live runs: (1) team strengths are 0 → replace with Dixon-Coles (`model/xpts.py`); (2) `cli._state_from_api` FT floor `max(1, …)` and `limit or 1` → use web-app formula, handle `limit: null`; (3) `--offline` can't read gz snapshots. User confirmed going ahead with the plan in docs/research/report.md. |

## After the gate passes
1. ~~`fplbot draft` sanity check~~ done 2026-11-25.
2. ~~`fplbot plan` against the live team~~ done 2026-11-25.
3. Run `fplbot run --force` once in dry-run and review the decision row in `data/fplbot.sqlite`.
4. Phase 3 (HiGHS, solver settings, noise voting), then 4-6 GWs of shadow mode.
5. Only with the user's OK, set `execute.dry_run: false`.

## Design decisions (don't undo without reason)
- **Official FPL API only, no scraping.** bootstrap-static already has prices, news, ep_next,
  xG/xA/DefCon per 90, team strengths. Add Understat/FBref only behind `project()`.
- **Deterministic solver decides; LLM only picks among solver plans by index.** The LLM must never
  generate moves — it can't reliably respect budget/shape/club constraints. Judge failure → plan 0.
- **Roll vs. transfer is emergent from the FT state variable**, not a rule. Tune via `ft_value`,
  `min_gain_per_transfer`, `decay`, `horizon` in config.yaml rather than adding heuristics.
- **`pulp>=2.8,<3` is pinned on purpose.** PuLP 4.x removed `LpVariable.dicts` / `LpStatus`;
  porting is possible but not done.
- `fplbot run` is idempotent: snapshots every call, decides only inside the pre-deadline window,
  at most once per GW (`decisions` table).

## Known gaps / backlog (rough priority)
1. Tune with the backtest: `ridge_sigma` (0.2 also shrinks top clubs: MCI att 0.25 vs 0.36 unshrunk),
   shrinkage `k`, bonus blend. Model under-projects by ~0.7 pts on average, mostly premiums.
2. Phase 3: HiGHS solver (drops the PuLP pin), FT value by state, bench weights, bank value, noise voting.
3. Chips: Free Hit revert, Bench Boost, Triple Captain (Wildcard/Free Hit unlimited transfers done).
4. Price-change timing. Snapshots store `transfers_in_event/out_event` for it.
5. Phase 4 LLM: news extractor with bounded adjustments, judge as tie-breaker only.
6. Selling price for players bought *within* the horizon uses current cost (ignores half-profit). Fine for planning.
7. Hosting / scheduling (open question, see Status). If GitHub Actions: secrets plus a `GH_PAT` with
   Secrets: write to persist the rotated token; the snapshot-commit step is moot now `data/raw/` is ignored.
8. Mirror account: user to create `.env.bot`, then `python scripts/viability_check.py --profile bot`.
9. Notifications (Telegram/email) after each decision.
10. Product idea: natural-language playstyle rules ("never double up on defence"). The LLM compiles
    them into a small typed rule schema that becomes solver constraints or penalties; the solver still
    picks players. Report the points each rule costs. Fits Phase 4 next to the news extractor.

## Gotchas
- Cloud sandbox couldn't reach fantasy.premierleague.com — use `--offline tests/fixtures` there.
- On synthetic data the bot rolls for several GWs: player quality is static and the draft is
  already optimal for the window. Don't "fix" that; judge rolling behaviour on real data.
- FPL requires the substitute GK in bench slot 12; `solve()` orders bench GK-first for that reason.
- Draft mode prints `FT=0` for the first GW — cosmetic.
- vaastav's `xP` is recomputed after each GW (leaky). The backtest rebuilds FPL's ep from form instead.
- New `project()` columns must not start with `xp_`: the solver reads every `xp_*` column as a GW.

## Map
```
fplbot/cli.py               entry points: ingest | draft | plan | run | backtest
fplbot/config.py            config.yaml + .env loading, squad-shape constants
fplbot/data/fpl_client.py   HTTP client (rate-limited, retries, auth header)
fplbot/data/auth.py         OIDC refresh → access token, persists rotated token to .env
fplbot/data/store.py        gz raw snapshots + SQLite (player_snapshots, decisions)
fplbot/model/team_strength.py  fit_team_model() -> TeamModel (Poisson ratings, promoted prior)
fplbot/model/odds.py        odds -> implied expected goals
fplbot/model/odds_api.py    optional Odds API blend for upcoming fixtures
fplbot/model/promoted_prior.py  estimates the promoted-club prior from history
fplbot/model/xpts.py        project(bootstrap, fixtures, gw, horizon, cfg, team_model) -> xp_<gw>, xmins, p_start
fplbot/eval/                backtest: history.py (data), pit.py (point-in-time), backtest.py (metrics)
fplbot/optimize/solver.py   SquadState, solve(), candidate_plans()
fplbot/judge/llm_judge.py   judge(plans, news, model) -> (index, reason)
fplbot/execute/executor.py  payload builders + execute(dry_run)
scripts/viability_check.py  read-only live checks of login, prices and payloads
tests/make_fixture.py       synthetic bootstrap/fixtures/results with real field names
```
