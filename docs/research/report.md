# fplbot: the case for a deterministic FPL manager with an LLM on a short leash

As of 25 Nov 2026. Built from six research passes over about 120 sources, plus live checks against the FPL API.

## The pitch

We can build a bot that manages an FPL team every week with no manual work, and it doesn't need machine learning to be good. The best public tools all share one shape. A team-strength model feeds player projections, those feed a mixed-integer solver, and humans or news checks adjust minutes. fplbot already has that shape. What it lacks is the tuning those tools have, plus one fix that is urgent because the bot crashes on live data today.

The LLM's job is narrow. It reads injury news and nudges minutes estimates within hard limits. It never picks players. LLMs asked to solve constrained problems directly get them wrong most of the time, and the research on that is clear.

Two things still stand between us and a working bot. We haven't confirmed that FPL gives us your selling prices, and we haven't checked the format of the transfer request. Both need your browser login and take about 20 minutes.

## Where we stand

> **Update, 26 Nov 2026.** Phases 0 to 2 are done. The viability gate passed (login, selling
> prices and payloads all work), the bugs below are fixed, and the bot runs end to end on live
> data in dry run. In a backtest of 2026/27 GW2-11 our next-gameweek error is 1.94 points
> against 2.33 for FPL's own `ep_next`. Phase 3 is next. The tables below are the original
> pitch, kept for the record.

### Headline numbers

| Number | What it means |
| --- | --- |
| 674 | Players in the live API this season |
| 109 | Data fields per player. The model uses 16 |
| 0 | Value FPL now returns for every team's attack and defence strength. This crashes the model |
| 110 | Finished matches this season with goals, xG and odds on football-data.co.uk, which is enough to replace those ratings |
| 4 / 4 | Offline tests passing |
| 27 Nov, 18:30 UTC | Next deadline (GW12) |

### Viability gate

| Check | Status | Notes |
| --- | --- | --- |
| Public data (1a) | Partial pass | All player fields are present. Team strengths are zeroed, so `project()` raises ZeroDivisionError |
| Login (1b) | Waiting on you | FPL's login server confirms the token URL and that refresh tokens are supported |
| Selling prices, bank, free transfers (1c) | Waiting on you | Essential. If FPL doesn't return selling prices, we stop |
| Transfer and lineup request format (1d) | Waiting on you | Needs one real transfer captured in DevTools |

### Bugs found in the code

- `model/xpts.py` divides by the team strength average, which is now 0.
- `cli.py` never lets free transfers drop below 1, so after you use your transfer the bot thinks it still has one.
- `fplbot --offline` can't read the gzipped snapshots that `fplbot ingest` writes.
- The bonus term predates the 2026/27 BPS changes, so it overrates defenders with high CBI counts.

## The plan, layer by layer

Each layer below says what fplbot does now, what it should do, and the evidence.

### 1. Team strength (urgent)

**Now.** Reads FPL's team strength fields, which are all 0.

**Change.** Fit a Dixon-Coles model on goals and xG from football-data.co.uk, weighting recent matches more (half-life about a year). Pull ratings toward last season's to stop early-season noise. For next gameweek only, blend in betting odds at 70 to 80 percent weight.

**Why.** Bookmaker odds are the most accurate public forecast of match outcomes. Hvattum and Arntzen found Elo ratings worse than odds. Solio, one of the newer FPL tools, builds its projections on sharp odds. The Dixon-Coles fit covers gameweeks the odds don't reach yet.

**Effort.** Half a day to a day for the model. One more day for the odds blend.

### 2. Player projections

**Now.** A hand-tuned formula on xG and xA per 90, blended 35 percent with FPL's own `ep_next`.

**Change.** Four upgrades, in order of payoff.

1. A real minutes model. Predict the chance of starting from recent starts, rest days and squad depth, gated by FPL's injury percentage. Minutes are the biggest source of error in every tool we looked at.
2. Shrink per-90 rates toward the position average and weight recent games more. A player with 200 minutes and two lucky goals stops looking like Haaland.
3. Treat defensive contribution as a probability of hitting the threshold (10 for defenders, 12 for midfielders and forwards). A quick check against 2025/26 data was close. Senesi predicted 70.8 percent, actual 70.3. Collins predicted 49.9, actual 50.
4. Recalibrate bonus for the 2026/27 BPS rules.

**Why not ML.** Weekly FPL points are mostly noise. The gap between good models is small.

| Model | Error | Note |
| --- | --- | --- |
| OpenFPL (open-source ML ensemble) | RMSE 1.52 | 3 to 4 point players, one GW ahead |
| FPL Review (commercial) | RMSE 1.59 | Same test. FPL Review was better at spotting who won't play |
| Onside | MAE 2.53 | Players with 60+ minutes, GW1 to 11 this season. Vendor's own numbers |
| FPL ep_next | MAE 2.71 | Same test |
| Last-4 form | MAE 2.97 | Same test |

ML buys about 5 percent at best and needs a data pipeline we don't have. We can add a small LightGBM correction later if the backtest shows room.

**Effort.** About a week, including a harness that freezes each week's projections and scores them against what happened.

### 3. The optimiser

**Now.** A PuLP and CBC model over 5 gameweeks. Transfers capped at 2 a week, a flat bench weight, and free transfers only valued at the end of the horizon.

**Change.** Copy the settings that FPL-Optimization-Tools (the strongest open solver) and FPL Review have converged on.

| Setting | fplbot now | Recommended | Source |
| --- | --- | --- | --- |
| Solver | CBC | HiGHS | Both top tools use it. PuLP 4 dropped the CBC call we use |
| Horizon | 5 GWs | 6 to 8, no transfers in the last 1 or 2 | FOT, FPL Review |
| Free transfer value | 1.0 at horizon end only | 2.0, 1.6, 1.3, 1.1 for the 2nd to 5th banked FT, counted every week | FOT |
| Bench weight | 0.10 flat | 0.03 sub keeper, 0.21 first sub, 0.06 second, near 0 third | FOT |
| Money in the bank | Ignored | 0.08 points per million | FOT |
| Transfers per week | Max 2 | Up to the free transfers you have, still no hits | With a cap of 2, a bank of 5 can never be used |
| Candidate plans | Differ in this week's buys | Differ in this week's buys and sells | FOT |

**Uncertainty.** Nobody serious uses formal robust optimisation for FPL. They add noise to the projections, re-solve 20 to 100 times and count which moves keep winning. FPL Review scores the top three moves 3, 2 and 1 in each run. A move that wins most runs is solid. A move that only wins the clean run is a coin flip. We should do the same and hand the vote counts to the LLM.

**Effort.** Three to four days for the settings and the voting. Chips (wildcard, free hit, bench boost, triple captain) are a bigger job and come later.

### 4. The LLM

**Now.** Picks one of K solver plans by index, sees only plan summaries and news strings.

**Change.** Give it a better job. It reads injury news and press conference quotes and returns, for each affected player, a start probability and a minutes multiplier, plus the exact quote it relied on. Code enforces the limits:

- Clamp every number.
- A low-confidence call can move FPL's injury percentage by 25 points at most.
- Drop any adjustment whose quote isn't in the source text word for word.
- Adjust at most 15 players a week.

The solver then re-runs with the new minutes. The judge role stays, but only as a tie-breaker between plans within about 1.5 expected points of the best. It sees the plans twice in shuffled order. If its two answers disagree, plan 0 wins.

**Why.** On ConstraintBench, frontier LLMs produced a valid answer only 44.5 to 65 percent of the time, and the best answer only 8.5 to 30.5 percent. An LLM-based FPL tool (ChatFPL) recommended Solanke after he'd changed clubs. Reading news and turning it into a number is the one part of this job LLMs are good at.

**Before we trust it.** Run it in shadow mode for 4 to 6 gameweeks. Keep it only if its start predictions beat FPL's injury percentage and its picks score at least as well as plan 0.

**Effort.** One to two weeks.

## Cheap edges worth taking

| Edge | Estimated gain | Effort | Note |
| --- | --- | --- | --- |
| Better free transfer and hit rules | 5 to 15 pts a season | Medium | Our estimate. No study has measured it |
| Make price-rising transfers before the overnight update | 2 to 8 pts a season | Small | Our estimate. FPL's API now publishes its own rise and fall forecast |
| Value team value | About 20 pts per £1m over a season | Small | Fantasy Football Scout, 2019. May be dated |
| Triple captain in the best double gameweek | Not measured | Small | Standard advice |
| Ownership-aware picks | Near zero for points | Small | Only matters when chasing a rank target |

The price forecast is the surprise. FPL's API now returns how close each player is to a price change, with a likelihood score from minus 5 to plus 5. On the day we checked, Hill was at 110 percent and Andrey Santos at minus 103 percent. We don't need to build a predictor, only to read theirs.

## Five ideas borrowed from other games

We surveyed 14 methods from baseball, NBA and NFL daily fantasy, F1 fantasy, cricket and betting. These five transfer best, all deterministic.

1. **Shrinkage and recency weighting, from baseball's Marcel system.** Weight recent seasons more and pull toward the average. Marcel gets most of the way to far more complex projection systems. Small effort, no new data.
2. **Separate minutes from per-minute output, from NBA daily fantasy.** When a starter is out, give his share of team xG to his replacement. Small to medium effort.
3. **Dixon-Coles team ratings, from betting.** This is the fix for the zeroed strengths. Medium effort.
4. **Force variety into the candidate plans, from Hunter, Vielma and Zaman's daily fantasy work.** Each plan must differ by one or two transfers, so the judge isn't choosing between near-copies. Combine with the noise voting. Small effort.
5. **Value what's left at the end of the horizon and add chip variables, from FPL research and F1 fantasy solvers.** This stops the bot burning transfers just before the planning window ends. Small to medium effort.

We passed on deep reinforcement learning. Dream11's cricket results came from its own staff with no significance tests, and it needs far more data than we have. Matthews and colleagues did reach the top 1 percent of 2.5 million FPL players with a Bayesian model in 2012, but their approach is heavy to build and our MILP gets most of the benefit.

## Roadmap

| Phase | What ships | Time | Gate to move on | Status |
| --- | --- | --- | --- | --- |
| 0. Finish the viability gate | Your login captured, selling prices checked, transfer format matched | 20 minutes of your time | Selling prices exist and payloads match. If not, we stop | Done 25 Nov |
| 1. Unblock live runs | Dixon-Coles team model, free transfer floor fix, offline snapshot fix | 1 to 2 days | `fplbot draft` and `plan` run cleanly on live data | Done 25 Nov |
| 2. Better projections | Minutes model, shrinkage, DefCon probability, bonus fix, odds blend, scoring harness | About 1 week | Our error beats `ep_next` on frozen projections | Done 26 Nov (1.94 vs 2.33) |
| 3. Optimiser upgrades | HiGHS, new settings, noise voting | 3 to 4 days | 4 to 6 gameweeks in dry-run shadow mode, then your OK to submit for real | Next |
| 4. LLM and extras | News extractor, tie-break judge, price timing, triple captain, then chips | 1 to 2 weeks | Shadow-mode results beat plan 0 | Not started |

Nothing gets submitted to FPL until phase 3 passes and you say so.

## Risks and open questions

- **Selling prices.** If `my-team` doesn't return them, the budget maths breaks and the project stops. This is the biggest unknown.
- **Login tokens.** FPL rotates the refresh token on every use. Running the bot on your laptop and in GitHub Actions at the same time will kill one of them.
- **FPL changes things mid-season.** The team strength fields went to zero without notice. Anything we read from the API can do the same, so the model must fail loudly, not quietly.
- **Estimates, not measurements.** The season gains for transfers and price timing are our estimates. The shrinkage settings and blend weights need tuning by backtest.
- **Scraping.** Fantasy Football Scout's terms ban automated extraction. The LLM should read FPL's own news fields and official club pages only.
- **Odds feed.** football-data.co.uk's upcoming-fixtures file was empty when we checked. The Odds API free tier (500 credits a month) is the fallback, at about 20 to 40 credits a gameweek.

## Sources

### Solvers
- FPL-Optimization-Tools solver code: https://raw.githubusercontent.com/sertalpbilal/FPL-Optimization-Tools/main/dev/solver.py
- FPL Review solver settings: https://docs.fplreview.com/the-model/solvers/settings/
- FPL Review sensitivity analysis: https://docs.fplreview.com/the-model/solvers/sensitivity-analysis/
- AIrsenal: https://github.com/alan-turing-institute/airsenal
- PuLP changelog: https://raw.githubusercontent.com/coin-or/pulp/master/HISTORY

### Papers
- Matthews et al. 2012, Bayesian FPL agent: https://ojs.aaai.org/index.php/AAAI/article/view/8259
- Ramezani and Dinh 2025, MILP for FPL: https://arxiv.org/html/2505.02170v3
- OpenFPL: https://arxiv.org/html/2508.09992v1
- Ridge vs LightGBM vs CNN for FPL: https://arxiv.org/html/2405.02412v1
- Hunter, Vielma and Zaman, daily fantasy integer programming: https://arxiv.org/abs/1604.01455v3
- ConstraintBench: https://arxiv.org/html/2602.22465v1
- Hvattum and Arntzen, Elo vs odds: https://ideas.repec.org/a/eee/intfor/v26yi3p460-470.html

### Projections and rules
- FPL Review xMins: https://docs.fplreview.com/the-model/projections/xmins
- Onside accuracy log: https://onsidearena.com/vs/fantasy-football-hub
- Marcel projections: https://blogs.fangraphs.com/index.php/behind-the-monkey/
- DefCon rules: https://www.premierleague.com/en/news/4361991
- 2026/27 rule changes: https://www.fantasyfootballscout.co.uk/2026/07/20/fpl-2026-27-5-rule-changes-new-features-announced
- vaastav FPL history: https://github.com/vaastav/Fantasy-Premier-League

### Betting and team models
- football-data.co.uk 2026/27 results: https://www.football-data.co.uk/mmz4281/2627/E0.csv
- penaltyblog: https://penaltyblog.readthedocs.io/en/latest/models/example.html
- The Odds API: https://the-odds-api.com/

### Decision edges and LLMs
- Value of team value: https://www.fantasyfootballscout.co.uk/2019/09/12/the-value-of-team-value/
- FPL price predictor explained: https://onefpl.com/blog/fpl-price-change-predictor-explained-2026-27
- ChatFPL and stale knowledge: https://thenextweb.com/news/ai-is-killing-fantasy-football-fpl
- Claude structured outputs: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
