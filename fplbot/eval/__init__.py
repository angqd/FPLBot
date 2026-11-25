"""Backtest harness: does `project()` beat FPL's own ep_next on frozen, point-in-time data?

  history.py   per-player per-GW history (vaastav/Fantasy-Premier-League + FPL event/live top-up)
               and the frozen ep baseline for each GW
  pit.py       point-in-time bootstrap/fixtures/results for a target GW (only data from GWs < g)
  backtest.py  run models + baselines, metrics, tuning hook (evaluate_configs)
"""
