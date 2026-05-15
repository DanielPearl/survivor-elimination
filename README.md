# Survivor Elimination forecast

Per-contestant elimination model that scores the active Kalshi
`KXSURVIVOR*` markets every poll cycle. The dashboard surfaces a
contestant-level watchlist with model probability, Kalshi market
probability, edge / EV per $1 contract, confidence score, suggested
side (YES / NO), and validators (active market, valid price,
season/episode parsing, contestant parsing, freshness, duplicate
contestants, missing model output).

## What this is

- A real supervised model. One row per (season, episode, contestant)
  for every modern-era Survivor season (41–49), labelled
  `eliminated_this_episode ∈ {0,1}`. Two models train on the panel —
  an L2-regularised logistic regression (interpretable) and a
  calibrated `HistGradientBoostingClassifier` (predictive). The
  lower-Brier model is used in production. The training panel
  lives at `data/raw/historical_boots.csv` (built-in seed in
  `src/survivor/data/seed_panel.py` if the CSV is missing).
- A current-season state file at `data/raw/current_state.json` you
  edit each week as the new episode airs. Per-contestant signal
  (tribe, visibility, idol, prior votes, alliance, etc.) feeds the
  same feature pipeline the trainer uses, so the live scorer has
  no train/serve skew.
- A Reddit ingestion module (`src/survivor/reddit/ingest.py`) that
  pulls mentions / boot-prediction sentences / sentiment / target
  share from `r/survivor` etc. Drops in five extra feature columns
  the model picks up. If Reddit creds aren't set the columns
  default to zero — model is still scored, just without that
  signal.
- A Kalshi scorer (`src/survivor/kalshi/markets.py`) that finds
  every open `KXSURVIVOR*` series via the SDK (no season number
  hard-coded), parses the contestant name + episode out of each
  market title, and joins it to the state to produce the live
  watchlist.

## Layout

```
config/config.yaml                  Paths, training params, validators
data/raw/historical_boots.csv       Optional — overrides the built-in seed
data/raw/current_state.json         Edit weekly: per-contestant pre-tribal state
data/outputs/watchlist.json         Live exporter target (dashboard reads this)
data/processed/artifacts/           model.joblib + metrics.json + coefficients.json
                                    + feature_importance.csv + holdout_predictions.csv
src/survivor/
  data/historical.py                Training-panel loader
  data/seed_panel.py                Hand-curated seed for S41–S49
  data/current_state.py             Live state file reader
  features/build_features.py        Feature vector builder (train + serve)
  models/train.py                   LR + calibrated GBT trainer
  models/predict.py                 Inference wrapper (cached load)
  reddit/ingest.py                  PRAW-based Reddit signal pull
  kalshi/markets.py                 Kalshi market discovery + normaliser
  trading/ev.py                     EV-per-contract calc
  trading/validators.py             Buy-gate validators
  dashboard/export_watchlist.py     End-to-end exporter (the cron entrypoint)
  utils/                            config + logging
scripts/
  run_daily_train.py                Re-train + rebuild watchlist
  run_live_monitor.py               Live-poll loop (systemd target)
deploy/
  deploy.sh                         Droplet bootstrap
  survivor-elimination-monitor.service
  survivor-elimination-train.service + .timer
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # editable-installs ../kalshi_sdk

# Train + offline watchlist (no Kalshi creds required).
python scripts/run_daily_train.py --offline

# Verify the dashboard renders the survivor tab.
cd ../"Trading dashboard"
python dashboard.py --config config/dashboard.local.yaml
# http://localhost:8080/?bot=survivor&tab=watchlist
```

## Production

The droplet runs two systemd units:

- `survivor-elimination-monitor.service` — long-running loop, polls
  Kalshi every 5 min, rewrites `watchlist.json`. The dashboard
  reads it on every page load.
- `survivor-elimination-train.timer` — fires
  `survivor-elimination-train.service` once a day at 06:30 UTC.
  Retrains on the latest `historical_boots.csv` (if you've added
  the latest finished season) + rebuilds the watchlist with the
  new model.

See `deploy/deploy.sh` for the full bootstrap.

## Updating each week

After an episode airs:

1. Add the new boot to `data/raw/historical_boots.csv` (or to
   `seed_panel.py` if you're using the built-in seed) — one
   `(season, episode, contestant)` row with `eliminated=1`.
2. Edit `data/raw/current_state.json`:
   - Remove the eliminated contestant.
   - Bump `current_episode` by 1.
   - Update per-contestant `visibility_score`, `times_targeted`,
     `negative_edit_score`, `prior_votes_against` based on the
     episode's content. Defaults are sane if you skip this — the
     model just won't differentiate contestants beyond what
     Reddit signal + market price give it.

The live monitor picks up the changes on its next tick (within 5
minutes); no restart needed.
