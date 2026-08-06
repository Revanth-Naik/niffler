# Niffler — Stock Prediction Platform

A free, platform-independent app that ingests stock pricing and trend data from
multiple sources, predicts market moves, and self-corrects by comparing
pre-market predictions against post-market actuals.

## Goal

1. Ingest daily stock price/trend data from free sources before market open.
2. Generate a prediction for each tracked ticker.
3. After market close, compare the prediction to the actual outcome.
4. Feed that error back into the model to improve future predictions.

## Current status

Working Streamlit prototype: a transparent heuristic prediction model, an
optional trained AI model that learns to correct the heuristic's own
mistakes, a Niffler-themed multi-page dashboard, a top-50 S&P 500 scanner,
a predicted-vs-actual accuracy tracker, and Dumbledore — an in-app guide
you can ask about any of it.

## Project structure

Trimmed to exactly what's needed to clone, run, and push — no dead code,
no superseded mockups, no generated artifacts.

```
niffler/
├── .github/workflows/      # scheduled predict/resolve/retrain jobs — see Automation below
├── src/
│   ├── ingestion/         # data source clients (yfinance, optional Alpha Vantage)
│   ├── prediction/
│   │   ├── model.py       # baseline momentum/RSI heuristic — predicted %, direction, confidence
│   │   ├── features.py    # technical + heuristic-derived features for the ML model
│   │   ├── ml_model.py    # loads/runs the trained ML correction model
│   │   ├── predictor.py   # unified entry point: ML if trained, else heuristic
│   │   ├── universe.py    # S&P 500 ticker list (live fetch + offline fallback)
│   │   └── synthetic.py   # deterministic fallback price series for offline/demo use
│   ├── tracking/
│   │   └── logger.py      # predictions_log.csv read/write + accuracy stats
│   ├── chatbot/
│   │   └── dumbledore.py  # in-app guide — free, local, template-based (no external AI API)
│   └── config.py          # ticker list + settings
├── scripts/
│   ├── fetch_daily.py     # pull raw OHLCV data
│   ├── train_model.py     # train/retrain the AI correction model
│   ├── run_predictions.py # run before market open — logs today's predictions
│   ├── record_actuals.py  # run after market close — fills in what actually happened
│   └── seed_demo_data.py  # backfill demo history so charts aren't empty on first run
├── web/
│   ├── streamlit_app.py   # home page: watchlist, hoard accuracy meter, nightly whisper
│   ├── pages/
│   │   ├── 1_Stock_lookup.py       # search any ticker, live prediction + its track record
│   │   ├── 2_Top_50_predictions.py # S&P 500 scan, ranked by predicted return
│   │   ├── 3_Accuracy_tracker.py   # hit rate over time, predicted vs actual, error charts
│   │   ├── 4_Model_insights.py     # AI vs heuristic holdout comparison, feature importances
│   │   └── 5_Ask_Dumbledore.py     # chat with the in-app guide
│   ├── theme.py            # shared Niffler palette, CSS, and themed chart helpers
│   └── data_helpers.py     # cached data access with offline fallback
├── data/                   # raw/, processed/, cache/ — gitignored except .gitkeep + the S&P 500 offline fallback list
├── models/                 # gitignored — created locally by train_model.py
├── .streamlit/config.toml  # dashboard theme
├── requirements.txt
└── .env.example
```

## Data source

[yfinance](https://github.com/ranaroussi/yfinance) — free, no API key required —
is the primary source for OHLCV data. Alpha Vantage (free tier, 25 requests/day)
is wired in as an optional second source for news/sentiment; set
`ALPHAVANTAGE_API_KEY` in `.env` to enable it.

If a live fetch fails (no network, rate limit, delisted ticker), the app
falls back to a deterministic synthetic price series so the UI stays usable
— pages will say when they're showing illustrative data instead of real data.

## The prediction model

`src/prediction/model.py` is a transparent baseline: it blends short-vs-long
moving average momentum with an RSI(14) mean-reversion signal into a
predicted next-session % move, direction, and confidence score. It's meant
as an honest, explainable starting point — easy to audit, easy to explain
in the UI.

## The AI correction model

`scripts/train_model.py` trains a gradient-boosted regressor
(scikit-learn) on technical features *plus the heuristic's own prediction*
as an input feature. That last part matters: it's what lets the AI model
learn to correct the heuristic's biases ("when the heuristic says +2% but
RSI is this extreme, it tends to overshoot") instead of learning from
scratch.

Once trained, `src/prediction/predictor.py` becomes the single entry point
the whole app calls — it uses the AI model automatically whenever one
exists, and falls back to the heuristic otherwise. Every prediction is
tagged with which one actually produced it (`source: "ml"` or
`"heuristic"`), shown as a small badge in the UI — nothing is silently
swapped out without you being able to see it.

**This is also the self-correction mechanism.** Re-running
`train_model.py` periodically — weekly is reasonable — retrains the model
on whatever history has accumulated since, including real predicted-vs-actual
outcomes once you've been running the daily loop for a while. It's not
continuous/autonomous learning (nothing retrains itself without you running
the script or scheduling it), but re-running it regularly is how the model
keeps adapting.

The **Model insights** page shows an honest comparison: the AI model's
error and hit rate on a holdout set it never trained on, side by side with
the heuristic's, plus which features it actually relies on. If the AI
model isn't beating the heuristic there, the page says so — it's not
built to oversell itself.

```bash
python scripts/train_model.py --synthetic       # quick offline test/demo, no network needed
python scripts/train_model.py                   # real data — scans S&P 500, trains on live history
python scripts/train_model.py --limit 30 --period 2y   # smaller/larger universe, longer history
```

## Dumbledore — the in-app guide

`src/chatbot/dumbledore.py` + the **Ask Dumbledore** page. Free and fully
local — no external AI API, no cost, no API key — so it isn't a real
language model. It's pattern matching over the question plus retrieval
from Niffler's own data (live predictions, the accuracy log, the trained
model's stats), filled into templates. It can explain why a ticker's
prediction looks the way it does, report accuracy/hit rate (overall or
per-ticker), explain terms like RSI or momentum, and describe how the AI
model was trained.

It deliberately does **not** answer "should I buy/sell" questions. This
app is deployed at a public URL, so a bot dispensing "sell now for profit"
calls would be unlicensed personalized financial advice reaching whoever
opens the link — not just you. It declines those with an explanation and
redirects to what it can honestly answer instead, keeping the same
"transparent, not overselling itself" character as the Model insights
page and the disclaimer below.

## Setup

```bash
cd niffler
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ALPHAVANTAGE_API_KEY if using it
```

## Run the dashboard

```bash
streamlit run web/streamlit_app.py
```

Opens at `http://localhost:8501` with five pages in the sidebar: stock
lookup, top 50 predictions, the accuracy tracker, model insights, and
Ask Dumbledore.

First time setup:

```bash
python scripts/seed_demo_data.py     # backfill demo predicted-vs-actual history
python scripts/train_model.py --synthetic   # train a first AI model to see the UI light up
```

Both use synthetic (not real market) data so the app is fully demoable
offline — replace with real history by running the daily loop and
retraining on live data once you have it.

## The daily loop

```bash
python scripts/run_predictions.py    # run before market open
python scripts/record_actuals.py     # run after market close
```

Running these daily builds up real predicted-vs-actual history in
`data/processed/predictions_log.csv`, which powers the accuracy tracker,
the home page's hoard meter, and — once you retrain on it — the AI model.

You don't have to run these by hand — see "Automation" below.

## Automation

Three scheduled GitHub Actions workflows in `.github/workflows/` run the
daily loop for you, on GitHub's servers, and commit the results back to the
repo so the live Streamlit Cloud app picks them up automatically:

| Workflow | Schedule (UTC, weekdays unless noted) | Runs |
|---|---|---|
| `predict.yml` | 13:00, Mon–Fri | `run_predictions.py` |
| `resolve.yml` | 21:30, Mon–Fri | `record_actuals.py` |
| `retrain.yml` | 12:00, Sunday | `train_model.py --limit 40 --period 1y` |

Each job installs dependencies, runs its script, and — only if the output
actually changed — commits and pushes using the repo's built-in
`GITHUB_TOKEN` (no personal access token needed).

**One-time setup:** on GitHub, go to your repo → Settings → Actions →
General → Workflow permissions, and select "Read and write permissions."
Without this the commit-and-push step fails with a 403, since the default
token is read-only.

You can also trigger any of the three manually from the Actions tab
("Run workflow" button) instead of waiting for the schedule. Times are
fixed UTC — chosen with a buffer so they land before/after market hours
across both EDT and EST.

## Roadmap

- [x] Project scaffold + git setup
- [x] Basic daily ingestion (yfinance)
- [x] Baseline prediction model (momentum + RSI heuristic)
- [x] Streamlit dashboard: watchlist, stock lookup, top 50 scanner, accuracy tracker
- [x] Predicted-vs-actual logging loop
- [x] AI correction model trained on technical features + heuristic output
- [x] Model insights page (honest AI vs heuristic comparison)
- [x] Scheduled daily runs + scheduled retraining (GitHub Actions)
- [x] Deploy somewhere reachable outside localhost (Streamlit Community Cloud)
- [x] AI chatbot (Dumbledore) for plain-English Q&A on predictions/tickers
- [ ] Optional Alpha Vantage news/sentiment ingestion
- [ ] Real data persistence for the live deployment (a small database instead of ephemeral CSV/joblib files)

## Disclaimer

Niffler is a personal/portfolio project. Predictions come from a simple
heuristic and an experimental ML model trained on limited history — they
are not investment advice.
