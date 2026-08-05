# Nifler — Stock Prediction Platform

A free, platform-independent app that ingests stock pricing and trend data from
multiple sources, predicts market moves, and self-corrects by comparing
pre-market predictions against post-market actuals.

## Goal

1. Ingest daily stock price/trend data from free sources before market open.
2. Generate a prediction for each tracked ticker.
3. After market close, compare the prediction to the actual outcome.
4. Feed that error back into the model to improve future predictions.

## Current status

Phase 1: data ingestion pipeline (local prototype).

## Project structure

```
nifler/
├── src/
│   ├── ingestion/       # data source clients + fetch orchestration
│   ├── prediction/      # prediction model (future phase)
│   └── config.py        # ticker list + settings
├── data/
│   ├── raw/             # raw daily pulls (CSV)
│   └── processed/       # cleaned/merged data (future)
├── scripts/
│   └── fetch_daily.py   # CLI entrypoint to run ingestion
├── tests/
├── requirements.txt
└── .env.example
```

## Data source

Phase 1 uses [yfinance](https://github.com/ranaroussi/yfinance) — free, no API
key required, pulls historical + recent OHLCV data and basic company info
from Yahoo Finance.

Alpha Vantage (free tier, 25 requests/day) is wired in as an optional second
source for news/sentiment and fundamentals — set `ALPHAVANTAGE_API_KEY` in
`.env` to enable it.

## Setup

```bash
cd nifler
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ALPHAVANTAGE_API_KEY if using it
```

## Run the ingestion pipeline

```bash
python scripts/fetch_daily.py --tickers AAPL,MSFT,GOOGL
```

This writes one CSV per run to `data/raw/`.

## Roadmap

- [x] Project scaffold + git setup
- [x] Basic daily ingestion (yfinance)
- [ ] Optional Alpha Vantage news/sentiment ingestion
- [ ] Pre-market prediction model (baseline)
- [ ] Post-market actual vs. predicted comparison + logging
- [ ] Feedback loop to retrain/correct the model
- [ ] Web page/dashboard
- [ ] Scheduled daily runs
