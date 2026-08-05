# Getting Niffler running — step by step

## 0. What you need

- Python 3.10 or newer
- Terminal access
- A GitHub account, if you want to push this (optional but assumed below)

Check your Python version first:

```bash
python3 --version
```

If it's below 3.10, install a newer one from python.org, or via Homebrew: `brew install python3`.

## 1. Find the project folder

The project is at:

```
/Users/revanthnaik/Library/Application Support/Claude/local-agent-mode-sessions/958f9781-d1e6-4803-841f-a22795c9b3fa/355b2323-21d0-4c76-a53d-2c72edd87124/local_bd44bc36-3c7d-4f3e-bee3-761149b90539/outputs/niffler
```

Open Terminal and go there:

```bash
cd "/Users/revanthnaik/Library/Application Support/Claude/local-agent-mode-sessions/958f9781-d1e6-4803-841f-a22795c9b3fa/355b2323-21d0-4c76-a53d-2c72edd87124/local_bd44bc36-3c7d-4f3e-bee3-761149b90539/outputs/niffler"
```

If you'd rather work somewhere easier to find, drag the `niffler` folder to somewhere like `~/Projects/` in Finder first, then `cd` to that new location instead — everything below works the same either way.

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. That means it's active. You'll need to run the `source` line again each time you open a new terminal window to work on this project.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Installs Streamlit, yfinance, scikit-learn, plotly, pandas, and everything else. Takes a minute or two.

## 4. (Optional) Alpha Vantage API key

Only needed if you want the optional news/sentiment data source. Skip this entirely if not — the app works fully without it.

```bash
cp .env.example .env
```

Then open `.env` in any text editor and paste in a free key from [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key).

## 5. Seed some demo data

This makes the app show something useful the first time you open it, using synthetic (not real) data so there's nothing to wait on:

```bash
python scripts/seed_demo_data.py
python scripts/train_model.py --synthetic
```

First command backfills ~60 days of demo predicted-vs-actual history per watchlist ticker. Second trains a first version of the AI correction model on synthetic data. Both print a short summary when done.

## 6. Run the app

```bash
streamlit run web/streamlit_app.py
```

Opens your browser to `http://localhost:8501` automatically. If it doesn't, open that URL yourself. Four pages in the left sidebar: home, stock lookup, top 50 predictions, accuracy tracker, and model insights.

To stop it: go back to Terminal and press `Ctrl+C`.

## 7. Push to GitHub

a. On github.com, create a new **empty** repository (don't check "add a README" — this project already has one).

b. Back in Terminal, still inside the `niffler` folder:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

Swap in your actual repo URL (GitHub shows it right after you create the repo). If it asks for a password and rejects it, GitHub now requires a **personal access token** instead of your account password for this — generate one at GitHub → Settings → Developer settings → Personal access tokens, and paste that in as the password when prompted. (Or install the GitHub CLI and run `gh auth login` once, which handles this for you.)

## 8. Run the real daily loop

This is what turns the demo data into real predicted-vs-actual history:

```bash
python scripts/run_predictions.py     # once, before market open
python scripts/record_actuals.py      # once, after market close
```

Run these manually each trading day, or automate them — see step 10.

## 9. Retrain the AI model on real data

Once you've run the daily loop for a couple of weeks and have some real history:

```bash
python scripts/train_model.py --limit 30 --period 1y
```

This scans real tickers via yfinance (takes a minute or two) and replaces the demo model with one trained on live history. Re-run this weekly or so — check the **Model insights** page in the app to see whether it's actually beating the plain heuristic before trusting it.

## 10. (Optional) Automate the daily loop

macOS/Linux, via cron:

```bash
crontab -e
```

Add two lines (adjust the path and times — these are in your local timezone, aimed just before market open and just after close):

```
30 8 * * 1-5  cd /path/to/niffler && .venv/bin/python scripts/run_predictions.py
5 16 * * 1-5  cd /path/to/niffler && .venv/bin/python scripts/record_actuals.py
```

If you'd rather not deal with cron, ask me — this is also something a scheduled task in Cowork can trigger.

## Troubleshooting

- **"command not found: streamlit"** — your virtual environment isn't active. Run `source .venv/bin/activate` again.
- **"No module named 'X'"** — same cause, or `pip install -r requirements.txt` didn't finish. Re-run it.
- **Port 8501 already in use** — another Streamlit app is running. Stop it, or run `streamlit run web/streamlit_app.py --server.port 8502`.
- **Pages keep saying "illustrative data"** — yfinance couldn't reach Yahoo Finance (no internet, or a restrictive network). The app still works, just not with real prices until that's resolved.
- **`git push` rejects your password** — see the personal access token note in step 7.
