# Using Niffler — a plain-English guide

This isn't setup documentation (see `GETTING_STARTED.md` for that) or a
technical README (see `README.md`). This is for anyone actually looking at
the app and wondering what they're looking at — written the way you'd
explain it to a friend, not a spec sheet.

## What Niffler actually does, in one paragraph

Every weekday before the market opens, Niffler predicts which direction a
handful of stocks will move that day. After the market closes, it checks
what actually happened and writes down whether it was right. It does this
forever, automatically, and shows you the real track record — not a
cherry-picked highlight reel. That's the whole idea: a prediction app that
grades its own homework, in public, including on the days it's wrong.

## A tour of the app

**Home** — your watchlist (five tickers by default, edit `src/config.py`
to change them), each with today's prediction and a plain-language reason
for it. Below that, "The hoard": a gauge showing the overall hit rate
across everything Niffler has predicted and later checked. "Niffler's
nightly whisper" is just a one-line mood summary of whether the watchlist
leans up or down tonight — flavor text, not a signal of its own.

**Stock lookup** — type any ticker and get a live prediction plus its
price history. If that ticker has any resolved predictions in its past,
you'll also see a chart comparing what was predicted against what actually
happened, day by day.

**Top 50 predictions** — scans the S&P 500 and ranks every ticker by
predicted move, biggest gainers first (with a toggle to flip to predicted
decliners). Useful as a starting point for your own research, not a list
of things to buy.

**Accuracy tracker** — the app's report card. Overall hit rate, how many
predictions have been checked, average error size, hit rate over time
(daily bars plus a smoothed 7-day line), a scatter plot of predicted vs.
actual move for every single prediction, and a per-ticker breakdown. This
is the page to check if you want the honest number, not the marketing
number.

**Model insights** — what's actually running under the hood. Whether
predictions are coming from the plain heuristic or the trained AI model,
how the AI model did on data it never trained on (its "holdout" score),
and which signals it leans on most. If the AI model isn't beating the
plain heuristic, this page says so — it's built to be honest about that,
not to talk up the AI.

**Dumbledore** — the floating chat button in the bottom-right corner on
every page. See below.

## Understanding the numbers

**Predicted %** is how much the model thinks a stock will move by the next
close, and **direction** is just the sign of that (up if zero or
positive, down if negative).

**Confidence** is *not* a probability of being right — it's how strongly
the model's own signals agree with each other. A 90%-confidence call can
still miss; it just means momentum and RSI were pointing the same way
when it made the call.

**Hit rate** is the percentage of predictions where the *direction* was
right — it doesn't care how close the size of the move was, only whether
up-vs-down was correct.

**Mean absolute error** is the flip side: how far off the predicted %
move was from the actual % move, on average, regardless of direction.

**Heuristic vs. AI badge** — every prediction is tagged with which model
actually produced it. The heuristic is a transparent momentum + RSI
formula, always available. The AI model, once trained, tries to learn
when to trust or override the heuristic — but only takes over once it
exists (`scripts/train_model.py`), and you can always see which one made
any given call.

## Talking to Dumbledore

Dumbledore is a chat widget that answers questions using Niffler's own
live data — not a general-purpose chatbot. It's good for things like:

- *"Why is AAPL predicted to move?"*
- *"What's the accuracy on TSLA?"*
- *"What is RSI?"* or *"What is momentum?"*
- *"How does the AI model work?"*

It will **not** tell you whether to buy, sell, or hold anything, even if
you ask in a roundabout way — that check happens before the question ever
reaches an AI model, so it can't be talked around by rephrasing. This is a
public demo app, not a licensed financial advisor, and nothing in it is
actually good enough to responsibly tell a stranger when to trade. Ask it
to explain what the signals show instead, and it will.

Under the hood, Dumbledore runs on a free template system by default (no
external API, always available), and can optionally be upgraded to a real
language model (Groq or a local Ollama server) for more natural
conversation — the no-advice rule holds either way.

## How it stays honest (the self-correction loop)

Three automated jobs, running on GitHub Actions (not a laptop that has to
stay on), keep the whole thing current with zero manual work:

1. **Every weekday morning, before the market opens** — log a prediction
   for each tracked ticker.
2. **Every weekday evening, after the market closes** — check what
   actually happened and grade the morning's predictions.
3. **Every Sunday** — retrain the AI model on everything that's
   accumulated since the last retrain, so it keeps adapting to recent
   history instead of running on a stale snapshot forever.

If a check-in ever gets missed (an outage, a bug), the next successful run
catches up on anything still pending — nothing gets permanently skipped.

## Reading the numbers honestly

A few things worth keeping in mind so the numbers don't get
over-interpreted:

- **Sample size matters.** Five tickers a day means the daily hit rate on
  the Accuracy tracker can swing between 20% and 80% purely from noise.
  The 7-day rolling line is a steadier read than any single day.
- **A hit rate near 50% is expected, not a bug.** Predicting daily stock
  direction is genuinely hard — professional models rarely do much better
  than a coin flip consistently. Niffler says this plainly on the Model
  insights page rather than dressing it up.
- **"AI" doesn't automatically mean "better."** The AI model only replaces
  the heuristic where it's demonstrably improving on it — check Model
  insights to see whether that's actually the case right now.
- **"Illustrative data" means the real data source was unreachable** (no
  network, a rate limit, a delisted ticker) and the app fell back to a
  synthetic price series so the page still shows something. It says so
  on-screen whenever this happens — it's never silent about it.

## The disclaimer, stated plainly

Niffler is a personal project. Its predictions come from a simple
formula and an experimental model trained on limited history. They are
not investment advice, and the app — including Dumbledore — will tell you
that directly if you ask it to weigh in on a trade.

## FAQ

**Why isn't the accuracy higher?**
Because next-day stock direction is close to a coin flip even for
professional models — Niffler doesn't have some hidden edge that would
make it dramatically better, and it doesn't pretend to.

**Is a ~50% hit rate useless, then?**
It means Niffler isn't beating the market, which is an honest and
expected result for a hobby project — not a claim that it's broken. The
point of this app was never "beat Wall Street"; it's "build something
that predicts and grades itself, out loud, without hiding the bad days."

**Can I get Dumbledore to give me a stock tip if I phrase it cleverly?**
No — the guardrail checks the question itself before it ever reaches an
AI model, so rewording it doesn't get around it.

**Why do some pages say "illustrative data"?**
The real-time data source (Yahoo Finance via `yfinance`) couldn't be
reached for that ticker at that moment. The app falls back to a
synthetic series so the page still works, and always tells you when
that's happening.

**What's the difference between the heuristic and the AI model?**
The heuristic is a fixed formula (momentum + RSI) — always on, always
explainable. The AI model is trained on real history and tries to
improve on the heuristic's calls, but only exists once someone runs
`scripts/train_model.py` (or the weekly automated retrain), and every
prediction is tagged with which one actually made it.
