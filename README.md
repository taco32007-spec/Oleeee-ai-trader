# Tasman AI Trader

Mobile dashboard and cloud **paper-trading only** experiment. No real orders, wallets, exchange accounts or API keys. Public BTC/USDT market data is fetched with GET from Binance's market-data-only API.

## Configuration

- Initial virtual capital: **100,000 USDT**.
- Closed candles: **5m**; pair: **BTCUSDT**, spot, long only, no leverage.
- Risk budget: **2% of current equity per trade**, including estimated commissions/slippage, capped by available cash. Actual risk may be lower because spot cannot borrow. Gaps can exceed the planned stop loss.
- Entry: EMA(9) crossing above EMA(21); exit: reverse cross, stop or target.
- Stop: max(2 × 14-candle mean true range, 0.5% price); target 2× stop distance.
- Commission 0.1% each side; adverse slippage 0.05% each side.
- Signals are generated after a candle closes and simulated at the next candle open. Delayed runs replay missing closed candles in order. This is an OHLC simulation, not exchange execution. If both stop and target touch, stop takes precedence.
- First startup begins at 100,000; historical profits are never fabricated. A strategy signal may take time; no trade is forced to fill the screen.

## Trained ML filter (v2)

EMA entries now require approval from a supervised logistic regression model in `ai_model.py`. It is trained with batch gradient descent and L2 regularization, using six price-derived features: 1/3/12-candle returns, EMA gap, 12-candle volatility and candle body. No paid API, LLM service or keys.

- Fetches ~6,000 public 5m candles (~20.8 days); fixed 15-minute label: next open to third candle close must beat both commissions and expected slippage. This label is not the profit outcome of the actual stop/target strategy.
- Chronological 75% training / 25% validation, purging 3 samples between sets. Scaler is fitted on training only. Latest 3 examples have no labels yet and are excluded. No shuffled split or future features.
- Stored model retrains every 12 hours using only candles closed by the signal time. Backfilled candles cannot use a future-trained model.
- New entries require validation log loss below a constant forecast based on training class frequency and model score >=55%. Insufficient data, single-class training, or failed validation block entries. EMA exits and protective stops are never blocked by ML.
- Dashboard shows the score, accuracy vs majority-class baseline, log losses, sample counts and validation period. Scores are uncalibrated estimates; a 60% score is not a verified 60% chance of trade profit. Historical metrics and the validation gate are experimental and do not prove an edge. A longer untouched forward paper test is still needed.
- Existing balance, open positions and closed trades are preserved on upgrade. An old, unfiltered pending buy is cancelled. ML counters start with the upgrade; older trades remain part of the portfolio and must not be attributed to ML.

It does not promise profits.

## Cloud hosting

GitHub Actions runs tests and simulation on a five-minute cron and saves `docs/state.json`; GitHub Pages serves the generated `docs` (frontend source: `index.html`). Enable Pages → Source: GitHub Actions. Run the workflow manually for initial deployment. Public repository / standard GitHub-hosted runners and Pages are used to avoid paid hosting. No custom credentials are needed: GitHub supplies an ephemeral GITHUB_TOKEN.

**Free scheduler limitations:** GitHub can delay/drop scheduled runs; 5m is the candle interval, not a guaranteed execution SLA. The dashboard marks stale data. Catch-up replays candle fills retrospectively, not live fills; gaps beyond available warmup/history pause the engine. Scheduled workflows may be disabled by GitHub after 60 days without repository activity. A failed market request retains the previous portfolio and publishes the error. Dashboard is public and read-only; all portfolio numbers are fictional.

## Development

Python 3.12, standard library only; no dependencies to install.

```sh
python -m unittest discover -s . -p test_bot.py -v
python bot.py
python -m http.server 8000 --directory docs
```

Repository state and history are the durable portfolio record. Do not edit `state.json` while a workflow runs. To change the engine, edit `bot.py` and run tests. To stop it, disable the workflow in Actions. Strategy and settings are intentionally read-only in the public dashboard.
