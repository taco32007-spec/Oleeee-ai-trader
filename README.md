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

Despite the project name, this version uses a transparent rules-based strategy, not a trained AI model. It does not promise profits.

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
