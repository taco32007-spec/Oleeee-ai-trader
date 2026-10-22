"""Tasman AI Trader: deterministic spot paper simulation; public GET market data only."""
import json
import math
import os
import shutil
import time
import urllib.request
from pathlib import Path
import ai_model as ml

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'docs' / 'state.json'
CAPITAL = 100_000.0
RISK = .02
FEE = .001
SLIPPAGE = .0005
INTERVAL = 300_000


def fresh():
    return dict(version=1, mode='PAPER', symbol='BTCUSDT', timeframe='5m', initial=CAPITAL,
                risk=RISK, cash=CAPITAL, position=None, pending=None, last=None,
                trades=[], equity=[], candles=[], updated=None, error=None)


def ema(values, span):
    result = [values[0]]
    for value in values[1:]:
        result.append(value * 2 / (span + 1) + result[-1] * (1 - 2 / (span + 1)))
    return result


def signal(rows, i):
    if i < 30:
        return None
    window = rows[max(0, i-200):i+1]
    closes = [r['c'] for r in window]
    fast, slow = ema(closes, 9), ema(closes, 21)
    tr = [max(r['h']-r['l'], abs(r['h']-window[j-1]['c']), abs(r['l']-window[j-1]['c']))
          for j, r in enumerate(window) if j > 0]
    distance = max(sum(tr[-14:])/14*2, closes[-1]*.005)
    if fast[-2] <= slow[-2] and fast[-1] > slow[-1]:
        return dict(side='buy', distance=distance)
    if fast[-2] >= slow[-2] and fast[-1] < slow[-1]:
        return dict(side='sell', distance=distance)
    return None


def close(s, price, timestamp, reason):
    p = s['position']
    fill = price * (1-SLIPPAGE)
    proceeds = p['qty'] * fill * (1-FEE)
    pnl = proceeds - p['cost']
    s['cash'] += proceeds
    s['trades'].append(dict(opened=p['opened'], closed=timestamp, entry=p['entry'], exit=fill,
                            qty=p['qty'], pnl=pnl, reason=reason, fees=p['entry_fee']+p['qty']*fill*FEE))
    s['position'] = None


def step(s, row, intent):
    # Signal from previous closed candle is executed at this candle's open.
    pending = s['pending']
    if pending and pending['side'] == 'sell' and s['position']:
        close(s, row['o'], row['t'], 'EMA reversal')
    elif pending and pending['side'] == 'buy' and not s['position']:
        entry = row['o'] * (1+SLIPPAGE)
        stop = entry - pending['distance']
        if stop > 0:
            # Include both commissions and expected exit slippage in 2% stop budget.
            risk_unit = entry*(1+FEE) - stop*(1-SLIPPAGE)*(1-FEE)
            qty = min(s['cash']*RISK/risk_unit, s['cash']/(entry*(1+FEE)))
            cost = qty*entry*(1+FEE)
            s['cash'] = max(0., s['cash']-cost)
            s['position'] = dict(qty=qty, entry=entry, stop=stop, target=entry+2*pending['distance'],
                                 cost=cost, opened=row['t'], entry_fee=qty*entry*FEE,
                                 planned_risk=qty*risk_unit)
    p = s['position']
    if p:
        # A gap can exceed the planned risk. If both levels touch, assume stop first.
        if row['o'] <= p['stop']:
            close(s, row['o'], row['t'], 'Stop gap')
        elif row['l'] <= p['stop']:
            close(s, p['stop'], row['t'], 'Stop loss')
        elif row['h'] >= p['target']:
            close(s, p['target'], row['t'], 'Take profit')
    s['pending'] = intent
    equity = s['cash'] + (s['position']['qty']*row['c'] if s['position'] else 0)
    s['equity'].append(dict(t=row['t']+INTERVAL, value=equity))
    s['last'] = row['t']


def ai_intent(s, rows, i):
    ai = s.setdefault('ai', dict(model=None, latest=None, approved=0, blocked=0))
    model = ai.get('model')
    asof = rows[i]['t']+INTERVAL
    if not model or model.get('version') != ml.VERSION or asof-model['asof'] >= ml.RETRAIN_MS:
        # Never train against later candles when replaying a delayed run.
        ratio = (1+SLIPPAGE)*(1+FEE)/((1-SLIPPAGE)*(1-FEE))
        model = ml.train(rows[:i+1],ratio)
        ai['model'] = model
    if model['asof'] > asof:
        raise ValueError('Model trained on future data; simulation paused')
    base = signal(rows,i)
    intent, score, reason = ml.decision(model,ml.features(rows,i),base)
    ai['latest'] = dict(t=asof, score=score, reason=reason, ema=base['side'] if base else 'wait')
    if base and base['side'] == 'buy':
        ai['approved' if intent else 'blocked'] += 1
    return intent


def process(s, rows, now):
    rows = sorted(rows, key=lambda r:r['t'])
    if len(rows) < 32:
        raise ValueError('Not enough closed candles')
    if any(b['t']-a['t'] != INTERVAL for a,b in zip(rows, rows[1:])):
        raise ValueError('Candle gap or duplicate; simulation paused')
    if now-(rows[-1]['t']+INTERVAL) > INTERVAL*2:
        raise ValueError('Market data is stale; simulation paused')
    if 'ai' not in s and s['last'] is not None:
        # On upgrade, cancel unfiltered entry intent; preserve portfolio and exits.
        if s.get('pending') and s['pending']['side'] == 'buy':
            s['pending'] = None
        existing = next((i for i,r in enumerate(rows) if r['t'] == s['last']),None)
        if existing is not None:
            ai_intent(s,rows,existing)
    s['version'] = 2
    # Start now, with empty history: never present backtest profit as live paper profit.
    if s['last'] is None:
        s['last'] = rows[-1]['t']
        s['pending'] = ai_intent(s, rows, len(rows)-1)
        s['equity'] = [dict(t=s['last']+INTERVAL, value=CAPITAL)]
    else:
        timestamps = [r['t'] for r in rows]
        if s['last'] not in timestamps:
            raise ValueError('Downtime exceeds available candle history; manual recovery required')
        for i, row in enumerate(rows):
            if row['t'] > s['last']:
                if i < 30:
                    raise ValueError('Insufficient indicator warmup after downtime')
                step(s, row, ai_intent(s, rows, i))
    s['candles'] = rows[-160:]
    s['updated'] = now
    s['error'] = None
    return s


def fetch():
    url = 'https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=1000'
    # No keys, signed endpoints, order methods, exchange SDKs, or account access exist.
    raw = []
    end = None
    # ~20.8 days of closed 5m candles; only six public market-data GETs.
    for _ in range(6):
        page_url = url + (f'&endTime={end}' if end is not None else '')
        with urllib.request.urlopen(page_url, timeout=25) as response:
            page = json.load(response)
        if not isinstance(page,list) or not page:
            raise ValueError('Empty or invalid candle response')
        raw = page + raw
        end = int(page[0][0])-1
    now = int(time.time()*1000)
    rows = []
    for k in raw:
        if int(k[6]) >= now:
            continue
        row = dict(t=int(k[0]), o=float(k[1]), h=float(k[2]), l=float(k[3]), c=float(k[4]))
        if not all(math.isfinite(row[x]) and row[x] > 0 for x in ('o','h','l','c')):
            raise ValueError('Invalid price')
        if not row['l'] <= min(row['o'],row['c']) <= max(row['o'],row['c']) <= row['h']:
            raise ValueError('Invalid OHLC')
        rows.append(row)
    return rows, now


def save(s):
    STATE.parent.mkdir(exist_ok=True)
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(s, ensure_ascii=False, allow_nan=False, separators=(',',':'))+'\n')
    os.replace(temp, STATE)


def main():
    STATE.parent.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / 'index.html', ROOT / 'docs' / 'index.html')
    s = json.loads(STATE.read_text()) if STATE.exists() else fresh()
    try:
        rows, now = fetch()
        # Work on a copy, so failed processing cannot persist half-applied fills.
        candidate = process(json.loads(json.dumps(s)), rows, now)
    except Exception as exc:
        s['error'] = f'{type(exc).__name__}: {exc}'
        s['attempted'] = int(time.time()*1000)
        save(s)
        print('PAPER PAUSED:', s['error'])
        return 1
    save(candidate)
    print(f"PAPER OK: last={candidate['last']}, closed trades={len(candidate['trades'])}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
