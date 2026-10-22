"""Reproducible Q-learning experiment on generated markets, isolated from BTC paper state.
Fixed train/validation/test seed ranges. Never tune against the final test results.
"""
import json
import math
import random
import statistics
import time
from pathlib import Path

VERSION = 'synthetic-q-v1'
CAPITAL = 100_000.
FEE = .001
SLIP = .0005
RISK = .02
BARS = 384
TRAIN_EPISODES = 1200
VALIDATION_SEEDS = tuple(range(50_000,50_040))
TEST_SEEDS = tuple(range(60_000,60_100))


def market(seed, bars=BARS, stress=False):
    rng = random.Random(seed)
    price = anchor = 100.
    regime, left, momentum = 0, 0, 0.
    rows = []
    for i in range(bars):
        if left <= 0:
            regime = rng.choice((-1,0,1))
            left = rng.randint(35,90)
            anchor = price
        sigma = rng.uniform(.0008,.0020) * (2. if stress else 1.)
        drift = regime*.0010 if regime else -.045*math.log(price/anchor)
        ret = drift + .12*momentum + rng.gauss(0,sigma)
        if stress and rng.random() < .015:
            ret += rng.choice((-1,1))*.025
        ret = max(-.08,min(.08,ret))
        close = price*math.exp(ret)
        wick = abs(rng.gauss(0,sigma))*.5
        rows.append(dict(o=price,c=close,h=max(price,close)*(1+wick),l=min(price,close)*(1-wick)))
        price, momentum, left = close, ret, left-1
    return rows


def observations(rows):
    # Causal features; no regime id, RNG state, future price or seed enters the agent.
    result=[];fast=slow=rows[0]['c']
    for i,r in enumerate(rows):
        fast += (r['c']-fast)*.2
        slow += (r['c']-slow)*2/22
        if i < 20:
            result.append(None);continue
        r3=r['c']/rows[i-3]['c']-1
        r12=r['c']/rows[i-12]['c']-1
        gap=fast/slow-1
        bucket=lambda x,b: 0 if x < -b else 2 if x > b else 1
        result.append((bucket(r3,.0015),bucket(r12,.003),bucket(gap,.002)))
    return result


class Wallet:
    def __init__(self, cost=1.):
        self.cash=CAPITAL;self.position=None;self.trades=[];self.cost=cost
        self.fees=0.;self.peak=CAPITAL;self.dd=0.

    def value(self,price):
        return self.cash+(self.position['qty']*price if self.position else 0.)

    def close(self,price):
        p=self.position
        fill=price*(1-SLIP*self.cost)
        fee=p['qty']*fill*FEE*self.cost
        proceeds=p['qty']*fill-fee
        self.cash+=proceeds;self.fees+=fee
        self.trades.append(proceeds-p['cost'])
        self.position=None

    def advance(self,row,action,terminal=False,protective=True):
        # Decision at prior close; execution at next open, then this bar's stop/target.
        if action==2 and self.position:
            self.close(row['o'])
        elif action==1 and not self.position:
            fill=row['o']*(1+SLIP*self.cost)
            distance=fill*.008
            stop=fill-distance
            risk_unit=fill*(1+FEE*self.cost)-stop*(1-SLIP*self.cost)*(1-FEE*self.cost)
            qty=min(self.cash*RISK/risk_unit,self.cash/(fill*(1+FEE*self.cost)))
            fee=qty*fill*FEE*self.cost
            cost=qty*fill+fee
            self.cash=max(0.,self.cash-cost);self.fees+=fee
            self.position=dict(qty=qty,cost=cost,stop=stop,target=fill+2*distance,risk=qty*risk_unit)
        p=self.position
        if p and protective:
            if row['o'] <= p['stop']:self.close(row['o'])
            elif row['l'] <= p['stop']:self.close(p['stop'])
            elif row['h'] >= p['target']:self.close(p['target'])
        if terminal and self.position:self.close(row['c'])
        value=self.value(row['c'])
        self.peak=max(self.peak,value);self.dd=max(self.dd,(self.peak-value)/self.peak)
        return value


def key(obs,wallet):
    return ','.join(map(str,(*obs,int(wallet.position is not None))))


def allowed(wallet):
    return (0,2) if wallet.position else (0,1)


def greedy(q,k,actions):
    values=q.get(k,[0.,0.,0.])
    # Stable ties choose wait; evaluation never writes to the learned table.
    return max(actions,key=lambda a:values[a])


def run(seed,q=None,training=False,epsilon=0.,gamma=.97,alpha=.08,cost=1.,stress=False,baseline=None,bars=BARS):
    rows=market(seed,bars,stress);obs=observations(rows);wallet=Wallet(cost)
    rng=random.Random(seed+999_999)
    q={} if q is None else q
    equity=[CAPITAL]
    for i in range(20,len(rows)-1):
        k=key(obs[i],wallet);actions=allowed(wallet)
        if baseline=='cash':action=0
        elif baseline=='hold':action=0 if wallet.position else 1
        elif baseline=='ema':
            # Trend-following baseline uses the same causal EMA-gap observation and wallet.
            action=(1 if obs[i][2]==2 else 0) if not wallet.position else (2 if obs[i][2]==0 else 0)
        else:action=rng.choice(actions) if training and rng.random()<epsilon else greedy(q,k,actions)
        previous=wallet.value(rows[i]['c'])
        terminal=i==len(rows)-2
        value=wallet.advance(rows[i+1],action,terminal,protective=baseline!='hold')
        equity.append(value)
        if training:
            values=q.setdefault(k,[0.,0.,0.])
            nk=key(obs[i+1],wallet)
            target=100*math.log(value/previous)
            if not terminal:target+=gamma*max(q.get(nk,[0.,0.,0.])[a] for a in allowed(wallet))
            values[action]+=alpha*(target-values[action])
    return dict(return_pct=(equity[-1]/CAPITAL-1)*100,dd_pct=wallet.dd*100,
                trades=len(wallet.trades),fees=wallet.fees,equity=equity)


def summarize(results):
    values=[r['return_pct'] for r in results]
    avg=statistics.mean(values)
    se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else 0.
    return dict(mean=avg,median=statistics.median(values),win_rate=sum(v>0 for v in values)/len(values)*100,
                worst=min(values),best=max(values),mean_dd=statistics.mean(r['dd_pct'] for r in results),
                worst_dd=max(r['dd_pct'] for r in results),mean_trades=statistics.mean(r['trades'] for r in results),
                mean_fees=statistics.mean(r['fees'] for r in results),ci95=[avg-1.96*se,avg+1.96*se],n=len(values))


def train(episodes=TRAIN_EPISODES):
    # Hyperparameters fixed before testing. Checkpoint selection sees validation only.
    q={};curve=[];checkpoints=[]
    for episode in range(episodes):
        epsilon=.30-(.27*episode/max(1,episodes-1))
        result=run(1_000+episode,q,True,epsilon)
        curve.append(result['return_pct'])
        if (episode+1) % max(1,episodes//3)==0 or episode+1==episodes:
            validation=summarize([run(seed,q) for seed in VALIDATION_SEEDS])
            checkpoints.append(dict(episode=episode+1,validation=validation,q={k:v[:] for k,v in q.items()}))
    chosen=max(checkpoints,key=lambda c:c['validation']['mean'])
    return chosen,checkpoints,curve


def build():
    chosen,checkpoints,curve=train()
    q=chosen['q']
    # Final test is evaluated once, after all training and checkpoint selection.
    results=[run(seed,q) for seed in TEST_SEEDS]
    baselines={name:summarize([run(seed,baseline=name) for seed in TEST_SEEDS]) for name in ('cash','hold','ema')}
    stress=summarize([run(seed,q,cost=2.,stress=True) for seed in range(70_000,70_050)])
    before=summarize([run(seed,{}) for seed in TEST_SEEDS])
    mean_curve=[statistics.mean([r['equity'][i] for r in results]) for i in range(len(results[0]['equity']))]
    stats=summarize(results)
    return dict(version=VERSION,mode='SYNTHETIC_ONLY',created=int(time.time()*1000),
                algorithm='Tabular Q-learning',capital=CAPITAL,bars=BARS,bar_minutes=5,
                episodes=TRAIN_EPISODES,steps=TRAIN_EPISODES*(BARS-21),selected_episode=chosen['episode'],
                seeds=dict(train=[1000,1000+TRAIN_EPISODES-1],validation=[50000,50039],test=[60000,60099],stress=[70000,70049]),
                validation=chosen['validation'],test=stats,before=before,baselines=baselines,stress=stress,
                checkpoints=[dict(episode=c['episode'],validation=c['validation']) for c in checkpoints],
                learning_curve=[dict(episode=i+1,return_pct=statistics.mean(curve[max(0,i-49):i+1])) for i in range(49,len(curve),50)],
                equity=[dict(t=i*300000,value=v) for i,v in enumerate(mean_curve)],
                example_seed=TEST_SEEDS[0],example_return=results[0]['return_pct'],
                example_equity=results[0]['equity'],test_returns=[r['return_pct'] for r in results],
                passed=stats['ci95'][0]>0,q=q,
                limitations='Simplified generated markets with learnable trends; same simulator distribution in train/test, correlated bars. No evidence of profitability on real BTC. Stress adds shocks and doubles costs. Buy-and-hold benchmark holds all cash in the asset without stop loss; its risk differs from the protected agent.')


def ensure(root):
    root=Path(root)
    report=root/'synthetic_report.json'
    if report.exists():
        result=json.loads(report.read_text())
        if result.get('version') != VERSION:
            raise ValueError('Synthetic report version mismatch; explicitly retrain before deployment')
    else:
        result=build()
        report.write_text(json.dumps(result,allow_nan=False,separators=(',',':'))+'\n')
    path=root/'docs'/'lab.json'
    path.parent.mkdir(exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,allow_nan=False,separators=(',',':'))+'\n')
    tmp.replace(path)
    return result

if __name__=='__main__':
    import sys
    root=Path(__file__).resolve().parent
    if '--retrain' in sys.argv:
        result=build()
        (root/'synthetic_report.json').write_text(json.dumps(result,allow_nan=False,separators=(',',':'))+'\n')
    result=ensure(root)
    print('SYNTHETIC TEST:',json.dumps(result['test']))
