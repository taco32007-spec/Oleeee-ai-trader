"""Small supervised ML filter, no external API or dependencies.

Predicts whether next-open -> third-candle-close return exceeds estimated costs.
Not a calibrated probability, profit guarantee, or a backtest of stop/target policy.
"""
import math

VERSION = 'logistic-v1'
HORIZON = 3
THRESHOLD = .55
RETRAIN_MS = 12*60*60*1000
FEATURES = ['return_1', 'return_3', 'return_12', 'ema_gap', 'volatility_12', 'candle_body']


def features(rows, i):
    if i < 60:
        return None
    closes = [r['c'] for r in rows[i-60:i+1]]
    fast = slow = closes[0]
    for c in closes[1:]:
        fast += (c-fast)*2/10
        slow += (c-slow)*2/22
    returns = [closes[j]/closes[j-1]-1 for j in range(len(closes)-12,len(closes))]
    avg = sum(returns)/len(returns)
    r = rows[i]
    return [closes[-1]/closes[-2]-1, closes[-1]/closes[-4]-1,
            closes[-1]/closes[-13]-1, fast/slow-1,
            math.sqrt(sum((v-avg)**2 for v in returns)/len(returns)),
            (r['c']-r['o'])/max(r['h']-r['l'],r['c']*1e-10)]


def sigmoid(z):
    return 1/(1+math.exp(-max(-35.,min(35.,z))))


def normalize(x, mean, scale):
    return [max(-8.,min(8.,(v-m)/d)) for v,m,d in zip(x,mean,scale)]


def fit(x, y, epochs=140):
    n = len(x)
    mean = [sum(r[j] for r in x)/n for j in range(len(FEATURES))]
    scale = [max(1e-8,math.sqrt(sum((r[j]-mean[j])**2 for r in x)/n)) for j in range(len(FEATURES))]
    data = [normalize(r,mean,scale) for r in x]
    prior = (sum(y)+1)/(n+2)
    bias = math.log(prior/(1-prior))
    weights = [0.]*len(FEATURES)
    for _ in range(epochs):
        grad = [0.]*len(weights)
        gb = 0.
        for row,label in zip(data,y):
            err = sigmoid(bias+sum(w*v for w,v in zip(weights,row)))-label
            gb += err
            for j,v in enumerate(row):
                grad[j] += err*v
        bias -= .12*gb/n
        weights = [w-.12*(g/n+.02*w) for w,g in zip(weights,grad)]
    return dict(mean=mean, scale=scale, weights=weights, bias=bias, prior=prior)


def predict(model, x):
    return sigmoid(model['bias']+sum(w*v for w,v in zip(model['weights'],normalize(x,model['mean'],model['scale']))))


def loss(predictions, labels):
    return -sum(y*math.log(max(p,1e-12))+(1-y)*math.log(max(1-p,1e-12))
                for p,y in zip(predictions,labels))/len(labels)


def train(rows, cost_ratio):
    # labels use only outcomes closed by the last supplied candle.
    samples = []
    for i in range(60,len(rows)-HORIZON):
        samples.append((i, features(rows,i), int(rows[i+HORIZON]['c']/rows[i+1]['o'] > cost_ratio)))
    if len(samples) < 600:
        return dict(version=VERSION, ready=False, eligible=False, reason='Недостаточно истории для обучения',
                    asof=rows[-1]['t']+300_000, samples=len(samples))
    cut = int(len(samples)*.75)
    # Purge overlapping future label horizons at the chronological boundary.
    training, validation = samples[:cut-HORIZON], samples[cut:]
    labels = [r[2] for r in training]
    if len(set(labels)) < 2:
        return dict(version=VERSION, ready=False, eligible=False, reason='В обучении только один класс',
                    asof=rows[-1]['t']+300_000, samples=len(samples))
    model = fit([r[1] for r in training],labels)
    targets = [r[2] for r in validation]
    scores = [predict(model,r[1]) for r in validation]
    ll = loss(scores,targets)
    baseline = loss([model['prior']]*len(targets),targets)
    eligible = len(set(targets)) == 2 and ll < baseline
    model.update(version=VERSION, ready=True, eligible=eligible,
                 reason='ML-фильтр включён' if eligible else 'Нет преимущества на проверке: новые покупки заблокированы',
                 asof=rows[-1]['t']+300_000, samples=len(samples),
                 train_n=len(training), validation_n=len(validation),
                 train_start=rows[training[0][0]]['t'], train_end=rows[training[-1][0]]['t'],
                 train_label_end=rows[training[-1][0]+HORIZON]['t'],
                 validation_start=rows[validation[0][0]]['t'], validation_end=rows[validation[-1][0]+HORIZON]['t'],
                 accuracy=sum((p>=.5)==bool(y) for p,y in zip(scores,targets))/len(targets),
                 baseline_accuracy=sum((model['prior']>=.5)==bool(y) for y in targets)/len(targets),
                 logloss=ll, baseline_logloss=baseline,
                 positive_rate=sum(targets)/len(targets), threshold=THRESHOLD)
    return model


def decision(model, x, intent):
    score = predict(model,x) if model.get('ready') and x is not None else None
    if not intent:
        return None, score, 'Нет сигнала EMA'
    if intent['side'] == 'sell':
        return intent, score, 'Выход по EMA; ML не блокирует закрытие'
    if not model.get('eligible') or score is None:
        return None, score, model['reason']
    if score < THRESHOLD:
        return None, score, 'Покупка отклонена: оценка модели ниже 55%'
    return dict(intent, ai_score=score, model_asof=model['asof']), score, 'Покупка разрешена ML-фильтром'
