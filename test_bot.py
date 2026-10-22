import unittest
import bot

class PaperTests(unittest.TestCase):
    def row(self,t=0,**kw):
        return dict(t=t,o=100,h=101,l=99,c=100,**kw)
    def test_risk_and_cash_cap(self):
        s=bot.fresh(); s['pending']={'side':'buy','distance':5}
        bot.step(s,self.row(),None)
        self.assertGreaterEqual(s['cash'],0)
        self.assertLessEqual(s['position']['planned_risk'],2000.00001)
        self.assertAlmostEqual(s['cash']+s['position']['cost'],100000)
    def test_stop_before_target_and_fees(self):
        s=bot.fresh(); s['pending']={'side':'buy','distance':5}
        bot.step(s,dict(t=0,o=100,h=120,l=90,c=100),None)
        self.assertIsNone(s['position'])
        self.assertEqual(s['trades'][0]['reason'],'Stop loss')
        self.assertAlmostEqual(s['cash'],98000,places=5)
        self.assertGreater(s['trades'][0]['fees'],0)
    def test_no_lookahead_or_duplicate(self):
        rows=[self.row(t=i*bot.INTERVAL) for i in range(100)]
        s=bot.process(bot.fresh(),rows,100*bot.INTERVAL)
        self.assertEqual(len(s['trades']),0)
        before=s.copy(); bot.process(s,rows,100*bot.INTERVAL)
        self.assertEqual(before,s)
        self.assertEqual(len(s['equity']),1)
    def test_stale_data_rejected(self):
        rows=[self.row(t=i*bot.INTERVAL) for i in range(100)]
        with self.assertRaises(ValueError): bot.process(bot.fresh(),rows,105*bot.INTERVAL)
    def test_catchup_each_candle_once(self):
        rows=[self.row(t=i*bot.INTERVAL) for i in range(100)]
        s=bot.process(bot.fresh(),rows[:90],90*bot.INTERVAL)
        bot.process(s,rows,100*bot.INTERVAL)
        self.assertEqual(len(s['equity']),11)
        bot.process(s,rows,100*bot.INTERVAL)
        self.assertEqual(len(s['equity']),11)
    def test_gap_rejected(self):
        rows=[self.row(t=i*bot.INTERVAL) for i in range(100) if i!=50]
        with self.assertRaises(ValueError): bot.process(bot.fresh(),rows,100*bot.INTERVAL)


class MLTests(unittest.TestCase):
    @staticmethod
    def history(n=900):
        import math
        rows=[]
        for i in range(n):
            c=100+5*math.sin(i*.17)+.002*i
            o=rows[-1]['c'] if rows else c
            rows.append(dict(t=i*bot.INTERVAL,o=o,c=c,h=max(o,c)+.2,l=min(o,c)-.2))
        return rows

    def test_model_learns_weights(self):
        import ai_model as ml
        x=[[float(i%2),0,0,0,0,0] for i in range(100)]
        model=ml.fit(x,[i%2 for i in range(100)])
        self.assertGreater(ml.predict(model,[1,0,0,0,0,0]),.8)
        self.assertLess(ml.predict(model,[0,0,0,0,0,0]),.2)

    def test_features_never_read_future(self):
        import ai_model as ml
        rows=self.history()
        self.assertEqual(ml.features(rows,300),ml.features(rows[:301],300))

    def test_chronology_purge_and_train_only_scaling(self):
        import ai_model as ml
        rows=self.history()
        model=ml.train(rows,1.003)
        self.assertTrue(model['ready'])
        self.assertLess(model['train_label_end'],model['validation_start'])
        self.assertLessEqual(model['validation_end']+bot.INTERVAL,model['asof'])
        altered=[dict(r) for r in rows]
        for r in altered:
            if r['t']>=model['validation_start']:
                for field in ('o','c','h','l'): r[field]*=10
        second=ml.train(altered,1.003)
        for key in ('weights','bias','mean','scale'):
            self.assertEqual(model[key],second[key])

    def test_no_model_no_buy_but_exit_allowed(self):
        import ai_model as ml
        model=ml.train(self.history(100),1.003)
        self.assertFalse(model['ready'])
        self.assertIsNone(ml.decision(model,None,dict(side='buy',distance=5))[0])
        self.assertEqual(ml.decision(model,None,dict(side='sell',distance=5))[0]['side'],'sell')

    def test_unvalidated_or_low_score_blocks_entry(self):
        import ai_model as ml
        model=dict(ready=True,eligible=False,reason='Validation failed',mean=[0]*6,scale=[1]*6,weights=[0]*6,bias=10,asof=0)
        self.assertIsNone(ml.decision(model,[0]*6,dict(side='buy'))[0])
        model.update(eligible=True,bias=-10)
        self.assertIsNone(ml.decision(model,[0]*6,dict(side='buy'))[0])
        model['bias']=10
        self.assertGreater(ml.decision(model,[0]*6,dict(side='buy'))[0]['ai_score'],.55)

    def test_upgrade_preserves_portfolio_and_cancels_old_entry(self):
        rows=self.history()
        s=bot.fresh();s.update(last=rows[-1]['t'],cash=98000,pending=dict(side='buy',distance=5))
        bot.process(s,rows,900*bot.INTERVAL)
        self.assertEqual(s['cash'],98000)
        self.assertEqual(s['trades'],[])
        self.assertIsNone(s['pending'])
        self.assertTrue(s['ai']['model']['ready'])
        self.assertEqual(s['version'],2)

    def test_future_model_rejected(self):
        import ai_model as ml
        rows=self.history(100)
        s=bot.fresh();s['ai']=dict(model=dict(version=ml.VERSION,asof=999999999),latest=None,approved=0,blocked=0)
        with self.assertRaises(ValueError):bot.ai_intent(s,rows,99)


class SyntheticTests(unittest.TestCase):
    def test_repeatable_generator_and_causal_observations(self):
        import synthetic_lab as lab
        rows=lab.market(123)
        self.assertEqual(rows,lab.market(123))
        self.assertNotEqual(rows,lab.market(124))
        self.assertEqual(lab.observations(rows)[100],lab.observations(rows[:101])[100])

    def test_cash_and_untrained_evaluation_do_not_trade(self):
        import synthetic_lab as lab
        q={}
        self.assertEqual(lab.run(60000,q)['return_pct'],0)
        self.assertEqual(q,{})
        self.assertEqual(lab.run(60000,baseline='cash')['fees'],0)

    def test_round_trip_costs_and_risk(self):
        import synthetic_lab as lab
        w=lab.Wallet();row=dict(o=100,c=100,h=100,l=100)
        w.advance(row,1)
        self.assertLessEqual(w.position['risk'],lab.CAPITAL*lab.RISK+1e-8)
        self.assertGreaterEqual(w.cash,0)
        w.advance(row,2)
        self.assertLess(w.cash,lab.CAPITAL)
        self.assertEqual(len(w.trades),1)
        self.assertAlmostEqual(w.cash-lab.CAPITAL,sum(w.trades))

    def test_stop_precedes_target(self):
        import synthetic_lab as lab
        w=lab.Wallet();w.advance(dict(o=100,c=100,h=110,l=90),1)
        self.assertIsNone(w.position)
        self.assertLess(w.trades[0],0)

    def test_test_policy_is_frozen(self):
        import copy
        import synthetic_lab as lab
        q={};lab.run(1000,q,training=True,epsilon=.3,bars=100)
        snapshot=copy.deepcopy(q)
        lab.run(60000,q)
        self.assertEqual(q,snapshot)
        self.assertGreater(len(q),0)

    def test_disjoint_seed_ranges(self):
        import synthetic_lab as lab
        train=set(range(1000,1000+lab.TRAIN_EPISODES))
        self.assertFalse(train.intersection(lab.VALIDATION_SEEDS))
        self.assertFalse(train.intersection(lab.TEST_SEEDS))
        self.assertFalse(set(lab.VALIDATION_SEEDS).intersection(lab.TEST_SEEDS))

    def test_frozen_report_matches_replayed_metrics(self):
        import json
        import synthetic_lab as lab
        report=json.loads((bot.ROOT/'synthetic_report.json').read_text())
        self.assertEqual(report['version'],lab.VERSION)
        self.assertEqual(len(report['test_returns']),100)
        result=lab.run(lab.TEST_SEEDS[0],report['q'])
        self.assertAlmostEqual(result['return_pct'],report['test_returns'][0],places=9)
        self.assertEqual(result['equity'],report['example_equity'])
        self.assertAlmostEqual(sum(report['test_returns'])/100,report['test']['mean'])

if __name__=='__main__': unittest.main()
