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

if __name__=='__main__': unittest.main()
