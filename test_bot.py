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

if __name__=='__main__': unittest.main()
