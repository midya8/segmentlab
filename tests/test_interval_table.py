import copy
import json
import math
import random
import unittest
from unittest.mock import patch
from segmentlab.core import infer,interval_table,segment_score

class IntervalTableTests(unittest.TestCase):
    def test_every_interval_exact_equality(self):
        rng=random.Random(27)
        for n in [1,2,7,19]:
            ys=[rng.gauss(0,1) for _ in range(n)]
            for minimum in range(1,n+1):
                scores,work=interval_table(ys,.7,.4,2.,minimum)
                self.assertEqual(len(scores),(n-minimum+1)*(n-minimum+2)//2)
                for (start,end),value in scores.items():self.assertEqual(value,segment_score(ys[start:end],.7,.4,2.))
    def test_large_offset_and_constant_exact_equality(self):
        for ys in [[1e6,1e6-.1,1e6-.2,1e6-.3],[7.]*12,[-1e6,1e6,-1e6,1e6]]:
            scores,_=interval_table(ys,.01,.3,.1,1)
            for (start,end),value in scores.items():self.assertEqual(value,segment_score(ys[start:end],.01,.3,.1))
    def test_actual_observation_reads_not_just_formula(self):
        class Counted(list):
            def __init__(self,ys):super().__init__(ys);self.reads=0
            def __getitem__(self,k):self.reads+=1;return super().__getitem__(k)
        ys=Counted([.1]*36);_,work=interval_table(ys,.35,0.,3.,3)
        # Two scalar reads per actual Welford update in this implementation.
        self.assertEqual(ys.reads,2*work['rolling_observation_updates'])
        self.assertEqual(work['rolling_observation_updates'],663)
        self.assertEqual(work['independent_rescan_observation_updates'],8330)
        self.assertEqual(work['avoided_observation_updates'],7667)
    def test_work_counts_exhaustive_lengths(self):
        for n in range(1,15):
            for minimum in range(1,n+1):
                table,work=interval_table([.1]*n,1.,0.,1.,minimum)
                self.assertEqual(work['independent_rescan_observation_updates'],sum(end-start for start,end in table))
                self.assertEqual(work['rolling_observation_updates'],sum(n-start for start in range(n-minimum+1)))
                self.assertGreaterEqual(work['avoided_observation_updates'],0)
    def test_exact_full_inference_against_rescan(self):
        import segmentlab.core as core
        ys=[-.3,.2,.4,1.2,1.5,-.4,.3];r=infer(ys,min_length=2,max_segments=3)
        original=core.interval_table
        def rescan(y,sigma,mu,tau,minimum):
            _,work=original(y,sigma,mu,tau,minimum)
            return {(start,end):segment_score(y[start:end],sigma,mu,tau) for start in range(len(y)) for end in range(start+minimum,len(y)+1)},work
        with patch.object(core,'interval_table',rescan):self.assertEqual(r,infer(ys,min_length=2,max_segments=3))
    def test_maximum_bound_and_no_alias(self):
        ys=[math.sin(i) for i in range(120)];before=ys[:];table,work=interval_table(ys,1.,0.,3.,1)
        self.assertEqual(len(table),7260);self.assertEqual(work['rolling_observation_updates'],7260);self.assertEqual(work['independent_rescan_observation_updates'],295240)
        self.assertEqual(ys,before);first=table[0,1]['posterior_mean'];table[0,2]['posterior_mean']=99;self.assertEqual(table[0,1]['posterior_mean'],first)
    def test_work_audit_only_added_public_field(self):
        r=infer([0.,.1,1.,1.1],min_length=2,max_segments=2)
        self.assertEqual(r['score_work']['intervals_scored'],6);json.dumps(r,allow_nan=False)
