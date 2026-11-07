import copy
import itertools
import json
import math
import subprocess
import sys
import unittest
from segmentlab import infer,boundary_decision
from segmentlab.core import lse
from test_core import enumerate_partitions


class DecisionTests(unittest.TestCase):
    def test_optimal_against_all_candidates_and_all_true_partitions(self):
        for n,minimum,cap in [(1,1,1),(6,1,4),(7,2,3),(8,3,2)]:
            data=[(-1)**i*(i+1)/5 for i in range(n)]
            parts=enumerate_partitions(data,minimum=minimum,cap=cap);z=lse(p[3] for p in parts)
            for false,missed in [(1.,1.),(.2,3.),(4.,.3)]:
                result=boundary_decision(data,false,missed,min_length=minimum,max_segments=cap)
                risks=[]
                for cuts,_,_,_ in parts:
                    loss=math.fsum(math.exp(score-z)*(false*len(set(cuts)-set(truth))+missed*len(set(truth)-set(cuts))) for truth,_,_,score in parts)
                    risks.append((loss,len(cuts),cuts))
                best=min(risks);self.assertAlmostEqual(result['decision']['expected_loss'],best[0],12)
                self.assertAlmostEqual(next(row[0] for row in risks if row[2]==result['decision']['cuts']),best[0],12)
                self.assertLessEqual(result['decision']['expected_loss'],result['map_comparison']['expected_loss']+1e-12)
    def test_unconstrained_threshold_matches_when_admissible(self):
        for cost in [.01,.2,1.,5.]:
            result=boundary_decision([-.7,-.4,.6,.9],cost,1.,min_length=1,max_segments=4)
            self.assertTrue(result['unconstrained_threshold']['admissible'])
            self.assertEqual(result['decision']['cuts'],result['unconstrained_threshold']['cuts'])
    def test_threshold_can_violate_count_cap(self):
        result=boundary_decision([0.]*6,.0001,1.,min_length=1,max_segments=2)
        self.assertFalse(result['unconstrained_threshold']['admissible'])
        self.assertEqual(len(result['decision']['cuts']),1)
        self.assertGreater(len(result['unconstrained_threshold']['cuts']),1)
    def test_constraints_and_risk_decomposition(self):
        result=boundary_decision([0.,0.,1.,1.,0.,0.],.01,1.,min_length=2,max_segments=2)
        cuts=result['decision']['cuts'];bounds=[0]+cuts+[6]
        self.assertTrue(all(b-a>=2 for a,b in zip(bounds,bounds[1:])))
        row=result['decision'];self.assertAlmostEqual(row['expected_loss'],.01*row['expected_false_cuts']+row['expected_missed_cuts'],14)
        self.assertAlmostEqual(row['expected_loss'],result['numerical_audit']['no_cut_base_loss']+result['numerical_audit']['selected_increment_sum'],13)
    def test_forced_partition_single_observation(self):
        for data,minimum in [([.2],1),([.2,.3,.4],2)]:
            result=boundary_decision(data,min_length=minimum,max_segments=12)
            self.assertEqual(result['decision']['cuts'],[]);self.assertEqual(result['decision']['expected_loss'],0)
    def test_cost_scaling_changes_loss_not_decision(self):
        data=[-.3,.2,.9,-.5,.2]
        a=boundary_decision(data,.2,3.,min_length=1,max_segments=3)
        b=boundary_decision(data,2.,30.,min_length=1,max_segments=3)
        self.assertEqual(a['decision']['cuts'],b['decision']['cuts'])
        self.assertAlmostEqual(a['decision']['expected_loss']*10,b['decision']['expected_loss'],12)
    def test_more_missed_cost_not_fewer_selected_cuts_unconstrained(self):
        data=[-.7,-.4,.6,.9];counts=[]
        for cost in [.1,1.,10.,100.]:counts.append(len(boundary_decision(data,1.,cost,min_length=1,max_segments=4)['decision']['cuts']))
        self.assertEqual(counts,sorted(counts))
    def test_input_no_mutation_original_inference(self):
        data=[0.,.2,.9];before=data[:];result=boundary_decision(data,min_length=1)
        self.assertEqual(data,before);self.assertEqual(result['inference'],infer(data,min_length=1))
    def test_maximum_n_and_extremes(self):
        for data,settings in [([0.]*120,dict(min_length=60,max_segments=12)),([-1e6,1e6],dict(noise_std=1e-4,prior_std=1e4,min_length=1,max_segments=2))]:
            result=boundary_decision(data,**settings);json.dumps(result,allow_nan=False)
    def test_invalid_costs_and_settings(self):
        for kw in [dict(false_cut_cost=0),dict(missed_cut_cost=-1),dict(false_cut_cost=True),dict(missed_cut_cost=float('nan')),dict(false_cut_cost=1e7),dict(min_length=0)]:
            with self.assertRaises((ValueError,TypeError)):boundary_decision([0.,1.],**kw)
    def test_cli(self):
        command=[sys.executable,'-m','segmentlab','decide','examples/sequence.json','--false-cut-cost','.2','--missed-cut-cost','3']
        result=subprocess.run(command,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['loss']['false_cut_cost'],.2)
        result=subprocess.run(command[:-1]+['0'],capture_output=True,text=True);self.assertEqual(result.returncode,2)
    def test_exact_ties_fewer_segments_then_lexicographic(self):
        from unittest.mock import patch
        fixture=infer([0.]*5,min_length=1,max_segments=2)
        for row in fixture['boundaries']:row['posterior_probability']=.5
        with patch('segmentlab.decision.infer',return_value=fixture):
            self.assertEqual(boundary_decision([0.]*5)['decision']['cuts'],[])
        for row in fixture['boundaries']:row['posterior_probability']=.75
        with patch('segmentlab.decision.infer',return_value=fixture):
            self.assertEqual(boundary_decision([0.]*5)['decision']['cuts'],[1])
    def test_audit_script(self):
        result=subprocess.run([sys.executable,'examples/decision_audit.py'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);obj=json.loads(result.stdout)
        self.assertEqual(len(obj['fixtures']),2)
        self.assertEqual(obj['fixtures']['map_not_boundary_optimal']['decision']['cuts'],[])
        self.assertFalse(obj['fixtures']['threshold_breaks_cap']['unconstrained_threshold']['admissible'])
