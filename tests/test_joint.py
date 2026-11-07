import copy
import itertools
import json
import math
import subprocess
import sys
import unittest
from segmentlab import infer,joint_cuts
from segmentlab.core import NEG,lse
from segmentlab.joint import _constrained_log_mass
from test_core import enumerate_partitions


class JointTests(unittest.TestCase):
    def test_every_pair_against_exhaustive_partitions(self):
        for n,minimum,cap in [(4,1,4),(6,1,3),(7,2,3),(8,3,2)]:
            data=[(-1)**i*(i+1)/5 for i in range(n)];pairs=[list(p) for p in itertools.combinations(range(1,n),2)]
            result=joint_cuts(data,pairs,min_length=minimum,max_segments=cap)
            parts=enumerate_partitions(data,minimum=minimum,cap=cap);z=lse(p[3] for p in parts)
            for row in result['pairs']:
                a=row['first_cut'];b=row['second_cut']
                la=lse(p[3] for p in parts if a in p[0]);lb=lse(p[3] for p in parts if b in p[0]);lj=lse(p[3] for p in parts if a in p[0] and b in p[0])
                for key,value in [('first_probability',la),('second_probability',lb),('joint_probability',lj)]:
                    self.assertAlmostEqual(row[key],0 if value==NEG else math.exp(value-z),12)
                self.assertEqual(row['joint_structurally_impossible'],lj==NEG)
                for key,condition in [('second_given_first',la),('first_given_second',lb)]:
                    if condition==NEG:self.assertIsNone(row[key])
                    else:self.assertAlmostEqual(row[key],0 if lj==NEG else math.exp(lj-condition),12)
    def test_required_cuts_do_not_forbid_intervening_cuts(self):
        data=[-.5,.3,-.2,.7];result=joint_cuts(data,[[1,3]],min_length=1,max_segments=4)
        parts=enumerate_partitions(data,minimum=1,cap=4);z=lse(p[3] for p in parts)
        included=[p for p in parts if 1 in p[0] and 3 in p[0]];self.assertEqual(len(included),2)
        only_three=next(p for p in included if p[0]==[1,3])
        self.assertGreater(result['pairs'][0]['joint_probability'],math.exp(only_three[3]-z))
    def test_exclusion_under_count_cap_not_product_of_marginals(self):
        result=joint_cuts([0.]*6,[[2,4]],min_length=1,max_segments=2)['pairs'][0]
        self.assertGreater(result['first_probability'],0);self.assertGreater(result['second_probability'],0)
        self.assertEqual(result['joint_probability'],0);self.assertTrue(result['joint_structurally_impossible'])
        self.assertLess(result['joint_minus_independence_product'],0);self.assertEqual(result['second_given_first'],0)
    def test_impossible_condition_is_null_not_zero(self):
        row=joint_cuts([0.]*7,[[1,3]],min_length=3,max_segments=2)['pairs'][0]
        self.assertTrue(row['first_condition_impossible']);self.assertIsNone(row['second_given_first'])
        self.assertIsNone(row['log_first_probability']);self.assertEqual(row['first_probability'],0)
        self.assertGreater(row['second_probability'],0);self.assertEqual(row['first_given_second'],0)
    def test_structural_impossibility_distinct_from_underflow(self):
        row=joint_cuts([-100.,-100.,100.,100.],[[1,3]],noise_std=.01,prior_std=100.,min_length=1,max_segments=3)['pairs'][0]
        self.assertEqual(row['joint_probability'],0.)
        self.assertFalse(row['joint_structurally_impossible']);self.assertIsNotNone(row['log_joint_probability'])
        self.assertTrue(math.isfinite(row['log_joint_probability']))
        json.dumps(row,allow_nan=False)
    def test_marginals_match_and_frechet_bounds(self):
        result=joint_cuts([-.7,-.4,.6,.9],[[1,2],[1,3],[2,3]],min_length=1,max_segments=4)
        marginal={b['cut_before_index']:b['posterior_probability'] for b in result['inference']['boundaries']}
        for row in result['pairs']:
            a=row['first_probability'];b=row['second_probability'];j=row['joint_probability']
            self.assertAlmostEqual(a,marginal[row['first_cut']],13);self.assertAlmostEqual(b,marginal[row['second_cut']],13)
            self.assertGreaterEqual(j+1e-12,max(0,a+b-1));self.assertLessEqual(j,min(a,b)+1e-12)
            self.assertAlmostEqual(j,a*row['second_given_first'],13)
            self.assertAlmostEqual(row['joint_minus_independence_product'],j-a*b,14)
    def test_no_constraints_original_evidence(self):
        result=infer([-.2,.5,1.,.3],min_length=1,max_segments=4)
        self.assertAlmostEqual(_constrained_log_mass(result,()),result['log_marginal_likelihood']+result['log_prior_admissibility_mass'],13)
    def test_reuse_unique_single_constraints_and_preserve_order(self):
        pairs=[[2,3],[1,3],[1,2]];result=joint_cuts([0.]*4,pairs,min_length=1)
        self.assertEqual([[r['first_cut'],r['second_cut']] for r in result['pairs']],pairs)
        self.assertEqual(result['query_work'],{'unique_single_cut_constraints':3,'pair_constraints':3,'constrained_dp_evaluations':6})
    def test_no_mutation_and_inference_unchanged(self):
        data=[0.,.1,1.,1.1];pairs=[[1,3]];before=copy.deepcopy((data,pairs))
        result=joint_cuts(data,pairs,min_length=1)
        self.assertEqual((data,pairs),before);self.assertEqual(result['inference'],infer(data,min_length=1))
    def test_validation(self):
        for pairs in [[],[[1,2]]*65,[[1,2],[1,2]],[[2,1]],[[1,1]],[[0,2]],[[1,4]],[[True,2]],[[1,2.]],[(1,2)],'bad',[[1]],[[1,2,3]]]:
            with self.assertRaises((ValueError,TypeError)):joint_cuts([0.]*4,pairs,min_length=1)
        with self.assertRaises(ValueError):joint_cuts([0.],[[1,2]])
    def test_maximum_n_and_pair_count(self):
        pairs=[list(p) for p in itertools.combinations(range(1,13),2)][:64]
        result=joint_cuts([0.]*120,pairs,min_length=60,max_segments=12)
        self.assertEqual(len(result['pairs']),64);json.dumps(result,allow_nan=False)
    def test_cli(self):
        command=[sys.executable,'-m','segmentlab','joint','examples/sequence.json','--pairs','[[1,3],[2,4]]']
        result=subprocess.run(command,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(len(json.loads(result.stdout)['pairs']),2)
        for value in ['[[1,1]]','not-json','{}']:
            result=subprocess.run(command[:-1]+[value],capture_output=True,text=True);self.assertEqual(result.returncode,2)
    def test_audit_script_retains_all_failure_directions(self):
        result=subprocess.run([sys.executable,'examples/joint_audit.py'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);obj=json.loads(result.stdout)
        self.assertEqual(len(obj['fixtures']),4)
        self.assertEqual(obj['fixtures']['count_cap_exclusion']['pairs'][0]['joint_probability'],0)
        self.assertFalse(obj['fixtures']['underflow_not_impossible']['pairs'][0]['joint_structurally_impossible'])
        self.assertIsNone(obj['fixtures']['impossible_condition']['pairs'][0]['second_given_first'])
