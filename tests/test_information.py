import copy
import itertools
import json
import math
import subprocess
import sys
import unittest
from segmentlab import infer,partition_information
from segmentlab.core import lse
from segmentlab.information import _mixture
from test_core import enumerate_partitions


class InformationTests(unittest.TestCase):
    def test_full_entropy_and_kl_against_exhaustive(self):
        for n,minimum,cap,p in [(1,1,1,.08),(6,1,4,.08),(7,2,3,.3),(8,3,2,.8)]:
            y=[(-1)**i*(i+1)/5 for i in range(n)]
            info=partition_information(y,min_length=minimum,max_segments=cap,cut_probability=p)['information']
            parts=enumerate_partitions(y,minimum=minimum,cap=cap,p=p);z=lse(row[3] for row in parts);pz=lse(row[2] for row in parts)
            h=-math.fsum(math.exp(row[3]-z)*(row[3]-z) for row in parts)
            hp=-math.fsum(math.exp(row[2]-pz)*(row[2]-pz) for row in parts)
            kl=math.fsum(math.exp(row[3]-z)*((row[3]-z)-(row[2]-pz)) for row in parts)
            self.assertAlmostEqual(info['posterior_partition_entropy'],h,12)
            self.assertAlmostEqual(info['conditioned_prior_partition_entropy'],hp,12)
            self.assertAlmostEqual(info['posterior_to_conditioned_prior_kl'],kl,12)
    def test_entropy_chain_rule_and_uniform_within_prior_count(self):
        result=partition_information([0.,.2,.3,.7,.8,.9],min_length=1,max_segments=4);info=result['information']
        self.assertAlmostEqual(info['posterior_partition_entropy'],info['posterior_count_entropy']+info['expected_partition_entropy_given_count'],14)
        for row in info['conditional_partition_entropy_by_count']:
            count=next(r['admissible_partition_count'] for r in result['inference']['segment_count_posterior'] if r['segments']==row['segments'])
            self.assertLessEqual(row['entropy_nats'],math.log(count)+1e-12)
    def test_forced_partition_zero_information(self):
        info=partition_information([.2,.4,.3],min_length=2,max_segments=12)['information']
        for key in ['posterior_partition_entropy','posterior_count_entropy','conditioned_prior_partition_entropy','posterior_to_conditioned_prior_kl','sum_binary_cut_entropies','cut_dependence_total_correlation']:
            self.assertAlmostEqual(info[key],0,13)
    def test_total_correlation_is_full_joint_gap(self):
        result=partition_information([0.]*6,min_length=1,max_segments=2);info=result['information']
        self.assertGreater(info['cut_dependence_total_correlation'],0)
        self.assertAlmostEqual(info['sum_binary_cut_entropies']-info['posterior_partition_entropy'],info['cut_dependence_total_correlation'],14)
        parts=enumerate_partitions([0.]*6,minimum=1,cap=2);z=lse(row[3] for row in parts)
        marginal={row['cut_before_index']:row['posterior_probability'] for row in result['inference']['boundaries']}
        kl=0.
        for cuts,_,_,score in parts:
            log_product=sum(math.log(q if index in cuts else 1-q) for index,q in marginal.items())
            kl+=math.exp(score-z)*(score-z-log_product)
        self.assertAlmostEqual(kl,info['cut_dependence_total_correlation'],12)
    def test_conditional_count_entropies_against_exhaustive(self):
        data=[-.3,.2,.9,-.4,.1];result=partition_information(data,min_length=1,max_segments=4)
        parts=enumerate_partitions(data,minimum=1,cap=4)
        for row in result['information']['conditional_partition_entropy_by_count']:
            selected=[p for p in parts if len(p[1])==row['segments']];z=lse(p[3] for p in selected)
            expected=-sum(math.exp(p[3]-z)*(p[3]-z) for p in selected)
            self.assertAlmostEqual(row['entropy_nats'],expected,12)
    def test_mixture_recursion_does_not_subtract_large_log_scores(self):
        mass,h=_mixture([(1e16,0.),(1e16,0.)]);self.assertAlmostEqual(h,math.log(2),14)
        mass,h=_mixture([(-1e16,2.),(-1e16,3.)]);self.assertAlmostEqual(h,2.5+math.log(2),14)
    def test_large_offsets_extremes_and_maximum_n(self):
        for data,settings in [([1e6,1e6-.1,1e6-.2],dict(prior_mean=1e6,noise_std=.01,prior_std=.1,min_length=1,max_segments=3)),([-1e6,1e6],dict(noise_std=1e-4,prior_std=1e4,min_length=1,max_segments=2)),([0.]*120,dict(min_length=60,max_segments=12))]:
            result=partition_information(data,**settings);json.dumps(result,allow_nan=False)
            self.assertGreaterEqual(result['information']['posterior_partition_entropy'],0)
    def test_translation_sign_symmetry(self):
        data=[-.3,.4,1.2,-.8]
        a=partition_information(data,min_length=1,max_segments=3)['information']
        for values,mu in [([-v for v in data],0),([v+7 for v in data],7)]:
            b=partition_information(values,prior_mean=mu,min_length=1,max_segments=3)['information']
            for key in ['posterior_partition_entropy','posterior_to_conditioned_prior_kl','cut_dependence_total_correlation']:self.assertAlmostEqual(a[key],b[key],12)
    def test_no_mutation_and_original_inference(self):
        data=[0.,.3,.9];before=data[:];result=partition_information(data,min_length=1)
        self.assertEqual(data,before);self.assertEqual(result['inference'],infer(data,min_length=1))
    def test_validation(self):
        for data,settings in [([],{}),([True],{}),([0.,1.],dict(noise_std=0)),([0.,1.],dict(max_segments=13))]:
            with self.assertRaises((ValueError,TypeError)):partition_information(data,**settings)
    def test_cli(self):
        result=subprocess.run([sys.executable,'-m','segmentlab','information','examples/sequence.json'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(result.stdout)['information']['units'],'nats')
        result=subprocess.run([sys.executable,'-m','segmentlab','information','README.md'],capture_output=True,text=True);self.assertEqual(result.returncode,2)
    def test_audit_script_preserves_failure_example(self):
        result=subprocess.run([sys.executable,'examples/information_audit.py'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);obj=json.loads(result.stdout)
        self.assertEqual(len(obj['fixtures']),4)
        self.assertEqual(obj['fixtures']['outlier_low_noise']['map']['cuts'],[12,13])
        self.assertEqual(obj['fixtures']['outlier_high_noise']['map']['cuts'],[])
