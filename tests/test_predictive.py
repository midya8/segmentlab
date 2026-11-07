import copy
import json
import math
import subprocess
import sys
import unittest
from statistics import NormalDist
from segmentlab import posterior_predictive
from segmentlab.core import segment_score,lse
from test_core import enumerate_partitions


class PredictiveTests(unittest.TestCase):
    def test_full_partition_reference_all_indices(self):
        for n,minimum,cap in [(1,1,1),(5,1,4),(7,2,3),(8,3,2)]:
            y=[(-1)**i*(i+1)/5 for i in range(n)]
            r=posterior_predictive(y,[[i,.17] for i in range(n)],min_length=minimum,max_segments=cap)
            parts=enumerate_partitions(y,minimum=minimum,cap=cap);z=lse(p[3] for p in parts)
            for out in r['queries']:
                mixture=[]
                for _,segments,_,score in parts:
                    a,b=next((a,b) for a,b in segments if a<=out['index']<b)
                    s=segment_score(y[a:b],.5,0.,3.)
                    mixture.append((math.exp(score-z),NormalDist(s['posterior_mean'],math.sqrt(.25+s['posterior_variance']))))
                density=sum(w*d.pdf(.17) for w,d in mixture);cdf=sum(w*d.cdf(.17) for w,d in mixture)
                self.assertAlmostEqual(out['log_density'],math.log(density),12);self.assertAlmostEqual(out['cdf'],cdf,12)
                self.assertAlmostEqual(out['cdf']+out['survival'],1.,13)
                m=r['mixtures'][out['index']]
                for q in m['quantiles']:self.assertAlmostEqual(sum(w*d.cdf(q['value']) for w,d in mixture),q['probability'],12)
    def test_single_partition_closed_form(self):
        r=posterior_predictive([.2,.4,.3],[[1,2.]],max_segments=1)
        s=segment_score([.2,.4,.3],.5,0.,3.);d=NormalDist(s['posterior_mean'],math.sqrt(.25+s['posterior_variance']))
        m=r['mixtures'][0];self.assertEqual(len(m['components']),1)
        self.assertAlmostEqual(m['variance'],d.variance,14)
        self.assertAlmostEqual(r['queries'][0]['log_density'],math.log(d.pdf(2.)),13)
        for q in m['quantiles']:self.assertAlmostEqual(q['value'],d.inv_cdf(q['probability']),13)
    def test_predictive_adds_noise_to_latent_variance(self):
        r=posterior_predictive([-.8,-1.,.7,1.,.6],[[i,0.] for i in range(5)],min_length=1,max_segments=4)
        for m,p in zip(r['mixtures'],r['inference']['pointwise']):
            self.assertAlmostEqual(m['mean'],p['latent_mean_posterior_mean'],13)
            self.assertAlmostEqual(m['variance'],.25+p['latent_mean_posterior_variance'],13)
            self.assertAlmostEqual(sum(c['weight'] for c in m['components']),1.,13)
    def test_stable_survival_does_not_subtract_rounded_cdf(self):
        r=posterior_predictive([0.],[[0,12.]],noise_std=1.,prior_std=1.,min_length=1)
        q=r['queries'][0];self.assertEqual(q['cdf'],1.);self.assertGreater(q['survival'],0.)
        self.assertAlmostEqual(q['survival'],.5*math.erfc(12/math.sqrt(3)),29)
    def test_finite_log_density_when_density_underflows(self):
        r=posterior_predictive([0.],[[0,1e6]],noise_std=1e-4,prior_std=1e-4,min_length=1)
        q=r['queries'][0];self.assertTrue(math.isfinite(q['log_density']));self.assertEqual(math.exp(q['log_density']),0.)
        self.assertEqual(q['survival'],0.);json.dumps(r,allow_nan=False)
    def test_offsets_sign_and_tail_symmetry(self):
        y=[-.8,0.,.9,-.3];a=posterior_predictive(y,[[1,.6]],min_length=1)
        b=posterior_predictive([v+7 for v in y],[[1,7.6]],prior_mean=7,min_length=1)
        c=posterior_predictive([-v for v in y],[[1,-.6]],min_length=1)
        self.assertAlmostEqual(a['queries'][0]['log_density'],b['queries'][0]['log_density'],12)
        self.assertAlmostEqual(a['queries'][0]['cdf'],c['queries'][0]['survival'],13)
        for x,z in zip(a['mixtures'][0]['quantiles'],reversed(c['mixtures'][0]['quantiles'])):self.assertAlmostEqual(x['value'],-z['value'],12)
    def test_query_order_repeat_indices_unique_mixtures(self):
        r=posterior_predictive([0.,1.,2.],[[2,3.],[0,0.],[2,-1.]],min_length=1)
        self.assertEqual([q['index'] for q in r['queries']],[2,0,2]);self.assertEqual([m['index'] for m in r['mixtures']],[0,2])
    def test_no_mutation_or_alias(self):
        y=[0.,1.,2.];queries=[[1,.7]];before=copy.deepcopy((y,queries))
        r=posterior_predictive(y,queries,min_length=1);self.assertEqual((y,queries),before)
        r['mixtures'][0]['components'][0]['mean']=99;r['inference']['data'][0]=88;self.assertEqual((y,queries),before)
    def test_strict_query_bounds_and_model_validation(self):
        for q in [[],[[0,0.]]*65,[[0,0.],[0,0.]],[[True,0.]],[[0,True]],[[-1,0.]],[[2,0.]],[[.5,0.]],[[0,float('nan')]],[[0,1e6+1]],[(0,0.)],[[0]],{},None]:
            with self.assertRaises(ValueError):posterior_predictive([0.,1.],q,min_length=1)
        with self.assertRaises(ValueError):posterior_predictive([0.,1.],[[0,0.]],noise_std=0,min_length=1)
    def test_extreme_model_quantile_residuals(self):
        for y,sigma,mu,tau in [([1e6,1e6-.1,1e6-.2],.01,1e6,.1),([-1e6,1e6],1e-4,0.,1e4)]:
            r=posterior_predictive(y,[[i,v] for i,v in enumerate(y)],noise_std=sigma,prior_mean=mu,prior_std=tau,min_length=1,max_segments=3)
            json.dumps(r,allow_nan=False)
            for m in r['mixtures']:
                values=[q['value'] for q in m['quantiles']];self.assertEqual(values,sorted(values))
                for q in m['quantiles']:self.assertLess(abs(q['cdf_minus_probability']),1e-5)
    def test_cli(self):
        args=[sys.executable,'-m','segmentlab','predictive','examples/sequence.json','--queries','[[2,0.0],[3,1.0]]']
        p=subprocess.run(args,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(len(json.loads(p.stdout)['queries']),2)
        for value in ['bad','[[true,0]]','[]']:
            p=subprocess.run(args[:-1]+[value],capture_output=True,text=True);self.assertEqual(p.returncode,2);self.assertNotIn('Traceback',p.stderr)
    def test_moment_gaussian_is_not_mixture(self):
        for p,expected in [(1e-5,[]),(1e-4,[2,3])]:
            r=posterior_predictive([0.,0.,4.,0.,0.],[[2,2.]],noise_std=.5,cut_probability=p,min_length=1,max_segments=3)
            self.assertEqual(r['inference']['map']['cuts'],expected)
            self.assertLess(r['queries'][0]['log_density_minus_moment_gaussian'],-1.5)
    def test_maximum_queries_and_indices(self):
        r=posterior_predictive([0.]*8,[[i//8,(i%8)/10] for i in range(64)],min_length=2,max_segments=3)
        self.assertEqual(len(r['queries']),64);self.assertEqual(len(r['mixtures']),8)
    def test_reproducible_audit(self):
        args=[sys.executable,'examples/predictive_audit.py']
        a=subprocess.run(args,capture_output=True,text=True);b=subprocess.run(args,capture_output=True,text=True)
        self.assertEqual(a.returncode,0,a.stderr);self.assertEqual(a.stdout,b.stdout)
        self.assertEqual(len(json.loads(a.stdout)['cases']),2)
