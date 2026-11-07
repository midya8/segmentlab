import copy
import itertools
import json
import math
import subprocess
import sys
import unittest
from segmentlab.core import infer,segment_score,lse,NEG


def enumerate_partitions(y,sigma=.5,mu=0.,tau=3.,p=.08,minimum=2,cap=8):
    n=len(y);parts=[]
    for bits in itertools.product([False,True],repeat=n-1):
        cuts=[j+1 for j,b in enumerate(bits) if b];bounds=[0]+cuts+[n];segments=list(zip(bounds,bounds[1:]));k=len(segments)
        if k>cap or any(b-a<minimum for a,b in segments):continue
        prior=(k-1)*math.log(p)+(n-k)*math.log1p(-p)
        score=prior+sum(segment_score(y[a:b],sigma,mu,tau)['log_marginal_likelihood'] for a,b in segments)
        parts.append((cuts,segments,prior,score))
    return parts

class CoreTests(unittest.TestCase):
    def test_single_observation_predictive(self):
        y=[1.2];r=segment_score(y,.5,-.2,2.)
        expected=-.5*math.log(2*math.pi*4.25)-1.4**2/(2*4.25)
        self.assertAlmostEqual(r['log_marginal_likelihood'],expected,13)
        self.assertAlmostEqual(r['posterior_variance'],1/(.25+4),14)
        self.assertAlmostEqual(r['posterior_mean'],(-.2/4+1.2/.25)/(.25+4),14)
    def test_sequential_predictive_independent_reference(self):
        ys=[.2,-.4,1.3,1.2];mu=.7;v=1.3**2;sigma=.8;logp=0.
        for y in ys:
            predictive=v+sigma*sigma;logp+=-.5*math.log(2*math.pi*predictive)-(y-mu)**2/(2*predictive)
            gain=v/predictive;mu+=gain*(y-mu);v*=1-gain
        r=segment_score(ys,sigma,.7,1.3)
        self.assertAlmostEqual(logp,r['log_marginal_likelihood'],13);self.assertAlmostEqual(mu,r['posterior_mean'],13);self.assertAlmostEqual(v,r['posterior_variance'],13)
    def test_exhaustive_partition_evidence_marginals_map(self):
        for n,minimum,cap in [(6,1,4),(7,2,3),(8,3,2),(1,1,1)]:
            y=[(-1)**i*(i+1)/5 for i in range(n)];r=infer(y,min_length=minimum,max_segments=cap)
            parts=enumerate_partitions(y,minimum=minimum,cap=cap);z=lse(p[3] for p in parts);priorz=lse(p[2] for p in parts)
            self.assertEqual(r['partition_count'],len(parts));self.assertAlmostEqual(r['log_marginal_likelihood'],z-priorz,12)
            self.assertAlmostEqual(r['log_prior_admissibility_mass'],priorz,13)
            best=max(parts,key=lambda x:x[3]);self.assertEqual(r['map']['cuts'],best[0]);self.assertAlmostEqual(r['map']['posterior_probability'],math.exp(best[3]-z),13)
            for b in r['boundaries']:self.assertAlmostEqual(b['posterior_probability'],sum(math.exp(p[3]-z) for p in parts if b['cut_before_index'] in p[0]),12)
            for s in r['intervals']:self.assertAlmostEqual(s['posterior_probability'],sum(math.exp(p[3]-z) for p in parts if (s['start'],s['end']) in p[1]),12)
            for k in r['segment_count_posterior']:self.assertAlmostEqual(k['posterior_probability'],sum(math.exp(p[3]-z) for p in parts if len(p[1])==k['segments']),12)
    def test_pointwise_variance_full_partition_reference(self):
        y=[-.8,-1.,.7,1.,.6];r=infer(y,min_length=1,max_segments=4);parts=enumerate_partitions(y,minimum=1,cap=4);z=lse(p[3] for p in parts)
        for t,out in enumerate(r['pointwise']):
            mix=[]
            for _,segments,_,score in parts:
                a,b=next((a,b) for a,b in segments if a<=t<b);s=segment_score(y[a:b],.5,0.,3.);mix.append((math.exp(score-z),s))
            mean=sum(p*s['posterior_mean'] for p,s in mix);var=sum(p*(s['posterior_variance']+(s['posterior_mean']-mean)**2) for p,s in mix)
            self.assertAlmostEqual(out['latent_mean_posterior_mean'],mean,12);self.assertAlmostEqual(out['latent_mean_posterior_variance'],var,12)
    def test_conditioned_prior_changes_evidence(self):
        y=[0.]*6;restricted=infer(y,min_length=3,max_segments=2);unrestricted=infer(y,min_length=1,max_segments=6)
        self.assertAlmostEqual(unrestricted['log_prior_admissibility_mass'],0.,13);self.assertLess(restricted['log_prior_admissibility_mass'],0.)
        self.assertEqual(restricted['partition_count'],2)
        for row in restricted['segment_count_posterior']:self.assertEqual(row['admissible_partition_count'],1)
    def test_single_partition_forced(self):
        r=infer([.2,.4,.3],min_length=2,max_segments=12)
        self.assertEqual(r['map']['cuts'],[]);self.assertEqual(r['partition_count'],1);self.assertEqual(r['map']['posterior_probability'],1.)
        self.assertAlmostEqual(r['log_marginal_likelihood'],segment_score([.2,.4,.3],.5,0.,3.)['log_marginal_likelihood'],13)
    def test_sum_rules(self):
        r=infer([-.5,.2,1.,1.3,.9,-.1,.4],min_length=1,max_segments=5)
        a=r['audits'];self.assertAlmostEqual(a['expected_segments'],a['interval_mass_sum'],12);self.assertAlmostEqual(a['expected_segments']-1,a['expected_cuts_from_boundaries'],12);self.assertAlmostEqual(a['count_probability_sum'],1.,13);self.assertLess(a['max_coverage_mass_error'],1e-12)
    def test_translation_and_sign_symmetry(self):
        y=[-.3,.4,1.2,-.8];a=infer(y,min_length=1,max_segments=3)
        b=infer([v+7 for v in y],prior_mean=7,min_length=1,max_segments=3);c=infer([-v for v in y],min_length=1,max_segments=3)
        self.assertEqual(a['map']['cuts'],b['map']['cuts']);self.assertEqual(a['map']['cuts'],c['map']['cuts'])
        self.assertAlmostEqual(a['log_marginal_likelihood'],b['log_marginal_likelihood'],12);self.assertAlmostEqual(a['log_marginal_likelihood'],c['log_marginal_likelihood'],12)
        for x,z in zip(a['pointwise'],b['pointwise']):self.assertAlmostEqual(x['latent_mean_posterior_mean']+7,z['latent_mean_posterior_mean'],12)
    def test_impossible_intervals_have_zero_mass(self):
        r=infer([0.]*7,min_length=3,max_segments=2)
        self.assertTrue(all(s['posterior_probability']==0 for s in r['intervals'] if s['start'] in [1,2] or s['end'] in [5,6] and s['start']>0))
        self.assertTrue(all(b['posterior_probability']==0 for b in r['boundaries'] if b['cut_before_index'] in [1,2,5,6]))
    def test_stability_large_offset_and_extremes(self):
        r=infer([1e6,1e6-.1,1e6-.2,1e6],noise_std=.01,prior_mean=1e6,prior_std=.1,min_length=1,max_segments=4)
        json.dumps(r,allow_nan=False);self.assertLess(r['audits']['max_coverage_mass_error'],1e-10)
        r=infer([-1e6,1e6],noise_std=1e-4,prior_std=1e4,min_length=1,max_segments=2);json.dumps(r,allow_nan=False);self.assertEqual(r['map']['cuts'],[1])
    def test_no_mutation_or_alias(self):
        y=[0.,1.,2.];before=y[:];r=infer(y,min_length=1);self.assertEqual(y,before);r['data'][0]=100;self.assertEqual(y,before)
    def test_bounds(self):
        for kw in [{'data':[]},{'data':[0.]*121},{'data':[True]},{'data':[float('nan')]},{'data':[1e6+1]},{'noise_std':0.},{'prior_std':float('inf')},{'min_length':True},{'min_length':4},{'max_segments':13},{'cut_probability':0.},{'cut_probability':1.}]:
            v={'data':[0.,1.,2.]};v.update(kw)
            with self.assertRaises((ValueError,TypeError)):infer(**v)
    def test_lse(self):
        self.assertEqual(lse([]),NEG);self.assertEqual(lse([NEG,NEG]),NEG);self.assertAlmostEqual(lse([-1000.,-1001.]),-1000+math.log1p(math.exp(-1)),13)
    def test_cli(self):
        r=subprocess.run([sys.executable,'-m','segmentlab','infer','examples/sequence.json'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout)['map']['cuts'],[3])
        r=subprocess.run([sys.executable,'-m','segmentlab','infer','README.md'],capture_output=True,text=True);self.assertEqual(r.returncode,2)
