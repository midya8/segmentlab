import copy
import itertools
import json
import math
import random
import subprocess
import sys
import unittest
from unittest.mock import patch
from segmentlab import infer, sample_posterior
from segmentlab.core import lse, NEG
from segmentlab.sampling import _prepare, _start_options, _probabilities, _pick
from test_core import enumerate_partitions


class SamplingTests(unittest.TestCase):
    def test_all_backward_path_probabilities_against_enumeration(self):
        for n,minimum,cap in [(1,1,1),(6,1,4),(7,2,3),(8,3,2),(6,1,1)]:
            y=[(-1)**i*(i+1)/5 for i in range(n)]
            result=infer(y,min_length=minimum,max_segments=cap)
            scores,forward,counts=_prepare(result);count_p=dict(_probabilities(counts))
            parts=enumerate_partitions(y,minimum=minimum,cap=cap);z=lse(row[3] for row in parts)
            total=0.
            for cuts,bounds,prior,score in parts:
                k=len(bounds);probability=count_p[k];end=n
                for start,stop in reversed(bounds):
                    self.assertEqual(stop,end)
                    options=dict(_probabilities(_start_options(scores,forward,k,end,minimum)))
                    probability*=options[start];end=start;k-=1
                self.assertEqual(end,0);self.assertEqual(k,0)
                self.assertAlmostEqual(probability,math.exp(score-z),13);total+=probability
            self.assertAlmostEqual(total,1.,13)
    def test_categorical_thresholds_zero_mass_and_underflow(self):
        options=[('impossible',NEG),('tiny',-10000),('a',0.),('b',math.log(2))]
        self.assertEqual(_pick(options,0.),'a')
        self.assertEqual(_pick(options,.3),'a');self.assertEqual(_pick(options,.34),'b')
        self.assertEqual(_pick(options,math.nextafter(1.,0.)),'b')
        self.assertEqual(_pick([('only',-1e20)],.9),'only')
        for uniform in [-.1,1.,float('nan')]:
            with self.assertRaises(ValueError):_pick(options,uniform)
        with self.assertRaises(ValueError):_pick([('none',NEG)],0.)
    def test_categorical_every_bin_midpoint(self):
        options=[(i,-i*.7) for i in range(30)];left=0.
        for key,probability in _probabilities(options):
            self.assertEqual(_pick(options,left+probability/2),key);left+=probability
        self.assertAlmostEqual(left,1.,14)
    def test_repeatable_and_global_rng_untouched(self):
        random.seed(445);before=random.getstate()
        a=sample_posterior([0.,.2,1.,1.1],draws=20,seed=23,min_length=1)
        self.assertEqual(random.getstate(),before)
        self.assertEqual(a,sample_posterior([0.,.2,1.,1.1],draws=20,seed=23,min_length=1))
        self.assertNotEqual(a['draws'],sample_posterior([0.,.2,1.,1.1],draws=20,seed=24,min_length=1)['draws'])
    def test_inference_unchanged_and_input_no_mutation(self):
        data=[0.,.2,1.,1.1];before=data[:]
        a=sample_posterior(data,draws=3,seed=1,min_length=1,max_segments=3)
        self.assertEqual(a['inference'],infer(data,min_length=1,max_segments=3));self.assertEqual(data,before)
        a['draws'][0]['segments'][0]['sampled_mean']=9
        self.assertNotEqual(a['draws'][1]['segments'][0]['sampled_mean'],9)
        a['draws'][0]['latent_mean'][0]=8;self.assertNotEqual(a['draws'][1]['latent_mean'][0],8)
    def test_whole_partition_constraints_probability_and_path(self):
        data=[-.2,.1,.2,1.,1.2,.9,-1.,-.8];settings=dict(min_length=2,max_segments=3)
        result=sample_posterior(data,draws=250,seed=192,**settings)
        parts=enumerate_partitions(data,minimum=2,cap=3);z=lse(row[3] for row in parts)
        reference={tuple(row[0]):row[3]-z for row in parts}
        for draw in result['draws']:
            self.assertIn(tuple(draw['cuts']),reference)
            self.assertAlmostEqual(draw['log_partition_posterior_probability'],reference[tuple(draw['cuts'])],12)
            self.assertAlmostEqual(draw['partition_posterior_probability'],math.exp(reference[tuple(draw['cuts'])]),13)
            cursor=0;values=[]
            for segment in draw['segments']:
                self.assertEqual(segment['start'],cursor);self.assertGreaterEqual(segment['end']-cursor,2)
                cursor=segment['end'];values.extend([segment['sampled_mean']]*(cursor-segment['start']))
            self.assertEqual(cursor,len(data));self.assertEqual(values,draw['latent_mean'])
            self.assertEqual(draw['cuts'],[s['end'] for s in draw['segments'][:-1]])
    def test_gaussian_conditional_parameters_and_independent_calls(self):
        observed=[]
        def fake(self,mean,std):
            observed.append((mean,std));return mean+std*1.25
        with patch('random.Random.gauss',fake):
            result=sample_posterior([-.2,.1,2.,2.1],draws=15,seed=5,min_length=1)
        expected=[]
        for draw in result['draws']:
            for segment in draw['segments']:
                mean=segment['conditional_mean'];std=math.sqrt(segment['conditional_variance'])
                expected.append((mean,std));self.assertEqual(segment['sampled_mean'],mean+std*1.25)
        self.assertEqual(observed,expected)
    def test_single_partition_gaussian_moments(self):
        result=sample_posterior([.2,.4,.3],draws=10000,seed=91,min_length=2)
        score=result['inference']['map']['segments'][0];values=[d['latent_mean'][0] for d in result['draws']]
        mean=math.fsum(values)/len(values);var=math.fsum((x-mean)**2 for x in values)/len(values)
        self.assertLess(abs(mean-score['posterior_mean']),.015)
        self.assertLess(abs(var-score['posterior_variance']),.004)
        self.assertTrue(all(d['cuts']==[] and d['partition_posterior_probability']==1 for d in result['draws']))
    def test_empirical_joint_draws_match_exact_marginals_and_mixture_moments(self):
        result=sample_posterior([-.7,-.4,.6,.9],draws=10000,seed=19,min_length=1,max_segments=4)
        exact=result['inference'];draws=result['draws'];count=len(draws)
        for boundary in exact['boundaries']:
            frequency=sum(boundary['cut_before_index'] in d['cuts'] for d in draws)/count
            self.assertLess(abs(frequency-boundary['posterior_probability']),.02)
        for row in exact['segment_count_posterior']:
            frequency=sum(len(d['segments'])==row['segments'] for d in draws)/count
            self.assertLess(abs(frequency-row['posterior_probability']),.02)
        for point in exact['pointwise']:
            values=[d['latent_mean'][point['index']] for d in draws];mean=math.fsum(values)/count
            variance=math.fsum((x-mean)**2 for x in values)/count
            self.assertLess(abs(mean-point['latent_mean_posterior_mean']),.02)
            self.assertLess(abs(variance-point['latent_mean_posterior_variance']),.02)
    def test_one_observation_maximum_size_and_extreme_inputs(self):
        for data,settings in [([1.],dict(min_length=1)),([0.]*120,dict(min_length=60,max_segments=12)),([-1e6,1e6],dict(noise_std=1e-4,prior_std=1e4,min_length=1,max_segments=2))]:
            result=sample_posterior(data,draws=2,seed=2**32-1,**settings)
            json.dumps(result,allow_nan=False)
            for draw in result['draws']:self.assertEqual(len(draw['latent_mean']),len(data))
    def test_validation(self):
        for kw in [dict(draws=0),dict(draws=10001),dict(draws=True),dict(seed=-1),dict(seed=2**32),dict(seed=True),dict(seed=1.),dict(noise_std=0),dict(min_length=0)]:
            with self.assertRaises((ValueError,TypeError)):sample_posterior([0.,1.],**kw)
        with self.assertRaises(ValueError):sample_posterior([])
    def test_cli_sample(self):
        command=[sys.executable,'-m','segmentlab','sample','examples/sequence.json','--draws','3','--seed','23']
        result=subprocess.run(command,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
        obj=json.loads(result.stdout);self.assertEqual(obj['draw_count'],3);self.assertEqual(obj['seed'],23)
        self.assertEqual(len(obj['draws']),3)
        for args in [['sample','examples/sequence.json','--draws','0'],['sample','README.md']]:
            result=subprocess.run([sys.executable,'-m','segmentlab']+args,capture_output=True,text=True);self.assertEqual(result.returncode,2)
    def test_reproducible_audit_script(self):
        result=subprocess.run([sys.executable,'examples/sample_audit.py'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);audit=json.loads(result.stdout)
        self.assertEqual(audit['draw_count'],10000);self.assertEqual(audit['partition_count'],8)
        self.assertEqual(len(audit['points']),4);self.assertEqual(len(audit['first_three_draws']),3)
        self.assertLess(max(audit['absolute_errors'].values()),.02)
