"""Independent backward draws under SegmentLab's finite Gaussian model.

Not MCMC: count and starts use exact finite sums in floating point. Standard
library pseudorandom and Gaussian draws have finite precision, not a symbolic
or calibrated real-data guarantee.
"""
import math
import random
from .core import infer, integer, lse, NEG


def _prepare(result):
    n=len(result['data']);settings=result['settings']
    minimum=settings['min_length'];cap=settings['max_segments_effective']
    scores={(s['start'],s['end']):s for s in result['intervals']}
    forward=[[NEG]*(n+1) for _ in range(cap+1)];forward[0][0]=0.
    for k in range(1,cap+1):
        for end in range(k*minimum,n+1):
            forward[k][end]=lse(forward[k-1][start]+scores[start,end]['log_marginal_likelihood']
                                for start in range((k-1)*minimum,end-minimum+1))
    p=settings['cut_probability']
    count_logs=[(k,forward[k][n]+(k-1)*math.log(p)+(n-k)*math.log1p(-p)) for k in range(1,cap+1)]
    return scores,forward,count_logs


def _start_options(scores,forward,k,end,minimum):
    return [(start,forward[k-1][start]+scores[start,end]['log_marginal_likelihood'])
            for start in range((k-1)*minimum,end-minimum+1)
            if forward[k-1][start]!=NEG]


def _probabilities(options):
    """Normalize positive finite-log mass, retaining finite-log underflow as0."""
    finite=[(key,value) for key,value in options if value!=NEG]
    if not finite:raise ValueError('no admissible sampling choice')
    maximum=max(value for _,value in finite)
    weights=[(key,math.exp(value-maximum)) for key,value in finite]
    total=math.fsum(weight for _,weight in weights)
    return [(key,weight/total) for key,weight in weights]


def _pick(options,uniform):
    if not 0<=uniform<1:raise ValueError('uniform must be in[0,1)')
    probabilities=_probabilities(options);cumulative=0.;last=None
    for key,probability in probabilities:
        if probability==0:continue
        last=key;cumulative+=probability
        if uniform<cumulative:return key
    # A rounded cumulative total may be just below1. Never select zero mass.
    return last


def sample_posterior(data,draws=100,seed=0,**settings):
    """Draw whole partitions, then their conditionally independent means.

    `draws`1..10000, integer seed0..2**32-1. Model settings match infer. The
    returned `inference` is unchanged, and no global random state is touched.
    Partition probability integrates means; it is not a sampled path density.
    """
    draws=integer(draws,'draws',1,10000);seed=integer(seed,'seed',0,2**32-1)
    result=infer(data,**settings);n=len(data);minimum=result['settings']['min_length']
    scores,forward,count_logs=_prepare(result);log_z=lse(value for _,value in count_logs)
    rng=random.Random(seed);output=[];p=result['settings']['cut_probability']
    for _ in range(draws):
        count=_pick(count_logs,rng.random());k=count;end=n;bounds=[]
        while k:
            start=_pick(_start_options(scores,forward,k,end,minimum),rng.random())
            bounds.append((start,end));end=start;k-=1
        bounds.reverse();segments=[];latent=[]
        log_mass=(count-1)*math.log(p)+(n-count)*math.log1p(-p)-log_z
        for start,end in bounds:
            score=scores[start,end];log_mass+=score['log_marginal_likelihood']
            mean=rng.gauss(score['posterior_mean'],math.sqrt(score['posterior_variance']))
            segments.append({'start':start,'end':end,'sampled_mean':mean,
                             'conditional_mean':score['posterior_mean'],
                             'conditional_variance':score['posterior_variance']})
            latent.extend([mean]*(end-start))
        output.append({'cuts':[end for _,end in bounds[:-1]],'segments':segments,
                       'latent_mean':latent,'log_partition_posterior_probability':log_mass,
                       'partition_posterior_probability':math.exp(log_mass)})
    return {'inference':result,'seed':seed,'draw_count':draws,'draws':output,
            'semantics':'independent pseudorandom complete-partition posterior draws and conditionally independent Gaussian segment means; no MCMC, burn-in or thresholded marginal cuts; finite floating-point weights and RNG can omit tiny masses; partition probabilities integrate means, not path density; latent means exclude future observation noise; no real-data calibration or misspecification guarantee'}
