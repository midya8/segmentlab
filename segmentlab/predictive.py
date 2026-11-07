"""Marginal replicate observations at existing design indices, not forecasts."""
import math
from statistics import NormalDist
from .core import infer, integer, real, lse


def _cdf(components,value,survival=False):
    sign=1 if survival else -1
    return math.fsum(c['weight']*.5*math.erfc(sign*(value-c['mean'])/math.sqrt(2*c['variance'])) for c in components)


def _quantile(components,probability):
    """Component quantiles bracket the mixture quantile. No grid truncation."""
    z=NormalDist().inv_cdf(probability)
    values=[c['mean']+z*math.sqrt(c['variance']) for c in components]
    lo=math.nextafter(min(values),-math.inf);hi=math.nextafter(max(values),math.inf)
    iterations=0
    for _ in range(80):
        mid=lo+(hi-lo)/2
        if mid==lo or mid==hi:break
        iterations+=1
        if _cdf(components,mid)<probability:lo=mid
        else:hi=mid
    value=lo+(hi-lo)/2;residual=_cdf(components,value)-probability
    if abs(residual)>1e-5:raise ValueError('predictive quantile numerically unresolved; rescale inputs')
    return {'probability':probability,'value':value,'cdf_minus_probability':residual,
            'final_bracket_width':hi-lo,'bisection_iterations':iterations}


def posterior_predictive(data,queries,**settings):
    """One to64 unique [index,value] queries. Fixed .05/.5/.95 marginal quantiles.

    A new observation shares the original index's partition and latent segment
    mean, with fresh independent noise of the supplied sigma. Fit uses all y.
    Returned marginal mixtures cannot be multiplied into a joint replicate law.
    """
    if not isinstance(data,list) or not 1<=len(data)<=120:raise ValueError('data requires1..120 observations')
    if not isinstance(queries,list) or not 1<=len(queries)<=64:raise ValueError('queries requires1..64 unique[index,value] pairs')
    validated=[]
    for query in queries:
        if not isinstance(query,list) or len(query)!=2:raise ValueError('query requires[index,value]')
        index=integer(query[0],'query index',0,len(data)-1);value=real(query[1],'query value',-1e6,1e6)
        if (index,value) in validated:raise ValueError('duplicate query')
        validated.append((index,value))
    result=infer(data,**settings);sigma2=result['settings']['noise_std']**2;indices={};output=[]
    for index,value in validated:
        if index not in indices:
            intervals=[s for s in result['intervals'] if s['start']<=index<s['end'] and s['posterior_probability']>0]
            mass=math.fsum(s['posterior_probability'] for s in intervals)
            components=[{'start':s['start'],'end':s['end'],'weight':s['posterior_probability']/mass,
                         'mean':s['posterior_mean'],'variance':s['posterior_variance']+sigma2} for s in intervals]
            mean=math.fsum(c['weight']*c['mean'] for c in components)
            variance=math.fsum(c['weight']*(c['variance']+(c['mean']-mean)**2) for c in components)
            indices[index]={'index':index,'raw_coverage_mass':mass,'components':components,'mean':mean,'variance':variance,
                            'quantiles':[_quantile(components,q) for q in (.05,.5,.95)]}
        mixture=indices[index];components=mixture['components'];mean=mixture['mean'];variance=mixture['variance']
        log_density=lse(math.log(c['weight'])-.5*(math.log(2*math.pi*c['variance'])+(value-c['mean'])**2/c['variance']) for c in components)
        gaussian_log=-.5*(math.log(2*math.pi*variance)+(value-mean)**2/variance)
        output.append({'index':index,'value':value,'log_density':log_density,'cdf':_cdf(components,value),
                       'survival':_cdf(components,value,True),'moment_matched_gaussian_log_density':gaussian_log,
                       'log_density_minus_moment_gaussian':log_density-gaussian_log})
    return {'inference':result,'mixtures':[indices[i] for i in sorted(indices)],'queries':output,
            'semantics':'same-design posterior predictive independent replicate at an existing index, conditional on all observed data; mixture over covering intervals integrates partition and segment-mean uncertainty, adds supplied observation-noise variance; fixed5/50/95 percent quantiles are marginal equal-tail summaries, not simultaneous bands; log density is per data unit, not event probability; moment Gaussian is a comparison, not replacement mixture; neither forecast, leave-one-out score, anomaly p-value nor calibrated coverage guarantee; returned marginals do not define joint replicate independence; floating point, interval masses may underflow, CDF/survival may round to0/1, quantile residuals reported'}
