"""Exact finite-partition inference under an explicit conjugate Gaussian model."""
import math

NEG=float('-inf')


def lse(values):
    v=list(values)
    if not v or max(v)==NEG:return NEG
    m=max(v);return m+math.log(math.fsum(math.exp(z-m) for z in v))


def real(v,name,lo,hi):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError(f'{name} must be finite in[{lo},{hi}]')
    return float(v)


def integer(v,name,lo,hi):
    if type(v) is not int or not lo<=v<=hi:raise ValueError(f'{name} must be integer in[{lo},{hi}]')
    return v


def segment_score(values,noise_std,prior_mean,prior_std):
    """Normal observations with known sigma; independent N(m0,tau²) means."""
    mean=0.;m2=0.
    for n,y in enumerate(values,1):
        delta=y-mean;mean+=delta/n;m2+=delta*(y-mean)
    if not values:raise ValueError('empty segment')
    sigma2=noise_std**2;tau2=prior_std**2
    log_marginal=-n/2*math.log(2*math.pi*sigma2)-.5*math.log1p(n*tau2/sigma2)-m2/(2*sigma2)-n*(mean-prior_mean)**2/(2*(sigma2+n*tau2))
    variance=1/(1/tau2+n/sigma2)
    return {'log_marginal_likelihood':log_marginal,'posterior_mean':prior_mean+n*tau2/(sigma2+n*tau2)*(mean-prior_mean),'posterior_variance':variance}


def infer(data,noise_std=.5,prior_mean=0.,prior_std=3.,cut_probability=.08,min_length=2,max_segments=8):
    if not isinstance(data,list) or not 1<=len(data)<=120:raise ValueError('data requires1..120 observations')
    y=[real(x,'observation',-1e6,1e6) for x in data];n=len(y)
    sigma=real(noise_std,'noise_std',1e-4,1e4);mu=real(prior_mean,'prior_mean',-1e6,1e6);tau=real(prior_std,'prior_std',1e-4,1e4)
    p=real(cut_probability,'cut_probability',1e-6,1-1e-6);minimum=integer(min_length,'min_length',1,n);cap=integer(max_segments,'max_segments',1,12)
    cap=min(cap,n//minimum)
    scores={}
    for start in range(n):
        for end in range(start+minimum,n+1):scores[start,end]=segment_score(y[start:end],sigma,mu,tau)
    forward=[[NEG]*(n+1) for _ in range(cap+1)];backward=[[NEG]*(n+1) for _ in range(cap+1)]
    best=[[NEG]*(n+1) for _ in range(cap+1)];pointer={};forward[0][0]=0.;best[0][0]=0.;backward[0][n]=0.
    counts=[[0]*(n+1) for _ in range(cap+1)];counts[0][0]=1
    for k in range(1,cap+1):
        for end in range(k*minimum,n+1):
            terms=[];options=[]
            for start in range((k-1)*minimum,end-minimum+1):
                score=scores[start,end]['log_marginal_likelihood']
                terms.append(forward[k-1][start]+score);options.append((best[k-1][start]+score,start))
                counts[k][end]+=counts[k-1][start]
            forward[k][end]=lse(terms)
            value,start=max(options,key=lambda v:v[0]);best[k][end]=value;pointer[k,end]=start
        for start in range(n-k*minimum,-1,-1):
            backward[k][start]=lse(scores[start,end]['log_marginal_likelihood']+backward[k-1][end] for end in range(start+minimum,n-(k-1)*minimum+1))
    log_prior=[NEG]+[(k-1)*math.log(p)+(n-k)*math.log1p(-p) for k in range(1,cap+1)]
    prior_z=lse(math.log(counts[k][n])+log_prior[k] for k in range(1,cap+1) if counts[k][n])
    joint_z=lse(forward[k][n]+log_prior[k] for k in range(1,cap+1))
    count_probs=[{'segments':k,'admissible_partition_count':counts[k][n],
                  'prior_probability':math.exp(math.log(counts[k][n])+log_prior[k]-prior_z),
                  'posterior_probability':math.exp(forward[k][n]+log_prior[k]-joint_z)} for k in range(1,cap+1)]
    intervals=[];coverage=[[] for _ in y];boundary=[0.]*(n+1)
    for (start,end),score in scores.items():
        terms=[]
        for left in range(cap):
            if forward[left][start]==NEG:continue
            for right in range(cap-left):
                if backward[right][end]==NEG:continue
                terms.append(forward[left][start]+score['log_marginal_likelihood']+backward[right][end]+log_prior[left+right+1])
        probability=math.exp(lse(terms)-joint_z) if terms else 0.
        interval={'start':start,'end':end,'posterior_probability':probability,**score};intervals.append(interval)
        if end<n:boundary[end]+=probability
        for t in range(start,end):coverage[t].append(interval)
    pointwise=[]
    for t,items in enumerate(coverage):
        total=math.fsum(s['posterior_probability'] for s in items)
        mean=math.fsum(s['posterior_probability']*s['posterior_mean'] for s in items)/total
        variance=math.fsum(s['posterior_probability']*(s['posterior_variance']+(s['posterior_mean']-mean)**2) for s in items)/total
        pointwise.append({'index':t,'latent_mean_posterior_mean':mean,'latent_mean_posterior_variance':variance,'segment_coverage_mass':total})
    k=max(range(1,cap+1),key=lambda k:best[k][n]+log_prior[k]);end=n;parts=[];kk=k
    while kk:
        start=pointer[kk,end];parts.append({'start':start,'end':end,**scores[start,end]});end=start;kk-=1
    parts.reverse();expected_segments=math.fsum(k['segments']*k['posterior_probability'] for k in count_probs)
    if max(abs(s['segment_coverage_mass']-1) for s in pointwise)>1e-8 or abs(math.fsum(k['posterior_probability'] for k in count_probs)-1)>1e-8:raise ValueError('partition probabilities numerically unresolved; rescale data or noise')
    return {'data':y,'settings':{'noise_std':sigma,'prior_mean':mu,'prior_std':tau,'cut_probability':p,'min_length':minimum,'max_segments_requested':max_segments,'max_segments_effective':cap},
            'log_prior_admissibility_mass':prior_z,'log_marginal_likelihood':joint_z-prior_z,
            'partition_count':sum(counts[k][n] for k in range(1,cap+1)),
            'segment_count_posterior':count_probs,'boundaries':[{'cut_before_index':j,'posterior_probability':boundary[j]} for j in range(1,n)],
            'intervals':intervals,'pointwise':pointwise,
            'map':{'segments':parts,'cuts':[s['end'] for s in parts[:-1]],'posterior_probability':math.exp(best[k][n]+log_prior[k]-joint_z)},
            'audits':{'expected_segments':expected_segments,'expected_cuts_from_boundaries':math.fsum(boundary),'interval_mass_sum':math.fsum(s['posterior_probability'] for s in intervals),'count_probability_sum':math.fsum(k['posterior_probability'] for k in count_probs),'max_coverage_mass_error':max(abs(s['segment_coverage_mass']-1) for s in pointwise)},
            'semantics':'offline exact finite-partition sums in floating point under independent Gaussian observations and segment means; Bernoulli gap-cut prior conditioned on min_length/max_segments admissibility; boundaries are cuts before zero-based index, intervals half-open; MAP is one complete partition, marginal peaks are not a joint partition; latent-mean variance excludes future observation noise; not an online detector, causal attribution or calibrated real-data guarantee'}
