"""Full discrete partition entropy, not a sum of binary cut entropies."""
import math
from .core import infer, lse, NEG


def _mixture(terms):
    """Disjoint prefix alternatives: log mass, normalized entropy recursion."""
    terms=[(log_mass,entropy) for log_mass,entropy in terms if log_mass!=NEG]
    if not terms:return NEG,0.
    maximum=max(log_mass for log_mass,_ in terms)
    weights=[math.exp(log_mass-maximum) for log_mass,_ in terms];total=math.fsum(weights)
    probabilities=[weight/total for weight in weights]
    entropy=math.fsum(probability*(child_entropy-math.log(probability))
                      for probability,(_,child_entropy) in zip(probabilities,terms) if probability)
    return maximum+math.log(total),entropy


def _entropy(probabilities):
    return -math.fsum(p*math.log(p) for p in probabilities if p)


def partition_information(data,**settings):
    """Discrete partition uncertainties in nats under the supplied model.

    Segment means are integrated out, not continuous entropy variables here.
    Conditional entropy recurrences avoid subtracting large log likelihoods.
    Posterior-to-prior KL uses the explicitly conditioned admissible prior.
    """
    result=infer(data,**settings);n=len(data);model=result['settings']
    minimum=model['min_length'];cap=model['max_segments_effective'];p=model['cut_probability']
    scores={(row['start'],row['end']):row['log_marginal_likelihood'] for row in result['intervals']}
    forward=[[NEG]*(n+1) for _ in range(cap+1)];entropy=[[0.]*(n+1) for _ in range(cap+1)];forward[0][0]=0.
    for k in range(1,cap+1):
        for end in range(k*minimum,n+1):
            forward[k][end],entropy[k][end]=_mixture(
                (forward[k-1][start]+scores[start,end],entropy[k-1][start])
                for start in range((k-1)*minimum,end-minimum+1))
    counts=result['segment_count_posterior'];post=[row['posterior_probability'] for row in counts]
    prior=[row['prior_probability'] for row in counts]
    # infer audits normalization; normalize its floating-point weights explicitly.
    post_total=math.fsum(post);prior_total=math.fsum(prior)
    post=[v/post_total for v in post];prior=[v/prior_total for v in prior]
    count_entropy=_entropy(post)
    conditional_mean=math.fsum(weight*entropy[row['segments']][n] for weight,row in zip(post,counts))
    posterior_entropy=count_entropy+conditional_mean
    log_prior=[(row['segments']-1)*math.log(p)+(n-row['segments'])*math.log1p(-p) for row in counts]
    expected_log_prior=math.fsum(weight*value for weight,value in zip(post,log_prior))-result['log_prior_admissibility_mass']
    prior_entropy=result['log_prior_admissibility_mass']-math.fsum(weight*value for weight,value in zip(prior,log_prior))
    raw_kl=-posterior_entropy-expected_log_prior
    binary=[]
    for row in result['boundaries']:
        q=row['posterior_probability']
        if not -1e-8<=q<=1+1e-8:raise ValueError('boundary probability numerically unresolved')
        q=min(1.,max(0.,q));binary.append({'cut_before_index':row['cut_before_index'],'probability_used':q,'entropy_nats':_entropy([q,1-q])})
    sum_binary=math.fsum(row['entropy_nats'] for row in binary);raw_dependence=sum_binary-posterior_entropy
    for name,value in [('KL',raw_kl),('dependence gap',raw_dependence),('prior entropy',prior_entropy)]:
        if value< -1e-8:raise ValueError(f'{name} numerically unresolved')
    return {'inference':result,'information':{'units':'nats','posterior_partition_entropy':posterior_entropy,
            'posterior_count_entropy':count_entropy,'expected_partition_entropy_given_count':conditional_mean,
            'conditional_partition_entropy_by_count':[{'segments':row['segments'],'entropy_nats':entropy[row['segments']][n]} for row in counts],
            'conditioned_prior_partition_entropy':max(0.,prior_entropy),
            'posterior_expected_log_conditioned_prior_mass':expected_log_prior,
            'posterior_to_conditioned_prior_kl':max(0.,raw_kl),
            'sum_binary_cut_entropies':sum_binary,'cut_dependence_total_correlation':max(0.,raw_dependence),
            'binary_cuts':binary,'raw_kl_before_roundoff_clamp':raw_kl,'raw_dependence_before_roundoff_clamp':raw_dependence,
            'count_probability_totals_before_normalizing':{'posterior':post_total,'prior':prior_total}},
            'semantics':'discrete full-partition entropy and posterior-to-admissible-conditioned-prior KL in nats; segment means integrated out, not differential entropy; H(partition)=H(count)+E H(partition|count); cut total correlation=sum binary marginal entropies minus full joint entropy, not pairwise-only correlation or causality; higher information or lower entropy does not imply a correct model or calibrated cuts; finite-floating-point weights can underflow; raw values retained, negative roundoff up to1e-8 clamped, larger inconsistency fails'}
