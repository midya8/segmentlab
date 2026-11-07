"""Exact inclusion queries, not independent Bernoulli cut approximations."""
import math
from .core import infer, integer, lse, NEG


def _constrained_log_mass(result,required):
    """Unnormalized joint mass of partitions containing every required gap.

    Other cuts remain allowed. An interval crossing a required cut is forbidden;
    the ordinary capped prefix recurrence then sums every permitted partition.
    """
    n=len(result['data']);settings=result['settings'];minimum=settings['min_length']
    cap=settings['max_segments_effective'];p=settings['cut_probability']
    scores={(s['start'],s['end']):s['log_marginal_likelihood'] for s in result['intervals']
            if not any(s['start']<cut<s['end'] for cut in required)}
    forward=[[NEG]*(n+1) for _ in range(cap+1)];forward[0][0]=0.
    for k in range(1,cap+1):
        for end in range(k*minimum,n+1):
            forward[k][end]=lse(forward[k-1][start]+scores[start,end]
                                for start in range((k-1)*minimum,end-minimum+1)
                                if (start,end) in scores and forward[k-1][start]!=NEG)
    return lse(forward[k][n]+(k-1)*math.log(p)+(n-k)*math.log1p(-p) for k in range(1,cap+1))


def _log_ratio(numerator,denominator):
    if numerator==NEG:return None
    ratio=numerator-denominator
    if ratio>1e-8:raise ValueError('joint probability numerically unresolved')
    return min(0.,ratio)


def _probability(log_probability):
    return 0. if log_probability is None else math.exp(log_probability)


def joint_cuts(data,pairs,**settings):
    """Evaluate1..64 unique interior-gap pairs under the full partition law.

    Pairs must be sorted a<b with integer zero-based cut-before indices. Other
    cuts are allowed. Impossible events have log probability None and zero
    mass; a conditional on an impossible event is None, not zero. Finite log
    probabilities remain available if exponentiation underflows.
    """
    if not isinstance(data,list) or not 1<=len(data)<=120:raise ValueError('data requires1..120 observations')
    if not isinstance(pairs,list) or not 1<=len(pairs)<=64:raise ValueError('pairs requires1..64 unique[a,b] pairs')
    normalized=[]
    for pair in pairs:
        if not isinstance(pair,list) or len(pair)!=2:raise ValueError('pair requires[a,b]')
        a=integer(pair[0],'first cut',1,len(data)-1);b=integer(pair[1],'second cut',1,len(data)-1)
        if a>=b:raise ValueError('pair requires first cut < second cut')
        if (a,b) in normalized:raise ValueError('duplicate pair')
        normalized.append((a,b))
    result=infer(data,**settings);joint_z=result['log_marginal_likelihood']+result['log_prior_admissibility_mass']
    single={cut:_constrained_log_mass(result,(cut,)) for cut in sorted({cut for pair in normalized for cut in pair})}
    output=[]
    for a,b in normalized:
        mass=_constrained_log_mass(result,(a,b));log_joint=_log_ratio(mass,joint_z)
        log_a=_log_ratio(single[a],joint_z);log_b=_log_ratio(single[b],joint_z)
        probability=_probability(log_joint);pa=_probability(log_a);pb=_probability(log_b)
        conditional_b=_log_ratio(mass,single[a]) if single[a]!=NEG else None
        conditional_a=_log_ratio(mass,single[b]) if single[b]!=NEG else None
        output.append({'first_cut':a,'second_cut':b,'first_probability':pa,'second_probability':pb,
                       'log_first_probability':log_a,'log_second_probability':log_b,
                       'joint_probability':probability,'log_joint_probability':log_joint,
                       'independence_product':pa*pb,'joint_minus_independence_product':probability-pa*pb,
                       'second_given_first':None if single[a]==NEG else _probability(conditional_b),
                       'first_given_second':None if single[b]==NEG else _probability(conditional_a),
                       'log_second_given_first':conditional_b,'log_first_given_second':conditional_a,
                       'joint_structurally_impossible':mass==NEG,
                       'first_condition_impossible':single[a]==NEG,'second_condition_impossible':single[b]==NEG})
    return {'inference':result,'pairs':output,'query_work':{'unique_single_cut_constraints':len(single),
            'pair_constraints':len(output),'constrained_dp_evaluations':len(single)+len(output)},
            'semantics':'exact floating-point posterior cut inclusion queries under the original full partition law; both queried cuts required, other cuts allowed; same conditioned prior and evidence denominator; independence product is a comparison, not a replacement joint law; pairwise queries do not determine the complete law; None logs denote structurally impossible events, None conditionals denote impossible conditions; finite log values preserve underflowed masses; no causal, online or real-data calibration guarantee'}
