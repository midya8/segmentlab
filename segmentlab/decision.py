"""Constrained posterior decisions under additive cut-inclusion loss."""
import math
from .core import infer,real


def _risk(cuts,probabilities,false_cut_cost,missed_cut_cost):
    selected=set(cuts)
    false=math.fsum(1-q for j,q in probabilities.items() if j in selected)
    missed=math.fsum(q for j,q in probabilities.items() if j not in selected)
    return {'expected_false_cuts':false,'expected_missed_cuts':missed,
            'expected_loss':false_cut_cost*false+missed_cut_cost*missed}


def boundary_decision(data,false_cut_cost=1.,missed_cut_cost=1.,**settings):
    """Minimize posterior expected boundary loss over admissible partitions.

    Costs are strictly positive supplied relative loss units, not money. Every
    candidate obeys infer's minimum length and count cap. Ties prefer fewer
    segments, then the lexicographically smallest cut list. Marginal boundary
    probabilities are enough for this additive loss despite dependent cuts.
    """
    false=real(false_cut_cost,'false_cut_cost',1e-6,1e6)
    missed=real(missed_cut_cost,'missed_cut_cost',1e-6,1e6)
    result=infer(data,**settings);n=len(data);minimum=result['settings']['min_length'];cap=result['settings']['max_segments_effective']
    probabilities={r['cut_before_index']:min(1.,max(0.,r['posterior_probability'])) for r in result['boundaries']}
    # Selecting j adds false*(1-q_j)-missed*q_j to the no-cut base loss.
    increments={j:false*(1-q)-missed*q for j,q in probabilities.items()}
    states={(0,0):(0.,())}
    for k in range(1,cap+1):
        for end in range(k*minimum,n+1):
            choices=[]
            for start in range((k-1)*minimum,end-minimum+1):
                previous=states.get((k-1,start))
                if previous is None:continue
                cost,cuts=previous
                if start:cost+=increments[start];cuts=cuts+(start,)
                choices.append((cost,cuts))
            if choices:states[k,end]=min(choices,key=lambda choice:(choice[0],choice[1]))
    value,count,cuts=min((states[k,n][0],k,states[k,n][1]) for k in range(1,cap+1))
    selected=list(cuts);bounds=[0]+selected+[n];map_cuts=result['map']['cuts']
    threshold=false/(false+missed)
    unconstrained=[j for j,q in probabilities.items() if q>threshold]
    ubounds=[0]+unconstrained+[n]
    valid=len(ubounds)-1<=cap and all(b-a>=minimum for a,b in zip(ubounds,ubounds[1:]))
    return {'inference':result,'loss':{'false_cut_cost':false,'missed_cut_cost':missed,'units':'supplied relative loss units, not money'},
            'decision':{'cuts':selected,'segments':[{'start':a,'end':b} for a,b in zip(bounds,bounds[1:])],
                        **_risk(selected,probabilities,false,missed)},
            'map_comparison':{'cuts':map_cuts,**_risk(map_cuts,probabilities,false,missed)},
            'unconstrained_threshold':{'probability_threshold':threshold,'cuts':unconstrained,'admissible':valid,
                                       **_risk(unconstrained,probabilities,false,missed)},
            'numerical_audit':{'no_cut_base_loss':missed*math.fsum(probabilities.values()),'selected_increment_sum':value,
                               'max_boundary_probability_clamp_change':max((abs(probabilities[r['cut_before_index']]-r['posterior_probability']) for r in result['boundaries']),default=0.)},
            'semantics':'minimizes model-posterior expected additive false-cut/missed-cut loss among min-length/count-capped partitions; exact matching of cut indices only, no tolerance window or location-distance loss; MAP instead minimizes whole-partition0/1 loss; threshold ignores constraints and is comparison only; ties use exact floating cost equality, then fewer segments and lexicographic cuts, no tolerance tie-merging; finite arithmetic may affect near ties; costs supplied, not learned; no real-world accuracy, causal or calibration guarantee'}
