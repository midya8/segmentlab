import random
from .core import infer


def demo():
    rng=random.Random(19)
    truth=[-2.]*12+[2.]*12+[-1.]*12
    data=[m+rng.gauss(0,.35) for m in truth]
    fitted=infer(data,noise_std=.35,prior_std=3.,cut_probability=.06,min_length=3,max_segments=6)
    assert fitted['map']['cuts']==[12,24]
    probabilities={b['cut_before_index']:b['posterior_probability'] for b in fitted['boundaries']}
    assert min(probabilities[12],probabilities[24])>.9
    # Same observed sequence, two incompatible assumed noise scales.
    # No true mean changes: one isolated outlier in an otherwise zero sequence.
    outlier=[0.]*12+[4.]+[0.]*11
    low=infer(outlier,noise_std=.15,prior_std=3.,cut_probability=.05,min_length=1,max_segments=6)
    high=infer(outlier,noise_std=2.,prior_std=3.,cut_probability=.05,min_length=1,max_segments=6)
    assert len(low['map']['cuts'])>=2 and not high['map']['cuts']
    return {'known_shift':{'truth_mean':truth,'true_cuts':[12,24],'result':fitted},
            'noise_assumption_failure':{'true_mean':[0.]*24,'low_assumed_noise':low,'high_assumed_noise':high},
            'protocol':'seed19 selected development illustration,36 observations, three known mean blocks; outlier counterexample deterministic; known sigma/prior/cut settings supplied, not learned or held-out benchmark; posterior confidence is conditional on a model, not a guarantee about events'}
