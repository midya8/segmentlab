"""A fixed-seed development audit, not a model-calibration experiment."""
import json
import math
import pathlib
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from segmentlab import sample_posterior


def audit():
    result=sample_posterior([-.7,-.4,.6,.9],draws=10000,seed=19,min_length=1,max_segments=4)
    draws=result['draws'];exact=result['inference'];total=len(draws)
    counts=[{'segments':row['segments'],'exact_probability':row['posterior_probability'],
             'sample_frequency':sum(len(d['segments'])==row['segments'] for d in draws)/total}
            for row in exact['segment_count_posterior']]
    cuts=[{'cut_before_index':row['cut_before_index'],'exact_probability':row['posterior_probability'],
           'sample_frequency':sum(row['cut_before_index'] in d['cuts'] for d in draws)/total}
          for row in exact['boundaries']]
    points=[]
    for row in exact['pointwise']:
        values=[d['latent_mean'][row['index']] for d in draws];mean=math.fsum(values)/total
        variance=math.fsum((x-mean)**2 for x in values)/total
        points.append({'index':row['index'],'exact_mean':row['latent_mean_posterior_mean'],
                       'sample_mean':mean,'exact_variance':row['latent_mean_posterior_variance'],
                       'sample_variance_divisor_draw_count':variance})
    errors={'max_count_frequency_error':max(abs(r['sample_frequency']-r['exact_probability']) for r in counts),
            'max_cut_frequency_error':max(abs(r['sample_frequency']-r['exact_probability']) for r in cuts),
            'max_latent_mean_error':max(abs(r['sample_mean']-r['exact_mean']) for r in points),
            'max_latent_variance_error':max(abs(r['sample_variance_divisor_draw_count']-r['exact_variance']) for r in points)}
    if max(errors.values())>=.02:raise ValueError('fixed-seed development audit regressed')
    return {'data':exact['data'],'settings':exact['settings'],'seed':result['seed'],'draw_count':total,
            'partition_count':exact['partition_count'],'counts':counts,'boundaries':cuts,'points':points,
            'absolute_errors':errors,'first_three_draws':draws[:3],
            'semantics':'all10000 draws contribute to every summary; first3 paths retained as examples, not selected for fit; finite fixed-seed regression comparisons to exact model-conditional sums, not independent validation data, confidence bounds or real-data calibration'}


if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))
