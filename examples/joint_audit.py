"""Fixed development examples: mutually exclusive cuts and finite-log mass."""
import json
import pathlib
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from segmentlab import joint_cuts


def audit():
    fixtures=[('count_cap_exclusion',[0.]*6,[[2,4]],dict(min_length=1,max_segments=2)),
              ('four_point_dependence',[-.7,-.4,.6,.9],[[1,2],[1,3],[2,3]],dict(min_length=1,max_segments=4)),
              ('underflow_not_impossible',[-100.,-100.,100.,100.],[[1,3]],dict(noise_std=.01,prior_std=100.,min_length=1,max_segments=3)),
              ('impossible_condition',[0.]*7,[[1,3]],dict(min_length=3,max_segments=2))]
    output={}
    for name,data,pairs,settings in fixtures:
        result=joint_cuts(data,pairs,**settings)
        output[name]={'data':data,'settings':result['inference']['settings'],
                      'partition_count':result['inference']['partition_count'],
                      'pairs':result['pairs'],'query_work':result['query_work']}
    excluded=output['count_cap_exclusion']['pairs'][0]
    assert excluded['joint_probability']==0 and excluded['independence_product']>0 and excluded['joint_structurally_impossible']
    underflow=output['underflow_not_impossible']['pairs'][0]
    assert underflow['joint_probability']==0 and underflow['log_joint_probability'] is not None and not underflow['joint_structurally_impossible']
    assert output['impossible_condition']['pairs'][0]['second_given_first'] is None
    assert output['four_point_dependence']['pairs'][0]['joint_probability']<output['four_point_dependence']['pairs'][0]['independence_product']
    return {'fixtures':output,'semantics':'exact model-conditional development examples; cap exclusion is structural, not an empirical discovery; underflow is not impossibility; independence product is not the actual joint probability; no causal or calibration claim'}


if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))
