"""Selected development comparisons of decision loss, not observed accuracy."""
import json
import pathlib
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from segmentlab import boundary_decision


def audit():
    fixtures=[('map_not_boundary_optimal',[.79,.94,.91,-.11,-.57,.34,-.75],1.,1.,dict(min_length=1,max_segments=3)),
              ('threshold_breaks_cap',[0.]*6,.0001,1.,dict(min_length=1,max_segments=2))]
    output={}
    for name,data,false,missed,settings in fixtures:
        result=boundary_decision(data,false,missed,**settings);inference=result.pop('inference')
        output[name]={'data':data,'settings':inference['settings'],'partition_count':inference['partition_count'],
                      'map_partition_probability':inference['map']['posterior_probability'],**result}
    assert output['map_not_boundary_optimal']['decision']['cuts']==[]
    assert output['map_not_boundary_optimal']['map_comparison']['cuts']==[3]
    assert output['map_not_boundary_optimal']['decision']['expected_loss']<output['map_not_boundary_optimal']['map_comparison']['expected_loss']
    assert not output['threshold_breaks_cap']['unconstrained_threshold']['admissible']
    return {'fixtures':output,'provenance':'seven-point example selected during development by searching seed71 uniform[-1,1] sequences for a MAP/boundary-loss decision difference; six-zero example chosen to expose a count-cap violation; neither is held-out performance',
            'semantics':'model-posterior expected loss, not observed real-world mistakes; MAP and additive-boundary decisions answer different loss questions; cheaper threshold risk outside the feasible set is not an admissible solution; costs are supplied relative loss units, not money; no quality ranking or calibration claim'}


if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))
