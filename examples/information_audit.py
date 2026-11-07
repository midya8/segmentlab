"""Selected fixed-model uncertainty examples, not calibration scores."""
import json
import pathlib
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from segmentlab import partition_information


def audit():
    fixtures=[('count_cap_dependence',[0.]*6,dict(min_length=1,max_segments=2)),
              ('four_point',[-.7,-.4,.6,.9],dict(min_length=1,max_segments=4)),
              ('outlier_low_noise',[0.]*12+[4.]+[0.]*11,dict(noise_std=.15,cut_probability=.05,min_length=1,max_segments=6)),
              ('outlier_high_noise',[0.]*12+[4.]+[0.]*11,dict(noise_std=2.,cut_probability=.05,min_length=1,max_segments=6))]
    output={}
    for name,data,settings in fixtures:
        result=partition_information(data,**settings);r=result['inference']
        output[name]={'data':data,'settings':r['settings'],'partition_count':r['partition_count'],
                      'map':r['map'],'information':result['information']}
    assert output['count_cap_dependence']['information']['sum_binary_cut_entropies']>output['count_cap_dependence']['information']['posterior_partition_entropy']
    assert output['outlier_low_noise']['map']['cuts']==[12,13]
    assert output['outlier_high_noise']['map']['cuts']==[]
    return {'fixtures':output,'semantics':'all discrete information quantities in nats under each supplied model; outlier contamination has no generating-mean shift; confidence/information is not model correctness; different noise settings define different models, not a statistical-quality ranking; no continuous entropy, causal or calibration claim'}


if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))
