"""Selected same-design replicate mixture failure of moment-Gaussian replacement."""
import json
import pathlib
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from segmentlab import posterior_predictive


def audit():
    output=[]
    for p in (1e-5,1e-4):
        r=posterior_predictive([0.,0.,4.,0.,0.],[[2,2.]],noise_std=.5,prior_mean=0.,prior_std=3.,cut_probability=p,min_length=1,max_segments=3)
        m=r['mixtures'][0];q=r['queries'][0];latent=r['inference']['pointwise'][2]
        assert abs(m['variance']-latent['latent_mean_posterior_variance']-.25)<1e-12
        assert q['log_density_minus_moment_gaussian']<-1.5
        assert max(abs(x['cdf_minus_probability']) for x in m['quantiles'])<1e-12
        output.append({'cut_probability':p,'map_cuts':r['inference']['map']['cuts'],'latent_variance':latent['latent_mean_posterior_variance'],
                       'mixture':m,'query':q})
    return {'cases':output,'provenance':'Selected during development by a grid of sigma in[.4,.5,.6,.7,.8,1] and cut p in[1e-6,1e-5,.0001,.001,.01,.1] on fixed five-point outlier data; two contrasting priors retained, not held-out performance or a quality ranking.',
            'interpretation':'Same-design replicates conditional on all observations. Moment Gaussian matches mean/variance but not mixture shape. Changing p changes the model. Neither forecast, LOO evaluation, anomaly p-value nor calibration claim.'}


if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))
