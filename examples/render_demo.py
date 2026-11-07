"""Reproducible SVG of all observations, posterior means and boundary marginals."""
import argparse
import html
import json
from pathlib import Path


def render(v):
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="860" viewBox="0 0 1120 860" role="img" aria-labelledby="title desc">','<title id="title">SegmentLab offline changepoint audit</title>','<desc id="desc">Known shift and noise-misspecification examples. Separate observation and boundary probability panels, all points shown; posterior means are model conditional.</desc>','<rect width="1120" height="860" fill="white"/>']
    def text(x,y,s,size=15,weight=400):out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="#251f21">{html.escape(s)}</text>')
    text(28,35,'SegmentLab | posterior confidence depends on the noise model',25)
    text(28,64,'Offline exact finite-partition inference. Synthetic development examples, not event detection or real-data calibration.')
    arms=[('Known shifts',v['known_shift']['result'],v['known_shift']['truth_mean']),('Outlier, assumed sigma = .15',v['noise_assumption_failure']['low_assumed_noise'],v['noise_assumption_failure']['true_mean']),('Same outlier, assumed sigma = 2',v['noise_assumption_failure']['high_assumed_noise'],v['noise_assumption_failure']['true_mean'])]
    for idx,(title,r,truth) in enumerate(arms):
        x=65+idx*365;w=300;n=len(r['data']);y=135;h=270
        text(x,102,title,18,600)
        for z in [-3,-1,1,3,5]:
            yy=y+h-(z+3)/8*h;out.append(f'<path d="M{x} {yy}h{w}" stroke="#eae9ea"/>');text(x-25,yy+4,str(z),12)
        def pos(t,z):return x+t/(n-1)*w,y+h-(z+3)/8*h
        for vals,color,dash in [(truth,'#999396','4 4'),([p['latent_mean_posterior_mean'] for p in r['pointwise']],'#73a89a','')]:
            points=' '.join(f'{pos(t,z)[0]:.3f},{pos(t,z)[1]:.3f}' for t,z in enumerate(vals));out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-dasharray="{dash}"/>')
        for t,z in enumerate(r['data']):
            px,py=pos(t,z);out.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="2.8" fill="#251f21"/>')
        for t in [0,n-1]:text(x+t/(n-1)*w-5,y+h+23,str(t),13)
        text(x+65,450,'Observation index (zero-based)',13)
        text(x,486,'Posterior cut probability',17,600)
        py=515;ph=140
        for probability in [0,.5,1]:
            yy=py+ph-probability*ph;out.append(f'<path d="M{x} {yy}h{w}" stroke="#eae9ea"/>');text(x-25,yy+4,str(probability),12)
        for b in r['boundaries']:
            bx=x+b['cut_before_index']/(n-1)*w;top=py+ph-b['posterior_probability']*ph
            out.append(f'<path d="M{bx:.3f} {py+ph}V{top:.3f}" stroke="#73a89a" stroke-width="3"/>')
        text(x,685,'Cut before index: 1 through '+str(n-1),13)
        text(x,718,'MAP cuts: '+str(r['map']['cuts']),16,600)
        text(x,745,f'MAP partition probability: {r["map"]["posterior_probability"]:.4f}',14)
        text(x,770,f'Expected segment count: {r["audits"]["expected_segments"]:.4f}',14)
    text(28,809,'Dots: all observations. Solid teal: posterior latent-mean average. Dashed gray: generating mean (not inferred).',14)
    text(28,836,'Shared observation y[-3,5] and probability[0,1] scales. Marginal cut peaks are not a joint partition. No hidden thinning.',14)
    out.append('</svg>');return '\n'.join(out)+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--check',action='store_true');a=p.parse_args();s=render(json.loads(Path(a.input).read_text()));path=Path(__file__).resolve().parents[1]/'assets/partition-audit.svg'
    if a.check:
        if path.read_text()!=s:raise SystemExit('SVG differs from demo')
    else:path.write_text(s)
