"""Deterministic fixed-fixture SVG: joint paths are not marginal averages."""
import argparse
import html
import json
import math
from pathlib import Path


NS='http://www.w3.org/2000/svg'


def validate(v):
    # This is the committed sample_audit fixture, not a general plotting API.
    if v['data']!=[-.7,-.4,.6,.9] or v['seed']!=19 or v['draw_count']!=10000 or v['partition_count']!=8:
        raise ValueError('renderer requires the committed seed19 four-point10000-draw audit')
    expected={'noise_std':.5,'prior_mean':0.,'prior_std':3.,'cut_probability':.08,'min_length':1,'max_segments_requested':4,'max_segments_effective':4}
    if v['settings']!=expected:raise ValueError('renderer requires the documented model settings')
    if [r['segments'] for r in v['counts']]!=[1,2,3,4] or [r['cut_before_index'] for r in v['boundaries']]!=[1,2,3] or [r['index'] for r in v['points']]!=[0,1,2,3]:
        raise ValueError('incomplete audit rows')
    if len(v['first_three_draws'])!=3:raise ValueError('requires first three unselected draws')
    for kind in ['counts','boundaries']:
        for row in v[kind]:
            for field in ['exact_probability','sample_frequency']:
                if not math.isfinite(row[field]) or not 0<=row[field]<=1:raise ValueError('invalid probability')
    for row in v['points']:
        for key in ['exact_variance','sample_variance_divisor_draw_count']:
            if not math.isfinite(row[key]) or not 0<=row[key]<=.25:raise ValueError('variance outside fixed plot domain')
    for draw in v['first_three_draws']:
        if len(draw['latent_mean'])!=4 or any(not math.isfinite(z) or not -1.2<=z<=1.2 for z in draw['latent_mean']):raise ValueError('path outside fixed plot domain')
        cursor=0;values=[]
        for segment in draw['segments']:
            if segment['start']!=cursor or not cursor<segment['end']<=4:raise ValueError('invalid joint segment')
            values.extend([segment['sampled_mean']]*(segment['end']-cursor));cursor=segment['end']
        if cursor!=4 or values!=draw['latent_mean'] or draw['cuts']!=[s['end'] for s in draw['segments'][:-1]]:raise ValueError('path does not match joint segments')
    for row in v['points']:
        if not math.isfinite(row['exact_mean']) or not -1.2<=row['exact_mean']<=1.2:raise ValueError('mean outside fixed plot domain')


def render(v):
    validate(v)
    manifest={'fixture':{'data':v['data'],'settings':v['settings'],'seed':v['seed'],'draw_count':v['draw_count']},
              'domains':{'paths':[-1.2,1.2],'probability':[0,1],'variance':[0,.25]},
              'paths':v['first_three_draws'],'counts':v['counts'],'boundaries':v['boundaries'],'points':v['points']}
    out=[f'<svg xmlns="{NS}" width="1120" height="1010" viewBox="0 0 1120 1010" role="img" aria-labelledby="title desc">',
         '<title id="title">SegmentLab joint posterior draw audit</title>',
         '<desc id="desc">First three complete sampled latent paths versus exact pointwise mean; exact and10000-draw count and boundary probabilities and latent variances. Zero K4 visits do not imply zero posterior mass. All points shown, no credible bands.</desc>',
         '<metadata id="plotted-data">'+html.escape(json.dumps(manifest,sort_keys=True,allow_nan=False))+'</metadata>',
         '<rect width="1120" height="1010" fill="white"/>']
    def text(x,y,s,size=14,weight=400,color='#251f21',anchor='start'):
        out.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{html.escape(str(s))}</text>')
    def line(x1,y1,x2,y2,color='#eae9ea',width=1,dash=''):
        out.append(f'<path d="M{x1:.3f} {y1:.3f}L{x2:.3f} {y2:.3f}" fill="none" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>')
    def axes(x,y,w,h,lo,hi,ticks,labels):
        for value in ticks:
            yy=y+h-(value-lo)/(hi-lo)*h;line(x,yy,x+w,yy);text(x-12,yy+4,f'{value:g}',12,anchor='end')
        for px,label in labels:text(px,y+h+22,label,13,anchor='middle')
        return lambda value:y+h-(value-lo)/(hi-lo)*h
    def poly(points,color,width=2,dash='',series=''):
        out.append(f'<polyline data-series="{series}" points="'+ ' '.join(f'{x:.3f},{y:.3f}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>')
    def circle(x,y,color,series,index,value,radius=4):
        out.append(f'<circle data-series="{series}" data-index="{index}" data-value="{value:.17g}" cx="{x:.3f}" cy="{y:.3f}" r="{radius}" fill="{color}"/>')
    text(28,36,'SegmentLab | a sampled partition is not a marginal average',25,600)
    text(28,63,'Seed19; all10,000 draws; four observed values[-.7,-.4,.6,.9]; all8 admissible partitions.',15)
    text(28,86,'sigma.5, prior mean0 / std3, gap-cut p.08, min_length1, cap4. Offline model-conditional development audit.',14)
    x=80;y=165;w=430;h=230
    text(x,128,'A. First three complete latent-mean draws',18,600)
    py=axes(x,y,w,h,-1.2,1.2,[-1.2,-.6,0,.6,1.2],[(x+(i+.5)/4*w,str(i)) for i in range(4)])
    for j,(draw,color,dash) in enumerate(zip(v['first_three_draws'],['#585254','#999396','#251f21'],['','6 4','2 4'])):
        points=[]
        for segment in draw['segments']:
            yy=py(segment['sampled_mean']);points.extend([(x+segment['start']/4*w,yy),(x+segment['end']/4*w,yy)])
        poly(points,color,2,dash,'joint-draw-'+str(j+1))
    points=[(x+(i+.5)/4*w,py(row['exact_mean'])) for i,row in enumerate(v['points'])]
    poly(points,'#73a89a',3,series='exact-latent-mean')
    for i,(px,yy) in enumerate(points):circle(px,yy,'#73a89a','exact-latent-mean',i,v['points'][i]['exact_mean'])
    for i,z in enumerate(v['data']):circle(x+(i+.5)/4*w,py(z),'#251f21','observation',i,z,3)
    text(x,440,'Observation index; path steps occur at segment boundaries.',13)
    text(x,465,'Teal: exact mean. Black dots: observations.',13,600)
    text(x,487,'Gray paths: draws1 solid,2 dashed,3 dotted. Not bands.',13)
    text(x,509,'First3 in generation order, not selected for fit.',13)
    x=625;y=165;w=430;h=230
    text(x,128,'B. Segment-count probability',18,600)
    py=axes(x,y,w,h,0,1,[0,.5,1],[(x+(i+.5)/4*w,str(i+1)) for i in range(4)])
    for i,row in enumerate(v['counts']):
        center=x+(i+.5)/4*w
        for offset,key,color,series in [(-15,'exact_probability','#73a89a','count-exact'),(15,'sample_frequency','#251f21','count-frequency')]:
            top=py(row[key]);out.append(f'<rect data-series="{series}" data-index="{i+1}" data-value="{row[key]:.17g}" x="{center+offset-9:.3f}" y="{top:.3f}" width="18" height="{y+h-top:.3f}" fill="{color}"/>')
        text(center,443,f"{row['exact_probability']:.6f}",12,600,'#73a89a','middle')
        text(center,464,f"{row['sample_frequency']:.4f}",12,400,'#251f21','middle')
    text(x,487,'K (segments). Teal: exact mass; black: draw frequency.',13)
    text(x,509,'K4: positive exact mass, zero visits. Not a zero posterior.',13,600)
    x=80;y=615;w=430;h=220
    text(x,575,'C. Cut probability, all interior gaps',18,600)
    py=axes(x,y,w,h,0,1,[0,.5,1],[(x+(i+.5)/3*w,str(i+1)) for i in range(3)])
    for i,row in enumerate(v['boundaries']):
        center=x+(i+.5)/3*w
        for offset,key,color,series in [(-13,'exact_probability','#73a89a','cut-exact'),(13,'sample_frequency','#251f21','cut-frequency')]:
            line(center+offset,y+h,center+offset,py(row[key]),color,3)
            circle(center+offset,py(row[key]),color,series,i+1,row[key])
        text(center,880,f"{row['exact_probability']:.6f} / {row['sample_frequency']:.4f}",12,anchor='middle')
    text(x,905,'Cut before zero-based index; exact / frequency below.',13)
    text(x,927,'Shared probability scale[0,1] with count panel.',13)
    x=625;y=615;w=430;h=220
    text(x,575,'D. Pointwise latent-mean variance',18,600)
    py=axes(x,y,w,h,0,.25,[0,.125,.25],[(x+(i+.5)/4*w,str(i)) for i in range(4)])
    for key,color,series,dash in [('exact_variance','#73a89a','variance-exact',''),('sample_variance_divisor_draw_count','#251f21','variance-sample','5 4')]:
        points=[]
        for i,row in enumerate(v['points']):
            px=x+(i+.5)/4*w;yy=py(row[key]);points.append((px,yy));circle(px,yy,color,series,i,row[key])
        poly(points,color,2,dash,series)
    for i,row in enumerate(v['points']):
        center=x+(i+.5)/4*w;text(center,880,f"{row['exact_variance']:.4f}",12,600,'#73a89a','middle');text(center,901,f"{row['sample_variance_divisor_draw_count']:.4f}",12,anchor='middle')
    text(x,927,'Exact teal / sampled black (divisor10,000). No sigma².',13)
    text(28,972,'All indices, counts and gaps shown. Numerical sums/RNG have finite precision; rare mass may not be sampled.',14)
    text(28,994,'Joint paths exclude future observation noise. No credible bands, event labels, online forecast or real-data calibration.',14)
    out.append('</svg>');return '\n'.join(out)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    result=render(json.loads(Path(args.input).read_text()));path=Path(__file__).resolve().parents[1]/'assets/joint-draw-audit.svg'
    if args.check:
        if path.read_text()!=result:raise SystemExit('SVG differs from sample audit')
    else:path.write_text(result)
