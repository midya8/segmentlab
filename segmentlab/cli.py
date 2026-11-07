import argparse
import json
from .core import infer
from .demo import demo


def main():
    p=argparse.ArgumentParser(description='Exact offline Gaussian changepoint partition audit');s=p.add_subparsers(dest='command',required=True);s.add_parser('demo');q=s.add_parser('infer');q.add_argument('input');a=p.parse_args()
    try:
        if a.command=='demo':v=demo()
        else:
            with open(a.input,encoding='utf8') as f:v=json.load(f)
            if not isinstance(v,dict) or set(v)!={'data','noise_std','prior_mean','prior_std','cut_probability','min_length','max_segments'}:raise ValueError('infer requires exact data/noise_std/prior_mean/prior_std/cut_probability/min_length/max_segments fields')
            v=infer(**v)
        print(json.dumps(v,indent=2,allow_nan=False))
    except (ValueError,TypeError,OSError,OverflowError) as e:p.exit(2,f'segmentlab: {e}\n')
