"""Readable command parsing, strict model payload loading and dispatch."""
import argparse
import json

from .core import infer
from .decision import boundary_decision
from .demo import demo
from .information import partition_information
from .joint import joint_cuts
from .sampling import sample_posterior


MODEL_FIELDS=frozenset({
    'data','noise_std','prior_mean','prior_std',
    'cut_probability','min_length','max_segments',
})


def build_parser():
    parser=argparse.ArgumentParser(
        description='Exact offline Gaussian changepoint partition audit',
        epilog='All model commands read one strict JSON object. Results are model-conditional, not online alerts or calibrated real-data claims.',
    )
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('demo',help='run fixed shift and noise-misspecification examples')
    descriptions={
        'infer':'full evidence, marginal probabilities, MAP and latent-mean moments',
        'sample':'seeded complete-partition and conditional Gaussian mean draws',
        'joint':'joint inclusion and conditional probabilities for requested cut pairs',
        'information':'full partition entropy, conditioned-prior KL and cut dependence',
        'decide':'constrained minimum posterior expected false/missed boundary loss',
    }
    for name,description in descriptions.items():
        command=commands.add_parser(name,help=description,description=description)
        command.add_argument('input',help='JSON file with exactly: '+', '.join(sorted(MODEL_FIELDS)))
        if name=='sample':
            command.add_argument('--draws',type=int,default=100,help='number of draws,1..10000 (default:100)')
            command.add_argument('--seed',type=int,default=0,help='local RNG seed,0..2**32-1 (default:0)')
        elif name=='joint':
            command.add_argument('--pairs',required=True,help='JSON list of1..64 unique[a,b] interior gaps,1<=a<b<N; other cuts allowed')
        elif name=='decide':
            command.add_argument('--false-cut-cost',type=float,default=1.,help='relative loss for a false cut,1e-6..1e6 (default:1); not money')
            command.add_argument('--missed-cut-cost',type=float,default=1.,help='relative loss for a missed cut,1e-6..1e6 (default:1); not money')
    return parser


def load_model(path):
    with open(path,encoding='utf8') as source:
        model=json.load(source)
    if not isinstance(model,dict) or set(model)!=MODEL_FIELDS:
        raise ValueError('model JSON requires exactly: '+', '.join(sorted(MODEL_FIELDS)))
    return model


def dispatch(args):
    if args.command=='demo':
        return demo()
    model=load_model(args.input)
    if args.command=='infer':
        return infer(**model)
    if args.command=='sample':
        return sample_posterior(**model,draws=args.draws,seed=args.seed)
    if args.command=='joint':
        return joint_cuts(**model,pairs=json.loads(args.pairs))
    if args.command=='information':
        return partition_information(**model)
    if args.command=='decide':
        return boundary_decision(**model,false_cut_cost=args.false_cut_cost,missed_cut_cost=args.missed_cut_cost)
    raise ValueError('unknown command')


def main():
    parser=build_parser()
    args=parser.parse_args()
    try:
        result=dispatch(args)
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,TypeError,OSError,OverflowError) as error:
        parser.exit(2,f'segmentlab: {error}\n')
