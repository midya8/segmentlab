import argparse
import contextlib
import copy
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from segmentlab.cli import MODEL_FIELDS,build_parser,load_model,dispatch

ROOT=pathlib.Path(__file__).resolve().parents[1]


class CliContractTests(unittest.TestCase):
    def test_all_top_level_help_commands_described(self):
        parser=build_parser();help_text=parser.format_help()
        for command in ['demo','infer','sample','joint','information','decide']:self.assertIn(command,help_text)
        for phrase in ['entropy','loss','Gaussian','model-conditional']:self.assertIn(phrase,help_text)
    def test_each_model_command_help_has_complete_schema(self):
        parser=build_parser()
        for command in ['infer','sample','joint','information','decide']:
            output=io.StringIO()
            with contextlib.redirect_stdout(output),self.assertRaises(SystemExit) as error:
                parser.parse_args([command,'--help'])
            self.assertEqual(error.exception.code,0)
            for field in MODEL_FIELDS:self.assertIn(field,output.getvalue())
    def test_option_defaults_and_bounds_in_help(self):
        parser=build_parser()
        sample=parser.parse_args(['sample','file.json']);self.assertEqual((sample.draws,sample.seed),(100,0))
        decide=parser.parse_args(['decide','file.json']);self.assertEqual((decide.false_cut_cost,decide.missed_cut_cost),(1.,1.))
        for command,phrases in [('sample',['10000','2**32-1']),('joint',['1<=a<b<N','other cuts allowed']),('decide',['not money','1e-6..1e6'])]:
            output=io.StringIO()
            with contextlib.redirect_stdout(output),self.assertRaises(SystemExit):parser.parse_args([command,'--help'])
            for phrase in phrases:self.assertIn(phrase,' '.join(output.getvalue().split()))
    def test_strict_loader_missing_extra_and_wrong_top_level(self):
        model=json.loads((ROOT/'examples/sequence.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            path=pathlib.Path(folder)/'input.json'
            for field in sorted(MODEL_FIELDS):
                value=copy.deepcopy(model);value.pop(field);path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):load_model(path)
            for value in [[],True,None,dict(model,extra=1),dict(model,draws=3),dict(model,pairs=[[1,3]])]:
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):load_model(path)
            path.write_text(json.dumps(model));self.assertEqual(load_model(path),model)
    def test_loader_malformed_json_and_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path=pathlib.Path(folder)/'input.json';path.write_text('{bad')
            with self.assertRaises(ValueError):load_model(path)
            with self.assertRaises(OSError):load_model(path.with_name('missing.json'))
    def test_each_dispatch_has_exact_arguments(self):
        model=json.loads((ROOT/'examples/sequence.json').read_text());parser=build_parser()
        for command,function,extra,options in [
            ('infer','infer',{},[]),('sample','sample_posterior',{'draws':3,'seed':23},['--draws','3','--seed','23']),
            ('joint','joint_cuts',{'pairs':[[1,3]]},['--pairs','[[1,3]]']),
            ('information','partition_information',{},[]),('decide','boundary_decision',{'false_cut_cost':.2,'missed_cut_cost':3.},['--false-cut-cost','.2','--missed-cut-cost','3'])]:
            args=parser.parse_args([command,'examples/sequence.json']+options)
            with patch('segmentlab.cli.load_model',return_value=model) as loader,patch('segmentlab.cli.'+function,return_value={'ok':command}) as call:
                self.assertEqual(dispatch(args),{'ok':command});loader.assert_called_once_with('examples/sequence.json');call.assert_called_once_with(**model,**extra)
    def test_demo_never_reads_model(self):
        with patch('segmentlab.cli.load_model') as loader,patch('segmentlab.cli.demo',return_value={'demo':True}):
            self.assertEqual(dispatch(build_parser().parse_args(['demo'])),{'demo':True});loader.assert_not_called()
    def test_unknown_dispatch_fails(self):
        with patch('segmentlab.cli.load_model',return_value={}):
            with self.assertRaises(ValueError):dispatch(argparse.Namespace(command='unknown',input='input.json'))
    def test_no_args_unknown_command_missing_input_and_required_pairs(self):
        parser=build_parser()
        for arguments in [[],['unknown'],['infer'],['sample'],['joint','input.json'],['decide','input.json','--false-cut-cost','word']]:
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:parser.parse_args(arguments)
            self.assertEqual(error.exception.code,2)
    def test_cli_errors_are_prefixed_without_traceback(self):
        result=subprocess.run([sys.executable,'-m','segmentlab','information','README.md'],capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertEqual(result.stdout,'');self.assertIn('segmentlab:',result.stderr);self.assertNotIn('Traceback',result.stderr)
