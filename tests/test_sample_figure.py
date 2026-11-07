import copy
import importlib.util
import json
import math
import pathlib
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET

ROOT=pathlib.Path(__file__).resolve().parents[1]

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'examples'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

render=load('render_samples').render
NS={'s':'http://www.w3.org/2000/svg'}

class SampleFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit=load('sample_audit').audit();cls.svg=render(cls.audit);cls.tree=ET.fromstring(cls.svg)
    def test_deterministic_committed_asset_and_cli(self):
        self.assertEqual(self.svg,render(self.audit))
        self.assertEqual(self.svg,(ROOT/'assets/joint-draw-audit.svg').read_text())
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path=pathlib.Path(folder)/'audit.json';path.write_text(json.dumps(self.audit))
            result=subprocess.run([sys.executable,'examples/render_samples.py',str(path),'--check'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
    def test_accessibility_and_full_numeric_manifest(self):
        self.assertEqual(self.tree.attrib['aria-labelledby'],'title desc')
        self.assertIn('joint posterior',self.tree.find('s:title',NS).text)
        self.assertIn('Zero K4 visits',self.tree.find('s:desc',NS).text)
        metadata=json.loads(self.tree.find('s:metadata',NS).text)
        for key in ['counts','boundaries','points']:self.assertEqual(metadata[key],self.audit[key])
        self.assertEqual(metadata['paths'],self.audit['first_three_draws'])
        self.assertEqual(metadata['fixture']['settings'],self.audit['settings'])
        self.assertEqual(metadata['domains']['probability'],[0,1])
    def test_exactly_three_unselected_joint_step_paths(self):
        paths=[p for p in self.tree.findall('s:polyline',NS) if p.attrib.get('data-series','').startswith('joint-draw-')]
        self.assertEqual(len(paths),3)
        for idx,(path,draw) in enumerate(zip(paths,self.audit['first_three_draws']),1):
            self.assertEqual(path.attrib['data-series'],f'joint-draw-{idx}')
            points=[tuple(map(float,p.split(','))) for p in path.attrib['points'].split()]
            self.assertEqual(len(points),2*len(draw['segments']))
            for j,segment in enumerate(draw['segments']):
                left,right=points[2*j:2*j+2];self.assertEqual(left[1],right[1])
                self.assertAlmostEqual(left[0],80+segment['start']/4*430,3)
                self.assertAlmostEqual(right[0],80+segment['end']/4*430,3)
                self.assertAlmostEqual(left[1],165+230-(segment['sampled_mean']+1.2)/2.4*230,3)
    def test_all_counts_and_positive_rare_mass_preserved(self):
        bars=[p for p in self.tree.findall('s:rect',NS) if p.attrib.get('data-series','').startswith('count-')]
        self.assertEqual(len(bars),8)
        for row in self.audit['counts']:
            for series,key in [('count-exact','exact_probability'),('count-frequency','sample_frequency')]:
                bar=next(b for b in bars if b.attrib['data-series']==series and int(b.attrib['data-index'])==row['segments'])
                self.assertEqual(float(bar.attrib['data-value']),row[key])
                self.assertAlmostEqual(float(bar.attrib['height']),230*row[key],3)
        rare=[b for b in bars if b.attrib['data-index']=='4'];self.assertGreater(float(rare[0].attrib['height']),0)
        self.assertEqual(float(rare[1].attrib['height']),0)
        self.assertIn('0.000115',self.svg);self.assertIn('zero visits',self.svg)
    def test_all_points_and_cuts_no_thinning(self):
        circles=self.tree.findall('s:circle',NS)
        for series,expected in [('observation',4),('exact-latent-mean',4),('cut-exact',3),('cut-frequency',3),('variance-exact',4),('variance-sample',4)]:
            self.assertEqual(sum(c.attrib.get('data-series')==series for c in circles),expected)
        for circle in circles:
            value=float(circle.attrib['data-value']);self.assertTrue(math.isfinite(value))
    def test_plot_coordinates_in_fixed_domains_and_no_input_mutation(self):
        before=copy.deepcopy(self.audit);self.assertEqual(render(self.audit),self.svg);self.assertEqual(self.audit,before)
        for circle in self.tree.findall('s:circle',NS):
            x=float(circle.attrib['cx']);y=float(circle.attrib['cy']);self.assertTrue(80<=x<=1055);self.assertTrue(165<=y<=835)
        self.assertNotIn('nan',self.svg.lower());self.assertNotIn('inf',self.svg.lower())
    def test_invalid_or_incomplete_fixture_fails_instead_of_clipping(self):
        mutations=[lambda v:v.update(seed=20),lambda v:v.update(draw_count=9999),lambda v:v['counts'].pop(),
                   lambda v:v['first_three_draws'].pop(),lambda v:v['points'][0].update(exact_variance=.3),
                   lambda v:v['boundaries'][0].update(exact_probability=float('nan')),
                   lambda v:v['points'][0].update(exact_mean=4.),lambda v:v['first_three_draws'][0]['latent_mean'].__setitem__(0,2.),
                   lambda v:v['first_three_draws'][0].update(cuts=[]),lambda v:v['settings'].update(noise_std=2.)]
        for mutate in mutations:
            v=copy.deepcopy(self.audit);mutate(v)
            with self.assertRaises(ValueError):render(v)
    def test_qualifications_visible_in_text(self):
        text=' '.join(e.text or '' for e in self.tree.findall('s:text',NS))
        for phrase in ['First3 in generation order','Shared probability scale[0,1]','No sigma²','No credible bands','zero visits','finite precision']:
            self.assertIn(phrase,text)
