import importlib.util,json,subprocess,sys
from pathlib import Path
import pytest
R=Path(__file__).resolve().parents[1]
def load(name):
 spec=importlib.util.spec_from_file_location(name,R/'scripts'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_aggregate_reproduction():
 result=load('verify_results').verify()
 assert result['model_seed_summaries']==22

def test_launcher_requires_explicit_data_paths():
 p=load('run').parser()
 with pytest.raises(SystemExit):p.parse_args(['stage1'])

def test_launcher_overrides_every_legacy_path(tmp_path):
 m=load('run');p=m.parser()
 for stage in ['stage1','nested','ensembles','paired']:
  argv=[stage,'--out',str(tmp_path/'out')]
  if stage in ['stage1','nested']:argv+=['--anatomy',str(tmp_path),'--fmri',str(tmp_path)]
  if stage!='stage1':argv+=['--original-out',str(tmp_path)]
  if stage=='paired':argv+=['--nested-out',str(tmp_path)]
  cmd=m.command(p.parse_args(argv))
  assert not any('/home/' in x for x in cmd)
  assert '--out' in cmd
  if stage in ['stage1','nested']:assert '--anatomy' in cmd and '--fmri' in cmd and '--aggregate-zip' in cmd

def test_source_lineage_is_preserved():
 import hashlib
 records=json.loads((R/'provenance/source_lineage.json').read_text())['files']
 for r in records:
  if not r['modified']:assert hashlib.sha256((R/r['path']).read_bytes()).hexdigest()==r['sha256']
 h=hashlib.sha256()
 for p in sorted((R/'code/stage1').rglob('*.py')):
  if 'tests' in p.parts:continue
  h.update(str(p.relative_to(R/'code/stage1')).encode());h.update(p.read_bytes())
 assert h.hexdigest()=='1405fb37fa064628a5b02ca11f85420bef38f8bc3cc08f3163822b7ac6eea99c'

def test_repo_outputs_are_isolated():
 m=load('run')
 with pytest.raises(Exception):m.output(str(R/'results/new_run'))
 assert m.output(str(R/'runs/new_run')).is_relative_to(R/'runs')
