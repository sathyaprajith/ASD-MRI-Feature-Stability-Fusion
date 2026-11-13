#!/usr/bin/env python3
"""Portable launcher. Original numerical code remains byte-identical."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STAGE1=ROOT/'code/stage1'
STAGE2=ROOT/'code/stage2/scripts'
MODELS=[r['name'] for r in json.loads((ROOT/'configs/experiments.json').read_text()) if r['stage']=='stage1']
def directory(value):
 p=Path(value).expanduser().resolve()
 if not p.is_dir():raise argparse.ArgumentTypeError(f'Directory does not exist: {p}')
 return p

def output(value):
 p=Path(value).expanduser().resolve()
 if p.is_relative_to(ROOT) and not p.is_relative_to(ROOT/'runs'):
  raise argparse.ArgumentTypeError('Put generated outputs under runs/ or outside this repository.')
 return p

def command(args):
 py=sys.executable
 if args.stage=='stage1':
  cmd=[py,str(STAGE1/'run.py'),'--anatomy',str(args.anatomy),'--fmri',str(args.fmri),'--out',str(args.out),'--aggregate-zip',str(args.out/'aggregate.zip'),'--models',*args.models,'--bootstraps',str(args.bootstraps),'--epochs',str(args.epochs),'--inner-splits',str(args.inner_splits),'--device',args.device]
 elif args.stage=='ensembles':
  cmd=[py,str(STAGE2/'quick_ensembles.py'),'--original-out',str(args.original_out),'--out',str(args.out)]
 elif args.stage=='nested':
  cmd=[py,str(STAGE2/'nested_search.py'),'--original-lab',str(STAGE1),'--original-out',str(args.original_out),'--anatomy',str(args.anatomy),'--fmri',str(args.fmri),'--out',str(args.out),'--aggregate-zip',str(args.out/'aggregate.zip'),'--seeds',*map(str,args.seeds),'--n-boot',str(args.n_boot),'--inner-splits',str(args.inner_splits)]
 else:
  cmd=[py,str(STAGE2/'compare_paired.py'),'--original-out',str(args.original_out),'--nested-out',str(args.nested_out),'--out',str(args.out),'--seeds',*map(str,args.seeds)]
 if getattr(args,'pilot',False):cmd.append('--pilot')
 return cmd

def parser():
 p=argparse.ArgumentParser(description=__doc__)
 sub=p.add_subparsers(dest='stage',required=True)
 for stage in ['stage1','ensembles','nested','paired']:
  q=sub.add_parser(stage)
  q.add_argument('--out',type=output,default=ROOT/'runs'/stage)
  if stage in ['stage1','nested']:
   q.add_argument('--anatomy',required=True,type=directory)
   q.add_argument('--fmri',required=True,type=directory)
   q.add_argument('--pilot',action='store_true')
   q.add_argument('--inner-splits',type=int,default=2 if stage=='stage1' else 3)
  if stage!='stage1':q.add_argument('--original-out',required=True,type=directory)
  if stage=='stage1':
   q.add_argument('--models',choices=MODELS,nargs='+',default=MODELS)
   q.add_argument('--bootstraps',type=int,default=8)
   q.add_argument('--epochs',type=int,default=65)
   q.add_argument('--device',default='cpu',help='cpu by default; cuda only with a compatible PyTorch installation')
  if stage=='nested':q.add_argument('--n-boot',type=int,default=8)
  if stage in ['nested','paired']:q.add_argument('--seeds',nargs='+',type=int,default=[17,41])
  if stage=='paired':q.add_argument('--nested-out',required=True,type=directory)
 return p

def main():
 a=parser().parse_args()
 env=os.environ.copy()
 for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']:env.setdefault(key,'4')
 subprocess.run(command(a),cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
