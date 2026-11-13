#!/usr/bin/env python3
"""Run separate synthetic suites to isolate the archived import layouts."""
from pathlib import Path
import os,subprocess,sys
R=Path(__file__).resolve().parents[1]
env=os.environ.copy()
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
for folder in ['code/stage1/tests','code/stage2/tests','tests']:
 subprocess.run([sys.executable,'-m','pytest','-q',str(R/folder)],cwd=R,env=env,check=True)
