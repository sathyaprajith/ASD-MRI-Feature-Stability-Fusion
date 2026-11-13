#!/usr/bin/env python3
"""Merge stage reports, never include participant-level prediction files."""
import argparse,json,zipfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/home/jovyan/ASD_TO_070_RESULTS')
    p.add_argument('--archive',default='/home/jovyan/ASD_TO_070_FINAL_AGGREGATES.zip')
    a=p.parse_args();root=Path(a.root);archive=Path(a.archive)
    folders=('quick','nested','paired')
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in folders:
            base=root/folder
            if not base.exists(): continue
            for f in base.rglob('*'):
                if not f.is_file():continue
                if any('private' in part.lower() for part in f.parts):continue
                if f.suffix.lower() not in ('.json','.csv','.md','.txt'):continue
                z.write(f,f.relative_to(root))
        z.writestr('DO_NOT_MISREAD.txt',
            'Retrospective ABIDE-I model development. Fixed mixtures and nested tuning do not prove\n'
            'independent generalization. Do not choose success by looking at selected test-site AUCs.\n'
            'Original patient-level predictions, MRI data and checkpoints are excluded.\n')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError('Corrupt export')
        print(f'Aggregate only: {archive} ({len(z.namelist())} files)')

if __name__=='__main__':main()
