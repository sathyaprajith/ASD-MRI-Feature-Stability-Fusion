#!/usr/bin/env python3
"""Record a deliberately reviewed release tree; does not grant publication rights."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','__pycache__','.pytest_cache','.venv'}
items=[]
for p in sorted(ROOT.rglob('*')):
 if not p.is_file() or any(x in p.relative_to(ROOT).parts for x in SKIP):continue
 rel=p.relative_to(ROOT).as_posix()
 if 'runs' in p.relative_to(ROOT).parts and rel!='runs/README.md':continue
 if rel=='provenance/RELEASE_MANIFEST.json':continue
 items.append({'path':rel,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(ROOT/'provenance/RELEASE_MANIFEST.json').write_text(json.dumps({'release':'0.1.0','files':items},indent=2)+'\n')
print('Recorded',len(items),'reviewed public files.')
