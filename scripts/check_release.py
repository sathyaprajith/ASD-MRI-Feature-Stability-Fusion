#!/usr/bin/env python3
"""Validate the public release manifest and reject prohibited release files."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
EXTENSIONS={'.nii','.gz','.npz','.npy','.pt','.pth','.ckpt','.joblib','.pkl','.pickle','.h5','.hdf5','.pem','.key','.zip'}
def files():
 return [p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.relative_to(ROOT).parts for x in ['.git','__pycache__','.pytest_cache','.venv']) and ('runs' not in p.relative_to(ROOT).parts or p.relative_to(ROOT).as_posix()=='runs/README.md')]
def check():
 manifest=json.loads((ROOT/'provenance/RELEASE_MANIFEST.json').read_text())
 for item in manifest['files']:
  p=ROOT/item['path']
  assert p.is_file(),item['path']
  assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['path']
 expected={x['path'] for x in manifest['files']}|{'provenance/RELEASE_MANIFEST.json'}
 actual={p.relative_to(ROOT).as_posix() for p in files()}
 assert actual==expected,{'extra':sorted(actual-expected),'missing':sorted(expected-actual)}
 for p in files():
  assert p.suffix.lower() not in EXTENSIONS,p.relative_to(ROOT)
  if p.suffix in {'.py','.md','.txt','.json','.yml','.cff'} or p.name.startswith('.'):
   text=p.read_text(errors='replace')
   # Patterns are assembled to avoid matching this scanner's own source.
   patterns=['gh'+'p_[A-Za-z0-9]{30,}','A'+'KIA[A-Z0-9]{16}','-----BEGIN '+ '(?:RSA |OPENSSH )?PRIVATE KEY-----']
   assert not any(re.search(x,text) for x in patterns),'Possible credential in '+str(p.relative_to(ROOT))
 return {'status':'passed','files':len(actual),'note':'Automated checks supplement, rather than certify, privacy and licensing review.'}
if __name__=='__main__':print(json.dumps(check(),indent=2))
