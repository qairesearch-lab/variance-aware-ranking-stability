"""Read-only SHA-256 verification, using Python standard library only."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'FILE_MANIFEST_v2.0.20.json').read_text())
errors=[]
for row in manifest['files']:
 p=root/row['path'];h=hashlib.sha256()
 if not p.is_file():errors.append(row['path']+': missing');continue
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 if h.hexdigest()!=row['sha256']:errors.append(row['path']+': SHA-256 mismatch')
print(json.dumps({'files_checked':len(manifest['files']),'errors':errors},ensure_ascii=False))
sys.exit(bool(errors))
