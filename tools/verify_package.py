"""Verify distributed files using Python's standard library."""
from pathlib import Path
import hashlib,json,sys,zipfile
root=Path(__file__).resolve().parents[1];errors=[];rows=[]
for line in (root/'checksums.txt').read_text().splitlines():
 digest,name=line.split('  ',1);p=root/name;rows.append(name)
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:errors.append(name)
with zipfile.ZipFile(root/'run_records.zip') as z:
 bad=z.testzip()
 if bad:errors.append(bad)
print(json.dumps({'distributed_files_checked':len(rows),'errors':errors}))
sys.exit(bool(errors))
