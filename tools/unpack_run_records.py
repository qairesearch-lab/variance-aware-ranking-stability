"""Restore recorded metrics/configuration/completion/timing files, without overwriting."""
from pathlib import Path
import zipfile,hashlib,json
root=Path(__file__).resolve().parents[1]
with zipfile.ZipFile(root/'run_records.zip') as z:
 for name in z.namelist():
  p=root/name
  if not p.resolve().is_relative_to(root):raise ValueError('Unexpected archive path')
  data=z.read(name)
  if p.exists():
   if p.read_bytes()!=data:raise RuntimeError('Existing record differs: '+name)
  else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 print(json.dumps({'restored_or_verified_files':len(z.namelist())}))
