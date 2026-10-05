"""Relocate archived drawing code; reads saved statistics, does not refit models."""
from pathlib import Path
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--figure',choices=['CF02','CF04'],required=True)
a=ap.parse_args();root=Path(__file__).resolve().parents[1]
name={'CF02':'render_CF02_historical_v2.0.5.py','CF04':'render_CF04_historical_v2.0.4.py'}[a.figure]
text=(Path(__file__).parent/name).read_text()
old="ROOT=Path('/Users/qijiansheng/PycharmProjects/CevicalCancerV2')"
if text.count(old)!=1:raise RuntimeError('Historical source relocation must match exactly once')
text=text.replace(old,'ROOT=Path('+repr(str(root))+')')
exec(compile(text,name,'exec'),{'__name__':'__main__'})
