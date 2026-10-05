"""Render archived figure sources from a portable package root."""
from pathlib import Path
import argparse,runpy
p=argparse.ArgumentParser();p.add_argument("--figure",choices=["CF02","CF04"],required=True);a=p.parse_args()
name={"CF02":"render_CF02_historical_v2.0.5.py","CF04":"render_CF04_historical_v2.0.4.py"}[a.figure]
runpy.run_path(str(Path(__file__).parent/name),run_name="__main__")
