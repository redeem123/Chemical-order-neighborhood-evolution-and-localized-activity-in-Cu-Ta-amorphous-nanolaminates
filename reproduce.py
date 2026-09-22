"""Run the bounded numerical reconstruction on a separate evidence directory."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
STAGES = ('core', 'labels', 'profiles', 'context', 'accounting', 'extended', 'unloading', 'entry')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='Extracted data folder containing MANIFEST.json and evidence/')
    p.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are refused')
    p.add_argument('--stage', choices=('all', 'hashes', *STAGES), default='all')
    a = p.parse_args()
    data, out = a.data.resolve(), a.output.resolve()
    if not (data/'MANIFEST.json').is_file() or not (data/'evidence').is_dir():
        p.error('Data must be the extracted dataset, not the code checkout or HDF5 file.')
    if out == data or out.is_relative_to(data):
        p.error('Keep generated output outside the immutable evidence folder.')
    out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'PAPER4_DATA_ROOT': str(data), 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'MPLBACKEND': 'Agg'}
    check = ROOT/'checks'
    evidence = data/'evidence'
    commands = {
        'hashes': [str(check/'rebuild.py'), '--stage', 'hashes'],
        'core': [str(check/'rebuild.py'), '--stage', 'all'],
        'labels': [str(check/'rebuild.py'), '--stage', 'labels'],
        'profiles': [str(check/'check_profile_definition.py'), '--root', str(data)],
        'context': [str(check/'check_revision.py')],
        'accounting': [str(check/'reviewer2_audit_20260921.py'), '--evidence', str(evidence), '--output', str(out/'accounting')],
        'extended': [str(check/'reviewer2_extended_check_20260921.py'), '--evidence', str(evidence), '--review', str(evidence/'reviewer2_20260921')],
        'unloading': [str(check/'rebuild_unloading_20260921.py'), '--data', str(evidence/'unloading_20260921'), '--output', str(out/'unloading'), '--verify'],
        'entry': [str(check/'verify_entry_20260922.py'), '--data', str(evidence/'entry_20260922'), '--output', str(out/'entry.json')],
    }
    report = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'Author-run numerical reconstruction, not independent physical validation', 'stages': {}}
    for stage in STAGES if a.stage == 'all' else (a.stage,):
        start = time.monotonic()
        with (out/(stage+'.log')).open('w') as log:
            result = subprocess.run([sys.executable, *commands[stage]], env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        report['stages'][stage] = {'returncode': result.returncode, 'seconds': round(time.monotonic()-start, 3), 'log': stage+'.log'}
        report['status'] = 'failed' if result.returncode else 'in_progress'
        (out/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
        print(stage, 'PASS' if result.returncode == 0 else 'FAIL', flush=True)
        if result.returncode:
            print((out/(stage+'.log')).read_text()[-4000:], file=sys.stderr)
            return result.returncode
    report['status'] = 'passed'
    (out/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
