"""Portable entry audit: reconstruct force onset, verify saved geometry receipts.

Raw full-workpiece pair searches require the original trajectories and are not
repeated by this reduced-package verifier. No production simulation is launched.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(text):
    """Independent stdlib parser; keep the last recorded duplicate timestep."""
    columns, rows = None, {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == 'Step':
            columns = fields if {'Time', 'v_fx', 'v_fy', 'v_fz'} <= set(fields) else None
            continue
        if columns is None or len(fields) != len(columns):
            continue
        try:
            values = list(map(float, fields))
        except ValueError:
            continue
        assert all(math.isfinite(v) for v in values), 'Nonfinite record'
        row = dict(zip(columns, values))
        step = int(row['Step'])
        assert step == row['Step']
        rows[step] = (row['Time'], row['v_fx'], row['v_fy'], row['v_fz'])
    return [(step, *rows[step]) for step in sorted(rows)]


def verify(data):
    onsets = read_csv(data/'onset.csv')
    prefixes = read_csv(data/'force_prefixes.csv')
    geometry = read_csv(data/'sampled_geometry.csv')
    index = json.loads((data/'execution/index.json').read_text())
    assert len(onsets) == len(index) == 18
    assert len({(r['condition'], r['realization']) for r in onsets}) == 18
    checks = []
    ramp_count = 0
    for onset in onsets:
        name = onset['run']
        entry = index[name]
        for key in ('log', 'launch'):
            assert digest(data/entry[key]) == entry[key+'_sha256'], (name, key)
        assert entry['log_sha256'] == onset['accepted_log_sha256']
        force = records((data/entry['log']).read_text())
        first = next(i for i, r in enumerate(force) if any(v != 0 for v in r[2:]))
        assert first == next(i for i, r in enumerate(force) if max(map(abs, r[2:])) > 1e-12)
        assert first > 0
        a, b = force[first-1:first+1]
        assert a[0] == int(onset['last_zero_step']) and b[0] == int(onset['first_reaction_step'])
        assert b[1] == float(onset['first_reaction_ps'])
        assert b[0]-a[0] == float(onset['bracket_steps'])
        ts = float(onset['translation_start_ps'])
        assert abs(.03*(b[1]-ts)-float(onset['first_reaction_travel_nm'])) < 1e-12
        reduced = [(int(r['step']), *[float(r[k]) for k in ('time_ps', 'fx_eV_A', 'fy_eV_A', 'fz_eV_A')])
                   for r in prefixes if r['run'] == name]
        assert reduced == force[:first+1], name
        ramp = [r for r in force if r[1] < ts-1e-9]
        assert len(ramp) == int(onset['ramp_force_records'])
        assert all(v == 0 for r in ramp for v in r[2:])
        assert abs(float(onset['sphere_x0_A'])-float(onset['relaxed_xmax_A'])-40) < 1e-8
        assert abs(float(onset['sphere_z0_A'])-float(onset['relaxed_zmax_A'])-20) < 1e-8
        ramp_count += len(ramp)
        checks.append(dict(run=name, first_step=b[0], first_ps=b[1], prior_step=a[0]))
    assert len(geometry) == len({(r['run'], r['step']) for r in geometry}) == 82
    ramps, calibration = [], []
    for row in geometry:
        frame = json.loads((data/'frames'/f"{row['run']}_{row['step']}.json").read_text())
        assert int(row['cutoff_pairs']) == frame['cutoff_pairs']
        assert row['source_sha256'] == frame['source_sha256']
        reasons = ast.literal_eval(row['reasons'])
        if any(s.startswith('ramp_') for s in reasons):
            ramps.append(row)
        if 'threshold_calibration' in reasons:
            calibration.append(row)
    assert len(ramps) == 45 and ramp_count == 4527
    assert len(calibration) == 2
    assert all(int(r['cutoff_pairs']) == 0 for r in ramps+calibration)
    # Synthetic parser checks: restart replacement and nonfinite rejection.
    test = 'Step Time v_fx v_fy v_fz\n1 .001 0 0 0\n2 .002 9 0 0\nStep Time v_fx v_fy v_fz\n2 .002 1 0 0\n'
    assert records(test)[1][2] == 1 and len(records(test)) == 2
    try:
        records('Step Time v_fx v_fy v_fz\n1 .001 nan 0 0\n')
    except AssertionError:
        pass
    else:
        raise AssertionError('Nonfinite fixture was accepted')
    return dict(status='PASS', onset_reconstructed_from_original_logs=checks,
                ramp_force_records=ramp_count, saved_ramp_frames=len(ramps),
                geometry_receipts_checked=len(geometry), calibration_endpoints=len(calibration),
                parser_fixtures='duplicate and nonfinite checks passed',
                scope='Force onset rebuilt from copied original logs; geometry receipts cross-checked, not independently recomputed from absent full trajectories.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
