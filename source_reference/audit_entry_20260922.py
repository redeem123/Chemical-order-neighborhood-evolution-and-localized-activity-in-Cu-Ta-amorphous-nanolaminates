"""Existing-output contact-entry audit; no MD and no edits to historical inputs."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from atomic_analysis import reference, dump, metal
from audit_sources import ROOT, P4, sha, write_csv

OUT = P4 / 'data/derived/entry_20260922'
CONDITIONS = ['Baseline', 'A10B05', 'A10B20', 'A25B10', 'A25B20', 'A50B20']


def number(text, key):
    vals = re.findall(r'^variable\s+' + re.escape(key) + r'\s+equal\s+([\d.eE+-]+)\s*$', text, re.M)
    if not vals:
        raise ValueError(key)
    return float(vals[-1])


def force_records(text):
    """Later recorded segment replaces earlier duplicate timestep, never averages."""
    header = None
    out = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == 'Step':
            header = fields if all(k in fields for k in ('Time', 'v_fx', 'v_fy', 'v_fz')) else None
        elif header and len(fields) == len(header):
            try:
                row = dict(zip(header, map(float, fields)))
            except ValueError:
                continue
            if not all(np.isfinite(v) for v in row.values()):
                raise ValueError('Nonfinite force record')
            step = int(row['Step'])
            assert step == row['Step']
            out[step] = row
    return [out[s] for s in sorted(out)]


def sources():
    result = {}
    for path in (P4/'data/run_inventory.csv',
                 P4/'data/replication_extension_20260912/data/run_inventory.csv',
                 P4/'data/replication_extension_20260917/data/run_inventory.csv'):
        for row in csv.DictReader(path.open()):
            if row['condition'] in CONDITIONS:
                result[(row['condition'], int(row['realization']))] = row
    assert len(result) == 18, len(result)
    return result


def geometry(frame, ref):
    x = metal(frame, ref)
    t = frame['xyz'][frame['types'] == 5]
    assert len(t) > 0
    ly = np.diff(ref['box'][1])[0]
    result = dict(tool_com_x_A=float(t[:, 0].mean()), tool_com_y_A=float(t[:, 1].mean()),
                  tool_com_z_A=float(t[:, 2].mean()), tool_xmin_A=float(t[:, 0].min()),
                  metal_xmax_A=float(x[:, 0].max()), atom_count=frame['n'])
    for species, kinds, cutoff in [('Cu', (1, 3), 5.4), ('Ta', (2, 4), 7.164)]:
        pts = x[np.isin(ref['types'], kinds)]
        tree = cKDTree(np.concatenate([pts + [0, dy, 0] for dy in (-ly, 0, ly)]))
        ds = tree.query(t, workers=1)[0]
        result[f'min_{species}_C_A'] = float(ds.min())
        result[f'{species}_C_pairs'] = int(tree.query_ball_point(t, cutoff, return_length=True, workers=1).sum())
    result['cutoff_pairs'] = result['Cu_C_pairs'] + result['Ta_C_pairs']
    return result


def run(frames):
    OUT.mkdir(parents=True, exist_ok=True)
    refs = {r: reference(r) for r in (1, 2, 3)}
    entries, samples, prefixes, hashes, selections = [], [], [], {}, {}
    for (condition, realization), row in sorted(sources().items(), key=lambda kv: (kv[0][1], CONDITIONS.index(kv[0][0]))):
        run_id = row['run_id']
        log = Path(row['log_path'])
        launch_path = log.parents[1]/'metadata/launch.json'
        text = log.read_text()
        launch = json.loads(launch_path.read_text())
        coefficients = set(re.findall(r'^pair_coeff ([1-4]) 5 lj/cut ([0-9.eE+-]+) ([0-9.eE+-]+) ([0-9.eE+-]+)\s*$', text, re.M))
        assert len(coefficients) == 4, (run_id, coefficients)
        for kind, eps, sigma, cutoff in coefficients:
            expected = (0.05477, 2.168, 5.4) if int(kind) in (1, 3) else (0.13, 2.86575, 7.164)
            assert np.allclose([float(eps), float(sigma), float(cutoff)], expected, atol=1e-12, rtol=0)
        ref = refs[realization]
        assert launch['structure_sha256'] == ref['sha256']
        hashes[str(log)] = sha(log)
        hashes[str(launch_path)] = sha(launch_path)
        hashes[ref['source_path']] = ref['sha256']
        expected_log = row.get('log_sha256')
        if run_id == 'S4_R01':
            # The legacy inventory points to the accepted replacement log but
            # retains the old failed-run hash. Use its dedicated source ledger;
            # never rewrite the historical inventory or silently accept drift.
            ledger = P4/'data/auxiliary_completion_20260917/recovery_packet_source_manifest.json'
            recorded = json.loads(ledger.read_text())['sources']
            expected_log = recorded[str(log)]
            assert hashes[str(launch_path)] == recorded[str(launch_path)]
            hashes[str(ledger)] = sha(ledger)
        if expected_log:
            assert hashes[str(log)] == expected_log, run_id
        x0, y0, z0 = [number(text, f'tool_{axis}0') for axis in 'xyz']
        bounds = np.vstack([ref['xyz'].min(axis=0), ref['xyz'].max(axis=0)])
        assert np.allclose([x0, y0, z0], [bounds[1, 0]+40, bounds[:, 1].mean(), bounds[1, 2]+20], atol=1e-8, rtol=0)
        ts = float(row['translation_start_ps'])
        forces = force_records(text)
        assert forces and abs(forces[0]['Time']-4) < 1e-8, (run_id, forces[0]['Time'])
        hits = [i for i, q in enumerate(forces) if max(abs(q[k]) for k in ('v_fx', 'v_fy', 'v_fz')) > 1e-12]
        first = hits[0]
        # Exact nonzero detection must select the same first printed record.
        assert first == next(i for i, q in enumerate(forces) if any(q[k] != 0 for k in ('v_fx', 'v_fy', 'v_fz')))
        assert first > 0
        prior, onset = forces[first-1], forces[first]
        ramp = [q for q in forces if q['Time'] < ts-1e-9]
        gap = (onset['Step']-prior['Step'])
        assert onset['Time'] > ts
        entry = dict(run=run_id, condition=condition, realization=realization,
                     translation_start_ps=ts, sphere_x0_A=x0, sphere_y0_A=y0, sphere_z0_A=z0,
                     relaxed_xmax_A=float(bounds[1, 0]), relaxed_zmax_A=float(bounds[1, 2]),
                     last_zero_step=int(prior['Step']), first_reaction_step=int(onset['Step']),
                     last_zero_ps=prior['Time'], first_reaction_ps=onset['Time'],
                     last_zero_travel_nm=.03*(prior['Time']-ts), first_reaction_travel_nm=.03*(onset['Time']-ts),
                     bracket_steps=gap, ramp_force_records=len(ramp),
                     ramp_max_abs_reaction_eV_A=max((max(abs(q[k]) for k in ('v_fx','v_fy','v_fz')) for q in ramp), default=0),
                     onset_fx_eV_A=onset['v_fx'], onset_fy_eV_A=onset['v_fy'], onset_fz_eV_A=onset['v_fz'],
                     legacy_inventory_log_sha256=row.get('log_sha256'), accepted_log_sha256=expected_log)
        entries.append(entry)
        for q in forces[:first+1]:
            prefixes.append(dict(run=run_id, step=int(q['Step']), time_ps=q['Time'],
                                 fx_eV_A=q['v_fx'], fy_eV_A=q['v_fy'], fz_eV_A=q['v_fz']))
        trajectories = log.parents[1]/'trajectories'
        available = {}
        for p in trajectories.glob('scratch.*.dump*'):
            m = re.fullmatch(r'scratch\.(\d+)\.dump(?:\.zst|\.gz)?', p.name)
            if m:
                s = int(m[1])
                # Prefer compressed source; duplicate state identity is checked on use.
                if s not in available or p.suffix == '.zst':
                    available[s] = p
        chosen = {}
        def choose(s, why):
            chosen.setdefault(s, []).append(why)
        ramp_steps = sorted(s for s in available if 4000 <= s < round(ts*1000))
        if ramp_steps:
            for i, why in [(0, 'ramp_first'), (len(ramp_steps)//2, 'ramp_middle'), (-1, 'ramp_last')]:
                choose(ramp_steps[i], why)
        before = [s for s in available if s <= onset['Step']]
        after = [s for s in available if s >= onset['Step']]
        assert before and after, run_id
        choose(max(before), 'at_or_before_first_reaction')
        choose(min(after), 'at_or_after_first_reaction')
        if run_id == 'scratch_base_R02':
            for s in (4000, 30000):
                assert s in available
                choose(s, 'threshold_calibration')
        selections[run_id] = {str(s): dict(path=str(available[s]), reasons=why) for s, why in sorted(chosen.items())}
        if frames:
            for step, reasons in sorted(chosen.items()):
                p = available[step]
                digest = sha(p)
                cache = OUT/'frames'/f'{run_id}_{step}.json'
                if cache.exists():
                    record = json.loads(cache.read_text())
                    assert record['source_sha256'] == digest and record['analysis_sha256'] == sha(__file__)
                else:
                    frame = dump(p)
                    assert frame['step'] == step and frame['sha256'] == digest
                    record = dict(run=run_id, step=step, time_ps=step*.001,
                                  travel_nm=.03*max(0, step*.001-ts), reasons=reasons,
                                  source_path=str(p), source_sha256=digest, analysis_sha256=sha(__file__),
                                  **geometry(frame, ref))
                    cache.parent.mkdir(exist_ok=True)
                    cache.write_text(json.dumps(record, indent=2)+'\n')
                hashes[str(p)] = digest
                samples.append(record)
                print(run_id, step, reasons, 'cutoff pairs', record['cutoff_pairs'], flush=True)
        print(run_id, 'onset', onset['Time'], 'ps', entry['first_reaction_travel_nm'], 'nm', flush=True)
    write_csv(OUT/'onset.csv', entries)
    write_csv(OUT/'force_prefixes.csv', prefixes)
    if samples:
        write_csv(OUT/'sampled_geometry.csv', samples)
    report = dict(source_hashes=hashes, analysis_sha256=sha(__file__), selected_frames=selections,
                  case_count=len(entries), parsed_frames=len(samples),
                  diagnostic='First recorded reaction with any component nonzero; identical using 1e-12 eV/A tolerance. Cadence-limited, not exact first pair time.',
                  ramp_scope='All available force records; first/middle/last saved ramp geometries only, not continuous geometry.',
                  geometry_scope='All workpiece species with periodic-y images, actual LJ cutoffs 5.4/7.164 A; identical original IDs/types.')
    (OUT/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', action='store_true')
    run(parser.parse_args().frames)
