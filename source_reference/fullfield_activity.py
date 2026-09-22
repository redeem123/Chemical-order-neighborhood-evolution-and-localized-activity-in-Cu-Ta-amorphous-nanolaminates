"""Extend the frozen finite-interval observable to every workpiece atom.

Descriptive display-only population extension. No new MD, predictor or model fit.
Use cuta-md for the same numerical runtime as the frozen sampled calculations.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from atomic_analysis import reference, dump, metal, mic, CONFIG
from audit_sources import P4, sha

OUT = P4 / 'data/derived/fullfield'
CASES = [('scratch_base_R02', 'Baseline'), ('S6_R02', 'A50B20')]


def batched_activity(x0, x1, cutoff, ly, cfg, batch_size=8192):
    """Same complete-neighbor least-squares definition, bounded-memory SVD batches.

    Neighbor ordering equals the original unique-parent ordering. The production
    sampled values are independently checked for every displayed interval.
    """
    support = np.concatenate([x0+[0, offset, 0] for offset in (-ly, 0, ly)])
    owner = np.tile(np.arange(len(x0)), 3)
    tree = cKDTree(support)
    out = np.full((len(x0), 4), np.nan)
    for start in range(0, len(x0), batch_size):
        centers = np.arange(start, min(start+batch_size, len(x0)))
        lists = tree.query_ball_point(x0[centers], cutoff, workers=1)
        neighbors = []
        for i, indices in zip(centers, lists):
            parents = np.unique(owner[indices])
            neighbors.append(parents[parents != i])
        counts = np.array([len(v) for v in neighbors])
        out[centers, 3] = counts
        for count in np.unique(counts):
            if count < cfg['min_neighbors']:
                continue
            local = np.flatnonzero(counts == count)
            ids = centers[local]
            js = np.stack([neighbors[i] for i in local])
            v0 = mic(x0[js]-x0[ids, None, :], ly)
            v1 = mic(x1[js]-x1[ids, None, :], ly)
            u, singular, vh = np.linalg.svd(v0, full_matrices=False)
            good = (singular[:, -1] > 0)
            condition = np.divide(singular[:, 0], singular[:, -1],
                                  out=np.full(len(ids), np.inf), where=good)
            good &= condition <= cfg['max_condition_number']
            ids = ids[good]
            # pinv(v0) @ v1, with no regularization and the same row convention.
            coeff = (np.swapaxes(u[good], 1, 2) @ v1[good]) / singular[good, :, None]
            transform = np.swapaxes(vh[good], 1, 2) @ coeff
            residual = v1[good] - v0[good] @ transform
            total = np.sum(residual**2, axis=(1, 2))
            out[ids, 0] = total
            out[ids, 1] = total/count
            out[ids, 2] = condition[good]
        if start % (batch_size*8) == 0 or centers[-1] == len(x0)-1:
            print(f'  evaluated {centers[-1]+1:,}/{len(x0):,} atoms', flush=True)
    return out


def compute(run, condition, station):
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = P4/f'data/derived/production_{run}.json'
    sampled_path = meta_path.with_suffix('.npz')
    meta = json.loads(meta_path.read_text())
    cfg = meta['effective_config']
    assert meta['code_sha256'] == sha(P4/'scripts/atomic_analysis.py')
    assert meta['config_sha256'] == sha(CONFIG)
    assert meta['realization'] == 2 and meta['condition'] == condition
    assert not meta['exclusions']
    k = next(i for i, row in enumerate(meta['motion_checks']) if row['station_nm'] == station)
    motion = meta['motion_checks'][k]
    endpoints = meta['validated_frames'][2*k:2*k+2]
    assert len(endpoints) == 2
    assert max(abs(motion[key]) for key in ('delta_x_error_A', 'delta_z_error_A')) < .005
    ref = reference(2)
    assert ref['sha256'] == meta['reference_sha256']
    inputs = {str(meta_path): sha(meta_path), str(sampled_path): sha(sampled_path),
              str(CONFIG): sha(CONFIG), ref['source_path']: ref['sha256'],
              str(Path(__file__).resolve()): sha(__file__),
              str(P4/'scripts/atomic_analysis.py'): sha(P4/'scripts/atomic_analysis.py')}
    for endpoint in endpoints:
        assert sha(endpoint['source_path']) == endpoint['source_sha256']
        inputs[endpoint['source_path']] = endpoint['source_sha256']
    target = OUT/f'{run}_station{station:02d}.npz'
    record_path = target.with_suffix('.json')
    if target.exists() and record_path.exists():
        existing = json.loads(record_path.read_text())
        if existing['inputs'] == inputs and existing['data_sha256'] == sha(target):
            print(f'Validated full-field cache: {run} {station} nm', flush=True)
            return existing
    started = time.perf_counter()
    frames = [dump(e['source_path']) for e in endpoints]
    for f, e in zip(frames, endpoints):
        assert f['step'] == e['step'] and f['sha256'] == e['source_sha256']
    x0, x1 = [metal(f, ref) for f in frames]
    tool_mask = frames[1]['types'] == 5
    tool_xyz = frames[1]['xyz'][tool_mask].copy()
    tool_ids = frames[1]['ids'][tool_mask].copy()
    box = frames[1]['box'].copy()
    del frames
    # Verify the station-to-frame join independently against the audited manifest.
    inventory = list(csv.DictReader((P4/'data/frame_manifest.csv').open()))
    joined = [next(row for row in inventory if row['run_id'] == run
                   and row['source_path'] == e['source_path']) for e in endpoints]
    assert abs(float(joined[0]['distance_nm']) - motion['start_distance_nm']) < 1e-8
    dt = float(joined[1]['time_ps']) - float(joined[0]['time_ps'])
    assert abs(dt - motion['interval_ps']) < 1e-8
    print(f'Computing all {ref["n"]:,} metal atoms: {run} {station} nm', flush=True)
    values = batched_activity(x0, x1, cfg['cutoff_A'], np.diff(ref['box'][1])[0], cfg)
    valid = np.isfinite(values[:, 1])
    assert np.all(values[valid, 1] >= 0)
    sampled = np.load(sampled_path)['data']
    sampled = sampled[sampled[:, 1] == station]
    ix = np.searchsorted(ref['ids'], sampled[:, 0].astype(np.int64))
    assert np.array_equal(ref['ids'][ix], sampled[:, 0])
    np.testing.assert_allclose(values[ix], sampled[:, 10:14], atol=1e-11, rtol=1e-11,
                               equal_nan=True)
    np.savez_compressed(target, ids=ref['ids'], types=ref['types'], xyz=x1,
                        values=values, valid=valid, box=box,
                        tool_xyz=tool_xyz, tool_ids=tool_ids)
    record = {'run_id': run, 'condition': condition, 'realization': 2,
              'station_nm': station, 'start_distance_nm': float(joined[0]['distance_nm']),
              'end_distance_nm': float(joined[1]['distance_nm']), 'interval_ps': dt,
              'inputs': inputs, 'endpoints': endpoints, 'cutoff_A': cfg['cutoff_A'],
              'min_neighbors': cfg['min_neighbors'], 'max_condition_number': cfg['max_condition_number'],
              'boundary': ['nonperiodic', 'periodic', 'nonperiodic'],
              'data_path': str(target), 'data_sha256': sha(target),
              'total_metal_atoms': ref['n'], 'valid_atoms': int(valid.sum()),
              'invalid_atoms': int((~valid).sum()), 'tool_atoms': int(tool_mask.sum()),
              'initial_surface_z_A': float(ref['xyz'][:, 2].max()),
              'sampled_comparison_atoms': len(sampled),
              'sampled_max_abs_difference': float(np.nanmax(abs(values[ix]-sampled[:, 10:14]))),
              'value_columns': ['Dmin_sum_A2', 'Dmin_mean_A2', 'fit_condition', 'reference_neighbor_count'],
              'interpretation': 'finite-temperature finite-interval activity, not irreversible STZ identification',
              'population': 'all original metal IDs including boundary atoms; tool excluded from fitting',
              'scope': 'descriptive full-field extension; frozen sampled responses and model scores unchanged',
              'wall_seconds': time.perf_counter()-started}
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    print(f'Completed {run} {station} nm: {valid.sum():,} valid, {(~valid).sum()} invalid; '
          f'sampled difference {record["sampled_max_abs_difference"]:.3g}; '
          f'{record["wall_seconds"]:.1f} s', flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', choices=[c[0] for c in CASES])
    p.add_argument('--station', type=int, choices=[6, 10, 14])
    args = p.parse_args()
    for run, condition in CASES:
        if args.run and args.run != run:
            continue
        for station in (6, 10, 14):
            if args.station and args.station != station:
                continue
            compute(run, condition, station)


if __name__ == '__main__':
    main()
