"""Portable NumPy-only checks of the packaged graphs, nulls and local profiles.

This does not reconstruct graphs from full trajectories, rerun MD, or validate
physical assumptions. Original author code is not imported. No source is edited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def statistic(labels, graph):
    """Count directed Ta-centered edges independently for each region/state."""
    output = np.empty((19, len(labels), 4))
    for state in range(19):
        lo, hi = graph['offsets'][state:state + 2]
        src, dst = graph['src'][lo:hi], graph['dst'][lo:hi]
        for region in range(4):
            mask = graph['masks'][region][src]
            centers, neighbors = graph['centers'][src[mask]], dst[mask]
            for first in range(0, len(labels), 32):
                block = labels[first:first + 32]
                ta_centers = block[:, centers] == 1
                count = ta_centers.sum(axis=1)
                assert np.all(count > 0)
                pairs = np.count_nonzero(ta_centers & (block[:, neighbors] == 1), axis=1)
                output[state, first:first + len(block), region] = 1 - pairs / count / graph['compositions'][region]
    return output


def profile_width(xref, current, origin, plane):
    if len(xref) < 50:
        return np.nan
    center = xref.mean(axis=0)
    affine = np.linalg.lstsq(xref - center, current - current.mean(axis=0), rcond=None)[0]
    if np.linalg.cond(affine) > 100:
        return np.nan
    normal = np.linalg.inv(affine)[:, 2]
    stretch = 1 / np.linalg.norm(normal)
    normal *= stretch
    distance = (current - current.mean(axis=0)) @ normal + (center[2] - plane) * stretch
    label = origin if origin[xref[:, 2] > plane].mean() > origin[xref[:, 2] < plane].mean() else 1 - origin
    grid = np.linspace(-6 * stretch, 6 * stretch, 161)
    weights = np.exp(-0.5 * ((distance[None, :] - grid[:, None]) / 0.75) ** 2)
    support = weights.sum(axis=1)
    fraction = (weights @ label) / np.maximum(support, 1e-100)
    crossings = []
    for level in (0.1, 0.5, 0.9):
        candidates = []
        for i in range(160):
            if support[i] >= 3 and support[i + 1] >= 3 and fraction[i] < level <= fraction[i + 1]:
                candidates.append(grid[i] + (level - fraction[i]) * (grid[i + 1] - grid[i]) / (fraction[i + 1] - fraction[i]))
        if not candidates:
            return np.nan
        crossings.append(min(candidates, key=abs))
    if not crossings[0] < crossings[1] < crossings[2]:
        return np.nan
    return (crossings[2] - crossings[0]) / stretch


def verify(root, regenerate_labels=False):
    manifest = json.loads((root / 'manifest.json').read_text())
    for item in manifest['files']:
        assert digest(root / item['path']) == item['sha256'], item['path']
    result = {'scope': 'Packaged reduced evidence only; not full MD or physical validation',
              'verified_package_files': len(manifest['files']), 'regenerated_full_label_assignments': regenerate_labels,
              'preparations': {}}
    for name in ('R01', 'R02'):
        folder = root / 'evidence' / name
        graph = np.load(folder / 'graphs.npz')
        states = json.loads((folder / 'states.json').read_text())
        stats = np.load(folder / 'mc_statistics.npz')
        observed = statistic(graph['labels'][:1], graph)[:, 0]
        assert np.allclose(observed, stats['observed'], atol=1e-13, rtol=0)
        nulls = []
        maximum = 0.
        for number in range(1, 5):
            batch = np.load(folder / f'batch_{number}.npz')
            labels = batch['labels']
            if regenerate_labels:
                population = np.load(folder / 'null_population.npz')
                ta, strata = population['ta'], population['strata']
                full = np.empty((256, len(ta)), dtype=np.uint8)
                rng = np.random.default_rng(int(batch['seed']))
                for group in np.unique(strata):
                    indices = np.flatnonzero(strata == group)
                    for draw in range(256):
                        full[draw, indices] = rng.permutation(ta[indices])
                    assert np.all(full[:, indices].sum(axis=1) == ta[indices].sum())
                assert np.array_equal(full[:, population['support']], labels)
                del full
            measured = statistic(labels, graph)
            maximum = max(maximum, float(np.max(abs(measured - batch['alpha']))))
            assert maximum < 1e-12
            nulls.append(measured)
        null = np.concatenate(nulls, axis=1)
        assert np.allclose(null, stats['null'], atol=1e-12, rtol=0)
        lookup = {(s['arm'], s['phase_index']): i for i, s in enumerate(states)}
        pair = np.array([null[lookup['vibration', p]] - null[lookup['baseline', p]] for p in range(9)])
        obs_pair = np.array([observed[lookup['vibration', p]] - observed[lookup['baseline', p]] for p in range(9)])
        assert np.allclose(pair, stats['paired_null'], atol=1e-12, rtol=0)
        increment_null = pair[-1, :, 0] - pair[0, :, 0]
        increment = float(obs_pair[-1, 0] - obs_pair[0, 0] - increment_null.mean())
        profile = np.load(folder / 'profiles.npz')
        packet = np.load(folder / 'coordinates.npz')
        packet_states = json.loads((folder / 'coordinate_states.json').read_text())
        state_lookup = {(s['arm'], s['phase_index']): i for i, s in enumerate(packet_states)}
        widths = {}
        origin = np.isin(packet['types'], (1, 2))
        for arm in ('baseline', 'vibration'):
            values = np.full((9, len(packet['tile_info'])), np.nan)
            for phase in range(9):
                for tile, info in enumerate(packet['tile_info']):
                    lo, hi = packet['tile_offsets'][tile:tile + 2]
                    indices = packet['tile_atoms'][lo:hi]
                    xref = packet['xyz'][0, indices]
                    delta = packet['xyz'][state_lookup[arm, phase], indices] - xref
                    delta[:, 1] -= float(packet['ly']) * np.rint(delta[:, 1] / float(packet['ly']))
                    values[phase, tile] = profile_width(xref, xref + delta, origin[indices], info[3])
            np.testing.assert_allclose(values, profile['width_' + arm], atol=1e-10, rtol=0, equal_nan=True)
            widths[arm] = values
        valid = np.isfinite(widths['baseline']).all(axis=0) & np.isfinite(widths['vibration']).all(axis=0)
        assert np.array_equal(valid, profile['valid'])
        changes = (widths['vibration'][-1] - widths['vibration'][0]) - (widths['baseline'][-1] - widths['baseline'][0])
        result['preparations'][name] = {
            'max_statistic_rebuild_error': maximum,
            'Cu_rich_interior_increment': increment,
            'increment_null_sd': float(increment_null.std(ddof=1)),
            'increment_mcse': float(increment_null.std(ddof=1) / np.sqrt(1024)),
            'common_valid_tiles': int(valid.sum()),
            'paired_width_increment_A': float(changes[valid].mean())}
        print(name + ' reduced evidence checked', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--regenerate-labels', action='store_true', help='Also regenerate all full-workpiece assignments from recorded seeds.')
    parser.add_argument('--output', type=Path, help='Optional JSON report; source data are never changed.')
    args = parser.parse_args()
    report = verify(args.root, args.regenerate_labels)
    rendered = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered)
