"""Cycle-start controls and sensitivity, separate from all frozen v1 outputs."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[k] = '1'
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from atomic_analysis import reference, dump, metal, neighbors, mic, frame_record, expected_tool
from cycle_structure import make_plan
from audit_sources import ROOT, P4, sha, write_csv

CFG = P4/'config/cycle_increment.json'
OUT = P4/'data/derived/cycle_increment'


def retention(start, current):
    return np.array([len(np.intersect1d(a, b))/len(a) if len(a) else np.nan
                     for a, b in zip(start, current)])


def width(x0, x, origin, z0, half, bandwidth, levels):
    if len(x0) < 50 or not np.any(x0[:, 2] > z0) or not np.any(x0[:, 2] < z0):
        return None
    center = x0.mean(axis=0)
    J = np.linalg.lstsq(x0-center, x-x.mean(axis=0), rcond=None)[0]
    if np.linalg.cond(J) > 100:
        return None
    normal = np.linalg.solve(J, [0., 0., 1.])
    stretch = 1/np.linalg.norm(normal)
    normal *= stretch
    d = (x-x.mean(axis=0))@normal+(center[2]-z0)*stretch
    label = origin if origin[x0[:, 2] > z0].mean() > origin[x0[:, 2] < z0].mean() else 1-origin
    grid = np.linspace(-half*stretch, half*stretch, 161)
    weights = np.exp(-.5*((grid[:, None]-d)/bandwidth)**2)
    counts = weights.sum(axis=1)
    profile = (weights@label)/np.maximum(counts, 1e-100)
    crossings = []
    for level in levels:
        ix = np.flatnonzero((profile[:-1] < level) & (profile[1:] >= level) &
                           (counts[:-1] >= 3) & (counts[1:] >= 3))
        if not len(ix):
            return None
        vals = grid[ix]+(level-profile[ix])*(grid[ix+1]-grid[ix])/(profile[ix+1]-profile[ix])
        crossings.append(float(vals[np.argmin(abs(vals))]))
    if not crossings[0] < crossings[1] < crossings[2]:
        return None
    return (crossings[2]-crossings[0])/stretch


def analyze(task):
    case, station, cfg = task
    name = f'{case["case"]}_s{station:02d}'
    out = OUT/name
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        old = json.loads((out/'manifest.json').read_text())
        if old['config_sha256'] == sha(CFG) and old['script_sha256'] == sha(__file__) and all(
            sha(p) == h for p, h in old['sources'].items()) and all(
            sha(out/n) == h for n, h in old['outputs'].items()):
            print(name, 'verified cached result', flush=True)
            return name
        raise RuntimeError('Existing v2 outputs changed; refuse silent replacement '+name)
    ref = reference(case['realization'])
    assert ref['sha256'] == case['row']['reference_sha256'] == case['baseline']['reference_sha256']
    x0 = ref['xyz']; top = x0[:, 2].max(); ly = np.diff(ref['box'][1])[0]
    ta = np.isin(ref['types'], [2, 4]); origin = np.isin(ref['types'], [1, 2])
    center = [float(case['row']['x_start'])-10*station, ref['box'][1].mean()]
    dx = x0[:, :2]-center
    cohort = np.all(abs(dx) <= 25, axis=1) & (top-x0[:, 2] <= 60) & (top-x0[:, 2] >= 0)
    centers = np.flatnonzero(cohort & (ref['ids'] % 3 == 0))
    imeta = ROOT/f'results/runs/relax_laminate_R{case["realization"]:02d}/metadata/relaxed_structure.manifest.json'
    interfaces = np.array(json.loads(imeta.read_text())['interface_coordinates_A'])
    dist = np.min(abs(x0[:, 2, None]-interfaces), axis=1)
    regions = [(name0+suffix, cohort & (origin == o) & ((dist > 3.6) if interior else (dist <= 3.6)))
               for o, name0 in [(True, 'Cu-rich'), (False, 'Ta-rich')]
               for interior, suffix in [(True, ' interior'), (False, ' interface-near')]]
    tiles = {}
    for variant in cfg['width_variants']:
        n = variant['tiles']
        if n in tiles:
            continue
        edge = np.linspace(-25, 25, n+1); tiles[n] = []
        for ii, z in enumerate(interfaces):
            if not top-54 <= z <= top-6:
                continue
            for ix in range(n):
                for iy in range(n):
                    selected = np.flatnonzero((abs(x0[:, 2]-z) <= 6) &
                        (dx[:, 0] >= edge[ix]) & (dx[:, 0] < edge[ix+1]) &
                        (dx[:, 1] >= edge[iy]) & (dx[:, 1] < edge[iy+1]))
                    tiles[n].append((ii, ix, iy, z, selected))
    sources = {str(CFG): sha(CFG), str(P4/'config/cycle_structure.json'): sha(P4/'config/cycle_structure.json'),
               str(Path(__file__).resolve()): sha(__file__), ref['source_path']: ref['sha256'], str(imeta): sha(imeta)}
    for rr in (case['row'], case['baseline']):
        for key in ('log_path', 'metadata_path'):
            sources[rr[key]] = sha(rr[key])
    for helper in ('atomic_analysis.py', 'cycle_structure.py'):
        p = P4/'scripts'/helper; sources[str(p)] = sha(p)
    def shells(x):
        broad = neighbors(x, centers, max(cfg['retention_cutoffs_A']), ly)
        ds = [np.linalg.norm(mic(x[js]-x[i], ly), axis=1) for i, js in zip(centers, broad)]
        return {cut: [js[d <= cut] for js, d in zip(broad, ds)] for cut in cfg['retention_cutoffs_A']}
    initial = shells(x0); starts = {}; observations = []; profiles = []; validated = []; motions = {}
    for item in case['items']:
        if item['stage'] != 'cycle':
            continue
        frame = dump(item['frame']['source_path']); x = metal(frame, ref)
        sources[frame['source_path']] = frame['sha256']; validated.append(frame_record(frame, ref))
        arm = item['arm']; k = item['phase_index']; ns = shells(x)
        if k == 0:
            starts[arm] = ns
        if arm not in starts:
            raise ValueError('Missing cycle-start reference')
        rr = case['row'] if arm == 'vibration' else case['baseline']
        actual = frame['xyz'][frame['types'] == 5].mean(axis=0)[[0, 2]]
        expected = expected_tool(rr, float(item['frame']['time_ps']), {})
        if arm in motions:
            a0, e0 = motions[arm]
            if np.max(abs((actual-a0)-(expected-e0))) > .005:
                raise ValueError('Motion mismatch')
        else:
            motions[arm] = actual, expected
        common = dict(case=case['case'], condition=case['condition'], realization=case['realization'], station_nm=station,
                      arm=arm, phase_index=k, phase_fraction=item['actual_vibration_phase_rad']/(2*np.pi),
                      distance_nm=float(item['frame']['distance_nm']), step=frame['step'])
        for cut, shell in ns.items():
            rc = retention(starts[arm][cut], shell); ri = retention(initial[cut], shell)
            src = np.repeat(np.arange(len(centers)), [len(js) for js in shell]); dst = np.concatenate(shell)
            for region, mask in regions:
                use = mask[centers]
                if not np.isfinite(rc[use]).all() or not np.isfinite(ri[use]).all():
                    raise ValueError('Undefined reference shell retention')
                tb = use[src] & ta[centers[src]]
                observations.append(dict(common, cutoff_A=cut, region=region, centers=int(use.sum()),
                    cycle_retention=float(rc[use].mean()), initial_retention=float(ri[use].mean()),
                    alpha_TaTa=float(1-ta[dst[tb]].mean()/ta[mask].mean())))
        for v in cfg['width_variants']:
            for ii, ix, iy, z, indices in tiles[v['tiles']]:
                xx = x0[indices]+mic(x[indices]-x0[indices], ly)
                value = width(x0[indices], xx, origin[indices], z, 6, v['kernel_A'], v['levels'])
                profiles.append(dict(common, variant=v['name'], interface=ii, tile_x=ix, tile_y=iy,
                                     width_corrected_A=value, valid=value is not None))
    write_csv(out/'neighbors.csv', observations); write_csv(out/'widths.csv', profiles)
    manifest = dict(config_sha256=sha(CFG), script_sha256=sha(__file__), sources=sources,
                    case=case, station_nm=station, validated_frames=validated,
                    outputs={n: sha(out/n) for n in ('neighbors.csv', 'widths.csv')})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(name, 'completed', flush=True)
    return name


def main():
    p = argparse.ArgumentParser(); p.add_argument('--workers', type=int, default=2)
    p.add_argument('--case'); p.add_argument('--station', type=int); p.add_argument('--plan-only', action='store_true')
    args = p.parse_args(); cfg = json.loads(CFG.read_text()); OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads((P4/'config/cycle_structure.json').read_text()); tasks = []
    for station in cfg['stations_nm']:
        plan = make_plan(dict(base, cycle_center_target_nm=station))
        for case in plan:
            if (not args.case or args.case == case['case']) and (not args.station or args.station == station):
                tasks.append((case, station, cfg))
    (OUT/'plan.json').write_text(json.dumps(dict(config_sha256=sha(CFG), tasks=[dict(case=t[0], station_nm=t[1]) for t in tasks]), indent=2)+'\n')
    print('Planned paired cycles:', len(tasks), flush=True)
    if args.plan_only:
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(analyze, tasks))


if __name__ == '__main__':
    main()
