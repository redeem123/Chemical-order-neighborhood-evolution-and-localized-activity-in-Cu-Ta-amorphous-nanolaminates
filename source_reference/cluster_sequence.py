"""Consecutive interval fields and explicitly operational activity components."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from atomic_analysis import reference, dump, metal, frame_record, expected_tool, CONFIG
from fullfield_activity import batched_activity
from audit_sources import P4, sha, write_csv

PROTOCOL = P4/'config/cluster_sequence.json'
OUT = P4/'data/derived/cluster_sequence'


def components(xyz, ly, cutoff):
    """Connected components with explicit periodic-y images and no x/z wrapping."""
    n = len(xyz)
    if not n:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    support = np.concatenate([xyz+[0, dy, 0] for dy in (-ly, 0, ly)])
    owner = np.tile(np.arange(n), 3)
    lists = cKDTree(support).query_ball_point(xyz, cutoff, workers=1)
    lengths = np.fromiter((len(v) for v in lists), int, n)
    src = np.repeat(np.arange(n), lengths)
    dst = owner[np.concatenate(lists)]
    graph = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n)).tocsr()
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=count)
    assert sizes.sum() == n
    return labels, sizes


def compute_field(run, steps, ref, cfg, inventory):
    dest = OUT/f'{run}_{steps[0]}_{steps[1]}.npz'
    meta_path = dest.with_suffix('.json')
    paths = [Path(inventory[run]['source_root'])/'trajectories'/f'scratch.{s}.dump.zst' for s in steps]
    inputs = {str(p): sha(p) for p in paths}
    inputs.update({str(PROTOCOL): sha(PROTOCOL), str(CONFIG): sha(CONFIG),
                   str(Path(__file__).resolve()): sha(__file__),
                   str(P4/'scripts/fullfield_activity.py'): sha(P4/'scripts/fullfield_activity.py'),
                   str(P4/'scripts/atomic_analysis.py'): sha(P4/'scripts/atomic_analysis.py'),
                   ref['source_path']: ref['sha256']})
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['inputs'] == inputs and meta['data_sha256'] == sha(dest):
            print(f'Validated cache {dest.name}', flush=True)
            return meta
    frames = [dump(p) for p in paths]
    records = [frame_record(f, ref) for f in frames]
    times = np.array(steps)*.001
    observed = np.array([[r['tool_com_x_A'], r['tool_com_z_A']] for r in records])
    expected = np.array([expected_tool(inventory[run], t, cfg) for t in times])
    # The known launch rounding offset is tested as well as interval displacement.
    assert np.max(abs((observed[1]-observed[0])-(expected[1]-expected[0]))) < .005
    x0, x1 = [metal(f, ref) for f in frames]
    tool = frames[1]['xyz'][frames[1]['types']==5].copy()
    box = frames[1]['box'].copy()
    if steps == [4000, 30000]:
        gaps = [float(cKDTree(metal(f, ref)).query(f['xyz'][f['types']==5])[0].min()) for f in frames]
        assert min(gaps) > 7.164
    else:
        gaps = None
    del frames
    print(f'Calculating consecutive field {run} {steps}', flush=True)
    values = batched_activity(x0, x1, 3.375, np.diff(ref['box'][1])[0], cfg)
    valid = np.isfinite(values[:, 1])
    np.savez_compressed(dest, ids=ref['ids'], types=ref['types'], xyz=x1, values=values,
                        valid=valid, tool_xyz=tool, box=box)
    start = float(inventory[run]['translation_start_ps'])
    meta = dict(run=run, steps=steps, inputs=inputs, endpoints=records,
                data_path=str(dest), data_sha256=sha(dest), interval_ps=float(times[1]-times[0]),
                start_distance_nm=.03*max(0, times[0]-start),
                end_distance_nm=.03*max(0, times[1]-start),
                motion_delta_error_A=((observed[1]-observed[0])-(expected[1]-expected[0])).tolist(),
                endpoint_min_tool_gap_A=gaps, valid_atoms=int(valid.sum()))
    meta_path.write_text(json.dumps(meta, indent=2)+'\n')
    return meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    inventory = {r['run_id']: r for r in csv.DictReader((P4/'data/run_inventory.csv').open())}
    # Field name is discovered from the existing audit, never an inferred remote path.
    for run in protocol['runs']:
        if 'source_root' not in inventory[run]:
            roots = [v for v in inventory[run].values() if v == f'/Volumes/CuTaData/CuTa-results/runs/{run}']
            assert len(roots) == 1
            inventory[run]['source_root'] = roots[0]
    ref = reference(2)
    depth = ref['xyz'][:, 2].max()-ref['xyz'][:, 2]
    eligible = (depth >= 0) & (depth <= protocol['eligible_reference_depth_A'])
    bg = compute_field(protocol['background_run'], protocol['background_steps'], ref, cfg, inventory)
    bgdata = np.load(bg['data_path'])
    background = bgdata['values'][eligible & bgdata['valid'], 1]
    thresholds = {str(q): float(np.quantile(background, q)) for q in protocol['threshold_quantiles']}
    sources = {str(PROTOCOL): sha(PROTOCOL), str(Path(__file__).resolve()): sha(__file__),
               ref['source_path']: ref['sha256'], str(CONFIG): sha(CONFIG)}
    sequences, rows, details, cadence = [], [], [], []
    for run in protocol['runs']:
        firstpath = P4/f'data/derived/fullfield/{run}_station10.json'
        first = json.loads(firstpath.read_text())
        assert first['inputs'][str(CONFIG)] == sha(CONFIG)
        assert sha(first['data_path']) == first['data_sha256']
        for endpoint in first['endpoints']:
            assert sha(endpoint['source_path']) == endpoint['source_sha256']
        steps = [e['step'] for e in first['endpoints']]
        duration = steps[1]-steps[0]
        root = Path(inventory[run]['source_root'])/'trajectories'
        available = sorted(int(p.name.split('.')[1]) for p in root.glob('scratch.*.dump.zst'))
        needed = [steps[0]+k*duration for k in range(4)]
        assert set(needed) <= set(available)
        cadence.append(dict(run=run, selected_steps=needed,
            available_frames_in_sequence=sum(needed[0]<=s<=needed[-1] for s in available),
            inventory_only=True, note='Only selected endpoints fully parsed and hashed in this extension.'))
        previous = {str(q): None for q in protocol['threshold_quantiles']}
        for i in range(3):
            if i == 0:
                m = {**first, 'run': run, 'steps': steps}
                sources[str(firstpath)] = sha(firstpath)
            else:
                m = compute_field(run, needed[i:i+2], ref, cfg, inventory)
            sources[m['data_path']] = m['data_sha256']
            a = np.load(m['data_path'])
            assert np.array_equal(a['ids'], ref['ids'])
            summaries = []
            for q in protocol['threshold_quantiles']:
                mask = eligible & a['valid'] & (a['values'][:, 1] > thresholds[str(q)])
                selected = np.flatnonzero(mask)
                labels, sizes = components(a['xyz'][mask], np.diff(ref['box'][1])[0], protocol['cluster_cutoff_A'])
                ids = a['ids'][mask]
                prev = previous[str(q)]
                recurrence = None if prev is None or not len(prev) else len(np.intersect1d(ids, prev))/len(prev)
                row = dict(run=run, interval=i+1, start_nm=m['start_distance_nm'], end_nm=m['end_distance_nm'],
                    quantile=q, threshold_A2=thresholds[str(q)], eligible_valid_atoms=int((eligible&a['valid']).sum()),
                    active_atoms=len(ids), components=len(sizes), largest_component_atoms=int(sizes.max()) if len(sizes) else 0,
                    depth_p95_nm=float(np.quantile(np.maximum(0, ref['xyz'][:,2].max()-a['xyz'][mask,2]),.95)/10) if len(ids) else None,
                    previous_active_id_recurrence=recurrence)
                rows.append(row); summaries.append(row); previous[str(q)] = ids
                if q == protocol['display_quantile']:
                    selection_path = OUT/f'{run}_interval{i+1}_clusters.npz'
                    np.savez_compressed(selection_path, ids=ids, indices=selected, labels=labels, sizes=sizes)
                    for j, size in enumerate(sizes):
                        details.append(dict(run=run, interval=i+1, component=j, atoms=int(size)))
            sequences.append(dict(run=run, condition=first['condition'], interval=i+1,
                field=m, selection_path=str(selection_path), selection_sha256=sha(selection_path),
                summaries=summaries))
            print(f'Components complete {run} interval {i+1}', flush=True)
    write_csv(OUT/'summary.csv', rows)
    write_csv(OUT/'components.csv', details)
    report = dict(protocol=protocol, protocol_sha256=sha(PROTOCOL), background=bg,
                  background_atoms=len(background), thresholds_A2=thresholds,
                  sources=sources, cadence_audit=cadence, sequences=sequences)
    (OUT/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    lines=[]
    for r in rows:
        if r['quantile'] != protocol['display_quantile']: continue
        condition = 'Baseline' if r['run'].startswith('scratch') else 'A50B20'
        recurrence = '--' if r['previous_active_id_recurrence'] is None else f"{100*r['previous_active_id_recurrence']:.1f}"
        lines.append(f"{condition} & {r['interval']} & {r['start_nm']:.2f}--{r['end_nm']:.2f} & {r['active_atoms']} & {r['components']} & {r['largest_component_atoms']} & {r['depth_p95_nm']:.2f} & {recurrence} \\\\")
    (P4/'tables/cluster_sequence_rows.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'thresholds':thresholds,'rows':[r for r in rows if r['quantile']==.99]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
