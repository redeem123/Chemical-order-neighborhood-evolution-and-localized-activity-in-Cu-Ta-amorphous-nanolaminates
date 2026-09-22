"""Paired material-stratified label controls on unchanged deformed graphs."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from atomic_analysis import reference, dump, metal, neighbors
from audit_sources import P4, ROOT, sha, write_csv

CFG = P4/'config/dynamic_order_control.json'
OUT = P4/'data/derived/dynamic_order_control'
REGIONS = ['Cu-rich interior', 'Cu-rich interface-near', 'Ta-rich interior', 'Ta-rich interface-near']


def read(path):
    return json.loads(Path(path).read_text())


def assignments(ta, strata, count, seed):
    """The first row is observed. All other rows conserve every stratum count."""
    rng = np.random.default_rng(seed)
    result = np.tile(np.asarray(ta, dtype=np.uint8), (count+1, 1))
    for group in np.unique(strata):
        ix = np.flatnonzero(strata == group)
        for k in range(1, count+1):
            result[k, ix] = rng.permutation(result[0, ix])
        assert np.all(result[:, ix].sum(axis=1) == result[0, ix].sum())
    return result


def statistics(labels, centers, src, dst, masks, compositions):
    """Edge-weighted Ta-Ta statistic for observed and paired synthetic labels."""
    central_ta = labels[:, centers[src]].astype(bool)
    neighbor_ta = labels[:, dst]
    result = np.full((len(labels), len(masks)), np.nan)
    for j, (mask, composition) in enumerate(zip(masks, compositions)):
        selected = central_ta & mask[src][None, :]
        denominator = selected.sum(axis=1)
        valid = (denominator > 0) & (composition > 0)
        numerator = (selected * neighbor_ta).sum(axis=1)
        result[valid, j] = 1 - numerator[valid]/denominator[valid]/composition
    return result


def prepare(case_name):
    """Read raw sources once; all later statistic rebuilds use the reduced graph."""
    cfg = read(CFG)
    target = OUT/case_name
    target.mkdir(parents=True, exist_ok=True)
    if (target/'graph_manifest.json').exists():
        m = read(target/'graph_manifest.json')
        assert m['config_sha256'] == sha(CFG) and m['script_sha256'] == sha(__file__)
        assert sha(target/'graphs.npz') == m['graph_sha256']
        for p, h in m['sources'].items():
            assert sha(p) == h, p
        print(case_name, 'verified existing graph cache', flush=True)
        return
    original = P4/f'data/derived/cycle_structure/{case_name}'
    manifest = read(original/'manifest.json')
    case = manifest['case']
    r = case['realization']
    ref = reference(r)
    x0 = ref['xyz']; top = x0[:, 2].max(); ly = np.diff(ref['box'][1])[0]
    origin = np.isin(ref['types'], [1, 2])
    ta = np.isin(ref['types'], [2, 4]).astype(np.uint8)
    ip = ROOT/f'results/runs/relax_laminate_R{r:02d}/metadata/relaxed_structure.manifest.json'
    interfaces = np.array(read(ip)['interface_coordinates_A'])
    distance = np.min(abs(x0[:, 2, None]-interfaces), axis=1)
    center = [float(case['row']['x_start'])-100, ref['box'][1].mean()]
    cohort = (np.all(abs(x0[:, :2]-center) <= 25, axis=1)
              & (top-x0[:, 2] <= 60) & (top-x0[:, 2] >= 0))
    centers = np.flatnonzero(cohort & (ref['ids'] % 3 == 0))
    masks = []; compositions = []
    for o in (True, False):
        for interior in (True, False):
            mask = cohort & (origin == o) & ((distance > 3.6) if interior else (distance <= 3.6))
            masks.append(mask[centers]); compositions.append(ta[mask].mean())
    masks = np.array(masks); compositions = np.array(compositions)
    strata = np.floor((x0[:, 2]-x0[:, 2].min())/cfg['z_bin_A']).astype(int)*2+origin.astype(int)
    labels = assignments(ta, strata, cfg['permutations'], cfg['seed']+r)
    sources = {str(CFG): sha(CFG), ref['source_path']: ref['sha256'], str(ip): sha(ip),
               str(original/'manifest.json'): sha(original/'manifest.json'),
               str(original/'regional.csv'): sha(original/'regional.csv')}
    for helper in ('atomic_analysis.py', 'audit_sources.py'):
        sources[str(P4/'scripts'/helper)] = sha(P4/'scripts'/helper)
    with (original/'regional.csv').open() as stream:
        previous = {(a['arm'], int(a['phase_index']), a['region']): a for a in csv.DictReader(stream) if a['stage'] != 'postpass'}
    items = [dict(arm='reference', phase_index=-1)] + [i for i in case['items'] if i['stage'] == 'cycle']
    assert len(items) == 19
    graphs = []; states = []; errors = []
    known_hashes = {a['source_path']: a['source_sha256'] for a in manifest['validated_frames']}
    for item in items:
        if item['arm'] == 'reference':
            x = x0; step = -1; run = f'relax_laminate_R{r:02d}'; source_hash = ref['sha256']
        else:
            f = dump(item['frame']['source_path'])
            assert f['sha256'] == known_hashes[f['source_path']]
            assert f['step'] == int(item['frame']['step'])
            x = metal(f, ref); step = f['step']; run = item['run']; source_hash = f['sha256']
            sources[f['source_path']] = source_hash
        ns = neighbors(x, centers, cfg['cutoff_A'], ly)
        src = np.repeat(np.arange(len(centers), dtype=np.int32), [len(js) for js in ns])
        dst = np.concatenate(ns).astype(np.int32)
        measured = statistics(labels[:1], centers, src, dst, masks, compositions)[0]
        for j, region in enumerate(REGIONS):
            old = previous[(item['arm'], item['phase_index'], region)]
            assert int(old['centers']) == int(masks[j].sum())
            errors.append(abs(measured[j]-float(old['alpha_TaTa'])))
        graphs.append((src, dst))
        states.append(dict(arm=item['arm'], phase_index=item['phase_index'], run_id=run, step=step, source_sha256=source_hash))
        print(case_name, item['arm'], item['phase_index'], 'graph checked', flush=True)
    assert max(errors) < 1e-12
    support = np.unique(np.concatenate([centers] + [b for a, b in graphs]))
    offsets = np.r_[0, np.cumsum([len(a) for a, b in graphs])]
    np.savez_compressed(target/'graphs.npz', labels=labels[:, support],
        support_ids=ref['ids'][support], centers=np.searchsorted(support, centers), masks=masks,
        compositions=compositions, src=np.concatenate([a for a, b in graphs]),
        dst=np.searchsorted(support, np.concatenate([b for a, b in graphs])), offsets=offsets)
    m = dict(case=case_name, realization=r, config_sha256=sha(CFG), script_sha256=sha(__file__),
        sources=sources, states=states, centers=len(centers), support_atoms=len(support),
        full_atoms=len(ta), permutations=cfg['permutations'], seed=cfg['seed']+r,
        full_label_assignments_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
        strata_count=int(len(np.unique(strata))), stratum_counts_conserved=True,
        maximum_observed_reconciliation_error=max(errors), graph_sha256=sha(target/'graphs.npz'))
    (target/'graph_manifest.json').write_text(json.dumps(m, indent=2)+'\n')


def summarize():
    """Portable rebuild: no raw file, coordinate, reference or external asset read."""
    cfg = read(CFG); all_rows = []; contrasts = []; output_hashes = {}; records = []
    for case_name in cfg['cases']:
        folder = OUT/case_name; m = read(folder/'graph_manifest.json')
        assert m['config_sha256'] == sha(CFG) and m['script_sha256'] == sha(__file__)
        assert sha(folder/'graphs.npz') == m['graph_sha256']
        a = np.load(folder/'graphs.npz'); fields = []
        for index, state in enumerate(m['states']):
            lo, hi = a['offsets'][index:index+2]
            fields.append(statistics(a['labels'], a['centers'], a['src'][lo:hi], a['dst'][lo:hi], a['masks'], a['compositions']))
        values = np.array(fields)
        assert np.isfinite(values).all()
        for i, state in enumerate(m['states']):
            for j, region in enumerate(REGIONS):
                v = values[i, :, j]
                all_rows.append(dict(case=case_name, realization=m['realization'], **state, region=region,
                    centers=int(a['masks'][j].sum()), observed=float(v[0]), null_mean=float(v[1:].mean()),
                    null_sd=float(v[1:].std(ddof=1)), excess=float(v[0]-v[1:].mean())))
        lookup = {(s['arm'], s['phase_index']): i for i, s in enumerate(m['states'])}
        paired = np.array([values[lookup[('vibration', p)]]-values[lookup[('baseline', p)]] for p in range(9)])
        for j, region in enumerate(REGIONS):
            change = paired[-1, :, j]-paired[0, :, j]
            excess = paired[:, 0, j]-paired[:, 1:, j].mean(axis=1)
            contrasts.append(dict(case=case_name, realization=m['realization'], region=region,
                mean_observed_paired=float(paired[:, 0, j].mean()),
                mean_null_paired=float(paired[:, 1:, j].mean()), mean_excess_paired=float(excess.mean()),
                mean_null_paired_sd=float(paired[:, 1:, j].mean(axis=0).std(ddof=1)),
                excess_start=float(excess[0]), excess_end=float(excess[-1]),
                observed_paired_increment=float(change[0]), null_paired_increment=float(change[1:].mean()),
                excess_paired_increment=float(change[0]-change[1:].mean()),
                null_paired_increment_sd=float(change[1:].std(ddof=1))))
        np.savez_compressed(folder/'statistics.npz', alpha=values, paired=paired)
        for name in ('graphs.npz', 'graph_manifest.json', 'statistics.npz'):
            output_hashes[f'{case_name}/{name}'] = sha(folder/name)
        records.append({k: m[k] for k in ('case', 'centers', 'support_atoms', 'permutations', 'maximum_observed_reconciliation_error')})
    write_csv(OUT/'phase_statistics.csv', all_rows)
    write_csv(OUT/'contrasts.csv', contrasts)
    table = []
    for v in contrasts:
        table.append(f"R{v['realization']:02d} & {v['region']} & {v['mean_observed_paired']:+.4f} & {v['mean_null_paired']:+.4f} & {v['mean_excess_paired']:+.4f} & {v['excess_paired_increment']:+.4f} & {v['null_paired_increment_sd']:.4f} \\\\")
    (P4/'tables/dynamic_order_rows.tex').write_text('\n'.join(table)+'\n')
    for name in ('phase_statistics.csv', 'contrasts.csv'):
        output_hashes[name] = sha(OUT/name)
    report = dict(status='passed', config_sha256=sha(CFG), script_sha256=sha(__file__),
        cases=records, reference_states=2, reused_trajectory_frames=36, new_trajectory_frames=0,
        phase_region_statistics=len(all_rows), contrasts=len(contrasts),
        table_sha256=sha(P4/'tables/dynamic_order_rows.tex'), outputs=output_hashes,
        inference='Conditional material-stratified controls; no p-values or independent-material error bars.')
    (OUT/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(contrasts, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'summarize', 'all'], default='all')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    assert 1 <= args.workers <= 2
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ('prepare', 'all'):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(prepare, read(CFG)['cases']))
    if args.stage in ('summarize', 'all'):
        summarize()


if __name__ == '__main__':
    main()
