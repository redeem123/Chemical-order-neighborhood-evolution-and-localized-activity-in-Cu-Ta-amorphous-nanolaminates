"""Rebuild limited post-review controls from hashed, existing evidence."""
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
from audit_sources import P4, sha, write_csv
from cluster_sequence import components
from horizontal_tables import matrix

CFG = P4/'config/reviewer_controls_20260913.json'
OUT = P4/'data/derived/reviewer_controls_20260913'
SEQUENCE = P4/'data/derived/sequence_campaign_20260912'


def graph(xyz, ly, cutoff):
    n = len(xyz)
    support = np.concatenate([xyz+[0, d, 0] for d in (-ly, 0, ly)])
    owner = np.tile(np.arange(n), 3)
    lists = cKDTree(support).query_ball_point(xyz, cutoff, workers=1)
    src = np.repeat(np.arange(n), [len(js) for js in lists])
    dst = owner[np.concatenate(lists)]
    return coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n)).tocsr()


def size_fraction(g):
    count, labels = connected_components(g, directed=False)
    sizes = np.bincount(labels, minlength=count)
    return float(sizes.max()/len(labels))


def checked(path, expected, inputs):
    digest = sha(path)
    assert digest == expected, str(path)
    inputs[str(path.relative_to(P4))] = digest


def matched_counts(cfg, inputs):
    rows, draws = [], []
    for r in cfg['component_realizations']:
        runs = [f'scratch_base_R{r:02d}', f'S6_R{r:02d}']
        metas = [json.loads((SEQUENCE/f'{run}.json').read_text()) for run in runs]
        for run in runs:
            path = SEQUENCE/f'{run}.json'; inputs[str(path.relative_to(P4))] = sha(path)
        for interval in cfg['intervals']:
            fields, original = [], []
            for run, meta in zip(runs, metas):
                name = f'data/derived/sequence_campaign_20260912/{run}_interval{interval}.npz'
                checked(P4/name, meta['selection_files'][name], inputs)
                f = np.load(P4/name, allow_pickle=False)
                xyz = f['xyz']; ly = meta['periodic_y_A']
                g = graph(xyz, ly, cfg['connection_cutoff_A'])
                row = meta['rows'][interval-1]
                assert len(xyz) == row['active_atoms']
                assert abs(size_fraction(g)-row['largest_fraction']) < 1e-14
                # Independent existing graph builder checks the original population.
                _, sizes = components(xyz, ly, cfg['connection_cutoff_A'])
                assert sizes.max() == row['largest_component_atoms']
                fields.append(g); original.append(row)
            n = original[0]['active_atoms']
            assert original[1]['active_atoms'] >= n
            rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], r, interval]))
            values = []
            for draw in range(cfg['draws']):
                ix = np.sort(rng.choice(fields[1].shape[0], n, replace=False))
                value = size_fraction(fields[1][ix][:, ix])
                # An induced subgraph must agree with rebuilding from retained coordinates.
                if draw == 0:
                    xyz = np.load(P4/f'data/derived/sequence_campaign_20260912/{runs[1]}_interval{interval}.npz')['xyz']
                    _, sizes = components(xyz[ix], metas[1]['periodic_y_A'], cfg['connection_cutoff_A'])
                    assert abs(value-sizes.max()/n) < 1e-14
                values.append(value)
                draws.append(dict(realization=r, interval=interval, draw=draw,
                                  matched_active_atoms=n, largest_fraction=value))
            rows.append(dict(realization=r, interval=interval, matched_active_atoms=n,
                vibration_active_atoms=original[1]['active_atoms'],
                baseline_fraction=original[0]['largest_fraction'],
                vibration_unthinned_fraction=original[1]['largest_fraction'],
                thinned_mean=float(np.mean(values)), thinned_sd=float(np.std(values, ddof=1)),
                thinned_min=float(min(values)), thinned_max=float(max(values))))
            print('Matched-count control', r, interval, rows[-1], flush=True)
    write_csv(OUT/'matched_counts.csv', rows); write_csv(OUT/'matched_count_draws.csv', draws)
    cols = [(f'R{r:02d}', f'Interval {i}') for r in cfg['component_realizations'] for i in cfg['intervals']]
    labels = ['Matched active atoms', 'A50B20 original active atoms', 'Baseline fraction (\%)',
              'A50B20 original fraction (\%)', 'A50B20 thinned fraction (\%)', 'Thinning range (\%)']
    cells = {}
    for row in rows:
        col = f"R{row['realization']:02d}", f"Interval {row['interval']}"
        vals = [str(row['matched_active_atoms']), str(row['vibration_active_atoms']),
                f"{100*row['baseline_fraction']:.1f}", f"{100*row['vibration_unthinned_fraction']:.1f}",
                f"{100*row['thinned_mean']:.1f} $\\pm$ {100*row['thinned_sd']:.1f}",
                f"{100*row['thinned_min']:.1f}--{100*row['thinned_max']:.1f}"]
        cells.update({(col, '', k):v for k,v in zip(labels, vals)})
    matrix('reviewer_matched_counts.tex', cols, [''], labels, cells)
    return rows


def relative_widths(cfg, inputs):
    rows = []
    for r in cfg['component_realizations']:
        for station in cfg['width_stations_nm']:
            folder = P4/f'data/derived/cycle_increment/S6_R{r:02d}_s{station:02d}'
            meta = json.loads((folder/'manifest.json').read_text())
            path = folder/'widths.csv'; checked(path, meta['outputs']['widths.csv'], inputs)
            raw = [row for row in csv.DictReader(path.open()) if row['variant'] == cfg['width_variant']]
            indexed = {(row['arm'], int(row['phase_index']), int(row['interface']), int(row['tile_x']), int(row['tile_y'])):row for row in raw}
            assert len(indexed) == len(raw)
            tiles = sorted(set(key[2:] for key in indexed))
            common = [t for t in tiles if all(indexed[(a,k,*t)]['valid'] == 'True'
                      for a in ('baseline','vibration') for k in range(9))]
            means = {(a,k): float(np.mean([float(indexed[(a,k,*t)]['width_corrected_A']) for t in common]))
                     for a in ('baseline','vibration') for k in (0,8)}
            b0,b1,v0,v1 = [means[key] for key in [('baseline',0),('baseline',8),('vibration',0),('vibration',8)]]
            assert min(b0,v0)>0 and len(common)>0
            rows.append(dict(realization=r, station_nm=station, common_tiles=len(common),
                baseline_start_A=b0, baseline_end_A=b1, vibration_start_A=v0, vibration_end_A=v1,
                baseline_change_A=b1-b0, vibration_change_A=v1-v0,
                paired_change_A=(v1-v0)-(b1-b0),
                baseline_fractional_percent=100*(b1-b0)/b0,
                vibration_fractional_percent=100*(v1-v0)/v0,
                paired_fractional_pp=100*((v1-v0)/v0-(b1-b0)/b0)))
    write_csv(OUT/'relative_widths.csv', rows)
    labels = ['Common tiles', 'Baseline start width (Å)', 'A50B20 start width (Å)',
              'Baseline width change (\%)', 'A50B20 width change (\%)',
              'Paired fractional change (pp)', 'Paired absolute change (Å)']
    cols = [(f"R{row['realization']:02d}", f"{row['station_nm']} nm") for row in rows]
    cells = {}
    for col,row in zip(cols,rows):
        keys=['common_tiles','baseline_start_A','vibration_start_A','baseline_fractional_percent',
              'vibration_fractional_percent','paired_fractional_pp','paired_change_A']
        vals=[str(row[keys[0]])]+[f'{row[k]:.3f}' for k in keys[1:3]]+[f'{row[k]:+.2f}' for k in keys[3:6]]+[f"{row[keys[6]]:+.3f}"]
        cells.update({(col,'',k):v for k,v in zip(labels,vals)})
    matrix('reviewer_relative_widths.tex', cols, [''], labels, cells)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CFG.read_text())
    inputs = {str(p.relative_to(P4)):sha(p) for p in (CFG,Path(__file__).resolve(),P4/'scripts/cluster_sequence.py',P4/'scripts/horizontal_tables.py')}
    counts = matched_counts(cfg, inputs)
    widths = relative_widths(cfg, inputs)
    near = {r['realization']:r for r in widths if r['station_nm']==10}
    macros = [r'\newcommand{\MatchedReversalCount}{'+str(sum(r['thinned_mean']<r['baseline_fraction'] for r in counts))+'}']
    for r, suffix in ((1,'One'),(2,'Two')):
        for name,key in [('RelativeBaseWidth','baseline_fractional_percent'),('RelativeOscWidth','vibration_fractional_percent'),('RelativePairedWidth','paired_fractional_pp')]:
            macros.append('\\newcommand{\\'+name+suffix+'}{'+f'{near[r][key]:.2f}'+'}')
    (P4/'tables/reviewer_numbers.tex').write_text('\n'.join(macros)+'\n')
    outputs = {str(p.relative_to(P4)):sha(p) for p in sorted(OUT.glob('*.csv'))}
    for name in ('reviewer_matched_counts.tex','reviewer_relative_widths.tex'):
        p=P4/'tables/horizontal'/name; outputs[str(p.relative_to(P4))]=sha(p)
    p=P4/'tables/reviewer_numbers.tex';outputs[str(p.relative_to(P4))]=sha(p)
    manifest = dict(config=cfg, inputs=inputs, outputs=outputs, matched_count_rows=len(counts),
                    relative_width_rows=len(widths), primary_results_unchanged=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
