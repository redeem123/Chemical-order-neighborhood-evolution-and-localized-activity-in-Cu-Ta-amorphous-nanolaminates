"""Observable-row tables generated from preserved records, not edited values."""
import argparse
import csv
import difflib
import json
import math
import re
from collections import Counter
from itertools import groupby, product
from pathlib import Path
from audit_sources import P4, sha
from table_presentation import CONDITIONS, TRANSFERS, records, cycle_records

OUT = P4/'tables/horizontal'
MISSING = r'\textemdash'
REPORTS = []
SOURCES = {}


def read_rows(name):
    path = P4/'tables'/name
    SOURCES[str(path.relative_to(P4))] = sha(path)
    result = []
    for line in path.read_text().splitlines():
        if '&' not in line or not line.rstrip().endswith(r'\\'): continue
        row = [v.strip() for v in line.strip().removesuffix(r'\\').split('&')]
        result.append(row)
    assert result and len({len(r) for r in result}) == 1, name
    return result


def balanced_blocks(columns, limit=9):
    count = math.ceil(len(columns)/limit)
    if count == 1: return [columns]
    # Keep every condition/preparation group together whenever possible.
    groups = [list(g) for _, g in groupby(columns, key=lambda c: c[0])]
    result = []; block = []
    target = math.ceil(len(columns)/count)
    for group in groups:
        if len(group) > limit:
            if block: result.append(block); block = []
            result.extend(group[i:i+limit] for i in range(0, len(group), limit))
            continue
        if block and (len(block)+len(group) > limit or len(block) >= target):
            result.append(block); block = []
        block.extend(group)
    if block: result.append(block)
    assert sum(result, []) == columns
    return result


def header(columns):
    result = []
    for level in range(len(columns[0])):
        parts = ['Observable' if level == len(columns[0])-1 else '']
        for prefix, group in groupby(columns, lambda c: c[:level+1]):
            cells = list(group); label = prefix[-1]
            # Explicit width permits long provenance labels to wrap without inflating the matrix.
            width = r'\dimexpr '+f'{len(cells)*.74/len(columns):.8f}'+r'\PaperFourColumnWidth+'+str(2*(len(cells)-1))+r'\tabcolsep\relax'
            parts.append(r'\multicolumn{'+str(len(cells))+r'}{>{\centering\arraybackslash}p{'+width+'}}{'+label+'}')
        result.append(' & '.join(parts)+r' \\')
    return result


def matrix(name, columns, groups, measures, cells, *, limit=9, expected_values=None):
    """cells[(column tuple, group, measure)] = exact frozen display value."""
    assert len(set(columns)) == len(columns)
    text = []; emitted = []; blocks = balanced_blocks(columns, limit)
    for b, cols in enumerate(blocks):
        if b: text.append(r'\par\medskip')
        n = len(cols)
        text.append(r'\begin{fullwidthtabular}['+str(n+1)+r']{L{.26}*{'+str(n)+'}{R{'+f'{.74/n:.8f}'+'}}}'+r'\toprule')
        text.extend(header(cols)); text.append(r'\midrule')
        for group in groups:
            if group:
                text.append(r'\multicolumn{'+str(n+1)+r'}{l}{\itshape '+group+r'} \\')
            for measure in measures:
                values = [cells.get((column, group, measure), MISSING) for column in cols]
                text.append(' & '.join([measure]+values)+r' \\')
                emitted.extend(cells[(column, group, measure)] for column in cols if (column, group, measure) in cells)
            if group != groups[-1]: text.append(r'\addlinespace[2pt]')
        text.append(r'\bottomrule\end{fullwidthtabular}')
    assert Counter(emitted) == Counter(cells.values()), name
    if expected_values is not None: assert Counter(emitted) == Counter(expected_values), name
    path = OUT/name
    path.write_text('\n'.join(text)+'\n')
    REPORTS.append(dict(name=name, sha256=sha(path), column_keys=columns, groups=groups, measures=measures,
                        populated_cells=len(cells), unavailable_cells=len(columns)*len(groups)*len(measures)-len(cells),
                        blocks=len(blocks), mappings=[dict(column=list(c), group=g, measure=m, value=v) for (c,g,m),v in cells.items()]))


def from_rows(source, keys, labels, *, group_index=None, group_prefix='', limit=9, aliases=None, skip_groups=None):
    raw = read_rows(source); aliases = aliases or {}
    if skip_groups is not None:
        raw = [r for r in raw if group_index is None or r[group_index] not in skip_groups]
    columns = list(dict.fromkeys(tuple(aliases.get(i, {}).get(r[i], r[i]) for i in keys) for r in raw))
    def suffix_order(value):
        numbers = tuple(map(int,re.findall(r'R(\d+)',value)))
        if len(numbers)==2 and numbers in TRANSFERS: return (0,TRANSFERS.index(numbers))
        if len(numbers)==1: return (1,numbers[0])
        return (2,value)
    if all(c[0] in CONDITIONS for c in columns):
        columns.sort(key=lambda c:(CONDITIONS.index(c[0]),*[suffix_order(v) for v in c[1:]]))
    elif all(re.fullmatch(r'R\d+',c[0]) for c in columns):
        columns.sort(key=lambda c:tuple(suffix_order(v) for v in c))
    groups = list(dict.fromkeys(group_prefix+r[group_index] if group_index is not None else '' for r in raw))
    measures = list(labels.values()); cells = {}; original = []
    for row in raw:
        col = tuple(aliases.get(i, {}).get(row[i], row[i]) for i in keys)
        group = group_prefix+row[group_index] if group_index is not None else ''
        for index, label in labels.items():
            key = col, group, label
            assert key not in cells, (source, key)
            cells[key] = row[index]; original.append(row[index])
    matrix(source, columns, groups, measures, cells, limit=limit, expected_values=original)


def relaxed_ptm_table():
    raw = read_rows('relaxed_ptm_rows.tex')
    cutoffs = ('0.08', '0.12')
    realizations = ('R01', 'R02', 'R03')
    measures = ('FCC', 'HCP', 'BCC', 'ICO', 'Other')
    values = {}
    for row in raw:
        assert len(row) == 8
        key = (row[0], row[1])
        assert key not in values and row[0] in realizations and row[1] in {'0.08', '0.10', '0.12'}
        values[key] = row[3:]
    assert set(values) == {(r, c) for r in realizations for c in ('0.08', '0.10', '0.12')}
    text = [
        r'\begin{fullwidthtabular}[11]{L{.12}*{10}{R{.088}}}\toprule',
        r' & \multicolumn{5}{c}{RMSD cutoff 0.08} & \multicolumn{5}{c}{RMSD cutoff 0.12} \\',
        'Realization & ' + ' & '.join(measures * 2) + r' \\',
        r'\midrule',
    ]
    for realization in realizations:
        cells = [values[(realization, cutoff)][i] for cutoff in cutoffs for i in range(len(measures))]
        text.append(realization + ' & ' + ' & '.join(cells) + r' \\')
    text.append(r'\bottomrule\end{fullwidthtabular}')
    path = OUT/'relaxed_ptm_rows.tex'
    path.write_text('\n'.join(text) + '\n')
    REPORTS.append(dict(name='relaxed_ptm_rows.tex', sha256=sha(path),
                        column_keys=[[r] for r in realizations], groups=list(cutoffs),
                        measures=list(measures), populated_cells=30,
                        unavailable_cells=0, blocks=1, mappings=[]))


def main_tables():
    plan_path = P4/'data/derived/sequence_campaign_20260912/plan.json'
    SOURCES[str(plan_path.relative_to(P4))] = sha(plan_path)
    plan = json.loads(plan_path.read_text())
    cols = [(c,) for c in CONDITIONS]
    measures = ['Amplitude (nm)', 'Frequency (GHz)', r'$A/h$', 'Period (ps)', r'$v_{z,\max}/v_x$', 'R01', 'R02', 'R03']; cells = {}
    for c in CONDITIONS:
        cases = [r for r in plan['cases'] if r['condition'] == c]
        cells[(c,), '', measures[0]] = '0' if c == 'Baseline' else f'{int(c[1:3])/100:.2f}'
        cells[(c,), '', measures[1]] = '--' if c == 'Baseline' else f"{1000*float(cases[0]['inventory']['frequency_per_ps']):.2f}"
        amplitude = 0 if c == 'Baseline' else int(c[1:3])/100
        frequency = float(cases[0]['inventory']['frequency_per_ps'])
        cells[(c,), '', measures[2]] = f'{amplitude/1.0:.2f}'
        cells[(c,), '', measures[3]] = '--' if c == 'Baseline' else f'{1/frequency:.3f}'
        cells[(c,), '', measures[4]] = f'{2*math.pi*frequency*amplitude/.03:.2f}'
        for r in (1, 2, 3): cells[(c,), '', f'R{r:02d}'] = 'Valid' if any(x['realization'] == r for x in cases) else MISSING
    matrix('loading_matrix.tex', cols, [''], measures, cells)

    path = P4/'data/derived/sequence_campaign_20260912/summary.csv'; SOURCES[str(path.relative_to(P4))] = sha(path)
    rows = list(csv.DictReader(path.open()))
    cols = [(c, f'R{r:02d}') for c in CONDITIONS for r in (1,2,3)]
    measures = ['Start travel (nm)', 'End travel (nm)', r'$N_{\mathrm{active}}$', 'Components', r'$S_{\max}$',
                r'$S_{\max}/N_{\mathrm{active}}$ (\%)', r'$d_{95}$ (nm)', r'Recurrence (\%)']
    groups = ['Interval 1', 'Interval 2', 'Interval 3']; cells = {}
    for row in rows:
        col = row['condition'], f"R{int(row['realization']):02d}"; group = 'Interval '+row['interval']
        vals = [f"{float(row['start_nm']):.2f}", f"{float(row['end_nm']):.2f}", row['active_atoms'], row['components'],
                row['largest_component_atoms'], f"{100*float(row['largest_fraction']):.1f}", f"{float(row['depth_p95_nm']):.2f}",
                '--' if not row['previous_active_id_recurrence'] else f"{100*float(row['previous_active_id_recurrence']):.1f}"]
        cells.update({(col, group, m):v for m,v in zip(measures, vals)})
    matrix('cluster_sequence_rows.tex', cols, groups, measures, cells)
    compact = [r'$N_{\mathrm{active}}$', r'$S_{\max}/N_{\mathrm{active}}$ (\%)', r'$d_{95}$ (nm)']
    matrix('cluster_sequence_compact.tex', cols, groups, compact,
           {k:v for k,v in cells.items() if k[2] in compact})
    for index, group in enumerate(groups, 1):
        matrix(f'cluster_sequence_full_{index}.tex', cols, [group], measures,
               {k:v for k,v in cells.items() if k[1] == group})

    rows = read_rows('current_activity_rows.tex'); cells = {}
    groups = ['6 nm', '10 nm', '14 nm']; measures = [r'$\langle\overline D^2_{\min}\rangle$ (Å$^2$)', '$P$']
    for row in rows:
        for i,g in enumerate(groups):
            for j,m in enumerate(measures): cells[(row[0],row[1]),g,m] = row[2+2*i+j]
    matrix('current_activity_rows.tex', cols, groups, measures, cells, expected_values=[v for r in rows for v in r[2:]])

    values = cycle_records(); cols = [(c,f'R{r:02d}') for c in CONDITIONS[1:] for r in (1,2,3)]
    measures = ['Initial-neighbor retention (pp)', 'Cycle-start retention (pp)', 'Within-cycle width (Å)']; cells = {}
    for (c,r), v in values.items():
        for f,m in zip(['history','cycle','width'],measures):
            cells[(c,f'R{r:02d}'),'',m] = f'{v[f]:+.3f}' if f == 'width' else f'{v[f]:+.2f}'
    matrix('cycle_contrast_matrix.tex', cols, [''], measures, cells)
    scores = records(); cols = [(c,) for c in CONDITIONS]
    groups = ['']; measures = [f'R{a:02d} $\\to$ R{b:02d}' for a,b in TRANSFERS]; cells = {}
    for (c,a,b),v in scores.items(): cells[(c,),'',f'R{a:02d} $\\to$ R{b:02d}'] = f"{v['gain']:+.3f}"
    matrix('prediction_benefit_matrix.tex', cols, groups, measures, cells)
    cells = {}
    for (c,a,b),v in scores.items():
        for field,g in [('base','Base model MSE'),('augmented','Augmented model MSE')]:
            cells[(c,),g,f'R{a:02d} $\\to$ R{b:02d}'] = f"{v[field]:.4f}"
    matrix('prediction_error_matrix.tex', cols, ['Base model MSE','Augmented model MSE'], measures, cells)
    for source in ('data/derived/model_scores.csv','data/replication_extension_20260912/prediction_scores.csv',
                   'data/derived/cycle_increment/summary.csv','data/replication_extension_20260912/cycle_summary.csv'):
        SOURCES[source] = sha(P4/source)


def build():
    REPORTS.clear(); SOURCES.clear(); OUT.mkdir(parents=True, exist_ok=True)
    main_tables()
    from_rows('chemical_rows.tex',[0],{2:r'$c_{\mathrm{Ta}}$',3:r'Observed $\alpha^R$',4:'Null mean',5:'Null SD'},group_index=1)
    from_rows('model_detail_rows.tex',[0,1],{2:'Training rows',3:'Test rows',4:r'Base $R^2$',5:r'Augmented $R^2$',6:'Transfer role'})
    from_rows('sensitivity_rows.tex',[0,1],{3:'MSE reduction (\%)'},group_index=2)
    from_rows('negative_control_rows.tex',[0,1],{2:r'Observed gain (\%)',3:r'Null mean (\%)',4:r'Null minimum (\%)',5:r'Null maximum (\%)'})
    from_rows('cluster_sequence_sensitivity.tex',[0,1],{3:'Threshold (Å$^2$)',4:r'$N_{\mathrm{active}}$',5:'Components',6:r'$S_{\max}$',7:r'Recurrence (\%)'},group_index=2,group_prefix='Percentile ',aliases={1:{str(i):f'Interval {i}' for i in (1,2,3)}},skip_groups={'99'})
    from_rows('cycle_regions_rows.tex',[0,1],{3:r'$\Delta\alpha^R$',4:r'$\Delta\mathrm{CN}$',5:r'Cross-origin $\Delta$ (pp)'},group_index=2)
    from_rows('preparation_rows.tex',[0],{1:'Source seeds (A/B)',3:'Removed atoms'})
    relaxed_ptm_table()
    from_rows('cycle_location_rows.tex',[0,1],{3:r'$\langle\Delta r_{\mathrm{init}}\rangle$ (pp)',4:r'$\Delta r_{\mathrm{cycle,end}}$ (pp)',5:r'$\Delta\widetilde w_0$ (Å)',6:r'$\delta\widetilde w_{\mathrm{end}}$ (Å)'},group_index=2,group_prefix='Travel station (nm) ')
    from_rows('cycle_definition_rows.tex',[0,1],{2:'Retention range (pp)',3:'Width-increment range (Å)',4:'Valid definitions'},limit=6)
    from_rows('cycle_structure_rows.tex',[0,1],{5:r'Late $\Delta r$ (pp)'})
    from_rows('review_coverage_rows.tex',[0],{1:'Runs',2:'Distinct frames',3:'Frame uses'},limit=5)
    from_rows('review_period_rows.tex',[0],{1:'Frequency (GHz)',2:'Period (ps)',3:'Travel per cycle (nm)'})
    from_rows('review_fixed_rows.tex',[0,1],{2:'Vibration travel (nm)',3:'Duration vib/base (ps)',4:r'$\Delta r$ (pp)',5:r'$\delta\widetilde w$ (Å)',6:'Valid tiles'},limit=6)
    from_rows('review_exchange_rows.tex',[0,1],{2:r'Loss (\%)',3:r'Gain (\%)',4:r'$\langle N_0\rangle$',5:r'$\langle N_1\rangle$',6:r'$\langle G-L\rangle$ (pp)'})
    from_rows('review_contact_rows.tex',[0,1],{3:'Duration (ps)',4:r'$\overline F_t$ (nN)',5:r'$\overline F_n$ (nN)'},group_index=2)
    station_labels={0:{str(i):f'{i} nm' for i in (6,10,14)}}
    from_rows('replication_cycles.tex',[0],{1:'Cumulative retention (pp)',2:'Cycle-start retention (pp)',3:r'$\Delta\alpha^R$',4:'Width increment (Å)',5:'Valid tiles'},aliases=station_labels)
    from_rows('replication_sensitivity.tex',[0],{1:'Retention range (pp)',2:'Width range (Å)'},aliases=station_labels)
    from_rows('replication_forces.tex',[0,1],{2:'Duration (ps)',3:r'$F_t$ (nN)',4:r'$F_n$ (nN)'},aliases=station_labels)
    from_rows('dynamic_order_display_rows.tex',[0],{2:'Mean observed contrast',3:'Mean null contrast (SD)',4:r'Mean excess $\langle\Delta u\rangle$',5:r'Excess increment $\delta\Delta u$',6:'Increment null SD'},group_index=1)
    assert len(REPORTS) == 30
    manifest = dict(status='passed', scope='Observable rows and grouped case columns. Frozen source display values retained verbatim; full-population sequence extension is separately validated.',
                    sources=SOURCES, tables=REPORTS, table_count=len(REPORTS), block_count=sum(r['blocks'] for r in REPORTS))
    (OUT/'layout.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['status','table_count','block_count']}))


def integration_patch():
    known = {r['name'] for r in json.loads((OUT/'layout.json').read_text())['tables']}
    result = ['*** Begin Patch']
    for p in [P4/'main_paper4.tex', P4/'supplementary.tex', *sorted((P4/'sections').glob('*.tex'))]:
        old = p.read_text()
        def replace(match):
            block = match[0]
            inp = re.search(r'\\inputtable\{tables/([^}]+)\}',block)
            if inp is None:
                if 'Validated realizations' in block:
                    return r'\input{tables/horizontal/loading_matrix.tex}'
                return block
            name = inp[1]
            if name not in known: return block
            new = r'\input{tables/horizontal/'+name+'}'
            if r'\begin{fullwidthlongtable}' in block:
                cap = re.search(r'(\\caption\{[^\n]+)',block)[1].removesuffix(r'\\').strip()
                new = '\\begin{table}[!htbp]\n\\centering\n'+cap+'\n'+new+'\n\\end{table}'
            return new
        new = re.sub(r'\\begin\{fullwidth(tabular|longtable)\}.*?\\end\{fullwidth\1\}',replace,old,flags=re.S)
        if new == old: continue
        diff = list(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True)))
        result.append('*** Update File: '+str(p))
        # apply_patch uses bare hunk separators rather than numbered unified ranges.
        result.extend(re.sub(r'^@@.*@@.*\n$', '@@\n', line).rstrip('\n') for line in diff[2:])
    result.append('*** End Patch')
    print('\n'.join(result))


def check():
    saved = json.loads((OUT/'layout.json').read_text())
    for p,digest in saved['sources'].items(): assert sha(P4/p)==digest,p
    for rec in saved['tables']:
        assert sha(OUT/rec['name'])==rec['sha256']
        assert len({(tuple(v['column']),v['group'],v['measure']) for v in rec['mappings']})==rec['populated_cells']
    print(json.dumps(dict(status='passed',tables=len(saved['tables']),cells=sum(r['populated_cells'] for r in saved['tables']))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--integration-patch',action='store_true'); parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.integration_patch: integration_patch()
    elif args.check: check()
    else: build()
