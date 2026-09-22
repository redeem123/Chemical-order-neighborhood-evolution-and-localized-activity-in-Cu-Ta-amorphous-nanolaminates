"""Pivot existing structural contrasts and prediction scores without refitting."""
import argparse
import csv
import json
import math
from pathlib import Path
from audit_sources import P4, sha

CONDITIONS = ['Baseline', 'A10B05', 'A10B20', 'A25B10', 'A25B20', 'A50B20']
TRANSFERS = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]
SOURCES = ['data/derived/model_scores.csv',
           'data/replication_extension_20260912/prediction_scores.csv',
           'data/derived/cycle_increment/summary.csv',
           'data/replication_extension_20260912/cycle_summary.csv']
MISSING = r'\textemdash'


def records():
    result = {}
    with (P4/SOURCES[0]).open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if (row['mode'] != 'production' or row['degree'] != '2'
                or float(row['ridge_lambda']) != 1 or row['station_nm']):
            continue
        if row['condition'] == 'A25B10' and (int(row['train_preparation']) == 1 or int(row['test_preparation']) == 1):
            continue
        key = (row['condition'], int(row['train_preparation']), int(row['test_preparation']))
        assert key not in result
        result[key] = dict(base=float(row['mse_base']), augmented=float(row['mse_chemical']),
                           gain=float(row['relative_mse_reduction_percent']))
    assert len(result) == 22
    with (P4/SOURCES[1]).open() as stream:
        for row in csv.DictReader(stream):
            if row['station_nm'] != 'pooled':
                continue
            key = ('A10B20', int(row['train']), int(row['test']))
            assert key not in result
            result[key] = dict(base=float(row['mse_base']), augmented=float(row['mse_augmented']),
                               gain=float(row['gain_percent']))
    assert len(result) == 26
    for value in result.values():
        assert all(math.isfinite(v) for v in value.values()) and value['base'] > 0
        expected = 100*(value['base']-value['augmented'])/value['base']
        assert math.isclose(expected, value['gain'], rel_tol=0, abs_tol=1e-12)
    return result


def cell(values, condition, transfer, field):
    value = values.get((condition, *transfer))
    if value is None:
        return MISSING
    return f"{value[field]:+.3f}" if field == 'gain' else f"{value[field]:.4f}"


def cycle_records():
    result = {}
    with (P4/SOURCES[2]).open() as stream:
        for row in csv.DictReader(stream):
            if (float(row['cutoff_A']) != 3.375 or row['region'] != 'Cu-rich interior'
                    or float(row['station_nm']) != 10):
                continue
            if row['condition'] == 'A25B10' and int(row['realization']) == 1:
                continue
            key = row['condition'], int(row['realization'])
            assert key not in result
            result[key] = dict(history=100*float(row['initial_retention_mean_difference']),
                               cycle=100*float(row['cycle_retention_end_difference']),
                               start=float(row['width_initial_offset_A']),
                               width=float(row['width_increment_end_A']))
    assert len(result) == 11
    with (P4/SOURCES[3]).open() as stream:
        selected = [r for r in csv.DictReader(stream)
                    if float(r['station_nm']) == 10 and float(r['cutoff_A']) == 3.375
                    and r['region'] == 'Cu-rich interior']
    assert len(selected) == 1 and ('A10B20', 3) not in result
    row = selected[0]
    result['A10B20', 3] = dict(history=float(row['initial_retention_mean_pp']),
                               cycle=float(row['cycle_retention_end_pp']),
                               width=float(row['width_increment_A']))
    assert len(result) == 12 and all(math.isfinite(x) for r in result.values() for x in r.values())
    return result


def cycle_cell(values, condition, realization, field):
    value = values.get((condition, realization))
    if value is None:
        return MISSING
    return f"{value[field]:+.3f}" if field in ('start', 'width') else f"{value[field]:+.2f}"


def build():
    values = records()
    matrix = []
    detail = []
    for condition in CONDITIONS:
        matrix.append(' & '.join([condition]+[cell(values, condition, t, 'gain') for t in TRANSFERS])+r' \\')
        for field, label in [('base', 'Base MSE'), ('augmented', 'Augmented MSE')]:
            detail.append(' & '.join([condition if field == 'base' else '', label]
                                    +[cell(values, condition, t, field) for t in TRANSFERS])+r' \\')
        if condition != CONDITIONS[-1]:
            detail.append(r'\addlinespace[3pt]')
    cycles = cycle_records()
    cycle_matrix = [' & '.join([condition] +
                              [cycle_cell(cycles, condition, realization, field)
                               for field in ('history', 'cycle', 'width') for realization in (1, 2, 3)]) + r' \\'
                    for condition in CONDITIONS[1:]]
    outputs = {}
    for name, lines in [('prediction_benefit_matrix.tex', matrix), ('prediction_error_matrix.tex', detail),
                        ('cycle_contrast_matrix.tex', cycle_matrix)]:
        path = P4/'tables'/name
        path.write_text('\n'.join(lines)+'\n')
        outputs[str(path.relative_to(P4))] = sha(path)
    report = dict(status='passed', sources={p: sha(P4/p) for p in SOURCES},
                  outputs=outputs, conditions=CONDITIONS, transfers=TRANSFERS,
                  populated_cells=26, unavailable_cells=10,
                  negative_cells=sum(v['gain'] < 0 for v in values.values()),
                  cycle_cases=12, cycle_populated_cells=36, cycle_unavailable_cells=9,
                  scope='Presentation-only pivots. Prediction gains in main text and raw errors in SI. Cycle contrasts include the existing completed A10B20 R03 result; original pre-cycle width offsets remain in SI Table S9. No pooling or refitting.')
    (P4/'reports/table_presentation_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def snapshot():
    target = P4/'reports/table_presentation_preservation_20260912.json'
    if target.exists():
        raise FileExistsError('Do not silently replace the pre-edit preservation snapshot')
    paths = list((P4/'tables').glob('*.tex'))
    for folder in ('data', 'config'):
        paths.extend(p for p in (P4/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    with (P4/'data/figure_asset_manifest.csv').open() as stream:
        paths.extend(P4/r['path'] for r in csv.DictReader(stream))
    paths.extend(P4/p for p in ('references.bib'))
    content = {str(p.relative_to(P4)): sha(p) for p in sorted(set(paths))}
    target.write_text(json.dumps(dict(scope='Pre-existing table values, data, fixed configurations, figure assets and bibliography before presentation-only changes.',
                                     sha256=content), indent=2)+'\n')
    print(f'Frozen presentation invariants for {len(content)} files')


def check():
    values = records()
    report = json.loads((P4/'reports/table_presentation_validation.json').read_text())
    for p, digest in {**report['sources'], **report['outputs']}.items():
        assert sha(P4/p) == digest, p
    rows = (P4/'tables/prediction_benefit_matrix.tex').read_text().splitlines()
    assert len(rows) == 6
    observed = 0
    for condition, line in zip(CONDITIONS, rows):
        fields = [v.strip() for v in line.removesuffix(r' \\').split('&')]
        assert fields[0] == condition and len(fields) == 7
        for transfer, rendered in zip(TRANSFERS, fields[1:]):
            assert rendered == cell(values, condition, transfer, 'gain')
            observed += rendered != MISSING
    assert observed == 26
    old_rows = (P4/'tables/current_prediction_rows.tex').read_text().splitlines()
    assert len(old_rows) == 26
    for line in old_rows:
        condition, transfer, base, augmented, gain = [s.strip() for s in line.removesuffix(r' \\').split('&')]
        import re
        a, b = map(int, re.findall(r'R(\d+)', transfer))
        value = values[(condition, a, b)]
        assert base == f"{value['base']:.4f}" and augmented == f"{value['augmented']:.4f}"
        assert gain == f"{value['gain']:.3f}"
    details = [line for line in (P4/'tables/prediction_error_matrix.tex').read_text().splitlines()
               if not line.startswith(r'\addlinespace')]
    assert len(details) == 12
    for i, condition in enumerate(CONDITIONS):
        for j, field in enumerate(('base', 'augmented')):
            parts = [s.strip() for s in details[2*i+j].removesuffix(r' \\').split('&')]
            assert parts[0] == (condition if j == 0 else '') and len(parts) == 8
            assert parts[2:] == [cell(values, condition, t, field) for t in TRANSFERS]
    cycles = cycle_records()
    rows = (P4/'tables/cycle_contrast_matrix.tex').read_text().splitlines()
    assert len(rows) == 5
    for condition, line in zip(CONDITIONS[1:], rows):
        parts = [s.strip() for s in line.removesuffix(r' \\').split('&')]
        assert parts[0] == condition and len(parts) == 10
        assert parts[1:] == [cycle_cell(cycles, condition, r, field)
                            for field in ('history', 'cycle', 'width') for r in (1, 2, 3)]
    old_cycles = (P4/'tables/cycle_increment_rows.tex').read_text().splitlines()
    assert len(old_cycles) == 11
    detailed_cycles = (P4/'tables/cycle_location_rows.tex').read_text().splitlines()
    for line in old_cycles:
        condition, realization, *cells = [s.strip() for s in line.removesuffix(r' \\').split('&')]
        assert cells == [cycle_cell(cycles, condition, int(realization[1:]), field)
                         for field in ('history', 'cycle', 'start', 'width')]
        # The entire original row, including its pre-cycle width offset,
        # remains visible at station 10 in the existing SI Table S9.
        detail = ' & '.join([condition, realization, '10']+cells)+r' \\'
        assert detail in detailed_cycles
    extension = next(line for line in (P4/'tables/replication_cycles.tex').read_text().splitlines()
                     if line.split('&')[0].strip() == '10')
    parts = [s.strip() for s in extension.removesuffix(r' \\').split('&')]
    assert parts[1:3] == [cycle_cell(cycles, 'A10B20', 3, f) for f in ('history', 'cycle')]
    assert parts[4] == cycle_cell(cycles, 'A10B20', 3, 'width')
    preserved = json.loads((P4/'reports/table_presentation_preservation_20260912.json').read_text())
    changed = [p for p, digest in preserved['sha256'].items() if sha(P4/p) != digest]
    assert not changed, changed
    report['preserved_files'] = len(preserved['sha256'])
    report['original_scores_reconciled'] = 26
    report['cycle_original_values_reconciled'] = 44
    report['cycle_extension_values_reconciled'] = 3
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.snapshot:
        snapshot()
    elif args.check:
        check()
    else:
        print(json.dumps(build(), indent=2))
