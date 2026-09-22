"""Executed preparation lineage and PTM of the actual relaxed references."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OVITO_THREAD_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
import io, json, re, sys
from pathlib import Path
import numpy as np
import ovito
from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier
from audit_sources import ROOT, P4, sha, write_csv
from atomic_analysis import reference
sys.path.insert(0, str(ROOT/'tools'))
from write_glass_data import build


def main():
    sources = {}; preparation = []; ptmrows = []
    def record(p, expected=None):
        p = Path(p); h = sha(p)
        if expected is not None:
            assert h == expected, p
        sources[str(p)] = h
        return p
    record(__file__); record(ROOT/'tools/write_glass_data.py')
    for r in (1, 2, 3):
        assembled = json.loads(record(ROOT/f'prepared_structures/scratch_campaign/laminate_R{r:02d}.manifest.json').read_text())
        launch = json.loads(record(ROOT/f'results/runs/relax_laminate_R{r:02d}/metadata/launch.json').read_text())
        assert assembled['data_sha256'] == launch['structure_sha256']
        record(launch['structure'], launch['structure_sha256'])
        log = record(ROOT/f'results/runs/relax_laminate_R{r:02d}/logs/lammps.log').read_text()
        assert f'ASSEMBLED_STRUCTURE_RELAXATION_TEMPLATE_COMPLETE case=relax_laminate_R{r:02d}' in log
        for command in ('boundary p p s', 'minimize 1.0e-6 1.0e-8 100000 1000000',
                        'fix relax all npt temp 300.0 300.0 0.1 x 0.0 0.0 1.0 y 0.0 0.0 1.0 couple xy',
                        'run 120000', 'fix stabilize all nvt temp 300.0 300.0 0.1', 'run 20000'):
            assert command in log, command
        # Time column verifies the implicit metal-unit timestep, not a guessed template default.
        thermo = [list(map(float, line.split())) for line in log.splitlines()
                  if re.match(r'^\s*\d+\s+[-\d.]+', line) and len(line.split()) == 8]
        assert any(abs(a[1]-120) < 1e-9 for a in thermo) and any(abs(a[1]-140) < 1e-9 for a in thermo)
        for material, key in [('A', 'top'), ('B', 'bottom')]:
            source_id = f'glass_{material}_R{r:02d}_xs78'
            provenance = json.loads(record(ROOT/f'prepared_structures/glass_sources/{source_id}.data.provenance.json').read_text())
            seed = json.loads(record(ROOT/provenance['seed_manifest'], provenance['seed_manifest_sha256']).read_text())
            inputpath = record(ROOT/provenance['output'], provenance['output_sha256'])
            atoms, box, _, _ = build(seed, jitter_frac=provenance['jitter_fraction'],
                                     volume_scale=provenance['volume_scale'], cross_section_A=78.)
            saved = np.loadtxt(io.StringIO(inputpath.read_text().split('Atoms # atomic')[1]))
            rebuilt = np.asarray(atoms)
            np.testing.assert_array_equal(saved[:, :2], rebuilt[:, :2])
            assert np.max(abs(saved[:, 2:]-rebuilt[:, 2:])) <= 5.000001e-7
            pl = json.loads(record(ROOT/f'results/runs/{source_id}/metadata/launch.json').read_text())
            assert pl['structure_sha256'] == provenance['output_sha256'] and pl['completed']
            record(assembled[key+'_source'], assembled[key+'_sha256'])
            rawlog = record(Path('/Volumes/CuTaData/CuTa-results/runs')/source_id/'logs/lammps.log').read_text()
            for command in ('timestep 0.001', 'velocity all create 300.0 41001 mom yes rot no dist gaussian',
                            'fix melt all npt temp 300.0 3253.0 0.1 z 0.0 0.0 1.0',
                            'fix liquid all npt temp 3253.0 3253.0 0.1 z 0.0 0.0 1.0',
                            'fix quench all npt temp 3253.0 300.0 0.1 z 0.0 0.0 1.0',
                            'run 590600', 'run 120000', f'GLASS_VALIDATION_TEMPLATE_COMPLETE case={source_id} seed=41001'):
                assert command in rawlog, (source_id, command)
            assert rawlog.count('\nrun 10000\n') == 2
            preparation.append(dict(realization=r, material=material, atom_count=32000, source_seed=seed['seed'],
                executed_velocity_seed=41001, heating_ps=10, melt_hold_ps=10, quench_ps=590.6,
                quench_K_per_ps=5., final_precursor_relaxation_ps=120,
                seed_geometry_reproduced=True, in_plane_strain=assembled['worst_in_plane_strain'],
                assembled_atoms=assembled['n_atoms'], overlap_removed=assembled['overlap_removed_count']))
        ref = reference(r); record(ref['source_path'], ref['sha256'])
        pipeline = import_file(ref['source_path'], atom_style='atomic')
        def boundary(frame, data):
            data.cell_.pbc = (True, True, False)
        pipeline.modifiers.append(boundary)
        mod = PolyhedralTemplateMatchingModifier(rmsd_cutoff=.1)
        for structure in mod.structures:
            structure.enabled = structure.id in (1, 2, 3, 4)
        pipeline.modifiers.append(mod)
        for cutoff in (.08, .10, .12):
            mod.rmsd_cutoff = cutoff
            data = pipeline.compute()
            ids = np.asarray(data.particles['Particle Identifier'])
            np.testing.assert_array_equal(np.sort(ids), ref['ids'])
            labels = np.asarray(data.particles['Structure Type'])
            counts = np.bincount(labels, minlength=5)
            assert counts.sum() == ref['n']
            row = dict(realization=r, rmsd_cutoff=cutoff, atoms=ref['n'])
            for j, name in enumerate(('Other', 'FCC', 'HCP', 'BCC', 'ICO')):
                row[name+'_count'] = int(counts[j]); row[name+'_percent'] = 100*counts[j]/ref['n']
            ptmrows.append(row)
        print('Preparation and relaxed PTM verified R', r, flush=True)
    d = P4/'data/derived'
    write_csv(d/'preparation_protocol.csv', preparation); write_csv(d/'relaxed_reference_ptm.csv', ptmrows)
    lines = [f"R{r['realization']:02d} & {r['rmsd_cutoff']:.2f} & {r['atoms']} & "+
             ' & '.join(f"{r[n+'_percent']:.3f}" for n in ('FCC','HCP','BCC','ICO','Other'))+r' \\' for r in ptmrows]
    (P4/'tables/relaxed_ptm_rows.tex').write_text('\n'.join(lines)+'\n')
    grouped=[]
    for realization in (1, 2, 3):
        group=sorted((r for r in preparation if r['realization']==realization), key=lambda r:r['material'])
        assert len(group)==2 and len({r['executed_velocity_seed'] for r in group})==1
        assert len({r['assembled_atoms'] for r in group})==1 and len({r['overlap_removed'] for r in group})==1
        grouped.append(f"R{realization:02d} & {group[0]['source_seed']}/{group[1]['source_seed']} & {group[0]['assembled_atoms']} & {group[0]['overlap_removed']}"+r' \\')
    lines = grouped
    (P4/'tables/preparation_rows.tex').write_text('\n'.join(lines)+'\n')
    report = dict(ovito_version='.'.join(map(str, ovito.version)), sources=sources,
        source_seed_geometry_reconstruction=True, source_velocity_seeds_distinct=False,
        preparation=preparation, ptm=ptmrows,
        scope='Separately seeded species/positions, common velocity seed and protocol. PTM local motif fractions are not crystalline volume fractions or irreversible defects.',
        outputs={str(p):sha(p) for p in (d/'preparation_protocol.csv', d/'relaxed_reference_ptm.csv', P4/'tables/relaxed_ptm_rows.tex', P4/'tables/preparation_rows.tex')})
    (P4/'reports/preparation_audit.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
