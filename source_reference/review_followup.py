"""New review-requested postprocessing; never overwrite frozen analyses or raw MD."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from audit_sources import P4, ROOT, sha, write_csv, moving_thermo_blocks
from atomic_analysis import reference, dump, metal, neighbors, mic, frame_record, fit_vectors, expected_tool
from cycle_increment import width

OUT = P4/'data/derived/review_followup'
CFG = P4/'config/review_followup.json'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return list(csv.DictReader(Path(path).open()))


def loss_gain(a, b):
    """Per-center losses/gains normalized by own initial coordination."""
    if len(a) != len(b) or any(len(x) == 0 for x in a):
        raise ValueError('Unmatched/empty reference neighborhoods')
    shared = np.array([len(np.intersect1d(x, y)) for x, y in zip(a, b)])
    n0 = np.array([len(x) for x in a]); n1 = np.array([len(x) for x in b])
    loss = (n0-shared)/n0; gain = (n1-shared)/n0
    assert np.allclose(gain-loss, (n1-n0)/n0, atol=1e-14)
    return loss, gain, n0, n1


def clipped_mean(t, values, start, end):
    if not np.all(np.diff(t) > 0) or not t[0] <= start < end <= t[-1]:
        raise ValueError('Unbracketed interval or nonmonotonic force clock')
    inside = (t > start) & (t < end)
    tt = np.r_[start, t[inside], end]
    yy = np.r_[np.interp(start, t, values), values[inside], np.interp(end, t, values)]
    return float(np.trapezoid(yy, tt)/(end-start)), tt, yy


def force_analysis():
    result = []; series = []; source_records = {}; sources = {str(CFG): sha(CFG)}
    for task in read(P4/'data/derived/cycle_increment/plan.json')['tasks']:
        case = task['case']; station = task['station_nm']
        for arm, rr in [('vibration', case['row']), ('baseline', case['baseline'])]:
            run = rr['run_id']
            if run not in source_records:
                path = rr['log_path']; digest = sha(path)
                assert digest == rr['log_sha256'], 'Executed log changed '+run
                text, blocks = moving_thermo_blocks(path)
                assert 'fix tool_react tool setforce 0.0 0.0 0.0' in text
                for k, comp in [('fx', 1), ('fz', 3)]:
                    assert f'variable {k} equal f_tool_react[{comp}]' in text
                assert len(blocks) == 1, 'Multiple segments require explicit reconciliation'
                header, data = blocks[0]; t = data[:, header.index('Time')]
                assert np.allclose(t*1000, data[:, header.index('Step')], atol=1e-5)
                assert np.all(np.diff(t) > 0)
                # Verify incremental translation, allowing the recorded constant COM offset.
                x = data[:, header.index('v_tool_x')]
                assert np.max(abs((x-x[0])+.3*(t-t[0]))) < .002
                source_records[run] = (header, data)
                sources[path] = digest
            header, data = source_records[run]; t = data[:, header.index('Time')]
            items = [i for i in case['items'] if i['stage'] == 'cycle' and i['arm'] == arm]
            start, end = [float(i['frame']['time_ps']) for i in (items[0], items[-1])]
            record = dict(case=case['case'], condition=case['condition'], realization=case['realization'],
                          station_nm=station, arm=arm, run=run, start_ps=start, end_ps=end,
                          duration_ps=end-start, sample_interval_ps=float(np.median(np.diff(t))))
            for label, col in [('Ft', 'v_fx'), ('Fn', 'v_fz')]:
                values = data[:, header.index(col)]*1.602176634
                mean, tt, yy = clipped_mean(t, values, start, end)
                # Retain both outermost samples so the exact endpoints remain bracketed.
                keep = np.unique(np.r_[np.arange(0, len(t), 2), len(t)-1])
                coarse, _, _ = clipped_mean(t[keep], values[keep], start, end)
                record[label+'_mean_nN'] = mean
                record[label+'_downsample_difference_nN'] = coarse-mean
                for tm, value in zip(tt, yy):
                    series.append(dict(case=case['case'], station_nm=station, arm=arm,
                        component=label, time_ps=tm, elapsed_fraction=(tm-start)/(end-start), force_nN=value))
            record['interior_samples'] = int(((t > start) & (t < end)).sum())
            result.append(record)
    write_csv(OUT/'contact_summary.csv', result); write_csv(OUT/'contact_traces.csv', series)
    report = dict(sources=sources, runs=len(source_records), paired_cycles=len(result)//2,
        definition='Signed pre-setforce tool force, +x opposes negative-x translation; +z is upward tool reaction. No absolute values.',
        force_conversion_nN_per_eV_A=1.602176634,
        force_documentation='https://docs.lammps.org/fix_setforce.html',
        maximum_downsample_difference_nN={k:max(abs(r[k+'_downsample_difference_nN']) for r in result) for k in ('Ft','Fn')},
        output_hashes={n:sha(OUT/n) for n in ('contact_summary.csv','contact_traces.csv')})
    (OUT/'contact_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Contact analysis:', report['runs'], 'runs,', report['paired_cycles'], 'paired cycles', flush=True)


def geometry(case):
    ref = reference(case['realization']); x0 = ref['xyz']; top = x0[:,2].max()
    assert ref['sha256'] == case['row']['reference_sha256'] == case['baseline']['reference_sha256']
    ip = ROOT/f'results/runs/relax_laminate_R{case["realization"]:02d}/metadata/relaxed_structure.manifest.json'
    interfaces = np.array(read(ip)['interface_coordinates_A'])
    origin = np.isin(ref['types'], [1,2]); d = x0[:,:2]-[float(case['row']['x_start'])-100, ref['box'][1].mean()]
    cohort = np.all(abs(d) <= 25, axis=1) & (top-x0[:,2] <= 60) & (top-x0[:,2] >= 0)
    distance = np.min(abs(x0[:,2,None]-interfaces), axis=1)
    centers = np.flatnonzero(cohort & origin & (distance > 3.6) & (ref['ids']%3 == 0))
    edges = np.linspace(-25,25,4); tiles = []
    for ii,z in enumerate(interfaces):
        if not top-54 <= z <= top-6: continue
        for ix in range(3):
            for iy in range(3):
                selected = np.flatnonzero((abs(x0[:,2]-z) <= 6) &
                    (d[:,0] >= edges[ix]) & (d[:,0] < edges[ix+1]) &
                    (d[:,1] >= edges[iy]) & (d[:,1] < edges[iy+1]))
                tiles.append((ii,ix,iy,z,selected))
    return ref, centers, origin, tiles, ip


def structural_task(task):
    case, kind = task; out = OUT/f'{kind}_{case["case"]}.json'
    if out.exists():
        old = read(out)
        if old['config_sha256'] == sha(CFG) and old['script_sha256'] == sha(__file__) and all(sha(p)==h for p,h in old['sources'].items()):
            print('Verified cache', out.name, flush=True); return
        raise ValueError('Stale new-analysis output '+str(out))
    ref, centers, origin, tiles, ip = geometry(case); ly = np.diff(ref['box'][1])[0]
    sources = {str(CFG):sha(CFG), ref['source_path']:ref['sha256'], str(ip):sha(ip)}
    for helper in ('atomic_analysis.py','cycle_increment.py'):
        hp=P4/'scripts'/helper; sources[str(hp)]=sha(hp)
    selected = {}
    if kind == 'cycle':
        for arm in ('vibration','baseline'):
            selected[arm] = [i['frame'] for i in case['items'] if i['stage']=='cycle' and i['arm']==arm and i['phase_index'] in (0,8)]
    else:
        prodpath=P4/f'data/derived/production_{case["case"]}.json'; prod=read(prodpath); sources[str(prodpath)]=sha(prodpath)
        fm=rows(P4/'data/frame_manifest.csv'); bypath={r['source_path']:r for r in fm}
        selected['vibration']=[bypath[f['source_path']] for f in prod['validated_frames'][2:4]]
        bas=[r for r in fm if r['run_id']==case['baseline']['run_id']]
        selected['baseline']=[min(bas,key=lambda b:abs(float(b['distance_nm'])-float(v['distance_nm']))) for v in selected['vibration']]
        assert max(abs(float(b['distance_nm'])-float(v['distance_nm'])) for b,v in zip(selected['baseline'],selected['vibration'])) <= .016
    output = []; widths = {}; validated=[]; affine=None
    for arm, rr in [('vibration',case['row']),('baseline',case['baseline'])]:
        records=selected[arm]; positions=[]; shells=[]; field=[]; com=[]
        for r in records:
            frame=dump(r['source_path']); x=metal(frame,ref); validated.append(dict(frame_record(frame,ref),run_id=rr['run_id']))
            sources[frame['source_path']]=frame['sha256']; positions.append(x)
            com.append(frame['xyz'][frame['types']==5].mean(axis=0)[[0,2]])
            shells.append(neighbors(x,centers,3.375,ly))
            values=[]
            for ii,ix,iy,z,ids in tiles:
                xx=ref['xyz'][ids]+mic(x[ids]-ref['xyz'][ids],ly)
                value=width(ref['xyz'][ids],xx,origin[ids],z,6,.75,[.1,.5,.9])
                values.append(np.nan if value is None else value)
            field.append(values)
        expected=[expected_tool(rr,float(r['time_ps']),{}) for r in records]
        assert np.max(abs((com[1]-com[0])-(expected[1]-expected[0]))) < .005
        loss,gain,n0,n1=loss_gain(*shells); widths[arm]=np.array(field)
        output.append(dict(arm=arm,run=rr['run_id'],start_step=int(records[0]['step']),end_step=int(records[1]['step']),
            start_nm=float(records[0]['distance_nm']),end_nm=float(records[1]['distance_nm']),
            duration_ps=float(records[1]['time_ps'])-float(records[0]['time_ps']),centers=len(centers),
            retention=float(1-loss.mean()),loss=float(loss.mean()),gain=float(gain.mean()),
            coordination_start=float(n0.mean()),coordination_end=float(n1.mean()),
            normalized_coordination_change=float(((n1-n0)/n0).mean())))
        if kind=='cycle' and case['case']=='S6_R02' and arm=='vibration':
            # Deliberately synthetic homogeneous dilation, preserving every atomic identity.
            fake=positions[0]*1.02; changed=neighbors(fake,centers,3.375,ly*1.02)
            al,ag,an0,an1=loss_gain(shells[0],changed); residuals=[]
            for i,js in zip(centers,shells[0]):
                v=mic(positions[0][js]-positions[0][i],ly)
                residuals.append(fit_vectors(v,v*1.02)[1])
            affine=dict(dilation=1.02,centers=len(centers),mean_loss=float(al.mean()),mean_gain=float(ag.mean()),
                maximum_Dmin_mean_A2=float(np.nanmax(residuals)),invalid_fits=int(np.isnan(residuals).sum()),
                meaning='Synthetic geometric sensitivity, not measured actual plastic rearrangement.')
    if kind=='fixed':
        assert abs(output[0]['duration_ps']-26.176) <= .03
        assert abs(output[0]['duration_ps']-output[1]['duration_ps']) <= 1.0
    valid=np.isfinite(widths['vibration']).all(axis=0)&np.isfinite(widths['baseline']).all(axis=0)
    assert valid.sum()>0
    delta=(widths['vibration'][1]-widths['vibration'][0])-(widths['baseline'][1]-widths['baseline'][0])
    report=dict(case=case['case'],condition=case['condition'],realization=case['realization'],kind=kind,
        config_sha256=sha(CFG),script_sha256=sha(__file__),sources=sources,validated_frames=validated,arms=output,
        valid_tiles=int(valid.sum()),total_tiles=len(valid),
        width_increment_A=float(delta[valid].mean()),width_tile_increments_A=[float(v) if ok else None for v,ok in zip(delta,valid)],
        retention_difference_pp=100*(output[0]['retention']-output[1]['retention']),affine_control=affine)
    out.write_text(json.dumps(report,indent=2)+'\n'); print(kind,case['case'],'completed',flush=True)


def inventory():
    uses=[]; sourcehash={}; originrecords={}
    def add(group,run,f,context):
        identity=(run,int(f['step'])); h=f['source_sha256']; path=f['source_path']
        if identity in sourcehash: assert sourcehash[identity]==h, 'Ambiguous run/timestep identity'
        sourcehash[identity]=h
        uses.append(dict(group=group,run_id=run,step=identity[1],source_sha256=h,source_path=path,context=context))
    for p in sorted((P4/'data/derived').glob('*.json')):
        m=read(p)
        if not isinstance(m,dict) or 'mode' not in m or m['mode']=='pilot' or 'validated_frames' not in m: continue
        group='Prediction production' if m['mode']=='production' else 'Prediction sensitivity'
        for f in m['validated_frames']: add(group,m['run_id'],f,p.stem)
    for m in read(P4/'data/derived/background_manifest.json'):
        for f in m['validated_frames']: add('Early sampled background',m['run_id'],f,'4--30 ps')
    m=read(P4/'data/derived/cluster_sequence/manifest.json')
    for seq in m['sequences']:
        for f in seq['field']['endpoints']: add('Activity components',seq['run'],f,f'interval {seq["interval"]}')
    for f in m['background']['endpoints']: add('Component calibration',m['background']['run'],f,'background')
    for p in sorted((P4/'data/derived/fullfield').glob('*_station10.json')):
        m=read(p)
        for f in m['endpoints']: add('Full-field interval',m['run_id'],f,p.stem)
    for folder,group in [('cycle_structure','Cumulative cycle and late states'),('cycle_increment','Cycle increments and sensitivity')]:
        for p in sorted((P4/f'data/derived/{folder}').glob('*/manifest.json')):
            m=read(p); lookup={i['frame']['source_path']:i for i in m['case']['items']}
            for f in m['validated_frames']:
                item=lookup[f['source_path']]; add(group,item['run'],f,p.parent.name)
    for p in sorted(OUT.glob('fixed_*.json'))+sorted(OUT.glob('cycle_S*.json')):
        m=read(p)
        for f in m['validated_frames']: add('Matched-duration structure' if m['kind']=='fixed' else 'Cycle loss and gain',f['run_id'],f,p.stem)
    # Hash-check each physical source once, after the identity-level reconciliation.
    for r in uses:
        p=r['source_path']
        if p not in originrecords: originrecords[p]=sha(p)
        assert originrecords[p]==r['source_sha256']
    groups=[]
    for group in dict.fromkeys(r['group'] for r in uses):
        rr=[r for r in uses if r['group']==group]
        groups.append(dict(group=group,unique_frames=len({(r['run_id'],r['step']) for r in rr}),
                           frame_uses=len(rr),runs=len({r['run_id'] for r in rr})))
    old={ (r['run_id'],r['step']) for r in uses if r['group'] in ('Prediction production','Prediction sensitivity') }
    assert len(old)==99
    write_csv(OUT/'frame_uses.csv',uses);write_csv(OUT/'frame_coverage.csv',groups)
    report=dict(identity='run/source identity plus timestep; identical key requires identical raw hash',
        groups=groups,all_unique_frames=len(sourcehash),all_frame_uses=len(uses),original_scope_unique_frames=len(old),
        exclusions='Relaxed references and illustration-only terminal frames are excluded. Variants never multiply UNIQUE frames. Uses count each stored endpoint/phase record once, including separate prediction-sensitivity manifests; nested operations reusing a cycle record are not expanded.',
        output_hashes={n:sha(OUT/n) for n in ('frame_uses.csv','frame_coverage.csv')})
    (OUT/'frame_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['forces','structure','inventory']);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='forces':
        from contact_followup import main as matched_forces
        matched_forces()
    elif args.stage=='inventory': inventory()
    else:
        plan=[t['case'] for t in read(P4/'data/derived/cycle_increment/plan.json')['tasks'] if t['station_nm']==10]
        tasks=[(c,'fixed') for c in plan]+[(c,'cycle') for c in plan if c['case'] in read(CFG)['cycle_loss_gain_cases']]
        with ProcessPoolExecutor(max_workers=args.workers) as pool: list(pool.map(structural_task,tasks))


if __name__=='__main__': main()
