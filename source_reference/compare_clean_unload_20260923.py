"""Read-only comparison of the clean R03 withdrawal rerun with frozen analysis.

The output is a temporary diagnostic, not a replacement for historical records.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'paper4/scripts'))
from atomic_analysis import reference, dump, metal, neighbors, mic
from analyze_unloading_20260921 import fields, retain, contacts
from dynamic_order_control import assignments, statistics, REGIONS
from material_profile_check import profile
from audit_sources import sha

OLD = ROOT/'paper4/data/derived/unloading_20260921/R03'
NEW = ROOT/'merged_workspaces/CuTa-results/validation/paper4_clean_cluster_20260922_download/verified/cases/P4_UNLOAD_R03_vibration_retry3/production'
OUT = ROOT/'paper4/data/derived/clean_rerun_20260923/unload_comparison.json'
TIMES = (0, 100, 150, 175, 200)

def main():
    receipt=json.loads((NEW/'completion.json').read_text())
    assert receipt['status']=='MD_COMPLETE' and receipt['launch']['returncode']==0
    for name, digest in receipt['artifact_sha256'].items():
        assert sha(NEW/name)==digest, name
    assert sorted(int(p.name.split('.')[1]) for p in NEW.glob('unload.*.dump'))==list(range(0,200001,5000))
    force=np.loadtxt(NEW/'force_trace.dat')
    assert force.shape==(20001,7) and np.isfinite(force).all()
    np.testing.assert_array_equal(force[:,0],np.arange(0,200001,10))
    oldm=json.loads((OLD/'manifest.json').read_text())
    old=np.load(OLD/'coordinates_graphs.npz')
    oldchem=np.load(OLD/'chemical.npz')['alpha']
    ref=reference(3); x0=ref['xyz']; ly=float(np.diff(ref['box'][1])[0])
    fullix=np.searchsorted(ref['ids'],old['ids']); assert np.array_equal(ref['ids'][fullix],old['ids'])
    centers=fullix[old['centers']]
    masks=old['masks'];comps=old['compositions']
    ta=np.isin(ref['types'],[2,4]).astype(np.uint8)
    origin=np.isin(ref['types'],[1,2]); initial=neighbors(x0,centers,3.375,ly)
    oldstart=old['xyz'][5]
    states=[]; graphs=[]; widths=np.full((2,5,27),np.nan)
    for k,t in enumerate(TIMES):
        path=NEW/f'unload.{1000*t}.dump'; frame=dump(path);x=metal(frame,ref)
        assert frame['step']==1000*t and frame['n']==563827
        ns=neighbors(x,centers,3.375,ly)
        src=np.repeat(np.arange(len(centers),dtype=np.int32),[len(js) for js in ns]);dst=np.concatenate(ns).astype(np.int32)
        graphs.append((src,dst));pairs,atoms=contacts(frame,ly)
        if k==0: branch=x.copy();branch_ns=ns
        if t==150: late_start=x.copy();late_ns=ns
        ri=retain(initial,ns);di=fields(x0,x,centers,initial,ly)
        rb=retain(branch_ns,ns);db=fields(branch,x,centers,branch_ns,ly)
        late=fields(late_start,x,centers,late_ns,ly) if t==200 else None
        if k==0:
            assert np.max(abs(mic(x[fullix]-oldstart,ly)))<1e-9
            assert np.array_equal(src,old['src'][old['offsets'][5]:old['offsets'][6]])
            olddst=old['ids'][old['dst'][old['offsets'][5]:old['offsets'][6]]]
            assert np.array_equal(ref['ids'][dst],olddst)
        regions={}
        for j,region in enumerate(REGIONS):
            use=masks[j]; selected=use[src]&ta[centers[src]].astype(bool)
            M=int(selected.sum());Q=int(ta[dst[selected]].sum())
            regions[region]=dict(retention=float(ri[use].mean()),D_initial_A2=float(di[use].mean()),
                                 branch_retention=float(rb[use].mean()),D_branch_A2=float(db[use].mean()),
                                 D_late_A2=float(late[use].mean()) if late is not None else None,Q=Q,M=M,
                                 alpha_observed=float(1-Q/M/comps[j]))
        for ti,info in enumerate(old['tile_info']):
            ix=fullix[old['tile_atoms'][old['tile_offsets'][ti]:old['tile_offsets'][ti+1]]]
            xx=x0[ix]+mic(x[ix]-x0[ix],ly)
            for ei,material in enumerate((False,True)):
                q=profile(x0[ix],xx,origin[ix],info[3],material);widths[ei,k,ti]=q['width']
        states.append(dict(time_ps=t,source_sha256=frame['sha256'],contact_pairs=pairs,contact_atoms=atoms,regions=regions))
        print('R03 clean vibration',t,'ps: geometry/graph complete',flush=True)
    common=np.isfinite(widths).all(axis=(0,1))&np.isfinite(old['widths'][:,0]).all(axis=(0,1))&np.isfinite(old['relaxed_widths']).all(0)
    assert common.any()
    for k,t in enumerate(TIMES):
        states[k]['width']=dict(physical_mean_A=float((widths[0,k,common]-old['widths'][0,0,k,common]).mean()),
                                material_mean_A=float((widths[1,k,common]-old['widths'][1,0,k,common]).mean()),
                                material_median_A=float(np.median(widths[1,k,common]-old['widths'][1,0,k,common])))
    # Regenerate exactly the frozen full-workpiece depth-conditioned assignments.
    support=np.unique(np.concatenate([centers]+[dst for _,dst in graphs]))
    localc=np.searchsorted(support,centers)
    localgraphs=[(src,np.searchsorted(support,dst)) for src,dst in graphs]
    strata=np.floor((x0[:,2]-x0[:,2].min())/2).astype(int)*2+origin.astype(int)
    alpha=[];seeds=[]
    for batch in range(1,5):
        archived=np.load(OLD/f'batch{batch}.npz');seed=int(archived['seed']);seeds.append(seed)
        full=assignments(ta,strata,256,seed)
        np.testing.assert_array_equal(full[1:,fullix],archived['labels'])
        labels=full[:,support];del full
        values=np.array([statistics(labels,localc,src,dst,masks,comps) for src,dst in localgraphs])
        assert np.isfinite(values).all()
        if batch==1:
            np.testing.assert_allclose(values[0,0],oldchem[1,0,0],atol=1e-12,rtol=0)
            alpha.append(values)
        else:alpha.append(values[:,1:])
        print('R03 clean vibration depth-label batch',batch,'complete',flush=True)
    chem=np.concatenate(alpha,axis=1)
    np.testing.assert_allclose(chem[0],oldchem[1,0],atol=1e-12,rtol=0)
    for k in range(len(TIMES)):
        for j,region in enumerate(REGIONS):
            v=chem[k,:,j];b=oldchem[0,k,:,j];paired=v-b
            states[k]['regions'][region].update(alpha_null_mean=float(v[1:].mean()),
                chemical_excess=float(v[0]-v[1:].mean()),paired_excess=float(paired[0]-paired[1:].mean()),
                paired_mcse=float(paired[1:].std(ddof=1)/np.sqrt(1024)))
    for j,region in enumerate(REGIONS):
        late_paired=(chem[4,:,j]-oldchem[0,4,:,j])-(chem[2,:,j]-oldchem[0,2,:,j])
        states[4]['regions'][region].update(
            paired_late_change=float(late_paired[0]-late_paired[1:].mean()),
            paired_late_mcse=float(late_paired[1:].std(ddof=1)/np.sqrt(1024)))
    old_reg={}
    import csv
    with (OLD/'regional.csv').open() as f:
        for row in csv.DictReader(f):
            if row['arm']=='vibration':old_reg[(int(row['time_ps']),row['region'])]=row
    comparisons=[]
    for k,t in enumerate(TIMES):
        for region in REGIONS:
            o=old_reg[t,region];n=states[k]['regions'][region]
            comparisons.append(dict(time_ps=t,region=region,
                                    retention_new_minus_old_pp=100*(n['retention']-float(o['initial_retention'])),
                                    D_new_minus_old_A2=n['D_initial_A2']-float(o['D_initial_A2']),
                                    Q_new_minus_old=n['Q']-int(o['Q']),M_new_minus_old=n['M']-int(o['M'])))
    result=dict(analysis_sha256=sha(__file__),scope='Clean rerun versus archived R03 A50B20 withdrawal; same initial state, different backend, not new preparation',
        old_manifest_sha256=sha(OLD/'manifest.json'),new_completion_sha256=sha(NEW/'completion.json'),
        old_status=oldm['checks'][1]['status'],new_status='clean_exit_0',
        initial_state_equal=True,original_depth_seeds=seeds,common_tiles=np.flatnonzero(common).tolist(),
        states=states,comparisons=comparisons)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print('WROTE',OUT,flush=True)

if __name__=='__main__':main()
