"""Versioned analysis of six saved withdrawal/hold branches. Never runs MD."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from atomic_analysis import reference, dump, metal, neighbors, mic, fit_vectors
from dynamic_order_control import assignments, statistics, REGIONS
from material_profile_check import profile
from audit_sources import ROOT, P4, sha, write_csv

OUT = P4/'data/derived/unloading_20260921'
E = P4/'review_evidence_20260921/evidence'
CAMPAIGN = ROOT/'development/paper14_followup_20260919/campaign.json'
TIMES = (0, 100, 150, 175, 200)

def read(p): return json.loads(Path(p).read_text())
def save(p, value): Path(p).write_text(json.dumps(value, indent=2)+'\n')

def fields(x0, x, centers, ns, ly):
    return np.array([fit_vectors(mic(x0[js]-x0[i], ly), mic(x[js]-x[i], ly), 4, 1e6)[1]
                     for i, js in zip(centers, ns)])

def retain(first, last):
    return np.array([len(np.intersect1d(a,b))/len(a) if len(a) else np.nan for a,b in zip(first,last)])

def contacts(f, ly):
    tool=f['xyz'][f['types']==5]; m=f['types']!=5
    x=f['xyz'][m]; types=f['types'][m]
    tree=cKDTree(np.concatenate([tool+[0,s,0] for s in (-ly,0,ly)]))
    counts=tree.query_ball_point(x,np.where(np.isin(types,[1,3]),5.4,7.164),return_length=True,workers=1)
    return int(counts.sum()), int(np.count_nonzero(counts))

def analyze(r):
    target=OUT/f'R{r:02d}'; target.mkdir(parents=True,exist_ok=True)
    if (target/'manifest.json').exists():
        m=read(target/'manifest.json')
        assert m['analysis_sha256']==sha(__file__),'Analysis changed: use a new output version'
        for p,h in m['sources'].items(): assert sha(p)==h,p
        for p,h in m['outputs'].items(): assert sha(target/p)==h,p
        print('R',r,'verified cache',flush=True);return
    started=time.time();c=read(CAMPAIGN);ref=reference(r);x0=ref['xyz'];ly=float(np.diff(ref['box'][1])[0])
    prior=E/'primary'/f'S6_R{r:02d}'
    a=np.load(prior/'coordinates.npz'); meta=read(prior/'coordinates.json')
    oldg=np.load(prior/'graphs.npz');gm=read(prior/'graph_manifest.json')
    packet_ix=np.searchsorted(ref['ids'],a['ids']);assert np.array_equal(ref['ids'][packet_ix],a['ids'])
    np.testing.assert_allclose(x0[packet_ix],a['xyz'][0],atol=0,rtol=0)
    centers=packet_ix[a['centers']];masks=a['masks'];compositions=a['compositions']
    assert np.array_equal(a['ids'][a['centers']],oldg['support_ids'][oldg['centers']])
    ta=np.isin(ref['types'],[2,4]).astype(np.uint8);origin=np.isin(ref['types'],[1,2])
    initial=neighbors(x0,centers,3.375,ly)
    sources={str(CAMPAIGN):sha(CAMPAIGN),ref['source_path']:ref['sha256']}
    for p in [prior/n for n in ('coordinates.npz','coordinates.json','graphs.npz','graph_manifest.json')]: sources[str(p)]=sha(p)
    for n in ('atomic_analysis.py','dynamic_order_control.py','material_profile_check.py','audit_sources.py'):
        p=P4/'scripts'/n;sources[str(p)]=sha(p)
    sources[str(P4/'config/material_spatial_20260916.json')]=sha(P4/'config/material_spatial_20260916.json')
    observations=[];profile_rows=[];state_rows=[];graphs=[];positions=[];per_atom=[];checks=[]
    widths=np.full((2,2,len(TIMES),27),np.nan);relaxed_width=np.full((2,27),np.nan)
    for ti,info in enumerate(a['tile_info']):
        ix=packet_ix[a['tile_atoms'][a['tile_offsets'][ti]:a['tile_offsets'][ti+1]]]
        for ei,material in enumerate((False,True)):
            q=profile(x0[ix],x0[ix],origin[ix],info[3],material);relaxed_width[ei,ti]=q['width']
    for ai,arm in enumerate(('baseline','vibration')):
        seg=next(s for s in c['segments'] if s['case']==f'P4_UNLOAD_R{r:02d}_{arm}')
        assert seg['reference_sha256']==ref['sha256']
        d=Path(seg['directory'])/'production';launch=read(d/'launch.json')
        for name in ('launch.json','lammps.log','force_trace.dat','final.dump'):
            sources[str(d/name)]=sha(d/name)
        if r==3 and arm=='vibration':
            recp=ROOT/'development/paper4_recovery_20260921/records/recovery.json';rec=read(recp)
            assert rec['status']=='RECOVERED_OUTPUTS_VALIDATED' and launch['returncode']==1
            for p,h in rec['original_hashes'].items():
                if Path(p).parent==d and Path(p).name in ('lammps.log','force_trace.dat','final.dump'):assert sha(p)==h
            assert rec['frozen_operational_checks']['status']=='PASS';sources[str(recp)]=sha(recp)
            status='recovered_outputs_exit_1'
        else:
            val=read(d/'validation.json'); assert val['status']=='PASS' and launch['returncode']==0
            for k,n in [('log_sha256','lammps.log'),('final_dump_sha256','final.dump')]:assert val[k]==sha(d/n)
            sources[str(d/'validation.json')]=sha(d/'validation.json');status='clean_exit_0'
        force=np.loadtxt(d/'force_trace.dat'); assert force.shape[0]==20001 and np.isfinite(force).all()
        np.testing.assert_array_equal(force[:,0],np.arange(0,200001,10))
        assert sorted(int(p.name.split('.')[1]) for p in d.glob('unload.*.dump'))==list(range(0,200001,5000))
        oldstate=next(i for i,s in enumerate(meta['states']) if s['arm']==arm and s['phase_index']==8)
        oldgraph=next(i for i,s in enumerate(gm['states']) if s['arm']==arm and s['phase_index']==8)
        for k,t in enumerate(TIMES):
            p=d/f'unload.{t*1000}.dump';f=dump(p);x=metal(f,ref);assert f['step']==t*1000
            sources[str(p)]=f['sha256'];ns=neighbors(x,centers,3.375,ly)
            src=np.repeat(np.arange(len(centers),dtype=np.int32),[len(js) for js in ns]);dst=np.concatenate(ns).astype(np.int32)
            graphs.append((src,dst));positions.append(x.copy())
            if k==0:
                branch=x.copy();branch_ns=ns
                err=float(np.max(abs(mic(x[packet_ix]-a['xyz'][oldstate],ly))));assert err<1e-9
                lo,hi=oldg['offsets'][oldgraph:oldgraph+2]
                assert np.array_equal(src,oldg['src'][lo:hi])
                assert np.array_equal(ref['ids'][dst],oldg['support_ids'][oldg['dst'][lo:hi]])
                checks.append(dict(arm=arm,loaded_coordinate_max_error_A=err,loaded_graph_exact=True,force_rows=len(force),status=status))
            if t==150:late_start=x.copy();late_ns=ns
            ri=retain(initial,ns);rb=retain(branch_ns,ns)
            di=fields(x0,x,centers,initial,ly);db=fields(branch,x,centers,branch_ns,ly)
            late=fields(late_start,x,centers,late_ns,ly) if t==200 else np.full(len(centers),np.nan)
            assert np.isfinite(ri).all() and np.isfinite(rb).all() and np.isfinite(di).all() and np.isfinite(db).all()
            if t==200: assert np.isfinite(late).all()
            pairs,atoms=contacts(f,ly)
            state_rows.append(dict(realization=r,arm=arm,time_ps=t,hold_ps=max(t-100,0),atoms=f['n'],cutoff_pairs=pairs,contacting_atoms=atoms,exit_status=status))
            for j,region in enumerate(REGIONS):
                use=masks[j]; central=use[src]&ta[centers[src]].astype(bool); M=int(central.sum());Q=int(ta[dst[central]].sum())
                observations.append(dict(realization=r,arm=arm,time_ps=t,region=region,centers=int(use.sum()),
                    initial_retention=float(ri[use].mean()),branch_retention=float(rb[use].mean()),
                    D_initial_A2=float(di[use].mean()),D_branch_A2=float(db[use].mean()),
                    D_late_A2=float(late[use].mean()),M=M,Q=Q,p=Q/M,alpha=1-Q/M/compositions[j]))
            per_atom.append(np.stack([ri,rb,di,db,late]))
            for ti,info in enumerate(a['tile_info']):
                ix=packet_ix[a['tile_atoms'][a['tile_offsets'][ti]:a['tile_offsets'][ti+1]]]
                xx=x0[ix]+mic(x[ix]-x0[ix],ly)
                for ei,material in enumerate((False,True)):
                    q=profile(x0[ix],xx,origin[ix],info[3],material);widths[ei,ai,k,ti]=q['width']
                    profile_rows.append(dict(realization=r,arm=arm,time_ps=t,tile=ti,interface=int(info[0]),estimator=('physical','material')[ei],
                        valid=q['valid'],reason=q['reason'],width_A=q['width'],stretch=q['stretch'],cross10_A=q['crossings'][0],cross50_A=q['crossings'][1],cross90_A=q['crossings'][2]))
            print(f'R{r:02d} {arm} {t} ps: {pairs} contact pairs; geometry complete',flush=True)
    # Freeze reduced full-neighbor-support data, including all reference-shell and profile atoms.
    support=np.unique(np.concatenate([centers,packet_ix]+[dst for _,dst in graphs]+initial))
    offsets=np.r_[0,np.cumsum([len(s) for s,_ in graphs])]
    reduced=dict(ids=ref['ids'][support],types=ref['types'][support],xyz0=x0[support],
        xyz=np.array([x[support] for x in positions]),centers=np.searchsorted(support,centers),
        masks=masks,compositions=compositions,ly=ly,src=np.concatenate([s for s,_ in graphs]),
        dst=np.searchsorted(support,np.concatenate([d for _,d in graphs])),offsets=offsets,
        tile_atoms=np.searchsorted(support,packet_ix[a['tile_atoms']]),tile_offsets=a['tile_offsets'],tile_info=a['tile_info'],
        per_atom=np.array(per_atom),widths=widths,relaxed_widths=relaxed_width)
    del positions
    np.savez_compressed(target/'coordinates_graphs.npz',**reduced)
    # Regenerate the original four independent depth-only batches over the entire material.
    strata=np.floor((x0[:,2]-x0[:,2].min())/2).astype(int)*2+origin.astype(int)
    olddir=E/'review_20260919'/f'S6_R{r:02d}';og=np.load(olddir/'graphs.npz')
    oldsupport=np.searchsorted(ref['ids'],og['support_ids'])
    assert np.array_equal(ref['ids'][oldsupport],og['support_ids'])
    alpha=[];seeds=[]
    for b in range(1,5):
        old=np.load(olddir/f'depth_only_batch{b}.npz');seed=int(old['seed']);seeds.append(seed)
        full=assignments(ta,strata,256,seed)
        assert np.array_equal(full[1:,oldsupport],old['labels'])
        labels=full[:,support];del full
        values=np.array([statistics(labels,reduced['centers'],reduced['src'][lo:hi],reduced['dst'][lo:hi],masks,compositions)
                         for lo,hi in zip(offsets[:-1],offsets[1:])])
        assert np.isfinite(values).all()
        np.savez_compressed(target/f'batch{b}.npz',labels=labels[1:],alpha=values[:,1:],seed=seed)
        alpha.append(values if b==1 else values[:,1:])
        print(f'R{r:02d} exact original depth assignments batch {b}: PASS',flush=True)
        sources[str(olddir/f'depth_only_batch{b}.npz')]=sha(olddir/f'depth_only_batch{b}.npz')
    vals=np.concatenate(alpha,axis=1).reshape(2,5,1025,4)
    for ai,arm in enumerate(('baseline','vibration')):
        # Compare full ensemble with the primary, already accepted loaded endpoint.
        oldalpha=np.load(prior/'turnover.npz')['alpha'][next(i for i,s in enumerate(gm['states']) if s['arm']==arm and s['phase_index']==8)]
        np.testing.assert_allclose(vals[ai,0],oldalpha,atol=1e-12,rtol=0)
    np.savez_compressed(target/'chemical.npz',alpha=vals,paired=vals[1]-vals[0],seeds=seeds)
    write_csv(target/'regional.csv',observations);write_csv(target/'profiles.csv',profile_rows);write_csv(target/'states.csv',state_rows)
    common=np.isfinite(widths).all(axis=(0,1,2))&np.isfinite(relaxed_width).all(0)
    save(target/'manifest.json',dict(realization=r,analysis_sha256=sha(__file__),sources=sources,
        states=state_rows,checks=checks,centers=len(centers),regional_centers=masks.sum(1).tolist(),
        support_atoms=len(support),full_atoms=len(x0),common_tiles=np.flatnonzero(common).tolist(),
        excluded_tiles=np.flatnonzero(~common).tolist(),depth_batch_seeds=seeds,seconds=time.time()-started,
        outputs={p.name:sha(p) for p in target.iterdir() if p.is_file() and p.name!='manifest.json'}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--realization',type=int,choices=[1,2,3]);args=p.parse_args()
    for r in ([args.realization] if args.realization else (1,2,3)):analyze(r)

if __name__=='__main__':main()
