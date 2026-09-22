"""Extract existing local evidence for the portable Paper 4 review audit."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json, argparse
import shutil
from pathlib import Path
import numpy as np
from atomic_analysis import reference, dump
from audit_sources import P4, sha
from reviewer2_audit_20260921 import read, write, save

OUT=P4/'data/derived/reviewer2_20260921'
CONDITIONS=['Baseline','A10B05','A10B20','A25B10','A25B20','A50B20']

def manifests():
    result={}
    roots=[P4/'data/derived/sequence_campaign_20260912',P4/'data/auxiliary_completion_20260917/sequence',
           P4/'data/replication_extension_20260917/data/derived/sequence']
    for root in roots:
        for path in sorted(root.glob('*.json')):
            m=read(path)
            if 'selection_files' not in m or 'rows' not in m:continue
            row=m['rows'][0];key=(row['condition'],int(row['realization']))
            # ADH100_R03 is the archived unit-strength A25B20 R03 identity.
            result[key]=(path,m)
    assert len(result)==18,len(result)
    return result

def depths():
    refs={r:reference(r) for r in (1,2,3)};records=[];sources={}
    packet=OUT/'activity';packet.mkdir(exist_ok=True)
    for (condition,r),(mp,m) in manifests().items():
        ref=refs[r];top=float(ref['xyz'][:,2].max());sources[str(mp.relative_to(P4))]=sha(mp)
        assert abs(top-m['initial_surface_z_A'])<1e-10
        for path,digest in m['selection_files'].items():
            src=P4/path;assert sha(src)==digest
            a=np.load(src);ix=np.searchsorted(ref['ids'],a['ids']);np.testing.assert_array_equal(ref['ids'][ix],a['ids'])
            interval=int(src.stem.rsplit('interval',1)[1]);row=m['rows'][interval-1]
            di=np.maximum(0,top-ref['xyz'][ix,2])/10;de=np.maximum(0,top-a['xyz'][:,2])/10
            assert abs(float(np.quantile(de,.95))-row['depth_p95_nm'])<1e-10
            np.savez_compressed(packet/(src.stem+'.npz'),ids=a['ids'],xyz=a['xyz'],activity=a['activity'],labels=a['labels'],sizes=a['sizes'],initial_depth_nm=di,endpoint_depth_nm=de)
            records.append(dict(condition=condition,realization=r,run=row['run'],interval=interval,
                active_atoms=len(di),start_nm=row['start_nm'],end_nm=row['end_nm'],
                d95_endpoint_nm=float(np.quantile(de,.95)),d95_initial_nm=float(np.quantile(di,.95)),
                source_sha256=digest))
        if r==2:
            ep=m['endpoints'][1];frame=dump(ep['source_path']);assert frame['sha256']==ep['source_sha256']
            np.savez_compressed(packet/(m['rows'][0]['run']+'_context.npz'),ids=frame['ids'],types=frame['types'],xyz=frame['xyz'])
    assert len(records)==54;write(OUT/'depth_definitions.csv',records)
    save(OUT/'depth_sources.json',dict(sources=sources,initial_sources={str(ref['source_path']):ref['sha256'] for ref in refs.values()},script_sha256=sha(__file__)))
    print('Both depth definitions reproduced for 54 fixed active-ID sets',flush=True)

def initial():
    dest=OUT/'initial';dest.mkdir(exist_ok=True);provenance=[];pilot=[]
    for r in (1,2,3):
        a=reference(r);np.savez_compressed(dest/f'R{r:02d}.npz',ids=a['ids'],types=a['types'],xyz=a['xyz'],box=a['box'])
        provenance.append(dict(realization=r,source_path=a['source_path'],source_sha256=a['sha256'],atoms=len(a['ids'])))
    for run in ('scratch_base_R01','S6_R01'):
        p=P4/f'data/derived/pilot_{run}.npz';prod=P4/f'data/derived/production_{run}.npz'
        a=np.load(p)['data'];b=np.load(prod)['data'];ka={tuple(v[:2]) for v in a};kb={tuple(v[:2])for v in b}
        pilot.append(dict(run=run,pilot_observations=len(a),production_observations=len(b),overlap_atom_station=len(ka&kb),pilot_sha256=sha(p),production_sha256=sha(prod)))
        shutil.copy2(p,dest/p.name)
    save(OUT/'initial_provenance.json',provenance);save(OUT/'pilot_overlap.json',pilot)
    print('Full initial coordinate populations and pilot overlap indexed',pilot,flush=True)

def render():
    # Render only archived active atoms with a documented, fixed context subset.
    os.environ['QT_QPA_PLATFORM']='offscreen'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import colorsys
    from functools import partial
    import render_illustrative_evidence as vis
    assert vis.ovito.version_string=='3.15.5'
    dest=P4/'figures/main/reviewer2_20260921';dest.mkdir(parents=True,exist_ok=True)
    vis.OUT=dest
    vis.TachyonRenderer=partial(vis.TachyonRenderer,antialiasing_samples=2,ambient_occlusion_samples=2)
    refs=reference(2);top=refs['xyz'][:,2].max();records=[]
    norm=Normalize(0,np.log1p(10/.01));cmap=plt.get_cmap('turbo')
    center_whole=np.array([234.0,78.0,116.0])
    center_slab=np.array([395.0,78.0,118.0])
    half=np.array([45.,35.,60.])
    for k,condition in enumerate(CONDITIONS):
        mp,m=manifests()[condition,2];row=m['rows'][0];run=row['run'];a=np.load(OUT/f'activity/{run}_interval1.npz')
        ctx=np.load(OUT/f'activity/{run}_context.npz');tool=ctx['xyz'][ctx['types']==5]
        center_cut=np.array([tool[:,0].mean(),tool[:,1].mean(),115.0])
        colors=cmap(norm(np.log1p(a['activity']/.01)))[:,:3]
        whole_xyz=np.vstack([a['xyz'],tool])
        whole_colors=np.vstack([colors,np.tile([.40,.43,.46],(len(tool),1))])
        whole_radii=np.concatenate([np.full(len(a['xyz']),2.0),np.full(len(tool),1.05)])
        whole=vis.render('case'+str(k)+'_whole',whole_xyz,whole_colors,target=center_whole,fov=152,size=(1400,850),radius=whole_radii)
        use=np.all(abs(a['xyz']-center_cut)<=half,axis=1);tm=np.all(abs(tool-center_cut)<=half,axis=1)
        slab_xyz=np.vstack([a['xyz'][use],tool[tm]])
        slab_colors=np.vstack([colors[use],np.tile([.40,.43,.46],(tm.sum(),1))])
        slab_radii=np.concatenate([np.full(use.sum(),1.2),np.full(tm.sum(),1.05)])
        heat=vis.render('case'+str(k)+'_activity',slab_xyz,slab_colors,target=center_slab,fov=68,size=(1400,900),radius=slab_radii)
        rank=np.empty(len(a['sizes']),int);rank[np.argsort(-a['sizes'],kind='stable')]=np.arange(len(rank))
        hues=np.array([colorsys.hsv_to_rgb(float(j*.61803398875%1),.60,.82) for j in rank[a['labels'][use]]])
        hues[rank[a['labels'][use]]==0]=[.72,.12,.42]
        component=vis.render('case'+str(k)+'_components',np.vstack([a['xyz'][use],tool[tm]]),
            np.vstack([hues,np.tile([.40,.43,.46],(tm.sum(),1))]),target=center_slab,fov=68,size=(1400,900),radius=slab_radii)
        fig,ax=plt.subplots(figsize=(5.2,2.9));vis.image_axis(ax,whole,10)
        fig.subplots_adjust(top=.99,bottom=.01,left=.01,right=.99)
        fig.savefig(dest/f'case{k}_whole.pdf');fig.savefig(dest/f'case{k}_whole.png',dpi=220);plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(4.4,5.0))
        vis.image_axis(axes[0],heat,2);vis.image_axis(axes[1],component,2)
        axes[0].set_title('Activity of selected atoms',fontsize=10,pad=2);axes[1].set_title('Connected components',fontsize=10,pad=2)
        fig.subplots_adjust(top=.96,bottom=.02,left=.02,right=.98,hspace=.10)
        fig.savefig(dest/f'case{k}_slabs.pdf');fig.savefig(dest/f'case{k}_slabs.png',dpi=220);plt.close(fig)
        records.append(dict(condition=condition,run=run,interval=1,steps=[ep['step'] for ep in m['endpoints'][:2]],
            start_nm=row['start_nm'],end_nm=row['end_nm'],duration_ps=(m['endpoints'][1]['step']-m['endpoints'][0]['step'])*.001,
            threshold_A2=row['threshold_A2'],slice_halfwidth_A=half.tolist(),center_A=center_cut.tolist(),eligible_initial_depth_A=60,
            full_active_atoms=len(a['ids']),displayed_active_atoms=int(use.sum()),largest_component_full_population=int(a['sizes'].max()),
            source_selection_sha256=sha(OUT/f'activity/{run}_interval1.npz'),context_sha256=sha(OUT/f'activity/{run}_context.npz')))
        print('OVITO figures',condition,'complete',flush=True)
    save(OUT/'activity_render.json',dict(ovito=vis.ovito.version_string,script_sha256=sha(__file__),records=records,
        views=vis.records,context='Neutral context omitted to maximize visual clarity of active atoms and tool; full active sets, no statistical thinning.',
        colors='Continuous activity only for selected high-activity atoms; magenta is largest full-population component, other component hues local to each case.'))

def controls():
    from audit_sources import moving_thermo_blocks
    from reviewer2_audit_20260921 import rows
    root=P4/'review_evidence_20260919/evidence/supporting/physical_controls'
    dest=OUT/'physical_forces';dest.mkdir(exist_ok=True);result=[]
    for q in read(root/'plan.json')['cases']:
        case=q['case'];lp=next(Path(p) for p in q['source_hashes'] if p.endswith('/logs/lammps.log'))
        mp=next(Path(p) for p in q['source_hashes'] if p.endswith('/metadata/launch.json'))
        assert sha(lp)==q['source_hashes'][str(lp)] and sha(mp)==q['source_hashes'][str(mp)]
        text,blocks=moving_thermo_blocks(lp);assert len(blocks)==1
        head,a=blocks[0];samples=a[:,[head.index(k) for k in ('Step','Time','v_tool_x','v_fx','v_fz')]]
        t=samples[:,1];start=q['t0_ps']+q['ramp_ps']+4/.03;end=q['t0_ps']+q['ramp_ps']+16/.03
        assert t[0]<=start<end<=t[-1] and np.all(np.diff(t)>0)
        rec=dict(case=case,realization=q['realization'],value=q['value'],family=q['family'],start_ps=start,end_ps=end,
            covered_start_nm=.03*(t[0]-q['t0_ps']-q['ramp_ps']),covered_end_nm=.03*(t[-1]-q['t0_ps']-q['ramp_ps']),
            first_force_step=int(samples[0,0]),last_force_step=int(samples[-1,0]),
            interior_records=int(((t>start)&(t<end)).sum()),launcher_returncode=q['launcher_returncode'],
            completion_marker=q['completion_marker'],log_sha256=sha(lp),launch_sha256=sha(mp))
        for field,col in [('Ft',3),('Fn',4)]:
            use=(t>start)&(t<end);tt=np.r_[start,t[use],end];values=samples[:,col]*1.602176634
            yy=np.r_[np.interp(start,t,values),values[use],np.interp(end,t,values)]
            val=float(np.trapezoid(yy,tt)/(end-start));assert abs(val-q['force'][field+'_mean_nN'])<1e-10
            rec[field+'_mean_nN']=val
        np.savez_compressed(dest/(case+'.npz'),samples=samples)
        shutil.copy2(lp,dest/(case+'.log'));shutil.copy2(mp,dest/(case+'_launch.json'));result.append(rec)
    write(OUT/'physical_force_coverage.csv',result)
    contrasts=rows(root/'contrasts.csv');without=[q for q in contrasts if q['family']=='adhesion' and q['realization']!='3']
    for q in without:
        assert (float(q['Ft_mean_nN'])<0)==(float(q['value'])<1)
        assert (float(q['dmin2_A2'])<0)==(float(q['value'])<1)
        assert (float(q['cycle_retention_pct'])>0)==(float(q['value'])<1)
    save(OUT/'physical_exclusion_check.json',dict(excluded='Entire R03 strength series for conservative check',
        unresolved_launcher='ADH050_R03 returncode 1 unchanged',all_four_R01_R02_strength_contrasts_retain_force_activity_retention_directions=True,
        records=without))
    print('Physical-control force windows reconstructed',len(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['depths','render','controls','initial']);args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    globals()[args.stage]()
