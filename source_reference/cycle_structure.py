"""Phase-resolved structural descriptors on fixed cohorts; never modify raw MD."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from atomic_analysis import reference, dump, metal, neighbors, mic, frame_record, expected_tool
from audit_sources import P4, ROOT, sha, write_csv

CFG=P4/'config/cycle_structure.json'
OUT=P4/'data/derived/cycle_structure'

def interface_width(x0,x,origin,z0,half,bandwidth):
    """Local-plane normal and real-space smoothed origin-fraction crossings."""
    if len(x0)<50: return None
    center=x0.mean(axis=0); J=np.linalg.lstsq(x0-center,x-x.mean(axis=0),rcond=None)[0]
    if np.linalg.cond(J)>100: return None
    normal=np.linalg.solve(J,np.array([0.,0.,1.]))
    stretch=1/np.linalg.norm(normal); normal*=stretch
    d=(x-x.mean(axis=0))@normal+(center[2]-z0)*stretch
    sign=1 if np.mean(origin[x0[:,2]>z0])>np.mean(origin[x0[:,2]<z0]) else -1
    label=origin if sign==1 else 1-origin
    grid=np.linspace(-half*stretch,half*stretch,161)
    weights=np.exp(-.5*((grid[:,None]-d)/bandwidth)**2)
    counts=weights.sum(axis=1); profile=(weights@label)/np.maximum(counts,1e-100)
    crossings=[]
    for level in (.1,.5,.9):
        ix=np.flatnonzero((profile[:-1]<level)&(profile[1:]>=level)&(counts[:-1]>=3)&(counts[1:]>=3))
        if not len(ix): return None
        vals=grid[ix]+(level-profile[ix])*(grid[ix+1]-grid[ix])/(profile[ix+1]-profile[ix])
        crossings.append(float(vals[np.argmin(abs(vals))]))
    if not crossings[0]<crossings[1]<crossings[2]: return None
    return dict(width_A=crossings[2]-crossings[0],normal_stretch=stretch,
                stretch_corrected_width_A=(crossings[2]-crossings[0])/stretch,
                midplane_shift_A=crossings[1],atoms=len(x0))

def make_plan(cfg):
    rows=list(csv.DictReader((P4/'data/run_inventory.csv').open()))
    rows=[r for r in rows if r['analysis_eligibility']=='selected endpoint intervals validated']
    frames=list(csv.DictReader((P4/'data/frame_manifest.csv').open())); plan=[]
    for row in rows:
        if row['condition']=='Baseline': continue
        base=next(r for r in rows if r['condition']=='Baseline' and r['realization']==row['realization'])
        f=float(row['frequency_per_ps']); phase=float(row['phase_origin']); start=float(row['translation_start_ps'])
        target=start+cfg['cycle_center_target_nm']/.03
        cycle=round(f*target+phase/(2*np.pi)-.5)
        zero=(2*np.pi*cycle-phase)/(2*np.pi*f)
        own=[r for r in frames if r['run_id']==row['run_id']]; bf=[r for r in frames if r['run_id']==base['run_id']]
        times=np.array([float(r['time_ps']) for r in own]); bd=np.array([float(r['distance_nm']) for r in bf])
        items=[]
        for phase_index,theta in enumerate(np.linspace(0,2*np.pi,cfg['phase_count'])):
            wanted=zero+theta/(2*np.pi*f); q=own[int(np.argmin(abs(times-wanted)))]; t=float(q['time_ps'])
            actual=2*np.pi*f*(t-zero); travel=float(q['distance_nm'])
            if abs(actual-theta)>cfg['maximum_phase_error_rad'] or not 4<=travel<=16: raise ValueError('Inadequate cycle sampling '+row['run_id'])
            b=bf[int(np.argmin(abs(bd-travel)))]; error=float(b['distance_nm'])-travel
            if abs(error)>cfg['maximum_baseline_travel_error_nm']: raise ValueError('Baseline travel mismatch')
            for arm,rr,fr in [('vibration',row,q),('baseline',base,b)]:
                items.append(dict(run=rr['run_id'],arm=arm,frame=fr,phase_index=phase_index,
                    target_phase_rad=float(theta),actual_vibration_phase_rad=actual,phase_error_rad=actual-theta,
                    travel_match_error_nm=error,stage='cycle'))
        post=min(own,key=lambda a:abs(float(a['distance_nm'])-cfg['postpass_target_nm']))
        post_eligible=abs(float(post['distance_nm'])-cfg['postpass_target_nm'])<=.06
        postbase=min(bf,key=lambda a:abs(float(a['distance_nm'])-float(post['distance_nm'])))
        for arm,rr,fr in ([('vibration',row,post),('baseline',base,postbase)] if post_eligible else []):
            items.append(dict(run=rr['run_id'],arm=arm,frame=fr,phase_index=-1,target_phase_rad=None,
                actual_vibration_phase_rad=None,phase_error_rad=None,travel_match_error_nm=None,stage='postpass'))
        plan.append(dict(case=row['run_id'],condition=row['condition'],realization=int(row['realization']),
            row=row,baseline=base,items=items,cycle_period_ps=1/f,
            postpass_eligible=post_eligible,postpass_exclusion='' if post_eligible else 'No snapshot within 0.06 nm of 16 nm'))
    return plan

def analyze(case,cfg):
    out=OUT/case['case']; out.mkdir(parents=True,exist_ok=True)
    ref=reference(case['realization']); x0=ref['xyz']; top=x0[:,2].max(); ly=np.diff(ref['box'][1])[0]
    assert case['row']['reference_matches_launch']=='True' and ref['sha256']==case['row']['reference_sha256']
    ta=np.isin(ref['types'],[2,4]); origin=np.isin(ref['types'],[1,2]); h=cfg['cohort_halfwidth_xy_A']
    center=np.array([float(case['row']['x_start'])-100,ref['box'][1].mean()])
    dx=x0[:,:2]-center; lateral=np.all(abs(dx)<=h,axis=1)
    allcohort=lateral&(top-x0[:,2]<=cfg['cohort_depth_A'])&(top-x0[:,2]>=0)
    centers=np.flatnonzero(allcohort&(ref['ids']%cfg['central_id_modulus']==0))
    ns0=neighbors(x0,centers,cfg['cutoff_A'],ly)
    imeta=ROOT/f'results/runs/relax_laminate_R{case["realization"]:02d}/metadata/relaxed_structure.manifest.json'
    interfaces=np.array(json.loads(imeta.read_text())['interface_coordinates_A'])
    distance=np.min(abs(x0[:,2,None]-interfaces),axis=1)
    masks=[(allcohort&(origin==o)&((distance>3.6) if interior else (distance<=3.6)),name)
        for o,name0 in [(True,'Cu-rich'),(False,'Ta-rich')] for interior,name in [(True,name0+' interior'),(False,name0+' interface-near')]]
    tiles=[]; edge=np.linspace(-h,h,cfg['interface_tiles_per_axis']+1)
    for interface,z in enumerate(interfaces):
        if not top-cfg['cohort_depth_A']+cfg['interface_halfwidth_A']<=z<=top-cfg['interface_halfwidth_A']: continue
        for ix in range(len(edge)-1):
            for iy in range(len(edge)-1):
                ixm=np.flatnonzero((abs(x0[:,2]-z)<=cfg['interface_halfwidth_A'])&
                    (dx[:,0]>=edge[ix])&(dx[:,0]<edge[ix+1])&(dx[:,1]>=edge[iy])&(dx[:,1]<edge[iy+1]))
                tiles.append((interface,ix,iy,z,ixm))
    sources={str(CFG):sha(CFG),str(Path(__file__).resolve()):sha(__file__),ref['source_path']:ref['sha256'],str(imeta):sha(imeta)}
    for rr in (case['row'],case['baseline']):
        for key in ('log_path','metadata_path'): sources[rr[key]]=sha(rr[key])
    stats=[]; widths=[]; validated=[]; first={}; maximum_motion_error=0.
    states=[dict(run=case['case'],arm='reference',stage='initial',phase_index=-1)]+case['items']
    for item in states:
        if item['stage']=='initial': x=x0; tool=None; step=-1; travel=0.
        else:
            fr=item['frame']; f=dump(fr['source_path']); x=metal(f,ref); tool=f['xyz'][f['types']==5]
            rec=frame_record(f,ref); validated.append(rec); sources[f['source_path']]=f['sha256']
            step=f['step']; travel=float(fr['distance_nm']); rr=case['row'] if item['arm']=='vibration' else case['baseline']
            actual=tool.mean(axis=0)[[0,2]]; expected=expected_tool(rr,float(fr['time_ps']),{})
            if item['arm'] in first:
                olda,olde=first[item['arm']]; err=float(np.max(abs((actual-olda)-(expected-olde))))
                maximum_motion_error=max(maximum_motion_error,err)
                if err>.005: raise ValueError('Executed motion mismatch')
            else: first[item['arm']]=(actual,expected)
        ns=ns0 if item['stage']=='initial' else neighbors(x,centers,cfg['cutoff_A'],ly)
        src=np.repeat(np.arange(len(centers)),[len(v) for v in ns]); dst=np.concatenate(ns)
        retains=np.array([len(np.intersect1d(a,b))/len(a) if len(a) else np.nan for a,b in zip(ns0,ns)])
        common={k:v for k,v in item.items() if k!='frame'}
        common.update(case=case['case'],condition=case['condition'],realization=case['realization'],step=step,distance_nm=travel)
        if tool is None: gap=None
        else:
            tree=cKDTree(np.concatenate([tool+[0,s,0] for s in (-ly,0,ly)]))
            gap=float(tree.query(x[allcohort],workers=1)[0].min())
        for mask,name in masks:
            selected=mask[centers]; bonds=selected[src]; tb=bonds&ta[centers[src]]; c=ta[mask].mean()
            stat=dict(common,region=name,centers=int(selected.sum()),regional_initial_Ta_fraction=float(c),
                alpha_TaTa=float(1-ta[dst[tb]].mean()/c),coordination=float(np.mean([len(js) for js,use in zip(ns,selected) if use])),
                retained_initial_neighbor_fraction=float(np.nanmean(retains[selected])),
                cross_origin_bond_fraction=float(np.mean(origin[dst[bonds]]!=origin[centers[src[bonds]]])),
                minimum_cohort_tool_gap_A=gap)
            stats.append(stat)
        for ii,ix,iy,z,ixm in tiles:
            # Unwrap displacement relative to original packet before affine fit.
            xx=x0[ixm]+mic(x[ixm]-x0[ixm],ly)
            result=interface_width(x0[ixm],xx,origin[ixm],z,cfg['interface_halfwidth_A'],cfg['profile_kernel_A'])
            widths.append(dict(common,interface_index=ii,tile_x=ix,tile_y=iy,valid=result is not None,
                **(result or dict(width_A=None,normal_stretch=None,stretch_corrected_width_A=None,midplane_shift_A=None,atoms=len(ixm)))))
        print(case['case'],item['arm'],item['stage'],step,'complete',flush=True)
    write_csv(out/'regional.csv',stats); write_csv(out/'interfaces.csv',widths)
    manifest=dict(case=case,config_sha256=sha(CFG),sources=sources,validated_frames=validated,
        maximum_tool_motion_error_A=maximum_motion_error,cohort_atoms=int(allcohort.sum()),central_atoms=len(centers),
        interface_tiles=len(tiles),outputs={n:sha(out/n) for n in ('regional.csv','interfaces.csv')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--case'); parser.add_argument('--plan-only',action='store_true'); args=parser.parse_args()
    cfg=json.loads(CFG.read_text()); OUT.mkdir(parents=True,exist_ok=True); plan=make_plan(cfg)
    (OUT/'plan.json').write_text(json.dumps(dict(protocol_sha256=sha(CFG),cases=plan),indent=2)+'\n')
    if args.plan_only:
        for c in plan: print(c['case'],c['condition'],c['cycle_period_ps'],max(abs(i['phase_error_rad'] or 0) for i in c['items']))
        return
    for case in plan:
        if not args.case or case['case']==args.case: analyze(case,cfg)

if __name__=='__main__': main()
