"""Versioned physical-vs-material bandwidth check on actual saved packets."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
import json
import numpy as np
from audit_sources import P4, sha, write_csv
from atomic_analysis import mic
from interval_evidence import profile as original_profile

CFG=P4/'config/material_spatial_20260916.json'
OUT=P4/'data/derived/material_spatial_20260916'


def profile(x0,x,origin,z0,material):
    cfg=json.loads(CFG.read_text())['profile']
    result=dict(valid=False,reason='',stretch=np.nan,width=np.nan,crossings=np.full(3,np.nan))
    if len(x0)<cfg['minimum_atoms']: result['reason']='too_few_atoms'; return result
    center=x0.mean(0); J=np.linalg.lstsq(x0-center,x-x.mean(0),rcond=None)[0]
    if np.linalg.cond(J)>cfg['max_condition_number']: result['reason']='affine_condition'; return result
    normal=np.linalg.solve(J,[0.,0.,1.]); stretch=1/np.linalg.norm(normal); normal*=stretch
    d=(x-x.mean(0))@normal+(center[2]-z0)*stretch
    label=origin if origin[x0[:,2]>z0].mean()>origin[x0[:,2]<z0].mean() else 1-origin
    grid=np.linspace(-6,6,161) if material else np.linspace(-6*stretch,6*stretch,161)
    distances=d/stretch if material else d
    weights=np.exp(-.5*((grid[:,None]-distances)/.75)**2)
    support=weights.sum(1); fraction=weights@label/np.maximum(support,1e-100)
    result.update(stretch=float(stretch),grid=grid,fraction=fraction,support=support)
    reasons=[]; crossings=[]
    for level in (.1,.5,.9):
        hit=(fraction[:-1]<level)&(fraction[1:]>=level)
        valid=hit&(support[:-1]>=3)&(support[1:]>=3)
        ix=np.flatnonzero(valid)
        if not len(ix):
            reasons.append(f'{level:g}_'+('insufficient_support' if hit.any() else 'no_upward_crossing'))
            crossings.append(np.nan)
        else:
            candidates=grid[ix]+(level-fraction[ix])*(grid[ix+1]-grid[ix])/(fraction[ix+1]-fraction[ix])
            crossings.append(float(candidates[np.argmin(abs(candidates))]))
    result['crossings']=np.array(crossings)
    if reasons: result['reason']=';'.join(reasons); return result
    if not crossings[0]<crossings[1]<crossings[2]:result['reason']='unordered_crossings';return result
    result.update(valid=True,reason='valid',width=float((crossings[2]-crossings[0])/(1 if material else stretch)))
    return result


def transforms():
    theta=np.deg2rad(27);R=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    shear=np.array([[1.,.15,.35],[0,1.,-.2],[.2,.1,1.]])
    yield 'translation',np.eye(3),np.array([2.,-3.,5.])
    yield 'rotation',R,np.zeros(3)
    yield 'tilted_shear',shear,np.zeros(3)
    for s in json.loads(CFG.read_text())['profile']['stretch_tests']:
        yield f'normal_{s:.1f}',np.diag([1.,1.,s]),np.zeros(3)
    yield 'rotation_shear_stretch',shear@np.diag([1.,1.,.7])@R,np.array([2.,-3.,5.])


def main():
    OUT.mkdir(parents=True,exist_ok=True); cfg=json.loads(CFG.read_text())
    rows=[];tile_rows=[];summary=[];affine=[];sources={};largest=0.;displays={}
    for r,name in enumerate(cfg['cases'],1):
        path=P4/f'data/review72_packets/{name}.npz'; meta=json.loads(path.with_suffix('.json').read_text())
        assert sha(path)==meta['packet_sha256']; sources[str(path.relative_to(P4))]=sha(path)
        a=np.load(path); origin=np.isin(a['types'],[1,2]); lookup={(s['arm'],s['phase_index']):i for i,s in enumerate(meta['states'])}
        widths=np.full((2,2,9,len(a['tile_info'])),np.nan); stretches=np.full((2,9,len(a['tile_info'])),np.nan)
        for ti,info in enumerate(a['tile_info']):
            ix=a['tile_atoms'][a['tile_offsets'][ti]:a['tile_offsets'][ti+1]]; x0=a['xyz'][0,ix]; labels=origin[ix]
            reference=profile(x0,x0,labels,info[3],True)
            for label,J,shift in transforms():
                q=profile(x0,(x0-x0.mean(0))@J+x0.mean(0)+shift,labels,info[3],True)
                assert q['valid']==reference['valid']
                error=abs(q['width']-reference['width']) if q['valid'] else np.nan
                if q['valid']:assert error<cfg['profile']['affine_tolerance_A']
                affine.append(dict(case=name,tile=ti,transform=label,valid=q['valid'],reason=q['reason'],width_A=q['width'],reference_width_A=reference['width'],absolute_error_A=error))
            for arm_i,arm in enumerate(('baseline','vibration')):
                for phase in range(9):
                    idx=lookup[arm,phase]; x=x0+mic(a['xyz'][idx,ix]-x0,float(a['ly']))
                    prior=original_profile(x0,x,labels,info[3])
                    for estimator,material in enumerate((False,True)):
                        q=profile(x0,x,labels,info[3],material)
                        if not material:
                            assert q['valid']==(prior is not None)
                            if q['valid']: largest=max(largest,abs(q['width']-prior['corrected_width']))
                        widths[estimator,arm_i,phase,ti]=q['width'];stretches[arm_i,phase,ti]=q['stretch']
                        rows.append(dict(case=name,realization=r,estimator='material' if material else 'physical',arm=arm,phase=phase,step=meta['states'][idx]['step'],tile=ti,interface=int(info[0])+1,valid=q['valid'],reason=q['reason'],stretch=q['stretch'],width_A=q['width'],cross10_A=q['crossings'][0],cross50_A=q['crossings'][1],cross90_A=q['crossings'][2]))
                        if ti==22 and phase in (0,8):
                            for key in ('grid','fraction','support','crossings'):
                                if key in q: displays[f'R{r}_{estimator}_{arm}_{phase}_{key}']=q[key]
        valid=np.isfinite(widths).all(axis=(1,2)); common=valid.all(0)
        assert common.any(); delta=(widths[:,1,-1]-widths[:,1,0])-(widths[:,0,-1]-widths[:,0,0])
        for ti,info in enumerate(a['tile_info']):
            tile_rows.append(dict(case=name,realization=r,tile=ti,interface=int(info[0])+1,physical_valid=bool(valid[0,ti]),material_valid=bool(valid[1,ti]),common=bool(common[ti]),physical_increment_A=float(delta[0,ti]),material_increment_A=float(delta[1,ti])))
        for estimator,label in enumerate(('physical','material')):
            selections=[('own_valid',valid[estimator]),('common',common)]
            for interface in sorted(set(a['tile_info'][:,0])):
                selections += [(f'interface_{int(interface)+1}',common&(a['tile_info'][:,0]==interface)),(f'leave_out_{int(interface)+1}',common&(a['tile_info'][:,0]!=interface))]
            for selection,use in selections:
                summary.append(dict(case=name,realization=r,estimator=label,selection=selection,tiles=int(use.sum()),increment_A=float(delta[estimator,use].mean()) if use.any() else np.nan))
        np.savez_compressed(OUT/f'{name}_widths.npz',widths=widths,stretches=stretches,valid=valid,common=common,paired_increments=delta,tile_info=a['tile_info'])
        print(name,'old/new/common tile counts',valid.sum(1).tolist(),int(common.sum()),'increments',delta[:,common].mean(1),flush=True)
    assert largest<1e-10
    for name,data in [('width_states',rows),('width_tiles',tile_rows),('width_summary',summary),('affine_tests',affine)]:write_csv(OUT/(name+'.csv'),data)
    np.savez_compressed(OUT/'selected_profiles.npz',**displays)
    manifest=dict(config_sha256=sha(CFG),script_sha256=sha(__file__),sources=sources,original_width_reconciliation_max_A=largest,affine_test_count=len(affine),affine_max_error_A=max(x['absolute_error_A'] for x in affine if x['valid']),no_new_MD=True)
    (OUT/'width_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
