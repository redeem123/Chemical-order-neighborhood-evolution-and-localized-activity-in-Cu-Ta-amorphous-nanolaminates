"""Independent check of the documented profile on supplied reduced coordinates.

Does not modify coordinates, masks, crossings or reference estimates.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,csv,json
from pathlib import Path
import numpy as np

def crossings(grid,f,support):
    selected=[];brackets=[]
    for a in (.1,.5,.9):
        candidates=[]
        for j in range(len(grid)-1):
            if f[j]<a<=f[j+1] and support[j]>=3 and support[j+1]>=3:
                t=(a-f[j])/(f[j+1]-f[j]);candidates.append((grid[j]+t*(grid[j+1]-grid[j]),j))
        if not candidates:return dict(valid=False,reason=f'no_supported_{a}_crossing')
        point,j=min(candidates,key=lambda v:abs(v[0]))
        selected.append(float(point));brackets.append([float(support[j]),float(support[j+1])])
    if not selected[0]<selected[1]<selected[2]:return dict(valid=False,reason='unordered')
    return dict(valid=True,crossings=selected,bracket_support=brackets)

def estimate(x0,x,origin,z,half,material):
    if len(x0)<50:return dict(valid=False,reason='atoms')
    J=np.linalg.lstsq(x0-x0.mean(0),x-x.mean(0),rcond=None)[0]
    if np.linalg.cond(J)>100:return dict(valid=False,reason='condition_number')
    cov=np.linalg.solve(J,np.array([0.,0.,1.]));stretch=1/np.linalg.norm(cov)
    q=(x-x.mean(0))@cov+x0[:,2].mean()-z
    labels=origin if origin[x0[:,2]>z].mean()>origin[x0[:,2]<z].mean() else 1-origin
    scale=1. if material else stretch
    grid=np.linspace(-half,half,round(2*half/.075)+1)*scale
    kernel=np.exp(-.5*((grid[:,None]-q*scale)/.75)**2)
    support=kernel.sum(axis=1);fraction=(kernel@labels)/np.maximum(support,1e-100)
    result=crossings(grid,fraction,support)
    if result['valid']:result['width_A']=(result['crossings'][2]-result['crossings'][0])/(1 if material else stretch)
    result['stretch']=float(stretch)
    return result

def main(root):
    # Labeled synthetic unit checks of the acceptance rule, not material evidence.
    grid=np.array([-2.,-1.,0.,1.,2.]);frac=np.array([0.,.2,.4,.6,1.]);sup=np.full(5,3.)
    assert crossings(grid,frac,sup)['valid']
    sup[0]=2.99;assert not crossings(grid,frac,sup)['valid']
    report=dict(support_boundary_tests='passed',primary={},extended={},central_R01=[],R02_tile25=[])
    e=root/'evidence'
    for rep in (1,2,3):
        name=f'S6_R{rep:02d}';folder=e/'primary'/name
        a=np.load(folder/'coordinates.npz');meta=json.loads((folder/'coordinates.json').read_text());expected=np.load(folder/'widths.npz')['widths'];lookup={(s['arm'],s['phase_index']):i for i,s in enumerate(meta['states'])}
        actual=np.full_like(expected,np.nan)
        for ai,arm in enumerate(('baseline','vibration')):
            for phase in range(9):
                for ti,info in enumerate(a['tile_info']):
                    ids=a['tile_atoms'][a['tile_offsets'][ti]:a['tile_offsets'][ti+1]];x0=a['xyz'][0,ids]
                    dx=a['xyz'][lookup[arm,phase],ids]-x0;dx[:,1]-=float(a['ly'])*np.rint(dx[:,1]/float(a['ly']))
                    for ei,material in enumerate((False,True)):
                        v=estimate(x0,x0+dx,np.isin(a['types'][ids],[1,2]),info[3],6,material)
                        if v['valid']:actual[ei,ai,phase,ti]=v['width_A']
                        row=dict(arm=arm,phase=phase,estimator='material' if material else 'physical',**v)
                        if rep==1 and ti==22 and phase in (0,8):report['central_R01'].append(row)
                        if rep==2 and ti==25:report['R02_tile25'].append(row)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-9,equal_nan=True)
        common=np.isfinite(actual).all(axis=(0,1,2))
        report['primary'][name]=dict(profiles=int(actual.size),common_tiles=int(common.sum()),maximum_error_A=float(np.nanmax(abs(actual-expected))))
        folder=e/'supporting'/('registration_R03' if rep==3 else 'registration_R01_R02')
        a=np.load(folder/(name+'_extended_width_coordinates.npz'));meta=json.loads((folder/(name+'_extended_states.json')).read_text());expected={(q['arm'],int(q['phase']),int(q['tile']),q['estimator']):q for q in csv.DictReader((folder/(name+'_support_states.csv')).open()) if q['packet_half_A']=='8'}
        errors=[];valid=np.ones(len(a['tile_info']),bool)
        for si,state in enumerate(meta[1:],1):
            for ti,info in enumerate(a['tile_info']):
                ids=a['tile_atoms'][a['tile_offsets'][ti]:a['tile_offsets'][ti+1]];x0=a['xyz'][0,ids];dx=a['xyz'][si,ids]-x0;dx[:,1]-=float(a['ly'])*np.rint(dx[:,1]/float(a['ly']))
                for material in (False,True):
                    v=estimate(x0,x0+dx,np.isin(a['types'][ids],[1,2]),info[3],8,material)
                    q=expected[state['arm'],state['phase_index'],ti,'material' if material else 'physical'];assert v['valid']==(q['valid']=='True')
                    valid[ti]&=v['valid']
                    if v['valid']:errors.append(abs(v['width_A']-float(q['width_A'])))
        assert max(errors)<1e-9 and valid.all()
        report['extended'][name]=dict(profiles=len(expected),common_tiles=int(valid.sum()),maximum_error_A=max(errors))
        print(name,'documented profile matches both packets',flush=True)
    assert [r['common_tiles'] for r in report['primary'].values()]==[27,26,27]
    assert any(not r['valid'] for r in report['R02_tile25'])
    report['status']='passed';print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();main(a.root)
