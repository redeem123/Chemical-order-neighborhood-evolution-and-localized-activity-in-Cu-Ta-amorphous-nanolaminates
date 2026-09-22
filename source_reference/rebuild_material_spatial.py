"""Portable checks of the material-bandwidth and spatial-null evidence.

Requires only NumPy plus the companion rebuild_selected_evidence.py. This is
numerical reconstruction from reduced inputs, not independent physical validation.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,csv,json
from pathlib import Path
import numpy as np
from rebuild_selected_evidence import digest,statistic

def width(x0,x,label,z0):
    A=x0-x0.mean(0);B=x-x.mean(0);F=np.linalg.lstsq(A,B,rcond=None)[0]
    if len(x0)<50 or np.linalg.cond(F)>100:return np.nan
    covector=np.linalg.inv(F)[:,2]
    # q = d/lambda, evaluated directly as the material-normal covector.
    q=B@covector+x0[:,2].mean()-z0
    labels=label if label[x0[:,2]>z0].mean()>label[x0[:,2]<z0].mean() else 1-label
    grid=np.linspace(-6,6,161);w=np.exp(-.5*((q[None,:]-grid[:,None])/.75)**2)
    support=w.sum(1);f=w@labels/np.maximum(support,1e-100);cross=[]
    for level in (.1,.5,.9):
        candidates=[]
        for i in range(160):
            if f[i]<level<=f[i+1] and min(support[i:i+2])>=3:candidates.append(grid[i]+(level-f[i])*(grid[i+1]-grid[i])/(f[i+1]-f[i]))
        if not candidates:return np.nan
        cross.append(min(candidates,key=abs))
    return cross[2]-cross[0] if cross[0]<cross[1]<cross[2] else np.nan

def edge_parts(labels,g,states):
    n=len(g['support_ids']);lookup={(s['arm'],s['phase_index']):i for i,s in enumerate(states)}
    edges=[g['src'][a:b].astype(np.int64)*n+g['dst'][a:b] for a,b in zip(g['offsets'][:-1],g['offsets'][1:])]
    def mq(edges,region):
        s=edges//n;t=edges%n;use=g['masks'][region][s];s=g['centers'][s[use]];t=t[use]
        central=labels[:,s].astype(bool);return central.sum(1),np.count_nonzero(central&(labels[:,t]==1),axis=1)
    result=np.full((2,9,len(labels),4,2),np.nan)
    for ai,arm in enumerate(('baseline','vibration')):
        first=edges[lookup[arm,0]]
        for j,c in enumerate(g['compositions']):
            m0,q0=mq(first,j)
            for p in range(9):
                last=edges[lookup[arm,p]];m1,_=mq(last,j);ml,ql=mq(np.setdiff1d(first,last),j);mg,qg=mq(np.setdiff1d(last,first),j)
                with np.errstate(divide='ignore',invalid='ignore'):
                    probability=q0/m0;gain=(probability*mg-qg)/(m1*c);loss=(ql-probability*ml)/(m1*c)
                good=(m0>0)&(m1>0)&(c>0);result[ai,p,good,j,0]=gain[good];result[ai,p,good,j,1]=loss[good]
    return result

def verify(root,regenerate):
    manifest=json.loads((root/'manifest.json').read_text())
    for item in manifest['files']:assert digest(root/item['path'])==item['sha256'],item['path']
    dest=root/'evidence/material_spatial';cfg=json.loads((dest/'design.json').read_text());records=[]
    for r in (1,2):
        name=f'S6_R{r:02d}';folder=root/f'evidence/R{r:02d}';g=np.load(folder/'graphs.npz');states=json.loads((folder/'states.json').read_text());a=np.load(folder/'coordinates.npz');sm=json.loads((folder/'coordinate_states.json').read_text());lookup={(s['arm'],s['phase_index']):i for i,s in enumerate(sm)}
        expected=np.load(dest/f'{name}_widths.npz');actual=np.full_like(expected['widths'][1],np.nan);labels=np.isin(a['types'],[1,2]);maximum=0.
        for ai,arm in enumerate(('baseline','vibration')):
            for phase in range(9):
                for ti,info in enumerate(a['tile_info']):
                    lo,hi=a['tile_offsets'][ti:ti+2];ix=a['tile_atoms'][lo:hi];x0=a['xyz'][0,ix];d=a['xyz'][lookup[arm,phase],ix]-x0;d[:,1]-=float(a['ly'])*np.rint(d[:,1]/float(a['ly']));actual[ai,phase,ti]=width(x0,x0+d,labels[ix],info[3])
        np.testing.assert_allclose(actual,expected['widths'][1],atol=1e-10,rtol=0,equal_nan=True)
        full=np.concatenate([g['labels'][:1]]+[np.load(folder/f'batch_{b}.npz')['labels'] for b in range(1,5)]);calculated=edge_parts(full,g,states);stored=np.load(dest/f'{name}_turnover.npz');np.testing.assert_allclose(calculated,stored['parts'],atol=1e-12,rtol=0,equal_nan=True)
        pmax=float(np.nanmax(abs(calculated-stored['parts'])));chemical_max=0.;spatial_parts_max=0.;defs=[]
        for di,(target,origin) in enumerate((h,o) for h in (16,32) for o in (0,.5)):
            sf=dest/name/f'block{target}_origin{origin:g}';pop=np.load(sf/'population.npz');ta=pop['ta'];strata=pop['strata'];order=np.argsort(strata,kind='stable');_,starts=np.unique(strata[order],return_index=True);groups=np.split(order,starts[1:])
            assert np.array_equal(ta[pop['support']],g['labels'][0])
            for b in range(1,5):
                batch=np.load(sf/f'batch_{b}.npz');seed=cfg['spatial_null']['seed_base']+10000*r+1000*di+b;assert int(batch['seed'])==seed
                if regenerate:
                    rng=np.random.default_rng(seed);full=np.tile(ta,(256,1))
                    for ix in groups:
                        count=int(ta[ix].sum())
                        if 0<count<len(ix):full[:,ix]=rng.permuted(np.broadcast_to(ta[ix],(256,len(ix))),axis=1)
                        assert np.all(full[:,ix].sum(1)==count)
                    assert np.array_equal(full[:,pop['support']],batch['labels']);np.testing.assert_allclose((full!=ta).mean(1),batch['changed_fraction'],rtol=0,atol=0);del full
                ll=np.concatenate([g['labels'][:1],batch['labels']]);measured=statistic(ll,g);np.testing.assert_allclose(measured,batch['alpha'],atol=1e-12,rtol=0);chemical_max=max(chemical_max,float(np.max(abs(measured-batch['alpha']))))
                parts=edge_parts(ll,g,states);np.testing.assert_allclose(parts,batch['parts'],atol=1e-12,rtol=0,equal_nan=True);spatial_parts_max=max(spatial_parts_max,float(np.nanmax(abs(parts-batch['parts']))))
            defs.append(sf.name);print(name,sf.name,'checked',flush=True)
        records.append(dict(case=name,material_width_max_error_A=float(np.nanmax(abs(actual-expected['widths'][1]))),primary_edge_max_error=pmax,spatial_statistic_max_error=chemical_max,spatial_edge_max_error=spatial_parts_max,definitions=defs))
    return dict(package_files=len(manifest['files']),regenerated_spatial_full_labels=regenerate,records=records,scope='Reduced numerical reconstruction; not MD, whole-workpiece coordinate reconstruction, or physical validation')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--regenerate-spatial',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args();result=verify(a.root,a.regenerate_spatial);s=json.dumps(result,indent=2)+'\n'
    if a.output:a.output.write_text(s)
    print(s)
