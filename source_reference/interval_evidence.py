"""Versioned Monte Carlo and actual interface-profile evidence; no MD."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,json
from pathlib import Path
import numpy as np
from audit_sources import P4,ROOT,sha,write_csv
from atomic_analysis import reference,mic
from dynamic_order_control import assignments,statistics,REGIONS
from cycle_increment import width

CFG=P4/'config/interval_evidence_20260916.json'
OUT=P4/'data/derived/interval_evidence_20260916'

def config():
    OUT.mkdir(parents=True,exist_ok=True)
    return json.loads(CFG.read_text())

def mc():
    cfg=config();records=[];batch_rows=[]
    for r,name in enumerate(cfg['cases'],1):
        folder=OUT/name;folder.mkdir(exist_ok=True)
        source=P4/f'data/derived/dynamic_order_control/{name}'
        gm=json.loads((source/'graph_manifest.json').read_text())
        assert sha(source/'graphs.npz')==gm['graph_sha256']
        g=np.load(source/'graphs.npz');ref=reference(r)
        support=np.searchsorted(ref['ids'],g['support_ids']);assert np.array_equal(ref['ids'][support],g['support_ids'])
        ta=np.isin(ref['types'],[2,4]).astype(np.uint8);origin=np.isin(ref['types'],[1,2])
        strata=np.floor((ref['xyz'][:,2]-ref['xyz'][:,2].min())/2).astype(int)*2+origin.astype(int)
        np.savez_compressed(folder/'null_population.npz',ta=ta,strata=strata,support=support)
        lookup={(s['arm'],s['phase_index']):i for i,s in enumerate(gm['states'])}
        observed=np.load(source/'statistics.npz')['alpha'][:,0,:]
        arrays=[];seeds=[]
        for b,seed in enumerate(cfg['chemical']['batch_seeds']):
            seed+=r*cfg['chemical']['realization_seed_offset'];seeds.append(seed)
            dest=folder/f'batch_{b+1}.npz'
            if dest.exists():
                old=np.load(dest);assert int(old['seed'])==seed
                vals=old['alpha'];labels=old['labels']
            else:
                full=assignments(ta,strata,cfg['chemical']['batch_size'],seed)
                labels=full[:,support];del full
                values=[]
                for i in range(len(gm['states'])):
                    lo,hi=g['offsets'][i:i+2]
                    values.append(statistics(labels,g['centers'],g['src'][lo:hi],g['dst'][lo:hi],g['masks'],g['compositions']))
                values=np.array(values);assert np.max(abs(values[:,0,:]-observed))<1e-13
                vals=values[:,1:,:]
                np.savez_compressed(dest,alpha=vals,labels=labels[1:],seed=seed)
            assert vals.shape==(19,256,4) and np.isfinite(vals).all();arrays.append(vals)
            print(name,'independent batch',b+1,'complete',flush=True)
        alpha=np.concatenate(arrays,axis=1)
        paired=np.array([alpha[lookup['vibration',p]]-alpha[lookup['baseline',p]] for p in range(9)])
        obs=np.array([observed[lookup['vibration',p]]-observed[lookup['baseline',p]] for p in range(9)])
        oldpaired=np.load(source/'statistics.npz')['paired']
        for label,count,null in [('original',64,oldpaired[:,1:,:])]+[('new',n,paired[:,:n,:]) for n in cfg['chemical']['prefix_counts']]:
            for j,region in enumerate(REGIONS):
                for stat,v,o in [('phase_mean',null[:,:,j].mean(0),obs[:,j].mean()),('cycle_increment',null[-1,:,j]-null[0,:,j],obs[-1,j]-obs[0,j])]:
                    records.append(dict(case=name,realization=r,draw_set=label,draws=count,region=region,statistic=stat,
                        observed=float(o),null_mean=float(v.mean()),excess=float(o-v.mean()),null_sd=float(v.std(ddof=1)),mcse=float(v.std(ddof=1)/np.sqrt(count))))
        for b in range(4):
            q=paired[:,b*256:(b+1)*256,:]
            for j,region in enumerate(REGIONS):
                batch_rows.append(dict(case=name,batch=b+1,seed=seeds[b],region=region,
                    mean_excess=float(obs[:,j].mean()-q[:,:,j].mean()),increment=float(obs[-1,j]-obs[0,j]-(q[-1,:,j]-q[0,:,j]).mean())))
        np.savez_compressed(folder/'mc_statistics.npz',observed=observed,null=alpha,paired_observed=obs,paired_null=paired)
        sources={str(p.relative_to(P4)):sha(p) for p in [source/'graphs.npz',source/'graph_manifest.json',source/'statistics.npz',CFG]}
        sources[ref['source_path']]=ref['sha256']
        (folder/'mc_manifest.json').write_text(json.dumps(dict(config_sha256=sha(CFG),script_sha256=sha(__file__),sources=sources,
            independent_batch_seeds=seeds,total_new_draws=1024,states=gm['states'],
            outputs={p.name:sha(p) for p in folder.glob('*.npz') if p.name.startswith(('batch_','mc_','null_'))}),indent=2)+'\n')
    write_csv(OUT/'mc_convergence.csv',records);write_csv(OUT/'mc_batches.csv',batch_rows)

def profile(x0,x,origin,z0):
    if len(x0)<50:return None
    center=x0.mean(0);J=np.linalg.lstsq(x0-center,x-x.mean(0),rcond=None)[0]
    if np.linalg.cond(J)>100:return None
    normal=np.linalg.solve(J,[0.,0.,1.]);stretch=1/np.linalg.norm(normal);normal*=stretch
    d=(x-x.mean(0))@normal+(center[2]-z0)*stretch
    label=origin if origin[x0[:,2]>z0].mean()>origin[x0[:,2]<z0].mean() else 1-origin
    grid=np.linspace(-6*stretch,6*stretch,161);weights=np.exp(-.5*((grid[:,None]-d)/.75)**2)
    counts=weights.sum(1);p=weights@label/np.maximum(counts,1e-100);cross=[]
    for level in (.1,.5,.9):
        ix=np.flatnonzero((p[:-1]<level)&(p[1:]>=level)&(counts[:-1]>=3)&(counts[1:]>=3))
        if not len(ix):return None
        vals=grid[ix]+(level-p[ix])*(grid[ix+1]-grid[ix])/(p[ix+1]-p[ix])
        cross.append(float(vals[np.argmin(abs(vals))]))
    if not cross[0]<cross[1]<cross[2]:return None
    raw=cross[2]-cross[0]
    return dict(grid=grid,profile=p,counts=counts,crossings=np.array(cross),stretch=stretch,physical_width=raw,corrected_width=raw/stretch)

def profiles():
    cfg=config();table=[];curves=[];summaries=[];tile_rows=[];errors=[];selection=[]
    for name in cfg['cases']:
        path=P4/f'data/review72_packets/{name}.npz';meta=json.loads(path.with_suffix('.json').read_text())
        assert sha(path)==meta['packet_sha256'];a=np.load(path);xyz=a['xyz'];origin=np.isin(a['types'],[1,2]);ly=float(a['ly'])
        lookup={(s['arm'],s['phase_index']):i for i,s in enumerate(meta['states'])};nt=len(a['tile_info'])
        results={};w={arm:np.full((9,nt),np.nan) for arm in ('baseline','vibration')}
        for arm in w:
            for phase in range(9):
                i=lookup[arm,phase]
                for tile,info in enumerate(a['tile_info']):
                    ix=a['tile_atoms'][a['tile_offsets'][tile]:a['tile_offsets'][tile+1]]
                    x0=xyz[0,ix];x=x0+mic(xyz[i,ix]-x0,ly)
                    q=profile(x0,x,origin[ix],info[3]);old=width(x0,x,origin[ix],info[3],6,.75,[.1,.5,.9])
                    assert (q is None)==(old is None)
                    if q is not None:
                        errors.append(abs(q['corrected_width']-old));w[arm][phase,tile]=q['corrected_width'];results[arm,phase,tile]=q
        valid=np.isfinite(w['baseline']).all(0)&np.isfinite(w['vibration']).all(0)
        top=max(a['tile_info'][:,0]);eligible=[i for i in np.flatnonzero(valid) if a['tile_info'][i,0]==top]
        tile=int(min(eligible,key=lambda i:((a['tile_info'][i,1]-1)**2+(a['tile_info'][i,2]-1)**2,tuple(a['tile_info'][i,1:3]))))
        selection.append(dict(case=name,realization=meta['realization'],representative_tile=tile,interface=int(top),tile_x=int(a['tile_info'][tile,1]),tile_y=int(a['tile_info'][tile,2]),valid_tiles=int(valid.sum()),excluded_tiles=np.flatnonzero(~valid).tolist(),packet_sha256=sha(path)))
        for arm in w:
            for phase in (0,8):
                q=results[arm,phase,tile]
                table.append(dict(case=name,realization=meta['realization'],arm=arm,phase_index=phase,tile=tile,
                    step=meta['states'][lookup[arm,phase]]['step'],stretch=q['stretch'],physical_width_A=q['physical_width'],corrected_width_A=q['corrected_width']))
                for k in range(161):curves.append(dict(case=name,arm=arm,phase_index=phase,tile=tile,distance_A=q['grid'][k],fraction=q['profile'][k],support=q['counts'][k]))
        delta=(w['vibration'][-1]-w['vibration'][0])-(w['baseline'][-1]-w['baseline'][0])
        for j in np.flatnonzero(valid):tile_rows.append(dict(case=name,realization=meta['realization'],tile=j,interface=int(a['tile_info'][j,0]),paired_increment_A=delta[j]))
        start=(w['vibration'][0,valid]-w['baseline'][0,valid]).mean();end=(w['vibration'][-1,valid]-w['baseline'][-1,valid]).mean()
        summaries.append(dict(case=name,realization=meta['realization'],valid_tiles=int(valid.sum()),start_offset_A=float(start),end_offset_A=float(end),paired_increment_A=float(delta[valid].mean())))
        folder=OUT/name;folder.mkdir(exist_ok=True)
        payload={}
        for (arm,phase,j),q in results.items():
            if phase in (0,8) and j==tile:
                for key,val in q.items():payload[f'{arm}_{phase}_{key}']=val
        payload.update(width_baseline=w['baseline'],width_vibration=w['vibration'],valid=valid,tile_info=a['tile_info'])
        np.savez_compressed(folder/'profiles.npz',**payload)
        print(name,'profiles checked;',valid.sum(),'valid tiles',flush=True)
    for name,data in [('profile_states',table),('profile_curves',curves),('profile_tiles',tile_rows),('profile_summary',summaries)]:write_csv(OUT/f'{name}.csv',data)
    (OUT/'profile_manifest.json').write_text(json.dumps(dict(config_sha256=sha(CFG),script_sha256=sha(__file__),
        selection=selection,maximum_original_width_error=max(errors),no_new_MD=True,
        sources={str(p.relative_to(P4)):sha(p) for n in cfg['cases'] for p in [P4/f'data/review72_packets/{n}.npz',P4/f'data/review72_packets/{n}.json']},
        helpers={n:sha(P4/'scripts'/n) for n in ['atomic_analysis.py','cycle_increment.py']},
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and (p.name.startswith('profile') and p.name!='profile_manifest.json')}),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['mc','profiles']);args=p.parse_args()
    mc() if args.stage=='mc' else profiles()
