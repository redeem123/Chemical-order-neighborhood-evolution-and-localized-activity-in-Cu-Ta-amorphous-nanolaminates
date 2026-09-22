"""Exact endpoint edge accounting and full-population spatial null checks. No MD."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
import argparse,csv,json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from audit_sources import P4,sha,write_csv
from atomic_analysis import reference
from dynamic_order_control import REGIONS

CFG=P4/'config/material_spatial_20260916.json'
OUT=P4/'data/derived/material_spatial_20260916'
OLD=P4/'data/derived/interval_evidence_20260916'

def counts(labels,centers,src,dst,masks):
    """Species order Cu-Cu, Cu-Ta, Ta-Cu, Ta-Ta; all edges directed."""
    result=np.zeros((len(labels),len(masks),4),dtype=np.int64)
    for j,mask in enumerate(masks):
        use=mask[src]; a=labels[:,centers[src[use]]]; b=labels[:,dst[use]]
        tt=(a*b).sum(1,dtype=np.int64); at=a.sum(1,dtype=np.int64); bt=b.sum(1,dtype=np.int64)
        result[:,j]=np.stack([use.sum()-at-bt+tt,bt-tt,at-tt,tt],axis=-1)
    return result

def decompose(start,end,lost,gained,composition):
    m0=start[...,2:].sum(-1);q0=start[...,3];m1=end[...,2:].sum(-1);q1=end[...,3]
    ml=lost[...,2:].sum(-1);ql=lost[...,3];mg=gained[...,2:].sum(-1);qg=gained[...,3]
    c=np.broadcast_to(composition,m0.shape);valid=(m0>0)&(m1>0)&(c>0)
    with np.errstate(divide='ignore',invalid='ignore'):
        p0=q0/m0; direct=-(q1/m1-p0)/c
        gain=-(qg-p0*mg)/m1/c;loss=(ql-p0*ml)/m1/c
    vals=np.stack([gain,loss],-1);vals[~valid]=np.nan;direct[~valid]=np.nan
    return vals,direct,valid

def tests():
    labels=np.array([[1,1,0,1,0],[0,0,0,0,0]],dtype=np.uint8);centers=np.array([0]);masks=np.array([[True]])
    def c(neighbors):return counts(labels,centers,np.zeros(len(neighbors),int),np.array(neighbors,int),masks)
    cases=[('unchanged',[1,2],[1,2]),('same_species',[1,2],[3,4]),('coordination',[1,2],[1,2,3]),('empty',[],[])]
    results=[]
    for name,a,b in cases:
        parts,d,valid=decompose(c(a),c(b),c(sorted(set(a)-set(b))),c(sorted(set(b)-set(a))),np.array([.5]))
        assert np.isnan(parts[~valid]).all() and np.isnan(d[~valid]).all()
        error=float(np.max(abs(parts.sum(-1)[valid]-d[valid]))) if valid.any() else 0
        assert error<1e-12
        if name in ('unchanged','same_species'):assert np.all(d[valid]==0)
        results.append(dict(test=name,closure_error=error,undefined=int((~valid).sum())))
    return results

def graph(case):
    folder=P4/f'data/derived/dynamic_order_control/{case}';m=json.loads((folder/'graph_manifest.json').read_text())
    assert sha(folder/'graphs.npz')==m['graph_sha256']
    g=dict(np.load(folder/'graphs.npz'));lookup={(s['arm'],s['phase_index']):i for i,s in enumerate(m['states'])}
    edges=[];n=len(g['support_ids'])
    for lo,hi in zip(g['offsets'][:-1],g['offsets'][1:]):
        ids=g['src'][lo:hi].astype(np.int64)*n+g['dst'][lo:hi];assert len(np.unique(ids))==len(ids);edges.append(ids)
    return g,m,lookup,edges

def account(case,labels):
    g,m,lookup,edges=graph(case);n=len(g['support_ids']);nlab=len(labels)
    def c(ids):return counts(labels,g['centers'],ids//n,ids%n,g['masks'])
    allcounts=np.array([c(e) for e in edges]);den=allcounts[...,2:].sum(-1)
    with np.errstate(divide='ignore',invalid='ignore'):alpha=1-allcounts[...,3]/den/g['compositions']
    alpha[den==0]=np.nan
    parts=np.full((2,9,nlab,4,2),np.nan);ledger=np.zeros((2,9,nlab,4,3,4),np.int64);largest=0.
    for ai,arm in enumerate(('baseline','vibration')):
        i0=lookup[arm,0];e0=edges[i0]
        for phase in range(9):
            i=lookup[arm,phase];et=edges[i]
            retained=c(np.intersect1d(e0,et));lost=c(np.setdiff1d(e0,et));gained=c(np.setdiff1d(et,e0))
            assert np.array_equal(retained+lost,allcounts[i0]);assert np.array_equal(retained+gained,allcounts[i])
            v,d,valid=decompose(allcounts[i0],allcounts[i],lost,gained,g['compositions'])
            direct=alpha[i]-alpha[i0];assert np.array_equal(np.isfinite(d),np.isfinite(direct))
            if valid.any():largest=max(largest,float(np.max(abs(v.sum(-1)[valid]-direct[valid]))))
            parts[ai,phase]=v;ledger[ai,phase]=np.stack([retained,lost,gained],-2)
    assert largest<json.loads(CFG.read_text())['turnover']['closure_tolerance']
    return dict(alpha=alpha,parts=parts,counts=ledger,maximum_closure_error=largest)

def summaries(case,definition,alpha,parts):
    _,_,lookup,_=graph(case);paired=np.array([alpha[lookup['vibration',p]]-alpha[lookup['baseline',p]] for p in range(9)])
    endpoint=parts[1,-1]-parts[0,-1];rows=[]
    values=[('phase_mean',paired.mean(0)),('cycle_increment',paired[-1]-paired[0]),('gain',endpoint[...,0]),('loss',endpoint[...,1])]
    for name,val in values:
        for j,region in enumerate(REGIONS):
            null=val[1:,j];assert np.isfinite(null).all() and np.isfinite(val[0,j])
            rows.append(dict(case=case,definition=definition,region=region,statistic=name,observed=float(val[0,j]),null_mean=float(null.mean()),excess=float(val[0,j]-null.mean()),null_sd=float(null.std(ddof=1)),mcse=float(null.std(ddof=1)/np.sqrt(len(null))),draws=len(null)))
    return rows

def primary():
    OUT.mkdir(parents=True,exist_ok=True);allrows=[];records=[]
    with (OUT/'edge_turnover_ledger.csv').open('w',newline='') as stream:
        fields=['case','arm','phase','assignment','region','M0','Q0','M1','Q1','p0','composition']+[f'{cat}_{species}' for cat in ('retained','lost','gained') for species in ('CuCu','CuTa','TaCu','TaTa')]+['gain_alpha','loss_alpha','direct_delta_alpha']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for case in json.loads(CFG.read_text())['cases']:
            g,m,lookup,_=graph(case);paths=[OLD/case/f'batch_{b}.npz' for b in range(1,5)]
            labels=np.concatenate([g['labels'][:1]]+[np.load(p)['labels'] for p in paths])
            result=account(case,labels);old=np.load(OLD/case/'mc_statistics.npz');expected=np.concatenate([old['observed'][:,None],old['null']],axis=1)
            assert np.max(abs(result['alpha']-expected))<1e-12
            np.savez_compressed(OUT/f'{case}_turnover.npz',**result)
            allrows+=summaries(case,'depth_only',result['alpha'],result['parts'])
            for ai,arm in enumerate(('baseline','vibration')):
                for phase in range(9):
                    for k in range(len(labels)):
                        for j,region in enumerate(REGIONS):
                            cc=result['counts'][ai,phase,k,j];c0=cc[0]+cc[1];c1=cc[0]+cc[2];m0=int(c0[2:].sum());m1=int(c1[2:].sum())
                            row=dict(case=case,arm=arm,phase=phase,assignment='observed' if k==0 else str(k),region=region,M0=m0,Q0=int(c0[3]),M1=m1,Q1=int(c1[3]),p0=float(c0[3]/m0) if m0 else np.nan,composition=float(g['compositions'][j]),gain_alpha=result['parts'][ai,phase,k,j,0],loss_alpha=result['parts'][ai,phase,k,j,1],direct_delta_alpha=result['alpha'][lookup[arm,phase],k,j]-result['alpha'][lookup[arm,0],k,j])
                            row.update({f'{cat}_{species}':int(cc[a,b]) for a,cat in enumerate(('retained','lost','gained')) for b,species in enumerate(('CuCu','CuTa','TaCu','TaTa'))});writer.writerow(row)
            records.append(dict(case=case,closure=result['maximum_closure_error'],sources={str(p.relative_to(P4)):sha(p) for p in paths},graph_sha256=m['graph_sha256']))
            print(case,'primary edge ledger checked',flush=True)
    write_csv(OUT/'turnover_summary.csv',allrows)
    (OUT/'turnover_manifest.json').write_text(json.dumps(dict(config_sha256=sha(CFG),script_sha256=sha(__file__),tests=tests(),records=records),indent=2)+'\n')

def population(case,di):
    cfg=json.loads(CFG.read_text())['spatial_null'];r=int(case[-2:]);ref=reference(r);g,m,_,_=graph(case)
    old=json.loads((OLD/case/'mc_manifest.json').read_text());assert old['sources'][ref['source_path']]==ref['sha256']
    target=cfg['target_block_A'][di//2];fraction=cfg['origin_fractions'][di%2];x=ref['xyz'];box=ref['box']
    lengths=np.diff(box[:2],axis=1).ravel();bins=np.maximum(1,np.floor(lengths/target+.5).astype(int));width=lengths/bins
    b=np.floor(np.mod((x[:,:2]-box[:2,0])/width-fraction,bins)).astype(int)
    depth=np.floor((x[:,2]-x[:,2].min())/cfg['depth_A']).astype(int);material=np.isin(ref['types'],[1,2]).astype(int)
    strata=((depth*2+material)*bins[0]+b[:,0])*bins[1]+b[:,1]
    ta=np.isin(ref['types'],[2,4]).astype(np.uint8);support=np.searchsorted(ref['ids'],g['support_ids']);assert np.array_equal(ref['ids'][support],g['support_ids'])
    order=np.argsort(strata,kind='stable');_,start,size=np.unique(strata[order],return_index=True,return_counts=True)
    groups=np.split(order,start[1:]);nc=np.array([ta[ix].sum() for ix in groups]);mixed=(nc>0)&(nc<size)
    info=dict(case=case,definition=f'block{target}_origin{fraction:g}',definition_index=di,target_A=target,origin_fraction=fraction,bins=bins.tolist(),actual_width_A=width.tolist(),population=len(ta),strata=len(groups),size_min=int(size.min()),size_median=float(np.median(size)),size_max=int(size.max()),mixed_strata=int(mixed.sum()),mutable_atom_fraction=float(size[mixed].sum()/len(ta)),source_path=ref['source_path'],source_sha256=ref['sha256'],graph_sha256=m['graph_sha256'])
    folder=OUT/case/info['definition'];folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'population.npz',ta=ta,strata=strata,support=support,ids=ref['ids'],block=b,depth=depth,material=material,box=box)
    (folder/'population.json').write_text(json.dumps(info,indent=2)+'\n');return folder

def spatial_batch(task):
    case,di,batch=task;cfg=json.loads(CFG.read_text())['spatial_null'];r=int(case[-2:]);target=cfg['target_block_A'][di//2];fraction=cfg['origin_fractions'][di%2]
    folder=OUT/case/f'block{target}_origin{fraction:g}';dest=folder/f'batch_{batch}.npz';seed=cfg['seed_base']+10000*r+1000*di+batch
    if dest.exists():
        a=np.load(dest);assert int(a['seed'])==seed;return str(dest)
    t=time.monotonic();pop=np.load(folder/'population.npz');ta=pop['ta'];strata=pop['strata'];support=pop['support'];count=cfg['batch_size'];rng=np.random.default_rng(seed)
    full=np.tile(ta,(count,1));order=np.argsort(strata,kind='stable');_,start=np.unique(strata[order],return_index=True);groups=np.split(order,start[1:])
    for ix in groups:
        nc=int(ta[ix].sum())
        if 0<nc<len(ix):full[:,ix]=rng.permuted(np.broadcast_to(ta[ix],(count,len(ix))),axis=1)
        assert np.all(full[:,ix].sum(1)==nc)
    changed=np.mean(full!=ta,axis=1);labels=full[:,support];del full
    g,_,_,_=graph(case);result=account(case,np.concatenate([g['labels'][:1],labels]))
    np.savez_compressed(dest,seed=seed,labels=labels,changed_fraction=changed,**result)
    print(case,folder.name,'batch',batch,'seconds',round(time.monotonic()-t,2),flush=True);return str(dest)

def spatial(workers):
    cfg=json.loads(CFG.read_text());tasks=[]
    for case in cfg['cases']:
        for di in range(4):
            population(case,di);tasks += [(case,di,b) for b in range(1,5)]
    # First fixed batch is a timing benchmark, not a parameter-selection pilot.
    spatial_batch(tasks[0])
    with ProcessPoolExecutor(max_workers=workers) as pool:list(pool.map(spatial_batch,tasks[1:]))
    rows=[];records=[]
    for case in cfg['cases']:
        for di in range(4):
            target=cfg['spatial_null']['target_block_A'][di//2];fraction=cfg['spatial_null']['origin_fractions'][di%2];definition=f'block{target}_origin{fraction:g}';folder=OUT/case/definition
            batches=[dict(np.load(folder/f'batch_{b}.npz')) for b in range(1,5)]
            alpha=np.concatenate([batches[0]['alpha'][:,:1]]+[b['alpha'][:,1:] for b in batches],axis=1)
            parts=np.concatenate([batches[0]['parts'][:,:,:1]]+[b['parts'][:,:,1:] for b in batches],axis=2)
            rows+=summaries(case,definition,alpha,parts);np.savez_compressed(folder/'statistics.npz',alpha=alpha,parts=parts)
            info=json.loads((folder/'population.json').read_text());info.update(stratum_counts_conserved=True,changed_fraction_mean=float(np.concatenate([b['changed_fraction'] for b in batches]).mean()),maximum_closure_error=float(max(b['maximum_closure_error'] for b in batches)),seeds=[int(b['seed']) for b in batches],outputs={p.name:sha(p) for p in folder.glob('*.npz')});records.append(info)
    write_csv(OUT/'spatial_summary.csv',rows)
    (OUT/'spatial_manifest.json').write_text(json.dumps(dict(config_sha256=sha(CFG),script_sha256=sha(__file__),tests=tests(),records=records,no_new_MD=True),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['primary','spatial']);p.add_argument('--workers',type=int,default=4);args=p.parse_args();assert 1<=args.workers<=4
    tests();primary() if args.stage=='primary' else spatial(args.workers)
