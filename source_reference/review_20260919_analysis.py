"""Read-only raw input, versioned output: joint population/null and frame checks.

Sparse directed adjacency is only an acceleration of pooled Ta-centered counts.
All labels are assigned on the full initial workpiece, then reduced to graph
support. Original seeds/assignment algorithms are reproduced and checked exactly.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import argparse,csv,json,itertools,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.sparse import csr_matrix
from audit_sources import P4,ROOT,sha,write_csv
from atomic_analysis import reference,dump,metal,neighbors
from dynamic_order_control import assignments,statistics,REGIONS

OUT=P4/'data/derived/review_20260919'
EXT=P4/'data/replication_extension_20260917'
CFG=P4/'config/review_20260919.json'
DEFS=['depth_only','block16_origin0','block16_origin0.5','block32_origin0','block32_origin0.5']
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(Path(p).open()))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def roots(r):
    return (EXT/'data/derived/interval_evidence',EXT/'data/derived/material_spatial',EXT/'data/derived/dynamic_order_control') if r==3 else (P4/'data/derived/interval_evidence_20260916',P4/'data/derived/material_spatial_20260916',P4/'data/derived/dynamic_order_control')

def extract(r):
    case=f'S6_R{r:02d}';folder=OUT/case;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'graphs.json').exists():
        m=read(folder/'graphs.json');assert sha(folder/'graphs.npz')==m['graph_sha256'];return
    root=EXT if r==3 else P4;mp=root/f'data/derived/cycle_increment/{case}_s10/manifest.json';m=read(mp);c=m['case']
    ref=reference(r);x=ref['xyz'];ta=np.isin(ref['types'],[2,4]).astype(np.uint8);origin=np.isin(ref['types'],[1,2])
    ip=ROOT/f'results/runs/relax_laminate_R{r:02d}/metadata/relaxed_structure.manifest.json'
    interfaces=np.asarray(read(ip)['interface_coordinates_A']);dist=np.min(abs(x[:,2,None]-interfaces),axis=1)
    center=[float(c['row']['x_start'])-100,ref['box'][1].mean()]
    eligible=np.all(abs(x[:,:2]-center)<=25,axis=1)&(x[:,2].max()-x[:,2]<=60)&(x[:,2].max()-x[:,2]>=0)
    centers=np.flatnonzero(eligible);masks=[];composition=[]
    for o in (True,False):
        for interior in (True,False):
            mask=eligible&(origin==o)&((dist>3.6) if interior else (dist<=3.6))
            masks.append(mask[centers]);composition.append(ta[mask].mean())
    old=np.load(roots(r)[2]/case/'graphs.npz')
    sample=(ref['ids'][centers]%3==0)
    np.testing.assert_array_equal(ref['ids'][centers[sample]],old['support_ids'][old['centers']])
    np.testing.assert_allclose(composition,old['compositions'],atol=0,rtol=0)
    np.testing.assert_array_equal(np.array(masks)[:,sample],old['masks'])
    inventory=rows(P4/'data/frame_manifest.csv')+rows(EXT/'data/frame_manifest.csv')
    bas={int(q['step']):q for q in inventory if q['run_id']==c['baseline']['run_id']}
    base=sorted(bas.values(),key=lambda q:float(q['distance_nm']))
    items=[q for q in c['items'] if q['stage']=='cycle'];lookup={(q['arm'],q['phase_index']):q for q in items}
    states=[];state_index={};choice={};registration=[]
    def add(q):
        path=q['source_path']
        if path not in state_index:
            state_index[path]=len(states);states.append(dict(q))
        return state_index[path]
    for arm in ('baseline','vibration'):
        choice[arm]=[add(lookup[arm,p]['frame']) for p in range(9)]
    for label,lower in [('earlier',True),('later',False)]:
        choices=[]
        for p in range(9):
            target=float(lookup['vibration',p]['frame']['distance_nm'])
            candidates=[q for q in base if (float(q['distance_nm'])<=target if lower else float(q['distance_nm'])>=target)]
            q=(max if lower else min)(candidates,key=lambda q:float(q['distance_nm']))
            choices.append(add(q));mismatch=float(q['distance_nm'])-target
            registration.append(dict(case=case,phase=p,choice=label,baseline_step=int(q['step']),vibration_step=int(lookup['vibration',p]['frame']['step']),mismatch_nm=mismatch,within_primary_tolerance=abs(mismatch)<=.016))
        choice[label]=choices
    sources={str(mp):sha(mp),ref['source_path']:ref['sha256'],str(ip):sha(ip)};graphs=[]
    ly=float(np.diff(ref['box'][1])[0])
    for q in states:
        f=dump(q['source_path']);xx=metal(f,ref)
        expected=m['sources'].get(f['source_path'])
        if expected:assert f['sha256']==expected
        sources[f['source_path']]=f['sha256'];q['source_sha256']=f['sha256']
        ns=neighbors(xx,centers,3.375,ly)
        graphs.append((np.repeat(np.arange(len(centers),dtype=np.int32),[len(js) for js in ns]),np.concatenate(ns).astype(np.int32)))
        print(case,'graph',len(graphs),'/',len(states),flush=True)
    support=np.unique(np.concatenate([centers]+[dst for src,dst in graphs]));offsets=np.r_[0,np.cumsum([len(src) for src,dst in graphs])]
    np.savez_compressed(folder/'graphs.npz',support_indices=support,support_ids=ref['ids'][support],centers=np.searchsorted(support,centers),masks=masks,compositions=composition,sampled=sample,observed=ta[support],src=np.concatenate([a for a,b in graphs]),dst=np.searchsorted(support,np.concatenate([b for a,b in graphs])),offsets=offsets)
    # Reconciliation of every sampled primary edge by original ID, not float statistics alone.
    gm=read(roots(r)[2]/case/'graph_manifest.json');fullmap={int(i):j for j,i in enumerate(ref['ids'][centers[sample]])}
    for arm in ('baseline','vibration'):
        for phase in range(9):
            j=next(i for i,s in enumerate(gm['states']) if s['arm']==arm and s['phase_index']==phase)
            lo,hi=old['offsets'][j:j+2];a,b=graphs[choice[arm][phase]];use=sample[a]
            ids=ref['ids'][centers[a[use]]];sa=np.array([fullmap[int(i)] for i in ids]);da=ref['ids'][b[use]]
            np.testing.assert_array_equal(sa,old['src'][lo:hi]);np.testing.assert_array_equal(da,old['support_ids'][old['dst'][lo:hi]])
    write_csv(folder/'registration.csv',registration)
    save(folder/'graphs.json',dict(case=case,states=states,choice=choice,sources=sources,graph_sha256=sha(folder/'graphs.npz'),config_sha256=sha(CFG),extractor_sha256=sha(__file__),full_eligible_centers=len(centers),sampled_centers=int(sample.sum()),primary_edges_reconcile=True))

def sparse_alpha(labels,g):
    # labels includes observed row; float64 dot products of integer 0/1 counts.
    lc=labels[:,g['centers']].T.astype(float);result=[]
    for lo,hi in zip(g['offsets'][:-1],g['offsets'][1:]):
        src=g['src'][lo:hi];dst=g['dst'][lo:hi]
        matrix=csr_matrix((np.ones(len(src)),(src,dst)),shape=(len(g['centers']),labels.shape[1]))
        nb=matrix@labels.T.astype(float);degree=np.asarray(matrix.sum(1)).ravel()
        values=[]
        for sample in (g['sampled'],np.ones(len(lc),bool)):
            for mask,c in zip(g['masks'],g['compositions']):
                use=sample&mask;den=(lc[use]*degree[use,None]).sum(0);num=(lc[use]*nb[use]).sum(0)
                values.append(np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)/c)
        result.append(1-np.array(values).T.reshape(len(labels),2,4))
    return np.array(result)

def batch(task):
    r,di,b=task;case=f'S6_R{r:02d}';definition=DEFS[di];folder=OUT/case;dest=folder/f'{definition}_batch{b}.npz'
    if dest.exists():return str(dest)
    t=time.monotonic();g=dict(np.load(folder/'graphs.npz'));meta=read(folder/'graphs.json');oldmc,oldsp,oldgraph=roots(r)
    source=(oldmc/case if di==0 else oldsp/case/definition);pop=np.load(source/('null_population.npz' if di==0 else 'population.npz'));old=np.load(source/f'batch_{b}.npz');seed=int(old['seed']);ta=pop['ta'];strata=pop['strata']
    if di==0:full=assignments(ta,strata,256,seed)[1:]
    else:
        rng=np.random.default_rng(seed);full=np.tile(ta,(256,1));order=np.argsort(strata,kind='stable');_,start=np.unique(strata[order],return_index=True)
        for ix in np.split(order,start[1:]):
            nc=int(ta[ix].sum())
            if 0<nc<len(ix):full[:,ix]=rng.permuted(np.broadcast_to(ta[ix],(256,len(ix))),axis=1)
            assert np.all(full[:,ix].sum(1)==nc)
    np.testing.assert_array_equal(full[:,pop['support']],old['labels'])
    labels=np.concatenate([g['observed'][None],full[:,g['support_indices']]]);del full
    alpha=sparse_alpha(labels,g)
    # The accelerated estimator must reproduce the existing sampled statistics.
    states=read(oldgraph/case/'graph_manifest.json')['states'];expected=old['alpha']
    for arm in ('baseline','vibration'):
        for p in range(9):
            j=next(i for i,s in enumerate(states) if s['arm']==arm and s['phase_index']==p)
            want=expected[j] if di==0 else expected[j,1:]
            np.testing.assert_allclose(alpha[meta['choice'][arm][p],1:,0],want,atol=1e-12,rtol=0)
    np.savez_compressed(dest,alpha=alpha,labels=labels[1:],seed=seed)
    save(dest.with_suffix('.json'),dict(source_batch=str(source/f'batch_{b}.npz'),source_sha256=sha(source/f'batch_{b}.npz'),population_path=str(source/('null_population.npz' if di==0 else 'population.npz')),population_sha256=sha(source/('null_population.npz' if di==0 else 'population.npz')),output_sha256=sha(dest),graph_sha256=meta['graph_sha256'],exact_label_reuse=True,primary_reconciliation_tolerance=1e-12,seconds=time.monotonic()-t))
    print(case,definition,b,'seconds',round(time.monotonic()-t),flush=True);return str(dest)

def summarize():
    result=[];registration=[];primary=[]
    def stats(v):
        null=v[1:];assert np.isfinite(v).all()
        return dict(observed=float(v[0]),null_mean=float(null.mean()),conditional=float(v[0]-null.mean()),conditional_sd=float(null.std(ddof=1)),mcse=float(null.std(ddof=1)/np.sqrt(len(null))),draws=len(null))
    for r in (1,2,3):
        case=f'S6_R{r:02d}';folder=OUT/case;choice=read(folder/'graphs.json')['choice'];g=np.load(folder/'graphs.npz')
        for definition in DEFS:
            batches=[np.load(folder/f'{definition}_batch{b}.npz') for b in range(1,5)]
            a=np.concatenate([batches[0]['alpha'][:,:1]]+[b['alpha'][:,1:] for b in batches],axis=1)
            p=a[choice['vibration']]-a[choice['baseline']]
            np.savez_compressed(folder/f'{definition}_statistics.npz',alpha=a,paired=p)
            for si,pop in enumerate(('sampled','all_centers')):
                for j,region in enumerate(REGIONS):
                    for name,v in [('phase_mean',p[:,:,si,j].mean(0)),('cycle_increment',p[-1,:,si,j]-p[0,:,si,j])]:
                        result.append(dict(case=case,definition=definition,population=pop,region=region,statistic=name,centers=int((g['masks'][j]&(g['sampled'] if si==0 else True)).sum()),**stats(v)))
            if definition in ('depth_only','block16_origin0.5'):
                for name in ('earlier','later'):
                    alt=a[choice['vibration']]-a[choice[name]]
                    for j,region in enumerate(REGIONS):
                        registration.append(dict(case=case,definition=definition,region=region,statistic='phase_mean',choice=name,**stats(alt[:,:,0,j].mean(0))))
                for st,en in itertools.product(('earlier','later'),repeat=2):
                    delta=(a[choice['vibration'][-1]]-a[choice[en][-1]])-(a[choice['vibration'][0]]-a[choice[st][0]])
                    for j,region in enumerate(REGIONS):registration.append(dict(case=case,definition=definition,region=region,statistic='cycle_increment',choice=st+'/'+en,**stats(delta[:,0,j])))
        for oldname in ('turnover_summary.csv','spatial_summary.csv'):
            primary+=rows(roots(r)[1]/oldname) if r==3 else [v for v in rows(roots(r)[1]/oldname) if v['case']==case]
    # 3 reps x 5 references x 2 populations x 4 regions x 2 statistics.
    assert len(result)==240 and len(registration)==144
    write_csv(OUT/'chemical_population_null_summary.csv',result);write_csv(OUT/'chemical_endpoint_summary.csv',registration)
    write_csv(OUT/'primary_full_regional_accounting.csv',primary)
    widths=[q for q in rows(EXT/'width_all.csv') if q['condition']=='A50B20'];assert len(widths)==54
    write_csv(OUT/'A50B20_all_54_width_combinations.csv',widths)
    save(OUT/'summary_manifest.json',dict(protocol_sha256=sha(CFG),script_sha256=sha(__file__),rows=len(result),endpoint_rows=len(registration),primary_regional_rows=len(primary),width_combinations=len(widths),no_MD=True,outputs={p.name:sha(p) for p in OUT.glob('*.csv')}))

def test():
    g=dict(centers=np.array([0,1]),src=np.array([0,0,0,1]),dst=np.array([1,2,3,2]),offsets=np.array([0,4]),masks=np.array([[1,1],[1,0],[0,1],[1,1]],bool),compositions=np.array([.5,.5,.5,.5]),sampled=np.array([1,1],bool))
    labels=np.array([[1,1,0,1],[0,0,1,1]],np.uint8)
    expect=statistics(labels,g['centers'],g['src'],g['dst'],g['masks'],g['compositions']);actual=sparse_alpha(labels,g)[0,:,0]
    np.testing.assert_allclose(expect,actual,atol=1e-14,equal_nan=True)
    assert np.isnan(actual[1]).all();assert actual[0,0]==0 # pooled p=2/4; not mean(2/3,0).
    print('sparse pooled counting and absent-Ta-center tests passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['extract','batches','summarize','test']);p.add_argument('--workers',type=int,default=3);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    test()
    if a.task=='extract':
        with ProcessPoolExecutor(max_workers=min(a.workers,3)) as pool:list(pool.map(extract,(1,2,3)))
    elif a.task=='batches':
        tasks=list(itertools.product((1,2,3),range(5),range(1,5)))
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(batch,tasks))
    elif a.task=='summarize':summarize()
