"""Portable, non-mutating numerical checks of the uncompressed review evidence."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
BASE=Path(os.environ.get('PAPER4_DATA_ROOT', Path(__file__).resolve().parents[1]))
E=BASE/'evidence';NEW=E/'review_20260919'
DEFS=['depth_only','block16_origin0','block16_origin0.5','block32_origin0','block32_origin0.5']
REGIONS=['Cu-rich interior','Cu-rich interface-near','Ta-rich interior','Ta-rich interface-near']
def read(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open()))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def hashes():
    m=read(BASE/'MANIFEST.json')
    for v in m['files']:assert digest(BASE/v['path'])==v['sha256'],v['path']
    return dict(files=len(m['files']),coverage=m['scope'])
def summary(v):
    n=v[1:];return dict(observed=v[0],null_mean=n.mean(),conditional=v[0]-n.mean(),conditional_sd=n.std(ddof=1),mcse=n.std(ddof=1)/np.sqrt(len(n)))
def check(q,v):
    for k,x in summary(v).items():assert abs(float(q[k])-x)<1e-12,(k,q,x)

def chemistry():
    target=rows(NEW/'chemical_population_null_summary.csv');end=rows(NEW/'chemical_endpoint_summary.csv');checks=0
    for r in (1,2,3):
        case=f'S6_R{r:02d}';folder=NEW/case;g=np.load(folder/'graphs.npz');choice=read(folder/'graphs.json')['choice']
        matrices=[]
        for lo,hi in zip(g['offsets'][:-1],g['offsets'][1:]):
            src,dst=g['src'][lo:hi],g['dst'][lo:hi]
            matrices.append(csr_matrix((np.ones(len(src)),(src,dst)),shape=(len(g['centers']),len(g['support_ids']))))
        for d in DEFS:
            batches=[np.load(folder/f'{d}_batch{b}.npz') for b in range(1,5)]
            labels=np.concatenate([g['observed'][None]]+[b['labels'] for b in batches]);a=np.zeros((len(matrices),1025,2,4))
            # Evaluate numerator and denominator separately for each region and
            # population, not by averaging the original residue statistic.
            for i,mat in enumerate(matrices):
                deg=np.asarray(mat.sum(1)).ravel();nn=mat@labels.T.astype(float);central=labels[:,g['centers']].T
                for si,pop in enumerate((g['sampled'],np.ones(len(g['centers']),bool))):
                    for j,mask in enumerate(g['masks']):
                        use=mask&pop;den=deg[use]@central[use];num=np.einsum('ij,ij->j',central[use],nn[use]);a[i,:,si,j]=1-num/den/g['compositions'][j]
            np.testing.assert_allclose(a,np.load(folder/f'{d}_statistics.npz')['alpha'],atol=1e-12,rtol=0)
            paired=a[choice['vibration']]-a[choice['baseline']]
            for q in [q for q in target if q['case']==case and q['definition']==d]:
                si=int(q['population']=='all_centers');j=REGIONS.index(q['region']);v=paired[:,:,si,j]
                check(q,v.mean(0) if q['statistic']=='phase_mean' else v[-1]-v[0]);checks+=1
            for q in [q for q in end if q['case']==case and q['definition']==d]:
                j=REGIONS.index(q['region'])
                if q['statistic']=='phase_mean':v=(a[choice['vibration']]-a[choice[q['choice']]])[:,:,0,j].mean(0)
                else:
                    st,en=q['choice'].split('/');v=((a[choice['vibration'][-1]]-a[choice[en][-1]])-(a[choice['vibration'][0]]-a[choice[st][0]]))[:,0,j]
                check(q,v);checks+=1
            print(case,d,'graph and paired summaries passed',flush=True)
        old=np.load(E/f'primary/{case}/turnover.npz');meta=read(E/f'primary/{case}/graph_manifest.json')['states'];lookup={(s['arm'],s['phase_index']):i for i,s in enumerate(meta)}
        for ai,arm in enumerate(('baseline','vibration')):
            direct=np.array([old['alpha'][lookup[arm,p]]-old['alpha'][lookup[arm,0]] for p in range(9)])
            np.testing.assert_allclose(old['parts'][ai].sum(-1),direct,atol=1e-12,rtol=0)
    assert checks==384;return dict(summary_checks=checks,graphs=3*5*27,assignments_per_reference=1024)

def labels():
    count=0
    for r in (1,2,3):
        case=f'S6_R{r:02d}';g=np.load(NEW/case/'graphs.npz')
        for d in DEFS:
            pop=np.load(E/f'nulls/{case}/{d}/population.npz');ta=pop['ta'];strata=pop['strata'];order=np.argsort(strata,kind='stable');_,starts=np.unique(strata[order],return_index=True);groups=np.split(order,starts[1:])
            for b in range(1,5):
                saved=np.load(NEW/case/f'{d}_batch{b}.npz');rng=np.random.default_rng(int(saved['seed']));full=np.tile(ta,(256,1))
                for ix in groups:
                    if d=='depth_only':
                        for k in range(256):full[k,ix]=rng.permutation(ta[ix])
                    elif 0<int(ta[ix].sum())<len(ix):full[:,ix]=rng.permuted(np.broadcast_to(ta[ix],(256,len(ix))),axis=1)
                    assert np.all(full[:,ix].sum(1)==ta[ix].sum())
                np.testing.assert_array_equal(full[:,g['support_indices']],saved['labels']);count+=1
            print(case,d,'full-population conservation and seeded labels passed',flush=True)
    return dict(batches=count,full_population_assignments=count*256)

def profile(x0,x,labels,z0,material):
    F=np.linalg.lstsq(x0-x0.mean(0),x-x.mean(0),rcond=None)[0]
    if len(x0)<50 or np.linalg.cond(F)>100:return np.nan
    cov=np.linalg.inv(F)[:,2];stretch=1/np.linalg.norm(cov);q=(x-x.mean(0))@cov+x0[:,2].mean()-z0
    grid=np.linspace(-6,6,161);dist=q
    if not material:grid=grid*stretch;dist=q*stretch
    label=labels if labels[x0[:,2]>z0].mean()>labels[x0[:,2]<z0].mean() else 1-labels
    w=np.exp(-.5*((grid[:,None]-dist)/.75)**2);support=w.sum(1);f=w@label/np.maximum(support,1e-100);cross=[]
    for level in (.1,.5,.9):
        ix=np.flatnonzero((f[:-1]<level)&(f[1:]>=level)&(support[:-1]>=3)&(support[1:]>=3))
        if not len(ix):return np.nan
        v=grid[ix]+(level-f[ix])*(grid[ix+1]-grid[ix])/(f[ix+1]-f[ix]);cross.append(v[np.argmin(abs(v))])
    return (cross[2]-cross[0])/(1 if material else stretch) if cross[0]<cross[1]<cross[2] else np.nan
def widths():
    largest=0.;checks=0
    for r in (1,2,3):
        folder=E/f'primary/S6_R{r:02d}';a=np.load(folder/'coordinates.npz');meta=read(folder/'coordinates.json');w=np.load(folder/'widths.npz');lookup={(s['arm'],s['phase_index']):i for i,s in enumerate(meta['states'])};actual=np.full_like(w['widths'],np.nan)
        for ai,arm in enumerate(('baseline','vibration')):
            for phase in range(9):
                for ti,info in enumerate(a['tile_info']):
                    ids=a['tile_atoms'][a['tile_offsets'][ti]:a['tile_offsets'][ti+1]];x0=a['xyz'][0,ids];v=a['xyz'][lookup[arm,phase],ids]-x0;v[:,1]-=float(a['ly'])*np.rint(v[:,1]/float(a['ly']));xx=x0+v
                    for ei in (0,1):actual[ei,ai,phase,ti]=profile(x0,xx,np.isin(a['types'][ids],[1,2]),info[3],bool(ei));checks+=1
        np.testing.assert_allclose(actual,w['widths'],atol=1e-9,rtol=0,equal_nan=True);largest=max(largest,float(np.nanmax(abs(actual-w['widths']))))
        print('R0'+str(r),'both width estimators passed',flush=True)
    return dict(profiles=checks,maximum_error_A=largest)
def cycles():
    expected=rows(E/'matrices/width_all.csv');lookup={(r['case'],int(r['station_nm']),r['variant']):r for r in expected};count=0;maxerr=0.
    for mp in (E/'cycles').glob('*/manifest.json'):
        m=read(mp);records=rows(mp.parent/'widths.csv');key=(m['case']['case'],m['station_nm'])
        for var in sorted({r['variant'] for r in records}):
            rr=[r for r in records if r['variant']==var];tile=lambda r:(r['interface'],r['tile_x'],r['tile_y'])
            common=set.intersection(*[{tile(r) for r in rr if r['arm']==arm and int(r['phase_index'])==p and r['valid']=='True'} for arm in ('baseline','vibration') for p in range(9)])
            means={arm:[np.mean([float(r['width_corrected_A']) for r in rr if r['arm']==arm and int(r['phase_index'])==p and tile(r) in common]) for p in (0,8)] for arm in ('baseline','vibration')}
            value=(means['vibration'][1]-means['vibration'][0])-(means['baseline'][1]-means['baseline'][0]);q=lookup[key+(var,)];assert len(common)==int(q['valid_tiles']);err=abs(value-float(q['width_increment_A']));assert err<1e-10;maxerr=max(err,maxerr);count+=1
    assert count==270;assert len(rows(NEW/'A50B20_all_54_width_combinations.csv'))==54
    return dict(width_combinations=count,A50B20_combinations=54,maximum_error_A=maxerr)
def forces():
    cache={};maxerr=0.;target=rows(E/'matrices/contact_all.csv')
    for q in target:
        run=q['run']
        if run not in cache:cache[run]=np.load(E/f'forces/{run}.npz')['samples']
        a=cache[run];t=a[:,1];st,en=float(q['start_ps']),float(q['end_ps']);mask=(t>st)&(t<en);assert t[0]<=st<en<=t[-1]
        for field,c in [('Ft',3),('Fn',4)]:
            y=a[:,c]*1.602176634;tt=np.r_[st,t[mask],en];yy=np.r_[np.interp(st,t,y),y[mask],np.interp(en,t,y)]
            trap=getattr(np,'trapezoid',None) or np.trapz;value=trap(yy,tt)/(en-st);err=abs(value-float(q[field+'_mean_nN']));assert err<1e-10;maxerr=max(err,maxerr)
    assert len(target)==90 and len(cache)==18;return dict(windows=len(target),runs=len(cache),maximum_error_nN=maxerr)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','hashes','chemistry','labels','widths','cycles','forces'],default='all');a=p.parse_args();result={}
    for stage in (['hashes','chemistry','widths','cycles','forces'] if a.stage=='all' else [a.stage]):
        result[stage]=globals()[stage]();print(stage,json.dumps(result[stage]),flush=True)
    print(json.dumps(dict(status='passed',checks=result),indent=2))
