"""Portable numerical reconstruction of the bounded withdrawal/hold extension.

python rebuild_unloading_20260921.py --data DIR --output NEW_DIR [--verify]
Only reduced coordinates/graphs, fixed masks and stored assignments are needed.
No simulation, source mutation, external paths or outcome-based selection.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

TIMES=[0,100,150,175,200]
REGIONS=['Cu-rich interior','Cu-rich interface-near','Ta-rich interior','Ta-rich interface-near']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def rows(p):return list(csv.DictReader(Path(p).open()))
def write(p,r):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def mic(d,ly):
    d=d.copy();d[...,1]-=ly*np.rint(d[...,1]/ly);return d
def shells(x,c,ly):
    support=np.concatenate([x+[0,k*ly,0] for k in (-1,0,1)])
    owner=np.tile(np.arange(len(x)),3);tree=cKDTree(support)
    return [np.setdiff1d(np.unique(owner[a]),i) for i,a in zip(c,tree.query_ball_point(x[c],3.375,workers=1))]
def dm(x0,x,c,ns,ly):
    ans=[]
    for i,js in zip(c,ns):
        A=mic(x0[js]-x0[i],ly);B=mic(x[js]-x[i],ly)
        if len(js)<4 or np.linalg.cond(A)>1e6:ans.append(np.nan);continue
        ans.append(float(np.mean(np.sum((B-A@np.linalg.lstsq(A,B,rcond=None)[0])**2,axis=1))))
    return np.array(ans)
def width(x0,x,origin,z,material):
    if len(x0)<50:return np.nan
    A=x0-x0.mean(0);B=x-x.mean(0);F=np.linalg.lstsq(A,B,rcond=None)[0]
    if np.linalg.cond(F)>100:return np.nan
    cov=np.linalg.inv(F)[:,2];stretch=1/np.linalg.norm(cov)
    q=B@cov+x0[:,2].mean()-z
    label=origin if origin[x0[:,2]>z].mean()>origin[x0[:,2]<z].mean() else 1-origin
    grid=np.linspace(-6,6,161);sigma=.75 if material else .75/stretch
    K=np.exp(-.5*((grid[:,None]-q)/sigma)**2);support=K.sum(1);f=K@label/np.maximum(support,1e-100);cross=[]
    for level in (.1,.5,.9):
        candidates=[grid[k]+(level-f[k])*(grid[k+1]-grid[k])/(f[k+1]-f[k])
                    for k in range(160) if f[k]<level<=f[k+1] and min(support[k:k+2])>=3]
        if not candidates:return np.nan
        cross.append(min(candidates,key=abs))
    return cross[2]-cross[0] if cross[0]<cross[1]<cross[2] else np.nan
def stats(v):
    return dict(observed=float(v[0]),null=float(v[1:].mean()),conditional=float(v[0]-v[1:].mean()),
        conditional_sd=float(v[1:].std(ddof=1)),mcse=float(v[1:].std(ddof=1)/np.sqrt(len(v)-1)))

def run(data,out,verify):
    out.mkdir(parents=True,exist_ok=True);paired=[];chemical=[];diagnostics=[];tiles=[];width_summary=[];checks=[]
    main=[];supp=[];max_width_error=0.;max_activity_error=0.;max_chemical_error=0.
    for r in (1,2,3):
        d=data/f'R{r:02d}';m=json.loads((d/'manifest.json').read_text())
        for n,h in m['outputs'].items():assert sha(d/n)==h,n
        a=np.load(d/'coordinates_graphs.npz');ch=np.load(d/'chemical.npz')['alpha'];reg=rows(d/'regional.csv')
        lookup={(v['arm'],int(v['time_ps']),v['region']):v for v in reg}
        common=np.isfinite(a['widths']).all(axis=(0,1,2))&np.isfinite(a['relaxed_widths']).all(0)
        assert np.flatnonzero(common).tolist()==m['common_tiles']
        assert all(v['cutoff_pairs']==0 for v in m['states'] if v['time_ps']>=100)
        if verify:
            x0=a['xyz0'];c=a['centers'];ly=float(a['ly']);n0=shells(x0,c,ly)
            labs=np.concatenate([np.isin(a['types'],[2,4])[None,:]]+[np.load(d/f'batch{b}.npz')['labels'] for b in range(1,5)])
            for ai,arm in enumerate(('baseline','vibration')):
                xb=a['xyz'][5*ai];nb=shells(xb,c,ly);nl=shells(a['xyz'][5*ai+2],c,ly)
                for k,t in enumerate(TIMES):
                    si=5*ai+k;x=a['xyz'][si];ns=shells(x,c,ly)
                    src=np.repeat(np.arange(len(c)),[len(js) for js in ns]);dst=np.concatenate(ns)
                    lo,hi=a['offsets'][si:si+2]
                    np.testing.assert_array_equal(src,a['src'][lo:hi]);np.testing.assert_array_equal(dst,a['dst'][lo:hi])
                    v=np.array([len(np.intersect1d(p,q))/len(p) for p,q in zip(n0,ns)])
                    np.testing.assert_allclose(v,a['per_atom'][si,0],atol=0,rtol=0)
                    v=np.array([len(np.intersect1d(p,q))/len(p) for p,q in zip(nb,ns)])
                    np.testing.assert_allclose(v,a['per_atom'][si,1],atol=0,rtol=0)
                    for field,base,shell in [(2,x0,n0),(3,xb,nb)]+([(4,a['xyz'][5*ai+2],nl)] if t==200 else []):
                        actual=dm(base,x,c,shell,ly);err=float(np.max(abs(actual-a['per_atom'][si,field])))
                        max_activity_error=max(max_activity_error,err)
                        np.testing.assert_allclose(actual,a['per_atom'][si,field],rtol=1e-12,atol=1e-10)
                    for ti,info in enumerate(a['tile_info']):
                        ix=a['tile_atoms'][a['tile_offsets'][ti]:a['tile_offsets'][ti+1]]
                        xx=x0[ix]+mic(x[ix]-x0[ix],ly)
                        for ei,material in enumerate((False,True)):
                            w=width(x0[ix],xx,np.isin(a['types'][ix],[1,2]),info[3],material)
                            old=a['widths'][ei,ai,k,ti]
                            np.testing.assert_allclose(w,old,atol=1e-9,rtol=0,equal_nan=True)
                            if np.isfinite(w):max_width_error=max(max_width_error,abs(w-old))
                    for j,mask in enumerate(a['masks']):
                        use=mask[src];sc=c[src[use]];de=dst[use]
                        # Chunk assignments to bound RAM; sum integer directed-edge counts.
                        for start in range(0,1025,64):
                            lab=labs[start:start+64];central=lab[:,sc].astype(bool)
                            M=central.sum(1);Q=(central&lab[:,de].astype(bool)).sum(1)
                            val=1-Q/M/a['compositions'][j]
                            err=float(np.max(abs(val-ch[ai,k,start:start+len(lab),j])));max_chemical_error=max(max_chemical_error,err)
                            assert err<1e-12
            print(f'R{r:02d}: graphs, per-atom fields, widths and 1024 chemical draws reconstructed',flush=True)
        wp=a['widths'][:,1]-a['widths'][:,0]
        for ei,est in enumerate(('physical','material')):
            for k,t in enumerate(TIMES):
                vals=wp[ei,k,common]
                width_summary.append(dict(realization=r,estimator=est,time_ps=t,tiles=int(common.sum()),
                    baseline_A=float(a['widths'][ei,0,k,common].mean()),vibration_A=float(a['widths'][ei,1,k,common].mean()),
                    paired_mean_A=float(vals.mean()),paired_median_A=float(np.median(vals)),negative_tiles=int((vals<0).sum()),
                    min_A=float(vals.min()),max_A=float(vals.max())))
                for ti in np.flatnonzero(common):tiles.append(dict(realization=r,estimator=est,time_ps=t,tile=int(ti),paired_A=float(wp[ei,k,ti])))
        for j,region in enumerate(REGIONS):
            p=ch[1,:,:,j]-ch[0,:,:,j]
            for k,t in enumerate(TIMES):
                B=lookup['baseline',t,region];V=lookup['vibration',t,region]
                row=dict(realization=r,region=region,time_ps=t,retention_pp=100*(float(V['initial_retention'])-float(B['initial_retention'])),
                    D_initial_A2=float(V['D_initial_A2'])-float(B['D_initial_A2']),**stats(p[k]))
                paired.append(row)
                for ai,arm in enumerate(('baseline','vibration')):chemical.append(dict(realization=r,arm=arm,region=region,time_ps=t,**stats(ch[ai,k,:,j])))
            for label,v in [('withdrawal_hold',p[-1]-p[0]),('late_hold',p[-1]-p[2])]:
                chemical.append(dict(realization=r,arm='paired_change',region=region,time_ps=label,**stats(v)))
            b=lookup['baseline',200,region];v=lookup['vibration',200,region]
            diagnostics.append(dict(realization=r,region=region,
                B_branch_retention_pct=100*float(b['branch_retention']),V_branch_retention_pct=100*float(v['branch_retention']),
                B_D_branch_A2=float(b['D_branch_A2']),V_D_branch_A2=float(v['D_branch_A2']),
                B_D_late_A2=float(b['D_late_A2']),V_D_late_A2=float(v['D_late_A2']),
                paired_chemical_late_change=stats(p[-1]-p[2])['conditional'],paired_chemical_late_mcse=stats(p[-1]-p[2])['mcse']))
        rows0=[v for v in paired if v['realization']==r and v['region']==REGIONS[0]]
        vals=[]
        for col in ('retention_pp','D_initial_A2'):
            vals.extend([f"{rows0[k][col]:+.2f}" for k in (0,4)])
        vals.extend([f'{wp[1,k,common].mean():+.3f}' for k in (0,4)])
        vals.extend([f"${rows0[k]['conditional']:+.4f}\\pm{rows0[k]['mcse']:.4f}$" for k in (0,4)])
        main.append(f'R{r:02d} & '+' & '.join(vals)+r' \\')
        checks.append(dict(realization=r,common_tiles=int(common.sum()),excluded_tiles=m['excluded_tiles'],raw_checks=m['checks'],centers=m['centers']))
    write(out/'paired.csv',paired);write(out/'chemical.csv',chemical);write(out/'late_diagnostics.csv',diagnostics)
    write(out/'width_summary.csv',width_summary);write(out/'tile_contrasts.csv',tiles)
    header=[r'\begin{fullwidthtabular}{lrrrrrrrr}',r'\toprule',
        r' & \multicolumn{2}{c}{$\Delta r_{\rm init}$ (pp)} & \multicolumn{2}{c}{$\Delta\langle\overline D^2_{\min}\rangle$ (\AA$^2$)} & \multicolumn{2}{c}{$\Delta W_q$ (\AA)} & \multicolumn{2}{c}{$\Delta u$} \\',
        r'Prep. & Loaded & Held & Loaded & Held & Loaded & Held & Loaded & Held \\',r'\midrule']
    (out/'main_table.tex').write_text('\n'.join(header+main+[r'\bottomrule',r'\end{fullwidthtabular}'])+'\n')
    for region in REGIONS:
        supp.append(r'\multicolumn{9}{l}{'+region+r'} \\')
        for d in diagnostics:
            if d['region']!=region:continue
            cols=[f"{d[k]:.2f}" for k in ('B_branch_retention_pct','V_branch_retention_pct','B_D_branch_A2','V_D_branch_A2')]
            cols += [f"{d[k]:.3f}" for k in ('B_D_late_A2','V_D_late_A2')]
            cols += [f"{d['paired_chemical_late_change']:+.4f}",f"{d['paired_chemical_late_mcse']:.4f}"]
            supp.append(f"R{d['realization']:02d} & "+' & '.join(cols)+r' \\')
    header=[r'\begin{fullwidthtabular}{lrrrrrrrr}',r'\toprule',
        r' & \multicolumn{2}{c}{$r_{\rm branch}(200)$ (\%)} & \multicolumn{2}{c}{$D_{0\to200}$ (\AA$^2$)} & \multicolumn{2}{c}{$D_{150\to200}$ (\AA$^2$)} & \multicolumn{2}{c}{$\Delta u_{200}-\Delta u_{150}$} \\',
        r'Prep. & B & V & B & V & B & V & Change & MCSE \\',r'\midrule']
    (out/'supp_table.tex').write_text('\n'.join(header+supp+[r'\bottomrule',r'\end{fullwidthtabular}'])+'\n')
    residual=[v for v in paired if v['time_ps']==200 and v['region']==REGIONS[0]]
    assert len(residual)==3 and all(v['retention_pp']<0 and v['D_initial_A2']>0 for v in residual)
    finalw=[v for v in width_summary if v['estimator']=='material' and v['time_ps']==200]
    sign_check={str(v['realization']):dict(retention_negative=v['retention_pp']<0,activity_positive=v['D_initial_A2']>0,
        width_positive=next(w['paired_mean_A'] for w in finalw if w['realization']==v['realization'])>0) for v in residual}
    save(out/'verification.json',dict(verified_reduced_reconstruction=verify,records=checks,
        maximum_activity_error_A2=max_activity_error,maximum_width_error_A=max_width_error,maximum_chemical_error=max_chemical_error,
        retained_signs_by_preparation=sign_check,without_recovered_R03='R01 and R02 retain all three geometry signs; opposite-sign R03 chemistry is not claimed without R03',
        scope='Author-run numerical reconstruction; not physical convergence, athermal irreversibility or independent reproduction',
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='verification.json'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args();run(a.data,a.output,a.verify)
