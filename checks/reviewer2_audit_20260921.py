"""Bounded, portable review audit. Reads sealed evidence; never starts MD.

Usage: python reviewer2_audit_20260921.py --evidence PATH --output NEW_DIRECTORY
The output directory contains numerical ledgers and data-derived profile plots.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

DEFS = ['depth_only', 'block16_origin0', 'block16_origin0.5',
        'block32_origin0', 'block32_origin0.5']
REGIONS = ['Cu-rich interior', 'Cu-rich interface-near',
           'Ta-rich interior', 'Ta-rich interface-near']

def read(path): return json.loads(Path(path).read_text())
def rows(path): return list(csv.DictReader(Path(path).open()))
def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def write(path, records):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
def summary(v):
    return dict(observed=float(v[0]), null_mean=float(v[1:].mean()),
                conditional=float(v[0]-v[1:].mean()),
                conditional_sd=float(v[1:].std(ddof=1)),
                mcse=float(v[1:].std(ddof=1)/np.sqrt(len(v)-1)))

def chemical(e, out):
    ledger=[]; reconciliation=[]; convergence=[]; occupancy=[]; weighted=[]
    closure=0.; observed_error=0.
    for r in (1,2,3):
        case=f'S6_R{r:02d}'; primary=e/'primary'/case
        a=np.load(primary/'turnover.npz'); g=np.load(primary/'graphs.npz')
        meta=read(primary/'graph_manifest.json')
        lookup={(s['arm'],s['phase_index']):i for i,s in enumerate(meta['states'])}
        # Each arm and assignment must close before subtraction between arms.
        for ai,arm in enumerate(('baseline','vibration')):
            for phase in range(9):
                counts=a['counts'][ai,phase]
                c0=counts[:,:,0]+counts[:,:,1]; c1=counts[:,:,0]+counts[:,:,2]
                M0=c0[...,2:].sum(-1); M1=c1[...,2:].sum(-1)
                p0=c0[...,3]/M0; p1=c1[...,3]/M1
                alpha0=1-p0/g['compositions']; alpha1=1-p1/g['compositions']
                direct=alpha1-alpha0
                err=float(np.max(abs(direct-a['parts'][ai,phase].sum(-1))))
                closure=max(closure,err); assert err<1e-12
                np.testing.assert_allclose(alpha1,a['alpha'][lookup[arm,phase]],atol=1e-12,rtol=0)
                gi=lookup[arm,phase];lo,hi=g['offsets'][gi:gi+2]
                src,dst=g['src'][lo:hi],g['dst'][lo:hi];labels=g['labels'][0]
                for j,mask in enumerate(g['masks']):
                    use=mask[src] & labels[g['centers'][src]].astype(bool)
                    assert int(use.sum())==int(M1[0,j])
                    assert int(labels[dst[use]].sum())==int(c1[0,j,3])
                if phase==8:
                    for j,region in enumerate(REGIONS):
                        ledger.append(dict(case=case,arm=arm,region=region,
                            centers=int(g['masks'][j].sum()),composition=float(g['compositions'][j]),
                            Q0=int(c0[0,j,3]),M0=int(M0[0,j]),Q1=int(c1[0,j,3]),M1=int(M1[0,j]),
                            p0=float(p0[0,j]),p1=float(p1[0,j]),alpha0=float(alpha0[0,j]),
                            alpha1=float(alpha1[0,j]),delta_alpha=float(direct[0,j]),
                            gained=float(a['parts'][ai,phase,0,j,0]),lost=float(a['parts'][ai,phase,0,j,1])))
        paired=a['parts'][1,-1]-a['parts'][0,-1]
        direct=(a['alpha'][lookup['vibration',8]]-a['alpha'][lookup['vibration',0]])-(a['alpha'][lookup['baseline',8]]-a['alpha'][lookup['baseline',0]])
        np.testing.assert_allclose(paired.sum(-1),direct,atol=1e-12,rtol=0)
        np.testing.assert_allclose(paired[0].sum(-1)-paired[1:].sum(-1).mean(0),direct[0]-direct[1:].mean(0),atol=1e-12,rtol=0)
        folder=e/'review_20260919'/case; gm=read(folder/'graphs.json'); choice=gm['choice']
        olddir='registration_R03' if r==3 else 'registration_R01_R02'
        old=np.load(e/'supporting'/olddir/f'{case}_residue_statistics.npz')['alpha']
        oldpaired=old[::2,3]-old[1::2,3] # archived order: vibration, baseline.
        for definition in DEFS:
            z=np.load(folder/f'{definition}_statistics.npz'); v=z['paired']
            if definition=='depth_only':
                for arm in ('baseline','vibration'):
                    for phase in range(9):
                        np.testing.assert_allclose(a['alpha'][lookup[arm,phase]],z['alpha'][choice[arm][phase],:,0,:],atol=1e-12,rtol=0)
            seeds=[int(np.load(folder/f'{definition}_batch{b}.npz')['seed']) for b in range(1,5)]
            pop=np.load(e/'nulls'/case/definition/'population.npz')
            _,inverse,size=np.unique(pop['strata'],return_inverse=True,return_counts=True)
            ta=np.bincount(inverse,weights=pop['ta']); mixed=(ta>0)&(ta<size)
            occupancy.append(dict(case=case,definition=definition,population=len(pop['ta']),
                occupied_strata=len(size),single_species_strata=int((~mixed).sum()),
                mixed_strata=int(mixed.sum()),mutable_atoms=int(size[mixed].sum()),
                mutable_fraction=float(size[mixed].sum()/size.sum()),
                size_min=int(size.min()),size_median=float(np.median(size)),size_max=int(size.max()),
                seeds=';'.join(map(str,seeds))))
            for pi,popname in enumerate(('original','all_centers')):
                for j,region in enumerate(REGIONS):
                    for stat,values in [('phase_mean',v[:,:,pi,j].mean(0)),('cycle_increment',v[-1,:,pi,j]-v[0,:,pi,j])]:
                        for b in range(4):
                            sample=np.r_[values[0],values[1+b*256:1+(b+1)*256]]
                            convergence.append(dict(case=case,definition=definition,population=popname,region=region,
                                statistic=stat,batch=b+1,seed=seeds[b],draws=256,**summary(sample)))
                        for n in (256,512,768,1024):
                            convergence.append(dict(case=case,definition=definition,population=popname,region=region,
                                statistic=stat,batch='nested',seed=';'.join(map(str,seeds[:n//256])),draws=n,
                                **summary(values[:n+1])))
                    # Physical-time weighting per arm; no resampling or new fit.
                    arms=[]
                    for arm in ('baseline','vibration'):
                        inds=choice[arm]; times=np.array([float(gm['states'][i]['time_ps']) for i in inds])
                        arms.append(np.trapezoid(z['alpha'][inds,:,pi,j],times,axis=0)/(times[-1]-times[0]))
                    weighted.append(dict(case=case,definition=definition,population=popname,region=region,
                        statistic='time_weighted',**summary(arms[1]-arms[0])))
                    weighted.append(dict(case=case,definition=definition,population=popname,region=region,
                        statistic='nine_frame_arithmetic',**summary(v[:,:,pi,j].mean(0))))
            if definition=='depth_only':
                err=float(np.max(abs(oldpaired[:,0]-v[:,0,1,0])));observed_error=max(err,observed_error);assert err<1e-12
                for stat,oldval,newval in [('phase_mean',oldpaired.mean(0),v[:,:,1,0].mean(0)),
                    ('cycle_increment',oldpaired[-1]-oldpaired[0],v[-1,:,1,0]-v[0,:,1,0])]:
                    oldseed=read(e/'supporting'/olddir/f'{case}_support_manifest.json')['label_seed']
                    for n,vals,seed in [(64,oldval,str(oldseed)),(1024,newval,';'.join(map(str,seeds)))]:
                        reconciliation.append(dict(case=case,statistic=stat,centers='all Cu-rich interior',
                            draws=n,seeds=seed,**summary(vals)))
    write(out/'arm_endpoints.csv',ledger);write(out/'center_ensemble_reconciliation.csv',reconciliation)
    write(out/'null_occupancy.csv',occupancy);write(out/'null_convergence.csv',convergence)
    write(out/'temporal_weighting.csv',weighted)
    return dict(maximum_arm_closure_error=closure,maximum_all_center_observed_mismatch=observed_error,
                regions=4,preparations=3,assignments_per_primary_reference=1024)

def profile(x0,x,origin,z,material):
    J=np.linalg.lstsq(x0-x0.mean(0),x-x.mean(0),rcond=None)[0]
    cov=np.linalg.solve(J,[0.,0.,1.]);stretch=1/np.linalg.norm(cov)
    q=(x-x.mean(0))@cov+x0[:,2].mean()-z
    scale=1 if material else stretch;grid=np.linspace(-6,6,161)*scale
    lab=origin if origin[x0[:,2]>z].mean()>origin[x0[:,2]<z].mean() else 1-origin
    w=np.exp(-.5*((grid[:,None]-q*scale)/.75)**2);support=w.sum(1);frac=w@lab/support
    crossings=[];brackets=[]
    for level in (.1,.5,.9):
        ix=np.flatnonzero((frac[:-1]<level)&(frac[1:]>=level)&(support[:-1]>=3)&(support[1:]>=3))
        if len(ix):
            values=grid[ix]+(level-frac[ix])*(grid[ix+1]-grid[ix])/(frac[ix+1]-frac[ix])
            k=int(np.argmin(abs(values)));jj=int(ix[k]);cross=float(values[k]);crossings.append(cross/scale)
            brackets.append(dict(level=level,selected_q=cross/scale,bracket_q_left=float(grid[jj]/scale),
                bracket_q_right=float(grid[jj+1]/scale),left_support=float(support[jj]),right_support=float(support[jj+1]),
                supported_crossing_count=len(ix)))
        else:crossings.append(np.nan)
    valid=bool(len(x0)>=50 and np.linalg.cond(J)<=100 and np.isfinite(crossings).all() and crossings[0]<crossings[1]<crossings[2])
    residual=x-x.mean(0)-(x0-x0.mean(0))@J
    return dict(grid=grid/scale,fraction=frac,support=support,crossings=crossings,brackets=brackets,
                valid=valid,stretch=float(stretch),condition=float(np.linalg.cond(J)),
                reference_condition=float(np.linalg.cond(x0-x0.mean(0))),determinant=float(np.linalg.det(J)),
                affine_rms_A=float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),
                width=float(crossings[2]-crossings[0]) if valid else np.nan)

def profiles(e,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8})
    state_rows=[];cross_rows=[];curve_rows=[];packets=[]
    fig,axes=plt.subplots(2,4,figsize=(12,5.4),sharex=True,sharey=True)
    maximum=0.
    for ri,(r,tile) in enumerate(((1,22),(2,25))):
        folder=e/f'primary/S6_R{r:02d}';a=np.load(folder/'coordinates.npz');m=read(folder/'coordinates.json');w=np.load(folder/'widths.npz')['widths']
        lookup={(s['arm'],s['phase_index']):i for i,s in enumerate(m['states'])}
        ix=a['tile_atoms'][a['tile_offsets'][tile]:a['tile_offsets'][tile+1]];x0=a['xyz'][0,ix];info=a['tile_info'][tile]
        packets.append(dict(realization=r,tile_zero_based=tile,interface=int(info[0]),tile_x=int(info[1]),tile_y=int(info[2]),interface_z_A=float(info[3]),atoms=len(ix)))
        for ai,arm in enumerate(('baseline','vibration')):
            for phase in range(9):
                x=a['xyz'][lookup[arm,phase],ix].copy();delta=x-x0;delta[:,1]-=float(a['ly'])*np.rint(delta[:,1]/float(a['ly']));x=x0+delta
                for ei,material in enumerate((False,True)):
                    v=profile(x0,x,np.isin(a['types'][ix],[1,2]),info[3],material)
                    expected=w[ei,ai,phase,tile];assert np.isnan(expected)==np.isnan(v['width'])
                    if v['valid']:maximum=max(maximum,abs(v['width']-expected));assert abs(v['width']-expected)<1e-9
                    ident=dict(realization=r,tile=tile,arm=arm,phase=phase,step=m['states'][lookup[arm,phase]]['step'],estimator='material' if material else 'physical')
                    state_rows.append(dict(ident,atoms=len(ix),**{k:v[k] for k in ('valid','stretch','condition','reference_condition','determinant','affine_rms_A','width')}))
                    cross_rows.extend(dict(ident,**q) for q in v['brackets'])
                    curve_rows.extend(dict(ident,q_A=float(q),fraction=float(f),support=float(s)) for q,f,s in zip(v['grid'],v['fraction'],v['support']))
                    if phase in (0,8):
                        ax=axes[ri,ai*2+(phase==8)];color='#315caa' if material else '#bd5936'
                        ax.plot(v['grid'],v['fraction'],color=color,lw=1.4,label='Material bandwidth' if material else 'Physical bandwidth')
                        for q in v['brackets']:ax.plot(q['selected_q'],q['level'],'o',ms=3,color=color)
                        if material:
                            ax.set_title(f'R0{r}, tile {tile}: {"V" if ai else "B"} {"end" if phase else "start"}\n'+f'$\\lambda_n$={v["stretch"]:.3f}, cond $F$={v["condition"]:.2f}')
                            for level in (.1,.5,.9):ax.axhline(level,color='.8',lw=.5,ls=':')
                            ax.set_xlim(-6,6);ax.set_ylim(-.03,1.03);ax.set_xlabel('Material-normal coordinate q (Å)')
                            if ai==0 and phase==0:ax.set_ylabel('Oriented layer-origin fraction')
        fig.tight_layout(rect=(0,0.07,1,1))
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.savefig(out/'influential_profiles.pdf');fig.savefig(out/'influential_profiles.png',dpi=220);plt.close(fig)
    write(out/'profile_states.csv',state_rows);write(out/'profile_crossing_brackets.csv',cross_rows)
    write(out/'profile_curves.csv',curve_rows);write(out/'profile_packet_identity.csv',packets)
    # Identify all displayed physical-width packets, rather than guessing from rounded widths.
    identity=[]
    for r in (1,2,3):
        folder=e/f'primary/S6_R{r:02d}';a=np.load(folder/'coordinates.npz');w=np.load(folder/'widths.npz')['widths']
        for tile,info in enumerate(a['tile_info']):
            vals=w[0,:,[0,8],tile].ravel()
            if abs(w[0,1,8,tile]-[9.770,5.451,5.988][r-1])<.0006:
                identity.append(dict(realization=r,tile=tile,interface=int(info[0]),tile_x=int(info[1]),tile_y=int(info[2]),width_vibration_end_A=float(w[0,1,8,tile])))
    assert len(identity)==3;write(out/'displayed_packet_identity.csv',identity)
    return dict(maximum_width_reproduction_error_A=maximum,displayed_packets=identity,
                excluded_states=[q for q in state_rows if not q['valid']])

def forces(e,out):
    result=[];paired=[];maximum=0.
    for q in rows(e/'matrices/contact_all.csv'):
        a=np.load(e/f'forces/{q["run"]}.npz')['samples'];t=a[:,1];start=float(q['start_ps']);end=float(q['end_ps'])
        assert np.all(np.diff(t)>0) and t[0]<=start<end<=t[-1]
        case=q['case'];station=int(float(q['station_nm']));m=read(e/f'cycles/{case}_s{station:02d}/manifest.json')
        items=[i for i in m['case']['items'] if i['stage']=='cycle' and i['arm']==q['arm']]
        assert abs(float(items[0]['frame']['time_ps'])-start)<1e-9
        rec=dict(case=case,condition=q['condition'],realization=q['realization'],station_nm=station,arm=q['arm'],run=q['run'],
            start_step=int(round(start*1000)),end_step=int(round(end*1000)),start_ps=start,end_ps=end,
            start_nm=float(items[0]['frame']['distance_nm']),end_nm=float(items[-1]['frame']['distance_nm']),
            record_step_min=int(np.diff(a[:,0]).min()),record_step_max=int(np.diff(a[:,0]).max()),
            records_strictly_inside=int(((t>start)&(t<end)).sum()),quadrature_points=int(((t>start)&(t<end)).sum())+2)
        winsteps=a[(t>=start)&(t<=end),0]
        rec.update(window_step_min=int(np.diff(winsteps).min()),window_step_max=int(np.diff(winsteps).max()),window_step_median=float(np.median(np.diff(winsteps))))
        for field,col in [('Ft',3),('Fn',4)]:
            estimates=[]
            for stride in (1,2):
                keep=np.unique(np.r_[np.arange(0,len(t),stride),len(t)-1]);tt=t[keep];v=a[keep,col]*1.602176634
                inside=(tt>start)&(tt<end);xx=np.r_[start,tt[inside],end];yy=np.r_[np.interp(start,tt,v),v[inside],np.interp(end,tt,v)]
                estimates.append(float(np.trapezoid(yy,xx)/(end-start)))
            err=abs(estimates[0]-float(q[field+'_mean_nN']));maximum=max(maximum,err);assert err<1e-10
            rec[field+'_mean_nN']=estimates[0];rec[field+'_coarsening_change_nN']=estimates[1]-estimates[0]
        result.append(rec)
    for case in sorted({q['case'] for q in result}):
        for station in (6,10,14):
            b=next(q for q in result if q['case']==case and q['station_nm']==station and q['arm']=='baseline')
            v=next(q for q in result if q['case']==case and q['station_nm']==station and q['arm']=='vibration')
            paired.append(dict(case=case,station_nm=station,Ft_difference_nN=v['Ft_mean_nN']-b['Ft_mean_nN'],
                Ft_paired_coarsening_change_nN=v['Ft_coarsening_change_nN']-b['Ft_coarsening_change_nN'],
                vibration_coarsening_nN=v['Ft_coarsening_change_nN'],baseline_coarsening_nN=b['Ft_coarsening_change_nN']))
    write(out/'force_window_audit.csv',result);write(out/'force_paired_coarsening.csv',paired)
    assert len(result)==90
    return dict(windows=len(result),maximum_mean_reproduction_error_nN=maximum)

def prediction(e,out):
    cfg=read(e/'prediction/specification.json');base=cfg['model_features_base'];aug=base+cfg['model_features_augmented'];arrays={};ledger=[]
    for p in (e/'prediction').glob('production_*.npz'):
        m=read(p.with_suffix('.json'));a=np.load(p);x=a['data'];cols=a['columns'].tolist()
        valid=np.isfinite(x[:,cols.index('Dmin_mean_A2')])&np.isfinite(x[:,[cols.index(k) for k in aug]]).all(1)
        arrays[m['condition'],int(m['realization'])]=(x[valid],cols)
        ledger.append(dict(condition=m['condition'],realization=m['realization'],total=len(x),valid=int(valid.sum()),excluded=int((~valid).sum())))
    def fit(x,z,y):
        mu=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1
        def expand(v):
            v=(v-mu)/sd
            return np.column_stack([v,*[v[:,i]*v[:,j] for i in range(v.shape[1]) for j in range(i,v.shape[1])]])
        a,b=expand(x),expand(z);mean=a.mean(0);scale=a.std(0);scale[scale<1e-12]=1
        a=(a-mean)/scale;b=(b-mean)/scale
        beta=np.linalg.solve(a.T@a+np.eye(a.shape[1]),a.T@(y-y.mean()))
        return b@beta+y.mean()
    maxerr=0.;scores=[]
    for q in rows(e/'matrices/prediction_all.csv'):
        tr,cols=arrays[q['condition'],int(q['train'])];te,tc=arrays[q['condition'],int(q['test'])];assert cols==tc
        y=np.log1p(tr[:,cols.index('Dmin_mean_A2')]/.01);test=np.log1p(te[:,cols.index('Dmin_mean_A2')]/.01)
        errors=[]
        for names in (base,aug):
            ix=[cols.index(k) for k in names];pred=fit(tr[:,ix],te[:,ix],y);errors.append(float(np.mean((test-pred)**2)))
        for val,key in zip(errors,('mse_base','mse_augmented')):
            err=abs(val-float(q[key]));maxerr=max(maxerr,err);assert err<1e-9
        scores.append(dict(condition=q['condition'],train=q['train'],test=q['test'],n_train=len(tr),n_test=len(te),mse_base=errors[0],mse_augmented=errors[1]))
    assert len(scores)==36;write(out/'prediction_reconstruction.csv',scores);write(out/'prediction_validity.csv',ledger)
    return dict(transfers=36,maximum_MSE_reproduction_error=maxerr,excluded_observations=sum(q['excluded'] for q in ledger))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    results={}
    for name,fn in [('chemical',chemical),('profiles',profiles),('forces',forces),('prediction',prediction)]:
        results[name]=fn(args.evidence,args.output);print(name,results[name],flush=True)
    # NaNs in invalid-profile CSVs are intentional, not imputed widths.
    results['profiles']['excluded_states']=[{k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in q.items()} for q in results['profiles']['excluded_states']]
    results['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    results['scope']='Existing-evidence numerical consistency, not independent physical validation.'
    save(args.output/'audit.json',results)

if __name__=='__main__':main()
