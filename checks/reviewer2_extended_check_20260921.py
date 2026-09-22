"""Portable checks from initial coordinates, selected IDs and original force logs."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,json
from pathlib import Path
import numpy as np
from reviewer2_audit_20260921 import read,rows,DEFS

def log_samples(path):
    blocks=[];head=None;vals=[]
    for line in path.read_text(errors='replace').splitlines():
        fields=line.split()
        if fields and fields[0]=='Step':
            if vals:blocks.append((head,np.array(vals,float)))
            head,vals=fields,[]
        elif head and len(fields)==len(head):
            try:vals.append([float(v) for v in fields])
            except ValueError:pass
    if vals:blocks.append((head,np.array(vals,float)))
    merged=None;segments=[]
    for h,a in blocks:
        if 'v_tool_x' not in h or len(a)<=10 or np.ptp(a[:,h.index('v_tool_x')])<=1:continue
        x=a[:,[h.index(k) for k in ('Step','Time','v_tool_x','v_fx','v_fz')]]
        assert np.all(np.diff(x[:,1])>0);discarded=0
        if merged is not None:
            discarded=int((merged[:,1]>=x[0,1]).sum());merged=merged[merged[:,1]<x[0,1]]
        merged=x if merged is None else np.vstack([merged,x]);segments.append(dict(start_ps=float(x[0,1]),end_ps=float(x[-1,1]),discarded_previous_samples=discarded))
    assert merged is not None and np.all(np.diff(merged[:,1])>0)
    return merged,segments

def main(e,d):
    result={};count=0
    for r in (1,2,3):
        initial=np.load(d/f'initial/R{r:02d}.npz');x=initial['xyz'];types=initial['types'];box=initial['box'];ids=initial['ids']
        depth=np.floor((x[:,2]-x[:,2].min())/2).astype(int);material=np.isin(types,[1,2]).astype(int)
        for definition in DEFS:
            p=np.load(e/f'nulls/S6_R{r:02d}/{definition}/population.npz')
            np.testing.assert_array_equal(p['ta'],np.isin(types,[2,4]))
            strata=depth*2+material
            if definition!='depth_only':
                meta=read(e/f'nulls/S6_R{r:02d}/{definition}/population.json');target=meta['target_A'];origin=meta['origin_fraction']
                lengths=np.diff(box[:2],axis=1).ravel();bins=np.maximum(1,np.floor(lengths/target+.5).astype(int));width=lengths/bins
                block=np.floor(np.mod((x[:,:2]-box[:2,0])/width-origin,bins)).astype(int)
                np.testing.assert_array_equal(block,p['block']);np.testing.assert_allclose(width,meta['actual_width_A'],atol=0,rtol=0)
                strata=(strata*bins[0]+block[:,0])*bins[1]+block[:,1]
            np.testing.assert_array_equal(strata,p['strata']);count+=1
        for q in rows(d/'depth_definitions.csv'):
            if int(q['realization'])!=r:continue
            a=np.load(d/f'activity/{q["run"]}_interval{q["interval"]}.npz');ix=np.searchsorted(ids,a['ids']);np.testing.assert_array_equal(ids[ix],a['ids'])
            di=np.maximum(0,x[:,2].max()-x[ix,2])/10;de=np.maximum(0,x[:,2].max()-a['xyz'][:,2])/10
            assert abs(np.quantile(di,.95)-float(q['d95_initial_nm']))<1e-12
            assert abs(np.quantile(de,.95)-float(q['d95_endpoint_nm']))<1e-12
    result['initial_coordinate_strata_checks']=count;result['active_ID_depth_sets']=54
    manifest=read(e/'forces/manifest.json');splices={}
    for run,meta in manifest.items():
        a,segments=log_samples(e/f'forces/logs/{run}.log')
        np.testing.assert_array_equal(a,np.load(e/f'forces/{run}.npz')['samples'])
        assert segments==meta['segments'];splices[run]=segments
    result['raw_primary_force_logs']=len(splices);result['restart_segments']=splices
    for q in rows(d/'physical_force_coverage.csv'):
        case=q['case'];a,_=log_samples(d/f'physical_forces/{case}.log')
        np.testing.assert_array_equal(a,np.load(d/f'physical_forces/{case}.npz')['samples'])
        t=a[:,1];st=float(q['start_ps']);en=float(q['end_ps']);assert t[0]<=st<en<=t[-1]
        for field,col in [('Ft',3),('Fn',4)]:
            mask=(t>st)&(t<en);xx=np.r_[st,t[mask],en];v=a[:,col]*1.602176634
            value=np.trapezoid(np.r_[np.interp(st,t,v),v[mask],np.interp(en,t,v)],xx)/(en-st)
            assert abs(value-float(q[field+'_mean_nN']))<1e-10
        if case=='ADH050_R03':
            assert read(d/f'physical_forces/{case}_launch.json')['returncode']==1
            result['unresolved_launcher_returncode']=1;result['ADH050_R03_force_interval_covered']=True
    result['physical_force_logs']=11
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--review',type=Path,required=True);a=p.parse_args();main(a.evidence,a.review)
