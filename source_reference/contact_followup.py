"""Matched force quadrature with explicit restart precedence."""
import json
import numpy as np
from review_followup import OUT, CFG, read, clipped_mean
from audit_sources import P4, sha, moving_thermo_blocks, write_csv


def load_force(rr):
    path=rr['log_path']; assert sha(path)==rr['log_sha256']
    text, blocks=moving_thermo_blocks(path)
    assert 'fix tool_react tool setforce 0.0 0.0 0.0' in text
    assert 'variable fx equal f_tool_react[1]' in text and 'variable fz equal f_tool_react[3]' in text
    merged=None; segments=[]
    for header,data in blocks:
        names=['Step','Time','v_tool_x','v_fx','v_fz']
        current=data[:,[header.index(n) for n in names]]
        assert np.all(np.diff(current[:,1])>0)
        assert np.allclose(current[:,1]*1000,current[:,0],atol=1e-5)
        assert np.max(abs((current[:,2]-current[0,2])+.3*(current[:,1]-current[0,1])))<.002
        discarded=0
        if merged is not None:
            discarded=int((merged[:,1]>=current[0,1]).sum())
            merged=merged[merged[:,1]<current[0,1]]
            assert abs((current[0,2]-merged[-1,2])+.3*(current[0,1]-merged[-1,1]))<.002
        merged=current if merged is None else np.vstack([merged,current])
        segments.append(dict(start_ps=float(current[0,1]),end_ps=float(current[-1,1]),discarded_previous_samples=discarded))
    assert np.all(np.diff(merged[:,1])>0)
    return merged,segments


def main():
    OUT.mkdir(parents=True,exist_ok=True); result=[]; series=[]; cache={}; records={}; sources={str(CFG):sha(CFG)}
    for task in read(P4/'data/derived/cycle_increment/plan.json')['tasks']:
        case=task['case']; station=task['station_nm']
        for arm,rr in [('vibration',case['row']),('baseline',case['baseline'])]:
            run=rr['run_id']
            if run not in cache:
                data,segments=load_force(rr);cache[run]=data;records[run]=segments
                sources[rr['log_path']]=sha(rr['log_path']);sources[rr['metadata_path']]=sha(rr['metadata_path'])
            data=cache[run];t=data[:,1]
            items=[i for i in case['items'] if i['stage']=='cycle' and i['arm']==arm]
            start,end=[float(i['frame']['time_ps']) for i in (items[0],items[-1])]
            r=dict(case=case['case'],condition=case['condition'],realization=case['realization'],station_nm=station,
                arm=arm,run=run,start_ps=start,end_ps=end,duration_ps=end-start,
                sample_interval_ps=float(np.median(np.diff(t))),interior_samples=int(((t>start)&(t<end)).sum()))
            assert r['interior_samples']>=20
            for label,col in [('Ft',3),('Fn',4)]:
                values=data[:,col]*1.602176634
                mean,tt,yy=clipped_mean(t,values,start,end)
                keep=np.unique(np.r_[np.arange(0,len(t),2),len(t)-1])
                coarse,_,_=clipped_mean(t[keep],values[keep],start,end)
                r[label+'_mean_nN']=mean;r[label+'_downsample_difference_nN']=coarse-mean
                for tm,val in zip(tt,yy):
                    series.append(dict(case=case['case'],station_nm=station,arm=arm,component=label,time_ps=tm,
                        elapsed_fraction=(tm-start)/(end-start),force_nN=val))
            result.append(r)
    write_csv(OUT/'contact_summary.csv',result);write_csv(OUT/'contact_traces.csv',series)
    sources[str(P4/'scripts/review_followup.py')]=sha(P4/'scripts/review_followup.py')
    sources[str(P4/'data/derived/cycle_increment/plan.json')]=sha(P4/'data/derived/cycle_increment/plan.json')
    report=dict(sources=sources,script_sha256=sha(__file__),runs=len(cache),paired_cycles=len(result)//2,
        restart_segments=records,restart_policy='Later executed segment replaces previous records at and after its first time. No averaging of duplicate times.',
        definition='Signed pre-setforce tool force, +x resists negative-x translation; +z is upward reaction.',
        documentation='https://docs.lammps.org/fix_setforce.html',conversion_nN_per_eV_A=1.602176634,
        maximum_downsample_difference_nN={k:max(abs(r[k+'_downsample_difference_nN']) for r in result) for k in ('Ft','Fn')},
        output_hashes={n:sha(OUT/n) for n in ('contact_summary.csv','contact_traces.csv')})
    (OUT/'contact_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
