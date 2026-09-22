"""Audit and analyze archived parameter controls; never invoke an MD engine."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import io
import json
import re
import subprocess
from pathlib import Path
import numpy as np
from atomic_analysis import reference, parse_dump_bytes, metal, neighbors, mic, activity
from audit_sources import ROOT, P4, sha, write_csv, moving_thermo_blocks, numeric_log_variable
from dynamic_order_control import assignments, statistics, REGIONS
from cycle_increment import retention, width
from review_followup import clipped_mean

CFG=P4/'config/existing_controls_20260916.json'
OUT=P4/'data/derived/existing_controls_20260916'
RAW=ROOT/'merged_workspaces/CuTa-results/runs'
RECOVERY=ROOT/'merged_workspaces/CuTa-results/.priority-recovery-pvZ3Ul'


def audit(case, r, value, family):
    mp=RAW/case/'metadata/launch.json';lp=RAW/case/'logs/lammps.log'
    m=json.loads(mp.read_text());text,blocks=moving_thermo_blocks(lp)
    assert len(blocks)==1 and not re.search(r'^ERROR:',text,re.M)
    h,a=blocks[0];assert np.isfinite(a).all() and np.all(np.diff(a[:,h.index('Time')])>0)
    assert np.allclose(a[:,h.index('Time')]*1000,a[:,h.index('Step')],rtol=0,atol=1e-5)
    marker=re.findall(r'^OSCILLATORY_SCRATCH_COMPLETE case='+re.escape(case)+r' .*$',text,re.M)
    assert len(marker)==1 and (RAW/case/'metadata/final.data').exists()
    physical=re.findall(r'^oscillation: alpha=([\d.eE+-]+) beta=([\d.eE+-]+) A=([\d.eE+-]+) A  f=([\d.eE+-]+) 1/ps  T=([\d.eE+-]+) ps  ramp=([\d.eE+-]+) ps',text,re.M)
    assert len(physical)==1,case
    alpha,beta,A,f,T,ramp=map(float,physical[0]);assert beta==2 and abs(f*T-1)<1e-10
    assert alpha==(.25 if family=='adhesion' else .1)
    t0=numeric_log_variable(text,'tstart');x0=numeric_log_variable(text,'tool_x0');z0=numeric_log_variable(text,'tool_z0')
    time=a[:,h.index('Time')];x=a[:,h.index('v_tool_x')]
    assert np.max(abs(x-(x0-.3*(time-t0-ramp))))<.0001
    expected_cu=[.05477*(value if family=='adhesion' else 1),2.168,5.4]
    expected_ta=[.13*(value if family=='adhesion' else 1),2.86575,7.164]
    for typ,expected in [(1,expected_cu),(3,expected_cu),(2,expected_ta),(4,expected_ta)]:
        matches=re.findall(r'^pair_coeff '+str(typ)+r' 5 lj/cut ([\d.eE+-]+) ([\d.eE+-]+) ([\d.eE+-]+)\s*$',text,re.M)
        assert len(matches)==1 and np.allclose(list(map(float,matches[0])),expected,rtol=0,atol=1e-12)
    sink=4 if family=='adhesion' else value
    assert re.search(r'^region r_th_bottom\s+block 4(?:\.0)? INF INF INF 4(?:\.0)? '+str(int(4+sink))+r'(?:\.0)? units box',text,re.M)
    assert 'fix sink_bottom g_th_bottom temp/rescale 10 300.0 300.0 0.1 1.0' in text
    ref=ROOT/f'results/runs/relax_laminate_R{r:02d}/metadata/relaxed_structure.data'
    assert sha(ref)==m['structure_sha256']
    pot=ROOT/'potentials/CuTa_LJ15_2014.adp.txt';assert sha(pot)==m['potential_sha256']
    template=ROOT/'simulation/in.scratch_oscillatory.lmp';assert sha(template)==m['input_sha256']
    folder=(RECOVERY/case/'trajectories') if family=='thermostat' else RAW/case/'trajectories'
    files=sorted(folder.glob('scratch.*.dump.zst'),key=lambda p:int(p.name.split('.')[1]))
    assert files,folder
    cfg=json.loads(CFG.read_text());target=t0+ramp+cfg['station_nm']/.03
    cycle=round(f*(target-t0)-.5);zero=t0+cycle/f
    steps=np.array([int(p.name.split('.')[1]) for p in files]);chosen=[]
    for k,theta in enumerate(np.linspace(0,2*np.pi,cfg['phase_count'])):
        wanted=zero+theta/(2*np.pi*f);i=int(np.argmin(abs(steps*.001-wanted)));step=int(steps[i]);phase=2*np.pi*f*(step*.001-zero)
        assert abs(phase-theta)<cfg['maximum_phase_error_rad']
        chosen.append(dict(phase_index=k,step=step,phase_rad=phase,path=str(files[i]),distance_nm=.03*(step*.001-t0-ramp)))
    assert len({x['step'] for x in chosen})==cfg['phase_count']
    tlo,thi=[t0+ramp+d/.03 for d in cfg['force_window_nm']]
    fmetrics={}
    for key,col in [('Ft',h.index('v_fx')),('Fn',h.index('v_fz'))]:
        fmetrics[key+'_mean_nN']=clipped_mean(time,a[:,col]*1.602176634,tlo,thi)[0]
    warn='' if m.get('returncode')==0 else 'Launcher exit 1; operational completion and exact MD marker present; analysis windows independently validated.'
    return dict(case=case,realization=r,family=family,value=value,A_A=A,f_per_ps=f,ramp_ps=ramp,t0_ps=t0,x0_A=x0,z0_A=z0,
                source_hashes={str(p):sha(p) for p in [mp,lp,ref,pot,template]},frames=chosen,force=fmetrics,warning=warn,
                launcher_returncode=m.get('returncode'),operational_completion=m.get('operational_completion'),completion_marker=marker[0])


def plan():
    cfg=json.loads(CFG.read_text());OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for family,key in [('adhesion','adhesion_cases'),('thermostat','thermostat_cases')]:
        for case,r,value in cfg[key]:rows.append(audit(case,r,value,family))
    for family in ('adhesion','thermostat'):
        for r in (1,2,3):
            group=[x for x in rows if x['family']==family and x['realization']==r]
            if not group:continue
            assert all([q['step'] for q in x['frames']]==[q['step'] for q in group[0]['frames']] for x in group)
    p=OUT/'plan.json'
    record=dict(config_sha256=sha(CFG),script_sha256=sha(__file__),cases=rows,no_MD=True)
    if p.exists():assert json.loads(p.read_text())==record,'Frozen plan mismatch'
    else:p.write_text(json.dumps(record,indent=2)+'\n')
    print('Audited',len(rows),'cases;',sum(len(x['frames']) for x in rows),'selected frames',flush=True)
    return rows


def analyze(row):
    cfg=json.loads(CFG.read_text());r=row['realization'];d=OUT/row['case'];d.mkdir(exist_ok=True)
    if (d/'manifest.json').exists():
        m=json.loads((d/'manifest.json').read_text());assert m['config_sha256']==sha(CFG) and m['script_sha256']==sha(__file__)
        assert all(sha(Path(p))==h for p,h in m['source_hashes'].items())
        assert all(sha(d/p)==h for p,h in m['outputs'].items());print(row['case'],'verified cached',flush=True);return
    ref=reference(r);x0=ref['xyz'];top=x0[:,2].max();ly=float(np.diff(ref['box'][1])[0])
    ta=np.isin(ref['types'],[2,4]);origin=np.isin(ref['types'],[1,2]);dx=x0[:,:2]-[row['x0_A']-100,ref['box'][1].mean()]
    cohort=np.all(abs(dx)<=25,axis=1)&(top-x0[:,2]<=60)&(top-x0[:,2]>=0)
    centers=np.flatnonzero(cohort&(ref['ids']%3==0))
    ip=ROOT/f'results/runs/relax_laminate_R{r:02d}/metadata/relaxed_structure.manifest.json'
    interfaces=np.array(json.loads(ip.read_text())['interface_coordinates_A']);dist=np.min(abs(x0[:,2,None]-interfaces),axis=1)
    masks=[];comps=[]
    for o in (True,False):
        for interior in (True,False):
            mask=cohort&(origin==o)&((dist>3.6) if interior else (dist<=3.6))
            masks.append(mask[centers]);comps.append(ta[mask].mean())
    masks=np.array(masks);comps=np.array(comps)
    strata=np.floor((x0[:,2]-x0[:,2].min())/2).astype(int)*2+origin.astype(int)
    labels=assignments(ta,strata,64,20260912+r)
    initial=neighbors(x0,centers,cfg['cutoff_A'],ly)
    z=max(z for z in interfaces if top-54<=z<=top-6);edges=np.linspace(-25,25,4);tiles=[]
    for ix in range(3):
        for iy in range(3):
            selected=np.flatnonzero((abs(x0[:,2]-z)<=6)&(dx[:,0]>=edges[ix])&(dx[:,0]<edges[ix+1])&(dx[:,1]>=edges[iy])&(dx[:,1]<edges[iy+1]))
            tiles.append(selected)
    regional=[];profiles=[];chem=[];kin=[];source=dict(row['source_hashes']);source[str(ip)]=sha(ip);motion=[]
    for q in row['frames']:
        path=Path(q['path']);raw=subprocess.run(['zstd','-dc',str(path)],capture_output=True,check=True).stdout
        f=parse_dump_bytes(raw);assert f['step']==q['step'];x=metal(f,ref)
        a=np.loadtxt(io.BytesIO(raw.split(b'\n',9)[9]));a=a[np.argsort(a[:,f['columns'].index('id')])]
        v=a[f['types']!=5][:,[f['columns'].index(k) for k in ('vx','vy','vz')]]
        mass=np.where(ta,180.95,63.546)[cohort];vv=v[cohort];vcom=np.sum(mass[:,None]*vv,axis=0)/mass.sum()
        temp=float(np.sum(mass[:,None]*(vv-vcom)**2)*1.0364269656262175e-4/((3*len(mass)-3)*8.617333262145e-5))
        kin.append(temp)
        theta=2*np.pi*row['f_per_ps']*(f['step']*.001-row['t0_ps'])
        expected=np.array([row['x0_A']-.3*(f['step']*.001-row['t0_ps']-row['ramp_ps']),row['z0_A']+row['A_A']*np.sin(theta)])
        actual=f['xyz'][f['types']==5].mean(0)[[0,2]];motion.append((actual,expected))
        assert np.max(abs((actual-motion[0][0])-(expected-motion[0][1])))<.005
        ns=neighbors(x,centers,cfg['cutoff_A'],ly)
        if q['phase_index']==0:start_shell=ns;start=x.copy()
        ri=retention(initial,ns);rc=retention(start_shell,ns)
        src=np.repeat(np.arange(len(centers)),[len(js) for js in ns]);dst=np.concatenate(ns)
        al=statistics(labels,centers,src,dst,masks,comps);assert np.isfinite(al).all();chem.append(al)
        for j,region in enumerate(REGIONS):
            use=masks[j];assert np.isfinite(ri[use]).all() and np.isfinite(rc[use]).all()
            regional.append(dict(case=row['case'],phase_index=q['phase_index'],region=region,centers=int(use.sum()),
                initial_retention=float(ri[use].mean()),cycle_retention=float(rc[use].mean()),alpha_observed=float(al[0,j]),
                alpha_null_mean=float(al[1:,j].mean()),alpha_excess=float(al[0,j]-al[1:,j].mean()),
                alpha_null_sd=float(al[1:,j].std(ddof=1))))
        for j,ix in enumerate(tiles):
            xx=x0[ix]+mic(x[ix]-x0[ix],ly)
            w=width(x0[ix],xx,origin[ix],z,6,.75,[.1,.5,.9])
            profiles.append(dict(case=row['case'],phase_index=q['phase_index'],tile=j,width_A=w,valid=w is not None))
        source[str(path)]=sha(path)
        print(row['case'],q['phase_index'],'fully parsed',flush=True)
    dm=activity(start,x,centers,cfg['cutoff_A'],ly,cfg['activity']);valid=np.isfinite(dm[:,1]);assert valid.any()
    np.savez_compressed(d/'arrays.npz',alpha=np.array(chem),kinetic_temperature_K=kin,center_ids=ref['ids'][centers],
                        region_masks=masks,composition=comps,dmin2=dm[:,1],valid_dmin2=valid,
                        endpoint_initial_retention=ri,endpoint_cycle_retention=rc)
    write_csv(d/'regional.csv',regional);write_csv(d/'widths.csv',profiles)
    m=dict(case=row,config_sha256=sha(CFG),script_sha256=sha(__file__),source_hashes=source,
           source_helpers={n:sha(P4/'scripts'/n) for n in ['atomic_analysis.py','dynamic_order_control.py','cycle_increment.py','review_followup.py','audit_sources.py']},
           cohort_atoms=int(cohort.sum()),center_atoms=len(centers),valid_dmin2=int(valid.sum()),invalid_dmin2=int((~valid).sum()),
           outputs={n:sha(d/n) for n in ['arrays.npz','regional.csv','widths.csv']})
    (d/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan-only',action='store_true');p.add_argument('--case');args=p.parse_args()
    rows=plan()
    if not args.plan_only:
        for row in rows:
            if not args.case or row['case']==args.case:analyze(row)
