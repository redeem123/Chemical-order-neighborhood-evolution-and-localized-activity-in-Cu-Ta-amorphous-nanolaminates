"""Read-only full selected-cycle comparison of clean ADH050 R03 and old result."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'paper4/scripts'))
from atomic_analysis import reference,dump,metal,neighbors,mic,activity
from audit_sources import moving_thermo_blocks,sha
from dynamic_order_control import assignments,statistics,REGIONS
from cycle_increment import retention,width
from review_followup import clipped_mean

P4=ROOT/'paper4'
NEW=ROOT/'merged_workspaces/CuTa-results/validation/paper4_clean_cluster_20260922_download/verified/cases/ADH050_R03_retry3/production'
OLD=P4/'data/derived/existing_controls_20260916'
OUT=ROOT/'paper4/data/derived/clean_rerun_20260923/adhesion_comparison.json'

def main():
    receipt=json.loads((NEW/'completion.json').read_text())
    assert receipt['status']=='MD_COMPLETE' and receipt['launch']['returncode']==0
    for name,digest in receipt['artifact_sha256'].items():assert sha(NEW/name)==digest,name
    assert len(list((NEW/'trajectories').glob('scratch.*.dump')))==197
    row=next(x for x in json.loads((OLD/'plan.json').read_text())['cases'] if x['case']=='ADH050_R03')
    cfg=json.loads((P4/'config/existing_controls_20260916.json').read_text())
    inputs=NEW.parents[2]/'inputs'
    assert sha(inputs/'ADH050_R03_retry3/production.in')==sha(ROOT/'simulation/in.scratch_oscillatory.lmp')==receipt['launch']['input_sha256']
    assert sha(inputs/'ADH050_R03_retry3/structure.data')==sha(ROOT/'results/runs/relax_laminate_R03/metadata/relaxed_structure.data')
    assert sha(inputs/'potential/CuTa_LJ15_2014.adp.txt')==sha(ROOT/'potentials/CuTa_LJ15_2014.adp.txt')
    log,blocks=moving_thermo_blocks(NEW/'logs/lammps.log');assert len(blocks)==1
    h,a=blocks[0];time=a[:,h.index('Time')];assert len(a)==2173 and np.all(np.diff(time)>0)
    tlo,thi=[row['t0_ps']+row['ramp_ps']+d/.03 for d in cfg['force_window_nm']]
    force={key+'_mean_nN':float(clipped_mean(time,a[:,h.index(col)]*1.602176634,tlo,thi)[0])
           for key,col in (('Ft','v_fx'),('Fn','v_fz'))}
    ref=reference(3);x0=ref['xyz'];top=x0[:,2].max();ly=float(np.diff(ref['box'][1])[0]);ta=np.isin(ref['types'],[2,4]);origin=np.isin(ref['types'],[1,2])
    dx=x0[:,:2]-[row['x0_A']-100,ref['box'][1].mean()]
    cohort=np.all(abs(dx)<=25,axis=1)&(top-x0[:,2]<=60)&(top-x0[:,2]>=0)
    centers=np.flatnonzero(cohort&(ref['ids']%3==0))
    old=np.load(OLD/'ADH050_R03/arrays.npz')
    np.testing.assert_array_equal(ref['ids'][centers],old['center_ids'])
    ip=ROOT/'results/runs/relax_laminate_R03/metadata/relaxed_structure.manifest.json'
    interfaces=np.array(json.loads(ip.read_text())['interface_coordinates_A']);dist=np.min(abs(x0[:,2,None]-interfaces),axis=1)
    masks=[];comps=[]
    for o in (True,False):
        for interior in (True,False):
            mask=cohort&(origin==o)&((dist>3.6) if interior else (dist<=3.6))
            masks.append(mask[centers]);comps.append(ta[mask].mean())
    masks=np.array(masks);comps=np.array(comps)
    np.testing.assert_array_equal(masks,old['region_masks']);np.testing.assert_allclose(comps,old['composition'],atol=0,rtol=0)
    strata=np.floor((x0[:,2]-x0[:,2].min())/2).astype(int)*2+origin.astype(int)
    labels=assignments(ta,strata,64,20260915)
    initial=neighbors(x0,centers,cfg['cutoff_A'],ly)
    z=max(z for z in interfaces if top-54<=z<=top-6);edges=np.linspace(-25,25,4);tiles=[]
    for ix in range(3):
        for iy in range(3):
            tiles.append(np.flatnonzero((abs(x0[:,2]-z)<=6)&(dx[:,0]>=edges[ix])&(dx[:,0]<edges[ix+1])&(dx[:,1]>=edges[iy])&(dx[:,1]<edges[iy+1])))
    widthvals=np.full((9,9),np.nan);alpha=[];kin=[];regional=[];framehashes={}
    for q in row['frames']:
        step=q['step'];path=NEW/f'trajectories/scratch.{step}.dump';f=dump(path);assert f['step']==step
        x=metal(f,ref);framehashes[str(step)]=f['sha256']
        # Dump parser sorts identities, as does the frozen velocity analysis.
        import io
        raw=path.read_bytes();data=np.loadtxt(io.BytesIO(raw.split(b'\n',9)[9]));data=data[np.argsort(data[:,f['columns'].index('id')])]
        v=data[f['types']!=5][:,[f['columns'].index(k) for k in ('vx','vy','vz')]]
        mass=np.where(ta,180.95,63.546)[cohort];vv=v[cohort];vcom=np.sum(mass[:,None]*vv,axis=0)/mass.sum()
        kin.append(float(np.sum(mass[:,None]*(vv-vcom)**2)*1.0364269656262175e-4/((3*len(mass)-3)*8.617333262145e-5)))
        ns=neighbors(x,centers,cfg['cutoff_A'],ly)
        if q['phase_index']==0:start_shell=ns;start=x.copy()
        ri=retention(initial,ns);rc=retention(start_shell,ns)
        src=np.repeat(np.arange(len(centers)),[len(js) for js in ns]);dst=np.concatenate(ns)
        al=statistics(labels,centers,src,dst,masks,comps);assert np.isfinite(al).all();alpha.append(al)
        regional.append(dict(phase_index=q['phase_index'],initial_retention_pct=float(ri.mean()*100),
                             cycle_retention_pct=float(rc.mean()*100),
                             observed_Cu=float(al[0,0]),null_Cu=float(al[1:,0].mean()),
                             excess_Cu=float(al[0,0]-al[1:,0].mean())))
        for j,ix in enumerate(tiles):
            xx=x0[ix]+mic(x[ix]-x0[ix],ly);w=width(x0[ix],xx,origin[ix],z,6,.75,[.1,.5,.9]);widthvals[q['phase_index'],j]=w if w is not None else np.nan
        print('ADH050 R03 clean phase',q['phase_index'],'parsed',flush=True)
    dm=activity(start,x,centers,cfg['cutoff_A'],ly,cfg['activity']);valid=np.isfinite(dm[:,1])
    base=np.load(OLD/'ADH100_R03/arrays.npz')
    strong=np.load(OLD/'ADH150_R03/arrays.npz')
    for arm in (base,strong):
        np.testing.assert_array_equal(ref['ids'][centers],arm['center_ids'])
        np.testing.assert_array_equal(masks,arm['region_masks'])
        np.testing.assert_allclose(comps,arm['composition'],atol=0,rtol=0)
    common=valid&base['valid_dmin2']&strong['valid_dmin2']
    oldabs=next(x for x in csv.DictReader((OLD/'absolute.csv').open()) if x['case']=='ADH050_R03')
    tilevalid=np.isfinite(widthvals).all(0)
    for case in ('ADH100_R03','ADH150_R03'):
        w=np.full((9,9),np.nan)
        for line in csv.DictReader((OLD/case/'widths.csv').open()):
            if line['valid']=='True':w[int(line['phase_index']),int(line['tile'])]=float(line['width_A'])
        tilevalid &= np.isfinite(w).all(0)
    alpha_array=np.array(alpha)
    u=alpha_array[:,0]-alpha_array[:,1:].mean(1)
    paired=alpha_array-base['alpha']
    chemical_regions={region:dict(mean_u=float(u[:,j].mean()),increment_u=float(u[-1,j]-u[0,j]))
                      for j,region in enumerate(REGIONS)}
    null_sd_Cu=dict(mean=float(paired[:,1:,0].mean(0).std(ddof=1)),
                    increment=float((paired[-1,1:,0]-paired[0,1:,0]).std(ddof=1)))
    result=dict(analysis_sha256=sha(__file__),scope='Same input and preparation; clean A100 Kokkos rerun versus archived Mac MPI run, not independent realization',
        old_manifest_sha256=sha(OLD/'ADH050_R03/manifest.json'),new_completion_sha256=sha(NEW/'completion.json'),
        new_source_hashes=framehashes,force=force,kinetic_temperature_K=float(np.mean(kin)),
        dmin2_A2=float(dm[common,1].mean()),common_centers=int(common.sum()),
        initial_retention_pct=float(ri.mean()*100),cycle_retention_pct=float(rc.mean()*100),
        chemical_mean_Cu=float(u[:,0].mean()),chemical_increment_Cu=float(u[-1,0]-u[0,0]),
        chemical_observed_mean_Cu=float(np.array(alpha)[:,0,0].mean()),
        chemical_null_mean_Cu=float(np.array(alpha)[:,1:,0].mean()),
        width_increment_A=float((widthvals[-1,tilevalid]-widthvals[0,tilevalid]).mean()),
        width_start_A=float(widthvals[0,tilevalid].mean()),width_end_A=float(widthvals[-1,tilevalid].mean()),
        common_tiles=int(tilevalid.sum()),regional=regional,chemical_regions=chemical_regions,
        conditional_label_sd_Cu=null_sd_Cu,
        old_absolute={k:float(v) for k,v in oldabs.items() if k in ('Ft_mean_nN','Fn_mean_nN','kinetic_temperature_K','dmin2_A2','initial_retention_pct','cycle_retention_pct','chemical_mean_Cu','chemical_increment_Cu','width_increment_A')})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+'\n');print('WROTE',OUT,flush=True)

if __name__=='__main__':main()
