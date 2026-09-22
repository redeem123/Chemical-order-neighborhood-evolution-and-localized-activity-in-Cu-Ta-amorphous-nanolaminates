"""Validated, finite-temperature, mean-normalized non-affine analysis.

No MD, minimization, energy evaluation or modification of source files occurs.
"""
from pathlib import Path
import csv, hashlib, io, json, re, subprocess, time, resource, sys
import numpy as np
from scipy.spatial import cKDTree
from audit_sources import ROOT, P4, sha, write_csv

CONFIG = P4/'config/analysis.json'

def parse_dump_bytes(raw):
    lines=raw.split(b'\n',9)
    if len(lines)!=10 or lines[0]!=b'ITEM: TIMESTEP' or lines[2]!=b'ITEM: NUMBER OF ATOMS':
        raise ValueError('Invalid single-frame dump header')
    step=int(lines[1]); n=int(lines[3]); boundary=lines[4].decode().split()[3:]
    if len(boundary)!=3 or not lines[4].startswith(b'ITEM: BOX BOUNDS '): raise ValueError('Triclinic or unknown box unsupported')
    box=np.array([[float(v) for v in line.split()] for line in lines[5:8]])
    if box.shape!=(3,2) or not np.isfinite(box).all() or np.any(box[:,1]<=box[:,0]): raise ValueError('Invalid box')
    columns=lines[8].decode().split()[2:]
    if not lines[8].startswith(b'ITEM: ATOMS '): raise ValueError('Missing atom header')
    # loadtxt rejects malformed text instead of accepting a numeric prefix.
    a=np.loadtxt(io.BytesIO(lines[9]),ndmin=2)
    if a.shape!=(n,len(columns)) or not np.isfinite(a).all(): raise ValueError('Atom count/schema/nonfinite mismatch')
    for col in ('id','type'):
        vals=a[:,columns.index(col)]
        if np.any(vals!=np.floor(vals)): raise ValueError('Noninteger ID/type')
    ids=a[:,columns.index('id')].astype(np.int64)
    if len(np.unique(ids))!=n or np.any(ids<=0): raise ValueError('Nonunique/nonpositive IDs')
    order=np.argsort(ids); a=a[order]; ids=ids[order]
    types=a[:,columns.index('type')].astype(np.int16)
    if not np.isin(types,[1,2,3,4,5]).all(): raise ValueError('Unknown species mapping')
    xyz=a[:,[columns.index(c) for c in ('x','y','z')]]
    return {'step':step,'ids':ids,'types':types,'xyz':xyz,'box':box,'boundary':boundary,'columns':columns,'n':n}

def dump(path):
    path=Path(path)
    if path.suffix=='.zst':
        proc=subprocess.run(['zstd','-dc',str(path)],capture_output=True)
        if proc.returncode: raise ValueError(proc.stderr.decode().strip())
        raw=proc.stdout
    elif path.suffix=='.gz':
        import gzip
        raw=gzip.decompress(path.read_bytes())
    else: raw=path.read_bytes()
    f=parse_dump_bytes(raw)
    m=re.search(r'scratch\.(\d+)\.dump',path.name)
    if m and int(m.group(1))!=f['step']: raise ValueError('Filename/header step mismatch')
    f['sha256']=sha(path); f['source_path']=str(path)
    return f

def reference(realization):
    path=ROOT/f'results/runs/relax_laminate_R{realization:02d}/metadata/relaxed_structure.data'
    raw=path.read_text(); n=int(re.search(r'(\d+) atoms',raw).group(1))
    box=np.array([list(map(float,re.search(r'([^\n]+) '+axis+'lo '+axis+'hi',raw).group(1).split())) for axis in 'xyz'])
    body=raw.split('Atoms # atomic',1)[1].split('Velocities',1)[0].strip()
    a=np.loadtxt(io.StringIO(body),ndmin=2)
    if a.shape[0]!=n or not np.isfinite(a).all(): raise ValueError('Reference count/nonfinite mismatch')
    if np.any(a[:,:2]!=np.floor(a[:,:2])): raise ValueError('Invalid reference ID/type')
    ids=a[:,0].astype(np.int64); order=np.argsort(ids); a=a[order]; ids=ids[order]
    if len(np.unique(ids))!=n or not np.isin(a[:,1],[1,2,3,4]).all(): raise ValueError('Reference identity/type mismatch')
    return {'ids':ids,'types':a[:,1].astype(np.int16),'xyz':a[:,2:5],'box':box,'n':n,'sha256':sha(path),'source_path':str(path)}

def metal(frame, ref):
    m=frame['types']!=5
    if not np.array_equal(frame['ids'][m],ref['ids']) or not np.array_equal(frame['types'][m],ref['types']):
        raise ValueError('Workpiece ID/type changed')
    if frame['boundary']!=['ss','pp','ss']: raise ValueError('Unexpected boundary')
    if not np.allclose(frame['box'][1],ref['box'][1],atol=1e-5,rtol=0): raise ValueError('Changed periodic box')
    return frame['xyz'][m]

def mic(v, ly):
    out=np.array(v,copy=True); out[...,1]-=ly*np.rint(out[...,1]/ly); return out

def neighbors(x, centers, cutoff, ly):
    """Complete support with periodic y images; unique parent IDs per center."""
    shifts=np.array([[0,-ly,0],[0,0,0],[0,ly,0]])
    support=np.concatenate([x+s for s in shifts]); owner=np.tile(np.arange(len(x)),3)
    tree=cKDTree(support)
    lists=tree.query_ball_point(x[centers],cutoff,workers=1)
    return [np.unique(owner[js])[np.unique(owner[js])!=i] for i,js in zip(centers,lists)]

def fit_vectors(v0,v1,minimum=4,max_condition=1e6):
    n=len(v0)
    if n<minimum: return np.nan,np.nan,np.nan,'too_few_neighbors'
    _,sv,_=np.linalg.svd(v0,full_matrices=False)
    if len(sv)<3 or sv[-1]<=0 or sv[0]/sv[-1]>max_condition: return np.nan,np.nan,np.nan,'rank_or_condition'
    # Row-vector convention. v1 = v0 @ J, including arbitrary affine motion.
    J=np.linalg.lstsq(v0,v1,rcond=None)[0]
    residual=v1-v0@J; total=float(np.sum(residual**2))
    return total,total/n,float(sv[0]/sv[-1]),'valid'

def activity(x0,x1,centers,cutoff,ly,cfg):
    ns=neighbors(x0,centers,cutoff,ly)
    out=[]
    for i,js in zip(centers,ns):
        v0=mic(x0[js]-x0[i],ly); v1=mic(x1[js]-x1[i],ly)
        out.append((*fit_vectors(v0,v1,cfg['min_neighbors'],cfg['max_condition_number'])[:3],len(js)))
    return np.asarray(out,dtype=float)

def shuffle_strata(labels,strata,rng):
    out=labels.copy()
    for key in np.unique(strata):
        ix=np.flatnonzero(strata==key); out[ix]=rng.permutation(labels[ix])
    return out

def chemical_features(ref,centers,cutoff,cfg):
    ly=np.diff(ref['box'][1])[0]; xyz=ref['xyz']; ta=np.isin(ref['types'],[2,4]).astype(float)
    shell=neighbors(xyz,centers,cutoff,ly)
    broad=neighbors(xyz,centers,cfg['composition_radius_A'],ly)
    features=np.array([[ta[i],len(js),ta[js].mean() if len(js) else np.nan,ta[bs].mean() if len(bs) else np.nan] for i,js,bs in zip(centers,shell,broad)])
    return features,shell

def frame_record(frame, ref):
    metal(frame,ref)
    tool=frame['xyz'][frame['types']==5]
    if not len(tool): raise ValueError('No tool atoms')
    return {'source_path':frame['source_path'],'source_sha256':frame['sha256'],'step':frame['step'],'atom_count':frame['n'],
            'schema':' '.join(frame['columns']),'boundary':' '.join(frame['boundary']),
            'validation':'fully-parsed/validated','workpiece_id_type_match':True,
            'tool_com_x_A':tool[:,0].mean(),'tool_com_y_A':tool[:,1].mean(),'tool_com_z_A':tool[:,2].mean()}

def expected_tool(row,t,cfg):
    start=float(row['translation_start_ps']); f=float(row['frequency_per_ps'])
    A=0 if row['condition']=='Baseline' else int(row['condition'][1:3])/10
    x=float(row['x_start'])-.3*max(0,t-start)
    theta=2*np.pi*f*t+float(row['phase_origin'])
    amp=1. if t>=start or f==0 else .5*(1-np.cos(np.pi*max(0,t-4)/(3/f)))
    z=float(row['z_reference'])+A*amp*np.sin(theta)
    return np.array([x,z])

def run_analysis(run, mode='pilot'):
    cfg=json.loads(CONFIG.read_text()); inventory=list(csv.DictReader((P4/'data/run_inventory.csv').open()))
    row=next(r for r in inventory if r['run_id']==run); realization=int(row['realization'])
    if row['reference_matches_launch']!='True': raise ValueError('Reference launch hash mismatch')
    manifests=[r for r in csv.DictReader((P4/'data/frame_manifest.csv').open()) if r['run_id']==run]
    if mode=='cutoff32': cfg['cutoff_A']=3.2
    if mode=='cutoff36': cfg['cutoff_A']=3.6
    if mode=='long': cfg['interval_ps']=cfg['sensitivity_interval_ps']
    target=P4/f'data/derived/{mode}_{run}.npz'; metadata=target.with_suffix('.json')
    codehash=sha(__file__); confighash=sha(CONFIG)
    if target.exists() and metadata.exists():
        m=json.loads(metadata.read_text())
        reference_unchanged=Path(m['reference_path']).exists() and sha(m['reference_path'])==m['reference_sha256']
        if m['code_sha256']==codehash and m['config_sha256']==confighash and reference_unchanged and all(Path(s['source_path']).exists() and sha(s['source_path'])==s['source_sha256'] for s in m['validated_frames']):
            print(f'{run}: validated cache reused',flush=True); return
    started=time.perf_counter(); ref=reference(realization); ly=np.diff(ref['box'][1])[0]; top=ref['xyz'][:,2].max()
    times=np.array([float(m['time_ps']) for m in manifests]); distances=np.array([float(m['distance_nm']) for m in manifests])
    stations=[10.] if mode=='pilot' else cfg['stations_nm']
    all_data=[]; records={}; failures=[]; motion=[]
    for station in stations:
        j0=int(np.argmin(abs(distances-station)))
        if abs(distances[j0]-station)>cfg['station_tolerance_nm']:
            failures.append({'station':station,'reason':'no matching start frame'}); continue
        j1=int(np.argmin(abs(times-(times[j0]+cfg['interval_ps']))))
        if abs(times[j1]-times[j0]-cfg['interval_ps'])/cfg['interval_ps']>cfg['interval_relative_tolerance']:
            failures.append({'station':station,'reason':'no matching end frame'}); continue
        try:
            f0=dump(manifests[j0]['source_path']); f1=dump(manifests[j1]['source_path'])
            for f in (f0,f1): records[f['source_path']]=frame_record(f,ref)
            x0=metal(f0,ref); x1=metal(f1,ref)
            dt=times[j1]-times[j0]
            observed=np.array([records[f1['source_path']]['tool_com_x_A']-records[f0['source_path']]['tool_com_x_A'],records[f1['source_path']]['tool_com_z_A']-records[f0['source_path']]['tool_com_z_A']])
            expected=expected_tool(row,times[j1],cfg)-expected_tool(row,times[j0],cfg)
            error=observed-expected
            motion.append({'station_nm':station,'interval_ps':dt,'delta_x_error_A':float(error[0]),'delta_z_error_A':float(error[1]),'start_distance_nm':distances[j0]})
            if np.max(abs(error))>0.005: raise ValueError(f'Tool motion mismatch {error}')
            planned_x=float(row['x_start'])-station*10; ycenter=np.mean(ref['box'][1])
            rx=ref['xyz'][:,0]-planned_x; ry=mic(ref['xyz']-[0,ycenter,0],ly)[:,1]; depth=top-ref['xyz'][:,2]
            mask=(abs(rx)<=cfg['roi_half_x_A'])&(abs(ry)<=cfg['roi_half_y_A'])&(depth>=0)&(depth<=cfg['roi_depth_A'])&(ref['xyz'][:,2]>8)&(ref['xyz'][:,0]>8)&(ref['ids']%cfg['id_modulus']==0)
            centers=np.flatnonzero(mask)
            chem,_=chemical_features(ref,centers,cfg['cutoff_A'],cfg)
            d=activity(x0,x1,centers,cfg['cutoff_A'],ly,cfg)
            # IDs and all predictors refer to the original relaxed state, before scratch.
            arr=np.column_stack([ref['ids'][centers],np.full(len(centers),station),chem,depth[centers],rx[centers],ry[centers],
                np.isin(ref['types'][centers],[1,2]),d,np.full(len(centers),dt)])
            all_data.append(arr)
            print(f'{run} {station:g} nm: {len(centers)} atoms; interval {dt:g} ps; motion residual {np.max(abs(error)):.3g} A',flush=True)
        except ValueError as e:
            failures.append({'station':station,'reason':str(e)}); print(f'{run} {station}: EXCLUDED {e}',flush=True)
    columns=['id','station_nm','is_Ta','initial_coordination','initial_shell_Ta_fraction','initial_8A_Ta_fraction','initial_depth_A','initial_dx_A','initial_dy_A','initial_A_layer','Dmin_sum_A2','Dmin_mean_A2','fit_condition','reference_neighbor_count','interval_ps']
    if not all_data: raise ValueError(f'{run}: no valid analysis, {failures}')
    data=np.concatenate(all_data); np.savez_compressed(target,data=data,columns=np.array(columns))
    meta={'run_id':run,'condition':row['condition'],'realization':realization,'mode':mode,'code_sha256':codehash,'config_sha256':confighash,
        'reference_path':ref['source_path'],'reference_sha256':ref['sha256'],'validated_frames':list(records.values()),'motion_checks':motion,'exclusions':failures,
        'rows':len(data),'unique_atoms':len(np.unique(data[:,0])),'valid_activity_rows':int(np.isfinite(data[:,11]).sum()),
        'wall_seconds':time.perf_counter()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'effective_config':cfg,'units':'A and ps; Dmin mean = sum / reference neighbor count','interpretation':'finite-temperature observed activity, not irreversible STZ identification'}
    metadata.write_text(json.dumps(meta,indent=2)); print(json.dumps({k:meta[k] for k in ('run_id','rows','valid_activity_rows','wall_seconds','peak_rss_bytes')}),flush=True)

if __name__=='__main__':
    run_analysis(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'pilot')
