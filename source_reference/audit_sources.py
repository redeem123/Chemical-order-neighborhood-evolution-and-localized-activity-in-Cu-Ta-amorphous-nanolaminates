"""Read-only source inventory. Generated evidence is confined to paper4/."""
from pathlib import Path
import csv, hashlib, json, re, subprocess, datetime
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P4 = ROOT / 'paper4'
RAW = Path('/Volumes/CuTaData/CuTa-results/runs')

# R03 was not part of the older Paper 3 two-realization provenance file.  The
# matching records below are therefore audited directly from their own raw log
# and launch metadata.  ADH050/ADH150 are retained in the source inventory but
# excluded from this study because they change the tool--metal interaction.
R03_SOURCE_CASES = (
    {'case': 'scratch_base_R03', 'analysis_condition': 'Baseline',
     'source_condition': 'Baseline', 'paper4_scope': 'candidate matching condition',
     'reason': 'Same baseline condition as R01/R02.'},
    {'case': 'S1_R03', 'analysis_condition': 'A10B05',
     'source_condition': 'A10B05', 'paper4_scope': 'candidate matching condition',
     'reason': 'Same amplitude and frequency condition as R01/R02.'},
    {'case': 'ADH100_R03', 'analysis_condition': 'A25B20',
     'source_condition': 'A25B20 with unit tool--metal interaction',
     'paper4_scope': 'candidate matching condition',
     'reason': 'Unit interaction and kinematics match the S5 A25B20 condition.'},
    {'case': 'ADH050_R03', 'analysis_condition': '',
     'source_condition': 'A25B20 with 0.5x tool--metal interaction',
     'paper4_scope': 'outside current study scope',
     'reason': 'Different tool--metal interaction scale.'},
    {'case': 'ADH150_R03', 'analysis_condition': '',
     'source_condition': 'A25B20 with 1.5x tool--metal interaction',
     'paper4_scope': 'outside current study scope',
     'reason': 'Different tool--metal interaction scale.'},
)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def write_csv(path, rows):
    if not rows: raise ValueError(f'No rows for {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def moving_thermo_blocks(log_path):
    """Read only translated LAMMPS thermo blocks without relying on Paper 3."""
    blocks, header, rows = [], None, []
    text = Path(log_path).read_text(errors='replace')
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == 'Step':
            if rows:
                blocks.append((header, np.asarray(rows, dtype=float)))
            header, rows = fields, []
        elif header and len(fields) == len(header):
            try:
                rows.append([float(value) for value in fields])
            except ValueError:
                pass
    if rows:
        blocks.append((header, np.asarray(rows, dtype=float)))
    moving = [
        (header, values) for header, values in blocks
        if 'v_tool_x' in header and len(values) > 10
        and np.ptp(values[:, header.index('v_tool_x')]) > 1.0
    ]
    if not moving:
        raise ValueError(f'No translated thermo block in {log_path}')
    return text, moving


def numeric_log_variable(text, name):
    matches = re.findall(r'^variable ' + re.escape(name) + r' equal ([\d.eE+-]+)\s*$', text, re.M)
    if not matches:
        raise ValueError(f'No numeric {name} variable found in executed log')
    return float(matches[-1])


def audit_r03_case(spec):
    """Derive a target-local R03 provenance record from the executed evidence."""
    case = spec['case']
    base = RAW / case
    log_path = base / 'logs/lammps.log'
    launch_path = base / 'metadata/launch.json'
    launch = json.loads(launch_path.read_text())
    text, blocks = moving_thermo_blocks(log_path)
    if len(blocks) != 1:
        raise ValueError(f'{case}: expected one translated segment, found {len(blocks)}')
    header, data = blocks[0]
    for name in ('Step', 'Time', 'v_tool_x'):
        if name not in header:
            raise ValueError(f'{case}: missing {name} in translated thermo block')
    completion = re.search(r'^\w*SCRATCH_COMPLETE case=' + re.escape(case) + r'\b.*$', text, re.M)
    if completion is None:
        raise ValueError(f'{case}: exact completion marker is absent')
    variables = launch['forwarded_variables']
    baseline = spec['analysis_condition'] == 'Baseline'
    amplitude = 0.0 if baseline else float(variables['A_case'])
    frequency = 0.0 if baseline else float(variables['f_case'])
    if baseline:
        phase_origin = 0.0
    else:
        for name in ('v_phase', 'v_tool_z'):
            if name not in header:
                raise ValueError(f'{case}: missing {name} for oscillatory provenance')
        time_ps = data[:, header.index('Time')]
        theta = np.unwrap(data[:, header.index('v_phase')] * (2.0 * np.pi))
        phase_origin = float(np.median(theta - 2.0 * np.pi * frequency * time_ps))
        phase_error = float(np.max(np.abs(theta - (2.0 * np.pi * frequency * time_ps + phase_origin))))
        z_reference = numeric_log_variable(text, 'tool_z0')
        z_error = float(np.max(np.abs(data[:, header.index('v_tool_z')] - (z_reference + amplitude * np.sin(theta)))))
        if phase_error > 1e-5 or z_error > 1e-3:
            raise ValueError(f'{case}: phase or tool-position provenance check failed')
    x_start = numeric_log_variable(text, 'tool_x0')
    z_reference = numeric_log_variable(text, 'tool_z0')
    dump_values = sorted(set(int(value) for value in re.findall(r'^dump traj all custom (\d+) ', text, re.M)))
    if len(dump_values) != 1:
        raise ValueError(f'{case}: ambiguous dump stride')
    time_ps = data[:, header.index('Time')]
    force_stride = int(round(np.median(np.diff(time_ps)) / 0.001))
    if not np.isclose(float(variables['dt_ps']), 0.001):
        raise ValueError(f'{case}: unexpected timestep')
    if not baseline:
        alpha, beta = float(variables['alpha']), float(variables['beta'])
        if not np.isclose(amplitude, alpha * float(variables['h0'])):
            raise ValueError(f'{case}: inconsistent amplitude metadata')
        if not np.isclose(frequency, beta * float(variables['vs']) / (2.0 * np.pi * amplitude)):
            raise ValueError(f'{case}: inconsistent frequency metadata')
    reference = Path(launch['structure'])
    if sha(reference) != launch['structure_sha256']:
        raise ValueError(f'{case}: relaxed reference hash mismatch')
    record = {
        'case': case,
        'condition': spec['analysis_condition'],
        'realization': 3,
        'source': 'direct R03 raw log and launch audit',
        'translation_start_step': int(data[0, header.index('Step')]),
        'time_origin': 'absolute LAMMPS Time in ps; dt=0.001 ps',
        'translation_start_ps': float(time_ps[0]),
        'x_start': x_start,
        'z_reference': z_reference,
        'phase_origin': phase_origin,
        'phase_formula': 'theta=2*pi*frequency_per_ps*time_ps+phase_origin; z=z_reference+A*sin(theta)',
        'frequency_per_ps': frequency,
        'force_stride': force_stride,
        'dump_stride': dump_values[0],
        'force_sample_interval_ps': float(np.median(np.diff(time_ps))),
        'trajectory_dump_interval_ps': dump_values[0] * 0.001,
        'restart_segments': len(blocks),
        'reset_timestep_commands': '; '.join(re.findall(r'^reset_timestep .*$', text, re.M)) or 'none',
        'logged_x_offset_A': float(data[0, header.index('v_tool_x')] - x_start),
        'window_start_ps': float(time_ps[0] + 4.0 / 0.03),
        'window_end_ps': float(time_ps[0] + 16.0 / 0.03),
        'log_sha256': sha(log_path),
        'launch_sha256': sha(launch_path),
    }
    source_record = {
        'case': case,
        'source_condition': spec['source_condition'],
        'target_condition': spec['analysis_condition'] or 'not mapped',
        'paper4_scope': spec['paper4_scope'],
        'reason': spec['reason'],
        'completion_marker': completion.group(),
        'launch_completed': launch.get('completed'),
        'launch_returncode': launch.get('returncode'),
        'reference_path': str(reference),
        'reference_sha256': launch['structure_sha256'],
        'log_sha256': record['log_sha256'],
        'launch_sha256': record['launch_sha256'],
        'translation_start_ps': record['translation_start_ps'],
        'frequency_per_ps': record['frequency_per_ps'],
        'dump_stride': record['dump_stride'],
        'thermo_segments': len(blocks),
    }
    return record, source_record, [{'start_step': int(data[0, header.index('Step')]),
                                    'end_step': int(data[-1, header.index('Step')]),
                                    'start_ps': float(time_ps[0]), 'end_ps': float(time_ps[-1])}]

def main():
    state = P4/'reports/initial_workspace_state.json'
    if not state.exists():
        tracked = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
        # Capture all existing Paper 1/3 files, including untracked submission files.
        protected = sorted({str(p.relative_to(ROOT)) for d in ('paper','paper3') for p in (ROOT/d).rglob('*') if p.is_file()})
        state.write_text(json.dumps({'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT).decode(),
            'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
            'protected_sha256':{p:sha(ROOT/p) for p in protected},
            'tracked_paths':tracked},indent=2))
    prompt=(P4/'Paper4_Master_Prompt_Deep_Research.md').read_text()
    bib=(P4/'Paper4_Verified_References.bib').read_text()
    block=prompt.split('```bibtex\n',1)[1].split('```',1)[0].strip()
    keys=re.findall(r'@\w+\{([^,]+),',bib)
    (P4/'reports/import_verification.json').write_text(json.dumps({
        'files':{n:{'download_sha256':sha(Path('/Users/hust-hwashin621m/Downloads')/n),'repo_sha256':sha(P4/n)} for n in ('Paper4_Master_Prompt_Deep_Research.md','Paper4_Verified_References.bib')},
        'entries':len(keys),'unique_keys':len(set(keys)), 'bib_matches_prompt_block':bib.strip()==block,
        'literature_access':'Author-supplied source cards; not an independent full-text re-verification'},indent=2))
    literature=[]
    for card in re.split(r'\n### R\d+\. ', prompt)[1:]:
        key=card.split('`')[1]
        access=re.search(r'\*\*Mức truy cập:\*\* `([^`]+)`',card)
        row={'key':key,'packet_access':access.group(1) if access else 'unverified','local_verification':'source packet read; access not upgraded'}
        for field,tag in [('supported_claim','Claim được hỗ trợ'),('limitations','Giới hạn'),('use','Dùng tại')]:
            m=re.search(r'\*\*'+tag+r':\*\* ([^\n]+)',card); row[field]=m.group(1) if m else ''
        literature.append(row)
    write_csv(P4/'reports/literature_evidence_matrix.csv',literature)
    rows=list(csv.DictReader((ROOT/'paper3/data/run_provenance.csv').open()))
    r03_sources=[]; r03_segments={}
    for spec in R03_SOURCE_CASES:
        record, source_record, segments = audit_r03_case(spec)
        r03_sources.append(source_record)
        r03_segments[record['case']] = segments
        if spec['analysis_condition']:
            rows.append(record)
    write_csv(P4/'data/r03_source_provenance.csv',r03_sources)
    (P4/'data/r03_restart_segments.json').write_text(json.dumps(r03_segments,indent=2)+'\n')
    inventory=[]; frames=[]; eligibility=[]
    for row in rows:
        run=row['case']; base=RAW/run; launchpath=base/'metadata/launch.json'; logpath=base/'logs/lammps.log'
        launch=json.loads(launchpath.read_text()) if launchpath.exists() else {}
        direct_r03=int(row['realization'])==3
        archived_launch=ROOT/f'paper3/evidence/{run}/launch.json'
        declared=launch if direct_r03 else json.loads(archived_launch.read_text())
        paths=sorted((base/'trajectories').glob('scratch.*.dump*'))
        bystep={}
        for p in paths:
            m=re.fullmatch(r'scratch\.(\d+)\.dump(?:\.zst|\.gz)?',p.name)
            if m: bystep.setdefault(int(m.group(1)),[]).append(p)
        selected=[]
        for step,ps in sorted(bystep.items()):
            # Prefer compressed source, record all duplicate alternatives without asserting equality.
            ps=sorted(ps,key=lambda p:(not p.name.endswith('.zst'),p.name)); p=ps[0]
            s=(step*.001-float(row['translation_start_ps']))*.03
            frames.append({'run_id':run,'step':step,'time_ps':step*.001,'distance_nm':s,'source_path':str(p),'size_bytes':p.stat().st_size,'duplicates':json.dumps([str(q) for q in ps[1:]]),'validation':'filename_inventory_only','branch_id':'unverified_single_directory'})
            if 4<=s<=16: selected.append(step)
        reference=ROOT/f"results/runs/relax_laminate_R{int(row['realization']):02d}/metadata/relaxed_structure.data"
        actualhash=sha(reference) if reference.exists() else 'missing'
        loghash=sha(logpath) if logpath.exists() else 'missing'
        launchhash=sha(launchpath) if launchpath.exists() else 'missing'
        rec={**row,'run_id':run,'source_path':str(base),'frame_count':len(bystep),'frames_4_16_nm':len(selected),
            'full_validation_count':0,'corrupt_frame_count':'unverified','duplicate_timesteps':sum(len(p)>1 for p in bystep.values()),
            'minimum_step':min(bystep) if bystep else '', 'maximum_step':max(bystep) if bystep else '',
            'max_gap_in_window_steps':max((b-a for a,b in zip(selected,selected[1:])),default=0),
            'log_path':str(logpath),'metadata_path':str(launchpath),'executed_input_path':declared.get('input','unverified'),
            'external_log_matches_paper3':('not applicable; direct Paper 4 R03 audit' if direct_r03 else loghash==row['log_sha256']),
            'external_launch_matches_paper3':('not applicable; direct Paper 4 R03 audit' if direct_r03 else launchhash==row['launch_sha256']),
            'reference_path':str(reference),'reference_sha256':actualhash,'reference_matches_launch':actualhash==declared.get('structure_sha256'),
            'preparation_origin':str(reference),'branch_id':'restart audit required' if int(row['restart_segments'])>1 else 'single logged segment',
            'parent_restart':'see executed log' if int(row['restart_segments'])>1 else 'none recorded',
            'atom_count':'unverified until parse','atom_type_mapping':'executed ADP mapping 1,3=Cu; 2,4=Ta; 5=C',
            'boundary_conditions':'executed change_box s p s; frame check pending','trajectory_schema':'unverified until parse',
            'integrity_status':'inventory only','paired_baseline_id':f"scratch_base_R{int(row['realization']):02d}",
            'analysis_eligibility':'candidate pending selected-frame validation' if selected else 'missing scratch-window frames',
            'exclusion_reason':'' if selected else 'no frames in 4-16 nm'}
        inventory.append(rec)
        eligibility.append({'run_id':run,'initial_chemistry':'candidate' if rec['reference_matches_launch'] else 'conflicting',
            'prediction':'candidate pending full frame and time audit' if selected else 'missing',
            'phase_event_duration':'not supported at sub-frame scale','causal_STZ':'not supported by saved coordinates',
            'internal_interface_mixing':'not attempted; outside selected question','force_power':'excluded; submitted Paper 3 scope'})
    write_csv(P4/'data/run_inventory.csv',inventory); write_csv(P4/'data/frame_manifest.csv',frames)
    write_csv(P4/'data/analysis_eligibility.csv',eligibility)
    print(json.dumps({'runs':len(inventory),'frames':len(frames),'candidate_runs':sum(r['frames_4_16_nm']>0 for r in inventory),'reference_matches':sum(r['reference_matches_launch'] for r in inventory),'external_logs_match':sum(r['external_log_matches_paper3'] is True for r in inventory),'direct_r03_runs':sum(int(r['realization'])==3 for r in inventory),'source_records':len(literature)},indent=2))

if __name__=='__main__': main()
