"""Reproducible physical views; never fit a model or change frozen responses.

Run with the hea-md Python (OVITO 3.15.5). All additions are descriptive
visualizations. Source identities, sampled populations and scales are recorded.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import ovito
from ovito.io import import_file
from ovito.data import DataCollection, Particles
from ovito.pipeline import Pipeline, StaticSource
from ovito.vis import Viewport, TachyonRenderer
from ovito.modifiers import CoordinationAnalysisModifier
from atomic_analysis import reference, dump, metal, CONFIG
from audit_sources import ROOT, P4, sha
from render_ovito_views import activity_case_data

OUT = P4 / 'figures/main/illustrative'
DERIVED = P4 / 'data/derived'
CU, TA = '#dea065', '#315caa'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.dpi': 600})
records, sources, figures = [], {}, []
DIRECTION = np.array([-.48, .78, -.51])
DIRECTION /= np.linalg.norm(DIRECTION)


def source(path, expected=None):
    path = Path(path)
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f'Source hash mismatch: {path}')
    sources[str(path)] = digest
    return path


def framing(xyz, size=(1200,900)):
    center=(xyz.max(axis=0)+xyz.min(axis=0))/2
    right=np.cross(DIRECTION,[0,0,1.]); right/=np.linalg.norm(right)
    up=np.cross(right,DIRECTION)
    extent=max(abs((xyz-center)@up).max(),abs((xyz-center)@right).max()*size[1]/size[0])
    return center,float(extent*1.15+2)


def render(name, xyz, colors, *, target=None, fov=None, radius=.95,
           transparency=None, size=(1200, 900)):
    """Render real coordinates with a documented orthographic physical scale."""
    assert xyz.shape[1] == 3 and np.isfinite(xyz).all() and len(xyz)
    center = (xyz.max(axis=0) + xyz.min(axis=0)) / 2 if target is None else target
    right = np.cross(DIRECTION, [0, 0, 1.]); right /= np.linalg.norm(right)
    up = np.cross(right, DIRECTION)
    projected = np.column_stack(((xyz-center) @ right, (xyz-center) @ up))
    if fov is None:
        fov = max(abs(projected[:, 1]).max(), abs(projected[:, 0]).max()*size[1]/size[0]) * 1.15 + 2
    assert abs(projected[:,1]).max()+float(np.max(radius)) < fov, f'{name}: vertical clipping'
    assert abs(projected[:,0]).max()+float(np.max(radius)) < fov*size[0]/size[1], f'{name}: horizontal clipping'
    data = DataCollection(); particles = Particles(); data.objects.append(particles)
    particles.create_property('Position', data=xyz)
    particles.create_property('Color', data=np.asarray(colors)[:, :3])
    particles.create_property('Radius', data=np.broadcast_to(radius, len(xyz)))
    if transparency is not None:
        particles.create_property('Transparency', data=transparency)
    pipeline = Pipeline(source=StaticSource(data=data)); pipeline.add_to_scene()
    viewport = Viewport(type=Viewport.Type.Ortho, camera_dir=tuple(DIRECTION),
                        camera_up=(0, 0, 1), camera_pos=tuple(center-600*DIRECTION), fov=float(fov))
    path = OUT / f'{name}_raw.png'
    try:
        viewport.render_image(filename=str(path), size=size, background=(1, 1, 1),
                              renderer=TachyonRenderer(shadows=True, ambient_occlusion=True))
    finally:
        pipeline.remove_from_scene()
    record = {'asset': name, 'raw_path': str(path), 'raw_sha256': sha(path),
              'atoms': len(xyz), 'camera_direction': DIRECTION.tolist(),
              'camera_target_A': np.asarray(center).tolist(), 'fov_half_height_A': float(fov),
              'render_pixels': list(size), 'position_min_A': xyz.min(axis=0).tolist(),
              'position_max_A': xyz.max(axis=0).tolist()}
    records.append(record)
    print(f'Rendered {name}: {len(xyz)} atoms', flush=True)
    return record


def image_axis(ax, record, ruler_nm=1):
    im = plt.imread(record['raw_path']); ax.imshow(im); ax.set_axis_off()
    h, w = im.shape[:2]
    length = ruler_nm*10*h/(2*record['fov_half_height_A'])
    ax.plot([.07*w, .07*w+length], [.93*h]*2, color='#223344', lw=2)
    ax.text(.07*w+length/2, .93*h-13, f'{ruler_nm:g} nm', ha='center', va='bottom', fontsize=12)


def save_panel(fig, name, caption, group, *, scope, **metadata):
    pdf, png = OUT/f'{name}.pdf', OUT/f'{name}.png'
    fig.savefig(pdf, bbox_inches='tight', pad_inches=.03)
    fig.savefig(png, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    item = {'name': name, 'caption': caption, 'figure': group,
            'pdf': str(pdf), 'png': str(png), 'pdf_sha256': sha(pdf),
            'png_sha256': sha(png), 'scope': scope, **metadata}
    return item


def physical_panel(record, name, caption, group, *, scope, legend=False,
                   norm=None, cmap=None, color_label=None, zoom=None, **metadata):
    fig = plt.figure(figsize=(4.4, 3.25))
    ax = fig.add_axes([.0, .17, .67, .80] if zoom is not None
                     else [.0, .17 if norm is not None else .01, 1, .80 if norm is not None else .9])
    image_axis(ax, record)
    if zoom is not None:
        # A physically specified central subvolume, not an outcome-selected hotspot.
        inset = fig.add_axes([.69, .38, .30, .42], facecolor='white')
        image_axis(inset, zoom, .5)
        inset.set_title('Cutaway', fontsize=11, pad=1)
        for spine in inset.spines.values(): spine.set_visible(True)
    if norm is not None:
        cax = fig.add_axes([.18, .12, .65, .035])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
        cax.set_xlabel(color_label, fontsize=12, labelpad=2); cax.tick_params(labelsize=11, length=2)
    if legend:
        fig.legend(handles=[Line2D([], [], marker='o', color='none', markerfacecolor=CU, label='Cu'),
                            Line2D([], [], marker='o', color='none', markerfacecolor=TA, label='Ta')],
                   loc='lower center', ncol=2, frameon=False, fontsize=12)
    return save_panel(fig, name, caption, group, scope=scope, **metadata)


def group(name, panels, caption, label, columns=2):
    """Preview only has headings; manuscript lettering belongs to subfigure."""
    rows = (len(panels)+columns-1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.4*columns, 3.5*rows), squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        ax.imshow(plt.imread(panel['png'])); ax.axis('off')
        ax.set_title(panel['caption'], fontsize=11, color='#183c61', pad=1)
    for ax in list(axes.flat)[len(panels):]: ax.axis('off')
    fig.subplots_adjust(wspace=.01, hspace=.06)
    pdf, png = P4/f'figures/main/{name}.pdf', P4/f'figures/main/{name}.png'
    fig.savefig(pdf, bbox_inches='tight', pad_inches=.03)
    fig.savefig(png, bbox_inches='tight', pad_inches=.03); plt.close(fig)
    tex = ['\\begin{figure*}[p]', '\\centering']
    width = '.48' if columns == 2 else '.315'
    for index, panel in enumerate(panels):
        tex += [f'\\begin{{subfigure}}[t]{{{width}\\textwidth}}\\centering',
                '\\includegraphics[width=\\linewidth]{'+str(Path(panel['pdf']).relative_to(P4))+'}',
                '\\caption{'+panel['caption']+'}\\end{subfigure}'+
                ('\\par\\medskip' if (index+1)%columns == 0 else '\\hfill')]
    tex += ['\\caption{'+caption+'}', '\\label{'+label+'}', '\\end{figure*}']
    tex_path = OUT/f'{name}.tex'; tex_path.write_text('\n'.join(tex)+'\n')
    figures.append({'figure': name, 'panels': panels, 'caption': caption, 'label': label,
                    'pdf': str(pdf), 'png': str(png), 'tex': str(tex_path),
                    'pdf_sha256': sha(pdf), 'png_sha256': sha(png), 'tex_sha256': sha(tex_path)})


def precursor_figure():
    assembly = source(ROOT/'prepared_structures/scratch_campaign/laminate_R01.manifest.json')
    lineage = json.loads(assembly.read_text())
    launch = json.loads(source(ROOT/'results/runs/relax_laminate_R01/metadata/launch.json').read_text())
    assert lineage['data_sha256'] == launch['structure_sha256']
    panels, rdf = [], {}
    for material, prefix in [('A', 'top'), ('B', 'bottom')]:
        path = source(lineage[f'{prefix}_source'], lineage[f'{prefix}_sha256'])
        pipeline = import_file(str(path), atom_style='atomic')
        initial = pipeline.compute()
        assert all(initial.cell.pbc), 'Bulk RDF requires periodic precursor cell'
        xyz = np.asarray(initial.particles.positions)
        types = np.asarray(initial.particles.particle_types)
        ta = np.isin(types, [2, 4])
        assert abs(ta.mean()-lineage[f'{prefix}_composition']['Ta']) < 1e-12
        pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=12., number_of_bins=300, partial=True))
        calculated = pipeline.compute(); table = calculated.tables['coordination-rdf']
        rdf[material] = {'xy': table.xy(), 'components': list(table.y.component_names)}
        colors = matplotlib.colors.to_rgba_array(np.where(ta, TA, CU))
        full = render(f'precursor_{material}', xyz, colors, radius=1.05)
        p = physical_panel(full, f'fig1_precursor_{material}',
                           'Cu-rich precursor' if material == 'A' else 'Ta-rich precursor',
                           'fig1_initial_geometry', scope='Actual R01 size-conditioned assembly precursor; not the relaxed laminate',
                           legend=True, run=path.parent.parent.name, Ta_fraction=float(ta.mean()))
        panels.append(p)
    source_path = DERIVED/'precursor_rdf.npz'
    np.savez_compressed(source_path, **{k: v['xy'] for k, v in rdf.items()})
    source(source_path)
    for material in ('A', 'B'):
        xy = rdf[material]['xy']; components = rdf[material]['components']
        assert len(components) == 3
        fig, ax = plt.subplots(figsize=(4.4, 3.1))
        for index, color in enumerate(('#c28243', '#747b85', TA)):
            label = components[index].replace('1','Cu').replace('2','Ta')
            ax.plot(xy[:,0], xy[:,index+1], color=color, lw=1.2, label=label)
        c = .2 if material=='A' else .8
        # OVITO partial components are unordered type pairs (Cu-Cu, Cu-Ta, Ta-Ta).
        assert components in (['1-1','1-2','2-2'], ['Cu-Cu','Cu-Ta','Ta-Ta']), components
        total = (1-c)**2*xy[:,1]+2*c*(1-c)*xy[:,2]+c*c*xy[:,3]
        ax.plot(xy[:,0], total, color='black', ls='--', lw=1, label='Total')
        ax.set(xlim=(1.5,10), xlabel=r'$r$ (Å)', ylabel=r'$g(r)$')
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        panels.insert(0 if material=='A' else 1, save_panel(fig, f'fig1_rdf_{material}',
                      'Cu-rich pair distributions' if material=='A' else 'Ta-rich pair distributions',
                      'fig1_initial_geometry', scope='Periodic assembly precursor RDF, 0.04 Å bins; not laminate evolution'))
    # Retain whole-laminate context and all three measured composition profiles.
    old = json.loads(source(DERIVED/'ovito_render_manifest.json').read_text())
    asset = next(r for r in old['records'] if r['asset_id']=='initial_relaxed_R01')
    raw = source(asset['output_path'], asset['output_sha256'])
    source(asset['source_path'], asset['source_sha256'])
    fig, ax = plt.subplots(figsize=(4.4, 3.1)); ax.imshow(plt.imread(raw)); ax.axis('off')
    panels.append(save_panel(fig, 'fig1_laminate', 'Relaxed laminate, R01', 'fig1_initial_geometry',
                             scope='Existing hash-verified full relaxed R01 architecture'))
    profiles = list(csv.DictReader(source(DERIVED/'initial_profiles.csv').open()))
    fig, ax = plt.subplots(figsize=(4.4,3.1))
    for r, color, ls in [(1,TA,'-'),(2,'#bc753a','--'),(3,'#559278',':')]:
        rows = [v for v in profiles if int(v['realization'])==r]
        ax.plot([float(v['Ta_fraction']) for v in rows],[float(v['z_A'])/10 for v in rows],
                label=f'R{r:02d}',color=color,ls=ls)
    ax.set(xlabel='Ta atomic fraction',ylabel='Initial height (nm)',xlim=(0,1))
    ax.legend(frameon=False,fontsize=9,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01)); fig.tight_layout()
    panels.append(save_panel(fig,'fig1_profiles','Laminate depth profiles','fig1_initial_geometry',
                             scope='Measured depth-bin composition of all three relaxed references'))
    group('fig1_initial_geometry',panels,'Assembly precursors and relaxed laminate structure.','fig:geometry')


def chemical_figure():
    cfg = json.loads(source(CONFIG).read_text()); panels = []
    inventory = list(csv.DictReader(source(P4/'data/run_inventory.csv').open()))
    for r in (1,2,3):
        ref = reference(r); source(ref['source_path'],ref['sha256'])
        xyz = ref['xyz']; ta = np.isin(ref['types'],[2,4]); origin = np.isin(ref['types'],[1,2])
        strata = np.floor((xyz[:,2]-xyz[:,2].min())/cfg['null_z_bin_A']).astype(int)*2+origin
        labels = ta.copy(); rng = np.random.default_rng(cfg['seed']+r)
        for key in np.unique(strata):
            ix = np.flatnonzero(strata==key); labels[ix] = rng.permutation(ta[ix])
            assert labels[ix].sum()==ta[ix].sum()
        row = next(v for v in inventory if v['run_id']==f'scratch_base_R{r:02d}')
        center = np.array([float(row['x_start'])-100,ref['box'][1].mean(),xyz[:,2].max()-9])
        mask = np.all(abs(xyz-center)<np.array([25,18,5]),axis=1)
        for title, lab in [('Observed',ta),('Shuffled',labels)]:
            colors = matplotlib.colors.to_rgba_array(np.where(lab[mask],TA,CU))
            # Ta is foreground; Cu remains faint contextual support, with no network edges invented.
            transparency = np.where(lab[mask],0.,.88)
            rendered = render(f'chemical_R{r:02d}_{title}',xyz[mask],colors,
                              transparency=transparency,radius=.9,target=center,fov=24.)
            panels.append(physical_panel(rendered,f'fig2_R{r:02d}_{title}',f'R{r:02d}, {title.lower()}',
                          'fig2_chemical_field',scope='Same fixed 5 x 3.6 x 1 nm upper-layer volume; full-workpiece constrained shuffle before selection',
                          legend=True,run=f'relax_laminate_R{r:02d}',seed=cfg['seed']+r,shuffle_index=0,
                          selected_atoms=int(mask.sum()),center_A=center.tolist(),halfwidth_A=[25,18,5]))
    # Columns are preparations; rows observed / same first reproducible shuffle.
    panels = [panels[i] for i in (0,2,4,1,3,5)]
    group('fig2_chemical_field',panels,'Ta distributions in matched upper-layer volumes.','fig:chemical',3)


def load_cases():
    cases = []
    predictions = np.load(source(DERIVED/'predictions.npz'))
    for run, condition in [('scratch_base_R02','Baseline'),('S6_R02','A50B20')]:
        case = activity_case_data(run,condition,2)
        for key in ('start_path','end_path','analysis_data_path','analysis_metadata_path'):
            source(case[key])
        ref = reference(2); source(ref['source_path'],ref['sha256'])
        frame = dump(case['end_path']); positions = metal(frame,ref)
        archive = np.load(case['analysis_data_path']); a=archive['data']
        a=a[np.isclose(a[:,1],10)]; a=a[np.argsort(a[:,0])]
        ids=a[:,0].astype(int); index=np.searchsorted(ref['ids'],ids)
        assert np.array_equal(ref['ids'][index],ids)
        key=f'production_{condition}_1to2_d2_l1'
        pred=predictions[key]; pred=pred[np.isclose(pred[:,1],10)]; pred=pred[np.argsort(pred[:,0])]
        assert np.array_equal(pred[:,0],a[:,0]) and len(np.unique(ids))==len(ids)
        assert np.allclose(pred[:,2],np.log1p(a[:,11]/.01),rtol=1e-12,atol=1e-12)
        assert np.isfinite(pred).all()
        case.update(array=a,xyz=positions[index],initial_xyz=ref['xyz'][index],pred=pred,ids=ids,key=key)
        cases.append(case)
    return cases


def response_figures():
    cases=load_cases(); chemistry=Normalize(0,1)
    for case in cases:
        ic=(case['initial_xyz'].min(axis=0)+case['initial_xyz'].max(axis=0))/2
        case['zoom_mask']=np.all(abs(case['initial_xyz']-ic)<[10,8,15],axis=1)
    common_fov=max(framing(case['xyz'])[1] for case in cases)
    common_zoom_fov=max(framing(case['xyz'][case['zoom_mask']],(600,600))[1] for case in cases)
    ymax=float(np.ceil(max(c['pred'][:,2:5].max() for c in cases)))
    ymin=float(min(0,np.floor(min(c['pred'][:,2:5].min() for c in cases))))
    activity=Normalize(ymin,ymax)
    rmax=float(np.ceil(max(abs(c['pred'][:,4]-c['pred'][:,2]).max() for c in cases)))
    residual=TwoSlopeNorm(vcenter=0,vmin=-rmax,vmax=rmax)
    fig5,fig7=[],[]
    for case in cases:
        xyz=case['xyz']; a=case['array']; condition=case['condition']; pred=case['pred']
        center=(xyz.min(axis=0)+xyz.max(axis=0))/2
        # Zoom volume fixed relative to original cohort center, independent of values.
        ic=(case['initial_xyz'].min(axis=0)+case['initial_xyz'].max(axis=0))/2
        zoom_mask=case['zoom_mask']
        zoom_center=(xyz[zoom_mask].min(axis=0)+xyz[zoom_mask].max(axis=0))/2
        metadata={'run':case['run_id'],'station_nm':10,'prediction_key':case['key'],
                  'realization':2,'interval_ps':float(np.asarray(case['interval_ps']).flat[0]),
                  'selected_atoms':len(a),'joined_by':'run + station + atom ID',
                  'observed_transform_verified':True,'zoom_atoms':int(zoom_mask.sum()),
                  'zoom_initial_center_A':ic.tolist(),'zoom_initial_halfwidth_A':[10,8,15]}
        for field,values,norm,cmap,label in [
            ('chemistry',a[:,4],chemistry,'viridis','Initial first-shell Ta fraction'),
            ('observed',pred[:,2],activity,'magma','Observed endpoint activity, Y'),
            ('predicted',pred[:,4],activity,'magma','Predicted Y, augmented model'),
            ('residual',pred[:,4]-pred[:,2],residual,'RdBu_r','Predicted Y − observed Y')]:
            colors=plt.get_cmap(cmap)(norm(values))
            rendered=render(f'{condition}_{field}',xyz,colors,target=center,fov=common_fov,radius=1.05)
            zoom=render(f'{condition}_{field}_zoom',xyz[zoom_mask],colors[zoom_mask],
                        target=zoom_center,fov=common_zoom_fov,radius=1.05,size=(600,600))
            panel=physical_panel(rendered,f'{condition}_{field}',f'{condition}, {field}',
                    'fig5_chemistry_activity' if field=='chemistry' else 'fig7_spatial_prediction',
                    scope='R02 station-10 evaluated atoms only on actual endpoint coordinates; no unsampled atoms imputed',
                    norm=norm,cmap=cmap,color_label=label,zoom=None,
                    color_limits=[float(norm.vmin),float(norm.vmax)],**metadata)
            if field in ('chemistry','observed'):
                paired=physical_panel(rendered,f'{condition}_{field}_paired',f'{condition}, {field}',
                        'fig5_chemistry_activity',scope=panel['scope'],norm=norm,cmap=cmap,
                        color_label=label,zoom=zoom,color_limits=panel['color_limits'],**metadata)
                fig5.append(paired)
            if field!='chemistry': fig7.append(panel)
    group('fig5_chemistry_activity',fig5,'Initial chemistry and later non-affinity on the same R02 atoms.','fig:chemistry-activity')
    group('fig7_spatial_prediction',fig7,'Observed activity, forward prediction and residuals in R02.','fig:prediction-calibration',3)


def distribution_figure():
    cases=[]
    for condition,stem in [('Baseline','scratch_base'),('A50B20','S6')]:
        for r in (1,2):
            path=source(DERIVED/f'production_{stem}_R{r:02d}.npz')
            a=np.load(path)['data']
            cases.append((condition,r,a))
    ymax=float(np.ceil(max(np.log1p(a[:,11]/.01).max() for _,_,a in cases)))
    edges=(np.linspace(0,1,41),np.linspace(0,ymax,51)); binned=[]
    for condition,r,a in cases:
        h,_,_=np.histogram2d(a[:,4],np.log1p(a[:,11]/.01),bins=edges)
        assert int(h.sum())==len(a)
        binned.append(h)
    maxcount=max(h.max() for h in binned); panels=[]
    for (condition,r,a),h in zip(cases,binned):
        fig,ax=plt.subplots(figsize=(4.4,3.4))
        im=ax.pcolormesh(*edges,np.ma.masked_equal(h.T,0),cmap='cividis',norm=LogNorm(1,maxcount),rasterized=True)
        medx,medy=[],[]
        y=np.log1p(a[:,11]/.01)
        for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
            mask=(a[:,4]>=lo)&((a[:,4]<hi) if hi<1 else (a[:,4]<=hi))
            if mask.sum()>=30: medx.append((lo+hi)/2); medy.append(np.median(y[mask]))
        ax.plot(medx,medy,'o-',color='#f36251',ms=3,lw=1,label='Bin median')
        ax.set(xlabel='Initial first-shell Ta fraction',ylabel='Endpoint activity, Y',xlim=(0,1),ylim=(0,ymax))
        ax.text(.03,.97,f'N = {len(a):,}',ha='left',va='top',transform=ax.transAxes,fontsize=9)
        ax.legend(loc='lower left',frameon=False,fontsize=8)
        fig.colorbar(im,ax=ax,label='Atom–station count',fraction=.045,pad=.025)
        fig.tight_layout()
        panels.append(save_panel(fig,f'fig6_{condition}_R{r:02d}',f'{condition}, R{r:02d}',
                      'fig6_chemical_distributions',scope='All three stations pooled descriptively within condition and preparation; no independence or causal claim',
                      observations=len(a),histogram_count=int(h.sum()),bin_shape=[40,50],
                      common_color_limits=[1,float(maxcount)],median_minimum_count=30,run=f'{condition}_R{r:02d}'))
    group('fig6_chemical_distributions',panels,'Joint distributions of initial chemical environment and endpoint activity.','fig:activity-distributions')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source(__file__); source(CONFIG); source(P4/'scripts/render_ovito_views.py')
    precursor_figure(); chemical_figure(); response_figures(); distribution_figure()
    result={'renderer':str(Path(__file__).resolve()),'renderer_sha256':sha(__file__),
            'ovito_version':ovito.version_string,'sources':[{'path':p,'sha256':s} for p,s in sources.items()],
            'records':records,'figures':figures,
            'scope':'Descriptive physical views and distributions of existing frozen data; no model refit or new MD'}
    (DERIVED/'illustrative_render_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Completed illustrative evidence with verified source and identity joins.',flush=True)


if __name__=='__main__': main()
