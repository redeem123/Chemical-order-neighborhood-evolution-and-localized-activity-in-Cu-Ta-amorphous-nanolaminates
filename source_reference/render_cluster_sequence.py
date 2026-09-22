"""OVITO activity components in translucent real-workpiece surfaces and full slabs."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch
import ovito
from ovito.data import DataCollection
from ovito.pipeline import Pipeline, StaticSource
from ovito.modifiers import ConstructSurfaceModifier
from ovito.vis import Viewport, TachyonRenderer
import render_illustrative_evidence as vis
from render_fullfield_activity import projection
from audit_sources import P4, sha

OUT = P4/'figures/main/cluster_sequence'
NAME = 'fig5_activity_sequence'
NORM = Normalize(0, np.log1p(10/.01), clip=True)
vis.OUT = OUT


def render_clusters(name, a, indices, target, fov, protocol, size=(1800,1150)):
    xyz, tool = a['xyz'], a['tool_xyz']
    field = a['values'][:,1]
    surface = DataCollection()
    surface.create_particles().create_property('Position', data=xyz)
    low, high = xyz.min(axis=0)-5, xyz.max(axis=0)+5
    matrix = np.column_stack([np.diag(high-low), low])
    surface.create_cell(matrix, pbc=(False,False,False)).vis.enabled = False
    surface.particles.vis.enabled = False
    mesh = Pipeline(source=StaticSource(data=surface))
    modifier = ConstructSurfaceModifier(radius=protocol['surface_probe_radius_A'], smoothing_level=0)
    modifier.vis.surface_color = (.77,.81,.86)
    modifier.vis.surface_transparency = protocol['surface_transparency']
    modifier.vis.show_cap = False
    mesh.modifiers.append(modifier)
    mesh.add_to_scene()
    active = DataCollection()
    particles = active.create_particles()
    positions = np.concatenate([xyz[indices], tool])
    colors = np.concatenate([plt.get_cmap('turbo')(NORM(np.log1p(field[indices]/.01)))[:,:3],
                             np.tile([.47,.49,.53],(len(tool),1))])
    particles.create_property('Position', data=positions)
    particles.create_property('Color', data=colors)
    particles.create_property('Radius', data=np.full(len(positions),1.05))
    pipe = Pipeline(source=StaticSource(data=active)); pipe.add_to_scene()
    viewport=Viewport(type=Viewport.Type.Ortho, camera_dir=tuple(vis.DIRECTION),
        camera_up=(0,0,1), camera_pos=tuple(target-600*vis.DIRECTION), fov=fov)
    path=OUT/f'{name}_whole_raw.png'
    try:
        viewport.render_image(filename=str(path),size=size,background=(1,1,1),
            renderer=TachyonRenderer(antialiasing_samples=3,ambient_occlusion_samples=4,
                                    shadows=True,ambient_occlusion=True))
    finally:
        pipe.remove_from_scene(); mesh.remove_from_scene()
    rec=dict(asset=name+'_whole',raw_path=str(path),raw_sha256=sha(path),
        atoms=len(positions), surface_input_atoms=len(xyz), active_atoms=len(indices),
        camera_direction=vis.DIRECTION.tolist(),camera_target_A=target.tolist(),
        fov_half_height_A=float(fov),render_pixels=list(size),
        surface_probe_radius_A=protocol['surface_probe_radius_A'],
        surface_transparency=protocol['surface_transparency'],
        surface_scope='Geometric alpha-shape context from all actual metal positions, open display boundaries. Not a defect mesh or quantitative surface metric.')
    vis.records.append(rec)
    print(f'Rendered {name} components and actual translucent surface',flush=True)
    return rec


def compose(seq, whole, slab, center, half):
    fig=plt.figure(figsize=(4.2,3.8))
    ax=fig.add_axes([0,.43,1,.57]); vis.image_axis(ax,whole,10)
    zoom=fig.add_axes([.10,.02,.80,.40]); vis.image_axis(zoom,slab,2)
    signs=np.array(list(itertools.product((-1,1),repeat=3)))
    projected=projection(center+signs*half,whole)
    for i,s in enumerate(signs):
        for j in range(i+1,len(signs)):
            if np.count_nonzero(s != signs[j])==1:
                ax.plot(*projected[[i,j]].T,color='#40546b',lw=.6,ls='--')
    bottom=projected[np.argmax(projected[:,1])]
    for point in ((.1,1),(.9,1)):
        fig.add_artist(ConnectionPatch(bottom,point,coordsA='data',coordsB='axes fraction',
            axesA=ax,axesB=zoom,lw=.6,color='#77879b',zorder=5,clip_on=False))
    for axis in (ax,zoom):
        for label in axis.texts: label.set_fontsize(15)
    field=seq['field']
    return vis.save_panel(fig,f"sequence_{seq['run']}_{seq['interval']}",
        f"{seq['condition']}, {field['end_distance_nm']:.2f} nm",NAME,
        scope='R02 consecutive finite-interval activity components with full-field central slab, not dislocations or irreversible STZs',
        run=seq['run'],interval_ps=field['interval_ps'],interval_index=seq['interval'],
        start_distance_nm=field['start_distance_nm'],end_distance_nm=field['end_distance_nm'],
        slab_center_A=center.tolist(),slab_halfwidth_A=half.tolist())


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--first-only',action='store_true'); args=parser.parse_args()
    assert ovito.version_string=='3.15.5'
    OUT.mkdir(parents=True,exist_ok=True)
    source=P4/'data/derived/cluster_sequence/manifest.json'
    evidence=json.loads(vis.source(source).read_text())
    protocol=evidence['protocol']
    vis.source(__file__); vis.source(vis.__file__)
    for p,h in evidence['sources'].items(): vis.source(p,h)
    cases=[]
    for seq in evidence['sequences']:
        a=np.load(vis.source(seq['field']['data_path'],seq['field']['data_sha256']))
        selected=np.load(vis.source(seq['selection_path'],seq['selection_sha256']))
        assert np.array_equal(a['ids'][selected['indices']],selected['ids'])
        cases.append((seq,a,selected['indices']))
    lo=np.min([a['xyz'].min(axis=0) for _,a,_ in cases],axis=0)
    hi=np.max([np.concatenate([a['xyz'],a['tool_xyz']]).max(axis=0) for _,a,_ in cases],axis=0)
    target=(lo+hi)/2
    right=np.cross(vis.DIRECTION,[0,0,1.]);right/=np.linalg.norm(right)
    up=np.cross(right,vis.DIRECTION)
    corners=np.array(list(itertools.product(*zip(lo,hi))))
    fov=float(max(abs((corners-target)@up).max(),abs((corners-target)@right).max()*1150/1800)*1.08+2)
    top=json.loads((P4/'data/derived/fullfield/scratch_base_R02_station10.json').read_text())['initial_surface_z_A']
    panels=[]
    for seq,a,indices in cases[:1] if args.first_only else cases:
        name=f"{seq['run']}_{seq['interval']}"
        whole=render_clusters(name,a,indices,target,fov,protocol)
        center=np.array([*a['tool_xyz'].mean(axis=0)[:2],top-15])
        half=np.array(protocol['slab_halfwidth_A'])
        mask=np.all(abs(a['xyz']-center)<=half,axis=1)
        toolmask=np.all(abs(a['tool_xyz']-center)<=half,axis=1)
        colors=plt.get_cmap('turbo')(NORM(np.log1p(a['values'][mask,1]/.01)))[:,:3]
        colors[~a['valid'][mask]]=.62
        slab=vis.render(name+'_slab',np.concatenate([a['xyz'][mask],a['tool_xyz'][toolmask]]),
            np.concatenate([colors,np.tile([.47,.49,.53],(toolmask.sum(),1))]),
            target=center,fov=58.,radius=1.05,size=(1500,1000))
        panel=compose(seq,whole,slab,center,half)
        panels.append(panel)
        (OUT/f'{name}_records.json').write_text(json.dumps({'records':vis.records[-2:],'panel':panel},indent=2))
    if args.first_only:return
    caption='Connected high-activity atoms and subsurface fields over three consecutive intervals.'
    tex=['\\begin{figure*}[p]','\\centering']
    for i,p in enumerate(panels):
        tex.extend(['\\begin{subfigure}[t]{.315\\textwidth}\\centering',
            '\\includegraphics[width=\\linewidth]{'+str(Path(p['pdf']).relative_to(P4))+'}',
            '\\caption{'+p['caption']+'}\\end{subfigure}'+('\\par\\medskip' if (i+1)%3==0 else '\\hfill')])
    tex.extend(['\\includegraphics[width=.92\\textwidth]{figures/main/cluster_sequence/colorbar.pdf}',
                '\\caption{'+caption+'}','\\label{fig:activity-sequence}','\\end{figure*}'])
    texpath=OUT/f'{NAME}.tex';texpath.write_text('\n'.join(tex)+'\n')
    fig=plt.figure(figsize=(8.5,.8));ax=fig.add_axes([.06,.58,.54,.20])
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=NORM,cmap='turbo'),cax=ax,orientation='horizontal')
    bar.set_ticks(np.log1p(np.array([0,.01,.1,1,10])/.01),labels=['0','0.01','0.1','1','≥10'])
    bar.ax.tick_params(labelsize=12);bar.set_label(r'Mean-normalized $D^2_{\min}$ (Å$^2$)',fontsize=13)
    fig.text(.65,.62,'3D: above-background atoms',fontsize=12)
    fig.text(.65,.18,'Slab: complete local heatmap',fontsize=12)
    for s in ('pdf','png'):fig.savefig(OUT/f'colorbar.{s}',dpi=600,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12.6,8.6))
    for ax,p in zip(axes.flat,panels):
        with Image.open(p['png']) as im:ax.imshow(np.array(im))
        ax.axis('off');ax.set_title(p['caption'],fontsize=14,color='#183c61',pad=1)
    fig.subplots_adjust(left=.01,right=.99,top=.965,bottom=.12,wspace=.025,hspace=.10)
    legend=fig.add_axes([.12,.005,.76,.085])
    with Image.open(OUT/'colorbar.png') as im:legend.imshow(np.array(im))
    legend.axis('off')
    figure=dict(figure=NAME,panels=panels,caption=caption,label='fig:activity-sequence',
                tex=str(texpath),tex_sha256=sha(texpath))
    for s in ('pdf','png'):
        p=P4/f'figures/main/{NAME}.{s}';fig.savefig(p,dpi=600,bbox_inches='tight',pad_inches=.03)
        figure[s]=str(p);figure[s+'_sha256']=sha(p)
    plt.close(fig)
    manifest=dict(renderer=str(Path(__file__).resolve()),renderer_sha256=sha(__file__),ovito_version=ovito.version_string,
        sources=[dict(path=p,sha256=h) for p,h in vis.sources.items()],records=vis.records,figures=[figure],
        extra_assets=[dict(path=str(OUT/f'colorbar.{s}'),sha256=sha(OUT/f'colorbar.{s}'),figure=NAME,role='shared_colorbar') for s in ('pdf','png')],
        scope=protocol['limitations'])
    (P4/'data/derived/cluster_sequence_render_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
