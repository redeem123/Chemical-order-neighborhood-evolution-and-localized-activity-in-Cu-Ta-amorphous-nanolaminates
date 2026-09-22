"""OVITO whole-workpiece interval heatmaps and geometry-selected cutaways."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import itertools
import json
import warnings
from functools import partial
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch
import ovito
import render_illustrative_evidence as vis
from audit_sources import P4, sha

OUT = P4/'figures/main/fullfield'
vis.OUT = OUT
# High-resolution output supplies edge detail without the default 12x12 ray
# sampling cost on a half-million-atom workpiece. Coordinates/fields are unchanged.
vis.TachyonRenderer = partial(vis.TachyonRenderer, antialiasing_samples=3,
                              ambient_occlusion_samples=4)
NAME = 'fig5_fullfield_evolution'
VMAX = float(np.log1p(10/.01))
NORM = Normalize(0, VMAX, clip=True)


def load_cases():
    cases = []
    for run, condition in [('scratch_base_R02', 'Baseline'), ('S6_R02', 'A50B20')]:
        for station in (6, 10, 14):
            path = P4/f'data/derived/fullfield/{run}_station{station:02d}.json'
            m = json.loads(vis.source(path).read_text())
            # Raw inputs were fully hash-checked when computing the reduced field.
            # Rendering consumes the signed-off reduced array and its lineage,
            # not a hidden reread of the complete trajectories or relaxed reference.
            for source, digest in m['inputs'].items():
                if str(Path(source)).startswith(str(P4)):
                    vis.source(source, digest)
            a = np.load(vis.source(m['data_path'], m['data_sha256']))
            assert len(a['ids']) == m['total_metal_atoms'] and np.unique(a['ids']).size == len(a['ids'])
            xyz, tool, valid = a['xyz'], a['tool_xyz'], a['valid']
            assert int(valid.sum()) == m['valid_atoms']
            center = np.array([*tool.mean(axis=0)[:2], m['initial_surface_z_A']-15])
            half = np.array([45., 6., 45.])
            slab = np.all(abs(xyz-center) <= half, axis=1)
            tool_slab = np.all(abs(tool-center) <= half, axis=1)
            colors = plt.get_cmap('turbo')(NORM(np.log1p(a['values'][:, 1]/.01)))[:, :3]
            colors[~valid] = [.62, .62, .62]
            all_xyz = np.concatenate([xyz, tool])
            all_colors = np.concatenate([colors, np.tile([.58, .61, .64], (len(tool), 1))])
            transparency = None
            zoom_xyz = np.concatenate([xyz[slab], tool[tool_slab]])
            zoom_colors = np.concatenate([colors[slab], np.tile([.58, .61, .64], (tool_slab.sum(), 1))])
            zoom_transparency = None
            cases.append(dict(meta=m, condition=condition, station=station,
                              xyz=all_xyz, colors=all_colors, transparency=transparency,
                              zoom_xyz=zoom_xyz, zoom_colors=zoom_colors,
                              zoom_transparency=zoom_transparency,
                              center=center, half=half, slab_atoms=int(slab.sum()),
                              saturated_atoms=int(np.sum(a['values'][valid, 1] > 10))))
    return cases


def projection(points, rec):
    direction = np.asarray(rec['camera_direction'])
    right = np.cross(direction, [0, 0, 1.]); right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    width, height = rec['render_pixels']
    relative = points - rec['camera_target_A']
    return np.column_stack([width/2+(relative@right)*height/(2*rec['fov_half_height_A']),
                            height/2-(relative@up)*height/(2*rec['fov_half_height_A'])])


def compose(case, whole, zoom):
    fig = plt.figure(figsize=(4.2, 4.0))
    full_ax = fig.add_axes([0, .45, 1, .55])
    vis.image_axis(full_ax, whole, 10)
    zoom_ax = fig.add_axes([.14, .01, .72, .43])
    vis.image_axis(zoom_ax, zoom, 2)
    for axis in (full_ax, zoom_ax):
        for label in axis.texts:
            label.set_fontsize(17)
    # Box is the fixed tool-relative slab, never an outcome-selected hotspot.
    signs = np.array(list(itertools.product((-1, 1), repeat=3)))
    projected = projection(case['center']+signs*case['half'], whole)
    for i, s in enumerate(signs):
        for j in range(i+1, len(signs)):
            if np.count_nonzero(s != signs[j]) == 1:
                full_ax.plot(*projected[[i, j]].T, color='#40546b', lw=.65, ls='--', alpha=.9)
    anchor = projected[np.argmax(projected[:, 1])]
    for target in ((.1, 1), (.9, 1)):
        fig.add_artist(ConnectionPatch(anchor, target, coordsA='data', coordsB='axes fraction',
                       axesA=full_ax, axesB=zoom_ax, lw=.65, color='#8d9aa8', zorder=0))
    zoom_ax.text(.5, -.03, 'Central slab', ha='center', va='top', fontsize=16,
                 color='#273b52', transform=zoom_ax.transAxes)
    m = case['meta']
    return vis.save_panel(fig, f'fullfield_{m["run_id"]}_{case["station"]:02d}',
        f'{case["condition"]}, {case["station"]} nm', NAME,
        scope='R02 full-workpiece finite-interval field; geometry-selected central slab; no model population change',
        run=m['run_id'], realization=2, station_nm=case['station'], interval_ps=m['interval_ps'],
        total_metal_atoms=m['total_metal_atoms'], valid_atoms=m['valid_atoms'], invalid_atoms=m['invalid_atoms'],
        sampled_comparison_atoms=m['sampled_comparison_atoms'],
        sampled_max_abs_difference=m['sampled_max_abs_difference'],
        slab_atoms=case['slab_atoms'], slab_center_A=case['center'].tolist(), slab_halfwidth_A=case['half'].tolist(),
        upper_saturation_atoms=case['saturated_atoms'], color_limits_A2=[0, 10],
        color_transform='log1p(mean_Dmin2_A2 / 0.01)', invalid_color='gray',
        tool_color='opaque gray actual carbon atoms; excluded from field',
        start_distance_nm=m['start_distance_nm'], end_distance_nm=m['end_distance_nm'])


def colorbar():
    fig = plt.figure(figsize=(8.5, .78))
    ax = fig.add_axes([.05, .60, .57, .20])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap='turbo'), cax=ax, orientation='horizontal')
    values = np.array([0, .01, .1, 1, 10])
    bar.set_ticks(np.log1p(values/.01), labels=['0', '0.01', '0.1', '1', '≥10'])
    bar.ax.tick_params(labelsize=12, length=2)
    bar.set_label(r'Mean-normalized $D^2_{\min}$ (Å$^2$)', fontsize=13, labelpad=3)
    fig.text(.68, .65, 'Gray: unresolved fit / tool', fontsize=12, va='center')
    fig.text(.68, .24, 'R02 · interval ≈26 ps', fontsize=12, va='center')
    for suffix in ('pdf', 'png'):
        fig.savefig(OUT/f'colorbar.{suffix}', dpi=600, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert ovito.version_string == '3.15.5'
    vis.source(__file__); vis.source(vis.__file__)
    cases = load_cases()
    size, zoom_size = (1800, 1150), (1500, 1000)
    # Fixed camera center/extent across all six whole-workpiece views.
    lo = np.min([c['xyz'].min(axis=0) for c in cases], axis=0)
    hi = np.max([c['xyz'].max(axis=0) for c in cases], axis=0)
    target = (lo+hi)/2
    direction = vis.DIRECTION
    right = np.cross(direction, [0, 0, 1.]); right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    fov = max(abs((corners-target)@up).max(), abs((corners-target)@right).max()*size[1]/size[0])*1.08+2
    zoom_fov = 58.
    panels = []
    for c in cases:
        name = f'{c["meta"]["run_id"]}_{c["station"]:02d}'
        whole = vis.render(name+'_whole', c['xyz'], c['colors'],
                    target=target, fov=fov, radius=1.05, transparency=c['transparency'], size=size)
        zoom = vis.render(name+'_slab', c['zoom_xyz'], c['zoom_colors'],
                    target=c['center'], fov=zoom_fov, radius=1.05,
                    transparency=c['zoom_transparency'], size=zoom_size)
        panels.append(compose(c, whole, zoom))
    colorbar()
    vis.group(NAME, panels, 'Full-workpiece non-affinity and central cutaways at three scratch stations.',
              'fig:fullfield-evolution', 3)
    figure = vis.figures[-1]
    # Add the shared legend to both preview and LaTeX, never baked-in subfigure letters.
    tex_path = Path(figure['tex'])
    tex = tex_path.read_text().replace('\\caption{Full-workpiece',
          '\\includegraphics[width=.92\\textwidth]{figures/main/fullfield/colorbar.pdf}\n\\caption{Full-workpiece')
    tex_path.write_text(tex)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.9))
    for ax, panel in zip(axes.flat, panels):
        ax.imshow(plt.imread(panel['png'])); ax.axis('off')
        ax.set_title(panel['caption'], fontsize=15, color='#183c61', pad=1)
    fig.subplots_adjust(left=.01, right=.99, top=.965, bottom=.115, wspace=.015, hspace=.08)
    legend = fig.add_axes([.13, .005, .74, .08]); legend.imshow(plt.imread(OUT/'colorbar.png')); legend.axis('off')
    for suffix in ('pdf', 'png'):
        fig.savefig(figure[suffix], dpi=600, bbox_inches='tight', pad_inches=.03)
        figure[suffix+'_sha256'] = sha(figure[suffix])
    plt.close(fig)
    figure['tex_sha256'] = sha(tex_path)
    result = {'renderer': str(Path(__file__).resolve()), 'renderer_sha256': sha(__file__),
              'ovito_version': ovito.version_string,
              'ray_sampling': {'antialiasing_samples': 3, 'ambient_occlusion_samples': 4},
              'sources': [{'path': p, 'sha256': digest} for p, digest in vis.sources.items()],
              'records': vis.records, 'figures': vis.figures,
              'extra_assets': [{'path': str(OUT/f'colorbar.{s}'), 'sha256': sha(OUT/f'colorbar.{s}'),
                                'figure': NAME, 'role': 'shared_colorbar'} for s in ('pdf', 'png')],
              'scope': 'All-atom descriptive interval-field extension; original statistics and predictions unchanged'}
    (P4/'data/derived/fullfield_render_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Full-field figure and manifest completed.', flush=True)


if __name__ == '__main__':
    main()
