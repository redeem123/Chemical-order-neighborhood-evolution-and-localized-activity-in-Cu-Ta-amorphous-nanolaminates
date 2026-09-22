"""Replace only the user-specified cartoon with an actual OVITO configuration."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import ovito
from pypdf import PdfReader, PdfWriter, Transformation, PageObject
from audit_sources import P4, sha
import render_illustrative_evidence as renderer

OUT = P4/'figures/main/loading_actual'


def main():
    assert '.'.join(map(str, ovito.version)) == '3.15.5'
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = P4/'data/derived/fullfield/S6_R02_station10.json'
    meta = json.loads(meta_path.read_text()); source = Path(meta['data_path'])
    assert sha(source) == meta['data_sha256']
    with np.load(source) as a:
        xyz = np.vstack([a['xyz'], a['tool_xyz']])
        colors = np.array([to_rgb('#dea065') if t in (1, 3) else to_rgb('#315caa') for t in a['types']])
        colors = np.vstack([colors, np.tile(to_rgb('#818b96'), (len(a['tool_xyz']), 1))])
        tool_center = a['tool_xyz'].mean(axis=0)
    renderer.OUT = OUT
    renderer.DIRECTION = np.array([-.12, .96, -.24]); renderer.DIRECTION /= np.linalg.norm(renderer.DIRECTION)
    result = renderer.render('loading_actual', xyz, colors, radius=1.1, size=(2400, 1100))
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    im = plt.imread(result['raw_path']); h, w = im.shape[:2]
    ax.imshow(im); ax.axis('off'); ax.set_ylim(1.09*h, -.05*h)
    length = 50*h/(2*result['fov_half_height_A'])
    ax.plot([.07*w, .07*w+length], [.99*h]*2, color='#172738', lw=1.5)
    ax.text(.07*w+length/2, 1.015*h, '5 nm', fontsize=8, ha='center', va='top')
    direction = renderer.DIRECTION; right = np.cross(direction, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, direction); center = np.array(result['camera_target_A'])
    scale = h/(2*result['fov_half_height_A'])
    projected = [(tool_center-center)@right*scale+w/2, h/2-(tool_center-center)@up*scale]
    y = .09*h
    ax.annotate('', (.46*w, y), (.64*w, y), arrowprops=dict(arrowstyle='->', lw=.8))
    ax.text(.55*w, y-.025*h, 'Scratch −x', ha='center', va='bottom', fontsize=8)
    x = min(.94*w, projected[0]+.095*w)
    ax.annotate('', (x, projected[1]-.13*h), (x, projected[1]+.12*h), arrowprops=dict(arrowstyle='<->', lw=.8))
    ax.text(x+.018*w, projected[1], 'z', va='center', fontsize=8)
    handles = [Line2D([], [], marker='o', ls='', ms=3.5, color=color, label=label)
               for label, color in [('Cu', '#dea065'), ('Ta', '#315caa'), ('Tool', '#818b96')]]
    ax.legend(handles=handles, loc='lower right', ncol=3, frameon=False, fontsize=7,
              handletextpad=.25, columnspacing=.8, borderpad=0)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT/'loading_actual.pdf', bbox_inches='tight', pad_inches=.01)
    fig.savefig(OUT/'loading_actual.png', dpi=600, bbox_inches='tight', pad_inches=.01)
    plt.close(fig)
    # Retain the existing right-hand diagram without drawing a replacement.
    old = P4/'figures/main/review_followup/protocol_schematic.pdf'
    page = PdfReader(old).pages[0]; width = float(page.mediabox.width); height = float(page.mediabox.height)
    x0 = .505*width
    crop = PageObject.create_blank_page(width=width-x0, height=height)
    crop.merge_transformed_page(page, Transformation().translate(tx=-x0), over=True)
    writer = PdfWriter(); writer.add_page(crop)
    with (OUT/'reference_states_retained.pdf').open('wb') as stream: writer.write(stream)
    outputs = {str(p.relative_to(P4)): sha(p) for p in OUT.iterdir() if p.is_file()}
    report = dict(scope='Only Fig. 1(g) left cartoon replaced at explicit user request. Actual A50B20 R02 endpoint; existing right reference timeline retained by vector crop.',
                  renderer_sha256=sha(__file__), ovito_version='3.15.5',
                  sources={str(meta_path.relative_to(P4)): sha(meta_path), str(source.relative_to(P4)): sha(source),
                           str(old.relative_to(P4)): sha(old), 'scripts/render_illustrative_evidence.py': sha(P4/'scripts/render_illustrative_evidence.py')},
                  run='S6_R02', realization=2, step=meta['endpoints'][-1]['step'], travel_nm=meta['end_distance_nm'],
                  coordinate_changes=False, atom_count=len(xyz), camera=result, outputs=outputs,
                  reference_crop=dict(source_fraction_left=.505, source_width_pt=width, source_height_pt=height))
    (P4/'data/derived/loading_actual_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('run', 'step', 'travel_nm', 'atom_count', 'coordinate_changes')}, indent=2))


if __name__ == '__main__': main()
