"""Read-only checks of the v2 center counts, kinetic context and legacy subset."""
import csv,json,os
from pathlib import Path
import numpy as np

ROOT=Path(os.environ.get('PAPER4_DATA_ROOT', Path(__file__).resolve().parents[1]))
E=ROOT/'evidence'
V=E/'review_20260919_v2'
def rows(path):return list(csv.DictReader(path.open()))

def main():
    counts=rows(V/'center_counts.csv')
    for q in counts:
        r=int(q['realization']);g=np.load(E/f'review_20260919/S6_R{r:02d}/graphs.npz')
        mask=g['masks'][0]
        assert int(q['sampled'])==int((mask&g['sampled']).sum())
        assert int(q['all_centers'])==int(mask.sum())
    histories=rows(V/'kinetic_R01_R02.csv')+rows(V/'kinetic_R03.csv')
    for q in rows(V/'kinetic_context.csv'):
        r=int(q['realization'])
        arms={a:np.array([float(z['affine_temperature_K']) for z in sorted(
            [z for z in histories if int(z['realization'])==r and z['arm']==a],
            key=lambda z:int(z['phase_index']))]) for a in ('baseline','vibration')}
        assert all(len(v)==9 for v in arms.values())
        d=arms['vibration']-arms['baseline']
        for key,value in [('minimum',d.min()),('maximum',d.max()),('mean',d.mean())]:
            assert abs(value-float(q[key+'_difference_K']))<1e-10
    ledger=rows(V/'late_subset_ledger.csv');assert len(ledger)==11
    late_count=0
    for q in ledger:
        folder=E/'late_subset'/q['case']
        inter=rows(folder/'interfaces.csv');regional=rows(folder/'regional.csv')
        def tile(z):return (z['interface_index'],z['tile_x'],z['tile_y'])
        states={(z['arm'],z['stage'],z['phase_index']) for z in inter}
        common=set.intersection(*[{tile(z) for z in inter
            if (z['arm'],z['stage'],z['phase_index'])==s and z['valid']=='True'} for s in states])
        assert len(common)==int(q['common_tiles'])
        means={(a,k):np.mean([float(z['stretch_corrected_width_A']) for z in inter
            if z['arm']==a and z['stage']=='cycle' and int(z['phase_index'])==k and tile(z) in common])
            for a in ('baseline','vibration') for k in range(9)}
        width=np.mean([means['vibration',k]-means['baseline',k] for k in range(9)])
        assert abs(width-float(q['width_mean_A']))<1e-12
        late={a:next((z for z in regional if z['arm']==a and z['stage']=='postpass'
            and z['region']=='Cu-rich interior'),None) for a in ('baseline','vibration')}
        if all(late.values()):
            value=100*(float(late['vibration']['retained_initial_neighbor_fraction'])-
                       float(late['baseline']['retained_initial_neighbor_fraction']))
            assert abs(value-float(q['late_retention_pp']))<1e-10
            for a in ('baseline','vibration'):
                assert abs(float(late[a]['distance_nm'])-float(q[a+'_late_nm']))<1e-12
            assert abs(float(q['vibration_late_nm'])-16)<=.06
            assert q['status']=='available';late_count+=1
        else:
            assert q['case']=='S6_R01' and not q['late_retention_pp']
            assert q['status']=='unavailable_in_frozen_diagnostic_inventory'
    assert late_count==10
    print(json.dumps(dict(status='passed',center_populations=len(counts)*2,
        kinetic_preparations=3,width_means=len(ledger),late_retention_pairs=late_count,
        late_unavailable_in_legacy_subset='S6_R01'),indent=2))

if __name__=='__main__':main()
