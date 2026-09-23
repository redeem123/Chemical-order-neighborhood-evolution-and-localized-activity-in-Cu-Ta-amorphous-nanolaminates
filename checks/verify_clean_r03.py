"""Bounded verification of the two same-start clean R03 source packets.

Checks archived raw-frame identities, recomputes adhesion force-window means,
and reconstructs selected structural observables from the supplied coordinates.
The label-permutation ensembles and interface-profile estimator are outside
this bounded check; no MD is run and no evidence file is changed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def header(path: Path):
    with path.open("rb") as stream:
        lines = [stream.readline().strip() for _ in range(4)]
    assert lines[0] == b"ITEM: TIMESTEP" and lines[2] == b"ITEM: NUMBER OF ATOMS", path
    return int(lines[1]), int(lines[3])


def thermo(path: Path):
    blocks, names, rows = [], None, []
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if fields and fields[0] == "Step":
            if rows:
                blocks.append((names, np.asarray(rows, dtype=float)))
            names, rows = fields, []
        elif names and len(fields) == len(names):
            try:
                rows.append([float(value) for value in fields])
            except ValueError:
                pass
    if rows:
        blocks.append((names, np.asarray(rows, dtype=float)))
    moving = [(names, values) for names, values in blocks
              if "v_tool_x" in names and len(values) > 10
              and np.ptp(values[:, names.index("v_tool_x")]) > 1]
    assert len(moving) == 1, f"Expected one moving thermo block: {path}"
    return moving[0]


def mean_on_window(times, values, start, end):
    assert np.all(np.diff(times) > 0) and times[0] <= start < end <= times[-1]
    inside = (times > start) & (times < end)
    t = np.r_[start, times[inside], end]
    v = np.r_[np.interp(start, times, values), values[inside],
              np.interp(end, times, values)]
    return float(np.trapezoid(v, t) / (end - start))


def metal_positions(path: Path, ref):
    with path.open("rt") as stream:
        lines = [stream.readline().strip() for _ in range(9)]
    assert lines[0] == "ITEM: TIMESTEP" and lines[2] == "ITEM: NUMBER OF ATOMS"
    assert lines[4] == "ITEM: BOX BOUNDS ss pp ss"
    columns = lines[8].split()[2:]
    assert columns[:5] == ["id", "type", "x", "y", "z"]
    periodic_y = np.fromstring(lines[6], sep=" ")
    np.testing.assert_allclose(periodic_y, ref["box"][1], atol=1e-5, rtol=0)
    values = np.loadtxt(path, skiprows=9)
    assert values.shape == (int(lines[3]), len(columns))
    assert np.isfinite(values).all()
    values = values[np.argsort(values[:, 0])]
    metal = values[values[:, 1] != 5]
    np.testing.assert_array_equal(metal[:, 0], ref["ids"])
    np.testing.assert_array_equal(metal[:, 1], ref["types"])
    return metal[:, 2:5]


def shells(x, centers, ly, cutoff=3.375):
    """Full-workpiece first shell with periodic images only along y."""
    images = np.concatenate([x + [0, shift, 0] for shift in (-ly, 0, ly)])
    owners = np.tile(np.arange(len(x), dtype=np.int32), 3)
    neighbors = cKDTree(images).query_ball_point(x[centers], cutoff, workers=1)
    return [np.setdiff1d(np.unique(owners[js]), [i], assume_unique=True)
            for i, js in zip(centers, neighbors)]


def retention(first, current):
    return np.asarray([len(np.intersect1d(a, b)) / len(a) if len(a) else np.nan
                       for a, b in zip(first, current)])


def nonaffinity(x0, x1, centers, first, ly):
    values = []
    for i, js in zip(centers, first):
        if len(js) < 4:
            values.append(np.nan)
            continue
        a = x0[js] - x0[i]
        b = x1[js] - x1[i]
        a[:, 1] -= ly * np.rint(a[:, 1] / ly)
        b[:, 1] -= ly * np.rint(b[:, 1] / ly)
        singular = np.linalg.svd(a, compute_uv=False)
        if singular[-1] <= 0 or singular[0] / singular[-1] > 1e6:
            values.append(np.nan)
            continue
        affine = np.linalg.lstsq(a, b, rcond=None)[0]
        values.append(float(np.sum((b - a @ affine) ** 2) / len(js)))
    return np.asarray(values)


def observed_order(first, centers, region, ta, composition):
    q = m = 0
    for i, js, selected in zip(centers, first, region):
        if selected and ta[i]:
            m += len(js)
            q += int(ta[js].sum())
    assert m > 0 and composition > 0
    return q, m, 1 - q / m / composition


def check_withdrawal(data: Path, case: Path, derived):
    evidence = data / "evidence"
    with np.load(evidence / "reviewer2_20260921/initial/R03.npz") as initial:
        ref = {key: initial[key] for key in ("ids", "types", "xyz", "box")}
    with np.load(evidence / "unloading_20260921/R03/coordinates_graphs.npz") as old:
        ids, old_start = old["ids"], old["xyz"][5]
        center_local, masks, compositions = old["centers"], old["masks"], old["compositions"]
    support = np.searchsorted(ref["ids"], ids)
    np.testing.assert_array_equal(ref["ids"][support], ids)
    centers = support[center_local]
    ta = np.isin(ref["types"], [2, 4])
    ly = float(np.diff(ref["box"][1])[0])
    relaxed_shell = shells(ref["xyz"], centers, ly)
    first_x = late_x = first_shell = late_shell = None
    max_error = 0.0
    for state in derived["states"]:
        time = state["time_ps"]
        x = metal_positions(case / f"unload.{time * 1000}.dump", ref)
        current = shells(x, centers, ly)
        if time == 0:
            np.testing.assert_allclose(x[support], old_start, atol=1e-9, rtol=0)
            first_x, first_shell = x.copy(), current
        if time == 150:
            late_x, late_shell = x.copy(), current
        initial_retained = retention(relaxed_shell, current)
        branch_retained = retention(first_shell, current)
        initial_d = nonaffinity(ref["xyz"], x, centers, relaxed_shell, ly)
        branch_d = nonaffinity(first_x, x, centers, first_shell, ly)
        late_d = nonaffinity(late_x, x, centers, late_shell, ly) if time == 200 else None
        for j, reported in enumerate(state["regions"].values()):
            use = masks[j]
            q, m, alpha = observed_order(current, centers, use, ta, compositions[j])
            assert (q, m) == (reported["Q"], reported["M"])
            for result, field in ((float(initial_retained[use].mean()), "retention"),
                                  (float(initial_d[use].mean()), "D_initial_A2"),
                                  (float(branch_retained[use].mean()), "branch_retention"),
                                  (float(branch_d[use].mean()), "D_branch_A2"),
                                  (alpha, "alpha_observed")):
                error = abs(result - reported[field])
                max_error = max(max_error, error)
                np.testing.assert_allclose(result, reported[field], atol=1e-8, rtol=0)
            if late_d is not None:
                result = float(late_d[use].mean())
                error = abs(result - reported["D_late_A2"])
                max_error = max(max_error, error)
                np.testing.assert_allclose(result, reported["D_late_A2"], atol=1e-8, rtol=0)
    return {"frames_reconstructed": len(derived["states"]), "centers": len(centers),
            "max_absolute_observable_error": max_error,
            "observables": "retention, mean-normalized non-affinity, observed Ta-Ta Q/M/alpha"}


def check_adhesion(data: Path, case: Path, derived):
    evidence = data / "evidence"
    with np.load(evidence / "reviewer2_20260921/initial/R03.npz") as initial:
        ref = {key: initial[key] for key in ("ids", "types", "xyz", "box")}
    with np.load(evidence / "supporting/physical_controls/ADH050_R03/arrays.npz") as old:
        ids = old["center_ids"]
        masks, compositions = old["region_masks"], old["composition"]
    centers = np.searchsorted(ref["ids"], ids)
    np.testing.assert_array_equal(ref["ids"][centers], ids)
    ta = np.isin(ref["types"], [2, 4])
    ly = float(np.diff(ref["box"][1])[0])
    relaxed_shell = shells(ref["xyz"], centers, ly)
    first_x = first_shell = last_x = None
    max_error = 0.0
    steps = sorted(map(int, derived["new_source_hashes"]))
    assert len(steps) == len(derived["regional"]) == 9
    for phase, step in enumerate(steps):
        x = metal_positions(case / f"scratch.{step}.dump", ref)
        current = shells(x, centers, ly)
        if phase == 0:
            first_x, first_shell = x.copy(), current
        if phase == 8:
            last_x = x.copy()
        reported = derived["regional"][phase]
        assert reported["phase_index"] == phase
        initial_retained = retention(relaxed_shell, current)
        cycle_retained = retention(first_shell, current)
        _, _, observed = observed_order(current, centers, masks[0], ta, compositions[0])
        for result, field in ((float(initial_retained.mean() * 100), "initial_retention_pct"),
                              (float(cycle_retained.mean() * 100), "cycle_retention_pct"),
                              (observed, "observed_Cu")):
            error = abs(result - reported[field])
            max_error = max(max_error, error)
            np.testing.assert_allclose(result, reported[field], atol=1e-8, rtol=0)
    dmin = nonaffinity(first_x, last_x, centers, first_shell, ly)
    assert int(np.isfinite(dmin).sum()) == derived["common_centers"]
    result = float(dmin[np.isfinite(dmin)].mean())
    max_error = max(max_error, abs(result - derived["dmin2_A2"]))
    np.testing.assert_allclose(result, derived["dmin2_A2"], atol=1e-8, rtol=0)
    return {"frames_reconstructed": len(steps), "centers": len(centers),
            "max_absolute_observable_error": max_error,
            "observables": "initial/cycle retention, observed Cu-rich Ta-Ta alpha, endpoint non-affinity"}


def verify(data: Path):
    root = data / "evidence" / "clean_rerun_20260923"
    assert root.is_dir(), "Clean R03 packet missing (requires dataset v1.1.0)"
    results = {}
    for name, reported in (("withdrawal", "unload_comparison.json"),
                           ("adhesion", "adhesion_comparison.json")):
        case = root / name
        derived = read(case / reported)
        receipt = read(case / "completion.json")
        assert receipt["status"] == "MD_COMPLETE"
        assert receipt["launch"]["returncode"] == 0
        assert sha(case / "completion.json") == derived["new_completion_sha256"]
        sources = ([(int(s["time_ps"] * 1000), s["source_sha256"])
                    for s in derived["states"]] if name == "withdrawal" else
                   [(int(step), digest) for step, digest in derived["new_source_hashes"].items()])
        assert len(sources) == (5 if name == "withdrawal" else 9)
        for step, digest in sources:
            path = case / (f"unload.{step}.dump" if name == "withdrawal" else
                           f"scratch.{step}.dump")
            assert sha(path) == digest, path
            assert header(path) == (step, 563827), path
        results[name] = {"frames_sha256_and_headers": len(sources), "clean_exit": True}
        results[name]["structural_reconstruction"] = (
            check_withdrawal(data, case, derived) if name == "withdrawal" else
            check_adhesion(data, case, derived))
        if name == "withdrawal":
            force = np.loadtxt(case / "force_trace.dat")
            assert force.shape == (20001, 7) and np.isfinite(force).all()
            np.testing.assert_array_equal(force[:, 0], np.arange(0, 200001, 10))
            results[name]["force_trace_rows"] = len(force)
        else:
            names, values = thermo(case / "lammps.log")
            plan = read(case / "selected_plan.json")
            start, end = (plan["t0_ps"] + plan["ramp_ps"] + d / 0.03
                          for d in plan["force_window_nm"])
            times = values[:, names.index("Time")]
            for key, column in (("Ft_mean_nN", "v_fx"), ("Fn_mean_nN", "v_fz")):
                computed = mean_on_window(times, values[:, names.index(column)] * 1.602176634,
                                          start, end)
                np.testing.assert_allclose(computed, derived["force"][key], rtol=0, atol=1e-10)
                results[name][key] = computed
            results[name]["force_window_ps"] = [start, end]
            results[name]["thermo_rows"] = len(values)
    return {"status": "passed", "scope": "selected structural observables and force window; not label-null or width reanalysis",
            "cases": results}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = verify(args.data)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
