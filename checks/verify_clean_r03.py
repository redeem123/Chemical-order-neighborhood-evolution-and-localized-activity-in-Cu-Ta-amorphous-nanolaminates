"""Bounded verification of the two same-start clean R03 source packets.

Checks the archived raw-frame identities and recomputes the reported adhesion
force-window means. Structural result JSONs are checked for provenance and
finiteness, not independently recomputed from the selected frames here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


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
        results[name] = {"frames_sha256_and_headers": len(sources),
                         "clean_exit": True,
                         "reported_structural_values": "checked for provenance, not recalculated"}
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
    return {"status": "passed", "scope": "source identity and force window; not independent structural reanalysis",
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
