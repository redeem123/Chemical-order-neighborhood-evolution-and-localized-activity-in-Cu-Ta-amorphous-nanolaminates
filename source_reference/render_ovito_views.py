#!/usr/bin/env python3
"""Render provenance-checked Paper 4 configuration views with OVITO.

The renderer uses archived configurations only. It produces a relaxed
species view, OVITO renderings of the initial broad- and first-shell
Ta-fraction fields, and full-track terminal local-shear fields for the six
predeclared visual conditions. The latter compare each actual final frame with
its launch-matched relaxed reference after giving the reference the final
scratch-cell geometry. They are qualitative configuration context, not the
matched 4--16 nm non-affinity analysis, a time-resolved pathway, or a force
mechanism. The renderer does not calculate new molecular-dynamics results.
"""
from __future__ import annotations

import csv
import json
import os
import re
import warnings
from pathlib import Path

# The Paper 3 renderer uses off-screen Tachyon rendering too.  This must be
# set before OVITO imports Qt in a non-interactive build environment.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import ovito
from ovito.io import export_file, import_file
from ovito.modifiers import AtomicStrainModifier, ColorCodingModifier
from ovito.pipeline import FileSource
from ovito.vis import TachyonRenderer, Viewport

from atomic_analysis import CONFIG, chemical_features, reference
from audit_sources import P4, sha


ASSET_DIR = P4 / "figures/main/ovito_assets"
MANIFEST_PATH = P4 / "data/derived/ovito_render_manifest.json"
CONDITION_ORDER = ("Baseline", "A10B05", "A10B20", "A25B10", "A25B20", "A50B20")

# The endpoint surface view keeps a fixed orthographic camera, crop, and
# magnification for every rendered final-surface configuration.
SURFACE_DIRECTION = np.array((-0.45, 0.65, -0.61), dtype=float)
SURFACE_DIRECTION /= np.linalg.norm(SURFACE_DIRECTION)
SURFACE_TARGET_A = np.array((324.0, 78.0, 111.0))
SURFACE_FOV_A = 140.0
SURFACE_SIZE = (1200, 1000)
SURFACE_X_RANGE_A = (180.0, 468.0)
SURFACE_Z_MIN_A = 85.0

# The initial-state view retains the Paper 3 color treatment while widening the
# fixed field of view enough to show the whole relaxed laminate.
INITIAL_DIRECTION = SURFACE_DIRECTION
INITIAL_TARGET_A = np.array((234.0, 78.0, 60.0))
INITIAL_FOV_A = 230.0
INITIAL_SIZE = (1200, 900)

# The initial chemical-field panel is an x-z slice through the deterministic
# ID-divisible-by-30 structural sample already used by initial_structure.py.
COMPOSITION_DIRECTION = np.array((0.0, -1.0, 0.0))
COMPOSITION_TARGET_A = np.array((234.0, 78.0, 60.0))
COMPOSITION_FOV_A = 96.0
COMPOSITION_SIZE = (1600, 600)
COMPOSITION_SLICE_HALF_WIDTH_A = 1.6
COMPOSITION_CMAP = "viridis"

# The endpoint activity maps use the hash-checked end frame of the 10 nm
# interval. Color values come from the frozen production arrays, mapped by
# persistent atom ID. Each selected condition is shown as an actual oblique
# 3D, side, and top OVITO view. The current-position crop is display-only;
# all numerical summaries retain the original-coordinate cohorts frozen in
# atomic_analysis.py.
ACTIVITY_STATION_NM = 10.0
ACTIVITY_CMAP = "magma"
ACTIVITY_CONTEXT_COLOR = np.array((0.79, 0.79, 0.79))
ACTIVITY_VIEW_ORDER = ("oblique", "side", "top")
ACTIVITY_VIEW_SPECS = {
    # The oblique direction is the same physical viewing direction as the
    # target-local surface contract, but it is centered on the actual 10 nm
    # endpoint and colored by the frozen Dmin2 field instead of species.
    "oblique": {
        "direction": SURFACE_DIRECTION.copy(),
        "camera_up": (0.0, 0.0, 1.0),
        "fov_A": 74.0,
        "size_px": (1200, 820),
        "y_half_width_A": 34.0,
        "x_range_relative_to_tool_A": (-52.0, 58.0),
        "z_range_relative_to_tool_A": (-84.0, 5.0),
        "target_offset_A": (3.0, 0.0, -39.0),
    },
    # This is an x-z view along the transverse y direction, matching the
    # scratch-side view used for the existing endpoint-field evidence.
    "side": {
        "direction": np.array((0.0, -1.0, 0.0)),
        "camera_up": (0.0, 0.0, 1.0),
        "fov_A": 60.0,
        "size_px": (1200, 780),
        "y_half_width_A": 12.0,
        "x_range_relative_to_tool_A": (-44.0, 50.0),
        "z_range_relative_to_tool_A": (-80.0, 5.0),
        "target_offset_A": (3.0, 0.0, -38.0),
    },
    # This looks down onto the scratch track. It widens the transverse context
    # but uses exactly the same stored atom-wise field and global color scale.
    "top": {
        "direction": np.array((0.0, 0.0, -1.0)),
        "camera_up": (0.0, 1.0, 0.0),
        "fov_A": 76.0,
        "size_px": (1200, 780),
        "y_half_width_A": 34.0,
        "x_range_relative_to_tool_A": (-52.0, 58.0),
        "z_range_relative_to_tool_A": (-84.0, 5.0),
        "target_offset_A": (3.0, 0.0, -39.0),
    },
}
ACTIVITY_CASES = (
    ("scratch_base_R01", "Baseline", 1),
    ("S1_R01", "A10B05", 1),
    ("S2_R01", "A10B20", 1),
    ("S4_R02", "A25B10", 2),
    ("S5_R02", "A25B20", 2),
    ("S6_R02", "A50B20", 2),
)
ACTIVITY_FIGURE_GROUP_BY_CASE = {
    ("Baseline", 1): "fig3",
    ("A10B05", 1): "fig3",
    ("A10B20", 1): "fig3",
    ("A25B10", 2): "fig4",
    ("A25B20", 2): "fig4",
    ("A50B20", 2): "fig4",
}

# These actual-species colors deliberately differ from the neutral-gray final
# surfaces used by Paper 3.  They encode Cu/Ta species only.
FINAL_CU_COLOR = np.array((0.10, 0.53, 0.58))
FINAL_TA_COLOR = np.array((0.77, 0.25, 0.45))

# Figures 3 and 4 retain the whole physical workpiece and full visible scratch
# track, following the Paper 3 visual contract. Unlike Paper 3's neutral
# morphology views, the atoms are colored by the actual final-versus-relaxed
# local shear strain. The fixed 0--1 range is a common display scale only.
FULL_TRACK_CMAP = "viridis"
FULL_TRACK_VIEW_ORDER = ("oblique", "side", "top")
FULL_TRACK_STRAIN_RANGE = (0.0, 1.0)
FULL_TRACK_REFERENCE_DIR = P4 / "build/ovito_fulltrack_references"
FULL_TRACK_RAW_DIR = P4 / "build/ovito_fulltrack_raw"
FULL_TRACK_SIDE_CENTER_Y_A = 78.0
FULL_TRACK_SIDE_HALF_WIDTH_A = 15.0
# Preserve the full x-z workpiece profile in the side view, matching the
# established full-scratch visual contract rather than the cropped 3D ROI.
FULL_TRACK_SIDE_CAMERA_POS_A = (234.0, -300.0, 80.0)
FULL_TRACK_SIDE_FOV_A = 80.0
FULL_TRACK_SIDE_SIZE = (1800, 600)
# The top view uses the same full-track framing as the side view: the complete
# 468 x 156 A workpiece fits in the 3:1 field without a local endpoint crop.
FULL_TRACK_TOP_CAMERA_POS_A = (234.0, 78.0, 500.0)
FULL_TRACK_TOP_FOV_A = 80.0
FULL_TRACK_TOP_SIZE = (1800, 600)


def require_file(path: Path, expected_sha256: str | None = None) -> str:
    """Return a verified file hash, failing rather than silently using a new input."""
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Source changed: {path}")
    return digest


def endpoint_step(path: Path) -> int:
    """Read the write_data timestep recorded in the immutable final frame."""
    first = path.open().readline()
    match = re.search(r"timestep = (\d+)", first)
    if match is None:
        raise ValueError(f"No endpoint timestep recorded in {path}")
    return int(match.group(1))


def eligible_surface_cases() -> tuple[tuple[str, str, int], ...]:
    """Return all target-local final configurations eligible for Paper 4."""
    rows = list(csv.DictReader((P4 / "data/run_inventory.csv").open()))
    eligible = [
        row for row in rows
        if row["analysis_eligibility"] == "selected endpoint intervals validated"
    ]
    unknown = {row["condition"] for row in eligible}.difference(CONDITION_ORDER)
    if unknown:
        raise ValueError(f"Unknown condition ordering: {sorted(unknown)}")
    ordered = sorted(
        eligible,
        key=lambda row: (CONDITION_ORDER.index(row["condition"]), int(row["realization"])),
    )
    return tuple((row["run_id"], row["condition"], int(row["realization"])) for row in ordered)


def display_only_surface_cases() -> tuple[tuple[str, str, int], ...]:
    """Return the one R01 final configuration retained only for visual coverage.

    S4_R01 has a real, hash-checked final configuration and a launch-matched
    relaxed reference, but the local archive has no saved 4--16 nm frames.
    It therefore cannot enter any endpoint metric or predictive comparison.
    """
    rows = list(csv.DictReader((P4 / "data/run_inventory.csv").open()))
    candidates = [row for row in rows if row["run_id"] == "S4_R01"]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one S4_R01 inventory record")
    row = candidates[0]
    if (row["condition"], row["realization"], row["analysis_eligibility"], row["reference_matches_launch"]) != (
        "A25B10", "1", "missing scratch-window frames", "True",
    ):
        raise ValueError("S4_R01 no longer has the approved display-only scope")
    if not (Path(row["source_path"]) / "metadata/final.data").is_file():
        raise FileNotFoundError("S4_R01 final configuration is unavailable")
    return ((row["run_id"], row["condition"], int(row["realization"])),)


def renderer() -> TachyonRenderer:
    """The same ray-tracing settings used by Paper 3 final-surface panels."""
    return TachyonRenderer(shadows=True, ambient_occlusion=True)


def surface_viewport() -> Viewport:
    return Viewport(
        type=Viewport.Type.Ortho,
        camera_dir=tuple(SURFACE_DIRECTION),
        camera_up=(0.0, 0.0, 1.0),
        camera_pos=tuple(SURFACE_TARGET_A - 600.0 * SURFACE_DIRECTION),
        fov=SURFACE_FOV_A,
    )


def initial_viewport() -> Viewport:
    return Viewport(
        type=Viewport.Type.Ortho,
        camera_dir=tuple(INITIAL_DIRECTION),
        camera_up=(0.0, 0.0, 1.0),
        camera_pos=tuple(INITIAL_TARGET_A - 600.0 * INITIAL_DIRECTION),
        fov=INITIAL_FOV_A,
    )


def composition_viewport() -> Viewport:
    """Return an x-z view for the initial local-composition field."""
    return Viewport(
        type=Viewport.Type.Ortho,
        camera_dir=tuple(COMPOSITION_DIRECTION),
        camera_up=(0.0, 0.0, 1.0),
        camera_pos=tuple(COMPOSITION_TARGET_A - 600.0 * COMPOSITION_DIRECTION),
        fov=COMPOSITION_FOV_A,
    )


def render_initial() -> dict:
    """Ray-trace the real relaxed R01 reference, colored by Cu/Ta species."""
    ref = reference(1)
    source = Path(ref["source_path"])
    source_sha256 = require_file(source, ref["sha256"])
    output = ASSET_DIR / "fig1_relaxed_R01_ovito.png"

    pipeline = import_file(str(source), atom_style="atomic")

    def appearance(frame, data):
        particle_types = np.asarray(data.particles["Particle Type"])
        if not set(particle_types.astype(int)) <= {1, 2, 3, 4}:
            raise ValueError("Unexpected type mapping in relaxed R01 reference")
        # The executed type map is 1/3 = Cu and 2/4 = Ta.  These are species
        # colors, not layer-origin colors or a chemical-network classifier.
        colors = np.where(
            np.isin(particle_types, (1, 3))[:, None],
            np.array((0.91, 0.30, 0.12)),
            np.array((0.20, 0.37, 0.77)),
        )
        data.particles_.create_property("Color", data=colors)
        data.particles.vis.radius = 1.25
        data.cell_.vis.enabled = False

    pipeline.modifiers.append(appearance)
    pipeline.add_to_scene()
    try:
        initial_viewport().render_image(
            filename=str(output),
            size=INITIAL_SIZE,
            background=(1.0, 1.0, 1.0),
            renderer=renderer(),
        )
    finally:
        pipeline.remove_from_scene()

    return {
        "asset_id": "initial_relaxed_R01",
        "kind": "relaxed_initial_geometry",
        "source_path": str(source),
        "source_sha256": source_sha256,
        "output_path": str(output),
        "output_sha256": require_file(output),
        "realization": 1,
        "atom_color_rule": "types 1/3 Cu orange; types 2/4 Ta blue",
        "camera": {
            "projection": "orthographic",
            "direction": INITIAL_DIRECTION.tolist(),
            "target_A": INITIAL_TARGET_A.tolist(),
            "fov_A": INITIAL_FOV_A,
            "size_px": list(INITIAL_SIZE),
        },
        "scope": "relaxed reference geometry only; no tool contact or loading claim",
    }


def initial_composition_centers(ref: dict) -> np.ndarray:
    """Return the valid central-slice atoms used for every initial field."""
    positions = ref["xyz"]
    y_mid = float(np.mean(ref["box"][1]))
    top = float(positions[:, 2].max())
    return np.flatnonzero(
        (np.abs(positions[:, 1] - y_mid) <= COMPOSITION_SLICE_HALF_WIDTH_A)
        & (positions[:, 0] > 8.0)
        & (positions[:, 0] < positions[:, 0].max() - 8.0)
        & (positions[:, 2] > 8.0)
        & (positions[:, 2] < top - 3.6)
    )


def render_initial_composition_field(realization: int, descriptor: str) -> dict:
    """Render one initial local Ta-fraction field on the matching relaxed reference."""
    ref = reference(realization)
    source = Path(ref["source_path"])
    source_sha256 = require_file(source, ref["sha256"])
    config_sha256 = require_file(CONFIG)
    cfg = json.loads(CONFIG.read_text())
    centers = initial_composition_centers(ref)
    features, _ = chemical_features(ref, centers, cfg["cutoff_A"], cfg)
    sample_ids = ref["ids"][centers]
    if descriptor == "8A":
        field_column = 3
        asset_id = f"initial_8A_Ta_fraction_R{realization:02d}"
        output = ASSET_DIR / f"fig2_initial_8A_Ta_fraction_R{realization:02d}_ovito.png"
        property_name = "Initial 8A Ta fraction"
        property_definition = (
            "Ta fraction among metal neighbours within 8 Å, excluding the central atom"
        )
        field_scope = "initial broad-composition covariate"
    elif descriptor == "first_shell":
        field_column = 2
        asset_id = f"initial_shell_Ta_fraction_R{realization:02d}"
        output = ASSET_DIR / f"fig2_initial_shell_Ta_fraction_R{realization:02d}_ovito.png"
        property_name = "Initial first-shell Ta fraction"
        property_definition = (
            "Ta fraction among metal neighbours within "
            f"{cfg['cutoff_A']:g} Å, excluding the central atom"
        )
        field_scope = "initial first-shell descriptor"
    else:
        raise ValueError(f"Unknown initial Ta-fraction descriptor: {descriptor}")
    values = features[:, field_column]
    if len(centers) < 100 or not np.isfinite(values).all():
        raise ValueError("Initial Ta-fraction slice is not a finite atom field")
    pipeline = import_file(str(source), atom_style="atomic")

    def composition_field(frame, data):
        identifiers = np.asarray(data.particles["Particle Identifier"], dtype=np.int64)
        positions = np.asarray(data.particles.positions)
        lookup = np.searchsorted(sample_ids, identifiers)
        within = lookup < len(sample_ids)
        matched = np.zeros(len(identifiers), dtype=bool)
        matched[within] = sample_ids[lookup[within]] == identifiers[within]
        field = np.full(len(identifiers), np.nan)
        field[matched] = values[lookup[matched]]
        keep = matched
        kept_field = field[keep]
        colors = plt.get_cmap(COMPOSITION_CMAP)(np.clip(kept_field, 0.0, 1.0))[:, :3]
        data.particles_.delete_elements(~keep)
        data.particles_.create_property(property_name, data=kept_field)
        data.particles_.create_property("Color", data=colors)
        data.particles.vis.radius = 1.55
        data.cell_.vis.enabled = False

    pipeline.modifiers.append(composition_field)
    pipeline.add_to_scene()
    try:
        composition_viewport().render_image(
            filename=str(output),
            size=COMPOSITION_SIZE,
            background=(1.0, 1.0, 1.0),
            renderer=renderer(),
        )
    finally:
        pipeline.remove_from_scene()

    return {
        "asset_id": asset_id,
        "kind": "relaxed_initial_composition_field",
        "source_path": str(source),
        "source_sha256": source_sha256,
        "analysis_config_path": str(CONFIG),
        "analysis_config_sha256": config_sha256,
        "output_path": str(output),
        "output_sha256": require_file(output),
        "realization": realization,
        "descriptor": descriptor,
        "atom_selection": "all valid metal atoms in a central 3.2 Å y slice",
        "property_definition": property_definition,
        "color_rule": "viridis, fixed from 0 to 1",
        "camera": {
            "projection": "orthographic",
            "direction": COMPOSITION_DIRECTION.tolist(),
            "target_A": COMPOSITION_TARGET_A.tolist(),
            "fov_A": COMPOSITION_FOV_A,
            "size_px": list(COMPOSITION_SIZE),
        },
        "scope": (
            f"{field_scope} in the relaxed reference only; it does not establish a "
            "mechanical or causal network"
        ),
    }


def render_initial_composition_colorbar() -> dict:
    """Render the shared fixed-scale legend for Figure 2's local-fraction fields."""
    config_sha256 = require_file(CONFIG)
    colorbar_pdf = ASSET_DIR / "fig2_initial_Ta_fraction_colorbar.pdf"
    colorbar_png = ASSET_DIR / "fig2_initial_Ta_fraction_colorbar.png"
    colorbar_figure, colorbar_axis = plt.subplots(figsize=(5.0, 0.42))
    ColorbarBase(
        colorbar_axis,
        cmap=plt.get_cmap(COMPOSITION_CMAP),
        norm=Normalize(0.0, 1.0),
        orientation="horizontal",
        ticks=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    colorbar_axis.set_xlabel("Initial local Ta fraction", labelpad=2)
    colorbar_figure.savefig(colorbar_pdf, bbox_inches="tight", pad_inches=0.02)
    colorbar_figure.savefig(colorbar_png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(colorbar_figure)

    return {
        "asset_id": "initial_Ta_fraction_colorbar",
        "kind": "composition_field_colorbar",
        "source_path": str(CONFIG),
        "source_sha256": config_sha256,
        "output_path": str(colorbar_pdf),
        "output_sha256": require_file(colorbar_pdf),
        "png_path": str(colorbar_png),
        "png_sha256": require_file(colorbar_png),
        "color_rule": "viridis, fixed from 0 to 1",
        "scope": "shared legend for the adjacent initial local Ta-fraction OVITO renderings",
    }


def render_initial_compositions() -> list[dict]:
    """Render Figure 2 fields across preparations and descriptor scales."""
    return [
        render_initial_composition_field(1, "8A"),
        render_initial_composition_field(1, "first_shell"),
        render_initial_composition_field(2, "8A"),
        render_initial_composition_field(3, "8A"),
        render_initial_composition_colorbar(),
    ]


def activity_case_data(run_id: str, condition: str, realization: int) -> dict:
    """Load one frozen 10 nm activity field with its audited endpoint pair."""
    inventory = {
        row["run_id"]: row for row in csv.DictReader((P4 / "data/run_inventory.csv").open())
    }
    if run_id not in inventory:
        raise KeyError(f"No run inventory record for {run_id}")
    row = inventory[run_id]
    if (row["condition"], int(row["realization"])) != (condition, realization):
        raise ValueError(f"Inventory mismatch for activity map {run_id}")
    if row["analysis_eligibility"] != "selected endpoint intervals validated":
        raise ValueError(f"Activity-map run is not endpoint eligible: {run_id}")

    array_path = P4 / f"data/derived/production_{run_id}.npz"
    metadata_path = array_path.with_suffix(".json")
    array_sha256 = require_file(array_path)
    metadata_sha256 = require_file(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    if metadata["run_id"] != run_id or metadata["mode"] != "production":
        raise ValueError(f"Unexpected production metadata for {run_id}")
    config_sha256 = require_file(CONFIG)
    if metadata["config_sha256"] != config_sha256:
        raise ValueError(f"Production configuration changed for {run_id}")

    with np.load(array_path, allow_pickle=False) as archive:
        columns = [str(column) for column in archive["columns"]]
        data = archive["data"]
    expected_columns = {
        "id", "station_nm", "Dmin_mean_A2", "interval_ps",
    }
    if not expected_columns.issubset(columns):
        raise ValueError(f"Missing frozen activity columns for {run_id}")
    column = {name: columns.index(name) for name in columns}
    station_mask = np.isclose(data[:, column["station_nm"]], ACTIVITY_STATION_NM)
    identifiers = data[station_mask, column["id"]].astype(np.int64)
    values = data[station_mask, column["Dmin_mean_A2"]]
    interval_ps = data[station_mask, column["interval_ps"]]
    if len(identifiers) < 100 or len(np.unique(identifiers)) != len(identifiers):
        raise ValueError(f"Invalid frozen station population for {run_id}")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"Non-positive or non-finite activity field for {run_id}")
    if not np.allclose(interval_ps, interval_ps[0]):
        raise ValueError(f"Inconsistent station interval for {run_id}")

    motion_index = next(
        (index for index, check in enumerate(metadata["motion_checks"])
         if np.isclose(float(check["station_nm"]), ACTIVITY_STATION_NM)),
        None,
    )
    if motion_index is None:
        raise ValueError(f"No validated 10 nm motion record for {run_id}")
    frames = metadata["validated_frames"]
    if len(frames) != 2 * len(metadata["motion_checks"]):
        raise ValueError(f"Frame-pair ordering is incomplete for {run_id}")
    start, end = frames[2 * motion_index: 2 * motion_index + 2]
    start_path = Path(start["source_path"])
    end_path = Path(end["source_path"])
    start_sha256 = require_file(start_path, start["source_sha256"])
    end_sha256 = require_file(end_path, end["source_sha256"])
    if abs(float(interval_ps[0]) - float(metadata["motion_checks"][motion_index]["interval_ps"])) > 1e-9:
        raise ValueError(f"Interval metadata mismatch for {run_id}")

    ordering = np.argsort(identifiers)
    return {
        "run_id": run_id,
        "condition": condition,
        "realization": realization,
        "identifiers": identifiers[ordering],
        "values": values[ordering],
        "interval_ps": float(interval_ps[0]),
        "start": start,
        "end": end,
        "start_path": start_path,
        "start_sha256": start_sha256,
        "end_path": end_path,
        "end_sha256": end_sha256,
        "analysis_data_path": array_path,
        "analysis_data_sha256": array_sha256,
        "analysis_metadata_path": metadata_path,
        "analysis_metadata_sha256": metadata_sha256,
        "analysis_config_sha256": config_sha256,
    }


def activity_viewport(tool_position: np.ndarray, view: str) -> tuple[Viewport, np.ndarray]:
    """Return one fixed tool-relative camera for an endpoint-field view."""
    if view not in ACTIVITY_VIEW_SPECS:
        raise ValueError(f"Unknown endpoint-field view: {view}")
    spec = ACTIVITY_VIEW_SPECS[view]
    direction = np.asarray(spec["direction"], dtype=float)
    target = tool_position + np.asarray(spec["target_offset_A"], dtype=float)
    viewport = Viewport(
        type=Viewport.Type.Ortho,
        camera_dir=tuple(direction),
        camera_up=tuple(spec["camera_up"]),
        camera_pos=tuple(target - 600.0 * direction),
        fov=float(spec["fov_A"]),
    )
    return viewport, target


def activity_color_limits(cases: list[dict]) -> tuple[float, float]:
    """Fix one display scale from all six predeclared 10 nm fields."""
    values = np.concatenate([case["values"] for case in cases])
    lower, upper = np.quantile(values, (0.01, 0.99))
    if not (np.isfinite(lower) and np.isfinite(upper) and 0 < lower < upper):
        raise ValueError("Cannot form a finite common logarithmic activity scale")
    return float(lower), float(upper)


def render_activity_map(case: dict, vmin: float, vmax: float, view: str) -> dict:
    """Map one frozen endpoint field to one actual OVITO view by atom ID."""
    run_id = case["run_id"]
    if view not in ACTIVITY_VIEW_SPECS:
        raise ValueError(f"Unknown endpoint-field view: {view}")
    spec = ACTIVITY_VIEW_SPECS[view]
    figure_group = ACTIVITY_FIGURE_GROUP_BY_CASE.get((case["condition"], case["realization"]))
    if figure_group is None:
        raise ValueError(f"No approved Figure 3/4 group for {run_id}")
    output = ASSET_DIR / (
        f"{figure_group}_Dmin_mean_{case['condition']}_R{case['realization']:02d}_10nm_{view}_ovito.png"
    )
    identifiers = case["identifiers"]
    values = case["values"]
    tool = np.array((
        float(case["end"]["tool_com_x_A"]),
        float(case["end"]["tool_com_y_A"]),
        float(case["end"]["tool_com_z_A"]),
    ))
    viewport, target = activity_viewport(tool, view)
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    pipeline = import_file(str(case["end_path"]))
    counts: dict[str, int] = {}

    def endpoint_field(frame, data):
        particle_ids = np.asarray(data.particles["Particle Identifier"], dtype=np.int64)
        positions = np.asarray(data.particles.positions)
        particle_types = np.asarray(data.particles["Particle Type"], dtype=np.int64)
        lookup = np.searchsorted(identifiers, particle_ids)
        valid_lookup = lookup < len(identifiers)
        matched = np.zeros(len(particle_ids), dtype=bool)
        matched[valid_lookup] = identifiers[lookup[valid_lookup]] == particle_ids[valid_lookup]

        relative_x = positions[:, 0] - tool[0]
        relative_z = positions[:, 2] - tool[2]
        context = (
            (particle_types != 5)
            & (np.abs(positions[:, 1] - tool[1]) <= spec["y_half_width_A"])
            & (relative_x >= spec["x_range_relative_to_tool_A"][0])
            & (relative_x <= spec["x_range_relative_to_tool_A"][1])
            & (relative_z >= spec["z_range_relative_to_tool_A"][0])
            & (relative_z <= spec["z_range_relative_to_tool_A"][1])
        )
        field = matched & context
        if field.sum() < 100:
            raise ValueError(f"Too few mapped field atoms in the endpoint crop for {run_id}")
        colors = np.tile(ACTIVITY_CONTEXT_COLOR, (len(particle_ids), 1))
        radii = np.full(len(particle_ids), 0.54)
        colors[field] = plt.get_cmap(ACTIVITY_CMAP)(norm(values[lookup[field]]))[:, :3]
        radii[field] = 1.38
        counts["context_atoms"] = int(context.sum())
        counts["field_atoms"] = int(field.sum())
        data.particles_.delete_elements(~context)
        data.particles_.create_property("Color", data=colors[context])
        data.particles_.create_property("Radius", data=radii[context])
        data.cell_.vis.enabled = False

    pipeline.modifiers.append(endpoint_field)
    pipeline.add_to_scene()
    try:
        viewport.render_image(
            filename=str(output),
            size=spec["size_px"],
            background=(1.0, 1.0, 1.0),
            renderer=renderer(),
        )
    finally:
        pipeline.remove_from_scene()

    return {
        "asset_id": f"activity_{case['condition']}_R{case['realization']:02d}_station10_{view}",
        "kind": "endpoint_nonaffinity_field",
        "run_id": run_id,
        "condition": case["condition"],
        "realization": case["realization"],
        "figure_group": figure_group,
        "view": view,
        "station_nm": ACTIVITY_STATION_NM,
        "interval_ps": case["interval_ps"],
        "source_path": str(case["end_path"]),
        "source_sha256": case["end_sha256"],
        "start_source_path": str(case["start_path"]),
        "start_source_sha256": case["start_sha256"],
        "endpoint_step": int(case["end"]["step"]),
        "analysis_data_path": str(case["analysis_data_path"]),
        "analysis_data_sha256": case["analysis_data_sha256"],
        "analysis_metadata_path": str(case["analysis_metadata_path"]),
        "analysis_metadata_sha256": case["analysis_metadata_sha256"],
        "analysis_config_path": str(CONFIG),
        "analysis_config_sha256": case["analysis_config_sha256"],
        "output_path": str(output),
        "output_sha256": require_file(output),
        "property_definition": "frozen Dmin_mean_A2 from the 10 nm endpoint interval, joined to the archived endpoint by persistent atom ID",
        "mapped_sample_atoms": int(len(identifiers)),
        "visual_field_atoms": counts["field_atoms"],
        "visual_context_atoms": counts["context_atoms"],
        "color_rule": {
            "colormap": ACTIVITY_CMAP,
            "normalization": "logarithmic",
            "vmin_A2": vmin,
            "vmax_A2": vmax,
            "limits_rule": "global 1st and 99th percentiles across the six predeclared 10 nm fields",
        },
        "camera": {
            "projection": "orthographic",
            "view": view,
            "direction": np.asarray(spec["direction"], dtype=float).tolist(),
            "camera_up": list(spec["camera_up"]),
            "tool_relative_target_offset_A": list(spec["target_offset_A"]),
            "target_A": target.tolist(),
            "fov_A": float(spec["fov_A"]),
            "size_px": list(spec["size_px"]),
            "y_slice_half_width_A": float(spec["y_half_width_A"]),
            "x_range_relative_to_tool_A": list(spec["x_range_relative_to_tool_A"]),
            "z_range_relative_to_tool_A": list(spec["z_range_relative_to_tool_A"]),
        },
        "scope": (
            f"actual {view} endpoint view for frozen original-cohort values; gray atoms are unmeasured endpoint context. "
            "The current-position crop is visual only and does not redefine the numerical cohort, identify STZ cores, "
            "or resolve motion between endpoints"
        ),
    }


def render_activity_colorbar(vmin: float, vmax: float) -> dict:
    """Render the shared logarithmic legend for Figures 3 and 4."""
    colorbar_pdf = ASSET_DIR / "endpoint_Dmin_mean_colorbar.pdf"
    colorbar_png = ASSET_DIR / "endpoint_Dmin_mean_colorbar.png"
    figure, axis = plt.subplots(figsize=(5.1, 0.42))
    ColorbarBase(
        axis,
        cmap=plt.get_cmap(ACTIVITY_CMAP),
        norm=LogNorm(vmin=vmin, vmax=vmax),
        orientation="horizontal",
    )
    axis.set_xlabel(r"Endpoint $\overline{D}_{\min}^{2}$ (Å$^{2}$)", labelpad=2)
    figure.savefig(colorbar_pdf, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(colorbar_png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return {
        "asset_id": "activity_Dmin_mean_colorbar",
        "kind": "endpoint_nonaffinity_colorbar",
        "source_path": str(CONFIG),
        "source_sha256": require_file(CONFIG),
        "output_path": str(colorbar_pdf),
        "output_sha256": require_file(colorbar_pdf),
        "png_path": str(colorbar_png),
        "png_sha256": require_file(colorbar_png),
        "color_rule": {
            "colormap": ACTIVITY_CMAP,
            "normalization": "logarithmic",
            "vmin_A2": vmin,
            "vmax_A2": vmax,
            "limits_rule": "global 1st and 99th percentiles across the six predeclared 10 nm fields",
        },
        "scope": "shared display legend for the adjacent endpoint non-affinity OVITO fields",
    }


def render_activity_fields() -> list[dict]:
    """Render the six fixed-condition endpoint fields in three matched views."""
    cases = [activity_case_data(*case) for case in ACTIVITY_CASES]
    if [(case["condition"], case["realization"]) for case in cases] != [
        ("Baseline", 1), ("A10B05", 1), ("A10B20", 1),
        ("A25B10", 2), ("A25B20", 2), ("A50B20", 2),
    ]:
        raise ValueError("The approved endpoint activity-map case set changed")
    if set(ACTIVITY_FIGURE_GROUP_BY_CASE) != set((case["condition"], case["realization"]) for case in cases):
        raise ValueError("The endpoint activity figure grouping changed")
    vmin, vmax = activity_color_limits(cases)
    return [
        *(render_activity_map(case, vmin, vmax, view) for case in cases for view in ACTIVITY_VIEW_ORDER),
        render_activity_colorbar(vmin, vmax),
    ]


def full_track_case_data(run_id: str, condition: str, realization: int) -> dict:
    """Resolve one approved terminal configuration and its matched reference."""
    inventory = {
        row["run_id"]: row for row in csv.DictReader((P4 / "data/run_inventory.csv").open())
    }
    if run_id not in inventory:
        raise KeyError(f"No run inventory record for {run_id}")
    row = inventory[run_id]
    if (row["condition"], int(row["realization"])) != (condition, realization):
        raise ValueError(f"Inventory mismatch for full-track field {run_id}")
    if row["analysis_eligibility"] != "selected endpoint intervals validated":
        raise ValueError(f"Full-track field is not an approved target condition: {run_id}")
    if row["reference_matches_launch"] != "True":
        raise ValueError(f"Launch-matched relaxed reference is unavailable for {run_id}")

    source = Path(row["source_path"]) / "metadata/final.data"
    reference_source = Path(row["reference_path"])
    source_sha256 = require_file(source)
    reference_sha256 = require_file(reference_source, row["reference_sha256"])
    reference_record = reference(realization)
    if Path(reference_record["source_path"]).resolve() != reference_source.resolve():
        raise ValueError(f"Relaxed reference path disagrees with the analysis record for {run_id}")
    if reference_record["sha256"] != reference_sha256:
        raise ValueError(f"Relaxed reference hash disagrees with the analysis record for {run_id}")

    step = endpoint_step(source)
    endpoint_time_ps = 0.001 * step
    travel_nm = 0.03 * (endpoint_time_ps - float(row["translation_start_ps"]))
    if not (np.isfinite(travel_nm) and travel_nm > 0):
        raise ValueError(f"Non-positive terminal travel for {run_id}")
    figure_group = ACTIVITY_FIGURE_GROUP_BY_CASE.get((condition, realization))
    if figure_group is None:
        raise ValueError(f"No approved Figure 3/4 group for {run_id}")
    return {
        "run_id": run_id,
        "condition": condition,
        "realization": realization,
        "figure_group": figure_group,
        "source_path": source,
        "source_sha256": source_sha256,
        "reference_source_path": reference_source,
        "reference_source_sha256": reference_sha256,
        "endpoint_step": step,
        "endpoint_time_ps": endpoint_time_ps,
        "translation_start_ps": float(row["translation_start_ps"]),
        "travel_nm": float(travel_nm),
    }


def remove_tool(frame, data):
    """Exclude the rigid diamond before local-strain correspondence."""
    data.particles_.delete_elements(np.asarray(data.particles["Particle Type"]) == 5)


def full_track_reference(case: dict) -> tuple[Path, str]:
    """Write the relaxed workpiece in the final scratch-cell geometry.

    The terminal cell includes the vacuum above the surface. Giving the
    relaxed reference the same cell avoids artificial top--bottom neighbours
    during the local-strain calculation while preserving all reference atom
    positions and identifiers.
    """
    FULL_TRACK_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    output = FULL_TRACK_REFERENCE_DIR / f"{case['run_id']}_reference.dump.gz"
    terminal = import_file(str(case["source_path"]), atom_style="atomic").compute()
    relaxed = import_file(str(case["reference_source_path"]), atom_style="atomic").compute()
    relaxed.cell_[...] = np.asarray(terminal.cell)
    export_file(
        relaxed,
        str(output),
        "lammps/dump",
        columns=[
            "Particle Identifier",
            "Particle Type",
            "Position.X",
            "Position.Y",
            "Position.Z",
        ],
    )
    return output, require_file(output)


def full_track_pipeline(case: dict, reference_path: Path) -> tuple[object, dict]:
    """Build one tool-free terminal strain pipeline and audit correspondence."""
    pipeline = import_file(str(case["source_path"]), atom_style="atomic")
    pipeline.modifiers.append(remove_tool)
    endpoint = pipeline.compute()
    relaxed = import_file(str(reference_path)).compute()
    identifiers = np.asarray(endpoint.particles["Particle Identifier"], dtype=np.int64)
    reference_identifiers = np.asarray(relaxed.particles["Particle Identifier"], dtype=np.int64)
    if len(identifiers) != len(np.unique(identifiers)):
        raise ValueError(f"Duplicate workpiece atom identifiers in {case['run_id']}")
    if not np.array_equal(np.sort(identifiers), np.sort(reference_identifiers)):
        raise ValueError(f"Reference atom identifiers do not match {case['run_id']}")
    endpoint_order = np.argsort(identifiers)
    reference_order = np.argsort(reference_identifiers)
    endpoint_types = np.asarray(endpoint.particles["Particle Type"], dtype=np.int64)
    reference_types = np.asarray(relaxed.particles["Particle Type"], dtype=np.int64)
    if not np.array_equal(endpoint_types[endpoint_order], reference_types[reference_order]):
        raise ValueError(f"Reference atom types do not match {case['run_id']}")

    strain = AtomicStrainModifier(
        cutoff=3.375,
        affine_mapping=AtomicStrainModifier.AffineMapping.Off,
        select_invalid_particles=True,
    )
    strain.reference = FileSource()
    strain.reference.load(str(reference_path))
    pipeline.modifiers.append(strain)
    strained = pipeline.compute()
    shear = np.asarray(strained.particles["Shear Strain"])
    invalid = np.asarray(strained.particles["Selection"]).astype(bool) | ~np.isfinite(shear)
    valid = ~invalid
    if valid.sum() < 1000:
        raise ValueError(f"Too few valid local-strain fits for {case['run_id']}")
    return pipeline, {
        "atoms_workpiece": int(len(identifiers)),
        "atom_ids_match": True,
        "types_match": True,
        "invalid_strain_count": int(invalid.sum()),
        "valid_shear_mean": float(shear[valid].mean()),
        "valid_fraction_above_0_3": float(np.mean(shear[valid] > 0.3)),
    }


def full_track_viewport(view: str) -> tuple[Viewport, tuple[int, int], TachyonRenderer]:
    """Return the three fixed full-workpiece cameras used by Figures 3 and 4."""
    if view == "oblique":
        return surface_viewport(), SURFACE_SIZE, renderer()
    if view == "side":
        return (
            Viewport(
                type=Viewport.Type.Ortho,
                camera_dir=(0.0, 1.0, 0.0),
                camera_pos=FULL_TRACK_SIDE_CAMERA_POS_A,
                fov=FULL_TRACK_SIDE_FOV_A,
            ),
            FULL_TRACK_SIDE_SIZE,
            TachyonRenderer(shadows=False, ambient_occlusion=False),
        )
    if view == "top":
        return (
            Viewport(
                type=Viewport.Type.Ortho,
                camera_dir=(0.0, 0.0, -1.0),
                camera_up=(0.0, 1.0, 0.0),
                camera_pos=FULL_TRACK_TOP_CAMERA_POS_A,
                fov=FULL_TRACK_TOP_FOV_A,
            ),
            FULL_TRACK_TOP_SIZE,
            TachyonRenderer(shadows=False, ambient_occlusion=False),
        )
    raise ValueError(f"Unknown full-track view: {view}")


def decorate_full_track_view(raw: Path, output: Path, view: str) -> None:
    """Add only the scale bar and scratch-direction annotation to an OVITO view."""
    image = plt.imread(raw)
    if view == "oblique":
        figure, axis = plt.subplots(figsize=(6, 5), dpi=200)
        figure.subplots_adjust(0, 0, 1, 1)
        axis.imshow(image)
        axis.axis("off")
        right = np.cross(SURFACE_DIRECTION, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        up = np.cross(right, SURFACE_DIRECTION)
        delta = np.array((-right[0], up[0]))
        delta = delta / np.linalg.norm(delta) * 190.0
        start = np.array((1030.0, 900.0))
        end = start + delta
        axis.annotate("", xy=end, xytext=start,
                      arrowprops=dict(arrowstyle="->", lw=2, color="black"))
        axis.text(980, 955, "Scratch direction", ha="right", fontsize=13,
                  bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=0.8))
        scale_end = 80.0 + 50.0 * SURFACE_SIZE[0] / 280.0
        axis.plot((80, scale_end), (920, 920), color="white", lw=4.5, solid_capstyle="butt")
        axis.plot((80, scale_end), (920, 920), color="black", lw=2.1, solid_capstyle="butt")
        axis.text((80 + scale_end) / 2, 890, "5 nm", ha="center", fontsize=13,
                  bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=0.8))
        figure.savefig(output, dpi=200)
        plt.close(figure)
        return

    if view == "side":
        figure, axis = plt.subplots(figsize=(6, 2), dpi=300)
        figure.subplots_adjust(0, 0, 1, 1)
        axis.imshow(image)
        axis.axis("off")
        pixels_per_A = FULL_TRACK_SIDE_SIZE[1] / (2.0 * FULL_TRACK_SIDE_FOV_A)
        scale_end = 90.0 + 100.0 * pixels_per_A
        axis.plot((90, scale_end), (540, 540), color="white", lw=2, solid_capstyle="butt")
        axis.text((90 + scale_end) / 2, 520, "10 nm", ha="center", fontsize=9, color="white")
        axis.annotate("Scratch direction", xy=(90, 70), xytext=(420, 70),
                      arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
                      fontsize=9, va="center")
        figure.savefig(output, dpi=300)
        plt.close(figure)
        return

    if view == "top":
        figure, axis = plt.subplots(figsize=(6, 2), dpi=300)
        figure.subplots_adjust(0, 0, 1, 1)
        axis.imshow(image)
        axis.axis("off")
        pixels_per_A = FULL_TRACK_TOP_SIZE[1] / (2.0 * FULL_TRACK_TOP_FOV_A)
        scale_end = 90.0 + 100.0 * pixels_per_A
        axis.plot((90, scale_end), (540, 540), color="white", lw=2, solid_capstyle="butt")
        axis.text((90 + scale_end) / 2, 520, "10 nm", ha="center", fontsize=9, color="white")
        axis.annotate("Scratch direction", xy=(90, 70), xytext=(420, 70),
                      arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
                      fontsize=9, va="center")
        figure.savefig(output, dpi=300)
        plt.close(figure)
        return

    raise ValueError(f"Unknown full-track decoration view: {view}")


def render_full_track_view(case: dict, pipeline, strain_stats: dict,
                           reference_path: Path, reference_sha256: str, view: str) -> dict:
    """Render one whole-workpiece terminal local-shear view from a real field."""
    if view not in FULL_TRACK_VIEW_ORDER:
        raise ValueError(f"Unknown full-track view: {view}")
    output = ASSET_DIR / (
        f"{case['figure_group']}_fulltrack_shear_{case['condition']}_R{case['realization']:02d}_{view}_ovito.png"
    )
    FULL_TRACK_RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw = FULL_TRACK_RAW_DIR / f"{case['run_id']}_{view}.png"
    viewport, size, render_engine = full_track_viewport(view)
    color = ColorCodingModifier(
        property="Shear Strain",
        start_value=FULL_TRACK_STRAIN_RANGE[0],
        end_value=FULL_TRACK_STRAIN_RANGE[1],
        gradient=ColorCodingModifier.Viridis(),
    )

    def display_field(frame, data):
        positions = np.asarray(data.particles.positions)
        invalid = (
            np.asarray(data.particles["Selection"]).astype(bool)
            | ~np.isfinite(np.asarray(data.particles["Shear Strain"]))
        )
        data.particles_["Color_"][invalid] = (0.65, 0.65, 0.65)
        if view == "oblique":
            keep = (
                (positions[:, 0] >= SURFACE_X_RANGE_A[0])
                & (positions[:, 0] <= SURFACE_X_RANGE_A[1])
                & (positions[:, 2] >= SURFACE_Z_MIN_A)
            )
        elif view == "side":
            keep = np.ones(len(positions), dtype=bool)
            keep &= np.abs(positions[:, 1] - FULL_TRACK_SIDE_CENTER_Y_A) <= FULL_TRACK_SIDE_HALF_WIDTH_A
        else:
            # The top camera sees the entire x-y workpiece and its actual
            # terminal surface; no station or lateral slice is applied.
            keep = np.ones(len(positions), dtype=bool)
        if keep.sum() < 1000:
            raise ValueError(f"Too few full-track display atoms for {case['run_id']} {view}")
        data.particles_.delete_elements(~keep)
        data.particles.vis.radius = 1.35
        data.cell_.vis.enabled = False

    pipeline.modifiers.append(color)
    pipeline.modifiers.append(display_field)
    pipeline.add_to_scene()
    try:
        viewport.render_image(
            filename=str(raw),
            size=size,
            background=(1.0, 1.0, 1.0),
            renderer=render_engine,
        )
    finally:
        pipeline.remove_from_scene()
        pipeline.modifiers.pop()
        pipeline.modifiers.pop()
    decorate_full_track_view(raw, output, view)

    if view == "oblique":
        camera = {
            "projection": "orthographic",
            "view": view,
            "direction": SURFACE_DIRECTION.tolist(),
            "camera_up": [0.0, 0.0, 1.0],
            "target_A": SURFACE_TARGET_A.tolist(),
            "fov_A": SURFACE_FOV_A,
            "size_px": list(SURFACE_SIZE),
            "x_range_A": list(SURFACE_X_RANGE_A),
            "z_min_A": SURFACE_Z_MIN_A,
        }
    elif view == "side":
        camera = {
            "projection": "orthographic",
            "view": view,
            "direction": [0.0, 1.0, 0.0],
            "camera_pos_A": list(FULL_TRACK_SIDE_CAMERA_POS_A),
            "fov_A": FULL_TRACK_SIDE_FOV_A,
            "size_px": list(FULL_TRACK_SIDE_SIZE),
            "full_xz_workpiece": True,
            "y_center_A": FULL_TRACK_SIDE_CENTER_Y_A,
            "y_half_width_A": FULL_TRACK_SIDE_HALF_WIDTH_A,
        }
    else:
        camera = {
            "projection": "orthographic",
            "view": view,
            "direction": [0.0, 0.0, -1.0],
            "camera_up": [0.0, 1.0, 0.0],
            "camera_pos_A": list(FULL_TRACK_TOP_CAMERA_POS_A),
            "fov_A": FULL_TRACK_TOP_FOV_A,
            "size_px": list(FULL_TRACK_TOP_SIZE),
            "full_xy_workpiece": True,
        }
    return {
        "asset_id": f"fulltrack_shear_{case['condition']}_R{case['realization']:02d}_{view}",
        "kind": "terminal_fulltrack_local_shear_field",
        "run_id": case["run_id"],
        "condition": case["condition"],
        "realization": case["realization"],
        "figure_group": case["figure_group"],
        "view": view,
        "source_path": str(case["source_path"]),
        "source_sha256": case["source_sha256"],
        "reference_source_path": str(case["reference_source_path"]),
        "reference_source_sha256": case["reference_source_sha256"],
        "generated_reference_path": str(reference_path),
        "generated_reference_sha256": reference_sha256,
        "endpoint_step": case["endpoint_step"],
        "endpoint_time_ps": case["endpoint_time_ps"],
        "translation_start_ps": case["translation_start_ps"],
        "terminal_travel_nm": case["travel_nm"],
        **strain_stats,
        "tool_removed": True,
        "cutoff_A": 3.375,
        "affine_mapping": "off",
        "invalid_strain_policy": "gray in rendering; excluded from reported visual diagnostics; never replaced by zero",
        "color_rule": {
            "colormap": FULL_TRACK_CMAP,
            "normalization": "fixed_linear",
            "vmin": FULL_TRACK_STRAIN_RANGE[0],
            "vmax": FULL_TRACK_STRAIN_RANGE[1],
            "values_above_vmax": "saturated",
        },
        "camera": camera,
        "output_path": str(output),
        "output_sha256": require_file(output),
        "scope": (
            "terminal full-track local-shear visualization from the actual final configuration and its launch-matched relaxed reference; "
            "terminal travel is condition-specific and outside the matched 4--16 nm endpoint analysis. Qualitative context only: "
            "not a matched numerical comparison, time-resolved STZ pathway, interfacial metric, or force mechanism"
        ),
    }


def render_full_track_colorbar() -> dict:
    """Render the shared fixed-scale legend for Figures 3 and 4."""
    colorbar_pdf = ASSET_DIR / "fulltrack_shear_colorbar.pdf"
    colorbar_png = ASSET_DIR / "fulltrack_shear_colorbar.png"
    figure, axis = plt.subplots(figsize=(4.8, 0.5))
    ColorbarBase(
        axis,
        cmap=plt.get_cmap(FULL_TRACK_CMAP),
        norm=Normalize(*FULL_TRACK_STRAIN_RANGE),
        orientation="horizontal",
    )
    axis.set_xlabel("Local shear strain; gray = unresolved", labelpad=2)
    figure.savefig(colorbar_pdf, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(colorbar_png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return {
        "asset_id": "fulltrack_shear_colorbar",
        "kind": "terminal_fulltrack_local_shear_colorbar",
        "source_path": str(CONFIG),
        "source_sha256": require_file(CONFIG),
        "output_path": str(colorbar_pdf),
        "output_sha256": require_file(colorbar_pdf),
        "png_path": str(colorbar_png),
        "png_sha256": require_file(colorbar_png),
        "color_rule": {
            "colormap": FULL_TRACK_CMAP,
            "normalization": "fixed_linear",
            "vmin": FULL_TRACK_STRAIN_RANGE[0],
            "vmax": FULL_TRACK_STRAIN_RANGE[1],
        },
        "scope": "shared display legend for the full-track terminal local-shear OVITO renderings",
    }


def render_full_track_fields() -> list[dict]:
    """Render six approved terminal configurations in whole-track 3D, side and top views."""
    cases = [full_track_case_data(*case) for case in ACTIVITY_CASES]
    expected_pairs = [
        ("Baseline", 1), ("A10B05", 1), ("A10B20", 1),
        ("A25B10", 2), ("A25B20", 2), ("A50B20", 2),
    ]
    if [(case["condition"], case["realization"]) for case in cases] != expected_pairs:
        raise ValueError("The approved full-track condition set changed")
    records = []
    for case in cases:
        reference_path, reference_sha256 = full_track_reference(case)
        pipeline, strain_stats = full_track_pipeline(case, reference_path)
        records.extend(
            render_full_track_view(case, pipeline, strain_stats, reference_path, reference_sha256, view)
            for view in FULL_TRACK_VIEW_ORDER
        )
    records.append(render_full_track_colorbar())
    return records


def render_final_surface(run_id: str, condition: str, realization: int) -> dict:
    """Render one approved final configuration with a common species palette."""
    inventory = {
        row["run_id"]: row for row in csv.DictReader((P4 / "data/run_inventory.csv").open())
    }
    if run_id not in inventory:
        raise KeyError(f"No run inventory record for {run_id}")
    row = inventory[run_id]
    if (row["condition"], int(row["realization"])) != (condition, realization):
        raise ValueError(f"Inventory mismatch for {run_id}")
    if row["analysis_eligibility"] == "selected endpoint intervals validated":
        display_role = "analysis_eligible"
        scope = (
            "qualitative final-surface context outside the 4--16 nm activity window; "
            "not a Dmin field, STZ trajectory, post-scratch interface metric, or force mechanism"
        )
    elif run_id == "S4_R01" and row["analysis_eligibility"] == "missing scratch-window frames":
        display_role = "display_only_missing_activity_window"
        scope = (
            "qualitative R01 final-surface context retained solely for six-condition visual coverage; "
            "no saved 4--16 nm frames exist, so it is excluded from activity, prediction, replication "
            "counts and mechanism claims"
        )
    else:
        raise ValueError(f"Unapproved final-surface rendering scope for {run_id}")
    run_root = Path(row["source_path"])
    source = run_root / "metadata/final.data"
    source_sha256 = require_file(source)
    step = endpoint_step(source)
    output = ASSET_DIR / f"fig3_final_{run_id}_ovito.png"

    pipeline = import_file(str(source), atom_style="atomic")

    def morphology(frame, data):
        positions = np.asarray(data.particles.positions)
        particle_types = np.asarray(data.particles["Particle Type"])
        # The rigid diamond is removed for topology readability, exactly as in
        # Paper 3.  The render is a material-surface context panel, not a tool
        # contact/adhesion visualization.
        excluded = (
            (particle_types == 5)
            | (positions[:, 0] < SURFACE_X_RANGE_A[0])
            | (positions[:, 0] > SURFACE_X_RANGE_A[1])
            | (positions[:, 2] < SURFACE_Z_MIN_A)
        )
        data.particles_.delete_elements(excluded)
        remaining = np.asarray(data.particles["Particle Type"])
        if not set(remaining.astype(int)) <= {1, 2, 3, 4}:
            raise ValueError(f"Unexpected type mapping after tool removal for {run_id}")
        colors = np.where(
            np.isin(remaining, (1, 3))[:, None],
            FINAL_CU_COLOR,
            FINAL_TA_COLOR,
        )
        data.particles_.create_property("Color", data=colors)
        data.particles.vis.radius = 1.35
        data.cell_.vis.enabled = False

    pipeline.modifiers.append(morphology)
    pipeline.add_to_scene()
    try:
        surface_viewport().render_image(
            filename=str(output),
            size=SURFACE_SIZE,
            background=(1.0, 1.0, 1.0),
            renderer=renderer(),
        )
    finally:
        pipeline.remove_from_scene()

    return {
        "asset_id": f"final_{run_id}",
        "kind": "final_surface_morphology",
        "run_id": run_id,
        "condition": condition,
        "realization": realization,
        "analysis_eligibility": row["analysis_eligibility"],
        "display_role": display_role,
        "source_path": str(source),
        "source_sha256": source_sha256,
        "endpoint_step": step,
        "output_path": str(output),
        "output_sha256": require_file(output),
        "tool_removed": True,
        "atom_color_rule": "types 1/3 Cu teal; types 2/4 Ta magenta",
        "camera": {
            "projection": "orthographic",
            "direction": SURFACE_DIRECTION.tolist(),
            "target_A": SURFACE_TARGET_A.tolist(),
            "fov_A": SURFACE_FOV_A,
            "size_px": list(SURFACE_SIZE),
            "x_range_A": list(SURFACE_X_RANGE_A),
            "z_min_A": SURFACE_Z_MIN_A,
        },
        "scope": scope,
    }


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    records = [render_initial(), *render_initial_compositions(), *render_full_track_fields()]

    payload = {
        "analysis_version": "paper4-ovito-adapter-v8",
        "based_on": "OVITO/Tachyon rendering of relaxed structure, initial local composition, and full-track terminal local-shear fields",
        "renderer_script": str(Path(__file__).resolve()),
        "renderer_script_sha256": sha(__file__),
        "ovito_version": list(ovito.version),
        "rendering_scope": "ray-traced visual assets only; full-track terminal fields use OVITO local shear strain against a launch-matched relaxed reference, while numerical endpoint non-affinity analysis remains custom NumPy/SciPy",
        "full_track_camera_contract": {
            "projection": "orthographic",
            "oblique_direction": SURFACE_DIRECTION.tolist(),
            "oblique_target_A": SURFACE_TARGET_A.tolist(),
            "oblique_fov_A": SURFACE_FOV_A,
            "oblique_size_px": list(SURFACE_SIZE),
            "side_direction": [0.0, 1.0, 0.0],
            "side_camera_pos_A": list(FULL_TRACK_SIDE_CAMERA_POS_A),
            "side_fov_A": FULL_TRACK_SIDE_FOV_A,
            "side_size_px": list(FULL_TRACK_SIDE_SIZE),
            "top_direction": [0.0, 0.0, -1.0],
            "top_camera_pos_A": list(FULL_TRACK_TOP_CAMERA_POS_A),
            "top_fov_A": FULL_TRACK_TOP_FOV_A,
            "top_size_px": list(FULL_TRACK_TOP_SIZE),
            "tool_removed": True,
            "same_geometry_and_magnification_within_each_view": True,
        },
        "full_track_field_cases": [
            {"run_id": run_id, "condition": condition, "realization": realization,
             "figure_group": ACTIVITY_FIGURE_GROUP_BY_CASE[(condition, realization)],
             "views": list(FULL_TRACK_VIEW_ORDER)}
            for run_id, condition, realization in ACTIVITY_CASES
        ],
        "records": records,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "ovito_version": payload["ovito_version"],
        "assets": [record["output_path"] for record in records],
        "manifest": str(MANIFEST_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
