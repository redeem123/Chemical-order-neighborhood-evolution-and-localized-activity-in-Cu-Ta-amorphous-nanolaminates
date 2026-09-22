# Chemical order, neighborhood evolution and localized activity in Cu–Ta amorphous nanolaminates during vibration-assisted nanoscratching

Analysis and bounded numerical reconstruction for the manuscript by **Viet-Anh
Ngo, Thanh-Huan Nguyen and Duc-Toan Nguyen**. This repository contains code only;
the versioned supporting dataset is deposited separately. It does not contain
third-party papers, confidential reviews, full MD trajectories or credentials.

## Reproduce the supporting numerical results

Use Python 3.10 or later in a separate environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python extract_data.py paper4-evidence.h5 --output data
python reproduce.py --data data --output output/verification --stage all
```

Download `paper4-evidence.h5` and its README/manifest from the data deposit once
its record is published. The lossless HDF5 container preserves the original
CSV, JSON, NPZ and log bytes, deduplicated by SHA-256. It is **not** a ZIP and is
not a replacement scientific estimator. Extraction verifies every file and
refuses overwriting an existing directory. About 0.4 GB of extracted data plus
the container and generated checks is needed. The supplied manifests, not
rounded manuscript tables, are the numerical inputs.

`reproduce.py` runs eight bounded stages. Each stage uses one numerical thread,
reads data from `--data`, and writes only to a new `--output` directory. The
`verification.json` record distinguishes each success or failure. No stage
starts MD, contacts a cluster, downloads data or uploads results.

| Stage | Reconstructed or checked scope |
|---|---|
| `core` | Checksums, 405 graphs, paired chemical statistics, both primary width estimators, 270 width-definition/station summaries, and 90 force windows |
| `labels` | 60 independently seeded batches of full-workpiece assignments; stratum species conservation |
| `profiles` | Crossing/support rules on primary and extended packets, including influential/excluded tiles |
| `context` | Center counts, affine-subtracted kinetic contrasts and the explicitly frozen late-state subset |
| `accounting` | Per-arm then paired edge closure, reference/population reconciliation, force quadrature and 36 fixed-design prediction transfers |
| `extended` | Spatial-stratum geometry, two depth definitions on the same active IDs, raw force logs/restart handling, and physical-control windows |
| `unloading` | Thirty selected-state graphs, retention, activity, interface widths and paired chemical assignments for six withdrawal/hold branches |
| `entry` | Original loading logs, first reactions and ramp samples; cross-check of archived geometry receipts |

Run one stage with `--stage core` or any stage name in the table. `--stage hashes`
checks dataset identity only. An existing output folder is refused to preserve
previous checks.

## Boundaries and adverse evidence

These are author-run computational consistency checks, not independent reviewer
execution or validation of the physical model. The dataset retains negative and
preparation-dependent chemical results, finite-temperature/finite-hold limits,
the shared precursor/tiling caveat and the R01 pilot's overlap with later test
directions. It does not establish irreversible STZs or permanent mixing.

The historical ADH050_R03 control and recovered R03 vibration withdrawal branch
retain their original nonzero launcher receipts and separate validation records.
Clean reruns conducted separately are not silently substituted into this release.
The selected raw windows and reconstruction scope are documented in the dataset.

Full-time trajectory processing, complete RDF/PTM regeneration, full-neighbor
completeness checks against absent trajectories, and every original rendering
are outside the portable reconstruction claim. Archived source code is provided
in `source_reference/` to document those author workflows; it requires the
original project layout and raw inputs. **Do not run those historical modules
as generic launchers or import them as a test suite.** The supported entry point
is `reproduce.py`. OVITO 3.15.5 was used for the atomistic rendering workflow.

## Source and provenance

- `checks/`: portable numerical reconstruction routines, preserving their
  original dated filenames for traceability.
- `source_reference/`: scientific extraction, analysis and rendering source;
  original workspace dependencies are not bundled.
- `provenance/source_manifest.json`: SHA-256 of the source files before the
  documented data-path-only adaptation of two portable checks.
- `checks/rebuild_documents_reviewer2_20260921.py`: optional XeLaTeX/BibTeX
  compilation of the separately supplied manuscript-source folder, not an MD
  analysis stage. Times New Roman and the supplied journal class are required.

The full paper title is retained here and in `CITATION.cff`; the repository slug
uses its concise title prefix. Cite the exact dataset version and code commit
used. No acceptance, journal submission or independent verification is implied
by this repository's existence.

## License

The authors release the code under the MIT license. The supporting dataset has
a separate CC BY 4.0 license. Installed third-party dependencies retain their
own licenses; their source code is not redistributed here.
