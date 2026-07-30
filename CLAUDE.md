# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A microscopy image analysis pipeline for Nikon-acquired images: segments "BAGs" (cell-like structures) in
brightfield/segmentation channel using Cellpose-SAM, detects diffraction-limited spots in a second channel using
Spotiflow, assigns spots to their containing BAG, and measures per-object morphology + spot counts/density. Runs
in either 2D (single projection) or 3D (pseudo-3D stitched z-stack) mode, controlled by `mode.do_3d` in the config.

The installable package is `spot-detector`, living under `src/spot_detector/`.

## Commands

Dependency management is via `uv` (recently migrated from conda — some conda/module-load artifacts still exist
in `src/bash_scripts/` and are legacy/not the current workflow).

```bash
# install/sync environment
uv sync

# run the pipeline against a config file
uv run spot-detector configs/config.yml
# equivalently
uv run python -m spot_detector.cli configs/config.yml

# run all tests
uv run pytest

# run a single test file / test
uv run pytest tests/test_segmentation.py
uv run pytest tests/test_segmentation.py::TestSegment3D::test_output_shape -v
```

There is no configured linter (no ruff/flake8 config in `pyproject.toml`); `.ruff_cache/` is present locally but
not wired into any command in this repo.

## Architecture

Pipeline entry point is `cli.py:main`, which loads `configs/config.yml` (via `config.py:load_config`) and calls
`run_pipeline.py:run_pipeline`. Processing is a strict fan-out:

```
run_pipeline (one call)
  -> _process_file (one per file in raw_data_dir)
       -> _process_scene (one per scene/FOV within a multi-scene file, via BioImage)
            1. segmentation_detection.segment_2d / segment_3d   (Cellpose-SAM)
            2. segmentation_detection.detect_spots_spotiflow    (Spotiflow)
            3. segmentation_detection.assign_spots_to_mask      (nearest-voxel lookup)
            4. obejct_measurement.measure_objects               (regionprops_table -> per-object DataFrame)
            5. qc_figures.make_qc_figure                        (per-scene multi-panel QC PNG)
       -> concatenates scene DataFrames, writes `{condition}_objects_{mode}.csv`,
          calls qc_figures.make_scene_summary_figure
  -> concatenates all file DataFrames, writes `_run_objects_{mode}.csv`,
     calls qc_figures.make_run_summary_figure
```

Key modules under `src/spot_detector/`:

- `config.py` — YAML loader validated via pydantic. `load_config` returns a `PipelineConfig` (not a plain
  dict), built from nested `BaseModel`s (`ModeConfig`, `PathsConfig`, `ChannelConfig`, `SegmentationConfig`,
  `DetectionConfig`), all frozen (`model_config = ConfigDict(frozen=True)`) since nothing downstream should
  mutate config after load. Pydantic migration (started 2026-07-28, full history/decisions in `todo.txt`
  item 5) is DONE as of 2026-07-29: every call site (`cli.py`, `run_pipeline.py`, `utils.py`, `qc_figures.py`)
  uses real attribute access (`config.section.key`), no dict-style `config["section"]["key"]` lookups remain
  anywhere in `src/`. `cellpose_model_path`/`spotiflow_model_path` (singular, renamed) live in
  `SegmentationConfig`/`DetectionConfig` respectively (co-located with each model's other settings; typed
  `FilePath`/`DirectoryPath` respectively — Cellpose-SAM's checkpoint is a single file, Spotiflow's is a
  folder), each paired with a `use_default_model: bool = False` flag and a local `@model_validator`
  requiring the path to be set unless the flag opts into a pretrained default. The "flag wins" precedence
  logic is DONE as of 2026-07-30 (see `todo.txt` item 5): `ModelBundle`'s `_load_cellpose`/
  `_load_spotiflow_from_config` both check `use_default_model` explicitly before attempting the configured
  path — if `True`, the custom path is skipped entirely (never attempted, not just allowed to fail) and a
  pretrained default loads instead, with a `logger.warning` if a path was configured anyway (so an ignored
  path is never silent).
- `utils.py` — `parse_condition_from_name` (strips a trailing `_<token><digits>` suffix from filenames to derive
  the experimental condition, e.g. `Treated-DrugA_FOV3` -> `Treated-DrugA`), and `ModelBundle`, a dataclass that
  loads + validates both models together. `ModelBundle.load(config)` is the only way to construct it — takes
  just the `PipelineConfig`, no separate `do_3d` argument (dropped once `config.mode.do_3d` was available
  everywhere internally). Spotiflow loading has a fallback chain: try the custom model path from config -> if
  load fails or the model's dimensionality (`model.config.is_3d`) doesn't match the pipeline's `do_3d` mode,
  fall back to a pretrained model (`synth_complex` for 2D, `smfish_3d` for 3D) - this fallback is now reached
  two ways: a caught exception from a genuinely broken custom path, or `use_default_model=True` skipping the
  custom path attempt outright (distinct code paths, both tested).
- `segmentation_detection.py` — the actual CV/ML calls. 2D segmentation runs Cellpose on a stdev-projection of the
  z-stack; 3D segmentation runs Cellpose per-plane on a min-subtracted stack and stitches with `stitch_threshold`.
  Both downscale by `segmentation.bin_factor` before inference and upscale masks back, then strip edge-touching
  objects (`cellpose.utils.remove_edge_masks`). `assign_spots_to_mask` does nearest-voxel label lookup and raises
  `ValueError` on a coordinate/mask dimensionality mismatch.
- `obejct_measurement.py` (filename typo, intentional/existing — don't "fix" it without also updating the import
  in `run_pipeline.py`) — turns masks + spot labels into a tidy per-object DataFrame via `skimage.regionprops_table`.
  2D and 3D modes populate disjoint sets of columns (e.g. `Volume_um3` is NaN in 2D, `Area_um2`/`Eccentricity` are
  NaN in 3D) rather than using separate schemas — this is intentional, keep both modes on one flat column set.
- `qc_figures.py` — all matplotlib/seaborn plotting. `SpotData` and `ImageData` are dataclasses that derive
  pixel/micron coordinate arrays from raw detector/image output (`__post_init__` does the unit conversion — treat
  them as read-only views, not places to add pipeline logic). Three figure builders correspond to the three levels
  of aggregation: `make_qc_figure` (per scene), `make_scene_summary_figure` (per condition/file), and
  `make_run_summary_figure` (whole run).

Config schema (`configs/config.yml`, validated by `config.py`'s `PipelineConfig`): `mode.do_3d`,
`paths.{raw_data_dir,out_dir}`, `channels.{segmentation_image,spot_image}` (channel indices into the raw
image — `channels.misc` was dropped, confirmed zero references in `src/`),
`segmentation.{use_default_model,cellpose_model_path,use_gpu,bin_factor,stitch_threshold}`,
`detection.{use_default_model,spotiflow_model_path,prob_thresh,min_distance}`. `raw_data_dir` and
`spotiflow_model_path` are pydantic `DirectoryPath`; `cellpose_model_path` is `FilePath` (Cellpose-SAM's
checkpoint is a single ~1.2GB file, not a folder, unlike Spotiflow's) — both fail fast at config-load time
if the path doesn't exist or is the wrong kind, instead of failing confusingly deep inside
`ModelBundle.load()`/`BioImage()` later. `out_dir` is a plain `Path` since the pipeline creates it via
`mkdir`. Both model paths are `Optional[...] = None`, required unless their section's `use_default_model`
flag is `true` (enforced by a local `@model_validator` in each of `SegmentationConfig`/`DetectionConfig`).
Model paths point outside the repo (`../_pipeline_assets/...`) — they're expected to exist in a sibling
directory on the machine running the pipeline, not to be committed here.

Output layout: `output/tables/{condition}_objects_{mode}.csv` and `output/tables/_run_objects_{mode}.csv` (rows
are one segmented object each), plus matching PNGs under `output/figures/`.

Input images are read via `bioio.BioImage`, which abstracts over Nikon `.nd2` and other formats (`.czi`, `.lif`,
OME-TIFF) — the specific `bioio-*` plugin used depends on file extension, handled transparently by `bioio`.

`notebooks/` contains exploratory/validation notebooks (`analysis.ipynb`, `pipeline_validation.ipynb`,
`spot_detection_pipeline.ipynb`) used for visualizing pipeline output and comparing detected spots against source
images — useful for understanding expected behavior but not part of the package.

`src/bash_scripts/` and `workflow/` (Snakemake) are an in-progress orchestration layer (repo setup, HPC conda/module
loading, raw-data staging to/from Dropbox, result upload) — several scripts are stubs or contain scratch notes
rather than working end-to-end automation; don't assume they run as-is.

## Testing conventions

Tests live in `tests/`, one file per source module (`test_segmentation.py` covers `segmentation_detection.py`,
`test_object_measurement.py` covers `obejct_measurement.py`, etc.), 113 tests total as of 2026-07-30. Tests
mock heavy ML dependencies (Cellpose/Spotiflow model calls) via `pytest-mock` rather than loading real models
or real microscopy files — keep new tests fast and offline.

`conftest.py` holds two shared factory fixtures, deliberately kept minimal: `make_config(**overrides)` builds
a real, validated `PipelineConfig` backed by real tmp_path files/dirs (so pydantic's `FilePath`/`DirectoryPath`
validators actually run), merging `**overrides` into a valid base dict — since `PipelineConfig` is frozen,
tests needing a different value (e.g. `mode.do_3d`) must call `make_config(mode={"do_3d": True})` to get a
new instance rather than mutate a shared one. `make_stack(shape)` returns a random `float32` array of the
given shape. Both were hoisted here specifically because their *implementation* (not just fixture name) was
identical across files. Several other same-named fixtures across test files (`base_params` in
`test_detection.py` vs `test_object_measurement.py`) look like duplicates but aren't — different call
signatures for different functions under test — and were deliberately left local rather than merged; check a
fixture's body, not just its name, before assuming it's safe to hoist into `conftest.py`.
