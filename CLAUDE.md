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

# one-time per clone: wire up git hooks (uv sync does NOT do this - .git/hooks/ is
# local to each clone and untracked, so nothing triggers it automatically)
uv run pre-commit install
uv run pre-commit install --hook-type pre-push

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

Ruff is configured (`[tool.ruff]`/`[tool.ruff.lint]` in `pyproject.toml`, `target-version = "py312"`,
`select = ["E", "F", "W", "I", "UP", "B", "RUF"]`, `ignore = ["E501"]` - line-length not enforced since
`ruff-format` already wraps real code sensibly and manual E501 fixups on long docstrings/log strings aren't
worth the friction) and wired into a pre-commit hook (`.pre-commit-config.yaml`): `ruff-check --fix` and
`ruff-format` run automatically on every commit (`git commit`). A local `pytest` hook (added 2026-07-30,
`language: system` since it just invokes `uv run pytest` in the existing environment rather than a separate
pip-installable hook package) runs the full suite, but only at the `pre-push` stage (`git push`), not on every
commit - deliberately split from the fast ruff checks since there's no CI in this repo (`.github/workflows/`
doesn't exist) and the full suite takes ~7s, too slow to tax every single local commit but still worth
gating before code leaves the machine. Requires `uv run pre-commit install --hook-type pre-push` once
(in addition to the default `pre-commit install`) for the `pre-push` stage to actually be installed.
An `nbstripout` hook (added 2026-09-10, `repo: https://github.com/kynan/nbstripout`) runs on every commit
too, stripping `outputs` + `execution_count` from every `.ipynb` under `notebooks/` — the committed-output
policy for the notebook consolidation (`todo.txt` item 9) is "strip everything", so notebook diffs stay
code-only and don't carry embedded image/itables-JSON blobs. `ruff-check`/`ruff-format` also lint the
notebooks (ruff globs `*.ipynb` by default). Like the ruff hooks, both modify files in place, so the first
commit after they fire aborts with "files were modified" — re-`git add` and re-commit.

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
  the experimental condition, e.g. `Treated-DrugA_FOV3` -> `Treated-DrugA`). Gotcha (2026-09-10): a file like
  `None_0.nd2` parses to the condition `"None"` — correct per the regex, but `"None"` (also `"NA"`, `"NaN"`,
  `"null"`, `"inf"`, `""`) is a default pandas NA token, so it survives in-run (all in-memory `pd.concat`) but
  reads back from `output/tables/*.csv` as `NaN`. Two-part mitigation: (1) README documents that conditions
  must not be named with those tokens; (2) notebooks read pipeline tables with
  `pd.read_csv(path, keep_default_na=False, na_values=[""])` — the exact inverse of `to_csv`'s `na_rep=""`,
  so numeric NaN still round-trips while string columns don't get NA-coerced. `parse_condition_from_name`
  itself is left alone (a pure string fn shouldn't know about pandas). `ModelBundle`, a dataclass that
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
  objects (`cellpose.utils.remove_edge_masks`). `block_reduce` zero-pads to the next multiple of `bin_factor`
  when a dimension isn't evenly divisible, so `segment_2d`/`segment_3d` capture the pre-bin spatial shape
  (`std_proj.shape` / `min_substracted.shape[-2:]`) and crop the upscaled mask back to it (`masks_resized[:h, :w]`)
  before edge-mask removal — otherwise the mask ends up oversized and misaligned with the unbinned spot
  coordinates downstream (`todo.txt` item 6, fixed 2026-09-09; a `logger.debug` fires when padding happens).
  `assign_spots_to_mask` does nearest-voxel label lookup and raises `DimensionMismatchError` on a
  coordinate/mask dimensionality mismatch.
- `obejct_measurement.py` (filename typo, intentional/existing — don't "fix" it without also updating the import
  in `run_pipeline.py`) — turns masks + spot labels into a tidy per-object DataFrame via `skimage.regionprops_table`.
  2D and 3D modes populate disjoint sets of columns (e.g. `Volume_um3` is NaN in 2D, `Area_um2`/`Eccentricity` are
  NaN in 3D) rather than using separate schemas — this is intentional, keep both modes on one flat column set.
- `qc_figures.py` / `qc_panels.py` — all matplotlib/seaborn plotting, split on the public/private boundary
  (2026-09-04, `todo.txt` item 7). `qc_figures.py` is the public API: the three figure builders, one per level
  of aggregation — `make_qc_figure` (per scene), `make_scene_summary_figure` (per condition/file),
  `make_run_summary_figure` (whole run). `run_pipeline.py` imports only from here; the split was a pure internal
  refactor, no caller changed. `qc_panels.py` holds everything private that `make_qc_figure` composes: `SpotData`
  and `ImageData` (dataclasses that derive pixel/micron coordinate arrays from raw detector/image output —
  `__post_init__` does the unit conversion, treat them as read-only views, not places to add pipeline logic),
  `_flow_to_rgb`, and the six `_panel_*` helpers. `make_scene_summary_figure`/`make_run_summary_figure` draw
  their panels inline (no `_panel_*` extraction) — see `todo.txt` item 1 for why.

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

`notebooks/`, not part of the package, is mid-consolidation to three (`todo.txt` item 9, in progress
2026-09-10):

- `pipeline_run.ipynb` — thin Jupyter front-end to `run_pipeline`, outputs shown inline (config table,
  run-summary figure, interactive `ipyfilechooser` browser over `output/`). Equivalent to
  `uv run spot-detector configs/config.yml`. **DONE** (committed 2026-09-10).
- `pipeline_tuning.ipynb` (renamed from `pipeline_validation.ipynb`) — single-scene: run the stages,
  inspect with `stackview` / optional napari, sweep `bin_factor`/`prob_thresh` to pick config values.
  Deliberately mirrors `run_pipeline._process_scene` with no divergence. Still being reworked.
- `pipeline_data_analysis.ipynb` (renamed from `analysis.ipynb`) — post-run analysis of
  `output/tables/*.csv`. Functionally done as of 2026-09-11 (markdown/docstring polish pending, see
  `todo.txt` item 9(c)): table picker (falls back to `_run_objects_{mode}.csv`, reads with
  `keep_default_na=False, na_values=[""]` — see the `utils.py` bullet above — then drops all-NaN columns),
  an optional `Condition -> metadata` mapping cell (gated behind a flag, off by default), `describe()` +
  an interactive per-group stats widget, `pygwalker` for free-form exploration, a PNG browser, and a
  general `plot_metric_by_group(df, metric, group, hue=None, scale="log", category_axis="auto",
  log_offset=None, ...)` template for building presentation-ready figures (auto-picks horizontal vs.
  vertical orientation from label length; `log_offset` loudly shifts non-positive values before
  log-scaling instead of matplotlib silently dropping them off the axis — design rationale in
  `LEARNING_NOTES.md` 2026-09-11).

The old `spot_detection_pipeline.ipynb` (a pre-package reimplementation) and `pipeline_param_optimalisation.ipynb`
(its param-sweep intent folded into `pipeline_tuning.ipynb`) have both been deleted.

`src/bash_scripts/` and `workflow/` (Snakemake) are an in-progress orchestration layer (repo setup, HPC conda/module
loading, raw-data staging to/from Dropbox, result upload) — several scripts are stubs or contain scratch notes
rather than working end-to-end automation; don't assume they run as-is.

## Testing conventions

Tests live in `tests/`, one file per source module (`test_segmentation.py` covers `segmentation_detection.py`,
`test_object_measurement.py` covers `obejct_measurement.py`, etc.), 192 tests collected as of 2026-09-09.
`test_segmentation.py` (13): its `mock_cellpose_2d`/`mock_cellpose_3d` fixtures use `model.eval.side_effect`
(a `fake_eval(img, **kwargs)` closure) rather than `return_value`, so the fake mask matches its input's shape
the way real Cellpose does — required for the non-divisible `bin_factor` tests (a fixed-size fake mask only
worked because `40 / 4 == 10` exactly). The `test_logs_when_padding_needed` / `test_no_log_when_divisible`
pair uses `caplog.set_level(logging.DEBUG, logger="spot_detector.segmentation_detection")` — mandatory, since
`caplog` captures at WARNING+ by default and the module logger inherits WARNING from root, so the
`logger.debug` padding message is filtered before any handler without it.
The qc-plotting split (`todo.txt` item 7) is mirrored in the tests: `test_qc_panels.py` covers `qc_panels.py`
(all 6 `_panel_*` helpers plus `SpotData`/`ImageData`/`_flow_to_rgb` — 62 tests, done), `test_qc_figures.py`
covers the three figure builders and is **DONE as of 2026-09-08** (9 tests, `todo.txt` item 4 closed).
`TestMakeQCFigure` (4): `test_smoke_2d`/`test_smoke_3d` render a real figure to a `tmp_path` PNG with all
6 panels running their happy path — asserted via `out_path.exists()` plus `"failed" not in caplog.text.lower()`,
since every panel's `except Exception` fallback logs a `logger.warning` containing "failed";
`test_dispatches_to_all_six_panels` `mocker.patch.multiple`s the 6 panels + spies `plt.subplots` to check
each gets the right `axes_flat[i]`; `test_closes_figure` asserts `plt.close` called once.
`TestMakeSceneSummaryFigure` (2) + `TestMakeRunSummaryFigure` (3): mostly smoke — a small real DataFrame
with the mode's expected columns (`Scene`/`Condition`, `Spot_Count`, `Area_um2`|`Volume_um3`,
`Spot_Density_per_um2`|`_per_um3`), assert the file is written. The `mode` branch is guarded *by omission* —
the 3D df deliberately lacks the 2D columns, so a broken `size_metric`/`norm_metric` selection raises a
seaborn `ValueError` instead of silently plotting the wrong column. `test_3d_selects_volume_columns` is the
one non-smoke: a production-shaped df (all 4 metric columns present) + `mocker.patch(...sns.scatterplot)`,
asserting Panel D is called with `x="Volume_um3"` — the case a plain smoke test can't catch because in
production the wrong column exists (all-NaN) rather than being absent. The summary builders have **no
per-panel try/except** (resilience is at the `run_pipeline.py` call site, `todo.txt` item 1), so a bad
column raises and fails the test loudly — no `caplog` guard needed there.
`test_qc_panels.py` is also the first test file to use nested test classes (`TestPanelZDistribution`'s
`TestIs3dTrue` / `TestIs3dFalse`), for a panel with two independent `is_3d` branches. Its per-class fixtures
(`valid_segmentation`, `valid_spot_detection`, `valid_ecdf`, `valid_spotmap`, `valid_zdist_2d`, plus the
original `valid_3d`; `todo.txt` item 8, done 2026-09-04) follow one convention: each returns a dict of valid
default args, and individual tests mutate one or two keys in place before calling
`_panel_x(ax=ax, **valid_y)` — a panel argument a given code path never reads is set to `...` (Ellipsis, not
`None` — `None` is a real value elsewhere in this module, e.g. `masks_2d`/`z_um`, so it would misleadingly
suggest the code branches on it). Tests
mock heavy ML dependencies (Cellpose/Spotiflow model calls) via `pytest-mock` rather than loading real models
or real microscopy files — keep new tests fast and offline.

`[tool.pytest.ini_options]` in `pyproject.toml` has a `filterwarnings` entry suppressing one
`MatplotlibDeprecationWarning` (`vert:` → `orientation:`, mpl 3.11): seaborn 0.13.2's `boxplot` passes the
deprecated kwarg to `Axes.bxp` — not our code, fixed upstream but unreleased, surfaces once
`test_qc_figures.py` started exercising real seaborn (all `_panel_*` tests mock `ax`). Scoped to that exact
message+category so a `vert`-unrelated mpl deprecation still shows. Drop when seaborn > 0.13.2 ships.

`conftest.py` holds two shared factory fixtures, deliberately kept minimal: `make_config(**overrides)` builds
a real, validated `PipelineConfig` backed by real tmp_path files/dirs (so pydantic's `FilePath`/`DirectoryPath`
validators actually run), merging `**overrides` into a valid base dict — since `PipelineConfig` is frozen,
tests needing a different value (e.g. `mode.do_3d`) must call `make_config(mode={"do_3d": True})` to get a
new instance rather than mutate a shared one. The merge is **shallow** (`{**base, **overrides}`): passing a
nested section (e.g. `detection={"prob_thresh": 0.4}`) *replaces the entire section*, dropping its other
keys — so `make_config(detection={"prob_thresh": 0.4})` fails validation (no `spotiflow_model_path`) unless
you also pass `"use_default_model": True` or the full section. `make_config(mode={"do_3d": True})` only
works because `mode` has a single key. `make_stack(shape)` returns a random `float32` array of the
given shape. Both were hoisted here specifically because their *implementation* (not just fixture name) was
identical across files. Several other same-named fixtures across test files (`base_params` in
`test_detection.py` vs `test_object_measurement.py`) look like duplicates but aren't — different call
signatures for different functions under test — and were deliberately left local rather than merged; check a
fixture's body, not just its name, before assuming it's safe to hoist into `conftest.py`.
