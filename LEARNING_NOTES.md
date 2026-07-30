# Learning Notes

Running log of things I'm learning about this codebase and about software engineering practices in general,
kept as I work through `nikon_cellpose_bags_spots` with Claude Code. Newest entries at the top.

---

## 2026-07-30 — "Flag wins" use_default_model precedence implemented, config migration fully closed

**Session goal:** implement the last open piece of `todo.txt` item 5 - the "flag wins" precedence logic
decided 2026-07-28 but never coded. I wrote the implementation and tests myself this session; Claude
reviewed each increment by actually running the tests, not just reading the diff.

**`_load_cellpose` implemented first.** Flattened from a 3-level nested `if/else` into a guard clause (return
early on the simple "custom model" case, fall through to the default-loading path otherwise) - removes a
duplicated `CellposeModel(gpu=...)` call that existed in both arms of the original nested `else`. Confirmed
via `inspect.signature(models.CellposeModel.__init__)` that omitting `pretrained_model` entirely isn't
undefined behavior - it has its own default (`'cpsam_v2'`), so "don't pass a path" genuinely means "load
Cellpose's own pretrained default," matching the 2026-07-28 design assumption ("Cellpose has its own robust
pretrained default, no fallback code needed on our side").

**`_load_spotiflow_from_config` implemented second, same shape.** Before this session, `use_default_model`
was checked nowhere for Spotiflow either - the existing fallback chain only *looked* like it respected the
flag, because an unset path made `Spotiflow.from_folder("None")` raise, which the pre-existing exception
handler happened to catch. That's a fundamentally different code path from "flag wins": it doesn't fire if a
path IS configured alongside `use_default_model=True` (previously: silently used the configured path anyway,
ignoring the flag entirely). Fixed by adding the same explicit `if not use_default_model: try/except ...`
guard before ever calling `Spotiflow.from_folder`.

- **A mocked test can fail in a way that reveals a real, unrelated bug in the test itself, not the code under
  test.** A new Spotiflow-focused test accidentally called `ModelBundle._load_cellpose` instead of
  `_load_spotiflow` (leftover from copy-pasting the cellpose version as a starting point). Running it didn't
  raise a clean assertion failure - it raised `RuntimeError: mmap can only be used with files saved with
  torch.save(...)`, because `_load_cellpose` then tried to construct a *real* `CellposeModel` against
  `make_config`'s dummy `cellpose_model.ckpt` (just an empty `.touch()`'d file), since the test never mocked
  `CellposeModel` at all. The traceback pointed at PyTorch's serialization internals, nothing like "wrong
  method called" - had to trace it back to the call site to find the actual one-line typo. Lesson: an
  unexpected, deep, unfamiliar-looking traceback from a test is often not a new discovery about the code under
  test - check the test's own call sites first, especially right after a copy-paste.
- **Testing "it was never attempted" vs. "it failed and recovered" are different claims, even when they reach
  the same final call.** Both the exception-driven fallback (already tested) and the new flag-driven fallback
  end up calling `Spotiflow.from_pretrained(fallback)` - so asserting just that would duplicate coverage that
  already existed. The assertion that actually proves flag-wins is `mock_from_folder.assert_not_called()` -
  proving the custom-path attempt never happened at all, which is the one thing that's actually new/different
  about this code path versus the pre-existing exception-based one.
- **When a parametrize axis you already need for one assertion also happens to cover a second, related
  assertion, fold it into the same test rather than writing a new one.** The flag-wins test already
  parametrizes over `configure_path` (True/False) to prove which fallback model gets chosen; that's the exact
  same axis needed to prove the "ignoring path" warning fires only when there was a path to ignore. Considered
  writing a separate `test_flag_triggers_warning_log` test, but that would've just rebuilt the same three
  config permutations to check one more mock call - folded the warning assertion into the existing
  parametrized test instead. Renamed it `test_flag_wins_behavior` (from `test_flag_wins_skips_custom_load_
  entirely`) since it now documents everything that happens when the flag wins, not just the one narrowest
  claim its original name promised.
- **A copy-pasted log message is as easy to miss as a copy-pasted method call.** `_load_spotiflow_from_config`
  originally said `logger.debug("Loading default Cellpose model...")` in its own default-loading branch -
  copy-paste residue from `_load_cellpose`, caught by reading the diff carefully rather than by any test
  (log message text isn't usually asserted on unless a test specifically checks it).

**Result: `todo.txt` item 5 (pydantic config migration) is now fully closed** - schema, call-site rewrite, and
flag-wins precedence logic are all implemented and tested. 113 tests total (up from 107), 6 new
(`TestLoadCellpose.test_flag_precedence`, `TestLoadSpotiflow.test_flag_wins_behavior`, 3 parametrize cases
each).

## 2026-07-29 — Pydantic config migration: call-site rewrite finished, conftest.py streamlined

**Session goal:** finish the call-site rewrite scoped 2026-07-28 (`config["section"]["key"]` ->
`config.section.key` across `cli.py`, `utils.py`, `run_pipeline.py`, `qc_figures.py`) and the matching test
fixture rewrite, then act on the `conftest.py` streamlining scoped the same day. I wrote the code this time;
Claude reviewed each file by actually running the affected tests, not just reading the diff - caught two real
bugs this way (see below), both mechanical slips from the dict->attribute rewrite itself, not design flaws.

**Call-site rewrite: done, verified file-by-file.** Confirmed via `grep -rn "config\["` across
`src/spot_detector/` returning zero hits - every call site now reads `config.section.key`.
`ModelBundle.load()` also dropped a now-redundant separate `do_3d` argument once `config.mode.do_3d` was
available internally everywhere it's needed; `run_pipeline.py`'s call site simplified to match.

- **A stray comma survives a mechanical dict->attribute rewrite and is invisible to every automated check.**
  `qc_figures.py`'s `_panel_ecdf` had `config['detection']['prob_thresh']` rewritten to
  `config.detection.prob_thresh` correctly in one spot, but a second occurrence one line below (inside an
  f-string brace, building the plot legend label) kept a trailing comma from the original dict-subscript
  edit: `f"thresh={config.detection.prob_thresh,}"`. That's syntactically valid Python - the comma makes it
  a 1-tuple, rendering as `"thresh=(0.3,)"` instead of `"thresh=0.3"`. Neither pytest nor a type checker
  flags this; it only shows up by reading the actual rendered figure. General lesson: a search-and-replace
  across similar-looking lines can leave a leftover token that changes *meaning* without breaking syntax -
  worth actually eyeballing the diff around every occurrence, not just trusting the substitution pattern held.
- **A mocked fixture is already active the moment pytest resolves it - calling its return value is a
  different operation entirely.** `test_model_bundle.py` had a `mock_validate_passtrhough` fixture (wraps
  `mocker.patch.object(ModelBundle, "_validate_spotiflow_mode", side_effect=lambda model, config: model)`),
  correctly requested as a fixture in three tests, but each test body then also called it directly:
  `mock_validate_passtrhough()`. That doesn't "activate" anything - the patch already replaced
  `ModelBundle._validate_spotiflow_mode` when the fixture ran. Calling the returned `MagicMock` with zero
  arguments instead *invokes* it, running the `side_effect` lambda with no `model`/`config`, raising
  `TypeError: missing 2 required positional arguments`. Confirmed by actually running the tests (3 failures,
  identical traceback each time) rather than assuming the fixture-injection pattern used correctly elsewhere
  in the same file (`mock_from_pretrained`, never called) meant this one was fine too. Fix was deleting the
  spurious call, not touching the fixture.
- **Reusing a config fixture across a file that also serializes it can break silently when the fixture's
  return type changes.** `test_cli.py` had `config_file.write_text(yaml.dump(full_config_dict))` when the
  fixture returned a plain dict; once the fixture was swapped for `make_config()` (returning a real
  `PipelineConfig`), the same line became `yaml.dump(config)` on a pydantic model. `yaml.dump` doesn't know
  how to serialize a `BaseModel` as config data - it falls back to PyYAML's unsafe `!!python/object`
  representer, producing something `load_config`'s `yaml.safe_load` could never parse back. The test still
  passed, because `load_config` itself was fully mocked (`return_value=config`) - nobody ever read the file's
  actual content. Fixed by replacing the write with `config_file.touch()`, since the file only needed to
  *exist* as a path for the `assert_called_once_with(config_file)` check. Lesson: a test can keep passing
  while quietly writing meaningless output, if nothing downstream ever reads that output back - worth asking
  "is this file's content actually exercised, or just its existence?" before trusting a passing assertion.
- **Test coupling should match the layer being tested, not whatever fixture happens to be lying around.**
  Considered deriving `test_detection.py`'s `base_params` (`prob_thresh`/`min_distance`/`do_3d`) from
  `make_config()` for consistency with the rest of the suite - rejected, because `detect_spots_spotiflow`
  itself takes three plain scalars, not a `PipelineConfig` (`segmentation_detection.py` doesn't import
  `config.py` at all). Threading `PipelineConfig` through would couple a low-level unit test to a schema it
  doesn't depend on, and would weaken the assertions: `base_params` deliberately uses non-default values
  (`0.5`/`10` vs `PipelineConfig`'s `0.3`/`1` defaults) specifically so `assert kwargs["prob_thresh"] == 0.5`
  proves the value actually flowed through, rather than passing coincidentally because both layers happen to
  default to the same number.

**`conftest.py` streamlining: done, matches the 2026-07-28 scoping almost exactly.** `make_config(**overrides)`
(tmp_path-backed factory building a real, validated `PipelineConfig`) is now shared by `test_cli.py`,
`test_run_pipeline.py`, `test_model_bundle.py`, `test_detection.py`. `make_stack(shape)` (parametrized
`np.random.rand(*shape).astype(np.float32)`) replaced the fixed-size `stack_2d`/`stack_3d` fixtures in both
`test_detection.py` and `test_segmentation.py` - also incidentally fixed the stale "40x40" docstring flagged
2026-07-28, since the fixed-size fixtures it was attached to no longer exist. `base_params` in
`test_detection.py` vs `test_object_measurement.py` confirmed still a false-friend name collision (different
call signatures for different functions) and deliberately left local, not merged.

- **A parametrized factory fixture is safe to share across files in a way a fixed-value fixture with the same
  name isn't.** This confirms the 2026-07-28 hypothesis directly: `stack_2d`/`stack_3d` (fixed 20x20 vs 40x40)
  couldn't be merged without one file silently inheriting the other's implicit size assumption, but
  `make_stack(shape)` has no implicit default at all - every caller states its own shape, so there's nothing
  for two files to disagree about. The general rule: hoist fixtures into `conftest.py` when the *behavior* is
  identical (a factory with explicit parameters has no hidden behavior to diverge on); don't hoist merely
  because the *name* matches across files.
- **Frozen pydantic models mean per-test config variation requires a new instance, not a mutated one.**
  Verified directly: `config.mode.do_3d = True` on an already-built `PipelineConfig` raises
  `pydantic.ValidationError` ("Instance is frozen"), and `model_copy(update={"mode": ModeConfig(do_3d=True)})`
  does work (returns a distinct object, original untouched) but skips re-validation on the replaced field -
  fine for a field no validator depends on, but not a safe general substitute for reconstructing via
  `PipelineConfig(**dict)` when the changed field could interact with a `@model_validator`. This is why
  `make_config` reconstructs from a full dict each call rather than mutating or `model_copy`-patching a
  cached instance.

**Formatting unified across all 9 test files** (was requested explicitly, so implemented directly rather than
just advised): every file now uses the same `# =====...` divider style already established in
`test_object_measurement.py`/`test_run_pipeline.py` - a `# Fixtures` divider before locally-defined fixtures
(omitted in files with none), and a divider naming the function/class under test before each test group.
Zero regressions - full suite (107 tests) passes before and after.

### Questions to follow up on

- The "flag wins" `use_default_model` precedence logic (design decided 2026-07-28, see `todo.txt` item 5) is
  still not implemented in `utils.py` - `ModelBundle`/`_load_cellpose`/`_load_spotiflow_from_config` load
  whatever path is configured unconditionally and never check the flag. Next config-related session should
  pick this up before considering the migration fully closed.

## 2026-07-28 — Pydantic config migration: last open question resolved, starting implementation

**Session goal:** pick up the pydantic config work scoped yesterday (see entry below). Recapped the design
against the current codebase (grep-verified the call-site inventory still holds: cli.py 1, run_pipeline.py
10, utils.py 3, qc_figures.py 2) and resolved the one thing that was still genuinely undecided.

- **Resolved:** qc_figures.py's `_panel_ecdf`/`make_qc_figure` take `prob_thresh: float` directly instead
  of the whole config object. Reasoning: once call sites become real attribute access instead of dict
  indexing, there's no reason for a plotting helper to depend on the entire `PipelineConfig` schema just
  to read one number - a narrower signature documents the actual dependency and doesn't couple plotting
  code to the pipeline config shape.
- Confirmed the rest of yesterday's design (schema/types, dropping `channels.misc`, `DirectoryPath` for
  the three model/data paths, build order) still holds with no changes.
- No design doc under `docs/` for this - todo.txt item 5 + this log are the design record, consistent with
  how the rest of this migration has been tracked.

**Implementation started - I'm writing the code this time, Claude advises + keeps todo.txt/CLAUDE.md/
LEARNING_NOTES.md current instead of writing snippets (see "Concepts learned" below).**

- `bin_factor` bug found and deferred: while adding a `Field(gt=0)` bound to `SegmentationConfig.bin_factor`,
  asked whether it needs to evenly divide the image shape. Confirmed yes, and worse than expected -
  `skimage.measure.block_reduce` silently zero-pads to the next multiple of `bin_factor` instead of erroring,
  and the later `masks.repeat(factor, ...)` upscale lands on that padded size, not the original image shape.
  No crash anywhere - `assign_spots_to_mask` still indexes in-bounds against the oversized mask, just wrong.
  Can't be caught by `PipelineConfig`/`SegmentationConfig` validation since it depends on each image's shape,
  which isn't known until `BioImage` opens the file. Logged as a new `todo.txt` item 6 (segmentation
  correctness), deliberately not fixed now to stay scoped to the config migration.
- All five nested config models frozen (`ConfigDict(frozen=True)`), not just `PipelineConfig` at the top
  level. Reasoning discussed: freezing doesn't meaningfully speed up attribute *reads* (pydantic v2 is
  already fast either way) - the real win is correctness, since top-level-only freezing still lets
  `config.paths.out_dir = x` succeed even though `config.paths = other_paths` would raise. Matches "nothing
  should override config after load."
- **Design decision: `use_default_model` flags, one per model, not a single coupled flag.** Wanted a way to
  run the pipeline with no custom-trained models (Cellpose ships a robust pretrained default; Spotiflow
  already has `from_pretrained` fallbacks in `utils.py`). First framing (one `paths.use_default_models` bool
  gating both paths at once) was rejected in favor of two independent flags, because the project already hit
  a real case where a user has a custom model for one stage but not the other (the `do_3d`/`spotiflow_models_
  path` mismatch bug from 2026-07-27's GPU profiling session). Decided:
    - `cellpose_models_path`/`spotiflow_models_path` move out of `PathsConfig` into `SegmentationConfig`/
      `DetectionConfig` respectively, each becoming `Optional[DirectoryPath] = None`, alongside a new
      `use_default_model: bool = False` field.
    - Each section gets its own local `@model_validator(mode="after")` enforcing "path must be set unless
      `use_default_model` is True" - kept local to each model (not a cross-section `PipelineConfig`
      validator) specifically because the paths now live in the same model as the flag that governs them.
    - Default `False`, not `True`: makes the fallback something you opt into explicitly, rather than a bare
      config silently defaulting to a generic model - matches the "be vocal about defaults" requirement.
      Actual logging of "using default model" is `utils.py`'s job (ModelBundle already logs the spotiflow
      fallback), not the schema's.
  Not yet implemented - agreed design, `config.py` still has the pre-restructure shape (paths.cellpose_
  models_path/spotiflow_models_path as required `DirectoryPath`).

**Restructuring implemented.** `cellpose_model_path`/`spotiflow_model_path` (renamed singular, matching
`config.yml`) now live in `SegmentationConfig`/`DetectionConfig` with `use_default_model: bool = False` and
a local `@model_validator(mode="after")` each, exactly as designed above. Two things caught while checking
the implementation against the real config, not just reading the diff:

- **Real bug found by actually running `load_config` against `configs/config.yml`, not just reading the
  code:** `cellpose_model_path` was typed `DirectoryPath` like the other two model/data paths, but the real
  checkpoint on disk (`../_pipeline_assets/cellpose_models/cpsam_pseudo3d_4x_20260506`) is a single 1.2GB
  file (Cellpose-SAM's checkpoint format), not a folder - `models.CellposeModel(pretrained_model=...)` takes
  a file path, unlike `Spotiflow.from_folder(...)` which genuinely does take a directory. Original schema
  design (2026-07-27) grouped all three model/data paths under one type without checking that assumption
  against what's actually on disk for each. Fixed to `FilePath`. Lesson: "these three fields look the same
  (they're all paths to ML model checkpoints)" isn't the same claim as "they have the same shape on disk" -
  worth checking the latter directly rather than pattern-matching from the former.
- Verified all four validator branches directly (`use_default_model` × path-set-or-not) rather than trusting
  the code read - `ValueError` raised inside `@model_validator(mode="after")` gets wrapped into pydantic's
  `ValidationError` automatically, confirmed via `pydantic.ValidationError` in each case. Also confirmed
  mutating a frozen model raises `pydantic_core.ValidationError` ("Instance is frozen"), and that
  `yaml.safe_load("")` on an empty file returns `None`, which `PipelineConfig(**None)` turns into a bare
  `TypeError` rather than a pydantic error - noted as a real edge case for `test_config.py` to cover, not
  fixed (out of scope for the schema work itself).

**`conftest.py` streamlining, scoped while starting `test_config.py` (not implemented, revisit after the
migration lands - see `todo.txt` item 4 for the full writeup):** checked actual fixture bodies across test
files before recommending anything, not just fixture names. Most same-named fixtures across files
(`base_config` in test_run_pipeline.py vs test_model_bundle.py, `base_params` in test_detection.py vs
test_object_measurement.py) turned out to be false friends - same name, unrelated shape/purpose - so
blind-merging by name would've been wrong. Real candidate: test_run_pipeline.py's and test_model_bundle.py's
`base_config` are both the exact "dict pretending to be a config" problem this migration exists to fix -
once both become real `PipelineConfig`s from one valid-config source of truth, sharing one fixture is a
natural side effect of finishing the migration rather than a separate refactor. For genuinely-similar-but-
different-sized fixtures (`stack_2d`/`stack_3d`, test_detection.py 20x20 vs test_segmentation.py 40x40),
suggested a factory fixture (returns a function taking `shape`) instead of merging into one fixed size,
so files stay decoupled while still sharing the array-construction logic.

**Design decision: `use_default_model` "flag wins" semantics.** While writing `test_segmentation_default_
model_true_with_path` (asserts the schema *preserves* a configured path when `use_default_model=True`,
rather than nulling it out), realized the schema alone doesn't answer a real question: if a path is
configured AND the flag is `True`, which one does the pipeline actually load? The schema only encodes "path
is optional when the flag is True" - it's silent on precedence. Two options: (a) flag wins - `True` always
loads the pretrained default, ignoring any configured path (logged as ignored, matching the existing
Spotiflow-fallback logging bar); (b) path wins - a present, valid path always gets used, and the flag only
matters when the path is genuinely absent. Picked (a), because (b) would make `use_default_model` a much
weaker knob than intended - it'd stop meaning "use the default" and start meaning "path is allowed to be
missing," which isn't the same feature. Flag-wins lets someone temporarily force default-model behavior
(e.g. to A/B against their trained model) without editing/deleting the configured path. Not yet
implemented - this is `ModelBundle.load()`/`_load_cellpose`/`_load_spotiflow_from_config` territory in
`utils.py`, untouched so far in this migration. See `todo.txt` item 5 for the tracked follow-up.
- **A schema-level test can pass while leaving a real behavioral question open.** `test_segmentation_
default_model_true_with_path` is a legitimate, correct test - it just tests a narrower thing (the field
survives construction) than its name might suggest to a future reader (which model loads). Worth being
precise about what a passing test actually proves versus what it sounds like it proves, especially when
the interesting behavior lives one layer down (`utils.py`) from what's currently implemented (`config.py`).

**Session closed out: `test_config.py` finished, 44 tests, full coverage of the schema** - `TestYAML`,
`TestDefaultModelsLoadLogic`, `TestFieldConstraints`, `TestDefaults`, `TestPathValidation`,
`TestRequiredFields`, `TestFrozen` (both nested-field mutation and top-level `PipelineConfig`
reassignment). Built almost entirely by writing the tests myself with Claude reviewing each increment by
actually running it, not just reading the diff - caught real bugs this way that a read-through alone
wouldn't have: a `use_default_model=True` test that couldn't actually raise (wrong condition), a copy-paste
bug deleting from the wrong section, a duplicate parametrize case pytest silently disambiguated with a
`0`/`1` suffix, a combined multi-field assertion that would've silently masked a regression in any one of
the three checks it bundled together, and a `result.section.field = x` typo that would've raised
`AttributeError` instead of testing anything. Recurring lesson across most of these: pytest (or Python)
often fails "successfully" in a way that looks like a normal test failure but is actually testing nothing
or the wrong thing - the empty-`pass`-stub trap and the bundled-assertion masking risk are the same root
issue in different shapes.

**Next session: the call-site rewrite.** `config.py`/`config.yml`/`test_config.py` are done; `cli.py`,
`utils.py` (including the still-unbuilt "flag wins" `use_default_model` logic), `run_pipeline.py`, and
`qc_figures.py` all still access config via the old `config["section"]["key"]` dict pattern and need
rewriting to `config.section.key`, per the build order in `todo.txt` item 5.

### Concepts learned

- **A collaboration-mode instruction is itself worth recording, not just the technical decisions.** Told
  explicitly: from here, write advice only and keep `todo.txt`/`CLAUDE.md`/`LEARNING_NOTES.md` current - no
  more code snippets from Claude, even for novel scaffolding (which is a step further than the earlier
  "full snippets for novel/architectural work, worked example + self-serve for repetitive edits" pattern
  from the logging session). Saved as a standing instruction rather than something to re-derive next session.
- **Checking a design decision by actually running it beats re-reading the diff.** The `cellpose_model_path`
  `DirectoryPath`-vs-`FilePath` bug above wasn't visible from reading `config.py` in isolation - it only
  showed up by loading the real `configs/config.yml` and hitting a real `ValidationError`. Same pattern as
  the `bin_factor`/`block_reduce` finding earlier this session: both were caught by executing code against
  real inputs (a real config file; a real non-divisible image shape) rather than reasoning about the code
  from its text alone.

## 2026-07-27 (cont'd) — Crash resilience framing + config validation scoped (implementation starts 2026-07-28)

**Session goal:** with item 1 closed, talked through two open items from `todo.txt` (crash resilience for
overnight runs, config schema/validation) before touching any code. Both ended in a scoped plan for
tomorrow rather than a diff today - decisions and reasoning captured here so they don't need re-deriving.

### Concepts learned

- **Not every crash is catchable, and that changes what "log what happened" can promise.** A CUDA OOM
  raised by PyTorch is a normal Python exception - existing `except Exception` blocks already catch it. But
  the OS OOM killer (or preemption, or power loss) terminates a process with `SIGKILL`, which cannot be
  caught, handled, or logged from inside the process by design. So resilience for an overnight run splits
  into two different problems: better exception handling helps with the first class, but the only defense
  against the second is making sure completed work is already durable on disk *before* the kill happens -
  the term for this design philosophy is "crash-only software" (Candea & Fox). Concretely for this repo:
  per-file CSVs already satisfy that; the run-level rollup (`_run_objects_{MODE}.csv`) doesn't, since it's
  only written once at the very end - flagged as a small, cheap, still-open fix (make it reconstructable
  from per-file CSVs already on disk) separate from the bigger resume question.
- **"Resume a killed run" and "redo with different params" look like the same feature (skip-if-exists) but
  are dangerously different if conflated.** A naive skip-if-exists resume, run in the same output folder
  with a *different* model/config than produced the existing files, would silently blend two runs' results
  under one filename with no visible error - worse than no resume feature at all, because nothing looks
  wrong. Decided to defer this entirely to the planned Snakemake migration (already on the roadmap,
  `workflow/`), since Snakemake's DAG + rule params already solve "only rebuild what actually needs
  rebuilding" correctly - hand-rolling a weaker version now would be worth throwing away later anyway.
- **"Validate at the boundary, don't propagate the new type" vs. "migrate everything to attribute access"
  is a real scope decision, not just an implementation detail.** First framing (pydantic model built and
  validated inside `load_config`, then `.model_dump()`'d back to a plain dict) would've kept every
  `config["section"]["key"]` call site in the codebase unchanged - smallest possible diff, but keeps the
  stringly-typed access pattern alive everywhere except the one function that loads the file. Chose instead
  to actually rewrite call sites to `config.section.key` - bigger diff (mapped: 4 source files / 11 call
  sites, 4 test files), but it's the version that actually fixes "config keys are accessed by string
  throughout" rather than just adding a validation step in front of the same pattern.
- **A strict schema forces dead/misdocumented config to get resolved, not just noticed.** Grepping for
  actual usage before writing the schema turned up two real findings that the audit's "config.py has no
  schema/validation" line alone wouldn't have surfaced on its own: `channels.misc` has zero references
  anywhere in `src/` (confirmed dead, dropped from the new schema), and `segmentation.use_gpu` wasn't just
  unread in code - the README/PKG-INFO actively documented behavior ("GPU use is optional and controlled
  per-run via segmentation.use_gpu") that the code didn't implement. Fixed the code side same-day; the
  schema migration is what forced the check that caught the doc/code mismatch in the first place.
- **Fixing dead config immediately can break a test whose fixture predates the fix - and that's fine to
  leave red if the fixture is about to be rewritten anyway.** Wiring up `use_gpu` in `utils.py` made
  `test_model_bundle.py`'s `base_config` fixture (a plain dict missing a `segmentation` key) raise
  `KeyError('segmentation')`. Decided not to patch the dict fixture, since that exact fixture is on the
  list to become a `PipelineConfig` instance tomorrow - a throwaway fix now would just be redone.

### Questions to follow up on

- Once `PipelineConfig` exists: should `_panel_ecdf`/`make_qc_figure` in `qc_figures.py` keep taking the
  whole `config` object just to read one field (`prob_thresh`), or switch to taking that float directly?
  Flagged as optional cleanup while already touching that call site, not required for the migration itself.
- Run-level CSV reconstructability (rebuilding `_run_objects_{MODE}.csv` from per-file CSVs already on
  disk, instead of only writing it once at the end) is still unimplemented - small, cheap, and independent
  of the pydantic work; could be picked up either before or after it.

## 2026-07-27 — Error handling & robustness, implemented and verified

**Session goal:** finish item 1 from the audit (per-scene/per-file failure isolation, failures CSV, the two
silent `qc_figures.py` excepts, the `coor=` typo). Claude explained the design and reviewed each increment;
I wrote the actual diffs myself.

### Concepts learned

- **"Kill the pipeline" means *not* catching, not catching-and-passing.** My first attempt wrapped
  `ModelBundle.load()` and `BioImage(filepath)` in `try/except: pass` placeholders, intending that to crash
  the run. It does the opposite — swallowing the exception lets execution fall through with `models`/`img`
  never assigned, so the real crash becomes an uninformative `UnboundLocalError` several lines later instead
  of the original, informative exception at the point of failure. The fix for "this should kill the pipeline"
  was to delete the `try/except` entirely, not to write a smarter one. Separately, this exposed that the two
  cases actually needed *different* treatment: `ModelBundle.load()` genuinely should crash (setup, runs once,
  fail-fast at the boundary); `BioImage(filepath)` should NOT be caught locally at all, because it's called
  per-file inside a loop that's *already* wrapped by an outer per-file try/except in `run_pipeline()` — the
  inner catch was pre-empting the outer one and needed to just not exist.
- **A shared mutable list threaded through as a parameter, appended to at multiple call depths, is a
  legitimate pattern for accumulating cross-cutting state (like failures) without changing return types.**
  Alternative would've been `_process_file` returning `(scene_df, failures)` tuples and stitching them
  together at each level — more "pure," but would've broken every existing caller's return-value handling.
  Passing one `failures: list` down from `run_pipeline()` into `_process_file()`, appended to at both the
  scene-level and file-level except blocks, kept both failure sources landing in one final DataFrame with
  zero return-signature disruption.
- **Order of operations around an early return matters as much as the exception handling itself.** First
  version of the failures-CSV write sat *after* `if not all_run_records: return None` — meaning a run where
  every single file failed (exactly the case you'd most want a failures report for) never got one, because
  the early return skipped straight past it. Moving the write above the check fixed it. General lesson:
  whenever a function has an early-exit guard, check whether anything *after* it should actually have run
  regardless of which branch triggered the exit.
- **Test doubles need to match the real control flow, not just the intent.** `test_handles_corrupted_file`
  gave `_process_file`'s mock a 3-item `side_effect` list (2 successes + 1 exception) but only created 2 real
  files on disk in `tmp_path` — the file loop in `run_pipeline()` only calls `_process_file` once per real
  file via `data_folder.iterdir()`, so the mock was only invoked twice, silently starving the third
  `side_effect` entry and producing a confusing `assert 2 == 3` instead of a clear "your fixture is wrong"
  signal. Fixed by adding a third `.touch()`'d file so file count matches side_effect length.
- **A "fixed" bug in a wired-up feature can reopen a different test hygiene bug.** `cli.py`'s `main()` now
  actually calls `configure_logging()` (closing the older confirmed bug from the 2026-07-15 audit). But
  `test_cli.py`'s test never mocked it, so *fixing* the wiring bug turned an inert function call into a real
  one — every test run started attaching real file+console handlers to the root logger and leaving untracked
  `output/logs/run_*.log` files in the repo root. Confirmed via `git status` before and after adding
  `mocker.patch("spot_detector.cli.configure_logging")`. General lesson: closing a "not wired up yet" bug can
  silently activate side effects in code that assumed it would stay inert — worth an explicit pass over
  anything that calls the newly-wired function, tests included.
- **Where a `try:` starts inside a loop determines what it actually protects.** `img.set_scene(scene)` was
  originally called *before* the per-scene `try:`, not inside it — meaning a scene-switch failure (corrupted
  scene metadata) wasn't isolated per-scene at all; it would propagate out of the whole file and get caught
  one level up by the per-*file* handler, discarding every already-successful scene for that file along with
  it. Moving the line inside the `try` block closed the gap.

### Questions to follow up on

- Still undecided: catch+skip (operational errors) vs. fail-fast (config/programmer errors) as *different*
  exception classes, rather than one bare `except Exception` at each level. `assign_spots_to_mask`'s
  dimension-mismatch `ValueError` is the concrete example — it'll fail identically on every scene, which is
  pure wasted compute under the current uniform-catch design.
- `make_qc_figure()` failing currently discards an already-computed `scene_df` along with it, since it's the
  last call in `_process_scene` and that function stays intentionally exception-transparent. Is that
  acceptable (encourages fixing the QC code instead of masking it) or should measurement data survive a
  rendering failure? Not decided — flagged in `todo.txt`.

## 2026-07-15 — Logging implementation, verified

**Session goal:** implement the logging plan (configure_logging, print->logger conversions across
run_pipeline.py/utils.py/qc_figures.py), then have Claude check the result against a fresh read of the code
and a real test run, not against what was discussed earlier in the conversation.

### Concepts learned

- **`logger.exception()` bundles two independent choices you can actually split apart.** It's shorthand for
  `logger.error(msg, exc_info=True)` — hardcoding both the severity (ERROR) and "attach a traceback"
  together. But `exc_info=True` works on *any* level. `utils.py`'s Spotiflow fallback is a deliberate,
  successful recovery (not a failure), so it stayed at `WARNING` — but `exc_info=True` was still added,
  because `str(e)` alone can hide *where* in the load chain a failure happened (bad path vs. corrupted
  checkpoint vs. version mismatch look identical as a bare message, but very different as a traceback).
  Severity should answer "does a human need to look at this"; `exc_info` should answer "would the traceback
  help them" — two separate questions, not one.
- **A search-and-replace done for one reason can silently break something unrelated.** Fixing `mode.upper()`
  inside log messages (for readability) accidentally spread to five output *filename* f-strings in
  `run_pipeline.py`, changing `_run_objects_2d.csv` to `_run_objects_2D.csv` and breaking the established
  lowercase naming convention. Caught by 4 failing tests in `test_run_pipeline.py` — a clean example of why
  "we have tests for this" matters even for changes that feel purely cosmetic (logging cleanup, in this case).
- **A function being defined correctly doesn't mean it's wired up.** `configure_logging()` in `cli.py` was
  well-written but (as far as a fresh read of the file on disk showed) never actually called from `main()` —
  meaning none of the logging work would have taken effect on a real run. Worth double-checking: did I
  actually verify this from disk, or am I trusting an editor buffer that hasn't been saved yet? Follow-up
  pending confirmation.
- **Don't trust your own earlier audit without re-checking.** The original codebase audit (start of this
  session) found several test files were empty stubs. By the time we got to the logging work, that was no
  longer true — a separate commit (`babe381`, "test_run_pipeline finished") had already filled in
  `test_run_pipeline.py`, and it turned out *none* of the test files were still empty. Claude initially wrote
  a stale claim into `todo.txt` based on the old audit instead of re-reading current state — good reminder
  that "I checked this earlier in the conversation" isn't the same as "this is still true now," especially
  across a long session with real edits happening between checks.

### Questions to follow up on

- Resolve the `configure_logging()` wiring question — check `cli.py` for unsaved changes vs. what's on disk.
- Once error-handling work starts: the per-scene `try/except` in `_process_file` already exists (added
  alongside the logging work) — next session should build on it (failures list/CSV, file-level catch, a
  failure-isolation test) rather than starting the design from scratch.

## 2026-07-15 — Error handling & robustness deep dive (implementation pending)

**Session goal:** understand the principles before implementing per-file/per-scene failure isolation myself.
Claude explained, I haven't written the code yet — will log the actual implementation lessons once done and
verified.

### Concepts covered

- **Fail fast at boundaries, fail isolated in loops — these are opposite instincts.** Config/setup problems
  (before the loop starts) should crash immediately and loudly. Per-item problems (inside the file/scene loop)
  should be caught, logged, and skipped so the rest of the batch survives. Conflating the two — e.g. catching a
  config bug per-scene — just reproduces the same crash hundreds of times instead of once.
- **Operational errors vs. programmer errors is the key mental model.** Operational = the code is right, but the
  world misbehaved (corrupt file, GPU OOM) → catch, log, continue. Programmer/invariant = an assumption was
  violated (e.g. `assign_spots_to_mask`'s dimension-mismatch `ValueError`, which means `do_3d` disagrees with the
  actual data) → will fail identically on every item, so isolating it per-scene just burns compute; better caught
  once, early, loudly.
- **A caught-and-silent exception is worse than an uncaught one.** `except Exception: pass` (as in
  [qc_figures.py:345](src/spot_detector/qc_figures.py#L345)) throws away the only information you'd have about
  what went wrong. `logging.exception(...)` (called from inside an `except` block) attaches the full traceback
  automatically — that's the minimum bar for "caught but not silent."
- **Exception chaining (`raise X from e`) preserves the original cause** when wrapping a low-level exception in a
  more contextual one (e.g. "scene 4 of Control_01.nd2 failed" wrapping the underlying `RuntimeError`).
- **Partial-progress writing is defense in depth independent of try/except.** Even perfect error isolation
  doesn't help if the *process* itself dies (OOM-killed, cluster preemption). Writing results incrementally
  rather than only concatenating at the very end protects against that separately.

### Questions to follow up on

- Where exactly does GPU OOM surface for this stack — `torch.cuda.OutOfMemoryError` or a `RuntimeError` with
  "out of memory" in the message? Depends on the torch version pinned via `uv.lock`; need to check before writing
  a specific `except` clause for it.
- Once implemented: does catching per-scene *and* per-file (two-level isolation) turn out to be worth the extra
  boilerplate, or is per-file alone good enough in practice? Will know after trying it.

## 2026-07-15 — Codebase improvement audit

**Session goal:** identify and rank improvement areas (testing, logging, error handling, code organization,
cluster/parallelization readiness) before changing anything.

### Concepts learned

- **Broad `except Exception` clauses hide two very different failure modes.** In [utils.py:59-66](src/spot_detector/utils.py#L59-L66),
  the Spotiflow model-loading fallback catches *any* exception and silently falls back to a pretrained model.
  That's fine for "no custom model configured" but identical in behavior to "custom model path has a typo" or
  "checkpoint file is corrupted." Lesson: a broad except is only safe if you also log *what* was caught, not just
  that something was caught — otherwise you can't tell a deliberate fallback from a masked bug.
- **Untested code paths accumulate silent bugs.** Found a live typo (`coor="gray"` instead of `color="gray"`) at
  [qc_figures.py:328](src/spot_detector/qc_figures.py#L328), inside a branch that only executes when a scene has
  0–1 detected spots. `qc_figures.py` is the largest module in the package (691 lines) and has zero test coverage
  — that's not a coincidence. The lesson generalizes: the size/complexity of a module and the cost of it being
  untested scale together, not independently.
- **Sequential batch pipelines lose *everything* on a late failure, not just the failing item.** In
  [run_pipeline.py](src/spot_detector/run_pipeline.py), the run-level CSV is only written after every file has
  been processed ([run_pipeline.py:54-57](src/spot_detector/run_pipeline.py#L54-L57)), and there's no try/except
  anywhere in the file/scene loops. This means a crash on item N out of M throws away all progress on items
  1..N-1 at the run level (per-file CSVs already flushed to disk survive, but the aggregate doesn't). This is a
  general pattern to watch for in any long-running batch job, not specific to this repo: write partial results
  incrementally, and isolate failures so one bad input can't take down the whole batch.
- **Config-as-plain-dict has a hidden cost.** `load_config` in [config.py](src/spot_detector/config.py) returns a
  bare `dict`, and consumers do string-keyed lookups (`config["paths"]["cellpose_models_path"]`) scattered across
  multiple files. A missing/typo'd key surfaces as a `KeyError` far from where the mistake was made, with no
  indication of which field is at fault. Schema validation at the boundary (where config is loaded) turns a
  confusing runtime crash into an immediate, specific error message.
- **"Convention over configuration" needs a guardrail, or it fails silently.** `experiment` is inferred from
  `data_folder.parent.name` ([run_pipeline.py:27](src/spot_detector/run_pipeline.py#L27)) — an undocumented rule
  that the experiment name is whatever directory *contains* the raw-data folder. I confirmed this actually broke
  in practice: [output/tables/None_objects_3d.csv](output/tables/None_objects_3d.csv) has an empty `Experiment`
  column in every row from a real run. Nothing validated or warned that the derived value was empty. Lesson:
  when metadata is inferred by convention rather than passed explicitly, add a check that fails loudly if the
  inferred value looks wrong (empty, unexpected type, etc.) — otherwise it just flows downstream into your data.
- **Dead config is worse than no config.** `configs/config.yml` declares `segmentation.use_gpu: true`
  ([config.yml:19](configs/config.yml#L19)), but [utils.py:47](src/spot_detector/utils.py#L47) hardcodes
  `gpu=True` and never reads that key. Editing the YAML gives false confidence that you're controlling GPU usage.
  General lesson: an unused config field is actively misleading, not just harmless — worth grepping for "is this
  config key actually read anywhere?" whenever auditing a YAML-driven pipeline.

### Questions to follow up on

- What's the cleanest way to add per-file (or per-scene) error isolation without silently hiding failures — i.e.
  fail loud in the log but keep processing the rest of the batch? (Relevant to error handling + logging.)
  - Update 2026-07-15: leaning towards a decorator or try/except at the `_process_file` level that logs the
    exception with traceback and appends a `status=failed` row rather than raising — but haven't validated this
    against how Snakemake would want to see failures reported.
- What would a minimal config schema look like here (dataclass? pydantic? just a `validate_config()` function)
  given the project deliberately avoided adding new dependencies during the conda→uv migration?
- How do other Snakemake-based imaging pipelines handle GPU scheduling across cluster nodes with mixed
  GPU/CPU availability? Relevant to the dead `use_gpu` flag and the broader cluster-readiness goal in `todo.txt`.
