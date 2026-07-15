# Learning Notes

Running log of things I'm learning about this codebase and about software engineering practices in general,
kept as I work through `nikon_cellpose_bags_spots` with Claude Code. Newest entries at the top.

---

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
