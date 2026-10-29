# Repository instructions for agents

Read ignored `AGENTS.local.md`, if present, at session start.

## Scope

- Keep `main` focused on accepted implementations, frozen evaluation settings,
  full evaluation campaigns, and concise accepted conclusions. Dataset checks
  live on persistent `dataset-validation`; development runs and tuning settings
  live on persistent `hyperparameter-tuning`; exploratory post-processing lives
  on persistent `analysis`. When an analysis supports an accepted result or
  retained figure, transfer its reproducible implementation to
  `scripts/analysis/` on `main`. Keep standard per-run artifact generation in
  `scripts/post_processing/` and explanatory visuals in `scripts/illustrations/`.
  Leave superseded or unadopted explorations on their workstream branch.
  Transfer selected code or decisions with their source commit; do not merge
  entire workstreams and their artifacts.

- Communicate and write code and documentation in English.
- This is an independent research repository, separate from coursework repositories.
  Start small, use targeted diffs, and generalize only when another experiment
  requires it.
- Use DeepXDE with a PyTorch backend for the controlled Burgers study.
  Keep problem definitions and sampling callbacks separate as needed; avoid
  introducing a general framework before experiments require it.
- Keep the experiment readable from top to bottom, with explicit optimizer stages
  and consistent parameter names. Each sampling class contains its numerical
  updates; keep logging and serialization outside the scientific flow.
- Preserve upstream attribution, licenses, pinned commits, and recorded
  deviations. Distinguish code reproduction from a modified benchmark study.
- Treat `references/pacmann/burgers/` as immutable upstream material. The retained
  original reference is `data/burgers_dataset/PACMANN_Burgers.npz`; generated
  references are selected by `problem.reference_path`.

## Environment and execution

- Use Python 3.12 and manage dependencies through uv in a local `venv`.
- The local `venv` and `uv.lock` use the pinned released dependency stack.
  Use the existing environment; obtain approval before dependency changes.
  Conda is unnecessary for the current wheel-based stack. Dev tools are
  ruff, ty, pytest, and pytest-cov; run focused tests with coverage.
- After dependency changes, run `uv lock --check` and
  `uv pip check --python venv/bin/python`.
- Verify GPU execution on the host before claiming CUDA validation. Do not
  interpret sandbox GPU restrictions as an environment failure.
- Write plain docstrings and concise intent comments before non-trivial blocks.

## Experiments and artifacts

- Classify work by purpose before observing outcomes: engineering validation,
  scientific development (including tuning), or frozen-setting evaluation.
  Engineering validation checks readiness; development selects scientific
  settings; evaluation tests research questions with frozen settings. Run duration
  and seed count alone do not determine scientific value.
- Keep scientific settings in YAML. Archive the resolved configuration in
  `output/burgers/nu_<viscosity>/<campaign>/<method>/<seed>/config.yaml` at run start.
  Retain older completed campaigns at their archived paths.
- Never overwrite a prior run or alter original reference datasets.
- Use archived configurations and fixed evaluation seeds for post-processing.
- Model snapshots support evaluation, not exact training resume.
- Use [research context](docs/research_context.md) for research questions, the
  study design, phases, accepted scientific decisions, and open dependencies.
  Keep setup and final environment choices in [README](README.md#setup-and-execution).
  Put comparisons and brief interpretation in
  [results](docs/results.md), and artifact meanings in
  [output format](docs/output_format.md). Keep detailed numerical records in outputs.
- Write authored documentation for external researchers and eventual paper
  preparation. Use commands from the repository root and repository-relative
  paths; omit personal hostnames, usernames, absolute checkout paths, and remote
  access or session-management instructions. Retain hardware, software,
  concurrency, provenance, and deviations that affect reproducibility or
  interpretation. Keep observations distinct from hypotheses and disclose
  analyses chosen after inspecting results.
- Put shared scientific figures in `docs/figures/` and reference them from
  documents and presentations. Keep per-run diagnostics with their run and
  campaign-specific diagnostics with their campaign. Reserve
  `docs/presentation/figures/` for assets specific to the presentation.
- Obtain approval for the scientific objective, comparisons, evaluation, and
  decision rules before execution. Keep campaign-specific reasoning not captured
  by configurations or artifacts in a concise, objective
  `output/burgers/nu_<viscosity>/<campaign>/README.md`, alongside its run directories.
  Include execution instructions and links to settings, the study phase, results,
  and `docs/output_format.md`. Summarize per-seed final outcomes and method
  aggregates there; keep study-level comparisons in `docs/results.md`.
  When several output directories share one design,
  document it once and link from the others.
- Do not duplicate run settings or maintain campaign diaries. Update research
  context for consequential study decisions; keep engineering validation history
  and operational incidents out of scientific documents unless they affect
  interpretation. Create report notes only when requested.
- Only physics points move; keep boundary, initial, and supervised constraints
  fixed in matched sampling comparisons. Keep point budgets matched unless
  budget is the variable explicitly under study.
- Retain planned seeds and report numerical failures. Distinguish external
  interruptions from scientific failures; follow the [recovery policy](#execution-recovery).
  Preserve completed artifacts; never select retry attempts by favorable error.
  Avoid unsupported novelty or
  superiority claims; diffusion and sharpness benefits are open questions.
- Track scientific run directories directly in Git, including configs, models,
  points, logs, predictions, and diagnostics. Reference datasets are tracked;
  preserve their provenance and restoration instructions.
  Setup-validation/compatibility runs are
  disposable and ignored; do not create compressed run archives by default.

## Execution recovery

- Retain numerical failures and planned seeds; do not silently retry them or
  replace seeds. External interruptions are not evidence of algorithm failure.
  An unchanged same-seed restart is recovery within the approved scope, with
  completed or scientifically meaningful artifacts preserved. Resolve occupied
  destinations through a separate attempt or explicitly authorized cleanup of
  disposable incomplete artifacts; never overwrite runs or select attempts by
  favorable error.
- Regenerate missing diagnostics from completed training artifacts when possible.
  Saved weights do not support exact training resume. Seek approval for changes
  to scientific settings, budgets, or seeds.

## Change control

- Propose a concise plan and obtain approval before edits, dependency changes,
  artifact-writing execution, or training. Existing explicit approval applies
  to its agreed scope; do not repeatedly request permission for routine steps
  within an approved campaign. Seek approval for material protocol changes.
- The main agent runs and monitors campaigns unless delegation is explicitly
  requested. Use polling intervals proportional to expected runtime. Do not
  stop healthy training merely to change the monitoring agent or operator.
- Obtain explicit approval before commits, branches, tags, or publication.
  Use Conventional Commits.
- Never modify or discard user changes without authorization.
- Never store secrets in code, logs, or documentation.
