# Repository instructions for agents

Read ignored `AGENTS.local.md`, if present, at session start.

## Scope

- Communicate and write code and documentation in English.
- This is an independent research repository, separate from coursework repositories.
  Start small, use targeted diffs, and generalize only when another experiment
  requires it.
- Use DeepXDE with a PyTorch backend for the initial upstream reproduction.
  Keep problem definitions and sampling callbacks separate as needed; avoid
  introducing a general framework before experiments require it.
- Keep the experiment readable from top to bottom, with explicit optimizer stages
  and consistent parameter names. Each sampling class contains its numerical
  updates; keep logging and serialization outside the scientific flow.
- Preserve upstream attribution, licenses, pinned commits, and recorded
  deviations. Distinguish code reproduction from a modified benchmark study.
- Treat `references/pacmann/burgers/` as immutable upstream material. Restore
  `data/burgers/Burgers.npz` using `references/upstream.md`.

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
  Follow [methodology](docs/methodology.md#work-classification-and-campaign-records);
  run duration and seed count alone do not determine scientific value.
- Keep scientific settings in YAML. Archive the resolved configuration in
  `output/<campaign>/<problem>/<method>/<seed>/config.yaml` at run start.
- Never overwrite a prior run or alter original reference datasets.
- Use archived configurations and fixed evaluation seeds for post-processing.
- Model snapshots support evaluation, not exact training resume.
- Use [research context](docs/research_context.md) for research questions, the
  study plan, accepted scientific decisions, and open dependencies. Keep
  [methodology](docs/methodology.md) for reusable procedures and final environment
  choices. Put scientific campaign protocols and interpretation in
  [experimental notes](docs/experimental_notes.md), compact comparisons in
  [results](docs/results.md), and artifact meanings in
  [output format](docs/output_format.md). Keep detailed numerical records in outputs.
- Before scientific campaigns, document objectives, controls, changed variables,
  metrics, reproducibility requirements, and decision criteria in
  `docs/experimental_notes.md`. Obtain approval for the experiment protocol.
- Record scientific development and evaluation outcomes, limitations, and
  decisions there at campaign level. Do not add engineering validation history
  or an operational incident diary. Create report notes only when requested.
- Each scientific campaign has a short README linking its stable experiment-notes
  anchor, results table, and `docs/output_format.md`; keep narratives in the notes.
- Only physics points move; keep boundary, initial, and supervised constraints
  fixed in matched sampling comparisons. Keep point budgets matched unless
  budget is the variable explicitly under study.
- Retain planned seeds and report numerical failures. Distinguish external
  interruptions from scientific failures; follow the
  [recovery policy](docs/methodology.md#execution-interruptions-and-recovery).
  Preserve completed artifacts; never select retry attempts by favorable error.
  Avoid unsupported novelty or
  superiority claims; diffusion and sharpness benefits are open questions.
- Track scientific run directories directly in Git, including configs, models,
  points, logs, predictions, and diagnostics. Reference datasets remain ignored
  with restoration instructions. Setup-validation/compatibility runs are
  disposable and ignored; do not create compressed run archives by default.

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
