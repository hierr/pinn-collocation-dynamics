# Repository instructions for agents

Read `MEMORY.md` and, if present, ignored `AGENTS.local.md` at session start.

## Scope

- Communicate and write code and documentation in English.
- This is a research repository. Start small, use targeted diffs,
  and generalize only when another experiment requires it.
- Use DeepXDE with a PyTorch backend for the initial upstream reproduction.
  Keep problem definitions and sampling callbacks separate as needed; avoid
  introducing a general framework before experiments require it.
- Preserve upstream attribution, licenses, pinned commits, and recorded
  deviations. Distinguish code reproduction from a modified benchmark study.

## Environment and execution

- Use Python 3.12 and manage dependencies through uv in a local `venv`.
- No environment exists yet. Inspect available tools and obtain approval
  before creating it or choosing dependency versions. Check whether conda is
  necessary; otherwise prefer uv. Do not add production testing tooling by default.
- After dependency changes, run `uv lock --check` and
  `uv pip check --python venv/bin/python`.
- Verify GPU execution on the host before claiming CUDA validation. Do not
  interpret sandbox GPU restrictions as an environment failure.
- Write plain docstrings and concise intent comments before non-trivial blocks.

## Experiments and artifacts

- Keep scientific settings in YAML. Archive the resolved configuration in
  `output/<campaign>/<problem>/<method>/<seed>/config.yaml` at run start.
- Never overwrite a prior run or alter original reference datasets.
- Use archived configurations and fixed evaluation seeds for post-processing.
- Before material runs, document objectives, controls, changed variables,
  metrics, reproducibility requirements, and decision criteria in
  `experimental/experimental_notes.md`. Obtain approval for the methodology.
- Record completed results and limitations there, with compact comparisons in
  `experimental/results.md`. Create report notes when a report requires them.
- Only physics points move; keep boundary, initial, and supervised constraints
  fixed in matched sampling comparisons. Keep point budgets matched unless
  budget is the variable explicitly under study.
- Retain planned seeds and report failures. Avoid unsupported novelty or
  superiority claims; diffusion and sharpness benefits are open questions.
- Datasets and outputs are ignored by Git. Agree on an artifact archive and
  retrieval policy before producing campaign results.

## Change control

- Propose a concise plan and obtain approval before edits, dependency changes,
  artifact-writing execution, or training. Existing explicit approval applies
  to its agreed scope.
- Obtain explicit approval before commits, branches, tags, or publication.
  Use Conventional Commits.
- Never modify or discard user changes without authorization.
- Update `MEMORY.md` for durable setup, structure, workflow, and policy decisions
  only, using dated `(Mem-xxxx)` bullets. Keep experiment details in notes.
- Never store secrets in code, logs, or documentation.
