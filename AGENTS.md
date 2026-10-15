# Repository instructions for agents

Read ignored `AGENTS.local.md`, if present, at session start.

## Scope

- Write code and documentation in English; keep scientific code minimal and readable.
- This persistent `dataset-validation` branch contains dataset generation and
  validation only. Use root `dataset_generation/` and `dataset_validation/`.
  Training and tuning workflows live on their respective branches; see README.
- Keep the collaborator's solver unchanged except for authorized path changes.
  Treat `references/pacmann/burgers/` as immutable upstream material and preserve
  attribution and licenses.
- Use the existing Python 3.12 uv-managed `venv` and locked dependencies.
  Obtain approval before dependency changes. Conda is unnecessary.

## Validation and artifacts

- Obtain approval for scientific objectives, comparisons, and decision rules
  before new artifact-writing execution. Continue within approved scope.
- Preserve original datasets and model inputs. Regenerate only derived comparisons.
  The retained source models support evaluation, not exact training resume.
- Put comparison artifacts in `data/burgers_dataset/reference_comparison/` with
  concise READMEs alongside the artifacts. Keep numerical records there, broader
  decisions in `docs/research_context.md`, accepted comparisons in
  `docs/results.md`, and artifact meanings in `docs/output_format.md`.
- The current check compares references at the original spatial coordinates and
  exact shared times, using Chebyshev barycentric spatial interpolation. Keep
  predictions fixed when changing the reference used to score a model.
- Numerical agreement at one viscosity does not validate all viscosities or
  certify absolute solver accuracy.
- Run focused pytest checks with coverage and ruff/ty for edited Python code.
  Preserve the collaborator's unrelated code and formatting.
- Obtain approval before commits, branches, tags, or publication; use
  Conventional Commits. Do not discard user changes.
