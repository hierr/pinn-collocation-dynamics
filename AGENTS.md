# Peer-review branch instructions

Read ignored `AGENTS.local.md`, if present.

- This branch is the compact PIML peer-review submission based on research
  checkpoint `6e7a638383fe0e04d9d999a2cc491c313f587cc8`. Keep the review route
  self-contained; full campaigns and development artifacts remain on their
  research branches.
- Preserve `src/`, the standard post-processing implementation, immutable
  `references/pacmann/burgers/`, and the pinned dependency files. Obtain approval
  for numerical changes or dependency changes.
- Keep all 250 original final records, including unfavorable outcomes, and the
  complete PACMANN-Adam seed-4000 example at nu = 0.01/pi. Never overwrite them.
- Use the existing Python 3.12 `venv`. Execute the commands documented in
  [README](README.md#setup-and-execution); scientific reasoning and code order
  are in [the review guide](docs/peer_review.md).
- The short demo is engineering validation. Its outputs are disposable and
  ignored. Select the method in the script and scientific settings in YAML;
  avoid new orchestration or command-line interfaces.
- Write portable, concise English documentation. Preserve source attribution,
  distinguish evidence from interpretation, and explain reduced demo budgets.
- Run relevant pytest checks with coverage, Ruff, and ty. Preserve safeguards
  for reference identity, numerical differentiation, paired populations, and
  fixed constraints.
- Obtain explicit approval for commits, tags, publication, and additional
  scientific protocols. Use Conventional Commits.
