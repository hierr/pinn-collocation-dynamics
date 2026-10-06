# Project memory

- [2026-10-05] (Mem-0001) The project is independent research on adaptive
  collocation and stochastic diffusion for PINNs, targeting a possible publication.
- [2026-10-05] (Mem-0002) Begin with a small DeepXDE/PyTorch Burgers reproduction
  and separate sampling callbacks. Generalize and add functionality only when
  required. The initial implementation is recorded in Mem-0007.
- [2026-10-05] (Mem-0003) Runs use
  `output/<campaign>/<problem>/<method>/<seed>/` with an archived configuration.
  Data and generated artifacts stay outside ordinary Git tracking; campaign
  execution requires a documented storage and retrieval policy.
- [2026-10-05] (Mem-0004) The initial upstream PACMANN commit is recorded in
  `references/upstream.md`; the unchanged Burgers reference snapshot is imported
  under `references/pacmann/burgers/` (see Mem-0006).
- [2026-10-05] (Mem-0005) Class-specific material belongs in the separate
  master's repository, which can link to this research repository at a stable
  tag or commit. This repository has no coursework directory or prior-class
  provenance; retain external PACMANN provenance for reproduction.
- [2026-10-05] (Mem-0006) Pinned upstream scripts, requirements, and license are
  immutable reference material, separate from maintained code. A tracked manifest
  records source URLs, sizes, and hashes; the ignored Burgers dataset lives at
  `data/burgers/Burgers.npz` with restoration instructions in source provenance.
  The initial executable reproduction infrastructure is recorded in Mem-0007.

- [2026-10-06] (Mem-0007) Python 3.12 and uv manage local `venv` with a
  tracked lock: DeepXDE 1.14.0, PyTorch 2.7.1/cu128, and released numerical
  dependencies. Conda is unnecessary. The minimal Burgers implementation separates
  problem definition, PACMANN callback, and single-run CLI. YAML selects scientific
  settings; CLI selects method and seed index. Phase/final weights are evaluation
  snapshots, not exact-resume checkpoints. Focused tests use pytest with coverage;
  ruff and ty provide static checks. Diffusion and campaign automation are deferred.

- [2026-10-06] (Mem-0008) Committed the minimal Burgers reproduction
  implementation as `a64639d`. Tested DeepXDE 1.15.0 and PyTorch 2.14.1/cu130
  in an isolated environment using the existing focused tests and short GPU runs
  for both methods. No source changes were needed for compatibility. Retained
  DeepXDE 1.14.0 and PyTorch 2.7.1/cu128 for upstream reproduction.
- [2026-10-06] (Mem-0009) After validation, removed the temporary compatibility
  environment, manifest/lock, configuration, and detailed validation records from
  tracked documentation. Final environment decisions belong in methodology;
  experimental notes and results are reserved for scientific experiment runs.
  Memory retains the actions taken and decisions made, without detailed results.

- [2026-10-06] (Mem-0010) The approved reproduction pilot uses a dedicated
  one-seed YAML and committed protocol before training. Pilot artifacts remain
  local, with a checksummed compressed archive and verified extraction; this is
  not off-machine backup. Expansion to ten seeds requires separate review.
