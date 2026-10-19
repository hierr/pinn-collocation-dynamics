# Burgers diffusion tuning at nu = pi/100

Prepared scientific development; training has not started. This viscosity
fills the gap between `0.1` and `0.01` in the planned sharpness comparison.
It was chosen after inspecting existing reference profiles.

## Reference and reproduction

The tracked reference
`data/burgers_dataset/burgers_nu_0p031415927_N1024.npz` is restored unchanged
from dataset commit `52b9359`. Its stored viscosity is `pi/100`
(`0.031415926535897934`); only its filename rounds the value. Generation used
Chebyshev degree 1024 (1,025 nodes), 201 snapshots through `t = 1`, Radau,
`rtol = 1e-11`, `atol = 1e-13`, and `max_step = 0.005`. Its SHA-256 is
`aad9aff8eecc32e1fe9e60ad2e4ca5f399dd5e96bf748851f085a2165923fb60`.

Finite values, grid and time coverage, viscosity, and initial/boundary values
were checked. Native-node Chebyshev differentiation over `[0, 0.99]` gives a
sampled peak spatial gradient of approximately `11.873` at `t = 0.515`, between
the `nu = 0.1` and `nu = 0.01` cases. These checks do not independently certify
numerical accuracy or bound continuous gradient maxima.

From the repository root, with the
[documented environment](../../../../README.md#setup-and-execution), execute:

```sh
venv/bin/python -m scripts.run_experiments.run_burgers_sigma_tuning
```

## Tuning protocol

The [five YAMLs](../../../../config/burgers/nu_0p031415927/) retain the shared
network, optimizer schedule, point budgets, and evaluation grids. Compare
ordinary PACMANN with diffusion at `1e-5`, `1e-4`, `1e-3`, and `1e-2`, using
paired development seeds 3000-3002: 15 initial runs. The runner executes three
seeds concurrently per candidate and candidates sequentially, with one numerical
CPU thread per worker. Hardware and dependency provenance will be recorded with
the executed runs.

Select the positive sigma with the lowest arithmetic mean final solution
relative L2 over all three seeds; exact ties favor the smaller sigma. Sigma
zero is the comparison baseline only. If a positive boundary wins, extend that
side by one and two decades, retaining completed candidates. Permit at most two
expansion rounds, then freeze the best tested positive value and disclose any
remaining boundary limitation. Preserve failures for review rather than ranking
only successful seeds. Evaluate residual error at the same solution-selected
sigma. A separate ten-seed evaluation requires frozen settings after tuning.

For a read-only summary, select `ACTION = "summarize"` in the tuning script.
Outputs use `output/burgers/nu_0p031415927/sigma_*/<method>/seed_<seed>/`.
See [study context](../../../../docs/research_context.md),
[results](../../../../docs/results.md), and
[artifact meanings](../../../../docs/output_format.md).
