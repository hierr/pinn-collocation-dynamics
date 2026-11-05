# PINN Collocation Dynamics — peer review

This branch is the compact class submission for the controlled study of PINN
collocation sampling. It compares five methods over five Burgers viscosities,
including PACMANN and PACMANN with Gaussian point diffusion.

Start with the [peer-review guide](docs/peer_review.md): the research questions,
method equations, study design, code-reading route, and interpretation. Then see
[results](docs/results.md) and the [presentation](docs/presentation/presentation.pdf).

## What to review

| Goal | Entry point |
| --- | --- |
| Understand the method and design | [Review guide](docs/peer_review.md) |
| Inspect the numerical implementation | [Problem](src/burgers.py), [samplers](src/sampling.py), [training](src/train.py) |
| Assess the evidence | [Results](docs/results.md), all 250 retained final records, and shared figures |
| Inspect one complete trained example | [PACMANN-Adam, seed 4000](output/burgers/burgers_nu_0p0031830989_10seed/pacmann_adam/seed_4000) |
| Run a short training demo | `scripts/run_peer_review.py` and `config/peer_review.yaml` |
| Interpret generated files | [Artifact dictionary](docs/output_format.md) |

The five modules in `src/` and the standard post-processing code are unchanged
from research checkpoint
[`6e7a638`](https://github.com/hierr/pinn-collocation-dynamics/tree/6e7a638383fe0e04d9d999a2cc491c313f587cc8).
This branch retains each evaluation run's original configuration, metadata,
status, and metrics. [PACMANN-Adam seed 4000 at nu = 0.01/pi](output/burgers/burgers_nu_0p0031830989_10seed/pacmann_adam/seed_4000) is also retained in
full. Larger artifacts, development work, and full campaign orchestration are
available at that fixed research checkpoint and the linked workstream commits.

## Getting this branch

For a compact checkout of this branch:

```bash
git clone --single-branch --branch peer-review --depth 1 \
  https://github.com/hierr/pinn-collocation-dynamics.git
cd pinn-collocation-dynamics
```

A normal full clone also downloads the earlier research history and its larger
artifacts. The review commands use the current branch's files and do not require
historical Git objects.

## Setup and execution

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). The existing lockfile pins
DeepXDE 1.14.0, PyTorch 2.7.1, NumPy 2.3.1, SciPy 1.16.0, and Matplotlib 3.10.3.
The implementation uses PyTorch backend and float64. These instructions are validated on Linux x86_64.

From the repository root:

```bash
uv venv --python 3.12 venv
VIRTUAL_ENV="$PWD/venv" uv sync --active --locked
```

**Run the short training demonstration:**

```bash
venv/bin/python -m scripts.run_peer_review
```

This is an engineering demonstration. It exercises training, point adaptation,
checkpointing, and evaluation; its errors do not reproduce the ten-seed study.
No particular accuracy or method advantage is expected from this short budget.

The script selects one method through `METHOD`, defaulting to
`pacmann_diffusion`, and one YAML through `CONFIG_PATH`. The demo uses seed 0 at
nu = 0.01/pi, 500 Adam steps, and one short L-BFGS stage. Its architecture, point
budgets, and sampler parameters match the corresponding research configuration;
training and evaluation budgets are reduced. All five sampler settings are
available in the YAML if another method is selected.


Outputs go to
`output/burgers/nu_0p0031830989/peer_review_demo/pacmann_diffusion/seed_0/`. The runner refuses to overwrite an existing run. To
repeat it, change the YAML's `campaign` identifier to `peer_review_demo_2` (and
use another suffix for subsequent attempts).

The commands use a CUDA GPU when available and CPU otherwise. To force CPU
execution, optionally prefix a Python command with `CUDA_VISIBLE_DEVICES=''`.

On an Intel Core 7 240H with Python 3.12.13, complete CPU execution took
**64.2 seconds** for the demonstration, including module imports and final
evaluation. This is an example execution time, not a scientific campaign timing.


## Contents

```text
src/                         Scientific implementation and artifact handling
scripts/run_peer_review.py   One-method training demonstration
scripts/post_processing/     Standard saved-run evaluation and diagnostics
config/                      Demo settings and five frozen study configurations
data/                        Original reference and five accepted generated references
docs/                        Review guide, results, artifact dictionary, and figures
output/                      Compact records for all 250 runs and one complete example
references/                  Immutable PACMANN source and source attribution
tests/                       Numerical and execution checks
```
