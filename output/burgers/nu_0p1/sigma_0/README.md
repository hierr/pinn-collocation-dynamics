# Burgers diffusion tuning at nu = 0.1

The 21-run development sweep completed without numerical failures. It compared
ordinary PACMANN (sigma = 0) with six positive settings on paired seeds
3000-3002. The first lower-boundary expansion added 1e-7 and 1e-6 after
1e-5 won the initial grid at its lower boundary. The final positive winner,
sigma = 1e-5, is interior among the seven tested positive values. The protocol
permits at most two boundary-expansion rounds; only the first was needed. Sigma
zero is a comparison baseline only. Select the positive value with the lowest
mean final solution relative L2 over all three seeds; exact ties favor the
smaller positive sigma.

## Protocol and reproduction

All configurations use the reference
data/burgers_dataset/burgers_nu_0p1_N1024.npz (stored viscosity 0.1), a
1024-by-199 solution grid, and a 1024-by-200 residual grid. The reference was
transferred unchanged from dataset commit dbeb289; structural and loader checks
do not independently certify its numerical accuracy at this viscosity.

The [seven YAML configurations](../../../../config/burgers/nu_0p1/) share the network,
optimizer, point budgets, evaluation grids, and seeds. From the repository
root, with the documented environment active, run:

    python -m scripts.run_experiments.run_burgers_sigma_tuning

The runner executes three seeds concurrently per candidate and candidates
sequentially on one NVIDIA GeForce RTX 5070 (12 GB), using float64 and one
numerical CPU thread per worker. To summarize archived runs without training,
set ACTION = "summarize" in the runner. Each candidate README links back here;
the full per-seed and aggregate outcomes follow.

## Solution error and executed training time

Solution relative L2 is reported in percent. Paired changes are candidate minus
the same-seed sigma-zero error in percentage points; negative values favor the
candidate. Training time is the mean recorded seconds per run; seeds within a
candidate ran concurrently.

| Sigma | Seed 3000 | Seed 3001 | Seed 3002 | Mean ± sample SD (%) | Paired change vs 0 (pp) | Improved seeds | Mean training time (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0169006 | 0.0109887 | 0.0252385 | 0.0177093 ± 0.00715926 | - | - | 9344.3 |
| 1e-7 | 0.0135284 | 0.0114627 | 0.0118461 | 0.0122791 ± 0.00109882 | -0.00543022 | 2/3 | 8951.1 |
| 1e-6 | 0.0116698 | 0.0163272 | 0.0108791 | 0.0129587 ± 0.00294386 | -0.00475061 | 2/3 | 9373.4 |
| **1e-5** | **0.00948322** | **0.00781694** | **0.0115330** | **0.00961106 ± 0.00186134** | **-0.00809823** | **3/3** | 9417.0 |
| 1e-4 | 0.0114115 | 0.0152684 | 0.0237857 | 0.0168219 ± 0.00633167 | -0.000887435 | 2/3 | 8736.4 |
| 1e-3 | 0.00882639 | 0.00839546 | 0.0278965 | 0.0150394 ± 0.0111366 | -0.00266984 | 2/3 | 9192.9 |
| 1e-2 | 0.0371935 | 0.0292609 | 0.0202227 | 0.0288924 ± 0.0084914 | +0.0111831 | 1/3 | 9310.8 |

The selected candidate lowers mean solution error by 45.73% relative to sigma
zero on these development seeds. The three-seed sample is small and does not
establish a held-out benefit or a result across viscosities.

## Residual error at the selected setting

Residual RMS uses the fixed interior midpoint grid. It was evaluated at the
solution-selected sigma; no separate residual selection was performed.

| Sigma | Seed 3000 | Seed 3001 | Seed 3002 | Mean ± sample SD |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00109487 | 0.000782608 | 0.00157782 | 0.00115176 ± 0.000400647 |
| 1e-7 | 0.000967790 | 0.000738979 | 0.00114608 | 0.000950949 ± 0.000204071 |
| 1e-6 | 0.000692394 | 0.000931870 | 0.000861906 | 0.000828723 ± 0.000123138 |
| **1e-5** | **0.000817199** | **0.000575135** | **0.000929151** | **0.000773829 ± 0.000180949** |
| 1e-4 | 0.000597629 | 0.00107085 | 0.00101145 | 0.000893307 ± 0.000257781 |
| 1e-3 | 0.00103128 | 0.000633777 | 0.00122422 | 0.000963093 ± 0.000301071 |
| 1e-2 | 0.00376117 | 0.00228843 | 0.00197006 | 0.00267322 ± 0.000955546 |

The selected setting has a paired mean residual-RMS change of -0.000377936
(sample SD 0.000237073) relative to sigma zero, with lower RMS in all three
seeds. This development comparison does not certify the reference numerically
or establish a general diffusion benefit. The run histories retain
actual optimizer work and timing. Results are for three-way GPU concurrency and
should not be compared directly with ten-way evaluation timing.

Training source was commit 0948ad936fb32502c19f0b174572866799ca1285.
The reference SHA-256 is
78fbdc52d18e6538668d127174d5a94ed5999ef8c295a5fa9414592f14f9a833.
See [study context](../../../../docs/research_context.md),
[results](../../../../docs/results.md), and
[artifact meanings](../../../../docs/output_format.md).
