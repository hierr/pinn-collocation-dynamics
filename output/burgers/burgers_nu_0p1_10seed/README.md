# Burgers ν = 0.1: frozen ten-seed evaluation

**Complete evaluation: all five methods and all 50 planned runs.** The
comparison uses paired seeds 4000–4009 and frozen diffusion **σ = 1e-5**.
This positive sigma was selected on separate development seeds 3000–3002 in
the [tuning checkpoint](https://github.com/hierr/pinn-collocation-dynamics/commit/1968b350a5d44adc9180782406357c2c8763be92);
ordinary PACMANN at sigma zero was only its comparison baseline. The selected
value was interior among tested positive settings after one approved lower
boundary expansion.

## Design and verification

The [frozen configuration](../../../config/burgers_nu_0p1_10seed.yaml)
compares fixed Hammersley, random resampling, RAD, PACMANN-Adam, and PACMANN
with diffusion. All methods use the same network, matched point budgets, fixed
boundary and initial constraints, optimizer schedule, and evaluation grids.
Only viscosity, reference dataset, and the separately selected sigma differ
from the preceding frozen comparison. The reference is
[data/burgers_dataset/burgers_nu_0p1_N1024.npz](../../../data/burgers_dataset/burgers_nu_0p1_N1024.npz),
with SHA-256
`78fbdc52d18e6538668d127174d5a94ed5999ef8c295a5fa9414592f14f9a833`.
It passed structural and loader checks, but has no independent numerical-
accuracy certification at this viscosity.

Run outcomes and archived metadata show that all 50 training and post-
processing jobs completed without numerical failures. The training source
revision recorded in every run is
[`e261b37`](https://github.com/hierr/pinn-collocation-dynamics/commit/e261b3720062491aad5898603a0d6b3826211dfe);
all runs used the same pinned environment, CUDA device, and reference
fingerprint. An independent CPU evaluation reloaded every final checkpoint and
reproduced solution MSE, solution relative L2, residual RMS, boundary RMSE, and
initial RMSE within 1e-12 + 1e-10·|saved|. Initial interior point designs and
constraint coordinates match across methods for each seed. Constraints stayed
fixed, and physics points did not move during L-BFGS stages.

Use final-model errors on all ten seeds. Retain every outcome. Tables report
solution relative L2 as percent, residual RMS in its original units, and
boundary and initial-condition RMSE separately. Means use all ten seeds and
sample SD. Supplementary trimmed means remove one minimum and one maximum
independently for solution L2 and residual RMS. Paired solution differences are
percentage points. Residual error does not select a separate sigma.

## Reproduction

Use the [documented environment](../../../README.md#setup-and-execution).
From the repository root, run:

```sh
venv/bin/python -u -m scripts.run_experiments.run_burgers_campaign
```

The runner uses the configuration above and launches ten concurrent seeds per
method, one numerical CPU thread per worker, on one NVIDIA GeForce RTX 5070
(12 GB) in float64. Method batches run sequentially; each batch finishes
training and evaluation before the next begins. Training time includes
compilation and optimizer work and excludes final evaluation. It reflects
ten-way GPU concurrency and is not comparable to three-way tuning timings.
Artifacts are retained at
`output/burgers/burgers_nu_0p1_10seed/<method>/seed_<seed>/`. Consolidated
comparisons are in [results](../../../docs/results.md#frozen-evaluation-at-nu-01);
see the [study context](../../../docs/research_context.md) and
[artifact dictionary](../../../docs/output_format.md).

## Per-seed outcomes

### [Fixed Hammersley](fixed_hammersley/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 2.94362e-09 | 0.0121160 | 0.000559872878 | 0.000161672997 | 0.000227218572 | 9651.9 |
| 4001 | 8.11286e-09 | 0.0201143 | 0.000976159886 | 0.000261432037 | 0.000354244727 | 9614.6 |
| 4002 | 3.45599e-09 | 0.0131282 | 0.000625218489 | 0.000204973550 | 0.000152025343 | 9875.0 |
| 4003 | 2.77758e-09 | 0.0117693 | 0.000489903983 | 0.000139888794 | 0.000243848258 | 9770.5 |
| 4004 | 6.80606e-09 | 0.0184232 | 0.000598432688 | 0.000304637876 | 0.000249789197 | 9747.6 |
| 4005 | 1.68856e-09 | 0.0091765 | 0.000659284232 | 0.000113228498 | 0.000132883937 | 9804.8 |
| 4006 | 1.95289e-09 | 0.0098686 | 0.000550620122 | 0.000149121931 | 0.000131705200 | 9780.8 |
| 4007 | 6.27862e-09 | 0.0176950 | 0.000549410972 | 0.000260385119 | 0.000251942357 | 9833.4 |
| 4008 | 4.46195e-09 | 0.0149169 | 0.000541319803 | 0.000214617466 | 0.000198952012 | 9797.6 |
| 4009 | 4.21036e-09 | 0.0144903 | 0.000896974143 | 0.000179188793 | 0.000203518017 | 9537.4 |
| Mean ± sample SD | 4.26885e-09 ± 2.1591e-09 | 0.01416982 ± 0.0036669 | 0.0006447197 ± 0.00016194 | 0.0001989147 ± 0.000061671 | 0.0002146128 ± 0.000067479 | 9741.36 ± 106.25 |
| 10% trimmed mean | — | 0.01405093 | 0.0006226417 | — | — | — |

Sampling updates: zero per seed.

### [Random resampling](random_resampling/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 5.14384e-09 | 0.0160163 | 0.000835204339 | 0.000203563938 | 0.000225857183 | 9813.9 |
| 4001 | 1.37034e-09 | 0.0082667 | 0.000552313536 | 0.000113183252 | 0.000160883719 | 9993.0 |
| 4002 | 3.74441e-09 | 0.0136650 | 0.000718107245 | 0.000198920334 | 0.000157593361 | 9820.2 |
| 4003 | 3.82859e-09 | 0.0138177 | 0.000595657182 | 0.000186575377 | 0.000203855576 | 9624.7 |
| 4004 | 1.78845e-09 | 0.0094440 | 0.000542589833 | 0.000145883479 | 0.000181410087 | 9714.8 |
| 4005 | 1.00215e-09 | 0.0070694 | 0.000587295109 | 0.000056077188 | 0.000135638827 | 9966.6 |
| 4006 | 3.58999e-09 | 0.0133803 | 0.000665027720 | 0.000188660261 | 0.000147259462 | 9973.2 |
| 4007 | 3.79386e-09 | 0.0137549 | 0.000626748149 | 0.000184471253 | 0.000250747379 | 9988.4 |
| 4008 | 6.96028e-09 | 0.0186308 | 0.000805110683 | 0.000262948523 | 0.000258910468 | 9936.1 |
| 4009 | 2.39017e-09 | 0.0109177 | 0.000630082883 | 0.000140553638 | 0.000140230179 | 9930.7 |
| Mean ± sample SD | 3.36121e-09 ± 1.8160e-09 | 0.01249627 ± 0.0035691 | 0.0006558137 ± 0.00010099 | 0.0001680837 ± 0.000056817 | 0.0001862386 ± 0.000045954 | 9876.17 ± 127.67 |
| 10% trimmed mean | — | 0.01240782 | 0.0006475428 | — | — | — |

Sampling updates: 700 per seed.

### [RAD](rad/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 3.10195e-09 | 0.0124375 | 0.000549992733 | 0.000176100138 | 0.000188130127 | 10059.3 |
| 4001 | 3.25504e-09 | 0.0127408 | 0.000456563350 | 0.000172661166 | 0.000236835512 | 10107.5 |
| 4002 | 1.62136e-09 | 0.0089920 | 0.000485209573 | 0.000095802305 | 0.000167217416 | 10064.1 |
| 4003 | 6.45132e-09 | 0.0179367 | 0.000552564923 | 0.000266961686 | 0.000262591436 | 10190.6 |
| 4004 | 5.52994e-09 | 0.0166065 | 0.000523030255 | 0.000241005440 | 0.000218457233 | 10166.6 |
| 4005 | 2.51049e-09 | 0.0111891 | 0.000534874587 | 0.000122095527 | 0.000118809207 | 9853.4 |
| 4006 | 1.89354e-09 | 0.0097175 | 0.000478229018 | 0.000134779032 | 0.000171360768 | 9554.9 |
| 4007 | 4.63434e-09 | 0.0152024 | 0.000689340932 | 0.000202647844 | 0.000250393442 | 10144.6 |
| 4008 | 6.09766e-09 | 0.0174381 | 0.000563256942 | 0.000256666138 | 0.000253857442 | 10039.7 |
| 4009 | 3.38667e-09 | 0.0129958 | 0.000577810031 | 0.000154073610 | 0.000225005992 | 10069.3 |
| Mean ± sample SD | 3.84823e-09 ± 1.7312e-09 | 0.01352564 ± 0.0031563 | 0.0005410872 ± 0.000065500 | 0.0001822793 ± 0.000058548 | 0.0002092659 ± 0.000046488 | 10025.01 ± 189.71 |
| 10% trimmed mean | — | 0.01354096 | 0.0005331210 | — | — | — |

Sampling updates: 700 per seed.

### [PACMANN without diffusion](pacmann_adam/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 6.60794e-09 | 0.0181531 | 0.000932534863 | 0.000167960724 | 0.000265791877 | 10276.3 |
| 4001 | 2.47444e-09 | 0.0111085 | 0.000747511642 | 0.000143167819 | 0.000194942341 | 10229.2 |
| 4002 | 5.92002e-09 | 0.0171822 | 0.000819809416 | 0.000239912962 | 0.000194664970 | 10239.6 |
| 4003 | 3.36453e-09 | 0.0129533 | 0.000848912412 | 0.000177438402 | 0.000195500264 | 10141.9 |
| 4004 | 5.19839e-09 | 0.0161010 | 0.000793099612 | 0.000223398890 | 0.000212744835 | 10225.1 |
| 4005 | 8.20638e-09 | 0.0202299 | 0.000953660463 | 0.000200055310 | 0.000263766876 | 10089.6 |
| 4006 | 4.38037e-09 | 0.0147800 | 0.000940558074 | 0.000187795776 | 0.000160766074 | 10101.8 |
| 4007 | 6.18033e-09 | 0.0175559 | 0.001021060020 | 0.000157890597 | 0.000190883166 | 9872.0 |
| 4008 | 4.45538e-09 | 0.0149059 | 0.000770589082 | 0.000221389024 | 0.000220637257 | 10214.9 |
| 4009 | 4.53677e-09 | 0.0150415 | 0.001232942990 | 0.000133904139 | 0.000137043668 | 9958.2 |
| Mean ± sample SD | 5.13246e-09 ± 1.6639e-09 | 0.01580112 ± 0.0026410 | 0.0009060679 ± 0.00014575 | 0.0001852914 ± 0.000035732 | 0.0002036741 ± 0.000040150 | 10134.86 ± 132.46 |
| 10% trimmed mean | — | 0.01583411 | 0.0008850280 | — | — | — |

Sampling updates: 700 per seed.

### [PACMANN with diffusion](pacmann_diffusion/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 2.44178e-09 | 0.0110350 | 0.000841742120 | 0.000144438778 | 0.000166087610 | 10248.6 |
| 4001 | 2.36104e-09 | 0.0108510 | 0.000683268572 | 0.000131929829 | 0.000170537972 | 10261.5 |
| 4002 | 2.34285e-09 | 0.0108091 | 0.000747498335 | 0.000108040788 | 0.000135793459 | 10078.9 |
| 4003 | 3.35327e-09 | 0.0129316 | 0.000713704324 | 0.000162945054 | 0.000193668400 | 10200.6 |
| 4004 | 5.13328e-09 | 0.0159998 | 0.000805557933 | 0.000229159898 | 0.000217456109 | 9465.2 |
| 4005 | 9.76173e-09 | 0.0220638 | 0.000976194318 | 0.000211908471 | 0.000327672471 | 10175.3 |
| 4006 | 3.36863e-09 | 0.0129612 | 0.001015203520 | 0.000181239708 | 0.000119088260 | 9327.7 |
| 4007 | 8.38613e-09 | 0.0204502 | 0.000873379566 | 0.000286556034 | 0.000230723988 | 10257.5 |
| 4008 | 6.91277e-09 | 0.0185671 | 0.000727339336 | 0.000287775812 | 0.000245153266 | 10228.6 |
| 4009 | 2.30356e-09 | 0.0107181 | 0.000682529580 | 0.000140934740 | 0.000177243340 | 10120.6 |
| Mean ± sample SD | 4.63650e-09 ± 2.7830e-09 | 0.01463868 ± 0.0043371 | 0.0008066418 ± 0.00011878 | 0.0001884929 ± 0.000063469 | 0.0001983425 ± 0.000060374 | 10036.46 ± 344.03 |
| 10% trimmed mean | — | 0.01420061 | 0.0007960856 | — | — | — |

Sampling updates: 700 per seed.

Every method used 35,000 Adam steps per seed. Mean ± sample SD L-BFGS work
was 16,713.6 ± 721.2 iterations and 24,811.4 ± 2,490.6 function evaluations
for fixed Hammersley; 16,363.6 ± 720.8 and 25,333.2 ± 1,514.8 for random
resampling; 16,737.6 ± 660.3 and 24,161.7 ± 2,714.4 for RAD; 16,557.8 ±
481.8 and 25,087.1 ± 2,712.2 for PACMANN; and 17,181.9 ± 494.0 and
25,184.1 ± 1,485.6 for PACMANN with diffusion. These observed counts are
recorded in each `training.json`; they differ across methods despite shared
configured stages.

The [study-level comparison](../../../docs/results.md#frozen-evaluation-at-nu-01)
reports the paired contrasts. On these held-out seeds, diffusion had lower mean
solution L2 and residual RMS than PACMANN without diffusion, with improvement
in seven of ten pairs on both metrics. This does not establish a general
diffusion benefit. The comparison also does not isolate sharpness. The generated
reference accuracy limitation remains material.
