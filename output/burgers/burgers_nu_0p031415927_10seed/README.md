# Burgers ν = π/100: frozen ten-seed evaluation

**Complete evaluation: all five methods and all 50 planned runs.** The
comparison uses paired seeds 4000–4009 and frozen **σ =
1e-4**, selected on separate development seeds 3000–3002. The [tuning
checkpoint](https://github.com/hierr/pinn-collocation-dynamics/commit/e089cb9d02def773fa16b9811e2f2f6ee5c59a7c)
contains all 15 initial-grid runs and selected the lowest-mean positive
candidate; its winner was interior, so no expansion was needed. Sigma zero was
a comparison baseline only, although it had lower mean solution error than all
positive candidates. See the [selection results](../../../docs/results.md#diffusion-selection-at-pi100)
and the [canonical tuning README](https://github.com/hierr/pinn-collocation-dynamics/blob/e089cb9d02def773fa16b9811e2f2f6ee5c59a7c/output/burgers/nu_0p031415927/sigma_0/README.md).
The tuning runs used source revision
[`cb6c6e8`](https://github.com/hierr/pinn-collocation-dynamics/commit/cb6c6e85642b82a70ce9e64013b74cd5633788ff).

## Design and verification

The [frozen configuration](../../../config/burgers_nu_0p031415927_10seed.yaml)
compares fixed Hammersley, upstream random resampling, RAD, PACMANN, and
PACMANN with diffusion. All methods retain the shared network, point budgets,
fixed boundary and initial constraints, optimizer schedule, and evaluation
grids. Only viscosity, reference dataset, and the development-selected sigma
differ from the preceding frozen comparison. Evaluation seeds are disjoint
from tuning seeds.

The N1024 reference
[data/burgers_dataset/burgers_nu_0p031415927_N1024.npz](../../../data/burgers_dataset/burgers_nu_0p031415927_N1024.npz)
was restored unchanged from dataset commit `52b9359`. Its stored viscosity is
**π/100 = 0.031415926535897934**, and its SHA-256 is
`aad9aff8eecc32e1fe9e60ad2e4ca5f399dd5e96bf748851f085a2165923fb60`.
Structural, constraint, and loader checks passed, but the reference has no
independent numerical-accuracy certification.

All 50 runs completed training and final evaluation with finite outcomes and
no recorded numerical failures. Every run records training source revision
[`09b380e`](https://github.com/hierr/pinn-collocation-dynamics/commit/09b380e147d9367ac0f9bd427e78f853dd22b984),
the same pinned software versions, CUDA device, and reference fingerprint.
Independent CPU evaluation of all final checkpoints reproduced solution MSE,
solution relative L2, residual RMS, boundary RMSE, and initial RMSE within
1e-12 + 1e-10·|saved|. Initial point designs and constraint coordinates match
across methods for each seed. All snapshots retain 2,500 interior points and
240 fixed constraints; physics points remain unchanged during L-BFGS.

Use final-model solution and independent residual errors on all ten seeds.
Retain every planned outcome. Report per-seed values, mean and sample SD,
paired differences and improvement counts. Supplementary trimmed means remove
one minimum and one maximum independently for solution L2 and residual RMS.
Solution relative L2 is reported as percent; paired solution differences are
percentage points; residual RMS retains its original units. Boundary and
initial-condition RMSE are separate diagnostics. Residual error uses the same
selected sigma; do not retune on evaluation seeds.

## Reproduction

Use the [documented environment](../../../README.md#setup-and-execution).
The runner selects this YAML and methods in the order listed above. From the
repository root, execute:

```sh
venv/bin/python -u -m scripts.run_experiments.run_burgers_campaign
```

The runner launches ten concurrent seeds per method, with one numerical CPU
thread per worker, on one NVIDIA GeForce RTX 5070 (12 GB) using float64.
Methods run sequentially; each batch completes training and evaluation before
the next begins. Training time includes compilation and optimizer work and
excludes final evaluation. It reflects ten-way GPU concurrency and is not
comparable to three-way tuning timings.

Runs are retained under
`output/burgers/burgers_nu_0p031415927_10seed/<method>/seed_<seed>/`.
Consolidated comparisons are in [results](../../../docs/results.md);
see the [study context](../../../docs/research_context.md) and
[artifact dictionary](../../../docs/output_format.md). Per-seed outcomes and
aggregates follow below.

## Per-seed outcomes

### [Fixed Hammersley](fixed_hammersley/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 1.28211e-09 | 0.0064221 | 0.000523284 | 0.000093690 | 0.000132867 | 10316.3 |
| 4001 | 2.11404e-09 | 0.0082466 | 0.000631318 | 0.000094506 | 0.000153032 | 10296.7 |
| 4002 | 8.94805e-10 | 0.0053651 | 0.000567997 | 0.000080465 | 0.000111105 | 10346.4 |
| 4003 | 4.07423e-09 | 0.0114482 | 0.000650962 | 0.000115901 | 0.000296624 | 10208.1 |
| 4004 | 2.56648e-09 | 0.0090863 | 0.000784958 | 0.000090372 | 0.000167901 | 10347.8 |
| 4005 | 2.80794e-09 | 0.0095041 | 0.000641819 | 0.000141881 | 0.000203663 | 10216.2 |
| 4006 | 1.98828e-09 | 0.0079975 | 0.000715249 | 0.000086451 | 0.000160749 | 10200.5 |
| 4007 | 4.72048e-09 | 0.0123228 | 0.000831065 | 0.000169393 | 0.000225581 | 10216.1 |
| 4008 | 4.55617e-09 | 0.0121064 | 0.000704624 | 0.000112043 | 0.000294968 | 10310.8 |
| 4009 | 5.89282e-09 | 0.0137682 | 0.000711756 | 0.000176160 | 0.000266152 | 10345.9 |
| Mean ± sample SD | 3.08974e-09 ± 1.6411e-09 | 0.00962674 ± 0.00273221 | 0.000676303 ± 0.000093294 | 0.000116086 ± 0.000034813 | 0.000201264 ± 0.000067139 | 10280.5 ± 62.8 |
| 10% trimmed mean | — | 0.00964176 | 0.000676085 | — | — | — |

Sampling updates: 0 per seed.

### [Random resampling](random_resampling/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 3.87537e-09 | 0.0111654 | 0.000910021 | 0.000163157 | 0.000174591 | 10081.3 |
| 4001 | 5.93097e-09 | 0.0138127 | 0.001002285 | 0.000167142 | 0.000218927 | 10184.7 |
| 4002 | 1.20034e-09 | 0.0062140 | 0.000665315 | 0.000070536 | 0.000116226 | 10133.8 |
| 4003 | 7.51416e-09 | 0.0155474 | 0.000954439 | 0.000173796 | 0.000303780 | 10182.4 |
| 4004 | 2.80946e-09 | 0.0095067 | 0.000675354 | 0.000097177 | 0.000165662 | 10005.4 |
| 4005 | 3.75975e-09 | 0.0109975 | 0.000625252 | 0.000111167 | 0.000251961 | 10093.1 |
| 4006 | 3.86635e-09 | 0.0111524 | 0.000696571 | 0.000090717 | 0.000202477 | 10119.4 |
| 4007 | 1.68531e-09 | 0.0073630 | 0.000606942 | 0.000092383 | 0.000163299 | 10183.1 |
| 4008 | 5.06687e-09 | 0.0127669 | 0.000720414 | 0.000151731 | 0.000298263 | 10108.5 |
| 4009 | 9.14267e-09 | 0.0171496 | 0.000819416 | 0.000206058 | 0.000297772 | 10155.8 |
| Mean ± sample SD | 4.48512e-09 ± 2.4948e-09 | 0.01156755 ± 0.00341113 | 0.000767601 ± 0.000143477 | 0.000132386 ± 0.000045367 | 0.000219296 ± 0.000066262 | 10124.7 ± 56.6 |
| 10% trimmed mean | — | 0.01153899 | 0.000758348 | — | — | — |

Sampling updates: 700 per seed.

### [RAD](rad/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 6.33421e-09 | 0.0142745 | 0.000822465 | 0.000220906 | 0.000248401 | 10503.1 |
| 4001 | 4.60735e-09 | 0.0121742 | 0.000776819 | 0.000146998 | 0.000166403 | 10506.1 |
| 4002 | 3.01236e-09 | 0.0098440 | 0.000532292 | 0.000136031 | 0.000198322 | 10548.3 |
| 4003 | 4.71046e-09 | 0.0123097 | 0.000735430 | 0.000112488 | 0.000209631 | 10512.0 |
| 4004 | 2.99674e-09 | 0.0098184 | 0.000641446 | 0.000117061 | 0.000166361 | 10439.5 |
| 4005 | 2.89205e-09 | 0.0096454 | 0.000705778 | 0.000171729 | 0.000152787 | 10529.8 |
| 4006 | 5.45274e-09 | 0.0132441 | 0.000801655 | 0.000160702 | 0.000247352 | 10374.0 |
| 4007 | 2.41813e-09 | 0.0088197 | 0.000450687 | 0.000087205 | 0.000203154 | 10096.2 |
| 4008 | 5.49018e-09 | 0.0132895 | 0.000701574 | 0.000125942 | 0.000254896 | 9907.7 |
| 4009 | 4.27356e-09 | 0.0117250 | 0.000654761 | 0.000095639 | 0.000214358 | 10232.8 |
| Mean ± sample SD | 4.21878e-09 ± 1.3317e-09 | 0.01151446 ± 0.00186473 | 0.000682291 ± 0.000117920 | 0.000137470 ± 0.000039746 | 0.000206166 ± 0.000036544 | 10364.9 ± 217.5 |
| 10% trimmed mean | — | 0.01150629 | 0.000693719 | — | — | — |

Sampling updates: 700 per seed.

### [PACMANN](pacmann_adam/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 6.03646e-09 | 0.0139350 | 0.001263674 | 0.000079023 | 0.000156230 | 10704.5 |
| 4001 | 7.38873e-09 | 0.0154170 | 0.001273494 | 0.000099726 | 0.000185850 | 10706.6 |
| 4002 | 8.58907e-09 | 0.0166222 | 0.001274321 | 0.000108622 | 0.000194609 | 10674.4 |
| 4003 | 1.72107e-08 | 0.0235297 | 0.001083700 | 0.000099282 | 0.000257198 | 10538.3 |
| 4004 | 2.17219e-08 | 0.0264341 | 0.001455458 | 0.000190488 | 0.000198194 | 10640.8 |
| 4005 | 8.56565e-09 | 0.0165996 | 0.000980867 | 0.000129489 | 0.000199772 | 10696.7 |
| 4006 | 4.60323e-09 | 0.0121688 | 0.000946430 | 0.000119165 | 0.000257122 | 10667.9 |
| 4007 | 4.19068e-09 | 0.0116107 | 0.000880040 | 0.000133388 | 0.000190218 | 10605.7 |
| 4008 | 7.25970e-09 | 0.0152818 | 0.001150439 | 0.000103889 | 0.000293734 | 10311.3 |
| 4009 | 7.84517e-09 | 0.0158861 | 0.000834797 | 0.000121941 | 0.000289692 | 10683.6 |
| Mean ± sample SD | 9.34113e-09 ± 5.6437e-09 | 0.01674851 ± 0.00471155 | 0.001114322 ± 0.000202999 | 0.000118501 ± 0.000030052 | 0.000222262 ± 0.000047904 | 10623.0 ± 121.3 |
| 10% trimmed mean | — | 0.01618003 | 0.001106621 | — | — | — |

Sampling updates: 700 per seed.

### [PACMANN + diffusion](pacmann_diffusion/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 2.21604e-09 | 0.0084432 | 0.000900044 | 0.000087418 | 0.000108184 | 10572.3 |
| 4001 | 7.93977e-09 | 0.0159816 | 0.001245002 | 0.000136768 | 0.000237584 | 10686.3 |
| 4002 | 1.19771e-08 | 0.0196288 | 0.001367532 | 0.000159018 | 0.000275181 | 10708.7 |
| 4003 | 1.06400e-08 | 0.0185006 | 0.001261939 | 0.000089296 | 0.000309248 | 10701.0 |
| 4004 | 1.90917e-08 | 0.0247821 | 0.001618625 | 0.000188541 | 0.000213257 | 10688.0 |
| 4005 | 8.19606e-09 | 0.0162375 | 0.000935418 | 0.000139303 | 0.000199068 | 10679.4 |
| 4006 | 4.53313e-09 | 0.0120758 | 0.000871158 | 0.000123314 | 0.000230580 | 10648.0 |
| 4007 | 6.41602e-09 | 0.0143664 | 0.001099130 | 0.000250143 | 0.000260864 | 10569.8 |
| 4008 | 5.80832e-09 | 0.0136692 | 0.000936990 | 0.000098620 | 0.000288481 | 10385.9 |
| 4009 | 7.03085e-09 | 0.0150390 | 0.001087881 | 0.000126611 | 0.000240808 | 10532.0 |
| Mean ± sample SD | 8.38490e-09 ± 4.6916e-09 | 0.01587242 ± 0.00444685 | 0.001132372 ± 0.000241172 | 0.000139903 ± 0.000049851 | 0.000236326 ± 0.000056318 | 10617.1 ± 102.8 |
| 10% trimmed mean | — | 0.01568737 | 0.001104242 | — | — | — |

Sampling updates: 700 per seed.

Every run used 35,000 Adam steps. Observed L-BFGS work per seed is reported
as mean ± sample SD; shared configured limits do not imply equal work.

| Method | L-BFGS iterations | L-BFGS function evaluations |
| --- | ---: | ---: |
| Fixed Hammersley | 16878.8 ± 350.3 | 27223.0 ± 1997.6 |
| Random resampling | 16888.1 ± 422.0 | 26128.7 ± 2015.8 |
| RAD | 16774.5 ± 735.4 | 26630.4 ± 3128.2 |
| PACMANN | 17223.9 ± 391.2 | 27478.0 ± 1613.3 |
| PACMANN + diffusion | 16971.7 ± 465.9 | 27752.0 ± 1186.1 |

Fixed Hammersley has the lowest mean solution error and residual RMS.
Diffusion lowers PACMANN's mean solution error, improving seven of ten paired
seeds, but raises its mean residual RMS, improving five of ten. These outcomes
do not establish a consistent diffusion benefit or isolate a sharpness effect.
See the [study-level comparison](../../../docs/results.md#frozen-evaluation-at-pi100)
for consolidated outcomes and paired contrasts. The reference-accuracy
limitation remains material.
