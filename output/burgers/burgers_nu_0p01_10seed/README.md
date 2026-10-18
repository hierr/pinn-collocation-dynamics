# Burgers ν = 0.01: frozen ten-seed evaluation

**Completed evaluation: all five methods and all 50 planned runs.** Fixed
Hammersley, random resampling, RAD, PACMANN-Adam, and PACMANN with diffusion
each have ten seeds and complete evaluation artifacts. The
[configuration](../../../config/burgers_nu_0p01_10seed.yaml) freezes the five
methods and ten paired evaluation seeds. The three baseline methods record
revision `c615f0436e00bdb82762f78a20fb49c4d74e7b60`; the two PACMANN methods
record `6582fa759bceff7b84b1b742ade8d7a75d56fd17`. The intervening revision
changed the runner selection and documentation, while `src/` and `config/`
remained unchanged.

## Design

Evaluate fixed Hammersley, upstream random resampling, RAD, PACMANN, and PACMANN
with diffusion at **ν = 0.01**, using
`data/burgers_dataset/burgers_nu_0p01_N1024.npz`. Diffusion uses **sigma = 1e-3**,
selected on development seeds 3000–3002 in `hyperparameter-tuning` commit
`2baa9dd`. The positive search winner was interior, so no expansion was needed.
See the [development result](../../../docs/results.md#diffusion-development-selection-at-nu-001).

Use evaluation seeds 4000–4009 for every method, excluding the tuning seeds.
These seed identifiers also occur in the earlier viscosity campaign; retain
them all, including unfavorable outcomes. Network, point budgets, fixed
constraints, optimizer stages, baseline sampling settings, evaluation grids,
and reporting rules match the
[shared comparison design](../burgers_nu_0p0031830989_10seed/README.md#design-and-comparisons).
Only viscosity, reference dataset, and the separately selected diffusion sigma
change. Do not retune on evaluation results.

The primary contrast is diffusion versus PACMANN for final solution relative L2.
Assess the same contrast for independent residual RMS, without a separate sigma
selection. Compare adaptive methods against fixed Hammersley and random
resampling as in the shared design. Report all per-seed outcomes, means, sample
SD, paired differences and improvement counts; supplementary trimmed means
remove one minimum and one maximum separately for solution L2 and residual RMS.
Boundary and initial-condition RMSE are constraint diagnostics. Report actual
optimizer work and timing, retaining numerical failures and all planned seeds.

This campaign evaluates held-out seeds at one viscosity. Comparison with the
earlier viscosity does not by itself establish a sharpness-dependent benefit;
explicit sharpness and feature-region evaluation remain pending. The generated
reference passed structural and loader checks, without independent numerical
accuracy certification at this viscosity. Baseline samplers retain upstream
settings while diffusion is locally tuned; disclose that tuning policy.

## Reproduction

Use the [documented environment](../../../README.md#setup-and-execution).
Select `config/burgers_nu_0p01_10seed.yaml` through `CONFIG_PATH` and the
required method batches through `METHODS` in the runner. The committed runner
selects the two PACMANN methods; the three baseline method directories are
already retained in this campaign. For a full repetition, select all five
methods in the order listed above and use a copy of the YAML with a new campaign
name.
Execute from the repository root:

```sh
venv/bin/python -m scripts.run_experiments.run_burgers_campaign
```

Each method launches ten concurrent seed processes, each with one numerical CPU
thread, sharing one GPU. The next method starts after all workers finish
training and final evaluation. Use the same execution conditions as the earlier
campaign: NVIDIA GeForce RTX 5070 (12 GB), float64. Timing includes compilation
and training, excludes final evaluation, and reflects shared-GPU concurrency.
It is not comparable to the three-way-concurrent development timings.

Set the runner's `METHODS` to a subset to stop after selected method batches.
The runner rejects occupied seed destinations and stops after a failed batch's
other workers finish. Preserve artifacts and follow the
[recovery policy](../../../AGENTS.md#execution-recovery). For a full repetition,
copy the YAML with a new campaign name and select that copy in the runner;
keep scientific settings fixed.

Outputs are under `<method>/seed_<seed>/` alongside this README. Per-seed and
aggregate results for all five methods follow; consolidated comparisons are in
[results](../../../docs/results.md#frozen-evaluation-at-nu-001). See
[study context](../../../docs/research_context.md#reference-validation-and-current-evidence)
and the [artifact dictionary](../../../docs/output_format.md).

## Verified outcomes

All 50 retained runs completed without numerical failures. Independent CPU
evaluation reproduced the five saved error metrics for all runs, including all
20 PACMANN checkpoints completed in this phase. Initial interior designs and
constraint coordinates match exactly across methods within each seed.
Constraints remain fixed, and physics points do not change during any L-BFGS
stage.

L2 is expressed as a percentage. SD is the sample standard deviation over all
ten planned seeds. Supplementary trimmed means remove one minimum and one
maximum independently for solution L2 and residual RMS; they do not replace
the full-seed summaries. Training time includes compilation and optimizer work,
excluding final evaluation. Actual optimizer work is retained in `training.json`.

### [Fixed Hammersley](fixed_hammersley/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 1.55786e-09 | 0.006568 | 0.000717407 | 5.31879e-05 | 9.64406e-05 | 9860.8 |
| 4001 | 8.54823e-09 | 0.015385 | 0.000975933 | 0.000140406 | 0.000272621 | 10014.4 |
| 4002 | 3.68692e-09 | 0.010104 | 0.000901633 | 9.71121e-05 | 0.00019483 | 9614.9 |
| 4003 | 9.54087e-09 | 0.016254 | 0.00121153 | 0.000125287 | 0.000204179 | 10038.7 |
| 4004 | 3.61468e-09 | 0.010005 | 0.000982779 | 0.000103999 | 0.00014496 | 9975.0 |
| 4005 | 3.93917e-09 | 0.010444 | 0.000911176 | 7.93961e-05 | 0.00017565 | 9952.4 |
| 4006 | 5.91946e-09 | 0.012803 | 0.000860622 | 9.89826e-05 | 0.000234447 | 10009.6 |
| 4007 | 3.62135e-09 | 0.010014 | 0.000826064 | 8.57494e-05 | 0.000217292 | 9953.0 |
| 4008 | 5.03032e-09 | 0.011802 | 0.000871417 | 0.00014071 | 0.000207061 | 10024.9 |
| 4009 | 5.36481e-09 | 0.012188 | 0.000906525 | 0.000108764 | 0.000204948 | 10014.9 |
| Mean ± SD | 5.08237e-09 ± 2.41842e-09 | 0.0115567 ± 0.00282369 | 0.000916508 ± 0.000128360 | 0.000103359 ± 0.0000273611 | 0.000195243 ± 0.0000483028 | 9945.9 ± 127.4 |
| Trimmed mean | — | 0.0115932 | 0.000904519 | — | — | — |

Sampling updates: zero per seed.

### [Random resampling](random_resampling/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 1.10435e-08 | 0.017487 | 0.00141039 | 0.000154216 | 0.000223996 | 10063.0 |
| 4001 | 1.18661e-08 | 0.018127 | 0.00135034 | 0.000172743 | 0.000206285 | 10098.9 |
| 4002 | 1.30291e-08 | 0.018994 | 0.00122969 | 9.74498e-05 | 0.000133886 | 10072.9 |
| 4003 | 1.26713e-08 | 0.018732 | 0.00102766 | 0.000132954 | 0.000208636 | 9787.0 |
| 4004 | 1.71057e-08 | 0.021764 | 0.0013609 | 0.000124841 | 0.000209764 | 9843.0 |
| 4005 | 1.19514e-08 | 0.018192 | 0.00139559 | 0.00013335 | 0.000220234 | 10114.5 |
| 4006 | 9.74878e-09 | 0.016430 | 0.00090242 | 8.06021e-05 | 0.000164535 | 9981.5 |
| 4007 | 9.64829e-09 | 0.016345 | 0.0011039 | 7.5171e-05 | 0.00023393 | 10066.2 |
| 4008 | 3.63542e-08 | 0.031728 | 0.00121578 | 0.000184118 | 0.00017566 | 10076.8 |
| 4009 | 1.06543e-08 | 0.017176 | 0.00106469 | 0.000133716 | 0.000195837 | 9957.5 |
| Mean ± SD | 1.44072e-08 ± 8.00069e-09 | 0.0194975 ± 0.00456948 | 0.00120614 ± 0.000175615 | 0.000128916 ± 0.0000363573 | 0.000197276 ± 0.0000307504 | 10006.1 ± 112.7 |
| Trimmed mean | — | 0.0183627 | 0.00121857 | — | — | — |

Sampling updates: 700 per seed.

### [RAD](rad/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 1.97284e-08 | 0.023373 | 0.00110785 | 0.000138692 | 0.000246656 | 9073.4 |
| 4001 | 1.35694e-08 | 0.019384 | 0.000925153 | 0.000123554 | 0.000267904 | 9320.4 |
| 4002 | 9.1463e-09 | 0.015914 | 0.000857204 | 0.000119156 | 0.000185808 | 9337.0 |
| 4003 | 6.19486e-09 | 0.013097 | 0.000912479 | 0.000114644 | 0.000154414 | 9448.9 |
| 4004 | 7.87385e-09 | 0.014766 | 0.000779329 | 0.000100216 | 0.000229829 | 9318.8 |
| 4005 | 6.37352e-09 | 0.013285 | 0.000824063 | 0.000102557 | 0.00014942 | 9418.7 |
| 4006 | 1.02606e-08 | 0.016856 | 0.000821751 | 0.000131393 | 0.00023252 | 9251.1 |
| 4007 | 5.27253e-09 | 0.012083 | 0.000880522 | 6.4301e-05 | 0.000176795 | 9443.3 |
| 4008 | 2.19451e-08 | 0.024651 | 0.000834613 | 0.000117867 | 0.000177876 | 9494.0 |
| 4009 | 1.29777e-08 | 0.018957 | 0.000959236 | 0.000164745 | 0.000218937 | 9451.6 |
| Mean ± SD | 1.13342e-08 ± 5.73904e-09 | 0.0172366 ± 0.00431413 | 0.000890220 ± 0.0000939736 | 0.000117713 ± 0.0000264021 | 0.000204016 ± 0.0000405491 | 9355.7 ± 125.7 |
| Trimmed mean | — | 0.0169540 | 0.000876878 | — | — | — |

Sampling updates: 700 per seed.

### [PACMANN without diffusion](pacmann_adam/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 1.099598e-08 | 0.0174495 | 0.001523533 | 0.0001802144 | 0.0001964841 | 10167.5 |
| 4001 | 1.297004e-07 | 0.05992893 | 0.001853498 | 0.0001262160 | 0.0002289090 | 10228.7 |
| 4002 | 6.196273e-08 | 0.04142201 | 0.002416573 | 0.0001263319 | 0.0002090710 | 10136.6 |
| 4003 | 1.624915e-08 | 0.02121198 | 0.001979454 | 0.0000719378 | 0.0001615769 | 10021.5 |
| 4004 | 2.667047e-08 | 0.02717573 | 0.001955405 | 0.0000969956 | 0.0001411954 | 10078.3 |
| 4005 | 1.242768e-08 | 0.01855073 | 0.001554052 | 0.0001006231 | 0.0001876612 | 10238.4 |
| 4006 | 5.094089e-08 | 0.03755771 | 0.001671433 | 0.0000724619 | 0.0001975204 | 10224.2 |
| 4007 | 6.280506e-08 | 0.04170260 | 0.001787265 | 0.0001381890 | 0.0003216585 | 10241.4 |
| 4008 | 3.215152e-08 | 0.02983779 | 0.001628041 | 0.0001252408 | 0.0002114375 | 10211.6 |
| 4009 | 3.157047e-08 | 0.02956694 | 0.001642684 | 0.0001197628 | 0.0002898054 | 10155.2 |
| Mean ± SD | 4.35474e-08 ± 3.57745e-08 | 0.0324404 ± 0.0130586 | 0.00180119 ± 0.000268183 | 0.000115797 ± 0.0000322503 | 0.000214532 ± 0.0000546788 | 10170.3 ± 74.3 |
| Trimmed mean | — | 0.0308782 | 0.00175898 | — | — | — |

Sampling updates: 700 per seed.

### [PACMANN with diffusion](pacmann_diffusion/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 1.998805e-08 | 0.02352616 | 0.002143818 | 0.0002316155 | 0.0003517909 | 10340.8 |
| 4001 | 1.051858e-08 | 0.01706649 | 0.001002463 | 0.0001369017 | 0.0002205528 | 10230.0 |
| 4002 | 2.888123e-08 | 0.02827963 | 0.002129822 | 0.0002237226 | 0.0003039760 | 10190.5 |
| 4003 | 3.713147e-08 | 0.03206542 | 0.001762671 | 0.0000690758 | 0.0002229725 | 10350.1 |
| 4004 | 4.901187e-08 | 0.03683973 | 0.001892818 | 0.0002245138 | 0.0003071160 | 10325.2 |
| 4005 | 1.208667e-08 | 0.01829444 | 0.001209751 | 0.0000954426 | 0.0002192400 | 10354.8 |
| 4006 | 9.238193e-09 | 0.01599408 | 0.001223920 | 0.0001001557 | 0.0001928588 | 10344.4 |
| 4007 | 5.151344e-08 | 0.03776819 | 0.001318669 | 0.0001351416 | 0.0002390373 | 10206.3 |
| 4008 | 2.231121e-08 | 0.02485578 | 0.001472758 | 0.0002289501 | 0.0003766330 | 10313.5 |
| 4009 | 1.842975e-08 | 0.02259048 | 0.001552828 | 0.0001852353 | 0.0002823205 | 10132.4 |
| Mean ± SD | 2.59110e-08 ± 1.53980e-08 | 0.0257280 ± 0.00785700 | 0.00157095 ± 0.000398408 | 0.000163075 ± 0.0000630832 | 0.000271650 ± 0.0000622357 | 10278.8 ± 81.1 |
| Trimmed mean | — | 0.0254398 | 0.00157040 | — | — | — |

Sampling updates: 700 per seed.

### Paired diffusion contrast and optimizer work

For the paired contrast, differences are PACMANN with diffusion minus PACMANN
without diffusion. Solution L2 differences are percentage points. Negative
values favor diffusion; counts are improved / tied / worse.

| Metric | Mean paired difference ± sample SD | Improved / tied / worse |
| --- | ---: | ---: |
| Solution relative L2 | -0.006712 percentage points ± 0.016198 | 7 / 0 / 3 |
| Residual RMS | -0.00023024 ± 0.00037751 | 9 / 0 / 1 |

Every run performed 35,000 Adam iterations. The L-BFGS driver can exceed its
configured per-stage iteration and function-evaluation limits, so the table
reports observed per-seed totals from `training.json`. Training seconds include
compilation and optimizer work, and exclude final evaluation.

| Method | L-BFGS iterations per seed, mean ± sample SD | L-BFGS function evaluations per seed, mean ± sample SD |
| --- | ---: | ---: |
| Fixed Hammersley | 16322.7 ± 348.6 | 25513.6 ± 1698.4 |
| Random resampling | 16603.6 ± 342.7 | 25799.7 ± 1379.3 |
| RAD | 15735.5 ± 506.9 | 19807.0 ± 1944.8 |
| PACMANN without diffusion | 16344.8 ± 276.0 | 24786.2 ± 904.8 |
| PACMANN with diffusion | 16325.0 ± 476.6 | 25666.4 ± 882.2 |

The comparisons are limited to this generated reference and viscosity.
The selected diffusion setting shows lower mean errors than PACMANN without
diffusion on these ten seeds, but does not establish a general diffusion benefit
or a sharpness-dependent effect. The reference passed structural and loader
checks without independent numerical-accuracy certification at this viscosity.
The three baseline sampling methods retain upstream settings while diffusion
was tuned on separate development seeds.
