# Burgers: nu = 0.003183098862

This compact review record retains all five methods and all ten evaluation seeds
(4000–4009), including unfavorable outcomes. Frozen diffusion sigma is `0.0001`.
The tables below are preserved from the completed research campaign.

See the [study design](../../../docs/peer_review.md#study-design),
[comparison](../../../docs/results.md),
[frozen configuration](../../../config/burgers_nu_0p0031830989_10seed.yaml), and
[artifact dictionary](../../../docs/output_format.md).

## Reproduction and retained evidence

Every seed retains its original configuration, metadata, status, and final
metrics. Only PACMANN-Adam seed 4000 at nu = 0.01/pi retains its complete run,
including model weights and predictions. The [complete campaign](https://github.com/hierr/pinn-collocation-dynamics/tree/6e7a638383fe0e04d9d999a2cc491c313f587cc8/output/burgers/burgers_nu_0p0031830989_10seed)
and full execution instructions are available at the fixed research checkpoint.

Use the two review commands in the [repository README](../../../README.md#setup-and-execution)
for a short training demonstration and an independent saved-model check.
Full campaigns used float64 on an RTX 5070 (12 GB), with ten concurrent seeds
per method and sequential method batches. Training seconds reflect that shared
GPU condition and exclude final evaluation; they are not serial CPU timings.

## Per-seed outcomes

Solution L2 is reported in percent. Means and sample SDs use all ten seeds.
Supplementary trimmed means remove one minimum and one maximum separately
for each error metric; all seeds remain in the primary and paired comparisons.

### [Fixed Hammersley](fixed_hammersley/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 3.20795e-06 | 0.29105 | 0.0280008 | 0.000188581 | 0.000568541 | 9505.7 |
| 4001 | 1.5728e-05 | 0.64445 | 0.0437026 | 8.5356e-05 | 0.000271495 | 9562.5 |
| 4002 | 6.91911e-06 | 0.42745 | 0.0296447 | 0.000115879 | 0.000270559 | 9354.8 |
| 4003 | 2.04641e-05 | 0.73511 | 0.0323068 | 0.000108367 | 0.000220698 | 9557.2 |
| 4004 | 2.05833e-05 | 0.73725 | 0.0329556 | 7.78325e-05 | 0.000240949 | 9395.7 |
| 4005 | 8.12516e-06 | 0.46320 | 0.0245709 | 0.000110127 | 0.000289011 | 9443.4 |
| 4006 | 0.154927 | 63.96177 | 4.17012 | 0.000327142 | 0.00206593 | 9000.4 |
| 4007 | 9.7826e-07 | 0.16073 | 0.0196895 | 0.000126174 | 0.00037224 | 9572.5 |
| 4008 | 1.92201e-06 | 0.22529 | 0.0307403 | 8.66774e-05 | 0.000236557 | 9468.9 |
| 4009 | 3.52307e-06 | 0.30501 | 0.0328799 | 0.000106927 | 0.000284051 | 9346.9 |
| Mean ± SD | 0.0155009 ± 0.0489895 | 6.79513 ± 20.08737 | 0.444461 ± 1.30908 | 0.000133306 ± 7.48243e-05 | 0.000482003 ± 0.00056573 | 9420.8 ± 169.6 |
| Trimmed mean | — | 0.47860 | 0.0318502 | — | — | — |

Sampling updates: 0 per seed.
### [Random resampling](random_resampling/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 7.14072e-06 | 0.43424 | 0.0232759 | 0.000124384 | 0.000145046 | 9101.1 |
| 4001 | 1.32556e-07 | 0.05916 | 0.0146084 | 0.00011287 | 0.000202492 | 9119.8 |
| 4002 | 8.30018e-07 | 0.14805 | 0.0119498 | 8.86308e-05 | 0.000305103 | 9147.6 |
| 4003 | 1.97019e-06 | 0.22809 | 0.0200279 | 0.000121474 | 0.000223617 | 9129.6 |
| 4004 | 7.97565e-06 | 0.45892 | 0.0117573 | 0.000214293 | 0.000357311 | 9089.0 |
| 4005 | 1.19249e-06 | 0.17745 | 0.0119512 | 0.000161141 | 0.000294145 | 9140.7 |
| 4006 | 3.14563e-05 | 0.91140 | 0.0267699 | 7.42782e-05 | 0.000171368 | 8752.4 |
| 4007 | 4.07979e-05 | 1.03795 | 0.0287328 | 0.000111822 | 0.000267596 | 9047.1 |
| 4008 | 1.79641e-05 | 0.68875 | 0.0398089 | 8.54762e-05 | 0.000191216 | 8842.9 |
| 4009 | 4.25907e-06 | 0.33536 | 0.0119489 | 0.00012914 | 0.000300202 | 8727.1 |
| Mean ± SD | 1.13719e-05 ± 1.42292e-05 | 0.44794 ± 0.33274 | 0.0200831 ± 0.00952523 | 0.000122351 ± 4.08487e-05 | 0.00024581 ± 6.893e-05 | 9009.7 ± 167.6 |
| Trimmed mean | — | 0.42278 | 0.0186581 | — | — | — |

Sampling updates: 700 per seed.
### [RAD](rad/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 4.45924e-07 | 0.10851 | 0.00632936 | 0.00013951 | 0.000285945 | 8938.0 |
| 4001 | 1.28637e-07 | 0.05828 | 0.00203292 | 0.00016351 | 0.000282582 | 8911.2 |
| 4002 | 1.16694e-06 | 0.17554 | 0.0192312 | 0.000157578 | 0.00035192 | 8932.1 |
| 4003 | 8.79019e-08 | 0.04818 | 0.00199634 | 0.000112769 | 0.000428856 | 8931.4 |
| 4004 | 1.22076e-07 | 0.05678 | 0.00375718 | 0.00026298 | 0.000329297 | 8936.6 |
| 4005 | 1.2745e-07 | 0.05801 | 0.00352121 | 0.000191178 | 0.000526106 | 8917.2 |
| 4006 | 0.000474156 | 3.53848 | 0.090849 | 0.000119447 | 0.000147056 | 8927.3 |
| 4007 | 2.2739e-07 | 0.07749 | 0.0120701 | 0.000137756 | 0.000217786 | 8919.2 |
| 4008 | 1.12151e-07 | 0.05442 | 0.00222902 | 0.00015422 | 0.000368333 | 8932.0 |
| 4009 | 2.10477e-07 | 0.07455 | 0.00690446 | 0.000186402 | 0.000482698 | 8936.2 |
| Mean ± SD | 4.76785e-05 ± 0.000149849 | 0.42503 ± 1.09462 | 0.0148921 ± 0.027241 | 0.000162535 ± 4.3527e-05 | 0.000342058 ± 0.000116568 | 8928.1 ± 9.2 |
| Trimmed mean | — | 0.08295 | 0.00700943 | — | — | — |

Sampling updates: 700 per seed.
### [PACMANN](pacmann_adam/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 7.80805e-08 | 0.04541 | 0.00530195 | 0.000167993 | 0.00022868 | 9107.1 |
| 4001 | 7.84028e-08 | 0.04550 | 0.0031081 | 0.000111992 | 0.000369017 | 9065.1 |
| 4002 | 0.114951 | 55.09497 | 2.92553 | 0.000878069 | 0.0137854 | 8965.9 |
| 4003 | 1.03934e-06 | 0.16567 | 0.00519094 | 0.000132565 | 0.000235839 | 8948.0 |
| 4004 | 1.27511e-07 | 0.05803 | 0.00283742 | 0.000170114 | 0.000261211 | 9121.4 |
| 4005 | 1.15184e-07 | 0.05515 | 0.0032305 | 0.000162694 | 0.000363078 | 8954.5 |
| 4006 | 0.152715 | 63.50343 | 2.98031 | 0.00150141 | 0.034423 | 8944.6 |
| 4007 | 5.23628e-08 | 0.03719 | 0.00564723 | 0.000165304 | 0.000273733 | 8977.9 |
| 4008 | 5.27358e-08 | 0.03732 | 0.00362157 | 7.54099e-05 | 0.000312793 | 9080.3 |
| 4009 | 3.11263e-07 | 0.09066 | 0.00452938 | 0.000142517 | 0.000266641 | 9079.1 |
| Mean ± SD | 0.0267667 ± 0.0571265 | 11.91333 ± 25.05309 | 0.593931 ± 1.24336 | 0.000350807 ± 0.000466864 | 0.00505194 ± 0.0111577 | 9024.4 ± 72.0 |
| Trimmed mean | — | 6.94909 | 0.36952 | — | — | — |

Sampling updates: 700 per seed.
### [PACMANN + diffusion](pacmann_diffusion/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 3.79704e-08 | 0.03166 | 0.00319881 | 0.000108014 | 0.000253213 | 8997.0 |
| 4001 | 8.83845e-08 | 0.04831 | 0.00418438 | 0.000199022 | 0.000340726 | 9099.7 |
| 4002 | 0.18444 | 69.78852 | 4.71596 | 0.000998999 | 0.0612308 | 8999.6 |
| 4003 | 1.05915e-06 | 0.16724 | 0.00428316 | 0.000136612 | 0.000289678 | 9103.8 |
| 4004 | 0.0169002 | 21.12525 | 0.794535 | 0.000326859 | 0.00116677 | 8961.6 |
| 4005 | 1.44869e-06 | 0.19559 | 0.00480385 | 0.000209052 | 0.000249669 | 8984.7 |
| 4006 | 0.0485264 | 35.79689 | 2.25145 | 0.00032623 | 0.00156724 | 8985.8 |
| 4007 | 4.52521e-08 | 0.03457 | 0.00682301 | 0.000169052 | 0.000358389 | 8998.7 |
| 4008 | 1.61956e-07 | 0.06540 | 0.0049314 | 0.000125169 | 0.000318012 | 9068.5 |
| 4009 | 5.65231e-08 | 0.03863 | 0.00373017 | 0.00014924 | 0.000346217 | 9002.7 |
| Mean ± SD | 0.0249869 ± 0.0581331 | 12.72921 ± 23.51803 | 0.77939 ± 1.55871 | 0.000274825 ± 0.00026586 | 0.00661207 ± 0.0191964 | 9020.2 ± 50.8 |
| Trimmed mean | — | 7.18399 | 0.384343 | — | — | — |

Sampling updates: 700 per seed.

[Study controls](../../../docs/peer_review.md#study-design)
· [Development selection](../../../docs/peer_review.md#study-design)
· [Artifact dictionary](../../../docs/output_format.md)
· [Environment](../../../README.md#setup-and-execution)
· [Optimizer semantics](../../../references/upstream.md#controlled-study-choices)
