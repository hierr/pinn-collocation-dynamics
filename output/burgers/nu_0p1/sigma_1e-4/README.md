# Burgers nu = 0.1, sigma = 1e-4

This candidate completed all three development seeds without numerical
failures. Its per-seed final solution relative L2 values are
0.0114115%, 0.0152684%, and 0.0237857%; residual RMS values are
0.000597629, 0.00107085, and 0.00101145 for seeds 3000-3002.

The selected positive coefficient is sigma = 1e-5; full per-seed and
aggregate outcomes, selection rationale, execution conditions, and limitations
are in the [shared campaign README](../sigma_0/README.md). The
[configuration](../../../../config/burgers/nu_0p1/burgers_nu_0p1_sigma_1e-4.yaml)
records this candidate.
