# Research context

The study asks whether the relative benefit of residual-guided collocation
changes with sharp, localized solution features, and whether adding diffusion
changes that benefit. Neither is assumed to hold.

The first milestone is an upstream Burgers reproduction followed by a diffusion
ablation and a matched low-discrepancy comparison. A later parameter study may
vary Burgers viscosity; changing the PDE requires a matching validated reference
solution. Benchmark and parameter selection remain to be agreed.

Distinguish solution gradients from residual-score gradients. Measure global
solution error and feature-region error independently of training points;
residual statistics and point histories are supporting diagnostics. Match
controls within each problem regime and report seed variability and runtime.
