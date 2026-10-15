# Source provenance

## PACMANN

- Authors: Coen Visser, Alexander Heinlein, Bianca Giovanardi.
- Paper: PACMANN: Point adaptive collocation method for artificial neural networks.
- DOI: https://doi.org/10.1016/j.cma.2025.118723
- Repository: https://github.com/CoenVisser/PACMANN
- Imported reference commit: `ecf06a6cf277b8beb6f8a99c88baa2535738fd7f`
- First target: `1D Burgers' equation/Setup_file.py`,
  `PINN_Burgers_function.py`, and `Burgers.npz` at that commit.
- Upstream scripts are Apache-2.0 licensed. Preserve notices and required
  attribution when importing code; verify dataset provenance separately.

## Imported snapshot

Unchanged scripts, upstream requirements, and the Apache-2.0 license are in
`references/pacmann/burgers/`. These files are reference material, not the
maintained training entrypoint. Do not format or edit them. Their requirements
are evidence of the released environment, not this project's dependency lock.
No project-wide license is assigned by importing the upstream license.

`references/pacmann/burgers/manifest.json` records commit-pinned URLs, upstream
and local paths, byte counts, Git blob SHA-1, and SHA-256 for all five files.
The import was verified against upstream Git blob IDs before installation.

The reference dataset is tracked at `data/burgers_dataset/PACMANN_Burgers.npz`.
The immutable import manifest records its historical local path,
`data/burgers/Burgers.npz`; restoration below uses the retained canonical path.
Its arrays are float64: `x` has shape (256, 1), `t` (100, 1),
and `usol` (256, 100). These header/archive checks do not validate numerical
accuracy. The upstream README attributes the Burgers scripts/data to DeepXDE
examples; the paper attributes the spectral reference solution to Raissi et al.
Dataset licensing has not been independently established; do not infer it from
the scripts' Apache-2.0 license.

### Restore the dataset

Run from the repository root using Python's standard library. The download is
checked in memory before writing; an existing different file is never replaced.

```bash
python3 - <<'PYDATA'
import hashlib
import json
import urllib.request
from pathlib import Path

manifest = json.loads(Path('references/pacmann/burgers/manifest.json').read_text())
item = next(f for f in manifest['files'] if f['local_path'] == 'data/burgers/Burgers.npz')
with urllib.request.urlopen(item['url'], timeout=60) as response:
    content = response.read()
assert len(content) == item['bytes'], 'Unexpected download size'
assert hashlib.sha256(content).hexdigest() == item['sha256'], 'Checksum mismatch'
target = Path('data/burgers_dataset/PACMANN_Burgers.npz')
if target.exists():
    assert target.read_bytes() == content, 'Conflicting local dataset'
else:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(content)
print('Dataset verified:', target)
PYDATA
```

## Reproduction observations to resolve

These are source observations, not silent corrections to the imported code.
The paper is DOI 10.1016/j.cma.2025.118723, Sections 2 and 3.1, Tables 1-3.

| Topic | Paper / released code | Next-checkpoint requirement |
| --- | --- | --- |
| Time domain | Paper: [0, 1]; code: [0, 0.99] | Record the chosen reproduction domain and separately inspect reference coordinates. |
| Coordinate Adam epsilon | Table 1: 1e-8; callback default: `10e-8` (1e-7) | Preserve released-code behavior for code reproduction; disclose difference. |
| Training phases | Five phases, each 7,000 network Adam steps at 1e-3 followed by L-BFGS configured for up to 3,000 iterations | Preserve alternating phases and record actual optimizer endpoints; do not substitute one long Adam stage. |
| Movement | Every 50 Adam steps; 15 coordinate-Adam steps at 1e-5 for the Burgers table case | Freeze network weights during movement; coordinate moments reset at each event; no movement in L-BFGS. |
| Escaped points | Code replaces points uniformly after all coordinate steps | Preserve this timing for reproduction, including the later diffusion ablation. |
| Point bookkeeping | 2,500 interior, 80 boundary, 160 initial; callback first uses `train_x[480:]` | Validate DeepXDE's actual arrays and counts; keep constraints fixed. |
| Network | Float64; four 64-neuron tanh layers; Glorot normal | Preserve precision, architecture, and initialization. |
| Sampling baseline | Paper uses Hammersley; callback-free default distribution comes from DeepXDE | Verify the pinned library default; do not silently replace it with Halton. |
| Error selection | Code exports error at minimum recorded training loss and minimum observed test errors | The controlled study instead evaluates the final checkpoint, using the agreed comparison policy. |
| Software | Paper mentions PyTorch 1.12.1; released requirements pin PyTorch 2.7.1 and DeepXDE 1.14.0 | Choose a compatible locked project environment and disclose differences. |
| Seeds / failures | Ten repetitions; paper describes replacing occasional failed initializations | Use explicit planned seeds and retain/report failures as repository policy requires; disclose the protocol difference. |

No published result has been reproduced.

## Controlled-study choices

The maintained implementation uses [0, 0.99], float64, explicit Hammersley
initialization, coordinate-Adam epsilon 1e-7 outside the square root, and fresh
coordinate moments at each movement event. Escaped points are replaced after
all movement steps. Network optimizer state resets at each alternating stage.
Campaign YAMLs specify seeds; failures are retained rather than replaced.

The released implementation starts with 2,740 PDE coordinates: 2,500 interior
points and 240 initial/boundary points. Upstream `train_x[480:]` selects the
2,500 interior points, so the adaptive method later uses fewer PDE points than
the static baseline. The controlled implementation instead uses the configured
interior PDE population from initialization for every method, while keeping the
same initial/boundary constraints fixed.

PyTorch is pinned to 2.7.1 with CUDA 12.8 wheels, rather than the paper's 1.12.1.
The runner updates DeepXDE's derived L-BFGS per-call limits alongside its public
options; version 1.14.0 otherwise leaves those limits at import-time defaults.
The full 3,000-iteration configuration retains 1,000 iterations per call.
DeepXDE checks `maxiter` between calls and derives a per-call function-evaluation
limit from `maxfun`; neither is a strict overall stage cap. For the PyTorch
backend, positive `maxls` enables strong-Wolfe line search with PyTorch's default
limit of 25; the configured value 50 is not passed as a numerical limit.
Actual optimizer iterations and function evaluations are archived.

The first frozen comparison retains this L-BFGS driver, including its possible
stage-limit overshoot, to match sigma development. Shared optimizer settings do
not imply identical computation; compare recorded work and runtime.

Random resampling follows the released callback: every 50 Adam steps it replaces
the interior population with pseudorandom uniform points, not another Hammersley
design. RAD uses 100,000 fresh pseudorandom candidates at the same cadence,
weights `abs(residual) / mean(abs(residual)) + 1`, and selects without replacement.
The maintained RAD callback batches residual predictions before global
normalization to limit memory. An exactly zero residual field uses uniform
probabilities instead of the released callback's undefined division. Both
methods retain the controlled study's matched PDE populations and fixed
constraints; neither updates points during L-BFGS.

Accepted training/evaluation code was transferred from development commit
`59e2de8`, and plot improvements from `95c02f8`. The source snapshot above remains
unchanged; tuning configurations and run artifacts remain on their branch.

Final-checkpoint evaluation uses a uniform grid with spatially interpolated
generated references and an independent residual grid. Errors are not selected
from training-history minima. These are controlled-study choices, not a claim
of published-result reproduction. See the
[accepted study controls](../docs/research_context.md#accepted-controls-and-evaluation-policy).
