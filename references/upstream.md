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
| Error selection | Code exports error at minimum recorded training loss and minimum observed test errors | Reproduce named outputs and additionally record final error; agree on comparison policy before runs. |
| Software | Paper mentions PyTorch 1.12.1; released requirements pin PyTorch 2.7.1 and DeepXDE 1.14.0 | Choose a compatible locked project environment and disclose differences. |
| Seeds / failures | Ten repetitions; paper describes replacing occasional failed initializations | Use explicit planned seeds and retain/report failures as repository policy requires; disclose the protocol difference. |

No published result has been reproduced.

## Historical reproduction choices

The retained original-reference runs use [0, 0.99], float64, explicit Hammersley sampling,
coordinate-Adam epsilon 1e-7 outside the square root, and fresh coordinate moments
at each event. Escaped points are replaced only after all 15 movement steps.
Network optimizer state resets at each alternating stage. Planned seeds are
1000–1009; failed seeds are retained rather than replaced.

DeepXDE initially includes 240 initial/boundary points in the PDE population:
2,740 PDE coordinates and 240 fixed constraint coordinates (duplicated in the
combined training array). Upstream `train_x[480:]` selects the 2,500 interior
points. The maintained callback selects the corresponding tail of `train_x_all`,
then replaces PDE anchors with those 2,500 points. The static baseline retains
2,740 PDE coordinates. This released-code population difference is preserved
and disclosed; a later matched-budget study must address it explicitly.

PyTorch is pinned to 2.7.1 with CUDA 12.8 wheels, rather than the paper's 1.12.1.
The runner updates DeepXDE's derived L-BFGS per-call limits alongside its public
options; version 1.14.0 otherwise leaves those limits at import-time defaults.
This matters for short validation limits; the full 3,000-iteration configuration
retains 1,000 iterations per call. Actual optimizer iterations are archived.
Final-weight errors and upstream history-selected errors are saved separately.
Engineering checks do not establish agreement with published numerical results.

New controlled runs match the interior PDE population across methods from
initialization, evaluate solution errors on a uniform grid with spatially
interpolated generated references, and report
final-model errors rather than history-selected errors. See the
[accepted study controls](../docs/research_context.md#accepted-controls-and-evaluation-policy).
