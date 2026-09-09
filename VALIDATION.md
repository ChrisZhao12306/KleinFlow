# Consolidation validation record

Date: 2026-09-09.

## Completed checks

| Check | Result |
|---|---|
| Local regression suite | **98 passed, 14 skipped** |
| Python syntax | All 75 Python files parsed successfully |
| Shell syntax | All 10 Shell scripts passed `bash -n` |
| Local package import paths | No references to missing project modules |
| Undefined names | No undefined names detected; harmless lint warnings and original formatting were retained |
| Dataset migration | All six raw data files match their source SHA-256 hashes |
| Real Planar and Tree data | Each has train=128, val=32, test=40; split membership, order, adjacency matrices, shapes, and dtypes match the original loaders |
| Static comparison of computation code | 161 retained functions/classes match their source ASTs, subject to the scope below |
| Environment file | Uses only HypDiff_gt, renamed Klein_FM with prefix removed; the dependency-body hash matches the source |
| Original repository preservation | Content hashes for all 481 non-Git files were unchanged, with no files added or deleted at the time of the audit |
| Repository contents | No old experiment logs, checkpoints, generated graphs, PID files, lock files, temporary downloads, or five-seed scripts were migrated; datasets are excluded from Git |

The local suite ran on Windows in a temporary verification environment: Python 3.8.20, pytest 8.3.5, NumPy 1.24.4, SciPy 1.10.1, NetworkX 3.1, and pyemd 1.0.0. This environment was used for verification and does not replace the repository's Linux `Klein_FM` environment file.

New Python caches created by tests are ignored by `.gitignore`. An automatic approval policy blocked cache cleanup with `blocked by policy`, so these local caches were retained. They were not copied from the original repositories and are excluded from the Git upload.

Command used:

```bash
python -B -m pytest -q -rs -p no:cacheprovider
```

Coverage includes all nine dataset routes and aliases, original argument defaults, explicit overrides, sampled search configurations and training arguments, frozen search baselines, result writing and reproduction commands, dry runs from other working directories, background scheduling and concurrency, failure handling, output directory placement, data loading, and locally available metric tests.

Static comparisons normalize relocated imports and docstrings and use Python's effective last definition when original files contain duplicate method names. They cover retained models, Flow Matching, geometry, data features and split functions, training computation, and evaluation functions. Routing, file paths, result metadata, and functions with removed unsupported dataset branches were inspected separately. Static equivalence does not replace numerical or CUDA training validation.

## Outstanding validation

The 14 skipped records include:

- Two Torch-dependent test modules and one configuration check gated on Torch.
- The real ORCA executable test and the PyGSP wavelet test; the target Linux executable or PyGSP was unavailable locally.
- Nine original-versus-consolidated numerical comparison cases; Torch/DGL was unavailable locally and source repository path parameters were not enabled.

The following checks require an accessible Linux + CUDA `Klein_FM` environment and have **not** been reported as passing:

1. Creating the environment on the target server and importing Torch/DGL with CUDA.
2. Compiling ORCA and running real Orbit and wavelet metrics.
3. Shortened training for all nine datasets, including encoder and Flow training, model saving, sampling, decoding, and final evaluation.
4. Comparing forward outputs, losses, gradients, updated parameters, sampling, and metrics against the original implementations.

Commands:

```bash
conda activate Klein_FM
python scripts/build_orca.py
python scripts/prepare_data.py
python -m pytest -q
python scripts/smoke_test.py --device cuda:0

export FLOW_KLEIN_SOURCE_FIXED=/path/to/Flow_Klein_fixed
export FLOW_KLEIN_SOURCE_V0901=/path/to/Flow_Klein_0901
export FLOW_KLEIN_TEST_DEVICE=cpu
python -m pytest -q tests/test_numerical_equivalence.py
```

Ordinary training and standard tests do not depend on the original repositories. Source paths are used only for explicitly enabled migration comparisons. Numerical comparisons use identical synthetic inputs in isolated subprocesses; `smoke_test.py` covers end-to-end execution with real data.

## GitHub publication checks

Before the initial upload, the ignore rules were anchored to `/data/` and `/outputs/` so that the `flow_klein/data/` source package is tracked. All 95 staged files were inspected: the source package is present, local datasets and runtime caches are excluded, and no Chinese text was found in file contents or decoded JSON strings. Documentation is in English.

The staged files were exported to a separate directory without local data or untracked files. The regression suite in that exported snapshot again finished with **98 passed, 14 skipped**. These checks do not change the outstanding Linux/CUDA and numerical validation listed above.
