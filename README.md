# KleinFlow

Graph generation with Flow Matching on the Klein model of hyperbolic geometry. The code is organized by function and preserves the two dataset-specific implementations from the original project.

## Datasets and implementation routes

| Dataset | Implementation | Evaluation |
|---|---|---|
| Planar, Tree | `v0901` | V.U.N., five structural metrics, and ratios |
| Grid | `v0901` | Degree / Clustering / Spectral MMD |
| Ego, Community, IMDBBINARY, MUTAG | `fixed` | Degree / Clustering / Spectral MMD |
| Ego-small, Community-small | `fixed` | Degree / Clustering / Orbit MMD |

The `v0901` implementation comes from the on-disk Flow_Klein_0901 project, and `fixed` comes from Flow_Klein_fixed. The training entry point selects the implementation by dataset. Configuration, preprocessing, encoders/decoders, training, sampling, and evaluation follow that route. Klein geometry, Conditional DiT, and Flow Matching share the same implementation.

Original experimental behavior is preserved, including Grid's separate validation set, the validation slices for standard fixed datasets, the fixed Planar/Tree benchmark splits and structural constraints, and Community-small's original split over an index space of 200. The Community-small source contains 100 graphs; the loader does not replace its special split with a new 128/32/40 split.

Aliases are case-insensitive and accept `-` or `_` in small dataset names. `SynEgo1000_origin` and `SynEgo1000_original` map to Ego; `SynCommunity1000_origin` maps to Community; `comm20` maps to Community-small. Unsupported datasets are rejected.

## Environment

Training targets Linux with CUDA. The repository contains a single `environment.yml`, copied from the original HypDiff_gt environment. Its name is now `Klein_FM`, the machine-specific `prefix` is removed, and all channels, dependencies, and versions are unchanged.

```bash
git clone https://github.com/ChrisZhao12306/KleinFlow.git
cd KleinFlow
conda env create -f environment.yml
conda activate Klein_FM
export PYTHON="$(command -v python)"
python scripts/build_orca.py
```

ORCA compilation requires `g++`. The Planar/Tree downloaders require `curl`. Use the original installation sources for the pinned Torch and DGL CUDA wheels. If environment creation cannot locate the pinned DGL CUDA wheel on the default index, use the original server's wheel source without changing the dependency versions.

Run directly from the source checkout; an editable installation is not required. The environment name change applies to this repository's environment file and does not rename an existing Conda environment.

## Data

The consolidated local project contains these copied source files:

- `data/benchmarks/planar/planar.pkl` and `data/benchmarks/tree/tree.pkl`
- `data/SynEgo1000_origin.pkl` and `data/SynCommunity1000_origin.pkl`
- `data/benchmarks/ego_small/ego_small.pkl`
- `data/benchmarks/community_small/community_12_21_100.pt`

Grid generates 100 grid graphs using the original method. IMDBBINARY and MUTAG retain the DGL GINDataset loader, with download caches under `data/dgl/`. Planar, Tree, and the two small benchmarks can be downloaded from their original sources.

```bash
python scripts/prepare_data.py --dry-run
python scripts/prepare_data.py
# Prepare selected datasets only.
python scripts/prepare_data.py planar tree ego-small community-small
```

Data files are excluded from Git. After cloning, copy the two `Syn*.pkl` files into `data/` separately. Prepare the remaining datasets with the commands above or copy them from an existing local dataset directory. The default data directory is resolved relative to the repository; set `FLOW_KLEIN_DATA_DIR` before startup to use another directory.

## Training

```bash
python scripts/train.py --dataset planar --device cuda:0
python scripts/train.py --dataset tree --device cuda:0
python scripts/train.py --dataset grid --device cuda:0
python scripts/train.py --dataset ego --device cuda:0
python scripts/train.py --dataset ego-small --device cuda:0
python scripts/train.py --dataset community --device cuda:0
python scripts/train.py --dataset community-small --device cuda:0
python scripts/train.py --dataset IMDBBINARY --device cuda:0
python scripts/train.py --dataset MUTAG --device cuda:0
```

Original effective Klein arguments and defaults are preserved, and explicit arguments take precedence. `--taskselect klein_graphtask` remains supported. Standard training defaults differ from the fixed parameters used by dataset-specific searches; reproduce an experiment with the complete command printed in the search results.

```bash
python scripts/train.py --dataset planar --help
python scripts/train.py --dataset planar --dry-run
python scripts/train.py --dataset MUTAG --epoch_number 2000 --epoch_diff 2000 \
  --device cuda:0 --directed False --graph_save_path /path/to/my_run
```

`--dry-run` prints the selected route and effective arguments without starting training or writing experiment files. The historical `--use_simple_dit` argument remains accepted for compatibility; both original pipelines actually use Conditional DiT.

Default outputs are written to `outputs/<dataset>/<timestamp_and_process_id>/`, including models, generated graphs, text evaluation results, and `metrics.json`. The JSON file records the implementation route, effective arguments, and original metric fields. Use `--graph_save_path` to select an output directory, or set `FLOW_KLEIN_OUTPUT_DIR` before startup to change the default output root.

Python and Shell entry points locate project files relative to their own locations and support invocation from another working directory. Explicit relative output paths are resolved from the caller's working directory.

## Hyperparameter search

The unified entry point preserves each source's sampling algorithm, search space, fixed parameters, default GPUs, and ranking rules:

```bash
python scripts/hyperparam_search.py --dataset planar --dry-run
python scripts/hyperparam_search.py --dataset grid --dry-run
python scripts/hyperparam_search.py --dataset ego-small --device cuda:0
python scripts/hyperparam_search.py --dataset MUTAG --device cuda:0 --num-experiments 60
```

| Dataset | Original default GPUs | Default experiments | Search stage |
|---|---|---:|---|
| Planar | cuda:4, 5, 6, 7 | 120 | Fourth |
| Tree | cuda:1, 2, 3 | 120 | Fourth |
| Grid | cuda:1 | 60 | Second |
| Ego | cuda:5 | 60 | Second |
| Community | cuda:1 | 60 | Sixth |
| Ego-small, Community-small | cuda:0 | 60 | Third |
| IMDBBINARY | cuda:1 | 60 | Second |
| MUTAG | cuda:2 | 60 | Third |

Background launch examples:

```bash
bash scripts/planar_hypsearch.sh --devices cuda:0 cuda:1
bash scripts/tree_hypsearch.sh --device cuda:0
bash scripts/ego_small_hypsearch.sh --device cuda:0
bash scripts/Grid_hypsearch.sh --device cuda:0
# The unified launcher accepts canonical lowercase dataset names.
bash scripts/launch_search.sh community --device cuda:0
```

Planar/Tree retain GPU isolation, parallel scheduling, and failure records. Other searches retain sequential execution. Background launchers print the log and PID locations; search results and models are stored under `outputs/search/`. Use each entry point's `--help` for source-specific options.

## Project layout

Executable entry points are in `scripts/`, and implementation code is in `flow_klein/`. Implementation modules are grouped under `config`, `data`, `geometry`, `models`, `training`, `evaluation`, `experiments`, and `utils`. Differences between the two routes live in `fixed.py` and `v0901.py`. Frozen search baselines are in `configs/`, orbit-counting source code is in `third_party/orca/`, and regression tests are in `tests/`.

This repository targets retraining. It does not provide legacy import-path mappings for checkpoints serialized as complete Python objects.

## Validation

```bash
python -m pytest -q
python scripts/smoke_test.py --dry-run
# Run shortened training for all nine datasets on Linux/CUDA after preparing data and ORCA.
python scripts/smoke_test.py --device cuda:0
```

The smoke test reduces epochs and model widths for validation without changing production defaults. It writes per-dataset logs and status under `outputs/smoke/`.

Optional numerical comparisons against the original implementations run only when the original repository paths are explicitly provided. Ordinary training and tests do not depend on the original repositories:

```bash
export FLOW_KLEIN_SOURCE_FIXED=/path/to/Flow_Klein_fixed
export FLOW_KLEIN_SOURCE_V0901=/path/to/Flow_Klein_0901
export FLOW_KLEIN_TEST_DEVICE=cpu  # Alternatively, use cuda:0.
python -m pytest -q tests/test_numerical_equivalence.py
```

Comparisons use isolated processes and identical synthetic inputs to check splits, ordering, features, parameter initialization, forward outputs, losses, gradients, updated parameters, Euler/Heun sampling, decoding, and metrics. Discrete outputs must match exactly; floating-point tolerances are `rtol=1e-5, atol=1e-6`. The smoke test separately covers end-to-end execution with real datasets.

See [VALIDATION.md](VALIDATION.md) for the checks actually performed and outstanding Linux/CUDA validation.
