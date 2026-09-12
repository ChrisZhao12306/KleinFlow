# KleinFlow

**Graph generation with conditional Flow Matching in Klein hyperbolic space.**

KleinFlow learns graph representations in the Klein ball and models their distribution with a conditional Transformer velocity field. Dataset-specific preprocessing, structural conditioning, decoding constraints, and evaluation protocols are selected through a single training interface.

## Method

The generation pipeline has three stages:

1. **Graph representation learning.** A graph encoder produces a Klein latent representation and a conditioning vector. A masked decoder learns node occupancy, adjacency, and auxiliary graph statistics.
2. **Conditional Flow Matching.** A conditional DiT predicts velocities in standardized tangent coordinates. Latent normalization statistics are estimated from training embeddings.
3. **Graph generation.** Euler or Heun integration produces latent samples, which are mapped to the Klein ball and decoded into graphs. Dataset configurations control edge budgets, connectivity repair, structural constraints, and candidate selection.

The geometry and Flow Matching implementation are shared across datasets. Architecture settings, data protocols, and decoding options are defined in the Python package.

## Installation

The training environment targets **Linux with CUDA**. Conda, `g++` for ORCA, and `curl` for benchmark downloads are required.

```bash
conda env create -f environment.yml
conda activate Klein_FM
python scripts/build_orca.py
```

The environment file pins the Python, Torch, DGL, and scientific computing dependencies. CUDA-enabled packages require a compatible CUDA runtime. Run commands from the source checkout; no editable installation is required.

## Datasets

| Dataset | Data preparation | Evaluation |
|---|---|---|
| Planar | Included in `data/benchmarks/planar/` | V.U.N. and structural MMD ratios |
| Tree | Included in `data/benchmarks/tree/` | V.U.N. and structural MMD ratios |
| Grid | Generate 100 grid graphs | Degree, clustering, spectral MMD |
| Ego | Included in `data/` | Degree, clustering, spectral MMD |
| Community | Included in `data/` | Degree, clustering, spectral MMD |
| Ego-small | Included in `data/benchmarks/ego_small/` | Degree, clustering, orbit MMD |
| Community-small | Included in `data/benchmarks/community_small/` | Degree, clustering, orbit MMD |
| IMDBBINARY | Download through DGL | Degree, clustering, spectral MMD |
| MUTAG | Download through DGL | Degree, clustering, spectral MMD |

Dataset names are case-insensitive. Both hyphens and underscores are accepted for `ego-small` and `community-small`.

The repository includes the six raw dataset files listed above. Prepare all datasets, including DGL downloads for IMDBBINARY and MUTAG:

```bash
python scripts/prepare_data.py
```

To prepare selected datasets only:

```bash
python scripts/prepare_data.py planar tree grid ego-small community-small imdbbinary mutag
```

Ego and Community are stored as `data/SynEgo1000_origin.pkl` and `data/SynCommunity1000_origin.pkl`. Missing Planar, Tree, Ego-small, and Community-small files can be downloaded by the preparation script. Use `--dry-run` to inspect the requested datasets without downloading or loading them.

Raw benchmarks are stored in `data/benchmarks/`, and DGL caches are stored in `data/dgl/`. Raw dataset files are tracked in Git; generated DGL caches and temporary downloads are excluded. Set `FLOW_KLEIN_DATA_DIR` before startup to use another data directory.

**Data protocols.** Planar and Tree use predefined splits of 128 training, 32 validation, and 40 test graphs. Ego-small uses a seeded split of the same sizes. Community-small applies a seed-0 permutation over an index space of 200 to its 100 available graphs; membership and ordering are defined by the benchmark loader. Grid reserves a separate validation subset. Dataset loaders determine node ordering, structural features, and padding before training.

## Training

```bash
python scripts/train.py --dataset planar --device cuda:0
python scripts/train.py --dataset ego-small --device cuda:0
python scripts/train.py --dataset MUTAG --device cuda:0
```

A training run fits the graph autoencoder and the Flow Matching model, then samples graphs and computes the dataset's evaluation metrics. Explicit command-line arguments take precedence over dataset defaults.

```bash
python scripts/train.py --dataset planar --help
python scripts/train.py --dataset planar --dry-run

python scripts/train.py --dataset MUTAG --device cuda:0 \
  --epoch_number 2000 --epoch_diff 2000 \
  --directed False --graph_save_path outputs/mutag_run
```

Key arguments include `--graphEmDim` for latent width, `--cond_dim` for conditioning width, `--epoch_number` and `--epoch_diff` for the two training stages, and `--flow_steps` and `--flow_integrator` for sampling. Inspect `--help` for the selected dataset before changing structural decoding options.

## Hyperparameter search

```bash
python scripts/hyperparam_search.py --dataset planar --dry-run
python scripts/hyperparam_search.py --dataset planar --devices cuda:0 cuda:1
python scripts/hyperparam_search.py --dataset ego-small --device cuda:0
python scripts/hyperparam_search.py --dataset MUTAG --device cuda:0 --num-experiments 60
```

Planar and Tree use 120 trials by default: three anchor configurations, 93 local variants, and 24 global samples. Their scheduler isolates each worker to a selected GPU. Other datasets use 60 sequential random-search trials. Search spaces, anchor configurations, and constant training parameters are packaged in `flow_klein/experiments/`.

Specify `--device` or `--devices` for the GPUs available on your machine. Search defaults use seed 42 for configuration sampling and seed 1432 for training. Use `--search-seed` and `--training-seed` to override them.

Background execution:

```bash
export PYTHON="$(command -v python)"
bash scripts/planar_hypsearch.sh --devices cuda:0 cuda:1
bash scripts/launch_search.sh community --device cuda:0
```

Search results include the effective parameters, ranking metrics, and complete reproduction commands. Standard training defaults and search settings serve different purposes; use the command recorded for a trial to reproduce that trial.

## Evaluation and outputs

For Planar and Tree, V.U.N. measures validity, uniqueness, and novelty relative to the training graphs. Structural comparisons include degree, clustering, orbit, spectral, and wavelet statistics, together with reference-normalized MMD ratios. Results are ranked by V.U.N. and then average ratio. Other datasets are ranked by their corresponding average MMD; lower MMD is better.

Training outputs are saved under `outputs/<dataset>/<run_id>/`. Each run includes model files, generated graphs, text metrics, and `metrics.json` with effective parameters, split metadata, seeds, and software versions. Search outputs are stored under `outputs/search/`.

Use `--graph_save_path` for an explicit training destination, or set `FLOW_KLEIN_OUTPUT_DIR` to change the default output root. Entry points support invocation from other working directories; explicit relative paths resolve from the caller's directory. Outputs and caches are excluded from Git.

## Code organization

```text
scripts/                  Training, data preparation, search, and build entry points
data/                     Raw graph datasets
flow_klein/
  config/                 Argument definitions and dataset defaults
  data/                   Graph loading, preprocessing, and structural profiles
  geometry/               Klein geometry and mathematical operations
  models/                 Encoders, decoders, conditional DiT, and Flow Matching
  training/               Optimization, sampling, and graph decoding
  evaluation/             Graph-distribution and structural metrics
  experiments/            Search spaces, anchors, and process scheduling
  utils/                  Shared utility namespace
third_party/orca/         Orbit-counting source code
```

Benchmark download locations and data-source attribution are recorded in the dataset loaders. Orbit statistics use the bundled ORCA implementation.
