# KleinFlow

## Installation

Run from the repository root on Linux with CUDA. Requires Conda, `g++`, and `curl`.

```bash
conda env create -f environment.yml
conda activate Klein_FM
python scripts/build_orca.py
python scripts/prepare_data.py
```

## Training

Choose a dataset and replace `cuda:0` with your GPU.

```bash
# Planar
python scripts/train.py --dataset planar --device cuda:0

# Tree
python scripts/train.py --dataset tree --device cuda:0

# Grid
python scripts/train.py --dataset grid --device cuda:0

# Ego
python scripts/train.py --dataset ego --device cuda:0

# Ego-small
python scripts/train.py --dataset ego-small --device cuda:0

# Community
python scripts/train.py --dataset community --device cuda:0

# Community-small
python scripts/train.py --dataset community-small --device cuda:0

# IMDBBINARY
python scripts/train.py --dataset IMDBBINARY --device cuda:0

# MUTAG
python scripts/train.py --dataset MUTAG --device cuda:0
```

Heun sampling uses the legacy parallel-transport implementation by default so
existing experiment commands remain reproducible. Select analytic parallel
transport along the connecting geodesic with:

```bash
python scripts/train.py --dataset MUTAG --device cuda:0 \
  --flow_integrator heun --flow_ptransp lorentz
```

The MUTAG search launcher runs in the background and accepts the same choice:

```bash
bash scripts/MUTAG_hypsearch.sh --device cuda:0 --flow-ptransp lorentz
```
