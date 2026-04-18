# MultiROCKET ECG Classification Ablation Study

This repository implements a reproducible ablation study for 12-lead ECG classification using MultiROCKET feature extraction. The project investigates how different dimensionality reduction strategies and classifier architectures affect performance on high-dimensional time-series features. Specifically, we compare autoencoder-based compression and structured pooling, paired with either a multi-layer perceptron (MLP) or an FT-Transformer, under a controlled 2×2 experimental design.

## Project Overview

We perform classification on the Chapman–Shaoxing 12-lead ECG dataset, focusing on four diagnostic classes:

- **AF** (Atrial Fibrillation)
- **SVT** (Supraventricular Tachycardia)
- **Sinus Brady** (Sinus Bradycardia)
- **Sinus Rhythm** (Sinus Rhythm (+ Sinus Irregularity))

All experiments share a common upstream pipeline:

raw ECG → MultiROCKET feature extraction → feature scaling → dimensionality reduction → classifier

This design isolates the impact of downstream modeling choices by keeping feature extraction fixed across all experiments.

## Experimental Design

The study follows a fully crossed 2×2 ablation over two independent components:

**Dimensionality Reduction**
- Autoencoder (nonlinear compression)
- Structured Pooling (deterministic aggregation of feature groups)

**Classifier**
- MLP (shallow multi-layer perceptron)
- FT-Transformer (feature tokenizer + transformer)

This results in four experimental conditions:

- A1: Autoencoder + MLP  
- A2: Autoencoder + FT-Transformer  
- B1: Structured Pooling + MLP  
- B2: Structured Pooling + FT-Transformer  

All conditions use identical data splits and MultiROCKET features, ensuring fair comparison.


## Repository Structure
data/ # Raw and processed ECG data
experiments/ # Experiment configs (A1, A2, B1, B2)
notebooks/ # Exploratory and analysis notebooks
scripts/ # Utility scripts (data prep)
src/ # Core pipeline code (models, training, reduction)


## Setup (Poetry)

Install [Poetry](https://python-poetry.org/docs/#installation), then from the project root:

```bash
poetry install
poetry shell
```

To add the optional MultiROCKET backend (sktime):

```bash
poetry add sktime
# or: poetry install --extras sktime  (if sktime is listed under [tool.poetry.extras])
```

**Without Poetry:** export a lockfile-based requirements file and use pip:

```bash
poetry export -f requirements.txt --without-hashes -o requirements.txt
pip install -r requirements.txt
```

## Raw Data

Place your raw ECG dataset in:

```
data/raw/
```

Do **not** commit large raw data files. Implement the dataset loader in one place:

- **`src/data/io.py`** → `load_raw_dataset()`  
  Replace the placeholder (which raises `NotImplementedError`) with your logic to load ECG time series and labels. The docstring in that function describes the expected return format.

## Running an Experiment

Each ablation condition has a YAML config under `experiments/`:

- **A1**: autoencoder + MLP  
- **A2**: autoencoder + FT-Transformer  
- **B1**: structured pooling + MLP  
- **B2**: structured pooling + FT-Transformer  

Run training (full pipeline: features → reduction → classifier). From the project root, with the Poetry env active:

```bash
python -m src.training.train --config experiments/A1_autoencoder_mlp/config.yaml
```

Other examples:

```bash
python -m src.training.train --config experiments/B1_pooling_mlp/config.yaml
python -m src.training.train --config experiments/B2_pooling_ft/config.yaml
```

The script will fail at `load_raw_dataset()` until you implement it. Artifacts (splits, MultiROCKET transformer, scaler, memmaps, checkpoints, metrics, confusion matrices) are written under `experiments/<condition>/` and shared model dirs under `models/`.

## Notebook Workflow

1. **00_data_inspection.ipynb** – Inspect raw data (call `load_raw_dataset()` after implementing it).
2. **01_preprocessing.ipynb** – Preprocessing and split creation.
3. **02_multirocket.ipynb** – MultiROCKET feature extraction and persistence.
4. **03_autoencoder.ipynb** – Autoencoder training and reduced features.
5. **04_structured_pooling.ipynb** – Structured pooling reduction.
6. **05_mlp_models.ipynb** – MLP classifier training/evaluation.
7. **06_ft_transformer.ipynb** – FT-Transformer training/evaluation.

Notebooks import from `src`; keep core logic in `src/` and use notebooks for exploration and visualization. Use the same Poetry environment for Jupyter (`poetry run jupyter notebook` or select the Poetry venv as the kernel).

### If you see `TypeError: C variable sklearn.utils._random.DEFAULT_SEED has wrong signature`

This can happen on Windows with the scikit-learn version pinned by sktime (e.g. 1.7.x). Fix it by forcing a reinstall of scikit-learn 1.8 in the Poetry env:

```bash
poetry run pip install --force-reinstall scikit-learn
```

Then restart the notebook kernel and re-run. (Requires Python 3.11+ for scikit-learn 1.8.)

## Reports

The `reports/` directory is in `.gitignore`. Share proposal, preliminary, and final reports (e.g. `reports/proposal/`, `reports/preliminary/`, `reports/final/`) via your team’s chosen channel (Drive, OneDrive, etc.).

**Collaborative project** — this repo is set up for 3 people. Use Poetry for a consistent environment; reports are gitignored (share via your preferred channel). Track progress with **[ANALYSIS_ROADMAP.md](ANALYSIS_ROADMAP.md)** (2×2 ablation checklist).
