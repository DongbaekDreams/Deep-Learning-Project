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

The study follows a fully crossed 2×3 ablation over two independent components:

**Dimensionality Reduction**
- Autoencoder (nonlinear compression)
- Structured Pooling (deterministic aggregation of feature groups)

**Classifier**
- MLP (shallow multi-layer perceptron)
- FT-Transformer (feature tokenizer + transformer)
- ConvTran (Position Encoding + Transformer)

This results in four basic experimental conditions:

- A1: Autoencoder + MLP  
- A2: Autoencoder + FT-Transformer
- A3: Autoencoder + ConvTran 
- B1: Structured Pooling + MLP  
- B2: Structured Pooling + FT-Transformer
- B3: Structured Pooling + ConvTran

**Class Imbalancing Methods**

To accomodate the undersampled class of SVT, several class imbalancing methods were also applied like;
- None
- WeightedCE
- Focal Loss
- SMOTEENN

Considering every Dimensionality Reduction methods + Classifier + Class Imbalancing Method leaves with a total of 2 x 3 x 4 experiments (24)

All conditions use identical data splits and MultiROCKET features, ensuring fair comparison.


## Repository Structure
data/ # Raw and processed ECG data
experiments/ # Experiment configs (A1, A2, A3, B1, B2, B3 )
notebooks/ # Exploratory and analysis notebooks
scripts/ # Utility scripts (data prep)
src/ # Core pipeline code (models, training, reduction)

The folder structure in experiments is of the form for example:
A1_autoencoder_SMOTEENN_super. The first part is the experiment dimensionality reduction and classifier followed by classifier name followed by the class imbalancing method.

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
8. **07_results_report.ipynb** - Aggregate and report all experiment results.
9. **08_convtran_amp.ipynb** - Experiment with ConvTran model and reduced features (pre-pooled and autoencoder)
10. **09_super_model_evaluation.ipynb** - Compilation of all above experiment in one super notebook that also creates initial comparison tables
11. **12_a1_curves.ipynb** - Experiments ran on A1 configuration to get training plots
12. **13_a1_per_class_curves.ipynb** - Experiments ran on A1 configuration to get training plots per class.
13. **pureconvtran-1sec.ipynb** - ConvTran ran on raw data, but the training data time series is all trimmed down to 1 second for computational convenience
14. **pureconvtran-1sec.ipynb** - ConvTran ran on raw data, but the training data time series is all trimmed down to 2.5 seconds for computational convenience

Notebooks import from `src`; keep core logic in `src/` and use notebooks for exploration and visualization. Use the same Poetry environment for Jupyter (`poetry run jupyter notebook` or select the Poetry venv as the kernel).

### If you see `TypeError: C variable sklearn.utils._random.DEFAULT_SEED has wrong signature`

This can happen on Windows with the scikit-learn version pinned by sktime (e.g. 1.7.x). Fix it by forcing a reinstall of scikit-learn 1.8 in the Poetry env:

```bash
poetry run pip install --force-reinstall scikit-learn
```

Then restart the notebook kernel and re-run. (Requires Python 3.11+ for scikit-learn 1.8.)


**Collaborative project** — this repo is set up for 3 people. Use Poetry for a consistent environment; reports are gitignored (share via your preferred channel). Track progress with **[ANALYSIS_ROADMAP.md](ANALYSIS_ROADMAP.md)** (2×2 ablation checklist).
