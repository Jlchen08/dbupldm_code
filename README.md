# DBUPLDM Code

Reference implementations and datasets for the experiments reported in the paper
*Dual Balanced Unified Pinball Loss Distribution Machine (DBUPLDM)*.

This repository contains the core implementation of each model used in the
experiments, together with the benchmark datasets, so that the reported results
can be reproduced and audited.

## Repository structure

```
dbupldm_code/
├── python/           Python implementations (neural and SVM baselines, runners)
├── matlab/           MATLAB implementations (DBUPLDM and margin-based baselines)
├── datasets/         Benchmark datasets (.mat, .txt, .xlsx)
└── README.md
```

## Models

Every model below is implemented independently and can be run on its own.

### MATLAB implementations (`matlab/`)

| File | Model |
|------|-------|
| `Unified_pin_ldm.m` | Unified pinball loss distribution machine (UP-LDM) |
| `Unified_pin_csldm.m` | Cost-sensitive unified pinball loss distribution machine (CS-LDM, and the dual-balanced variant DBUPLDM when driven by the dual balance factor) |
| `Unified_pin_svm.m` | Unified pinball loss SVM (UP-SVM) |
| `Unified_pin_cssvm.m` | Cost-sensitive unified pinball loss SVM (CS-SVM) |
| `pin_svm.m` | Pinball loss SVM (Pin-SVM) |

Supporting routines:

| File | Role |
|------|------|
| `Balance_factor.m` | Dual balance factor computation |
| `Function_Kernel.m`, `svkernel.m` | Kernel functions (linear, RBF) |
| `svdatanorm.m` | Feature normalization |
| `myAUC.m` | AUC computation |
| `tune_tau.m`, `tune_para_svm.m` | Hyperparameter tuning |
| `createfigure.m` | Figure helper used by the tuning routine |
| `main.m` | Example driver showing how the models are called |

### Python implementations (`python/`)

| File | Model |
|------|-------|
| `mlp_focal_model.py` | MLP with Focal Loss (`MLPClassifier`, `FocalLoss`, class weights) |
| `imbalanced_baselines.py` | DeepSMOTE, MLP + LDAM, MLP + Balanced Softmax (\(T=1.0\)) |
| `svm_model.py` | SVM baseline (RBF kernel, balanced class weights) |
| `data_loader.py` | Dataset loading and preprocessing (`StandardScaler`) |
| `trainer.py` | Training loop, optimizer, scheduler, early stopping |
| `run_experiment.py` | End-to-end runner for the MLP/Focal and SVM experiments |
| `run_table7_baselines.py` | End-to-end runner for DeepSMOTE / LDAM / Balanced Softmax |

## Datasets (`datasets/`)

Benchmark datasets used in the experiments. File formats:

- `*.mat` — tabular datasets (last column is the class label).
- `*.txt` — text-format datasets (`monks_*`, `SPECT_*`, `german`, `plrx`, `noise_0_*`).
- `*.xlsx` — tabular datasets used by the MATLAB drivers (`aps_failure_test_set`,
  `Breast`, `creditcard`, `diabetes_prediction`, `patients`).

## Environment

Python dependencies:

```bash
pip install -r python/requirements.txt
```

Tested with Python 3.8+, NumPy, pandas, scikit-learn, SciPy, and PyTorch.
The MATLAB implementations require only base MATLAB (no additional toolboxes).

## Running the experiments

### Python

MLP+Focal and SVM:

```bash
cd python
python run_experiment.py --dataset-dir ../datasets --results-dir results
```

DeepSMOTE, LDAM, and Balanced Softmax:

```bash
cd python
python run_table7_baselines.py \
  --dataset-dir ../datasets \
  --results-dir results/table7 \
  --seed 42 \
  --folds 5
```

The second command writes `fold_level_results.json` (one record per dataset,
method, and fold), `fold_assignments.json` (the exact train/test indices of every
fold), `summary.csv`, and `run_config.json` (seed and all hyperparameters).
Reported accuracies, AUC, and F1-scores are the arithmetic means of the fold-level
records; no fold is removed or selected.

### MATLAB

Open MATLAB, add `matlab/` and `datasets/` to the path, and run `main.m`.
Set `kernel = 1` for the linear kernel or `kernel = 2` for the RBF kernel.

## Reproducibility notes

- The Python experiments use a fixed random seed (default `42`).
- Feature scaling is fit on the training folds only to avoid test leakage.
- Hyperparameters and the random seed used for each run are saved next to the results.

## License and contact

This code is released for research use. For questions, please open an issue in
this repository.
