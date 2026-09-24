"""Run the reproducible DeepSMOTE/LDAM/Balanced-Softmax Table 7 package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data_loader import DatasetLoader
from imbalanced_baselines import BaselineConfig, train_classifier

METHODS = ("mlp_focal", "deepsmote", "ldam", "balanced_softmax")
TABLE7_DATASETS = {
    "Australian.mat", "Diabetes.mat", "Echocardiogram.mat", "Ecoli.mat", "Fertility.mat",
    "German.mat", "Haberman.mat", "Ionosphere.mat", "Monk_1.mat", "Monk_2.mat", "Monk_3.mat",
    "Pima_Indians.mat", "Plrx.mat", "Sonar.mat", "Statlog_Heart.mat", "Votes.mat", "WDBC.mat",
    "Ecoil.mat", "Ecoli.mat",
}


def run_dataset(loader, dataset_name, config, folds, device):
    raw_x, raw_y = loader.load_mat_file(str(Path(loader.dataset_path) / dataset_name))
    raw_y = loader.label_encoder.fit_transform(raw_y)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.seed)
    rows = []
    assignments = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(raw_x, raw_y), start=1):
        assignments.append({"fold": fold, "train_indices": train_idx.tolist(), "test_indices": test_idx.tolist()})
        # Fit the scaler on each training fold only to prevent test-fold leakage.
        train_x = loader.scaler.fit_transform(raw_x[train_idx]).astype(np.float32)
        test_x = loader.scaler.transform(raw_x[test_idx]).astype(np.float32)
        train_y, test_y = raw_y[train_idx], raw_y[test_idx]
        for method in METHODS:
            metrics, metadata = train_classifier(train_x, train_y, test_x, test_y, method, config, device)
            rows.append({
                "dataset": dataset_name,
                "fold": fold,
                "method": method,
                "seed": config.seed,
                **metrics,
                **metadata,
            })
    return rows, assignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--results-dir", default="results/table7")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--all-datasets", action="store_true", help="run every .mat file instead of the 17 Table 7 datasets")
    args = parser.parse_args()
    config = BaselineConfig(seed=args.seed)
    loader = DatasetLoader(args.dataset_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    datasets = loader.get_all_datasets()
    if not args.all_datasets:
        selected = {Path(name).stem.lower().replace("_", "") for name in TABLE7_DATASETS}
        datasets = [dataset for dataset in datasets if any(
            token in Path(dataset).stem.lower().replace("_", "") for token in selected
        )]
    if not datasets:
        raise FileNotFoundError("No Table 7 datasets found; provide the .mat files with --dataset-dir")
    rows = []
    fold_assignments = {}
    for dataset in datasets:
        dataset_rows, assignments = run_dataset(loader, dataset, config, args.folds, args.device)
        rows.extend(dataset_rows)
        fold_assignments[dataset] = assignments
        pd.DataFrame(rows).to_json(results_dir / "fold_level_results.json", orient="records", indent=2)
    with (results_dir / "fold_assignments.json").open("w", encoding="utf-8") as handle:
        json.dump(fold_assignments, handle, indent=2)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["dataset", "method"], as_index=False)[["accuracy", "auc_roc", "f1_score"]].agg(["mean", "std"])
    summary.to_csv(results_dir / "summary.csv")
    with (results_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": config.to_dict(), "folds": args.folds, "device": args.device, "datasets": datasets}, handle, indent=2)


if __name__ == "__main__":
    main()
