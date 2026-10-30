import os, torch, gc, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,  
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)

CELL_TYPE_NAMES = [
    "ATL",
    "CNT",
    "DCT",
    "DTL",
    "EC",
    "FIB",
    "IC",
    "IMM",
    "NEU",
    "PC",
    "PEC",
    "POD",
    "PT",
    "PapE",
    "TAL",
    "VSM/P",
]

def compute_per_cell_metrics(y_true, y_pred, seed=None, fold=None):
    """
    Compute metrics separately for each of the 16 renal cell types.

    y_true and y_pred must have shape:
        [number_of_tiles, 16]
    """
    if y_true.shape[1] != len(CELL_TYPE_NAMES):
        raise ValueError(
            f"Expected {len(CELL_TYPE_NAMES)} targets, "
            f"but received {y_true.shape[1]}"
        )

    records = []

    for cell_idx, cell_type in enumerate(CELL_TYPE_NAMES):
        true_values = y_true[:, cell_idx]
        pred_values = y_pred[:, cell_idx]

        valid = np.isfinite(true_values) & np.isfinite(pred_values)
        true_values = true_values[valid]
        pred_values = pred_values[valid]

        if len(true_values) == 0:
            continue

        # Pearson correlation
        if (
            len(true_values) >= 3
            and np.std(true_values) > 0
            and np.std(pred_values) > 0
        ):
            pearson_r, pearson_p = pearsonr(
                true_values,
                pred_values,
            )
            spearman_r, spearman_p = spearmanr(
                true_values,
                pred_values,
            )
        else:
            pearson_r = np.nan
            pearson_p = np.nan
            spearman_r = np.nan
            spearman_p = np.nan

        # Per-cell regression metrics
        if len(true_values) >= 2:
            cell_r2 = r2_score(true_values, pred_values)
        else:
            cell_r2 = np.nan

        cell_mae = mean_absolute_error(
            true_values,
            pred_values,
        )

        cell_rmse = np.sqrt(
            mean_squared_error(
                true_values,
                pred_values,
            )
        )

        records.append(
            {
                "Seed": seed,
                "Fold": fold,
                "Cell_Index": cell_idx,
                "Cell_Type": cell_type,
                "N_Samples": len(true_values),

                # Cell abundance statistics
                "Mean_True_Abundance": np.mean(true_values),
                "Median_True_Abundance": np.median(true_values),
                "Std_True_Abundance": np.std(
                    true_values,
                    ddof=1,
                ) if len(true_values) > 1 else np.nan,
                "Nonzero_Fraction": np.mean(
                    true_values > 0
                ),

                # Predicted abundance
                "Mean_Predicted_Abundance": np.mean(
                    pred_values
                ),
                "Bias": np.mean(
                    pred_values - true_values
                ),

                # Prediction performance
                "R2": cell_r2,
                "MAE": cell_mae,
                "RMSE": cell_rmse,
                "Pearson_R": pearson_r,
                "Pearson_P": pearson_p,
                "Spearman_R": spearman_r,
                "Spearman_P": spearman_p,
            }
        )

    return pd.DataFrame(records)

def make_index_from_csv(emb_folder, label_root, csv_path):
    """
    Read a CSV file with columns [ID, Group_ID].
    For each ID, load its embedding and label (.pt files with same name)
    from emb_folder and label_root respectively.
    """
    df = pd.read_csv(csv_path)
    sample_ids, groups, ys, emb_paths = [], [], [], []

    for _, row in df.iterrows():
        emb_name = row["ID"]
        group_id = row["Group_ID"]
        base_name = os.path.splitext(emb_name)[0] + ".pt"
        emb_path = os.path.join(emb_folder, base_name)
        label_path = os.path.join(label_root, base_name)

        if not os.path.exists(emb_path):
            print(f" Missing embedding: {emb_path}")
            continue
        if not os.path.exists(label_path):
            print(f" Missing label: {label_path}")
            continue

        y = torch.load(label_path, map_location="cpu").numpy()
        sample_ids.append(os.path.splitext(base_name)[0])
        groups.append(group_id)
        ys.append(y)
        emb_paths.append(emb_path)

    return (
        np.array(sample_ids),
        np.array(groups),
        np.vstack(ys),
        emb_paths,
    )


def get_embedding(path, dtype=np.float32):
    """Load a single .pt embedding and cast to e.g. float16/32."""
    emb = torch.load(path, map_location="cpu").numpy()
    return emb.astype(dtype)

def load_regression_data(emb_folder, label_root, max_debug=None):
    """
    Now also returns `groups`: the sample_folder for each embedding.
    """
    print("→ Enter load_regression_data()")
    embs = sorted(f for f in os.listdir(emb_folder) if f.endswith(".pt"))
    if max_debug:
        embs = embs[:max_debug]
    print(f"  found {len(embs)} .pt files in {emb_folder}")
    loaded = 0

    X, y, sample_ids, groups = [], [], [], []
    for fn in embs:
        base = os.path.splitext(fn)[0]
        if "_" not in base:
            continue
        sample_folder, barcode = base.rsplit("_", 1)

        emb_path   = os.path.join(emb_folder, fn)
        label_path = os.path.join(label_root, sample_folder, "cell_types", f"{barcode}.pt")
        if not os.path.exists(label_path):
            continue

        emb = torch.load(emb_path,   map_location="cpu").numpy()
        lbl = torch.load(label_path, map_location="cpu").numpy()

        X.append(emb)
        y.append(lbl)
        sample_ids.append(base)
        groups.append(sample_folder)      # ← record group here
        # increment and print counter
        loaded += 1
        if loaded % 100 == 0:
            print(f"Loaded {loaded} embeddings so far")

    X = np.vstack(X)
    y = np.vstack(y)
    print(f" Exit load_regression_data(): X.shape={X.shape}, y.shape={y.shape}")

    return X, y, np.array(sample_ids), np.array(groups)

def compute_abundance_performance_correlation(per_cell_df):
    """
    Across the 16 cell types, determine whether more abundant cell types
    are predicted more accurately.
    """
    summary = (
        per_cell_df
        .groupby(["Cell_Index", "Cell_Type"], as_index=False)
        .agg(
            Mean_True_Abundance=(
                "Mean_True_Abundance",
                "mean",
            ),
            Median_True_Abundance=(
                "Median_True_Abundance",
                "mean",
            ),
            Nonzero_Fraction=(
                "Nonzero_Fraction",
                "mean",
            ),
            Pearson_R=(
                "Pearson_R",
                "mean",
            ),
            Spearman_R=(
                "Spearman_R",
                "mean",
            ),
            R2=(
                "R2",
                "mean",
            ),
            MAE=(
                "MAE",
                "mean",
            ),
            RMSE=(
                "RMSE",
                "mean",
            ),
        )
    )

    records = []

    abundance_metrics = [
        "Mean_True_Abundance",
        "Median_True_Abundance",
        "Nonzero_Fraction",
    ]

    performance_metrics = [
        "Pearson_R",
        "Spearman_R",
        "R2",
        "MAE",
        "RMSE",
    ]

    for abundance_metric in abundance_metrics:
        for performance_metric in performance_metrics:
            x = summary[abundance_metric].to_numpy()
            y = summary[performance_metric].to_numpy()

            valid = np.isfinite(x) & np.isfinite(y)
            x_valid = x[valid]
            y_valid = y[valid]

            if (
                len(x_valid) >= 3
                and np.std(x_valid) > 0
                and np.std(y_valid) > 0
            ):
                pearson_r_value, pearson_p = pearsonr(
                    x_valid,
                    y_valid,
                )

                spearman_r_value, spearman_p = spearmanr(
                    x_valid,
                    y_valid,
                )
            else:
                pearson_r_value = np.nan
                pearson_p = np.nan
                spearman_r_value = np.nan
                spearman_p = np.nan

            records.append(
                {
                    "Abundance_Metric": abundance_metric,
                    "Performance_Metric": performance_metric,
                    "N_Cell_Types": len(x_valid),
                    "Pearson_R": pearson_r_value,
                    "Pearson_P": pearson_p,
                    "Spearman_R": spearman_r_value,
                    "Spearman_P": spearman_p,
                }
            )

    return summary, pd.DataFrame(records)

def compute_regression_metrics(y_true, y_pred):
    # Assume shape: (N_samples, N_cell_types)
    r2   = r2_score(y_true, y_pred, multioutput='uniform_average')
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps)))
    # print(y_true.shape, y_pred.shape, "y_pred")
    # ⇨ ROW-WISE PEARSON R ACROSS CELL TYPES (i.e., column-wise across samples)
    pearsons = []
    for i in range(y_true.shape[1]):  # iterate over cell types
        try:
            r, _ = pearsonr(y_true[:, i], y_pred[:, i])
            pearsons.append(r)
        except:
            pearsons.append(np.nan)

    return {
        "r2":         r2,
        "mae":        mae,
        "rmse":       rmse,
        "mape":       mape,
        "pearson_r":  np.nanmean(pearsons)  # mean of column-wise Pearson r
    }

def run_cv_regression(X, y, sample_ids, groups, n_splits=5, seeds=(0,1,2)):
    # records, fold_results = [], []
    records = []
    fold_results = []
    per_cell_records = []
    M = X.shape[1]            # embedding dimension
    C = y.shape[1]            # number of regression targets
    alpha = 100.0 / (M * C)   # HEST-style regularization
    print(f"Using analytic ridge alpha = {alpha:.6f} (M={M}, C={C})")

    for seed in seeds:
        print(f"\n CV with seed={seed}")
        gss = GroupShuffleSplit(n_splits=n_splits, test_size=1/n_splits, random_state=seed)

        for fold_idx, (tr_idx, te_idx) in enumerate(gss.split(X, y, groups), 1):
            print(f" Fold {fold_idx}/{n_splits}: train={len(tr_idx)} test={len(te_idx)}")

            # Split data
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            # Scale features
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)

            # Train model
            model = Ridge(alpha=alpha, solver="svd")
            model.fit(X_tr_scaled, y_tr)
            y_pred = model.predict(X_te_scaled)

            # Compute metrics
            m = compute_regression_metrics(y_te, y_pred)
            print(f" r2={m['r2']:.3f}, mae={m['mae']:.3f}, rmse={m['rmse']:.3f}, "
                  f"mape={m['mape']:.3f}, pearson={m['pearson_r']:.3f}")

            records.append(
                {
                    **m,
                    "Seed": seed,
                    "Fold": fold_idx,
                }
            )

            # Save metrics separately for all 16 cell types
            per_cell_df = compute_per_cell_metrics(
                y_true=y_te,
                y_pred=y_pred,
                seed=seed,
                fold=fold_idx,
            )

            per_cell_records.append(per_cell_df)

            fold_results.append(
                {
                    "sample_ids": sample_ids[te_idx],
                    "y_true": y_te,
                    "y_pred": y_pred,
                }
            )
            # Clean up
            del X_tr, X_te, X_tr_scaled, X_te_scaled, y_tr, y_te, y_pred
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # return pd.DataFrame(records), fold_results
    per_cell_results_df = pd.concat(
        per_cell_records,
        ignore_index=True,
    )

    return (
        pd.DataFrame(records),
        per_cell_results_df,
        fold_results,
    )

def run_bootstrap_regression(fold_results, B=1000, seed=0):
    sample_map = defaultdict(list)
    for fr in fold_results:
        for sid, yt, yp in zip(fr["sample_ids"], fr["y_true"], fr["y_pred"]):
            sample_map[sid].append((yt, yp))
    all_ids = list(sample_map.keys())
    sample_size = len(fold_results[0]["sample_ids"])
    print(f"\n Bootstrap: {len(all_ids)} unique samples, B={B}")

    rng = np.random.default_rng(seed)
    boot_recs = []

    for b in range(B):
        if b % 100 == 0:
            print(f"  → bootstrap iteration {b}/{B}")
        chosen = rng.choice(all_ids, size=sample_size, replace=True)
        yt_all, yp_all = [], []
        for sid in chosen:
            i = rng.integers(len(sample_map[sid]))
            yt_i, yp_i = sample_map[sid][i]
            yt_all.append(yt_i)
            yp_all.append(yp_i)

        yt_all = np.vstack(yt_all)
        yp_all = np.vstack(yp_all)
        boot_recs.append(compute_regression_metrics(yt_all, yp_all))

    boot_df = pd.DataFrame(boot_recs)
    ci_summary = pd.DataFrame({
        metric: [
            boot_df[metric].mean(),
            boot_df[metric].quantile(0.025),
            boot_df[metric].quantile(0.975)
        ]
        for metric in boot_df.columns
    }, index=["mean", "ci_lo", "ci_hi"])
    print("\n Bootstrap CI summary:\n", ci_summary)
    return boot_df, ci_summary
fm_names = ["UNI", "UNI2-h", "Virchow", "Virchow2", "SP22M", "SP85M", "H-optimus-0", "H-optimus-1", "Prov-Gigapath", "Hibou-B", "Hibou-L"]


def run_regression_probe(
    emb_root,
    label_root,
    csv_file,
    out_dir="results_regression",
    models=None,
    n_splits=5,
    seeds=(0, 1, 2),
    bootstrap_iters=1000
):
    """
    Runs ridge regression probing with GroupShuffleSplit cross-validation and bootstrap CI.
    Can be imported and reused as part of a package.
    """

    all_models = [
        "UNI", "UNI2-h", "Virchow", "Virchow2",
        "SP22M", "SP85M", "H-optimus-0", "H-optimus-1",
        "Prov-Gigapath", "Hibou-B", "Hibou-L"
    ]
    models = models or all_models

    os.makedirs(out_dir, exist_ok=True)

    for fm in models:
        print(f"\n=== Running regression probe for model: {fm} ===")

        embedding_dir = os.path.join(emb_root, fm)
        label_dir = label_root

        if not os.path.exists(embedding_dir):
            print(f"[skip] Missing embeddings dir: {embedding_dir}")
            continue
        if not os.path.exists(label_dir):
            print(f"[skip] Missing labels dir: {label_dir}")
            continue
        if not os.path.exists(csv_file):
            print(f"[skip] Missing CSV file: {csv_file}")
            continue

        print("→ Reading CSV and matching embeddings/labels…")
        sample_ids, groups, y, emb_paths = make_index_from_csv(
            emb_folder=embedding_dir,
            label_root=label_dir,
            csv_path=csv_file
        )

        if len(sample_ids) == 0:
            print(f"[skip] No valid samples found for {fm}")
            continue

        print(f"Loaded {len(sample_ids)} samples | unique groups: {len(np.unique(groups))}")

        # Load all embeddings into memory
        X = np.vstack([get_embedding(p) for p in emb_paths])

        print("→ Running Ridge regression CV and bootstrap…")
        # cv_df, fold_results = run_cv_regression(
        #     X, y, sample_ids, groups,
        #     n_splits=n_splits,
        #     seeds=tuple(seeds)
        # )
        cv_df, per_cell_df, fold_results = run_cv_regression(
            X,
            y,
            sample_ids,
            groups,
            n_splits=n_splits,
            seeds=tuple(seeds),
        )
        fm_outdir = os.path.join(out_dir, fm)
        os.makedirs(fm_outdir, exist_ok=True)

        # --- Save CV results ---
        # out_cv = os.path.join(fm_outdir, f"cv_results_LR_{fm}.csv")
        # cv_df.to_csv(out_cv, index=False)
        # print(f" Saved regression CV results → {out_cv}")
        # Save fold-wise cell-type-specific metrics
        out_per_cell = os.path.join(
            fm_outdir,
            f"per_cell_metrics_LR_{fm}.csv",
        )

        per_cell_df.to_csv(
            out_per_cell,
            index=False,
        )

        # Average the metrics across seeds and folds for each cell type,
        # and calculate whether prediction performance is associated
        # with cell abundance.
        per_cell_summary, abundance_correlation_df = (
            compute_abundance_performance_correlation(
                per_cell_df
            )
        )

        out_per_cell_summary = os.path.join(
            fm_outdir,
            f"per_cell_summary_LR_{fm}.csv",
        )

        out_abundance_correlation = os.path.join(
            fm_outdir,
            f"abundance_performance_correlation_LR_{fm}.csv",
        )

        per_cell_summary.to_csv(
            out_per_cell_summary,
            index=False,
        )

        abundance_correlation_df.to_csv(
            out_abundance_correlation,
            index=False,
        )

        print(
            f" Saved per-cell summary → "
            f"{out_per_cell_summary}"
        )

        print(
            f" Saved abundance-performance correlations → "
            f"{out_abundance_correlation}"
        )
        
        print(
            f" Saved per-cell metrics → {out_per_cell}"
        )
        # --- Run bootstrapping ---
        boot_df, ci_summary = run_bootstrap_regression(
            fold_results, B=bootstrap_iters, seed=1
        )

        # --- Save bootstrap results ---
        out_boot = os.path.join(fm_outdir, f"bootstrap_replicates_LR_{fm}.csv")
        out_ci   = os.path.join(fm_outdir, f"bootstrap_CI_LR_{fm}.csv")
        boot_df.to_csv(out_boot, index=False)
        ci_summary.to_csv(out_ci)
        print(f" Saved bootstrap results → {out_boot}, {out_ci}")


if __name__ == "__main__":
    main()
