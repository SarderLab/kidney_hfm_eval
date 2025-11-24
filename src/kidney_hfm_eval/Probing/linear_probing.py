import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import os
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import random, argparse
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, multilabel_confusion_matrix
)
from sklearn.preprocessing import label_binarize

seed = 42
random.seed(seed)
np.random.seed(seed)              # for any legacy numpy.random calls
os.environ["OMP_NUM_THREADS"] = "1"
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
VERBOSE = 1  # 0=silent, 1=key info, 2=deep debug

def vprint(*args, level=1, **kwargs):
    if VERBOSE >= level:
        print(*args, **kwargs)

def class_counts(y):
    u, c = np.unique(y, return_counts=True)
    return dict(zip(u, c))

def load_from_csv(csv_path, embeddings_dir):
    """
    Read rows from a CSV file with columns [ID, class, Group_ID].
    For each ID, load its .pt embedding from embeddings_dir.
    """
    df = pd.read_csv(csv_path)
    embeds, groups, sample_ids, labels = [], [], [], []
    
    for _, row in df.iterrows():
        fn = row["ID"]
        label = row["class"]
        group = row["Group_ID"]
        
        # load .pt embedding
        # pt_path = os.path.join(embeddings_dir, fn.replace(".png, .jpg, .jpeg", ".pt"))
        base = os.path.splitext(fn)[0]  # removes .png, .jpg, .jpeg, etc.
        pt_path = os.path.join(embeddings_dir, base + ".pt")

        emb = torch.load(pt_path).cpu().numpy()
        
        embeds.append(emb)
        groups.append(group)
        sample_ids.append(fn)
        labels.append(label)
    
    return (
        np.vstack(embeds),
        np.array(groups),
        np.array(sample_ids),
        np.array(labels)
    )

def compute_metrics(y_true, y_pred, y_prob=None, classes=None):
    m = {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc":               matthews_corrcoef(y_true, y_pred),
    }
    classes = list(classes) if classes is not None else [0, 1]
    uniq = np.unique(y_true)

    if len(classes) == 2:  # ── binary ───────────────────────────────────────────
        m["f1"]        = f1_score(y_true, y_pred, zero_division=0)
        m["precision"] = precision_score(y_true, y_pred, zero_division=0)
        m["recall"]    = recall_score(y_true, y_pred, zero_division=0)

        # force a 2x2 confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=classes).ravel()
        denom = tn + fp
        m["specificity"] = (tn / denom) if denom > 0 else np.nan

        if y_prob is not None:
            # choose the column that corresponds to positive label (1 by convention)
            if y_prob.ndim == 1:
                pos_prob = y_prob
            else:
                pos_label = 1 if 1 in classes else classes[-1]
                pos_idx = classes.index(pos_label)
                pos_prob = y_prob[:, pos_idx]
            try:
                m["auroc"] = roc_auc_score(y_true, pos_prob)
            except ValueError:
                m["auroc"] = np.nan
            try:
                m["auprc"] = average_precision_score(y_true, pos_prob)
            except ValueError:
                m["auprc"] = np.nan                
                
    else:  # ── multiclass (macro) ────────────────────────────────────────────
        m.update({
            "f1":        f1_score(y_true, y_pred, average='macro', zero_division=0),
            "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "recall":    recall_score(y_true, y_pred, average='macro', zero_division=0),
        })
        cm = multilabel_confusion_matrix(y_true, y_pred, labels=classes)
        tn = cm[:, 0, 0]; fp = cm[:, 0, 1]
        spec = np.where((tn+fp) > 0, tn/(tn+fp), np.nan)
        m["specificity"] = np.nanmean(spec)

        if (y_prob is not None) and (classes is not None):
            try:
                y_bin = label_binarize(y_true, classes=classes)
                m["auroc"] = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
            except ValueError:
                m["auroc"] = np.nan
            try:
                y_bin = label_binarize(y_true, classes=classes)
                m["auprc"] = average_precision_score(y_bin, y_prob, average='macro')
            except ValueError:
                m["auprc"] = np.nan

    return m

def evaluate_logistic(X_train, y_train, X_test, y_test, classes=None):
    M = X_train.shape[1]
    C_classes = len(np.unique(y_train))
    lam = 100.0 / (M * C_classes)
    C_param = 1.0 / lam

    lr = LogisticRegression(
        penalty='l2',
        C=C_param,
        solver='lbfgs',
        max_iter=1000
    )
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    probs = lr.predict_proba(X_test)       # shape = (n_samples, n_classes)
    # vprint(f"[LR] classes_={lr.classes_} aligned_to={classes} probs_shape={probs.shape}", level=2)
    # assert probs.shape[1] == len(classes)
    # assert np.allclose(probs.sum(axis=1), 1, atol=1e-6), "probs rows not summing to 1"
    if classes is None:
        classes = list(lr.classes_)
    order = [list(lr.classes_).index(c) for c in classes]
    probs = probs[:, order]
    vprint(f"[LR] classes_={lr.classes_} aligned_to={classes} probs_shape={probs.shape}", level=2)
    assert probs.shape[1] == len(classes)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-6)

    return compute_metrics(y_test, preds, probs, classes), y_test, probs

def run_cross_validation(X, y, groups, sample_ids, classes, seeds=(0,1,2), n_splits=5):
    """
    Performs StratifiedGroupKFold CV over (X,y) with grouping=groups,
    returns a DataFrame of per‐fold metrics plus a list of dicts for bootstrap.
    """
    records = []
    fold_results = []

    for seed in seeds:
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # — LogisticRegression —
            lr_m, y_true_lr, y_prob_lr = evaluate_logistic(
                X_tr_s, y_tr, X_te_s, y_te, classes
            )
            print(y_true_lr.shape, "y_true_lr", y_prob_lr.shape, "y_prob_lr")
            records.append({
                **lr_m,
                "Model": "LogisticRegression",
                "Seed": seed,
                "Fold": fold_idx
            })
            fold_results.append({
                "ids":    sample_ids[test_idx],
                "y_true":    y_true_lr,
                "y_prob":    y_prob_lr
            })

    return pd.DataFrame(records), fold_results


def run_bootstrap(fold_results, classes, B=1000, seed=0, sample_frac=None):
    """
    Group‐wise bootstrap:
     1) Build a map WSI_ID → list of (y_true, y_prob) from all folds.
     2) Determine how many IDs to sample (either a fraction of unique IDs, or
        default = avg. number of samples per fold).
     3) For each bootstrap run:
        • Randomly pick that many IDs (no replacement).
        • For each chosen ID, pick one (y_true, y_prob) at random.
        • Compute metrics on the assembled set.
     Returns the full bootstraps DataFrame and a CI summary DataFrame.
    """
    # 1) build mapping
    grp_map = defaultdict(list)
    for fr in fold_results:
        for sampleid, yt, yp in zip(fr["ids"], fr["y_true"], fr["y_prob"]):
            grp_map[sampleid].append((yt, yp))
    all_ids = list(grp_map.keys())
    counts = [len(v) for v in grp_map.values()]
    print("Tiles with 1 prediction:", counts.count(1))
    print("Tiles with 2 predictions:", counts.count(2))
    print("Tiles with 3 predictions:", counts.count(3))
    print("Total predictions:", sum(counts))
    print("Dataset size (#unique tiles):", len(counts))
    print("Average predictions per tile:", sum(counts) / len(counts))
    print("=============================================")

    # 2) sample_size
    if sample_frac is None:
        total_test_samples = sum(len(fr["y_true"]) for fr in fold_results)
        sample_size = total_test_samples // len(fold_results)
    else:
        sample_size = int(len(all_ids) * sample_frac)

    rng = np.random.default_rng(seed)
    boot_recs = []
    print(f"Bootstrap sampling {sample_size} unique IDs per run")
    for b in range(B):
        if b and (b % 100 == 0):
            print(f"  → bootstrap iteration {b}/{B}")
        chosen = rng.choice(all_ids, size=sample_size, replace=True)

        yb, pb = [], []
        for sid in chosen:
            # yt_i, yp_i = random.choice(grp_map[gid])
            i = rng.integers(len(grp_map[sid]))
            yt_i, yp_i = grp_map[sid][i]
            yb.append(yt_i)
            pb.append(yp_i)

        yb = np.array(yb)
        pb = np.vstack(pb)
        # preds = np.argmax(pb, axis=1)
        pred_idx = np.argmax(pb, axis=1)
        preds = np.asarray(classes)[pred_idx]
        boot_recs.append(compute_metrics(yb, preds, pb, classes))

    boot_df = pd.DataFrame(boot_recs)
    # 5) build CI summary
    summary = pd.DataFrame({
        metric: [
            boot_df[metric].mean(),
            boot_df[metric].quantile(0.025),
            boot_df[metric].quantile(0.975)
        ]
        for metric in boot_df.columns
    }, index=["mean", "ci_lo", "ci_hi"])

    return boot_df, summary

# def main():
#     parser = argparse.ArgumentParser(description="Run CV + bootstrap evaluation on embeddings from CSV")
#     parser.add_argument(
#         "--models", nargs="+", default=[],
#         help="Which FMs to run (space-separated). Default = all available"
#     )
#     parser.add_argument(
#         "--emb_dir", type=str, required=True,
#         help="Root directory containing per-model embeddings"
#     )
#     parser.add_argument(
#     "--out_dir", type=str, default="results",
#     help="Directory where results will be saved"
#     )
#     parser.add_argument(
#         "--csv_file", type=str, required=True,
#         help="Single CSV file (with ID, Group_ID, class) used by all foundation models"
#     )

#     parser.add_argument("--splits", type=int, default=5, help="Number of CV folds")
#     parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2], help="Random seeds for CV")
#     parser.add_argument("--bootstrap", type=int, default=1000, help="Number of bootstrap replicates")
#     args = parser.parse_args()

#     # default FM list if not provided
#     all_fms = ["UNI", "UNI2-h", "Virchow", "Virchow2",
#                "SP22M", "SP85M", "H-optimus-0", "H-optimus-1",
#                "Prov-Gigapath", "Hibou-B", "Hibou-L"]
#     fm_names = args.models if args.models else all_fms

#     for fm in fm_names:
#         print(f"\n=== Running {fm} ===")
#         csv_path = args.csv_file
#         embeddings_dir = os.path.join(args.emb_dir, fm)

#         if not os.path.exists(csv_path):
#             print(f"[skip] Missing CSV: {csv_path}")
#             continue
#         if not os.path.exists(embeddings_dir):
#             print(f"[skip] Missing embeddings dir: {embeddings_dir}")
#             continue

#         # load
#         X, groups, samples, y = load_from_csv(csv_path, embeddings_dir)
#         classes = sorted(np.unique(y))

#         # ── CROSS-VALIDATION ──
#         cv_df, fold_results = run_cross_validation(X, y, groups, samples,
#                                                    classes, seeds=args.seeds,
#                                                    n_splits=args.splits)
#         fm_outdir = os.path.join(args.out_dir, fm)
#         os.makedirs(fm_outdir, exist_ok=True)

#         cv_out   = os.path.join(fm_outdir, f"cv_results_LR_{fm}.csv")
#         boot_out = os.path.join(fm_outdir, f"bootstrap_replicates_LR_{fm}.csv")
#         ci_out   = os.path.join(fm_outdir, f"bootstrap_CI_LR_{fm}.csv")

#         # save cross-validation
#         cv_df.to_csv(cv_out, index=False)

#         # run and save bootstrap
#         boot_df, ci_summary = run_bootstrap(fold_results, classes, B=args.bootstrap, seed=1)
#         boot_df.to_csv(boot_out, index=False)
#         ci_summary.to_csv(ci_out)

#         print(f"✅ Saved results for {fm} → {cv_out}, {boot_out}, {ci_out}")

def run_linear_probe(csv_file, emb_root, out_dir, models=None, n_splits=5, seeds=(0,1,2), bootstrap=1000):
    all_models = ["UNI", "UNI2-h", "Virchow", "Virchow2", "SP22M", "SP85M",
                  "H-optimus-0", "H-optimus-1", "Prov-Gigapath", "Hibou-B", "Hibou-L"]
    models = models or all_models

    os.makedirs(out_dir, exist_ok=True)
    for fm in models:
        print(f"\n=== Running {fm} ===")
        embeddings_dir = os.path.join(emb_root, fm)
        if not (os.path.exists(csv_file) and os.path.exists(embeddings_dir)):
            print(f"[skip] Missing CSV or embeddings dir for {fm}")
            continue

        X, groups, samples, y = load_from_csv(csv_file, embeddings_dir)
        classes = sorted(set(y))
        cv_df, fold_results = run_cross_validation(X, y, groups, samples, classes, seeds=seeds, n_splits=n_splits)
        fm_outdir = os.path.join(out_dir, fm)
        os.makedirs(fm_outdir, exist_ok=True)
        cv_df.to_csv(os.path.join(fm_outdir, f"cv_results_LR_{fm}.csv"), index=False)
        boot_df, ci_summary = run_bootstrap(fold_results, classes, B=bootstrap, seed=1)
        boot_df.to_csv(os.path.join(fm_outdir, f"bootstrap_replicates_LR_{fm}.csv"), index=False)
        ci_summary.to_csv(os.path.join(fm_outdir, f"bootstrap_CI_LR_{fm}.csv"))
        print(f"✅ Saved results for {fm}")


if __name__ == "__main__":
    main()
