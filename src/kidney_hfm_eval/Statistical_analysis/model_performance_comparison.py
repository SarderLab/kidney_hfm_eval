import glob, os, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, friedmanchisquare
from statsmodels.stats.multitest import multipletests
from itertools import combinations

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
MODEL_ORDER = [
    "UNI", "UNI2-h", "Virchow", "Virchow2",
    "SP22M", "SP85M", "H-optimus-0", "H-optimus-1",
    "Prov-Gigapath", "Hibou-B", "Hibou-L"
]
COLOR_PALETTE = sns.color_palette("tab20", n_colors=len(MODEL_ORDER))

MODEL_COLORS = {model: COLOR_PALETTE[i] for i, model in enumerate(MODEL_ORDER)}

def sort_models_by_cld(cld_map, series_map):
    """
    Sort models by CLD letter first (a < b < c < ...),
    then within group by mean performance (high → low).
    """
    grouped = {}
    for m, g in cld_map.items():
        grouped.setdefault(g, []).append(m)

    # sort groups alphabetically a < b < c ...
    sorted_models = []
    for g in sorted(grouped.keys()):
        # sort models within group by descending mean performance
        models = grouped[g]
        models_sorted = sorted(models, key=lambda m: series_map[m].mean(), reverse=True)
        sorted_models.extend(models_sorted)

    return sorted_models


def generate_cld_groups(p_holm, models, alpha=0.05):
    """
    True Compact Letter Display Groups based on complete-link rule.
    All models within a group must be mutually NOT significantly different.
    """

    # Significant differences matrix
    sig = p_holm < alpha
    np.fill_diagonal(sig.values, False)

    # Start with each model in its own group
    groups = {m: set([m]) for m in models}

    # Try to merge groups only when fully NSD
    for a, b in combinations(models, 2):
        if not sig.loc[a, b]:  # NSD → candidate for grouping

            merged = groups[a] | groups[b]

            # Check every pair in this merged set
            fully_nsd = True
            for x, y in combinations(merged, 2):
                if sig.loc[x, y]:  # any significant → fail
                    fully_nsd = False
                    break

            # If valid, merge groups
            if fully_nsd:
                for m in merged:
                    groups[m] = merged

    # Assign group letters based on unique merged sets
    unique_groups = {}
    letter = ord('a')
    cld_map = {}

    for m in models:
        group = tuple(sorted(groups[m]))
        if group not in unique_groups:
            unique_groups[group] = chr(letter)
            letter += 1
        cld_map[m] = unique_groups[group]

    return cld_map

def relabel_cld_by_performance(cld_map, series_map):
    """
    Reassign CLD labels so that:
    best-performing group = 'a'
    second best group = 'b'
    third best group = 'c'
    ...
    """

    # Step 1: group models by original CLD letter
    group_to_models = {}
    for m, g in cld_map.items():
        group_to_models.setdefault(g, []).append(m)

    # Step 2: compute mean performance of each CLD group
    group_mean = {
        g: np.mean([series_map[m].mean() for m in group_to_models[g]])
        for g in group_to_models
    }

    # Step 3: sort groups by performance (descending)
    sorted_groups = sorted(group_mean.keys(), key=lambda g: group_mean[g], reverse=True)

    # Step 4: assign new CLD letters
    new_cld_map = {}
    new_letter = ord('a')

    for g in sorted_groups:
        for m in group_to_models[g]:
            new_cld_map[m] = chr(new_letter)
        new_letter += 1

    return new_cld_map

def plot_box_ungrouped(series_map, models, metric_name="MCC", save_dir=None, suffix=""):
    df_long = pd.concat(
        [pd.DataFrame({"Model": m, metric_name: series_map[m].values})
         for m in models],
        ignore_index=True
    )

    plt.figure(figsize=(16, 8))
    ax = sns.boxplot(
        data=df_long,
        x="Model",
        y=metric_name,
        order=models,
        palette={m: MODEL_COLORS[m] for m in models},   # <--- per-model color
        linewidth=1.8,
        fliersize=3,
        whiskerprops={"linewidth": 1.6},
        capprops={"linewidth": 1.6},
        boxprops={"linewidth": 1.8},
        medianprops={"linewidth": 2.2, "color": "black"},
    )

    # ---- Add Legend ----
    handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m], lw=8) for m in models]
    ax.legend(handles, models, title="Models", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.title(f"{metric_name} – Ungrouped Boxplot", fontsize=20)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_plot(save_dir, f"ungrouped_boxplot_{metric_name}{suffix}.png")
    plt.show()


def plot_violin_ungrouped(series_map, models, metric_name="MCC", save_dir=None, suffix=""):
    df_long = pd.concat(
        [pd.DataFrame({"Model": m, metric_name: series_map[m].values})
         for m in models],
        ignore_index=True
    )

    plt.figure(figsize=(16, 8))
    ax = sns.violinplot(
        data=df_long,
        x="Model",
        y=metric_name,
        order=models,
        palette={m: MODEL_COLORS[m] for m in models},
        inner="quartile",
        cut=0,
        density_norm="width",
        linewidth=1.5,
        saturation=1.0,
        alpha=0.9
    )

    # ---- Add Legend ----
    handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m], lw=8) for m in models]
    ax.legend(handles, models, title="Models", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.title(f"{metric_name} – Ungrouped Violin Plot", fontsize=20)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_plot(save_dir, f"ungrouped_violin_{metric_name}{suffix}.png")
    plt.show()
    
def plot_box_grouped(series_map, models, cld_map, metric_name="MCC",
                     save_dir=None, suffix=""):

    sorted_models = sort_models_by_cld(cld_map, series_map)

    df_long = pd.concat(
        [pd.DataFrame({
            "Model": m,
            metric_name: series_map[m].values,
            "CLD": cld_map[m]
        }) for m in sorted_models],
        ignore_index=True
    )

    plt.figure(figsize=(18, 8))
    ax = sns.boxplot(
        data=df_long,
        x="Model",
        y=metric_name,
        order=sorted_models,
        palette={m: MODEL_COLORS[m] for m in sorted_models},
        linewidth=1.8,
        fliersize=3,
        whiskerprops={"linewidth": 1.6},
        capprops={"linewidth": 1.6},
        boxprops={"linewidth": 1.8},
        medianprops={"linewidth": 2.2, "color": "black"},
    )

    # ---- Add CLD letters above each model ----
    ymin, ymax = ax.get_ylim()
    ypos = ymax + 0.05 * (ymax - ymin)

    for i, m in enumerate(sorted_models):
        ax.text(i, ypos, cld_map[m], ha="center",
                fontsize=14, fontweight="bold")

    # ---- Expand y-limit to ensure visibility ----
    ax.set_ylim(ymin, ymax + 0.10 * (ymax - ymin))

    # plt.title(f"{metric_name} – Boxplot Grouped by CLD (Sorted)", fontsize=20)
    plt.title(f"Model Performance with CLD-Based Grouping and Ranking", fontsize=20)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_plot(save_dir, f"grouped_boxplot_CLD_sorted_{metric_name}{suffix}.png")
    plt.show()

def plot_violin_grouped(series_map, models, cld_map, metric_name="MCC",
                        save_dir=None, suffix=""):

    sorted_models = sort_models_by_cld(cld_map, series_map)

    df_long = pd.concat(
        [pd.DataFrame({
            "Model": m,
            metric_name: series_map[m].values,
            "CLD": cld_map[m]
        }) for m in sorted_models],
        ignore_index=True
    )

    plt.figure(figsize=(18, 8))
    ax = sns.violinplot(
        data=df_long,
        x="Model",
        y=metric_name,
        order=sorted_models,
        palette={m: MODEL_COLORS[m] for m in sorted_models},
        inner="quartile",
        cut=0,
        density_norm="width",
        saturation=1.0,
        alpha=0.9,
        linewidth=1.5
    )

    # ---- Add CLD letters above each model ----
    ymin, ymax = ax.get_ylim()
    ypos = ymax + 0.05 * (ymax - ymin)

    for i, m in enumerate(sorted_models):
        ax.text(i, ypos, cld_map[m], ha="center",
                fontsize=14, fontweight="bold")

    # ---- Expand y-limit to ensure visibility ----
    ax.set_ylim(ymin, ymax + 0.10 * (ymax - ymin))

    # plt.title(f"{metric_name} – Violin Plot Grouped by CLD (Sorted)", fontsize=20)
    plt.title(f"Model Performance with CLD-Based Grouping and Ranking", fontsize=20)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_plot(save_dir, f"grouped_violinplot_CLD_sorted_{metric_name}{suffix}.png")
    plt.show()

def make_cld_table(series_map, cld_map, metric="mcc"):
    rows = []
    for model, series in series_map.items():
        rows.append([
            cld_map.get(model),
            model,
            series.min(),
            series.quantile(0.25),
            series.median(),
            series.mean(),
            series.quantile(0.75),
            series.max()
        ])
    df = pd.DataFrame(rows, columns=[
        "CLD", "Model",
        "Min", "1st Quart.", "Median", "Mean", "3rd Quart.", "Max"
    ])

    # Sort by CLD group first, then by mean within each group
    df = df.sort_values(["CLD", "Mean"], ascending=[True, False]).reset_index(drop=True)

    return df

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_plot(save_dir, filename, dpi=300):
    """Save current matplotlib figure to directory."""
    if save_dir:
        full_path = os.path.join(save_dir, filename)
        plt.savefig(full_path, bbox_inches="tight", dpi=dpi)
        print(f" Saved figure: {full_path}")


def load_replicate_series(repl_glob, metric="mcc"):
    """Load replicate CSV files containing 'mcc' column and return a dict of Series."""
    rep_files = glob.glob(repl_glob)
    if not rep_files:
        raise FileNotFoundError(f"No replicate files found with pattern: {repl_glob}")

    series_map = {}
    rep_counts = {}
    for fp in rep_files:
        model = os.path.basename(fp).split('_')[-1].replace('.csv', '')
        df = pd.read_csv(fp)
        if metric not in df.columns:
            raise KeyError(f" Column '{metric}' not found in file: {fp}")
        s = df[metric].astype(float).dropna().reset_index(drop=True)
        series_map[model] = s
        rep_counts[model] = len(s)
    return series_map, rep_counts


def make_summary_df(series_map, model_order, metric="mcc"):
    """Compute mean and 95% CI for each model."""
    models = [m for m in model_order if m in series_map]
    rows = []
    for m in models:
        s = series_map[m]
        rows.append((m, s.mean(), s.quantile(0.025), s.quantile(0.975)))
    return pd.DataFrame(rows, columns=['model', f"mean_{metric}", 'ci_lo', 'ci_hi']).set_index('model')

def plot_summary(summary_df, save_dir=None, metric_name="MCC", suffix=""):
    """Plot mean metric with 95% CI and prevent negative yerr values."""
    plt.figure(figsize=(10, 6))

    # Compute non-negative error bars
    mean_col = f"mean_{metric_name.lower()}"
    err_lower = (summary_df[mean_col] - summary_df["ci_lo"]).clip(lower=0)
    err_upper = (summary_df["ci_hi"] - summary_df[mean_col]).clip(lower=0)
    yerr = np.vstack([err_lower, err_upper])

    # Detect and report any inverted CIs (for debugging)
    bad = (summary_df["ci_hi"] < summary_df[mean_col]) | (summary_df["ci_lo"] > summary_df[mean_col])
    if bad.any():
        print(" Some CI bounds inverted due to rounding:")
        print(summary_df[bad])

    # Plot safely
    plt.bar(
        summary_df.index,
        summary_df[mean_col],
        yerr=yerr,
        capsize=5,
        edgecolor="black"
    )

    # Decorate
    plt.ylabel(metric_name, fontsize=16)
    plt.xlabel("Model", fontsize=16)
    plt.title(f"Mean {metric_name} per Model (95% CI)", fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.tight_layout()

    # Save and close
    save_plot(save_dir, f"mean_{metric_name}_barplot{suffix}.png")
    plt.close()


def plot_violin(series_map, models, save_dir=None, metric_name="MCC", suffix=""):
    """Plot violin distribution of replicate MCCs."""
    df_long = pd.concat(
        [pd.DataFrame({"Model": m, metric_name: series_map[m].values}) for m in models],
        ignore_index=True
    )
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_long, x="Model", y= metric_name, order=models, inner="box", cut=0, scale="width")
    # sns.stripplot(data=df_long, x="Model", y="MCC", order=models, color="black", size=2, alpha=0.5)
    plt.ylabel(metric_name, fontsize=16)
    plt.xlabel("Model", fontsize=16)
    plt.title(f"Distribution of {metric_name} Replicates per Model", fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.tight_layout()
    save_plot(save_dir, f"violin_{metric_name}_distribution{suffix}.png")
    plt.show()


def run_wilcoxon_posthoc(df_paired, models, alpha=0.05, save_dir=None, metric_name="MCC", suffix=""):
    """Run Friedman + Wilcoxon + Holm correction and plot heatmaps."""
    k = len(models)
    friedman_stat, friedman_p = friedmanchisquare(*[df_paired[m].values for m in models])
    print(f"Friedman test: Q={friedman_stat:.4f}, df={k-1}, p={friedman_p:.6g}")

    if friedman_p > alpha:
        print("Friedman not significant; skipping post-hoc pairwise tests.")
        return None, None, None

    print("\nRunning pairwise Wilcoxon signed-rank tests with Holm–Bonferroni correction...")
    pairs = list(itertools.combinations(models, 2))
    raw_p_list, stat_list = [], []

    for a, b in pairs:
        x, y = df_paired[a].values, df_paired[b].values
        try:
            res = wilcoxon(x, y, alternative='two-sided', zero_method='wilcox')
            stat_list.append(res.statistic)
            raw_p_list.append(res.pvalue)
        except ValueError:
            stat_list.append(np.nan)
            raw_p_list.append(np.nan)

    valid_mask = ~np.isnan(raw_p_list)
    valid_pairs = [pairs[i] for i, ok in enumerate(valid_mask) if ok]
    valid_pvals = [raw_p_list[i] for i, ok in enumerate(valid_mask) if ok]

    # Holm correction
    adj_p_list = multipletests(valid_pvals, alpha=alpha, method='holm')[1]
    hb_map = {pair: p_adj for pair, p_adj in zip(valid_pairs, adj_p_list)}

    # Build symmetric matrices
    p_raw = pd.DataFrame(np.ones((k, k)), index=models, columns=models)
    p_holm = pd.DataFrame(np.ones((k, k)), index=models, columns=models)
    for (a, b), p in zip(valid_pairs, valid_pvals):
        p_raw.loc[a, b] = p_raw.loc[b, a] = p
        p_holm.loc[a, b] = p_holm.loc[b, a] = hb_map[(a, b)]
    np.fill_diagonal(p_raw.values, 0)
    np.fill_diagonal(p_holm.values, 0)

    # --- Raw p-value heatmap ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(p_raw, annot=True, fmt=".3f", cmap="viridis_r",
                xticklabels=models, yticklabels=models,
                cbar_kws={'label': 'Raw p-value'})
    plt.title("Pairwise Wilcoxon signed-rank (Raw p-values)", fontsize=18)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_plot(save_dir, f"wilcoxon_raw_pvalues_{metric_name}_{suffix}.png")
    plt.show()

    # --- Holm–Bonferroni adjusted p-value heatmap ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(p_holm, annot=True, fmt=".3f", cmap="viridis_r",
                xticklabels=models, yticklabels=models,
                cbar_kws={'label': 'Holm-adjusted p-value'})
    plt.title("Pairwise Wilcoxon signed-rank (Holm–Bonferroni corrected)", fontsize=18)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_plot(save_dir, f"wilcoxon_holm_corrected_{metric_name}_{suffix}.png")
    plt.show()

    # Save numeric tables too
    if save_dir:
        p_raw.to_csv(os.path.join(save_dir, f"wilcoxon_raw_pvalues_{metric_name}_{suffix}.csv"))
        p_holm.to_csv(os.path.join(save_dir, f"wilcoxon_holm_corrected_{metric_name}_{suffix}.csv"))
        print(f" Saved p-value matrices to {save_dir}")

    return friedman_p, p_raw, p_holm

def plot_cld_violin(series_map, models, cld_map, metric_name="MCC", save_dir=None, suffix=""):
    df_long = pd.concat(
        [pd.DataFrame({
            "Model": m,
            metric_name: series_map[m].values,
            "CLD": cld_map[m]
        }) for m in models],
        ignore_index=True
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_long, x="Model", y=metric_name, order=models)
    
    # Add CLD letters above boxes
    positions = range(len(models))
    for i, m in enumerate(models):
        y = df_long[df_long["Model"] == m][metric_name].max() + 0.02
        plt.text(i, y, cld_map[m], ha="center", fontsize=14, fontweight="bold")

    plt.title(f"{metric_name} distribution with CLD grouping")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_plot(save_dir, f"CLD_violin_{metric_name}{suffix}.png")
    plt.show()

def run_replicate_pipeline(args: dict):
    """
    Run full replicate analysis using an argument dictionary.
    Expected keys in `args`:
      - repl_glob: glob pattern for replicate CSVs
      - save_dir: directory to save outputs
      - models: optional list of models to include manually
      - model_order: optional list defining order of models in plots
      - alpha: significance level for tests (default=0.05)
    """
    repl_glob = args.get("repl_glob")
    save_dir = args.get("save_dir")
    alpha = args.get("alpha", 0.05)
    verbose = args.get("verbose", True)
    metric = args.get("metric", "mcc").lower()
    manual_models = args.get("models", None)
    model_order = args.get("model_order", MODEL_ORDER)
    tag = args.get("tag", "")  # e.g., "LR", "kNN"
    suffix = f"_{tag}" if tag else ""

    if not repl_glob:
        raise ValueError(" Missing required argument: 'repl_glob'")

    if verbose:
        print(f"\n Loading replicate data from: {repl_glob}")
        print(f" Using metric: {metric.upper()}")

    # Step 1: Load all replicate CSVs and extract the chosen metric
    series_map, rep_counts = load_replicate_series(repl_glob, metric=metric)

    # Step 2: Select models (now series_map exists)
    if manual_models:
        models = [m for m in manual_models if m in series_map]
        if verbose:
            print(f"Using user-specified models: {models}")
    else:
        models = [m for m in model_order if m in series_map]
        if verbose:
            print(f"Detected models from directory: {models}")

    if not models:
        raise ValueError(" No valid models found in the directory or args['models'].")

    # Step 3: Create output directory if needed
    if save_dir:
        ensure_dir(save_dir)

    # Step 4: Compute summary stats and save
    summary_df = make_summary_df(series_map, model_order, metric)

    # --- Step 4: Save summary ---
    summary_path = os.path.join(save_dir, f"summary_{metric}_95CI{suffix}.csv")
    summary_df.to_csv(summary_path)
    print(f" Saved summary table: {summary_path}")


    # Step 6: Statistical comparison (Friedman + Wilcoxon)
    df_paired = pd.DataFrame({m: series_map[m] for m in models}).dropna()
    if verbose:
        print(f" Paired matrix shape: {df_paired.shape}")

    # --- Step 5: Plots ---
    plot_summary(summary_df.loc[models], save_dir, metric_name=metric.upper(), suffix=suffix)
    plot_violin(series_map, models, save_dir, metric_name=metric.upper(), suffix=suffix)

    # --- Step 6: Stats ---
    # run_wilcoxon_posthoc(df_paired, models, alpha=alpha, save_dir=save_dir, suffix=suffix)
    # friedman_p, p_raw, p_holm = run_wilcoxon_posthoc(df = pd.data
    #     df_paired, models, alpha=alpha, save_dir=save_dir, suffix=suffix)
    friedman_p, p_raw, p_holm = run_wilcoxon_posthoc(
        df_paired,
        models,
        alpha=alpha,
        save_dir=save_dir,
        metric_name=metric.upper(),  # <-- FIXED
        suffix=suffix
    )

    # -----------------------------
    # Step 7: Compact Letter Display (CLD)
    # -----------------------------
    cld_map = generate_cld_groups(p_holm.loc[models, models], models, alpha=alpha)
    cld_map = relabel_cld_by_performance(cld_map, series_map)

    print("\nFinal CLD groups (performance-ranked):")
    for m, g in cld_map.items():
        print(f"  {m}: {g}")

    # --- Save CLD summary table ---
    cld_table = make_cld_table(series_map, cld_map, metric)
    cld_path = os.path.join(save_dir, f"CLD_summary_{metric}{suffix}.csv")
    cld_table.to_csv(cld_path, index=False)
    print(f" Saved CLD summary table: {cld_path}")

    # --- Generate ALL plots with correct CLD ---
    plot_box_grouped(series_map, models, cld_map,
                     metric_name=metric.upper(), save_dir=save_dir, suffix=suffix)

    plot_violin_grouped(series_map, models, cld_map,
                        metric_name=metric.upper(), save_dir=save_dir, suffix=suffix)

    # CLD letters + raw order (optional)
    plot_cld_violin(series_map, models, cld_map,
                    metric_name=metric.upper(), save_dir=save_dir, suffix=suffix)

    print(f" Saved CLD summary table: {cld_path}")
    # --- CLD Annotated Plot ---
    plot_cld_violin(series_map, models, cld_map,
                    metric_name=metric.upper(),
                    save_dir=save_dir,
                    suffix=suffix)

    return summary_df.loc[models]

if __name__ == "__main__":
    args = {
        "repl_glob": "/blue/pinaki.sarder/harishwarreddy.k/Validation_of_Hist_FMs_new/Probing/Patch/Tubule/binary_bootstrap_replicates_LR_*.csv",
        "save_dir": "/blue/pinaki.sarder/harishwarreddy.k/Validation_of_Hist_FMs_new/Probing/Stats/Tubule_LR"
    }
    run_replicate_pipeline(args)
