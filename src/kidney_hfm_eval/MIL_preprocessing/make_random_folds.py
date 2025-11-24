import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path

def generate_random_folds(csv_path, seeds, n_splits, out_path):
    """
    Generate random k-fold splits for multiple seeds and save to JSON.
    Each seed produces a list of {trainval, test} index dictionaries.
    """
    df = pd.read_csv(csv_path)
    df['ID'] = df['ID'].str.strip()
    N = len(df)

    all_splits = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N)                 # random order of indices 0..N-1
        folds = np.array_split(perm, n_splits)    # near-equal sized folds

        seed_folds = []
#         for i in range(n_splits):
#             test_idx = folds[i]
#             trainval_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i]) if n_splits > 1 else np.array([], dtype=int)

#             seed_folds.append({
#                 "trainval": trainval_idx.tolist(),
#                 "test": test_idx.tolist()
#             })
        for i in range(n_splits):
            test_idx = folds[i]
            trainval_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i]) if n_splits > 1 else np.array([], dtype=int)

            # ✅ Map index positions to actual IDs
            seed_folds.append({
                "trainval": df.loc[trainval_idx, "ID"].tolist(),
                "test": df.loc[test_idx, "ID"].tolist()
            })
        all_splits[str(seed)] = seed_folds

    # ensure output directory exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_splits, f, indent=2)

    print(f"✅ Saved random {n_splits}-fold indices for seeds {seeds} → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random k-fold splits for multiple seeds.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--out_path", type=str, required=True, help="Path to save JSON output (e.g., outer_folds.json).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="List of random seeds.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for cross-validation.")

    args = parser.parse_args()
    generate_random_folds(args.csv_path, args.seeds, args.n_splits, args.out_path)