import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import argparse

def load_embeddings_from_folder_pt(folder_path: str, ext: str = ".pt") -> Dict[str, np.ndarray]:
    embeddings: Dict[str, np.ndarray] = {}
    for fname in os.listdir(folder_path):
        if not fname.endswith(ext):
            continue
        key = Path(fname).stem
        full_path = os.path.join(folder_path, fname)
        tensor = torch.load(full_path)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"File '{fname}' did not contain a torch.Tensor.")
        arr = tensor.squeeze().cpu().numpy().astype(np.float32)
        embeddings[key] = arr
    return embeddings


def normalize_vectors(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def compute_topk_accuracy(
    orig_embeddings: Dict[str, np.ndarray],
    aug_embeddings: Dict[str, np.ndarray],
    topk_list: List[int] = [1, 3, 5]
) -> Dict[int, float]:
    orig_keys = sorted(orig_embeddings.keys())
    aug_keys  = sorted(aug_embeddings.keys())
    if set(orig_keys) != set(aug_keys):
        missing_in_aug = set(orig_keys) - set(aug_keys)
        missing_in_orig = set(aug_keys) - set(orig_keys)
        raise ValueError(
            f"Filename mismatch:\n"
            f"  In original but not in aug: {sorted(missing_in_aug)}\n"
            f"  In aug but not in original: {sorted(missing_in_orig)}"
        )

    n = len(orig_keys)
    orig_matrix = np.stack([orig_embeddings[k] for k in orig_keys], axis=0)
    aug_matrix  = np.stack([aug_embeddings[k]  for k in orig_keys], axis=0)

    orig_normed = normalize_vectors(orig_matrix)
    aug_normed  = normalize_vectors(aug_matrix)

    sim_matrix = orig_normed.dot(aug_normed.T)
    sorted_inds = np.argsort(-sim_matrix, axis=1)

    results = {}
    for k in topk_list:
        hits = 0
        for i in range(n):
            topk_inds = sorted_inds[i, :k]
            if i in topk_inds:
                hits += 1
        results[k] = hits / n
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute top-k accuracies for embeddings under augmentations.")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Path to base folder (e.g., Copy_detection). Each subfolder is a model (UNI, Virchow, ...).")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="Path to save output CSV file with top-k results.")
    args = parser.parse_args()

    BASE_DIR = args.base_dir

    model_names = sorted([name for name in os.listdir(BASE_DIR)
                          if os.path.isdir(os.path.join(BASE_DIR, name))])
    if len(model_names) == 0:
        raise RuntimeError(f"No subfolders found in {BASE_DIR}.")

    records = []
    for model_name in tqdm(model_names, desc="Models", unit="model"):
        print(f"{model_name}, model_name")
        model_path  = os.path.join(BASE_DIR, model_name)
        orig_folder = os.path.join(model_path, "images")
        aug_base    = os.path.join(model_path, "aug_images")

        if not os.path.isdir(orig_folder) or not os.path.isdir(aug_base):
            print(f"Skipping {model_name}: missing 'images' or 'aug_images' folder.")
            continue

        try:
            orig_embeddings = load_embeddings_from_folder_pt(orig_folder)
        except Exception as e:
            print(f"Skipping model '{model_name}': {e}")
            continue
        if len(orig_embeddings) == 0:
            continue

        # Loop over each augmentation type (geo, noise, deform, color)
        aug_types = sorted([d for d in os.listdir(aug_base) if os.path.isdir(os.path.join(aug_base, d))])

        for aug_type in tqdm(aug_types, desc=f"Augmentations ({model_name})", leave=False, unit="aug"):
            aug_folder = os.path.join(aug_base, aug_type)

            try:
                aug_embeddings = load_embeddings_from_folder_pt(aug_folder)
                topk_acc = compute_topk_accuracy(orig_embeddings, aug_embeddings, topk_list=[1,3,5])
            except Exception as e:
                print(f"  [Warning] Error on {model_name}/{aug_type}: {e}")
                continue

            records.append({
                "model": model_name,
                "augmentation": aug_type,   # now store geo/noise/deform/color
                "top1_accuracy": topk_acc[1],
                "top3_accuracy": topk_acc[3],
                "top5_accuracy": topk_acc[5]
            })

    if len(records) == 0:
        raise RuntimeError("No valid records generated. Check folders and .pt files.")

    df = pd.DataFrame.from_records(records).sort_values(by=["model"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f" Saved results â†’ {args.out_csv}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
