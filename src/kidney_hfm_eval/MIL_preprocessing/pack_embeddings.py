import os, sys, glob, json
import numpy as np
import torch, argparse
from pathlib import Path
from tqdm import tqdm
import pickle

def _load_pt_safe(path):
    """Return a 1D torch.Tensor or None if unreadable/invalid."""
    try:
        if os.path.getsize(path) == 0:
            raise EOFError("zero-byte file")
        obj = torch.load(path, map_location="cpu")
    except (EOFError, RuntimeError, ValueError, pickle.UnpicklingError, OSError) as e:  # noqa: F821
        return None, f"{type(e).__name__}: {e}"

    # Accept common shapes/containers
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, (list, tuple)) and len(obj) and isinstance(obj[0], torch.Tensor):
        t = obj[0]
    elif isinstance(obj, dict):
        for k in ("embedding", "feat", "features", "vector"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                t = obj[k]; break
        else:
            return None, "No tensor in dict"
    else:
        return None, f"Unsupported object type: {type(obj)}"

    # Squeeze [1, D] to [D]
    if t.ndim == 2 and t.shape[0] == 1:
        t = t[0]
    if t.ndim != 1:
        return None, f"Expected 1D tensor, got {tuple(t.shape)}"
    return t, None

def pack_patient_dir(patient_dir, out_file):
    """
    Reads all .pt vectors under patient_dir, stacks to [N, D],
    saves as .npy (uncompressed, float32). Returns (N, D, stats).
    """
    pt_files = sorted(glob.glob(os.path.join(patient_dir, "*.pt")))
    if not pt_files:
        return (0, 0, {"total": 0, "loaded": 0, "skipped": 0, "reasons": {}})

    vecs = []
    stats = {"total": len(pt_files), "loaded": 0, "skipped": 0, "reasons": {}}

    for f in pt_files:
        t, err = _load_pt_safe(f)
        if t is None:
            stats["skipped"] += 1
            key = (err or "Unknown")
            stats["reasons"][key] = stats["reasons"].get(key, 0) + 1
            continue
        vecs.append(t.detach().to(torch.float32).numpy())
        stats["loaded"] += 1

    if not vecs:
        return (0, 0, stats)

    # Ensure consistent D across all
    D = vecs[0].shape[0]
    bad = [i for i, v in enumerate(vecs) if v.shape[0] != D]
    if bad:
        stats["reasons"]["Inconsistent D"] = len(bad)
        vecs = [v for v in vecs if v.shape[0] == D]

    if not vecs:
        return (0, 0, stats)

    arr = np.stack(vecs, axis=0).astype(np.float32, copy=False)
    np.save(out_file, arr)
    return arr.shape + (stats,)

# def main(root_dir, out_root):
#     Path(out_root).mkdir(parents=True, exist_ok=True)
#     patients = [p for p in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, p))]
#     index = {}
#     summary = {"patients": 0, "with_data": 0, "without_data": 0, "skipped_files": 0}

#     for p in tqdm(sorted(patients), desc="Packing patients"):
#         p_dir = os.path.join(root_dir, p)
#         out_file = os.path.join(out_root, f"{p}.npy")

#         # Fast path if valid npy already exists
#         if os.path.exists(out_file):
#             try:
#                 arr = np.load(out_file, mmap_mode="r")
#                 index[p] = {"file": out_file, "shape": arr.shape}
#                 summary["patients"] += 1
#                 summary["with_data"] += int(arr.shape[0] > 0)
#                 summary["without_data"] += int(arr.shape[0] == 0)
#                 continue
#             except Exception:
#                 print(f"[INFO] {out_file} is invalid, rebuilding…")
#                 try: os.remove(out_file)
#                 except OSError: pass

#         try:
#             N, D, stats = pack_patient_dir(p_dir, out_file)
#             summary["patients"] += 1
#             summary["skipped_files"] += stats.get("skipped", 0)
#             if N > 0:
#                 index[p] = {"file": out_file, "shape": (N, D), "stats": stats}
#                 summary["with_data"] += 1
#             else:
#                 summary["without_data"] += 1
#                 # ensure we don't leave a bogus npy
#                 if os.path.exists(out_file):
#                     try: os.remove(out_file)
#                     except OSError: pass
#                 if stats["skipped"] > 0:
#                     print(f"[WARN] {p}: all files skipped; reasons: {stats['reasons']}")
#         except Exception as e:
#             print(f"[WARN] skipping {p}: {e}")

#     with open(os.path.join(out_root, "_manifest.json"), "w") as f:
#         json.dump(index, f, indent=2)
#     with open(os.path.join(out_root, "_summary.json"), "w") as f:
#         json.dump(summary, f, indent=2)
#     print(f"Packed {len(index)} patients into {out_root}")
#     print(f"Summary: {summary}")

# # if __name__ == "__main__":
# #     if len(sys.argv) < 3:
# #         print("Usage: pack_patients_to_npy.py <root_dir> <out_root>")
# #         sys.exit(1)
# #     main(sys.argv[1], sys.argv[2])
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Pack per-patient .pt embeddings into .npy files with manifest and summary."
#     )
#     parser.add_argument(
#         "--root_dir", type=str, required=True,
#         help="Path to input root directory containing patient subfolders with .pt embeddings."
#     )
#     parser.add_argument(
#         "--out_root", type=str, required=True,
#         help="Output directory where packed .npy files and JSON manifests will be saved."
#     )
#     parser.add_argument(
#         "--manifest_name", type=str, default="_manifest.json",
#         help="Filename for the manifest JSON file (default: _manifest.json)."
#     )
#     parser.add_argument(
#         "--summary_name", type=str, default="_summary.json",
#         help="Filename for the summary JSON file (default: _summary.json)."
#     )

#     args = parser.parse_args()

#     Path(args.out_root).mkdir(parents=True, exist_ok=True)
#     main(args.root_dir, args.out_root)

#     # Rename output files if custom names provided
#     manifest_path = os.path.join(args.out_root, args.manifest_name)
#     summary_path  = os.path.join(args.out_root, args.summary_name)
#     if os.path.exists(os.path.join(args.out_root, "_manifest.json")):
#         os.rename(os.path.join(args.out_root, "_manifest.json"), manifest_path)
#     if os.path.exists(os.path.join(args.out_root, "_summary.json")):
#         os.rename(os.path.join(args.out_root, "_summary.json"), summary_path)
def pack_all_patients(root_dir, out_root, manifest_name="_manifest.json", summary_name="_summary.json"):
    """
    High-level wrapper: iterates over patient dirs, packs each into .npy,
    and writes manifest/summary JSON files.
    """
    Path(out_root).mkdir(parents=True, exist_ok=True)
    patients = [p for p in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, p))]
    index, summary = {}, {"patients": 0, "with_data": 0, "without_data": 0, "skipped_files": 0}

    for p in tqdm(sorted(patients), desc="Packing patients"):
        p_dir = os.path.join(root_dir, p)
        out_file = os.path.join(out_root, f"{p}.npy")

        # Fast path: skip if valid npy exists
        if os.path.exists(out_file):
            try:
                arr = np.load(out_file, mmap_mode="r")
                index[p] = {"file": out_file, "shape": arr.shape}
                summary["patients"] += 1
                summary["with_data"] += int(arr.shape[0] > 0)
                summary["without_data"] += int(arr.shape[0] == 0)
                continue
            except Exception:
                print(f"[INFO] {out_file} is invalid, rebuilding…")
                try:
                    os.remove(out_file)
                except OSError:
                    pass

        try:
            N, D, stats = pack_patient_dir(p_dir, out_file)
            summary["patients"] += 1
            summary["skipped_files"] += stats.get("skipped", 0)
            if N > 0:
                index[p] = {"file": out_file, "shape": (N, D), "stats": stats}
                summary["with_data"] += 1
            else:
                summary["without_data"] += 1
                if os.path.exists(out_file):
                    try:
                        os.remove(out_file)
                    except OSError:
                        pass
                if stats["skipped"] > 0:
                    print(f"[WARN] {p}: all files skipped; reasons: {stats['reasons']}")
        except Exception as e:
            print(f"[WARN] skipping {p}: {e}")

    # Save manifests
    with open(os.path.join(out_root, manifest_name), "w") as f:
        json.dump(index, f, indent=2)
    with open(os.path.join(out_root, summary_name), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Packed {len(index)} patients into {out_root}")
    print(f"Summary: {summary}")
    return index, summary


def main():
    parser = argparse.ArgumentParser(description="Pack per-patient .pt embeddings into .npy files.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to input directory containing patient subfolders.")
    parser.add_argument("--out_root", type=str, required=True, help="Output directory for packed .npy files and JSONs.")
    parser.add_argument("--manifest_name", type=str, default="_manifest.json", help="Name for manifest JSON.")
    parser.add_argument("--summary_name", type=str, default="_summary.json", help="Name for summary JSON.")
    args = parser.parse_args()

    pack_all_patients(args.root_dir, args.out_root, args.manifest_name, args.summary_name)


if __name__ == "__main__":
    main()
