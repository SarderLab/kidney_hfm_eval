import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MemmapPatientBagsDataset(Dataset):
    """
    Drop-in alternative to LazyPatientBagsDataset, but each patient is a single
    <patient>.npy file (saved earlier). Uses np.load(..., mmap_mode='r') so data
    is not fully loaded into RAM.
    """
    def __init__(self, packed_root, split_patients, labels_map, on_disk_dtype="float32"):
        self.packed_root = packed_root
        self.patients = list(split_patients)
        # build labels aligned to patients
        self.labels_map = {str(k).strip(): float(v) for k, v in labels_map.items()}
        self.labels = [torch.tensor(self.labels_map[p], dtype=torch.float32) for p in self.patients]
        self.on_disk_dtype = on_disk_dtype  # informational only

        # Expose an `index` dict to be compatible with your old code
        # Map each patient -> [path_to_npy] (list so your code indexing still works)
        self.index = {}
        missing = []
        for p in self.patients:
            f = os.path.join(self.packed_root, f"{p}.npy")
            if os.path.exists(f):
                self.index[p] = [f]
            else:
                missing.append(p)
        if missing:
            raise FileNotFoundError(
                f"Missing packed npy for {len(missing)} patients. "
                f"First few: {missing[:5]}"
            )
        # prune any patients that somehow ended up without a file
        self.patients = [p for p in self.patients if p in self.index]
        self.labels   = [torch.tensor(self.labels_map[p], dtype=torch.float32) for p in self.patients]

    @classmethod
    def from_lists(cls, packed_root, patient_list, label_list, on_disk_dtype="float32", **_):
        labels_map = {str(p).strip(): float(l) for p, l in zip(patient_list, label_list)}
        return cls(packed_root, split_patients=list(labels_map.keys()), labels_map=labels_map,
                   on_disk_dtype=on_disk_dtype)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pid = self.patients[idx]
        y   = self.labels[idx]  # torch.float32 (regression)

        # Memory-map without reading fully into RAM
        path = self.index[pid][0]
        mm = np.load(path, mmap_mode="r")      # shape [N, D], dtype float32/16
        x  = torch.from_numpy(mm)              # zero-copy CPU tensor view

        return x, y
