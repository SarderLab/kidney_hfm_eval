import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

def infer_vector_size_from_dataset(ds):
    first_patient = ds.patients[0]
    first_path = ds.index[first_patient][0]  # points to <patient>.npy
    if first_path.endswith(".npy"):
        arr = np.load(first_path, mmap_mode="r")  # [N, D]
        return int(arr.shape[-1])
    else:
        t = torch.load(first_path, map_location="cpu")
        return int(t.shape[-1] if t.ndim > 1 else t.numel())

def get_params_groups(model):
    reg_params, no_reg = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith('.bias') or p.ndim == 1:
            no_reg.append(p)
        else:
            reg_params.append(p)
    return [
        {'params': reg_params},
        {'params': no_reg, 'weight_decay': 0.0}
    ]

def cosine_scheduler(base, final, epochs, niter_per_epoch, warmup_epochs=0, start_warmup_value=0.0):
    warmup_iters = warmup_epochs * niter_per_epoch
    total = epochs * niter_per_epoch
    main_iters = max(total - warmup_iters, 1)
    warmup = np.linspace(start_warmup_value, base, warmup_iters, endpoint=False) if warmup_iters > 0 else np.array([])
    iters = np.arange(main_iters)
    schedule = final + 0.5 * (base - final) * (1 + np.cos(np.pi * iters / main_iters))
    return np.concatenate((warmup, schedule))[:total]

def calculate_metrics(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    def _scalar(x):
        # convert numpy scalars / 0-d arrays to Python float
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return float(x.ravel()[0])
            # if it's truly a vector (per-class), take macro mean
            return float(np.nanmean(x))
        try:
            return float(x)
        except Exception:
            return np.nan

    # number of classes inferred from labels seen
    # classes = np.unique(y_true)
    # C = len(classes)
    # number of classes: prefer model output dimension if available
    if y_prob is not None and np.asarray(y_prob).ndim == 2:
        C = y_prob.shape[1]   # number of columns = number of classes
    else:
        C = len(np.unique(y_true))

    # ---- specificity: only well-defined as a single scalar for binary ----
    specificity = np.nan
    if C == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ---- averaging for multiclass ----
    avg = None if C == 2 else "macro"   # or "weighted" if you prefer

    auroc = np.nan
    auprc = np.nan

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            y_prob = np.asarray(y_prob)

            if C == 2:
                # accept either [N] (positive-class prob) OR [N,2]
                if y_prob.ndim == 2:
                    y_prob_bin = y_prob[:, 1]
                else:
                    y_prob_bin = y_prob
                auroc = roc_auc_score(y_true, y_prob_bin)
                auprc = average_precision_score(y_true, y_prob_bin)

            else:
                # y_prob must be [N, C]
                # AUROC macro OVR
                auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

                # AUPRC macro across classes (OVR)
                # Y = label_binarize(y_true, classes=np.arange(C))  # assumes labels 0..C-1
                Cprob = y_prob.shape[1]  # number of columns output by the model (0..C-1)
                classes = np.arange(Cprob)  # expects 0..C-1
                Y = label_binarize(y_true, classes=classes)

                # if labels aren't 0..C-1, change classes=... accordingly
                # per_class_ap = []
                # for c in range(C):
                #     # skip if class absent in this sample set (can happen in small folds)
                #     if Y[:, c].sum() == 0:
                #         continue
                #     per_class_ap.append(average_precision_score(Y[:, c], y_prob[:, c]))
                # auprc = float(np.mean(per_class_ap)) if len(per_class_ap) else np.nan
                per_class_ap = []
                for c in range(Cprob):
                    if Y[:, c].sum() == 0:
                        continue
                    per_class_ap.append(average_precision_score(Y[:, c], y_prob[:, c]))
                auprc = float(np.mean(per_class_ap)) if len(per_class_ap) else np.nan

        except ValueError:
            auroc = np.nan
            auprc = np.nan

    return {
        "accuracy": _scalar(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _scalar(balanced_accuracy_score(y_true, y_pred)),
        "precision": _scalar(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": _scalar(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "specificity": _scalar(specificity),
        "f1_score": _scalar(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "mcc": _scalar(matthews_corrcoef(y_true, y_pred)),
        "auroc": _scalar(auroc),
        "auprc": _scalar(auprc),
    }



def cast_to_model_dtype(x, model):
    """
    Move tensor `x` to the same device and dtype as `model`'s parameters.
    """
    # grab the first parameter to infer device & dtype
    p = next(model.parameters())
    return x.to(device=p.device, dtype=p.dtype)
