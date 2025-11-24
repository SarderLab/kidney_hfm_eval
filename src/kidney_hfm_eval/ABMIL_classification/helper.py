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

# def calculate_metrics(y_true, y_pred, y_prob=None):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
#     auroc = auprc = None
#     if y_prob is not None and len(np.unique(y_true)) > 1:
#         try:
#             auroc = roc_auc_score(y_true, y_prob)
#             auprc = average_precision_score(y_true, y_prob)
#         except ValueError:
#             auroc = auprc = np.nan
#     return {
#         'accuracy': accuracy_score(y_true, y_pred),
#         'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
#         'precision': precision_score(y_true, y_pred, zero_division=0),
#         'recall': recall_score(y_true, y_pred, zero_division=0),
#         'specificity': tn/(tn+fp) if (tn+fp)>0 else 0.0,
#         'f1_score': f1_score(y_true, y_pred, zero_division=0),
#         'mcc': matthews_corrcoef(y_true, y_pred),
#         'auroc': auroc,
#         'auprc': auprc
#     }
def calculate_metrics(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    auroc = auprc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auroc = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
        except ValueError:
            auroc = auprc = np.nan
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'specificity': tn/(tn+fp) if (tn+fp)>0 else 0.0,
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'auroc': auroc,
        'auprc': auprc
    }

def cast_to_model_dtype(x, model):
    """
    Move tensor `x` to the same device and dtype as `model`'s parameters.
    """
    # grab the first parameter to infer device & dtype
    p = next(model.parameters())
    return x.to(device=p.device, dtype=p.dtype)
