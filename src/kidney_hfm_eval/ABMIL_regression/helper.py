import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

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

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
    try:
        pr, _ = pearsonr(y_true, y_pred)
    except Exception:
        pr = np.nan
    try:
        sr, _ = spearmanr(y_true, y_pred)
    except Exception:
        sr = np.nan
    # ---- MAPE ----
    # avoid division by zero
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return {
        'mse': mse, 'rmse': rmse, 'mae': mae,
        'r2': r2, 'pearson_r': pr, 'spearman_r': sr,
        'mape': mape
    }

def cast_to_model_dtype(x, model):
    """
    Move tensor `x` to the same device and dtype as `model`'s parameters.
    """
    # grab the first parameter to infer device & dtype
    p = next(model.parameters())
    return x.to(device=p.device, dtype=p.dtype)
