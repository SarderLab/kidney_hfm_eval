import warnings
warnings.filterwarnings('ignore')
import argparse, random
import os, json
import numpy as np
import pandas as pd
import torch, csv
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torch.nn.functional as F
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
from .dataloader import MemmapPatientBagsDataset
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from .model import Attention
from torch.utils.data import WeightedRandomSampler, DataLoader
from sklearn.model_selection import KFold
# device = torch.device("cuda" if args.cuda else "cpu")
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from .helper import infer_vector_size_from_dataset, get_params_groups, cosine_scheduler, calculate_metrics, cast_to_model_dtype
from itertools import product

# cuDNN determinism
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_fold(train_loader, model, optimizer, criterion, epoch, cuda):
    model.train()
    total_loss = 0.0
    all_true, all_pred = [], []
    n_iters = len(train_loader)

    for i, (data, label) in enumerate(train_loader):
        # it = (epoch-1)*n_iters + i
        # if i % 40 == 0:
        #     print(f"   epoch {epoch}, batch {i}/{n_iters}", flush=True)

        # update lr & wd
        for j, pg in enumerate(optimizer.param_groups):
            pg['lr'] = optimizer.defaults['lr']
            if j == 0:
                pg['weight_decay'] = optimizer.defaults['weight_decay']

        target = label.float()
        if cuda:
            data, target = data.cuda(), target.cuda()
        data  = cast_to_model_dtype(data, model)
        # label = label.long()  # labels stay ints
        optimizer.zero_grad()
        x = data.view(data.size(1), data.size(-1))
        preds, A = model(x)                 # now logits is [1,2]
        # loss = criterion(logits, bag_label)  # bag_label is 0 or 1
        preds = preds.view(-1)
        # print(logits, target, 'logits, target')
        loss = criterion(preds, target.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        # probs = F.softmax(logits, dim=1)
        # pred = torch.argmax(probs, dim=1)
        all_true.append(target.item())
        all_pred.append(preds.detach().cpu().item())
        # all_prob.append(probs[:,1].item())

    avg_loss = total_loss / n_iters
    metrics = calculate_metrics(all_true, all_pred)
    return avg_loss, metrics


def test_fold(test_loader, model, criterion, cuda):
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []

    with torch.no_grad():
        for data, label in test_loader:
            target = label.float()
            if cuda:
                data, target = data.cuda(), target.cuda()
            data  = cast_to_model_dtype(data, model)
            x = data.view(data.size(1), data.size(-1))
            preds, A = model(x)
            preds = preds.view(-1)
            loss = criterion(preds, target.view(-1))
            total_loss += loss.item()
            all_true.append(target.item())
            all_pred.append(preds.detach().cpu().item())

    avg_loss = total_loss / len(test_loader)
    metrics  = calculate_metrics(all_true, all_pred)
    return avg_loss, metrics, np.array(all_true), np.array(all_pred)

def train_until_stopping(train_loader, val_loader, model, optimizer, criterion, 
                         cuda, patience, num_epochs, seed=None, outer_fold_idx=None, 
                         inner_fold_idx=None, log_file=None, hp_dict=None):
    best_r2 = -np.inf
    best_state   = None
    no_imp       = 0

    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch}—best_val_R2 so far = {best_r2:.6f}", flush=True)  # FIX

        # 1) train one epoch and print its MCC
        train_loss, train_metrics = train_fold(train_loader, model, optimizer, criterion, epoch, cuda)
        print(f"        → train RMSE  this epoch = {train_metrics['rmse']:.4f}", flush=True)

        # 2) evaluate on val set and print its MCC
        val_loss, val_metrics, _, _ = test_fold(val_loader, model, criterion, cuda)
        val_r2 = val_metrics.get('r2', np.nan)
        val_r2_num = -np.inf if (val_r2 is None or np.isnan(val_r2)) else float(val_r2)
        print(f"        → val   RMSE this epoch = {val_metrics['rmse']:.4f} | R² = {val_r2_num:.4f}", flush=True)

        if val_r2_num > best_r2:
            best_r2 = val_r2_num
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
        
        if log_file is not None:
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                  seed if seed is not None else '',
                  outer_fold_idx if outer_fold_idx is not None else '',
                  inner_fold_idx if inner_fold_idx is not None else '',
                  epoch,
                  hp_dict.get("dropout", ""),
                  hp_dict.get("lr", ""),
                  hp_dict.get("weight_decay", ""),
                  hp_dict.get("epochs", ""),
                  hp_dict.get("M", ""),
                  hp_dict.get("L", ""),
                  f"{train_loss:.6f}",
                  f"{train_metrics['mse']:.6f}",
                  f"{train_metrics['rmse']:.6f}",
                  f"{train_metrics['mae']:.6f}",
                  f"{train_metrics['r2']:.6f}",
                  f"{train_metrics['pearson_r']:.6f}",
                  f"{train_metrics['spearman_r']:.6f}",
                  f"{train_metrics['mape']:.6f}",
                  f"{val_loss:.6f}",
                  f"{val_metrics['mse']:.6f}",
                  f"{val_metrics['rmse']:.6f}",
                  f"{val_metrics['mae']:.6f}",
                  f"{val_metrics.get('r2', np.nan):.6f}",
                  f"{val_metrics.get('pearson_r', np.nan):.6f}",
                  f"{val_metrics.get('spearman_r', np.nan):.6f}",
                  f"{val_metrics['mape']:.6f}",
                ])

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_r2

def log_outer_epoch(log_file, seed, outer_fold_idx, epoch, train_loss, tm, hp_dict=None):
    hp_dict = hp_dict or {}
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            seed if seed is not None else '',
            outer_fold_idx if outer_fold_idx is not None else '',
            'outer',
            epoch,
            hp_dict.get("dropout", ""),
            hp_dict.get("lr", ""),
            hp_dict.get("weight_decay", ""),
            hp_dict.get("epochs", ""),
            hp_dict.get("M", ""),       # add this
            hp_dict.get("L", ""),       # add this
            f"{train_loss:.6f}",
            f"{tm['mse']:.6f}",
            f"{tm['rmse']:.6f}",
            f"{tm['mae']:.6f}",
            f"{tm.get('r2', np.nan):.6f}",
            f"{tm.get('pearson_r', np.nan):.6f}",
            f"{tm.get('spearman_r', np.nan):.6f}",
            "", "", "", "", "", "", ""  # val_* placeholders
        ])

def run_mil_pipeline(args):
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    args.cuda = torch.cuda.is_available()
    fm_names = args.models

    for fm in fm_names:
        print(fm, "_______________________________fm__________________________________")
        fm_root = os.path.join(args.root_dir, fm)
        if not os.path.isdir(fm_root):
            raise FileNotFoundError(f"Missing folder for foundation model '{fm}': {fm_root}")
        set_seed(args.seed)
        device = torch.device("cuda" if args.cuda else "cpu")
        criterion = nn.MSELoss()
        cuda = args.cuda
        df = pd.read_csv(args.csv_path)
        df['ID'] = df['ID'].str.strip()
        patients = df['ID'].tolist()
        labels   = df['value'].astype(float).tolist()
        # --- Default values from argparse ---
        # default_params = {
        #     "dropout": 0.5,
        #     "weight_decay": 0.04,
        #     "lr": 5e-4,
        #     "epochs": args.epochs,
        # }
        default_params = {
            "dropout": 0.6,
            "weight_decay": 0.01,
            "lr": 5e-4,
            "epochs": args.epochs,
            "M": 512,
            "L": 256,
        }

        # full_param_grid = {
        #     "dropout": [0.25, 0.5, 0.75],
        #     "weight_decay": [0.01, 0.04, 0.1],
        #     "lr": [1e-3, 5e-4, 1e-4],
        #     "epochs": [30, 40],
        # }
        full_param_grid = {
            "lr": [2e-5, 5e-5, 1e-4],
            "M":  [256, 512, 1024],
            "L":  [32, 128, 256],
        }

        # --- Determine which parameters to tune ---
        tune_keys = [k.strip() for k in args.tune_params.split(",") if k.strip() in full_param_grid]
        # Optional: warn if user provided invalid tune_params
        invalid = [k.strip() for k in args.tune_params.split(",") if k.strip() not in full_param_grid]
        if invalid:
            print(f" Ignored invalid tune params: {invalid}. Valid options are {list(full_param_grid.keys())}")

        # --- Construct grid dynamically ---
        if tune_keys:
            # Only use tuned params from full grid
            param_grid = {k: full_param_grid[k] for k in tune_keys}
        else:
            param_grid = {}  # no tuning at all

        print(f"Tuning these parameters: {list(param_grid.keys())}")        
        
        best_params, best_score = None, -np.inf
        
        results = []
        all_y_true_list = []
        all_y_pred_list = []
        all_y_ids_list  = []
        fm_log_dir = os.path.join(args.log_file_path, fm)
        os.makedirs(fm_log_dir, exist_ok=True)
        log_file = os.path.join(fm_log_dir, f'training_log_{fm}.csv')
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                # 'seed','outer_fold','inner_fold','epoch',
                'seed','outer_fold','inner_fold','epoch', 'dropout', 'lr', 'weight_decay', 'epochs','M','L',
                'train_loss','train_mse','train_rmse','train_mae','train_r2','train_pearson_r','train_spearman_r','train_mape',
                'val_loss','val_mse','val_rmse','val_mae','val_r2','val_pearson_r','val_spearman_r', 'val_mape'
        ])
        with open(args.outer_fold, 'r') as f:
            all_fold_indices = json.load(f)

        for seed_str, fold_indices in all_fold_indices.items():
            seed = int(seed_str)
            set_seed(seed)                       # reproducibility
            g = torch.Generator()
            g.manual_seed(seed)

            def _seed_worker(worker_id):
                np.random.seed(seed + worker_id)
                random.seed(seed + worker_id)

            for fold_idx, indices in enumerate(fold_indices, 1):
                trainval_idx, test_idx = indices['trainval'], indices['test']
                tv_p = [p for p in patients if p in trainval_idx]
                te_p = [p for p in patients if p in test_idx]
                tv_l = df.set_index("ID").loc[tv_p, "value"].astype(float).tolist()
                te_l = df.set_index("ID").loc[te_p, "value"].astype(float).tolist()
                print(tv_l, te_l, "tvl", "tel")                
                print(f"[MIL seed={seed} outer_fold={fold_idx}]")
                print("   Train+Val IDs:", tv_p)
                print("   Test IDs:     ", te_p)

                # ----- inner hyperparameter tuning -----
                # INNER CV to pick best hp
                inner_cv = KFold(n_splits=args.inner_folds, shuffle=True, random_state=seed)
                print(f"[Outer seed={seed} fold={fold_idx}] starting inner CV", flush=True)
                print(f"Tuning {len(list(product(*param_grid.values())))} combinations")
                # best_dropout, best_score = None, -np.inf
                if not param_grid:
                    # no tuning; use defaults directly
                    best_params = default_params.copy()
                    best_score = -np.inf
                else:
                    best_params, best_score = None, -np.inf
                    for hp in product(*param_grid.values()):
                        # unpack tuple of current combination into a dict
                        hp_dict = dict(zip(param_grid.keys(), hp))
                        for k, v in default_params.items():
                            if k not in hp_dict:
                                hp_dict[k] = v

                        print(f"\n=== Inner CV: {hp_dict} ===", flush=True)

                        val_scores = []
                        for inner_fold_idx, (train_idx, val_idx) in enumerate(inner_cv.split(tv_p, tv_l), start=1):
                            print( "Inner‐CV training")
                            tr_ds = MemmapPatientBagsDataset.from_lists(fm_root,
                                                                   [tv_p[i] for i in train_idx],
                                                                   [tv_l[i] for i in train_idx],
                                                                    map_location="cpu",
                                                                    max_cache=8,
                                                                    dtype=None)
                            vl_ds = MemmapPatientBagsDataset.from_lists(fm_root,
                                                                    [tv_p[i] for i in val_idx],
                                                                    [tv_l[i] for i in val_idx],
                                                                    map_location="cpu",
                                                                    max_cache=4)

                            tr_ld = DataLoader(
                                tr_ds,
                                batch_size=args.batch_size,     # 1
                                shuffle=True,
                                drop_last=False,
                                num_workers=4,                  # tune: 4–8 typical
                                pin_memory=args.cuda,           # True when using GPU
                                persistent_workers=True,worker_init_fn=_seed_worker,
                                generator=g# keeps workers alive across epochs
                            )
                            vl_ld = DataLoader(
                                vl_ds,
                                batch_size=args.batch_size,
                                num_workers=4,
                                pin_memory=args.cuda,
                                persistent_workers=True,worker_init_fn=_seed_worker,
                                generator=g
                            )

                            vector_size = infer_vector_size_from_dataset(tr_ds)

#                             model = Attention(vector_size, M=512, L=256, attention_branches=1, dropout=hp_dict["dropout"]).to(device)

#                             optimizer = optim.AdamW(get_params_groups(model),
#                                                     lr=hp_dict["lr"],
#                                                     weight_decay=hp_dict["weight_decay"])

#                             lr_sched = cosine_scheduler(hp_dict["lr"], args.lr_end,
#                                                         hp_dict["epochs"], len(tr_ld),
#                                                         args.warmup_epochs)
#                             wd_sched = cosine_scheduler(hp_dict["weight_decay"],
#                                                         args.weight_decay_end,
#                                                         hp_dict["epochs"], len(tr_ld),
#                                                         args.warmup_epochs)

#                             val_score  = train_until_stopping(
#                                 tr_ld, vl_ld, model, optimizer,
#                                 nn.MSELoss(), lr_sched, wd_sched,
#                                 cuda, args.patience, args.tol, 
#                                 num_epochs=hp_dict["epochs"],
#                                 seed=seed, outer_fold_idx=fold_idx, 
#                                 inner_fold_idx=inner_fold_idx, 
#                                 log_file=log_file, hp_dict=hp_dict)
                            model = Attention(vector_size, M=hp_dict["M"], L=hp_dict["L"], dropout=hp_dict["dropout"]).to(device)

                            optimizer = optim.AdamW(
                                get_params_groups(model),
                                lr=hp_dict["lr"],
                                weight_decay=hp_dict["weight_decay"],
                                betas=(0.99, 0.9999),
                                eps=1e-4)

                            val_score = train_until_stopping(
                                tr_ld, vl_ld, model, optimizer,
                                nn.MSELoss(), 
                                cuda, args.patience,
                                num_epochs=hp_dict["epochs"],
                                seed=seed, outer_fold_idx=fold_idx,
                                inner_fold_idx=inner_fold_idx,
                                log_file=log_file, hp_dict=hp_dict)
                            
                            val_scores.append(val_score)
                        mean_score = float(np.mean(val_scores))
                        if mean_score > best_score:
                            best_score, best_params  = mean_score, hp_dict.copy()

                if best_params is None:
                    print(" No valid inner‐CV combination found; using defaults.")
                    best_params = default_params.copy()
                print(f"\n Best inner-CV params: {best_params}, mean val-R² = {best_score:.4f}\n", flush=True)
                # ------------------------------------------------------------
                # LOG BEST HYPERPARAMETERS FOR THIS OUTER FOLD
                # ------------------------------------------------------------
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        seed,                      # seed
                        fold_idx,                 # outer fold index
                        'best_hp',                # mark inner_fold column as "best_hp"
                        '',                       # epoch (empty)
                        best_params.get("dropout", ""),
                        best_params.get("lr", ""),
                        best_params.get("weight_decay", ""),
                        best_params.get("epochs", ""),
                        best_params.get("M", ""),
                        best_params.get("L", ""),
                        '', '', '', '', '', '', '',   # train_* metrics left blank
                        '', '', '', '', '', '', ''    # val_* metrics left blank
                    ])

                print(f"→ best params = {best_params} (val‐R²={best_score:.4f})", flush=True)
                print(f"--- Outer retrain: seed={seed}, outer_fold={fold_idx}, using tuned params ---", flush=True)

                # retrain on full train+val
                train_ds = MemmapPatientBagsDataset.from_lists(
                    fm_root, tv_p, tv_l, map_location="cpu", max_cache=16
                )
                te_ds = MemmapPatientBagsDataset.from_lists(
                    fm_root, te_p, te_l, map_location="cpu", max_cache=8
                )
                tr_ld = DataLoader(
                    train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
                    num_workers=4, pin_memory=args.cuda, persistent_workers=True, worker_init_fn=_seed_worker,generator=g
                )
                te_ld = DataLoader(
                    te_ds, batch_size=args.batch_size,
                    num_workers=4, pin_memory=args.cuda, persistent_workers=True, worker_init_fn=_seed_worker,generator=g
                )
                vector_size_outer = infer_vector_size_from_dataset(train_ds)
                model = Attention(vector_size_outer, M=best_params["M"], L=best_params["L"],
                                  dropout=best_params["dropout"]).to(device)
                opt   = optim.AdamW(get_params_groups(model), lr=best_params["lr"], weight_decay=best_params["weight_decay"],betas=(0.99, 0.9999), eps=1e-4)
                # lr_s = cosine_scheduler(best_params["lr"], args.lr_end, best_params["epochs"], len(tr_ld), args.warmup_epochs)
                # wd_s = cosine_scheduler(best_params["weight_decay"], args.weight_decay_end, best_params["epochs"], len(tr_ld), args.warmup_epochs)
                print( "Outer‐fold final training")
                for epoch in range(1, best_params["epochs"]+1):
                    _avg_loss, _trm = train_fold(tr_ld, model, opt, criterion, epoch, cuda)
                    log_outer_epoch(log_file, seed, fold_idx, epoch, _avg_loss, _trm)
                    if epoch % 5 == 0 or epoch == 1 or epoch == best_params["epochs"]:
                        print(f"[outer seed={seed} fold={fold_idx}] epoch {epoch:03d} "
                              f"train RMSE={_trm['rmse']:.4f} R2={_trm['r2']:.4f}", flush=True)

                    # _, test_metrics, y_true, y_prob = test_fold(te_ld, model, criterion, cuda)
                _, train_metrics, _, _ = test_fold(tr_ld, model, criterion, cuda)
                print(f"[Outer seed={seed} fold={fold_idx}]   train R2 = {train_metrics['r2']:.4f}", flush=True)
                _, test_metrics, y_true, y_pred = test_fold(te_ld, model, criterion, cuda)
                print(f"[Outer seed={seed} fold={fold_idx}]   test  R2 = {test_metrics['r2']:.4f}", flush=True)

                print(
                    f"[Outer seed={seed} fold={fold_idx}] "
                    f"train_rmse={train_metrics['rmse']:.4f}, train_r2={train_metrics['r2']:.4f}; "
                    f"test_rmse={test_metrics['rmse']:.4f},  test_r2={test_metrics['r2']:.4f}",
                    flush=True
                    )
                all_y_true_list.append(y_true)
                all_y_pred_list.append(y_pred)
                all_y_ids_list.append(np.array(te_p))
                results.append({
                  'seed': seed, 'fold': fold_idx,
                  **best_params,
                  'train_mse':  train_metrics['mse'],
                  'train_rmse': train_metrics['rmse'],
                  'train_mae':  train_metrics['mae'],
                  'train_r2':   train_metrics['r2'],
                  'train_pearson_r': train_metrics['pearson_r'],
                  'train_spearman_r': train_metrics['spearman_r'],
                  'train_mape': train_metrics['mape'],
                  'test_mse':   test_metrics['mse'],
                  'test_rmse':  test_metrics['rmse'],
                  'test_mae':   test_metrics['mae'],
                  'test_r2':    test_metrics['r2'],
                  'test_pearson_r': test_metrics['pearson_r'],
                  'test_spearman_r': test_metrics['spearman_r'],
                })
        fm_log_dir = os.path.join(args.log_file_path, fm)
        os.makedirs(fm_log_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(os.path.join(fm_log_dir, f'cv_results_MIL_{fm}.csv'), index=False)
        print(f" Done, results in cv_results_MIL_{fm}.csv", flush=True)


        all_y_true = np.concatenate(all_y_true_list)
        all_y_pred = np.concatenate(all_y_pred_list)
        all_y_ids = np.concatenate(all_y_ids_list)   # shape == (N_images * num_seeds,)
        unique_ids = np.unique(all_y_ids)             
        n = len(all_y_true)
        print(n, 'n', args.k_folds* args.num_seeds, "folds, seeds")
        sample_size = n // (args.k_folds* args.num_seeds)   # or choose your own
        print(sample_size, "sample_size")
        records = list(zip(all_y_ids, all_y_true, all_y_pred))
        # Group by patient ID
        grouped = defaultdict(list)
        for pid, y_t, y_p in records:
            grouped[pid].append((y_t, y_p))
        unique_ids = np.array(sorted(grouped.keys()))
        print(unique_ids, "unique_ids")
        rng = np.random.default_rng(args.seed)
        all_boot_metrics = []

        for b in range(args.bootstrap):
            if b % 100 == 0:
                print(f"bootstrap iteration {b}/{args.bootstrap}", flush=True)
            chosen_ids = rng.choice(unique_ids, size=sample_size, replace=True)
            if b < 10:
                print(f"Bootstrap {b:2d}: sampled IDs → {chosen_ids.tolist()}", flush=True)
            sel_true, sel_pred = [], []
            for pid in chosen_ids:
                vals = grouped[pid]
                idx = rng.integers(len(vals))
                yt_i, yp_i = vals[idx]
                sel_true.append(yt_i)
                sel_pred.append(yp_i)

            m = calculate_metrics(sel_true, sel_pred)  # returns mse/rmse/mae/r2/pearson/spearman
            m['bootstrap_idx'] = b
            all_boot_metrics.append(m)

        # 3) save the full bootstrap‐replicate table
        boot_df = pd.DataFrame(all_boot_metrics)
        boot_df.to_csv(os.path.join(fm_log_dir,f'bootstrap_replicates_MIL_{fm}.csv'), index=False)
        print(f" Saved bootstrap replicate metrics → bootstrap_replicates_MIL_{fm}.csv", flush=True)

        # 4) compute mean + 95% CI for each metric
        metrics = ['mse','rmse','mae','r2','pearson_r','spearman_r', 'mape']

        summary = {}
        for m in metrics:
            vals = boot_df[m]
            summary[m] = {
                'mean':   vals.mean(),
                'ci_lo':  vals.quantile(0.025),
                'ci_hi':  vals.quantile(0.975)
            }
        summary_df = pd.DataFrame.from_dict(summary, orient='index')
        summary_df.to_csv(os.path.join(fm_log_dir, f'bootstrap_CI_MIL_{fm}.csv'))
        print(f" Saved bootstrap summary (mean + 95% CI) → bootstrap_CI_MIL_{fm}.csv", flush=True)
        
def main():
    parser = argparse.ArgumentParser(description='MIL nested-bootstrap + permutation test')
    parser.add_argument('--root_dir', required=True, default = "", help="Embeddings folders")
    parser.add_argument('--csv_path', required=True, help="CSV with patient_name + label")
    parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--lr', type=float, default=0.0005)
    # parser.add_argument('--lr_end', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.04)
    # parser.add_argument('--weight_decay_end', type=float, default=0.4)
    # parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--patience', type=int, default=20, help="Number of epochs with no improvement after which training will be stopped")
    # parser.add_argument('--tol', type=float, default=0.0001, help="Minimum decrease in loss to qualify as an improvement")
    parser.add_argument('--inner_folds', type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000, help="Number of bootstrap replicates")
    parser.add_argument("--outer_fold", required=True, help="path to json file containing the fold indices")
    parser.add_argument('--log_file_path', required=True)
    parser.add_argument('--models', nargs="+", default=[
        "UNI", "UNI2-h", "Virchow", "Virchow2",
        "SP22M", "SP85M", "H-optimus-0", "H-optimus-1", "Hibou-B", "Hibou-L", "Prov-Gigapath"
    ])
    parser.add_argument("--tune_params", type=str, default="lr,M,L")
    args = parser.parse_args()
    run_mil_pipeline(args)
    
if __name__ == "__main__":
    main()
