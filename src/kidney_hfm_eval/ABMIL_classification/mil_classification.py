from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')
import argparse
import os, json
import numpy as np
import pandas as pd
import torch, csv
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torch.nn.functional as F
from .dataloader import MemmapPatientBagsDataset
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
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from .model import Attention
from torch.utils.data import WeightedRandomSampler, DataLoader
from sklearn.model_selection import StratifiedKFold
# device = torch.device("cuda" if args.cuda else "cpu")
import random
from .helper import infer_vector_size_from_dataset, get_params_groups, calculate_metrics, cast_to_model_dtype
from itertools import product


def set_seed(seed: int):
    # 1) Python
    random.seed(seed)
    # 2) NumPy
    np.random.seed(seed)
    # 3) PyTorch CPU + CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 4) cuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


def train_fold(train_loader, model, optimizer, criterion, epoch, cuda):
    model.train()
    total_loss = 0.0
    all_true, all_pred, all_prob = [], [], []
    n_iters = len(train_loader)

    for i, (data, label) in enumerate(train_loader):
        # constant LR and WD (no scheduler needed)
        for j, pg in enumerate(optimizer.param_groups):
            pg['lr'] = optimizer.defaults['lr']
            if j == 0:
                pg['weight_decay'] = optimizer.defaults['weight_decay']

        bag_label = label.long()
        if cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data  = cast_to_model_dtype(data, model)
        label = label.long()  # labels stay ints
        optimizer.zero_grad()
        # x = data.view(data.size(1), data.size(-1))
        logits, A = model(data)   # data should be [B,K,D] or [K,D] # now logits is [1,2]
        # loss = criterion(logits, bag_label)  # bag_label is 0 or 1
        target = bag_label.view(-1)
        # print(logits, target, 'logits, target')
        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        probs = F.softmax(logits, dim=1)          # [B, C]
        pred  = torch.argmax(probs, dim=1)        # [B]
        all_true.extend(bag_label.view(-1).tolist())
        all_pred.extend(pred.view(-1).tolist())
        all_prob.extend(probs.detach().cpu().numpy())   # appends rows, each is length C
        
    all_prob = np.vstack(all_prob)   # shape [N, C]
    avg_loss = total_loss / n_iters
    metrics = calculate_metrics(
        np.array(all_true),
        np.array(all_pred),
        np.array(all_prob)
    )
    return avg_loss, metrics



def test_fold(test_loader, model, criterion, cuda):
    model.eval()
    total_loss = 0.0
    all_true, all_prob = [], []

    with torch.no_grad():
        for data, label in test_loader:
            bag_label = label.long()
            if cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data  = cast_to_model_dtype(data, model)
            # x = data.view(data.size(1), data.size(-1))
            logits, A = model(data)
            loss = criterion(logits, bag_label.view(-1))
            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)              # [B, C]
            pred  = torch.argmax(probs, dim=1)            # [B]
            all_true.extend(bag_label.view(-1).tolist())
            all_prob.extend(probs.detach().cpu().numpy())
    all_prob = np.vstack(all_prob)  # [N, C]
    y_pred = np.argmax(all_prob, axis=1)
    avg_loss = total_loss / len(test_loader)
    # y_pred = (np.array(all_prob) > 0.5).astype(int)
    metrics = calculate_metrics(np.array(all_true), y_pred, np.array(all_prob))
    return avg_loss, metrics, np.array(all_true), np.array(all_prob)


def train_until_stopping(train_loader, val_loader, model, optimizer, criterion,
                         cuda, patience, num_epochs,
                         seed=None, outer_fold_idx=None, inner_fold_idx=None,
                         log_file=None, hp_dict=None):
    best_val_mcc = -np.inf
    best_state   = None
    no_imp       = 0

    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch}—best_val_mcc so far = {best_val_mcc:.4f}", flush=True)

        # 1) train one epoch and print its MCC
        train_loss, train_metrics = train_fold(train_loader, model, optimizer, criterion, epoch, cuda)
        print(f"        → train MCC this epoch = {train_metrics['mcc']:.4f}", flush=True)

        # 2) evaluate on val set and print its MCC
        val_loss, val_metrics, _, _ = test_fold(val_loader, model, criterion, cuda)
        mcc = val_metrics['mcc']
        print(f"        → val   MCC this epoch = {mcc:.4f}", flush=True)

        # 3) early stopping logic
        if mcc > best_val_mcc:
            best_val_mcc = mcc
            best_state   = {k: v.cpu() for k, v in model.state_dict().items()}
            no_imp       = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
        if log_file is not None:
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                  # seed or '',
                  # outer_fold_idx or '',
                  # inner_fold_idx or '',
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
                  f"{train_loss:.4f}",
                  f"{train_metrics['accuracy']:.4f}",
                  f"{train_metrics['precision']:.4f}",
                  f"{train_metrics['recall']:.4f}",
                  f"{train_metrics['f1_score']:.4f}",
                  f"{train_metrics['auroc']:.4f}",
                  f"{train_metrics['mcc']:.4f}",
                  f"{val_loss:.4f}",
                  f"{val_metrics['accuracy']:.4f}",
                  f"{val_metrics['precision']:.4f}",
                  f"{val_metrics['recall']:.4f}",
                  f"{val_metrics['f1_score']:.4f}",
                  f"{val_metrics['auroc']:.4f}",
                  f"{val_metrics['mcc']:.4f}",
                ])

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_mcc

def log_outer_epoch_cls(log_file, seed, outer_fold_idx, epoch, train_loss, tm, hp_dict=None):
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
            f"{tm.get('accuracy', np.nan):.6f}",
            f"{tm.get('precision', np.nan):.6f}",
            f"{tm.get('recall', np.nan):.6f}",
            f"{tm.get('f1_score', np.nan):.6f}",
            f"{tm.get('auroc', np.nan):.6f}",
            f"{tm.get('mcc', np.nan):.6f}",
            "", "", "", "", "", "", ""  # val_* placeholders
        ])

def run_mil_pipeline(args):
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    args.cuda = torch.cuda.is_available()
    fm_names = args.models


    for fm in fm_names: 
        print(fm, "_______________________________fm__________________________________")
        # --- FM-specific root directory ---
        fm_root = os.path.join(args.root_dir, fm)
        if not os.path.isdir(fm_root):
            raise FileNotFoundError(f"Missing folder for foundation model '{fm}': {fm_root}")
        set_seed(args.seed)
        device = torch.device("cuda" if args.cuda else "cpu")
        criterion = nn.CrossEntropyLoss()
        cuda = args.cuda
        df = pd.read_csv(args.csv_path)
        df['ID'] = df['ID'].str.strip()
        patients = df['ID'].tolist()
        # labels   = df['class'].astype(int).tolist()
        labels_raw = df["class"].astype(int)
        C = labels_raw.nunique()   # infer number of classes from CSV
        if C == 2:
            uniq = sorted(df["class"].astype(int).unique().tolist())
            if uniq != [0, 1]:
                mapping = {uniq[0]: 0, uniq[1]: 1}
                df["class"] = df["class"].map(mapping).astype(int)
                print("Mapped binary labels to {0,1}:", mapping)

        labels = df['class'].astype(int).tolist()
        assert set(labels).issubset(set(range(C))), "Labels must be in [0, C-1]"

        default_params = {
            "dropout": 0.6,
            "weight_decay": 0.01,
            "lr": 5e-4,
            "epochs": args.epochs,
            "M": 512,
            "L": 256,
        }

        full_param_grid = {
            "lr": [5e-5, 1e-4, 2e-4],
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
        all_y_prob_list = []
        all_y_ids_list  = []
        fm_log_dir = os.path.join(args.log_file_path, fm)
        os.makedirs(fm_log_dir, exist_ok=True)
        log_file = os.path.join(fm_log_dir, f'training_log_{fm}.csv')
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'seed','outer_fold','inner_fold','epoch', 'dropout', 'lr', 'weight_decay', 'epochs','M','L',
                'train_loss','train_accuracy','train_precision','train_recall','train_f1','train_auroc','train_mcc',
                'val_loss','val_accuracy','val_precision','val_recall','val_f1','val_auroc','val_mcc'
            ])
        with open(args.outer_fold, 'r') as f:
            all_fold_indices = json.load(f)
        num_seeds = len(all_fold_indices)
        k_folds   = len(next(iter(all_fold_indices.values())))  # assumes all seeds have same #folds
        fold_counts = [len(v) for v in all_fold_indices.values()]
        assert len(set(fold_counts)) == 1, "Mismatch in number of folds across seeds"

        for seed_str, fold_indices in all_fold_indices.items():
            seed = int(seed_str)
            set_seed(seed)                       # reproducibility
            g = torch.Generator()
            g.manual_seed(seed)

            def _seed_worker(worker_id):
                np.random.seed(seed + worker_id)
                random.seed(seed + worker_id)

            for fold_idx, indices in enumerate(fold_indices, 1):
                trainval_ids = indices["trainval"]  # list of strings like "K1300468"
                test_ids     = indices["test"]

                tv_p = [str(x).strip() for x in trainval_ids]
                te_p = [str(x).strip() for x in test_ids]

                # map ID -> class using your CSV
                id2y = dict(zip(df["ID"].astype(str).str.strip(), df["class"].astype(int)))

                tv_l = [id2y[p] for p in tv_p]
                te_l = [id2y[p] for p in te_p]
                print(tv_p[:3], te_p[:3], "example train/test IDs")
                print(tv_l[:3], te_l[:3], "example train/test labels")

                print(tv_l, te_l, "tvl", "tel")
                print(f"[MIL seed={seed} outer_fold={fold_idx}]")
                print("   Train+Val IDs:", tv_p)
                print("   Test IDs:     ", te_p)

                # ----- inner hyperparameter tuning -----
                # INNER CV to pick best hp
                inner_cv = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=seed)
                print(f"[Outer seed={seed} fold={fold_idx}] starting inner CV", flush=True)
                print(f"Tuning {len(list(product(*param_grid.values())))} combinations")
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
                            print(f" → Inner fold {inner_fold_idx}/{args.inner_folds}")

                            tr_ds = MemmapPatientBagsDataset.from_lists(
                                fm_root, [tv_p[i] for i in train_idx], [tv_l[i] for i in train_idx],
                                map_location="cpu", max_cache=8)
                            vl_ds = MemmapPatientBagsDataset.from_lists(
                                fm_root, [tv_p[i] for i in val_idx], [tv_l[i] for i in val_idx],
                                map_location="cpu", max_cache=4)

                            tr_ld = DataLoader(tr_ds, batch_size=1, shuffle=True,
                                               num_workers=4, pin_memory=args.cuda, persistent_workers=True,
                                               worker_init_fn=_seed_worker, generator=g)
                            vl_ld = DataLoader(vl_ds, batch_size=1,
                                               num_workers=4, pin_memory=args.cuda, persistent_workers=True,
                                               worker_init_fn=_seed_worker, generator=g)

                            vector_size = infer_vector_size_from_dataset(tr_ds)
                            model = Attention(vector_size, M=hp_dict["M"], L=hp_dict["L"], dropout=hp_dict["dropout"], n_classes=C).to(device)

                            optimizer = optim.AdamW(
                                get_params_groups(model),
                                lr=hp_dict["lr"],
                                weight_decay=hp_dict["weight_decay"],
                                betas=(0.95, 0.99),
                                eps=1e-4)

                            val_mcc = train_until_stopping(
                                tr_ld, vl_ld, model, optimizer,
                                nn.CrossEntropyLoss(), 
                                cuda, args.patience,
                                num_epochs=hp_dict["epochs"],
                                seed=seed, outer_fold_idx=fold_idx,
                                inner_fold_idx=inner_fold_idx,
                                log_file=log_file, hp_dict=hp_dict)

                            val_scores.append(val_mcc)

                        mean_mcc = np.mean(val_scores)
                        print(f"→ mean inner MCC = {mean_mcc:.4f} for {hp_dict}", flush=True)

                        if mean_mcc > best_score:
                            best_score = mean_mcc
                            best_params = hp_dict
                if best_params is None:
                    print(" No valid inner‐CV combination found; using defaults.")
                    best_params = default_params.copy()

                print(f"\n Best inner-CV params: {best_params}, mean val-MCC = {best_score:.4f}\n", flush=True)
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

                print(f"→ best params = {best_params} (val‐MCC={best_score:.4f})", flush=True)
                print(f"--- Outer retrain: seed={seed}, outer_fold={fold_idx}, using tuned params ---", flush=True)
                train_ds = MemmapPatientBagsDataset.from_lists(
                    fm_root, tv_p, tv_l, map_location="cpu", max_cache=16
                )
                te_ds = MemmapPatientBagsDataset.from_lists(
                    fm_root, te_p, te_l, map_location="cpu", max_cache=8
                )
                tr_ld = DataLoader(
                    train_ds, batch_size=1, shuffle=True, drop_last=False,
                    num_workers=4, pin_memory=args.cuda, persistent_workers=True, worker_init_fn=_seed_worker, generator=g
                )
                te_ld = DataLoader(
                    te_ds, batch_size=1,
                    num_workers=4, pin_memory=args.cuda, persistent_workers=True,worker_init_fn=_seed_worker, generator=g
                )
                vector_size_outer = infer_vector_size_from_dataset(train_ds)
                model = Attention(vector_size_outer, M=best_params["M"], L=best_params["L"],
                                  dropout=best_params["dropout"], n_classes=C).to(device)
                opt   = optim.AdamW(get_params_groups(model), lr=best_params["lr"], weight_decay=best_params["weight_decay"],betas=(0.99, 0.9999), eps=1e-4)
                # lr_s = cosine_scheduler(best_params["lr"], args.lr_end, best_params["epochs"], len(tr_ld), args.warmup_epochs)
                # wd_s = cosine_scheduler(best_params["weight_decay"], args.weight_decay_end, best_params["epochs"], len(tr_ld), args.warmup_epochs)

                print( "Outer‐fold final training")
                for epoch in range(1, best_params["epochs"]+1):
                    _avg_loss, _trm = train_fold(tr_ld, model, opt, criterion, epoch, cuda)
                    # log_outer_epoch_cls(log_file, seed, fold_idx, epoch, _avg_loss, _trm)
                    log_outer_epoch_cls(log_file, seed, fold_idx, epoch, _avg_loss, _trm, hp_dict=best_params)

                    if epoch % 5 == 0 or epoch == 1 or epoch == best_params["epochs"]:
                        print(f"[outer seed={seed} fold={fold_idx}] epoch {epoch:03d} "
                              f"train MCC={_trm['mcc']:.4f}", flush=True)

                _, train_metrics, _, _ = test_fold(tr_ld, model, criterion, cuda)
                print(f"[Outer seed={seed} fold={fold_idx}]   train MCC = {train_metrics['mcc']:.4f}", flush=True)
                _, test_metrics, y_true, y_prob = test_fold(te_ld, model, criterion, cuda)
                print(f"[Outer seed={seed} fold={fold_idx}]   test  MCC = {test_metrics['mcc']:.4f}", flush=True)

                print(
                    f"[Outer seed={seed}, fold={fold_idx}] "
                    f"train_mcc={train_metrics['mcc']:.4f} "
                    f"test_mcc={test_metrics['mcc']:.4f}",
                    flush=True
                    )
                all_y_true_list.append(y_true)
                all_y_prob_list.append(y_prob)
                all_y_ids_list.append(np.array(te_p))
                results.append({
                  'seed': seed, 'fold': fold_idx,
                  **best_params,
                  'train_accuracy':  train_metrics['accuracy'],
                  'train_precision': train_metrics['precision'],
                  'train_recall':    train_metrics['recall'],
                  'train_f1':        train_metrics['f1_score'],
                  'train_auroc':     train_metrics['auroc'],
                  'train_mcc':       train_metrics['mcc'],
                  'test_accuracy':   test_metrics['accuracy'],
                  'test_precision':  test_metrics['precision'],
                  'test_recall':     test_metrics['recall'],
                  'test_f1':         test_metrics['f1_score'],
                  'test_auroc':      test_metrics['auroc'],
                  'test_mcc':        test_metrics['mcc'],
                })
        os.makedirs(fm_log_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(os.path.join(fm_log_dir, f'cv_results_MIL_{fm}.csv'), index=False)
        print(f" Done, results in cv_results_MIL_{fm}.csv", flush=True)

        all_y_true = np.concatenate(all_y_true_list)
        all_y_prob = np.concatenate(all_y_prob_list)
        all_y_ids = np.concatenate(all_y_ids_list)   # shape == (N_images * num_seeds,)
        unique_ids = np.unique(all_y_ids)             
        n = len(all_y_true)
        print(n, 'n', k_folds* num_seeds, "folds, sseds")
        sample_size = n // (k_folds* num_seeds)   # or choose your own
        print(sample_size, "sample_size")
        records = list(zip(all_y_ids, all_y_true, all_y_prob))
        # Group by patient ID
        grouped = defaultdict(list)
        for pid, y_t, y_p in records:
            grouped[pid].append((y_t, y_p))
        # ensure reproducible ordering
        unique_ids = np.array(sorted(grouped.keys()))
        print(unique_ids, "unique_ids")
        # 2) run bootstrap
        rng = np.random.default_rng(args.seed)
        all_boot_metrics = []

        for b in range(args.bootstrap):
            if b % 100 == 0:
                print(f"bootstrap iteration {b}/{args.bootstrap}", flush=True)
            chosen_ids = rng.choice(unique_ids, size=sample_size, replace=True)
            if b < 10:
                print(f"Bootstrap {b:2d}: sampled IDs → {chosen_ids.tolist()}", flush=True)
            sel_true, sel_prob = [], []
            for pid in chosen_ids:
                vals = grouped[pid]
                idx = rng.integers(len(vals))
                yt_i, yp_i = vals[idx]
                sel_true.append(yt_i)
                sel_prob.append(yp_i)

            y_tb = np.array(sel_true)
            y_pb = np.array(sel_prob)
            # y_predb = (y_pb > 0.5).astype(int)
            y_predb = np.argmax(y_pb, axis=1)   # if y_pb is [N, C]
            is_binary = (C == 2)
            avg = None if is_binary else "macro"
            # is_binary = (C == 2)

            if is_binary:
                labels_present = np.unique(y_tb)

                if len(labels_present) < 2:
                    prec = precision_score(y_tb, y_predb, average="macro", zero_division=0)
                    rec  = recall_score(y_tb, y_predb, average="macro", zero_division=0)
                    f1   = f1_score(y_tb, y_predb, average="macro", zero_division=0)
                    spec = np.nan
                    auroc = np.nan
                    auprc = np.nan
                else:
                    # IMPORTANT: this assumes your binary labels are exactly {0,1}
                    tn, fp, fn, tp = confusion_matrix(y_tb, y_predb, labels=[0, 1]).ravel()

                    prec = precision_score(y_tb, y_predb, average="binary", pos_label=1, zero_division=0)
                    rec  = recall_score(y_tb, y_predb, average="binary", pos_label=1, zero_division=0)
                    f1   = f1_score(y_tb, y_predb, average="binary", pos_label=1, zero_division=0)

                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    auroc = roc_auc_score(y_tb, y_pb[:, 1])
                    auprc = average_precision_score(y_tb, y_pb[:, 1])

            else:
                prec = precision_score(y_tb, y_predb, average="macro", zero_division=0)
                rec  = recall_score(y_tb, y_predb, average="macro", zero_division=0)
                f1   = f1_score(y_tb, y_predb, average="macro", zero_division=0)
                spec = np.nan
                tmp = calculate_metrics(y_tb, y_predb, y_pb)
                auroc = tmp["auroc"]
                auprc = tmp["auprc"]

            scores = y_predb
            acc  = accuracy_score(y_tb, y_predb)
            bacc = balanced_accuracy_score(y_tb, y_predb)
            mcc  = matthews_corrcoef(y_tb, y_predb)

            all_boot_metrics.append({
                "bootstrap_idx": b,
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "precision": prec,
                "recall": rec,
                "specificity": spec,
                "f1": f1,
                "mcc": mcc,
                "auroc": auroc,
                "auprc": auprc,
            })

        # 3) save the full bootstrap‐replicate table
        boot_df = pd.DataFrame(all_boot_metrics)
        boot_df.to_csv(os.path.join(fm_log_dir,f'bootstrap_replicates_MIL_{fm}.csv'), index=False)
        print(f" Saved bootstrap replicate metrics → bootstrap_replicates_MIL_{fm}.csv", flush=True)

        # 4) compute mean + 95% CI for each metric
        metrics = ['accuracy','balanced_accuracy','precision','recall',
                   'specificity','f1','mcc','auroc','auprc']

        summary = {}
        for m in metrics:
            vals = boot_df[m]
            summary[m] = {
                'mean':   vals.mean(),
                'ci_lo':  vals.quantile(0.025),
                'ci_hi':  vals.quantile(0.975)
            }
        summary_df = pd.DataFrame.from_dict(summary, orient='index')
        summary_df = summary_df.T
        summary_df.to_csv(os.path.join(fm_log_dir, f'bootstrap_CI_MIL_{fm}.csv'))
        print(f" Saved bootstrap summary (mean + 95% CI) → bootstrap_CI_MIL_{fm}.csv", flush=True)

        
def main():
    parser = argparse.ArgumentParser(description='MIL nested-bootstrap + permutation test')
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--lr_end', type=float, default=1e-6)
    # parser.add_argument('--weight_decay_end', type=float, default=0.4)
    # parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--patience', type=int, default=20)
    # parser.add_argument('--tol', type=float, default=0.0001)
    parser.add_argument('--inner_folds', type=int, default=4)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--outer_fold', required=True)
    parser.add_argument('--log_file_path', required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument('--models', nargs="+", default=[
        "UNI", "UNI2-h", "Virchow", "Virchow2",
        "SP22M", "SP85M", "H-optimus-0", "H-optimus-1", "Hibou-B", "Hibou-L", "Prov-Gigapath"
    ])
    # parser.add_argument("--tune_params", type=str, default="dropout,weight_decay,lr,epochs")
    parser.add_argument("--tune_params", type=str, default="lr,M,L")

    args = parser.parse_args()
    run_mil_pipeline(args)
    
if __name__ == "__main__":
    main()