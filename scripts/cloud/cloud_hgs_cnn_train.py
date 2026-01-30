"""
HGS-Optimized 1D-CNN (CPU-only)

Flow:
- Ask: Load prepared data (NPZ)?
  - Yes -> load train/test NPZ (X_train,y_train,X_test,y_test) and run HGS HPO + final train
  - No  -> call BBFS filter which internally calls data_prep_interactive,
           then uses returned selected-feature arrays to run HGS HPO + final train

Outputs:
- Prints per-agent (candidate) scores each HGS iteration
- Prints best config and final test metrics

Kumar - PhD IDS Cloud Tier
"""

from __future__ import annotations

import os
import sys
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np

# ---------------------------------------------------------
# Ensure project root is on sys.path (so `scripts.*` imports work)
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import BBFS filter callable
from scripts.cloud.cloud_bbfs_filter import run_cloud_filter

# ---------------------------------------------------------
# TensorFlow CPU (silence logs)
# ---------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D,
    Flatten, Dense, Dropout, Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import (
    StratifiedShuffleSplit,
    train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------------------------------------------------------
# Prompts
# ---------------------------------------------------------
def prompt_str(msg: str, default: Optional[str] = None) -> str:
    if default is None:
        return input(msg).strip()
    x = input(f"{msg} [default: {default}] ").strip()
    return x if x else default

def prompt_yes_no(msg: str, default_yes: bool = True) -> bool:
    default = "Y/n" if default_yes else "y/N"
    x = input(f"{msg} ({default}): ").strip().lower()
    if not x:
        return default_yes
    return x in ("y", "yes", "1", "true")

def prompt_int(msg: str, default: int) -> int:
    x = input(f"{msg} [default: {default}] ").strip()
    return int(x) if x else default

def prompt_float(msg: str, default: float) -> float:
    x = input(f"{msg} [default: {default}] ").strip()
    return float(x) if x else default


# ---------------------------------------------------------
# Data helpers
# ---------------------------------------------------------
def load_prepared_npz(train_npz: str, test_npz: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(train_npz, allow_pickle=True) as d:
        X_train = d["X_train"].astype(np.float32)
        y_train = d["y_train"].astype(np.int32)
    with np.load(test_npz, allow_pickle=True) as d:
        X_test = d["X_test"].astype(np.float32)
        y_test = d["y_test"].astype(np.int32)
    return X_train, y_train, X_test, y_test

def drop_constant_cols(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    std = X_train.std(axis=0)
    keep = std > 0
    return X_train[:, keep], X_test[:, keep], keep

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_cnn_shape(X: np.ndarray) -> np.ndarray:
    # (N,F) -> (N,F,1)
    if X.ndim == 2:
        return X[..., np.newaxis]
    if X.ndim == 3:
        return X
    raise ValueError(f"Unexpected X shape {X.shape}. Need (N,F) or (N,F,1).")


# ---------------------------------------------------------
# MAC estimation (approx) for 1D CNN + Dense
# ---------------------------------------------------------
def estimate_macs_1dcnn(model: tf.keras.Model) -> int:
    """
    Rough MAC count:
    - Conv1D: out_steps * out_ch * (k * in_ch)
    - Dense : in_dim * out_dim
    Ignores BN/Activation/Dropout.
    """
    macs = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            # output shape: (None, steps, out_ch)
            out_shape = layer.output_shape
            if isinstance(out_shape, list):
                out_shape = out_shape[0]
            steps = int(out_shape[1])
            out_ch = int(out_shape[2])

            k = int(layer.kernel_size[0])
            in_ch = int(layer.input_shape[-1])

            macs += steps * out_ch * (k * in_ch)

        elif isinstance(layer, tf.keras.layers.Dense):
            in_dim = int(layer.input_shape[-1])
            out_dim = int(layer.output_shape[-1])
            macs += in_dim * out_dim

    return int(macs)


# ---------------------------------------------------------
# Hyperparameter space + decoding
# ---------------------------------------------------------
SPACE = {
    "filters1": [64, 128, 256],
    "filters2": [64, 128, 256],
    "filters3": [128, 256, 384],
    "k1": [3, 5, 7],
    "k2": [3, 5, 7],
    "k3": [3, 5, 7],
    "pool": [2, 2, 2, 3],
    "dense1": [128, 256, 384],
    "dense2": [64, 128, 256],
    "drop1": (0.20, 0.60),     # continuous
    "drop2": (0.20, 0.60),     # continuous
    "l2":    (1e-5, 1e-3),     # log-uniform
    "lr":    (1e-4, 5e-3),     # log-uniform
    "batch": [256, 512, 1024],
}
SPACE_KEYS = list(SPACE.keys())
DIM = len(SPACE_KEYS)

def _pick_discrete(u: float, choices: List[Any]) -> Any:
    idx = min(int(u * len(choices)), len(choices)-1)
    return choices[idx]

def _log_map(u: float, lo: float, hi: float) -> float:
    log_lo, log_hi = math.log(lo), math.log(hi)
    return math.exp(log_lo + u * (log_hi - log_lo))

def _lin_map(u: float, lo: float, hi: float) -> float:
    return lo + u * (hi - lo)

def decode_vec(vec01: np.ndarray) -> Dict[str, Any]:
    assert len(vec01) == DIM
    d: Dict[str, Any] = {}
    for i, k in enumerate(SPACE_KEYS):
        spec = SPACE[k]
        u = float(np.clip(vec01[i], 0.0, 1.0))
        if isinstance(spec, tuple):
            lo, hi = spec
            if k in ("l2", "lr"):
                d[k] = _log_map(u, lo, hi)
            else:
                d[k] = _lin_map(u, lo, hi)
        elif isinstance(spec, list):
            d[k] = _pick_discrete(u, spec)
        else:
            raise ValueError("Bad SPACE spec")
    d["drop1"] = float(np.clip(round(d["drop1"], 2), 0.0, 0.9))
    d["drop2"] = float(np.clip(round(d["drop2"], 2), 0.0, 0.9))
    return d


# ---------------------------------------------------------
# Model builder
# ---------------------------------------------------------
def build_model(cfg: Dict[str, Any], n_feats: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(n_feats, 1)),

        Conv1D(filters=cfg["filters1"], kernel_size=cfg["k1"], padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=cfg["pool"], padding='same'),

        Conv1D(filters=cfg["filters2"], kernel_size=cfg["k2"], padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=cfg["pool"], padding='same'),

        Conv1D(filters=cfg["filters3"], kernel_size=cfg["k3"], padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=cfg["pool"], padding='same'),

        Flatten(),

        Dense(cfg["dense1"], kernel_regularizer=l2(cfg["l2"])),
        BatchNormalization(),
        Activation('relu'),
        Dropout(cfg["drop1"]),

        Dense(cfg["dense2"], kernel_regularizer=l2(cfg["l2"])),
        BatchNormalization(),
        Activation('relu'),
        Dropout(cfg["drop2"]),

        Dense(1, activation='sigmoid')
    ])
    opt = Adam(learning_rate=cfg["lr"])
    model.compile(optimizer=opt, loss='binary_crossentropy')
    return model


# ---------------------------------------------------------
# Evaluation config
# ---------------------------------------------------------
@dataclass
class EvalCfg:
    warmup_epochs: int = 10
    patience: int = 2
    verbose_fit: int = 0
    val_size: float = 0.15
    seed: int = 42
    sample_frac: float = 0.20
    min_per_class: int = 50
    fresh_subset_each_eval: bool = False


def stratified_subsample(
    X: np.ndarray, y: np.ndarray,
    frac: float,
    min_per_class: int = 50,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    frac = float(np.clip(frac, 0.01, 1.0))
    n = len(y)
    n_target = max(int(round(frac * n)), 2 * min_per_class)
    n_target = min(n_target, n)

    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_target,
        random_state=seed
    )
    idx_sub, _ = next(sss.split(X, y))
    return X[idx_sub], y[idx_sub]


def evaluate_candidate(
    vec01: np.ndarray,
    X_tr: np.ndarray, y_tr: np.ndarray,
    eval_cfg: EvalCfg
) -> Tuple[float, Dict[str, Any]]:

    set_seed(eval_cfg.seed)
    cfg = decode_vec(vec01)

    # Subsample (stable or fresh)
    sub_seed = int(time.time()) % 1_000_000 if eval_cfg.fresh_subset_each_eval else eval_cfg.seed
    X_sub_all, y_sub_all = stratified_subsample(
        X_tr, y_tr, frac=eval_cfg.sample_frac,
        min_per_class=eval_cfg.min_per_class, seed=sub_seed
    )

    # split subset into train/val
    X_subtr, X_val, y_subtr, y_val = train_test_split(
        X_sub_all, y_sub_all,
        test_size=eval_cfg.val_size,
        random_state=eval_cfg.seed,
        stratify=y_sub_all
    )

    model = build_model(cfg, n_feats=X_subtr.shape[1])
    bs = cfg["batch"]

    es = EarlyStopping(
        monitor='val_loss',
        patience=eval_cfg.patience,
        restore_best_weights=True,
        verbose=0
    )

    model.fit(
        X_subtr, y_subtr,
        epochs=eval_cfg.warmup_epochs,
        batch_size=bs,
        validation_data=(X_val, y_val),
        verbose=eval_cfg.verbose_fit,
        callbacks=[es]
    )

    y_prob = model.predict(X_val, batch_size=bs, verbose=0).ravel()
    y_hat = (y_prob >= 0.5).astype(np.int32)
    f1 = f1_score(y_val, y_hat, zero_division=0)

    params = model.count_params()
    macs = estimate_macs_1dcnn(model)

    info = {"cfg": cfg, "f1": float(f1), "params": int(params), "macs": int(macs)}
    return float(f1), info


# ---------------------------------------------------------
# HGS
# ---------------------------------------------------------
@dataclass
class HGSConfig:
    pop_size: int = 24
    iters: int = 20
    alpha: float = 0.6          # weight on F1 vs compactness
    w_params: float = 0.5
    w_macs: float = 0.5
    lower: float = 0.0
    upper: float = 1.0
    seed: int = 42
    elite_frac: float = 0.10
    stagnate_iters: int = 2


def hgs_optimize(
    X_train_raw: np.ndarray, y_train: np.ndarray,
    hgs_cfg: HGSConfig, eval_cfg: EvalCfg
) -> Dict[str, Any]:

    rng = np.random.default_rng(hgs_cfg.seed)
    N, D = hgs_cfg.pop_size, DIM
    P = rng.uniform(hgs_cfg.lower, hgs_cfg.upper, size=(N, D))

    global_params: List[int] = []
    global_macs: List[int] = []

    Infos: List[Dict[str, Any]] = []
    for i in range(N):
        _, info = evaluate_candidate(P[i], X_train_raw, y_train, eval_cfg)
        Infos.append(info)
        global_params.append(info["params"])
        global_macs.append(info["macs"])

    def _global_norm(values: np.ndarray, ref_list: List[int]) -> np.ndarray:
        arr = np.asarray(ref_list, dtype=np.float64)
        vmin, vmax = np.min(arr), np.max(arr)
        if vmax == vmin:
            return np.zeros_like(values, dtype=np.float64)
        return (values - vmin) / (vmax - vmin)

    def _fitnesses(infos: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        params = np.array([it["params"] for it in infos], dtype=np.float64)
        macs = np.array([it["macs"] for it in infos], dtype=np.float64)
        f1s = np.array([it["f1"] for it in infos], dtype=np.float64)

        n_params = _global_norm(params, global_params)
        n_macs = _global_norm(macs, global_macs)
        LW = hgs_cfg.w_params * n_params + hgs_cfg.w_macs * n_macs

        eps = 1e-4
        fit = hgs_cfg.alpha * f1s + (1.0 - hgs_cfg.alpha) * (1.0 - LW) - eps * LW
        return fit, LW

    fit, LW = _fitnesses(Infos)
    best_idx = int(np.argmax(fit))
    gbest = P[best_idx].copy()
    gbest_info = Infos[best_idx]
    gbest_fit = float(fit[best_idx])

    print("\n[HGS] === Iter 00 (initial population) ===")
    print("agent |   F1_val  |    fitness |   params (M) |   MACs (M)")
    for i, info in enumerate(Infos):
        print(f"{i:5d} | {info['f1']:.6f} | {fit[i]:10.6f} | "
              f"{info['params']/1e6:11.3f} | {info['macs']/1e6:9.3f}")
    print(f"[HGS] init best -> fit={gbest_fit:.6f} | F1={gbest_info['f1']:.6f} | "
          f"Params={gbest_info['params']:,} | MACs={gbest_info['macs']:,}")

    no_improve = 0
    n_elite = max(1, int(np.ceil(hgs_cfg.elite_frac * N)))

    for t in range(1, hgs_cfg.iters + 1):
        order = np.argsort(-fit)
        elites_idx = order[:n_elite]
        elites = P[elites_idx].copy()

        a = 2.0 * (1 - t / hgs_cfg.iters)  # decreasing
        b = rng.uniform(-1, 1, size=(N, D))
        c = rng.uniform(0, 1, size=(N, D))

        center = np.mean(elites, axis=0)

        newP = np.empty_like(P)
        for i in range(N):
            e1 = elites[rng.integers(0, n_elite)]
            e2 = elites[rng.integers(0, n_elite)]

            term1 = a * b[i] * (e1 - np.abs(P[i]))
            term2 = (1 - a) * c[i] * (e2 - center)
            cand = P[i] + term1 + term2

            if rng.random() < 0.25:
                cand = cand + (gbest - cand) * rng.uniform(0.2, 0.6, size=D)

            newP[i] = np.clip(cand, hgs_cfg.lower, hgs_cfg.upper)

        newInfos: List[Dict[str, Any]] = []
        for i in range(N):
            _, info = evaluate_candidate(newP[i], X_train_raw, y_train, eval_cfg)
            newInfos.append(info)
            global_params.append(info["params"])
            global_macs.append(info["macs"])

        newFit, _ = _fitnesses(newInfos)

        replace_mask = newFit > fit
        P[replace_mask] = newP[replace_mask]
        for idx in np.where(replace_mask)[0]:
            Infos[idx] = newInfos[idx]
        fit[replace_mask] = newFit[replace_mask]

        # Elitism keep elites
        worst_now = np.argsort(fit)[:n_elite]
        P[worst_now] = elites
        for k, idx in enumerate(worst_now):
            el_idx = elites_idx[min(k, n_elite - 1)]
            Infos[idx] = Infos[el_idx]
            fit[idx] = fit[el_idx]

        print(f"\n[HGS] === Iter {t:02d} (candidate population) ===")
        print("agent |   F1_val  |    fitness |   params (M) |   MACs (M)")
        for i, info in enumerate(newInfos):
            print(f"{i:5d} | {info['f1']:.6f} | {newFit[i]:10.6f} | "
                  f"{info['params']/1e6:11.3f} | {info['macs']/1e6:9.3f}")

        bi = int(np.argmax(fit))
        if fit[bi] > gbest_fit + 1e-9:
            gbest_fit = float(fit[bi])
            gbest = P[bi].copy()
            gbest_info = Infos[bi]
            no_improve = 0
        else:
            no_improve += 1

        print(f"[HGS] iter {t:02d} -> best_fit={gbest_fit:.6f} | "
              f"best_F1={gbest_info['f1']:.6f} | "
              f"Params={gbest_info['params']:,} | MACs={gbest_info['macs']:,}")

        # Diversify if stagnant
        if no_improve >= hgs_cfg.stagnate_iters:
            no_improve = 0
            order_now = np.argsort(fit)
            worst_k = max(1, N // 4)
            worst_idx = order_now[:worst_k]
            for idx in worst_idx:
                if rng.random() < 0.5:
                    noise = rng.normal(0, 0.15, size=D)
                    cand = gbest + noise
                else:
                    cand = rng.uniform(hgs_cfg.lower, hgs_cfg.upper, size=D)
                P[idx] = np.clip(cand, hgs_cfg.lower, hgs_cfg.upper)
                _, info_i = evaluate_candidate(P[idx], X_train_raw, y_train, eval_cfg)
                Infos[idx] = info_i
                global_params.append(info_i["params"])
                global_macs.append(info_i["macs"])
            fit, _ = _fitnesses(Infos)
            bi = int(np.argmax(fit))
            if fit[bi] > gbest_fit:
                gbest_fit = float(fit[bi])
                gbest = P[bi].copy()
                gbest_info = Infos[bi]
            print(f"[HGS] diversify -> best_fit={gbest_fit:.6f} | best_F1={gbest_info['f1']:.6f}")

    return {"best_vec": gbest, "best_info": gbest_info, "best_fit": gbest_fit}


# ---------------------------------------------------------
# Final train on full training set + evaluate on test
# ---------------------------------------------------------
def final_train(
    best_cfg: Dict[str, Any],
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    epochs: int = 30,
    seed: int = 42
) -> Tuple[tf.keras.Model, Dict[str, Any]]:

    set_seed(seed)

    model = build_model(best_cfg, n_feats=X_tr.shape[1])

    initial_lr = best_cfg["lr"]
    final_lr = max(1e-5, initial_lr / 100.0)
    decay_rate = (final_lr / initial_lr) ** (1.0 / max(1, epochs - 1))

    def lr_fn(epoch, lr):
        return float(initial_lr * (decay_rate ** epoch))

    lrs = LearningRateScheduler(lr_fn, verbose=0)
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

    model.fit(
        X_tr, y_tr,
        epochs=epochs,
        batch_size=best_cfg["batch"],
        validation_split=0.1,
        callbacks=[lrs, es],
        verbose=1
    )

    y_prob = model.predict(X_te, batch_size=best_cfg["batch"], verbose=0).ravel()
    y_hat = (y_prob >= 0.5).astype(np.int32)

    metrics = {
        "acc": float(accuracy_score(y_te, y_hat)),
        "prec": float(precision_score(y_te, y_hat, zero_division=0)),
        "rec": float(recall_score(y_te, y_hat, zero_division=0)),
        "f1": float(f1_score(y_te, y_hat, zero_division=0)),
        "conf": confusion_matrix(y_te, y_hat),
        "report": classification_report(y_te, y_hat, zero_division=0),
        "params": int(model.count_params()),
        "macs": int(estimate_macs_1dcnn(model)),
    }

    print("\n[FINAL TEST]")
    print(f"Acc={metrics['acc']:.4f} | Prec={metrics['prec']:.4f} | Rec={metrics['rec']:.4f} | F1={metrics['f1']:.4f}")
    print(f"Params={metrics['params']:,} | MACs={metrics['macs']:,}")
    print("\nConfusion Matrix:")
    print(metrics["conf"])
    print("\nClassification Report:")
    print(metrics["report"])

    return model, metrics


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    print("\n=== CLOUD HGS-OPTIMIZED CNN ===")

    load_data = prompt_yes_no("Load prepared data (NPZ)?", default_yes=True)

    if load_data:
        train_npz = prompt_str("Train NPZ path:", default=os.path.expanduser("~/Jupyter/PhD/PreparedData/train_prepared.npz"))
        test_npz = prompt_str("Test  NPZ path:", default=os.path.expanduser("~/Jupyter/PhD/PreparedData/test_prepared.npz"))
        X_train, y_train, X_test, y_test = load_prepared_npz(os.path.expanduser(train_npz), os.path.expanduser(test_npz))

        # Keep behavior consistent with your pipeline: remove constant cols
        X_train, X_test, _ = drop_constant_cols(X_train, X_test)

    else:
        print("\nLoad:NO → Running BBFS filter (which calls data_prep_interactive internally) ...")
        X_train, y_train, X_test, y_test, selected, selected_names = run_cloud_filter(return_selected_data=True)
        print(f"\nUsing BBFS-selected features for HGS-CNN. Selected #features: {len(selected)}")

    # Convert to CNN shape
    X_train_cnn = ensure_cnn_shape(X_train.astype(np.float32))
    X_test_cnn = ensure_cnn_shape(X_test.astype(np.float32))

    print("\nData shapes for CNN:")
    print("X_train:", X_train_cnn.shape, "y_train:", y_train.shape)
    print("X_test :", X_test_cnn.shape, "y_test :", y_test.shape)

    # ---- HGS settings (interactive) ----
    seed = prompt_int("Seed?", default=42)

    eval_cfg = EvalCfg(
        warmup_epochs=prompt_int("HGS warmup epochs per eval?", default=10),
        patience=prompt_int("EarlyStop patience per eval?", default=2),
        verbose_fit=0,
        val_size=prompt_float("Eval val split?", default=0.15),
        seed=seed,
        sample_frac=prompt_float("Eval sample fraction?", default=0.20),
        min_per_class=prompt_int("Eval min per class?", default=50),
        fresh_subset_each_eval=prompt_yes_no("Fresh subset each evaluation?", default_yes=False)
    )

    hgs_cfg = HGSConfig(
        pop_size=prompt_int("HGS population size?", default=24),
        iters=prompt_int("HGS iterations?", default=20),
        alpha=prompt_float("HGS alpha (weight on F1)?", default=0.6),
        w_params=prompt_float("HGS w_params?", default=0.5),
        w_macs=prompt_float("HGS w_macs?", default=0.5),
        seed=seed,
        elite_frac=prompt_float("Elite fraction?", default=0.10),
        stagnate_iters=prompt_int("Stagnation iters before diversify?", default=2)
    )

    print("\nRunning HGS hyperparameter optimization ...")
    t0 = time.time()
    result = hgs_optimize(X_train_cnn, y_train, hgs_cfg, eval_cfg)
    t1 = time.time()

    best_cfg = result["best_info"]["cfg"]
    print("\n=== Best configuration (decoded) ===")
    for k, v in best_cfg.items():
        print(f"{k:>8}: {v}")
    print(f"HGS best val-F1: {result['best_info']['f1']:.6f}")
    print(f"HGS time (s): {t1 - t0:.2f}")

    final_epochs = prompt_int("Final training epochs?", default=30)
    final_model, test_metrics = final_train(
        best_cfg,
        X_train_cnn, y_train,
        X_test_cnn, y_test,
        epochs=final_epochs,
        seed=seed
    )

    save_model = prompt_yes_no("Save final trained model?", default_yes=False)
    if save_model:
        out_dir = prompt_str("Output directory:", default=os.path.expanduser("~/Jupyter/PhD/CloudModels"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "cloud_hgs_cnn_model.keras")
        final_model.save(out_path)
        print(f"Saved model: {out_path}")


if __name__ == "__main__":
    main()
