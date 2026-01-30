"""
EDGE IDS Pipeline (CPU-only)

Features:
- Load data from NPZ OR run interactive raw data prep
- BBFS wrapper feature selection using sklearn LogisticRegression
- Recall-biased custom F1 objective (beta > 1)
- Iteration-wise and agent-wise feature subsets + scores printed
- Final LR training + confidence-based routing
- Suppresses sklearn LogisticRegression warnings

Author: Kumar (PhD IDS – Edge Tier)
"""

from __future__ import annotations

import os
import time
import random
import warnings
import importlib.util
from typing import List, Dict, Tuple, Optional

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.exceptions import ConvergenceWarning


# =========================================================
# Silence sklearn warnings (important for BBFS loops)
# =========================================================
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*lbfgs.*")
warnings.filterwarnings("ignore", message=".*max_iter.*")
warnings.filterwarnings("ignore", message=".*converge.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*n_jobs.*has no effect.*")


# =========================================================
# Prompt utilities
# =========================================================
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


def print_class_counts(y: np.ndarray, title: str) -> None:
    cls, cnt = np.unique(y, return_counts=True)
    print(f"\n{title}:")
    for c, n in zip(cls, cnt):
        print(f"  Class {c}: {n}")


# =========================================================
# Data loading helpers
# =========================================================
def load_from_npz(train_npz: str, test_npz: str):
    with np.load(train_npz, allow_pickle=True) as d:
        X_train = d["X_train"].astype(np.float32)
        y_train = d["y_train"].astype(np.int32)
        feature_names = d["feature_names"].astype(object).tolist()

    with np.load(test_npz, allow_pickle=True) as d:
        X_test = d["X_test"].astype(np.float32)
        y_test = d["y_test"].astype(np.int32)

    return X_train, y_train, X_test, y_test, feature_names


def run_data_prep_interactive() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    script_path = os.path.abspath(os.path.join(os.getcwd(), "scripts", "data_prep_interactive.py"))
    if not os.path.exists(script_path):
        raise FileNotFoundError("data_prep_interactive.py not found")

    spec = importlib.util.spec_from_file_location("data_prep_interactive", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    X_train, y_train, X_test, y_test, feature_names, _ = mod.main(return_data=True)
    return X_train, y_train, X_test, y_test, feature_names


# =========================================================
# Custom recall-biased F1 (EDGE objective)
# =========================================================
def custom_f1(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 1.2) -> float:
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)


# =========================================================
# Wrapper evaluation with caching
# =========================================================
def evaluate_subset(
    subset: List[int],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model: LogisticRegression,
    cache: Dict[Tuple[int, ...], float],
    beta: float
) -> float:

    key = tuple(sorted(subset))
    if key in cache:
        return cache[key]

    if len(subset) == 0:
        cache[key] = 0.0
        return 0.0

    clf = clone(model)
    clf.fit(X_tr[:, subset], y_tr)
    y_pred = clf.predict(X_val[:, subset])

    score = custom_f1(y_val, y_pred, beta=beta)
    cache[key] = score
    return score


# =========================================================
# BBFS (CPU-friendly, verbose)
# =========================================================
def bbfs_wrapper(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model: LogisticRegression,
    num_iterations: int,
    population_size: int,
    top_fraction: float,
    q: float,
    beta: float,
    seed: int,
    max_print_features: int = 200
) -> List[int]:

    random.seed(seed)
    np.random.seed(seed)

    n_features = X_train.shape[1]
    q_iters = max(1, int(q * num_iterations))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.3,
        stratify=y_train,
        random_state=seed
    )

    population = [
        random.sample(range(n_features), random.randint(1, n_features))
        for _ in range(population_size)
    ]

    cache: Dict[Tuple[int, ...], float] = {}
    performance = [
        evaluate_subset(p, X_tr, y_tr, X_val, y_val, model, cache, beta)
        for p in population
    ]

    best_subset = population[int(np.argmax(performance))]
    best_score = max(performance)
    worst_counter = [0] * population_size

    for it in range(num_iterations):
        sorted_idx = np.argsort(performance)
        k_top = max(1, int(population_size * top_fraction))
        top_idx = sorted_idx[-k_top:]
        worst_idx = sorted_idx[:max(1, population_size - k_top)]

        for i in range(population_size):
            j = int(random.choice(top_idx))
            candidate = list(set(population[i]) | set(population[j]))
            candidate = [f for f in candidate if random.random() > 0.3]

            score = evaluate_subset(candidate, X_tr, y_tr, X_val, y_val, model, cache, beta)

            if score >= performance[i]:
                population[i] = candidate
                performance[i] = score
                worst_counter[i] = 0
            else:
                worst_counter[i] += 1

        for i in worst_idx:
            if worst_counter[i] >= q_iters:
                population[i] = random.sample(range(n_features), random.randint(1, n_features))
                performance[i] = evaluate_subset(population[i], X_tr, y_tr, X_val, y_val, model, cache, beta)
                worst_counter[i] = 0

        it_best = int(np.argmax(performance))
        if performance[it_best] > best_score:
            best_score = performance[it_best]
            best_subset = population[it_best]

        # -------- Verbose logging --------
        print(f"\nIteration {it + 1}/{num_iterations}")
        for idx, (subset, score) in enumerate(zip(population, performance)):
            subset_sorted = sorted(subset)
            shown = subset_sorted[:max_print_features]
            suffix = " ..." if len(subset_sorted) > max_print_features else ""
            print(f"Agent {idx:02d}: #F={len(subset_sorted)} | "
                  f"Features={shown}{suffix} | Score={score:.5f}")

        print(f"Best at Iter {it + 1}: Agent {it_best:02d}, "
              f"#F={len(population[it_best])}, Score={performance[it_best]:.5f}")
        print("-" * 65)

    return sorted(best_subset)


# =========================================================
# Final EDGE LR + confidence routing
# =========================================================
def train_and_route(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected: List[int],
    confidence_threshold: float
):

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    )

    t0 = time.time()
    clf.fit(X_train[:, selected], y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = clf.predict(X_test[:, selected])
    y_prob = clf.predict_proba(X_test[:, selected])
    pred_time = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n=== EDGE LR PERFORMANCE ===")
    print(f"Train time (s): {train_time:.4f}")
    print(f"Pred  time (s): {pred_time:.4f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    confidence = np.max(y_prob, axis=1)
    high = confidence >= confidence_threshold
    low = ~high

    print("\n=== CONFIDENCE ROUTING ===")
    print(f"Threshold: {confidence_threshold}")
    print(f"High-confidence flows: {high.sum()}")
    print(f"Low-confidence → Cloud: {low.sum()}")

    if high.sum() > 0:
        print("High-confidence CM:\n", confusion_matrix(y_test[high], y_pred[high]))
    if low.sum() > 0:
        print("Low-confidence CM:\n", confusion_matrix(y_test[low], y_pred[low]))


# =========================================================
# Main
# =========================================================
def main():
    print("\n=== EDGE IDS (CPU + BBFS) ===")
    seed = prompt_int("Random seed?", default=42)

    use_npz = prompt_yes_no("Load from NPZ?", default_yes=True)

    if use_npz:
        train_npz = prompt_str(
            "Train NPZ path:",
            default=os.path.expanduser("~/Jupyter/PhD/PreparedData/train_prepared.npz")
        )
        test_npz = prompt_str(
            "Test NPZ path:",
            default=os.path.expanduser("~/Jupyter/PhD/PreparedData/test_prepared.npz")
        )
        X_train, y_train, X_test, y_test, feature_names = load_from_npz(train_npz, test_npz)
    else:
        X_train, y_train, X_test, y_test, feature_names = run_data_prep_interactive()

    print_class_counts(y_train, "Train distribution")
    print_class_counts(y_test, "Test distribution")

    num_iter = prompt_int("BBFS iterations?", default=20)
    pop_size = prompt_int("BBFS population size?", default=30)
    top_frac = prompt_float("BBFS top_fraction?", default=0.7)
    q = prompt_float("BBFS q (fraction of iterations)?", default=0.15)
    beta = prompt_float("Custom F1 beta?", default=1.2)
    conf_th = prompt_float("Confidence threshold?", default=0.98)

    base_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    )

    print("\nRunning BBFS wrapper ...")
    t0 = time.time()
    selected = bbfs_wrapper(
        X_train, y_train,
        model=base_model,
        num_iterations=num_iter,
        population_size=pop_size,
        top_fraction=top_frac,
        q=q,
        beta=beta,
        seed=seed
    )
    print(f"\nBBFS completed in {time.time() - t0:.2f}s")

    print("\nSelected feature indices:", selected)
    print("Selected feature names:",
          [feature_names[i] for i in selected])

    train_and_route(
        X_train, y_train,
        X_test, y_test,
        selected,
        confidence_threshold=conf_th
    )


if __name__ == "__main__":
    main()
