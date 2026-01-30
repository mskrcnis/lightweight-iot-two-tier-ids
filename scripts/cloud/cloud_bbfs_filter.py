"""
CLOUD FILTER (CPU-only) — callable module + runnable script

Key capability:
- run_cloud_filter(return_selected_data=True) returns selected-feature datasets directly:
    X_train_sel, y_train, X_test_sel, y_test, selected_indices, selected_feature_names

Also supports running as a script:
    python scripts/cloud/cloud_bbfs_filter.py
"""

from __future__ import annotations

import os
import time
import random
import warnings
import importlib.util
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.stats import levy

from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from sklearn.utils import check_random_state
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*converge.*")
warnings.filterwarnings("ignore", message=".*max_iter.*")


# ---------------- Prompt helpers ----------------
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


# ---------------- Data loading ----------------
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
        raise FileNotFoundError(f"Expected data prep script at: {script_path}")

    spec = importlib.util.spec_from_file_location("data_prep_interactive", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create import spec for data_prep_interactive.py")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    X_train, y_train, X_test, y_test, feature_names, _scaler = mod.main(return_data=True)
    return X_train, y_train, X_test, y_test, feature_names


# ---------------- Drop constant cols ----------------
def drop_constant_cols(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    std = X_train.std(axis=0)
    keep = std > 0
    X_train2 = X_train[:, keep]
    X_test2 = X_test[:, keep]
    names2 = [n for n, k in zip(feature_names, keep) if k]
    return X_train2, X_test2, names2, keep


# ---------------- CMI via supervised KMeans binning ----------------
def _elbow_k(ks: np.ndarray, mis: np.ndarray) -> int:
    ks = np.asarray(ks)
    mis = np.asarray(mis)
    if len(ks) < 3 or np.allclose(mis, mis[0]):
        return int(ks[np.argmax(mis)])

    x1, y1 = ks[0], mis[0]
    x2, y2 = ks[-1], mis[-1]
    num = np.abs((y2 - y1) * ks - (x2 - x1) * mis + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    return int(ks[np.argmax(num / den)])


def _mi_of_binned_col(x_binned: np.ndarray, y: np.ndarray) -> float:
    return float(mutual_info_classif(x_binned.reshape(-1, 1), y, discrete_features=True)[0])


def select_k_for_feature(
    x_col: np.ndarray,
    y: np.ndarray,
    k_min: int = 2,
    k_max: int = 15,
    n_init: str | int = "auto",
    random_state: int = 42
):
    x_col = np.asarray(x_col).reshape(-1, 1)
    rs = check_random_state(random_state)

    uniq = np.unique(x_col)
    if uniq.size <= k_min:
        z = np.searchsorted(uniq, x_col.ravel())
        mi = _mi_of_binned_col(z.astype(int), y)
        return int(uniq.size), (np.array([uniq.size]), np.array([mi]))

    ks, mis = [], []
    k_hi = int(min(k_max, len(x_col) - 1))
    for k in range(max(k_min, 2), k_hi + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=k, n_init=n_init, random_state=int(rs.randint(0, 1e9)))
            labels = km.fit_predict(x_col)
        mi = _mi_of_binned_col(labels.astype(int), y)
        ks.append(k)
        mis.append(mi)

    ks = np.array(ks)
    mis = np.array(mis)
    best_k = _elbow_k(ks, mis)
    return int(best_k), (ks, mis)


def build_binned_matrix(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_min: int = 2,
    k_max: int = 5,
    n_init: str | int = "auto",
    random_state: int = 42,
    verbose: bool = True
):
    n, p = X_train.shape
    X_binned = np.zeros((n, p), dtype=np.int32)
    rs = check_random_state(random_state)

    for i in range(p):
        xi = X_train[:, i]
        best_k, _ = select_k_for_feature(
            xi, y_train,
            k_min=k_min, k_max=k_max,
            n_init=n_init, random_state=int(rs.randint(0, 1e9))
        )
        if verbose:
            print(f"Feature {i}: chosen k = {best_k}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if best_k <= 1:
                labels = np.zeros(n, dtype=np.int32)
            else:
                km = KMeans(n_clusters=best_k, n_init=n_init, random_state=int(rs.randint(0, 1e9)))
                labels = km.fit_predict(xi.reshape(-1, 1)).astype(np.int32)

        X_binned[:, i] = labels

    return X_binned


def compute_CMI(X_binned: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    scores = np.zeros(X_binned.shape[1], dtype=float)
    for i in range(X_binned.shape[1]):
        scores[i] = mutual_info_score(y_train, X_binned[:, i])
    return scores


# ---------------- Fisher scores ----------------
def calculate_fisher_score(feature_vector: np.ndarray, labels: np.ndarray) -> float:
    unique_classes = np.unique(labels)
    overall_mean = np.mean(feature_vector)

    numerator = 0.0
    denominator = 0.0
    for c in unique_classes:
        idx = np.where(labels == c)[0]
        Nc = len(idx)
        class_mean = np.mean(feature_vector[idx])
        class_variance = np.var(feature_vector[idx])
        numerator += Nc * (class_mean - overall_mean) ** 2
        denominator += Nc * class_variance

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_fisher_scores(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    p = X_train.shape[1]
    scores = np.zeros(p, dtype=float)
    for i in range(p):
        scores[i] = calculate_fisher_score(X_train[:, i], y_train)

    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val > min_val:
        scores = (scores - min_val) / (max_val - min_val)
    else:
        scores = np.zeros_like(scores)
    return scores


# ---------------- BBFS fitness + search ----------------
def subset_fitness(
    subset: List[int],
    mi_scores: np.ndarray,
    fisher_scores: np.ndarray,
    corr_abs: np.ndarray,
    penalty_lambda: float
) -> float:
    if len(subset) == 0:
        return 0.0

    mi_sum = float(np.sum(mi_scores[subset]))
    fisher_sum = float(np.sum(fisher_scores[subset]))

    corr_penalty = 0.0
    for a in range(len(subset)):
        ia = subset[a]
        for b in range(a + 1, len(subset)):
            ib = subset[b]
            corr_penalty += float(corr_abs[ia, ib])

    return (penalty_lambda / 2.0) * mi_sum + (penalty_lambda / 2.0) * fisher_sum - (1.0 - penalty_lambda) * corr_penalty


def levy_flight(Lambda: float) -> int:
    return int(levy.rvs(loc=0, scale=Lambda))


def bbfs_filter(
    num_features: int,
    mi_scores: np.ndarray,
    fisher_scores: np.ndarray,
    corr_abs: np.ndarray,
    penalty_lambda: float,
    num_iterations: int,
    population_size: int,
    top_fraction: float,
    q: float,
    Lambda: float,
    seed: int,
    max_print_features: int = 200
) -> List[int]:

    random.seed(seed)
    np.random.seed(seed)

    q_iters = max(1, int(q * num_iterations))

    population: List[List[int]] = [
        random.sample(range(num_features), k=random.randint(1, num_features))
        for _ in range(population_size)
    ]

    cache: Dict[Tuple[int, ...], float] = {}

    def eval_cached(subset: List[int]) -> float:
        key = tuple(sorted(subset))
        if key in cache:
            return cache[key]
        s = subset_fitness(subset, mi_scores, fisher_scores, corr_abs, penalty_lambda)
        cache[key] = s
        return s

    performance = [eval_cached(s) for s in population]
    best_i = int(np.argmax(performance))
    best_subset = population[best_i]
    best_score = performance[best_i]
    worst_counter = [0] * population_size

    for it in range(num_iterations):
        sorted_idx = np.argsort(performance)
        k_top = max(1, int(population_size * top_fraction))
        top_idx = sorted_idx[-k_top:]
        worst_idx = sorted_idx[:max(1, population_size - k_top)]

        for i in range(population_size):
            j = i
            while j == i:
                j = (i + levy_flight(Lambda)) % population_size
            j = max(0, min(population_size - 1, j))

            new_subset = population[i][:]

            if j in top_idx:
                observed = set(population[j]) - set(new_subset)
                if observed:
                    improvements = []
                    base = eval_cached(new_subset)
                    for fk in observed:
                        temp = new_subset + [fk]
                        sc = eval_cached(temp)
                        improvements.append((fk, sc - base))

                    total_imp = sum(im for _, im in improvements)
                    for fk, imp in improvements:
                        Pk = (imp / total_imp) if total_imp > 0 else 0.0
                        if random.random() < Pk:
                            new_subset.append(fk)

            base = eval_cached(new_subset)
            for fk in new_subset[:]:
                reduced = [f for f in new_subset if f != fk]
                sc = eval_cached(reduced)
                if sc >= base and random.random() < 0.8:
                    new_subset.remove(fk)

            population[i] = new_subset
            performance[i] = eval_cached(new_subset)

        for idx in worst_idx:
            worst_counter[idx] += 1
            if worst_counter[idx] >= q_iters:
                population[idx] = random.sample(range(num_features), k=random.randint(1, num_features))
                performance[idx] = eval_cached(population[idx])
                worst_counter[idx] = 0

        it_best = int(np.argmax(performance))
        if performance[it_best] > best_score or (
            performance[it_best] == best_score and len(population[it_best]) < len(best_subset)
        ):
            best_score = performance[it_best]
            best_subset = population[it_best]

        # ---- print agentwise ----
        print(f"\nIteration {it + 1}/{num_iterations}")
        for a_idx, (subset, score) in enumerate(zip(population, performance)):
            ss = sorted(subset)
            shown = ss[:max_print_features]
            suffix = " ..." if len(ss) > max_print_features else ""
            print(f"Agent {a_idx:02d}: #F={len(ss)} | Features={shown}{suffix} | Score={score:.5f}")
        print(f"Best at Iter {it + 1}: Agent {it_best:02d}, #F={len(population[it_best])}, Score={performance[it_best]:.5f}")
        print("-" * 70)

    return sorted(best_subset)


def run_cloud_filter(return_selected_data: bool = True):
    """
    Interactively runs: load prepared NPZ OR raw via data_prep_interactive
    Then computes CMI+Fisher+Corr and runs BBFS.
    Finally computes:
        X_train_sel = X_train[:, selected]
        X_test_sel  = X_test[:, selected]
    and returns those if requested.
    """
    print("\n=== CLOUD FILTER (BBFS + CMI/Fisher/Corr) ===")

    seed = prompt_int("Random seed?", default=42)

    # This filter module itself also supports NPZ vs raw
    use_npz = prompt_yes_no("Use prepared NPZ for filter input?", default_yes=True)
    if use_npz:
        train_npz = prompt_str("Train NPZ path:", default=os.path.expanduser("~/Jupyter/PhD/PreparedData/train_prepared.npz"))
        test_npz = prompt_str("Test NPZ path:", default=os.path.expanduser("~/Jupyter/PhD/PreparedData/test_prepared.npz"))
        X_train, y_train, X_test, y_test, feature_names = load_from_npz(os.path.expanduser(train_npz), os.path.expanduser(test_npz))
    else:
        X_train, y_train, X_test, y_test, feature_names = run_data_prep_interactive()

    print_class_counts(y_train, "Train distribution")
    print_class_counts(y_test, "Test distribution (unbalanced)")

    X_train, X_test, feature_names, _ = drop_constant_cols(X_train, X_test, feature_names)
    print(f"\nAfter dropping constant columns: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"#features kept: {len(feature_names)}")

    k_min = prompt_int("CMI binning k_min?", default=2)
    k_max = prompt_int("CMI binning k_max?", default=5)
    verbose_binning = prompt_yes_no("Print chosen k per feature?", default_yes=True)

    print("\nBuilding binned matrix for CMI ...")
    X_binned = build_binned_matrix(X_train, y_train, k_min=k_min, k_max=k_max, random_state=seed, verbose=verbose_binning)
    mi_scores = compute_CMI(X_binned, y_train)
    fisher_scores = compute_fisher_scores(X_train, y_train)
    corr_abs = np.abs(np.corrcoef(X_train, rowvar=False))

    penalty_lambda = prompt_float("penalty_lambda (λ)?", default=0.974)
    num_iterations = prompt_int("BBFS iterations?", default=50)
    population_size = prompt_int("BBFS population size?", default=50)
    top_fraction = prompt_float("BBFS top_fraction?", default=0.7)
    q = prompt_float("BBFS q (fraction of iterations)?", default=0.15)
    Lambda = prompt_float("Levy Lambda?", default=1.5)
    max_print_features = prompt_int("Max features to print per agent (truncate)?", default=200)

    print("\nRunning BBFS (FILTER) ...")
    t0 = time.time()
    selected = bbfs_filter(
        num_features=X_train.shape[1],
        mi_scores=mi_scores,
        fisher_scores=fisher_scores,
        corr_abs=corr_abs,
        penalty_lambda=penalty_lambda,
        num_iterations=num_iterations,
        population_size=population_size,
        top_fraction=top_fraction,
        q=q,
        Lambda=Lambda,
        seed=seed,
        max_print_features=max_print_features
    )
    fs_time = time.time() - t0

    selected_names = [feature_names[i] for i in selected]
    print("\n=== BBFS FILTER RESULT ===")
    print(f"Feature selection time (s): {fs_time:.2f}")
    print(f"Selected #features: {len(selected)}")
    print("Selected indices:", selected)
    print("Selected names:", selected_names)

    X_train_sel = X_train[:, selected]
    X_test_sel = X_test[:, selected]

    if return_selected_data:
        return X_train_sel, y_train, X_test_sel, y_test, selected, selected_names

    # if user wants to save, keep it optional
    save_out = prompt_yes_no("Save selected datasets to NPZ?", default_yes=True)
    if save_out:
        out_dir = prompt_str("Output directory:", default=os.path.expanduser("~/Jupyter/PhD/CloudFilter"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "cloud_filter_selected.npz")
        np.savez(
            out_path,
            X_train=X_train_sel, y_train=y_train,
            X_test=X_test_sel, y_test=y_test,
            selected_indices=np.array(selected, dtype=np.int32),
            selected_feature_names=np.array(selected_names, dtype=object)
        )
        print(f"Saved: {out_path}")

    return None


def main():
    run_cloud_filter(return_selected_data=False)


if __name__ == "__main__":
    main()
