"""
Interactive data preparation for binary IDS (CPU-only):
- Select dataset from data/raw
- Auto-drop identifier columns: StartTime, LastTime, SrcAddr, DstAddr, sIpId, dIpId
- Ask extra columns to drop + ask label column
- Optional stratified fraction
- Stratified train/test split
- Normalize: fit scaler on train, transform test
- Optional balancing on TRAIN only:
    (1) SMOTE
    (2) SMOTE + ENN
    (3) SMOTE + ENN + LOF outlier removal
Outputs:
- Balanced train (X_train, y_train)
- Unbalanced test (X_test, y_test)
- feature_names, scaler
Optionally saves NPZ.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours


DEFAULT_DROP_COLS = [
    "StartTime",
    "LastTime",
    "SrcAddr",
    "DstAddr",
    "sIpId",
    "dIpId",
]


# -----------------------------
# Utilities
# -----------------------------
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


def print_class_counts(y: np.ndarray, title: str = "Class distribution") -> None:
    classes, counts = np.unique(y, return_counts=True)
    print(f"\n{title}:")
    for c, n in zip(classes, counts):
        print(f"  Class {c}: {n}")
    total = counts.sum()
    if total > 0 and len(classes) == 2:
        attack = counts[classes.tolist().index(1)] if 1 in classes else 0
        print(f"  Attack ratio: {attack/total:.4%}")


def parse_csv_list(s: str) -> List[str]:
    if not s.strip():
        return []
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [x.strip() for x in s.split() if x.strip()]


def safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns to numeric where possible.
    Non-numeric values become NaN (later filled).
    """
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# -----------------------------
# Dataset loading
# -----------------------------
def list_datasets_in_raw(raw_dir: str) -> list[str]:
    if not os.path.isdir(raw_dir):
        return []
    exts = (".csv", ".parquet", ".feather", ".tsv")
    files = [
        f for f in os.listdir(raw_dir)
        if os.path.isfile(os.path.join(raw_dir, f)) and f.lower().endswith(exts)
    ]
    files.sort()
    return files


def _read_any(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".feather"):
        return pd.read_feather(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_dataset_interactive() -> pd.DataFrame:
    print("\n=== DATASET LOADING ===")

    raw_dir = os.path.abspath(os.path.join(os.getcwd(), "data", "raw"))
    files = list_datasets_in_raw(raw_dir)

    if files:
        print(f"\nDatasets found in: {raw_dir}\n")
        for i, f in enumerate(files, start=1):
            print(f"{i:02d}. {f}")

        print("\nChoose a dataset number from the list above.")
        print("If your dataset is not shown, enter 0 to browse/provide a path.")
        choice = prompt_int("Enter choice (number):", default=1)

        if choice != 0:
            if choice < 1 or choice > len(files):
                raise ValueError("Invalid dataset selection number.")
            selected_path = os.path.join(raw_dir, files[choice - 1])
            print(f"\nSelected: {selected_path}")
            df = _read_any(selected_path)
            print(f"Shape: {df.shape}")
            return df

    print("\nNo datasets found in data/raw OR you selected browse mode.")
    default_path = os.path.expanduser("~/Jupyter/PhD/WUSTL-IIoT/wustl_corrected.csv")
    path = prompt_str("Enter dataset file path to load:", default=default_path)
    path = os.path.expanduser(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = _read_any(path)
    print(f"Loaded dataset: {path}")
    print(f"Shape: {df.shape}")
    return df


# -----------------------------
# Sampling / Columns / Labels
# -----------------------------
def stratified_fraction(df: pd.DataFrame, y_col: str, frac: float, seed: int) -> pd.DataFrame:
    if y_col not in df.columns:
        raise ValueError(f"Stratify column '{y_col}' not found in dataframe.")
    if not (0 < frac <= 1.0):
        raise ValueError("frac must be in (0, 1].")
    if frac == 1.0:
        return df

    idx = np.arange(len(df))
    y = df[y_col].to_numpy()

    idx_small, _ = train_test_split(
        idx,
        train_size=frac,
        random_state=seed,
        stratify=y
    )
    return df.iloc[idx_small].reset_index(drop=True)


def choose_columns_and_label(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    print("\n=== COLUMN SELECTION ===")

    existing_defaults = [c for c in DEFAULT_DROP_COLS if c in df.columns]
    if existing_defaults:
        df = df.drop(columns=existing_defaults, errors="ignore")
        print(f"Automatically removed identifier columns: {existing_defaults}")
    else:
        print("No default identifier columns found to remove.")

    cols = list(df.columns)
    print("\nRemaining columns:")
    for i, c in enumerate(cols):
        print(f"{i:02d}: {c}")

    extra_input = prompt_str(
        "\nEnter any OTHER columns to remove (comma separated), or press Enter to skip:",
        default=""
    )
    extra_drop = parse_csv_list(extra_input)

    existing_extra = [c for c in extra_drop if c in df.columns]
    missing_extra = [c for c in extra_drop if c not in df.columns]

    if existing_extra:
        df = df.drop(columns=existing_extra, errors="ignore")
        print(f"Removed additional columns: {existing_extra}")
    if missing_extra:
        print(f"Warning: these columns were not found and ignored: {missing_extra}")

    print(f"Shape after column removal: {df.shape}")

    default_label = "Target" if "Target" in df.columns else cols[-1]
    label_col = prompt_str("\nEnter class/label column name:", default=default_label)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")
    return df, label_col


def make_binary_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Enforce binary labels: Benign=0, Attack=1.
    If label already numeric {0,1}, no change.
    If string, map benign tokens to 0, else 1.
    """
    y = df[label_col]

    if pd.api.types.is_numeric_dtype(y):
        uniq = sorted(pd.unique(y.dropna()))
        if set(uniq).issubset({0, 1}):
            return df
        raise ValueError(f"Label column '{label_col}' is numeric but not binary: unique={uniq[:20]}")

    y_str = y.astype(str).str.lower()
    benign_tokens = {"benign", "normal", "0"}
    mapped = np.where(y_str.isin(benign_tokens), 0, 1)

    df2 = df.copy()
    df2[label_col] = mapped
    return df2


# -----------------------------
# Split + Normalize + Balance
# -----------------------------
def split_and_normalize(
    df: pd.DataFrame,
    label_col: str,
    test_size: float,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:

    print("\n=== SPLIT + NORMALIZE ===")
    X_df = df.drop(columns=[label_col])
    y = df[label_col].to_numpy()

    # numeric coercion
    X_df = safe_numeric_frame(X_df)
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(0.0)

    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
    print_class_counts(y_train, "Train (before balance)")
    print_class_counts(y_test, "Test (unbalanced)")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def balance_train_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    choice: str,
    seed: int,
    smote_k: int = 5,
    enn_k: int = 5,
    lof_k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:

    print("\n=== BALANCING (TRAIN ONLY) ===")
    smote = SMOTE(random_state=seed, k_neighbors=smote_k)

    if choice == "1":
        Xb, yb = smote.fit_resample(X_train, y_train)
        print_class_counts(yb, "Train (after SMOTE)")
        return Xb, yb

    enn = EditedNearestNeighbours(n_neighbors=enn_k)

    if choice == "2":
        X_sm, y_sm = smote.fit_resample(X_train, y_train)
        Xb, yb = enn.fit_resample(X_sm, y_sm)
        print_class_counts(yb, "Train (after SMOTE+ENN)")
        return Xb, yb

    if choice == "3":
        X_sm, y_sm = smote.fit_resample(X_train, y_train)
        X_se, y_se = enn.fit_resample(X_sm, y_sm)

        lof = LocalOutlierFactor(n_neighbors=lof_k, novelty=True)
        lof.fit(X_se)
        pred = lof.predict(X_se)  # -1 outlier, 1 inlier
        mask = pred == 1

        Xb = X_se[mask]
        yb = y_se[mask]
        print_class_counts(yb, "Train (after SMOTE+ENN+LOF)")
        return Xb, yb

    raise ValueError("Invalid balancing choice. Use 1/2/3.")


def maybe_save_npz(
    out_dir: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str]
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train_prepared.npz")
    test_path = os.path.join(out_dir, "test_prepared.npz")

    np.savez(train_path, X_train=X_train, y_train=y_train, feature_names=np.array(feature_names, dtype=object))
    np.savez(test_path, X_test=X_test, y_test=y_test, feature_names=np.array(feature_names, dtype=object))

    print(f"\nSaved:\n  {train_path}\n  {test_path}")


# -----------------------------
# Main interactive flow
# -----------------------------
def main(return_data: bool = False):
    seed = prompt_int("Random seed?", default=42)

    df = load_dataset_interactive()
    df, label_col = choose_columns_and_label(df)

    print("\n=== FRACTION (OPTIONAL DEMO MODE) ===")
    use_fraction = prompt_yes_no("Load only a fraction of the dataset (stratified)?", default_yes=False)
    if use_fraction:
        frac = prompt_float("Enter fraction (e.g., 0.1 for 10%)", default=0.1)
        df = stratified_fraction(df, y_col=label_col, frac=frac, seed=seed)
        print(f"Using stratified fraction. New shape: {df.shape}")

    df = make_binary_labels(df, label_col=label_col)

    test_size = prompt_float("Test size fraction?", default=0.2)
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_normalize(
        df, label_col=label_col, test_size=test_size, seed=seed
    )

    do_balance = prompt_yes_no("Balance training data?", default_yes=True)
    if do_balance:
        print("\nChoose balancing method:")
        print("  1) SMOTE")
        print("  2) SMOTE + ENN")
        print("  3) SMOTE + ENN + LOF")
        choice = prompt_str("Enter choice (1/2/3):", default="2")

        smote_k = prompt_int("SMOTE k_neighbors?", default=5)
        enn_k = prompt_int("ENN n_neighbors?", default=5)
        lof_k = prompt_int("LOF n_neighbors?", default=10)

        X_train, y_train = balance_train_data(
            X_train, y_train,
            choice=choice,
            seed=seed,
            smote_k=smote_k,
            enn_k=enn_k,
            lof_k=lof_k
        )
    else:
        print("\nSkipping balancing. Train remains imbalanced.")

    print("\n=== FINAL OUTPUTS ===")
    print("Train (prepared) :", X_train.shape, y_train.shape)
    print("Test  (prepared) :", X_test.shape, y_test.shape)
    print_class_counts(y_train, "Train (final)")
    print_class_counts(y_test, "Test (final, unbalanced)")

    save = prompt_yes_no("Save prepared arrays as NPZ?", default_yes=True)
    if save:
        default_out = os.path.expanduser("~/Jupyter/PhD/PreparedData")
        out_dir = prompt_str("Output directory:", default=default_out)
        maybe_save_npz(out_dir, X_train, y_train, X_test, y_test, feature_names)

    if return_data:
        return X_train, y_train, X_test, y_test, feature_names, scaler


if __name__ == "__main__":
    main(return_data=False)
