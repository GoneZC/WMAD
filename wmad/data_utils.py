"""Common data preparation utilities for WMAD experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Directory configuration
# ---------------------------------------------------------------------------


DEFAULT_DATA_DIR = Path("./data").resolve()


@dataclass(frozen=True)
class ViewConfig:
    filename_template: str
    drop_column: str


VIEW_CONFIGS: Dict[str, ViewConfig] = {
    "view1": ViewConfig("Part_B_sampled_{ratio}.csv", "Rndrng_NPI_"),
    "view2": ViewConfig("Part_D_sampled_{ratio}.csv", "Prscrbr_NPI_"),
    "view3": ViewConfig("DMEPOS_sampled_{ratio}.csv", "Rfrg_NPI_"),
    "labels": ViewConfig("labels_sampled_{ratio}.csv", "NPI"),
}


RESTORE_ANOMALY_COUNTS: Dict[float, int] = {
    0.90: 200,
    0.99: 100,
    0.999: 50,
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _resolve_ratio_suffix(fraud_ratio: float) -> str:
    """Format ratio value to match filename suffix (e.g. 0.05 -> '0.05')."""

    if 0 < fraud_ratio < 1:
        return f"{fraud_ratio:.2f}".rstrip("0").rstrip(".")
    return str(fraud_ratio)


def _load_csv(path: Path, drop_column: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if drop_column and drop_column in df.columns:
        df = df.drop(columns=[drop_column])
    return df


def load_multiview_data(
    fraud_ratio: float,
    data_dir: Path = DEFAULT_DATA_DIR,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load three views plus labels for the specified fraud ratio.

    Parameters
    ----------
    fraud_ratio: float
        The sampled fraud ratio (e.g. 0.05 for 5%).
    data_dir: Path
        Directory containing the sampled CSV files.
    scaler: Optional[MinMaxScaler]
        Optional scaler instance to reuse. If ``None``, a new scaler is fitted
        for each view separately (consistent with previous workflow).

    Returns
    -------
    Tuple of ``(X1, X2, X3, y)`` as numpy arrays.
    """

    ratio_suffix = _resolve_ratio_suffix(fraud_ratio)
    resolved_dir = data_dir.resolve()

    def build_path(config: ViewConfig) -> Path:
        filename = config.filename_template.format(ratio=ratio_suffix)
        return resolved_dir / filename

    df1 = _load_csv(build_path(VIEW_CONFIGS["view1"]), VIEW_CONFIGS["view1"].drop_column)
    df2 = _load_csv(build_path(VIEW_CONFIGS["view2"]), VIEW_CONFIGS["view2"].drop_column)
    df3 = _load_csv(build_path(VIEW_CONFIGS["view3"]), VIEW_CONFIGS["view3"].drop_column)
    labels_df = _load_csv(build_path(VIEW_CONFIGS["labels"]), VIEW_CONFIGS["labels"].drop_column)

    scaler = scaler or MinMaxScaler()

    X1 = scaler.fit_transform(df1.to_numpy())
    X2 = scaler.fit_transform(df2.to_numpy())
    X3 = scaler.fit_transform(df3.to_numpy())
    y = labels_df.to_numpy().ravel()

    return X1, X2, X3, y


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------


def compute_cv_splits(
    n_samples: int,
    n_splits: int,
    seed: int,
    val_ratio: float,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate (train, val, test) index tuples for KFold cross-validation."""

    indices = np.arange(n_samples)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for train_indices, test_indices in kf.split(indices):
        train_idx, val_idx = train_test_split(
            indices[train_indices], test_size=val_ratio, random_state=seed
        )
        splits.append((train_idx, val_idx, indices[test_indices]))

    return splits


# ---------------------------------------------------------------------------
# Unlabeled injection helper
# ---------------------------------------------------------------------------


def inject_unlabeled_labels(
    y_train: np.ndarray,
    unlabeled_ratio: float,
    *,
    restore_counts: Optional[Dict[float, int]] = None,
    min_anomalies: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """Inject unlabeled samples and optionally restore anomalies.

    Parameters mirror the logic previously implemented in
    ``experiment_unlabeled_ratios.py``.
    """

    rng = np.random.default_rng(random_state)

    restore_counts = restore_counts or RESTORE_ANOMALY_COUNTS

    y_original = y_train.astype(float, copy=True)
    y_augmented = y_original.copy()

    num_nan = int(unlabeled_ratio * len(y_augmented))
    nan_indices = rng.choice(len(y_augmented), num_nan, replace=False)
    y_augmented[nan_indices] = np.nan

    target_restore = restore_counts.get(unlabeled_ratio, 0)
    candidate_indices = [idx for idx in nan_indices if y_original[idx] == 1]

    if candidate_indices and target_restore > 0:
        if len(candidate_indices) > target_restore:
            restore_indices = rng.choice(candidate_indices, target_restore, replace=False)
        else:
            restore_indices = candidate_indices
        y_augmented[restore_indices] = 1

    current_anomalies = int(np.nansum(y_augmented == 1))
    if current_anomalies < min_anomalies:
        raise ValueError(
            "Insufficient anomalies after injecting unlabeled labels. "
            "Consider adjusting parameters."
        )

    return y_augmented


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def split_views(
    X_views: Sequence[np.ndarray],
    indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    """Split multiple views according to the provided index triplet."""

    train_idx, val_idx, test_idx = indices

    def take(view: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return view[idx]

    train_views = tuple(take(view, train_idx) for view in X_views)
    val_views = tuple(take(view, val_idx) for view in X_views)
    test_views = tuple(take(view, test_idx) for view in X_views)

    return train_views, val_views, test_views


def summarize_labels(y: np.ndarray) -> Dict[str, int]:
    """Return counts of unlabeled, normal, and anomalous samples for logging."""

    return {
        "unlabeled": int(np.isnan(y).sum()),
        "normal": int(np.sum(y == 0)),
        "anomalous": int(np.sum(y == 1)),
        "total": int(len(y)),
    }


