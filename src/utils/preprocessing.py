"""Data preprocessing utilities"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import logging

from .constants import (
    DataConstants,
    ModelConstants,
    NormalizationMethod,
    MissingValueStrategy,
)

logger = logging.getLogger(__name__)


class MissingValueHandler:
    """Handle missing values in data"""

    def __init__(
        self,
        missing_indicators: Optional[List[float]] = None,
        strategy: Union[str, MissingValueStrategy] = MissingValueStrategy.MASK,
    ):
        """
        Initialize missing value handler

        Args:
            missing_indicators: List of values to treat as missing (default: from constants)
            strategy: Strategy for handling missing values
        """
        self.missing_indicators = missing_indicators or DataConstants.MISSING_VALUES

        if isinstance(strategy, str):
            strategy = MissingValueStrategy(strategy)
        self.strategy = strategy

    def create_mask(self, data: np.ndarray) -> np.ndarray:
        """
        Create boolean mask for missing values

        Args:
            data: Input data array

        Returns:
            Boolean mask (True = valid, False = missing)
        """
        mask = np.ones(data.shape, dtype=bool)
        for missing_val in self.missing_indicators:
            mask &= ~np.isclose(data, missing_val)
        mask &= ~np.isnan(data)
        return mask

    def handle(
        self, data: np.ndarray, fill_value: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Handle missing values based on strategy

        Args:
            data: Input data array
            fill_value: Value to use for filling (for mean/median strategies)

        Returns:
            Processed data and mask (if strategy is MASK)
        """
        mask = self.create_mask(data)

        if self.strategy == MissingValueStrategy.MASK:
            # Keep original data, return mask
            return data, mask

        elif self.strategy == MissingValueStrategy.DROP:
            # Return only valid data
            return data[mask], None

        elif self.strategy == MissingValueStrategy.ZERO:
            # Fill with zero
            data_filled = data.copy()
            data_filled[~mask] = 0.0
            return data_filled, None

        elif self.strategy == MissingValueStrategy.MEAN:
            # Fill with mean of valid values
            if fill_value is None:
                fill_value = np.mean(data[mask])
            data_filled = data.copy()
            data_filled[~mask] = fill_value
            return data_filled, None

        elif self.strategy == MissingValueStrategy.MEDIAN:
            # Fill with median of valid values
            if fill_value is None:
                fill_value = np.median(data[mask])
            data_filled = data.copy()
            data_filled[~mask] = fill_value
            return data_filled, None

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def replace_missing_values(self, data: np.ndarray, replace_with: float = np.nan) -> np.ndarray:
        """
        Replace missing value indicators with a specific value (e.g., np.nan)

        Args:
            data: Input data
            replace_with: Value to replace missing indicators with

        Returns:
            Data with replaced values
        """
        data_clean = data.copy()
        for missing_val in self.missing_indicators:
            data_clean = np.where(np.isclose(data_clean, missing_val), replace_with, data_clean)
        return data_clean


class DataNormalizer:
    """Normalize data using various methods"""

    def __init__(self, method: Union[str, NormalizationMethod] = NormalizationMethod.STANDARD):
        """
        Initialize normalizer

        Args:
            method: Normalization method
        """
        if isinstance(method, str):
            method = NormalizationMethod(method)
        self.method = method

        # Create scaler
        if method == NormalizationMethod.STANDARD:
            self.scaler = StandardScaler()
        elif method == NormalizationMethod.MINMAX:
            self.scaler = MinMaxScaler()
        elif method == NormalizationMethod.ROBUST:
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        self.fitted = False

    def fit(self, data: np.ndarray) -> "DataNormalizer":
        """
        Fit normalizer to data

        Args:
            data: Training data (n_samples, n_features)

        Returns:
            Self
        """
        # Handle 1D data
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self.scaler.fit(data)
        self.fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data

        Args:
            data: Input data

        Returns:
            Normalized data
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before transform")

        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        transformed = self.scaler.transform(data)

        # Restore original shape
        if len(original_shape) == 1:
            transformed = transformed.ravel()

        return transformed

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform data

        Args:
            data: Input data

        Returns:
            Normalized data
        """
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data

        Args:
            data: Normalized data

        Returns:
            Original scale data
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")

        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        original = self.scaler.inverse_transform(data)

        if len(original_shape) == 1:
            original = original.ravel()

        return original

    def save_params(self) -> Dict:
        """Save scaler parameters"""
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before saving")

        params = {
            "method": self.method.value,
            "mean_": getattr(self.scaler, "mean_", None),
            "scale_": getattr(self.scaler, "scale_", None),
            "min_": getattr(self.scaler, "min_", None),
            "data_min_": getattr(self.scaler, "data_min_", None),
            "data_max_": getattr(self.scaler, "data_max_", None),
            "center_": getattr(self.scaler, "center_", None),
        }
        return {k: v for k, v in params.items() if v is not None}


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = ModelConstants.TRAIN_RATIO,
    val_ratio: float = ModelConstants.VAL_RATIO,
    test_ratio: float = ModelConstants.TEST_RATIO,
    stratify: bool = True,
    random_seed: int = ModelConstants.RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets

    Args:
        X: Features
        y: Labels
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        stratify: Whether to stratify split by labels
        random_seed: Random seed

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Validate ratios
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1.0"

    stratify_param = y if stratify else None

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), stratify=stratify_param, random_state=random_seed
    )

    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    stratify_temp = y_temp if stratify else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_size), stratify=stratify_temp, random_state=random_seed
    )

    logger.info(f"Data split - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    logger.info(
        f"Class distribution - Train: {np.bincount(y_train)}, "
        f"Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_class_weights(y: np.ndarray, method: str = "balanced") -> Dict[int, float]:
    """
    Calculate class weights for imbalanced data

    Args:
        y: Labels
        method: Method for calculating weights ('balanced', 'sqrt')

    Returns:
        Dictionary of class weights {class_label: weight}
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    total = len(y)

    if method == "balanced":
        # Inverse frequency: n_samples / (n_classes * n_samples_for_class)
        weights = total / (len(unique_classes) * counts)
    elif method == "sqrt":
        # Square root of inverse frequency
        weights = np.sqrt(total / counts)
    else:
        raise ValueError(f"Unknown method: {method}")

    class_weights = {int(cls): float(weight) for cls, weight in zip(unique_classes, weights)}

    logger.info(f"Class weights ({method}): {class_weights}")
    logger.info(f"Class distribution: {dict(zip(unique_classes, counts))}")

    return class_weights


def analyze_class_imbalance(y: np.ndarray) -> Dict:
    """
    Analyze class imbalance in labels

    Args:
        y: Labels

    Returns:
        Dictionary with imbalance statistics
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    total = len(y)

    distribution = {int(cls): int(count) for cls, count in zip(unique_classes, counts)}
    ratios = {int(cls): float(count / total) for cls, count in zip(unique_classes, counts)}

    # Imbalance ratio (majority / minority)
    imbalance_ratio = max(counts) / min(counts) if len(counts) > 1 else 1.0

    stats = {
        "total_samples": total,
        "num_classes": len(unique_classes),
        "distribution": distribution,
        "ratios": ratios,
        "imbalance_ratio": float(imbalance_ratio),
    }

    logger.info(f"Class imbalance analysis:")
    logger.info(f"  Total samples: {total}")
    logger.info(f"  Distribution: {distribution}")
    logger.info(f"  Ratios: {ratios}")
    logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}")

    return stats
