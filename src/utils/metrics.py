"""Evaluation metrics for model performance"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    average: str = "binary",
) -> Dict[str, float]:
    """
    Compute classification metrics

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities (0-1)
        threshold: Classification threshold
        average: Averaging method for multi-class

    Returns:
        Dictionary of metrics
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Flatten if needed
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_pred_binary = y_pred_binary.flatten()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred_binary, average=average, zero_division=0),
    }

    # Add AUC metrics (only for binary classification)
    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred)
        metrics["auc_pr"] = average_precision_score(y_true, y_pred)
    except ValueError as e:
        logger.warning(f"Could not compute AUC: {e}")
        metrics["auc_roc"] = 0.0
        metrics["auc_pr"] = 0.0

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """
    Compute confusion matrix

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Confusion matrix (2x2 for binary)
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_true = y_true.flatten()
    y_pred_binary = y_pred_binary.flatten()

    cm = confusion_matrix(y_true, y_pred_binary)
    return cm


def compute_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Compute per-class metrics

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary with per-class metrics
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_true = y_true.flatten()
    y_pred_binary = y_pred_binary.flatten()

    cm = confusion_matrix(y_true, y_pred_binary)

    # For binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

        class_metrics = {
            "class_0": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "precision": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
                "recall": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                "support": int(tn + fp),
            },
            "class_1": {
                "true_positives": int(tp),
                "false_negatives": int(fn),
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                "support": int(tp + fn),
            },
        }

        # Add F1 scores
        class_metrics["class_0"]["f1_score"] = (
            2
            * class_metrics["class_0"]["precision"]
            * class_metrics["class_0"]["recall"]
            / (class_metrics["class_0"]["precision"] + class_metrics["class_0"]["recall"])
            if (class_metrics["class_0"]["precision"] + class_metrics["class_0"]["recall"]) > 0
            else 0.0
        )

        class_metrics["class_1"]["f1_score"] = (
            2
            * class_metrics["class_1"]["precision"]
            * class_metrics["class_1"]["recall"]
            / (class_metrics["class_1"]["precision"] + class_metrics["class_1"]["recall"])
            if (class_metrics["class_1"]["precision"] + class_metrics["class_1"]["recall"]) > 0
            else 0.0
        )

        return class_metrics

    return {}


def get_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, target_names=None
) -> str:
    """
    Get classification report

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        target_names: Names for classes

    Returns:
        Classification report string
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_true = y_true.flatten()
    y_pred_binary = y_pred_binary.flatten()

    if target_names is None:
        target_names = ["No Fire", "Fire"]

    report = classification_report(y_true, y_pred_binary, target_names=target_names)
    return report


def compute_spatial_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, grid_shape: Tuple[int, int], threshold: float = 0.5
) -> Dict:
    """
    Compute spatial metrics (for grid-based predictions)

    Args:
        y_true: Ground truth labels (flattened grid)
        y_pred: Predicted probabilities (flattened grid)
        grid_shape: Original grid shape (height, width)
        threshold: Classification threshold

    Returns:
        Dictionary with spatial metrics
    """
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Reshape to grid
    y_true_grid = y_true.reshape(grid_shape)
    y_pred_grid = y_pred_binary.reshape(grid_shape)

    # Compute spatial error map
    error_map = np.abs(y_true_grid - y_pred_grid)

    # Regional accuracy (divide grid into regions)
    h, w = grid_shape
    num_regions_h = 4
    num_regions_w = 4

    region_h = h // num_regions_h
    region_w = w // num_regions_w

    regional_metrics = []
    for i in range(num_regions_h):
        for j in range(num_regions_w):
            region_y_true = y_true_grid[
                i * region_h : (i + 1) * region_h, j * region_w : (j + 1) * region_w
            ].flatten()
            region_y_pred = y_pred_grid[
                i * region_h : (i + 1) * region_h, j * region_w : (j + 1) * region_w
            ].flatten()

            if len(region_y_true) > 0:
                acc = accuracy_score(region_y_true, region_y_pred)
                regional_metrics.append(
                    {"region": f"{i}_{j}", "accuracy": float(acc), "num_samples": len(region_y_true)}
                )

    return {
        "error_map": error_map,
        "mean_spatial_error": float(np.mean(error_map)),
        "max_spatial_error": float(np.max(error_map)),
        "regional_metrics": regional_metrics,
    }


class MetricsTracker:
    """Track metrics during training and evaluation"""

    def __init__(self):
        """Initialize metrics tracker"""
        self.history = {
            "train": [],
            "val": [],
            "test": [],
        }

    def update(self, split: str, metrics: Dict[str, float], epoch: Optional[int] = None):
        """
        Update metrics for a split

        Args:
            split: Data split ('train', 'val', 'test')
            metrics: Dictionary of metrics
            epoch: Optional epoch number
        """
        record = {"epoch": epoch, **metrics}
        self.history[split].append(record)

    def get_best(self, split: str, metric: str = "auc_roc", mode: str = "max") -> Dict:
        """
        Get best metrics for a split

        Args:
            split: Data split
            metric: Metric to optimize
            mode: 'max' or 'min'

        Returns:
            Best metrics record
        """
        if len(self.history[split]) == 0:
            return {}

        records = self.history[split]
        if mode == "max":
            best_record = max(records, key=lambda x: x.get(metric, -float("inf")))
        else:
            best_record = min(records, key=lambda x: x.get(metric, float("inf")))

        return best_record

    def get_latest(self, split: str) -> Dict:
        """
        Get latest metrics for a split

        Args:
            split: Data split

        Returns:
            Latest metrics record
        """
        if len(self.history[split]) == 0:
            return {}

        return self.history[split][-1]

    def summary(self) -> Dict:
        """Get summary of all metrics"""
        summary = {}

        for split in ["train", "val", "test"]:
            if len(self.history[split]) > 0:
                latest = self.get_latest(split)
                best_auc = self.get_best(split, "auc_roc", "max")

                summary[split] = {
                    "latest": latest,
                    "best_auc_roc": best_auc.get("auc_roc", 0.0),
                    "best_auc_epoch": best_auc.get("epoch", -1),
                }

        return summary
