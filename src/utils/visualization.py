"""Visualization utilities for model evaluation"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Plot confusion matrix

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        class_names: Names for classes
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_true = y_true.flatten()
    y_pred_binary = y_pred_binary.flatten()

    cm = confusion_matrix(y_true, y_pred_binary)

    if class_names is None:
        class_names = ["No Fire", "Fire"]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Plot ROC curve

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"ROC curve saved to {save_path}")

    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Plot precision-recall curve

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(recall, precision, color="darkgreen", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")

    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color="navy", lw=2, linestyle="--", label=f"Baseline ({baseline:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"PR curve saved to {save_path}")

    plt.close()


def plot_training_history(
    history: dict,
    metrics: List[str] = ["loss", "accuracy", "auc"],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
):
    """
    Plot training history

    Args:
        history: Training history dictionary from model.fit()
        metrics: List of metrics to plot
        save_path: Path to save figure
        figsize: Figure size
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)

    if num_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Plot train
        if metric in history:
            ax.plot(history[metric], label=f"Train {metric}", linewidth=2)

        # Plot validation
        val_metric = f"val_{metric}"
        if val_metric in history:
            ax.plot(history[val_metric], label=f"Val {metric}", linewidth=2, linestyle="--")

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f"{metric.capitalize()} vs Epoch", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Training history saved to {save_path}")

    plt.close()


def plot_spatial_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grid_shape: Tuple[int, int],
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    title: str = "Spatial Prediction Error",
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Plot spatial error map

    Args:
        y_true: Ground truth labels (flattened)
        y_pred: Predicted probabilities (flattened)
        grid_shape: Grid shape (height, width)
        threshold: Classification threshold
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Reshape to grid
    y_true_grid = y_true.reshape(grid_shape)
    y_pred_grid = y_pred_binary.reshape(grid_shape)
    y_pred_prob_grid = y_pred.reshape(grid_shape)

    # Create error map
    error_map = np.abs(y_true_grid - y_pred_grid)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot ground truth
    im1 = axes[0].imshow(y_true_grid, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot predictions (probabilities)
    im2 = axes[1].imshow(y_pred_prob_grid, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[1].set_title("Predicted Probability", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot error map
    im3 = axes[2].imshow(error_map, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("Prediction Error", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Spatial error map saved to {save_path}")

    plt.close()


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Prediction Distribution",
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot prediction probability distribution by class

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    fig, ax = plt.subplots(figsize=figsize)

    # Split by class
    y_pred_class0 = y_pred[y_true == 0]
    y_pred_class1 = y_pred[y_true == 1]

    # Plot histograms
    ax.hist(y_pred_class0, bins=50, alpha=0.6, label="No Fire (Class 0)", color="blue", density=True)
    ax.hist(y_pred_class1, bins=50, alpha=0.6, label="Fire (Class 1)", color="red", density=True)

    ax.axvline(x=0.5, color="black", linestyle="--", linewidth=2, label="Threshold (0.5)")

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Prediction distribution saved to {save_path}")

    plt.close()


def plot_class_distribution(
    y: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Class Distribution",
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Plot class distribution

    Args:
        y: Labels
        class_names: Names for classes
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    y = y.flatten()
    unique, counts = np.unique(y, return_counts=True)

    if class_names is None:
        class_names = [f"Class {i}" for i in unique]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(class_names, counts, color=["steelblue", "coral"])

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}\n({count/len(y)*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Class distribution saved to {save_path}")

    plt.close()
