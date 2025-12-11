"""Model evaluation utilities"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from .metrics import (
    compute_metrics,
    compute_confusion_matrix,
    compute_class_metrics,
    get_classification_report,
    compute_spatial_metrics,
)
from .visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_spatial_error,
    plot_prediction_distribution,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(
        self,
        output_dir: str,
        threshold: float = 0.5,
        class_names: Optional[list] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize model evaluator

        Args:
            output_dir: Directory to save evaluation results
            threshold: Classification threshold
            class_names: Names for classes
            grid_shape: Grid shape for spatial evaluation (optional)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.threshold = threshold
        self.class_names = class_names or ["No Fire", "Fire"]
        self.grid_shape = grid_shape

        logger.info(f"Model evaluator initialized. Output: {self.output_dir}")

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        split_name: str = "test",
        plot_spatial: bool = False,
    ) -> Dict:
        """
        Comprehensive evaluation

        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            split_name: Name of data split (e.g., 'test', 'val')
            plot_spatial: Whether to plot spatial error map

        Returns:
            Dictionary with all metrics
        """
        logger.info(f"Evaluating {split_name} set...")

        # Create split directory
        split_dir = self.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, threshold=self.threshold)
        logger.info(f"{split_name} metrics: {metrics}")

        # Confusion matrix
        cm = compute_confusion_matrix(y_true, y_pred, threshold=self.threshold)
        metrics["confusion_matrix"] = cm.tolist()

        # Per-class metrics
        class_metrics = compute_class_metrics(y_true, y_pred, threshold=self.threshold)
        metrics["class_metrics"] = class_metrics

        # Classification report
        report = get_classification_report(
            y_true, y_pred, threshold=self.threshold, target_names=self.class_names
        )
        logger.info(f"\n{split_name} Classification Report:\n{report}")

        # Save report
        with open(split_dir / "classification_report.txt", "w") as f:
            f.write(report)

        # Plot confusion matrix
        plot_confusion_matrix(
            y_true,
            y_pred,
            threshold=self.threshold,
            class_names=self.class_names,
            save_path=str(split_dir / "confusion_matrix.png"),
            title=f"Confusion Matrix - {split_name.capitalize()}",
        )

        # Plot ROC curve
        plot_roc_curve(
            y_true,
            y_pred,
            save_path=str(split_dir / "roc_curve.png"),
            title=f"ROC Curve - {split_name.capitalize()}",
        )

        # Plot PR curve
        plot_precision_recall_curve(
            y_true,
            y_pred,
            save_path=str(split_dir / "pr_curve.png"),
            title=f"Precision-Recall Curve - {split_name.capitalize()}",
        )

        # Plot prediction distribution
        plot_prediction_distribution(
            y_true,
            y_pred,
            save_path=str(split_dir / "prediction_distribution.png"),
            title=f"Prediction Distribution - {split_name.capitalize()}",
        )

        # Spatial metrics (if grid shape provided)
        if plot_spatial and self.grid_shape is not None:
            spatial_metrics = compute_spatial_metrics(
                y_true, y_pred, grid_shape=self.grid_shape, threshold=self.threshold
            )

            metrics["spatial_metrics"] = {
                "mean_spatial_error": spatial_metrics["mean_spatial_error"],
                "max_spatial_error": spatial_metrics["max_spatial_error"],
                "regional_metrics": spatial_metrics["regional_metrics"],
            }

            # Plot spatial error
            plot_spatial_error(
                y_true,
                y_pred,
                grid_shape=self.grid_shape,
                threshold=self.threshold,
                save_path=str(split_dir / "spatial_error.png"),
                title=f"Spatial Error Map - {split_name.capitalize()}",
            )

        # Save metrics to JSON
        metrics_json = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
        with open(split_dir / "metrics.json", "w") as f:
            json.dump(metrics_json, f, indent=2)

        logger.info(f"Evaluation complete. Results saved to {split_dir}")

        return metrics

    def compare_models(self, results: Dict[str, Dict], save_name: str = "comparison"):
        """
        Compare multiple models

        Args:
            results: Dictionary mapping model names to their metrics
            save_name: Name for comparison file
        """
        import pandas as pd

        # Create comparison DataFrame
        comparison_data = []

        for model_name, metrics in results.items():
            row = {"model": model_name}
            row.update(metrics)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Save to CSV
        csv_path = self.output_dir / f"{save_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Model comparison saved to {csv_path}")

        # Print comparison
        logger.info(f"\nModel Comparison:\n{df.to_string()}")

        return df

    def evaluate_threshold(
        self, y_true: np.ndarray, y_pred: np.ndarray, thresholds: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Evaluate different classification thresholds

        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            thresholds: List of thresholds to evaluate

        Returns:
            DataFrame with metrics for each threshold
        """
        import pandas as pd

        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)

        results = []

        for threshold in thresholds:
            metrics = compute_metrics(y_true, y_pred, threshold=threshold)
            metrics["threshold"] = threshold
            results.append(metrics)

        df = pd.DataFrame(results)

        # Save results
        csv_path = self.output_dir / "threshold_analysis.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Threshold analysis saved to {csv_path}")

        # Find optimal threshold (max F1 score)
        optimal_idx = df["f1_score"].idxmax()
        optimal_threshold = df.loc[optimal_idx, "threshold"]
        optimal_f1 = df.loc[optimal_idx, "f1_score"]

        logger.info(f"Optimal threshold: {optimal_threshold:.2f} (F1={optimal_f1:.4f})")

        return df
