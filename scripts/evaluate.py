"""Evaluation script for trained forestfire prediction model"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data import ForestfireDataset
from src.utils import (
    setup_logger,
    ModelEvaluator,
)

import tensorflow as tf


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate forestfire prediction model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved model (.keras or .h5)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test", "both"],
        default="test",
        help="Which data split to evaluate on",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--find-optimal-threshold",
        action="store_true",
        help="Find optimal threshold using validation set",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID (e.g., '0' or '0,1')",
    )

    return parser.parse_args()


def setup_gpu(gpu_id: str):
    """Setup GPU configuration"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found, using CPU")


def load_model(model_path: str, logger):
    """Load trained model"""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading model from {model_path}")

    try:
        # Load with custom objects if needed
        from src.training import FocalLoss, WeightedBinaryCrossentropy

        custom_objects = {
            "FocalLoss": FocalLoss,
            "WeightedBinaryCrossentropy": WeightedBinaryCrossentropy,
        }

        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects=custom_objects,
        )

        logger.info("Model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def evaluate_split(
    model,
    dataset: ForestfireDataset,
    data: dict,
    evaluator: ModelEvaluator,
    split_name: str,
    batch_size: int,
    logger,
):
    """Evaluate model on a data split"""
    logger.info(f"Evaluating on {split_name} set...")

    # Create TensorFlow dataset
    tf_dataset = dataset.create_tf_dataset(
        data,
        batch_size=batch_size,
        shuffle=False,
    )

    # Get predictions
    logger.info("Running inference...")
    predictions = model.predict(tf_dataset, verbose=1)

    # Evaluate
    logger.info("Computing metrics...")
    metrics = evaluator.evaluate(
        y_true=data["labels"],
        y_pred=predictions,
        split_name=split_name,
        plot_spatial=False,  # Can enable if grid_shape is available
    )

    # Print summary
    logger.info("=" * 80)
    logger.info(f"{split_name.upper()} SET RESULTS:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
    logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR:    {metrics['auc_pr']:.4f}")
    logger.info("=" * 80)

    return metrics, predictions


def main():
    """Main evaluation function"""
    args = parse_args()

    # Setup GPU
    setup_gpu(args.gpu)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(
        "forestfire_eval",
        log_file=str(output_dir / "evaluation.log"),
        log_level="INFO",
    )

    logger.info("=" * 80)
    logger.info("Starting forestfire prediction evaluation")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Output directory: {output_dir}")

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Load model
    model = load_model(args.model_path, logger)

    # Print model summary
    logger.info("\nModel Summary:")
    model.summary(print_fn=logger.info)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = ForestfireDataset(
        data_dir=args.data_dir,
        spatial_features=config["data"]["spatial_features"],
        climate_features=config["data"]["climate_features"],
        image_size=config["data"]["image_size"],
    )

    # Prepare data
    train_data, val_data, test_data = dataset.prepare_data(
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        stratify=True,
        random_seed=config["random_seed"],
    )

    # Find optimal threshold if requested
    threshold = args.threshold
    if args.find_optimal_threshold:
        logger.info("Finding optimal threshold on validation set...")

        # Create temporary evaluator for threshold finding
        temp_evaluator = ModelEvaluator(
            output_dir=str(output_dir / "threshold_analysis"),
            threshold=0.5,
        )

        # Get validation predictions
        val_dataset = dataset.create_tf_dataset(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
        )
        val_predictions = model.predict(val_dataset, verbose=1)

        # Find optimal threshold
        threshold_df = temp_evaluator.evaluate_threshold(
            y_true=val_data["labels"],
            y_pred=val_predictions,
        )

        # Get optimal threshold
        optimal_idx = threshold_df["f1_score"].idxmax()
        threshold = threshold_df.loc[optimal_idx, "threshold"]
        optimal_f1 = threshold_df.loc[optimal_idx, "f1_score"]

        logger.info(f"Optimal threshold: {threshold:.3f} (F1={optimal_f1:.4f})")

    # Create evaluator with threshold
    evaluator = ModelEvaluator(
        output_dir=str(output_dir),
        threshold=threshold,
    )

    # Evaluate on requested splits
    results = {}

    if args.split in ["val", "both"]:
        val_metrics, val_predictions = evaluate_split(
            model=model,
            dataset=dataset,
            data=val_data,
            evaluator=evaluator,
            split_name="val",
            batch_size=args.batch_size,
            logger=logger,
        )
        results["val"] = val_metrics

    if args.split in ["test", "both"]:
        test_metrics, test_predictions = evaluate_split(
            model=model,
            dataset=dataset,
            data=test_data,
            evaluator=evaluator,
            split_name="test",
            batch_size=args.batch_size,
            logger=logger,
        )
        results["test"] = test_metrics

    # Compare splits if both were evaluated
    if args.split == "both":
        logger.info("\nComparing validation and test sets...")
        comparison = evaluator.compare_models(
            results={"validation": results["val"], "test": results["test"]},
            save_name="val_vs_test_comparison",
        )

    logger.info("=" * 80)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
