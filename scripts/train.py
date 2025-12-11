"""Training script for forestfire prediction model"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data import ForestfireDataset
from src.models import create_multimodal_model
from src.training import get_loss_function, create_callbacks
from src.utils import (
    setup_logger,
    calculate_class_weights,
    analyze_class_imbalance,
    plot_training_history,
    ModelEvaluator,
)

import tensorflow as tf


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train forestfire prediction model")
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
        default="./outputs",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
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


def main():
    """Main training function"""
    args = parse_args()

    # Setup GPU
    setup_gpu(args.gpu)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(
        "forestfire_train",
        log_file=str(output_dir / "train.log"),
        log_level="INFO",
    )

    logger.info("=" * 80)
    logger.info("Starting forestfire prediction training")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Override config with command line arguments
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr

    # Log training configuration
    logger.info("Training configuration:")
    logger.info(f"  Epochs: {config['training']['epochs']}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  Loss: {config['training']['loss']}")

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

    # Analyze class imbalance
    logger.info("Analyzing class imbalance...")
    class_stats = analyze_class_imbalance(train_data["labels"])

    # Calculate class weights
    class_weights = calculate_class_weights(train_data["labels"], method="balanced")

    # Create TensorFlow datasets
    train_dataset = dataset.create_tf_dataset(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )

    val_dataset = dataset.create_tf_dataset(
        val_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    # Create model
    logger.info("Creating model...")
    model = create_multimodal_model(config)

    # Print model summary
    logger.info("\nModel Summary:")
    model.summary(print_fn=logger.info)

    # Get loss function
    loss_fn = get_loss_function(config["training"], class_weights)

    # Compile model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["training"]["learning_rate"],
        clipnorm=config["training"].get("grad_clip_norm", 1.0),
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc", curve="ROC"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
        ],
    )

    logger.info("Model compiled successfully")

    # Create callbacks
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    callbacks = create_callbacks(
        config=config["training"],
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
    )

    # Train model
    logger.info("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config["training"]["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("Training completed!")

    # Plot training history
    logger.info("Plotting training history...")
    plot_training_history(
        history.history,
        metrics=["loss", "accuracy", "auc"],
        save_path=str(output_dir / "training_history.png"),
    )

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    evaluator = ModelEvaluator(
        output_dir=str(output_dir / "evaluation"),
        threshold=config["evaluation"]["threshold"],
    )

    # Get predictions
    val_predictions = model.predict(val_dataset)
    evaluator.evaluate(
        y_true=val_data["labels"],
        y_pred=val_predictions,
        split_name="val",
    )

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_dataset = dataset.create_tf_dataset(
        test_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    test_predictions = model.predict(test_dataset)
    evaluator.evaluate(
        y_true=test_data["labels"],
        y_pred=test_predictions,
        split_name="test",
    )

    # Find optimal threshold
    logger.info("Finding optimal threshold...")
    threshold_df = evaluator.evaluate_threshold(
        y_true=val_data["labels"],
        y_pred=val_predictions,
    )

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
