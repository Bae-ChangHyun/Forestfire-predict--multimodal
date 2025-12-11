"""Inference script for forestfire prediction on new data"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data import ForestfireDataset, load_test_grid_data
from src.utils import setup_logger

import tensorflow as tf


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run inference on new forestfire data")
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
        help="Directory containing input data",
    )
    parser.add_argument(
        "--climate-file",
        type=str,
        required=True,
        help="CSV file with climate data for prediction",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output CSV file for predictions (default: auto-generated)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Save probability scores in addition to binary predictions",
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


def prepare_climate_data(
    climate_file: str,
    climate_features: list,
    dataset: ForestfireDataset,
    logger,
) -> np.ndarray:
    """
    Prepare climate data for inference

    Args:
        climate_file: Path to climate CSV file
        climate_features: List of climate feature names
        dataset: ForestfireDataset instance for normalization
        logger: Logger instance

    Returns:
        Normalized climate data array
    """
    df = pd.read_csv(climate_file)
    logger.info(f"Loaded climate data: {len(df)} samples")

    # Check for required columns
    missing_features = set(climate_features) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing climate features in CSV: {missing_features}")

    # Extract features
    climate_data = df[climate_features].values

    # Handle missing values if needed
    if dataset.missing_handler is not None:
        climate_data, _ = dataset.missing_handler.handle(climate_data)

    # Normalize if normalizer is fitted
    if dataset.climate_normalizer is not None:
        climate_data = dataset.climate_normalizer.transform(climate_data)

    # Reshape for model input (batch, features, 1)
    climate_data = climate_data.reshape(-1, len(climate_features), 1)

    logger.info(f"Climate data prepared: shape {climate_data.shape}")

    return climate_data


def prepare_spatial_data(
    data_dir: str,
    image_size: int,
    spatial_features: list,
    num_samples: int,
    logger,
) -> dict:
    """
    Prepare spatial data for inference

    Args:
        data_dir: Data directory
        image_size: Image size
        spatial_features: List of spatial feature names
        num_samples: Number of samples (from climate data)
        logger: Logger instance

    Returns:
        Dictionary with spatial data
    """
    # Load test grid data
    test_data = load_test_grid_data(data_dir, image_size)

    # Check if we have all required features
    missing_features = set(spatial_features) - set(test_data.keys())
    if missing_features:
        raise ValueError(f"Missing spatial features: {missing_features}")

    # Verify shapes match
    for feature, data in test_data.items():
        if len(data) != num_samples:
            logger.warning(
                f"Spatial feature {feature} has {len(data)} samples, "
                f"but climate data has {num_samples} samples. "
                f"Will use first {min(len(data), num_samples)} samples."
            )

    # Trim to match climate data size
    spatial_data = {}
    for feature in spatial_features:
        if feature in test_data:
            spatial_data[feature] = test_data[feature][:num_samples]

    logger.info(f"Spatial data prepared: {len(spatial_data)} features")

    return spatial_data


def run_inference(
    model,
    climate_data: np.ndarray,
    spatial_data: dict,
    batch_size: int,
    logger,
) -> np.ndarray:
    """
    Run inference on prepared data

    Args:
        model: Trained model
        climate_data: Climate data array
        spatial_data: Dictionary with spatial data
        batch_size: Batch size
        logger: Logger instance

    Returns:
        Prediction probabilities
    """
    # Prepare inputs
    inputs = spatial_data.copy()
    inputs["climate"] = climate_data

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Run inference
    logger.info("Running inference...")
    predictions = model.predict(dataset, verbose=1)

    return predictions


def save_predictions(
    predictions: np.ndarray,
    threshold: float,
    output_file: str,
    save_probabilities: bool,
    climate_file: str,
    logger,
):
    """
    Save predictions to CSV

    Args:
        predictions: Prediction probabilities
        threshold: Classification threshold
        output_file: Output file path
        save_probabilities: Whether to save probabilities
        climate_file: Original climate file (for metadata)
        logger: Logger instance
    """
    # Load original data for metadata
    original_df = pd.read_csv(climate_file)

    # Create output dataframe
    output_df = pd.DataFrame()

    # Copy relevant columns from original data if available
    if "grid_id" in original_df.columns:
        output_df["grid_id"] = original_df["grid_id"]
    if "latitude" in original_df.columns:
        output_df["latitude"] = original_df["latitude"]
    if "longitude" in original_df.columns:
        output_df["longitude"] = original_df["longitude"]
    if "date" in original_df.columns:
        output_df["date"] = original_df["date"]

    # Add predictions
    output_df["fire_probability"] = predictions.flatten()
    output_df["fire_predicted"] = (predictions.flatten() >= threshold).astype(int)

    if save_probabilities:
        output_df["probability_class_0"] = 1 - predictions.flatten()
        output_df["probability_class_1"] = predictions.flatten()

    # Add metadata
    output_df["threshold"] = threshold
    output_df["prediction_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to CSV
    output_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")

    # Print summary statistics
    num_fire_predicted = output_df["fire_predicted"].sum()
    avg_probability = output_df["fire_probability"].mean()
    max_probability = output_df["fire_probability"].max()

    logger.info("=" * 80)
    logger.info("PREDICTION SUMMARY:")
    logger.info(f"  Total samples:        {len(output_df)}")
    logger.info(f"  Fire predicted:       {num_fire_predicted} ({num_fire_predicted/len(output_df)*100:.2f}%)")
    logger.info(f"  Average probability:  {avg_probability:.4f}")
    logger.info(f"  Max probability:      {max_probability:.4f}")
    logger.info("=" * 80)


def main():
    """Main inference function"""
    args = parse_args()

    # Setup GPU
    setup_gpu(args.gpu)

    # Setup logger
    logger = setup_logger("forestfire_predict", log_level="INFO")

    logger.info("=" * 80)
    logger.info("Starting forestfire prediction inference")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Climate data: {args.climate_file}")

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Initialize dataset (for normalizers)
    logger.info("Initializing dataset...")
    dataset = ForestfireDataset(
        data_dir=args.data_dir,
        spatial_features=config["data"]["spatial_features"],
        climate_features=config["data"]["climate_features"],
        image_size=config["data"]["image_size"],
    )

    # Load and fit normalizers on training data
    # This ensures predictions use the same normalization as training
    try:
        _, _, _ = dataset.prepare_data(
            train_ratio=config["data"]["train_ratio"],
            val_ratio=config["data"]["val_ratio"],
            test_ratio=config["data"]["test_ratio"],
            stratify=True,
            random_seed=config["random_seed"],
        )
        logger.info("Dataset normalizers fitted on training data")
    except Exception as e:
        logger.warning(f"Could not fit normalizers on training data: {e}")
        logger.warning("Predictions may not be properly normalized")

    # Load model
    model = load_model(args.model_path, logger)

    # Prepare climate data
    climate_data = prepare_climate_data(
        climate_file=args.climate_file,
        climate_features=config["data"]["climate_features"],
        dataset=dataset,
        logger=logger,
    )

    # Prepare spatial data
    spatial_data = prepare_spatial_data(
        data_dir=args.data_dir,
        image_size=config["data"]["image_size"],
        spatial_features=config["data"]["spatial_features"],
        num_samples=len(climate_data),
        logger=logger,
    )

    # Run inference
    predictions = run_inference(
        model=model,
        climate_data=climate_data,
        spatial_data=spatial_data,
        batch_size=args.batch_size,
        logger=logger,
    )

    # Generate output filename if not provided
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"predictions_{timestamp}.csv"

    # Save predictions
    save_predictions(
        predictions=predictions,
        threshold=args.threshold,
        output_file=args.output_file,
        save_probabilities=args.save_probabilities,
        climate_file=args.climate_file,
        logger=logger,
    )

    logger.info("=" * 80)
    logger.info("Inference complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
