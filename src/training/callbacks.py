"""Training callbacks for model training"""

import os
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_callbacks(config: dict, checkpoint_dir: str, log_dir: str):
    """
    Create training callbacks from configuration

    Args:
        config: Training configuration
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for logs

    Returns:
        List of callbacks
    """
    callbacks = []

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Early stopping
    early_stopping_config = config.get("early_stopping", {})
    if early_stopping_config:
        patience = early_stopping_config.get("patience", 10)
        metric = early_stopping_config.get("metric", "val_loss")
        mode = early_stopping_config.get("mode", "min")

        early_stopping = keras.callbacks.EarlyStopping(
            monitor=metric,
            patience=patience,
            mode=mode,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stopping)
        logger.info(f"Early stopping: monitor={metric}, patience={patience}, mode={mode}")

    # Model checkpoint
    checkpoint_config = config.get("checkpoint", {})
    if checkpoint_config:
        save_best = checkpoint_config.get("save_best", True)
        metric = checkpoint_config.get("metric", "val_loss")
        mode = checkpoint_config.get("mode", "min")

        checkpoint_path = Path(checkpoint_dir) / "best_model.h5"
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=metric,
            mode=mode,
            save_best_only=save_best,
            save_weights_only=False,
            verbose=1,
        )
        callbacks.append(model_checkpoint)
        logger.info(f"Model checkpoint: path={checkpoint_path}, monitor={metric}, mode={mode}")

        # Save last model as well
        if checkpoint_config.get("save_last", True):
            last_checkpoint_path = Path(checkpoint_dir) / "last_model.h5"
            last_checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=str(last_checkpoint_path),
                save_best_only=False,
                save_weights_only=False,
                verbose=0,
            )
            callbacks.append(last_checkpoint)

    # Learning rate scheduler
    lr_scheduler_config = config.get("lr_scheduler", {})
    if lr_scheduler_config:
        scheduler_type = lr_scheduler_config.get("type", "reduce_on_plateau")

        if scheduler_type == "reduce_on_plateau":
            patience = lr_scheduler_config.get("patience", 5)
            factor = lr_scheduler_config.get("factor", 0.5)
            min_lr = lr_scheduler_config.get("min_lr", 1e-6)

            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                verbose=1,
            )
            callbacks.append(reduce_lr)
            logger.info(f"ReduceLROnPlateau: factor={factor}, patience={patience}, min_lr={min_lr}")

        elif scheduler_type == "cosine":
            # Cosine annealing (requires custom callback)
            epochs = config.get("epochs", 100)
            initial_lr = config.get("learning_rate", 0.001)

            cosine_schedule = keras.experimental.CosineDecay(
                initial_learning_rate=initial_lr, decay_steps=epochs
            )

            lr_scheduler = keras.callbacks.LearningRateScheduler(
                lambda epoch: cosine_schedule(epoch), verbose=1
            )
            callbacks.append(lr_scheduler)
            logger.info(f"Cosine decay: initial_lr={initial_lr}, epochs={epochs}")

    # TensorBoard
    use_tensorboard = config.get("use_tensorboard", True)
    if use_tensorboard:
        tensorboard_dir = Path(log_dir) / "tensorboard"
        os.makedirs(tensorboard_dir, exist_ok=True)

        tensorboard = keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq="epoch",
        )
        callbacks.append(tensorboard)
        logger.info(f"TensorBoard: log_dir={tensorboard_dir}")

    # CSV logger
    csv_log_path = Path(log_dir) / "training_log.csv"
    csv_logger = keras.callbacks.CSVLogger(str(csv_log_path), append=True)
    callbacks.append(csv_logger)
    logger.info(f"CSV logger: path={csv_log_path}")

    # Progress bar (optional)
    # callbacks.append(keras.callbacks.ProgbarLogger())

    return callbacks


class MetricsLogger(keras.callbacks.Callback):
    """Custom callback to log additional metrics"""

    def __init__(self, log_dir: str):
        """
        Initialize metrics logger

        Args:
            log_dir: Directory to save logs
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.train_metrics = []
        self.val_metrics = []

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at end of epoch"""
        if logs is None:
            return

        # Collect train metrics
        train_log = {"epoch": epoch}
        val_log = {"epoch": epoch}

        for key, value in logs.items():
            if key.startswith("val_"):
                val_log[key] = float(value)
            else:
                train_log[key] = float(value)

        self.train_metrics.append(train_log)
        self.val_metrics.append(val_log)

        # Save to file
        import json

        with open(self.log_dir / "train_metrics.json", "w") as f:
            json.dump(self.train_metrics, f, indent=2)

        with open(self.log_dir / "val_metrics.json", "w") as f:
            json.dump(self.val_metrics, f, indent=2)


class GradientClippingCallback(keras.callbacks.Callback):
    """Callback for gradient clipping"""

    def __init__(self, clip_norm: float = 1.0):
        """
        Initialize gradient clipping callback

        Args:
            clip_norm: Maximum gradient norm
        """
        super().__init__()
        self.clip_norm = clip_norm

    def on_train_batch_begin(self, batch, logs=None):
        """Clip gradients before training batch"""
        # Note: Gradient clipping is usually done in optimizer
        # This is a placeholder for custom clipping logic
        pass
