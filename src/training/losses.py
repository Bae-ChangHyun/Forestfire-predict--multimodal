"""Custom loss functions for forestfire prediction"""

import tensorflow as tf
from tensorflow import keras
from typing import Optional


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        name: str = "focal_loss",
        **kwargs,
    ):
        """
        Initialize focal loss

        Args:
            alpha: Weighting factor for class imbalance (0-1)
            gamma: Focusing parameter (higher = more focus on hard examples)
            from_logits: Whether predictions are logits or probabilities
            label_smoothing: Label smoothing factor
            name: Loss name
        """
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        """
        Compute focal loss

        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted probabilities or logits

        Returns:
            Focal loss value
        """
        # Convert to float
        y_true = tf.cast(y_true, tf.float32)

        # Apply label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Get probabilities
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Compute cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

        # Compute focal weight
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, self.gamma)

        # Compute alpha weight
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        # Focal loss
        focal_loss = alpha_weight * focal_weight * cross_entropy

        return tf.reduce_mean(focal_loss)

    def get_config(self):
        """Get config for serialization"""
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config


class WeightedBinaryCrossentropy(keras.losses.Loss):
    """Weighted binary crossentropy for class imbalance"""

    def __init__(
        self,
        pos_weight: float = 1.0,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        name: str = "weighted_bce",
        **kwargs,
    ):
        """
        Initialize weighted BCE

        Args:
            pos_weight: Weight for positive class
            from_logits: Whether predictions are logits
            label_smoothing: Label smoothing factor
            name: Loss name
        """
        super().__init__(name=name, **kwargs)
        self.pos_weight = pos_weight
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        """
        Compute weighted BCE

        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities or logits

        Returns:
            Weighted BCE loss
        """
        # Convert to float
        y_true = tf.cast(y_true, tf.float32)

        # Apply label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Compute BCE
        if self.from_logits:
            # Use numerically stable version
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        else:
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

        # Apply class weights
        weight = y_true * self.pos_weight + (1 - y_true) * 1.0
        weighted_bce = weight * bce

        return tf.reduce_mean(weighted_bce)

    def get_config(self):
        """Get config for serialization"""
        config = super().get_config()
        config.update(
            {
                "pos_weight": self.pos_weight,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config


def get_loss_function(loss_config: dict, class_weights: Optional[dict] = None):
    """
    Factory function to create loss function from config

    Args:
        loss_config: Loss configuration dictionary
        class_weights: Optional class weights {0: weight_neg, 1: weight_pos}

    Returns:
        Loss function
    """
    loss_type = loss_config.get("loss", "bce")

    if loss_type == "focal":
        alpha = loss_config.get("focal_alpha", 0.25)
        gamma = loss_config.get("focal_gamma", 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_type == "weighted_bce":
        if class_weights is not None:
            pos_weight = class_weights.get(1, 1.0)
        else:
            pos_weight = 1.0
        return WeightedBinaryCrossentropy(pos_weight=pos_weight)

    elif loss_type == "bce":
        # Standard binary crossentropy
        if class_weights is not None:
            # Use sample weights during training instead
            return keras.losses.BinaryCrossentropy()
        else:
            return keras.losses.BinaryCrossentropy()

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
