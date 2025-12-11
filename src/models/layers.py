"""Custom layers and building blocks for models"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Optional, Tuple


class ConvBlock(layers.Layer):
    """Convolutional block with optional BatchNorm and Dropout"""

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        padding: str = "same",
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize convolutional block

        Args:
            filters: Number of filters
            kernel_size: Kernel size
            strides: Stride size
            padding: Padding mode
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0 = no dropout)
            name: Layer name
        """
        super().__init__(name=name, **kwargs)

        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=not use_batch_norm,  # No bias if using BN
        )

        self.batch_norm = layers.BatchNormalization() if use_batch_norm else None
        self.activation_fn = layers.Activation(activation)
        self.dropout = layers.Dropout(dropout_rate) if dropout_rate > 0 else None

    def call(self, inputs, training=None):
        """Forward pass"""
        x = self.conv(inputs)

        if self.batch_norm is not None:
            x = self.batch_norm(x, training=training)

        x = self.activation_fn(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        return x

    def get_config(self):
        """Get layer config for serialization"""
        config = super().get_config()
        return config


class ResidualBlock(layers.Layer):
    """Residual block with skip connection"""

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize residual block

        Args:
            filters: Number of filters
            kernel_size: Kernel size
            strides: Stride size
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
            name: Layer name
        """
        super().__init__(name=name, **kwargs)

        self.conv1 = ConvBlock(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            use_batch_norm=use_batch_norm,
            dropout_rate=0.0,  # No dropout in first conv
            activation="relu",
        )

        self.conv2 = layers.Conv2D(
            filters=filters, kernel_size=kernel_size, padding="same", use_bias=not use_batch_norm
        )

        self.batch_norm2 = layers.BatchNormalization() if use_batch_norm else None
        self.dropout = layers.Dropout(dropout_rate) if dropout_rate > 0 else None

        # Shortcut connection
        self.shortcut = None
        self.strides = strides

    def build(self, input_shape):
        """Build layer (create shortcut if needed)"""
        # Check if shortcut is needed (dimension mismatch)
        if input_shape[-1] != self.conv2.filters or self.strides != 1:
            self.shortcut = layers.Conv2D(
                filters=self.conv2.filters,
                kernel_size=1,
                strides=self.strides,
                padding="same",
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass"""
        # Main path
        x = self.conv1(inputs, training=training)
        x = self.conv2(x)

        if self.batch_norm2 is not None:
            x = self.batch_norm2(x, training=training)

        # Shortcut path
        shortcut = inputs
        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)

        # Add residual
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        return x


class DenseBlock(layers.Layer):
    """Dense block with BatchNorm and Dropout"""

    def __init__(
        self,
        units: int,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize dense block

        Args:
            units: Number of units
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
            name: Layer name
        """
        super().__init__(name=name, **kwargs)

        self.dense = layers.Dense(units, use_bias=not use_batch_norm)
        self.batch_norm = layers.BatchNormalization() if use_batch_norm else None
        self.activation_fn = layers.Activation(activation)
        self.dropout = layers.Dropout(dropout_rate) if dropout_rate > 0 else None

    def call(self, inputs, training=None):
        """Forward pass"""
        x = self.dense(inputs)

        if self.batch_norm is not None:
            x = self.batch_norm(x, training=training)

        x = self.activation_fn(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        return x


class SpatialAttention(layers.Layer):
    """Spatial attention mechanism"""

    def __init__(self, kernel_size: int = 7, name: Optional[str] = None, **kwargs):
        """
        Initialize spatial attention

        Args:
            kernel_size: Kernel size for convolution
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.kernel_size = kernel_size

        self.conv = layers.Conv2D(
            filters=1, kernel_size=kernel_size, padding="same", activation="sigmoid"
        )

    def call(self, inputs):
        """Forward pass"""
        # Compute channel-wise statistics
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

        # Concatenate
        concat = tf.concat([avg_pool, max_pool], axis=-1)

        # Compute attention map
        attention = self.conv(concat)

        # Apply attention
        return inputs * attention


class ChannelAttention(layers.Layer):
    """Channel attention mechanism (Squeeze-and-Excitation)"""

    def __init__(self, reduction_ratio: int = 16, name: Optional[str] = None, **kwargs):
        """
        Initialize channel attention

        Args:
            reduction_ratio: Reduction ratio for bottleneck
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.reduction_ratio = reduction_ratio
        self.global_avg_pool = layers.GlobalAveragePooling2D()

    def build(self, input_shape):
        """Build layer"""
        channels = input_shape[-1]
        reduced_channels = max(channels // self.reduction_ratio, 1)

        self.fc1 = layers.Dense(reduced_channels, activation="relu")
        self.fc2 = layers.Dense(channels, activation="sigmoid")

        super().build(input_shape)

    def call(self, inputs):
        """Forward pass"""
        # Global average pooling
        x = self.global_avg_pool(inputs)

        # FC layers
        x = self.fc1(x)
        x = self.fc2(x)

        # Reshape for broadcasting
        x = tf.reshape(x, [-1, 1, 1, tf.shape(inputs)[-1]])

        # Apply attention
        return inputs * x


class CBAM(layers.Layer):
    """Convolutional Block Attention Module (Channel + Spatial)"""

    def __init__(
        self,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize CBAM

        Args:
            reduction_ratio: Channel attention reduction ratio
            kernel_size: Spatial attention kernel size
            name: Layer name
        """
        super().__init__(name=name, **kwargs)

        self.channel_attention = ChannelAttention(reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def call(self, inputs):
        """Forward pass"""
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x
