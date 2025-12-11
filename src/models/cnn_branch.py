"""CNN branch models for spatial feature extraction"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Optional

from .layers import ConvBlock, ResidualBlock, CBAM


class SimpleCNNBranch(keras.Model):
    """Simple CNN branch (baseline)"""

    def __init__(
        self,
        num_filters: List[int] = [32, 64, 128, 256],
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.3,
        output_dim: int = 128,
        name: str = "simple_cnn_branch",
        **kwargs,
    ):
        """
        Initialize simple CNN branch

        Args:
            num_filters: List of filter numbers for each layer
            kernel_size: Kernel size
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
            output_dim: Output feature dimension
            name: Model name
        """
        super().__init__(name=name, **kwargs)

        self.conv_blocks = []
        for i, filters in enumerate(num_filters):
            self.conv_blocks.append(
                ConvBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=0.0,
                    name=f"conv_block_{i}",
                )
            )
            self.conv_blocks.append(layers.MaxPooling2D(pool_size=2, name=f"pool_{i}"))

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(output_dim, activation="relu", name="dense_output")
        self.dropout = layers.Dropout(dropout_rate, name="dropout_output")

    def call(self, inputs, training=None):
        """Forward pass"""
        x = inputs

        for block in self.conv_blocks:
            if isinstance(block, ConvBlock):
                x = block(x, training=training)
            else:
                x = block(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x, training=training)

        return x


class ResNetBranch(keras.Model):
    """ResNet-style CNN branch with residual connections"""

    def __init__(
        self,
        num_filters: List[int] = [64, 128, 256],
        num_blocks_per_stage: List[int] = [2, 2, 2],
        use_batch_norm: bool = True,
        dropout_rate: float = 0.3,
        output_dim: int = 128,
        name: str = "resnet_branch",
        **kwargs,
    ):
        """
        Initialize ResNet branch

        Args:
            num_filters: Number of filters for each stage
            num_blocks_per_stage: Number of residual blocks per stage
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
            output_dim: Output feature dimension
            name: Model name
        """
        super().__init__(name=name, **kwargs)

        # Initial convolution
        self.conv_init = ConvBlock(
            filters=num_filters[0],
            kernel_size=7,
            strides=2,
            use_batch_norm=use_batch_norm,
            dropout_rate=0.0,
            name="conv_init",
        )
        self.pool_init = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="pool_init")

        # Residual stages
        self.residual_stages = []
        for stage_idx, (filters, num_blocks) in enumerate(zip(num_filters, num_blocks_per_stage)):
            stage_blocks = []

            # First block in stage (with stride=2 for downsampling, except first stage)
            strides = 2 if stage_idx > 0 else 1
            stage_blocks.append(
                ResidualBlock(
                    filters=filters,
                    strides=strides,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=0.0,
                    name=f"stage_{stage_idx}_block_0",
                )
            )

            # Remaining blocks
            for block_idx in range(1, num_blocks):
                stage_blocks.append(
                    ResidualBlock(
                        filters=filters,
                        use_batch_norm=use_batch_norm,
                        dropout_rate=0.0,
                        name=f"stage_{stage_idx}_block_{block_idx}",
                    )
                )

            self.residual_stages.append(stage_blocks)

        # Output layers
        self.global_avg_pool = layers.GlobalAveragePooling2D(name="global_avg_pool")
        self.dense = layers.Dense(output_dim, activation="relu", name="dense_output")
        self.dropout = layers.Dropout(dropout_rate, name="dropout_output")

    def call(self, inputs, training=None):
        """Forward pass"""
        x = self.conv_init(inputs, training=training)
        x = self.pool_init(x)

        # Residual stages
        for stage_blocks in self.residual_stages:
            for block in stage_blocks:
                x = block(x, training=training)

        x = self.global_avg_pool(x)
        x = self.dense(x)
        x = self.dropout(x, training=training)

        return x


class AttentionCNNBranch(keras.Model):
    """CNN branch with attention mechanisms"""

    def __init__(
        self,
        num_filters: List[int] = [32, 64, 128, 256],
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        use_attention: bool = True,
        dropout_rate: float = 0.3,
        output_dim: int = 128,
        name: str = "attention_cnn_branch",
        **kwargs,
    ):
        """
        Initialize attention CNN branch

        Args:
            num_filters: List of filter numbers
            kernel_size: Kernel size
            use_batch_norm: Whether to use batch normalization
            use_attention: Whether to use attention (CBAM)
            dropout_rate: Dropout rate
            output_dim: Output feature dimension
            name: Model name
        """
        super().__init__(name=name, **kwargs)

        self.blocks = []
        for i, filters in enumerate(num_filters):
            # Conv block
            self.blocks.append(
                ConvBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=0.0,
                    name=f"conv_block_{i}",
                )
            )

            # Attention
            if use_attention:
                self.blocks.append(CBAM(name=f"cbam_{i}"))

            # Pooling
            self.blocks.append(layers.MaxPooling2D(pool_size=2, name=f"pool_{i}"))

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(output_dim, activation="relu", name="dense_output")
        self.dropout = layers.Dropout(dropout_rate, name="dropout_output")

    def call(self, inputs, training=None):
        """Forward pass"""
        x = inputs

        for block in self.blocks:
            if isinstance(block, (ConvBlock,)):
                x = block(x, training=training)
            else:
                x = block(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x, training=training)

        return x


def create_cnn_branch(
    branch_type: str = "simple",
    input_shape: tuple = (64, 64, 3),
    output_dim: int = 128,
    **kwargs,
) -> keras.Model:
    """
    Factory function to create CNN branch

    Args:
        branch_type: Type of CNN branch ('simple', 'resnet', 'attention')
        input_shape: Input shape (height, width, channels)
        output_dim: Output feature dimension
        **kwargs: Additional arguments for specific branch type

    Returns:
        CNN branch model
    """
    # Create input layer
    inputs = layers.Input(shape=input_shape)

    # Select branch type
    if branch_type == "simple":
        branch = SimpleCNNBranch(output_dim=output_dim, **kwargs)
    elif branch_type == "resnet":
        branch = ResNetBranch(output_dim=output_dim, **kwargs)
    elif branch_type == "attention":
        branch = AttentionCNNBranch(output_dim=output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown branch type: {branch_type}")

    # Build model
    outputs = branch(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"{branch_type}_cnn_branch")

    return model
