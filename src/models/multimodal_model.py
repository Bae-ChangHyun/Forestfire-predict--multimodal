"""Multimodal forestfire prediction model"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Optional, Tuple

from .cnn_branch import create_cnn_branch
from .layers import DenseBlock
from .attention import MultiModalAttentionFusion, GatedFusion


class MultimodalForestfireModel(keras.Model):
    """Multimodal model for forestfire prediction"""

    def __init__(
        self,
        # Spatial features config
        spatial_features: List[str] = ["height", "ndvi", "slope", "landuse", "popden"],
        spatial_input_shape: Tuple[int, int, int] = (64, 64, 3),
        cnn_branch_type: str = "resnet",  # simple, resnet, attention
        cnn_output_dim: int = 128,
        # Climate features config
        climate_input_dim: int = 4,
        climate_hidden_dims: List[int] = [64, 64],
        # Fusion config
        fusion_method: str = "attention",  # concat, attention, gating
        use_cross_attention: bool = True,
        attention_heads: int = 4,
        # Output config
        output_dim: int = 1,
        output_activation: str = "sigmoid",
        # Regularization
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        name: str = "multimodal_forestfire",
        **kwargs,
    ):
        """
        Initialize multimodal forestfire prediction model

        Args:
            spatial_features: List of spatial feature names
            spatial_input_shape: Input shape for spatial features (H, W, C)
            cnn_branch_type: Type of CNN branch
            cnn_output_dim: Output dimension of CNN branches
            climate_input_dim: Input dimension of climate features
            climate_hidden_dims: Hidden layer dimensions for climate branch
            fusion_method: Method to fuse features ('concat', 'attention', 'gating')
            use_cross_attention: Whether to use cross-modal attention
            attention_heads: Number of attention heads
            output_dim: Output dimension (1 for binary classification)
            output_activation: Output activation function
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
            name: Model name
        """
        super().__init__(name=name, **kwargs)

        self.spatial_features = spatial_features
        self.fusion_method = fusion_method
        self.use_cross_attention = use_cross_attention

        # Create CNN branches for each spatial feature
        self.cnn_branches = {}
        for feature_name in spatial_features:
            branch = create_cnn_branch(
                branch_type=cnn_branch_type,
                input_shape=spatial_input_shape,
                output_dim=cnn_output_dim,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate,
            )
            self.cnn_branches[feature_name] = branch

        # Climate feature branch (Dense layers)
        self.climate_layers = []
        for i, hidden_dim in enumerate(climate_hidden_dims):
            self.climate_layers.append(
                DenseBlock(
                    units=hidden_dim,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=dropout_rate if i == len(climate_hidden_dims) - 1 else 0.0,
                    name=f"climate_dense_{i}",
                )
            )

        # Fusion layer
        if fusion_method == "attention" and use_cross_attention:
            self.fusion = MultiModalAttentionFusion(
                num_modalities=len(spatial_features) + 1,  # +1 for climate
                num_heads=attention_heads,
                key_dim=cnn_output_dim,
                dropout_rate=dropout_rate,
                fusion_method="concat",
                name="attention_fusion",
            )
        elif fusion_method == "gating":
            # Gated fusion
            self.fusion = GatedFusion(
                output_dim=cnn_output_dim, activation="tanh", name="gated_fusion"
            )
        else:
            # Simple concatenation
            self.fusion = None

        # Calculate fusion output dimension
        if fusion_method == "attention" and use_cross_attention:
            fusion_dim = cnn_output_dim * (len(spatial_features) + 1)
        elif fusion_method == "gating":
            fusion_dim = cnn_output_dim
        else:  # concat
            fusion_dim = cnn_output_dim * len(spatial_features) + climate_hidden_dims[-1]

        # Final classification layers
        self.final_dense = layers.Dense(128, activation="relu", name="final_dense")
        self.final_dropout = layers.Dropout(dropout_rate, name="final_dropout")
        self.output_layer = layers.Dense(
            output_dim, activation=output_activation, name="output_layer"
        )

    def call(self, inputs: Dict[str, tf.Tensor], training=None):
        """
        Forward pass

        Args:
            inputs: Dictionary with keys:
                - For each spatial feature: feature_name (e.g., 'height', 'ndvi')
                - 'climate': climate features
            training: Training mode

        Returns:
            Output predictions
        """
        # Process each spatial feature through CNN
        spatial_features = []
        for feature_name in self.spatial_features:
            if feature_name not in inputs:
                raise ValueError(f"Missing input for spatial feature: {feature_name}")

            features = self.cnn_branches[feature_name](inputs[feature_name], training=training)
            spatial_features.append(features)

        # Process climate features
        climate_input = inputs.get("climate")
        if climate_input is None:
            raise ValueError("Missing 'climate' input")

        # Reshape climate input if needed (add dimension for dense layers)
        if len(climate_input.shape) == 2:
            climate_input = tf.expand_dims(climate_input, axis=-1)

        climate_features = climate_input
        for layer in self.climate_layers:
            climate_features = layer(climate_features, training=training)

        # Flatten if still 3D
        if len(climate_features.shape) == 3:
            climate_features = layers.Flatten()(climate_features)

        # Fusion
        if self.fusion is not None:
            # Combine all features for attention/gating fusion
            all_features = spatial_features + [climate_features]

            if isinstance(self.fusion, (MultiModalAttentionFusion,)):
                fused = self.fusion(all_features, training=training)
            elif isinstance(self.fusion, (GatedFusion,)):
                fused = self.fusion(all_features)
            else:
                fused = tf.concat(all_features, axis=-1)
        else:
            # Simple concatenation
            fused = tf.concat(spatial_features + [climate_features], axis=-1)

        # Final classification
        x = self.final_dense(fused)
        x = self.final_dropout(x, training=training)
        output = self.output_layer(x)

        return output

    def get_config(self):
        """Get model config for serialization"""
        config = super().get_config()
        config.update(
            {
                "spatial_features": self.spatial_features,
                "fusion_method": self.fusion_method,
                "use_cross_attention": self.use_cross_attention,
            }
        )
        return config


def create_multimodal_model(config: Dict) -> keras.Model:
    """
    Factory function to create multimodal model from config

    Args:
        config: Configuration dictionary

    Returns:
        Compiled model
    """
    # Extract model config
    model_config = config.get("model", {})
    data_config = config.get("data", {})

    # Spatial features
    spatial_features = data_config.get("spatial_features", [
        "height", "slope", "ndvi", "landuse", "population_density"
    ])

    # Image size
    image_size = data_config.get("image_size", 64)
    spatial_input_shape = (image_size, image_size, 3)

    # CNN config
    cnn_config = model_config.get("cnn", {})
    cnn_branch_type = cnn_config.get("backbone", "resnet")
    cnn_output_dim = 128

    # Climate config
    climate_features = data_config.get("climate_features", ["humidity", "rainfall", "temp", "windspeed"])
    climate_input_dim = len(climate_features)

    # Fusion config
    fusion_config = model_config.get("fusion", {})
    fusion_method = fusion_config.get("method", "concat")
    use_cross_attention = fusion_config.get("use_cross_attention", False)
    attention_heads = fusion_config.get("attention_heads", 4)

    # Regularization
    dropout_rate = cnn_config.get("dropout_rate", 0.3)
    use_batch_norm = cnn_config.get("use_batch_norm", True)

    # Create model
    model = MultimodalForestfireModel(
        spatial_features=spatial_features,
        spatial_input_shape=spatial_input_shape,
        cnn_branch_type=cnn_branch_type,
        cnn_output_dim=cnn_output_dim,
        climate_input_dim=climate_input_dim,
        fusion_method=fusion_method,
        use_cross_attention=use_cross_attention,
        attention_heads=attention_heads,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
    )

    return model
