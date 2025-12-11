"""Cross-modal attention mechanisms for multimodal fusion"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Optional


class CrossModalAttention(layers.Layer):
    """Cross-modal attention between two modalities"""

    def __init__(
        self,
        num_heads: int = 4,
        key_dim: int = 64,
        dropout_rate: float = 0.1,
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize cross-modal attention

        Args:
            num_heads: Number of attention heads
            key_dim: Dimension of key/query
            dropout_rate: Dropout rate
            name: Layer name
        """
        super().__init__(name=name, **kwargs)

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate

        # Multi-head attention layer
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate
        )

        self.layer_norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, query, key_value, training=None):
        """
        Forward pass

        Args:
            query: Query modality features (batch, seq_len, dim)
            key_value: Key/Value modality features (batch, seq_len, dim)
            training: Training mode

        Returns:
            Attended features
        """
        # Ensure inputs are 3D (batch, seq, features)
        query_3d = tf.expand_dims(query, axis=1) if len(query.shape) == 2 else query
        kv_3d = tf.expand_dims(key_value, axis=1) if len(key_value.shape) == 2 else key_value

        # Cross attention: query attends to key_value
        attention_output = self.attention(
            query=query_3d, key=kv_3d, value=kv_3d, training=training
        )

        # Dropout
        attention_output = self.dropout(attention_output, training=training)

        # Residual connection + layer norm
        output = self.layer_norm(query_3d + attention_output)

        # Squeeze if input was 2D
        if len(query.shape) == 2:
            output = tf.squeeze(output, axis=1)

        return output


class BiDirectionalCrossAttention(layers.Layer):
    """Bi-directional cross-modal attention"""

    def __init__(
        self,
        num_heads: int = 4,
        key_dim: int = 64,
        dropout_rate: float = 0.1,
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize bi-directional cross attention

        Args:
            num_heads: Number of attention heads
            key_dim: Key dimension
            dropout_rate: Dropout rate
            name: Layer name
        """
        super().__init__(name=name, **kwargs)

        self.attention_a_to_b = CrossModalAttention(
            num_heads=num_heads, key_dim=key_dim, dropout_rate=dropout_rate, name="attn_a_to_b"
        )

        self.attention_b_to_a = CrossModalAttention(
            num_heads=num_heads, key_dim=key_dim, dropout_rate=dropout_rate, name="attn_b_to_a"
        )

    def call(self, features_a, features_b, training=None):
        """
        Forward pass

        Args:
            features_a: Features from modality A
            features_b: Features from modality B
            training: Training mode

        Returns:
            Tuple of (enhanced_a, enhanced_b)
        """
        # A attends to B
        enhanced_a = self.attention_a_to_b(query=features_a, key_value=features_b, training=training)

        # B attends to A
        enhanced_b = self.attention_b_to_a(query=features_b, key_value=features_a, training=training)

        return enhanced_a, enhanced_b


class MultiModalAttentionFusion(layers.Layer):
    """Multi-modal attention fusion for multiple modalities"""

    def __init__(
        self,
        num_modalities: int,
        num_heads: int = 4,
        key_dim: int = 64,
        dropout_rate: float = 0.1,
        fusion_method: str = "concat",  # concat, mean, weighted_sum
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize multi-modal attention fusion

        Args:
            num_modalities: Number of modalities to fuse
            num_heads: Number of attention heads
            key_dim: Key dimension
            dropout_rate: Dropout rate
            fusion_method: Method to fuse attended features
            name: Layer name
        """
        super().__init__(name=name, **kwargs)

        self.num_modalities = num_modalities
        self.fusion_method = fusion_method

        # Create attention layers for each pair
        self.attention_layers = []
        for i in range(num_modalities):
            self.attention_layers.append(
                CrossModalAttention(
                    num_heads=num_heads,
                    key_dim=key_dim,
                    dropout_rate=dropout_rate,
                    name=f"attn_modality_{i}",
                )
            )

        # Fusion layer
        if fusion_method == "weighted_sum":
            self.fusion_weights = None  # Will be created in build()

    def build(self, input_shape):
        """Build layer"""
        if self.fusion_method == "weighted_sum":
            # Learnable fusion weights
            self.fusion_weights = self.add_weight(
                name="fusion_weights",
                shape=(self.num_modalities,),
                initializer="ones",
                trainable=True,
            )

        super().build(input_shape)

    def call(self, modality_features: List[tf.Tensor], training=None):
        """
        Forward pass

        Args:
            modality_features: List of feature tensors from each modality
            training: Training mode

        Returns:
            Fused features
        """
        if len(modality_features) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(modality_features)}"
            )

        # Compute context: mean of all modalities
        context = tf.reduce_mean(tf.stack(modality_features, axis=0), axis=0)

        # Each modality attends to context
        attended_features = []
        for i, features in enumerate(modality_features):
            attended = self.attention_layers[i](query=features, key_value=context, training=training)
            attended_features.append(attended)

        # Fuse attended features
        if self.fusion_method == "concat":
            fused = tf.concat(attended_features, axis=-1)
        elif self.fusion_method == "mean":
            fused = tf.reduce_mean(tf.stack(attended_features, axis=0), axis=0)
        elif self.fusion_method == "weighted_sum":
            # Softmax normalize weights
            weights = tf.nn.softmax(self.fusion_weights)
            weighted_features = [w * f for w, f in zip(tf.unstack(weights), attended_features)]
            fused = tf.reduce_sum(tf.stack(weighted_features, axis=0), axis=0)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return fused


class GatedFusion(layers.Layer):
    """Gated fusion mechanism for multimodal features"""

    def __init__(
        self,
        output_dim: Optional[int] = None,
        activation: str = "tanh",
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize gated fusion

        Args:
            output_dim: Output dimension (default: same as input)
            activation: Activation function
            name: Layer name
        """
        super().__init__(name=name, **kwargs)

        self.output_dim = output_dim
        self.activation = activation

    def build(self, input_shape):
        """Build layer"""
        # Expect list of input shapes
        if not isinstance(input_shape, list):
            raise ValueError("GatedFusion expects a list of input tensors")

        # Determine output dim
        if self.output_dim is None:
            self.output_dim = input_shape[0][-1]

        # Create transformation and gate layers for each modality
        self.transforms = []
        self.gates = []

        for i, shape in enumerate(input_shape):
            self.transforms.append(
                layers.Dense(self.output_dim, activation=self.activation, name=f"transform_{i}")
            )
            self.gates.append(
                layers.Dense(self.output_dim, activation="sigmoid", name=f"gate_{i}")
            )

        super().build(input_shape)

    def call(self, inputs: List[tf.Tensor]):
        """
        Forward pass

        Args:
            inputs: List of feature tensors

        Returns:
            Gated fused features
        """
        # Transform and gate each modality
        gated_features = []
        for i, features in enumerate(inputs):
            transformed = self.transforms[i](features)
            gate = self.gates[i](features)
            gated = transformed * gate
            gated_features.append(gated)

        # Sum gated features
        fused = tf.reduce_sum(tf.stack(gated_features, axis=0), axis=0)

        return fused
