"""Model modules"""

from .layers import ConvBlock, ResidualBlock, DenseBlock, CBAM
from .cnn_branch import SimpleCNNBranch, ResNetBranch, AttentionCNNBranch, create_cnn_branch
from .attention import (
    CrossModalAttention,
    BiDirectionalCrossAttention,
    MultiModalAttentionFusion,
    GatedFusion,
)
from .multimodal_model import MultimodalForestfireModel, create_multimodal_model
from .gnn import (
    GCNModel,
    GATModel,
    GraphSAGEModel,
    ResidualGCN,
    create_gnn_model,
)
from .hybrid_model import HybridGNNCNN, create_hybrid_model

__all__ = [
    # Layers
    "ConvBlock",
    "ResidualBlock",
    "DenseBlock",
    "CBAM",
    # CNN branches
    "SimpleCNNBranch",
    "ResNetBranch",
    "AttentionCNNBranch",
    "create_cnn_branch",
    # Attention
    "CrossModalAttention",
    "BiDirectionalCrossAttention",
    "MultiModalAttentionFusion",
    "GatedFusion",
    # Multimodal model
    "MultimodalForestfireModel",
    "create_multimodal_model",
    # GNN models
    "GCNModel",
    "GATModel",
    "GraphSAGEModel",
    "ResidualGCN",
    "create_gnn_model",
    # Hybrid model
    "HybridGNNCNN",
    "create_hybrid_model",
]
