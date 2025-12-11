"""Hybrid GNN-CNN model for forestfire prediction"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import Dict, List, Optional, Literal
import logging

from .cnn_branch import create_cnn_branch
from .gnn import create_gnn_model
from .attention import GatedFusion

logger = logging.getLogger(__name__)


class HybridGNNCNN(nn.Module):
    """
    Hybrid model combining CNN for local feature extraction
    and GNN for spatial relationship modeling
    """

    def __init__(
        self,
        # CNN parameters
        spatial_features: List[str],
        cnn_branch_type: str = "resnet",
        cnn_output_dim: int = 128,
        image_size: int = 64,
        # Climate parameters
        climate_features: int = 4,
        climate_hidden_dim: int = 64,
        # GNN parameters
        gnn_type: str = "gat",
        gnn_hidden_channels: int = 64,
        gnn_num_layers: int = 3,
        gnn_dropout: float = 0.3,
        # Fusion parameters
        fusion_method: str = "gating",
        fusion_hidden_dim: int = 128,
        # General parameters
        dropout: float = 0.3,
        grid_height: int = 278,
        grid_width: int = 400,
    ):
        """
        Initialize Hybrid GNN-CNN model

        Args:
            spatial_features: List of spatial feature names
            cnn_branch_type: CNN branch type ('simple', 'resnet', 'attention')
            cnn_output_dim: Output dimension of CNN branches
            image_size: Input image size
            climate_features: Number of climate features
            climate_hidden_dim: Hidden dimension for climate branch
            gnn_type: GNN type ('gcn', 'gat', 'sage', 'residual_gcn')
            gnn_hidden_channels: Hidden dimension for GNN
            gnn_num_layers: Number of GNN layers
            gnn_dropout: GNN dropout rate
            fusion_method: Fusion method ('concat', 'gating', 'attention')
            fusion_hidden_dim: Hidden dimension for fusion
            dropout: General dropout rate
            grid_height: Grid height
            grid_width: Grid width
        """
        super().__init__()

        self.spatial_features = spatial_features
        self.num_spatial_features = len(spatial_features)
        self.climate_features = climate_features
        self.fusion_method = fusion_method
        self.grid_height = grid_height
        self.grid_width = grid_width

        # CNN branches for spatial features
        self.cnn_branches = nn.ModuleDict()
        for feature in spatial_features:
            input_shape = (image_size, image_size, 1)
            self.cnn_branches[feature] = create_cnn_branch(
                branch_type=cnn_branch_type,
                input_shape=input_shape,
                output_dim=cnn_output_dim,
            )

        # Climate feature branch (Dense layers)
        self.climate_branch = nn.Sequential(
            nn.Linear(climate_features, climate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(climate_hidden_dim, climate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Calculate total feature dimension for each node
        # CNN features + climate features
        total_node_features = cnn_output_dim * self.num_spatial_features + climate_hidden_dim

        # GNN for spatial relationship modeling
        self.gnn = create_gnn_model(
            model_type=gnn_type,
            in_channels=total_node_features,
            hidden_channels=gnn_hidden_channels,
            num_layers=gnn_num_layers,
            dropout=gnn_dropout,
            pooling=None,  # We'll do custom pooling
        )

        # Fusion layer
        if fusion_method == "concat":
            # Simple concatenation
            self.fusion = nn.Sequential(
                nn.Linear(gnn_hidden_channels + cnn_output_dim * self.num_spatial_features, fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif fusion_method == "gating":
            # Gated fusion
            self.fusion = GatedFusion(
                dim1=gnn_hidden_channels,
                dim2=cnn_output_dim * self.num_spatial_features,
                output_dim=fusion_hidden_dim,
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )

        logger.info(
            f"Hybrid GNN-CNN initialized: CNN={cnn_branch_type}, GNN={gnn_type}, "
            f"Fusion={fusion_method}"
        )

    def extract_cnn_features(self, spatial_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract CNN features from spatial inputs

        Args:
            spatial_inputs: Dictionary mapping feature names to tensors

        Returns:
            Concatenated CNN features [batch, total_cnn_dim]
        """
        cnn_features = []

        for feature_name in self.spatial_features:
            if feature_name in spatial_inputs:
                feature_data = spatial_inputs[feature_name]

                # Ensure 4D input: [batch, height, width, channels]
                if len(feature_data.shape) == 3:
                    feature_data = feature_data.unsqueeze(-1)

                # CNN branch
                feature_out = self.cnn_branches[feature_name](feature_data)
                cnn_features.append(feature_out)

        # Concatenate all CNN features
        cnn_features = torch.cat(cnn_features, dim=-1)

        return cnn_features

    def prepare_graph_features(
        self,
        spatial_inputs: Dict[str, torch.Tensor],
        climate_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare node features for GNN

        Args:
            spatial_inputs: Dictionary with spatial features [batch, H, W, C]
            climate_input: Climate features [batch, features]

        Returns:
            Node features [batch * num_nodes, total_features]
        """
        batch_size = climate_input.shape[0]
        num_nodes = self.grid_height * self.grid_width

        # Extract CNN features at each spatial location
        # For simplicity, we'll downsample CNN outputs to match grid
        # In practice, you might want to use adaptive pooling or other methods

        node_features_list = []

        # Spatial features (CNN features at each location)
        for feature_name in self.spatial_features:
            if feature_name in spatial_inputs:
                feature_data = spatial_inputs[feature_name]

                # Ensure 4D: [batch, H, W, C]
                if len(feature_data.shape) == 3:
                    feature_data = feature_data.unsqueeze(-1)

                # Flatten spatial dimensions
                # [batch, H, W, C] -> [batch, H*W, C]
                feature_flat = feature_data.view(batch_size, -1, feature_data.shape[-1])

                # Pad or trim to match num_nodes
                if feature_flat.shape[1] < num_nodes:
                    # Pad with zeros
                    padding = torch.zeros(
                        batch_size,
                        num_nodes - feature_flat.shape[1],
                        feature_flat.shape[-1],
                        device=feature_data.device,
                    )
                    feature_flat = torch.cat([feature_flat, padding], dim=1)
                elif feature_flat.shape[1] > num_nodes:
                    # Trim
                    feature_flat = feature_flat[:, :num_nodes, :]

                node_features_list.append(feature_flat)

        # Concatenate spatial features
        if node_features_list:
            spatial_node_features = torch.cat(node_features_list, dim=-1)
        else:
            spatial_node_features = torch.zeros(batch_size, num_nodes, 0)

        # Climate features (same for all nodes)
        climate_features = self.climate_branch(climate_input)

        # Expand to all nodes
        climate_features = climate_features.unsqueeze(1).expand(-1, num_nodes, -1)

        # Concatenate spatial and climate features
        node_features = torch.cat([spatial_node_features, climate_features], dim=-1)

        # Reshape to [batch * num_nodes, features]
        node_features = node_features.view(batch_size * num_nodes, -1)

        return node_features, batch_size

    def forward(
        self,
        spatial_inputs: Dict[str, torch.Tensor],
        climate_input: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            spatial_inputs: Dictionary with spatial features
            climate_input: Climate features [batch, features]
            edge_index: Edge index [2, num_edges]
            edge_attr: Optional edge attributes

        Returns:
            Predictions [batch, 1]
        """
        batch_size = climate_input.shape[0]

        # Extract CNN features (global representation)
        cnn_features = self.extract_cnn_features(spatial_inputs)

        # Prepare node features for GNN
        node_features, _ = self.prepare_graph_features(spatial_inputs, climate_input)

        # Create batch vector for graph pooling
        num_nodes = self.grid_height * self.grid_width
        batch_vector = torch.arange(batch_size, device=climate_input.device)
        batch_vector = batch_vector.repeat_interleave(num_nodes)

        # GNN forward pass
        gnn_output = self.gnn(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=None,  # We'll handle pooling ourselves
        )

        # Pool GNN output
        gnn_pooled = global_mean_pool(gnn_output, batch_vector)

        # Fusion
        if self.fusion_method == "concat":
            fused = torch.cat([gnn_pooled, cnn_features], dim=-1)
            fused = self.fusion(fused)
        elif self.fusion_method == "gating":
            fused = self.fusion(gnn_pooled, cnn_features)

        # Output
        output = self.output_layer(fused)
        output = torch.sigmoid(output)

        return output


def create_hybrid_model(config: Dict) -> HybridGNNCNN:
    """
    Create hybrid GNN-CNN model from config

    Args:
        config: Configuration dictionary

    Returns:
        Hybrid model
    """
    model_config = config.get("model", {})
    data_config = config.get("data", {})

    model = HybridGNNCNN(
        # Data parameters
        spatial_features=data_config.get("spatial_features", []),
        image_size=data_config.get("image_size", 64),
        climate_features=len(data_config.get("climate_features", [])),
        # CNN parameters
        cnn_branch_type=model_config.get("cnn", {}).get("backbone", "resnet"),
        cnn_output_dim=model_config.get("cnn", {}).get("output_dim", 128),
        # GNN parameters
        gnn_type=model_config.get("gnn", {}).get("type", "gat"),
        gnn_hidden_channels=model_config.get("gnn", {}).get("hidden_channels", 64),
        gnn_num_layers=model_config.get("gnn", {}).get("num_layers", 3),
        gnn_dropout=model_config.get("gnn", {}).get("dropout", 0.3),
        # Fusion parameters
        fusion_method=model_config.get("fusion", {}).get("method", "gating"),
        fusion_hidden_dim=model_config.get("fusion", {}).get("hidden_dim", 128),
        # General parameters
        dropout=model_config.get("dropout", 0.3),
    )

    return model
