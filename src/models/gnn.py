"""Graph Neural Network models for forestfire prediction"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)
from torch_geometric.data import Batch
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)


class GCNModel(nn.Module):
    """Graph Convolutional Network for forestfire prediction"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        pooling: str = "mean",
    ):
        """
        Initialize GCN model

        Args:
            in_channels: Number of input features
            hidden_channels: Hidden dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            pooling: Graph pooling method ('mean', 'max', 'add')
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.pooling = pooling

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Batch normalization
        if use_batch_norm:
            self.batch_norms = nn.ModuleList()
            for _ in range(num_layers):
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.output_layer = nn.Linear(hidden_channels, 1)

        logger.info(f"GCN initialized: {num_layers} layers, {hidden_channels} hidden")

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge index [2, num_edges]
            edge_attr: Edge attributes (optional)
            batch: Batch vector (optional)

        Returns:
            Predictions [batch_size, 1] or [num_nodes, 1]
        """
        # GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_attr)

            if self.use_batch_norm:
                x = self.batch_norms[i](x)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph pooling (if batch is provided)
        if batch is not None:
            if self.pooling == "mean":
                x = global_mean_pool(x, batch)
            elif self.pooling == "max":
                x = global_max_pool(x, batch)
            elif self.pooling == "add":
                x = global_add_pool(x, batch)
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

        # Output layer
        x = self.output_layer(x)
        x = torch.sigmoid(x)

        return x


class GATModel(nn.Module):
    """Graph Attention Network for forestfire prediction"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        pooling: str = "mean",
    ):
        """
        Initialize GAT model

        Args:
            in_channels: Number of input features
            hidden_channels: Hidden dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            pooling: Graph pooling method
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.pooling = pooling
        self.num_heads = num_heads

        # GAT layers
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=num_heads,
                dropout=dropout,
                concat=True,
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            )

        # Last layer (no concatenation)
        self.convs.append(
            GATConv(
                hidden_channels * num_heads,
                hidden_channels,
                heads=1,
                dropout=dropout,
                concat=False,
            )
        )

        # Batch normalization
        if use_batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels * num_heads))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.output_layer = nn.Linear(hidden_channels, 1)

        logger.info(
            f"GAT initialized: {num_layers} layers, {hidden_channels} hidden, "
            f"{num_heads} heads"
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """Forward pass"""
        # GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if self.use_batch_norm:
                x = self.batch_norms[i](x)

            if i < len(self.convs) - 1:  # Not the last layer
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Final activation
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph pooling
        if batch is not None:
            if self.pooling == "mean":
                x = global_mean_pool(x, batch)
            elif self.pooling == "max":
                x = global_max_pool(x, batch)
            elif self.pooling == "add":
                x = global_add_pool(x, batch)

        # Output
        x = self.output_layer(x)
        x = torch.sigmoid(x)

        return x


class GraphSAGEModel(nn.Module):
    """GraphSAGE for forestfire prediction"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        pooling: str = "mean",
        aggregator: str = "mean",
    ):
        """
        Initialize GraphSAGE model

        Args:
            in_channels: Number of input features
            hidden_channels: Hidden dimension
            num_layers: Number of SAGE layers
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            pooling: Graph pooling method
            aggregator: Aggregation method ('mean', 'max', 'lstm')
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.pooling = pooling

        # SAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))

        # Batch normalization
        if use_batch_norm:
            self.batch_norms = nn.ModuleList()
            for _ in range(num_layers):
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.output_layer = nn.Linear(hidden_channels, 1)

        logger.info(
            f"GraphSAGE initialized: {num_layers} layers, {hidden_channels} hidden, "
            f"{aggregator} aggregation"
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """Forward pass"""
        # SAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if self.use_batch_norm:
                x = self.batch_norms[i](x)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph pooling
        if batch is not None:
            if self.pooling == "mean":
                x = global_mean_pool(x, batch)
            elif self.pooling == "max":
                x = global_max_pool(x, batch)
            elif self.pooling == "add":
                x = global_add_pool(x, batch)

        # Output
        x = self.output_layer(x)
        x = torch.sigmoid(x)

        return x


class ResidualGCN(nn.Module):
    """GCN with residual connections"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        pooling: str = "mean",
    ):
        """Initialize Residual GCN"""
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.pooling = pooling

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GCN layers with residual connections
        self.convs = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.residual_projs.append(nn.Linear(hidden_channels, hidden_channels))

        # Batch normalization
        if use_batch_norm:
            self.batch_norms = nn.ModuleList()
            for _ in range(num_layers):
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.output_layer = nn.Linear(hidden_channels, 1)

        logger.info(f"Residual GCN initialized: {num_layers} layers")

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """Forward pass"""
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # GCN layers with residual connections
        for i, conv in enumerate(self.convs):
            identity = x

            # GCN convolution
            x = conv(x, edge_index, edge_weight=edge_attr)

            if self.use_batch_norm:
                x = self.batch_norms[i](x)

            x = F.relu(x)

            # Residual connection
            x = x + self.residual_projs[i](identity)

            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph pooling
        if batch is not None:
            if self.pooling == "mean":
                x = global_mean_pool(x, batch)
            elif self.pooling == "max":
                x = global_max_pool(x, batch)
            elif self.pooling == "add":
                x = global_add_pool(x, batch)

        # Output
        x = self.output_layer(x)
        x = torch.sigmoid(x)

        return x


def create_gnn_model(
    model_type: Literal["gcn", "gat", "sage", "residual_gcn"],
    in_channels: int,
    hidden_channels: int = 64,
    num_layers: int = 3,
    dropout: float = 0.3,
    use_batch_norm: bool = True,
    pooling: str = "mean",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create GNN model

    Args:
        model_type: Type of GNN ('gcn', 'gat', 'sage', 'residual_gcn')
        in_channels: Number of input features
        hidden_channels: Hidden dimension
        num_layers: Number of layers
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
        pooling: Pooling method
        **kwargs: Additional model-specific arguments

    Returns:
        GNN model
    """
    if model_type == "gcn":
        return GCNModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            pooling=pooling,
        )
    elif model_type == "gat":
        num_heads = kwargs.get("num_heads", 4)
        return GATModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            pooling=pooling,
        )
    elif model_type == "sage":
        aggregator = kwargs.get("aggregator", "mean")
        return GraphSAGEModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            pooling=pooling,
            aggregator=aggregator,
        )
    elif model_type == "residual_gcn":
        return ResidualGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            pooling=pooling,
        )
    else:
        raise ValueError(f"Unknown GNN model type: {model_type}")
