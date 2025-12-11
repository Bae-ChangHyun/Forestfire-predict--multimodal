"""Graph data construction for GNN models"""

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import logging

from ..utils.constants import DataConstants

logger = logging.getLogger(__name__)


class GridGraphBuilder:
    """Build graph structure from spatial grid data"""

    def __init__(
        self,
        grid_height: int = DataConstants.GRID_HEIGHT,
        grid_width: int = DataConstants.GRID_WIDTH,
        connectivity: str = "4-neighbor",
    ):
        """
        Initialize grid graph builder

        Args:
            grid_height: Height of the grid
            grid_width: Width of the grid
            connectivity: Type of connectivity ('4-neighbor' or '8-neighbor')
        """
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.connectivity = connectivity
        self.num_nodes = grid_height * grid_width

        logger.info(f"GridGraphBuilder initialized: {grid_height}x{grid_width} grid")
        logger.info(f"Total nodes: {self.num_nodes}, Connectivity: {connectivity}")

    def grid_to_node_id(self, row: int, col: int) -> int:
        """
        Convert grid coordinates to node ID

        Args:
            row: Row index
            col: Column index

        Returns:
            Node ID
        """
        return row * self.grid_width + col

    def node_id_to_grid(self, node_id: int) -> Tuple[int, int]:
        """
        Convert node ID to grid coordinates

        Args:
            node_id: Node ID

        Returns:
            Tuple of (row, col)
        """
        row = node_id // self.grid_width
        col = node_id % self.grid_width
        return row, col

    def build_edge_index(self) -> torch.Tensor:
        """
        Build edge index for grid graph

        Returns:
            Edge index tensor of shape [2, num_edges]
        """
        edges = []

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                node_id = self.grid_to_node_id(row, col)

                if self.connectivity == "4-neighbor":
                    # Right, Down, Left, Up
                    neighbors = [
                        (row, col + 1),  # Right
                        (row + 1, col),  # Down
                        (row, col - 1),  # Left
                        (row - 1, col),  # Up
                    ]
                elif self.connectivity == "8-neighbor":
                    # 8 directions
                    neighbors = [
                        (row, col + 1),      # Right
                        (row + 1, col),      # Down
                        (row, col - 1),      # Left
                        (row - 1, col),      # Up
                        (row + 1, col + 1),  # Down-Right
                        (row + 1, col - 1),  # Down-Left
                        (row - 1, col + 1),  # Up-Right
                        (row - 1, col - 1),  # Up-Left
                    ]
                else:
                    raise ValueError(f"Unknown connectivity: {self.connectivity}")

                # Add valid neighbors
                for n_row, n_col in neighbors:
                    if 0 <= n_row < self.grid_height and 0 <= n_col < self.grid_width:
                        neighbor_id = self.grid_to_node_id(n_row, n_col)
                        edges.append([node_id, neighbor_id])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        logger.info(f"Built edge index: {edge_index.shape[1]} edges")

        return edge_index

    def build_edge_attr(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Build edge attributes (distances)

        Args:
            edge_index: Edge index tensor

        Returns:
            Edge attribute tensor of shape [num_edges, 1]
        """
        num_edges = edge_index.shape[1]
        edge_attr = []

        for i in range(num_edges):
            src_node = edge_index[0, i].item()
            dst_node = edge_index[1, i].item()

            src_row, src_col = self.node_id_to_grid(src_node)
            dst_row, dst_col = self.node_id_to_grid(dst_node)

            # Euclidean distance
            distance = np.sqrt((src_row - dst_row) ** 2 + (src_col - dst_col) ** 2)
            edge_attr.append([distance])

        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return edge_attr


class ForestfireGraphDataset:
    """Dataset class for graph-based forestfire prediction"""

    def __init__(
        self,
        spatial_features: List[str],
        climate_features: List[str],
        grid_height: int = DataConstants.GRID_HEIGHT,
        grid_width: int = DataConstants.GRID_WIDTH,
        connectivity: str = "4-neighbor",
        include_edge_attr: bool = True,
    ):
        """
        Initialize graph dataset

        Args:
            spatial_features: List of spatial feature names
            climate_features: List of climate feature names
            grid_height: Grid height
            grid_width: Grid width
            connectivity: Graph connectivity
            include_edge_attr: Whether to include edge attributes
        """
        self.spatial_features = spatial_features
        self.climate_features = climate_features
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.connectivity = connectivity
        self.include_edge_attr = include_edge_attr

        # Build graph structure
        self.graph_builder = GridGraphBuilder(grid_height, grid_width, connectivity)
        self.edge_index = self.graph_builder.build_edge_index()

        if include_edge_attr:
            self.edge_attr = self.graph_builder.build_edge_attr(self.edge_index)
        else:
            self.edge_attr = None

        logger.info("ForestfireGraphDataset initialized")

    def prepare_node_features(
        self,
        spatial_data: Dict[str, np.ndarray],
        climate_data: np.ndarray,
    ) -> torch.Tensor:
        """
        Prepare node features from spatial and climate data

        Args:
            spatial_data: Dictionary with spatial features (batch, height, width, channels)
            climate_data: Climate features (batch, features, 1)

        Returns:
            Node features tensor of shape [batch, num_nodes, num_features]
        """
        batch_size = climate_data.shape[0]
        num_nodes = self.grid_height * self.grid_width

        # Flatten spatial features
        spatial_features_list = []
        for feature_name in self.spatial_features:
            if feature_name in spatial_data:
                # Shape: (batch, height, width, channels)
                feature_data = spatial_data[feature_name]

                # Flatten spatial dimensions
                # Shape: (batch, height * width, channels)
                feature_flat = feature_data.reshape(batch_size, num_nodes, -1)
                spatial_features_list.append(feature_flat)

        # Concatenate spatial features
        if spatial_features_list:
            spatial_features = np.concatenate(spatial_features_list, axis=-1)
        else:
            spatial_features = np.zeros((batch_size, num_nodes, 0))

        # Repeat climate features for each node
        # Shape: (batch, features, 1) -> (batch, features) -> (batch, 1, features)
        climate_features = climate_data.squeeze(-1)
        climate_features = np.expand_dims(climate_features, axis=1)

        # Broadcast to all nodes
        # Shape: (batch, num_nodes, features)
        climate_features = np.repeat(climate_features, num_nodes, axis=1)

        # Concatenate spatial and climate features
        node_features = np.concatenate([spatial_features, climate_features], axis=-1)

        # Convert to tensor
        node_features = torch.tensor(node_features, dtype=torch.float)

        logger.info(f"Node features prepared: shape {node_features.shape}")

        return node_features

    def create_graph_batch(
        self,
        spatial_data: Dict[str, np.ndarray],
        climate_data: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Batch:
        """
        Create batch of graph data

        Args:
            spatial_data: Dictionary with spatial features
            climate_data: Climate features
            labels: Optional labels

        Returns:
            PyTorch Geometric Batch object
        """
        # Prepare node features
        node_features = self.prepare_node_features(spatial_data, climate_data)
        batch_size = node_features.shape[0]

        # Create individual graphs
        graph_list = []

        for i in range(batch_size):
            # Node features for this sample
            x = node_features[i]  # Shape: [num_nodes, num_features]

            # Create graph data
            data = Data(
                x=x,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr if self.include_edge_attr else None,
            )

            # Add label if available
            if labels is not None:
                # Repeat label for all nodes
                y = torch.full((x.shape[0],), labels[i], dtype=torch.float)
                data.y = y

            graph_list.append(data)

        # Create batch
        batch = Batch.from_data_list(graph_list)

        return batch

    def graph_to_grid(
        self,
        node_features: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Convert node features back to grid format

        Args:
            node_features: Node features [batch * num_nodes, features]
            batch_size: Batch size

        Returns:
            Grid features [batch, height, width, features]
        """
        num_nodes = self.grid_height * self.grid_width
        num_features = node_features.shape[-1]

        # Reshape to batch
        node_features = node_features.view(batch_size, num_nodes, num_features)

        # Reshape to grid
        grid_features = node_features.view(
            batch_size, self.grid_height, self.grid_width, num_features
        )

        return grid_features


def create_graph_from_sample(
    spatial_data: Dict[str, np.ndarray],
    climate_data: np.ndarray,
    label: Optional[float] = None,
    grid_height: int = DataConstants.GRID_HEIGHT,
    grid_width: int = DataConstants.GRID_WIDTH,
) -> Data:
    """
    Create a single graph from a data sample

    Args:
        spatial_data: Dictionary with spatial features (height, width, channels)
        climate_data: Climate features (features, 1)
        label: Optional label
        grid_height: Grid height
        grid_width: Grid width

    Returns:
        PyTorch Geometric Data object
    """
    # Build graph structure
    graph_builder = GridGraphBuilder(grid_height, grid_width)
    edge_index = graph_builder.build_edge_index()
    edge_attr = graph_builder.build_edge_attr(edge_index)

    # Prepare node features
    num_nodes = grid_height * grid_width
    node_features_list = []

    # Flatten spatial features
    for feature_name, feature_data in spatial_data.items():
        # Shape: (height, width, channels) -> (num_nodes, channels)
        feature_flat = feature_data.reshape(num_nodes, -1)
        node_features_list.append(feature_flat)

    # Concatenate spatial features
    if node_features_list:
        spatial_features = np.concatenate(node_features_list, axis=-1)
    else:
        spatial_features = np.zeros((num_nodes, 0))

    # Repeat climate features for each node
    climate_features = climate_data.squeeze(-1)  # Shape: (features,)
    climate_features = np.tile(climate_features, (num_nodes, 1))  # Shape: (num_nodes, features)

    # Concatenate all features
    node_features = np.concatenate([spatial_features, climate_features], axis=-1)
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Create graph data
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    # Add label if available
    if label is not None:
        y = torch.full((num_nodes,), label, dtype=torch.float)
        data.y = y

    return data
