"""Training script for GNN/Hybrid models using PyTorch"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data import ForestfireDataset, ForestfireGraphDataset
from src.models import create_gnn_model, create_hybrid_model
from src.utils import (
    setup_logger,
    calculate_class_weights,
    analyze_class_imbalance,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train GNN/Hybrid forestfire model")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs_gnn",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["gnn", "hybrid"],
        default="hybrid",
        help="Model type to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID",
    )

    return parser.parse_args()


def setup_device(gpu_id: str):
    """Setup device (CPU/GPU)"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute evaluation metrics"""
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
    }

    return metrics


def train_epoch_gnn(
    model,
    graph_dataset,
    data_dict,
    optimizer,
    criterion,
    device,
    logger,
):
    """Train one epoch for GNN model"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Create graph batch
    batch = graph_dataset.create_graph_batch(
        spatial_data={k: v for k, v in data_dict.items() if k in graph_dataset.spatial_features},
        climate_data=data_dict["climate"],
        labels=data_dict["labels"],
    )

    # Move to device
    batch = batch.to(device)

    # Forward pass
    optimizer.zero_grad()
    outputs = model(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        batch=batch.batch,
    )

    # Compute loss (pool labels to match batch size)
    # Since each graph has multiple nodes with same label, we take first label per graph
    num_graphs = data_dict["labels"].shape[0]
    labels_pooled = torch.tensor(data_dict["labels"], dtype=torch.float, device=device).view(-1, 1)

    loss = criterion(outputs, labels_pooled)

    # Backward pass
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    all_preds.extend(outputs.detach().cpu().numpy())
    all_labels.extend(labels_pooled.detach().cpu().numpy())

    avg_loss = total_loss
    return avg_loss, np.array(all_preds), np.array(all_labels)


def train_epoch_hybrid(
    model,
    graph_dataset,
    data_dict,
    optimizer,
    criterion,
    device,
    batch_size,
    logger,
):
    """Train one epoch for Hybrid model"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    num_samples = data_dict["labels"].shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)

        # Prepare batch data
        batch_data = {}
        for key in data_dict:
            batch_data[key] = data_dict[key][start_idx:end_idx]

        # Prepare spatial inputs
        spatial_inputs = {}
        for feature in graph_dataset.spatial_features:
            if feature in batch_data:
                spatial_inputs[feature] = torch.tensor(
                    batch_data[feature], dtype=torch.float, device=device
                )

        # Climate input
        climate_input = torch.tensor(
            batch_data["climate"].squeeze(-1), dtype=torch.float, device=device
        )

        # Labels
        labels = torch.tensor(
            batch_data["labels"], dtype=torch.float, device=device
        ).view(-1, 1)

        # Build edge index (shared across batch)
        edge_index = graph_dataset.graph_builder.build_edge_index().to(device)
        edge_attr = graph_dataset.graph_builder.build_edge_attr(edge_index).to(device) if graph_dataset.include_edge_attr else None

        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            spatial_inputs=spatial_inputs,
            climate_input=climate_input,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / num_batches
    return avg_loss, np.array(all_preds), np.array(all_labels)


def evaluate(
    model,
    graph_dataset,
    data_dict,
    criterion,
    device,
    model_type,
    batch_size,
    logger,
):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        if model_type == "gnn":
            # GNN evaluation
            batch = graph_dataset.create_graph_batch(
                spatial_data={k: v for k, v in data_dict.items() if k in graph_dataset.spatial_features},
                climate_data=data_dict["climate"],
                labels=data_dict["labels"],
            )
            batch = batch.to(device)

            outputs = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
            )

            labels = torch.tensor(data_dict["labels"], dtype=torch.float, device=device).view(-1, 1)
            loss = criterion(outputs, labels)

            all_preds = outputs.cpu().numpy()
            all_labels = labels.cpu().numpy()
            total_loss = loss.item()

        else:  # hybrid
            # Hybrid evaluation (batch-wise)
            num_samples = data_dict["labels"].shape[0]
            num_batches = (num_samples + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)

                batch_data = {}
                for key in data_dict:
                    batch_data[key] = data_dict[key][start_idx:end_idx]

                spatial_inputs = {}
                for feature in graph_dataset.spatial_features:
                    if feature in batch_data:
                        spatial_inputs[feature] = torch.tensor(
                            batch_data[feature], dtype=torch.float, device=device
                        )

                climate_input = torch.tensor(
                    batch_data["climate"].squeeze(-1), dtype=torch.float, device=device
                )

                labels = torch.tensor(
                    batch_data["labels"], dtype=torch.float, device=device
                ).view(-1, 1)

                edge_index = graph_dataset.graph_builder.build_edge_index().to(device)
                edge_attr = graph_dataset.graph_builder.build_edge_attr(edge_index).to(device) if graph_dataset.include_edge_attr else None

                outputs = model(
                    spatial_inputs=spatial_inputs,
                    climate_input=climate_input,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                )

                loss = criterion(outputs, labels)
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            total_loss /= num_batches

    return total_loss, np.array(all_preds), np.array(all_labels)


def main():
    """Main training function"""
    args = parse_args()

    # Setup device
    device = setup_device(args.gpu)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(
        "gnn_train",
        log_file=str(output_dir / "train.log"),
        log_level="INFO",
    )

    logger.info("=" * 80)
    logger.info(f"Starting {args.model_type.upper()} model training")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Override config
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr

    # Log configuration
    logger.info("Training configuration:")
    logger.info(f"  Model type: {args.model_type}")
    logger.info(f"  Epochs: {config['training']['epochs']}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = ForestfireDataset(
        data_dir=args.data_dir,
        spatial_features=config["data"]["spatial_features"],
        climate_features=config["data"]["climate_features"],
        image_size=config["data"]["image_size"],
    )

    # Prepare data
    train_data, val_data, test_data = dataset.prepare_data(
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        stratify=True,
        random_seed=config["random_seed"],
    )

    # Analyze class imbalance
    logger.info("Analyzing class imbalance...")
    class_stats = analyze_class_imbalance(train_data["labels"])

    # Calculate class weights
    class_weights = calculate_class_weights(train_data["labels"], method="balanced")
    logger.info(f"Class weights: {class_weights}")

    # Create graph dataset
    logger.info("Creating graph dataset...")
    graph_dataset = ForestfireGraphDataset(
        spatial_features=config["data"]["spatial_features"],
        climate_features=config["data"]["climate_features"],
        grid_height=config["data"]["grid_height"],
        grid_width=config["data"]["grid_width"],
        connectivity=config["model"]["gnn"]["connectivity"],
        include_edge_attr=config["model"]["gnn"]["include_edge_attr"],
    )

    # Create model
    logger.info("Creating model...")
    if args.model_type == "gnn":
        # Calculate input features
        # Each node has: spatial_features_channels + climate_features
        # Assuming 1 channel per spatial feature for simplicity
        in_channels = len(config["data"]["spatial_features"]) + len(config["data"]["climate_features"])

        model = create_gnn_model(
            model_type=config["model"]["gnn"]["type"],
            in_channels=in_channels,
            hidden_channels=config["model"]["gnn"]["hidden_channels"],
            num_layers=config["model"]["gnn"]["num_layers"],
            dropout=config["model"]["gnn"]["dropout"],
            pooling=config["model"]["gnn"]["pooling"],
            num_heads=config["model"]["gnn"].get("num_heads", 4),
            aggregator=config["model"]["gnn"].get("aggregator", "mean"),
        )
    else:  # hybrid
        model = create_hybrid_model(config)

    model = model.to(device)
    logger.info(f"Model created: {model.__class__.__name__}")

    # Loss function with class weights
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=config["training"]["lr_scheduler"]["patience"],
        factor=config["training"]["lr_scheduler"]["factor"],
        min_lr=config["training"]["lr_scheduler"]["min_lr"],
    )

    # Training loop
    logger.info("Starting training...")
    best_val_auc = 0
    patience_counter = 0
    patience = config["training"]["early_stopping"]["patience"]

    for epoch in range(config["training"]["epochs"]):
        # Train
        if args.model_type == "gnn":
            train_loss, train_preds, train_labels = train_epoch_gnn(
                model, graph_dataset, train_data, optimizer, criterion, device, logger
            )
        else:
            train_loss, train_preds, train_labels = train_epoch_hybrid(
                model, graph_dataset, train_data, optimizer, criterion, device,
                config["training"]["batch_size"], logger
            )

        train_metrics = compute_metrics(train_labels.flatten(), train_preds.flatten())

        # Validate
        val_loss, val_preds, val_labels = evaluate(
            model, graph_dataset, val_data, criterion, device, args.model_type,
            config["training"]["batch_size"], logger
        )
        val_metrics = compute_metrics(val_labels.flatten(), val_preds.flatten())

        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{config['training']['epochs']} - "
            f"Train Loss: {train_loss:.4f}, Train AUC: {train_metrics['auc_roc']:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val AUC: {val_metrics['auc_roc']:.4f}"
        )

        # Learning rate scheduling
        scheduler.step(val_metrics['auc_roc'])

        # Save best model
        if val_metrics['auc_roc'] > best_val_auc:
            best_val_auc = val_metrics['auc_roc']
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            logger.info(f"Best model saved! Val AUC: {best_val_auc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    logger.info("Evaluating on test set...")
    model.load_state_dict(torch.load(output_dir / "best_model.pth"))

    test_loss, test_preds, test_labels = evaluate(
        model, graph_dataset, test_data, criterion, device, args.model_type,
        config["training"]["batch_size"], logger
    )
    test_metrics = compute_metrics(test_labels.flatten(), test_preds.flatten())

    logger.info("=" * 80)
    logger.info("TEST SET RESULTS:")
    logger.info(f"  Loss:      {test_loss:.4f}")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {test_metrics['f1_score']:.4f}")
    logger.info(f"  AUC-ROC:   {test_metrics['auc_roc']:.4f}")
    logger.info("=" * 80)

    logger.info(f"Training complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
