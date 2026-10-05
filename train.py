#!/usr/bin/env python3
"""
train.py
Training Pipeline for CPG Vulnerability Prediction using GGNN & PyG.
Key Highlights:
1. Custom Focal Loss (gamma=2.0) with optional class weighting to tackle severe class imbalance.
2. AdamW optimizer with weight decay and learning rate scheduling.
3. Validation tracking: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
4. Model checkpointing to save best weights (best_model.pth).
5. Comprehensive test set evaluation at conclusion.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Optional

from src.data.cpg_dataset import CPGVulnerabilityDataset
from src.models.gnn_model import CPGGGNNModel


# ==============================================================================
# Custom Focal Loss for Class Imbalance
# ==============================================================================
class FocalLoss(nn.Module):
    """
    Multi-class / Binary Focal Loss:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy examples and focuses training on hard negative/positive samples.
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted class logits [batch_size, num_classes]
            targets: True labels [batch_size]
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        targets_col = targets.view(-1, 1)
        target_probs = probs.gather(1, targets_col).squeeze(-1)
        target_log_probs = log_probs.gather(1, targets_col).squeeze(-1)

        # Modulating factor (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - target_probs, self.gamma)
        loss = -focal_weight * target_log_probs

        if self.alpha is not None:
            alpha_tensor = self.alpha.to(logits.device)
            alpha_t = alpha_tensor.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ==============================================================================
# Metric Evaluation Helper
# ==============================================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    """Evaluates model on given DataLoader and returns metrics dictionary."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []

    for batch in loader:
        batch = batch.to(device)
        edge_type = getattr(batch, "edge_type", None)
        logits = model(batch.x, batch.edge_index, edge_type=edge_type, batch=batch.batch)
        
        y = batch.y.view(-1)
        loss = criterion(logits, y)
        total_loss += loss.item() * batch.num_graphs

        probs = F.softmax(logits, dim=-1)[:, 1]
        preds = torch.argmax(logits, dim=-1)

        all_targets.extend(y.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0.0
    rec = recall_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0.0

    try:
        # ROC-AUC requires at least one sample of each class
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    }


# ==============================================================================
# Training Engine
# ==============================================================================
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[*] Training on device: {device}")

    # 1. Load PyG Datasets
    val_samples = max(args.max_samples // 8, 30) if args.max_samples else None
    test_samples = max(args.max_samples // 8, 30) if args.max_samples else None
    train_dataset = CPGVulnerabilityDataset(root=args.data_dir, split="train", max_samples=args.max_samples)
    val_dataset = CPGVulnerabilityDataset(root=args.data_dir, split="val", max_samples=val_samples)
    test_dataset = CPGVulnerabilityDataset(root=args.data_dir, split="test", max_samples=test_samples)

    print(f"[+] Loaded splits -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Calculate class balance for Focal Loss alpha
    y_train = [int(data.y.item()) for data in train_dataset]
    n_neg = sum(1 for y in y_train if y == 0)
    n_pos = sum(1 for y in y_train if y == 1)
    total = len(y_train)

    if n_pos > 0 and n_neg > 0:
        # Inverse class frequency alpha: [weight_neg, weight_pos]
        alpha_neg = total / (2.0 * n_neg)
        alpha_pos = total / (2.0 * n_pos)
        alpha_tensor = torch.tensor([alpha_neg, alpha_pos], dtype=torch.float32)
        print(f"[*] Class weights: Benign (0)={alpha_neg:.3f}, Vulnerable (1)={alpha_pos:.3f}")
    else:
        alpha_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)

    # 2. Instantiate Model
    model = CPGGGNNModel(
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_edge_types=args.num_edge_types,
        dropout=args.dropout
    ).to(device)

    # 3. Setup Focal Loss & AdamW Optimizer
    criterion = FocalLoss(gamma=args.gamma, alpha=alpha_tensor)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # 4. Training Loop
    best_val_f1 = -1.0
    best_val_auc = -1.0
    best_epoch = 0

    print("\n" + "=" * 70)
    print("                     STARTING TRAINING LOOP")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            edge_type = getattr(batch, "edge_type", None)
            logits = model(batch.x, batch.edge_index, edge_type=edge_type, batch=batch.batch)
            y = batch.y.view(-1)

            loss = criterion(logits, y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in recurrent GGNN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

        train_loss /= max(len(train_dataset), 1)

        # Validation step
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["f1"])

        elapsed = time.time() - t_start
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"Val AUC: {val_metrics['roc_auc']:.4f}")

        # Checkpoint: Save best model weights
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": {
                    "in_channels": args.in_channels,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "num_edge_types": args.num_edge_types,
                    "dropout": args.dropout
                }
            }
            os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
            torch.save(checkpoint, args.save_path)
            print(f"    --> [SAVED] New best model saved to {args.save_path} (F1: {best_val_f1:.4f})")

    print("\n" + "=" * 70)
    print(f"[+] Training finished! Best model occurred at Epoch {best_epoch} with Val F1: {best_val_f1:.4f}")

    # 5. Final Evaluation on Test Set using Best Model
    if os.path.exists(args.save_path):
        print(f"[*] Loading best checkpoint from {args.save_path} for final test evaluation...")
        checkpoint = torch.load(args.save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device)
    print("\n" + "=" * 50)
    print("               FINAL TEST SET RESULTS")
    print("=" * 50)
    print(f"  Test Loss      : {test_metrics['loss']:.4f}")
    print(f"  Test Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  Test Precision : {test_metrics['precision']:.4f}")
    print(f"  Test Recall    : {test_metrics['recall']:.4f}")
    print(f"  Test F1-Score  : {test_metrics['f1']:.4f}")
    print(f"  Test ROC-AUC   : {test_metrics['roc_auc']:.4f}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train GGNN on CPGs for Vulnerability Prediction.")
    parser.add_argument("--data_dir", type=str, default="data", help="Dataset directory containing clean_splits/")
    parser.add_argument("--save_path", type=str, default="models/best_model.pth", help="Checkpoint save destination")
    parser.add_argument("--in_channels", type=int, default=768, help="Node embedding dim from CodeBERT")
    parser.add_argument("--hidden_dim", type=int, default=256, help="GGNN hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="GGNN unrolling timesteps")
    parser.add_argument("--num_edge_types", type=int, default=3, help="Number of edge relations (AST, CFG, DFG)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Graph mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss focusing parameter")
    parser.add_argument("--max_samples", type=int, default=None, help="Quick test sample limit (None for full)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
