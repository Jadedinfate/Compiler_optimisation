#!/usr/bin/env python3
"""
model.py
PyTorch Geometric (PyG) GGNN Model for Vulnerability Prediction.
Architecture:
1. Backbone: Gated Graph Neural Network (Relational GatedGraphConv) with typed edges
   (AST, CFG, DFG/PDG) and GRU recurrence unrolling.
2. Pooling: Global Soft Attention Readout (GlobalAttention) to compute a graph-level
   representation and prevent over-smoothing.
3. Classifier: Multi-Layer Perceptron (MLP) head with Linear -> ReLU -> Dropout -> Linear
   producing binary vulnerability logits (0: benign, 1: vulnerable).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch_geometric.nn import GatedGraphConv, GlobalAttention


class RelationalGatedGraphBlock(nn.Module):
    """
    Relational Gated Graph Neural Network module.
    Applies relation-specific linear transformations for typed edges (AST, CFG, DFG)
    followed by Gated Recurrent Unit (GRU) state updates across T unrolling steps.
    """
    def __init__(self, hidden_dim: int, num_layers: int = 4, num_edge_types: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_edge_types = num_edge_types

        # Per-relation transformation matrices for typed message passing
        self.relation_weights = nn.Parameter(
            torch.Tensor(num_edge_types, hidden_dim, hidden_dim)
        )
        # Recurrent cell for state update
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        # PyG GatedGraphConv as alternative / fallback backbone
        self.pyg_ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.relation_weights)
        self.gru.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node feature tensor [num_nodes, hidden_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_type: Relational edge type tensor [num_edges]
        Returns:
            Updated node representations [num_nodes, hidden_dim]
        """
        # If no edge_type provided or empty, fallback directly to PyG GatedGraphConv
        if edge_type is None or edge_index.size(1) == 0:
            return self.pyg_ggnn(x, edge_index)

        h = x
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Clamp edge_type within valid relation bounds
        edge_type = torch.clamp(edge_type, 0, self.num_edge_types - 1)

        # Message passing unrolled over T timesteps
        for t in range(self.num_layers):
            messages = torch.zeros((num_nodes, self.hidden_dim), device=x.device, dtype=x.dtype)

            # Compute relation-specific messages for each edge type
            for r in range(self.num_edge_types):
                mask = (edge_type == r)
                if not mask.any():
                    continue

                r_src = src[mask]
                r_dst = dst[mask]

                # Message from source node under relation r: h_u @ W_r
                msg = torch.matmul(h[r_src], self.relation_weights[r])
                # Scatter-add messages into destination nodes
                messages.index_add_(0, r_dst, msg)

            # GRU update: h_v^(t) = GRU(m_v, h_v^(t-1))
            h = self.gru(messages, h)

        return h


class CPGGGNNModel(nn.Module):
    """
    End-to-End Vulnerability Prediction Model.
    Code Property Graph -> Relational GGNN -> Global Soft Attention -> MLP Classifier.
    """
    def __init__(
        self,
        in_channels: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_edge_types: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_edge_types = num_edge_types
        self.dropout_rate = dropout

        # 1. Input Node Projection (CodeBERT 768-d -> hidden_dim)
        self.node_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # 2. GGNN Backbone with Typed Relational Message Passing
        self.ggnn = RelationalGatedGraphBlock(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_edge_types=num_edge_types
        )

        # 3. Global Soft Attention Pooling
        # Soft attention gating network: computes scalar importance weight for each node
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        # Node feature transformation prior to pooling
        self.feat_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.pool = GlobalAttention(gate_nn=self.gate_nn, nn=self.feat_nn)

        # 4. MLP Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for CPGGGNNModel.
        Args:
            x: Node features [total_nodes, in_channels]
            edge_index: Graph edges [2, total_edges]
            edge_type: Edge relation types [total_edges]
            batch: Batch vector mapping nodes to graph index [total_nodes]
            return_attention: If True, returns node attention scores for explainability
        Returns:
            logits: Classification logits [batch_size, num_classes]
            (Optional) attention_weights: Soft attention weights [total_nodes, 1]
        """
        # Handle unbatched single graph inference
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 1. Project input CodeBERT embeddings
        h = self.node_proj(x)

        # 2. Relational GGNN message passing
        h = self.ggnn(h, edge_index, edge_type=edge_type)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # 3. Compute Soft Attention Weights & Pool
        attn_scores = self.gate_nn(h)
        graph_embed = self.pool(h, batch)

        # 4. MLP Classifier Head
        logits = self.classifier(graph_embed)

        if return_attention:
            return logits, attn_scores

        return logits


def get_model(
    in_channels: int = 768,
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_edge_types: int = 3,
    dropout: float = 0.3
) -> CPGGGNNModel:
    """Factory helper to construct the CPGGGNNModel with standard hyperparameters."""
    return CPGGGNNModel(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_edge_types=num_edge_types,
        num_classes=2,
        dropout=dropout
    )


if __name__ == "__main__":
    # Sanity check forward pass
    print("[*] Running sanity check on CPGGGNNModel...")
    model = get_model()
    
    num_nodes = 20
    num_edges = 40
    dummy_x = torch.randn(num_nodes, 768)
    dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges))
    dummy_edge_type = torch.randint(0, 3, (num_edges,))
    dummy_batch = torch.zeros(num_nodes, dtype=torch.long)

    logits, attn = model(
        dummy_x,
        dummy_edge_index,
        dummy_edge_type,
        dummy_batch,
        return_attention=True
    )
    print(f"[+] Output logits shape: {logits.shape}")  # [1, 2]
    print(f"[+] Soft attention weights shape: {attn.shape}")  # [20, 1]
    assert logits.shape == (1, 2), "Mismatch in logit dimensions!"
    print("[+] Model sanity check passed successfully!")
