#!/usr/bin/env python3
"""
infer.py
Single-file Static Vulnerability Prediction CLI.
Usage:
    python infer.py --file path/to/target.c --model best_model.pth
Steps:
1. Takes a raw .c or .cpp source file.
2. Extracts CPG (AST, CFG, DFG) via Joern CLI or fallback static analyzer.
3. Featurizes statements using CodeBERT.
4. Loads best_model.pth and executes the trained GGNN model.
5. Prints terminal prediction: [SAFE] or [VULNERABLE] with confidence %.
"""

import os
import sys
import shutil
import argparse
import tempfile
import subprocess
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional

from src.data.cpg_dataset import (
    CodeBERTNodeEmbedder,
    parse_joern_dot,
    parse_joern_json,
    build_cpg_from_code
)
from src.models.gnn_model import CPGGGNNModel


def extract_cpg_with_joern(file_path: str, work_dir: str) -> Optional[Tuple[List[str], List[Tuple[int, int]], List[int]]]:
    """
    Attempts to run real Joern CLI to parse C/C++ into a CPG and export DOT graph.
    Returns parsed (node_codes, edges, edge_types) if successful, None otherwise.
    """
    if not shutil.which("joern-parse") or not shutil.which("joern-export"):
        return None

    cpg_bin = os.path.join(work_dir, "cpg.bin")
    export_dir = os.path.join(work_dir, "export")

    try:
        # 1. Parse file into CPG binary
        parse_cmd = ["joern-parse", file_path, "--output", cpg_bin]
        res = subprocess.run(parse_cmd, capture_output=True, text=True, timeout=60)
        if res.returncode != 0:
            return None

        # 2. Export CPG into DOT representations
        export_cmd = ["joern-export", cpg_bin, "--repr", "cpg14", "--out", export_dir]
        res = subprocess.run(export_cmd, capture_output=True, text=True, timeout=60)
        if res.returncode != 0:
            return None

        # 3. Locate exported DOT file
        dot_files = [f for f in os.listdir(export_dir) if f.endswith(".dot")]
        if not dot_files:
            return None

        dot_path = os.path.join(export_dir, dot_files[0])
        return parse_joern_dot(dot_path)
    except Exception:
        return None


def get_graph_for_file(file_path: str) -> Tuple[List[str], List[Tuple[int, int]], List[int], str]:
    """
    Extracts CPG representation from C/C++ file.
    Tries Joern CLI first; gracefully falls back to deterministic AST/CFG/DFG extractor.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        source_code = f.read()

    with tempfile.TemporaryDirectory() as tmp_dir:
        joern_result = extract_cpg_with_joern(file_path, tmp_dir)
        if joern_result is not None and len(joern_result[0]) > 0:
            node_codes, edges, edge_types = joern_result
            method_used = "Joern CLI Engine"
        else:
            node_codes, edges, edge_types = build_cpg_from_code(source_code)
            method_used = "Semantic CPG Extractor (AST + CFG + DFG)"

    return node_codes, edges, edge_types, method_used


def predict_file(file_path: str, model_path: str, device_name: str, show_attention: bool = True):
    """Executes end-to-end inference for a single C/C++ source file."""
    if not os.path.exists(file_path):
        print(f"[!] Error: Target source file does not exist: {file_path}")
        sys.exit(1)

    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print("\n" + "=" * 65)
    print(f"[*] Analyzing Source File: {file_path}")
    print("=" * 65)

    # 1. Graph Extraction
    node_codes, edges, edge_types, method = get_graph_for_file(file_path)
    print(f"[*] Graph Extraction Engine : {method}")
    print(f"[*] Graph Statistics        : {len(node_codes)} Nodes | {len(edges)} Relational Edges")

    # 2. Featurization with CodeBERT
    embedder = CodeBERTNodeEmbedder(device=str(device))
    x = embedder.embed_texts(node_codes).to(device)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
        edge_type = torch.tensor(edge_types, dtype=torch.long).to(device)
    else:
        edge_index = torch.zeros((2, 1), dtype=torch.long).to(device)
        edge_type = torch.zeros((1,), dtype=torch.long).to(device)

    # 3. Model Loading
    in_channels = 768
    hidden_dim = 256
    num_layers = 4
    num_edge_types = 3
    dropout = 0.3

    if not os.path.exists(model_path):
        candidate = os.path.join("models", os.path.basename(model_path))
        if os.path.exists(candidate):
            model_path = candidate

    if os.path.exists(model_path):
        print(f"[*] Loading trained weights from : {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        cfg = checkpoint.get("config", {})
        in_channels = cfg.get("in_channels", in_channels)
        hidden_dim = cfg.get("hidden_dim", hidden_dim)
        num_layers = cfg.get("num_layers", num_layers)
        num_edge_types = cfg.get("num_edge_types", num_edge_types)
        dropout = cfg.get("dropout", dropout)

        model = CPGGGNNModel(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_edge_types=num_edge_types,
            dropout=dropout
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"[!] Warning: Model checkpoint '{model_path}' not found on disk.")
        print("[!] Initializing CPGGGNNModel architecture for zero-shot testing verification.")
        model = CPGGGNNModel(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_edge_types=num_edge_types,
            dropout=dropout
        ).to(device)

    # 4. Inference Forward Pass
    model.eval()
    with torch.no_grad():
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        logits, attn_scores = model(
            x,
            edge_index,
            edge_type=edge_type,
            batch=batch,
            return_attention=True
        )
        probs = F.softmax(logits, dim=-1).squeeze(0)
        pred_class = int(torch.argmax(probs).item())
        conf_benign = float(probs[0].item()) * 100.0
        conf_vuln = float(probs[1].item()) * 100.0

    # 5. Output Result
    print("\n" + "=" * 65)
    print("                    PREDICTION VERDICT")
    print("=" * 65)
    if pred_class == 1:
        print(f"  >>> RESULT      : \033[91m[VULNERABLE]\033[0m")
        print(f"  >>> CONFIDENCE  : \033[91m{conf_vuln:.2f}%\033[0m")
    else:
        print(f"  >>> RESULT      : \033[92m[SAFE]\033[0m")
        print(f"  >>> CONFIDENCE  : \033[92{conf_benign:.2f}%\033[0m")

    print("-" * 65)
    print(f"  Class 0 (Benign/Safe) Probability      : {conf_benign:.2f}%")
    print(f"  Class 1 (Vulnerable) Probability      : {conf_vuln:.2f}%")

    # 6. Attention / Bug Localization (Soft attention weights)
    if show_attention and len(node_codes) > 0 and attn_scores is not None:
        attn_norm = F.softmax(attn_scores.squeeze(-1), dim=0).cpu().numpy()
        top_k = min(5, len(node_codes))
        top_indices = attn_norm.argsort()[::-1][:top_k]

        print("\n" + "-" * 65)
        print("  Top Attention Statements (Most Suspicious Nodes):")
        for rank, idx in enumerate(top_indices, 1):
            score = attn_norm[idx] * 100.0
            statement = node_codes[idx].replace("\n", " ").strip()
            print(f"    [{rank}] ({score:5.2f}% weight): {statement[:60]}")

    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Static Vulnerability Prediction CLI using PyG GGNN.")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to raw C/C++ source file")
    parser.add_argument("--model", "-m", type=str, default="models/best_model.pth", help="Trained model weights (.pth)")
    parser.add_argument("--device", "-d", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--no_attention", action="store_true", help="Disable suspicious statement ranking")
    args = parser.parse_args()

    predict_file(
        file_path=args.file,
        model_path=args.model,
        device_name=args.device,
        show_attention=not args.no_attention
    )


if __name__ == "__main__":
    main()
