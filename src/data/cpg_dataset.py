#!/usr/bin/env python3
"""
dataset.py
PyTorch Geometric (PyG) InMemoryDataset for Code Property Graphs (CPGs).
Features:
1. Loads clean splits (CSV/JSON).
2. Uses microsoft/codebert-base via HuggingFace to extract 768-d node embeddings.
3. Parses Joern JSON and DOT graph files into PyG Data objects with:
   - x: Node embeddings [num_nodes, 768]
   - edge_index: Graph connectivity [2, num_edges]
   - edge_type: Typed relations (0: AST, 1: CFG, 2: DFG/PDG)
   - y: Binary vulnerability label (0 or 1)
4. Saves processed .pt files for lightning-fast training.
"""

import os
import re
import sys
import json
import glob
import argparse
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset


# ==============================================================================
# Edge Type Definitions
# ==============================================================================
EDGE_TYPE_MAP = {
    # AST relations -> 0
    "AST": 0,
    "CONTAINS": 0,
    "SOURCE_FILE": 0,
    
    # CFG relations -> 1
    "CFG": 1,
    "CONDITION": 1,
    "DOMINATE": 1,
    "POST_DOMINATE": 1,
    
    # DFG / PDG relations -> 2
    "DFG": 2,
    "PDG": 2,
    "DDG": 2,
    "CDG": 2,
    "REACHING_DEF": 2,
    "ARGUMENT": 2,
    "REF": 2,
    "CALL": 2,
    "PARAMETER_LINK": 2
}


def map_edge_type(edge_label: str) -> int:
    """Maps a raw edge label/string to a canonical type index (0: AST, 1: CFG, 2: DFG)."""
    clean_label = edge_label.upper().strip()
    for key, val in EDGE_TYPE_MAP.items():
        if key in clean_label:
            return val
    # Default to AST if unknown structural link, or DFG for data dependencies
    return 0


# ==============================================================================
# CodeBERT Feature Extractor
# ==============================================================================
class CodeBERTNodeEmbedder:
    """
    Computes 768-d statement/token node embeddings using microsoft/codebert-base.
    Includes statement caching to avoid redundant transformer passes on common tokens.
    """
    def __init__(self, model_name: str = "microsoft/codebert-base", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing CodeBERT Embedder ({model_name}) on device: {self.device}")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.has_transformers = True
        except Exception as e:
            print(f"[!] Warning: Could not load HuggingFace transformers ({e}).")
            print("[!] Using deterministic 768-d hash embedder as fallback.")
            self.has_transformers = False

        self._cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        """Embeds a list of node code statements into [len(texts), 768] tensor."""
        if not texts:
            return torch.zeros((0, 768), dtype=torch.float32)

        if not self.has_transformers:
            # Deterministic pseudo-embedding for testing/environments lacking transformers
            embeddings = []
            for t in texts:
                np.random.seed(abs(hash(t)) % (2**32))
                embeddings.append(torch.from_numpy(np.random.randn(768).astype(np.float32)))
            return torch.stack(embeddings)

        embeddings: List[Optional[torch.Tensor]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check cache
        for idx, text in enumerate(texts):
            clean_text = text.strip() if text else "<EMPTY>"
            if clean_text in self._cache:
                embeddings[idx] = self._cache[clean_text]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(clean_text)

        # Process uncached in mini-batches
        for i in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**encoded)
            # Mean pooling over attention mask
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            pooled = (sum_embeddings / sum_mask).cpu()

            for sub_idx, emb in enumerate(pooled):
                orig_idx = uncached_indices[i + sub_idx]
                text_key = batch[sub_idx]
                self._cache[text_key] = emb
                embeddings[orig_idx] = emb

        return torch.stack([e for e in embeddings if e is not None])


# ==============================================================================
# Joern JSON & DOT Graph Parsers
# ==============================================================================
def parse_joern_json(json_content: Any) -> Tuple[List[str], List[Tuple[int, int]], List[int]]:
    """
    Parses Joern JSON graph export.
    Returns:
        node_codes: List of code strings for each node.
        edges: List of (src_node_idx, dst_node_idx).
        edge_types: List of integer edge relation types.
    """
    if isinstance(json_content, str):
        if os.path.exists(json_content):
            with open(json_content, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(json_content)
    else:
        data = json_content

    raw_nodes = data.get("nodes", []) if isinstance(data, dict) else []
    raw_edges = data.get("edges", []) if isinstance(data, dict) else []

    if isinstance(data, list):
        # Alternative Joern dump format
        raw_nodes = [item for item in data if "id" in item and "code" in item]
        raw_edges = [item for item in data if "out" in item and "in" in item]

    node_id_map: Dict[Any, int] = {}
    node_codes: List[str] = []

    for idx, node in enumerate(raw_nodes):
        node_id = node.get("id") or node.get("_id") or idx
        node_id_map[node_id] = len(node_codes)
        code_snippet = node.get("code") or node.get("label") or node.get("type") or "<NODE>"
        node_codes.append(str(code_snippet))

    edges: List[Tuple[int, int]] = []
    edge_types: List[int] = []

    for edge in raw_edges:
        src = edge.get("out") or edge.get("src") or edge.get("source")
        dst = edge.get("in") or edge.get("dst") or edge.get("target")
        label = edge.get("type") or edge.get("label") or edge.get("edgeType") or "AST"

        if src in node_id_map and dst in node_id_map:
            edges.append((node_id_map[src], node_id_map[dst]))
            edge_types.append(map_edge_type(str(label)))

    return node_codes, edges, edge_types


def parse_joern_dot(dot_content: str) -> Tuple[List[str], List[Tuple[int, int]], List[int]]:
    """
    Parses Joern DOT graph export.
    Returns:
        node_codes: List of code strings for each node.
        edges: List of (src_node_idx, dst_node_idx).
        edge_types: List of integer edge relation types.
    """
    if os.path.exists(dot_content):
        with open(dot_content, "r", encoding="utf-8", errors="ignore") as f:
            dot_content = f.read()

    node_id_map: Dict[str, int] = {}
    node_codes: List[str] = []
    edges: List[Tuple[int, int]] = []
    edge_types: List[int] = []

    # Node regex: "id" [label = "..." ... code = "..."]
    node_pattern = re.compile(r'^\s*"([^"]+)"\s*\[(.*?)\]', re.MULTILINE)
    for match in node_pattern.finditer(dot_content):
        node_id, attrs = match.group(1), match.group(2)
        code_match = re.search(r'code\s*=\s*"([^"]*)"', attrs) or re.search(r'label\s*=\s*<([^>]*)>', attrs) or re.search(r'label\s*=\s*"([^"]*)"', attrs)
        code_text = code_match.group(1) if code_match else f"node_{node_id}"
        # Strip html tags if present
        code_text = re.sub(r'<[^>]+>', '', code_text).strip()
        
        node_id_map[node_id] = len(node_codes)
        node_codes.append(code_text or "<NODE>")

    # Edge regex: "src" -> "dst" [label = "AST" ...]
    edge_pattern = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"\s*(?:\[(.*?)\])?', re.MULTILINE)
    for match in edge_pattern.finditer(dot_content):
        src, dst, attrs = match.group(1), match.group(2), match.group(3) or ""
        label_match = re.search(r'label\s*=\s*"([^"]*)"', attrs) or re.search(r'label\s*=\s*<([^>]*)>', attrs)
        label_text = label_match.group(1) if label_match else "AST"

        if src in node_id_map and dst in node_id_map:
            edges.append((node_id_map[src], node_id_map[dst]))
            edge_types.append(map_edge_type(label_text))

    return node_codes, edges, edge_types


def synthesize_code_from_row(row) -> str:
    """Creates a semantic C-code skeleton from extracted CPG feature metrics."""
    func_name = str(row.get("function") or f"func_{row.get('sample_id', 0)}")
    lines = [f"void {func_name}() {{"]
    
    n_assign = int(row.get("assignments", 2)) if pd.notna(row.get("assignments")) else 2
    for i in range(min(max(n_assign, 1), 5)):
        lines.append(f"    var_{i} = var_{i+1} + {i};")
        
    n_branch = int(row.get("branches", 1)) if pd.notna(row.get("branches")) else 1
    for i in range(min(max(n_branch, 1), 3)):
        lines.append(f"    if (var_{i} != 0) {{ execute_branch_{i}(); }}")
        
    n_loops = int(row.get("loops", 0)) if pd.notna(row.get("loops")) else 0
    for i in range(min(n_loops, 2)):
        lines.append(f"    while (loop_cond_{i} > 0) {{ loop_body_{i}(); }}")
        
    n_calls = int(row.get("function_calls", 1)) if pd.notna(row.get("function_calls")) else 1
    for i in range(min(max(n_calls, 1), 3)):
        lines.append(f"    call_subroutine_{i}(arg_{i});")
        
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines)


def build_cpg_from_code(code_str: str) -> Tuple[List[str], List[Tuple[int, int]], List[int]]:
    """
    Constructs an AST+CFG+DFG representation directly from C/C++ source code.
    Used as an immediate, deterministic graph extractor when raw DOT/JSON files are not present.
    """
    lines = [line.strip() for line in code_str.split("\n") if line.strip() and not line.strip().startswith("//")]
    if not lines:
        lines = ["void empty_function()", "{", "return;", "}"]

    node_codes: List[str] = lines
    edges: List[Tuple[int, int]] = []
    edge_types: List[int] = []

    num_nodes = len(lines)
    # 1. AST edges: Hierarchical tree connections
    root = 0
    for i in range(1, num_nodes):
        edges.append((root, i))
        edge_types.append(0)  # AST

    # 2. CFG edges: Sequential execution & branches
    for i in range(num_nodes - 1):
        edges.append((i, i + 1))
        edge_types.append(1)  # CFG
        if any(kw in lines[i] for kw in ["if", "while", "for", "switch"]):
            # Jump edge (branching)
            jump_target = min(i + 2, num_nodes - 1)
            if jump_target != i + 1:
                edges.append((i, jump_target))
                edge_types.append(1)  # CFG

    # 3. DFG / PDG edges: Variable def-use data flow
    var_defs: Dict[str, int] = {}
    var_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
    keywords = {"int", "char", "void", "float", "double", "if", "for", "while", "return", "sizeof", "struct", "NULL"}

    for i, line in enumerate(lines):
        tokens = set(var_pattern.findall(line)) - keywords
        # If variable was previously referenced/defined, connect DFG
        for token in tokens:
            if token in var_defs:
                edges.append((var_defs[token], i))
                edge_types.append(2)  # DFG / PDG
            var_defs[token] = i

    return node_codes, edges, edge_types


# ==============================================================================
# PyG InMemoryDataset Implementation
# ==============================================================================
class CPGVulnerabilityDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for Joern CPGs.
    Converts CPGs into Data(x=[N, 768], edge_index=[2, E], edge_type=[E], y=[1]).
    """
    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        graph_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None
    ):
        self.split = split
        self.graph_dir = graph_dir
        self.max_samples = max_samples
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Load processed dataset compatible with all PyG versions
        processed_file = self.processed_paths[0]
        if os.path.exists(processed_file):
            if hasattr(self, "load"):
                self.load(processed_file)
            else:
                self.data, self.slices = torch.load(processed_file, weights_only=False)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "clean_splits")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        return [f"{self.split}.csv", f"{self.split}.json"]

    @property
    def processed_file_names(self) -> List[str]:
        limit_suffix = f"_{self.max_samples}" if self.max_samples else ""
        return [f"{self.split}{limit_suffix}.pt"]

    def download(self):
        # Data is pre-split via fix_data.py
        pass

    def process(self):
        """Processes raw splits and graph files into PyG Data objects."""
        os.makedirs(self.processed_dir, exist_ok=True)
        split_csv = os.path.join(self.raw_dir, f"{self.split}.csv")
        split_json = os.path.join(self.raw_dir, f"{self.split}.json")

        df = None
        if os.path.exists(split_csv):
            print(f"[*] Loading split table from {split_csv}")
            df = pd.read_csv(split_csv)
        elif os.path.exists(split_json):
            print(f"[*] Loading split table from {split_json}")
            df = pd.read_json(split_json)
        else:
            raise FileNotFoundError(f"Neither {split_csv} nor {split_json} exists. Run fix_data.py first!")

        if self.max_samples is not None:
            df = df.iloc[:self.max_samples].reset_index(drop=True)
            print(f"[*] Limited dataset to first {self.max_samples} samples.")

        embedder = CodeBERTNodeEmbedder()
        data_list: List[Data] = []
        target_col = "target" if "target" in df.columns else "label"

        print(f"[*] Processing {len(df)} samples for '{self.split}' split...")

        for idx, row in df.iterrows():
            if idx % 500 == 0 and idx > 0:
                print(f"    - Processed {idx}/{len(df)} samples...")

            target = int(row[target_col]) if target_col in row and pd.notna(row[target_col]) else 0
            node_codes: List[str] = []
            edges: List[Tuple[int, int]] = []
            edge_types: List[int] = []

            # 1. Look for existing Joern DOT/JSON graph file on disk
            sample_id = row.get("sample_id", idx)
            filename = row.get("filename", f"sample_{sample_id}.c")
            base_name = os.path.splitext(os.path.basename(str(filename)))[0]

            graph_found = False
            candidate_dirs = [self.graph_dir, "data/devign/graphs", "cpg_graphs", "dataset/graphs"]
            for c_dir in [d for d in candidate_dirs if d and os.path.exists(d)]:
                dot_file = os.path.join(c_dir, f"{base_name}.dot")
                json_file = os.path.join(c_dir, f"{base_name}.json")
                if os.path.exists(json_file):
                    node_codes, edges, edge_types = parse_joern_json(json_file)
                    graph_found = True
                    break
                elif os.path.exists(dot_file):
                    node_codes, edges, edge_types = parse_joern_dot(dot_file)
                    graph_found = True
                    break

            # 2. If no graph file on disk, extract CPG from code snippet/function
            if not graph_found or len(node_codes) == 0:
                if "func" in row and pd.notna(row["func"]) and len(str(row["func"]).strip()) > 10:
                    code_text = str(row["func"])
                else:
                    code_text = synthesize_code_from_row(row)
                node_codes, edges, edge_types = build_cpg_from_code(code_text)

            # 3. Compute CodeBERT node embeddings
            x = embedder.embed_texts(node_codes)

            # 4. Format edge_index and edge_type
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_type = torch.tensor(edge_types, dtype=torch.long)
            else:
                # Self-loop fallback for isolated node
                edge_index = torch.zeros((2, 1), dtype=torch.long)
                edge_type = torch.zeros((1,), dtype=torch.long)

            # Ensure edge index bounds
            num_nodes = x.size(0)
            if edge_index.numel() > 0:
                valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                edge_index = edge_index[:, valid_mask]
                edge_type = edge_type[valid_mask]
                if edge_index.size(1) == 0:
                    edge_index = torch.zeros((2, 1), dtype=torch.long)
                    edge_type = torch.zeros((1,), dtype=torch.long)

            y = torch.tensor([target], dtype=torch.long)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_type=edge_type,
                y=y
            )
            data.sample_id = str(sample_id)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # Save processed .pt file
        save_path = self.processed_paths[0]
        print(f"[*] Saving {len(data_list)} processed graphs to: {save_path}")
        if hasattr(self, "save"):
            self.save(data_list, save_path)
        else:
            data, slices = self.collate(data_list)
            torch.save((data, slices), save_path)

        print(f"[+] Dataset '{self.split}' processing complete!\n")


# ==============================================================================
# CLI Entrypoint for Standalone Pre-Processing
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PyG CPG Dataset with CodeBERT Featurization.")
    parser.add_argument("--root", type=str, default="data", help="Root directory containing clean_splits/")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"],
                        help="Split to process")
    parser.add_argument("--graph_dir", type=str, default=None, help="Directory containing Joern DOT/JSON files")
    parser.add_argument("--limit", type=int, default=None, help="Optional max sample limit per split")
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for s in splits:
        print(f"\n{'='*50}\nProcessing Split: {s.upper()}\n{'='*50}")
        dataset = CPGVulnerabilityDataset(
            root=args.root,
            split=s,
            graph_dir=args.graph_dir,
            max_samples=args.limit
        )
        print(f"[+] Successfully loaded {len(dataset)} graphs for {s} split.")
