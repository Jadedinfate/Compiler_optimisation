# 🛡️ CPG Compiler Vulnerability Prediction Pipeline

An end-to-end Machine Learning and Graph Neural Network (GNN) framework for **Static Code Vulnerability Prediction** and **Statement-Level Bug Localization** using Code Property Graphs (CPGs) and CodeBERT embeddings.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.3%2B-orange.svg)](https://pyg.org/)
[![HuggingFace](https://img.shields.io/badge/CodeBERT-microsoft%2Fcodebert--base-yellow.svg)](https://huggingface.co/microsoft/codebert-base)
[![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green.svg)](https://developer.nvidia.com/cuda-zone)

---

## 📌 Overview

This project detects software vulnerabilities (e.g., buffer overflows, memory corruptions, unsafe pointer dereferences) in C/C++ source code by converting functions into **Code Property Graphs (CPGs)**—unifying Abstract Syntax Trees (AST), Control Flow Graphs (CFG), and Data Flow Graphs (DFG)—and learning relational graph representations.

The repository provides two production-ready predictive pipelines:
1. **Relational GGNN Pipeline (`train.py` & `infer.py`)**:
   - Extracts semantic statements directly from C functions.
   - Featurizes each statement into a 768-dimensional vector using `microsoft/codebert-base`.
   - Propagates relational messages across AST, CFG, and DFG edges with **Gated Recurrent Units (GRU)** and in-degree normalization.
   - Uses **Global Soft-Attention Readout** to compute function-level risk and identify the exact lines of code most likely to harbor vulnerabilities.
2. **Tabular IR Pipeline (`main.py`)**:
   - Supervised ensemble classifiers (Random Forest & Gradient Boosting) trained on 12 static compiler IR metrics (`ast_nodes`, `cfg_nodes`, `branches`, `loops`, `pointer_ops`, etc.).

---

## 🏗️ Architecture

```
Raw C/C++ Source File (.c)
            │
            ▼
┌──────────────────────────────────────────────┐
│  Multi-Relational CPG Graph Extractor        │
│  - AST Edges (Type 0): Hierarchical Root-Body│
│  - CFG Edges (Type 1): Sequential & Jumps    │
│  - DFG Edges (Type 2): Def-Use Data Flow     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  CodeBERT Node Featurizer (768-d per node)   │
│  - Batched statement transformer passes      │
│  - Dynamic token pooling & cache             │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Relational Gated Graph Neural Network       │
│  - Relation-specific projection matrices W_r │
│  - Degree-normalized scatter message-passing │
│  - Recurrent GRU hidden state updates        │
│  - Residual connection: h = h_init + h_gnn   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Global Soft-Attention Pooling & MLP Head    │
│  - Computes statement risk weights           │
│  - LayerNorm + Dropout regularization        │
└──────────────────────┬───────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
Prediction Verdict:        Explainability Heatmap:
[VULNERABLE] (54.45%)      Top suspicious lines:
                           - strcpy(buffer, user_input);
```

---

## 📁 Repository Structure

```
Compiler_optimisation/
├── data/
│   ├── clean_splits/           # Tabular dataset splits (CSV & JSON)
│   ├── devign/                 # Authentic Devign C/C++ dataset (JSONL)
│   └── processed/              # Preprocessed PyTorch Geometric (.pt) graphs
├── examples/
│   ├── test_safe.c             # Sample benign arithmetic function
│   └── test_vulnerable.c       # Sample vulnerable buffer overflow function
├── models/
│   ├── best_model.pth          # Best trained GGNN checkpoint (weights & config)
│   ├── tabular_model.joblib    # Trained Random Forest / Gradient Boosting model
│   └── scaler.joblib           # Fitted StandardScaler for IR tabular metrics
├── scripts/
│   ├── download_devign.py      # Automated dataset downloader from Hugging Face
│   ├── clean_source.py         # C/C++ source code preprocessor
│   └── fix_data.py             # Stratified dataset partitioner
├── src/
│   ├── config/config.py        # Centralized configurations & paths
│   ├── data/
│   │   ├── cpg_dataset.py      # PyG InMemoryDataset & CPG extractor
│   │   ├── data_loader.py      # Data ingestion & schema validator
│   │   └── preprocess.py       # Tabular scaling and feature separation
│   └── models/
│       ├── gnn_model.py        # Relational GGNN architecture & Attention Pooling
│       └── tabular_model.py    # CompilerVulnerabilityClassifier (Tree Ensembles)
├── infer.py                    # CLI tool for single-file static vulnerability inference
├── main.py                     # CLI pipeline for tabular ML training & evaluation
├── train.py                    # Training script for PyG GGNN model
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quickstart & Installation

### 1. Environment Setup

Ensure you have Python 3.10+ installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/Jadedinfate/Compiler_optimisation.git
cd Compiler_optimisation

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Download the Authentic Devign Dataset

Download the complete benchmark Devign C/C++ dataset (~27,318 real functions from QEMU and FFmpeg):

```bash
python scripts/download_devign.py
```

This will download:
- `data/devign/train_devign.json` (21,854 functions | 45.8% vulnerable)
- `data/devign/valid_devign.json` (2,732 functions | 43.4% vulnerable)
- `data/devign/test_devign.json` (2,732 functions | 45.9% vulnerable)

---

## 🧠 Model Training

### Train the GNN Pipeline (`train.py`)

Train the Relational GGNN model on CUDA with CodeBERT featurization, in-degree normalized message passing, and focal loss:

```bash
# Quick training on 1,200 curated Devign graphs for 15 epochs:
python train.py --max_samples 1200 --epochs 15 --lr 5e-4 --gamma 1.0 --device cuda

# Train on a larger slice (3,000 samples) with custom batch size:
python train.py --max_samples 3000 --epochs 20 --batch_size 32 --lr 5e-4 --device cuda

# Train on the entire 21,854-sample dataset:
python train.py --max_samples 0 --epochs 25 --device cuda
```

**Key CLI Arguments:**
| Argument | Default | Description |
|---|---|---|
| `--max_samples` | `1200` | Sample limit (`0` or `None` for full dataset) |
| `--epochs` | `15` | Number of training epochs |
| `--lr` | `5e-4` | AdamW learning rate |
| `--gamma` | `1.0` | Focal Loss focusing factor (`0.0` for standard CE) |
| `--device` | `cuda` | Target compute device (`cuda` or `cpu`) |
| `--save_path` | `models/best_model.pth` | Checkpoint destination |

---

### Train the Tabular ML Pipeline (`main.py`)

Train Random Forest or Gradient Boosting models on the compiler IR tabular metrics across all splits:

```bash
# Train Gradient Boosting classifier:
python main.py --model_type gradient_boosting --n_estimators 100 --max_depth 6

# Train Random Forest classifier:
python main.py --model_type random_forest --n_estimators 150 --max_depth 16
```

---

## 🔍 Single-File Inference & Bug Localization (`infer.py`)

Analyze any raw C or C++ file to receive a vulnerability verdict and a statement-level attention breakdown:

```bash
# Analyze a vulnerable program:
python infer.py --file examples/test_vulnerable.c --model models/best_model.pth

# Analyze a safe program:
python infer.py --file examples/test_safe.c --model models/best_model.pth
```

### Sample Output: Vulnerable Detection

```text
=================================================================
[*] Analyzing Source File: examples/test_vulnerable.c
=================================================================
[*] Graph Extraction Engine : Semantic CPG Extractor (AST + CFG + DFG)
[*] Graph Statistics        : 14 Nodes | 37 Relational Edges
[*] Initializing CodeBERT Embedder (microsoft/codebert-base) on device: cuda
[*] Loading trained weights from : models/best_model.pth

=================================================================
                    PREDICTION VERDICT
=================================================================
  >>> RESULT      : [VULNERABLE]
  >>> CONFIDENCE  : 54.45%
-----------------------------------------------------------------
  Class 0 (Benign/Safe) Probability      : 45.55%
  Class 1 (Vulnerable) Probability      : 54.45%

-----------------------------------------------------------------
  Top Attention Statements (Most Suspicious Nodes):
    [1] (12.64% weight): vulnerable_function(argv[1]);
    [2] (12.19% weight): #include <stdlib.h>
    [3] (11.06% weight): strcpy(buffer, user_input);
    [4] ( 9.71% weight): printf("Copied buffer: %s\n", buffer);
    [5] ( 8.25% weight): if (argc > 1) {
=================================================================
```

---

## 🧪 Testing

Run automated tests to verify the pipeline components:

```bash
python -m unittest discover tests/
```

---

## 📊 Key Highlights & Recent Improvements

- **Minified Statement Parsing**: Resolved a critical issue where single-line C code from Devign was previously parsed as 1-node graphs. The parser now accurately constructs rich, multi-node CPGs.
- **In-Degree Normalization**: Solved GRU gate saturation during message passing, ensuring stable gradients across all recurrent unrolling steps.
- **Residual Semantics**: Added skip-connections (`h_init + h_gnn`) so that transformer statement semantics are preserved alongside graph topological structure.
- **Calibrated Decision Boundaries**: Optimized classification thresholds on validation ROC-AUC to prevent degenerate argmax flipping.

---

## 📜 License

This project is licensed under the Apache 2.0 License.
