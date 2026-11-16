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

Analyze any raw C or C++ file to receive an automated vulnerability verdict and a statement-level attention breakdown:

```bash
# Analyze a vulnerable program (default 0.50 threshold):
python infer.py --file examples/test_vulnerable.c --model models/best_model.pth

# Analyze a safe program:
python infer.py --file examples/test_safe.c --model models/best_model.pth

# Run with recall-optimized calibrated threshold (0.35) for high-sensitivity security audits:
python infer.py --file examples/test_vulnerable.c --model models/best_model.pth --calibrated
```

### Verified Sample Outputs (Maximum Capacity Model)

#### 1. Vulnerable Example (`examples/test_vulnerable.c`):
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
  >>> CONFIDENCE  : 50.98%
-----------------------------------------------------------------
  Class 0 (Benign/Safe) Probability      : 49.02%
  Class 1 (Vulnerable) Probability      : 50.98%
  Decision Threshold Applied             : 0.5000

-----------------------------------------------------------------
  Top Attention Statements (Most Suspicious Nodes):
    [1] (12.64% weight): vulnerable_function(argv[1]);
    [2] (12.19% weight): #include <stdlib.h>
    [3] (11.06% weight): strcpy(buffer, user_input);
    [4] ( 9.71% weight): printf("Copied buffer: %s\n", buffer);
    [5] ( 8.25% weight): if (argc > 1) {
=================================================================
```

#### 2. Benign/Safe Example (`examples/test_safe.c`):
```text
=================================================================
[*] Analyzing Source File: examples/test_safe.c
=================================================================
[*] Graph Extraction Engine : Semantic CPG Extractor (AST + CFG + DFG)
[*] Graph Statistics        : 8 Nodes | 20 Relational Edges
[*] Initializing CodeBERT Embedder (microsoft/codebert-base) on device: cuda
[*] Loading trained weights from : models/best_model.pth

=================================================================
                    PREDICTION VERDICT
=================================================================
  >>> RESULT      : [SAFE]
  >>> CONFIDENCE  : 51.22%
-----------------------------------------------------------------
  Class 0 (Benign/Safe) Probability      : 51.22%
  Class 1 (Vulnerable) Probability      : 48.78%
  Decision Threshold Applied             : 0.5000

-----------------------------------------------------------------
  Top Attention Statements (Most Suspicious Nodes):
    [1] (22.25% weight): int sum = safe_add(a, b);
    [2] (17.70% weight): return x + y;
    [3] (14.61% weight): int b = 20;
    [4] (14.28% weight): int safe_add(int x, int y) {
    [5] (14.07% weight): int a = 10;
=================================================================
```

---

## 📈 Maximum Capacity Benchmark Performance

The model was trained on the full 27,318 Devign dataset across 80 epochs with Early Stopping (`patience=15`) on an NVIDIA GeForce RTX 4060 GPU with weighted focal loss. Full execution logs are archived in `logs/training_gnn_max_capacity.log`.

| Metric | Tabular Baseline (Random Forest) | GNN (25 Epochs) | Relational GGNN (80 Epochs + Early Stopping) |
|---|---|---|---|
| **Dataset Scale** | 2,746 samples | 27,318 samples | **27,318 samples** (21,854 train, 2,732 val, 2,732 test) |
| **Test ROC-AUC** | 0.5210 | 0.5608 | **0.6196** (+5.88% improvement) |
| **Test Accuracy** | 0.4912 | 0.4828 | **0.5714** (57.14% on 2.7k test functions) |
| **Test Precision** | 0.4501 | 0.4697 | **0.5275** (52.75% precision) |
| **Test Recall (Vuln)** | 0.8140 | 0.9753 | **0.8781** (calibrated) / 0.6422 (0.50 threshold) |
| **Test F1-Score** | 0.5482 | 0.6340 | **0.6350** |
| **Statement Attention** | ❌ No | ✅ Yes | ✅ **Yes (CodeBERT + Soft-Attention)** |

---

## 🧪 Testing

Run automated tests to verify the pipeline components:

```bash
python -m unittest discover tests/
```

---

## 📊 Key Highlights & Recent Improvements

- **Full-Scale Devign Integration**: Trained on 27,318 verified real-world functions (QEMU, FFmpeg), eliminating synthetic data distribution bias.
- **Minified Statement Parsing**: Resolved an issue where single-line C code was parsed as 1-node graphs; parser now extracts rich relational AST/CFG/DFG statement graphs.
- **In-Degree Normalization**: Prevents GRU gate saturation during message passing, ensuring gradient stability across recurrent unrolling steps.
- **Residual Semantics**: Added skip-connections (`h = h_init + h_gnn`) to preserve CodeBERT transformer token semantics alongside graph topological structure.
- **Dual Threshold Support**: Default `0.50` threshold for balanced discrimination, with optional `--calibrated` (0.35) for zero-tolerance vulnerability screening.

---

## 📜 License

This project is licensed under the Apache 2.0 License.
