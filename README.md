# Zero-Shot Information Retrieval: A Comparative Analysis of LLMs and Classical Retrieval Methods
**CENG543 NLP/IR Term Project - Izmir Institute of Technology**

## 📖 Overview
This project evaluates the **zero-shot re-ranking performance** of various Large Language Models (LLMs) against classical lexical retrievers (BM25) and modern dense retrieval methods (Sentence-BERT) on the **BEIR benchmark**.

The goal is to analyze how well generative models adapt to diverse domains (general, scientific, and argumentative) without task-specific fine-tuning.

## 📊 Datasets
* **Natural Questions (NQ):** General open-domain QA
* **SciFact:** Scientific claim verification
* **Arguana:** Counter-argument retrieval

## 🤖 Models Compared

| Category | Model | Implementation |
| :--- | :--- | :--- |
| **Lexical Baseline** | BM25 | via `Pyserini` / `BEIR` |
| **Semantic Baseline** | Sentence-BERT | via `sentence-transformers` |
| **Commercial LLM** | GPT-3.5-Turbo | via OpenAI API |
| **Open-Source LLM** | Llama 3 (8B Instruct) | via Hugging Face / Ollama |
| **Lightweight LLM** | Phi-3 / Gemma | via Hugging Face |

## 📉 Evaluation Metrics
* **NDCG@10:** Normalized Discounted Cumulative Gain (Primary Metric)
* **Precision@3:** Top-3 accuracy
* **Spearman's ρ:** Ranking correlation with ground truth

## 🛠 Requirements
* **Python 3.10** (Recommended)
* **Java 11+** (Required for Pyserini/Lucene backend)
* Dependencies listed in `requirements.txt`

## ⚙️ Installation

### 1. Create Conda Environment
We strongly recommend using Conda to manage Python and Java dependencies together.

```bash
conda create -n ceng543_ir python=3.10 -y
conda activate ceng543_ir

```

### 2. Install System Dependencies (Java 11)

Install OpenJDK 11 directly into the Conda environment to avoid system conflicts:

```bash
conda install -c conda-forge openjdk=11 -y

```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure Java Home (Crucial)

Point `JAVA_HOME` to the Conda environment's Java installation:

**macOS / Linux:**

```bash
export JAVA_HOME=$CONDA_PREFIX

```

**Windows:**

```cmd
set JAVA_HOME=%CONDA_PREFIX%

```

### 5. Verify Installation

```bash
python -c "import torch; import beir; from pyserini.search.lucene import LuceneSearcher; print('Environment ready! ✅')"

```

## LLM Setup (Local Inference)

This project uses [Ollama](https://ollama.com) to run LLMs locally without API costs.

### 1. Install Ollama

**macOS (Homebrew):**
```bash
brew install ollama
```

**Or download directly from:** https://ollama.com/download

### 2. Start Ollama Service
```bash
ollama serve
```

Keep this running in a separate terminal, or run as background service.

### 3. Download Llama 3 Model
```bash
ollama pull llama3:8b
```

> **Note:** This downloads approximately 4.7GB. The model will be cached locally for future use.

### 4. Verify Installation
```bash
python -c "import ollama; print(ollama.chat(model='llama3:8b', messages=[{'role': 'user', 'content': 'Hello!'}])['message']['content'])"
```

### Supported Models

| Model | Size | Command |
|-------|------|---------|
| Llama 3 8B | 4.7GB | `ollama pull llama3:8b` |
| Llama 3 70B | 40GB | `ollama pull llama3:70b` |
| Mistral 7B | 4.1GB | `ollama pull mistral` |
| Phi-3 Mini | 2.2GB | `ollama pull phi3` |

## 📂 Project Structure

```bash
.
├── README.md
├── requirements.txt
├── datasets/                    # Downloaded BEIR datasets
├── src/
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25_baseline.py      # BM25 candidate retrieval
│   │   └── dense_retrieval.py    # Sentence-BERT re-ranking
│   ├── reranking/
│   │   ├── __init__.py
│   │   ├── llm_reranker.py       # LLM-based re-ranking
│   │   └── prompts.py            # Prompt templates
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py            # NDCG, Precision, Spearman
├── notebooks/
│   └── experiments.ipynb    # Main experimentation notebook
└── results/                 # Evaluation outputs

```

## 🚀 Usage

### 1. Run Retrieval Pipeline (BM25)

This script downloads the dataset, indexes it using Pyserini, and retrieves top-100 candidates.

```python
# Example snippet from src/step1_bm25_baseline.py
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval

model = BM25(index_name="scifact_index", hostname="localhost", initialize=True)
retriever = EvaluateRetrieval(model)
results = retriever.retrieve(corpus, queries)

```

Run via command line:

```bash
python src/step1_bm25_baseline.py

```

### 2. Pipeline Overview

```mermaid
graph LR
    A[Query] --> B(BM25 Retrieval)
    B --> C{Top-100 Candidates}
    C --> D[Re-Ranker Model]
    D --> E[Sentence-BERT]
    D --> F[GPT-3.5 / Llama 3]
    D --> G[Phi-3 / Gemma]
    E & F & G --> H[Evaluation Metrics]
    H --> I[NDCG@10 / Precision@3]

```

*(Note: If Mermaid is not supported in your viewer, see the ASCII diagram below)*

```text
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Query     │────▶│  BM25 Top-100│────▶│  Re-ranker  │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    ▼                           ▼                           ▼
            ┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
            │ Sentence-BERT │         │   GPT-3.5/Llama │         │   Phi-3/Gemma   │
            └───────────────┘         └─────────────────┘         └─────────────────┘
                    │                           │                           │
                    └───────────────────────────┼───────────────────────────┘
                                                ▼
                                    ┌─────────────────────┐
                                    │  Evaluation Metrics │
                                    │  NDCG@10, P@3, ρ    │
                                    └─────────────────────┘

```

## 📚 References

* Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models," 2021.
* Liu et al., "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Rankers," 2023.
* Pradeep et al., "The MonoT5 Ranker," 2020.
* Reimers and Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," 2019.

---

**Author:** Kübra Holt

**Institution:** Izmir Institute of Technology

**Date:** January 2026
