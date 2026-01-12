# Content Generation & RAG Module

This module corresponds to **Task 3 (Content Generation)** of the DebateTroll project. It implements an intelligent rebuttal generation pipeline that leverages Strategy Planning integration and Retrieval-Augmented Generation (RAG) to produce high-quality, evidence-grounded debate content.

## Features

*   **Controllable Generation**: Generates rebuttals based on specific strategies, audience types, and styles (Academic, Public, Socratic).
*   **Dual-Retrieval Architecture**:
    1.  **ColBERTv2**: High-precision, late-interaction retrieval for fine-grained evidence logical matching.
    2.  **Scalable Dense Retrieval**: Efficient embedding-based retrieval (all-MiniLM-L6-v2) with streaming metadata, capable of handling large-scale corpuses (OpenCaselist) on limited hardware.
*   **Multi-Dataset Support**: Compatible with **OpenDebateEvidence (OpenCaselist)** and **DebateSum** datasets.
*   **Satire Mode**: Includes a "SatireBot" bonus module for stylistic rewriting.

## Directory Structure

```text
Content_Generation/
├── content_generator.py    # Main pipeline orchestrator
├── llm_client.py           # LLM API wrapper (DeepSeek/OpenAI)
├── prompts.py              # Prompt templates for generation & planning
├── schema.py               # JSON data structures
├── requirements.txt        # Module dependencies
├── retrieval/              # Retrieval implementations
│   ├── colbertv2_retriever.py
│   ├── simple_embedding_retriever.py  # Dense retrieval implem.
│   └── opencaselist.py     # Dataset loader
└── scripts/                # Utility scripts for data & experiments
    ├── build_*.py          # Index construction scripts
    ├── download_*.py       # Data downloaders
    └── run_*_demo.py       # Execution demos
```

## Installation

1.  **Environment Setup**:
    ```bash
    conda create -n content_gen python=3.10
    conda activate content_gen
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    Set your LLM API key in your environment variables:
    ```bash
    export LLM_API_KEY="your-api-key-here"
    export LLM_API_BASE="https://api.deepseek.com" # or other base URL
    ```

## Usage Guide

### 1. Data Preparation
Before running the generator, you need to prepare the evidence data. 

*   **Option A: OpenCaselist (Large Scale)**
    ```bash
    # Download raw data
    python scripts/download_opencaselist.py
    # Process into JSONL collection
    python scripts/prepare_opencaselist_collection.py
    ```

*   **Option B: DebateSum (Smaller/Backup)**
    ```bash
    python scripts/prepare_debatesum.py
    ```

### 2. Building the Retriever Index
The system supports two retrieval modes. Choose one based on your resource constraints.

*   **Mode 1: Scalable Dense Retrieval (Recommended for Speed/Low-Resource)**
    Builds a lightweight FAISS-compatible index with streaming metadata.
    ```bash
    python scripts/build_large_dense_index.py \
        --collection_path data/opencaselist_collection.jsonl \
        --output_dir data/dense_index
    ```

*   **Mode 2: ColBERTv2 (High Precision)**
    Builds a full ColBERT index (requires significantly more storage/RAM).
    ```bash
    python scripts/build_colbert_index.py \
        --collection_path data/opencaselist_collection.jsonl \
        --index_name opencaselist_colbert
    ```

### 3. Running the Generator (Demo)

We provide a demo script that loads the index and generates rebuttals for a given argument.

**Example Command (using Dense Retrieval):**
```bash
python scripts/run_dense_rag_demo.py \
    --index_path data/dense_index/embeddings.pt \
    --meta_path data/dense_index/metadata.jsonl \
    --text "Social media censorship is necessary to protect democracy." \
    --query "Social media censorship harms free speech" \
    --style "Academic"
```

### 4. Integration via Python
To use the `ContentGenerator` in your own code:

```python
from content_generator import ContentGenerator
from retrieval.simple_embedding_retriever import SimpleEmbeddingRetriever

# 1. Initialize Retriever
retriever = SimpleEmbeddingRetriever(
    index_path="data/dense_index/embeddings.pt",
    metadata_path="data/dense_index/metadata.jsonl"
)

# 2. Initialize Generator
generator = ContentGenerator(
    retriever=retriever,
    llm_config={"model": "deepseek-chat"}
)

# 3. Generate Rebuttal
result = generator.generate_rebuttal(
    context="The opponent argues that universal basic income leads to inflation.",
    fallacy_type="Slippery Slope",
    plan={"strategy": "Highlight lack of empirical evidence"},
    style="Public"
)

print(result.full_text)
print(result.citations)
```
