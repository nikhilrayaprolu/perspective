# Running the Qwen3-Embedding DPP Retrieval & Reranking Pipeline

This guide provides step-by-step instructions on setting up your environment, downloading the 1M Wikipedia corpus, running the end-to-end vector search using DiskANN, and performing perspective-aware DPP reranking.

---

## 1. Prerequisites and Installation

### A. Python Environment
We recommend using a python virtual environment (Python 3.8+).

```bash
# Create and activate virtual environment
python3 -m venv berds_env
source berds_env/bin/activate
```

### B. Installing Required Packages
Install the packages required by the BERDS benchmark, plus the `diskannpy` library:

```bash
# Install base requirements
pip install -r requirements.txt

# Install diskannpy for DiskANN support
pip install diskannpy
```

*Note: `diskannpy` contains native C++ bindings for DiskANN. Ensure your environment has standard build tools (`g++`, `cmake`) and header files if compiling from source.*

---

## 2. Dataset and Corpus Download (Bandwidth Optimized)

The pipeline is fully optimized to **avoid downloading the massive 10GB+ database embedding files** from the `maknee/wikipedia_qwen_4b` repository. Instead, it downloads only the necessary files:

1. **DiskANN Index Directory** (via `--download_data` flag): Downloads only the native pre-built DiskANN index files (`diskann/*`), which are extremely fast and lightweight.
2. **Text Corpus Streaming**: On the first run, the script streams **only the `text` and `title` columns** from the Hugging Face parquet dataset (completely skipping the 10GB embedding column) and caches it locally as a lightweight JSONL file (`corpus_text.jsonl`, ~550 MB).
3. **Dynamic Candidate Encoding**: During the query search, only the top candidate documents (default: 100 candidates) are dynamically encoded on the GPU using your query embedder. This takes a fraction of a second and eliminates the need to download 1M pre-computed database vectors.

---

## 3. Running the Pipeline

We provide two scripts to run the pipeline:
1. **End-to-End Search & DPP Rerank** (`reranking/retrieve_and_dpp_qwen.py`)
2. **Stage 2 Rerank Only** (`reranking/dpp_rerank_qwen.py`)

### Option A: End-to-End Wikipedia Search and Rerank (Recommended)
This script runs the query embedding generation, queries the 1M article DiskANN index to retrieve candidates, dynamically encodes the candidates, constructs the DPP kernel, and selects the final documents.

```bash
# You can pass a local JSONL file:
python reranking/retrieve_and_dpp_qwen.py \
    --data ./Data/kialo/kialo.test.jsonl \
    --output_file ./outputs/kialo_dpp_wiki.jsonl \
    --download_data \
    --local_wiki_dir ./data/wikipedia_qwen_4b \
    --diskann_index_path ./data/wikipedia_qwen_4b/diskann \
    --stage1_k 100 \
    --topk 5 \
    --theta 2.0 \
    --lambda_val 0.6 \
    --mode greedy \
    --device cuda

# OR you can pass the Hugging Face dataset name directly:
python reranking/retrieve_and_dpp_qwen.py \
    --data kialo \
    --output_file ./outputs/kialo_dpp_wiki.jsonl \
    --download_data \
    --local_wiki_dir ./data/wikipedia_qwen_4b \
    --diskann_index_path ./data/wikipedia_qwen_4b/diskann
```

### Option B: Automated Benchmark Script
You can run the automated bash script, which processes all three benchmark datasets (`Arguana`, `Kialo`, `OpinionQA`) and automatically evaluates the final diversity results. It will look for local files first, and automatically fetch from Hugging Face if they are not present:

```bash
# Make the script executable
chmod +x run_end_to_end_wiki.sh

# Run the pipeline
./run_end_to_end_wiki.sh
```

### Option C: Stage 2 Rerank Only (On Existing JSONL Candidate Files)
If you already have pre-retrieved candidate documents in a JSONL file under the `"ctxs"` field (e.g. from BM25 or Contriever) and only want to run the DPP reranking:

```bash
# You can pass a local file:
python reranking/dpp_rerank_qwen.py \
    --data /path/to/retrieval_outputs/wiki/bm25/kialo.jsonl \
    --output_file /path/to/retrieval_outputs/wiki/bm25/kialo_reranked.jsonl \
    --model_name Qwen/Qwen3-Embedding-0.6B \
    --topk 5 \
    --theta 2.0 \
    --lambda_val 0.5 \
    --mode greedy

# OR pass the Hugging Face dataset name directly:
python reranking/dpp_rerank_qwen.py \
    --data kialo \
    --output_file ./outputs/kialo_reranked.jsonl \
    --model_name Qwen/Qwen3-Embedding-0.6B
```

---

## 4. Hyperparameter Tuning Guide

| Parameter | Type | Default | Explanation |
| :--- | :--- | :--- | :--- |
| `--stage1_k` | int | `100` | Number of candidate documents retrieved by DiskANN in Stage 1. |
| `--topk` | int | `5` | Number of final diverse documents to output in the `"ctxs"` field. |
| `--theta` | float | `2.0` | **Relevance scale (Inverse Temperature)**. Higher values increase the importance of the core query relevance score. Lower values decrease it. |
| `--lambda_val`| float | `0.5` | **Stance diversity weight**. `0.0` uses pure semantic document similarity. `1.0` uses pure perspective-coverage similarity. |
| `--mode` | str | `greedy` | **Selection strategy**. `greedy` uses deterministic Cholesky-based greedy MAP optimization. `sample` uses probabilistic k-DPP sampling. |
| `--diskann_threads`| int | `16` | Number of threads for the native DiskANN search operations. |
| `--diskann_nodes_to_cache`| int | `10000` | Number of index nodes cached in memory to accelerate search. |

---

## 5. Evaluation

After producing the reranked JSONL files, evaluate their perspective coverage and precision:

```bash
PYTHONPATH=. python Eval/eval_vllm.py \
    --data ./outputs/kialo_dpp_wiki.jsonl \
    --output_file ./outputs/kialo_dpp_wiki.mistralpred \
    --instructions Eval/instructions_chat.txt \
    --model timchen0618/Mistral_BERDS_evaluator_full \
    --model_type mistral \
    --topk 5
```
