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

## 2. Dataset and Corpus Download

The pipeline uses the **`maknee/wikipedia_qwen_4b`** dataset on Hugging Face. This dataset contains:
1. **`base` split**: 1,000,000 Wikipedia articles with pre-computed 2560-dimensional embeddings.
2. **`diskann` directory**: Pre-built DiskANN index files (`index_*.index`, `gt_*.fbin`, etc.).

The scripts will **automatically download** these files when you run them with the `--download_data` flag, so you do not need to download them manually.

---

## 3. Running the Pipeline

We provide two scripts to run the pipeline:
1. **End-to-End Search & DPP Rerank** (`reranking/retrieve_and_dpp_qwen.py`)
2. **Stage 2 Rerank Only** (`reranking/dpp_rerank_qwen.py`)

### Option A: End-to-End Wikipedia Search and Rerank (Recommended)
This script runs the query embedding generation, queries the 1M article DiskANN index to retrieve candidates, fetches their pre-computed embeddings, constructs the DPP kernel, and selects the final documents.

```bash
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
```

### Option B: Automated Benchmark Script
You can run the automated bash script, which processes all three benchmark datasets (`Arguana`, `Kialo`, `OpinionQA`) and automatically evaluates the final diversity results:

```bash
# Make the script executable
chmod +x run_end_to_end_wiki.sh

# Run the pipeline
./run_end_to_end_wiki.sh
```

### Option C: Stage 2 Rerank Only (On Existing JSONL Candidate Files)
If you already have pre-retrieved candidate documents in a JSONL file under the `"ctxs"` field (e.g. from BM25 or Contriever) and only want to run the DPP reranking:

```bash
python reranking/dpp_rerank_qwen.py \
    --data /path/to/retrieval_outputs/wiki/bm25/kialo.jsonl \
    --output_file /path/to/retrieval_outputs/wiki/bm25/kialo_reranked.jsonl \
    --model_name Qwen/Qwen3-Embedding-0.6B \
    --topk 5 \
    --theta 2.0 \
    --lambda_val 0.5 \
    --mode greedy
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
