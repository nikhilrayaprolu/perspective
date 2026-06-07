import os
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

import diskannpy as dap

class Qwen3Embedding:
    """
    Wrapper for Qwen3-Embedding model. Supports both Hugging Face transformers and sentence-transformers.
    """
    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B", engine="transformers", device="cuda", attn_implementation="sdpa"):
        self.engine = engine
        self.device = device
        self.model_name = model_name
        self.attn_implementation = attn_implementation
        
        logger.info(f"Initializing Qwen3Embedding with model '{model_name}' using '{engine}' engine on device '{device}'...")
        
        if self.engine == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name, device=device)
                logger.info("SentenceTransformer model loaded successfully.")
            except ImportError:
                logger.warning("sentence-transformers package not installed. Falling back to transformers engine.")
                self.engine = "transformers"
            except Exception as e:
                logger.error(f"Failed to load via sentence-transformers: {e}. Falling back to transformers engine.")
                self.engine = "transformers"
                
        if self.engine == "transformers":
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            
            kwargs = {}
            if self.attn_implementation in ["flash_attention_2", "sdpa"]:
                kwargs["attn_implementation"] = self.attn_implementation
            
            if "cuda" in device and torch.cuda.is_available():
                kwargs["torch_dtype"] = torch.bfloat16
            else:
                kwargs["torch_dtype"] = torch.float32
                
            self.model = AutoModel.from_pretrained(model_name, **kwargs).to(device)
            self.model.eval()
            logger.info("Transformers AutoModel and Tokenizer loaded successfully.")

    def last_token_pool(self, last_hidden_states, attention_mask):
        """
        Extract the embedding of the last actual token (before padding) from causal LM hidden states.
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            dev = last_hidden_states.device
            return last_hidden_states[torch.arange(batch_size, device=dev), sequence_lengths]

    @torch.no_grad()
    def embed_texts(self, texts, is_query=False, instruction=None, batch_size=32):
        """
        Embed list of texts.
        If is_query is True and an instruction is provided, prefix the text with the instruction format.
        """
        formatted_texts = []
        for text in texts:
            if is_query and instruction:
                formatted_texts.append(f"Instruct: {instruction}\nQuery: {text}")
            else:
                formatted_texts.append(text)

        if self.engine == "sentence-transformers":
            embeddings = self.model.encode(formatted_texts, convert_to_numpy=True, show_progress_bar=False)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-10, norms)
            return embeddings / norms
        else:
            # Using transformers directly
            embeddings_list = []
            for i in range(0, len(formatted_texts), batch_size):
                batch_texts = formatted_texts[i:i+batch_size]
                batch_dict = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=8192, 
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**batch_dict)
                embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                embeddings_list.append(embeddings.cpu().numpy())
                
            return np.concatenate(embeddings_list, axis=0)


class VectorIndex:
    """
    Interface for DiskANN vector index.
    """
    def __init__(self, diskann_path, dimension=2560):
        self.dimension = dimension
        logger.info(f"Loading DiskANN index from: {diskann_path}")
        self.index = dap.StaticDiskIndex(diskann_path)

    def search(self, query_embeddings, k):
        """
        Search for top k nearest neighbors.
        query_embeddings: numpy array of shape (num_queries, dimension)
        """
        indices = []
        distances = []
        for query in query_embeddings:
            res = self.index.search(query, k_neighbors=k)
            indices.append(res.neighbors)
            distances.append(res.distances)
        return np.array(distances), np.array(indices)


def greedy_dpp_map(L, k):
    """
    Greedy MAP inference for Determinantal Point Process (DPP).
    """
    N = L.shape[0]
    selected_items = []
    
    c = np.zeros((k, N))
    d2 = np.diagonal(L).copy()
    
    for step in range(k):
        for item in selected_items:
            d2[item] = -np.inf
            
        j = np.argmax(d2)
        if d2[j] <= 1e-10:
            break
            
        selected_items.append(int(j))
        
        if len(selected_items) == k:
            break
            
        d_j = np.sqrt(d2[j])
        for i in range(N):
            if i not in selected_items:
                sum_c = 0.0
                for l in range(step):
                    sum_c += c[l, i] * c[l, j]
                val = (L[i, j] - sum_c) / d_j
                c[step, i] = val
                d2[i] = d2[i] - val * val
                
    return selected_items


def dpp_sample(L, k=None):
    """
    Sample a subset from a DPP with kernel matrix L.
    """
    val, vec = np.linalg.eigh(L)
    val = np.maximum(val, 0)
    N = L.shape[0]
    
    if k is not None:
        k = min(k, N)
        e = np.zeros((N + 1, k + 1))
        e[:, 0] = 1.0
        for n in range(1, N + 1):
            for l in range(1, min(n, k) + 1):
                e[n, l] = e[n - 1, l] + val[n - 1] * e[n - 1, l - 1]
                
        V = []
        l = k
        for n in range(N, 0, -1):
            if l == 0:
                break
            p = val[n - 1] * e[n - 1, l - 1] / e[n, l]
            if np.random.rand() < p:
                V.append(n - 1)
                l -= 1
        V_matrix = vec[:, V]
    else:
        V = []
        for n in range(N):
            p = val[n] / (val[n] + 1.0)
            if np.random.rand() < p:
                V.append(n)
        if len(V) == 0:
            return []
        V_matrix = vec[:, V]
        
    sampled_items = []
    k_selected = V_matrix.shape[1]
    
    for step in range(k_selected, 0, -1):
        probs = np.sum(V_matrix ** 2, axis=1)
        probs = probs / np.sum(probs)
        j = np.random.choice(N, p=probs)
        sampled_items.append(int(j))
        
        if step == 1:
            break
            
        row_j = V_matrix[j, :]
        u = row_j / np.linalg.norm(row_j)
        V_matrix = V_matrix - V_matrix @ np.outer(u, u)
        V_matrix, _ = np.linalg.qr(V_matrix)
        
    return sampled_items


def download_wiki_embeddings(repo_id="maknee/wikipedia_qwen_4b", local_dir="data/wikipedia_qwen_4b", download_index=False, index_variant="index_32_100_320"):
    """
    Download Parquet and/or DiskANN index files from Hugging Face.
    """
    os.makedirs(local_dir, exist_ok=True)
    
    if download_index:
        logger.info(f"Downloading DiskANN index files (variant '{index_variant}') from HF repo '{repo_id}' to '{local_dir}'...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[f"diskann/{index_variant}_*"],
            local_dir=local_dir
        )
    else:
        logger.info(f"Downloading Parquet base files from HF repo '{repo_id}' to '{local_dir}'...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["parquet/*"],
            local_dir=local_dir
        )
def load_queries(data_path_or_repo):
    """
    Load queries and perspectives from either a local JSONL file or a Hugging Face dataset.
    """
    if os.path.exists(data_path_or_repo) and os.path.isfile(data_path_or_repo):
        logger.info(f"Loading queries from local file: {data_path_or_repo}")
        queries = []
        with open(data_path_or_repo, 'r') as f:
            for line in f:
                queries.append(json.loads(line))
        return queries

    name_mapping = {
        "arguana": "timchen0618/Arguana",
        "arguana_generated": "timchen0618/Arguana",
        "arguana_generated.jsonl": "timchen0618/Arguana",
        "kialo": "timchen0618/Kialo",
        "kialo.jsonl": "timchen0618/Kialo",
        "opinionqa": "timchen0618/OpinionQA",
        "opinionqa.jsonl": "timchen0618/OpinionQA"
    }
    
    repo_id = data_path_or_repo
    basename = os.path.basename(data_path_or_repo).lower()
    if basename in name_mapping:
        repo_id = name_mapping[basename]
        
    logger.info(f"Local file '{data_path_or_repo}' not found. Attempting to load from Hugging Face dataset '{repo_id}'...")
    try:
        from datasets import load_dataset
        dataset = load_dataset(repo_id)
        split = 'test' if 'test' in dataset else list(dataset.keys())[0]
        logger.info(f"Successfully loaded HF dataset '{repo_id}' (using split '{split}').")
        
        queries = []
        for row in dataset[split]:
            queries.append({
                "question": row.get("question", ""),
                "perspectives": row.get("perspectives", []),
                "ctxs": row.get("ctxs", [])
            })
        return queries
    except Exception as e:
        if "Invalid pattern" in str(e) or "fsspec" in str(e):
            logger.error("\n" + "="*80 + "\n"
                         "Dependency Version Conflict Detected!\n"
                         "This is a known issue caused by an incompatible version of the 'fsspec' library.\n"
                         "To fix this, please run the following command in your Colab notebook or terminal:\n\n"
                         "    !pip install -U datasets huggingface_hub fsspec\n\n"
                         "And then restart your Python runtime (Runtime -> Restart session).\n" + "="*80 + "\n")
        raise RuntimeError(
            f"Could not load queries from path or Hugging Face repo '{data_path_or_repo}'.\n"
            f"Original Error: {e}\n"
            "If this is a dataset/fsspec version conflict, try running: pip install -U datasets huggingface_hub fsspec"
        )


def get_local_corpus_text(repo_id="maknee/wikipedia_qwen_4b", local_dir="data/wikipedia_qwen_4b"):
    """
    Load the Wikipedia text column by streaming from Hugging Face (avoiding downloading embeddings)
    and cache it locally in a simple JSONL file for extremely fast random-access lookups.
    """
    corpus_cache_file = os.path.join(local_dir, "corpus_text.jsonl")
    if os.path.exists(corpus_cache_file):
        logger.info(f"Loading cached corpus texts from {corpus_cache_file}...")
        texts = []
        with open(corpus_cache_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                texts.append(data.get('text', ''))
        return texts, None

    logger.info(f"Local corpus text cache not found. Streaming text column from HF repo '{repo_id}' (this avoids downloading massive embedding files)...")
    from datasets import load_dataset
    # Load dataset in streaming mode, selecting only 'text' column
    streamed_ds = load_dataset(repo_id, split="train", columns=["text"], streaming=True)
    
    texts = []
    logger.info("Caching texts locally to corpus_text.jsonl...")
    with open(corpus_cache_file, 'w') as f:
        for idx, row in enumerate(streamed_ds):
            text = row.get("text", "")
            texts.append(text)
            f.write(json.dumps({"text": text}) + "\n")
            if (idx + 1) % 100000 == 0:
                logger.info(f"Cached {idx + 1} documents...")
                
    logger.info(f"Successfully cached {len(texts)} documents to {corpus_cache_file}.")
    return texts, None


def main():
    parser = argparse.ArgumentParser(description="End-to-End Retrieval & DPP Reranking on 1M Wikipedia Qwen Corpus")
    parser.add_argument("--data", type=str, required=True, help="Path to input BERDS JSONL file containing questions and perspectives")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the retrieved & reranked JSONL file")
    parser.add_argument("--wiki_repo_id", type=str, default="maknee/wikipedia_qwen_4b", help="HF Repo ID containing Wikipedia text and Qwen 4B embeddings")
    parser.add_argument("--local_wiki_dir", type=str, default="data/wikipedia_qwen_4b", help="Local directory to cache/download Wikipedia dataset")
    parser.add_argument("--download_data", action="store_true", help="Download the Wikipedia corpus and index before running")
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-4B", help="Model to generate query embeddings (should match corpus dimension: 2560)")
    parser.add_argument("--engine", type=str, default="transformers", choices=["transformers", "sentence-transformers"], help="Library engine to use for query inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run query inference on")
    
    parser.add_argument("--diskann_index_path", type=str, default=None, help="Path to DiskANN index files")
    parser.add_argument("--index_variant", type=str, default="index_32_100_320", help="Specific DiskANN index variant to download/use (e.g., index_32_100_320, index_32_100_640)")
    
    parser.add_argument("--stage1_k", type=int, default=100, help="Number of candidate documents to retrieve in Stage 1")
    parser.add_argument("--topk", type=int, default=5, help="Number of diverse documents to output in the final set")
    parser.add_argument("--theta", type=float, default=2.0, help="Scaling factor (inverse temperature) for quality score in DPP")
    parser.add_argument("--lambda_val", type=float, default=0.5, help="Weight (0 to 1) for perspective similarity vs content similarity in DPP matrix")
    parser.add_argument("--mode", type=str, default="greedy", choices=["greedy", "sample"], help="Reranking method: 'greedy' or 'sample'")
    
    parser.add_argument("--quality_instruction", type=str, default="Given a query, retrieve relevant documents.", help="Instruction prompt for quality embedding")
    parser.add_argument("--perspective_instruction", type=str, default="Retrieve documents that support or discuss the perspective: {perspective}", help="Instruction template for perspective embedding")
    
    args = parser.parse_args()

    # 1. Download data if requested
    if args.download_data:
        download_wiki_embeddings(
            repo_id=args.wiki_repo_id,
            local_dir=args.local_wiki_dir,
            download_index=True,
            index_variant=args.index_variant
        )

    # 2. Load BERDS query dataset
    queries_data = load_queries(args.data)
    logger.info(f"Loaded {len(queries_data)} queries.")

    # 3. Load Wikipedia corpus texts and titles (avoiding downloading embeddings)
    corpus_texts, corpus_titles = get_local_corpus_text(args.wiki_repo_id, args.local_wiki_dir)
    
    # 4. Load DiskANN search index
    if not args.diskann_index_path:
        # Guess default path inside the local folder using the variant
        args.diskann_index_path = os.path.join(args.local_wiki_dir, "diskann", args.index_variant)
    vector_index = VectorIndex(diskann_path=args.diskann_index_path, dimension=2560)

    # 5. Initialize Query Embedder
    embedding_model = Qwen3Embedding(
        model_name=args.model_name,
        engine=args.engine,
        device=args.device
    )

    reranked_results = []

    # 6. Retrieve and DPP rerank
    for idx, inst in enumerate(tqdm(queries_data, desc="Retrieving and Reranking")):
        question = inst.get("question", "")
        perspectives = inst.get("perspectives", [])

        # Step A: Embed query with quality instruction
        query_quality_emb = embedding_model.embed_texts([question], is_query=True, instruction=args.quality_instruction, batch_size=1)

        # Step B: Stage 1 Search (Retrieve Top-N candidates)
        # Vector index search returns distances (cosine similarities if normalized) and indices
        distances, indices = vector_index.search(query_quality_emb, args.stage1_k)
        candidate_indices = indices[0]
        quality_scores = distances[0]

        # Step C: Extract candidate text, title, and pre-computed embeddings
        candidate_texts = [corpus_texts[int(i)] for i in candidate_indices]
        candidate_titles = [corpus_titles[int(i)] for i in candidate_indices] if corpus_titles else [f"Doc_{i}" for i in candidate_indices]
        
        # Dynamically compute candidate document embeddings from candidate_texts
        # (This is fast for stage1_k=100 and avoids downloading the 10GB+ database embeddings)
        candidate_embs = embedding_model.embed_texts(candidate_texts, is_query=False, batch_size=32)

        # Step D: Compute document content similarity matrix S_doc_sim (N x N)
        S_doc_sim = candidate_embs @ candidate_embs.T

        # Step E: Compute perspective relevance matrix R (N x m)
        if len(perspectives) > 0 and args.lambda_val > 0.0:
            persp_query_embs = []
            for p in perspectives:
                inst_p = args.perspective_instruction.format(perspective=p)
                emb = embedding_model.embed_texts([question], is_query=True, instruction=inst_p, batch_size=1)
                persp_query_embs.append(emb)
            persp_query_embs = np.concatenate(persp_query_embs, axis=0) # Shape: (m, D)
            
            # Compute relevance to each perspective query
            R = candidate_embs @ persp_query_embs.T  # Shape: (N, m)
            
            # Compute S_persp_sim as cosine similarity between rows of R
            norms = np.linalg.norm(R, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-10, norms)
            R_normalized = R / norms
            S_persp_sim = R_normalized @ R_normalized.T
        else:
            S_persp_sim = np.zeros((args.stage1_k, args.stage1_k))
            args.lambda_val = 0.0

        # Step F: Form combined similarity matrix S
        S = (1.0 - args.lambda_val) * S_doc_sim + args.lambda_val * S_persp_sim
        S = np.clip(S, -1.0, 1.0)
        S = (S + 1.0) / 2.0  # Shift to [0, 1]

        # Step G: Build DPP Matrix L
        q_scaled = np.exp(args.theta * quality_scores)
        L = np.outer(q_scaled, q_scaled) * S
        L += np.eye(args.stage1_k) * 1e-6  # stability jitter

        # Step H: Select final documents
        k = min(args.topk, args.stage1_k)
        if args.mode == "greedy":
            selected_indices = greedy_dpp_map(L, k)
        else:
            selected_indices = dpp_sample(L, k)
            if len(selected_indices) < k:
                remaining = [i for i in range(args.stage1_k) if i not in selected_indices]
                sorted_remaining = sorted(remaining, key=lambda idx: quality_scores[idx], reverse=True)
                selected_indices.extend(sorted_remaining[:k - len(selected_indices)])

        # Step I: Build the 'ctxs' output list
        new_ctxs = []
        for rank_idx, doc_idx in enumerate(selected_indices):
            global_idx = candidate_indices[doc_idx]
            new_ctxs.append({
                "title": str(candidate_titles[doc_idx]),
                "text": str(candidate_texts[doc_idx]),
                "score": float(quality_scores[doc_idx]),
                "dpp_rank": rank_idx + 1,
                "global_index": int(global_idx)
            })

        inst["ctxs"] = new_ctxs
        reranked_results.append(inst)

    # Save output file
    logger.info(f"Writing retrieved & reranked results to '{args.output_file}'...")
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output_file, 'w') as f:
        for inst in reranked_results:
            f.write(json.dumps(inst) + '\n')
            
    logger.info("Done! End-to-end pipeline execution finished.")

if __name__ == "__main__":
    main()
