import os
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class Qwen3Embedding:
    """
    Wrapper for Qwen3-Embedding model. Supports both Hugging Face transformers and sentence-transformers.
    """
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", engine="transformers", device="cuda", attn_implementation="sdpa"):
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
            # Ensure they are L2 normalized
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-10, norms)
            return embeddings / norms
        else:
            # Using transformers directly
            embeddings_list = []
            
            def _forward_pass(chunk_texts):
                batch_dict = self.tokenizer(
                    chunk_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=8192, 
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**batch_dict)
                embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                return embeddings.float().cpu().numpy()

            i = 0
            curr_batch_size = batch_size
            while i < len(formatted_texts):
                chunk_texts = formatted_texts[i:i+curr_batch_size]
                try:
                    emb = _forward_pass(chunk_texts)
                    embeddings_list.append(emb)
                    i += curr_batch_size
                except (torch.cuda.OutOfMemoryError if hasattr(torch.cuda, "OutOfMemoryError") else Exception, RuntimeError) as e:
                    is_oom = False
                    if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(e, torch.cuda.OutOfMemoryError):
                        is_oom = True
                    elif "out of memory" in str(e).lower():
                        is_oom = True
                    
                    if is_oom:
                        torch.cuda.empty_cache()
                        if curr_batch_size > 1:
                            new_batch_size = curr_batch_size // 2
                            logger.warning(f"CUDA Out of Memory. Reducing batch size from {curr_batch_size} to {new_batch_size} and retrying.")
                            curr_batch_size = new_batch_size
                        else:
                            logger.error("CUDA Out of Memory even with batch_size=1. Cannot proceed.")
                            raise e
                    else:
                        raise e
                
            return np.concatenate(embeddings_list, axis=0)

def greedy_dpp_map(L, k):
    """
    Greedy MAP inference for Determinantal Point Process (DPP).
    Selects k items from L that maximize the log-determinant log det(L_Y).
    L: PSD kernel matrix (N x N)
    k: number of items to select
    """
    N = L.shape[0]
    selected_items = []
    
    # c maintains the Cholesky factor rows of the submatrix
    c = np.zeros((k, N))
    d2 = np.diagonal(L).copy()  # d2[i] represents L_ii
    
    for step in range(k):
        # Set variance of already selected items to -inf to exclude them
        for item in selected_items:
            d2[item] = -np.inf
            
        j = np.argmax(d2)
        if d2[j] <= 1e-10:
            # If the best item has zero or negative variance, we stop
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
    If k is specified, samples exactly a k-DPP.
    L: PSD kernel matrix (N x N)
    k: target subset size
    """
    val, vec = np.linalg.eigh(L)
    val = np.maximum(val, 0)  # Remove tiny negative eigenvalues due to numerical errors
    N = L.shape[0]
    
    if k is not None:
        # k-DPP sampling using elementary symmetric polynomials
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
        # Standard DPP sampling
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


def main():
    parser = argparse.ArgumentParser(description="Diverse Document Reranking using Qwen3-Embedding and Determinantal Point Process (DPP)")
    parser.add_argument("--data", type=str, required=True, help="Path to input JSONL file containing retrieved results")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the reranked output JSONL file")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Hugging Face model ID for Qwen3-Embedding")
    parser.add_argument("--engine", type=str, default="transformers", choices=["transformers", "sentence-transformers"], help="Library engine to use for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"], help="Attention implementation for transformers")
    
    parser.add_argument("--topk", type=int, default=5, help="Number of documents to retrieve/rank in the final set")
    parser.add_argument("--theta", type=float, default=2.0, help="Scaling factor (inverse temperature) for quality relevance score in DPP diagonal")
    parser.add_argument("--lambda_val", type=float, default=0.5, help="Weight (0 to 1) for perspective similarity vs content similarity. 1.0 means pure perspective diversity.")
    parser.add_argument("--mode", type=str, default="greedy", choices=["greedy", "sample"], help="Reranking method: 'greedy' for deterministic MAP, 'sample' for probabilistic DPP sampling")
    
    parser.add_argument("--quality_instruction", type=str, default="Given a query, retrieve relevant documents.", help="Instruction prompt for quality/relevance embedding of the user query")
    parser.add_argument("--perspective_instruction", type=str, default="Retrieve documents that support or discuss the perspective: {perspective}", help="Instruction template for perspective-based embedding of the query")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding generation")
    
    args = parser.parse_args()

    # Load input data
    data = load_queries(args.data)
    logger.info(f"Successfully loaded {len(data)} instances.")

    # Initialize the embedding model
    embedding_model = Qwen3Embedding(
        model_name=args.model_name,
        engine=args.engine,
        device=args.device,
        attn_implementation=args.attn_implementation
    )

    reranked_data = []

    for idx, inst in enumerate(tqdm(data, desc="Reranking instances")):
        question = inst.get("question", "")
        perspectives = inst.get("perspectives", [])
        docs = inst.get("ctxs", [])

        if not docs:
            logger.warning(f"Instance {idx} has no documents in 'ctxs'. Skipping.")
            reranked_data.append(inst)
            continue

        N = len(docs)
        k = min(args.topk, N)

        # 1. Format document texts and generate embeddings
        doc_texts = []
        for doc in docs:
            if 'title' in doc and doc['title']:
                doc_texts.append(f"{doc['text']} {doc['title']}")
            elif 'wikipedia_title' in doc and doc['wikipedia_title']:
                doc_texts.append(f"{doc['text']} {doc['wikipedia_title']}")
            else:
                doc_texts.append(doc['text'])
                
        doc_embeddings = embedding_model.embed_texts(doc_texts, is_query=False, batch_size=args.batch_size)

        # 2. Compute document content similarity matrix S_doc_sim (N x N)
        S_doc_sim = doc_embeddings @ doc_embeddings.T

        # 3. Compute quality relevance score for each document
        query_quality_emb = embedding_model.embed_texts([question], is_query=True, instruction=args.quality_instruction, batch_size=1)
        quality_scores = (doc_embeddings @ query_quality_emb.T).squeeze(axis=1)  # Shape: (N,)

        # 4. Compute perspective relevance matrix R (N x m)
        if len(perspectives) > 0 and args.lambda_val > 0.0:
            persp_queries = []
            for p in perspectives:
                # Format instruction with specific perspective
                inst_p = args.perspective_instruction.format(perspective=p)
                persp_queries.append(question)
            
            # Since instructions can vary per query, we embed them individually or batch them if instruction varies
            persp_query_embs = []
            for p in perspectives:
                inst_p = args.perspective_instruction.format(perspective=p)
                emb = embedding_model.embed_texts([question], is_query=True, instruction=inst_p, batch_size=1)
                persp_query_embs.append(emb)
            persp_query_embs = np.concatenate(persp_query_embs, axis=0)  # Shape: (m, D)
            
            # R[i, k] is the cosine similarity between doc i and perspective-aware query k
            R = doc_embeddings @ persp_query_embs.T  # Shape: (N, m)
            
            # Compute S_persp_sim as cosine similarity between rows of R
            norms = np.linalg.norm(R, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-10, norms)
            R_normalized = R / norms
            S_persp_sim = R_normalized @ R_normalized.T
        else:
            S_persp_sim = np.zeros((N, N))
            # If no perspectives are present, fall back to pure content similarity
            args.lambda_val = 0.0

        # 5. Form the combined similarity matrix S
        S = (1.0 - args.lambda_val) * S_doc_sim + args.lambda_val * S_persp_sim

        # Ensure similarity values are in a stable range [0, 1] for DPP
        S = np.clip(S, -1.0, 1.0)
        # Shift or map cosine similarity to [0, 1] if needed, but raw is PSD.
        # Shift is safer to avoid negative entries in similarity matrix which might cause issues in some contexts,
        # but cosine similarity itself is PSD. To ensure it is purely non-negative similarity:
        S = (S + 1.0) / 2.0

        # 6. Form the DPP kernel matrix L
        # q_scaled_i = exp(theta * quality_i)
        q_scaled = np.exp(args.theta * quality_scores)
        L = np.outer(q_scaled, q_scaled) * S

        # Add a tiny diagonal perturbation (jitter) for numerical stability
        L += np.eye(N) * 1e-6

        # 7. Select top k documents
        if args.mode == "greedy":
            selected_indices = greedy_dpp_map(L, k)
        else:
            selected_indices = dpp_sample(L, k)
            # In case sampling returns fewer than k elements due to numerical thresholding
            if len(selected_indices) < k:
                remaining = [i for i in range(N) if i not in selected_indices]
                # Fill the rest with the highest quality documents that were not selected
                sorted_remaining = sorted(remaining, key=lambda idx: quality_scores[idx], reverse=True)
                selected_indices.extend(sorted_remaining[:k - len(selected_indices)])

        # 8. Reconstruct the ctxs field with selected documents
        # We store the selected documents in order of selection/ranking
        new_ctxs = []
        for rank_idx, doc_idx in enumerate(selected_indices):
            doc = docs[doc_idx].copy()
            # Add/overwrite score with the quality score and the ranking index for analysis
            doc["score"] = float(quality_scores[doc_idx])
            doc["dpp_rank"] = rank_idx + 1
            new_ctxs.append(doc)
            
        inst["ctxs"] = new_ctxs
        reranked_data.append(inst)

    # Save output
    logger.info(f"Writing reranked results to '{args.output_file}'...")
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output_file, 'w') as f:
        for inst in reranked_data:
            f.write(json.dumps(inst) + '\n')
            
    logger.info("Reranking completed successfully.")

if __name__ == "__main__":
    main()
