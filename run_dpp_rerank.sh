#!/bin/bash

# Configuration
CORPUS="wiki"
RETRIEVER="bm25"
ROOT_DIR="/path/to/retrieval_outputs/${CORPUS}/${RETRIEVER}"
MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
TOPK=5
THETA=2.0
LAMBDA_VAL=0.5
MODE="greedy"

echo "=== Starting Qwen3-Embedding DPP Reranking ==="
echo "Corpus: ${CORPUS} | Retriever: ${RETRIEVER}"
echo "Model: ${MODEL_NAME}"
echo "Theta (relevance weight): ${THETA} | Lambda (perspective weight): ${LAMBDA_VAL}"
echo "Selection Mode: ${MODE} | Top K: ${TOPK}"

# Loop over BERDS datasets
for DATA_NAME in "arguana_generated" "kialo" "opinionqa"
do
    INPUT_FILE="${ROOT_DIR}/${DATA_NAME}.jsonl"
    OUTPUT_FILE="${ROOT_DIR}/${DATA_NAME}_dpp_qwen.jsonl"
    
    echo "----------------------------------------"
    echo "Processing ${DATA_NAME}..."
    
    if [ ! -f "${INPUT_FILE}" ]; then
        echo "Warning: Input file ${INPUT_FILE} not found. Skipping."
        continue
    fi

    # Run the reranker
    python reranking/dpp_rerank_qwen.py \
        --data "${INPUT_FILE}" \
        --output_file "${OUTPUT_FILE}" \
        --model_name "${MODEL_NAME}" \
        --engine transformers \
        --device cuda \
        --topk ${TOPK} \
        --theta ${THETA} \
        --lambda_val ${LAMBDA_VAL} \
        --mode ${MODE}
        
    echo "Output saved to ${OUTPUT_FILE}."
    
    # Run Evaluation
    echo "Running automatic evaluation on ${DATA_NAME}..."
    EVAL_OUTPUT="${ROOT_DIR}/${DATA_NAME}_dpp_qwen.mistralpred"
    
    PYTHONPATH=. python Eval/eval_vllm.py \
        --data "${OUTPUT_FILE}" \
        --output_file "${EVAL_OUTPUT}" \
        --instructions Eval/instructions_chat.txt \
        --model timchen0618/Mistral_BERDS_evaluator_full \
        --model_type mistral \
        --topk ${TOPK}
        
    echo "Evaluation complete. Results saved in ${EVAL_OUTPUT}."
done

echo "=== All tasks finished ==="
