#!/bin/bash

# Configuration
WIKI_DIR="./data/wikipedia_qwen_4b"
OUTPUT_DIR="./outputs"
TOPK=5
STAGE1_K=100
THETA=2.0
LAMBDA_VAL=0.6
DEVICE="cuda"

echo "=== Starting End-to-End Wikipedia Retrieval & DPP Reranking ==="
echo "Local Wikipedia Directory: ${WIKI_DIR}"
echo "Stage 1 Candidate Retrieval count: ${STAGE1_K}"
echo "Stage 2 Diverse Selection count (Top K): ${TOPK}"
echo "Parameters -> Theta: ${THETA} | Lambda: ${LAMBDA_VAL}"

mkdir -p ${OUTPUT_DIR}

# Loop over the BERDS benchmark datasets
# (Expected files from standard BERDS test data: "arguana_generated.jsonl" "kialo.jsonl" "opinionqa.jsonl")
for DATA_NAME in "arguana_generated" "kialo" "opinionqa"
do
    # Assuming standard BERDS dataset files are in an input folder, adjust path as necessary
    INPUT_FILE="./Data/${DATA_NAME}/${DATA_NAME}.test.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/${DATA_NAME}_dpp_wiki.jsonl"
    
    echo "----------------------------------------"
    echo "Processing dataset: ${DATA_NAME}"
    
    # Fall back to HF dataset name if local file is missing
    if [ ! -f "${INPUT_FILE}" ]; then
        echo "Local file ${INPUT_FILE} not found. Will fetch from Hugging Face dataset '${DATA_NAME}'..."
        INPUT_FILE="${DATA_NAME}"
    fi

    # Run the end-to-end retrieval and reranker
    # Set --download_data on the first iteration or keep it to ensure all cached files are checked
    python reranking/retrieve_and_dpp_qwen.py \
        --data "${INPUT_FILE}" \
        --output_file "${OUTPUT_FILE}" \
        --download_data \
        --local_wiki_dir "${WIKI_DIR}" \
        --diskann_index_path "${WIKI_DIR}/diskann" \
        --stage1_k ${STAGE1_K} \
        --topk ${TOPK} \
        --theta ${THETA} \
        --lambda_val ${LAMBDA_VAL} \
        --device "${DEVICE}"
        
    echo "Output saved to ${OUTPUT_FILE}."
    
    # Run the BERDS automatic evaluator
    EVAL_OUTPUT="${OUTPUT_DIR}/${DATA_NAME}_dpp_wiki.mistralpred"
    echo "Running BERDS automatic evaluation..."
    
    PYTHONPATH=. python Eval/eval_vllm.py \
        --data "${OUTPUT_FILE}" \
        --output_file "${EVAL_OUTPUT}" \
        --instructions Eval/instructions_chat.txt \
        --model timchen0618/Mistral_BERDS_evaluator_full \
        --model_type mistral \
        --topk ${TOPK}
        
    echo "Evaluation complete. Results saved in ${EVAL_OUTPUT}."
done

echo "=== End-to-End Pipeline Tasks Finished ==="
