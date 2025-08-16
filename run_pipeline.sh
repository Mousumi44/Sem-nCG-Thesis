#!/bin/bash
# Usage: ./run_pipeline.sh <MODEL_NAME>

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <MODEL_NAME>"
    echo "Available models: sbert-mini, use, simcse, senticse, roberta, gemma, llama3.2, mistral, etc."
    exit 1
fi

MODEL_NAME="$1"

echo "Running nDCG-abs evaluation for model: $MODEL_NAME"
python src/ndcg_abs.py "$MODEL_NAME"

if [ $? -eq 0 ]; then
    echo "nDCG-abs evaluation completed successfully!"
    echo "Results saved to output/nDCG_scores.csv"
else
    echo "nDCG-abs evaluation failed for model: $MODEL_NAME"
    exit 1
fi
