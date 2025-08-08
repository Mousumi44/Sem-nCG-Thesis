#!/bin/bash
# Usage: ./run_pipeline.sh <MODEL_NAME>

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <MODEL_NAME>"
    exit 1
fi

MODEL_NAME="$1"

python src/main.py "$MODEL_NAME"
if [ $? -eq 0 ]; then
    python src/save_results.py
else
    echo "main.py failed, not running save_results.py"
    exit 1
fi
