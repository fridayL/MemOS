#!/bin/bash

cd "$(dirname "$0")/halumem"

echo "============================================"
echo "  HaluMem Evaluation Pipeline"
echo "============================================"

# Step 1: Add dialogues + Search memories + Generate QA responses
echo ""
echo "[Step 1/2] Running eval_memos.py (Add + Search + QA Response)..."
echo "This step may take a long time depending on the number of users."
echo ""
conda run --no-capture-output -n memos_env python -u eval_memos.py
if [ $? -ne 0 ]; then
    echo "Error running eval_memos.py"
    exit 1
fi

# Step 2: LLM-as-Judge evaluation + Metric aggregation
echo ""
echo "[Step 2/2] Running evaluation.py (LLM-as-Judge + Metrics)..."
echo ""
conda run --no-capture-output -n memos_env python -u evaluation.py --frame memos --version default
if [ $? -ne 0 ]; then
    echo "Error running evaluation.py"
    exit 1
fi

echo ""
echo "============================================"
echo "  All steps completed successfully!"
echo "  Results: results/memos-default/memos_eval_stat_result.json"
echo "============================================"
