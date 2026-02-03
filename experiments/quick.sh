#!/bin/bash
# Quick script to run REAP pruning on Step-3.5-Flash model
# Usage: bash experiments/quick.sh <GPU_IDS>
# Example: bash experiments/quick.sh 0,1,2,3

set -e

# GPU devices (required argument)
export CUDA_VISIBLE_DEVICES=${1:-0}
FIRST_DEVICE=$(echo "$1" | cut -d',' -f1)
port=$((8000 + FIRST_DEVICE))

# Model configuration
model_name="stepfun-ai/Step-3.5-Flash"
pruning_method="reap"
seed=42
compression_ratio=0.4  # 40% pruning
dataset_name="lkevincc0/glm47-math-code-calibration-1024"
num_samples=1024

# Evaluation flags (set to true to enable)
run_lm_eval=true
run_evalplus=true
run_livecodebench=false
run_math=true
run_wildbench=false

# Expert preservation settings
singleton_super_experts="false"
singleton_outlier_experts="false"

output_file_name="observations_${num_samples}_cosine-seed_${seed}.pt"
server_log_file_name="quick-step3p5-${FIRST_DEVICE}.log"

echo "=============================================="
echo "REAP Pruning: Step-3.5-Flash"
echo "=============================================="
echo "Model: $model_name"
echo "Dataset: $dataset_name"
echo "Compression ratio: $compression_ratio"
echo "Pruning method: $pruning_method"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "vLLM port: $port"
echo "Evaluations: lm_eval=$run_lm_eval, evalplus=$run_evalplus, livecodebench=$run_livecodebench, math=$run_math, wildbench=$run_wildbench"
echo "=============================================="

# Run pruning
python src/reap/prune.py \
    --model-name $model_name \
    --dataset-name $dataset_name \
    --compression-ratio $compression_ratio \
    --prune-method $pruning_method \
    --profile false \
    --vllm_port $port \
    --server-log-file-name $server_log_file_name \
    --do-eval false \
    --distance_measure cosine \
    --seed $seed \
    --output_file_name ${output_file_name} \
    --singleton_super_experts ${singleton_super_experts} \
    --singleton_outlier_experts ${singleton_outlier_experts} \
    --samples_per_category ${num_samples} \
    --record_pruning_metrics_only true

# Prepare model directory path for evaluation
short_model_name=$(echo $model_name | cut -d'/' -f2)
short_dataset_name=$(echo $dataset_name | cut -d'/' -f2)

pruned_model_dir_name="${pruning_method}"
if [[ "${singleton_super_experts}" == "true" ]]; then
    pruned_model_dir_name="${pruned_model_dir_name}-perserve_super"
elif [[ "${singleton_outlier_experts}" == "true" ]]; then
    pruned_model_dir_name="${pruned_model_dir_name}-perserve_outlier"
fi
pruned_model_dir_name="${pruned_model_dir_name}-seed_${seed}-${compression_ratio}"

model_dir="artifacts/${short_model_name}/${short_dataset_name}/pruned_models/${pruned_model_dir_name}"

echo "=============================================="
echo "Evaluating pruned model: ${model_dir}"
echo "=============================================="

bash experiments/eval.sh \
    $model_dir \
    $seed \
    $port \
    $server_log_file_name \
    ${run_lm_eval} \
    ${run_evalplus} \
    ${run_livecodebench} \
    ${run_math} \
    ${run_wildbench}

echo "=============================================="
echo "Finished! Pruned model saved to: ${model_dir}"
echo "=============================================="
