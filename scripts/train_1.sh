#!/bin/bash
#SBATCH --job-name e3
#SBATCH -p tmpgpu
#SBATCH -w gpu01
#SBATCH --ntasks-per-node=8 # 每个节点上的任务数（即GPU数量）
#SBATCH --gres=gpu:8 # 每个任务请求1个GPU
#SBATCH --cpus-per-task=8
#conda activate e3nn
#module load anaconda/2021a
#module load cuda/12.3

export PYTHONNOUSERSITE=True    # prevent using packages from base

CUDA_VISIBLE_DEVICES=0 python -u main_re.py \
    --output-dir 'models/ForceCon/der.log' \
    --model-name 'graph_attention_transformer_nonlinear_l2_e3_noNorm_dx' \
    --input-irreps '86x0e' \
    --data-path 'datasets' \
    --run-fold 4 \
    --batch-size 1 \
    --epochs 150 \
    --radius 8.0 \
    --num-basis 86 \
    --drop-path 0.0 \
    --weight-decay 5e-3 \
    --lr 5e-5 \
    --min-lr 1e-6\
    --no-model-ema \
    --no-amp

# 设置MASTER_ADDR和MASTER_PORT
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

# 使用srun运行分布式训练脚本
# srun python -u main.py \
#     --output-dir 'models/ForceCon/' \
#     --model-name 'graph_attention_transformer_nonlinear_l2_e3_noNorm' \
#     --input-irreps '86x0e' \
#     --data-path 'datasets' \
#     --run-fold 10 \
#     --batch-size 64 \
#     --epochs 150 \
#     --radius 8.0 \
#     --num-basis 86 \
#     --drop-path 0.0 \
#     --weight-decay 5e-3 \
#     --lr 5e-5 \
#     --min-lr 1e-6 
