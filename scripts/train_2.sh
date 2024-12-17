#!/bin/bash
#SBATCH --job-name e3
#SBATCH -p tmpgpu
#SBATCH -w gpu01
#SBATCH --ntasks-per-node=8 # ÿ���ڵ��ϵ�����������GPU������
#SBATCH --gres=gpu:8 # ÿ����������1��GPU
#SBATCH --cpus-per-task=8
#conda activate e3nn
#module load anaconda/2021a
#module load cuda/12.3

export PYTHONNOUSERSITE=True    # prevent using packages from base
#CUDA_VISIBLE_DEVICES=1 
python -u ../main_re.py \
    --output-dir 'models/ForceCon2/der.log' \
    --model-name 'graph_attention_transformer_nonlinear_l2_e3_noNorm_dx' \
    --input-irreps '86x0e' \
    --data-path '../datasets' \
    --run-fold 4 \
    --batch-size 1 \
    --epochs 10 \
    --radius 8.0 \
    --num-basis 86 \
    --drop-path 0.0 \
    --weight-decay 5e-3 \
    --lr 5e-5 \
    --min-lr 1e-6\
    --no-model-ema \
    --no-amp

# ����MASTER_ADDR��MASTER_PORT
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

# ʹ��srun���зֲ�ʽѵ���ű�
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
