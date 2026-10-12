#!/bin/bash
#SBATCH --job-name=mage_faced_single_scale_1_s100
#SBATCH --output=../MAGE_private_artifacts/logs/mage_faced_single_scale_1_s100_%j.out
#SBATCH --error=../MAGE_private_artifacts/logs/mage_faced_single_scale_1_s100_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00

mkdir -p ../MAGE_private_artifacts/logs
# Activate the mage environment before running or submitting this script.

unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_GPU
unset SLURM_MEM_PER_NODE

echo "=== GPU Check ==="
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"
echo "================="

PRETRAIN_DIR="ckpts/mage.ckpt"

# --- CONFIGURATION NOTES ---
# --active_head_index:
#    0 = Scale 0 (Patch Size 50)
#    1 = Scale 1 (Patch Size 100)
#    2 = Scale 2 (Patch Size 200)
#   -1 = Ensemble (all heads + Aggregator)

python downstream/run_downstream_full.py \
  --task FACED \
  --scheme mage \
  --split 4fold \
  --tag custom_64d_6L \
  --pretrain_ckpt "$PRETRAIN_DIR" \
  --epoch_select best \
  --load_pretrained \
  --batch_size 64 \
  --epochs 50 \
  --max_lr_head 1e-3 \
  --max_lr_enc 1e-4 \
  --num_workers 16 \
  --seq_len 2000 \
  --devices 0 \
  --freeze_encoder \
  --mage_patch_sizes 50 100 200 \
  --mage_k_temporal 3 \
  --mage_temporal_extra_convs 2 \
  --mage_use_spectral \
  --mage_use_channel_attn \
  --mage_aggregator_hidden_dim 32 \
  --mage_use_frequency \
  --mage_aggregator_freq_dim 16 \
  --mage_no_inter_attn_enc \
  --rope_time \
  --scale_div 1.0 \
  --use_chan_conv \
  --chan_max_norm 1.0 \
  --head_type eegpt_probe \
  --dropout 0.5 \
  --in_chans 32 \
  --seed 100 \
  --faced_path "${FACED_PATH:-data/FACED}" \
  --log_root "${MAGE_OUTPUT_ROOT:-../MAGE_private_artifacts}/results_seed100/single_scale_faced" \
  --active_head_index 1
