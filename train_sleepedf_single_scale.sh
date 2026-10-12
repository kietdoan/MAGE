#!/bin/bash
#SBATCH --job-name=mage_sleep_ss_s100
#SBATCH --output=../MAGE_private_artifacts/logs/mage_sleep_ss_s100_%j.out
#SBATCH --error=../MAGE_private_artifacts/logs/mage_sleep_ss_s100_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00

# Usage: sbatch train_sleepedf_single_scale.sh <scale_index 0|1|2> <fold 0..4>
mkdir -p ../MAGE_private_artifacts/logs
# Activate the mage environment before running or submitting this script.

unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_GPU
unset SLURM_MEM_PER_NODE

SCALE_INDEX=${1:?"usage: sbatch train_sleepedf_single_scale.sh <scale_index 0|1|2> <fold 0..4>"}
FOLD=${2:?"usage: sbatch train_sleepedf_single_scale.sh <scale_index 0|1|2> <fold 0..4>"}

echo "=== GPU Check ==="
nvidia-smi
echo "SCALE_INDEX=$SCALE_INDEX FOLD=$FOLD"
echo "================="

PRETRAIN_DIR="ckpts/mage.ckpt"

# --active_head_index: 0=Scale0(P50) 1=Scale1(P100) 2=Scale2(P200)  -1=Ensemble
python downstream/run_downstream_full.py \
  --task SLEEPEDF \
  --scheme mage \
  --split 5fold \
  --tag custom_64d_6L \
  --pretrain_ckpt "$PRETRAIN_DIR" \
  --epoch_select best \
  --load_pretrained \
  --batch_size 64 \
  --epochs 50 \
  --max_lr_head 1e-3 \
  --max_lr_enc 1e-4 \
  --num_workers 6 \
  --seq_len 6000 \
  --devices 0 \
  --freeze_encoder \
  --val_ratio 0.2 \
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
  --in_chans 2 \
  --seed 100 \
  --fold "$FOLD" \
  --sleepedf_path "${SLEEPEDF_PATH:-data/SLEEPEDF}" \
  --log_root "${MAGE_OUTPUT_ROOT:-../MAGE_private_artifacts}/results_seed100/single_scale_sleepedf" \
  --active_head_index "$SCALE_INDEX"
