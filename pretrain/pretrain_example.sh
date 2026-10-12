DATA_ROOT="${DATA_ROOT:-../../MAGE_private_artifacts/data/pretrain}"

python run_pretraining.py \
  --tag little \
  --pretrain_max_lr 1e-4 \
  --batch_size 64 \
  --max_epochs 50 \
  --devices 0 \
  --seed 7 \
  --num_workers 32 \
  --data_backend lmdb \
  --train_lmdb "${DATA_ROOT}/train.lmdb" \
  --valid_lmdb "${DATA_ROOT}/val.lmdb" \
  --log_root "${MAGE_OUTPUT_ROOT:-../../MAGE_private_artifacts}/pretrain" \
  --time_length 6000 \
  --num_channels 19 \
  --patch_sizes 50,100,200 \
  --montage 19 \
  --mask_ratio 0.7 \
  --d_model 64 \
  --encoder_depth 6 \
  --predictor_depth 3 \
  --encoder_heads 4 \
  --predictor_heads 4 \
  --k_temporal 3 \
  --temporal_extra_convs 2 \
  --use_frequency \
  --aggregator_hidden_dim 32 \
  --aggregator_freq_dim 16 \
  --decoder_hidden_ratio 2.0 \
  --dropout 0.1 \
  --use_channel_attn \
  --use_rope_time \
  --no_inter_attn_pred \
  --no_inter_attn_enc \
  --loss_schedule scale_cosine \
  --use_loss_r \
  --use_align_vis \
  --use_align_mask \
  --lambda_recon 5.0 \
  --use_ln \
  --agg_noise_std 0.3 \
  --scale_dropout 0.3 \
  --entropy_lambda 0.0

echo "MAGE pretraining job completed!"
