#!/usr/bin/env bash
CODED_TYPE=$1
OUTPUT_DIR="local/coded_vits_reconstruction_"$CODED_TYPE""
DATA_PATH='../dataset/shared_list/SSV2_recon/train.csv'
VAL_PATH='../dataset/shared_list/SSV2_recon/val.csv'
CROSS_MODEL_PATH='local/vits_naive_learned_ssv2_scratch/checkpoint-99.pth'
set -x
unset SLURM_PROCID
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="0, 1"

python3 -m torch.distributed.launch --nproc_per_node=2 \
        --master_port="$MASTER_PORT" --nnodes=1 \
        run_coded_reconstruction_training.py \
        --data_path ${DATA_PATH} \
        --val_path ${VAL_PATH} \
        --mask_type tube \
        --mask_ratio 0.0 \
        --decoder_mask_type run_cell \
        --decoder_mask_ratio 0.0 \
        --model coded_pretrain_videomae_super_tiny_patch8_112 \
        --decoder_depth 4 \
        --batch_size 8 \
        --num_sample 4 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_workers 10 \
        --clip_grad 1.0 \
        --lr 1e-3 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 5 \
        --save_ckpt_freq 10 \
        --epochs 50 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --acc_iters 16 \
        --data_root '/localdisk2/dataset/mmdataset' \
        --local-rank 0 \
        --coded_template_folder "./decorrelation_training_wd0_norm_new" \
        --coded_type "$CODED_TYPE" \
        --cross_model_path "$CROSS_MODEL_PATH"
