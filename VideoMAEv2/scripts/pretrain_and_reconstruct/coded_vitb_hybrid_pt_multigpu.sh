#!/usr/bin/env bash
CODED_TYPE=$1
OUTPUT_DIR="local/coded_vitb_pt_"$CODED_TYPE""
DATA_PATH='../dataset/shared_list/pretrained_combine.csv'
set -x
unset SLURM_PROCID
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="0, 1, 2, 3"

python3 -m torch.distributed.launch --nproc_per_node=4 \
        --master_port="$MASTER_PORT" --nnodes=1 \
        run_coded_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.85 \
        --decoder_mask_type run_cell \
        --decoder_mask_ratio 0.5 \
        --model coded_pretrain_videomae_base_patch8_112 \
        --decoder_depth 4 \
        --batch_size 32 \
        --num_sample 4 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_workers 10 \
        --lr 1e-3 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 30 \
        --save_ckpt_freq 20 \
        --epochs 300 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --acc_iters 16 \
        --data_root '/local_scratch/26293167/mmdataset/' \
        --local-rank 0 \
        --coded_template_folder "./decorrelation_training_wd0_norm_new" \
        --coded_type "$CODED_TYPE"
