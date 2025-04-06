# set CUDA_HOME
export CUDA_HOME=/software/cuda/12.1/
# add CUDA Toolkit to PATH
export PATH=$PATH:$CUDA_HOME/bin
# add CUDA lib to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

#!/usr/bin/env bash
set -x
unset SLURM_PROCID
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="0"

OUTPUT_DIR='local/k400/svc2d'
DATA_PATH='../dataset/shared_list/K400'
# MODEL_PATH='video_super_tiny_pretrain/checkpoint-59.pth'


# Recipe: https://github.com/facebookresearch/SlowFast/blob/main/configs/Kinetics/MVIT_B_16x4_CONV.yaml
# batch_size can be adjusted according to the graphics card
python3 -m torch.distributed.launch --nproc_per_node=1 \
        --master_port="$MASTER_PORT" --nnodes=1 \
        run_coded_class_finetuning.py \
        --model svc2d \
        --data_set Kinetics-400 \
        --nb_classes 400 \
        --data_root '/local_scratch/26477563/mmdataset'  \
        --data_path ${DATA_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 64 \
        --num_sample 2 \
        --input_size 112 \
        --short_side_size 112 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --opt adamw \
        --lr 2e-4 \
        --clip_grad 1.0 \
        --num_workers 12 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.001 \
        --layer_decay 1.0 \
        --epochs 200 \
        --test_num_segment 2 \
        --test_num_crop 3 \
        --local-rank 0 \
        --update_freq 1 \
        --warmup_epochs 30 \
        --coded_template_folder "./decorrelation_training_wd0_norm_new" \
        --coded_type "pami" \
        --validation
