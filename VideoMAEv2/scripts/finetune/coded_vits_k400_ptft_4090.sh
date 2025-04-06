# set CUDA_HOME
export CUDA_HOME=/workspace/cuda12.6
# add CUDA Toolkit to PATH
# add CUDA Toolkit to PATH
export PATH=$PATH:$CUDA_HOME/bin
# add CUDA lib to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export MODEL_PATH='local/coded_vits_pt_decor/checkpoint-299.pth'
CODED_TYPE="decor_fix"
CROSS_MODEL_PATH=""
CROSS_NAME=""
#!/usr/bin/env bash
set -x
unset SLURM_PROCID
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="0, 1"

OUTPUT_DIR="local/k400/vits_"$CODED_TYPE"_"$CROSS_NAME"_k400_ptft_no_clip"
DATA_PATH='../dataset/shared_list/K400'
# MODEL_PATH='video_super_tiny_pretrain/checkpoint-59.pth'

# Recipe: https://github.com/facebookresearch/SlowFast/blob/main/configs/Kinetics/MVIT_B_16x4_CONV.yaml
# batch_size can be adjusted according to the graphics card
python3 -m torch.distributed.launch --nproc_per_node=2 \
        --master_port="$MASTER_PORT" --nnodes=1 \
        run_coded_class_finetuning.py \
        --model coded_vit_small_patch8_112 \
        --finetune ${MODEL_PATH} \
        --data_set Kinetics-400 \
        --nb_classes 400 \
        --data_root '/localdisk2/dataset/mmdataset' \
        --data_path ${DATA_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 16 \
        --num_sample 2 \
        --input_size 112 \
        --short_side_size 112 \
        --save_ckpt_freq 20 \
        --num_frames 16 \
        --opt adamw \
        --lr 2e-3 \
        --num_workers 12 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --layer_decay 0.65 \
        --epochs 100 \
        --dist_eval \
        --test_num_segment 2 \
        --test_num_crop 3 \
        --local-rank 0 \
        --update_freq 8 \
        --warmup_epochs 15 \
        --coded_template_folder "./decorrelation_training_wd0_norm_new" \
        --coded_type "$CODED_TYPE" \
        --cross_model "$CROSS_MODEL_PATH"
