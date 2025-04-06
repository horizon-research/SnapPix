import numpy as np
import time
import torch
from timm.models import create_model
from coded.coded_AR.models.c2d_pt import C2DPtSpeed as  C2DPt
from coded.coded_AR.models.c2d import C2D
from coded.coded_AR.models.c3d import C3D
from fvcore.nn import FlopCountAnalysis, parameter_count
import models.modeling_finetune
from coded.constant import K710_SSV2_MEAN, K710_SSV2_STD

# Function to create a model based on the specified target
def get_model(test_target):
    if test_target == "small-image-vit":
        model = create_model(
            "vit_small_patch8_112",
            pretrained=False,
            num_classes=174,
            all_frames=1,
            tubelet_size=1,
            drop_rate=0.0,
            drop_path_rate=0.0,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_checkpoint=None,
            use_mean_pooling=True,
            init_scale=0.001,
            img_size=112,
        )
    elif test_target == "large-image-vit":
        model = create_model(
            "vit_large_patch8_112",
            pretrained=False,
            num_classes=174,
            all_frames=1,
            tubelet_size=1,
            drop_rate=0.0,
            drop_path_rate=0.0,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_checkpoint=None,
            use_mean_pooling=True,
            init_scale=0.001,
            img_size=112,
        )
    elif test_target == "base-image-vit":
        model = create_model(
            "vit_base_patch8_112",
            pretrained=False,
            num_classes=174,
            all_frames=1,
            tubelet_size=1,
            drop_rate=0.0,
            drop_path_rate=0.0,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_checkpoint=None,
            use_mean_pooling=True,
            init_scale=0.001,
            img_size=112,
        )
    elif test_target == "tpami":
        model = C2DPt(
            num_classes=174,
            sample_size=112,
            sample_duration=16
        )
    elif test_target == "tiny-video-vit":
        model = create_model(
            "vit_tiny_patch8_112",
            pretrained=False,
            num_classes=174,
            all_frames=16,
            tubelet_size=2,
            drop_rate=0.0,
            drop_path_rate=0.0,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_checkpoint=None,
            use_mean_pooling=True,
            init_scale=0.001,
            img_size=112,
        )
    elif test_target == "super-tiny-video-vit":
        model = create_model(
            "vit_super_tiny_patch8_112",
            pretrained=False,
            num_classes=174,
            all_frames=16,
            tubelet_size=2,
            drop_rate=0.0,
            drop_path_rate=0.0,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_checkpoint=None,
            use_mean_pooling=True,
            init_scale=0.001,
            img_size=112,
        )
    elif test_target == "tpami-C2D":
            model = C2D(
                num_classes=174,
                sample_size=112,
                sample_duration=16
            )
    elif test_target == "tpami-C3D":
        model = C3D(
            num_classes=174,
            sample_size=112,
            sample_duration=16
        )
    else:
        raise ValueError(f"Unknown model target: {test_target}")

    return model.cuda().eval()

# Function to calculate and print FLOPs, parameter count, and FPS
def evaluate_model(test_target, bs=64, warmup=10, test_num=100, save_name=None):
    print(f"\nEvaluating model: {test_target}")
    model = get_model(test_target)
    
    # Define input tensor shape based on the model type
    if test_target in ["tpami", "tpami-C2D"]:
        # input_tensor = torch.randn(bs, 1, 112, 112).cuda()
        # input is U(0,1)
        input_tensor = torch.randn(bs, 1, 112, 112).float().cuda()
    elif test_target in ["small-image-vit", "large-image-vit", "base-image-vit"]:
        input_tensor = torch.rand(bs, 1, 1, 112, 112).float().cuda()
    else:
        input_tensor = torch.rand(bs, 1, 16, 112, 112).float().cuda()

    # prepare norm mean and std
    if test_target not in ["small-image-vit", "large-image-vit", "base-image-vit"]:
        mean = torch.tensor(K710_SSV2_MEAN).cuda()
        std = torch.tensor(K710_SSV2_STD).cuda()
    else: # get a torch.int(bs, 1, 1, 112, 112) map betweeon 1-16, which int but float type
        mean = torch.tensor(K710_SSV2_MEAN).expand(bs, 1, 1, 112, 112).float().cuda()
        std = torch.tensor(K710_SSV2_STD).expand(bs, 1, 1, 112, 112).float().cuda()
        # mutlipy std by a 1-16 random int map, which is a torch.float(bs, 1, 1, 112, 112)
        # but it is tiled by a 8x8 random tile map
        per_pixel_norm_map = torch.randint(1, 17, (bs, 1, 1, 8, 8)).float().cuda()
        std = std * per_pixel_norm_map.repeat(1, 1, 1, 14, 14)

    # Compute FLOPs and parameter count
    # flop_analysis = FlopCountAnalysis(model, input_tensor)
    # flops = flop_analysis.total()
    # param_count = parameter_count(model)

    # Warmup runs to stabilize GPU performance
    for _ in range(warmup):
        with torch.no_grad():
            # norm 
            input_tensor = (input_tensor - mean) / std
            _ = model(input_tensor)
            torch.cuda.synchronize()

    # Measure FPS
    start = time.time()
    for _ in range(test_num):
        with torch.no_grad():
            # norm
            input_tensor = (input_tensor - mean) / std
            _ = model(input_tensor)
            torch.cuda.synchronize()
    end = time.time()

    fps = test_num * bs / (end - start)
    print(f"Model: {test_target}")
    # print(f"Total FLOPs (G): {flops / 1e9}")
    # print(f"Total Parameters (M): {param_count[''] / 1e6}")
    print(f"FPS: {fps}")

    # also write to file
    with open(save_name, "a") as f:
        f.write(f"Model: {test_target}\n")
        # f.write(f"Total FLOPs (G): {flops / 1e9}\n")
        # f.write(f"Total Parameters (M): {param_count[''] / 1e6}\n")
        f.write(f"FPS: {fps}\n")
        # add latency
        f.write(f"Latency (ms):  {1000 / fps}\n\n")


import argparse

# Define the modes and corresponding parameters
modes = {
    "4090_1": ( ["super-tiny-video-vit", "base-image-vit", "small-image-vit", "tpami", "tpami-C2D", "tpami-C3D"], [1, 1, 1, 1, 1, 1], 100, 1000),
    "4090_64": ( ["super-tiny-video-vit", "base-image-vit", "small-image-vit", "tpami", "tpami-C2D", "tpami-C3D"], [64, 64, 64, 64, 64, 64], 20, 200),
    "jetson_1": (["super-tiny-video-vit", "base-image-vit", "small-image-vit", "tpami", "tpami-C2D", "tpami-C3D"], [1, 1, 1, 1, 1, 1], 20, 200),
    "jetson_16": (["super-tiny-video-vit", "base-image-vit", "small-image-vit", "tpami", "tpami-C2D", "tpami-C3D"], [16, 16, 16, 16, 16, 16], 5, 50),
}
parser = argparse.ArgumentParser(description="Evaluate models with different modes")
parser.add_argument("--mode", choices=modes.keys(), required=True, help="Mode of evaluation (e.g., 4090_1, 4090_64, jetson_1, jetson_16)")
args = parser.parse_args()

test_targets, batch_sizes, warmup,  test_num= modes[args.mode]

# clean log
save_name = "model_analysis_results_" + args.mode + ".txt"
with open(save_name, "w") as f:
    f.write("")
# Evaluate each model
for i in range(len(test_targets)):
    target = test_targets[i]
    bs = batch_sizes[i]

    # if target in ["super-tiny-video-vit"]:
    #     # power test
    #     evaluate_model(target, bs=bs, warmup=warmup, test_num=10000, save_name=save_name)
    # else:
    #     continue
    evaluate_model(target, bs=bs , warmup=warmup, test_num=test_num, save_name=save_name)
