# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import json
import os
import random
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path

import deepspeed
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.utils import ModelEma
from coded.coded_AR.models.c3d import C3D
from coded.coded_AR.models.c2d_pt import C2DPt

from coded.coded_AR.models.binarized_modules import Exposuref as pami_coded_layer
from coded.coded_layer import CodedLayer as our_coded_layer
from coded.coded_layer import CodedBaseline as our_coded_baseline_layer
from coded.coded_layer import CodedAblatLayer as our_coded_ablat_layer
# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
import utils
from dataset import build_dataset
from engine_for_coded_finetuning import (
    final_test,
    merge,
    train_one_epoch,
    validation_one_epoch,
)
from optim_factory import (
    LayerDecayValueAssigner,
    create_optimizer,
    get_parameter_groups,
)
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_samples_collate


from run_decorrelation_training import record_pattern

def get_args():
    parser = argparse.ArgumentParser(
        'VideoMAE fine-tuning and evaluation script for action classification',
        add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model parameters
    parser.add_argument(
        '--model',
        default='',
        type=str,
        metavar='MODEL',
        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default=1)
    parser.add_argument(
        '--input_size', default=112, type=int, help='images input size')

    parser.add_argument(
        '--with_checkpoint', action='store_true', default=False)

    parser.add_argument(
        '--drop',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Dropout rate (default: 0.)')
    parser.add_argument(
        '--attn_drop_rate',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Attention dropout rate (default: 0.)')
    parser.add_argument(
        '--drop_path',
        type=float,
        default=0.1,
        metavar='PCT',
        help='Drop path rate (default: 0.1)')
    parser.add_argument(
        '--head_drop_rate',
        type=float,
        default=0.0,
        metavar='PCT',
        help='cls head dropout rate (default: 0.)')

    parser.add_argument(
        '--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument(
        '--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument(
        '--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument(
        '--opt',
        default='adamw',
        type=str,
        metavar='OPTIMIZER',
        help='Optimizer (default: "adamw"')
    parser.add_argument(
        '--opt_eps',
        default=1e-8,
        type=float,
        metavar='EPSILON',
        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument(
        '--opt_betas',
        default=None,
        type=float,
        nargs='+',
        metavar='BETA',
        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument(
        '--clip_grad',
        type=float,
        default=None,
        metavar='NORM',
        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        metavar='M',
        help='SGD momentum (default: 0.9)')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.05,
        help='weight decay (default: 0.05)')
    parser.add_argument(
        '--weight_decay_end',
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        metavar='LR',
        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument(
        '--warmup_lr',
        type=float,
        default=1e-8,
        metavar='LR',
        help='warmup learning rate (default: 1e-6)')
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-6,
        metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=5,
        metavar='N',
        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=-1,
        metavar='N',
        help='num of steps to warmup LR, will overload warmup_epochs if set > 0'
    )

    # Augmentation parameters
    parser.add_argument(
        '--color_jitter',
        type=float,
        default=0.4,
        metavar='PCT',
        help='Color jitter factor (default: 0.4)')
    parser.add_argument(
        '--num_sample', type=int, default=2, help='Repeated_aug (default: 2)')
    parser.add_argument(
        '--aa',
        type=str,
        default='rand-m7-n4-mstd0.5-inc1',
        metavar='NAME',
        help=
        'Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'
    ),
    parser.add_argument(
        '--smoothing',
        type=float,
        default=0.1,
        help='Label smoothing (default: 0.1)')
    parser.add_argument(
        '--train_interpolation',
        type=str,
        default='bicubic',
        help=
        'Training interpolation (random, bilinear, bicubic default: "bicubic")'
    )

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=112)
    parser.add_argument('--test_num_segment', type=int, default=10)
    parser.add_argument('--test_num_crop', type=int, default=3)

    # * Random Erase params
    parser.add_argument(
        '--reprob',
        type=float,
        default=0.25,
        metavar='PCT',
        help='Random erase prob (default: 0.25)')
    parser.add_argument(
        '--remode',
        type=str,
        default='pixel',
        help='Random erase mode (default: "pixel")')
    parser.add_argument(
        '--recount',
        type=int,
        default=1,
        help='Random erase count (default: 1)')
    parser.add_argument(
        '--resplit',
        action='store_true',
        default=False,
        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument(
        '--mixup',
        type=float,
        default=0.8,
        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument(
        '--cutmix',
        type=float,
        default=1.0,
        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument(
        '--cutmix_minmax',
        type=float,
        nargs='+',
        default=None,
        help='cutmix min/max ratio, overrides alpha and enables cutmix if set')
    parser.add_argument(
        '--mixup_prob',
        type=float,
        default=1.0,
        help=
        'Probability of performing mixup or cutmix when either/both is enabled'
    )
    parser.add_argument(
        '--mixup_switch_prob',
        type=float,
        default=0.5,
        help=
        'Probability of switching to cutmix when both mixup and cutmix enabled'
    )
    parser.add_argument(
        '--mixup_mode',
        type=str,
        default='batch',
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"'
    )

    # * Finetuning params
    parser.add_argument(
        '--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument(
        '--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument(
        '--data_path',
        default='/your/data/path/',
        type=str,
        help='dataset path')
    parser.add_argument(
        '--data_root', default='', type=str, help='dataset path root')
    parser.add_argument(
        '--eval_data_path',
        default=None,
        type=str,
        help='dataset path for evaluation')
    parser.add_argument(
        '--nb_classes',
        default=400,
        type=int,
        help='number of the classification types')
    parser.add_argument(
        '--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--sparse_sample', default=False, action='store_true')
    parser.add_argument(
        '--data_set',
        default='Kinetics-400',
        choices=[
            'Kinetics-400', 'Kinetics-600', 'Kinetics-700', 'SSV2', 'UCF101',
            'HMDB51', 'Diving48', 'Kinetics-710', 'MIT'
        ],
        type=str,
        help='dataset')
    parser.add_argument(
        '--fname_tmpl',
        default='img_{:05}.jpg',
        type=str,
        help='filename_tmpl for rawframe dataset')
    parser.add_argument(
        '--start_idx',
        default=1,
        type=int,
        help='start_idx for rwaframe dataset')

    parser.add_argument(
        '--output_dir',
        default='',
        help='path where to save, empty for no saving')
    parser.add_argument(
        '--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument(
        '--device',
        default='cuda',
        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument(
        '--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument(
        '--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument(
        '--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument(
        '--validation', action='store_true', help='Perform validation only')
    parser.add_argument(
        '--dist_eval',
        action='store_true',
        default=False,
        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help=
        'Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        '--world_size',
        default=1,
        type=int,
        help='number of distributed processes')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument(
        '--dist_url',
        default='env://',
        help='url used to set up distributed training')

    parser.add_argument(
        '--enable_deepspeed', action='store_true', default=False)
    
    parser.add_argument(
        '--debug', action='store_true', default=False)
    parser.add_argument('--coded_type', default="decor_fix", type=str,
                        choices=["decor_fix", "decor_learn", "long", "short", "naive", "naive_learned", "sparse", "pami", "no_norm", "global", "cross_ar", "cross_rec", "cross_big_rec"])
    parser.add_argument('--coded_template_folder', type=str)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--cross_model_path', default='', type=str)
    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        parser = deepspeed.add_config_arguments(parser)
        ds_init = deepspeed.initialize
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(
        is_train=True, test_mode=False, args=args)
    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(
            is_train=False, test_mode=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    if not args.debug:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval and not args.debug:
        if len(dataset_val) % num_tasks != 0:
            print(
                'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                'This will slightly alter validation results as extra duplicate entries are added to achieve '
                'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.num_sample > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None

    if args.debug:
        persistent_workers = False
    else:
        persistent_workers = True

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
        persistent_workers=persistent_workers)

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=persistent_workers)
    else:
        data_loader_val = None


    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes)


    if args.model not in ['svc2d']:
        model = create_model(
            args.model,
            img_size=args.input_size,
            pretrained=False,
            num_classes=args.nb_classes,
            all_frames=1,
            tubelet_size=1,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            head_drop_rate=args.head_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            with_cp=args.with_checkpoint,
        )
    elif args.model == 'svc2d':
        model = C2DPt(
            num_classes=args.nb_classes,
            sample_size=args.input_size,
            sample_duration=args.num_frames,
        )

    # initialize coded layer
    if args.coded_type in ["decor_fix", "decor_learn"]:
        coded_layer = our_coded_layer(
            init_pattern_path=os.path.join(args.coded_template_folder, "pattern_4.pth"),
        )
        if args.coded_type == "decor_fix":
            coded_layer.opt_pattern = False
        else:
            coded_layer.opt_pattern = True
    elif args.coded_type == "long":
        coded_layer = our_coded_baseline_layer(long_exposure=True)
    elif args.coded_type == "short":
        coded_layer = our_coded_baseline_layer(long_exposure=False)
    elif args.coded_type == "naive":
        coded_layer = our_coded_layer(
            init_pattern_path=os.path.join(args.coded_template_folder, "pattern_init.pth"),
        )
        coded_layer.opt_pattern = False
    elif args.coded_type == "naive_learned":
        coded_layer = our_coded_layer(
            init_pattern_path=None, # random initialization
        )
        coded_layer.opt_pattern = True
    elif args.coded_type == "sparse":
        coded_layer = our_coded_layer(
            init_pattern_path=os.path.join(args.coded_template_folder, "pattern_init_sparse.pth"),
        )
        coded_layer.opt_pattern = False
    elif args.coded_type == "pami":
        coded_layer = pami_coded_layer()
    elif args.coded_type == "no_norm":
        coded_layer = our_coded_ablat_layer(init_pattern_path=os.path.join(args.coded_template_folder, "pattern_init.pth"), pixel_wise_norm=False)
        coded_layer.opt_pattern = False
    elif args.coded_type == "global":
        coded_layer = our_coded_ablat_layer(init_pattern_path=None, pixel_wise_norm=True, global_pattern=True)
        coded_layer.opt_pattern = False
    elif args.coded_type in ["cross_ar", "cross_rec", "cross_big_rec"]:
        coded_layer = our_coded_layer(
                    init_pattern_path=None, # random initialization
                )
        if  not os.path.exists(args.cross_model_path):
            raise ValueError("Cross model path does not exist: %s" % args.cross_model_path)
        print("Loading checkpoint from %s" % args.cross_model_path)
        checkpoint = torch.load(args.cross_model_path, map_location='cpu')
        # 仅加载特定的权重
        # 
        if 'coded_layer.coded_weight' in checkpoint['model']:
            coded_layer.coded_weight.data = checkpoint['model']['coded_layer.coded_weight']
        elif 'coded_layer.weight' in checkpoint['model']:
            coded_layer.coded_weight.data = checkpoint['model']['coded_layer.weight']
        else:
            raise KeyError("Neither 'coded_layer.coded_weight' nor 'coded_layer.weight' found in the checkpoint.")

        coded_layer.opt_pattern = False

        # print "saving pattern"
        print("Saving Cross pattern")

        record_pattern(coded_layer, 0, args.output_dir)
    else:
        raise ValueError("Unknown init coded layer type: %s" % args.coded_type)
    


    model.coded_layer = coded_layer

    if args.model not in ['svc2d', 'c3d']:
        patch_size = model.patch_embed.patch_size
    else:
        patch_size = (8,8)
    print("Patch size = %s" % str(patch_size))

    args.window_size = (args.num_frames // args.tubelet_size,
                        args.input_size // patch_size[0],
                        args.input_size // patch_size[1])

    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        for old_key in list(checkpoint_model.keys()):
            if old_key.startswith('_orig_mod.'):
                new_key = old_key[10:]
                checkpoint_model[new_key] = checkpoint_model.pop(old_key)

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        utils.load_state_dict(
            model, checkpoint_model, prefix=args.model_prefix)


    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * num_tasks
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.lr * total_batch_size / 256
    #########scale the lr#############
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    #########scale the lr#############
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" %
          num_training_steps_per_epoch)
    

    if args.model not in ['svc2d']:
        num_layers = model_without_ddp.get_num_layers()
        if args.layer_decay < 1.0:
            assigner = LayerDecayValueAssigner(
                list(args.layer_decay**(num_layers + 1 - i)
                    for i in range(num_layers + 2)))
        else:
            assigner = None

        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        skip_weight_decay_list = model.no_weight_decay()
        print("Skip weight decay list: ", skip_weight_decay_list)
    else:
        assigner = None
        skip_weight_decay_list = model.no_weight_decay()

    if args.enable_deepspeed:
        # raise error
        raise ValueError("DeepSpeed is not supported for this script!")
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args,
            model=model,
            model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" %
              model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            if args.model not in ['svc2d', 'c3d']:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu], find_unused_parameters=True)
            else:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu], find_unused_parameters=True)
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                # sync_bn(model)

            model_without_ddp = model.module



        optimizer = create_optimizer(
            args,
            model_without_ddp,
            skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id
            if assigner is not None else None,
            get_layer_scale=assigner.get_scale
            if assigner is not None else None)
        loss_scaler = NativeScaler()



    # Check if coded_layer.coded_weight is in the skip list for weight decay
    # skip_param_found = False
    # for param_group in optimizer.param_groups:
    #     # Check if this param group has weight decay set to 0
    #     if param_group.get('weight_decay', 1) == 0:
    #         # Iterate over all parameters in this group to see if coded_weight is included
    #         for param in param_group['params']:
    #             if param is model.coded_layer.weight: # pami with lr scale fix
    #             # if param is model.coded_layer.coded_weight: #my
    #                 # print lr scale
    #                 print("lr scale for coded_weight = %.8f" % param_group['lr_scale'])
    #                 skip_param_found = True
    #                 break

    # if skip_param_found:
    #     print("Weight decay was successfully skipped for model.coded_layer.coded_weight.")
    # else:
    #     print("model.coded_layer.coded_weight is not in the skip list for weight decay.")
    
    # import ipdb; ipdb.set_trace()


    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay,
                                                args.weight_decay_end,
                                                args.epochs,
                                                num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" %
          (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        model_ema=model_ema)
    if args.validation:
        test_stats = validation_one_epoch(data_loader_val, model, device)
        print(
            f"{len(dataset_val)} val images: Top-1 {test_stats['acc1']:.2f}%, Top-5 {test_stats['acc5']:.2f}%, loss {test_stats['loss']:.4f}"
        )
        exit(0)
    
    

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch *
                                args.update_freq)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
        )
        if args.output_dir and args.save_ckpt:
            _epoch = epoch + 1
            if _epoch % args.save_ckpt_freq == 0 or _epoch == args.epochs:
                utils.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    model_ema=model_ema)
        if data_loader_val is not None:
            test_stats = validation_one_epoch(data_loader_val, model, device)
            print(
                f"Accuracy of the network on the {len(dataset_val)} val images: {test_stats['acc1']:.2f}%"
            )
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best",
                        model_ema=model_ema)

            print(f'Max accuracy: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.update(
                    val_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(
                    val_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(
                    val_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {
                **{f'train_{k}': v
                   for k, v in train_stats.items()},
                **{f'val_{k}': v
                   for k, v in test_stats.items()}, 'epoch': epoch,
                'n_parameters': n_parameters
            }
        else:
            log_stats = {
                **{f'train_{k}': v
                   for k, v in train_stats.items()}, 'epoch': epoch,
                'n_parameters': n_parameters
            }
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                    os.path.join(args.output_dir, "log.txt"),
                    mode="a",
                    encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
