import os
import json
import numpy as np # noqa
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, RGB2Gray, LowResolution)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop
from spatio_temporal_transforms import Coded, Averaged, OneFrame, ToTemporal, ToRepeat
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompos # noqa
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test

if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file, indent=2)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    from torch.backends import cudnn
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu" if opt.no_cuda else "cuda")
    if not opt.no_cuda:
        criterion = criterion.to(device)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        if opt.dataset in ['gtea', 'kth2']:
            spatial_transform = Compose([
                crop_method,
                RandomHorizontalFlip(),
                ToTensor(opt.norm_value), norm_method,
            ])
        elif opt.dataset in ['sth']:
            spatial_transform = Compose([
                crop_method,
                RGB2Gray(),
                ToTensor(opt.norm_value), norm_method,
            ])
        else:
            spatial_transform = Compose([
                crop_method,
                RandomHorizontalFlip(),
                RGB2Gray(),
                ToTensor(opt.norm_value), norm_method,
            ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        if opt.compress == 'mask':
            spatio_temporal_transform = Coded(opt.mask_path)
        elif opt.compress == 'avg':
            spatio_temporal_transform = Averaged()
        elif opt.compress == 'one':
            spatio_temporal_transform = OneFrame()
        elif opt.compress == 'spatial':
            if opt.dataset in ['gtea', 'kth2']:
                spatial_transform = Compose([
                    crop_method,
                    RandomHorizontalFlip(),
                    LowResolution(opt.spatial_compress_size, use_cv2=opt.use_cv2),
                    ToTensor(opt.norm_value), norm_method,
                ])
            elif opt.dataset in ['sth']:
                spatial_transform = Compose([
                    crop_method,
                    RGB2Gray(),
                    LowResolution(opt.spatial_compress_size, use_cv2=opt.use_cv2),
                    ToTensor(opt.norm_value), norm_method,
                ])
            else:
                spatial_transform = Compose([
                    crop_method,
                    RandomHorizontalFlip(),
                    RGB2Gray(),
                    LowResolution(opt.spatial_compress_size, use_cv2=opt.use_cv2),
                    ToTensor(opt.norm_value), norm_method,
                ])
            spatio_temporal_transform = None
        elif opt.compress == 'temporal':
            spatio_temporal_transform = Compose([
                Coded(opt.mask_path),
                ToTemporal(opt.mask_path),
            ])
        elif opt.compress == 'mask_3d':
            spatio_temporal_transform = Compose([
                ToRepeat(Coded(opt.mask_path), opt.sample_duration),
            ])
        elif opt.compress == 'avg_3d':
            spatio_temporal_transform = Compose([
                ToRepeat(Averaged(), opt.sample_duration),
            ])
        elif opt.compress == 'one_3d':
            spatio_temporal_transform = Compose([
                ToRepeat(OneFrame(), opt.sample_duration),
            ])
        else:
            spatio_temporal_transform = None
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform,
                                         spatio_temporal_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'top1', 'top5', 'lr', 'batch_time', 'data_time'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'top1', 'top5', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        if opt.optimizer == 'sgd':
            if opt.model in ["c2d_pt"]:
                for i, (name, param) in enumerate(model.named_parameters()):
                    print('{}: {}({})'.format(i, name, param.shape))
                params = []
                for i, p in enumerate(parameters):
                    if i < 128:
                        params.append({
                            "params": p,
                            "lr": opt.learning_rate * 64 * opt.lr_pt_rate})
                    else:
                        params.append({"params": p})
                optimizer = optim.SGD(
                    params,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=opt.nesterov)
            elif opt.model in ['c2d_pt_exp']:
                for i, (name, param) in enumerate(model.named_parameters()):
                    print('{}: {}({})'.format(i, name, param.shape))
                params = []
                for i, p in enumerate(parameters):
                    if 0 < i < 129:
                        params.append({
                            "params": p,
                            "lr": opt.learning_rate * 64 * opt.lr_pt_rate})
                    else:
                        params.append({"params": p})
                optimizer = optim.SGD(
                    params,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=opt.nesterov)
            else:
                optimizer = optim.SGD(
                    parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=opt.nesterov)
        elif opt.optimizer == 'adam':
            if opt.model in ['c2d_pt_exp']:
                for i, (name, param) in enumerate(model.named_parameters()):
                    print('{}: {}({})'.format(i, name, param.shape))
                params = []
                for i, p in enumerate(parameters):
                    if 0 < i < 129:
                        params.append({
                            "params": p,
                            "lr": opt.learning_rate * 64 * opt.lr_pt_rate})
                    else:
                        params.append({"params": p})
            elif opt.model in ['resnet18_pt_exp',
                               'resnet34_pt_exp',
                               'resnet50_pt_exp',
                               'resnet101_pt_exp',
                               'resnet152_pt_exp',
                               'resnext50_32x4d_pt_exp',
                               'resnext101_32x8d_pt_exp',
                               'wide_resnet50_2_pt_exp',
                               'wide_resnet101_2_pt_exp']:
                for i, (name, param) in enumerate(model.named_parameters()):
                    print('{}: {}({})'.format(i, name, param.shape))
                params = []
                for i, p in enumerate(parameters):
                    if 0 < i < 17:
                        params.append({
                            "params": p,
                            "lr": opt.learning_rate * 64 * opt.lr_pt_rate})
                    else:
                        params.append({"params": p})
                optimizer = optim.Adam(
                    params,
                    lr=opt.learning_rate,
                    weight_decay=opt.weight_decay)
            else:
                optimizer = optim.Adam(
                    parameters,
                    lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(
                parameters,
                lr=opt.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)
    if not opt.no_val:
        if opt.dataset in ['gtea', 'kth2']:
            spatial_transform = Compose([
                Scale(opt.sample_size),
                CenterCrop(opt.sample_size),
                ToTensor(opt.norm_value), norm_method,
            ])
        else:
            spatial_transform = Compose([
                Scale(opt.sample_size),
                CenterCrop(opt.sample_size),
                RGB2Gray(),
                ToTensor(opt.norm_value), norm_method,
            ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()
        if opt.compress == 'mask':
            spatio_temporal_transform = Coded(opt.mask_path)
        elif opt.compress == 'avg':
            spatio_temporal_transform = Averaged()
        elif opt.compress == 'one':
            spatio_temporal_transform = OneFrame()
        elif opt.compress == 'spatial':
            if opt.dataset in ['gtea', 'kth2']:
                spatial_transform = Compose([
                    Scale(opt.sample_size),
                    CenterCrop(opt.sample_size),
                    LowResolution(opt.spatial_compress_size, use_cv2=opt.use_cv2),
                    ToTensor(opt.norm_value), norm_method,
                ])
            else:
                spatial_transform = Compose([
                    Scale(opt.sample_size),
                    CenterCrop(opt.sample_size),
                    RGB2Gray(),
                    LowResolution(opt.spatial_compress_size, use_cv2=opt.use_cv2),
                    ToTensor(opt.norm_value), norm_method,
                ])
            spatio_temporal_transform = None
            temporal_transform = TemporalCenterCrop(opt.sample_duration)
        elif opt.compress == 'temporal':
            spatio_temporal_transform = Compose([
                Coded(opt.mask_path),
                ToTemporal(opt.mask_path),
            ])
        else:
            spatio_temporal_transform = None
            temporal_transform = TemporalCenterCrop(opt.sample_duration)
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform,
            spatio_temporal_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'),
            ['epoch', 'loss', 'top1', 'top5'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    if opt.load_path:
        check = False
        print('loading checkpoint {}'.format(opt.load_path))
        checkpoint = torch.load(opt.load_path)
        # assert opt.arch == checkpoint['arch']
        # model.load_state_dict(checkpoint['state_dict'])

        # for name, param in model.named_parameters():
        #     if name == 'module.exp.weight':
        #         param.requires_grad = False
        #         print('{}({}) is fixed'.format(name, param.shape))
        #         check = True
        # if not check:
        #     raise
        for i, (name, param) in enumerate(model.named_parameters()):
            for (prev_name, prev_param) in checkpoint['state_dict'].items():
                if name == prev_name:
                    param.data = prev_param.data
                    print('{}: {}({}) -> {}({})'.format(
                        i, prev_name, prev_param.shape, name, param.shape))
                    if name == 'module.exp.weight':
                        param.requires_grad = False
                    check = True
        if not check:
            raise

    if opt.init_path:
        check = False
        print('initilize checkpoint {}'.format(opt.init_path))
        checkpoint = torch.load(opt.init_path)

        for (name, param), i in zip(
                model.named_parameters(),
                range(opt.init_level)):
            for (prev_name, prev_param) in checkpoint['state_dict'].items():
                if name == prev_name:
                    param.data = prev_param.data
                    print('{}: {}({}) -> {}({})'.format(
                        i, prev_name, prev_param.shape, name, param.shape))
                    check = True
        if not check:
            raise
    if opt.fixed_mask:
        check = False
        for i, (name, param) in enumerate(model.named_parameters()):
            if name == 'module.exp.weight':
                print(name, 'FIXED!')
                param.requires_grad = False
                check = True
        if not check:
            raise

    model.to(device)
    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger, device)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger, device)

        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)

    if opt.test:
        if opt.dataset in ['gtea', 'kth2']:
            spatial_transform = Compose([
                Scale(int(opt.sample_size / opt.scale_in_test)),
                CornerCrop(opt.sample_size, opt.crop_position_in_test),
                ToTensor(opt.norm_value), norm_method,
            ])
        else:
            spatial_transform = Compose([
                Scale(int(opt.sample_size / opt.scale_in_test)),
                CornerCrop(opt.sample_size, opt.crop_position_in_test),
                RGB2Gray(),
                ToTensor(opt.norm_value), norm_method,
            ])
        temporal_transform = LoopPadding(opt.sample_duration)
        if opt.compress == 'mask':
            spatio_temporal_transform = Coded(opt.mask_path)
        elif opt.compress == 'avg':
            spatio_temporal_transform = Averaged()
        elif opt.compress == 'one':
            spatio_temporal_transform = OneFrame()
        elif opt.compress == 'spatial':
            if opt.dataset in ['gtea', 'kth2']:
                spatial_transform = Compose([
                    Scale(int(opt.sample_size / opt.scale_in_test)),
                    CornerCrop(opt.sample_size, opt.crop_position_in_test),
                    LowResolution(opt.spatial_compress_size, use_cv2=opt.use_cv2),
                    ToTensor(opt.norm_value), norm_method,
                ])
            else:
                spatial_transform = Compose([
                    Scale(int(opt.sample_size / opt.scale_in_test)),
                    CornerCrop(opt.sample_size, opt.crop_position_in_test),
                    RGB2Gray(),
                    LowResolution(opt.spatial_compress_size, use_cv2=opt.use_cv2),
                    ToTensor(opt.norm_value), norm_method,
                ])
            spatio_temporal_transform = None
        elif opt.compress == 'temporal':
            spatio_temporal_transform = Compose([
                Coded(opt.mask_path),
                ToTemporal(opt.mask_path),
            ])
        else:
            spatio_temporal_transform = None
        target_transform = VideoID()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform,
                                 spatio_temporal_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt, test_data.class_names, device)
