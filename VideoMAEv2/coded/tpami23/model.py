import torch
from torch import nn

from models import (
    resnet, pre_act_resnet, wide_resnet, resnext, densenet, c3d, c2d, c2d_exp,
    c2d_coord, c2d_pt, c2d_pt2, c2d_pt5, c2d_pt7, c2d_pt_exp, c2d_pt2_exp,
    c2d_pt5_exp, c2d_pt_exp_avg, c2d_pt_exp_sep, c3d_pt_exp, c2d_pt_exp_init,
    resnet_exp, resnet_pt_exp,
)


def generate_model(opt):
    assert opt.model in [
        'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet',
        'c3d', 'c2d', 'c2d_exp', 'c2d_coord',
        'c2d_pt', 'c2d_pt2', 'c2d_pt5', 'c2d_pt7',
        'c2d_pt_exp', 'c2d_pt2_exp', 'c2d_pt5_exp',
        'c2d_pt_exp_avg', 'c2d_pt_exp_sep',
        'c3d_pt_exp', 'c2d_pt_exp_init',
        'resnet18_exp',
        'resnet34_exp',
        'resnet50_exp',
        'resnet101_exp',
        'resnet152_exp',
        'resnext50_32x4d_exp',
        'resnext101_32x8d_exp',
        'wide_resnet50_2_exp',
        'wide_resnet101_2_exp',
        'resnet18_pt_exp',
        'resnet34_pt_exp',
        'resnet50_pt_exp',
        'resnet101_pt_exp',
        'resnet152_pt_exp',
        'resnext50_32x4d_pt_exp',
        'resnext101_32x8d_pt_exp',
        'wide_resnet50_2_pt_exp',
        'wide_resnet101_2_pt_exp',
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet import get_fine_tuning_parameters

        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'wideresnet':
        assert opt.model_depth in [50]

        from models.wide_resnet import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = wide_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                k=opt.wide_resnet_k,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        from models.resnext import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = resnext.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnext.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'preresnet':
        assert opt.model_depth in [18, 34, 50, 101, 152, 200]

        from models.pre_act_resnet import get_fine_tuning_parameters

        if opt.model_depth == 18:
            model = pre_act_resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = pre_act_resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = pre_act_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = pre_act_resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = pre_act_resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = pre_act_resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'densenet':
        assert opt.model_depth in [121, 169, 201, 264]

        from models.densenet import get_fine_tuning_parameters

        if opt.model_depth == 121:
            model = densenet.densenet121(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 169:
            model = densenet.densenet169(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 201:
            model = densenet.densenet201(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 264:
            model = densenet.densenet264(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    elif opt.model == 'c3d':
            model = c3d.C3D(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'c2d':
            model = c2d.C2D(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'c2d_pt':
            model = c2d_pt.C2DPt(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'c2d_pt2':
            model = c2d_pt2.C2DPt(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'c2d_pt5':
            model = c2d_pt5.C2DPt(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'c2d_pt7':
            model = c2d_pt7.C2DPt(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'c2d_exp':
            model = c2d_exp.C2DExp(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'c2d_pt_exp':
            model = c2d_pt_exp.C2DPtExp(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'c2d_pt_exp_init':
            model = c2d_pt_exp_init.C2DPtExp(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'c3d_pt_exp':
            model = c3d_pt_exp.C3DPtExp(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'c2d_pt_exp_avg':
            model = c2d_pt_exp_avg.C2DPtExpAvg(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'c2d_pt_exp_sep':
            model = c2d_pt_exp_sep.C2DPtExpSep(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'c2d_pt5_exp':
            model = c2d_pt5_exp.C2DPtExp(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'c2d_pt2_exp':
            model = c2d_pt2_exp.C2DPtExp(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'c2d_coord':
            model = c2d_coord.C2DCoord(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    elif opt.model == 'resnet18_exp':
        model = resnet_exp.resnet18(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)
    elif opt.model == 'resnet34_exp':
        model = resnet_exp.resnet34(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)
    elif opt.model == 'resnet50_exp':
        model = resnet_exp.resnet50(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)
    elif opt.model == 'resnet101_exp':
        model = resnet_exp.resnet101(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)
    elif opt.model == 'resnet152_exp':
        model = resnet_exp.resnet152(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)
    elif opt.model == 'resnext50_32x4d_exp':
        model = resnet_exp.resnext50_32x4d(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)
    elif opt.model == 'resnext101_32x8d_exp':
        model = resnet_exp.resnext101_32x8d(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)
    elif opt.model == 'wide_resnet50_2_exp':
        model = resnet_exp.wide_resnet50_2(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)
    elif opt.model == 'wide_resnet101_2_exp':
        model = resnet_exp.wide_resnet101_2(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)

    elif opt.model == 'resnet18_pt_exp':
        model = resnet_pt_exp.resnet18(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)
    elif opt.model == 'resnet34_pt_exp':
        model = resnet_pt_exp.resnet34(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)
    elif opt.model == 'resnet50_pt_exp':
        model = resnet_pt_exp.resnet50(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)
    elif opt.model == 'resnet101_pt_exp':
        model = resnet_pt_exp.resnet101(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)
    elif opt.model == 'resnet152_pt_exp':
        model = resnet_pt_exp.resnet152(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)
    elif opt.model == 'resnext50_32x4d_pt_exp':
        model = resnet_pt_exp.resnext50_32x4d(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)
    elif opt.model == 'resnext101_32x8d_pt_exp':
        model = resnet_pt_exp.resnext101_32x8d(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)
    elif opt.model == 'wide_resnet50_2_pt_exp':
        model = resnet_pt_exp.wide_resnet50_2(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)
    elif opt.model == 'wide_resnet101_2_pt_exp':
        model = resnet_pt_exp.wide_resnet101_2(
            pretrained=False, progress=True,
            num_classes=opt.n_classes,
            sample_duration=opt.sample_duration)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'])

            if opt.model == 'densenet':
                model.module.classifier = nn.Linear(
                    model.module.classifier.in_features, opt.n_finetune_classes)
                model.module.classifier = model.module.classifier.cuda()
            else:
                model.module.fc = nn.Linear(model.module.fc.in_features,
                                            opt.n_finetune_classes)
                model.module.fc = model.module.fc.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'])

            if opt.model == 'densenet':
                model.classifier = nn.Linear(
                    model.classifier.in_features, opt.n_finetune_classes)
            else:
                model.fc = nn.Linear(model.fc.in_features,
                                     opt.n_finetune_classes)

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    return model, model.parameters()
