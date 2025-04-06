from datasets.kinetics import Kinetics
from datasets.activitynet import ActivityNet
from datasets.ucf101 import UCF101
from datasets.hmdb51 import HMDB51
from datasets.kth import KTH
from datasets.kth2 import KTH2
from datasets.something2 import Something2
from datasets.something2_init import Something2Init
from datasets.gtea import GTEA
from datasets.jester import Jester
from datasets.real import REAL


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform,
                     spatio_temporal_transform=None):
    assert opt.dataset in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'kth', 'kth2',
        'sth', 'sth_init', 'gtea', 'jester',
    ]

    if opt.dataset == 'kinetics':
        training_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'activitynet':
        training_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            'training',
            False,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'ucf101':
        training_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'hmdb51':
        training_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'kth':
        training_data = KTH(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'kth2':
        training_data = KTH2(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'sth':
        training_data = Something2(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'jester':
        training_data = Jester(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'sth_init':
        training_data = Something2Init(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'gtea':
        training_data = GTEA(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform,
            test_split=opt.test_split)

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform,
                       spatio_temporal_transform=None):
    assert opt.dataset in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'kth', 'kth2',
        'sth', 'sth_init', 'gtea', 'jester',
    ]

    if opt.dataset == 'kinetics':
        validation_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'activitynet':
        validation_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            'validation',
            False,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        validation_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'hmdb51':
        validation_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'kth':
        validation_data = KTH(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'kth2':
        validation_data = KTH2(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'sth':
        validation_data = Something2(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'jester':
        validation_data = Jester(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'sth_init':
        validation_data = Something2Init(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'gtea':
        validation_data = GTEA(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform,
            test_split=opt.test_split)
    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform,
                 spatio_temporal_transform=None):
    assert opt.dataset in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'kth', 'kth2',
        'sth', 'sth_init', 'gtea', 'jester',
        'real',
    ]
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'kinetics':
        test_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'activitynet':
        test_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            subset,
            True,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        test_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            subset,
            10,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'hmdb51':
        test_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'kth':
        test_data = KTH(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'kth2':
        test_data = KTH2(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'sth':
        test_data = Something2(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'jester':
        test_data = Jester(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'sth_init':
        test_data = Something2Init(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'gtea':
        test_data = GTEA(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            test_split=opt.test_split)
    elif opt.dataset == 'real':
        test_data = REAL(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            spatio_temporal_transform=spatio_temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)

    return test_data
