import torch
import torch.utils.data as data
from PIL import Image
import os
import numpy as np


class GTEA(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 spatio_temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 test_split=2,
                 n_classes=61):
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.spatio_temporal_transform = spatio_temporal_transform
        self.target_transform = target_transform
        self.load_data(root_path, n_classes, sample_duration, subset, test_split)

    def load_data(self, root_path, n_classes, sample_duration, subset, test_split):
        video_path = root_path
        if subset == 'training':
            sets = [i for i in range(1, 5) if i != test_split]
        else:
            sets = [test_split]
        # data61_16/S1_X.npy
        self.X = np.concatenate([np.load(os.path.join(video_path, 'S%s_X.npy' % s)) for s in sets])
        self.y = np.concatenate([np.load(os.path.join(video_path, 'S%s_y.npy' % s)) for s in sets])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        index *= 16
        clip = [Image.fromarray(arr) for arr in self.X[index]]
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        if self.spatio_temporal_transform is not None:
            clip = self.spatio_temporal_transform(clip)
        else:
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = int(self.y[index])
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.X) // 16
