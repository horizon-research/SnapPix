import torch
import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import json
# method_decorator
from functools import partial, update_wrapper
# threadsafe_generator
import threading


# ##################################################################
#                        method_decorator
# ##################################################################
class classonlymethod(classmethod):
    def __get__(self, instance, cls=None):
        if instance is not None:
            raise AttributeError("This method is available only on the class, not on instances.")
        return super().__get__(instance, cls)


def _update_method_wrapper(_wrapper, decorator):
    # _multi_decorate()'s bound_method isn't available in this scope. Cheat by
    # using it on a dummy function.
    @decorator
    def dummy(*args, **kwargs):
        pass
    update_wrapper(_wrapper, dummy)


def _multi_decorate(decorators, method):
    """
    Decorate `method` with one or more function decorators. `decorators` can be
    a single decorator or an iterable of decorators.
    """
    if hasattr(decorators, '__iter__'):
        # Apply a list/tuple of decorators if 'decorators' is one. Decorator
        # functions are applied so that the call order is the same as the
        # order in which they appear in the iterable.
        decorators = decorators[::-1]
    else:
        decorators = [decorators]

    def _wrapper(self, *args, **kwargs):
        # bound_method has the signature that 'decorator' expects i.e. no
        # 'self' argument, but it's a closure over self so it can call
        # 'func'. Also, wrap method.__get__() in a function because new
        # attributes can't be set on bound method objects, only on functions.
        bound_method = partial(method.__get__(self, type(self)))
        for dec in decorators:
            bound_method = dec(bound_method)
        return bound_method(*args, **kwargs)

    # Copy any attributes that a decorator adds to the function it decorates.
    for dec in decorators:
        _update_method_wrapper(_wrapper, dec)
    # Preserve any existing attributes of 'method', including the name.
    update_wrapper(_wrapper, method)
    return _wrapper


def method_decorator(decorator, name=''):
    """
    Convert a function decorator into a method decorator
    """
    # 'obj' can be a class or a function. If 'obj' is a function at the time it
    # is passed to _dec,  it will eventually be a method of the class it is
    # defined on. If 'obj' is a class, the 'name' is required to be the name
    # of the method that will be decorated.
    def _dec(obj):
        if not isinstance(obj, type):
            return _multi_decorate(decorator, obj)
        if not (name and hasattr(obj, name)):
            raise ValueError(
                "The keyword argument `name` must be the name of a method "
                "of the decorated class: %s. Got '%s' instead." % (obj, name)
            )
        method = getattr(obj, name)
        if not callable(method):
            raise TypeError(
                "Cannot decorate '%s' as it isn't a callable attribute of "
                "%s (%s)." % (name, obj, method)
            )
        _wrapper = _multi_decorate(decorator, method)
        setattr(obj, name, _wrapper)
        return obj

    # Don't worry about making _dec look similar to a list/tuple as it's rather
    # meaningless.
    if not hasattr(decorator, '__iter__'):
        update_wrapper(_dec, decorator)
    # Change the name to aid debugging.
    obj = decorator if hasattr(decorator, '__name__') else decorator.__class__
    _dec.__name__ = 'method_decorator(%s)' % obj.__name__
    return _dec
# ##################################################################
#                        method_decorator
# ##################################################################


# ##################################################################
#                        threadsafe_generator
# ##################################################################
class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe."""
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g
# ##################################################################
#                        threadsafe_generator
# ##################################################################


class KTH2(data.Dataset):
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
                 n_classes=6):
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.spatio_temporal_transform = spatio_temporal_transform
        self.target_transform = target_transform
        self.root_path = root_path
        self.sample_duration = sample_duration
        """
        Training:   person11, 12, 13, 14, 15, 16, 17, 18
        Validation: person19, 20, 21, 23, 24, 25, 01, 04
        Test:       person22, 02, 03, 05, 06, 07, 08, 09, 10
        """
        if subset == 'training':
            self.sets = [11, 12, 13, 14, 15, 16, 17, 18]
        elif subset == 'validation':
            self.sets = [19, 20, 21, 23, 24, 25, 1, 4]
        else:
            self.sets = [22, 2, 3, 5, 6, 7, 8, 9, 10]
            self.sets = [6]
        self.n = None
        self.g = self.gen_generator()

    def pre_process(self, X, y):
        clip = [Image.fromarray(arr) for arr in X]
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        if self.spatio_temporal_transform is not None:
            clip = self.spatio_temporal_transform(clip)
        else:
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = int(y)
        return clip, target

    @method_decorator(threadsafe_generator)
    def gen_generator(self):
        while True:
            for s in self.sets:
                Xs = np.load(os.path.join(self.root_path, 'P%02d_X_%s.npy' % (s, self.sample_duration)))
                ys = np.load(os.path.join(self.root_path, 'P%02d_y_%s.npy' % (s, self.sample_duration)))
                # for X, y in zip(Xs, ys):
                for i, (X, y) in enumerate(zip(Xs, ys)):
                    if i % 16 == 0:
                        yield self.pre_process(X, y)

    def __getitem__(self, index):
        return next(self.g)

    def __len__(self):
        if self.n is None:
            with open(os.path.join(self.root_path, 'size.json'), 'r') as f:
                size = json.load(f)
            # self.n = sum([size[str(s)] for s in self.sets])
            self.n = sum([size[str(s)]//16 for s in self.sets])
        return self.n

# class KTH2(data.Dataset):
#     """
#     Args:
#         root (string): Root directory path.
#         spatial_transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
#             and returns a transformed version
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         loader (callable, optional): A function to load an video given its path and frame indices.
#      Attributes:
#         classes (list): List of the class names.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#         imgs (list): List of (image path, class_index) tuples
#     """
#
#     def __init__(self,
#                  root_path,
#                  annotation_path,
#                  subset,
#                  n_samples_for_each_video=1,
#                  spatial_transform=None,
#                  temporal_transform=None,
#                  spatio_temporal_transform=None,
#                  target_transform=None,
#                  sample_duration=16,
#                  n_classes=6):
#         self.spatial_transform = spatial_transform
#         self.temporal_transform = temporal_transform
#         self.spatio_temporal_transform = spatio_temporal_transform
#         self.target_transform = target_transform
#         self.load_data(root_path, n_classes, sample_duration, subset)
#
#     def load_data(self, root_path, n_classes, sample_duration, subset):
#         """
#         Training:   person11, 12, 13, 14, 15, 16, 17, 18
#         Validation: person19, 20, 21, 23, 24, 25, 01, 04
#         Test:       person22, 02, 03, 05, 06, 07, 08, 09, 10
#         """
#         video_path = root_path
#         if subset == 'training':
#             sets = [11, 12, 13, 14, 15, 16, 17, 18]
#         elif subset == 'validation':
#             sets = [19, 20, 21, 23, 24, 25, 1, 4]
#         else:
#             sets = [22, 2, 3, 5, 6, 7, 8, 9, 10]
#         self.X = np.concatenate([
#             np.load(os.path.join(video_path, 'P%02d_X_%s.npy' % (s, sample_duration))) for s in sets])
#         self.y = np.concatenate([
#             np.load(os.path.join(video_path, 'P%02d_y_%s.npy' % (s, sample_duration))) for s in sets])
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is class_index of the target class.
#         """
#         clip = [Image.fromarray(arr) for arr in self.X[index]]
#         if self.spatial_transform is not None:
#             self.spatial_transform.randomize_parameters()
#             clip = [self.spatial_transform(img) for img in clip]
#         if self.spatio_temporal_transform is not None:
#             clip = self.spatio_temporal_transform(clip)
#         else:
#             clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
#
#         target = int(self.y[index])
#         # if self.target_transform is not None:
#         #     target = self.target_transform(target)
#
#         return clip, target
#
#     def __len__(self):
#         return len(self.X)
