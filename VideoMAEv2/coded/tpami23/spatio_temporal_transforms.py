import numpy as np
import torch
from random import randint
from torch import nn
from torch.nn import functional as F


def split_to_patch(arr, target_shape=(8, 8)):
    """
    shape: (t, 112, 112, 1) -> (-1, t, 8, 8, 1)
    """
    hs, ws, _ = arr.shape
    ht, wt = target_shape
    hp = hs // ht
    wp = ws // wt

    patch = []
    for w in range(wp):
        for h in range(hp):
            patch.append(arr[(ht*h):(ht*(h+1)), (wt*w):(wt*(w+1)), :])
    return np.array(patch).astype(np.float32)


def concat_patch(arr, shape=(112, 112), t=16, c=1):
    """
    shape: (-1, 16, 8, 8, 1) -> (16, 112, 112, 1)
    """
    hs, ws = shape
    hp = hs // 8
    wp = ws // 8

    wl = []
    for w in range(wp):
        hl = arr[w*hp:(w+1)*hp].reshape(-1, t, 8, 8, c)
        wl.append(np.concatenate(hl, axis=1).reshape(t, hs, 8, c))
    return np.concatenate(np.array(wl), axis=2)


def predict(model, data):
    model.eval()
    with torch.no_grad():
        data = torch.from_numpy(data).to("cuda").float()
        output = model(data).cpu().detach().numpy()
    return output


def reconstruct(arr, model, mask_mean):
    arr = arr.reshape((112, 112, 1)).astype(np.float32) / 255
    patchs = split_to_patch(arr).reshape(-1, 64)
    output = predict(model, patchs)
    output = output.reshape(-1, 16, 8, 8, 1)
    out_video = concat_patch(output).reshape((1, 16, 112, 112)).astype(np.float32)
    out_video = np.clip(out_video * 255, 0, 255).astype(np.float32)
    return out_video

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.linear4 = nn.Linear(1024, 1024)
        self.linear5 = nn.Linear(1024, 1024)

    def forward(self, x):
        x = F.relu(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        x = self.linear5(x)
        return x


class Reconstruct(object):

    def __init__(self, mask_path):
        self.mask_mean = np.load(mask_path).mean()
        print(self.mask_mean, 1/self.mask_mean)
        checkpoint = torch.load("./model_okawra_gray/model_okawra_gray_100_nostate.pth")
        model = Decoder()
        for name, p in model.named_parameters():
            for name_prev, p_prev in checkpoint.items():
                if name == name_prev:
                    p.data = p_prev.data
        self.model = model
        self.model.to("cuda")

    def __call__(self, tensor):
        img = tensor.numpy()
        video = reconstruct(img, self.model, self.mask_mean)
        return torch.from_numpy(video).float()


class Coded(object):

    def __init__(self, mask_path):
        self.mask = np.load(mask_path)

    def __call__(self, clip):
        length = len(clip)
        image = torch.zeros(1, 112, 112).float()
        for t, m in zip(clip, self.mask):
            m = torch.from_numpy(np.transpose(m, (2, 0, 1))).float()
            image += t.float() * m
        image /= length
        return image


class ToTemporal(object):

    def __init__(self, mask_path, size=4, duration=16):
        self.mask = np.load(mask_path)
        self.size = size
        self.duration = duration

    def __call__(self, tensor):
        img = tensor.numpy()
        size = self.size
        duration = self.duration

        out = []
        for i in range(duration):
            out.append(
                img.reshape(112, 112)[(self.mask[i] == 1).reshape(112, 112)]
                .reshape(112//size, 112//size)
                .repeat(size, axis=0).repeat(size, axis=1)
                .reshape(1, 112, 112)
            )

        out = torch.from_numpy(
            np.array(out).astype(np.float32)
        ).float().permute(1, 0, 2, 3)
        return out

    def randomize_parameters(self):
        pass


class Averaged(object):

    def __call__(self, clip):
        length = len(clip)
        image = torch.zeros(1, 112, 112).float()
        for t in clip:
            image += t
        image /= length
        return image


class OneFrame(object):

    def __call__(self, clip, fixed=True):
        image = torch.zeros(1, 112, 112).float()
        # image = clip[randint(0, len(clip)-1)]
        # image = clip[0]
        if fixed:
            image = clip[int(len(clip)//2)]
        else:
            image = clip[randint(0, len(clip)-1)]
        return image


class ToRepeat(object):

    def __init__(self, func, duration=16):
        self.duration = duration
        self.func = func

    def __call__(self, clip):
        tensor = self.func(clip)
        img = tensor.numpy()

        out = []
        for i in range(self.duration):
            out.append(img)

        out = torch.from_numpy(
            np.array(out).astype(np.float32)
        ).float().permute(1, 0, 2, 3)
        return out

    def randomize_parameters(self):
        pass
