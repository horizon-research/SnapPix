# import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
# from functools import partial

__all__ = [
    'C3D'
]


class C3D(nn.Module):

    def __init__(self,
                 sample_size,
                 sample_duration,
                 num_classes=400):
        self.inplanes = 64
        self.sample_duration = sample_duration
        super().__init__()
        self.activation = F.relu

        self.conv1 = nn.Conv3d(1, 64, 3, 1, padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        # self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv3d(64, 128, 3, 1, padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3a = nn.Conv3d(128, 256, 3, 1, padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, 3, 1, padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        # self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4a = nn.Conv3d(256, 512, 3, 1, padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, 3, 1, padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(512)
        # self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5a = nn.Conv3d(512, 512, 3, 1, padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, 3, 1, padding=(1, 1, 1))
        self.bn5 = nn.BatchNorm3d(512)
        # self.pool5 = nn.MaxPool2d(2, 2, padding=(1, 0))
        if self.sample_duration == 16:
            self.fc6 = nn.Linear(512*4*4, 4096)
        elif self.sample_duration == 64:
            self.fc6 = nn.Linear(512*2*4*4, 4096)
        else:
            raise
        self.bn6 = nn.BatchNorm2d(4096)
        self.fc7 = nn.Linear(4096, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = F.max_pool3d(x, 2, 2)

        print(x.size())

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = F.max_pool3d(x, 2, 2)

        x = self.conv3a(x)
        x = self.activation(x)
        x = self.conv3b(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = F.max_pool3d(x, 2, 2)

        x = self.conv4a(x)
        x = self.activation(x)
        x = self.conv4b(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = F.max_pool3d(x, 2, 2)

        x = self.conv5a(x)
        x = self.activation(x)
        x = self.conv5b(x)
        x = self.bn5(x)
        x = self.activation(x)
        x = F.max_pool3d(x, 2, 2, padding=(0, 1, 1))

        if self.sample_duration == 16:
            x = x.view(-1, 512*4*4)
        elif self.sample_duration == 64:
            x = x.view(-1, 512*2*4*4)
        else:
            raise
        x = F.dropout2d(x, 0.5, training=self.training)
        x = F.relu(self.fc6(x))
        x = F.dropout2d(x, 0.5, training=self.training)
        x = self.fc7(x)

        return F.log_softmax(x, dim=1)


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters
