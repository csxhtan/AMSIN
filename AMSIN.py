import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import module_util as mutil
from time import time
from torchstat import stat

from thop import profile
# from thop.profile import profile
from thop import clever_format


# from PIL import Image
# import torchvision.transforms as transforms

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, channel_in, channel_out):
        super(ResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.eca = eca_layer(feature)
        self.se = SELayer(feature)
        self.conv3 = nn.Conv2d((feature + channel_in), channel_out, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)

        residual = self.conv2(residual)
        residual = self.relu(residual)

        residual_eca = self.eca(residual)
        input = torch.cat((x, residual_eca), dim=1)
        out = self.conv3(input)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        # return y.expand_as(x)


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init != 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
        elif net_structure == 'Resnet':
            return ResBlock(channel_in, channel_out)
        else:
            return None

    return constructor


class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1 / 16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)

            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac


class InvNet(nn.Module):
    def __init__(self, sp1=12, sp2=12, subnet_constructor=None, block_num=[], down_num=2):
        super(InvNet, self).__init__()

        operations = []

        for i in range(down_num):
            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, sp1 + sp2, sp1)
                operations.append(b)

        self.operations = nn.ModuleList(operations)
        self.tanh = nn.Tanh()

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out


class InvDDNet(nn.Module):

    def __init__(self):
        super(InvDDNet, self).__init__()
        self.down_num = 1
        self.haar3 = HaarDownsampling(3).cuda()
        self.ps = nn.PixelShuffle(2)
        self.pus = nn.PixelUnshuffle(2)
        self.net1 = InvNet(12, 12, subnet('Resnet', 'xavier'), [3], self.down_num).cuda()
        self.net2 = InvNet(24, 12, subnet('Resnet', 'xavier'), [3], self.down_num).cuda()
        self.net3 = InvNet(48, 12, subnet('Resnet', 'xavier'), [4], self.down_num).cuda()
        self.net4 = InvNet(24, 12, subnet('Resnet', 'xavier'), [8], self.down_num).cuda()
        self.net5 = InvNet(12, 12, subnet('Resnet', 'xavier'), [8], self.down_num).cuda()

    def forward(self, x1, x2, rev=False):
        if rev:
            x1_b = self.pus(x1)  # 12
            x2_b = self.haar3(x2)  # 12
            b_fuse = self.net5(torch.cat([x1_b, x2_b], dim=1), rev=True)  # 24
            x1_b = b_fuse[:, :12]  # 12
            x2_b = b_fuse[:, 12:]  # 12
            x1_c = self.pus(x1_b[:, :6])  # 24
            x2_c = self.haar3(x2_b[:, :3])  # 12
            c_fuse = self.net4(torch.cat([x1_c, x2_c], dim=1), rev=True)  # 36
            x1_c = c_fuse[:, :24]  # 24
            x2_c = c_fuse[:, 24:]  # 12
            x1_d = self.pus(x1_c[:, :12])  # 48
            x2_d = self.haar3(x2_c[:, :3])  # 12
            d_fuse = self.net3(torch.cat([x1_d, x2_d], dim=1), rev=True)  # 60
            x1_d = d_fuse[:, :48]  # 48
            x2_d = d_fuse[:, 48:]  # 12
            outc = torch.cat([self.ps(x1_d), self.haar3(x2_d, rev=True)], dim=1)  # 15
            x1_c = torch.cat([outc[:, :12], x1_c[:, 12:]], dim=1)  # 24
            x2_c = torch.cat([outc[:, 12:], x2_c[:, 3:]], dim=1)  # 12
            out_c_fuse = self.net2(torch.cat([x1_c, x2_c], dim=1), rev=True)  # 36
            outb = torch.cat([self.ps(out_c_fuse[:, :24]), self.haar3(out_c_fuse[:, 24:], rev=True)], dim=1)  # 9
            x1_b = torch.cat([outb[:, :6], x1_b[:, 6:]], dim=1)  # 12
            x2_b = torch.cat([outb[:, 6:], x2_b[:, 3:]], dim=1)  # 12
            out = self.net1(torch.cat([x1_b, x2_b], dim=1), rev=True)  # 24
            out = torch.cat([self.ps(out[:, :12]), self.haar3(out[:, 12:], rev=True)], dim=1)  # 6
            return out, time()
        else:
            x1_b = self.pus(x1)  # 12
            x2_b = self.haar3(x2)  # 12
            b_fuse = self.net1(torch.cat([x1_b, x2_b], dim=1))  # 24
            x1_b = b_fuse[:, :12]  # 12
            x2_b = b_fuse[:, 12:]  # 12
            x1_c = self.pus(x1_b[:, :6])  # 24
            x2_c = self.haar3(x2_b[:, :3])  # 12
            c_fuse = self.net2(torch.cat([x1_c, x2_c], dim=1))  # 36
            x1_c = c_fuse[:, :24]  # 24
            x2_c = c_fuse[:, 24:]  # 12
            x1_d = self.pus(x1_c[:, :12])  # 48
            x2_d = self.haar3(x2_c[:, :3])  # 12
            d_fuse = self.net3(torch.cat([x1_d, x2_d], dim=1))  # 60
            x1_d = d_fuse[:, :48]  # 48
            x2_d = d_fuse[:, 48:]  # 12
            outc = torch.cat([self.ps(x1_d), self.haar3(x2_d, rev=True)], dim=1)  # 15
            x1_c = torch.cat([outc[:, :12], x1_c[:, 12:]], dim=1)  # 24
            x2_c = torch.cat([outc[:, 12:], x2_c[:, 3:]], dim=1)  # 12
            out_c_fuse = self.net4(torch.cat([x1_c, x2_c], dim=1))  # 36
            outb = torch.cat([self.ps(out_c_fuse[:, :24]), self.haar3(out_c_fuse[:, 24:], rev=True)], dim=1)  # 9
            x1_b = torch.cat([outb[:, :6], x1_b[:, 6:]], dim=1)  # 12
            x2_b = torch.cat([outb[:, 6:], x2_b[:, 3:]], dim=1)  # 12
            out = self.net5(torch.cat([x1_b, x2_b], dim=1))  # 24
            out = torch.cat([self.ps(out[:, :12]), self.haar3(out[:, 12:], rev=True)], dim=1)  # 6
            return out, outb, outc, time()


if __name__ == '__main__':
    print('here')

