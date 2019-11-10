"""
    Pytorch implementation of Self-Supervised GAN
    Reference: "Self-Supervised GANs via Auxiliary Rotation Loss"
    Authors: Ting Chen,
                Xiaohua Zhai,
                Marvin Ritter,
                Mario Lucic and
                Neil Houlsby
    https://arxiv.org/abs/1811.11212 CVPR 2019.
    Script Author: Vandit Jain. Github:vandit15
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        if weight_mat.is_cuda:
            u = u.cuda()
        v = weight_mat.t() @ u
        v = v / v.norm()
        u = weight_mat @ v
        u = u / u.norm()
        weight_sn = weight_mat / (u.t() @ weight_mat @ v)
        weight_sn = weight_sn.view(*size)

        return weight_sn, Variable(u.data)

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        input_size = weight.size(0)
        u = Variable(torch.randn(input_size, 1) * 0.1, requires_grad=False)
        setattr(module, name + '_u', u)
        setattr(module, name, fn.compute_weight(module)[0])

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)

def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

    return module


def log_sum_exp(x, axis = 1):
    m = torch.max(x, keepdim = True)
    return m + torch.logsumexp(x - m, dim = 1, keepdim = True)


class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, padding, kernel_size = 4, stride = 2,
                spectral_normed = False):
        super(conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                padding = padding)
        if spectral_normed:
            self.conv = spectral_norm(self.conv)

    def forward(self, input):
        out = self.conv(input)
        return out


class deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, padding, kernel_size = (4,4), stride = (2,2),
                spectral_normed = False, iter = 1):
        super(deconv2d, self).__init__()

        self.devconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                        stride, padding = padding)
        if spectral_normed:
            self.devconv = spectral_norm(self.deconv)

    def forward(self, input):
        out = self.devconv(input)
        return out    


def conv_cond_concat(x, y):
    x_shapes = list(x.size())
    y_shapes = list(y.size())
    return torch.cat((x,y*torch,ones(x_shapes[0],x_shapes[1],x_shapes[2],y_shapes[3])))


class Residual_G(nn.Module):
    def __init__(self, in_channels, out_channels = 256, kernel_size = 3, stride = 1, 
                spectral_normed = False, up_sampling = False):
        super(Residual_G, self).__init__()
        self.up_sampling = up_sampling
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.conv1 = conv2d(in_channels, out_channels, spectral_normed = spectral_normed,
                            kernel_size = kernel_size, stride = stride, padding = 1)
        self.conv2 = conv2d(out_channels, out_channels, spectral_normed= spectral_normed, 
                            kernel_size = kernel_size, stride = stride, padding = 1)

    def forward(self, x):
        input = x
        x = self.relu(self.batch_norm1(x))
        if self.up_sampling:
            x = self.upsample(x)
        x = self.conv1(x)
        x = self.batch_norm2(x)
        x = self.conv2(self.relu(x))
        if self.up_sampling:
            return self.upsample(input) + x
        else:
            return input + x


class Residual_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 3, stride = 1,
                spectral_normed = False, down_sampling = False, is_start = False):
        super(Residual_D, self).__init__()
        self.down_sampling = down_sampling
        self.is_start = is_start

        self.avgpool_short = nn.AvgPool2d(2, 2, padding = 1)
        self.conv_short = conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0,
                                spectral_normed = False)
        self.conv1 = conv2d(in_channels, out_channels, spectral_normed = spectral_normed,
                            kernel_size = kernel, stride = stride, padding = 1)
        self.conv2 = conv2d(out_channels, out_channels, spectral_normed = spectral_normed,
                            kernel_size = kernel, stride = stride, padding = 1)
        self.avgpool2 = nn.AvgPool2d(2, 2, padding = 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        input = x
        if self.is_start:
            conv1 = self.relu(self.conv1(x))
            conv2 = self.relu(self.conv2(conv1))
            if self.down_sampling:
                conv2 = self.avgpool2(conv2)
        else:
            conv1 = self.conv1(self.relu(x))
            conv2 = self.conv2(self.relu(conv1))
            if self.down_sampling:
                conv2 = self.avgpool2(conv2)

        if self.down_sampling:
            input = self.avgpool_short(input)
        resi = self.conv_short(input)

        return resi + conv2

