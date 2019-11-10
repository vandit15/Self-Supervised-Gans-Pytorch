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
import os.path
import tarfile, sys, math
from six.moves import urllib
from ops import conv2d, deconv2d, Residual_G, Residual_D
import numpy as np
import scipy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.autograd import grad as torch_grad


class Discriminator(nn.Module):
    def __init__(self, spectral_normed, num_rotation,
                ssup, channel, resnet = False):
        super(Discriminator, self).__init__()
        self.resnet = resnet
        self.num_rotation = num_rotation
        self.ssup = ssup

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.conv1 = conv2d(channel, 64, kernel_size = 3, stride = 1, padding = 1,
                            spectral_normed = spectral_normed)
        self.conv2 = conv2d(64, 128, spectral_normed = spectral_normed,
                            padding = 0)
        self.conv3 = conv2d(128, 256, spectral_normed = spectral_normed,
                            padding = 0)
        self.conv4 = conv2d(256, 512, spectral_normed = spectral_normed,
                            padding = 0)
        self.fully_connect_gan1 = nn.Linear(512, 1)
        self.fully_connect_rot1 = nn.Linear(512, 4)
        self.softmax = nn.Softmax()

        self.re1 = Residual_D(channel, 128, spectral_normed = spectral_normed,
                            down_sampling = True, is_start = True)
        self.re2 = Residual_D(128, 128, spectral_normed = spectral_normed,
                            down_sampling = True)
        self.re3 = Residual_D(128, 128, spectral_normed = spectral_normed)
        self.re4 = Residual_D(128, 128, spectral_normed = spectral_normed)
        self.fully_connect_gan2 = nn.Linear(128, 1)
        self.fully_connect_rot2 = nn.Linear(128, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.resnet == False:
            conv1 = self.lrelu(self.conv1(x))
            conv2 = self.lrelu(self.conv2(conv1))
            conv3 = self.lrelu(self.conv3(conv2))
            conv4 = self.lrelu(self.conv4(conv3))
            conv4 = torch.view(conv4.size(0)*self.num_rotation, -1)
            gan_logits = self.fully_connect_gan1(conv4)
            if self.ssup:
                rot_logits = self.fully_connect_rot1(conv4)
                rot_prob = self.softmax(rot_logits)
        else:
            re1 = self.re1(x)
            re2 = self.re2(re1)
            re3 = self.re3(re2)
            re4 = self.re4(re3)
            re4 = self.relu(re4)
            re4 = torch.sum(re4,dim = (2,3))
            gan_logits = self.fully_connect_gan2(re4)
            if self.ssup:
                rot_logits = self.fully_connect_rot2(re4)
                rot_prob = self.softmax(rot_logits)

        if self.ssup:
            return self.sigmoid(gan_logits), gan_logits, rot_logits, rot_prob
        else:
            return self.sigmoid(gan_logits), gan_logits



class Generator(nn.Module):
    def __init__(self, z_size, channel, resnet = False, output_size = 32):
        super(Generator, self).__init__()
        s = 4
        self.output_size = output_size
        if self.output_size == 32:
            s = 4
        if self.output_size == 48:
            s = 6
        self.s = s
        self.z_size = z_size
        self.resnet = resnet
        self.fully_connect = nn.Linear(z_size, s*s*256)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.deconv1 = deconv2d(256, 256, padding = 0)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = deconv2d(256, 128, padding = 0) 
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = deconv2d(128, 64, padding = 0)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = conv2d(64, channel, padding = 1, kernel_size = 3, stride = 1)
        self.conv_res4 = conv2d(256,channel, padding = 1, kernel_size = 3, stride = 1)

        self.re1 = Residual_G(256, 256, up_sampling = True)
        self.re2 = Residual_G(256, 256, up_sampling = True)
        self.re3 = Residual_G(256, 256, up_sampling = True)
        self.bn = nn.BatchNorm2d(256)

    def forward(self, x):
        d1 = self.fully_connect(x)
        d1 = d1.view(-1, 256, self.s, self.s)
        if self.resnet == False:
            d1 = self.relu(d1)
            d2 = self.relu(self.bn1(self.deconv1(d1)))
            d3 = self.relu(self.bn2(self.deconv2(d2)))
            d4 = self.relu(self.bn3(self.deconv3(d3)))
            d5 = self.conv4(d4)
        else:
            d2 = self.re1(d1)
            d3 = self.re2(d2)
            d4 = self.re3(d3)
            d4 = self.relu(self.bn(d4))
            d5 = self.conv_res4(d4)

        return self.tanh(d5)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.z_size))

