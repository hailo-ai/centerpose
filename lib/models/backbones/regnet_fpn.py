# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math

import torch
import torch.distributed as dist
import torch.nn as nn
from pycls.models.model_zoo import build_model, get_weights_file

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class RegNetFPN(nn.Module):

    def __init__(self, config_string, fpn_level=None):
        self.deconv_with_bias = False
        if fpn_level is None:
            self.fpn_level = 4
        else:
            self.fpn_level = fpn_level

        super(RegNetFPN, self).__init__()
        # make sure weights are downloaded just once in multi-process training
        if dist.is_initialized():
            if dist.get_rank() == 0:
                get_weights_file(config_string)
            dist.monitored_barrier()

        self.base = build_model(config_string, pretrained=True)

        self.inplanes = list(self.base.s4.children())[-1].f.c_bn.weight.size()[0]

        # used for deconv layers
        self.deconv_layer1 = self._make_deconv_layer(256, 4)
        self.deconv_layer2 = self._make_deconv_layer(128, 4)
        self.deconv_layer3 = self._make_deconv_layer(64, 4)

        self.smooth_layer1 = FakeDeformConv(256, 256)
        self.smooth_layer2 = FakeDeformConv(128, 128)
        self.smooth_layer3 = FakeDeformConv(64, 64)

        c3_channels = list(self.base.s3.children())[-1].f.c_bn.weight.size()[0]
        self.project_layer1 = FakeDeformConv(c3_channels, 256)

        c2_channels = list(self.base.s2.children())[-1].f.c_bn.weight.size()[0]
        self.project_layer2 = FakeDeformConv(c2_channels, 128)

        c1_channels = list(self.base.s1.children())[-1].f.c_bn.weight.size()[0]
        self.project_layer3 = FakeDeformConv(c1_channels, 64)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError('Invalid deconv_kernel value:', deconv_kernel)

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_filters, num_kernels):

        layers = []

        kernel, padding, output_padding = \
            self._get_deconv_cfg(num_kernels)

        planes = num_filters
        # fc = DCN(self.inplanes, planes,
        #          kernel_size=(3, 3), stride=1,
        #          padding=1, dilation=1, deformable_groups=1)
        fc = nn.Conv2d(self.inplanes, planes,
                       kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=False)
        fill_fc_weights(fc)
        up = nn.ConvTranspose2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=kernel,
            stride=2,
            padding=padding,
            output_padding=output_padding,
            bias=self.deconv_with_bias)
        fill_up_weights(up)

        layers.append(fc)
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))
        layers.append(up)
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.base.stem(x)

        c1 = self.base.s1(x)
        c2 = self.base.s2(c1)
        c3 = self.base.s3(c2)
        c4 = self.base.s4(c3)

        # print('x', x.shape)
        # print('c1', c1.shape)
        # print('c2', c2.shape)
        # print('c3', c3.shape)
        # print('c4', c4.shape)

        p4 = c4

        deconv_p4 = self.deconv_layer1(p4)
        project_layer_c3 = self.project_layer1(c3)
        p3 = self.smooth_layer1(deconv_p4 + project_layer_c3)
        deconv_layer_p3 = self.deconv_layer2(p3)
        project_layer_c2 = self.project_layer2(c2)
        p2 = self.smooth_layer2(deconv_layer_p3 + project_layer_c2)
        deconv_layer_p2 = self.deconv_layer3(p2)
        project_layer_c1 = self.project_layer3(c1)
        p1 = self.smooth_layer3(deconv_layer_p2 + project_layer_c1)

        if self.fpn_level == 4:
            pyramid_level = p1
        elif self.fpn_level == 8:
            pyramid_level = deconv_layer_p2
        elif self.fpn_level == 16:
            pyramid_level = deconv_layer_p3
        elif self.fpn_level == 32:
            pyramid_level = p4
        else:
            raise ValueError('Invalid FPN level ', self.fpn_level)

        return pyramid_level


class FakeDeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(FakeDeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        # self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.conv = nn.Conv2d(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1)
        for name, m in self.actf.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


def get_pose_net(num_layers, cfg):
    model = RegNetFPN(cfg.MODEL.CONFIG_STRING, fpn_level=num_layers)
    return model
