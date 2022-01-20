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
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


BN_MOMENTUM = 0.1


model_param = {
    "RepVGG-A0": dict(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[0.75, 0.75, 0.75, 2.5],
        override_groups_map=None,
    ),
    "RepVGG-A1": dict(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
    ),
    "RepVGG-A2": dict(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[1.5, 1.5, 1.5, 2.75],
        override_groups_map=None,
    ),
    "RepVGG-B0": dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
    ),
    "RepVGG-B1": dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=None,
    ),
}


activations = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ReLU6": nn.ReLU6,
    "SELU": nn.SELU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "PReLU": nn.PReLU,
    None: nn.Identity,
}


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


def act_layers(name):
    assert name in activations.keys()
    if name == "LeakyReLU":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif name == "GELU":
        return nn.GELU()
    elif name == "PReLU":
        return nn.PReLU()
    else:
        return activations[name](inplace=True)


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class RepVGGConvModule(nn.Module):
    """
    RepVGG Conv Block from paper RepVGG: Making VGG-style ConvNets Great Again
    https://arxiv.org/abs/2101.03697
    https://github.com/DingXiaoH/RepVGG
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        activation="ReLU",
        padding_mode="zeros",
        deploy=False,
        **kwargs
    ):
        super(RepVGGConvModule, self).__init__()
        assert activation is None or isinstance(activation, str)
        self.activation = activation

        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        if deploy:
            self.rbr_reparam = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=False,
                    padding_mode=padding_mode,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )

        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=padding_11,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )
            print("RepVGG Block, identity = ", self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you
    #   do to the other models.  May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )


class RepVGG(nn.Module):

    def __init__(self, config_string, fpn_level=None):
        super(RepVGG, self).__init__()
        self.deconv_with_bias = False
        if fpn_level is None:
            self.fpn_level = 16
        else:
            self.fpn_level = fpn_level
        width_multiplier = model_param[config_string]["width_multiplier"]
        num_blocks = model_param[config_string]["num_blocks"]
        self.override_groups_map = (
            model_param[config_string]["override_groups_map"] or dict()
        )
        self.activation = "ReLU"
        self.deploy = True
        self.cur_layer_idx = 1
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGConvModule(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            activation=self.activation,
            deploy=self.deploy,
        )
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks[0], stride=2
        )
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), num_blocks[1], stride=2
        )
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), num_blocks[2], stride=2
        )
        self.stage4 = self._make_stage(
            int(512 * width_multiplier[3]), num_blocks[3], stride=2
        )
        self.inplanes = int(512 * width_multiplier[3])

        # used for deconv layers
        self.deconv_layer1 = self._make_deconv_layer(int(256 * width_multiplier[2]), 4)
        self.deconv_layer2 = self._make_deconv_layer(int(128 * width_multiplier[1]), 4)
        self.deconv_layer3 = self._make_deconv_layer(int(128 * width_multiplier[1]), 4)

        self.smooth_layer1 = FakeDeformConv(int(256 * width_multiplier[2]), int(256 * width_multiplier[2]))
        self.smooth_layer2 = FakeDeformConv(int(128 * width_multiplier[1]), int(128 * width_multiplier[1]))
        self.smooth_layer3 = FakeDeformConv(int(128 * width_multiplier[1]), int(128 * width_multiplier[1]))

        self.project_layer1 = FakeDeformConv(int(256 * width_multiplier[2]), int(256 * width_multiplier[2]))
        self.project_layer2 = FakeDeformConv(int(128 * width_multiplier[1]), int(128 * width_multiplier[1]))
        self.project_layer3 = FakeDeformConv(int(128 * width_multiplier[1]), int(128 * width_multiplier[1]))

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepVGGConvModule(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=cur_groups,
                    activation=self.activation,
                    deploy=self.deploy,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

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
        x = self.stage0(x)

        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        p4 = c4

        deconv_p4 = self.deconv_layer1(p4)
        project_layer_c3 = self.project_layer1(c3)
        p3 = self.smooth_layer1(deconv_p4 + project_layer_c3)
        deconv_layer_p3 = self.deconv_layer2(p3)
        project_layer_c2 = self.project_layer2(c2)
        p2 = self.smooth_layer2(deconv_layer_p3 + project_layer_c2)
        deconv_layer_p2 = self.deconv_layer3(p2)
    
        return deconv_layer_p2

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

def get_repvgg(num_layers, cfg):
    model = RepVGG(cfg.MODEL.CONFIG_STRING, fpn_level=num_layers)
    return model
