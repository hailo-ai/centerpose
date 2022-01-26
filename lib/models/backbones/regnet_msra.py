import torch
import torch.distributed as dist
import torch.nn as nn
from pycls.models.model_zoo import build_model, get_weights_file


BN_MOMENTUM = 0.1


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class PoseResNet(nn.Module):

    def __init__(self, config_string, split_deconv, **kwargs):
        self.deconv_with_bias = False

        super(PoseResNet, self).__init__()
        self.base = self._make_regnet(config_string)
        self.inplanes = list(self.base.s4.children())[-1].f.c_bn.weight.size()[0]

        # used for deconv layers
        new_block = []
        if split_deconv:
            planes = 256
            new_block.append(nn.Conv2d(in_channels=self.inplanes,
                                       out_channels=planes,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False))
            new_block.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            new_block.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        self.new_block = nn.Sequential(*new_block)

        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 256, 256],
            [4, 4, 4],
        )

    def _make_regnet(self, config_string):
        # make sure weights are downloaded just once in multi-process training
        if dist.is_initialized():
            if dist.get_rank() == 0:
                get_weights_file(config_string)
            dist.monitored_barrier()

        m = build_model(config_string, pretrained=True)

        # Remove avgpool & fc
        m.head = nn.Sequential()
        return m

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.base(x)
        x = self.new_block(x)
        x = self.deconv_layers(x)

        return x

    def init_weights(self, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')
            for _, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


def get_regnet(num_layers, cfg):
    model = PoseResNet(cfg.MODEL.CONFIG_STRING, cfg.MODEL.SPLIT_DECONV)
    model.init_weights()
    return model
