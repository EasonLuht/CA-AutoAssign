# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, trunc_normal_init, xavier_init, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import BaseModule, auto_fp16, ModuleList

from ..builder import NECKS

class ChannelAttention(nn.Module):
    """
        tanh通道注意力
    """
    def __init__(self, inplanes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(inplanes, inplanes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(inplanes // ratio, inplanes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
        tanh通道注意力
    """
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 1 if kernel_size == 3 else 3

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.tanh(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(inplanes=in_planes, ratio=ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


@NECKS.register_module()
class AutoCBAM_FPN_Pro(BaseModule):
    """FPN_CARAFE is a more flexible implementation of FPN. It allows more
    choice for upsample methods during the top-down pathway.

    It can reproduce the performance of ICCV 2019 paper
    CARAFE: Content-Aware ReAssembly of FEatures
    Please refer to https://arxiv.org/abs/1905.02188 for more details.

    Args:
        in_channels (list[int]): Number of channels for each input feature map.
        out_channels (int): Output channels of feature pyramids.
        num_outs (int): Number of output_NEUDET stages.
        start_level (int): Start level of feature pyramids.
            (Default: 0)
        end_level (int): End level of feature pyramids.
            (Default: -1 indicates the last level).
        norm_cfg (dict): Dictionary to construct and config norm layer.
        activate (str): Type of activation function in ConvModule
            (Default: None indicates w/o activation).
        order (dict): Order of components in ConvModule.
        upsample (str): Type of upsample layer.
        upsample_cfg (dict): Dictionary to construct and config upsample layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 norm_cfg=None,
                 act_cfg=None,
                 order=('conv', 'norm', 'act'),
                 upsample_cfg=dict(
                     type='carafe',
                     up_kernel=5,
                     up_group=1,
                     encoder_kernel=3,
                     encoder_dilation=1),
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(AutoCBAM_FPN_Pro, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_bias = norm_cfg is None
        self.upsample_cfg = upsample_cfg.copy()
        self.upsample = self.upsample_cfg.get('type')
        self.relu = nn.ReLU(inplace=False)

        self.order = order
        assert order in [('conv', 'norm', 'act'), ('act', 'conv', 'norm')]

        assert self.upsample in [
            'nearest', 'bilinear', 'deconv', 'pixel_shuffle', 'carafe', None
        ]
        if self.upsample in ['deconv', 'pixel_shuffle']:
            assert hasattr(
                self.upsample_cfg,
                'upsample_kernel') and self.upsample_cfg.upsample_kernel > 0
            self.upsample_kernel = self.upsample_cfg.pop('upsample_kernel')

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = ModuleList()
        self.fpn_convs = ModuleList()
        self.upsample_modules = ModuleList()
        self.auto_chanelattn = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=norm_cfg,
                bias=self.with_bias,
                act_cfg=act_cfg,
                inplace=False,
                order=self.order)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                bias=self.with_bias,
                act_cfg=act_cfg,
                inplace=False,
                order=self.order)
            auto_chanelattn = CBAM(out_channels, ratio=16, kernel_size=7)
            if i != self.backbone_end_level - 1:
                upsample_cfg_ = self.upsample_cfg.copy()
                upsample_cfg_.update(channels=out_channels, scale_factor=2)
                upsample_module = build_upsample_layer(upsample_cfg_)
                self.upsample_modules.append(upsample_module)
            self.lateral_convs.append(l_conv)
            self.auto_chanelattn.append(auto_chanelattn)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_out_levels = (
            num_outs - self.backbone_end_level + self.start_level)
        if extra_out_levels >= 1:
            for i in range(extra_out_levels):
                in_channels = (
                    self.in_channels[self.backbone_end_level -
                                     1] if i == 0 else out_channels)
                extra_l_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=self.with_bias,
                    act_cfg=act_cfg,
                    inplace=False,
                    order=self.order)
                upsampler_cfg_ = dict(
                    channels=out_channels,
                    scale_factor=2,
                    **self.upsample_cfg)
                upsampler_cfg_['type'] = self.upsample
                upsample_module = build_upsample_layer(upsampler_cfg_)
                auto_chanelattn = CBAM(out_channels, ratio=16, kernel_size=7)
                extra_fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.with_bias,
                    act_cfg=act_cfg,
                    inplace=False,
                    order=self.order)
                self.upsample_modules.append(upsample_module)
                self.auto_chanelattn.append(auto_chanelattn)
                self.fpn_convs.append(extra_fpn_conv)
                self.lateral_convs.append(extra_l_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of module."""
        super(AutoCBAM_FPN_Pro, self).init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()

    def slice_as(self, src, dst):
        """Slice ``src`` as ``dst``

        Note:
            ``src`` should have the same or larger size than ``dst``.

        Args:
            src (torch.Tensor): Tensors to be sliced.
            dst (torch.Tensor): ``src`` will be sliced to have the same
                size as ``dst``.

        Returns:
            torch.Tensor: Sliced tensor.
        """
        assert (src.size(2) >= dst.size(2)) and (src.size(3) >= dst.size(3))
        if src.size(2) == dst.size(2) and src.size(3) == dst.size(3):
            return src
        else:
            return src[:, :, :dst.size(2), :dst.size(3)]

    def tensor_add(self, a, b, i):
        """Add tensors ``a`` and ``b`` that might have different sizes."""
        if a.size() == b.size():
            b = b
        else:
            b = self.slice_as(b, a)
        b = b * self.auto_chanelattn[i](b)
        c = a + b
        return c

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if i <= self.backbone_end_level - self.start_level:
                input = inputs[min(i + self.start_level, len(inputs) - 1)]
            else:
                input = laterals[-1]
            lateral = lateral_conv(input)
            laterals.append(lateral)

        # build top-down path
        # for i in range(len(laterals) - 1, 0, -1):
        #     if self.upsample is not None:
        #         upsample_feat = self.upsample_modules[i - 1](laterals[i])
        #     else:
        #         upsample_feat = laterals[i]
        #     laterals[i - 1] = self.tensor_add(laterals[i - 1], upsample_feat)

        for i in range(0, len(laterals) - 1):
            if self.upsample is not None:
                upsample_feat = self.upsample_modules[i](laterals[i+1])
            else:
                upsample_feat = laterals[i+1]
            laterals[i] = self.tensor_add(laterals[i], upsample_feat, i)

        # build outputs
        num_conv_outs = len(self.fpn_convs)
        outs = []
        for i in range(num_conv_outs):
            out = self.fpn_convs[i](laterals[i])
            outs.append(out)
        return tuple(outs)





