# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, trunc_normal_init, MaxPool2d
from mmcv.runner import BaseModule, auto_fp16

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
        # self.w = nn.Parameter(torch.FloatTensor(1, inplanes, 1, 1), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        self.w1.data.fill_(0.5)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        # w_1 = nn.Sigmoid(self.w)
        # w_2 = 1 - w_1
        #
        # out = w_1 * avg_out + w_2 * max_out
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

        self.conv = nn.Conv2d(4, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x, feat):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out, feat], dim=1)
        x = self.conv(x)
        return self.tanh(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(inplanes=in_planes, ratio=ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x, feat):
        out = x * self.ca(x)
        result = x * self.sa(out, feat)
        return result

class extra_attention(nn.Module):
    """
    空间模块
    """
    def __init__(self, kernel_size=3):
        super(extra_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 1 if kernel_size == 3 else 3

        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = self.conv(avg_out)

        return self.sigmoid(x)

class extra_NEM(nn.Module):
    # def __init__(self, in_planes, ratio=16, kernel_size=7):
    def __init__(self, kernel_size=7):
        super(extra_NEM, self).__init__()

        self.sa = extra_attention(kernel_size)

    def forward(self, x, x_up):
        result = x_up - x_up * self.sa(x)
        avg_feat = torch.mean(result, dim=1, keepdim=True)
        max_feat, _ = torch.max(result, dim=1, keepdim=True)
        result = torch.cat([avg_feat, max_feat], dim=1)
        return result

@NECKS.register_module()
class AutoCBAM_FPNplus(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output_NEUDET channels (used at each scale)
        num_outs (int): Number of output_NEUDET scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output_NEUDET feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(AutoCBAM_FPNplus, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        # self.backbone_end_level：理解为 用来计数需要计算的FPN的最后层数，要么为给定值，要么为输入的channel的数量
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
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.auto_chanelattn = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            # in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, 同nn.Conv2d
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            auto_chanelattn = CBAM(out_channels, ratio=16, kernel_size=7)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.auto_chanelattn.append(auto_chanelattn)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        # 判断是否有除原层数之外的多余层数
        # faster_rcnn中 self.add_extra_convs 为 false
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        # 设计额外模块
        self.extra_conv = ConvModule(
                self.in_channels[0],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
        self.extra_NEM = extra_NEM(kernel_size=7)
        # self.extra_down = nn.MaxPool2d(
        #     kernel_size=3,
        #     stride=2,
        #     padding=1
        # )
        self.extra_downs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            extra_down = ConvModule(
                2,
                2,
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.extra_downs.append(extra_down)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        """
            inputs: (B, C, W, H)
        """
        assert len(inputs) == len(self.in_channels)

        # for x_1 in inputs:
        #     print(x_1.shape)

        # build laterals
        # 将backbone输出的特征进行压缩
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)

        # 额外补充模块
        extra_map = self.extra_conv(inputs[0])
        extra_shape = inputs[0].shape[2:]
        extra_up_feat = F.interpolate(laterals[1], size=extra_shape, **self.upsample_cfg)
        extra_map = self.extra_NEM(extra_up_feat, extra_map)
        extras = []
        for i in range(0, used_backbone_levels - 1, 1):
            if i == 0:
                extra_down = self.extra_downs[i](extra_map)
            else:
                extra_down = self.extra_downs[i](extras[i-1])
            extras.append(extra_down)

        for i in range(0, used_backbone_levels - 1, 1):
            prev_shape = laterals[i].shape[2:]
            # 进行上采样
            up_feat = F.interpolate(laterals[i + 1], size=prev_shape, **self.upsample_cfg)
            up_feat = up_feat * self.auto_chanelattn[i](up_feat, extras[i])
            laterals[i] += up_feat
            # extra_map = self.extra_down(extra_map)
            # laterals[i] += extra_map

        # Auto_Substract
        # used_backbone_levels = len(laterals)

        # 额外补充模块
        # extra_map = self.extra_conv(inputs[0])
        # extra_shape = inputs[0].shape[2:]
        # extra_up_feat = F.interpolate(laterals[1], size=extra_shape, **self.upsample_cfg)
        # extra_map = self.extra_NEM(extra_up_feat, extra_map)
        # extras = []
        # for i in range(0, used_backbone_levels - 1, 1):
        #     if i == 0:
        #         extra_down = self.extra_downs[i](extra_map)
        #     else:
        #         extra_down = self.extra_downs[i](extras[i-1])
        #     extras.append(extra_down)

        # for i in range(0, used_backbone_levels-1, 1):
        #     prev_shape = laterals[i].shape[2:]
        #     # 进行上采样
        #     up_feat = F.interpolate(laterals[i+1], size=prev_shape, **self.upsample_cfg)
        #     up_feat = up_feat * self.auto_chanelattn[i](up_feat, laterals[i])
        #     laterals[i] += up_feat
        #     laterals[i] += extras[i]


        # for i in range(used_backbone_levels - 1, 0, -1):
        #     #当没有指定时，指定size
        #     prev_shape = laterals[i - 1].shape[2:]
        #     up_feat = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
        #     up_feat = up_feat * self.auto_chanelattn[i-1](up_feat)
        #     laterals[i - 1] += up_feat

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)





