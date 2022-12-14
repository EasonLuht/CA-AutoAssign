# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, trunc_normal_init
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


class CAM(nn.Module):
    def __init__(self, inplanes, ratio=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc11 = nn.Conv2d(inplanes, inplanes // ratio, 1, bias=False)
        self.fc12 = nn.Conv2d(inplanes // ratio, inplanes, 1, bias=False)
        self.fc21 = nn.Conv2d(inplanes, inplanes // ratio, 1, bias=False)
        self.fc22 = nn.Conv2d(inplanes // ratio, inplanes, 1, bias=False)

        self.relu = nn.ReLU()
        self.w1 = nn.Conv2d(inplanes, 8, 1, bias=False)
        self.w2 = nn.Conv2d(inplanes, 8, 1, bias=False)
        self.w = nn.Conv2d(8*2, 2, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        B, C, _, _ = x.shape
        avg_out = self.fc12(self.relu(self.fc11(self.avg_pool(x))))
        max_out = self.fc22(self.relu(self.fc21(self.max_pool(x))))

        avg_w = self.w1(avg_out)
        max_w = self.w2(max_out)
        w = torch.cat([avg_w, max_w], dim=1)
        w = self.softmax(w)
        out = avg_out * w[:, 0:1, :, :] + max_out * w[:, 1:2, :, :]

        return self.sigmoid(out)

class NGM(nn.Module):
    def __init__(self, kernel_size=3):
        super(NGM, self).__init__()

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

class AGM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(AGM, self).__init__()
        self.ca = CAM(inplanes=in_planes, ratio=ratio)
        self.sa = NGM(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = x * self.sa(out)
        return result


class CSM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CSM, self).__init__()
        self.pixelshuffle = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=in_channel//4, out_channels=out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        x_shuffle = self.pixelshuffle(x)
        x_shuffle = self.conv(x_shuffle)

        return x_shuffle

@NECKS.register_module()
class CA_FPN(BaseModule):
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
        super(CA_FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

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
        self.ReLU = nn.ReLU()

        for i in range(self.start_level, self.backbone_end_level):
            if in_channels[i] > out_channels:
                l_conv = CSM(
                    in_channels[i],
                    out_channels
                )
            else:
                l_conv = ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False)
            auto_chanelattn = AGM(out_channels, ratio=16, kernel_size=7)
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


    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        """
            inputs: (B, C, W, H)
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Auto_Substract
        used_backbone_levels = len(laterals)
        for i in range(0, used_backbone_levels-1, 1):
            prev_shape = laterals[i].shape[2:]
            up_feat = F.interpolate(laterals[i+1], size=prev_shape, **self.upsample_cfg)
            up_feat = up_feat * self.auto_chanelattn[i](up_feat)
            ## up_feat = laterals[i] * self.auto_chanelattn[i](up_feat)
            # ----------------------------------------------------------------
            # gate = self.auto_chanelattn[i](up_feat)
            # up_feat = laterals[i] * torch.clamp(gate, max=0) + up_feat * torch.clamp(gate, min=0)
            # ----------------------------------------------------------------
            # up_feat = up_feat * self.auto_chanelattn[i](up_feat, laterals[i])
            laterals[i] = laterals[i] + up_feat

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # outs = laterals

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

        # from tools.visualization.backbone_featuremap import draw_feature_map
        # draw_feature_map(outs, save_dir="output_feature/fpn/assign_fpn_CBAM")
        return tuple(outs)





