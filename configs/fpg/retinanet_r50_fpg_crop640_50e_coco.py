_base_ = '../nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py'

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    neck=dict(
        _delete_=True,
        type='FPG',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        inter_channels=256,
        num_outs=5,
        add_extra_convs=True,
        start_level=1,
        stack_times=9,
        paths=['bu'] * 9,
        same_down_trans=None,
        same_up_trans=dict(
            type='conv',
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            inplace=False,
            order=('act', 'conv', 'norm')),
        across_lateral_trans=dict(
            type='conv',
            kernel_size=1,
            norm_cfg=norm_cfg,
            inplace=False,
            order=('act', 'conv', 'norm')),
        across_down_trans=dict(
            type='interpolation_conv',
            mode='nearest',
            kernel_size=3,
            norm_cfg=norm_cfg,
            order=('act', 'conv', 'norm'),
            inplace=False),
        across_up_trans=None,
        across_skip_trans=dict(
            type='conv',
            kernel_size=1,
            norm_cfg=norm_cfg,
            inplace=False,
            order=('act', 'conv', 'norm')),
        output_trans=dict(
            type='last_conv',
            kernel_size=3,
            order=('act', 'conv', 'norm'),
            inplace=False),
        norm_cfg=norm_cfg,
        skip_inds=[(0, 1, 2, 3), (0, 1, 2), (0, 1), (0, ), ()]))

# evaluation = dict(interval=2)
evaluation = dict(interval=1, save_best='auto', metric='bbox')