_base_ = './reppoints_moment_r50_fpn_1x_coco.py'
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(neck=dict(
        type='AutoCBAM_Channel_FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        relu_before_extra_convs=True,
        num_outs=5,
        norm_cfg=norm_cfg),
        bbox_head=dict(norm_cfg=norm_cfg))
# optimizer = dict(lr=0.01)
optimizer = dict(lr=0.0025)