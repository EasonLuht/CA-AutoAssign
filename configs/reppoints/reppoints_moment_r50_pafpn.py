_base_ = './reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py'

model = dict(
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5))