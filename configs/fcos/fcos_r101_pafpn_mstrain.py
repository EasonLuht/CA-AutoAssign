_base_ = './fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py'

model = dict(
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5))
