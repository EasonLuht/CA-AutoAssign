_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco.py'
# model settings
model = dict(
    neck=dict(
        type='AutoCBAM_Channel_FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        # start_level=1,
        # add_extra_convs=True,
        # relu_before_extra_convs=True,
        num_outs=5, ),
)