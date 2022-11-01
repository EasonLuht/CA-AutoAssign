_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='AutoAssign',
    backbone=dict(
        # _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,),
    # neck=dict(
    #     type='FPN',
    #     in_channels=[96, 192, 384, 768],
    #     out_channels=256,
    #     start_level=1,
    #     add_extra_convs=True,
    #     relu_before_extra_convs=True,
    #     num_outs=5, ),
    neck=dict(
        type='FPG',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        inter_channels=256,
        num_outs=5,
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
        skip_inds=[(0, 1, 2, 3), (0, 1, 2), (0, 1), (0,), ()]),
    bbox_head=dict(
        type='AutoAssignHead',
        num_classes=9,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_bbox=dict(type='GIoULoss', loss_weight=5.0)),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(640, 640)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=64),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(train=dict(pipeline=train_pipeline),
            val=dict(pipeline=test_pipeline),
            test=dict(pipeline=test_pipeline))

optimizer = dict(_delete_=True, type='AdamW', lr=0.00005, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
# lr_config = dict(step=[27, 33])
# runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)
runner = dict(max_epochs=72)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[14, 22, 50])

fp16 = dict(loss_scale=dict(init_scale=512))

# do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )
