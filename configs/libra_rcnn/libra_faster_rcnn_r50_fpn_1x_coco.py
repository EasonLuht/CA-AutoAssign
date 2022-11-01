_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# model settings
model = dict(
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        dict(
            type='BFP',
            in_channels=256,
            num_levels=5,
            refine_level=2,
            refine_type='non_local')
    ],
    roi_head=dict(
        bbox_head=dict(
            loss_bbox=dict(
                _delete_=True,
                type='BalancedL1Loss',
                alpha=0.5,
                gamma=1.5,
                beta=1.0,
                loss_weight=1.0))),)
    # model training and testing settings
    # train_cfg=dict(
    #     rpn=dict(sampler=dict(neg_pos_ub=5), allowed_border=-1),
    #     rcnn=dict(
    #         sampler=dict(
    #             _delete_=True,
    #             type='CombinedSampler',
    #             num=512,
    #             pos_fraction=0.25,
    #             add_gt_as_proposals=True,
    #             pos_sampler=dict(type='InstanceBalancedPosSampler'),
    #             neg_sampler=dict(
    #                 type='IoUBalancedNegSampler',
    #                 floor_thr=-1,
    #                 floor_fraction=0,
    #                 num_bins=3)))))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      #   img_scale=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                      #            (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                      #            (1333, 736), (1333, 768), (1333, 800)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      # img_scale=[(1333, 400), (1333, 500), (1333, 600)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      # crop_size=(600, 384),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                        # img_scale=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                        #          (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                        #          (1333, 736), (1333, 768), (1333, 800)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        # img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(train=dict(pipeline=train_pipeline),
            test=dict(pipeline=test_pipeline))


optimizer = dict(type='SGD', lr=0.00375, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[16, 22])
# runner = dict(type='EpochBasedRunner', max_epochs=24)

# lr_config = dict(step=[27, 33])
# runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)
runner = dict(max_epochs=36)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[14, 22])
    # step=[7, 14, 21])

