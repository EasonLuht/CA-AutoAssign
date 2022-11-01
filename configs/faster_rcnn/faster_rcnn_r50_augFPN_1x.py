_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(neck=dict(
        type='HighFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,),
        roi_head=dict(
            bbox_roi_extractor=dict(
                type='SoftRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),),
        rpn_head=dict(
                type='RPNHead_Aug',),
        test_cfg=dict(
            rpn=dict(
                nms_across_levels=False,
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)
            # soft-nms is also supported for rcnn testing
            # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
        )
    )