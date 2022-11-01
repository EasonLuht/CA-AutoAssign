checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = None
# load_from = 'work_dirs/tood_r50_fpn_mstrain_2x_coco_tianchi/best_bbox_mAP_epoch_18.pth'
load_from='work_dirs/autoassign_swin_Channel_AGM/best_bbox_mAP_epoch_32.pth'
resume_from = None
# resume_from = 'work_dirs/autoassign_swin_Channel_AGM/latest.pth'
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
