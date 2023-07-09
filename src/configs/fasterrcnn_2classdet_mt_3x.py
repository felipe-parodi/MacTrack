default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

# model settings
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            r"Y:\\MacTrack\\results\\mactrack_detection\\fasterrcnn2class_best_bbox_mAP_epoch_50.pth"
        )),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))

# dataset settings
dataset_type = 'MacTrackDet'
data_root = 'data/mactrack_det/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='RandomFlip', prob=0.25, direction='vertical'),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=r"C:\\Users\\Felipe Parodi\\Documents\\felipe_code\\MacTrack\\data\\detection\\train.json",
        data_prefix=dict(img=r"Y:\\MacTrack\\data\\for_pseudo_labeling\\imgs\\"),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=r"C:\\Users\\Felipe Parodi\\Documents\\felipe_code\\MacTrack\\data\\detection\\test.json",
        data_prefix=dict(img=r"Y:\\MacTrack\\data\\for_pseudo_labeling\\imgs\\"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=r"C:\\Users\\Felipe Parodi\\Documents\\felipe_code\\MacTrack\\data\\detection\\test.json",
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# training schedule for 2x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
###

# data = dict(
#     samples_per_gpu=4,
#     workers_per_gpu=1,
#     train=dict(
#         type='MacTrackDet',
#         ann_file=r"C:\\Users\\Felipe Parodi\\Documents\\felipe_code\\MacTrack\\data\\detection\\train.json",
#         img_prefix=r"Y:\\MacTrack\\data\\for_pseudo_labeling\\imgs\\\\",
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(
#                 type='Resize',
#                 img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
#                            (1333, 768), (1333, 800)],
#                 multiscale_mode='value',
#                 keep_ratio=True),
#             dict(type='RandomFlip', flip_ratio=0.5),
#             dict(
#                 type='Normalize',
#                 mean=[123.675, 116.28, 103.53],
#                 std=[58.395, 57.12, 57.375],
#                 to_rgb=True),
#             dict(type='Pad', size_divisor=32),
#             dict(type='DefaultFormatBundle'),
#             dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
#         ]),
#     val=dict(
#         type='MacTrackDet',
#         ann_file=r"C:\\Users\\Felipe Parodi\\Documents\\felipe_code\\MacTrack\\data\\detection\\val.json",
#         img_prefix=r"Y:\\MacTrack\\data\\for_pseudo_labeling\\imgs\\\\",
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(
#                 type='MultiScaleFlipAug',
#                 img_scale=(1333, 800),
#                 flip=False,
#                 transforms=[
#                     dict(type='Resize', keep_ratio=True),
#                     dict(type='RandomFlip'),
#                     dict(
#                         type='Normalize',
#                         mean=[123.675, 116.28, 103.53],
#                         std=[58.395, 57.12, 57.375],
#                         to_rgb=True),
#                     dict(type='Pad', size_divisor=32),
#                     dict(type='ImageToTensor', keys=['img']),
#                     dict(type='Collect', keys=['img'])
#                 ])
#         ]),
#     test=dict(
#         type='MacTrackDet',
#         ann_file=r"C:\\Users\\Felipe Parodi\\Documents\\felipe_code\\MacTrack\\data\\detection\\test.json",
#         img_prefix=r"Y:\\MacTrack\\data\\for_pseudo_labeling\\imgs\\\\",
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(
#                 type='MultiScaleFlipAug',
#                 img_scale=(1333, 800),
#                 flip=False,
#                 transforms=[
#                     dict(type='Resize', keep_ratio=True),
#                     dict(type='RandomFlip'),
#                     dict(
#                         type='Normalize',
#                         mean=[123.675, 116.28, 103.53],
#                         std=[58.395, 57.12, 57.375],
#                         to_rgb=True),
#                     dict(type='Pad', size_divisor=32),
#                     dict(type='ImageToTensor', keys=['img']),
#                     dict(type='Collect', keys=['img'])
#                 ])
#         ]))
# evaluation = dict(metric=['bbox'], save_best='auto', interval=10)
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=250)
# checkpoint_config = dict(interval=1)
# log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
# custom_hooks = [dict(type='NumClassCheckHook')]
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# load_from = r"Y:\\MacTrack\\results\\mactrack_detection\\fasterrcnn2class_best_bbox_mAP_epoch_50.pth"
# resume_from = None
# workflow = [('train', 1)]
# opencv_num_threads = 0
# mp_start_method = 'fork'
# auto_scale_lr = dict(enable=False, base_batch_size=2)
# work_dir = r"Y:\\MacTrack\\results\\mactrack_detection\\\\"
# auto_resume = False
# gpu_ids = 0
# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(type='LoggerHook', interval=50),
#     param_scheduler=dict(type='ParamSchedulerHook'),
#     checkpoint=dict(
#         type='CheckpointHook',
#         interval=10,
#         save_best='mAP',
#         rule='greater',
#         max_keep_ckpts=1),
#     sampler_seed=dict(type='DistSamplerSeedHook'))
# # 