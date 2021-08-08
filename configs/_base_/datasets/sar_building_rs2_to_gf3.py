'''
Author: Shuailin Chen
Created Date: 2021-07-11
Last Modified: 2021-07-13
	content: 
'''

dataset_type = 'Sar_building'
data_root = 'data/ade20k/sar_building_rs2'
dst_img_dir = 'data/ade20k/sar_building_gf3'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='FourierDomainAdaption', dst_img_dir=dst_img_dir, LB=0.01),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='FourierDomainAdaption', dst_img_dir=dst_img_dir, LB=0.01),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=dst_img_dir,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=dst_img_dir,
        img_dir='images/test',
        ann_dir='annotations/test',
        pipeline=test_pipeline)
    )