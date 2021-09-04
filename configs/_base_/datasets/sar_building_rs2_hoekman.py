'''
Author: Shuailin Chen
Created Date: 2021-07-11
Last Modified: 2021-09-04
	content: strong augs for labeled and unlabeled data
'''

dataset_type = 'Sar_multich'
data_root = 'data/ade20k/sar_building'
train_pipeline = [
    dict(type='LoadNpyFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='PhotoMetricDistortion'),
    # dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadNpyFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
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
        img_dir='npy/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline,
        split='split/RS2/training.txt',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='npy/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline,
        split='split/RS2/validation.txt',
        ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='npy/test',
        ann_dir='annotations/test',
        pipeline=test_pipeline,
        split='split/RS2/test.txt',
        )
    )