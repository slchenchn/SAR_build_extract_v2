'''
Author: Shuailin Chen
Created Date: 2021-08-28
Last Modified: 2021-09-02
	content: 
'''

_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/semi_sar_building_rs2.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

model = dict(
    type='ReCo',
    momentum = 0.99,
    strong_thres = 0.97,
    weak_thres = 0.7,
    tmperature = 0.5,
    num_queries = 256,
    num_negatives = 512,
    unlabeled_aug = dict(strong=
            [dict(type='BatchPhotoMetricDistortion'),
            dict(type='BatchNormalize', **img_norm_cfg)
            ],
            weak=[dict(type='BatchNormalize', **img_norm_cfg),
            ]
    ),

    decode_head=dict(
        type = 'Depthwise2SeparableASPPHead',
        rep_channels = 512,
        num_classes=2,
    ),

    auxiliary_head=dict(
        num_classes=2,
    )
)

runner = dict(type='SemiIterBasedRunner', max_iters=4000)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.001, by_epoch=False)
checkpoint_config = dict(by_epoch=False, interval=300000)
evaluation = dict(interval=400, metric='mIoU')

