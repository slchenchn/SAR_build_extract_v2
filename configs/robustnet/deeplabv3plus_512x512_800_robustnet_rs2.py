'''
Author: Shuailin Chen
Created Date: 2021-07-11
Last Modified: 2021-08-28
	content: 
'''

_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/sar_building_rs2.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]


# change num_classes
# norm_cfg = dict(type='BN', requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        norm_cfg = norm_cfg,
        out_indices=(0, 1, 2, 3),
    ),
    decode_head=dict(
        norm_cfg = norm_cfg,
        num_classes=2,
    ),
    auxiliary_head=[
        dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='NewFCNHead',
        in_channels=(1024, 1),  
        in_index=(0, 1,),
        channels=256,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        input_transform = 'multiple_select',
        loss_decode=dict(
            type='RelaxedInstanceWhiteningLoss', relax_denom=256, loss_weight=0))
    ]

    # auxiliary_head=
    #     dict(
    #     type='NewFCNHead',
    #     in_channels=(1024, 1),  
    #     in_index=(0, 1,),
    #     channels=256,
    #     num_convs=0,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     input_transform = 'multiple_select',
    #     loss_decode=dict(
    #         type='RelaxedInstanceWhiteningLoss', relax_denom=64, loss_weight=0.6)),
    
)

find_unused_parameters = True

# for schedule# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.005)
lr_config = dict(policy='poly', power=0.9, min_lr=0.001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=800)
checkpoint_config = dict(by_epoch=False, interval=100)
evaluation = dict(interval=100, metric='mIoU')
